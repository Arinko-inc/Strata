#!/usr/bin/env python3
"""Pictures in batch slots (strata-mesh IMPL-PLAN-remote-batch-vision.md 7.2): requests with synthetic pictures
(SVE1 embeddings, no image encoder) decode in batch slots beside text requests, greedy, and every slot's tokens must
equal the same request decoded alone (GEN / GENI).  It drives the engine's lines directly, as tools/batch_test.py does.

The scenes (run in this order, each one's slots compared with their solo records):
  S1  a picture after a text: BGEN A T1 -> BADM -> BGENI B P1 (P1's picture near the prompt's start, its filler after
      the picture: P1's read takes two chunks, and the text slot's windows run between them and after it)
  S4  the text slot's conversation goes on: BGEN A F (F = T1 + its answer + a next turn) continues from slot A's
      state (the K the S1 windows wrote); compared with F decoded alone after T1 alone
  S2  a text after a picture: BGENI A P1 -> BADM -> BGEN B T1
  S3  two pictures and two texts (--batch 4): BGENI 0 P1, BGENI 1 P2, BGEN 2 T2, BGEN 3 T3
P1 is a 32x32 grid (1,024 tokens, the text after it 992 positions back), P2 an 8x40 grid (320 tokens, nx != ny).
The picture sits near the prompt's start (cell 4) so that a text slot's cells (about 30..130) fall inside the
picture's cells: with one position table for the whole device the text slot would rotate by the picture's grid.

A pass: every slot of every scene IDENTICAL (exit 0); a slot that differs: exit 2; a setup failure (ERR, a slot
that ended at its admission, no tokens, the engine not running the batch asked for, S4 not continued from its
slot): exit 1.

--expect-differ (the negative control: STRATA_SLOT_POSITIONS=0 on every process, one table per device as before
the fix): the scenes' target slots must DIFFER from their solo records - S1's text slot and S2's picture slot (and
with S3, slot 0).  Exit 0 when every target differs, 2 when one does not (that scene does not tell the fix from
the bug), 1 on a setup failure.

  python3 tools/batch_vision_test.py --exe build-cuda/strata --config ~/b/rb-main3-vision.json --batch 4 \
      --extra "--pcie-frac 0 --adapt-every 1000000 --batch-groups 2" --out ~/b/rb-vision-g2
  (one GPU: --config strata-<model>.json --extra "--pcie-frac 0 --adapt-every 1000000")
  --dry-run: the plan and the prompts' layout, without an engine (the tokenizer only if the config's is there)
"""
import argparse
import json
import math
import os
import struct
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

IMAGE_PAD = 248056          # qwen4exp.ple.image_token_id (src/program/generate.cpp kImagePad)
SVE1 = 0x31455653
FILLER = ("The committee reviewed item %d of the long agenda, noting the budget, the schedule and the open risks, "
          "and asked the staff to report back next quarter. ")
NEXT_TURN = "Now summarize your answer in three short sentences."
SCENE_ORDER = ["S1", "S4", "S2", "S3"]
GRIDS = {"P1": (32, 32), "P2": (8, 40)}   # nx, ny


class SetupError(Exception):
    """Not a verdict on the slots: the run did not test what it asked for."""


class NoVerdict(Exception):
    """One scene ran but cannot tell (its slots' cells, or where it continued from): the next scenes still run."""


def text_prompt(q):
    return f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def image_prompt(n, filler):
    return ("<|im_start|>user\n<|vision_start|>" + "<|image_pad|>" * n + "<|vision_end|>\n" + filler +
            "Describe the picture.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")


def filler_text(tok, n_tokens):
    if n_tokens <= 0:
        return ""
    per = max(1, len(tok.encode(FILLER % 100))) if tok else 30
    return "".join(FILLER % i for i in range(n_tokens // per + 1)) + "\n\n"


def write_sve(path, nx, ny, width, phase):
    """One SVE1 record: int32 'SVE1', n, nx, ny, width, then n x width float32 (deterministic, about +-0.2)."""
    n = nx * ny
    try:
        import numpy as np
        rows = (np.sin(np.arange(n * width, dtype=np.float64) * 0.013 + phase) * 0.2).astype("<f4").tobytes()
    except ImportError:
        from array import array
        a = array("f", (math.sin(i * 0.013 + phase) * 0.2 for i in range(n * width)))
        if sys.byteorder != "little":
            a.byteswap()
        rows = a.tobytes()
    with open(path, "wb") as f:
        f.write(struct.pack("<5i", SVE1, n, nx, ny, width))
        f.write(rows)


def read_sve_header(path):
    with open(path, "rb") as f:
        return struct.unpack("<5i", f.read(20))


def model_width(cfg, given):
    if given:
        return given
    args = cfg["args"]
    if "--native" not in args:
        raise SetupError("the config has no --native GGUF to read the embedding width from: give --width")
    from gguf_reader import GGUFFile
    md = GGUFFile(Path(args[args.index("--native") + 1])).metadata
    arch = md.get("general.architecture")
    w = md.get(f"{arch}.embedding_length")
    if not w:
        raise SetupError(f"no {arch}.embedding_length in the GGUF: give --width")
    return int(w)


def parse_done(line):
    f = line.split()
    return {"generated": int(f[1]), "prompt": int(f[2]), "finish": f[5],
            "reused": int(f[8]) if len(f) > 8 else None}


def command(kind, max_new, ids, keys=(), emb=None, slot=None):
    head = ([kind, str(slot)] if slot is not None else [kind]) + [str(max_new)] + list(keys)
    if emb:
        head.append(str(emb))
    return " ".join(head + [",".join(map(str, ids))])


def solo(eng, out, req, keys=()):
    """GEN / GENI alone: the tokens and the DONE fields."""
    eng.send(command("GENI" if req["emb"] else "GEN", req["max_new"], req["ids"], keys, req["emb"]))
    got, t0 = [], time.time()
    for line in out:
        if line.startswith("T "):
            got.append(int(line.split()[1]))
        elif line.startswith("DONE"):
            d = parse_done(line)
            print(f"  solo {req['name']}: {len(got)} tokens in {time.time() - t0:.1f}s, prompt {d['prompt']}, "
                  f"reused {d['reused']}, {d['finish']}", flush=True)
            if len(got) < min(8, req["max_new"]):
                raise SetupError(f"solo {req['name']} made {len(got)} tokens: too few to compare ({line[:80]})")
            return got, d
        elif line.startswith("ERR"):
            raise SetupError(f"solo {req['name']}: {line}")
    raise SetupError("the engine's output ended (it exited?)")


def scene(eng, out, items):
    """Admit items [(slot, req, keys)] in order (each after the last one's BADM), then wait for every slot's BDONE.
    The tokens per slot (the admission's T line and the BT lines) and each admission's DONE fields."""
    got = {slot: [] for slot, _, _ in items}
    adm, pending, ended = {}, set(), {}
    t0 = time.time()

    def take(line):
        if line.startswith("BT "):
            _, s, y = line.split()
            if int(s) in got:
                got[int(s)].append(int(y))
        elif line.startswith("BDONE "):
            f = line.split()
            pending.discard(int(f[1]))
            ended[int(f[1])] = line
            if len(f) > 3 and f[3] == "nonfinite":
                raise SetupError(f"slot {f[1]} ended non-finite: {line}")
        elif line.startswith("ERR"):
            raise SetupError(line)

    for slot, req, keys in items:
        eng.send(command("BGENI" if req["emb"] else "BGEN", req["max_new"], req["ids"], keys, req["emb"], slot))
        for line in out:
            if line.startswith("T "):
                got[slot].append(int(line.split()[1]))
            elif line.startswith("DONE"):
                adm[slot] = parse_done(line)
            elif line.startswith("BADM "):
                f = line.split()
                if int(f[1]) != slot:
                    raise SetupError(f"BADM for slot {f[1]}, slot {slot} was asked: {line}")
                if f[2] != "1":
                    raise SetupError(f"slot {slot} ({req['name']}) ended at its admission (BADM 0, "
                                     f"DONE {adm.get(slot)}): the scene does not run it beside the others")
                pending.add(slot)
                break
            else:
                take(line)
        else:
            raise SetupError("the engine's output ended (it exited?)")
    for line in out:
        if not pending:
            break
        take(line)
        if not pending:
            break
    if pending:
        raise SetupError(f"the engine's output ended with slots {sorted(pending)} still running")
    print(f"  {len(items)} slots in {time.time() - t0:.1f}s; admissions: " +
          ", ".join(f"slot {s} {r['name']} prompt {adm[s]['prompt']} reused {adm[s]['reused']}" for s, r, _ in items),
          flush=True)
    return got, adm


def compare(name, slot, req, mine, ref):
    if not mine or not ref:
        raise SetupError(f"{name} slot {slot} ({req['name']}): no tokens (batch {len(mine)}, solo {len(ref)})")
    same = mine == ref
    at = next((k for k in range(min(len(mine), len(ref))) if mine[k] != ref[k]), None)
    if not same and at is None:
        at = min(len(mine), len(ref))   # one is the other's prefix: a length difference
    verdict = "IDENTICAL" if same else f"DIFFERS at {at}"
    print(f"  {name} slot {slot} ({req['name']}): {len(mine)} tokens, solo {len(ref)}: {verdict}", flush=True)
    return {"slot": slot, "request": req["name"], "batch": mine, "solo": ref, "identical": same, "first_diff": at}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exe", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--batch", type=int, default=4, help="the engine's --batch (S3 needs 4)")
    ap.add_argument("--extra", default="", help='more engine arguments, e.g. "--pcie-frac 0 --adapt-every 1000000"')
    ap.add_argument("--scenes", default="", help="a comma list of S1,S2,S3,S4 (default: all; --expect-differ: S1,S2)")
    ap.add_argument("--pair", default="0,1", help="the slots A,B of S1 / S2 / S4 (default 0,1)")
    ap.add_argument("--max-new-text", type=int, default=96)
    ap.add_argument("--max-new-image", type=int, default=64)
    ap.add_argument("--filler", type=int, default=3000,
                    help="about this many filler tokens after P1's picture (3000: P1's read takes 2 chunks of 2048)")
    ap.add_argument("--width", type=int, default=0, help="the embedding width (default: the --native GGUF's)")
    ap.add_argument("--expect-differ", action="store_true",
                    help="the negative control: the scenes' target slots must differ (run with STRATA_SLOT_POSITIONS=0)")
    ap.add_argument("--no-slot-positions", action="store_true",
                    help="STRATA_SLOT_POSITIONS=0 for this engine (the workers need it too: stages.sh --no-slot-positions)")
    ap.add_argument("--keep-parking", action="store_true",
                    help="do not add --conversation-cache-mib 0 (a parked conversation could serve S4 instead of its slot)")
    ap.add_argument("--mt-min", default="1", help="STRATA_IQ_MT_MIN for the engine (1: exact; empty: the default)")
    ap.add_argument("--out", default="", help="a directory for the embeddings, results.json and the engine log")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    scenes = [s.strip().upper() for s in (a.scenes or ("S1,S2" if a.expect_differ else ",".join(SCENE_ORDER))).split(",")
              if s.strip()]
    bad = [s for s in scenes if s not in SCENE_ORDER]
    if bad:
        ap.error(f"unknown scenes {bad}")
    if "S4" in scenes and "S1" not in scenes:
        ap.error("S4 continues S1's text slot: ask for S1 too")
    if a.expect_differ and "S4" in scenes:
        ap.error("--expect-differ has no target in S4 (S1's text slot already differs): leave it out")
    if "S3" in scenes and a.batch < 4:
        ap.error("S3 needs --batch 4")
    scenes = [s for s in SCENE_ORDER if s in scenes]
    A, B = (int(x) for x in a.pair.split(","))
    if A == B or not (0 <= A < a.batch and 0 <= B < a.batch):
        ap.error("--pair: two different slots below --batch")

    cfg_path = Path(a.config).expanduser()
    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else None
    tok = None
    tok_dir = Path(cfg["tokenizer"]) if cfg and cfg.get("tokenizer") else None
    if tok_dir and (tok_dir / "vocab.json").exists():
        from batch_test import tokenizer
        tok = tokenizer(tok_dir)
    elif not a.dry_run:
        raise SystemExit(f"no tokenizer: {a.config} " + ("is not there" if cfg is None else f"names {tok_dir}"))

    from batch_test import QUESTIONS
    texts = {"T1": QUESTIONS[0], "T2": QUESTIONS[1], "T3": QUESTIONS[2]}
    pfill = filler_text(tok, a.filler)

    def enc(s):
        return tok.encode(s, parse_special=True) if tok else None

    out_dir = Path(a.out).expanduser().resolve() if a.out else \
        Path(os.environ.get("TMPDIR", "/tmp")) / f"batch_vision_test-{os.getpid()}"
    reqs = {}
    for name, q in texts.items():
        reqs[name] = {"name": name, "ids": enc(text_prompt(q)), "emb": None, "max_new": a.max_new_text}
    for name, (nx, ny) in GRIDS.items():
        reqs[name] = {"name": name, "ids": enc(image_prompt(nx * ny, pfill if name == "P1" else "")),
                      "emb": str(out_dir / f"{name}.sve"), "max_new": a.max_new_image, "grid": (nx, ny)}

    # the layout: where the pictures sit, and whether a text slot's cells fall inside P1's picture
    print(f"batch_vision_test: scenes {','.join(scenes)}, pair {A},{B}, --batch {a.batch}"
          + (", EXPECT-DIFFER" if a.expect_differ else "") + (", STRATA_SLOT_POSITIONS=0" if a.no_slot_positions else ""))
    if tok:
        pad = tok.encode("<|image_pad|>", parse_special=True)
        if pad != [IMAGE_PAD]:
            raise SystemExit(f"<|image_pad|> encodes to {pad}, the engine reads {IMAGE_PAD}: another model?")
        for name in ("P1", "P2"):
            ids = reqs[name]["ids"]
            start = ids.index(IMAGE_PAD)
            nx, ny = reqs[name]["grid"]
            print(f"  {name}: {nx}x{ny} = {nx * ny} picture tokens at cells {start}..{start + nx * ny - 1}, "
                  f"prompt {len(ids)} tokens, the text after it {max(nx, ny) - nx * ny} positions back")
        t1 = len(reqs["T1"]["ids"])
        p1s = reqs["P1"]["ids"].index(IMAGE_PAD)
        p1e = p1s + 32 * 32
        print(f"  T1: prompt {t1} tokens; its slot writes cells {t1}..{t1 + a.max_new_text - 1} "
              f"({'inside' if p1s < t1 and t1 + a.max_new_text <= p1e else 'NOT inside'} P1's picture cells)")
        if not (p1s < t1 and t1 + a.max_new_text <= p1e):
            raise SystemExit("the text slot's cells are not inside P1's picture: S1 could not tell the bug apart")
    else:
        print("  (no tokenizer here: the prompts' layout is not shown)")
    plan = {
        "S1": [(A, "T1", ()), (B, "P1", ())],
        "S4": [(A, "F", ())],
        "S2": [(A, "P1", ()), (B, "T1", ())],
        "S3": [(0, "P1", ()), (1, "P2", ()), (2, "T2", ()), (3, "T3", ())],
    }
    targets = {"S1": A, "S2": A, "S3": 0}
    if a.dry_run:
        print("  solo records first (S4's F right after T1, before any slot holds T1; every record but T1's with ckpt=0)")
        for s in scenes:
            print(f"  {s}: " + " -> ".join(f"{'BGENI' if n.startswith('P') else 'BGEN'} {sl} {n}"
                                         f"{' ' + ' '.join(k) if k else ''}" for sl, n, k in plan[s])
                  + (f"   (target slot {targets[s]})" if a.expect_differ else ""))
        print("dry run: no engine started, no file written")
        return 0

    cfg_args = list(cfg["args"])
    extra = a.extra.split()
    if "--vision" not in cfg_args + extra:
        extra.append("--vision")
    if not a.keep_parking:
        extra += ["--conversation-cache-mib", "0"]
    out_dir.mkdir(parents=True, exist_ok=True)
    width = model_width(cfg, a.width)
    phase = {"P1": 0.0, "P2": 1.7}
    for name, (nx, ny) in GRIDS.items():
        write_sve(reqs[name]["emb"], nx, ny, width, phase[name])
        h = read_sve_header(reqs[name]["emb"])
        assert h == (SVE1, nx * ny, nx, ny, width), h
    print(f"  embeddings: width {width}, {out_dir}/P1.sve, P2.sve")

    env = {}
    if a.mt_min:
        env["STRATA_IQ_MT_MIN"] = a.mt_min
    if a.no_slot_positions:
        env["STRATA_SLOT_POSITIONS"] = "0"
    os.environ.setdefault("BATCH_TEST_LOG", str(out_dir / "engine.log"))
    from batch_test import Engine
    eng = Engine(a.exe, cfg, a.batch, env, extra)
    out = eng.lines()
    results = {"scenes": {}, "pair": [A, B], "batch": a.batch, "extra": extra, "expect_differ": a.expect_differ,
               "no_slot_positions": a.no_slot_positions, "engine_info": eng.info, "width": width}
    rc = 0
    try:
        # ---- the solo records, before any slot holds a conversation: a plain GEN also continues from an idle slot
        print("solo:", flush=True)
        ref = {}
        f_continued = True
        need = sorted({n for s in scenes for _, n, _ in plan[s] if n != "F"}, key=lambda n: (n != "T1", n))
        for name in need:
            # T1 keeps its checkpoints (F continues from it); every other record keeps none (ckpt=0)
            ref[name], _ = solo(eng, out, reqs[name], () if name == "T1" else ("ckpt=0",))
            if name == "T1" and "S4" in scenes:
                # F: T1, its answer, a next turn - decoded alone right after T1 (it continues T1's live state).
                # ckpt=0: it leaves no checkpoint deeper than slot A will hold, so S4 can only continue from the slot
                answer = ref["T1"]
                im_end = tok.encode("<|im_end|>", parse_special=True)
                rest = ("\n" if answer[-1:] == im_end else "<|im_end|>\n") + \
                    f"<|im_start|>user\n{NEXT_TURN}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
                reqs["F"] = {"name": "F", "ids": reqs["T1"]["ids"] + answer + enc(rest), "emb": None,
                             "max_new": a.max_new_text}
                ref["F"], fd = solo(eng, out, reqs["F"], ("ckpt=0",))
                if (fd["reused"] or 0) <= len(reqs["T1"]["ids"]):
                    f_continued = False
                    print(f"  note: solo F reused {fd['reused']} tokens (not T1's live state): its record was read "
                          "from the prompt, not continued (a difference in S4 is then no verdict)",
                          flush=True)
        # ---- the scenes
        for s in scenes:
            print(f"{s}:", flush=True)
            items = [(sl, reqs[n], k) for sl, n, k in plan[s]]
            got, adm = scene(eng, out, items)
            rows = [compare(s, sl, r, got[sl], ref[r["name"]]) for sl, r, _ in items]
            entry = {"slots": rows, "admissions": {str(k): v for k, v in adm.items()}}
            results["scenes"][s] = entry
            try:
                if s == "S1":
                    t1 = len(reqs["T1"]["ids"])
                    n_a = len(got[A])
                    entry["text_cells"] = [t1, t1 + n_a - 1]
                    print(f"  S1: slot {A} wrote cells {t1}..{t1 + n_a - 1}; P1's admission reused "
                          f"{adm[B]['reused']} of {adm[B]['prompt']} prompt tokens", flush=True)
                    p1s = reqs["P1"]["ids"].index(IMAGE_PAD)
                    if not (p1s < t1 and t1 + n_a - 1 < p1s + 32 * 32):
                        raise NoVerdict(f"S1: slot {A}'s cells {t1}..{t1 + n_a - 1} are not inside P1's picture cells "
                                         f"{p1s}..{p1s + 32 * 32 - 1}: the scene cannot tell the bug apart")
                    if (adm[B]["reused"] or 0) > 0:
                        print("  note: P1's admission did not read its whole prompt (a checkpoint): fewer chunks ran "
                              "beside slot A's windows", flush=True)
                if s == "S4":
                    reused, t1 = adm[A]["reused"] or 0, len(reqs["T1"]["ids"])
                    entry["continued_from_slot"] = reused > t1
                    if reused <= t1:
                        raise NoVerdict(f"S4: F's admission reused {reused} tokens, not more than T1's prompt ({t1}): "
                                        f"it did not continue from slot {A}'s state")
                    if not f_continued and not rows[0]["identical"]:
                        raise NoVerdict("S4: slot differs, but the solo F was read from its prompt, not continued: "
                                        "the difference can be read-vs-window K, not the position tables")
                if a.expect_differ:
                    tgt = targets[s]
                    hit = next(r for r in rows if r["slot"] == tgt)
                    # a token that differs where both have one: a slot that only ended sooner or later is not the bug
                    differs = not hit["identical"] and hit["first_diff"] < min(len(hit["batch"]), len(hit["solo"]))
                    entry["target"] = tgt
                    entry["target_differs"] = differs
                    print(f"  {s}: target slot {tgt} ({hit['request']}) "
                          + ("DIFFERS (the scene tells the bug apart)" if differs else
                             "does NOT differ in a token: this scene does not tell the bug apart"), flush=True)
                    if not differs:
                        rc = max(rc, 2)
                elif not all(r["identical"] for r in rows):
                    rc = max(rc, 2)
            except NoVerdict as e:
                print(f"  NO VERDICT: {e}", flush=True)
                entry["no_verdict"] = str(e)
                rc = max(rc, 1)
    except SetupError as e:
        print(f"SETUP FAILURE: {e}", flush=True)
        results["setup_error"] = str(e)
        rc = max(rc, 1)
    finally:
        try:
            eng.send("QUIT")
            eng.p.wait(timeout=180)   # the next run needs the GPUs back
        except Exception:
            eng.p.kill()
            eng.p.wait(timeout=60)
        results["exit"] = rc
        (out_dir / "results.json").write_text(json.dumps(results, indent=1))
    # what the engine said about the position tables (shown, not judged: the wording is the engine's)
    try:
        said = [l.rstrip() for l in open(eng.log_path, errors="replace")
                if "position table" in l or "STRATA_SLOT_POSITIONS" in l or "batch_slots" in l]
        for l in said[:6]:
            print("  engine log: " + l[:200])
    except OSError:
        pass
    verdict = {0: "PASS" if not a.expect_differ else "PASS (every target slot differs: the scenes tell the bug apart)",
               1: "SETUP FAILURE (no verdict)",
               2: "FAIL" if not a.expect_differ else "FAIL (a target slot is identical: that scene does not tell the bug apart)"}
    print(f"{verdict[rc]} - results {out_dir / 'results.json'}, engine log {eng.log_path}")
    return rc


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SetupError as e:
        print(f"SETUP FAILURE: {e}")
        sys.exit(1)
