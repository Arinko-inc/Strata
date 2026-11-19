#!/usr/bin/env python3
"""Small load generator for the OpenAI-compatible server: C clients at once, each a different prompt.

    python3 tools/serve_load.py http://127.0.0.1:8080 --clients 1,2,4,8 --rounds 3 --max-tokens 256

For every client count it sends C streamed chat requests at the same moment (each prompt starts with its own nonce, so
the prompt cache cannot help), and reports per round and as medians: total tokens/s (all generated tokens over the
wall time of the burst), per-request tokens/s, time to first token (p50 / p95 over all requests of that count), and
the request latency (p50 / p95).  With --save DIR it writes every answer, so a later run can be compared byte for
byte with `--compare DIR` (greedy answers of a request must not depend on who else ran).

Exit status: 0 only when every request really succeeded.  It is 1 when any request failed (HTTP/connection error, an
error event inside the stream), when fewer or more answers came back than were sent, when a stream did not end with
`data: [DONE]`, when the finish_reason is missing or not one of FINISH_OK, when the usage is missing or says 0 output
tokens, when --compare was given and the reference file of an answer is missing, or when an answer differs from its
reference.  The last two lines of the output are `result: ...` (how many requests were fine) and `failures: ...` (the
breakdown by cause).

Standard library only.  The prompts are fixed texts, so two runs of the tool send the same requests.
"""
import argparse, json, os, statistics, sys, threading, time, urllib.request

TOPICS = [
    "the history of lighthouses and how their lenses worked",
    "how a mechanical watch keeps time",
    "why bridges are built as arches and as suspension spans",
    "the water cycle and what a drought does to a river valley",
    "how bread rises, from yeast to crust",
    "the design of a medieval castle and why walls were stepped",
    "how a transistor switches, explained for a curious teenager",
    "the life of a honey bee colony through one year",
]


# the finish_reason values a healthy answer ends with ("cancel" is sent as "stop" by the server)
FINISH_OK = ("stop", "length", "tool_calls")


def problems(r):
    """The ways a request result is not a success (empty list = fine).  Names are the keys of the `failures:` line."""
    p = []
    if r.get("err"):
        p.append("request_error")
    if r.get("stream_error"):
        p.append("stream_error")
    if not r.get("done"):
        p.append("no_done")
    fin = r.get("finish")
    if fin is None:
        p.append("no_finish")
    elif fin not in FINISH_OK:
        p.append("bad_finish")
    u = r.get("usage_tokens")
    if u is None:
        p.append("no_usage")
    elif u <= 0:
        p.append("zero_tokens")
    return p


def prompt_for(i: int, words: int, nonce: str = "") -> str:
    topic = TOPICS[i % len(TOPICS)]
    return (f"[request {i}{nonce}] Write a clear, well organised essay of about {words} words on {topic}. "
            f"Use plain prose, no lists.")


def one(url, key, model, i, words, max_tokens, res, extra, think, nonce=""):
    body = {"model": model, "stream": True, "max_tokens": max_tokens, "temperature": 0,
            "messages": [{"role": "user", "content": prompt_for(i, words, nonce)}]}
    if not think:
        body["reasoning_effort"] = "none"
        body["chat_template_kwargs"] = {"enable_thinking": False}
    body.update(extra)
    req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json",
                                          **({"Authorization": "Bearer " + key} if key else {})})
    t0 = time.perf_counter()
    first, n, text = None, 0, []
    err = None
    done, finish, usage_tokens, stream_error = False, None, None, None
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                d = line[5:].strip()
                if d == "[DONE]":
                    done = True
                    break
                try:
                    j = json.loads(d)
                except ValueError:
                    continue
                if not isinstance(j, dict):
                    continue
                if j.get("error"):
                    stream_error = str(j["error"])[:200]
                for ch in j.get("choices") or []:
                    if ch.get("finish_reason") is not None:
                        finish = ch["finish_reason"]
                    dl = ch.get("delta") or {}
                    piece = (dl.get("content") or "") + (dl.get("reasoning_content") or "")
                    if piece:
                        if first is None:
                            first = time.perf_counter() - t0
                        n += 1                      # one SSE chunk = one token (the server flushes per token)
                        text.append(piece)
                if j.get("usage"):
                    usage_tokens = j["usage"].get("completion_tokens", usage_tokens)
                    if usage_tokens is not None:
                        n = usage_tokens
    except Exception as e:                          # noqa: BLE001
        err = repr(e)
    res[i] = {"first": first, "tokens": n, "total": time.perf_counter() - t0, "text": "".join(text), "err": err,
              "done": done, "finish": finish, "usage_tokens": usage_tokens, "stream_error": stream_error}


def pct(v, p):
    v = sorted(v)
    if not v:
        return float("nan")
    k = (len(v) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("--clients", default="1,2,4,8")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--words", type=int, default=800, help="the essay length asked for (longer = longer prompt? no: the answer)")
    ap.add_argument("--model", default="strata")
    ap.add_argument("--key", default=os.environ.get("STRATA_KEY", ""))
    ap.add_argument("--nonce", default=str(int(time.time())), help="prefix of the per-request nonce (every round reads fresh prompts); '' = identical prompts every round; fix it to compare two runs")
    ap.add_argument("--think", action="store_true")
    ap.add_argument("--stagger", type=float, default=0.0, help="seconds between the starts of the clients of a burst")
    ap.add_argument("--save", help="write every answer (clients/round/request) here")
    ap.add_argument("--compare", help="compare every answer with the files a --save run wrote")
    ap.add_argument("--json", help="write the table as JSON")
    a = ap.parse_args()
    extra = {}
    out = []
    bad = 0
    sent = 0                                        # requests started
    fails = {}                                      # failure name -> number of requests (or files, for the counts)
    failed_reqs = 0
    for c in [int(x) for x in a.clients.split(",")]:
        rows = []
        for rd in range(a.rounds):
            res = {}
            th = []
            t0 = time.perf_counter()
            for i in range(c):
                t = threading.Thread(target=one, args=(a.url, a.key, a.model, i, a.words, a.max_tokens, res, extra, a.think, f" {a.nonce}-{c}-{rd}" if a.nonce else ""))
                t.start()
                th.append(t)
                if a.stagger:
                    time.sleep(a.stagger)
            for t in th:
                t.join()
            wall = time.perf_counter() - t0
            sent += c
            missing = [i for i in range(c) if i not in res]
            if missing or len(res) != c:
                fails["count_mismatch"] = fails.get("count_mismatch", 0) + max(len(missing), abs(len(res) - c))
                print(f"  COUNT MISMATCH c={c} round={rd}: sent {c}, got {len(res)} results (missing {missing})", flush=True)
            for i in range(c):
                r = res.get(i)
                if r is None:                       # a thread that died before it stored anything (counted above)
                    failed_reqs += 1
                    continue
                pr = problems(r)
                for name in pr:
                    fails[name] = fails.get(name, 0) + 1
                if pr:
                    failed_reqs += 1
                    print(f"  FAILED c={c} round={rd} req={i}: {', '.join(pr)}", flush=True)
            toks = sum(r["tokens"] for r in res.values())
            errs = [r["err"] or ("stream error: " + r["stream_error"]) for r in res.values() if r["err"] or r["stream_error"]]
            firsts = [r["first"] for r in res.values() if r["first"] is not None]
            tot = [r["total"] for r in res.values()]
            per = [r["tokens"] / max(r["total"] - (r["first"] or 0), 1e-6) for r in res.values()]
            rows.append({"wall": wall, "toks": toks, "tps": toks / wall, "first": firsts, "lat": tot, "per": per, "err": errs})
            if a.save:
                d = os.path.join(a.save, f"c{c}", f"r{rd}")
                os.makedirs(d, exist_ok=True)
                for i, r in res.items():
                    open(os.path.join(d, f"{i}.txt"), "w", encoding="utf-8").write(r["text"])
            if a.compare:
                for i, r in res.items():
                    p = os.path.join(a.compare, f"c{c}", f"r{rd}", f"{i}.txt")
                    if not os.path.exists(p):
                        print(f"  MISSING REFERENCE c={c} round={rd} req={i}: {p}", flush=True)
                        fails["compare_missing"] = fails.get("compare_missing", 0) + 1
                    else:
                        ref = open(p, encoding="utf-8").read()
                        if ref != r["text"]:
                            n = next((k for k, (x, y) in enumerate(zip(ref, r["text"])) if x != y), min(len(ref), len(r["text"])))
                            print(f"  DIFF c={c} round={rd} req={i}: first difference at char {n}")
                            bad += 1
            print(f"c={c} round={rd}: {toks} tokens in {wall:.1f}s = {toks / wall:.1f} tok/s total"
                  + (f"  ERRORS {errs[:2]}" if errs else ""), flush=True)
        allf = [x for r in rows for x in r["first"]]
        alll = [x for r in rows for x in r["lat"]]
        allp = [x for r in rows for x in r["per"]]
        s = {"clients": c, "total_tps": statistics.median(r["tps"] for r in rows),
             "per_req_tps": statistics.median(allp) if allp else float("nan"),
             "first_p50": pct(allf, .5), "first_p95": pct(allf, .95),
             "lat_p50": pct(alll, .5), "lat_p95": pct(alll, .95), "errors": sum(len(r["err"]) for r in rows)}
        out.append(s)
    print("\nclients  total tok/s  per-req tok/s  first p50/p95 (s)  latency p50/p95 (s)  errors")
    for s in out:
        print(f"{s['clients']:>7}  {s['total_tps']:>11.1f}  {s['per_req_tps']:>13.1f}  "
              f"{s['first_p50']:>7.2f}/{s['first_p95']:<7.2f}  {s['lat_p50']:>8.2f}/{s['lat_p95']:<8.2f}  {s['errors']}")
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)
    if a.compare:
        miss = fails.get("compare_missing", 0)
        print("compare:", "ALL IDENTICAL" if not bad and not miss else
              ", ".join(x for x in (f"{bad} DIFFERENT" if bad else "", f"{miss} REFERENCE FILES MISSING" if miss else "") if x))
    if bad:
        fails["diff"] = bad
    print(f"result: {sent - failed_reqs}/{sent} requests ok")
    print("failures: " + (", ".join(f"{k}={v}" for k, v in fails.items()) if fails else "none"))
    return 1 if (bad or fails or failed_reqs) else 0


if __name__ == "__main__":
    sys.exit(main())
