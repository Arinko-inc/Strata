"""Keep long prompt heads in the idle batch slots (Arinko-inc patch, 2026-11-25).

A remote stage's main process holds ONE conversation branch in its main session: a prompt that does not continue it
(a 13-token probe, another client's system prompt) drops every checkpoint of the branch, root and pin included
(src/program/generate.cpp, `off_path` and `session_from0`).  The batch slots are different: each idle slot keeps its
own sessions and checkpoints on every stage, and a later prompt that starts with what a slot holds is copied back
from it (`slot_source`, `copy_from_slot`) - while the slot keeps its copy.  A request that ran alone never went
through a slot (the solo path, GEN), so its head lived only in the main session and the next unrelated request
erased it: measured on the 3-stage mesh 2026-11-25, one 13-token request between two requests with the same
~27K-token system prompt made the second read all of it again (9 s), while a head that had gone through a slot
(4 requests at once) survived another head in between (0.3 s).

This module decides, without the engine's help:
  - `KeepHeads.decide`: a request that is alone and whose reusable head (the engine's root / turn / pin checkpoints)
    is at least STRATA_KEEP_HEADS_MIN tokens longer than what any idle slot already serves of it goes through a
    batch slot (BGEN) instead of the solo path.  Alone in its slot it goes back to the solo path after its first
    token (StrataEngine._may_go_solo, the existing #857 hand-back), and the slot stays a cache of its prompt.
  - `KeepHeads.pick`: which free slot a BGEN overwrites - the slots whose overwrite would lose a head of at least
    STRATA_KEEP_HEADS_MIN tokens that no other slot (and not the new prompt itself) keeps come last.
  - `KeepHeads.check_reuse`: a solo request that a slot was expected to serve, but whose DONE says the engine reused
    less, makes the server forget that slot's tokens (the engine dropped them: a remote link made again, #879, a
    checkpoint the worker could not keep).
  - `KeepHeads.held_after_admission`: a BGEN that did not continue in its slot (max_tokens 1, an answer that ended
    at its first token, a request cancelled while its prompt was read) left the slot as it was in the engine
    (generate.cpp: `cont` false skips copy_to_slot), so the server keeps what it knew the slot held.

Opt-in: STRATA_KEEP_HEADS=1 in the server's environment (anything else: off, the default path unchanged).
STRATA_KEEP_HEADS_MIN=<tokens> (default 8192).  It acts only where the idle slots are caches and a slot request can
go back to the solo path: no pad rows (StrataEngine.slot_gs == 0; a remote stage's main says INFO batch_pad=0),
INFO slot_cache=1, no conversation cache, STRATA_PARALLEL_SOLO not 0 and STRATA_SLOT_CKPT_MOVE off.  Requests with
pictures are never sent to a slot for this (the engine keeps no slot cache of them).

The checkpoints the server assumes a slot holds after an admission that continued (the engine's own rules,
generate.cpp ~18113-18170 and ~19733-19815): the root - the first <|im_start|> after token 0 when it sits at
ROOT_MIN tokens or more (--prompt-cache-root, default 2048) - the turn - the last <|im_start|> - unless the request
had ckpt=0, the request's pin=N unless ckpt=0, and the slot's whole stream (prompt + the tokens it decoded).
"""
from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass

from serve.auto_prefix import lcp  # noqa: F401  (the common start of two id sequences; C-speed slice compares)

DEFAULT_MIN = 8192     # tokens of head a slot must keep (or would lose) before this module acts
ROOT_MIN = 2048        # the engine's --prompt-cache-root default: a shorter system prompt gets no root checkpoint
REUSE_SLACK = 64       # check_reuse: a DONE that reused at most this many tokens less than expected still counts
SAY_AGAIN_S = 2.0      # the same log line is not printed again within this many seconds (pick_slot is polled)


@dataclass
class SlotMeta:
    """What the server knows of the admission that last wrote a slot (besides StrataEngine.slot_held)."""
    pin: int = 0         # the pin=N it carried (0: none) - the engine took a pinned checkpoint there
    ckpt: bool = True    # False: it had ckpt=0 (no turn and no pin checkpoint; the root still)
    img: bool = False    # it had pictures: the engine never keeps such a slot as a cache (generate.cpp 19732, 15175)
    cvec: bool = True    # its experimental_speed_projection (request_cvec): with a control vector loaded, the engine
                         # resumes a prompt only from a slot of the same setting (generate.cpp ~16655, sl.cvec)


@dataclass
class Decision:
    """KeepHeads.decide's answer for a request that is alone."""
    to_slot: bool        # True: send it through a batch slot (BGEN) although it is alone
    slot: int | None     # to_slot False: the idle slot expected to serve `served` tokens (None: none does)
    served: int          # the tokens of this prompt the best idle slot serves (as far as the server knows)
    head: int            # the prompt's reusable head (root / turn / pin)
    reason: str          # one line for the log ("[strata] keep heads: " + reason)


def env_on(env=None) -> bool:
    """STRATA_KEEP_HEADS: exactly "1" (after strip) turns the feature on; unset or anything else is off."""
    env = os.environ if env is None else env
    return str(env.get("STRATA_KEEP_HEADS", "0")).strip() == "1"


def env_min(env=None) -> int:
    """STRATA_KEEP_HEADS_MIN: a positive integer; unset, not a number, 0 or negative -> DEFAULT_MIN."""
    env = os.environ if env is None else env
    try:
        n = int(str(env.get("STRATA_KEEP_HEADS_MIN", "")).strip())
    except ValueError:
        return DEFAULT_MIN
    return n if n > 0 else DEFAULT_MIN


def request_pin(sampling: dict) -> int:
    """The pin=N the request carries: sampling["strata_prefix"]["tokens"] when it is a positive int (not a bool), as
    StrataEngine.sampling_keys sends it; 0 otherwise (absent, unresolved, a request with ckpt=0 still returns it)."""
    prefix = sampling.get("strata_prefix") if isinstance(sampling, dict) else None
    if isinstance(prefix, dict):
        pn = prefix.get("tokens")
        if isinstance(pn, int) and not isinstance(pn, bool) and pn > 0:
            return pn
    return 0


def request_cvec(sampling: dict) -> bool:
    """The request's projection setting as StrataEngine.projection_key sends it: sampling["experimental_speed_projection"]
    when it is a bool, else True (the engine's cvec=1 when absent)."""
    on = sampling.get("experimental_speed_projection") if isinstance(sampling, dict) else None
    return on if isinstance(on, bool) else True


def cvec_loaded(info: dict) -> bool:
    """Whether the engine runs with a control vector (INFO cvec=..., as Service's history reads it): only then does
    the projection setting split the slots (the engine's want_cvec is true for every request otherwise)."""
    return str((info or {}).get("cvec", 0)) not in ("0", "", "None")


def request_ckpt(sampling: dict) -> bool:
    """False when the request asked for no checkpoint (sampling["strata_checkpoint"] is False: the engine's ckpt=0)."""
    return not (isinstance(sampling, dict) and sampling.get("strata_checkpoint") is False)


def turn_points(ids, turn_id: int, root_min: int = ROOT_MIN) -> tuple[int, int]:
    """(root, last_turn) of a token sequence, as the engine places those checkpoints (0: none).
    last_turn: the index of the last `turn_id` at index >= 1.  root: the index of the first `turn_id` at index >= 1,
    when it is below last_turn and at least root_min (generate.cpp 18131-18139).  A checkpoint at index i covers
    ids[:i]."""
    if not isinstance(ids, list):
        ids = list(ids)
    try:
        first = ids.index(turn_id, 1)                    # C-speed scans: no Python loop over a 260K-token prompt
    except ValueError:
        return 0, 0
    last = len(ids) - 1 - ids[::-1].index(turn_id)       # >= first >= 1
    return (first if first < last and first >= root_min else 0), last


def head_of(prompt, pin: int, ckpt: bool, turn_id: int, root_min: int = ROOT_MIN) -> int:
    """The prompt's reusable head: the deepest checkpoint the engine would leave that a LATER request can resume
    from - max(root, last_turn if ckpt, pin if ckpt and 0 < pin < len(prompt) - 1).  0: none."""
    root, last = turn_points(prompt, turn_id, root_min)
    head = root
    if ckpt:
        head = max(head, last)
        if 0 < pin < len(prompt) - 1:
            head = max(head, pin)
    return head


def served_by(prompt, held, meta: SlotMeta, turn_id: int, root_min: int = ROOT_MIN) -> int:
    """How many tokens of `prompt` a slot that holds `held` (StrataEngine.slot_held) serves, as far as the server
    knows: the largest of the slot's checkpoints - its whole stream when it is a start of the prompt, its root, its
    turn (when meta.ckpt), meta.pin (when meta.ckpt) - that is a start of the prompt and shorter than it.
    0 for an empty `held` or meta.img."""
    if not len(held) or meta.img:
        return 0
    root, last = turn_points(held, turn_id, root_min)
    cands = [len(held), root]
    if meta.ckpt:
        cands.append(last)
        if 0 < meta.pin < len(held):
            cands.append(meta.pin)
    n = lcp(prompt, held)
    return max((c for c in cands if 0 < c < len(prompt) and c <= n), default=0)


def value_of(held, meta: SlotMeta, turn_id: int, root_min: int = ROOT_MIN) -> int:
    """What a slot is worth keeping: the deepest of its root / turn (when meta.ckpt) / pin (when meta.ckpt).
    0 for an empty `held` or meta.img."""
    if not len(held) or meta.img:
        return 0
    root, last = turn_points(held, turn_id, root_min)
    return max(root, last if meta.ckpt else 0, meta.pin if meta.ckpt else 0)


def loss_of(b: int, prompt, slot_held, metas, turn_id: int, root_min: int = ROOT_MIN, images: bool = False) -> int:
    """The tokens of head an admission of `prompt` into slot b would lose for good: value_of(slot b) minus the most
    of it that survives - what the new prompt shares with it (served_by(prompt, slot b): the admission copies those
    checkpoints back in) or what another slot c != b serves of slot b's stream (served_by(slot_held[b], slot c)).
    `images`: the new prompt has pictures - the engine neither copies a slot's checkpoints into such a read
    (generate.cpp ~16647: a slot source needs req_imgs empty) nor keeps the slot as a cache after it, so what it
    shares with slot b does not survive.  Never below 0."""
    held = slot_held[b]
    value = value_of(held, metas[b], turn_id, root_min)
    if value <= 0:
        return 0
    survives = 0 if images else served_by(prompt, held, metas[b], turn_id, root_min)
    for c in range(len(slot_held)):
        if c != b and survives < value:
            survives = max(survives, served_by(held, slot_held[c], metas[c], turn_id, root_min))
    return max(0, value - survives)


class KeepHeads:
    """One per engine process (StrataEngine sets it up with the slot tables, so a restart starts it again empty)."""

    def __init__(self, batch: int, env=None, turn_id: int = 248045, root_min: int = ROOT_MIN):
        env = os.environ if env is None else env
        self.enabled = env_on(env)
        self.min_tokens = env_min(env) if self.enabled else DEFAULT_MIN
        self.turn_id = int(turn_id)
        self.root_min = int(root_min)
        self.parallel_solo = str(env.get("STRATA_PARALLEL_SOLO", "")) != "0"
        move = env.get("STRATA_SLOT_CKPT_MOVE")
        self.ckpt_move = bool(move) and str(move)[0] != "0"          # the engine's slot_ckpt_move_env()
        self.metas: list[SlotMeta] = [SlotMeta() for _ in range(max(0, int(batch)))]
        self._said: dict[str, float] = {}
        self._lock = threading.Lock()

    def usable(self, info: dict, slot_gs: int) -> bool:
        """Whether this engine fits the feature (enabled aside): idle slots are caches (slot_gs == 0), a slot request
        can go back to the solo path (INFO slot_cache=1 and STRATA_PARALLEL_SOLO not 0), no conversation cache
        (INFO conversation_cache_mib > 0 with conversation_cache_slots != 0, as Service.conversation_cache_on) and
        STRATA_SLOT_CKPT_MOVE off (a give-back would move the slot's checkpoints out of it)."""
        info = info or {}
        if slot_gs != 0 or not info.get("slot_cache") or not self.parallel_solo or self.ckpt_move:
            return False
        mib, slots = info.get("conversation_cache_mib"), info.get("conversation_cache_slots")
        return not (isinstance(mib, int) and mib > 0 and slots != 0)

    def applies(self, info: dict, slot_gs: int) -> bool:
        """enabled and usable(): pick_slot uses pick() only then, as decide() acts only then."""
        return self.enabled and self.usable(info, slot_gs)

    def meta(self, b: int) -> SlotMeta:
        """The record of slot b (a default one for a slot this object does not know)."""
        return self.metas[b] if 0 <= b < len(self.metas) else SlotMeta()

    def decide(self, prompt, sampling: dict, embeddings, slot_held, slot_busy, info: dict,
               slot_gs: int) -> Decision | None:
        """For a request that is ALONE on its first leg (no token out yet), with max_tokens > 1: None when the feature
        does not apply (off, not usable(), pictures); else a Decision.  to_slot is True when
        head_of(prompt) - max(served_by(prompt, slot b) over the idle slots) >= min_tokens."""
        if not self.enabled or embeddings or not self.usable(info, slot_gs) or len(prompt) < self.min_tokens:
            return None
        head = head_of(prompt, request_pin(sampling), request_ckpt(sampling), self.turn_id, self.root_min)
        served, best = 0, None
        split = cvec_loaded(info)
        want = request_cvec(sampling)
        for b in range(len(slot_held)):
            if slot_busy[b] or not len(slot_held[b]):
                continue
            if split and self.meta(b).cvec != want:      # the engine skips it for this request (and keeps it)
                continue
            n = served_by(prompt, slot_held[b], self.meta(b), self.turn_id, self.root_min)
            if n > served:
                served, best = n, b
        if head - served >= self.min_tokens:
            why = (f"a {len(prompt)}-token prompt alone goes through a batch slot "
                   f"(head {head} tokens, the idle slots serve {served})")
            return Decision(True, best, served, head, why)
        why = (f"a {len(prompt)}-token prompt alone stays on the solo path "
               f"(head {head} tokens, the idle slots serve {served})")
        return Decision(False, best, served, head, why)

    def check_reuse(self, decision: Decision | None, reused) -> int | None:
        """After the solo leg's DONE: the slot to forget (its slot_held set to []) when `decision` expected slot b to
        serve `served` tokens and the engine reused fewer than served - REUSE_SLACK; None otherwise (no decision,
        to_slot, no slot, reused not an int)."""
        if decision is None or decision.to_slot or decision.slot is None:
            return None
        if not isinstance(reused, int) or isinstance(reused, bool):
            return None
        if reused >= decision.served - REUSE_SLACK:
            return None
        if 0 <= decision.slot < len(self.metas):
            self.metas[decision.slot] = SlotMeta()
        return decision.slot

    def note_admitted(self, slot: int, sampling: dict, embeddings) -> None:
        """An admission continued in `slot` (BADM <slot> 1): remember its pin, ckpt and pictures (SlotMeta)."""
        if 0 <= slot < len(self.metas):
            self.metas[slot] = SlotMeta(pin=request_pin(sampling), ckpt=request_ckpt(sampling), img=bool(embeddings),
                                        cvec=request_cvec(sampling))

    def held_after_admission(self, prev_held, cont: bool, yielded: bool, failed: bool, erred: bool):
        """What slot_held should be after an admission: `prev_held` (what it held before the BGEN) when the admission
        did not continue in the slot and left it untouched in the engine (not cont, not yielded, not failed, no ERR);
        None (leave the server's value as it is) otherwise."""
        if cont or yielded or failed or erred:
            return None
        return list(prev_held or [])

    def after_drained_badm(self, slot: int, line, prev_held, sampling: dict, embeddings, asked: bool, done,
                           restore: bool = True):
        """An admission whose consumer left before its BADM was read (an answer that ended at its first token):
        generate_batched's `finally` drained the control lines up to `line`.  The same as the normal path does after
        BADM: `BADM <slot> 1` -> note_admitted, `BADM <slot> 0` -> held_after_admission.  The drain does not see a
        YIELDED line, so `asked` (a BYIELD was sent for this read) counts as yielded; `done` is the DONE line the drain
        read for THIS request (not the shared self.last, which another slot's BDONE can rewrite): one that said
        "nonfinite", or none at all, keeps the record empty.  `restore` False (not applies(): no slot is a cache)
        never gives the record back.  Returns what slot_held should be (None: leave it)."""
        f = line.split() if isinstance(line, str) else []
        if f[:1] != ["BADM"] or len(f) < 3:
            return None                                  # an ERR, or the engine ended: nothing is known
        if f[2] == "1":
            self.note_admitted(slot, sampling, embeddings)
            return None
        d = done.split() if isinstance(done, str) else []
        if not restore or len(d) < 6 or d[5] == "nonfinite":
            return None
        return self.held_after_admission(prev_held, False, bool(asked), f[3:4] == ["failed"], False)

    def pick(self, prompt, free: list[int], slot_held, base_key, images: bool = False) -> tuple[int, str | None]:
        """The free slot for an admission of `prompt` (pick_slot's last step, `free` in pick_slot's order) and a log
        line when this rule changed the pick (None: the same slot as base_key alone).  Order: slots whose loss_of()
        is below min_tokens first, by base_key (pick_slot's own order: the less busy group, empty, least recently
        used); then the others by their loss, then base_key.  `images`: the prompt has pictures (loss_of)."""
        held = [h if isinstance(h, list) else list(h) for h in slot_held]
        metas = [self.meta(b) for b in range(len(held))]
        loss = {b: loss_of(b, prompt, held, metas, self.turn_id, self.root_min, images) for b in free}

        def key(b):
            return (0, base_key(b)) if loss[b] < self.min_tokens else (1, loss[b], base_key(b))
        chosen, plain = min(free, key=key), min(free, key=base_key)
        if chosen == plain:
            return chosen, None
        return chosen, (f"slot {chosen} for a {len(prompt)}-token prompt instead of slot {plain}, whose overwrite "
                        f"would lose a {loss[plain]}-token head no other slot keeps")

    def say(self, line: str) -> None:
        """Print "[strata] keep heads: <line>" unless the same line was printed in the last SAY_AGAIN_S seconds."""
        now = time.monotonic()
        with self._lock:
            if now - self._said.get(line, -1e9) < SAY_AGAIN_S:
                return
            self._said = {k: v for k, v in self._said.items() if now - v < SAY_AGAIN_S}
            self._said[line] = now
        print(f"[strata] keep heads: {line}", flush=True)
