"""Automatic shared prefix inside one message (Arinko-inc patch, 2026-11-25).

The engine resumes a prompt only from a conversation checkpoint, and it takes checkpoints at message boundaries: the
end of the system prompt (the root), the last <|im_start|> (the turn) and, when a request asks for it, `pin=N`
(`"strata_prefix"`).  A client that puts a long shared document and its varying question in ONE user message gets
no checkpoint between the two, so every request reads the whole document again, one after another: measured on the
3-stage mesh, 4 such requests of ~40K tokens took 88 s and 8 took 183 s, against 12.5 s and 24.3 s with the document
in a system message or with `"strata_prefix"` marking its end (~/tools/llm-agent-bench, 2026-11-24/25).

This module finds that boundary without the client's help.  It keeps the last few prompts' ids; a new prompt that
shares a long start with one of them, AND whose shared start runs far past the last <|im_start|> inside it (so no
root or turn checkpoint can serve it), gets `pin` at that start, rounded down to a quantum so the requests that follow
ask for the same position.  The request that finds it reads the document once more and leaves a pinned checkpoint;
the requests after it resume there.

Off with STRATA_AUTO_PREFIX=0.  A request's own "strata_prefix" always wins; requests with images are skipped.
"""
from __future__ import annotations

import collections
import os
import threading
from array import array

DEFAULT_MIN = 4096     # shared tokens past the last turn marker before a pin is worth its extra cut (~0.3 s)
DEFAULT_KEEP = 8       # recent prompts compared against
QUANTUM = 256          # pin positions are rounded down to this, so neighbours of one document agree on one position
MARGIN = 16            # tokens kept clear of the divergence: the tokenizer may merge across it differently next time


def _int_env(env, name: str, default: int) -> int:
    try:
        v = int(env.get(name, default))
    except (TypeError, ValueError):
        return default
    return v if v > 0 else default


def lcp(a, b) -> int:
    """Length of the common start of two id sequences (arrays or lists of the same kind).  Slice comparisons run in
    C, so a 260K-token prompt costs a few milliseconds instead of a Python loop's tenth of a second."""
    n = min(len(a), len(b))
    if a[:n] == b[:n]:
        return n
    lo, hi = 0, n            # a[:lo] == b[:lo]; a[:hi] != b[:hi]
    while hi - lo > 64:
        mid = (lo + hi) // 2
        if a[lo:mid] == b[lo:mid]:
            lo = mid
        else:
            hi = mid
    while lo < hi and a[lo] == b[lo]:
        lo += 1
    return lo


class AutoPrefix:
    def __init__(self, turn_id: int, env=None):
        env = os.environ if env is None else env
        self.enabled = str(env.get("STRATA_AUTO_PREFIX", "1")).strip() != "0"
        self.min_tokens = _int_env(env, "STRATA_AUTO_PREFIX_MIN", DEFAULT_MIN)
        self.turn_id = int(turn_id)
        self._recent: collections.deque = collections.deque(maxlen=_int_env(env, "STRATA_AUTO_PREFIX_KEEP", DEFAULT_KEEP))
        self._lock = threading.Lock()

    def pick(self, ids) -> int:
        """The pin position for this prompt (0: none), and remember the prompt for the next ones."""
        if not self.enabled or len(ids) < self.min_tokens:
            return 0
        cur = array("q", ids)
        with self._lock:
            best = max((lcp(cur, r) for r in self._recent), default=0)
            self._recent.append(cur)
        # the pin must leave the prompt's tail to read (the engine ignores pin >= n - 1) and sit clear of the divergence
        pin = min(best - MARGIN, len(ids) - 2)
        pin -= pin % QUANTUM
        if pin < self.min_tokens:
            return 0
        # the last turn marker inside the shared start (not before the rounded pin: the rounding can step back over a
        # marker right at the divergence): the root / turn checkpoints already cover everything up to it
        last_turn = 0
        for i in range(min(best, len(cur)) - 1, -1, -1):
            if cur[i] == self.turn_id:
                last_turn = i
                break
        if pin - last_turn < self.min_tokens:
            return 0
        return pin
