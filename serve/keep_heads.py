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
    raise NotImplementedError


def request_pin(sampling: dict) -> int:
    """The pin=N the request carries: sampling["strata_prefix"]["tokens"] when it is a positive int (not a bool), as
    StrataEngine.sampling_keys sends it; 0 otherwise (absent, unresolved, a request with ckpt=0 still returns it)."""
    raise NotImplementedError


def request_ckpt(sampling: dict) -> bool:
    """False when the request asked for no checkpoint (sampling["strata_checkpoint"] is False: the engine's ckpt=0)."""
    raise NotImplementedError


def turn_points(ids, turn_id: int, root_min: int = ROOT_MIN) -> tuple[int, int]:
    """(root, last_turn) of a token sequence, as the engine places those checkpoints (0: none).
    last_turn: the index of the last `turn_id` at index >= 1.  root: the index of the first `turn_id` at index >= 1,
    when it is below last_turn and at least root_min (generate.cpp 18131-18139).  A checkpoint at index i covers
    ids[:i]."""
    raise NotImplementedError


def head_of(prompt, pin: int, ckpt: bool, turn_id: int, root_min: int = ROOT_MIN) -> int:
    """The prompt's reusable head: the deepest checkpoint the engine would leave that a LATER request can resume
    from - max(root, last_turn if ckpt, pin if ckpt and 0 < pin < len(prompt) - 1).  0: none."""
    raise NotImplementedError


def served_by(prompt, held, meta: SlotMeta, turn_id: int, root_min: int = ROOT_MIN) -> int:
    """How many tokens of `prompt` a slot that holds `held` (StrataEngine.slot_held) serves, as far as the server
    knows: the largest of the slot's checkpoints - its whole stream when it is a start of the prompt, its root, its
    turn (when meta.ckpt), meta.pin (when meta.ckpt) - that is a start of the prompt and shorter than it.
    0 for an empty `held` or meta.img."""
    raise NotImplementedError


def value_of(held, meta: SlotMeta, turn_id: int, root_min: int = ROOT_MIN) -> int:
    """What a slot is worth keeping: the deepest of its root / turn (when meta.ckpt) / pin (when meta.ckpt).
    0 for an empty `held` or meta.img."""
    raise NotImplementedError


def loss_of(b: int, prompt, slot_held, metas, turn_id: int, root_min: int = ROOT_MIN) -> int:
    """The tokens of head an admission of `prompt` into slot b would lose for good: value_of(slot b) minus the most
    of it that survives - what the new prompt shares with it (served_by(prompt, slot b): the admission copies those
    checkpoints back in) or what another slot c != b serves of slot b's stream (served_by(slot_held[b], slot c)).
    Never below 0."""
    raise NotImplementedError


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
        raise NotImplementedError

    def decide(self, prompt, sampling: dict, embeddings, slot_held, slot_busy, info: dict,
               slot_gs: int) -> Decision | None:
        """For a request that is ALONE on its first leg (no token out yet), with max_tokens > 1: None when the feature
        does not apply (off, not usable(), pictures); else a Decision.  to_slot is True when
        head_of(prompt) - max(served_by(prompt, slot b) over the idle slots) >= min_tokens."""
        raise NotImplementedError

    def check_reuse(self, decision: Decision | None, reused) -> int | None:
        """After the solo leg's DONE: the slot to forget (its slot_held set to []) when `decision` expected slot b to
        serve `served` tokens and the engine reused fewer than served - REUSE_SLACK; None otherwise (no decision,
        to_slot, no slot, reused not an int)."""
        raise NotImplementedError

    def note_admitted(self, slot: int, sampling: dict, embeddings) -> None:
        """An admission continued in `slot` (BADM <slot> 1): remember its pin, ckpt and pictures (SlotMeta)."""
        raise NotImplementedError

    def held_after_admission(self, prev_held, cont: bool, yielded: bool, failed: bool, erred: bool):
        """What slot_held should be after an admission: `prev_held` (what it held before the BGEN) when the admission
        did not continue in the slot and left it untouched in the engine (not cont, not yielded, not failed, no ERR);
        None (leave the server's value as it is) otherwise."""
        raise NotImplementedError

    def pick(self, prompt, free: list[int], slot_held, base_key) -> tuple[int, str | None]:
        """The free slot for an admission of `prompt` (pick_slot's last step, `free` in pick_slot's order) and a log
        line when this rule changed the pick (None: the same slot as base_key alone).  Order: slots whose loss_of()
        is below min_tokens first, by base_key (pick_slot's own order: the less busy group, empty, least recently
        used); then the others by their loss, then base_key."""
        raise NotImplementedError

    def say(self, line: str) -> None:
        """Print "[strata] keep heads: <line>" unless the same line was printed in the last SAY_AGAIN_S seconds."""
        now = time.monotonic()
        with self._lock:
            if now - self._said.get(line, -1e9) < SAY_AGAIN_S:
                return
            self._said = {k: v for k, v in self._said.items() if now - v < SAY_AGAIN_S}
            self._said[line] = now
        print(f"[strata] keep heads: {line}", flush=True)
