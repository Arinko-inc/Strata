"""serve/test_keep_heads.py - long prompt heads kept in the idle batch slots (serve/keep_heads.py, STRATA_KEEP_HEADS=1).

    python3 -m unittest serve.test_keep_heads -v

The pure rules (no engine), pick_slot with the rule on and off, and the server path over the fake engine of
serve/test_parallel.py (FakeEngineCase: GEN / BGEN / BT / BDONE / BADM / BSTOP lines, `--log` of each command).
The cases are numbered as in strata-mesh IMPL-PLAN-2026-11-25-keep-heads.md section 5.3.
"""
from __future__ import annotations

import os
import random
import sys
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve import keep_heads  # noqa: E402
from serve.keep_heads import Decision, KeepHeads, SlotMeta  # noqa: E402
from serve.server import StrataEngine  # noqa: E402
from serve.test_parallel import FakeEngineCase  # noqa: E402

T = 248045                       # Qwen's <|im_start|>
ON = {"STRATA_KEEP_HEADS": "1", "STRATA_KEEP_HEADS_MIN": "1000"}
INFO = {"slot_cache": 1, "engine": "0.1.39"}       # the Qwen3.8 path's INFO (the fake engine says engine= too)


def sysp(n, seed=1):
    """A system prompt: <|im_start|> and n plain tokens; the next <|im_start|> lands at index n + 1 (the root)."""
    return [T] + [seed * 1_000_000 + i for i in range(n)]


def turn(q, n=5):
    """One message: <|im_start|> and n tokens that depend on q."""
    return [T] + [q * 1000 + i for i in range(n)]


def prompt(sys_n, *qs, seed=1):
    out = sysp(sys_n, seed)
    for q in qs:
        out += turn(q)
    return out


def held_of(p, n=3):
    """What a slot holds after it served prompt `p`: the prompt and n decoded tokens."""
    return list(p) + [7] * n


def kh_on(batch=4, **env):
    return KeepHeads(batch, env={**ON, **env}, turn_id=T)


class Env(unittest.TestCase):
    """STRATA_KEEP_HEADS / STRATA_KEEP_HEADS_MIN, and KeepHeads.usable()."""

    def test_01_switch(self):
        for v in (None, "0", "true", "yes", "", "2", "on"):
            env = {} if v is None else {"STRATA_KEEP_HEADS": v}
            with self.subTest(v=v):
                self.assertFalse(keep_heads.env_on(env))
                self.assertFalse(KeepHeads(4, env=env).enabled)
        for v in ("1", " 1 "):
            with self.subTest(v=v):
                self.assertTrue(keep_heads.env_on({"STRATA_KEEP_HEADS": v}))
                self.assertTrue(KeepHeads(4, env={"STRATA_KEEP_HEADS": v}).enabled)

    def test_02_min(self):
        for v in (None, "abc", "0", "-5", "", "1.5"):
            env = {} if v is None else {"STRATA_KEEP_HEADS_MIN": v}
            with self.subTest(v=v):
                self.assertEqual(keep_heads.env_min(env), 8192)
        self.assertEqual(keep_heads.env_min({"STRATA_KEEP_HEADS_MIN": "4096"}), 4096)
        self.assertEqual(keep_heads.env_min({"STRATA_KEEP_HEADS_MIN": " 64 "}), 64)
        self.assertEqual(KeepHeads(4, env={"STRATA_KEEP_HEADS": "1", "STRATA_KEEP_HEADS_MIN": "4096"}).min_tokens, 4096)
        self.assertEqual(KeepHeads(4, env={"STRATA_KEEP_HEADS": "1"}).min_tokens, 8192)

    def test_03_usable(self):
        kh = kh_on()
        self.assertTrue(kh.usable(INFO, 0))
        self.assertFalse(kh.usable(INFO, 2))                                    # pad rows: the idle slots are written
        self.assertFalse(kh.usable({}, 0))
        self.assertFalse(kh.usable({"slot_cache": 0}, 0))
        self.assertFalse(kh_on(STRATA_PARALLEL_SOLO="0").usable(INFO, 0))
        self.assertTrue(kh_on(STRATA_PARALLEL_SOLO="1").usable(INFO, 0))
        self.assertFalse(kh.usable({**INFO, "conversation_cache_mib": 512, "conversation_cache_slots": 4}, 0))
        self.assertTrue(kh.usable({**INFO, "conversation_cache_mib": 512, "conversation_cache_slots": 0}, 0))
        self.assertTrue(kh.usable({**INFO, "conversation_cache_mib": 0, "conversation_cache_slots": 4}, 0))
        self.assertFalse(kh_on(STRATA_SLOT_CKPT_MOVE="1").usable(INFO, 0))
        self.assertTrue(kh_on(STRATA_SLOT_CKPT_MOVE="0").usable(INFO, 0))

    def test_03b_usable_only_on_the_checked_engine(self):
        """Only the Qwen3.8 path without a control vector (review 2026-11-25 #3): the Qwen3.6 engine (src/qwen35,
        INFO batch_slots=N slot_cache=1 and nothing else: no engine=) resumes a read with pictures from a slot, and
        a loaded control vector splits the slots by projection."""
        kh = kh_on()
        self.assertFalse(kh.usable({"batch_slots": 4, "slot_cache": 1}, 0))    # Qwen3.6's INFO
        self.assertFalse(kh.usable({**INFO, "cvec": "speed.gguf"}, 0))
        self.assertTrue(kh.usable({**INFO, "cvec": "0"}, 0))
        self.assertTrue(kh.usable({**INFO, "cvec": 0}, 0))


class Rules(unittest.TestCase):
    """turn_points, head_of, served_by, value_of, loss_of, request_pin, request_ckpt."""

    def test_04_turn_points(self):
        self.assertEqual(keep_heads.turn_points([1, 2, 3], T), (0, 0))
        self.assertEqual(keep_heads.turn_points([], T), (0, 0))
        self.assertEqual(keep_heads.turn_points([T, 1, 2, 3], T), (0, 0))               # only index 0: it does not count
        p = sysp(3000) + turn(1) + turn(2)
        self.assertEqual(keep_heads.turn_points(p, T), (3001, len(p) - 6))
        self.assertEqual(p[len(p) - 6], T)
        p = sysp(100) + turn(1) + turn(2)                                               # the first below root_min
        self.assertEqual(keep_heads.turn_points(p, T), (0, len(p) - 6))
        self.assertEqual(keep_heads.turn_points(p, T, root_min=50), (101, len(p) - 6))
        self.assertEqual(keep_heads.turn_points([1, 2, T, 3], T, root_min=1), (0, 2))   # one T at index >= 1: no root
        self.assertEqual(keep_heads.turn_points([T, 1, 2, T, 3], T, root_min=1), (0, 3))

    def test_05_head_of(self):
        p = prompt(3000, 1, 2)                       # root 3001, last turn 3007, 3013 tokens
        self.assertEqual(len(p), 3013)
        self.assertEqual(keep_heads.head_of(p, 0, False, T), 3001)
        self.assertEqual(keep_heads.head_of(p, 0, True, T), 3007)
        self.assertEqual(keep_heads.head_of(p, 3010, True, T), 3010)                    # a deeper pin wins
        self.assertEqual(keep_heads.head_of(p, 3012, True, T), 3007)                    # pin >= len - 1: not counted
        self.assertEqual(keep_heads.head_of(p, 3011, True, T), 3011)
        self.assertEqual(keep_heads.head_of(p, 3010, False, T), 3001)                   # ckpt=0: root only
        self.assertEqual(keep_heads.head_of(p, 100, True, T), 3007)                     # a shallower pin does not matter
        self.assertEqual(keep_heads.head_of(prompt(100, 1, 2), 0, False, T), 0)

    def test_request_pin_and_ckpt(self):
        self.assertEqual(keep_heads.request_pin({"strata_prefix": {"tokens": 500}}), 500)
        for bad in ({}, {"strata_prefix": None}, {"strata_prefix": {"messages": 2}}, {"strata_prefix": {"tokens": True}},
                    {"strata_prefix": {"tokens": 0}}, {"strata_prefix": {"tokens": -3}},
                    {"strata_prefix": {"tokens": "5"}}, {"strata_prefix": 5}):
            with self.subTest(bad=bad):
                self.assertEqual(keep_heads.request_pin(bad), 0)
        self.assertEqual(keep_heads.request_pin(None), 0)
        self.assertTrue(keep_heads.request_ckpt({}))
        self.assertTrue(keep_heads.request_ckpt({"strata_checkpoint": True}))
        self.assertTrue(keep_heads.request_ckpt({"strata_checkpoint": 0}))              # only `false` asks for ckpt=0
        self.assertFalse(keep_heads.request_ckpt({"strata_checkpoint": False}))

    def test_06_served_by(self):
        held = held_of(prompt(3000, 1, 2))                                              # root 3001, turn 3007
        meta = SlotMeta()
        # (i) the held stream is a start of the prompt
        nxt = held + turn(3)
        self.assertEqual(keep_heads.served_by(nxt, held, meta, T), len(held))
        # (ii) the same system prompt, another question: the root
        other = prompt(3000, 8, 9)
        self.assertEqual(keep_heads.served_by(other, held, meta, T), 3001)
        # (iii) the same conversation's next turn (the reply is rendered again): the turn
        again = held[:3007] + [T, 4, 4, 4] + turn(3)
        self.assertEqual(keep_heads.served_by(again, held, meta, T), 3007)
        self.assertEqual(keep_heads.served_by(again, held, SlotMeta(ckpt=False), T), 3001)     # ckpt=0: root only
        # (iv) a pin counts only when the slot remembers it (and ckpt)
        body = sysp(3000) + [T] + list(range(5000, 10000))                              # one T after the system: no root
        h2 = held_of(body)
        want = body[:4500] + [1, 2, 3, 4]
        self.assertEqual(keep_heads.served_by(want, h2, SlotMeta(), T), 3001)           # its turn is all it has
        self.assertEqual(keep_heads.served_by(want, h2, SlotMeta(pin=4000), T), 4000)
        self.assertEqual(keep_heads.served_by(want, h2, SlotMeta(pin=4000, ckpt=False), T), 0)
        self.assertEqual(keep_heads.served_by(want, h2, SlotMeta(pin=4600), T), 3001)   # a pin past the common start
        # (v) pictures: the engine keeps no such slot
        self.assertEqual(keep_heads.served_by(nxt, held, SlotMeta(img=True), T), 0)
        # (vi) the very same prompt: only what is shorter than it
        same = prompt(3000, 1, 2)
        self.assertEqual(keep_heads.served_by(same, same, meta, T), 3007)
        self.assertEqual(keep_heads.served_by(held, held, meta, T), 3007)               # len(held) itself is not shorter
        # (vii) empty
        self.assertEqual(keep_heads.served_by(nxt, [], meta, T), 0)

    def test_07_value_and_loss(self):
        a = held_of(prompt(3000, 1, 2))
        self.assertEqual(keep_heads.value_of(a, SlotMeta(), T), 3007)
        self.assertEqual(keep_heads.value_of(a, SlotMeta(ckpt=False), T), 3001)
        self.assertEqual(keep_heads.value_of(a, SlotMeta(pin=3500), T), 3500)
        self.assertEqual(keep_heads.value_of(a, SlotMeta(pin=3500, ckpt=False), T), 3001)
        self.assertEqual(keep_heads.value_of(a, SlotMeta(img=True), T), 0)
        self.assertEqual(keep_heads.value_of([], SlotMeta(), T), 0)
        empty = [[] for _ in range(3)]
        metas = [SlotMeta(), SlotMeta(), SlotMeta()]
        # a head no other slot keeps and the new prompt does not share: lost whole
        held = [a, [], []]
        newp = prompt(3000, 8, 9, seed=2)
        self.assertEqual(keep_heads.loss_of(0, newp, held, metas, T), 3007)
        # the same head in another slot too: nothing is lost
        held = [a, list(a), []]
        self.assertEqual(keep_heads.loss_of(0, newp, held, metas, T), 0)
        self.assertEqual(keep_heads.loss_of(1, newp, held, metas, T), 0)
        # the new prompt has the same system prompt and first message: nothing is lost (it copies the turn back in)
        held = [a, [], []]
        self.assertEqual(keep_heads.loss_of(0, prompt(3000, 1, 2, 3), held, metas, T), 0)
        # the same system prompt, another question: only the question's turn is lost
        self.assertEqual(keep_heads.loss_of(0, prompt(3000, 8, 9), held, metas, T), 6)
        # pictures: a slot the engine does not keep is worth nothing
        self.assertEqual(keep_heads.loss_of(0, newp, [a, [], []], [SlotMeta(img=True), SlotMeta(), SlotMeta()], T), 0)
        # an empty slot
        self.assertEqual(keep_heads.loss_of(1, newp, [a, [], []], metas, T), 0)
        del empty


class Decide(unittest.TestCase):
    """KeepHeads.decide / check_reuse / note_admitted / held_after_admission."""

    def decide(self, kh, p, held, sampling=None, embeddings=None, busy=None, info=INFO, gs=0):
        return kh.decide(p, sampling or {}, embeddings, held, busy or [False] * len(held), info, gs)

    def test_08_none_when_it_does_not_apply(self):
        p = prompt(30000, 1, 2)
        held = [[], [], [], []]
        self.assertIsNotNone(self.decide(kh_on(), p, held))
        self.assertIsNone(self.decide(KeepHeads(4, env={}, turn_id=T), p, held))        # off
        self.assertIsNone(self.decide(kh_on(), p, held, embeddings="/tmp/emb"))         # pictures
        self.assertIsNone(self.decide(kh_on(), prompt(500, 1, 2), held))                # shorter than min
        self.assertIsNone(self.decide(kh_on(), p, held, info={}))                       # not usable
        self.assertIsNone(self.decide(kh_on(), p, held, gs=2))

    def test_09_all_idle_slots_empty(self):
        p = prompt(30000, 1, 2)
        d = self.decide(kh_on(), p, [[], [], [], []])
        self.assertTrue(d.to_slot)
        self.assertIsNone(d.slot)
        self.assertEqual((d.served, d.head), (0, 30007))
        self.assertIn("alone goes through a batch slot", d.reason)

    def test_10_the_same_system_prompt_is_held(self):
        held = [[], held_of(prompt(30000, 1, 2)), [], []]
        d = self.decide(kh_on(), prompt(30000, 7, 8), held)
        self.assertFalse(d.to_slot)
        self.assertEqual((d.slot, d.served), (1, 30001))
        self.assertEqual(d.head, 30007)

    def test_11_the_next_turn_of_a_held_conversation(self):
        first = prompt(3000, 1, 2)
        held_ = held_of(first)
        for grown, to_slot in ((500, False), (1500, True)):
            nxt = held_ + turn(2, n=grown) + turn(3)                # the history grew by `grown` tokens
            d = self.decide(kh_on(), nxt, [held_, [], [], []])
            with self.subTest(grown=grown):
                self.assertEqual(d.to_slot, to_slot)
                self.assertEqual(d.served, len(held_))
                self.assertEqual(d.head - d.served, grown + 1)

    def test_12_a_pin_is_counted_only_when_the_slot_remembers_it(self):
        a = [T] + list(range(100, 200)) + [T] + list(range(10_000, 30_000)) + [T, 1, 2, 3, 4]   # no root (T at 101 < 2048)
        held = held_of(a)
        b = a[:15500] + [T] + list(range(7000, 7200)) + [T, 9, 9, 9]
        sampling = {"strata_prefix": {"tokens": 15000}}
        kh = kh_on()
        kh.note_admitted(0, sampling, None)
        d = self.decide(kh, b, [held, [], [], []], sampling)
        self.assertFalse(d.to_slot)                                  # the slot serves the pin; the rest is below min
        self.assertEqual((d.slot, d.served), (0, 15000))
        kh2 = kh_on()                                                # the same content, but no pin remembered
        d2 = self.decide(kh2, b, [held, [], [], []], sampling)
        self.assertTrue(d2.to_slot)
        self.assertEqual(d2.served, 0)
        self.assertGreater(d2.head, 15000)

    def test_13_ckpt_0_counts_the_root_only(self):
        kh = kh_on()
        p = prompt(30000, 1, 2)
        no = {"strata_checkpoint": False}
        d = self.decide(kh, p, [[], [], [], []], no)
        self.assertTrue(d.to_slot)
        self.assertEqual(d.head, 30001)                              # not the turn at 30007
        held = [held_of(prompt(30000, 5, 6)), [], [], []]
        d = self.decide(kh, p, held, no)
        self.assertFalse(d.to_slot)
        self.assertEqual((d.slot, d.served, d.head), (0, 30001, 30001))
        # a long prompt whose system prompt is short has no root: nothing to keep for a ckpt=0 request
        self.assertFalse(self.decide(kh, prompt(1500, 1, 2) + list(range(5000, 9000)), [[], [], [], []], no).to_slot)
        # a busy slot serves nothing
        d = self.decide(kh, p, held, busy=[True, False, False, False])
        self.assertTrue(d.to_slot)
        self.assertIsNone(d.slot)

    def test_14_check_reuse(self):
        kh = kh_on()
        held = [[], held_of(prompt(30000, 1, 2)), [], []]
        kh.note_admitted(1, {"strata_prefix": {"tokens": 100}}, None)
        d = self.decide(kh, prompt(30000, 7, 8), held)
        self.assertEqual((d.slot, d.served), (1, 30001))
        self.assertIsNone(kh.check_reuse(d, 30001))
        self.assertIsNone(kh.check_reuse(d, 30001 - 64))                # within the slack
        self.assertEqual(kh.metas[1].pin, 100)
        self.assertEqual(kh.check_reuse(d, 30001 - 65), 1)
        self.assertEqual(kh.metas[1], SlotMeta())                       # its record is back to the default
        for bad in (None, True, False, "30001", 2.5):
            with self.subTest(reused=bad):
                self.assertIsNone(kh.check_reuse(d, bad))
        self.assertIsNone(kh.check_reuse(None, 0))
        self.assertIsNone(kh.check_reuse(Decision(True, 1, 30001, 30007, "x"), 0))     # went through a slot itself
        self.assertIsNone(kh.check_reuse(Decision(False, None, 0, 30007, "x"), 0))     # no slot was expected to serve

    def test_15_held_after_admission(self):
        kh = kh_on()
        prev = [1, 2, 3]
        got = kh.held_after_admission(prev, False, False, False, False)
        self.assertEqual(got, prev)
        self.assertIsNot(got, prev)                                     # a copy
        self.assertEqual(kh.held_after_admission([], False, False, False, False), [])
        self.assertIsNone(kh.held_after_admission(prev, True, False, False, False))
        self.assertIsNone(kh.held_after_admission(prev, False, True, False, False))
        self.assertIsNone(kh.held_after_admission(prev, False, False, True, False))
        self.assertIsNone(kh.held_after_admission(prev, False, False, False, True))

    def test_16_note_admitted(self):
        kh = kh_on()
        self.assertEqual(kh.metas[2], SlotMeta())
        kh.note_admitted(2, {"strata_prefix": {"tokens": 700}, "strata_checkpoint": False}, "/tmp/e.bin")
        self.assertEqual(kh.metas[2], SlotMeta(pin=700, ckpt=False, img=True))
        kh.note_admitted(2, {}, None)                                   # the next admission overwrites it
        self.assertEqual(kh.metas[2], SlotMeta(pin=0, ckpt=True, img=False))
        self.assertEqual(kh.metas[0], SlotMeta())
        kh.note_admitted(9, {}, None)                                   # a slot it does not know: no error


class Pick(unittest.TestCase):
    """KeepHeads.pick and StrataEngine.pick_slot with the rule on (off: exactly the old order)."""

    @staticmethod
    def base_for(held, used):
        return lambda b: (bool(held[b]), used[b])

    def pick(self, kh, p, held, used=None, free=None):
        used = used or [float(i) for i in range(len(held))]
        free = list(range(len(held))) if free is None else free
        return kh.pick(p, free, held, self.base_for(held, used))

    def test_17_nothing_to_protect(self):
        held = [[], [], [], []]
        for used in ([3.0, 2.0, 1.0, 0.0], [0.0, 1.0, 2.0, 3.0]):
            b, note = self.pick(kh_on(), prompt(30000, 1, 2), held, used)
            self.assertEqual(b, min(range(4), key=lambda x: (False, used[x])))
            self.assertIsNone(note)

    def test_18_a_unique_long_head_goes_last(self):
        held = [held_of(prompt(5000, 1, 2, seed=11)),            # the oldest, with a head only it keeps
                held_of(prompt(100, 1, 2, seed=12)),             # a short middle (nothing to lose)
                held_of(prompt(5000, 3, 4, seed=13)),
                held_of(prompt(5000, 5, 6, seed=14))]
        b, note = self.pick(kh_on(), prompt(5000, 8, 9, seed=20), held, used=[1.0, 2.0, 3.0, 4.0])
        self.assertEqual(b, 1)
        self.assertIsInstance(note, str)
        self.assertIn("slot 1", note)
        self.assertIn("instead of slot 0", note)

    def test_19_every_slot_has_a_unique_long_head(self):
        held = [held_of(prompt(9000, 1, 2, seed=11)), held_of(prompt(2500, 1, 2, seed=12)),
                held_of(prompt(4000, 1, 2, seed=13)), held_of(prompt(7000, 1, 2, seed=14))]
        b, note = self.pick(kh_on(), prompt(5000, 8, 9, seed=20), held, used=[1.0, 2.0, 3.0, 4.0])
        self.assertEqual(b, 1)                                   # the shortest head
        self.assertIsInstance(note, str)
        self.assertIn("slot 1 for a 5013-token prompt instead of slot 0", note)
        self.assertIn("lose a 9007-token head", note)           # what slot 0 (the one the old order picks) would lose

    def test_20_a_head_in_two_slots_goes_first(self):
        a = held_of(prompt(5000, 1, 2, seed=11))
        held = [held_of(prompt(5000, 3, 4, seed=12)), list(a), held_of(prompt(5000, 5, 6, seed=13)), list(a)]
        b, _ = self.pick(kh_on(), prompt(5000, 8, 9, seed=20), held, used=[1.0, 2.0, 3.0, 4.0])
        self.assertIn(b, (1, 3))
        b, _ = self.pick(kh_on(), prompt(5000, 8, 9, seed=20), held, used=[1.0, 4.0, 3.0, 2.0])
        self.assertEqual(b, 3)                                   # the older of the two copies

    def test_21_a_head_the_new_prompt_shares_is_not_lost(self):
        a = held_of(prompt(5000, 1, 2, seed=11))
        held = [a, held_of(prompt(5000, 3, 4, seed=12)), held_of(prompt(5000, 5, 6, seed=13)),
                held_of(prompt(5000, 7, 8, seed=14))]
        b, note = self.pick(kh_on(), prompt(5000, 1, 2, 3, seed=11), held, used=[1.0, 2.0, 3.0, 4.0])
        self.assertEqual(b, 0)
        self.assertIsNone(note)

    def test_21b_a_request_with_pictures_shares_nothing(self):
        """A request with pictures neither resumes from a slot nor stays in one (generate.cpp: the slot source needs
        req_imgs empty; such a slot is no cache): the head it shares with a slot is lost too (review 2026-11-25, P2)."""
        held = [held_of(prompt(5000, 1, 2, seed=11)),           # the head the new prompt shares, newer
                held_of(prompt(2000, 3, 4, seed=12))]           # an unrelated, shorter head, older
        used = [2.0, 1.0]
        p = prompt(5000, 8, 9, seed=11)                          # the same system prompt, another conversation
        b, _ = self.pick(kh_on(2), p, held, used)
        self.assertEqual(b, 0)                                   # text: slot 0's root is copied back in
        b, note = kh_on(2).pick(p, [0, 1], held, self.base_for(held, used), images=True)
        self.assertEqual(b, 1)                                   # pictures: slot 0 would lose 5000+, slot 1 2000+
        self.assertIsNone(note)                                  # (the old order picks slot 1 too)
        held[1] = held_of(prompt(7000, 3, 4, seed=12))
        b, note = kh_on(2).pick(p, [0, 1], held, self.base_for(held, used), images=True)
        self.assertEqual(b, 0)
        self.assertIn("instead of slot 1", note)
        e = self.engine(2, 1, kh_on(2), pad_rows=False)
        e.slot_held, e.slot_used = [list(h) for h in held], [2.0, 1.0]
        e.slot_held[1] = held_of(prompt(2000, 3, 4, seed=12))
        with mock.patch("builtins.print"):
            self.assertEqual(e.pick_slot(p), 0)
            self.assertEqual(e.pick_slot(p, images=True), 1)

    def test_22_below_min_the_old_order_stays(self):
        held = [held_of(prompt(300, 1, 2, seed=11)), held_of(prompt(300, 3, 4, seed=12)), [], held_of(prompt(300, 5, 6, seed=13))]
        for used in ([0.0, 1.0, 2.0, 3.0], [3.0, 2.0, 1.0, 0.0], [2.0, 0.0, 3.0, 1.0]):
            b, note = self.pick(kh_on(), prompt(300, 8, 9, seed=20), held, used)
            self.assertEqual(b, 2)                               # an empty slot first, as before
            self.assertIsNone(note)
        held[2] = held_of(prompt(300, 7, 8, seed=14))
        b, note = self.pick(kh_on(), prompt(300, 8, 9, seed=20), held, [2.0, 0.0, 3.0, 1.0])
        self.assertEqual(b, 1)
        self.assertIsNone(note)

    @staticmethod
    def engine(batch, groups, with_kh, pad_rows=True, info=INFO):
        e = object.__new__(StrataEngine)            # pick_slot reads only the slot tables (as test_slot_alloc)
        e.info = dict(info)                         # (and INFO slot_cache, for KeepHeads.applies)
        gs = batch // groups
        e.batch = batch
        e.slot_order = [g * gs + t for t in range(gs) for g in range(groups)]
        e.slot_group = [b // gs for b in range(batch)]
        e.slot_groups = groups
        e.slot_busy = [False] * batch
        e.slot_held = [[] for _ in range(batch)]
        e.slot_used = [0.0] * batch
        e.slot_gs = gs if groups > 1 and pad_rows else 0       # INFO batch_pad=0 (a remote stage): 0
        if with_kh is not None:
            e.keep_heads = with_kh
        return e

    def test_23_off_it_is_the_old_pick_slot(self):
        rng = random.Random(23)
        for groups in (1, 2):
            e_old = self.engine(4, groups, None)                                   # no attribute at all
            e_off = self.engine(4, groups, KeepHeads(4, env={}))                   # switched off
            e_zero = self.engine(4, groups, KeepHeads(4, env={"STRATA_KEEP_HEADS": "0"}))
            seen = set()
            for _ in range(2000):
                prompt_ = [rng.choice((1, 2, 3, e_old.TURN_ID)) for _ in range(rng.randint(2, 40))]
                held, used, busy = [], [], []
                for _b in range(4):
                    r = rng.random()
                    if r < 0.3:
                        held.append([])
                    elif r < 0.6:
                        held.append(prompt_[:rng.randint(1, len(prompt_))] + [rng.randint(1, 3)] * rng.randint(0, 3))
                    else:
                        held.append([rng.choice((1, 2, 3, e_old.TURN_ID)) for _ in range(rng.randint(1, 40))])
                    used.append(float(rng.randint(0, 5)))
                    busy.append(rng.random() < 0.3)
                got = []
                for e in (e_old, e_off, e_zero):
                    e.slot_held = [list(h) for h in held]
                    e.slot_used = list(used)
                    e.slot_busy = list(busy)
                    got.append(e.pick_slot(list(prompt_)))
                seen.add(got[0])
                self.assertEqual(got[0], got[1], (groups, prompt_, held, used, busy))
                self.assertEqual(got[0], got[2], (groups, prompt_, held, used, busy))
            self.assertGreater(len(seen), 2)                                        # the trials were not trivial

    def test_24_on_the_whole_held_stream_is_still_picked_first(self):
        for groups in (1, 2):
            e = self.engine(4, groups, kh_on(), pad_rows=False)
            p = list(range(1, 701)) + [5, 6, 7]
            e.slot_held[2] = list(range(1, 701))
            e.slot_held[0] = held_of(prompt(5000, 1, 2, seed=11))
            e.slot_held[1] = held_of(prompt(5000, 3, 4, seed=12))
            e.slot_held[3] = held_of(prompt(5000, 5, 6, seed=13))
            e.slot_used = [0.0, 1.0, 9.0, 2.0]
            self.assertEqual(e.pick_slot(p), 2)
            e.slot_busy[2] = True                                   # now it has to give up a head: the least bad one
            self.assertIn(e.pick_slot(p), (0, 1, 3))

    def test_24b_on_pick_slot_protects_a_long_head(self):
        e = self.engine(4, 1, kh_on(), pad_rows=False)
        e.slot_held = [held_of(prompt(5000, 1, 2, seed=11)), held_of(prompt(100, 1, 2, seed=12)),
                       held_of(prompt(5000, 3, 4, seed=13)), held_of(prompt(5000, 5, 6, seed=14))]
        e.slot_used = [1.0, 2.0, 3.0, 4.0]
        with mock.patch("builtins.print"):
            self.assertEqual(e.pick_slot(prompt(5000, 8, 9, seed=20)), 1)
        e_old = self.engine(4, 1, None)
        e_old.slot_held, e_old.slot_used = e.slot_held, e.slot_used
        self.assertEqual(e_old.pick_slot(prompt(5000, 8, 9, seed=20)), 0)

    def test_24d_on_a_request_with_pictures_skips_the_held_prefix_step(self):
        """pick_slot's first step (a slot whose whole stream or last turn starts the prompt) does not look at pictures,
        but the engine neither resumes a read with pictures from a slot nor keeps that slot as a cache: with the rule
        on, a request with pictures goes through KeepHeads.pick (images=True) over every free slot (Astra 2026-11-25)."""
        a = held_of(prompt(5000, 1, 2, seed=11))                # the long text head A
        p = prompt(5000, 1, 2, 3, seed=11)                       # agrees with A up to A's last turn
        for groups in (1, 2):
            e = self.engine(4, groups, kh_on(), pad_rows=False)
            e.slot_held = [list(a), [], held_of(prompt(300, 4, 5, seed=12)), []]
            e.slot_used = [5.0, 1.0, 2.0, 3.0]
            with mock.patch("builtins.print"):
                self.assertEqual(e.pick_slot(p), 0, groups)                       # text: the first step, as before
                b = e.pick_slot(p, images=True)
                self.assertNotEqual(b, 0, groups)                                 # pictures: A stays
                self.assertEqual(e.slot_held[b], [], groups)                      # (an empty slot first)
                for x in (1, 2, 3):                                               # only A's slot free: it is used
                    e.slot_busy[x] = True
                self.assertEqual(e.pick_slot(p, images=True), 0, groups)
            for kh, info in ((None, INFO), (KeepHeads(4, env={}), INFO), (kh_on(), {"slot_cache": 0})):
                e = self.engine(4, groups, kh, pad_rows=False, info=info)         # off / not usable: as before
                e.slot_held = [list(a), [], held_of(prompt(300, 4, 5, seed=12)), []]
                e.slot_used = [5.0, 1.0, 2.0, 3.0]
                self.assertEqual(e.pick_slot(p, images=True), 0, (groups, info))

    def test_23b_on_but_not_usable_it_is_the_old_pick_slot(self):
        """STRATA_KEEP_HEADS=1 where usable() is false (Qwen3.6's INFO, a control vector, slot_cache=0,
        STRATA_PARALLEL_SOLO=0): pick_slot picks what it picks with no KeepHeads at all, with and without pictures
        (review 2026-11-25 #3)."""
        rng = random.Random(231)
        cases = [(kh_on(), {"batch_slots": 4, "slot_cache": 1}), (kh_on(), {**INFO, "cvec": "speed.gguf"}),
                 (kh_on(), {**INFO, "slot_cache": 0}), (kh_on(STRATA_PARALLEL_SOLO="0"), INFO)]
        for groups in (1, 2):
            e_old = self.engine(4, groups, None, pad_rows=False)
            ons = [self.engine(4, groups, kh, pad_rows=False, info=info) for kh, info in cases]
            seen = set()
            for _ in range(1000):
                prompt_ = [rng.choice((1, 2, 3, e_old.TURN_ID)) for _ in range(rng.randint(2, 600))]
                held, used, busy = [], [], []
                for _b in range(4):
                    r = rng.random()
                    if r < 0.3:
                        held.append([])
                    elif r < 0.6:
                        held.append(prompt_[:rng.randint(1, len(prompt_))] + [rng.randint(1, 3)] * rng.randint(0, 3))
                    else:
                        held.append([rng.choice((1, 2, 3, e_old.TURN_ID)) for _ in range(rng.randint(1, 600))])
                    used.append(float(rng.randint(0, 5)))
                    busy.append(rng.random() < 0.3)
                images = rng.random() < 0.5
                got = []
                for e in [e_old] + ons:
                    e.slot_held = [list(h) for h in held]
                    e.slot_used = list(used)
                    e.slot_busy = list(busy)
                    got.append(e.pick_slot(list(prompt_), images=images))
                seen.add(got[0])
                self.assertEqual(got, [got[0]] * len(got), (groups, images, held, used, busy))
            self.assertGreater(len(seen), 2)

    def test_24c_on_but_not_usable_it_is_the_old_pick_slot(self):
        """pick_slot uses the rule only where decide() does (KeepHeads.usable): INFO slot_cache=0 or missing, a
        conversation cache, STRATA_PARALLEL_SOLO=0, STRATA_SLOT_CKPT_MOVE=1 (review 2026-11-25, P3)."""
        held = [held_of(prompt(5000, 1, 2, seed=11)), held_of(prompt(100, 1, 2, seed=12)),
                held_of(prompt(5000, 3, 4, seed=13)), held_of(prompt(5000, 5, 6, seed=14))]
        cases = [(kh_on(), {"slot_cache": 0}), (kh_on(), {}),
                 (kh_on(), {"slot_cache": 1, "conversation_cache_mib": 512, "conversation_cache_slots": 4}),
                 (kh_on(STRATA_PARALLEL_SOLO="0"), INFO), (kh_on(STRATA_SLOT_CKPT_MOVE="1"), INFO)]
        for kh, info in cases:
            e = self.engine(4, 1, kh, pad_rows=False, info=info)
            e.slot_held, e.slot_used = [list(h) for h in held], [1.0, 2.0, 3.0, 4.0]
            with mock.patch("builtins.print") as said:
                self.assertEqual(e.pick_slot(prompt(5000, 8, 9, seed=20)), 0, info)
            said.assert_not_called()

    def test_25_speed(self):
        n = 260_000
        base = list(range(1000, 1000 + n))
        for i in (0, 100_000, 259_000):
            base[i] = T
        p = base[:] + [5, 5, 5]
        held = []
        for k in range(4):
            h = base[:] + [1, 2, 3]
            h[250_000 + k] = 77                      # they differ near the end: the compares run nearly the whole way
            held.append(h)
        kh = kh_on()
        kh.min_tokens = 1000
        t0 = time.perf_counter()
        d = kh.decide(p, {}, None, held, [False] * 4, INFO, 0)
        b, note = kh.pick(p, [0, 1, 2, 3], held, lambda x: (True, float(x)))
        took = time.perf_counter() - t0
        self.assertIsNotNone(d)
        self.assertIn(b, range(4))
        self.assertLess(took, 0.2, f"decide + pick took {took:.3f} s")


class ServerPath(FakeEngineCase):
    """generate_batched over the fake engine: the commands it sends with STRATA_KEEP_HEADS unset and set."""

    SYSTEM = "You are a careful assistant for the long-prompt tests. " * 5            # 280 bytes

    def start_kh(self, on=True, **kw):
        env = {"STRATA_KEEP_HEADS": "1", "STRATA_KEEP_HEADS_MIN": "64"} if on else {}
        with mock.patch.dict(os.environ, env, clear=False):
            if not on:
                for k in ("STRATA_KEEP_HEADS", "STRATA_KEEP_HEADS_MIN"):
                    os.environ.pop(k, None)
            self.start(4, slot_cache=True, says_groups=2, more=["--batch-groups", "2", "--says-no-pad"], **kw)
        self.engine.keep_heads.turn_id = 256                  # ByteTokenizer's <|im_start|>
        self.engine.keep_heads.root_min = 1

    def ask(self, system, user, max_tokens=64, **extra):
        import json
        import urllib.request
        body = {"messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "max_tokens": max_tokens, "reasoning_effort": "none", **extra}
        req = urllib.request.Request(self.base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read().decode())["choices"][0]["message"]["content"]

    def commands(self):
        path = self.log
        text = path.read_text() if path.exists() else ""
        return [line for line in text.splitlines() if line.startswith(("GEN", "BGEN"))]

    def wait_idle(self):
        for _ in range(100):
            if not any(self.engine.slot_busy):
                return
            time.sleep(0.05)
        self.fail("a slot stayed busy")

    def test_26_off_a_long_request_alone_is_a_plain_gen(self):
        self.start_kh(on=False)
        self.assertFalse(self.engine.keep_heads.enabled)
        reply = self.ask(self.SYSTEM + "S1 ", "LONGREPLY please")
        self.wait_idle()
        cmds = self.commands()
        self.assertEqual([c.split()[0] for c in cmds], ["GEN"], cmds)
        self.assertEqual(reply, "ok, " + "la " * 15 + "done.")

    def test_27_on_the_first_request_goes_through_a_slot_and_back(self):
        self.start_kh()
        reply = self.ask(self.SYSTEM + "S1 ", "LONGREPLY please")
        self.wait_idle()
        cmds = self.commands()
        self.assertEqual([c.split()[0] for c in cmds[:2]], ["BGEN", "GEN"], cmds)
        self.assertEqual(cmds[1].split()[1], "None")
        self.assertEqual(reply, "ok, " + "la " * 15 + "done.")

    def test_28_on_the_same_system_prompt_stays_on_the_solo_path_and_a_short_reuse_is_forgotten(self):
        self.start_kh()
        self.ask(self.SYSTEM + "S1 ", "LONGREPLY please")
        self.wait_idle()
        n = len(self.commands())
        slot = int(self.commands()[0].split()[1])
        self.assertTrue(self.engine.slot_held[slot])
        # (an answer cut at max_tokens: the solo leg's code after its loop runs, where check_reuse is called)
        self.ask(self.SYSTEM + "S1 ", "another question, LONGREPLY", max_tokens=5)
        self.wait_idle()
        cmds = self.commands()[n:]
        self.assertEqual([c.split()[0] for c in cmds], ["GEN"], cmds)
        self.assertEqual(self.engine.slot_held[slot], [])         # the fake engine reused 0: the record is forgotten
        self.assertEqual(self.engine.keep_heads.metas[slot], SlotMeta())

    def test_28b_an_answer_that_ends_at_its_stop_token_also_reaches_check_reuse(self):
        """Service stops reading at the stop token, which closes the generator at its `yield t` inside the solo leg's
        loop: the check runs in the `finally` too, after the drained DONE (review 2026-11-25, P2)."""
        self.start_kh()
        self.ask(self.SYSTEM + "S1 ", "LONGREPLY please")
        self.wait_idle()
        slot = int(self.commands()[0].split()[1])
        self.ask(self.SYSTEM + "S1 ", "another question, LONGREPLY")
        self.wait_idle()
        self.assertEqual(self.engine.slot_held[slot], [])
        self.assertEqual(self.engine.keep_heads.metas[slot], SlotMeta())

    def test_28c_on_an_admission_whose_first_token_is_the_stop_token_keeps_the_record(self):
        """A BGEN whose first token is the stop token: Service closes the generator before the BADM is read (the
        `finally` drains it).  `BADM <slot> 0`: the engine left the slot as it was, so does the record (review
        2026-11-25, P2)."""
        import threading
        self.start_kh()
        self.ask(self.SYSTEM + "S1 ", "LONGREPLY please")
        self.wait_idle()
        slot = int(self.commands()[0].split()[1])
        before = list(self.engine.slot_held[slot])
        meta = self.engine.keep_heads.metas[slot]
        self.assertTrue(before)
        n = len(self.commands())
        for b in range(4):                                       # only `slot` is free: not alone, a plain BGEN
            self.engine.slot_busy[b] = b != slot
        try:
            ids = list(b"a prompt whose answer is already here: ok, done.")   # the fake's reply: only its stop token
            g = self.engine.generate(ids, 64, {}, threading.Event())
            got = []
            for t in g:                                          # as Service: it stops reading at the stop token
                if t is None:
                    continue
                got.append(t)
                if t in self.svc.stop_ids:
                    g.close()
                    break
        finally:
            for b in range(4):
                self.engine.slot_busy[b] = False
        self.assertEqual(len(got), 1)
        self.assertEqual([c.split()[:2] for c in self.commands()[n:]], [["BGEN", str(slot)]])
        self.assertEqual(self.engine.slot_held[slot], before)
        self.assertEqual(self.engine.keep_heads.metas[slot], meta)

    def only_free(self, slot, fn):
        """Run fn() with every other slot marked busy (not alone: a plain BGEN into `slot`)."""
        for b in range(4):
            self.engine.slot_busy[b] = b != slot
        try:
            return fn()
        finally:
            for b in range(4):
                self.engine.slot_busy[b] = False

    def read_to_stop(self, ids, max_new=64):
        """As Service: read the generator, close it at the stop token."""
        import threading
        g = self.engine.generate(ids, max_new, {}, threading.Event())
        got = []
        for t in g:
            if t is None:
                continue
            got.append(t)
            if t in self.svc.stop_ids:
                g.close()
                break
        return got

    def test_28d_on_with_a_control_vector_it_does_nothing(self):
        """An engine with a control vector loaded (INFO cvec=...): the projection splits the slots (generate.cpp
        ~16655), so the rule is off there (usable() false) - the same commands and record as without it (reviews
        2026-11-25 #2 and #3)."""
        self.start_kh()
        self.ask(self.SYSTEM + "S1 ", "LONGREPLY please")        # (no control vector yet: through a slot)
        self.wait_idle()
        slot = int(self.commands()[0].split()[1])
        before = list(self.engine.slot_held[slot])
        self.assertTrue(before)
        self.engine.info["cvec"] = "speed.gguf"
        n = len(self.commands())
        for kw in ({"experimental_speed_projection": False}, {}):
            self.ask(self.SYSTEM + "S1 ", "LONGREPLY please", **kw)
            self.wait_idle()
            self.assertEqual(self.engine.slot_held[slot], before, kw)
        self.assertEqual([c.split()[0] for c in self.commands()[n:]], ["GEN", "GEN"])

    def test_28e_on_but_slot_cache_0_an_admission_does_not_restore_the_record(self):
        """INFO slot_cache=0: no slot is a cache, so the record is not given back after an admission that did not
        continue - neither after a drained BADM nor after max_tokens 1 (review 2026-11-25 #2, P3)."""
        self.start_kh()
        self.ask(self.SYSTEM + "S1 ", "LONGREPLY please")
        self.wait_idle()
        slot = int(self.commands()[0].split()[1])
        held = list(self.engine.slot_held[slot])
        self.assertTrue(held)
        self.engine.info["slot_cache"] = 0
        got = self.only_free(slot, lambda: self.read_to_stop(list(b"a prompt whose answer is already here: ok, done.")))
        self.assertEqual(len(got), 1)
        self.assertEqual(self.engine.slot_held[slot], [])
        self.engine.slot_held[slot] = list(held)
        self.only_free(slot, lambda: self.read_to_stop(list(b"another short prompt"), max_new=1))
        self.assertEqual(self.engine.slot_held[slot], [])
        self.assertIn(f"BGEN {slot}", "\n".join(self.commands()[-2:]))

    def test_28f_on_the_drained_badm_is_judged_by_its_own_done(self):
        """A consumer that left during the read: the `finally` drains this admission's DONE (nonfinite) and BADM 0,
        and another slot's BDONE can rewrite the shared self.last in between.  The record stays empty because the
        request's own DONE said nonfinite (review 2026-11-25 #2, P3)."""
        self.start_kh()
        self.ask(self.SYSTEM + "S1 ", "LONGREPLY please")
        self.wait_idle()
        slot = int(self.commands()[0].split()[1])
        self.assertTrue(self.engine.slot_held[slot])
        eng = self.engine
        real_control, real_drain = eng._control, eng._drain_control

        def control(*a, **kw):                                   # a heartbeat first: the consumer leaves at it
            yield None
            yield from real_control(*a, **kw)

        def drain(*a, **kw):                                     # another slot's BDONE lands right after
            line = real_drain(*a, **kw)
            eng._apply_bdone("BDONE 3 5 stop 1.0".split())
            return line

        def go():
            import threading
            g = eng.generate(list(b"NFADMIT a prompt the engine fails"), 64, {}, threading.Event())
            self.assertIsNone(next(g))
            g.close()
        with mock.patch.object(eng, "_control", control), mock.patch.object(eng, "_drain_control", drain):
            self.only_free(slot, go)
        self.assertEqual(eng.last.get("finish"), "stop")          # (the shared figures were rewritten)
        self.assertEqual(eng.slot_held[slot], [])

    def test_28g_on_but_parallel_solo_0_the_record_is_the_base_one(self):
        """STRATA_KEEP_HEADS=1 with STRATA_PARALLEL_SOLO=0 (usable() false): an admission that does not continue
        leaves the record as the base (7590c991e) does - empty after max_tokens 1 and after a first token that is
        the stop token (the base clears it at BGEN and never gives it back; checked on the base tree with the same
        fake engine, review 2026-11-25 #3)."""
        with mock.patch.dict(os.environ, {"STRATA_PARALLEL_SOLO": "0"}):
            self.start_kh()
            self.assertFalse(self.engine.keep_heads.usable(self.engine.info, getattr(self.engine, "slot_gs", 0)))
            metas = [SlotMeta(m.pin, m.ckpt, m.img) for m in self.engine.keep_heads.metas]
            for ids, max_new in ((list(b"another short prompt"), 1),
                                 (list(b"a prompt whose answer is already here: ok, done."), 64)):
                self.engine.slot_held[0] = list(range(1, 400))
                self.only_free(0, lambda: self.read_to_stop(ids, max_new=max_new))
                self.assertEqual(self.engine.slot_held[0], [], max_new)
            self.assertEqual([c.split()[:2] for c in self.commands()], [["BGEN", "0"], ["BGEN", "0"]])
            self.assertEqual([SlotMeta(m.pin, m.ckpt, m.img) for m in self.engine.keep_heads.metas], metas)

    def test_29_on_max_tokens_1_does_not_forget_a_held_head(self):
        self.start_kh()
        self.ask(self.SYSTEM + "S1 ", "LONGREPLY please")
        self.wait_idle()
        held = [list(h) for h in self.engine.slot_held]
        self.assertTrue(any(held))
        metas = [SlotMeta(m.pin, m.ckpt, m.img) for m in self.engine.keep_heads.metas]
        for b in range(4):                                       # fill every slot, as a parked request would
            self.engine.slot_busy[b] = b != 0
        free_slot = [b for b in range(4) if not self.engine.slot_busy[b]]
        self.assertEqual(free_slot, [0])
        self.engine.slot_held[0] = held[0] or [1, 2, 3]
        before = list(self.engine.slot_held[0])
        import threading
        out = {}

        def go():
            out["r"] = self.ask("A different system prompt entirely. " * 8, "hi", max_tokens=1)
        t = threading.Thread(target=go)
        t.start()
        t.join(30)
        for b in range(4):
            self.engine.slot_busy[b] = False
        self.assertFalse(t.is_alive())
        self.assertEqual(self.engine.slot_held[0], before)        # the admission left the slot alone: so does the record
        self.assertEqual([SlotMeta(m.pin, m.ckpt, m.img) for m in self.engine.keep_heads.metas], metas)

    def test_30_on_two_at_once_both_complete(self):
        self.start_kh()
        import threading
        out = {}

        def go(k, system, user):
            out[k] = self.ask(system, user)
        a = threading.Thread(target=go, args=("a", self.SYSTEM + "S1 ", "LONGREPLY please"))
        a.start()
        time.sleep(0.15)
        b = threading.Thread(target=go, args=("b", "short system prompt", "hello"))
        b.start()
        a.join(30)
        b.join(30)
        self.assertEqual(out.get("a"), "ok, " + "la " * 15 + "done.")
        self.assertEqual(out.get("b"), "ok, done.")
        self.wait_idle()
        self.assertFalse(any(self.engine.slot_busy))
        self.assertEqual(self.engine.waiting, 0)


if __name__ == "__main__":
    unittest.main()
