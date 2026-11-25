"""serve/warm_heads.py(再起動後の頭の温め。STRATA_WARM_HEADS=1)の試験。GPU も実エンジンも使わない。

Env / Store / Select は WarmHeads だけを直接使う。ServerPath は serve.test_parallel の偽エンジンの上で、実物の
StrataEngine と Service を動かし、エンジンへ届く命令(BGEN と max_new)を見る。
"""
import base64
import json
import os
import tempfile
import threading
import time
import unittest
import urllib.request
from array import array
from pathlib import Path
from unittest import mock

import serve.test_parallel as tp
from serve.server import StrataEngine
from serve.test_parallel import FakeEngineCase
from serve.warm_heads import (FILE_MAX_BYTES, WarmHeads, default_path, env_int, env_on, head_key)

TURN = 256                       # ByteTokenizer の <|im_start|>


def make_env(path, **extra):
    env = {"STRATA_WARM_HEADS": "1", "STRATA_WARM_HEADS_FILE": str(path), "STRATA_WARM_HEADS_MIN": "64"}
    env.update({k: str(v) for k, v in extra.items()})
    return env


def ids_of(seed, n):
    return [1000 + seed * 10000 + i for i in range(n)]


def two_notes(wh, ids, reused=None):
    reused = len(ids) - 1 if reused is None else reused
    wh.note(ids, reused, images=False, finish="stop")
    wh.note(ids, reused, images=False, finish="stop")


class Env(unittest.TestCase):
    def test_flag_is_exactly_one(self):
        self.assertFalse(env_on({}))
        for v in ("0", "true", "", "yes", "11", "on"):
            self.assertFalse(env_on({"STRATA_WARM_HEADS": v}), v)
        for v in ("1", " 1 ", "1\n"):
            self.assertTrue(env_on({"STRATA_WARM_HEADS": v}), v)

    def test_numbers_out_of_range_or_not_numbers_give_the_default(self):
        self.assertEqual(env_int({}, "X", 7, 1, 10), 7)
        self.assertEqual(env_int({"X": "abc"}, "X", 7, 1, 10), 7)
        self.assertEqual(env_int({"X": ""}, "X", 7, 1, 10), 7)
        self.assertEqual(env_int({"X": "0"}, "X", 7, 1, 10), 7)
        self.assertEqual(env_int({"X": "11"}, "X", 7, 1, 10), 7)
        self.assertEqual(env_int({"X": " 3 "}, "X", 7, 1, 10), 3)
        wh = WarmHeads(None, TURN, {"STRATA_WARM_HEADS": "1", "STRATA_WARM_HEADS_MAX": "9",
                                    "STRATA_WARM_HEADS_MIN": "x"})
        self.assertEqual(wh.max_heads, 1)
        self.assertEqual(wh.min_tokens, 16384)

    def test_default_path(self):
        self.assertEqual(default_path("/var/log/strata/engine.log"), "/var/log/strata/warm-heads.json")
        self.assertTrue(default_path(None).endswith(os.path.join(".cache", "strata", "warm-heads.json")))

    def test_disabled_does_nothing(self):
        with tempfile.TemporaryDirectory() as d:
            wh = WarmHeads(None, TURN, {"STRATA_WARM_HEADS_FILE": os.path.join(d, "w.json")})
            self.assertFalse(wh.enabled)
            ids = ids_of(1, 200)
            wh.note(ids, 199, images=False, finish="stop")
            wh.note(ids, 199, images=False, finish="stop")
            wh.arriving(ids)
            wh.on_ready(object())
            self.assertEqual(wh.saved(), [])
            self.assertIsNone(wh._thread)
            self.assertEqual(os.listdir(d), [])


class Store(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.path = self.dir / "sub" / "warm-heads.json"
        self.env = make_env(self.path)

    def tearDown(self):
        self.tmp.cleanup()

    def wh(self, turn=TURN, **extra):
        return WarmHeads(None, turn, dict(self.env, **{k: str(v) for k, v in extra.items()}))

    def saved_file(self):
        wh = self.wh()
        two_notes(wh, ids_of(1, 200))
        wh.save_now()
        return json.loads(self.path.read_text())

    def test_save_and_load_round_trip(self):
        wh = self.wh()
        ids = ids_of(1, 200)
        two_notes(wh, ids)
        wh.note(ids, 199, images=False, finish="stop")
        wh.save_now()
        self.assertTrue(self.path.exists())
        self.assertEqual(oct(self.path.stat().st_mode & 0o777), oct(0o600))
        self.assertEqual(oct(self.path.parent.stat().st_mode & 0o777), oct(0o700))
        self.assertEqual(sorted(os.listdir(self.path.parent)), ["warm-heads.json"])      # 一時ファイルが残らない
        got = self.wh().load(None)
        self.assertEqual(len(got), 1)
        h = got[0]
        self.assertEqual((h.n, h.reuses), (199, 3))
        self.assertEqual(list(h.ids), ids[:199])
        self.assertEqual((h.n, h.sha), head_key(ids[:199]))
        self.assertAlmostEqual(h.last_used, wh.saved()[0].last_used, places=3)

    def bad(self, text, why=None):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(text, bytes):
            self.path.write_bytes(text)
        else:
            self.path.write_text(text)
        wh = self.wh()
        with mock.patch("builtins.print") as pr:
            got = wh.load(None)
        self.assertEqual(got, [])
        self.assertFalse(self.path.exists())
        self.assertTrue(Path(str(self.path) + ".bad").exists())
        said = " ".join(str(c.args[0]) for c in pr.call_args_list)
        self.assertIn("is unusable", said)
        self.assertIn("no warm-up this start", said)
        if why:
            self.assertIn(why, said)
        Path(str(self.path) + ".bad").unlink()

    def test_bad_files_are_moved_aside_and_do_not_raise(self):
        good = self.saved_file()
        self.path.unlink()
        # 1. JSON の途中で切れる
        self.bad(json.dumps(good)[:100], "not valid JSON")
        # 2. version 違い
        self.bad(json.dumps(dict(good, version=2)), "unknown version")
        # 3. base64 の誤り
        g = json.loads(json.dumps(good))
        g["heads"][0]["ids"] = "!!!not base64!!!"
        self.bad(json.dumps(g), "base64")
        # 4. sha256 の不一致(ids を 1 つ書き換える)
        g = json.loads(json.dumps(good))
        raw = bytearray(base64.b64decode(g["heads"][0]["ids"]))
        raw[0] ^= 1
        g["heads"][0]["ids"] = base64.b64encode(bytes(raw)).decode()
        self.bad(json.dumps(g), "sha256")
        # 5. 長さの不一致
        g = json.loads(json.dumps(good))
        g["heads"][0]["n"] += 1
        self.bad(json.dumps(g), "length")
        # 6. 4 MiB 超
        self.bad(b" " * (FILE_MAX_BYTES + 1), "MiB")
        # 7. JSON だが頭の形が違う
        self.bad(json.dumps(dict(good, heads="x")), "not a list")
        self.bad("[1, 2]", "unknown version")

    def test_id_out_of_range_is_bad(self):
        self.saved_file()
        wh = self.wh()
        with mock.patch("builtins.print"):
            self.assertEqual(wh.load(1100), [])          # ids は 1000 台から: n_vocab 1100 を超える
        self.assertTrue(Path(str(self.path) + ".bad").exists())

    def test_another_turn_id_is_kept_and_not_used(self):
        self.saved_file()
        other = self.wh(turn=999)
        with mock.patch("builtins.print"):
            self.assertEqual(other.load(None), [])
        self.assertTrue(self.path.exists())
        self.assertFalse(Path(str(self.path) + ".bad").exists())
        # 次に保存するときは上書きされる
        two_notes(other, ids_of(2, 150))
        other.save_now()
        self.assertEqual(json.loads(self.path.read_text())["turn_id"], 999)

    def test_missing_file_is_empty(self):
        self.assertEqual(self.wh().load(None), [])

    def test_save_with_nothing_to_save_writes_nothing(self):
        wh = self.wh()
        wh.note(ids_of(1, 200), 199, images=False, finish="stop")           # 1 回だけ: REUSES に届かない
        wh.save_now()
        self.assertFalse(self.path.exists())


class Select(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "w.json"

    def tearDown(self):
        self.tmp.cleanup()

    def wh(self, **extra):
        return WarmHeads(None, TURN, make_env(self.path, **extra))

    def test_a_short_reuse_is_not_remembered(self):
        wh = self.wh()
        two_notes(wh, ids_of(1, 200), reused=63)
        self.assertEqual(wh.saved(), [])
        self.assertEqual(wh._mem, {})

    def test_images_and_errors_are_not_remembered(self):
        wh = self.wh()
        ids = ids_of(1, 200)
        for _ in range(3):
            wh.note(ids, 199, images=True, finish="stop")
            wh.note(ids, 199, images=False, finish="error")
        self.assertEqual(wh._mem, {})

    def test_the_second_reuse_makes_it_a_candidate(self):
        wh = self.wh()
        ids = ids_of(1, 200)
        wh.note(ids, 150, images=False, finish="stop")
        self.assertEqual(wh.saved(), [])
        wh.note(ids, 150, images=False, finish="length")
        got = wh.saved()
        self.assertEqual([(h.n, h.reuses) for h in got], [(150, 2)])

    def test_reused_beyond_the_prompt_is_cut_to_the_prompt_minus_one(self):
        wh = self.wh()
        ids = ids_of(1, 100)
        two_notes(wh, ids, reused=500)
        self.assertEqual([h.n for h in wh.saved()], [99])

    def test_longer_than_the_limit_is_not_remembered(self):
        wh = self.wh(STRATA_WARM_HEADS_MAX_TOKENS=100)
        two_notes(wh, ids_of(1, 200))
        self.assertEqual(wh._mem, {})

    def test_max_one_keeps_the_most_reused(self):
        wh = self.wh()
        a, b = ids_of(1, 200), ids_of(2, 300)
        two_notes(wh, a)
        for _ in range(4):
            wh.note(b, 299, images=False, finish="stop")
        self.assertEqual([h.n for h in wh.saved()], [299])
        wh4 = self.wh(STRATA_WARM_HEADS_MAX=2)
        two_notes(wh4, a)
        for _ in range(4):
            wh4.note(b, 299, images=False, finish="stop")
        self.assertEqual([h.n for h in wh4.saved()], [299, 199])

    def test_a_head_past_the_ttl_drops_out(self):
        wh = self.wh()
        two_notes(wh, ids_of(1, 200))
        self.assertEqual(len(wh.saved()), 1)
        for h in wh._mem.values():
            h.last_used = time.time() - 8 * 86400
        self.assertEqual(wh.saved(), [])

    def test_more_than_sixteen_drop_the_least_used_first(self):
        wh = self.wh()
        keep = ids_of(99, 200)
        two_notes(wh, keep)
        for i in range(30):
            wh.note(ids_of(i, 200), 199, images=False, finish="stop")
        self.assertEqual(len(wh._mem), 16)
        self.assertEqual([h.n for h in wh.saved()], [199])
        self.assertEqual(wh.saved()[0].reuses, 2)

    def test_note_does_not_write_a_file(self):
        wh = self.wh()
        two_notes(wh, ids_of(1, 200))
        self.assertFalse(self.path.exists())
        self.assertEqual(os.listdir(self.tmp.name), [])


FAKE_FAST = tp.FAKE_BATCH
FAKE_SLOW = tp.FAKE_BATCH.replace("STEP = 0.02", "STEP = 0.04")      # 塊 1 つの読みが 20 ms: 3900 トークンで約 2.4 秒


class ServerPath(FakeEngineCase):
    """実物の StrataEngine・Service を偽エンジンの上で動かし、温めがエンジンに送る命令を見る。"""

    SYSTEM = "You are a careful assistant for the warm-up tests. " * 70          # 約 3,600 バイト
    OTHER = "A short system prompt that shares nothing with the other one."

    def setUp(self):
        self.wdir = tempfile.TemporaryDirectory()
        self.file = Path(self.wdir.name) / "warm-heads.json"
        self.calls = []
        orig = StrataEngine.generate
        calls = self.calls

        def spy(engine, ids, max_new, sampling, cancel, *a, **kw):
            calls.append((len(ids), max_new))
            return orig(engine, ids, max_new, sampling, cancel, *a, **kw)
        p = mock.patch.object(StrataEngine, "generate", spy)
        p.start()
        self.addCleanup(p.stop)
        p2 = mock.patch.object(tp, "FAKE_BATCH", FAKE_SLOW)
        p2.start()
        self.addCleanup(p2.stop)
        p3 = mock.patch.object(StrataEngine, "TURN_ID", TURN)          # WarmHeads と keep-heads が <|im_start|> = 256 を使う
        p3.start()
        self.addCleanup(p3.stop)

    def tearDown(self):
        super().tearDown()
        self.wdir.cleanup()

    def tearDown_fake(self):
        if getattr(self, "httpd", None):
            self.httpd.shutdown()
            self.httpd.server_close()
            self.httpd = None
        if getattr(self, "engine", None):
            self.engine.unload()
            self.engine = None
        if getattr(self, "tmp", None):
            self.tmp.cleanup()
            self.tmp = None

    def start_wh(self, on=True, keep=True, slots=4, reuse=False, **extra):
        env = {}
        if on:
            env.update(make_env(self.file, STRATA_WARM_HEADS_IDLE_S="0.5", **extra))
        if keep:
            env.update({"STRATA_KEEP_HEADS": "1", "STRATA_KEEP_HEADS_MIN": "64"})
        keys = ("STRATA_WARM_HEADS", "STRATA_WARM_HEADS_FILE", "STRATA_KEEP_HEADS", "STRATA_KEEP_HEADS_MIN")
        with mock.patch.dict(os.environ, env, clear=False):
            for k in keys:
                if k not in env:
                    os.environ.pop(k, None)
            self.start(slots, slot_cache=True, reuse=reuse, says_groups=2, more=["--batch-groups", "2", "--says-no-pad"])
        self.engine.keep_heads.root_min = 1

    def ask(self, system, user="please answer", max_tokens=16):
        body = {"messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "max_tokens": max_tokens, "reasoning_effort": "none"}
        req = urllib.request.Request(self.base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read().decode())["choices"][0]["message"]["content"]

    def prompt_ids(self, system, user="please answer"):
        """この system と user の要求が Service からエンジンに渡る ids(温めなしの偽エンジンで 1 回流して取る)。"""
        seen = []
        orig = StrataEngine.generate

        def grab(engine, ids, *a, **kw):
            seen.append(list(ids))
            return orig(engine, ids, *a, **kw)
        with mock.patch.object(StrataEngine, "generate", grab), mock.patch.object(tp, "FAKE_BATCH", FAKE_FAST):
            self.start_wh(on=False, keep=False)
            self.ask(system, user)
        self.tearDown_fake()
        self.assertEqual(len(seen), 1)
        del self.calls[:]
        return seen[0]

    def put_head(self, ids, n):
        """ファイルに、ids の先頭 n 個の頭を (回数 2 で) 置く。"""
        wh = WarmHeads(None, TURN, make_env(self.file))
        two_notes(wh, ids[:n + 1], reused=n)
        wh.save_now()
        self.assertTrue(self.file.exists())

    def wait_for(self, text, timeout=20.0):
        t0 = time.time()
        while time.time() - t0 < timeout:
            if any(text in line for line in self.engine.warm_heads.log_lines):
                return
            time.sleep(0.05)
        self.fail(f"{text!r} not in {self.engine.warm_heads.log_lines}")

    def count(self, text):
        return sum(text in line for line in self.engine.warm_heads.log_lines)

    def commands(self):
        text = self.log.read_text() if self.log.exists() else ""
        return [line for line in text.splitlines() if line.startswith(("GEN", "BGEN"))]

    def test_01_ready_reads_the_saved_head_through_a_slot_with_max_new_2(self):
        ids = self.prompt_ids(self.SYSTEM)
        n = len(ids) - 30
        self.put_head(ids, n)
        self.start_wh()
        self.wait_for("read in")
        self.assertEqual(self.calls, [(n + 1, 2)])
        cmds = self.commands()
        self.assertEqual(cmds[0].split()[0], "BGEN")
        self.assertEqual(int(cmds[0].split()[2]), n + 1)
        self.assertTrue(any("reading a %d-token head (1 of 1)" % n in x for x in self.engine.warm_heads.log_lines))
        # 温めは履歴にも合計にも数えない
        self.assertEqual(self.svc.totals["requests"], 0)

    def test_02_without_keep_heads_it_reads_with_max_new_1(self):
        """計画書は GEN 1 と書いたが、--batch のエンジンでは max_new 1 の要求も BGEN で届く(generate_batched)。"""
        ids = self.prompt_ids(self.SYSTEM)
        n = len(ids) - 30
        self.put_head(ids, n)
        self.start_wh(keep=False)
        self.wait_for("read in")
        self.assertEqual(self.calls, [(n + 1, 1)])

    def test_03_another_head_stops_it_and_it_is_read_again_when_idle(self):
        ids = self.prompt_ids(self.SYSTEM)
        n = len(ids) - 30
        self.put_head(ids, n)
        self.start_wh()
        wh = self.engine.warm_heads
        for _ in range(100):
            if wh._active is not None:
                break
            time.sleep(0.02)
        self.assertIsNotNone(wh._active, wh.log_lines)
        t0 = time.time()
        reply = self.ask(self.OTHER)
        took = time.time() - t0
        self.assertEqual(reply, "ok, done.")
        self.assertIn("stopped: a request with another head came", " ".join(wh.log_lines))
        self.assertLess(took, 2.0)                       # 温めの残り(約 2 秒)を待たずに答えが出た
        self.wait_for("read in")                         # 0.5 秒空いたあと、頭をもう一度読む
        self.assertEqual(self.count("reading a"), 2)
        self.assertIn("will try again", " ".join(wh.log_lines))
        self.assertEqual([c[1] for c in self.calls], [2, 16, 2])
        # 履歴に数えるのは本物の要求だけ
        self.assertEqual(self.svc.totals["requests"], 1)

    def test_04_the_same_head_does_not_stop_it(self):
        ids = self.prompt_ids(self.SYSTEM)
        n = len(ids) - 30
        self.put_head(ids, n)
        self.start_wh()
        wh = self.engine.warm_heads
        for _ in range(100):
            if wh._active is not None:
                break
            time.sleep(0.02)
        self.assertIsNotNone(wh._active)
        self.assertEqual(self.ask(self.SYSTEM), "ok, done.")
        self.assertNotIn("stopped", " ".join(wh.log_lines))
        self.assertEqual(self.count("reading a"), 1)
        self.assertEqual(self.count("read in"), 1)

    def test_05_without_the_flag_no_thread_no_file_no_extra_command(self):
        ids = self.prompt_ids(self.SYSTEM)
        n = len(ids) - 30
        self.put_head(ids, n)
        before = (self.file.read_bytes(), sorted(os.listdir(self.wdir.name)))
        self.start_wh(on=False)
        self.assertFalse(self.engine.warm_heads.enabled)
        time.sleep(1.0)
        self.assertEqual(self.calls, [])
        self.assertEqual(self.commands(), [])
        self.ask(self.SYSTEM)
        self.assertEqual([c[1] for c in self.calls], [16])
        self.assertIsNone(self.engine.warm_heads._thread)
        self.assertEqual(self.engine.warm_heads.saved(), [])
        self.assertEqual((self.file.read_bytes(), sorted(os.listdir(self.wdir.name))), before)

    def test_06_without_the_flag_and_without_a_file_nothing_is_created(self):
        self.start_wh(on=False)
        self.ask(self.SYSTEM)
        self.assertEqual(os.listdir(self.wdir.name), [])
        self.assertIsNone(self.engine.warm_heads._thread)

    def test_07_a_slot_that_already_serves_the_head_is_not_read_again(self):
        ids = self.prompt_ids(self.SYSTEM)
        n = len(ids) - 30
        self.start_wh(on=False)                          # 旗なしで起こし、あとから温めを手で回す
        eng = self.engine
        eng.slot_held[0] = ids[:n] + [TURN] + [7, 8, 9]
        wh = WarmHeads(None, TURN, make_env(self.file))
        two_notes(wh, ids[:n + 1], reused=n)
        wh._warm(eng)
        self.assertIn("skipped: slot 0 serves it", wh.log_lines[-1])
        self.assertEqual(self.calls, [])

    def test_08_one_request_at_a_time_engine_is_not_warmed(self):
        ids = self.prompt_ids(self.SYSTEM)
        n = len(ids) - 30
        self.put_head(ids, n)
        self.start_wh(slots=1)
        self.assertEqual(self.engine.batch, 0)
        self.wait_for("not used")
        self.assertEqual(self.calls, [])
        self.assertEqual(self.commands(), [])

    def test_09_a_head_that_is_reused_twice_is_saved_by_the_thread(self):
        self.start_wh()
        self.wait_for("nothing to warm up")
        wh = self.engine.warm_heads
        ids = ids_of(3, 400)
        self.assertFalse(self.file.exists())
        wh.note(ids, 399, images=False, finish="stop")
        wh.note(ids, 399, images=False, finish="stop")
        t0 = time.time()
        while not self.file.exists() and time.time() - t0 < 5:
            time.sleep(0.05)
        self.assertTrue(self.file.exists())
        self.assertEqual(json.loads(self.file.read_text())["heads"][0]["n"], 399)

    def test_10_a_finished_request_with_a_reuse_is_counted(self):
        """Service._run の口: DONE の reused が note() に渡る(--reuse の偽エンジンは新しい prompt の最初の 3 つを返す)。"""
        self.start_wh(reuse=True, STRATA_WARM_HEADS_MIN=1)
        self.wait_for("nothing to warm up")
        wh = self.engine.warm_heads
        self.ask(self.SYSTEM)
        self.ask(self.SYSTEM)
        t0 = time.time()
        while time.time() - t0 < 5 and not wh.saved():
            time.sleep(0.05)
        self.assertEqual([(h.n, h.reuses) for h in wh.saved()], [(3, 2)])

    def test_11_without_a_reuse_nothing_is_counted(self):
        self.start_wh(STRATA_WARM_HEADS_MIN=1)
        self.wait_for("nothing to warm up")
        self.ask(self.SYSTEM)
        self.ask(self.SYSTEM)
        self.assertEqual(self.engine.warm_heads._mem, {})


if __name__ == "__main__":
    unittest.main()
