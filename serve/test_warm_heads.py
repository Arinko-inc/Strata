"""serve/warm_heads.py(再起動後の頭の温め。STRATA_WARM_HEADS=1)の試験。GPU も実エンジンも使わない。

Env / Store / Select は WarmHeads だけを直接使う。ServerPath は serve.test_parallel の偽エンジンの上で、実物の
StrataEngine と Service を動かし、エンジンへ届く命令(BGEN と max_new)を見る。
"""
import base64
import json
import os
import sys
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
            wh.leaving()
            wh.expect()
            self.assertEqual((wh._inflight, wh._expect_at), (0, float("-inf")))
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


class Stub:
    """WarmHeads._warm が触るだけの偽のエンジン(制御の行は持たない)。generate は送った (長さ, max_new) を記録する。"""

    def __init__(self, ctx=4096, info=None, batch=4, busy=False):
        self.gen, self.starting, self.batch, self.max_context = 1, False, batch, ctx
        self.info = {"engine": "0.1.39"} if info is None else info
        self.slot_cv, self.waiting, self.admitting = threading.Condition(), 0, 0
        self.slot_busy = [busy] * batch
        self.ctl = threading.Lock()
        self.keep_heads, self.slot_gs = None, 0
        self.sent = []

    def alive(self):
        return True

    def generate(self, ids, max_new, sampling, cancel):
        self.sent.append((len(ids), max_new))
        started = getattr(cancel, "_strata_queue_started", None)
        if started is not None:
            started()
        yield from ()


class Review(unittest.TestCase):
    """別系統レビュー(2026-11-26 warm-sol1)の指摘の再現。"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "w.json"

    def tearDown(self):
        self.tmp.cleanup()

    def wh(self, **extra):
        w = WarmHeads(None, TURN, make_env(self.path, **extra))
        w._say = lambda line, w=w: w.log_lines.append(line)          # 試験の出力を汚さない
        return w

    def heads_in_file(self):
        return [h["n"] for h in json.loads(self.path.read_text())["heads"]]

    def test_p2_saves_while_the_warm_up_waits_for_an_idle_engine(self):
        """[P2] 温めが空きを待つあいだも、新しく条件を満たした頭を保存する。"""
        a, b = ids_of(1, 200), ids_of(2, 300)
        first = self.wh()
        two_notes(first, a)
        first.save_now()
        wh = self.wh()
        eng = Stub(busy=True)                            # ずっと空かない
        wh.on_ready(eng)
        for _ in range(3):
            wh.note(b, 299, images=False, finish="stop")
        t0 = time.time()
        while time.time() - t0 < 4 and self.heads_in_file() != [299]:
            time.sleep(0.05)
        eng.gen = 2                                      # 待ちを終わらせる
        self.assertEqual(self.heads_in_file(), [299])
        self.assertEqual(eng.sent, [])

    def test_p2_an_expired_last_head_leaves_the_file(self):
        """[P2] 最後の頭が期限切れになったら、ファイルからも消える(実行中)。"""
        wh = self.wh()
        two_notes(wh, ids_of(1, 200))
        wh._loaded = True
        wh._maybe_save()
        self.assertEqual(self.heads_in_file(), [199])
        for h in wh._mem.values():
            h.last_used = time.time() - 8 * 86400
        wh._maybe_save()
        self.assertEqual(self.heads_in_file(), [])

    def test_p2_an_expired_head_in_the_file_is_removed_at_the_start(self):
        """[P2] 起動のとき、ファイルの頭がすでに期限切れなら、ファイルから消える。"""
        wh = self.wh()
        two_notes(wh, ids_of(1, 200))
        wh.save_now()
        data = json.loads(self.path.read_text())
        data["heads"][0]["last_used"] = time.time() - 8 * 86400
        self.path.write_text(json.dumps(data))
        again = self.wh()
        again._ensure_loaded(Stub())
        again._maybe_save()
        self.assertEqual(self.heads_in_file(), [])

    def test_p2_a_save_cut_to_4_mib_is_not_written_again_and_again(self):
        """[P2] 4 MiB に収めるために頭を外した保存のあと、要求が無いのに同じファイルを書き直さない。"""
        wh = self.wh(STRATA_WARM_HEADS_MAX=2)
        two_notes(wh, ids_of(1, 200))
        two_notes(wh, ids_of(2, 200))
        wh._loaded = True
        writes = []
        real = os.replace

        def spy(src, dst):
            writes.append(dst)
            return real(src, dst)
        with mock.patch("serve.warm_heads.FILE_MAX_BYTES", 2000), mock.patch("serve.warm_heads.os.replace", spy):
            wh._maybe_save()
            self.assertEqual(len(self.heads_in_file()), 1)
            for _ in range(3):
                wh._maybe_save()
        self.assertEqual(len(writes), 1)

    def test_p3_a_failed_save_of_counts_is_tried_again(self):
        """[P3] 回数だけが変わった保存が OSError で失敗しても、あとでやり直す。"""
        wh = self.wh()
        ids = ids_of(1, 200)
        two_notes(wh, ids)
        wh._loaded = True
        wh._maybe_save()
        wh.note(ids, 199, images=False, finish="stop")                     # 回数だけ 3 に
        wh._last_save = time.monotonic() - 700
        with mock.patch("serve.warm_heads.os.replace", side_effect=OSError("disk full")):
            wh._maybe_save()
        self.assertEqual(json.loads(self.path.read_text())["heads"][0]["reuses"], 2)
        wh._retry_after = 0.0
        wh._maybe_save()
        self.assertEqual(json.loads(self.path.read_text())["heads"][0]["reuses"], 3)

    def test_p3_a_head_too_long_for_the_context_is_skipped_and_the_next_is_read(self):
        """[P3] 文脈に収まらない頭だけを飛ばし、次の頭は読む。"""
        wh = self.wh(STRATA_WARM_HEADS_MAX=2)
        for _ in range(3):
            wh.note(ids_of(1, 301), 300, images=False, finish="stop")
        two_notes(wh, ids_of(2, 101), reused=100)
        eng = Stub(ctx=250)
        wh._warm(eng)
        self.assertEqual(eng.sent, [(101, 1)])

    def test_p3_the_engine_margin_of_8_tokens_is_counted(self):
        """[P3] エンジンは prompt + max_new + 8 > 文脈 の命令を断る(server.CTX_SLACK)。それを送らない。"""
        wh = self.wh()
        two_notes(wh, ids_of(1, 101), reused=100)
        eng = Stub(ctx=105)                              # 101 + 1 + 8 > 105
        wh._warm(eng)
        self.assertEqual(eng.sent, [])
        eng = Stub(ctx=110)                              # ちょうど収まる
        wh._warm(eng)
        self.assertEqual(eng.sent, [(101, 1)])

    def test_reads_that_finished_do_not_count_toward_tries(self):
        """STRATA_WARM_HEADS_TRIES は止められた読み込みの数。読み終えた頭は数えない(MAX が TRIES より多くても全部読む)。"""
        wh = self.wh(STRATA_WARM_HEADS_MAX=2, STRATA_WARM_HEADS_TRIES=1)
        two_notes(wh, ids_of(1, 101), reused=100)
        two_notes(wh, ids_of(2, 151), reused=150)
        eng = Stub()
        wh._warm(eng)
        self.assertEqual(sorted(eng.sent), [(101, 1), (151, 1)])

    def test_p2_the_qwen36_engine_path_is_not_warmed(self):
        """[P2] Qwen3.6 の経路(INFO に engine= が無い)では温めない(確かめていない経路。ログに 1 行)。"""
        wh = self.wh()
        two_notes(wh, ids_of(1, 200))
        eng = Stub(info={"batch_slots": 4, "slot_cache": 1})
        wh._warm(eng)
        self.assertEqual(eng.sent, [])
        self.assertIn("not used", wh.log_lines[-1])

    def test_p2_requests_and_a_coming_request_keep_it_from_starting(self):
        """[P2] arriving() から leaving() までと、expect() から EXPECT_S 秒は、エンジンが空いていても始めない。
        登録したがまだ送っていない温めは、頭が同じ要求でも止める。"""
        wh = self.wh()
        eng = Stub()
        self.assertTrue(wh._idle(eng))
        wh.arriving(ids_of(1, 10))
        self.assertFalse(wh._idle(eng))
        wh.leaving()
        self.assertTrue(wh._idle(eng))
        wh.expect()
        self.assertFalse(wh._idle(eng))
        with mock.patch.object(WarmHeads, "EXPECT_S", 0.0):
            self.assertTrue(wh._idle(eng))
        from serve.warm_heads import _Active
        head = ids_of(1, 100)
        a = _Active(threading.Event(), head, 100)
        wh._active = a
        wh.arriving(head + [TURN, 5])                    # 同じ頭。まだ送っていない -> 止める
        self.assertTrue(a.cancel.is_set())
        b = _Active(threading.Event(), head, 100, started=True)
        wh._active = b
        wh.arriving(head + [TURN, 5])                    # 送ったあと、同じ頭 -> 止めない
        self.assertFalse(b.cancel.is_set())
        wh.arriving(ids_of(2, 120))                      # 送ったあと、違う頭 -> 止める
        self.assertTrue(b.cancel.is_set())

    def test_p2_replicas_and_split_are_not_warmed(self):
        """[P2] 複数レプリカ・split の構成では温めない(start_replicas が旗を切る。ログに 1 行)。"""
        import serve.server as server
        import serve.warm_heads as whm
        seen = []

        class Fake:
            def __init__(self, *a, **kw):
                self.warm_heads = WarmHeads(None, TURN, make_env(self.path_))
                seen.append(self.warm_heads.enabled)
            path_ = self.path
        try:
            with mock.patch.object(server, "StrataEngine", Fake), \
                    mock.patch.object(server, "engine_args", lambda c: []), \
                    mock.patch.object(server, "child_env", lambda c, offline=False: {}), \
                    mock.patch("builtins.print"):
                server.start_replicas({"args": []}, [[0], [1]], "strata", [], False)
        finally:
            whm.not_here(None)
        self.assertEqual(seen, [False, False])
        self.assertTrue(WarmHeads(None, TURN, make_env(self.path)).enabled)


FAKE_FAST = tp.FAKE_BATCH
FAKE_SLOW = tp.FAKE_BATCH.replace("STEP = 0.02", "STEP = 0.04")      # 塊 1 つの読みが 20 ms: 3900 トークンで約 2.4 秒
# VRAM の命令に答える偽エンジン(読込の途中に来た VRAM は、その読込のあとで答える。本物と同じく制御の行は 1 本)
FAKE_VRAM = FAKE_SLOW.replace('    if line.startswith(("GEN ", "BGEN ")):\n',
                              '    if line.startswith("VRAM"):\n'
                              '        print("VRAM expert_slots=8 vram_free_mib=100", flush=True)\n'
                              '        continue\n'
                              '    if line.startswith(("GEN ", "BGEN ")):\n', 1)
assert FAKE_VRAM != FAKE_SLOW


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

    def warm_again(self, ids, n):
        """今のエンジンのまま、頭を覚えさせて READY をもう一度知らせる(温めをもう 1 回回す)。"""
        wh = self.engine.warm_heads
        two_notes(wh, ids[:n + 1], reused=n)
        wh.on_ready(self.engine)

    def test_12_p1_it_does_not_start_while_the_service_holds_its_turn(self):
        """[P1] Service の番(fifo。ensure_loaded の再起動と VRAM、/v1/vram、/load が持つ)を誰かが持つあいだは、
        温めを始めない(制御の行の応答を取り合わない)。"""
        ids = self.prompt_ids(self.SYSTEM)
        n = len(ids) - 30
        self.start_wh()
        self.wait_for("nothing to warm up")
        with self.svc.fifo:
            self.warm_again(ids, n)
            time.sleep(1.5)
            self.assertEqual(self.commands(), [])
        self.wait_for("read in")
        self.assertEqual(self.calls, [(n + 1, 2)])

    def test_13_p1_a_restart_with_a_vram_reserve_and_a_warm_up_do_not_mix_their_lines(self):
        """[P1] 要求がエンジンを起こし直し、続けて VRAM の予約を戻す(ensure_loaded)。その間に温めが始まると、
        VRAM と BGEN の応答を取り違えて、VRAM は時間切れ、温めは BADM を待ち続けた。"""
        import serve.server as server
        ids = self.prompt_ids(self.SYSTEM)
        n = len(ids) - 30
        self.put_head(ids, n)
        with mock.patch.object(tp, "FAKE_BATCH", FAKE_VRAM):
            self.start_wh()
        self.wait_for("read in")
        self.svc.vram_reserve = 100
        old = self.engine.proc
        old.kill()
        old.wait(timeout=10)
        t0 = time.time()
        while self.engine.alive() and time.time() - t0 < 5:
            time.sleep(0.05)
        self.assertFalse(self.engine.alive())
        got = {}
        real_vram = StrataEngine.vram

        def slow_vram(engine, reserve, timeout=120.0):
            time.sleep(0.8)                              # 再起動のあとの片付け(画像の会話の復元など)の分
            try:
                got["out"] = real_vram(engine, reserve, timeout=5.0)
            except Exception as e:                      # noqa: BLE001
                got["err"] = e
                raise
            return got["out"]
        script = Path(self.tmp.name) / "fake_strata.py"
        real_popen = server.subprocess.Popen
        with mock.patch.object(server.subprocess, "Popen",
                               lambda cmd, **kw: real_popen([sys.executable, str(script), *cmd[1:]], **kw)), \
                mock.patch.object(StrataEngine, "vram", slow_vram):
            self.svc.load()
        self.assertNotIn("err", got)
        self.assertEqual(got["out"].get("expert_slots"), 8)
        t0 = time.time()
        while self.count("read in") < 2 and time.time() - t0 < 20:
            time.sleep(0.05)
        self.assertEqual(self.count("read in"), 2, self.engine.warm_heads.log_lines)
        self.assertEqual(self.ask(self.OTHER), "ok, done.")

    def test_14_p2_a_request_that_came_while_it_decided_goes_first(self):
        """[P2] 空きを確かめてから温めを登録するまでに来た(別の頭の)要求は、温めのあとに回されない。"""
        ids = self.prompt_ids(self.SYSTEM)
        n = len(ids) - 30
        self.start_wh()
        self.wait_for("nothing to warm up")
        fired, arrived = threading.Event(), threading.Event()
        out = {}
        real_wait, real_arriving = WarmHeads._wait_idle, WarmHeads.arriving

        def go():
            t0 = time.time()
            out["reply"] = self.ask(self.OTHER)
            out["took"] = time.time() - t0

        def wait(wh, engine, gen, idle_s):
            ok = real_wait(wh, engine, gen, idle_s)
            if ok and not fired.is_set():                # 空きを確かめた直後に、要求が来る
                fired.set()
                threading.Thread(target=go, daemon=True).start()
                arrived.wait(10)
            return ok

        def arriving(wh, ids_):
            real_arriving(wh, ids_)
            arrived.set()
            t0 = time.time()                             # 温めが先に制御を取れるだけの間を置く(最悪の順)
            while time.time() - t0 < 1.0 and not self.commands():
                time.sleep(0.01)
        with mock.patch.object(WarmHeads, "_wait_idle", wait), mock.patch.object(WarmHeads, "arriving", arriving):
            self.warm_again(ids, n)
            t0 = time.time()
            while "took" not in out and time.time() - t0 < 20:
                time.sleep(0.05)
        self.assertEqual(out.get("reply"), "ok, done.")
        first = self.commands()[0].split()
        self.assertNotEqual(int(first[2]), n + 1, self.commands())     # 最初にエンジンに入ったのは要求
        self.assertLess(out["took"], 2.0)


if __name__ == "__main__":
    unittest.main()
