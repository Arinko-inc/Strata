"""F20 (2026-11-20): the picture reader's three fixes, tried with a stand-in reader process (no real encoder, no GPU).

  high 5  a GPU reader that did not start leaves the CPU reader running; the GPU reader is started again when no picture
          is being read, and the CPU reader stops once it says READY (never in the middle of a picture)
  high 4  a request's cache files are held until it has made its own file for the engine (the 64-picture race of the
          2026-11-18 review)
  copy    one picture's file for the engine is a hard link to the cache file (a copy when the link is refused); several
          are joined in blocks; nothing is left behind when it fails or the request ends

    python -m pytest -q serve/test_vision_f20.py
"""
import base64
import contextlib
import errno
import io
import json
import os
import queue
import tempfile
import threading
import time
import unittest
import urllib.request
from pathlib import Path
from unittest import mock

import serve.server as server
from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, MockEngine, Service, Vision

ROOT = Path(__file__).resolve().parents[1]


class FakeReader:
    """A `strata-vision` process object: READY (or an error line), then every `ENC <image> <out>` writes <out> in its
    working directory (named after the reader and the image) and answers `OK 3 ...`.  `gate`: an Event the answers wait for."""

    def __init__(self, kind, cwd, ready=True, gate=None):
        self.kind, self.cwd, self.gate = kind, Path(cwd), gate
        self.answers: queue.Queue = queue.Queue()
        self.answers.put("READY 8\n" if ready else "ERR connection to 8797 refused\n")
        self.lines, self.quit, self.dead = [], False, not ready
        self.stdin = self.stdout = self
        self.returncode = 1 if not ready else None

    # stdin
    def write(self, text):
        self.lines.append(text)
        if text.startswith("QUIT"):
            self.quit = True
            self.answers.put("")

    def flush(self):
        if not self.lines or not self.lines[-1].startswith("ENC"):
            return
        _, img, out = self.lines[-1].split()
        data = (self.cwd / img).read_bytes()

        def answer():
            if self.gate is not None:
                self.gate.wait(10)
            (self.cwd / out).write_bytes(f"{self.kind}:".encode() + data)
            self.answers.put("OK 3 1 1 1\n")
        threading.Thread(target=answer, daemon=True).start()

    # stdout
    def readline(self):
        return self.answers.get()

    def poll(self):
        return self.returncode if (self.dead or self.quit) else None

    def kill(self):
        self.dead = True
        self.answers.put("")

    def wait(self, timeout=None):
        return 0


class Farm:
    """The popen stand-in: which readers were started (by kind), and whether the GPU one starts."""

    def __init__(self):
        self.gpu_ok = False
        self.procs: list[FakeReader] = []
        self.gate = None                  # given to the next readers
        self.on_gpu_start = None          # called (in the starting thread) before a GPU reader answers

    def popen(self, what, args, **kw):
        kind = "gpu" if "--gpu" in args else "cpu"
        if kind == "gpu" and self.on_gpu_start is not None:
            self.on_gpu_start()
        proc = FakeReader(kind, kw["cwd"], ready=kind == "cpu" or self.gpu_ok, gate=self.gate)
        self.procs.append(proc)
        return proc

    def kinds(self):
        return [p.kind for p in self.procs]


def vision_cfg(**extra):
    return {"exe": "/x/remote_reader.py", "mmproj": "/x/m.gguf", "model": "/x/g.gguf", "gpu": True,
            "cpu_exe": "/x/strata-vision", **extra}


class VisionTestCase(unittest.TestCase):
    def setUp(self):
        self.farm = Farm()
        patcher = mock.patch.object(server, "popen", self.farm.popen)
        patcher.start()
        self.addCleanup(patcher.stop)
        norm = mock.patch.object(Vision, "normalize", staticmethod(lambda data: data))
        norm.start()
        self.addCleanup(norm.stop)
        self.vision = None

    def make(self, cfg=None, quiet=True):
        out = io.StringIO()
        with contextlib.redirect_stdout(out if quiet else io.StringIO()):
            self.vision = Vision(cfg or vision_cfg())
        self.log = out.getvalue()
        self.addCleanup(self.vision.shutdown)
        return self.vision

    def read(self, v, data):
        """One picture's embeddings file content."""
        path, _ = v.encode(data)
        return path.read_bytes()


class GpuRetry(VisionTestCase):
    def test_a_gpu_reader_that_does_not_start_falls_back_and_says_why(self):
        v = self.make()
        self.assertEqual(self.farm.kinds(), ["gpu", "cpu"])
        self.assertEqual(v.device, "cpu")
        st = v.status()
        self.assertEqual(st["device"], "cpu")
        self.assertEqual(st["last_switch"]["to"], "cpu")
        self.assertIn("8797", st["last_switch"]["reason"])
        self.assertGreater(st["last_switch"]["at"], time.time() - 60)
        self.assertIn("tried again every 60 s", self.log)
        self.assertEqual(self.read(v, b"a"), b"cpu:a")

    def test_a_gpu_reader_that_starts_shows_no_switch(self):
        self.farm.gpu_ok = True
        v = self.make()
        self.assertEqual(v.status(), {"device": "gpu", "last_switch": None})
        self.assertEqual(v.retry_gpu(), "nothing")
        self.assertEqual(self.read(v, b"a"), b"gpu:a")

    def test_the_cpu_reader_by_choice_is_never_retried(self):
        v = self.make({**vision_cfg(), "gpu": False, "cpu_exe": None})
        self.assertEqual(v.status()["device"], "cpu")
        self.assertIsNone(v.status()["last_switch"])
        self.assertEqual(v.retry_gpu(), "nothing")
        self.assertEqual(self.farm.kinds(), ["cpu"])

    def test_the_gpu_comes_back_and_the_cpu_reader_stops(self):
        v = self.make()
        cpu = v.proc
        self.assertEqual(v.retry_gpu(), "failed")                  # the far side is still down
        self.assertEqual((v.device, v.proc), ("cpu", cpu))
        self.assertFalse(cpu.quit)
        self.assertIn("8797", v.status()["gpu_retry_error"])
        self.farm.gpu_ok = True
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(v.retry_gpu(), "gpu")
        self.assertIn("started again", out.getvalue())
        self.assertEqual(v.device, "gpu")
        self.assertTrue(cpu.quit)                                 # asked to QUIT, no picture of its own was cut off
        self.assertIs(v.proc, self.farm.procs[-1])
        st = v.status()
        self.assertEqual((st["device"], st["last_switch"]["to"]), ("gpu", "gpu"))
        self.assertNotIn("gpu_retry_error", st)
        self.assertEqual(self.read(v, b"b"), b"gpu:b")
        self.assertEqual(v.retry_gpu(), "nothing")                # nothing more to come back to

    def test_a_failure_is_said_once_not_every_minute(self):
        v = self.make()
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            for _ in range(4):
                self.assertEqual(v.retry_gpu(), "failed")
        self.assertEqual(out.getvalue().count("still does not start"), 1)

    def test_no_switch_while_a_picture_is_being_read(self):
        v = self.make()
        self.farm.gpu_ok = True
        gate = threading.Event()
        v.proc.gate = gate                                         # the CPU reader's next answer waits
        result = []
        t = threading.Thread(target=lambda: result.append(self.read(v, b"slow")))
        t.start()
        for _ in range(200):
            if v.lock.locked():
                break
            time.sleep(0.01)
        started = len(self.farm.procs)
        self.assertEqual(v.retry_gpu(), "busy")
        self.assertEqual(len(self.farm.procs), started)           # no GPU reader was even started
        self.assertEqual(v.device, "cpu")
        gate.set()
        t.join(10)
        self.assertEqual(result, [b"cpu:slow"])                    # the picture finished on the reader it began on
        self.assertEqual(v.retry_gpu(), "gpu")

    def test_a_picture_that_begins_while_the_gpu_reader_starts_is_finished_first(self):
        v = self.make()
        self.farm.gpu_ok = True
        cpu = v.proc
        gate, result, enc = threading.Event(), [], []
        cpu.gate = gate

        def a_picture_arrives():                                   # in the middle of the GPU reader's start
            enc.append(threading.Thread(target=lambda: result.append(self.read(v, b"late"))))
            enc[0].start()
            for _ in range(200):
                if v.lock.locked():
                    return
                time.sleep(0.01)
        self.farm.on_gpu_start = a_picture_arrives
        got = []
        r = threading.Thread(target=lambda: got.append(v.retry_gpu()))
        r.start()
        time.sleep(0.3)
        self.assertTrue(r.is_alive())                              # the switch waits for the picture
        self.assertEqual((v.device, v.proc), ("cpu", cpu))
        self.assertFalse(cpu.quit)
        gate.set()
        r.join(10)
        enc[0].join(10)
        self.assertEqual(got, ["gpu"])
        self.assertEqual(result, [b"cpu:late"])
        self.assertTrue(cpu.quit)
        self.assertEqual(v.device, "gpu")

    def test_gpu_retry_s_zero_turns_it_off_and_a_bad_value_is_the_default(self):
        v = self.make({**vision_cfg(), "gpu_retry_s": 0})
        self.assertEqual(v.gpu_retry_s, 0)
        self.assertEqual(v.retry_gpu(), "nothing")
        self.assertNotIn("tried again", self.log)
        self.assertEqual(Vision.gpu_retry_of({}), 60.0)
        self.assertEqual(Vision.gpu_retry_of({"gpu_retry_s": 5}), 5.0)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            for bad in (-1, "x", True):
                self.assertEqual(Vision.gpu_retry_of({"gpu_retry_s": bad}), 60.0)
        self.assertEqual(out.getvalue().count("gpu_retry_s"), 3)

    def test_recovery_and_a_second_failure_in_a_row(self):
        v = self.make()
        self.farm.gpu_ok = True
        self.assertEqual(v.retry_gpu(), "gpu")
        self.farm.gpu_ok = False                                   # the far side goes down again: the GPU reader dies
        v.proc.kill()
        with contextlib.redirect_stdout(io.StringIO()):
            v.restart()                                            # what the next picture's start does
        self.assertEqual(v.device, "cpu")
        self.assertEqual(v.status()["last_switch"]["to"], "cpu")
        self.assertEqual(self.read(v, b"c"), b"cpu:c")
        self.farm.gpu_ok = True
        self.assertEqual(v.retry_gpu(), "gpu")
        self.assertEqual(self.read(v, b"d"), b"gpu:d")

    def test_an_unloaded_reader_tries_the_gpu_at_its_next_start(self):
        v = self.make()
        v.unload()
        self.farm.gpu_ok = True
        self.assertEqual(self.farm.kinds(), ["gpu", "cpu"])
        self.assertEqual(v.retry_gpu(), "gpu")                     # nothing is running: no reader is started for it
        self.assertEqual(self.farm.kinds(), ["gpu", "cpu"])
        v.restart()
        self.assertEqual(self.farm.kinds(), ["gpu", "cpu", "gpu"])
        self.assertEqual(v.device, "gpu")

    def test_the_gpu_reader_started_during_an_unload_is_not_kept(self):
        v = self.make()
        self.farm.gpu_ok = True
        self.farm.on_gpu_start = v.unload                          # the elastic idle stop comes while the GPU reader starts
        self.assertEqual(v.retry_gpu(), "gpu")
        self.assertTrue(self.farm.procs[-1].quit)                  # stopped again, not left holding the GPU
        self.assertIsNone(v.proc)
        self.assertTrue(v.stopped)

    def test_the_service_looks_for_the_gpu_while_idle(self):
        v = self.make({**vision_cfg(), "gpu_retry_s": 0.05})
        tok = ByteTokenizer()
        svc = Service(MockEngine(tok, "ok", max_context=4096), tok, ChatTemplate(ROOT / "serve/chat_template.jinja"), vision=v)
        self.farm.gpu_ok = True
        svc.start_vision_gpu_retry()
        for _ in range(100):
            if v.device == "gpu":
                break
            time.sleep(0.05)
        self.assertEqual(v.device, "gpu")

    def test_health_shows_the_device_and_the_switch(self):
        v = self.make()
        tok = ByteTokenizer()
        svc = Service(MockEngine(tok, "ok", max_context=4096), tok, ChatTemplate(ROOT / "serve/chat_template.jinja"), vision=v)
        httpd = server.serve(svc, port=0)
        self.addCleanup(httpd.server_close)
        self.addCleanup(httpd.shutdown)
        url = f"http://127.0.0.1:{httpd.server_address[1]}/health"
        with urllib.request.urlopen(url, timeout=10) as r:
            body = json.load(r)
        self.assertEqual(body["vision"]["device"], "cpu")
        self.assertIn("8797", body["vision"]["last_switch"]["reason"])
        self.assertTrue(body["images"])
        svc.vision = None                                          # a server without pictures: no vision field
        with urllib.request.urlopen(url, timeout=10) as r:
            self.assertNotIn("vision", json.load(r))


def data_url(n):
    return "data:image/png;base64," + base64.b64encode(f"image {n}".encode()).decode()


class Hold(VisionTestCase):
    def cpu_vision(self):
        return self.make({**vision_cfg(), "gpu": False, "cpu_exe": None})

    def test_held_files_survive_another_request_filling_the_cache(self):
        v = self.cpu_vision()
        first = v.encode_all([f"image {i}".encode() for i in range(64)], hold=True)      # the 11-18 review's case
        extra, _ = v.encode(b"another request's one new picture")                          # its encode evicts the oldest
        self.assertTrue(all(p.exists() for p, _ in first))
        self.assertTrue(extra.exists())
        v.release(first)
        v.encode(b"and one more")
        self.assertEqual(len(v.cache), 64)                                                  # back to the limit, oldest first
        self.assertEqual(len(list(v.dir.glob("*.sve"))), 64)

    def test_without_hold_the_old_behaviour_is_the_same(self):
        v = self.cpu_vision()
        first = v.encode_all([f"image {i}".encode() for i in range(64)])
        v.encode(b"extra")
        self.assertFalse(first[0][0].exists())                                              # (what hold= is for)

    def test_two_requests_hold_the_same_picture_twice(self):
        v = self.cpu_vision()
        a = v.encode_all([b"same"], hold=True)
        b = v.encode_all([b"same"], hold=True)
        v.release(a)
        for i in range(70):
            v.encode(f"other {i}".encode())
        self.assertTrue(a[0][0].exists())                                                   # the second request still holds it
        v.release(b)
        v.release(b)                                                                        # a double release is harmless
        v.encode(b"last")
        self.assertFalse(a[0][0].exists())

    def test_a_failed_or_cancelled_request_gives_its_files_back(self):
        v = self.cpu_vision()
        cancel = threading.Event()
        real = v.encode

        def encode(src, keep=(), hold=False):
            out = real(src, keep=keep, hold=hold)
            if src == b"b":
                cancel.set()
            return out
        v.encode = encode
        with self.assertRaises(server.ClientGone):
            v.encode_all([b"a", b"b", b"c"], cancel, hold=True)
        self.assertEqual(v.refs, {})

    def test_the_request_holds_until_its_own_file_is_made(self):
        """The 11-18 reproduction through Service.prepare: another request adds a picture between the encode turn and
        the request's read of the files; with 70 pictures, the first one would be evicted and the read would fail."""
        v = self.cpu_vision()
        tok = ByteTokenizer()
        svc = Service(MockEngine(tok, "ok", max_context=1 << 20), tok, ChatTemplate(ROOT / "serve/chat_template.jinja"), vision=v)
        n = 66
        msgs = [{"role": "user", "content": [{"type": "image", "source": data_url(i)} for i in range(n)]
                 + [{"type": "text", "text": "what are these"}]}]
        real = server.make_combined_embeddings

        def after_another_request(vision_dir, parts):
            v.encode(b"a picture another request adds")            # takes its turn after this request's encode turn
            return real(vision_dir, parts)
        with mock.patch.object(server, "make_combined_embeddings", after_another_request):
            svc.prepare(msgs, None, {})
        want = b"".join(f"cpu:image {i}".encode() for i in range(n))
        self.assertEqual(svc.embeddings.path.read_bytes(), want)
        self.assertEqual(v.refs, {})                                # given back when prepare returned
        svc.drop_embeddings()

    def test_the_hold_is_given_back_when_prepare_refuses(self):
        v = self.cpu_vision()
        tok = ByteTokenizer()
        svc = Service(MockEngine(tok, "ok", max_context=64), tok, ChatTemplate(ROOT / "serve/chat_template.jinja"), vision=v)
        msgs = [{"role": "user", "content": [{"type": "image", "source": data_url(1)}, {"type": "text", "text": "x"}]}]
        with self.assertRaisesRegex(ValueError, "no room|exceeds"):
            svc.prepare(msgs, None, {})
        self.assertEqual(v.refs, {})
        self.assertEqual(list(v.dir.glob("req-*")), [])


class Copy(VisionTestCase):
    def test_one_picture_is_a_hard_link(self):
        v = self.make({**vision_cfg(), "gpu": False, "cpu_exe": None})
        cache, _ = v.encode(b"one")
        path = server.make_combined_embeddings(v.dir, [cache])
        self.assertEqual(path.parent, v.dir)
        self.assertEqual(path.read_bytes(), b"cpu:one")
        self.assertEqual(os.stat(path).st_ino, os.stat(cache).st_ino)
        self.assertEqual(os.stat(cache).st_nlink, 2)
        path.unlink()                                               # the request's end: the cache file stays
        self.assertEqual(cache.read_bytes(), b"cpu:one")

    def test_a_link_that_is_refused_is_a_copy(self):
        v = self.make({**vision_cfg(), "gpu": False, "cpu_exe": None})
        cache, _ = v.encode(b"one")
        with mock.patch.object(os, "link", side_effect=OSError(errno.EXDEV, "cross-device link")):
            path = server.make_combined_embeddings(v.dir, [cache])
        self.assertEqual(path.read_bytes(), b"cpu:one")
        self.assertNotEqual(os.stat(path).st_ino, os.stat(cache).st_ino)
        self.assertEqual(os.stat(cache).st_nlink, 1)
        path.unlink()

    def test_several_pictures_are_joined_in_blocks_not_read_whole(self):
        with tempfile.TemporaryDirectory() as t:
            parts = []
            for i, size in enumerate((3, (3 << 20) + 5, 7)):
                p = Path(t) / f"{i}.sve"
                p.write_bytes(bytes([65 + i]) * size)
                parts.append(p)
            with mock.patch.object(Path, "read_bytes", side_effect=AssertionError("a whole file was read")):
                path = server.make_combined_embeddings(Path(t), parts)
            try:
                self.assertEqual(path.open("rb").read(), b"".join(p.open("rb").read() for p in parts))
            finally:
                path.unlink()

    def test_a_failed_write_leaves_no_file(self):
        with tempfile.TemporaryDirectory() as t:
            good = Path(t) / "good.sve"
            good.write_bytes(b"x" * 100)
            target = Path(t) / "req-target.sve"
            with mock.patch.object(server, "combined_embeddings_path", return_value=target):
                with self.assertRaises(OSError):
                    server.make_combined_embeddings(Path(t), [good, Path(t) / "missing.sve"])
            self.assertFalse(target.exists())
            self.assertEqual(sorted(p.name for p in Path(t).glob("req-*")), [])

    def test_a_request_that_ends_leaves_nothing_in_the_vision_directory(self):
        v = self.make({**vision_cfg(), "gpu": False, "cpu_exe": None})
        tok = ByteTokenizer()
        svc = Service(MockEngine(tok, "ok", max_context=4096), tok, ChatTemplate(ROOT / "serve/chat_template.jinja"), vision=v)
        for n in (1, 3):
            msgs = [{"role": "user", "content": [{"type": "image", "source": data_url(i)} for i in range(n)]
                     + [{"type": "text", "text": "what"}]}]
            svc.prepare(msgs, None, {})
            self.assertEqual(len(list(v.dir.glob("req-*"))), 1)
            svc.drop_embeddings()                                   # the handler's last step, also after a client left
            self.assertEqual(list(v.dir.glob("req-*")), [])
        self.assertEqual(len(list(v.dir.glob("*.sve"))), 3)         # the cache is intact


if __name__ == "__main__":
    unittest.main()
