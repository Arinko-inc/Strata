"""Tests of tools/serve_load.py's exit status, against a fake OpenAI-compatible HTTP server (a thread).

    ~/strata-wt/.venv-f20/bin/python -m pytest -q tools/test_serve_load.py
"""
import json, os, re, sys, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import serve_load  # noqa: E402


def sse(obj):
    return b"data: " + json.dumps(obj).encode() + b"\n\n"


def piece(text):
    return sse({"choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}]})


def last(finish="stop", tokens=3, usage=True):
    j = {"choices": [{"index": 0, "delta": {}, "finish_reason": finish}]}
    if usage:
        j["usage"] = {"prompt_tokens": 5, "completion_tokens": tokens, "total_tokens": 5 + tokens}
    return sse(j)


# mode -> (status, body); `bad` is the request index that gets the mode, the other requests answer normally
def normal_body(i):
    return piece(f"hello {i} ") + piece("world") + last() + b"data: [DONE]\n\n"


MODES = {
    "ok": lambda i: (200, normal_body(i)),
    "http_error": lambda i: (500, b'{"error": "boom"}'),
    "no_done": lambda i: (200, piece("hi") + last()),
    "stream_error": lambda i: (200, sse({"error": {"message": "engine died"}}) + b"data: [DONE]\n\n"),
    "no_usage": lambda i: (200, piece("hi") + last(usage=False) + b"data: [DONE]\n\n"),
    "zero_tokens": lambda i: (200, last(tokens=0) + b"data: [DONE]\n\n"),
    "no_finish": lambda i: (200, piece("hi") + sse({"usage": {"completion_tokens": 2}}) + b"data: [DONE]\n\n"),
    "bad_finish": lambda i: (200, piece("hi") + last(finish="error") + b"data: [DONE]\n\n"),
}


class Fake(BaseHTTPRequestHandler):
    mode = "ok"        # set per test
    bad = None         # index of the request that gets `mode` (None = all of them)

    def log_message(self, *a):
        pass

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n))
        i = int(re.search(r"\[request (\d+)", body["messages"][0]["content"]).group(1))
        mode = type(self).mode if type(self).bad in (None, i) else "ok"
        status, data = MODES[mode](i)
        self.send_response(status)
        self.send_header("Content-Type", "text/event-stream" if status == 200 else "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@pytest.fixture
def server():
    class H(Fake):
        pass
    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield H, f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()
    srv.server_close()


def run(url, capsys, monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["serve_load.py", url, "--clients", "2", "--rounds", "1", "--nonce", "t", *args])
    rc = serve_load.main()
    return rc, capsys.readouterr().out


def lines(out):
    return out.strip().splitlines()[-2:]


def test_normal(server, capsys, monkeypatch):
    H, url = server
    rc, out = run(url, capsys, monkeypatch)
    assert rc == 0
    assert lines(out) == ["result: 2/2 requests ok", "failures: none"]
    assert "clients  total tok/s" in out          # the table stays


def test_one_request_error(server, capsys, monkeypatch):
    H, url = server
    H.mode, H.bad = "http_error", 1
    rc, out = run(url, capsys, monkeypatch)
    assert rc == 1
    assert lines(out)[0] == "result: 1/2 requests ok"
    assert "request_error=1" in lines(out)[1]


def test_all_requests_fail_even_with_empty_compare(server, capsys, monkeypatch, tmp_path):
    H, url = server
    H.mode = "http_error"
    rc, out = run(url, capsys, monkeypatch, "--compare", str(tmp_path))
    assert rc == 1
    assert "compare_missing=2" in lines(out)[1]
    assert "ALL IDENTICAL" not in out


def test_no_done(server, capsys, monkeypatch):
    H, url = server
    H.mode, H.bad = "no_done", 0
    rc, out = run(url, capsys, monkeypatch)
    assert rc == 1
    assert lines(out)[1] == "failures: no_done=1"


def test_compare_file_missing(server, capsys, monkeypatch, tmp_path):
    H, url = server
    save = tmp_path / "ref"
    rc, _ = run(url, capsys, monkeypatch, "--save", str(save))
    assert rc == 0
    rc, out = run(url, capsys, monkeypatch, "--compare", str(save))
    assert rc == 0 and "ALL IDENTICAL" in out
    os.remove(save / "c2" / "r0" / "1.txt")
    rc, out = run(url, capsys, monkeypatch, "--compare", str(save))
    assert rc == 1
    assert lines(out)[1] == "failures: compare_missing=1"
    assert "ALL IDENTICAL" not in out


def test_compare_difference(server, capsys, monkeypatch, tmp_path):
    H, url = server
    save = tmp_path / "ref"
    run(url, capsys, monkeypatch, "--save", str(save))
    (save / "c2" / "r0" / "0.txt").write_text("other", encoding="utf-8")
    rc, out = run(url, capsys, monkeypatch, "--compare", str(save))
    assert rc == 1
    assert lines(out)[1] == "failures: diff=1"


def test_stream_error_then_done(server, capsys, monkeypatch):
    H, url = server
    H.mode = "stream_error"
    rc, out = run(url, capsys, monkeypatch)
    assert rc == 1
    f = lines(out)[1]
    assert "stream_error=2" in f and "no_finish=2" in f and "no_usage=2" in f
    assert lines(out)[0] == "result: 0/2 requests ok"
    assert " 2\n" in out                           # the table's errors column counts them (not "no errors")


@pytest.mark.parametrize("mode,name", [("no_usage", "no_usage"), ("zero_tokens", "zero_tokens"),
                                       ("no_finish", "no_finish"), ("bad_finish", "bad_finish")])
def test_usage_and_finish(server, capsys, monkeypatch, mode, name):
    H, url = server
    H.mode, H.bad = mode, 0
    rc, out = run(url, capsys, monkeypatch)
    assert rc == 1
    assert f"{name}=1" in lines(out)[1]


def test_length_and_tool_calls_are_fine(server, capsys, monkeypatch):
    H, url = server
    MODES["len"] = lambda i: (200, piece("a") + last(finish="length", tokens=1) + b"data: [DONE]\n\n")
    MODES["tc"] = lambda i: (200, piece("a") + last(finish="tool_calls", tokens=1) + b"data: [DONE]\n\n")
    try:
        for m in ("len", "tc"):
            H.mode = m
            rc, out = run(url, capsys, monkeypatch)
            assert rc == 0, out
    finally:
        del MODES["len"], MODES["tc"]


def test_count_mismatch(server, capsys, monkeypatch):
    H, url = server
    orig = serve_load.one

    def lossy(a, k, m, i, *rest):
        if i == 1:
            return                                  # a request whose result never arrives
        orig(a, k, m, i, *rest)
    monkeypatch.setattr(serve_load, "one", lossy)
    rc, out = run(url, capsys, monkeypatch)
    assert rc == 1
    assert lines(out)[0] == "result: 1/2 requests ok"
    assert lines(out)[1] == "failures: count_mismatch=1"


def test_connection_refused(capsys, monkeypatch):
    rc, out = run("http://127.0.0.1:1", capsys, monkeypatch)
    assert rc == 1
    assert "request_error=2" in lines(out)[1]
