#!/usr/bin/env python3
"""The far side of a picture reader on another PC: listens on TCP and, for each connection from
tools/vision/remote_reader.py, starts a real `strata-vision` on this PC's GPU and relays its lines.

    reader_server.py --listen 0.0.0.0:8797 --exe <strata-vision> --mmproj <mmproj.gguf> --model <model.gguf>
                     [--work-dir DIR] [--max-sessions 2]

The relay sends the arguments the server gave it; --mmproj and --model in them are the server PC's paths, so this
side replaces them with its own (--exe/--mmproj/--model above) and passes the other flags (--gpu, --threads,
--max-tokens, --min-tokens, --flash-attn) through.  Which card is used is this process's environment
(CUDA_VISIBLE_DEVICES and so on), set by its service unit.

Wire protocol (one TCP connection = one strata-vision process; no authentication, for a LAN):
  relay -> here   "HELLO 1 <JSON list of arguments>\\n"
                  "ENC <n>\\n" + n bytes of the image (already normalized by the server)
                  "QUIT\\n"
  here -> relay   strata-vision's own stdout lines ("READY <n_embd>", "OK ...", "ERR ...")
                  "SVE <n>\\n" + n bytes of the embeddings file, just before the "OK ..." line it belongs to
                  "LOG <text>\\n" for each stderr line of strata-vision
The connection is closed when strata-vision ends (QUIT, a start that failed, a crash).

Standard library only (Python 3.8+).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import socketserver
import subprocess
import sys
import tempfile
import threading
import time

PROTOCOL = 1
MAX_IMAGE = 256 << 20          # a picture the server normalized is far smaller; this only stops a garbage length
QUIT_WAIT_S = 10

# the reader's flags that take a value and are passed through (the paths are replaced)
VALUE_FLAGS = {"--threads", "--max-tokens", "--min-tokens", "--flash-attn"}
PATH_FLAGS = {"--mmproj", "--model"}


def log(msg: str) -> None:
    print(f"[reader-server] {msg}", file=sys.stderr, flush=True)


def reader_argv(opts, client_args: list) -> tuple[list, list]:
    """-> (argv for strata-vision, notes).  The client's paths are dropped; the rest of its flags are kept."""
    argv = [opts.exe, "--mmproj", opts.mmproj, "--model", opts.model]
    notes = []
    i = 0
    while i < len(client_args):
        a = str(client_args[i])
        if a in PATH_FLAGS:
            theirs = str(client_args[i + 1]) if i + 1 < len(client_args) else ""
            mine = opts.mmproj if a == "--mmproj" else opts.model
            if os.path.basename(theirs) != os.path.basename(mine):
                notes.append(f"{a}: the server asked for {os.path.basename(theirs)}, this side reads {os.path.basename(mine)}")
            i += 2
        elif a in VALUE_FLAGS and i + 1 < len(client_args):
            argv += [a, str(client_args[i + 1])]
            i += 2
        elif a == "--gpu":
            argv.append(a)
            i += 1
        else:
            notes.append(f"ignored argument {a!r}")
            i += 1
    return argv, notes


class Session(socketserver.StreamRequestHandler):
    """One relay connection: one strata-vision."""

    def setup(self):
        super().setup()
        self.request.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.request.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        for name, value in (("TCP_KEEPIDLE", 10), ("TCP_KEEPINTVL", 5), ("TCP_KEEPCNT", 3)):
            if hasattr(socket, name):
                self.request.setsockopt(socket.IPPROTO_TCP, getattr(socket, name), value)
        self.send_lock = threading.Lock()

    def send(self, data: bytes) -> None:
        with self.send_lock:
            self.wfile.write(data)
            self.wfile.flush()

    def send_line(self, text: str) -> None:
        self.send(text.rstrip("\r\n").encode("utf-8", "replace") + b"\n")

    def handle(self):
        peer = f"{self.client_address[0]}:{self.client_address[1]}"
        sem = self.server.sessions
        if not sem.acquire(blocking=False):
            log(f"{peer}: refused, {self.server.opts.max_sessions} sessions already running")
            try:
                self.send_line(f"ERR the remote picture reader on {socket.gethostname()} is busy "
                               f"({self.server.opts.max_sessions} connections already)")
            except OSError:
                pass
            return
        work = None
        proc = None
        try:
            hello = self.rfile.readline().decode("utf-8", "replace").split(" ", 2)
            if len(hello) != 3 or hello[0] != "HELLO" or hello[1] != str(PROTOCOL):
                self.send_line(f"ERR the remote picture reader speaks protocol {PROTOCOL}; this relay does not")
                return
            argv, notes = reader_argv(self.server.opts, json.loads(hello[2]))
            for n in notes:
                log(f"{peer}: {n}")
                self.send_line(f"LOG [reader-server] {n}")
            work = tempfile.mkdtemp(prefix="strata-vision-remote-", dir=self.server.opts.work_dir)
            log(f"{peer}: starting {' '.join(argv)}")
            try:
                proc = subprocess.Popen(argv, cwd=work, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE, bufsize=0)
            except OSError as e:
                log(f"{peer}: strata-vision did not start ({e})")
                self.send_line(f"ERR the remote picture reader could not start strata-vision on {socket.gethostname()} ({e})")
                return
            threading.Thread(target=self.pump_stderr, args=(proc,), daemon=True).start()
            if not self.until_ready(proc):
                return
            self.serve_lines(proc, work, peer)
        except (OSError, ValueError) as e:       # the relay went away, or a bad HELLO
            log(f"{peer}: {e}")
        finally:
            if proc is not None:
                self.stop(proc)
                log(f"{peer}: strata-vision ended ({proc.returncode})")
            if work:
                shutil.rmtree(work, ignore_errors=True)
            sem.release()

    def pump_stderr(self, proc):
        for raw in iter(proc.stderr.readline, b""):
            text = raw.decode("utf-8", "replace").rstrip("\r\n")
            print(f"[strata-vision {proc.pid}] {text}", file=sys.stderr, flush=True)
            try:
                self.send_line("LOG " + text)
            except OSError:
                pass

    def until_ready(self, proc) -> bool:
        """strata-vision's lines up to READY go to the relay as they are.  False when it ended before READY."""
        while True:
            line = proc.stdout.readline()
            if not line:
                return False
            text = line.decode("utf-8", "replace")
            self.send_line(text)
            if text.startswith("READY"):
                return True

    def serve_lines(self, proc, work: str, peer: str) -> None:
        img, out = os.path.join(work, "in.img"), os.path.join(work, "out.sve")
        while True:
            line = self.rfile.readline()
            if not line:                                   # the relay went away
                return
            text = line.decode("utf-8", "replace").rstrip("\r\n")
            if text == "QUIT":
                return
            if not text.startswith("ENC "):
                self.send_line("ERR expected: ENC <image> <output>")
                continue
            size = int(text[4:])
            if not 0 <= size <= MAX_IMAGE:
                raise ValueError(f"an image of {size} bytes")
            data = self.rfile.read(size)
            if len(data) != size:
                return
            t0 = time.monotonic()
            with open(img, "wb") as f:
                f.write(data)
            try:
                proc.stdin.write(b"ENC in.img out.sve\n")
                proc.stdin.flush()
            except OSError:
                return
            reply = proc.stdout.readline()
            if not reply:                                  # strata-vision died under the picture
                return
            reply_text = reply.decode("utf-8", "replace")
            if reply_text.startswith("OK"):
                try:
                    with open(out, "rb") as f:
                        sve = f.read()
                except OSError as e:
                    self.send_line(f"ERR the remote reader's output could not be read ({e})")
                    continue
                self.send(f"SVE {len(sve)}\n".encode() + sve)
            self.send_line(reply_text)
            for p in (img, out):
                try:
                    os.unlink(p)
                except OSError:
                    pass
            log(f"{peer}: {reply_text.strip()} ({(time.monotonic() - t0) * 1000:.0f} ms here)")

    @staticmethod
    def stop(proc) -> None:
        try:
            proc.stdin.write(b"QUIT\n")
            proc.stdin.flush()
            proc.stdin.close()
        except OSError:
            pass
        try:
            proc.wait(timeout=QUIT_WAIT_S)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--listen", default="0.0.0.0:8797", help="address:port to listen on")
    ap.add_argument("--exe", required=True, help="this PC's strata-vision")
    ap.add_argument("--mmproj", required=True, help="this PC's mmproj .gguf")
    ap.add_argument("--model", required=True, help="this PC's text model .gguf (first split)")
    ap.add_argument("--work-dir", default=None, help="where the per-connection folders go (default: the temp dir)")
    ap.add_argument("--max-sessions", type=int, default=2,
                    help="strata-vision processes at once; one more connection is refused with an ERR line")
    opts = ap.parse_args(argv)
    for what in ("exe", "mmproj", "model"):
        setattr(opts, what, os.path.abspath(getattr(opts, what)))      # strata-vision runs in its own folder
        if not os.path.isfile(getattr(opts, what)):
            ap.error(f"--{what} {getattr(opts, what)}: no such file")
    host, _, port = opts.listen.rpartition(":")
    srv = Server((host or "0.0.0.0", int(port)), Session)
    srv.opts = opts
    srv.sessions = threading.BoundedSemaphore(max(1, opts.max_sessions))
    log(f"listening on {opts.listen}: {opts.exe} with {opts.mmproj}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
