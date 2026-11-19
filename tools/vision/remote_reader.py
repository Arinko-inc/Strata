#!/usr/bin/env python3
"""A picture reader that reads on another PC: the same stdin/stdout lines as `strata-vision`, with the work done by
tools/vision/reader_server.py on a PC whose GPU is free.  Point the config's vision "exe" at this file; keep "gpu",
"max_tokens" and the rest as for a local reader (they are passed to the far side).

    remote_reader.py [--remote HOST:PORT] --mmproj M --model G [--gpu] [--threads N] [--max-tokens N] [--min-tokens N]

The far side is --remote when it comes first, else the environment's STRATA_VISION_REMOTE (the config's "env"
section reaches the reader).  --mmproj and --model are this PC's paths and are replaced by the far side's own.

Like strata-vision: "READY <n_embd>" once started, then per stdin line "ENC <image> <out>" -> "OK ..." or
"ERR ...", "QUIT" to end.  The image is read here (relative to the working directory, as the server names it) and
the embeddings file is written here before the "OK" line.  The far side's log lines go to stderr.  When the
connection is lost the reader ends with status 1, so the server starts it again.

Standard library only (Python 3.8+).
"""

from __future__ import annotations

import json
import os
import queue
import socket
import sys
import threading

PROTOCOL = 1
CONNECT_S = 10

out_lock = threading.Lock()
pending: queue.Queue = queue.Queue()        # (image, output) of the ENC lines sent and not answered yet
quitting = threading.Event()
answered = threading.Semaphore(0)           # one release per answer to an ENC: the next ENC waits for it


def say(text: str) -> None:
    with out_lock:
        sys.stdout.write(text.rstrip("\r\n") + "\n")
        sys.stdout.flush()


def note(text: str) -> None:
    sys.stderr.write(text.rstrip("\r\n") + "\n")
    sys.stderr.flush()


def die(code: int) -> None:
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    finally:
        os._exit(code)


def parse_enc(line: str):
    """strata-vision's split: the image path ends at the last space before the output path."""
    if not line.startswith("ENC "):
        return None
    rest = line[4:]
    sp = rest.rfind(" ")
    if sp <= 0 or not rest[sp + 1:]:
        return None
    return rest[:sp], rest[sp + 1:]


def write_atomic(path: str, data: bytes) -> None:
    tmp = f"{path}.part{os.getpid()}"
    try:
        with open(tmp, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
    except BaseException:                       # a full disk: the half-written file does not stay beside the cache
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def from_far_side(rfile, where: str) -> None:
    """The far side's messages, until it closes the connection; then this reader ends (status 1 unless QUIT was sent).
    A connection reset in the middle of a frame (an SVE's body or its OK line) ends it the same way: the exception
    must not end only this thread, or the main thread would wait on stdin and the server for its encode timeout."""
    ready = [False]
    try:
        read_frames(rfile, ready)
    except (OSError, ValueError) as e:
        note(f"remote_reader: reading from the picture reader at {where} failed ({e})")
    if quitting.is_set():
        die(0)
    if ready[0]:
        note(f"remote_reader: the connection to the picture reader at {where} was lost")
    die(1)


def read_frames(rfile, ready: list) -> None:
    """Relays the far side's lines and files until the connection ends (returns) or breaks (raises)."""
    while True:
        line = rfile.readline()
        if not line:
            return
        text = line.decode("utf-8", "replace").rstrip("\r\n")
        if text.startswith("LOG "):
            note(text[4:])
            continue
        if text.startswith("SVE "):
            size = int(text[4:])
            data = rfile.read(size)
            if len(data) != size:
                return
            reply = rfile.readline().decode("utf-8", "replace").rstrip("\r\n")
            if not reply:
                return
            try:
                _, out = pending.get_nowait()
            except queue.Empty:
                say("ERR the remote picture reader sent a result nobody asked for")
                continue
            try:
                write_atomic(out, data)
            except OSError as e:
                say(f"ERR cannot write {out} ({e})")
                answered.release()
                continue
            say(reply)
            answered.release()
            continue
        answer = False
        if ready[0] and (text.startswith("OK") or text.startswith("ERR")):
            try:
                img, out = pending.get_nowait()
                # the far side's file names, said as this side's (an ERR names the picture)
                text = text.replace(" in.img", " " + img).replace(" out.sve", " " + out)
                answer = True
            except queue.Empty:
                pass
        if text.startswith("READY"):
            ready[0] = True
        say(text)
        if answer:
            answered.release()


def main(argv: list) -> int:
    where = None
    if len(argv) >= 2 and argv[0] == "--remote":
        where, argv = argv[1], argv[2:]
    where = where or os.environ.get("STRATA_VISION_REMOTE", "").strip()
    if not where:
        say("ERR remote_reader: no picture reader to connect to (set STRATA_VISION_REMOTE=host:port, or --remote host:port)")
        return 2
    host, _, port = where.rpartition(":")
    try:
        sock = socket.create_connection((host, int(port)), timeout=CONNECT_S)
    except (OSError, ValueError) as e:
        say(f"ERR cannot reach the remote picture reader at {where} ({e})")
        return 1
    sock.settimeout(None)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
    for name, value in (("TCP_KEEPIDLE", 10), ("TCP_KEEPINTVL", 5), ("TCP_KEEPCNT", 3)):
        if hasattr(socket, name):
            sock.setsockopt(socket.IPPROTO_TCP, getattr(socket, name), value)
    rfile, wfile = sock.makefile("rb"), sock.makefile("wb")
    send_lock = threading.Lock()

    def send(data: bytes) -> bool:
        with send_lock:
            try:
                wfile.write(data)
                wfile.flush()
                return True
            except OSError:
                return False

    if not send(f"HELLO {PROTOCOL} {json.dumps(argv)}\n".encode("utf-8")):
        say(f"ERR the remote picture reader at {where} closed the connection")
        return 1
    threading.Thread(target=from_far_side, args=(rfile, where), daemon=True).start()

    stdin = sys.stdin.buffer
    while True:
        raw = stdin.readline()
        if not raw:
            break
        line = raw.decode("utf-8", "replace").rstrip("\r\n")
        if line == "QUIT":
            break
        enc = parse_enc(line)
        if enc is None:
            say("ERR expected: ENC <image> <output>")
            continue
        img, out = enc
        try:
            with open(img, "rb") as f:
                data = f.read()
        except OSError:
            say(f"ERR cannot read the image {img}")
            continue
        pending.put((img, out))
        if not send(f"ENC {len(data)}\n".encode() + data):
            die(1)
        # one picture at a time, as the server sends them: the far side reads ahead only a little, and a client that
        # piled pictures into it would also keep it from seeing this side leave while its encoder hangs
        answered.acquire()
    quitting.set()
    send(b"QUIT\n")
    # the far side closes the connection when its strata-vision has ended; from_far_side then exits with 0
    threading.Event().wait(15)
    die(0)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
