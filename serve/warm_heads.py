"""再起動後の温め(Arinko-inc のパッチ、2026-11-26。旗 STRATA_WARM_HEADS=1)。

エンジンを再起動すると、枠(batch slot)と主の履歴にあった長い頭(strata-ask の system と道具の定義、約 5.6 万トークン)は
消え、再起動後の最初の要求がそれを全部読み直す(約 17 秒)。keep_heads.py は、実行中の頭を枠に残す。
このモジュールは、よく再利用された長い頭の ids をディスクに残し、serve がエンジンの READY を受けて、
エンジンが空いたとき、その頭を 1 本だけ読ませて枠に戻す。

- `WarmHeads.note`: 要求が終わったとき(`Service._run` の finally)に呼ばれる。再利用された頭を数えるだけで、I/O をしない
- `WarmHeads.save_now`: 回数の多い頭を MAX 本、JSON 1 つに原子的に書く(WarmHeads のスレッドだけが呼ぶ)
- `WarmHeads.on_ready`: エンジンの READY ごとに呼ばれる(`StrataEngine.__init__` の最後)。スレッドは最初の 1 回で始める
- `WarmHeads.arriving`: 要求がエンジンに入る前に呼ばれる。温めている頭と違う頭の要求が来たら、温めを止める

送る列は 頭 + [TURN_ID]。最後の `<|im_start|>` に turn のチェックポイントが置かれるので、頭の長さちょうどで再利用できる。
keep-heads が使える構成では max_new を 2 にする(1 では枠に写らない。keep_heads.py の held_after_admission)。

旗が無い(STRATA_WARM_HEADS が ちょうど 1 でない)ときは、スレッドもファイルも作らず、note・arriving・on_ready は
何もしない。設定(環境変数。どれも範囲外・数でないときは既定値):
  STRATA_WARM_HEADS_FILE        保存先(既定: エンジンのログと同じディレクトリの warm-heads.json。ログが無ければ
                                ~/.cache/strata/warm-heads.json)
  STRATA_WARM_HEADS_MIN         これより短い頭は覚えない(16384)
  STRATA_WARM_HEADS_REUSES      同じ頭の再利用がこの回数に達したら保存する(2)
  STRATA_WARM_HEADS_MAX         保存する頭の数 1〜4(1)
  STRATA_WARM_HEADS_MAX_TOKENS  1 つの頭の上限(131072)
  STRATA_WARM_HEADS_TTL_DAYS    最後の再利用からこの日数を過ぎた頭は捨てる(7)
  STRATA_WARM_HEADS_IDLE_S      止めた温めをやり直す前に、エンジンが空いている秒数(60)
  STRATA_WARM_HEADS_TRIES       エンジンの起動 1 回あたりの読み込みの上限(3)
"""
from __future__ import annotations

import base64
import binascii
import hashlib
import json
import os
import sys
import tempfile
import threading
import time
from array import array
from dataclasses import dataclass

from serve.auto_prefix import lcp
from serve.keep_heads import served_by

DEFAULT_MIN = 16384
DEFAULT_REUSES = 2
DEFAULT_MAX = 1
DEFAULT_MAX_TOKENS = 131072
DEFAULT_TTL_DAYS = 7
DEFAULT_IDLE_S = 60.0
DEFAULT_TRIES = 3
FILE_VERSION = 1
FILE_MAX_BYTES = 4 << 20
SAVE_EVERY_S = 600.0
MEM_HEADS = 16                  # メモリに持つ頭の数の上限


def env_on(env=None) -> bool:
    """STRATA_WARM_HEADS: ちょうど "1"(前後の空白は可)のときだけ有効。未設定・それ以外は無効。"""
    env = os.environ if env is None else env
    return str(env.get("STRATA_WARM_HEADS", "0")).strip() == "1"


def env_int(env, name: str, default: int, lo: int, hi: int) -> int:
    """整数の設定。未設定・数でない・lo..hi の外 -> default。"""
    env = os.environ if env is None else env
    try:
        v = int(str(env.get(name, "")).strip())
    except ValueError:
        return default
    return v if lo <= v <= hi else default


def _env_float(env, name: str, default: float, lo: float, hi: float) -> float:
    try:
        v = float(str(env.get(name, "")).strip())
    except ValueError:
        return default
    return v if lo <= v <= hi else default


def default_path(log_path: str | None) -> str:
    """保存先の既定: エンジンのログと同じディレクトリの warm-heads.json。ログが無ければ ~/.cache/strata/。"""
    if log_path:
        return os.path.join(os.path.dirname(os.path.abspath(log_path)), "warm-heads.json")
    return os.path.join(os.path.expanduser("~"), ".cache", "strata", "warm-heads.json")


def _le_bytes(ids) -> bytes:
    """ids を little-endian の int32 にしたバイト列。int32 に入らない値・数でないものは OverflowError / TypeError。"""
    a = ids if isinstance(ids, array) and ids.typecode == "i" else array("i", ids)
    if a.itemsize != 4:
        raise ValueError("array('i') is not 4 bytes on this platform")
    if sys.byteorder == "big":
        a = array("i", a)
        a.byteswap()
    return a.tobytes()


def head_key(ids) -> tuple[int, str]:
    """頭の鍵: (長さ, ids を little-endian の int32 にしたバイト列の sha256 の hex)。"""
    return len(ids), hashlib.sha256(_le_bytes(ids)).hexdigest()


@dataclass
class Head:
    n: int
    sha: str
    reuses: int
    last_used: float
    ids: "array"            # array('i')。長さ n

    @property
    def key(self) -> tuple[int, str]:
        return self.n, self.sha


@dataclass
class _Active:
    """今温めている頭(arriving が見る)。"""
    cancel: threading.Event
    head: list
    n: int


class WarmHeads:
    """1 プロセスに 1 つ(StrataEngine が作り、restart をまたいで残す)。"""

    POLL = 0.2              # エンジンが空くのを見る間隔(秒)
    TICK = 1.0              # 保存の要否を見る間隔(秒)

    def __init__(self, log_path: str | None, turn_id: int, env=None):
        env = os.environ if env is None else env
        self.enabled = env_on(env)
        self.turn_id = int(turn_id)
        self.path = str(env.get("STRATA_WARM_HEADS_FILE", "")).strip() or default_path(log_path)
        self.min_tokens = env_int(env, "STRATA_WARM_HEADS_MIN", DEFAULT_MIN, 1, 1 << 30)
        self.reuses = env_int(env, "STRATA_WARM_HEADS_REUSES", DEFAULT_REUSES, 1, 1 << 30)
        self.max_heads = env_int(env, "STRATA_WARM_HEADS_MAX", DEFAULT_MAX, 1, 4)
        self.max_tokens = env_int(env, "STRATA_WARM_HEADS_MAX_TOKENS", DEFAULT_MAX_TOKENS, 1, 1 << 30)
        self.ttl_days = env_int(env, "STRATA_WARM_HEADS_TTL_DAYS", DEFAULT_TTL_DAYS, 1, 36500)
        self.idle_s = _env_float(env, "STRATA_WARM_HEADS_IDLE_S", DEFAULT_IDLE_S, 0.0, 3600.0)
        self.tries = env_int(env, "STRATA_WARM_HEADS_TRIES", DEFAULT_TRIES, 1, 100)
        self.log_lines: list[str] = []          # 試験用: 出したログの行(最後の 200 行)
        self.last_problem: str | None = None    # 試験用: load が使えないと見た理由
        self._lock = threading.Lock()           # _mem・_active・_dirty
        self._mem: dict[tuple[int, str], Head] = {}
        self._active: _Active | None = None
        self._dirty = False
        self._cv = threading.Condition()
        self._thread: threading.Thread | None = None
        self._engine = None
        self._ready_seq = 0
        self._loaded = False
        self._n_vocab: int | None = None
        self._saved_sig: tuple | None = None
        self._last_save = 0.0
        self._retry_after = 0.0
        self._attempts = 0

    # ---- ログ ----
    def _say(self, line: str) -> None:
        text = f"warm heads: {line}"
        self.log_lines.append(text)
        del self.log_lines[:-200]
        print(f"[strata] {text}", flush=True)

    # ---- 覚える(要求のスレッドから。I/O なし) ----
    def note(self, ids, reused, images: bool, finish) -> None:
        """要求の終わり: reused >= MIN の頭を数える。画像つき・finish が error・数でない reused は覚えない。
        メモリの更新だけをして、ファイルには書かない。要求の finally から呼ばれるので、例外は外に出さない。"""
        if not self.enabled or images or finish == "error":
            return
        try:
            self._note(ids, reused)
        except Exception:                               # noqa: BLE001 - 数え損ねても要求は止めない
            pass

    def _note(self, ids, reused) -> None:
        if not isinstance(reused, int) or isinstance(reused, bool):
            return
        n = min(reused, len(ids) - 1)
        if n < self.min_tokens or n > self.max_tokens:
            return
        try:
            arr = array("i", ids[:n])
            key = (n, hashlib.sha256(_le_bytes(arr)).hexdigest())
        except (OverflowError, TypeError, ValueError):
            return
        now = time.time()
        with self._lock:
            h = self._mem.get(key)
            if h is not None:
                h.reuses += 1
                h.last_used = now
            else:
                self._mem[key] = Head(n, key[1], 1, now, arr)
                self._evict_locked()
            self._dirty = True

    def _saved_locked(self) -> list[Head]:
        now = time.time()
        ttl = self.ttl_days * 86400.0
        c = [h for h in self._mem.values() if h.reuses >= self.reuses and now - h.last_used <= ttl]
        c.sort(key=lambda h: (-h.reuses, -h.last_used))
        return c[:self.max_heads]

    def saved(self) -> list[Head]:
        """試験用: 今保存の対象の頭(回数 >= REUSES かつ TTL 以内の、回数の多い順に MAX 本)。"""
        with self._lock:
            return self._saved_locked()

    def _evict_locked(self) -> None:
        if len(self._mem) <= MEM_HEADS:
            return
        keep = {h.key for h in self._saved_locked()}        # 保存の対象は捨てない
        for key in sorted((k for k in self._mem if k not in keep),
                          key=lambda k: (self._mem[k].reuses, self._mem[k].last_used)):
            if len(self._mem) <= MEM_HEADS:
                break
            del self._mem[key]

    # ---- ファイル ----
    def _bad(self, why: str) -> list[Head]:
        self.last_problem = why
        bad = self.path + ".bad"
        try:
            os.replace(self.path, bad)
        except OSError:
            pass
        self._say(f"{self.path} is unusable ({why}); moved to {bad}, no warm-up this start")
        return []

    def load(self, n_vocab: int | None) -> list[Head]:
        """ファイルから頭を読む。無ければ []。中身が正しくなければ <FILE>.bad に名前を変えて [](例外は出さない)。
        turn_id か n_vocab が今のエンジンと違うファイルは、使わずに残して []。"""
        self.last_problem = None
        try:
            size = os.stat(self.path).st_size
        except FileNotFoundError:
            return []
        except OSError as e:
            self._say(f"cannot read {self.path}: {e}; no warm-up this start")
            return []
        if size > FILE_MAX_BYTES:
            return self._bad(f"larger than {FILE_MAX_BYTES >> 20} MiB")
        try:
            with open(self.path, "rb") as f:
                raw = f.read(FILE_MAX_BYTES + 1)
        except OSError as e:
            self._say(f"cannot read {self.path}: {e}; no warm-up this start")
            return []
        if len(raw) > FILE_MAX_BYTES:
            return self._bad(f"larger than {FILE_MAX_BYTES >> 20} MiB")
        try:
            data = json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return self._bad("not valid JSON")
        if not isinstance(data, dict) or data.get("version") != FILE_VERSION:
            return self._bad("unknown version")
        if data.get("turn_id") != self.turn_id:
            self._say(f"{self.path} was written for another tokenizer (turn_id {data.get('turn_id')}); "
                      "not used, replaced at the next save")
            return []
        file_vocab = data.get("n_vocab")
        if n_vocab is not None and file_vocab is not None and file_vocab != n_vocab:
            self._say(f"{self.path} was written for another vocabulary (n_vocab {file_vocab}); "
                      "not used, replaced at the next save")
            return []
        try:
            rows = data["heads"]
            if not isinstance(rows, list):
                return self._bad("heads is not a list")
            out = []
            for r in rows:
                n, sha = r["n"], r["sha256"]
                if not (isinstance(n, int) and not isinstance(n, bool) and n > 0 and isinstance(sha, str)):
                    return self._bad("a head has a wrong field")
                reuses, last = r["reuses"], r["last_used"]
                if isinstance(reuses, bool) or not isinstance(reuses, int) or isinstance(last, bool) \
                        or not isinstance(last, (int, float)):
                    return self._bad("a head has a wrong field")
                try:
                    blob = base64.b64decode(r["ids"], validate=True)
                except (binascii.Error, ValueError, TypeError):
                    return self._bad("ids is not base64")
                if len(blob) != n * 4:
                    return self._bad("ids does not match the length")
                a = array("i")
                if a.itemsize != 4:
                    return self._bad("this platform has no 4-byte array")
                a.frombytes(blob)
                if sys.byteorder == "big":
                    a.byteswap()
                if min(a) < 0 or (n_vocab is not None and max(a) >= n_vocab):
                    return self._bad("an id is out of range")
                if hashlib.sha256(_le_bytes(a)).hexdigest() != sha:
                    return self._bad("sha256 does not match")
                out.append(Head(n, sha, reuses, float(last), a))
            return out
        except (KeyError, TypeError, ValueError, AttributeError) as e:
            return self._bad(f"unreadable ({type(e).__name__})")

    def save_now(self) -> None:
        """保存の対象の頭を JSON 1 つに原子的に書く(一時ファイル -> fsync -> os.replace。権限 0600)。
        WarmHeads のスレッドから呼ぶ。失敗しても例外は出さず、60 秒あとにやり直す。"""
        with self._lock:
            heads = self._saved_locked()
            self._dirty = False
        sig = tuple(h.key for h in heads)
        if not heads:
            return
        while True:
            payload = {"version": FILE_VERSION, "turn_id": self.turn_id, "n_vocab": self._n_vocab,
                       "saved_at": int(time.time()),
                       "heads": [{"n": h.n, "sha256": h.sha, "reuses": h.reuses, "last_used": h.last_used,
                                  "ids": base64.b64encode(_le_bytes(h.ids)).decode("ascii")} for h in heads]}
            text = json.dumps(payload, separators=(",", ":")).encode("utf-8")
            if len(text) <= FILE_MAX_BYTES or len(heads) <= 1:
                break
            heads = heads[:-1]                  # 4 MiB に収まるまで、回数の少ない頭から外す
            sig = tuple(h.key for h in heads)
        if len(text) > FILE_MAX_BYTES:
            self._say(f"not saved: one head is larger than {FILE_MAX_BYTES >> 20} MiB as a file")
            self._retry_after = time.monotonic() + 3600
            return
        tmp = None
        try:
            d = os.path.dirname(os.path.abspath(self.path))
            os.makedirs(d, mode=0o700, exist_ok=True)
            fd, tmp = tempfile.mkstemp(prefix=".warm-heads-", suffix=".tmp", dir=d)
            with os.fdopen(fd, "wb") as f:
                f.write(text)
                f.flush()
                os.fsync(f.fileno())
            os.chmod(tmp, 0o600)
            os.replace(tmp, self.path)
            tmp = None
        except OSError as e:
            self._say(f"could not save {self.path}: {e}")
            self._retry_after = time.monotonic() + 60
            return
        finally:
            if tmp is not None:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
        self._saved_sig = sig
        self._last_save = time.monotonic()
        self._say(f"saved {len(heads)} head(s) to {self.path}")

    def _maybe_save(self) -> None:
        """保存の対象の顔ぶれが変わったときはすぐ、回数だけが変わったときは SAVE_EVERY_S 秒たってから書く。"""
        if not self._loaded or time.monotonic() < self._retry_after:
            return
        with self._lock:
            sig = tuple(h.key for h in self._saved_locked())
            dirty = self._dirty
        if not sig:
            return
        if sig != self._saved_sig or (dirty and time.monotonic() - self._last_save >= SAVE_EVERY_S):
            self.save_now()

    def _ensure_loaded(self, engine) -> None:
        if self._loaded:
            return
        nv = (getattr(engine, "info", None) or {}).get("n_vocab")
        self._n_vocab = nv if isinstance(nv, int) and not isinstance(nv, bool) and nv > 0 else None
        heads = self.load(self._n_vocab)
        with self._lock:
            for h in heads:
                cur = self._mem.get(h.key)
                if cur is None:
                    self._mem[h.key] = h
                else:
                    cur.reuses = max(cur.reuses, h.reuses)
                    cur.last_used = max(cur.last_used, h.last_used)
            self._evict_locked()
            self._saved_sig = tuple(x.key for x in self._saved_locked())
        self._last_save = time.monotonic()
        self._loaded = True

    # ---- 温める ----
    def on_ready(self, engine) -> None:
        """エンジンの READY ごとに呼ぶ(StrataEngine.__init__ の最後)。旗が無ければ何もしない。"""
        if not self.enabled:
            return
        with self._cv:
            self._engine = engine
            self._ready_seq += 1
            if self._thread is None:
                self._thread = threading.Thread(target=self._loop, name="warm-heads", daemon=True)
                self._thread.start()
            self._cv.notify_all()

    def arriving(self, ids) -> None:
        """要求がエンジンに入る前に呼ぶ。温めている頭で始まらない要求なら、温めを止める(頭が同じなら止めない:
        その要求は温めの残りを待ち、枠か主の履歴から頭を再利用する)。"""
        if not self.enabled:
            return
        with self._lock:
            a = self._active
        if a is None or a.cancel.is_set():
            return
        try:
            if lcp(ids if isinstance(ids, list) else list(ids), a.head) < a.n:
                a.cancel.set()
        except Exception:                               # noqa: BLE001 - 判定できなければ止めない
            pass

    def _loop(self) -> None:
        seen = 0
        while True:
            with self._cv:
                if self._ready_seq == seen:
                    self._cv.wait(timeout=self.TICK)
                new = self._ready_seq != seen
                seen = self._ready_seq
                engine = self._engine
            try:
                if new and engine is not None:
                    self._ensure_loaded(engine)
                self._maybe_save()
                if new and engine is not None:
                    self._warm(engine)
                    self._maybe_save()
            except Exception as e:                      # noqa: BLE001 - 温めの失敗でサーバーを止めない
                self._say(f"failed: {type(e).__name__}: {e}")

    def _idle(self, engine) -> bool:
        with engine.slot_cv:
            if engine.waiting or engine.admitting or any(engine.slot_busy):
                return False
        return not engine.ctl.locked()

    def _wait_idle(self, engine, gen: int, idle_s: float) -> bool:
        """エンジンが idle_s 秒つづけて空くのを待つ。世代が変わった・エンジンが死んだときは False。"""
        since = None
        while True:
            if engine.gen != gen:
                return False
            starting = getattr(engine, "starting", False)
            if not starting and not engine.alive():
                return False
            if not starting and self._idle(engine):
                now = time.monotonic()
                since = now if since is None else since
                if now - since >= idle_s:
                    return True
            else:
                since = None
            time.sleep(self.POLL)

    def _warm(self, engine) -> None:
        gen = engine.gen
        self._attempts = 0
        if not getattr(engine, "batch", 0):
            self._say("not used: the engine runs one request at a time (no --batch)")
            return
        heads = self.saved()
        if not heads:
            self._say("nothing to warm up (no saved head)")
            return
        for i, h in enumerate(heads, 1):
            r = self._warm_one(engine, gen, h, i, len(heads))
            if r in ("gone", "failed"):
                return

    def _warm_one(self, engine, gen: int, h: Head, idx: int, total: int) -> str:
        head = list(h.ids)
        w = head + [self.turn_id]
        if h.n + 1 > int(getattr(engine, "max_context", 0) or 0) - 1:
            self._say(f"failed: the {h.n}-token head does not fit the engine's context ({engine.max_context})")
            return "failed"
        kh = getattr(engine, "keep_heads", None)
        applies = bool(kh is not None and kh.applies(engine.info or {}, getattr(engine, "slot_gs", 0)))
        first = True
        while True:
            if not self._wait_idle(engine, gen, 0.0 if first else self.idle_s):
                return "gone"
            first = False
            if applies:
                with engine.slot_cv:
                    for b in range(engine.batch):
                        if engine.slot_busy[b]:
                            continue
                        if served_by(w, engine.slot_held[b], kh.meta(b), self.turn_id, kh.root_min) >= h.n:
                            self._say(f"skipped: slot {b} serves it")
                            return "skipped"
            if self._attempts >= self.tries:
                self._say(f"giving up: {self._attempts} reads were stopped in this engine start")
                return "failed"
            self._attempts += 1
            r = self._read(engine, gen, h, w, 2 if applies else 1, idx, total)
            if r != "stopped" or not applies:
                return r
            self._say(f"will try again when the engine has been idle {self.idle_s:g} s")

    def _read(self, engine, gen: int, h: Head, w: list, max_new: int, idx: int, total: int) -> str:
        cancel = threading.Event()
        started = [False]
        cancel._strata_queue_started = lambda: started.__setitem__(0, True)     # type: ignore[attr-defined]
        with self._lock:
            self._active = _Active(cancel, w[:-1], h.n)
        self._say(f"reading a {h.n}-token head ({idx} of {total})")
        t0 = time.monotonic()
        g = None
        try:
            g = engine.generate(w, max_new, {}, cancel)
            for _ in g:
                pass
        except Exception as e:                          # noqa: BLE001 - EngineDied など。この世代では止める
            if not cancel.is_set():
                self._say(f"failed: {type(e).__name__}: {e}")
                return "failed"
        finally:
            with self._lock:
                self._active = None
            if g is not None:
                try:
                    g.close()
                except Exception:                       # noqa: BLE001
                    pass
        if cancel.is_set():
            self._say("stopped: a request with another head came")
            return "stopped"
        self._say(f"read in {time.monotonic() - t0:.1f} s")
        return "done"
