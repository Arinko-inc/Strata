// src/net/stage_link_deferred_test.cpp - L1's deferred windows over the remote stage's link (protocol 8's optional
// bits: SlotRun flag kStageSlotDeferred, SlotCommit b = kStageCommitDeferred, the hello's kStageBatchDeferred), on
// loopback, no GPU.  strata-mesh IMPL-PLAN-2026-11-26-batch-drafts.md 3.2 (the link's part; the worker's table of held
// windows is stage_deferred_test.cpp, protocol 8 itself stage_link_batch_test.cpp).
//   0. the constants;
//   1. kStageBatchDeferred reaches the main process only from a worker with slot_commit_deferred (serve_stage clears it
//      otherwise, whatever the hello handler wrote);
//   2. a worker without slot_commit_deferred: a kStageSlotDeferred window is "bad slot window", a deferred commit is the
//      next SlotRun's "bad slot commit", and the link goes on;
//   3. the main side refuses flags 3 (kStageSlotKeepAll | kStageSlotDeferred) and 4 before anything goes out;
//   4. a worker with it (answering at once, as the end): two deferred windows, committed by tag in the other order -
//      slot_run sees flag 2, slot_commit_deferred gets each tag and keep[] whole, slot_commit is not called; a plain
//      SlotCommit (b = 0) still goes to slot_commit;
//   5. slot_commit_deferred's failure (an unknown tag) is the next SlotRun's error, and the link goes on;
//   6. a worker that answers late (a relay): a deferred window is answered late, and the commit of the window before it
//      is served while this window's reply is still held (it does not wait for the late replies);
//   7. a hand-spoken SlotRun with flags 3, and a SlotCommit with b = 2, are refused by a worker with the handler.
#include "strata/net/stage_link.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>
#endif

using namespace strata::net;

namespace {
int g_fail = 0;
void check(bool ok, const char* what) {
    std::printf("  %-78s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}
bool contains(const std::string& s, const std::string& part) { return s.find(part) != std::string::npos; }

constexpr int64_t kHbf = 4, kD = 4, kChunk = 8, kMaxT = 8;
StageHello hello_of() {
    StageHello h;
    h.n_embd = 4;
    h.hc = 1;
    h.n_layers = 4;
    h.n_expert = 2;
    h.layer_begin = 2;
    h.max_t = (int32_t) kMaxT;
    h.max_context = 64;
    h.chunk = kChunk;
    h.handoff_floats = kHbf;
    h.pack_hash = 0x5354524d;
    return h;
}

// ---- a main process's end spoken by hand (7.)
#if defined(_WIN32)
using sock = SOCKET;
constexpr sock kNone = INVALID_SOCKET;
void close_sock(sock s) { (void) ::closesocket(s); }
#else
using sock = int;
constexpr sock kNone = -1;
void close_sock(sock s) { (void) ::close(s); }
#endif
void send_n(sock fd, const void* p, size_t n) {
#if defined(_WIN32)
    (void) ::send(fd, (const char*) p, (int) n, 0);
#else
    (void) ::send(fd, p, n, MSG_NOSIGNAL);
#endif
}
bool recv_n(sock fd, void* p, size_t n) {
    char* c = (char*) p;
    while (n > 0) {
        const long long r = ::recv(fd, c, (int) n, 0);
        if (r <= 0) return false;
        c += r;
        n -= (size_t) r;
    }
    return true;
}
struct Raw {
    sock s = kNone;
    bool open(const std::string& where) {
        sockaddr_in a{};
        a.sin_family = AF_INET;
        a.sin_port = htons((unsigned short) std::atoi(where.substr(where.rfind(':') + 1).c_str()));
        a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        s = ::socket(AF_INET, SOCK_STREAM, 0);
        if (s == kNone || ::connect(s, (const sockaddr*) &a, sizeof a) != 0) return false;
#if defined(_WIN32)
        const DWORD ms = 5000;
        (void) setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, (const char*) &ms, sizeof ms);
#else
        timeval tv{};
        tv.tv_sec = 5;
        (void) setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);
#endif
        return true;
    }
    void header(StageMsg type, int64_t a, int64_t b, int64_t c, uint64_t bytes) {
        StageHeader h;
        h.type = (uint32_t) type;
        h.a = a;
        h.b = b;
        h.c = c;
        h.bytes = bytes;
        send_n(s, &h, sizeof h);
    }
    bool reply(StageHeader& r, std::string& payload) {
        if (!recv_n(s, &r, sizeof r)) return false;
        payload.assign((size_t) std::min<uint64_t>(r.bytes, (uint64_t) 1 << 20), '\0');
        return recv_n(s, payload.data(), payload.size());
    }
    bool hello(const StageHello& mine) {
        header(StageMsg::Hello, 0, 0, 0, sizeof mine);
        send_n(s, &mine, sizeof mine);
        StageHeader r;
        std::string p;
        return reply(r, p) && r.type == (uint32_t) StageMsg::HelloOk;
    }
    // an fp32 SlotRun of S rows on slots 0.. (positions 0: the fake handlers do not check them)
    void slot_run(int S, int64_t flags, int64_t tag) {
        std::vector<char> body((size_t) (S * 16 + S * kHbf * 4), 0);
        for (int t = 0; t < S; ++t) std::memcpy(body.data() + t * 4, &t, 4);
        header(StageMsg::SlotRun, S, flags, tag, body.size());
        send_n(s, body.data(), body.size());
    }
    ~Raw() {
        if (s != kNone) close_sock(s);
    }
};

// one worker on a free loopback port, until stop() (as stage_link_batch_test's)
struct Worker {
    std::atomic<bool> stop_flag{false};
    std::atomic<int> served{-1};
    StageStats st;
    std::thread t;
    std::string where;
    bool start(const StageHandlers& h, const StageBuffers& buf) {
        std::mt19937 rng(std::random_device{}());
        for (int attempt = 0; attempt < 8 && where.empty(); ++attempt) {
            const int port = 20000 + (int) (rng() % 40000);
            served = -1;
            t = std::thread([&, port] { served = serve_stage("127.0.0.1", port, "", h, buf, st, &stop_flag); });
            const std::string at = "127.0.0.1:" + std::to_string(port);
            for (int i = 0; i < 100 && served.load() < 0; ++i) {
                StageHello bad = hello_of();
                bad.protocol = 4;
                StageHello peer;
                std::string e;
                StageClient c;
                if (c.connect(at, bad, peer, e) || contains(e, "the worker refused:")) { where = at; break; }
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
            }
            if (where.empty()) stop();
        }
        return !where.empty();
    }
    void stop() {
        if (!t.joinable()) return;
        stop_flag = true;
        t.join();
        stop_flag = false;
    }
    ~Worker() { stop(); }
};

// a gate a handler waits at (10 s at most) until the test opens it
struct Gate {
    std::mutex mu;
    std::condition_variable cv;
    bool is_open = true;
    void close() {
        std::lock_guard<std::mutex> lk(mu);
        is_open = false;
    }
    void open() {
        {
            std::lock_guard<std::mutex> lk(mu);
            is_open = true;
        }
        cv.notify_all();
    }
    void wait() {
        std::unique_lock<std::mutex> lk(mu);
        cv.wait_for(lk, std::chrono::seconds(10), [&] { return is_open; });
    }
};
// what the handlers did, in order
struct Log {
    std::mutex mu;
    std::vector<std::string> ev;
    void add(const std::string& e) {
        std::lock_guard<std::mutex> lk(mu);
        ev.push_back(e);
    }
    std::vector<std::string> all() {
        std::lock_guard<std::mutex> lk(mu);
        return ev;
    }
    void clear() {
        std::lock_guard<std::mutex> lk(mu);
        ev.clear();
    }
    bool has(const std::string& e) {
        std::lock_guard<std::mutex> lk(mu);
        for (const std::string& x : ev)
            if (x == e) return true;
        return false;
    }
    // waits (2 s at most) for entry `e`
    bool wait_for(const std::string& e) {
        for (int i = 0; i < 200; ++i) {
            if (has(e)) return true;
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        return false;
    }
};
std::string keep_text(const int32_t* keep, int n) {
    std::string s;
    for (int i = 0; i < n; ++i) s += (i ? "," : "") + std::to_string(keep[i]);
    return s;
}

// a window over S rows on `slots`, its rows all `v`; the reply's rows (the handlers copy them, plus the slot)
struct Win {
    int S = 0;
    std::vector<int32_t> sl, tk;
    std::vector<int64_t> ps;
    std::vector<float> in, out;
};
Win make(std::vector<int32_t> slots, float v) {
    Win w;
    w.S = (int) slots.size();
    w.sl = std::move(slots);
    w.tk.assign((size_t) w.S, 7);
    w.ps.assign((size_t) w.S, 0);
    for (int t = 0; t < w.S; ++t) w.ps[(size_t) t] = t;
    w.in.assign((size_t) (w.S * kHbf), v);
    w.out.assign((size_t) (w.S * kHbf), -1.0f);
    return w;
}
bool rows_ok(const Win& w) {
    for (int t = 0; t < w.S; ++t)
        for (int64_t j = 0; j < kHbf; ++j)
            if (w.out[(size_t) (t * kHbf + j)] != w.in[(size_t) (t * kHbf + j)] + (float) w.sl[(size_t) t]) return false;
    return true;
}
}  // namespace

int main() {
    std::printf("stage_link_deferred_test (protocol %u)\n", kStageProtocol);

    // ---- 0. the constants (protocol 8's optional bits: the number stays 8)
    check(kStageProtocol == 8, "0. the protocol is still 8");
    check(kStageSlotDeferred == 2 && kStageCommitDeferred == 1 && kStageBatchDeferred == 8u,
          "0. SlotRun flag bit 1, SlotCommit b bit 0, the hello's batch_flags bit 3");
    check((kStageSlotDeferred & kStageSlotKeepAll) == 0 &&
              (kStageBatchDeferred & (kStageBatchKeepAll | kStageBatchOverlap | kStageBatchSlotPositions)) == 0,
          "0. the new bits are apart from the old ones");

    float run_in[kMaxT * kHbf] = {}, run_out[kMaxT * kHbf] = {};
    float pf_in0[kChunk * kD] = {}, pf_in1[kChunk * kD] = {}, pf_out0[kChunk * kD] = {}, pf_out1[kChunk * kD] = {};
    StageBuffers buf;
    buf.run_in = run_in;
    buf.run_out = run_out;
    buf.run_floats = (size_t) (kMaxT * kHbf);
    buf.pf_in[0] = pf_in0;
    buf.pf_in[1] = pf_in1;
    buf.pf_out[0] = pf_out0;
    buf.pf_out[1] = pf_out1;
    buf.pf_floats = (size_t) (kChunk * kD);
    buf.handoff_floats = kHbf;
    buf.pf_row_floats = kD;

    // the handlers every worker here has; slot_run copies the rows plus the slot into run_out (answered at once)
    Log log;
    StageHandlers base;
    base.hello = [](const StageHello&, StageHello& m, std::string&) {
        m = hello_of();
        m.batch_slots = 4;
        m.batch_flags = kStageBatchKeepAll | kStageBatchDeferred;   // (serve_stage keeps bit 3 only with the handler)
        return true;
    };
    base.run = [](int, const int32_t*, int64_t, std::string&) { return true; };
    base.commit = [](int, std::string&) { return true; };
    base.prefill = [](const int64_t*, int64_t T, int64_t, int64_t, int64_t, const float* in, float* out, std::string&) {
        std::memcpy(out, in, (size_t) (T * kD) * sizeof(float));
        return true;
    };
    base.reset = [](const std::vector<int64_t>&, std::string&) { return true; };
    base.slot_run = [&](int S, const int32_t* slots, const int32_t*, const int64_t*, int64_t flags, std::string&) {
        for (int t = 0; t < S; ++t)
            for (int64_t j = 0; j < kHbf; ++j) run_out[t * kHbf + j] = run_in[t * kHbf + j] + (float) slots[t];
        log.add("run:" + std::to_string(flags) + ":" + std::to_string(slots[0]));
        return true;
    };
    base.slot_commit = [&](const int32_t* keep, int n, std::string&) {
        log.add("commit:" + keep_text(keep, n));
        return true;
    };
    base.slot_save = [](int, int64_t, std::string&) { return true; };
    base.slot_load = [](int, int64_t, int64_t, std::string&) { return true; };

    StageHello mine = hello_of();
    mine.batch_slots = 4;
    mine.batch_flags = kStageBatchKeepAll;
    auto window = [](StageClient& c, Win& w, int64_t flags, int64_t tag, std::string& e) {
        return c.slot_run(w.S, w.sl.data(), w.tk.data(), w.ps.data(), flags, tag, w.in.data(), w.in.size(),
                          w.out.data(), w.out.size(), e);
    };

    // ---- 1, 2. a worker without slot_commit_deferred
    {
        Worker wk;
        check(wk.start(base, buf), "1. a worker without slot_commit_deferred listens on loopback");
        StageClient c;
        StageHello peer;
        std::string e;
        check(c.connect(wk.where, mine, peer, e) && peer.batch_flags == kStageBatchKeepAll,
              "1. its hello's kStageBatchDeferred is cleared (the hello handler wrote it)");
        Win w = make({0, 1}, 1.0f);
        e.clear();
        check(!window(c, w, kStageSlotDeferred, 1, e) && contains(e, "bad slot window") && c.connected(),
              "2. a kStageSlotDeferred window: bad slot window");
        e.clear();
        check(window(c, w, 0, 2, e) && rows_ok(w), "2. ... and the link goes on (a window without a flag)");
        const int32_t keep[2] = {1, 1};
        e.clear();
        const bool sent = c.slot_commit(keep, 2, 2, e, true);
        Win w2 = make({0}, 2.0f);
        std::string e2;
        check(sent && !window(c, w2, 0, 3, e2) && contains(e2, "bad slot commit"),
              "2. a deferred SlotCommit (one way): the next SlotRun's error is bad slot commit");
        e2.clear();
        check(window(c, w2, kStageSlotKeepAll, 4, e2) && rows_ok(w2), "2. ... and the link goes on");
        c.close();
    }

    // ---- 3..5, 7. a worker with slot_commit_deferred that answers at once (as the end)
    {
        std::atomic<bool> fail_next{false};
        StageHandlers h = base;
        h.slot_commit_deferred = [&](const int32_t* keep, int n, int64_t tag, std::string& e) {
            log.add("dcommit:" + std::to_string(tag) + ":" + keep_text(keep, n));
            if (fail_next.exchange(false)) {
                e = "the test's deferred commit: no window has tag " + std::to_string(tag);
                return false;
            }
            return true;
        };
        Worker wk;
        check(wk.start(h, buf), "4. a worker with slot_commit_deferred listens on loopback");
        StageClient c;
        StageHello peer;
        std::string e;
        check(c.connect(wk.where, mine, peer, e) && peer.batch_flags == (kStageBatchKeepAll | kStageBatchDeferred),
              "1. its hello keeps kStageBatchDeferred");

        // 3. the main side's own check
        Win w = make({0, 1}, 1.0f);
        e.clear();
        const uint64_t out0 = c.stats().bytes_out;
        check(!window(c, w, kStageSlotKeepAll | kStageSlotDeferred, 9, e) && contains(e, "flags 3") &&
                  c.stats().bytes_out == out0 && c.connected(),
              "3. flags 3 are refused before anything goes out");
        e.clear();
        check(!window(c, w, 4, 9, e) && contains(e, "flags 4") && c.connected(), "3. so is flag bit 2 (4)");

        // 4. two deferred windows (slots 0 and 1; 2 and 3), committed by tag in the other order
        log.clear();
        Win a = make({0, 0, 1, 1}, 3.0f), b = make({2, 3}, 5.0f);
        e.clear();
        bool ok = window(c, a, kStageSlotDeferred, 11, e) && rows_ok(a) && window(c, b, kStageSlotDeferred, 12, e) &&
                  rows_ok(b);
        if (!ok) std::printf("    (%s)\n", e.c_str());
        const int32_t kb[4] = {0, 0, 1, 1}, ka[2] = {2, 1};
        ok = ok && c.slot_commit(kb, 4, 12, e, true) && c.slot_commit(ka, 2, 11, e, true);
        Win probe = make({3}, 0.0f);   // (a commit is one way: a window after it shows it was served)
        ok = ok && window(c, probe, kStageSlotKeepAll, 13, e);
        const std::vector<std::string> got = log.all();
        const std::vector<std::string> want = {"run:2:0", "run:2:2", "dcommit:12:0,0,1,1", "dcommit:11:2,1", "run:1:3"};
        check(ok && got == want,
              "4. slot_run gets flag 2; slot_commit_deferred gets each tag and keep[] whole, in the order sent");
        if (got != want) {
            std::string s;
            for (const std::string& x : got) s += " " + x;
            std::printf("    (the log:%s)\n", s.c_str());
        }
        log.clear();
        const int32_t kp[1] = {1};
        ok = window(c, probe, 0, 14, e) && c.slot_commit(kp, 1, 14, e) && window(c, probe, kStageSlotKeepAll, 15, e);
        check(ok && log.has("commit:1") && !log.has("dcommit:14:1"),
              "4. a plain SlotCommit (b = 0) still goes to slot_commit");

        // 5. a failed deferred commit: the next SlotRun's error, then the link goes on
        fail_next = true;
        e.clear();
        const bool sent = c.slot_commit(kp, 1, 99, e, true);
        std::string e2;
        check(sent && !window(c, probe, kStageSlotKeepAll, 16, e2) && contains(e2, "no window has tag 99"),
              "5. slot_commit_deferred's failure is the next SlotRun's error");
        e2.clear();
        check(window(c, probe, kStageSlotKeepAll, 17, e2) && c.connected(), "5. ... and the link goes on");
        c.close();

        // 7. hand-spoken: flags 3, and a SlotCommit with b = 2
        Raw r;
        const bool open = r.open(wk.where) && r.hello(mine);
        auto answer = [&](const char* want_text) {
            StageHeader rh;
            std::string p;
            return r.reply(rh, p) && rh.type == (uint32_t) StageMsg::Error && contains(p, want_text);
        };
        auto serves_on = [&](int64_t tag) {
            r.slot_run(1, 0, tag);
            StageHeader rh;
            std::string p;
            return r.reply(rh, p) && rh.type == (uint32_t) StageMsg::SlotRunOk && rh.c == tag;
        };
        r.slot_run(2, kStageSlotKeepAll | kStageSlotDeferred, 21);
        check(open && answer("bad slot window") && serves_on(22), "7. a SlotRun with flags 3: bad slot window");
        const int32_t k1[1] = {1};
        r.header(StageMsg::SlotCommit, 1, 2, 22, 4);
        send_n(r.s, k1, 4);
        r.slot_run(1, 0, 23);
        check(answer("bad slot commit") && serves_on(24), "7. a SlotCommit with b = 2: the next SlotRun's bad slot commit");
    }

    // ---- 6. a worker that answers late (a relay): the deferred commit does not wait for the late replies
    {
        Gate gate;
        std::mutex kept_mu;
        std::deque<std::vector<float>> kept;
        StageHandlers h = base;
        h.slot_run = [&](int S, const int32_t* slots, const int32_t*, const int64_t*, int64_t flags, std::string&) {
            std::vector<float> rows((size_t) (S * kHbf));
            for (int t = 0; t < S; ++t)
                for (int64_t j = 0; j < kHbf; ++j) rows[(size_t) (t * kHbf + j)] = run_in[t * kHbf + j] + (float) slots[t];
            log.add("run:" + std::to_string(flags) + ":" + std::to_string(slots[0]));
            std::lock_guard<std::mutex> lk(kept_mu);
            kept.push_back(std::move(rows));
            return true;
        };
        h.slot_run_reply = [&](int S, float* out, std::string& e) {
            gate.wait();
            std::vector<float> rows;
            {
                std::lock_guard<std::mutex> lk(kept_mu);
                if (kept.empty()) { e = "the test's late reply: no window kept"; return false; }
                rows = std::move(kept.front());
                kept.pop_front();
            }
            if (rows.size() != (size_t) (S * kHbf)) { e = "the test's late reply: S differs"; return false; }
            std::memcpy(out, rows.data(), rows.size() * sizeof(float));
            log.add("reply");
            return true;
        };
        h.slot_commit_deferred = [&](const int32_t* keep, int n, int64_t tag, std::string&) {
            log.add("dcommit:" + std::to_string(tag) + ":" + keep_text(keep, n));
            return true;
        };
        Worker wk;
        check(wk.start(h, buf), "6. a worker that answers late listens on loopback");
        StageClient c;
        StageHello peer;
        std::string e;
        check(c.connect(wk.where, mine, peer, e), "6. (the main process connects)");
        log.clear();
        Win a = make({0, 0}, 1.0f), b = make({1, 1}, 2.0f);
        bool ok = c.slot_run_send(a.S, a.sl.data(), a.tk.data(), a.ps.data(), kStageSlotDeferred, 31, a.in.data(),
                                  a.in.size(), e) &&
                  c.slot_run_recv(31, a.S, a.out.data(), a.out.size(), e) && rows_ok(a);
        gate.close();   // window b's reply waits here
        ok = ok && c.slot_run_send(b.S, b.sl.data(), b.tk.data(), b.ps.data(), kStageSlotDeferred, 32, b.in.data(),
                                   b.in.size(), e);
        const int32_t ka[2] = {2, 0};
        ok = ok && c.slot_commit(ka, 2, 31, e, true);
        const bool served_early = log.wait_for("dcommit:31:2,0");
        const int replies_then = [&] {
            int n = 0;
            for (const std::string& x : log.all()) n += x == "reply" ? 1 : 0;
            return n;
        }();
        gate.open();
        ok = ok && c.slot_run_recv(32, b.S, b.out.data(), b.out.size(), e) && rows_ok(b);
        if (!ok) std::printf("    (%s)\n", e.c_str());
        check(ok && log.has("run:2:0") && log.has("run:2:1"), "6. deferred windows are answered late, rows in place");
        check(ok && served_early && replies_then == 1,
              "6. window a's deferred commit is served while window b's late reply is still held");
        c.close();
    }

    // ---- 8. (合流 D) slot_run_tag: each SlotRun's tag, right before its slot_run; the bytes on the wire as before
    {
        StageHandlers h = base;
        h.slot_commit_deferred = [&](const int32_t* keep, int n, int64_t tag, std::string&) {
            log.add("dcommit:" + std::to_string(tag) + ":" + keep_text(keep, n));
            return true;
        };
        h.slot_run_tag = [&](int64_t tag) { log.add("tag:" + std::to_string(tag)); };
        h.hello = [](const StageHello&, StageHello& m, std::string&) {   // (its saved rows a slot: 2)
            m = hello_of();
            m.batch_slots = 4;
            m.batch_flags = with_deferred_rows(kStageBatchKeepAll | kStageBatchDeferred, 2);
            return true;
        };
        Worker wk;
        check(wk.start(h, buf), "8. a worker with slot_run_tag listens on loopback");
        StageClient c;
        StageHello peer;
        std::string e;
        check(c.connect(wk.where, mine, peer, e) && (peer.batch_flags & kStageBatchDeferred) != 0 &&
                  stage_deferred_rows(peer.batch_flags) == 2,
              "8. its hello says 2 rows a slot (bits 8..11) with kStageBatchDeferred");
        log.clear();
        Win a = make({0, 1}, 1.0f), b = make({2}, 2.0f), k = make({3}, 3.0f);
        const uint64_t out0 = c.stats().bytes_out;
        bool ok = window(c, a, kStageSlotDeferred, 41, e) && window(c, b, 0, 42, e);
        const int32_t kb[3] = {0, 0, 1}, ka[2] = {1, 1};
        ok = ok && c.slot_commit(kb, 3, 42, e) && c.slot_commit(ka, 2, 41, e, true) && window(c, k, kStageSlotKeepAll, 43, e);
        const std::vector<std::string> got = log.all();
        const std::vector<std::string> want = {"tag:41", "run:2:0", "tag:42", "run:0:2", "commit:0,0,1", "dcommit:41:1,1",
                                               "tag:43", "run:1:3"};
        check(ok && got == want, "8. slot_run_tag gets each window's tag before its slot_run (deferred, plain, KeepAll)");
        if (got != want) {
            std::string t;
            for (const std::string& x : got) t += " " + x;
            std::printf("    (%s; the log:%s)\n", e.c_str(), t.c_str());
        }
        // the same messages to a worker without the hook: the same bytes out (the hook changes nothing on the wire)
        const uint64_t sent_with = c.stats().bytes_out - out0;
        c.close();
        StageHandlers h2 = base;
        h2.slot_commit_deferred = h.slot_commit_deferred;
        Worker wk2;
        check(wk2.start(h2, buf), "8. the same worker without slot_run_tag listens on loopback");
        StageClient c2;
        check(c2.connect(wk2.where, mine, peer, e), "8. (the main process connects)");
        log.clear();
        const uint64_t out1 = c2.stats().bytes_out;
        ok = window(c2, a, kStageSlotDeferred, 41, e) && window(c2, b, 0, 42, e) && c2.slot_commit(kb, 3, 42, e) &&
             c2.slot_commit(ka, 2, 41, e, true) && window(c2, k, kStageSlotKeepAll, 43, e);
        const std::vector<std::string> want2 = {"run:2:0", "run:0:2", "commit:0,0,1", "dcommit:41:1,1", "run:1:3"};
        check(ok && log.all() == want2 && c2.stats().bytes_out - out1 == sent_with,
              "8. without slot_run_tag: the same handlers in the same order, the same bytes out");
        c2.close();
        // a worker without slot_commit_deferred: serve_stage clears the rows with the bit
        StageHandlers h3 = base;
        h3.hello = h.hello;
        Worker wk3;
        check(wk3.start(h3, buf), "8. a worker without slot_commit_deferred listens on loopback");
        StageClient c3;
        check(c3.connect(wk3.where, mine, peer, e) && peer.batch_flags == kStageBatchKeepAll,
              "8. ... its hello keeps neither kStageBatchDeferred nor the rows");
        c3.close();
        check(stage_deferred_rows(with_deferred_rows(kStageBatchDeferred, 20)) == 15 &&
                  stage_deferred_rows(with_deferred_rows(kStageBatchDeferredRowsMask, 0)) == 0 &&
                  (with_deferred_rows(kStageBatchKeepAll, 3) & 0xffu) == kStageBatchKeepAll,
              "8. with_deferred_rows: clamped to 4 bits, the other bits kept");
    }

    std::printf(g_fail == 0 ? "all passed (0 failed)\n" : "%d FAILED\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
