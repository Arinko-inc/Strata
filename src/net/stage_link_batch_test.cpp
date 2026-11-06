// src/net/stage_link_batch_test.cpp - protocol 8's batch slots over the remote stage's link, on loopback, no GPU
// (strata-mesh IMPL-PLAN-remote-batch.md section 3; the remote stage's link itself: stage_link_test.cpp).
//
// What the skeleton fixed - the constants, the hello's size and layout (192 bytes, the token where every protocol
// has it), the message numbers - and then, against a worker on loopback whose handlers record what they get:
//   1. a protocol 7 hello refused both ways, the reason naming both protocols;
//   2. the hello's batch_slots and batch_flags arrive, both ways (a worker without slot handlers answers 0);
//   3. SlotRun round trips: S = 1..8, fp32 and bf16 rows, slots / tokens / positions / flags / tag arrive, the rows
//      come back (the handler changes them by a fixed rule);
//   4. three SlotRuns sent back to back (slot_run_send x 3): the replies come back in the order sent, with their tags;
//      a reply asked for out of order, and a reply whose tag is not the window's, break the link;
//   5. SlotCommit carries keep[] whole; a handler's failure is the next SlotRun's error, also across a Reset;
//   6. SlotSave needs the session at L (a SlotRun does not move it); after a SlotLoad the session is at L (a CkptSave
//      at L is taken);
//   7. a SlotLoad of a checkpoint id the worker does not hold: `missing`, and the link goes on;
//   8. malformed slot messages (S = 0, S past the window, S near 2^62, bytes that do not add up, n past kStageSlotMax,
//      an unknown flag bit, a SlotSave with a payload) are answered with an error and the link goes on;
//   9. a worker without slot handlers names --batch in its error;
//  10. stage_link_lost does not count the new errors as a lost link.
// Stage 2 (strata-mesh IMPL-PLAN-remote-batch-stage2.md 4.1), against a worker that answers keep-all windows late:
//  11. the next window runs before the reply before it is out; the replies keep the order sent, tags and rows;
//  12. a window without kStageSlotKeepAll is not answered late;
//  13. a late window behind a prompt chunk's reply still going out runs after that reply;
//  14. a late reply's failure is that window's error, the later ones are answered, the link goes on;
//  15. SlotCommit, Reset and SlotSave after a late window are served once its reply is out;
//  16. StageSlotPipe: poll 0 then 1 (in any order), the rows in place, outstanding(), the worker's error (-1, and every
//      later call), a number sent twice or out of range, stop() with nothing out (the link serves on) and with a
//      window out (the read is cut at once, the link is down);
//  17. main -> relay -> end on loopback: a relay's failure after its send (window A's reply held just before its read):
//      A's rows, B's error, C's rows in order and the links standing; the wrong form breaks the relay's next link.
#include "strata/net/stage_link.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <initializer_list>
#include <mutex>
#include <random>
#include <set>
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

// ---- raw sockets (a main process's end spoken by hand, a worker's end faked)
#if defined(_WIN32)
using sock = SOCKET;
constexpr sock kNone = INVALID_SOCKET;
void close_sock(sock s) { (void) ::closesocket(s); }
void shut_sock(sock s) { (void) ::shutdown(s, SD_BOTH); }
using sock_len = int;
#else
using sock = int;
constexpr sock kNone = -1;
void close_sock(sock s) { (void) ::close(s); }
void shut_sock(sock s) { (void) ::shutdown(s, SHUT_RDWR); }
using sock_len = socklen_t;
#endif
void set_rcv_timeout(sock s, int seconds) {
#if defined(_WIN32)
    const DWORD ms = (DWORD) seconds * 1000;
    (void) setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, (const char*) &ms, sizeof ms);
#else
    timeval tv{};
    tv.tv_sec = seconds;
    (void) setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);
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
void send_n(sock fd, const void* p, size_t n) {
#if defined(_WIN32)
    (void) ::send(fd, (const char*) p, (int) n, 0);
#else
    (void) ::send(fd, p, n, MSG_NOSIGNAL);
#endif
}
int port_of(const std::string& where) { return std::atoi(where.substr(where.rfind(':') + 1).c_str()); }

// a main process's end by hand: headers a StageClient never sends.  Reads wait 5 s at most.
struct Raw {
    sock s = kNone;
    bool open(const std::string& where) {
        sockaddr_in a{};
        a.sin_family = AF_INET;
        a.sin_port = htons((unsigned short) port_of(where));
        a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        s = ::socket(AF_INET, SOCK_STREAM, 0);
        if (s == kNone || ::connect(s, (const sockaddr*) &a, sizeof a) != 0) return false;
        set_rcv_timeout(s, 5);
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
    void bytes(size_t n, char fill = 0) {
        std::vector<char> z(n, fill);
        if (n > 0) send_n(s, z.data(), n);
    }
    // the next message, its payload as bytes (up to 1 MiB); false: none within 5 s, or the worker closed it
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
        return reply(r, p) && r.type == (uint32_t) StageMsg::HelloOk && p.size() == sizeof(StageHello);
    }
    // one fp32 SlotRun of S rows on slots 0.., its rows all `v`
    void slot_run(int S, int64_t flags, int64_t tag, float v) {
        std::vector<int32_t> sl((size_t) S), tk((size_t) S);
        std::vector<int64_t> ps((size_t) S);
        std::vector<float> rows((size_t) (S * kHbf), v);
        for (int t = 0; t < S; ++t) { sl[(size_t) t] = t; tk[(size_t) t] = 100 + t; ps[(size_t) t] = 3 + t; }
        header(StageMsg::SlotRun, S, flags, tag, (uint64_t) S * 16 + rows.size() * 4);
        send_n(s, sl.data(), sl.size() * 4);
        send_n(s, tk.data(), tk.size() * 4);
        send_n(s, ps.data(), ps.size() * 8);
        send_n(s, rows.data(), rows.size() * 4);
    }
    ~Raw() {
        if (s != kNone) close_sock(s);
    }
};

// a worker faked by hand: answers the hello with hello_of(), then every SlotRun with its rows and the tag `tag_delta`
// off (a worker out of step)
struct Fake {
    sock l = kNone, s = kNone;
    std::thread t;
    std::string where;
    bool start(int64_t tag_delta) {
        sockaddr_in a{};
        a.sin_family = AF_INET;
        a.sin_port = 0;
        a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        l = ::socket(AF_INET, SOCK_STREAM, 0);
        sock_len len = (sock_len) sizeof a;
        if (l == kNone || ::bind(l, (const sockaddr*) &a, sizeof a) != 0 || ::listen(l, 4) != 0 ||
            ::getsockname(l, (sockaddr*) &a, &len) != 0)
            return false;
        where = "127.0.0.1:" + std::to_string(ntohs(a.sin_port));
        t = std::thread([this, tag_delta] {
            s = ::accept(l, nullptr, nullptr);
            if (s == kNone) return;
            set_rcv_timeout(s, 10);
            StageHeader h;
            StageHello theirs;
            if (!recv_n(s, &h, sizeof h) || !recv_n(s, &theirs, sizeof theirs)) return;
            StageHeader r;
            r.type = (uint32_t) StageMsg::HelloOk;
            r.bytes = sizeof(StageHello);
            StageHello mine = hello_of();
            mine.batch_slots = 4;
            send_n(s, &r, sizeof r);
            send_n(s, &mine, sizeof mine);
            for (;;) {
                if (!recv_n(s, &h, sizeof h)) return;
                std::vector<char> body((size_t) h.bytes);
                if (!body.empty() && !recv_n(s, body.data(), body.size())) return;
                if (h.type != (uint32_t) StageMsg::SlotRun) continue;
                const size_t rows = (size_t) h.a * (size_t) kHbf * 4;
                StageHeader ok;
                ok.type = (uint32_t) StageMsg::SlotRunOk;
                ok.c = h.c + tag_delta;
                ok.bytes = rows;
                send_n(s, &ok, sizeof ok);
                send_n(s, body.data() + body.size() - rows, rows);
            }
        });
        return true;
    }
    void stop() {
        if (l != kNone) { shut_sock(l); close_sock(l); l = kNone; }
        if (s != kNone) shut_sock(s);
        if (t.joinable()) t.join();
        if (s != kNone) close_sock(s);
        s = kNone;
    }
    ~Fake() { stop(); }
};

// what the worker's handlers saw
struct Seen {
    std::mutex mu;
    StageHello main_hello;
    int S = 0;
    std::vector<int32_t> slots, tokens, keep;
    std::vector<int64_t> pos;
    int64_t flags = -1;
    int save_slot = -1, load_slot = -1;
    int64_t save_L = -1, load_L = -1, load_id = -1;
    std::set<int64_t> held;   // checkpoint ids saved (CkptSave)
    int commits = 0;
};

// one worker on a free loopback port, until stop()
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
            // until it listens: the worker's own refusal of a protocol 4 hello (not the system's "connection refused")
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
};

// protocol 8 stage 2's tests: a gate a handler waits at (10 s at most) until the test opens it
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
    void clear() {
        std::lock_guard<std::mutex> lk(mu);
        ev.clear();
    }
    int count(const std::string& prefix) {
        std::lock_guard<std::mutex> lk(mu);
        int n = 0;
        for (const std::string& e : ev) n += e.rfind(prefix, 0) == 0 ? 1 : 0;
        return n;
    }
    // every one of `want` is in the log, in this order
    bool order(std::initializer_list<const char*> want) {
        std::lock_guard<std::mutex> lk(mu);
        size_t at = 0;
        for (const char* w : want) {
            while (at < ev.size() && ev[at] != w) ++at;
            if (at == ev.size()) {
                std::string all;
                for (const std::string& e : ev) all += " " + e;
                std::printf("    (not in order: %s; the log:%s)\n", w, all.c_str());
                return false;
            }
            ++at;
        }
        return true;
    }
};

// the handler's rule for a batch window's rows: row t's values doubled plus its slot (small integers: exact in bf16)
float slot_rule(float in, int32_t slot) { return in * 2.0f + (float) slot; }
}  // namespace

int main() {
    std::printf("stage_link_batch_test (protocol %u)\n", kStageProtocol);

    // ---- what the skeleton fixed
    check(kStageProtocol == 8, "the protocol is 8");
    check(sizeof(StageHello) == 192, "the hello is 192 bytes, as in every protocol");
    check(offsetof(StageHello, batch_slots) == 120 && offsetof(StageHello, batch_flags) == 124 &&
              offsetof(StageHello, token) == 128,
          "batch_slots / batch_flags at 120 / 124, the token at 128");
    check((uint32_t) StageMsg::SlotRun == 16 && (uint32_t) StageMsg::SlotRunOk == 17 &&
              (uint32_t) StageMsg::SlotCommit == 18 && (uint32_t) StageMsg::SlotSave == 19 &&
              (uint32_t) StageMsg::SlotSaveOk == 20 && (uint32_t) StageMsg::SlotLoad == 21 &&
              (uint32_t) StageMsg::SlotLoadOk == 22,
          "the batch slot messages are 16..22");
    check(kStageSlotKeepAll == 1 && kStageBatchKeepAll == 1u && kStageBatchOverlap == 2u && kStageSlotMax == 64,
          "the flags and the slot bound");

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

    Seen seen;
    std::atomic<int32_t> worker_protocol{(int32_t) kStageProtocol};
    std::atomic<bool> fail_commit{false};
    StageHandlers h;
    h.hello = [&](const StageHello& main_hello, StageHello& mine, std::string&) {
        mine = hello_of();
        mine.protocol = (uint32_t) worker_protocol.load();
        mine.batch_slots = 3;
        mine.batch_flags = kStageBatchKeepAll;
        std::lock_guard<std::mutex> lk(seen.mu);
        seen.main_hello = main_hello;
        return true;
    };
    h.run = [&](int T, const int32_t*, int64_t, std::string&) {
        for (int64_t i = 0; i < T * kHbf; ++i) run_out[i] = run_in[i] + 1.0f;
        return true;
    };
    h.commit = [](int, std::string&) { return true; };
    h.prefill = [](const int64_t*, int64_t T, int64_t, int64_t, int64_t, const float* in, float* out, std::string&) {
        std::memcpy(out, in, (size_t) (T * kD) * sizeof(float));
        return true;
    };
    h.reset = [](const std::vector<int64_t>&, std::string&) { return true; };
    h.ckpt_save = [&](int64_t id, int64_t, const std::vector<int64_t>&, std::string&) {
        std::lock_guard<std::mutex> lk(seen.mu);
        seen.held.insert(id);
        return 1;
    };
    h.ckpt_restore = [](int64_t, int64_t, const std::vector<int64_t>&, std::string&) { return true; };
    h.slot_run = [&](int S, const int32_t* slots, const int32_t* tokens, const int64_t* pos, int64_t flags,
                     std::string&) {
        std::lock_guard<std::mutex> lk(seen.mu);
        seen.S = S;
        seen.slots.assign(slots, slots + S);
        seen.tokens.assign(tokens, tokens + S);
        seen.pos.assign(pos, pos + S);
        seen.flags = flags;
        for (int t = 0; t < S; ++t)
            for (int64_t j = 0; j < kHbf; ++j) run_out[t * kHbf + j] = slot_rule(run_in[t * kHbf + j], slots[t]);
        return true;
    };
    h.slot_commit = [&](const int32_t* keep, int n, std::string& e) {
        std::lock_guard<std::mutex> lk(seen.mu);
        ++seen.commits;
        seen.keep.assign(keep, keep + n);
        if (fail_commit.exchange(false)) {
            e = "the test's commit failed";
            return false;
        }
        return true;
    };
    h.slot_save = [&](int slot, int64_t L, std::string&) {
        std::lock_guard<std::mutex> lk(seen.mu);
        seen.save_slot = slot;
        seen.save_L = L;
        return true;
    };
    h.slot_load = [&](int slot, int64_t L, int64_t id, std::string& e) {
        std::lock_guard<std::mutex> lk(seen.mu);
        if (id != 0 && seen.held.count(id) == 0) {
            e = std::string(kStageCkptMissing) + std::to_string(id);
            return false;
        }
        seen.load_slot = slot;
        seen.load_L = L;
        seen.load_id = id;
        return true;
    };

    Worker wk;
    check(wk.start(h, buf), "a worker listens on loopback");
    if (wk.where.empty()) {
        std::printf("stage_link_batch_test: no loopback port to listen on\n");
        return 1;
    }
    const std::string& where = wk.where;
    std::vector<std::string> new_errors;   // (10.) every error the new messages gave
    auto note = [&](const std::string& e) { new_errors.push_back(e); return true; };

    // ---- 1. a protocol 7 peer is refused for its protocol, with the reason, both ways
    {
        StageClient c;
        StageHello peer;
        std::string err;
        StageHello seven = hello_of();
        seven.protocol = 7;
        check(!c.connect(where, seven, peer, err) && contains(err, "the worker refused:") &&
                  contains(err, "protocol 7/8") && contains(err, "same Strata build") && !stage_link_lost(err),
              "1. a protocol 7 main process is refused by a protocol 8 worker, with the reason");
        c.close();
        worker_protocol = 7;   // (a worker that answers with a protocol 7 hello)
        err.clear();
        check(!c.connect(where, hello_of(), peer, err) && contains(err, "protocol 8/7") && contains(err, "differ") &&
                  !stage_link_lost(err),
              "1. a protocol 8 main process refuses a protocol 7 worker's hello, not as a lost link");
        c.close();
        worker_protocol = (int32_t) kStageProtocol;
    }

    // ---- 2. the hello's batch fields, both ways.  (The worker serves one main process at a time: a client closes
    // before another one connects, else the new one waits for the take-over time.)
    StageClient c;
    StageHello peer;
    std::string err;
    StageHello mine = hello_of();
    mine.batch_slots = 4;
    mine.batch_flags = kStageBatchKeepAll;
    check(c.connect(where, mine, peer, err), "2. a protocol 8 main process is served");
    {
        std::lock_guard<std::mutex> lk(seen.mu);
        check(seen.main_hello.batch_slots == 4 && seen.main_hello.batch_flags == kStageBatchKeepAll,
              "2. the worker's hello handler gets the main process's batch_slots and batch_flags");
    }
    check(peer.batch_slots == 3 && peer.batch_flags == kStageBatchKeepAll,
          "2. the main process gets the worker's batch_slots and batch_flags");

    // ---- 3. SlotRun round trips, fp32 and bf16, S = 1..8
    auto window = [&](StageClient& cl, int S, int64_t flags, int64_t tag, float base) -> bool {
        std::vector<int32_t> sl((size_t) S), tk((size_t) S);
        std::vector<int64_t> ps((size_t) S);
        std::vector<float> in((size_t) (S * kHbf)), out((size_t) (S * kHbf), -1.0f);
        for (int t = 0; t < S; ++t) {
            sl[(size_t) t] = t / 2;   // two rows a slot, contiguous
            tk[(size_t) t] = 1000 + 7 * t;
            ps[(size_t) t] = ((int64_t) 1 << 33) + 5 * t;   // (past 32 bits: the positions are int64 on the wire)
            for (int64_t j = 0; j < kHbf; ++j) in[(size_t) (t * kHbf + j)] = base + (float) (t * kHbf + j);
        }
        std::string e;
        if (!cl.slot_run(S, sl.data(), tk.data(), ps.data(), flags, tag, in.data(), in.size(), out.data(), out.size(), e)) {
            std::printf("    (slot_run S=%d: %s)\n", S, e.c_str());
            return false;
        }
        for (int t = 0; t < S; ++t)
            for (int64_t j = 0; j < kHbf; ++j)
                if (out[(size_t) (t * kHbf + j)] != slot_rule(in[(size_t) (t * kHbf + j)], sl[(size_t) t])) return false;
        std::lock_guard<std::mutex> lk(seen.mu);
        return seen.S == S && seen.slots == sl && seen.tokens == tk && seen.pos == ps && seen.flags == flags;
    };
    {
        bool all = c.rows() == StageRows::F32;
        for (int S = 1; S <= (int) kMaxT; ++S) all = window(c, S, S % 2 == 0 ? kStageSlotKeepAll : 0, 40 + S, 1.0f) && all;
        check(all, "3. fp32: S = 1..8, slots / tokens / positions / flags arrive, the rows come back");
        c.close();
        StageClient b;
        StageHello mb = mine;
        mb.row_formats = kStageRowsAll;
        StageHello pb;
        std::string eb;
        bool ball = b.connect(where, mb, pb, eb) && b.rows() == StageRows::BF16;
        for (int S = 1; ball && S <= (int) kMaxT; ++S) ball = window(b, S, 0, 900 + S, 3.0f);
        check(ball, "3. bf16: the same, the rows converted both ways (values bf16 holds exactly)");
        check(b.stats().slot_runs == (uint64_t) kMaxT && b.stats().ms_slot_run > 0,
              "3. the main side counts its batch windows");
        b.close();
        err.clear();
        check(c.connect(where, mine, peer, err), "3. (the fp32 main process connects again)");
    }

    // ---- 4. three windows back to back; the replies in order with their tags
    {
        const int Ss[3] = {2, 5, 3};
        const int64_t tags[3] = {1001, 1002, 1003};
        std::vector<int32_t> sl[3], tk[3];
        std::vector<int64_t> ps[3];
        std::vector<float> in[3], out[3];
        bool sent = true;
        for (int k = 0; k < 3; ++k) {
            const int S = Ss[k];
            sl[k].assign((size_t) S, (int32_t) k);
            tk[k].assign((size_t) S, 7);
            ps[k].assign((size_t) S, 9);
            in[k].resize((size_t) (S * kHbf));
            out[k].assign((size_t) (S * kHbf), -1.0f);
            for (size_t i = 0; i < in[k].size(); ++i) in[k][i] = (float) (10 * k + (int) i);
            sent = c.slot_run_send(S, sl[k].data(), tk[k].data(), ps[k].data(), 0, tags[k], in[k].data(), in[k].size(),
                                   err) && sent;
        }
        check(sent, "4. three SlotRuns go out before any reply is read");
        std::string e;
        bool early = !c.slot_save(0, 0, e) && contains(e, "outstanding") && c.connected();
        note(e);
        check(early, "4. a SlotSave while their replies are outstanding is refused here, the link stands");
        bool ordered = true;
        for (int k = 0; k < 3; ++k) {
            ordered = c.slot_run_recv(tags[k], Ss[k], out[k].data(), out[k].size(), err) && ordered;
            for (size_t i = 0; i < out[k].size(); ++i) ordered = ordered && out[k][i] == slot_rule(in[k][i], k);
        }
        check(ordered, "4. the replies come back in the order sent, each with its tag and rows");
        // a reply asked for out of order: the link is out of step and fails
        c.close();
        StageClient d;
        StageHello pd;
        std::string ed;
        bool broke = d.connect(where, mine, pd, ed);
        for (int k = 0; broke && k < 2; ++k)
            broke = d.slot_run_send(Ss[k], sl[k].data(), tk[k].data(), ps[k].data(), 0, tags[k], in[k].data(),
                                    in[k].size(), ed);
        broke = broke && !d.slot_run_recv(tags[1], Ss[1], out[1].data(), out[1].size(), ed) &&
                contains(ed, "out of step") && !d.connected();
        note(ed);
        check(broke, "4. a reply asked for out of order fails the link (out of step)");
        d.close();
        err.clear();
        check(c.connect(where, mine, peer, err), "4. (the main process connects again)");
        // a worker whose reply carries another tag
        Fake fake;
        bool tagged = fake.start(1);
        StageClient f;
        std::string ef;
        StageHello pf;
        tagged = tagged && f.connect(fake.where, mine, pf, ef) && pf.batch_slots == 4;
        tagged = tagged && !f.slot_run(Ss[0], sl[0].data(), tk[0].data(), ps[0].data(), 0, 77, in[0].data(),
                                       in[0].size(), out[0].data(), out[0].size(), ef) &&
                 contains(ef, "came with tag 78") && !f.connected();
        note(ef);
        check(tagged, "4. a reply whose tag is not the window's fails the link");
        f.close();
        fake.stop();
    }

    // ---- 5. SlotCommit: keep[] whole; a failure is the next SlotRun's error, also across a Reset
    {
        const int32_t keep[5] = {1, 0, 3, 2, 0};
        bool ok = window(c, 4, 0, 50, 1.0f) && c.slot_commit(keep, 5, 50, err);
        ok = ok && window(c, 1, 0, 51, 1.0f);   // (the commit is in before the next window's reply)
        {
            std::lock_guard<std::mutex> lk(seen.mu);
            ok = ok && seen.keep == std::vector<int32_t>(keep, keep + 5);
        }
        check(ok, "5. SlotCommit carries keep[] whole");
        fail_commit = true;
        const int32_t one[1] = {1};
        bool held = c.slot_commit(one, 1, 51, err);   // (the handler clears fail_commit as it fails)
        held = held && c.reset({}, err);
        std::string e;
        std::vector<int32_t> sl(1, 0), tk(1, 0);
        std::vector<int64_t> ps(1, 0);
        std::vector<float> in((size_t) kHbf, 1.0f), out((size_t) kHbf);
        held = held && !c.slot_run(1, sl.data(), tk.data(), ps.data(), 0, 52, in.data(), in.size(), out.data(),
                                   out.size(), e) &&
               contains(e, "remote stage (worker): ") && contains(e, "the test's commit failed") && c.connected();
        note(e);
        check(held, "5. a failed SlotCommit is the next SlotRun's error, after a Reset too, and the link stands");
        check(window(c, 2, 0, 53, 1.0f), "5. ... and only that one's: the next window is served");
    }

    // ---- 6. SlotSave at the session's position; SlotLoad moves it to L
    {
        int64_t tok[5] = {1, 2, 3, 4, 5};
        float rows[5 * kD] = {}, back[5 * kD] = {};
        bool ok = c.reset({}, err) && c.prefill_send(tok, 5, 0, 0, rows, (size_t) kD, 0, err) &&
                  c.prefill_recv(back, (size_t) kD, 5, 0, err);
        ok = ok && window(c, 2, 0, 60, 1.0f);   // (a window does not move the session)
        std::string e;
        const bool wrong = !c.slot_save(1, 4, e) && contains(e, "a slot save at 4") && contains(e, "is at 5") &&
                           c.connected();
        note(e);
        check(ok && wrong, "6. a SlotSave away from the session's position is an error (a SlotRun did not move it)");
        ok = c.slot_save(1, 5, err);
        {
            std::lock_guard<std::mutex> lk(seen.mu);
            ok = ok && seen.save_slot == 1 && seen.save_L == 5;
        }
        check(ok, "6. a SlotSave at the session's position reaches the handler");
        bool missing = true;
        ok = c.slot_load(2, 9, 0, missing, err) && !missing;
        {
            std::lock_guard<std::mutex> lk(seen.mu);
            ok = ok && seen.load_slot == 2 && seen.load_L == 9 && seen.load_id == 0;
        }
        bool stored = false;
        ok = ok && c.ckpt_save(7, 9, {7}, stored, err) && stored;
        check(ok, "6. after a SlotLoad the session is at its L: a CkptSave there is taken");
        e.clear();
        check(!c.ckpt_save(8, 5, {7, 8}, stored, e) && contains(e, "is at 9"), "6. ... and not one at the old position");

        // ---- 7. a checkpoint id the worker does not hold
        e.clear();
        missing = false;
        const bool m = !c.slot_load(1, 9, 99, missing, e) && missing && contains(e, kStageCkptMissing) && c.connected();
        note(e);
        check(m, "7. a SlotLoad of an id the worker does not hold: missing, the link stands");
        missing = true;
        ok = c.slot_load(1, 9, 7, missing, err) && !missing;
        {
            std::lock_guard<std::mutex> lk(seen.mu);
            ok = ok && seen.load_id == 7 && seen.load_slot == 1;
        }
        check(ok, "7. ... one it holds is loaded");
    }
    check(c.stats().slot_commits >= 2 && c.stats().slot_saves == 1 && c.stats().slot_loads == 2,
          "the main side counts its slot commits, saves and loads");

    // ---- 8. malformed slot messages: an error each, the link goes on
    c.close();
    {
        Raw r;
        bool ok = r.open(where) && r.hello(hello_of());
        check(ok, "8. a hand-spoken main process is served");
        auto answer = [&](const char* want) {
            StageHeader rh;
            std::string p;
            const bool got = r.reply(rh, p) && rh.type == (uint32_t) StageMsg::Error && contains(p, want);
            note("remote stage (worker): " + p);
            return got;
        };
        auto serves_on = [&](int64_t tag) {
            r.slot_run(2, 0, tag, 1.0f);
            StageHeader rh;
            std::string p;
            return r.reply(rh, p) && rh.type == (uint32_t) StageMsg::SlotRunOk && rh.c == tag &&
                   p.size() == (size_t) (2 * kHbf * 4);
        };
        r.header(StageMsg::SlotRun, 0, 0, 1, 0);
        check(ok && answer("bad slot window") && serves_on(2), "8. S = 0: bad slot window, and the link goes on");
        r.header(StageMsg::SlotRun, kMaxT + 1, 0, 3, (uint64_t) (kMaxT + 1) * (16 + kHbf * 4));
        r.bytes((size_t) ((kMaxT + 1) * (16 + kHbf * 4)));
        check(answer("bad slot window") && serves_on(4), "8. S past the window: the same");
        r.header(StageMsg::SlotRun, (int64_t) 1 << 62, 0, 5, 0);
        check(answer("bad slot window") && serves_on(6), "8. S = 2^62 (its products wrap): the same");
        r.header(StageMsg::SlotRun, 2, 0, 7, 2 * (16 + kHbf * 4) - 4);
        r.bytes((size_t) (2 * (16 + kHbf * 4) - 4));
        check(answer("bad slot window") && serves_on(8), "8. bytes that do not add up: the same");
        r.header(StageMsg::SlotRun, 2, 2, 9, 2 * (16 + kHbf * 4));
        r.bytes((size_t) (2 * (16 + kHbf * 4)));
        check(answer("bad slot window") && serves_on(10), "8. an unknown flag bit: the same");
        r.header(StageMsg::SlotCommit, kStageSlotMax + 1, 0, 10, (uint64_t) (kStageSlotMax + 1) * 4);
        r.bytes((size_t) (kStageSlotMax + 1) * 4);
        r.slot_run(2, 0, 11, 1.0f);
        check(answer("bad slot commit") && serves_on(12),
              "8. n past kStageSlotMax (one way): the next SlotRun's error, then served");
        r.header(StageMsg::SlotCommit, 0, 0, 12, 0);
        r.slot_run(1, 0, 13, 1.0f);
        check(answer("bad slot commit") && serves_on(14), "8. n = 0: the same");
        r.header(StageMsg::SlotSave, 0, 0, 0, 4);
        r.bytes(4);
        check(answer("bad slot message") && serves_on(15), "8. a SlotSave with a payload: bad slot message");
        r.header(StageMsg::SlotLoad, kStageSlotMax, 0, 0, 0);
        check(answer("bad slot message") && serves_on(16), "8. a SlotLoad of slot kStageSlotMax: the same");
    }

    // ---- 9. a worker without slot handlers
    {
        StageHandlers h0 = h;
        h0.slot_run = nullptr;
        h0.slot_commit = nullptr;
        h0.slot_save = nullptr;
        h0.slot_load = nullptr;
        float run_in0[kMaxT * kHbf] = {}, run_out0[kMaxT * kHbf] = {};
        StageBuffers buf0 = buf;
        buf0.run_in = run_in0;
        buf0.run_out = run_out0;
        h0.run = [](int, const int32_t*, int64_t, std::string&) { return true; };
        Worker w0;
        check(w0.start(h0, buf0), "9. a worker without slot handlers listens");
        StageClient c0;
        StageHello p0;
        std::string e;
        const bool hello0 = c0.connect(w0.where, mine, p0, e) && p0.batch_slots == 0 && p0.batch_flags == 0;
        check(hello0, "9. its hello says 0 batch slots, whatever its hello handler wrote");
        std::vector<int32_t> sl(1, 0), tk(1, 0);
        std::vector<int64_t> ps(1, 0);
        std::vector<float> in((size_t) kHbf, 1.0f), out((size_t) kHbf);
        e.clear();
        bool named = !c0.slot_run(1, sl.data(), tk.data(), ps.data(), 0, 1, in.data(), in.size(), out.data(),
                                  out.size(), e) &&
                     contains(e, "--batch") && c0.connected();
        note(e);
        e.clear();
        named = named && !c0.slot_save(0, 0, e) && contains(e, "--batch");
        note(e);
        e.clear();
        bool missing = true;
        named = named && !c0.slot_load(0, 0, 0, missing, e) && !missing && contains(e, "--batch");
        note(e);
        const int32_t one[1] = {1};
        e.clear();
        named = named && c0.slot_commit(one, 1, 2, err) &&
                !c0.slot_run(1, sl.data(), tk.data(), ps.data(), 0, 3, in.data(), in.size(), out.data(), out.size(),
                             e) &&
                contains(e, "--batch");
        note(e);
        int32_t tok1 = 1;
        float r1[kHbf] = {1, 2, 3, 4}, b1[kHbf] = {};
        named = named && c0.run(1, &tok1, 0, r1, kHbf, b1, kHbf, err);
        check(named, "9. every slot message is answered naming --batch, and solo windows still run");
        c0.close();
        w0.stop();
    }

    // ---- 10. none of the new errors counts as a lost link
    err.clear();
    check(c.connect(where, mine, peer, err), "10. (the main process connects again)");
    {
        bool none = !new_errors.empty();
        for (const std::string& e : new_errors) {
            if (e.empty()) {
                std::printf("    (an empty error)\n");
                none = false;
            }
            if (stage_link_lost(e) && !contains(e, "failed earlier")) {
                std::printf("    (counted as lost: %s)\n", e.c_str());
                none = false;
            }
        }
        // (the client's own refusals, before anything goes out)
        std::string e;
        std::vector<int32_t> sl(1, 0);
        std::vector<int64_t> ps(1, 0);
        std::vector<float> in((size_t) kHbf), out((size_t) kHbf);
        if (!none) std::printf("    (above: the worker's and the link's errors)\n");
        none = none && !c.slot_run(1, sl.data(), sl.data(), ps.data(), 2, 1, in.data(), in.size(), out.data(),
                                   out.size(), e) &&
               !stage_link_lost(e) && c.connected();
        e.clear();
        none = none && !c.slot_run(2, sl.data(), sl.data(), ps.data(), 0, 1, in.data(), in.size(), out.data(),
                                   out.size(), e) &&
               !stage_link_lost(e) && c.connected();
        e.clear();
        const int32_t keep[1] = {0};
        none = none && !c.slot_commit(keep, kStageSlotMax + 1, 1, e) && !stage_link_lost(e) && c.connected();
        e.clear();
        none = none && !c.slot_run_recv(1, 1, out.data(), out.size(), e) && !stage_link_lost(e) && c.connected();
        std::printf("    (%zu errors of the new messages looked at)\n", new_errors.size());
        check(none, "10. stage_link_lost counts none of the new errors, nor the client's own refusals");
        check(window(c, 3, 0, 99, 2.0f), "10. ... and the link still serves windows");
    }

    c.close();
    wk.stop();
    check(wk.st.slot_runs > 0 && wk.st.slot_commits > 0 && wk.st.slot_saves > 0 && wk.st.slot_loads > 0,
          "the worker counts its slot messages");

    // ==== protocol 8 stage 2 (strata-mesh IMPL-PLAN-remote-batch-stage2.md 4.1): late replies and StageSlotPipe
    // A worker that answers kStageSlotKeepAll windows late, as a relay does: slot_run computes the rows (slot_rule) and
    // keeps them, slot_run_reply hands them out on the reply thread - held at `gate` until the test opens it.
    {
        float l_in[kMaxT * kHbf] = {}, l_out[kMaxT * kHbf] = {};
        float l_pf_in0[kChunk * kD] = {}, l_pf_in1[kChunk * kD] = {}, l_pf_out0[kChunk * kD] = {},
              l_pf_out1[kChunk * kD] = {};
        StageBuffers lbuf = buf;
        lbuf.run_in = l_in;
        lbuf.run_out = l_out;
        lbuf.pf_in[0] = l_pf_in0;
        lbuf.pf_in[1] = l_pf_in1;
        lbuf.pf_out[0] = l_pf_out0;
        lbuf.pf_out[1] = l_pf_out1;
        Gate gate, chunk_gate;
        Log log;
        std::mutex kept_mu;
        std::deque<std::pair<int32_t, std::vector<float>>> kept;   // (marker, rows) of the windows answered late
        std::atomic<bool> fail_reply{false};
        std::atomic<int> late_replies{0};
        StageHandlers hl;
        hl.hello = [](const StageHello&, StageHello& m, std::string&) {
            m = hello_of();
            m.batch_slots = 3;
            m.batch_flags = kStageBatchKeepAll | kStageBatchOverlap;
            return true;
        };
        hl.run = [](int, const int32_t*, int64_t, std::string&) { return true; };
        hl.commit = [](int, std::string&) { return true; };
        hl.prefill = [](const int64_t*, int64_t T, int64_t, int64_t, int64_t, const float* in, float* out, std::string&) {
            std::memcpy(out, in, (size_t) (T * kD) * sizeof(float));
            return true;
        };
        hl.prefill_reply = [&](float*, int64_t, int64_t, std::string&) {
            chunk_gate.wait();
            log.add("chunk-reply");
            return true;
        };
        hl.reset = [&](const std::vector<int64_t>&, std::string&) { log.add("reset"); return true; };
        hl.slot_run = [&](int S, const int32_t* slots, const int32_t* tokens, const int64_t*, int64_t flags,
                          std::string&) {
            std::vector<float> rows((size_t) (S * kHbf));
            for (int t = 0; t < S; ++t)
                for (int64_t j = 0; j < kHbf; ++j)
                    rows[(size_t) (t * kHbf + j)] = slot_rule(l_in[t * kHbf + j], slots[t]);
            log.add("run:" + std::to_string(tokens[0]));
            if ((flags & kStageSlotKeepAll) == 0) {
                std::memcpy(l_out, rows.data(), rows.size() * sizeof(float));
                return true;
            }
            std::lock_guard<std::mutex> lk(kept_mu);
            kept.emplace_back(tokens[0], std::move(rows));
            return true;
        };
        hl.slot_run_reply = [&](int S, float* out, std::string& e) {
            ++late_replies;
            gate.wait();
            std::pair<int32_t, std::vector<float>> w;
            {
                std::lock_guard<std::mutex> lk(kept_mu);
                if (kept.empty()) { e = "the test's late reply: no window kept"; return false; }
                w = std::move(kept.front());
                kept.pop_front();
            }
            if (fail_reply.exchange(false)) {
                log.add("reply-failed:" + std::to_string(w.first));
                e = "the test's late reply failed";
                return false;
            }
            if (w.second.size() != (size_t) (S * kHbf)) { e = "the test's late reply: S differs"; return false; }
            std::memcpy(out, w.second.data(), w.second.size() * sizeof(float));
            log.add("reply:" + std::to_string(w.first));
            return true;
        };
        hl.slot_commit = [&](const int32_t*, int, std::string&) { log.add("commit"); return true; };
        hl.slot_save = [&](int, int64_t, std::string&) { log.add("save"); return true; };
        hl.slot_load = [](int, int64_t, int64_t, std::string&) { return true; };
        Worker lw;
        check(lw.start(hl, lbuf), "11. a worker that answers late listens on loopback");
        StageHello lmine = hello_of();
        lmine.batch_slots = 3;
        lmine.batch_flags = kStageBatchKeepAll;

        // a window's inputs: S rows, row t on slot t % 3, its first token the window's marker
        struct Win {
            int S = 0;
            std::vector<int32_t> sl, tk;
            std::vector<int64_t> ps;
            std::vector<float> in, out;
        };
        auto make = [](int S, int32_t marker, float base) {
            Win w;
            w.S = S;
            w.sl.resize((size_t) S);
            w.tk.resize((size_t) S);
            w.ps.resize((size_t) S);
            w.in.resize((size_t) (S * kHbf));
            w.out.assign((size_t) (S * kHbf), -1.0f);
            for (int t = 0; t < S; ++t) {
                w.sl[(size_t) t] = t % 3;
                w.tk[(size_t) t] = marker + t;
                w.ps[(size_t) t] = 10 + t;
                for (int64_t j = 0; j < kHbf; ++j) w.in[(size_t) (t * kHbf + j)] = base + (float) (t * kHbf + j);
            }
            return w;
        };
        auto rows_ok = [](const Win& w) {
            for (int t = 0; t < w.S; ++t)
                for (int64_t j = 0; j < kHbf; ++j)
                    if (w.out[(size_t) (t * kHbf + j)] != slot_rule(w.in[(size_t) (t * kHbf + j)], w.sl[(size_t) t]))
                        return false;
            return true;
        };
        auto send = [](StageClient& cl, Win& w, int64_t flags, int64_t tag) {
            std::string e;
            if (cl.slot_run_send(w.S, w.sl.data(), w.tk.data(), w.ps.data(), flags, tag, w.in.data(), w.in.size(), e))
                return true;
            std::printf("    (slot_run_send: %s)\n", e.c_str());
            return false;
        };
        auto recv = [](StageClient& cl, Win& w, int64_t tag, std::string& e) {
            return cl.slot_run_recv(tag, w.S, w.out.data(), w.out.size(), e);
        };
        // waits (2 s at most) until `n` entries starting with `what` are in the log
        auto wait_for = [&](const std::string& what, int n) {
            for (int i = 0; i < 200; ++i) {
                if (log.count(what) >= n) return true;
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
            return false;
        };

        // ---- 11. late replies keep the windows' order; the next window runs before the reply before it is out
        StageClient lc;
        StageHello lpeer;
        std::string le;
        check(lc.connect(lw.where, lmine, lpeer, le), "11. (the main process connects)");
        {
            log.clear();
            gate.close();
            Win w1 = make(2, 1000, 1.0f), w2 = make(3, 2000, 50.0f), w3 = make(1, 3000, 90.0f);
            bool ok = send(lc, w1, kStageSlotKeepAll, 101) && send(lc, w2, kStageSlotKeepAll, 102) &&
                      send(lc, w3, kStageSlotKeepAll, 103);
            // window 2 runs while window 1's reply waits at the gate; window 3 waits for window 1's reply to be out
            const bool second_ran = wait_for("run:", 2);
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            const bool held = log.count("run:") == 2 && log.count("reply:") == 0;
            gate.open();
            std::string e;
            ok = ok && recv(lc, w1, 101, e) && recv(lc, w2, 102, e) && recv(lc, w3, 103, e);
            if (!ok) std::printf("    (%s)\n", e.c_str());
            check(ok && second_ran && held, "11. window 2 runs before window 1's reply is out (window 3 does not)");
            check(ok && rows_ok(w1) && rows_ok(w2) && rows_ok(w3) &&
                      log.order({"run:1000", "run:2000", "reply:1000", "run:3000"}) &&
                      log.order({"reply:1000", "reply:2000", "reply:3000"}),
                  "11. the replies come back in the order sent, with their tags and rows");
        }

        // ---- 12. a window without kStageSlotKeepAll is answered at once, not late
        {
            log.clear();
            const int before = late_replies.load();
            gate.close();   // (a late reply would wait here: this one must not)
            Win w = make(2, 4000, 7.0f);
            std::string e;
            const bool ok = send(lc, w, 0, 201) && recv(lc, w, 201, e);
            gate.open();
            check(ok && rows_ok(w) && late_replies.load() == before, "12. a window without bit 0 is not answered late");
        }

        // ---- 13. a late window behind a prompt chunk's reply still going out runs after that reply
        {
            log.clear();
            chunk_gate.close();
            int64_t toks[2] = {5, 6};
            float rows[2 * kD] = {1, 2, 3, 4, 5, 6, 7, 8}, back[2 * kD] = {};
            std::string e;
            Win w = make(1, 5000, 3.0f);
            bool ok = lc.prefill_send(toks, 2, 0, 0, rows, (size_t) kD, 0, e) && send(lc, w, kStageSlotKeepAll, 301);
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            const bool waited = log.count("run:") == 0;
            chunk_gate.open();
            ok = ok && lc.prefill_recv(back, (size_t) kD, 2, 0, e) && recv(lc, w, 301, e);
            if (!ok) std::printf("    (%s)\n", e.c_str());
            check(ok && waited && rows_ok(w) && back[7] == 8.0f && log.order({"chunk-reply", "run:5000", "reply:5000"}),
                  "13. a late window waits for a prompt chunk's reply still going out");
        }

        // ---- 14. a late reply's failure is that window's error; the later windows are answered in order
        {
            log.clear();
            fail_reply = true;
            Win w1 = make(1, 6000, 1.0f), w2 = make(2, 7000, 2.0f), w3 = make(1, 8000, 3.0f);
            std::string e1, e2, e3;
            const bool sent = send(lc, w1, kStageSlotKeepAll, 401) && send(lc, w2, kStageSlotKeepAll, 402) &&
                              send(lc, w3, kStageSlotKeepAll, 403);
            const bool r1 = recv(lc, w1, 401, e1), r2 = recv(lc, w2, 402, e2), r3 = recv(lc, w3, 403, e3);
            check(sent && !r1 && contains(e1, "the test's late reply failed") && !stage_link_lost(e1) && r2 && r3 &&
                      rows_ok(w2) && rows_ok(w3) && lc.connected(),
                  "14. a late reply's failure: that window's error, the next ones answered, the link goes on");
        }

        // ---- 15. SlotCommit, Reset and SlotSave after a late window are served once its reply is out
        lc.close();
        {
            log.clear();
            gate.close();
            Raw r;
            bool ok = r.open(lw.where) && r.hello(lmine);
            r.slot_run(2, kStageSlotKeepAll, 501, 4.0f);
            const int32_t keep[2] = {1, 1};
            r.header(StageMsg::SlotCommit, 2, 0, 501, sizeof keep);
            send_n(r.s, keep, sizeof keep);
            r.header(StageMsg::Reset, 0, 0, 0, 0);
            r.header(StageMsg::SlotSave, 1, 0, 0, 0);
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            const bool held = log.count("run:") == 1 && log.count("commit") == 0 && log.count("reset") == 0 &&
                              log.count("save") == 0;
            gate.open();
            StageHeader h1, h2, h3;
            std::string p1, p2, p3;
            ok = ok && r.reply(h1, p1) && h1.type == (uint32_t) StageMsg::SlotRunOk && h1.c == 501 &&
                 p1.size() == (size_t) (2 * kHbf * 4) && r.reply(h2, p2) && h2.type == (uint32_t) StageMsg::ResetOk &&
                 r.reply(h3, p3) && h3.type == (uint32_t) StageMsg::SlotSaveOk;
            check(ok && held && log.order({"run:100", "reply:100", "commit", "reset", "save"}),
                  "15. SlotCommit, Reset and SlotSave wait for the late reply before them");
        }

        // ---- 16. StageSlotPipe
        {
            StageClient pc;
            std::string e;
            check(pc.connect(lw.where, lmine, lpeer, e), "16. (the main process connects)");
            gate.close();
            Win a = make(2, 9000, 1.0f), b = make(1, 9100, 20.0f);
            bool ok = true;
            {
                StageSlotPipe pipe(pc);
                ok = pipe.send(3, a.S, a.sl.data(), a.tk.data(), a.ps.data(), kStageSlotKeepAll, 601, a.in.data(),
                               a.in.size(), a.out.data(), a.out.size(), e) &&
                     pipe.send(5, b.S, b.sl.data(), b.tk.data(), b.ps.data(), kStageSlotKeepAll, 602, b.in.data(),
                               b.in.size(), b.out.data(), b.out.size(), e);
                std::this_thread::sleep_for(std::chrono::milliseconds(30));
                const bool none_yet = pipe.poll(3, e) == 0 && pipe.poll(5, e) == 0 && pipe.poll(4, e) == 0 &&
                                      pipe.outstanding() == 2;
                gate.open();
                // (polled out of the order sent: window 5 first)
                int p5 = 0, p3 = 0;
                for (int i = 0; i < 500 && p5 == 0; ++i) {
                    p5 = pipe.poll(5, e);
                    if (p5 == 0) std::this_thread::sleep_for(std::chrono::milliseconds(2));
                }
                const int left = pipe.outstanding();
                p3 = pipe.poll(3, e);
                check(ok && none_yet && p5 == 1 && left == 1 && p3 == 1 && pipe.outstanding() == 0 && rows_ok(a) &&
                          rows_ok(b),
                      "16. pipe: poll 0 while out, then 1 with the rows in place, in any order; outstanding counts");
                check(pipe.poll(3, e) == 0, "16. pipe: a window taken is not in again");
                pipe.stop();
                pipe.stop();
            }
            // a pipe stopped with nothing out leaves the link as it was
            {
                Win w = make(1, 9200, 5.0f);
                const bool on = pc.slot_run(w.S, w.sl.data(), w.tk.data(), w.ps.data(), 0, 603, w.in.data(),
                                            w.in.size(), w.out.data(), w.out.size(), e) &&
                                rows_ok(w);
                check(on && pc.connected(), "16. pipe: stopped with nothing out, the link serves on");
            }
            // the worker's Error: -1 with its words, and every later call too
            {
                StageSlotPipe pipe(pc);
                fail_reply = true;
                Win w = make(1, 9300, 6.0f), w2 = make(1, 9400, 6.0f);
                ok = pipe.send(0, w.S, w.sl.data(), w.tk.data(), w.ps.data(), kStageSlotKeepAll, 604, w.in.data(),
                               w.in.size(), w.out.data(), w.out.size(), e);
                int p = 0;
                for (int i = 0; i < 500 && p == 0; ++i) {
                    p = pipe.poll(0, e);
                    if (p == 0) std::this_thread::sleep_for(std::chrono::milliseconds(2));
                }
                std::string e2;
                const bool later = pipe.poll(1, e2) == -1 && contains(e2, "the test's late reply failed") &&
                                   !pipe.send(1, w2.S, w2.sl.data(), w2.tk.data(), w2.ps.data(), kStageSlotKeepAll, 605,
                                              w2.in.data(), w2.in.size(), w2.out.data(), w2.out.size(), e2);
                check(ok && p == -1 && contains(e, "the test's late reply failed") && later,
                      "16. pipe: the worker's error is -1 with its words, and so is every later call");
            }
            check(pc.connected(), "16. pipe: (a worker's error leaves the link standing)");
            // stop() with a window out: the read waiting on the worker (its reply held) is cut at once
            {
                StageSlotPipe pipe(pc);
                gate.close();
                Win w = make(1, 9500, 6.0f);
                ok = pipe.send(7, w.S, w.sl.data(), w.tk.data(), w.ps.data(), kStageSlotKeepAll, 606, w.in.data(),
                               w.in.size(), w.out.data(), w.out.size(), e);
                std::this_thread::sleep_for(std::chrono::milliseconds(50));   // (the reader is in its read)
                const int out = pipe.outstanding();
                const auto t0 = std::chrono::steady_clock::now();
                pipe.stop();
                const double ms =
                    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
                std::string e4;
                const int p7 = pipe.poll(7, e4);
                check(ok && out == 1 && ms < 1000.0 && p7 == -1 && contains(e4, "stopped with window 7 out") &&
                          !pc.connected(),
                      "16. pipe: stop() cuts a read waiting for a held reply at once; the link is down");
                std::printf("    (stop() took %.1f ms: %s)\n", ms, e4.c_str());
                gate.open();
            }
            pc.close();
            // a number sent twice (and one out of range): an error, and the link is cut as the pipe stops
            check(pc.connect(lw.where, lmine, lpeer, e), "16. (the main process connects again)");
            {
                StageSlotPipe pipe(pc);
                gate.close();
                Win w = make(1, 9600, 6.0f), w2 = make(1, 9700, 6.0f);
                ok = pipe.send(9, w.S, w.sl.data(), w.tk.data(), w.ps.data(), kStageSlotKeepAll, 607, w.in.data(),
                               w.in.size(), w.out.data(), w.out.size(), e);
                std::string e2, e3;
                const bool twice = !pipe.send(9, w2.S, w2.sl.data(), w2.tk.data(), w2.ps.data(), kStageSlotKeepAll, 608,
                                              w2.in.data(), w2.in.size(), w2.out.data(), w2.out.size(), e2) &&
                                   contains(e2, "sent again");
                StageSlotPipe pipe2(pc);   // (out of range, on a pipe of its own: nothing goes out)
                const bool range = !pipe2.send(kStageSlotMax, w2.S, w2.sl.data(), w2.tk.data(), w2.ps.data(),
                                               kStageSlotKeepAll, 608, w2.in.data(), w2.in.size(), w2.out.data(),
                                               w2.out.size(), e3) &&
                                   contains(e3, "batch window number");
                check(ok && twice && range, "16. pipe: a number sent again before its reply was taken, or out of "
                                            "range, is an error");
                pipe.stop();
                check(!pc.connected(), "16. pipe: stopped with a window out, the link is down");
                gate.open();
            }
            // reply rows of another width than the window's (1 row of 4 in, 8 floats out; Codex [P2]): refused before
            // anything goes out, so no reply is left unread on a link that looks in step
            pc.close();
            check(pc.connect(lw.where, lmine, lpeer, e), "16. (the main process connects again)");
            {
                StageSlotPipe pipe(pc);
                Win w = make(1, 9800, 6.0f);
                std::vector<float> wide((size_t) (2 * kHbf), -1.0f);
                std::string we;
                const bool refused = !pipe.send(2, w.S, w.sl.data(), w.tk.data(), w.ps.data(), kStageSlotKeepAll, 609,
                                                w.in.data(), w.in.size(), wide.data(), wide.size(), we) &&
                                     contains(we, "hand-off width") && pipe.outstanding() == 0;
                pipe.stop();
                std::string e5;
                const bool on = pc.connected() &&
                                pc.slot_run(w.S, w.sl.data(), w.tk.data(), w.ps.data(), 0, 610, w.in.data(),
                                            w.in.size(), w.out.data(), w.out.size(), e5) &&
                                rows_ok(w);
                if (!on) std::printf("    (%s / %s)\n", we.c_str(), e5.c_str());
                check(refused && on, "16. pipe: reply rows wider than the window's are refused before the send; the "
                                     "link serves on");
            }
            pc.close();
        }

    // ---- 17. a relay's failure after its send: main -> relay -> end on loopback (Codex 1 [P2]).  The relay's
    // handlers are the shape a relay worker writes (IMPL-PLAN-remote-batch-stage2.md 2.3.3): slot_run sends the window
    // on and queues {tag, S, a failure after the send}; slot_run_reply reads the next worker's replies, on the reply
    // thread only, in the order sent - a failed window's too, before its error.  `wrong`: the failed window's reply is
    // read on the serving thread instead (the form that breaks the link: it overtakes the read before it).
    auto relay_case = [&](bool wrong, std::string& ea, std::string& eb, std::string& ec, bool& rows_a, bool& rows_c,
                          bool& next_up, bool& d_on) -> bool {
        // the end worker: slot_rule on the rows it gets
        float e_in[kMaxT * kHbf] = {}, e_out[kMaxT * kHbf] = {};
        StageBuffers ebuf = buf;
        ebuf.run_in = e_in;
        ebuf.run_out = e_out;
        StageHandlers he;
        he.hello = [](const StageHello&, StageHello& m, std::string&) {
            m = hello_of();
            m.batch_slots = 3;
            m.batch_flags = kStageBatchKeepAll;
            return true;
        };
        he.run = [](int, const int32_t*, int64_t, std::string&) { return true; };
        he.commit = [](int, std::string&) { return true; };
        he.prefill = [](const int64_t*, int64_t, int64_t, int64_t, int64_t, const float*, float*, std::string&) {
            return true;
        };
        he.reset = [](const std::vector<int64_t>&, std::string&) { return true; };
        he.slot_run = [&](int S, const int32_t* slots, const int32_t*, const int64_t*, int64_t, std::string&) {
            for (int t = 0; t < S; ++t)
                for (int64_t j = 0; j < kHbf; ++j) e_out[t * kHbf + j] = slot_rule(e_in[t * kHbf + j], slots[t]);
            return true;
        };
        he.slot_commit = [](const int32_t*, int, std::string&) { return true; };
        he.slot_save = [](int, int64_t, std::string&) { return true; };
        he.slot_load = [](int, int64_t, int64_t, std::string&) { return true; };
        Worker end;
        if (!end.start(he, ebuf)) return false;

        // the relay: its own layers add 1 to every value, then the end's rule
        float r_in[kMaxT * kHbf] = {}, r_out[kMaxT * kHbf] = {}, r_mine[kMaxT * kHbf] = {};
        StageBuffers rbuf = buf;
        rbuf.run_in = r_in;
        rbuf.run_out = r_out;
        StageClient next;
        struct Q {
            int64_t tag = 0;
            int S = 0;
            std::string failed;   // a failure after the send: the window's reply is still read, then this error
        };
        std::mutex qmu;
        std::deque<Q> q;
        int64_t next_tag = 0;
        std::atomic<int> runs{0};
        Gate hold;   // window A's reply thread waits here, just before its read
        std::atomic<bool> held_once{false};
        StageHandlers hr;
        hr.hello = [&](const StageHello&, StageHello& m, std::string& e) {
            m = hello_of();
            m.batch_slots = 3;
            m.batch_flags = kStageBatchKeepAll | kStageBatchOverlap;
            if (next.connected()) return true;
            StageHello nm = hello_of(), np;
            nm.batch_slots = 3;
            nm.batch_flags = kStageBatchKeepAll;
            return next.connect(end.where, nm, np, e);
        };
        hr.run = [](int, const int32_t*, int64_t, std::string&) { return true; };
        hr.commit = [](int, std::string&) { return true; };
        hr.prefill = [](const int64_t*, int64_t, int64_t, int64_t, int64_t, const float*, float*, std::string&) {
            return true;
        };
        hr.reset = [](const std::vector<int64_t>&, std::string&) { return true; };
        hr.slot_run = [&](int S, const int32_t* slots, const int32_t* tokens, const int64_t* pos, int64_t flags,
                          std::string& e) {
            for (int64_t i = 0; i < S * kHbf; ++i) r_mine[i] = r_in[i] + 1.0f;   // (this worker's layers)
            const int64_t tag = ++next_tag;
            if (!next.slot_run_send(S, slots, tokens, pos, flags, tag, r_mine, (size_t) (S * kHbf), e))
                return false;   // nothing went on (or the link broke): answered at once
            // this worker's own commit, after the send: the test fails window B's (marker 2000)
            const std::string failed = tokens[0] == 2000 ? "the relay's own commit failed (test)" : "";
            if ((flags & kStageSlotKeepAll) == 0) {   // answered at once: the reply read here, as in stage 1
                std::string re;
                const bool got = next.slot_run_recv(tag, S, r_out, (size_t) (S * kHbf), re);
                if (!failed.empty()) { e = failed; return false; }
                if (!got) e = re;
                return got;
            }
            if (wrong && !failed.empty()) {   // the wrong form: read here, ahead of the reply thread
                std::string re;
                (void) next.slot_run_recv(tag, S, r_out, (size_t) (S * kHbf), re);
                ++runs;
                e = failed;
                return false;
            }
            {
                std::lock_guard<std::mutex> lk(qmu);
                q.push_back(Q{tag, S, failed});
            }
            ++runs;
            return true;
        };
        hr.slot_run_reply = [&](int S, float* rows_out, std::string& e) {
            if (!held_once.exchange(true)) hold.wait();   // (window A only)
            Q w;
            {
                std::lock_guard<std::mutex> lk(qmu);
                if (q.empty() || q.front().S != S) { e = "the relay's queue is out of step"; return false; }
                w = q.front();
                q.pop_front();
            }
            std::string re;
            const bool got = next.slot_run_recv(w.tag, S, rows_out, (size_t) (S * kHbf), re);
            if (!w.failed.empty()) { e = w.failed; return false; }
            if (!got) e = re;
            return got;
        };
        hr.slot_commit = [](const int32_t*, int, std::string&) { return true; };
        hr.slot_save = [](int, int64_t, std::string&) { return true; };
        hr.slot_load = [](int, int64_t, int64_t, std::string&) { return true; };
        hr.disconnected = [&] { next.close(); };
        Worker relay;
        if (!relay.start(hr, rbuf)) { end.stop(); return false; }

        StageClient mc;
        StageHello mm = hello_of(), mp;
        mm.batch_slots = 3;
        mm.batch_flags = kStageBatchKeepAll;
        std::string e;
        bool ok = mc.connect(relay.where, mm, mp, e);
        hold.close();
        Win a = make(2, 1000, 1.0f), b = make(1, 2000, 10.0f), c3 = make(2, 3000, 20.0f), d = make(1, 4000, 30.0f);
        ok = ok && send(mc, a, kStageSlotKeepAll, 11) && send(mc, b, kStageSlotKeepAll, 12) &&
             send(mc, c3, kStageSlotKeepAll, 13);
        // B has gone on to the end worker and failed after it (A's reply thread still waits before its read)
        for (int i = 0; i < 200 && runs.load() < 2; ++i) std::this_thread::sleep_for(std::chrono::milliseconds(10));
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
        hold.open();
        const bool ra = recv(mc, a, 11, ea), rb = recv(mc, b, 12, eb), rc = recv(mc, c3, 13, ec);
        // the end's rule over the relay's (+1): the rows main gets
        auto relayed = [](const Win& w) {
            for (int t = 0; t < w.S; ++t)
                for (int64_t j = 0; j < kHbf; ++j)
                    if (w.out[(size_t) (t * kHbf + j)] !=
                        slot_rule(w.in[(size_t) (t * kHbf + j)] + 1.0f, w.sl[(size_t) t]))
                        return false;
            return true;
        };
        rows_a = ra && relayed(a);
        rows_c = rc && relayed(c3);
        if (!rb && eb.empty()) eb = "(none)";
        if (rb) eb.clear();
        next_up = next.connected();
        std::string ed;
        d_on = mc.connected() && send(mc, d, kStageSlotKeepAll, 14) && recv(mc, d, 14, ed) && relayed(d);
        mc.close();
        relay.stop();
        end.stop();
        return ok && runs.load() >= 2;
    };
    {
        std::string ea, eb, ec;
        bool rows_a = false, rows_c = false, next_up = false, d_on = false;
        const bool ran = relay_case(false, ea, eb, ec, rows_a, rows_c, next_up, d_on);
        check(ran && rows_a && contains(eb, "the relay's own commit failed (test)") && !stage_link_lost(eb) && rows_c &&
                  next_up && d_on,
              "17. a relay's failure after its send: A's rows, B's error, C's rows, in order; the links stand");
        if (!(rows_a && rows_c)) std::printf("    (A: %s; C: %s)\n", ea.c_str(), ec.c_str());
        ea.clear(), eb.clear(), ec.clear();
        const bool ran2 = relay_case(true, ea, eb, ec, rows_a, rows_c, next_up, d_on);
        check(ran2 && !rows_a && !next_up,
              "17. ... and the wrong form (B's reply read on the serving thread) breaks the relay's next link");
        std::printf("    (the wrong form: A: %s)\n", ea.c_str());
    }
        lw.stop();
    }

    std::printf("%s (%d failed)\n", g_fail == 0 ? "all passed" : "FAILED", g_fail);
    return g_fail == 0 ? 0 : 1;
}
