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
#include "strata/net/stage_link.hpp"

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

    std::printf("%s (%d failed)\n", g_fail == 0 ? "all passed" : "FAILED", g_fail);
    return g_fail == 0 ? 0 : 1;
}
