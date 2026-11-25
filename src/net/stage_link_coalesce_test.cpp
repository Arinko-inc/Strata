// src/net/stage_link_coalesce_test.cpp - STRATA_LINK_COALESCE (include/strata/net/stage_link_coalesce.hpp), no GPU.
// POSIX only: it uses socketpair (and, for the short writes, SIGALRM); on Windows it only prints a skip line.
//
//   1. send_msg_rows puts out the bytes of the header, p1 and the rows' wire form (fp32 / bf16 / fp8, 0..100 rows, two
//      widths, slices of 25 rows), also with the first slice and the rest in separate sendmsg calls;
//   2. send_parts through short writes (SIGALRM without SA_RESTART on a small socket buffer): every byte, in order;
//      a send timeout is "remote stage: send failed: ..." as send_all words it;
//   3. recv_parts: a message in random pieces into 4 sinks; the peer closing and the receive timeout have recv_all's
//      error text;
//   4. what recv_ids_rows does (ids and one slice of rows in one recvmsg, decoded after) built from the public calls;
//   5. a worker (serve_stage) and a client (StageClient) on loopback with the flag off, on, and flipped at random
//      from another thread while windows, prompt chunks and batch windows go through (the bytes are the same either
//      way, so every mix of old and new ends must give the same rows);
//   6. on() and force_for_test.
// Sections 5 and the existing stage_link_test / stage_link_batch_test also run with STRATA_LINK_COALESCE=1 in CTest.
#include "strata/net/stage_link.hpp"
#include "strata/net/stage_link_coalesce.hpp"

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

#if defined(_WIN32)
int main() {
    std::printf("stage_link_coalesce_test: skipped (POSIX only)\n");
    return 0;
}
#else
#include <csignal>
#include <poll.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

using namespace strata::net;
namespace co = strata::net::coalesce;

namespace {

int g_failed = 0;
void check(bool ok, const std::string& what) {
    std::printf("  %s %s\n", what.c_str(), ok ? "ok" : "FAILED");
    if (!ok) ++g_failed;
}

#if defined(MSG_NOSIGNAL)
constexpr int kFlags = MSG_NOSIGNAL;
#else
constexpr int kFlags = 0;
#endif

bool pair_of(int sv[2]) { return ::socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0; }

#if defined(SO_NOSIGPIPE)
void no_sigpipe(int fd) {
    int one = 1;
    (void) setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &one, sizeof one);
}
#else
void no_sigpipe(int) {}
#endif

// n bytes from fd (until n, or the peer closes)
void read_n(int fd, std::vector<unsigned char>& out, size_t n, size_t piece = 1 << 16, int pause_us = 0) {
    out.resize(n);
    size_t got = 0;
    while (got < n) {
        const ssize_t r = ::recv(fd, out.data() + got, std::min(piece, n - got), 0);
        if (r <= 0) {
            if (r < 0 && errno == EINTR) continue;
            break;
        }
        got += (size_t) r;
        if (pause_us > 0) std::this_thread::sleep_for(std::chrono::microseconds(pause_us));
    }
    out.resize(got);
}

// values with the cases a row format cares about: NaN, infinities, zero blocks, denormals, a wide range
std::vector<float> values(size_t n, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> u(-3.0f, 3.0f);
    std::vector<float> v(n);
    for (size_t i = 0; i < n; ++i) {
        const uint32_t k = rng() % 97;
        float x = u(rng);
        if (k == 0) x = std::numeric_limits<float>::quiet_NaN();
        else if (k == 1) x = std::numeric_limits<float>::infinity();
        else if (k == 2) x = -std::numeric_limits<float>::infinity();
        else if (k == 3) x = std::numeric_limits<float>::denorm_min() * (float) (rng() % 1000);
        else if (k == 4) x = 0.0f;
        else if (k == 5) x *= 1.0e30f;
        else if (k == 6) x *= 1.0e-30f;
        v[i] = x;
    }
    // a block of zeros (the scale of an all-zero fp8 block)
    for (size_t i = 128; i < std::min<size_t>(n, 384); ++i) v[i] = 0.0f;
    return v;
}

const char* fmt_name(StageRows f) { return f == StageRows::F32 ? "fp32" : f == StageRows::BF16 ? "bf16" : "fp8"; }

// ---- 1.
void test_bytes() {
    std::printf("1. send_msg_rows: the bytes of header + p1 + rows\n");
    const StageRows fmts[] = {StageRows::F32, StageRows::BF16, StageRows::FP8};
    const size_t widths[] = {12804, 10240};
    const size_t row_counts[] = {0, 1, 4, 6, 100};
    const size_t n1s[] = {0, 16, 48};
    int cases = 0, bad = 0;
    for (StageRows f : fmts)
        for (size_t width : widths)
            for (size_t rows : row_counts)
                for (size_t n1 : n1s) {
                    const size_t step = rows == 100 ? 25 : 256;   // 100 rows go as 4 slices
                    std::vector<float> src = values(rows * width, (uint32_t) (rows * 7 + n1));
                    std::vector<unsigned char> p1(n1);
                    for (size_t i = 0; i < n1; ++i) p1[i] = (unsigned char) (i * 31 + 5);
                    StageHeader h;
                    h.type = 5;
                    h.a = (int64_t) rows;
                    h.b = 42;
                    h.bytes = n1 + (f == StageRows::F32 ? rows * width * 4 : stage_rows_bytes(f, rows, width));
                    // what send_msg(h, p1) and send_rows(f, ...) write
                    std::vector<unsigned char> want(sizeof h + n1);
                    std::memcpy(want.data(), &h, sizeof h);
                    if (n1) std::memcpy(want.data() + sizeof h, p1.data(), n1);
                    if (rows > 0) {
                        if (f == StageRows::F32) {
                            const size_t o = want.size();
                            want.resize(o + rows * width * 4);
                            std::memcpy(want.data() + o, src.data(), rows * width * 4);
                        } else {
                            // slice by slice, as send_rows encodes it
                            for (size_t r = 0; r < rows; r += step) {
                                const size_t k = std::min(step, rows - r);
                                const size_t o = want.size();
                                want.resize(o + stage_rows_bytes(f, k, width));
                                stage_rows_encode(f, src.data() + r * width, k, width, want.data() + o);
                            }
                        }
                    }
                    int sv[2];
                    if (!pair_of(sv)) { check(false, "socketpair"); return; }
                    no_sigpipe(sv[0]);
                    std::vector<unsigned char> got;
                    std::thread rd([&] { read_n(sv[1], got, want.size()); });
                    std::vector<unsigned char> buf;
                    double conv = 0;
                    std::string err;
                    const bool ok = co::send_msg_rows(sv[0], kFlags, h, p1.data(), n1, f, src.data(), rows, width, step,
                                                      buf, conv, err);
                    ::close(sv[0]);
                    rd.join();
                    ::close(sv[1]);
                    ++cases;
                    if (!ok || got != want) {
                        ++bad;
                        std::printf("    %s width %zu rows %zu n1 %zu: %s (got %zu bytes, want %zu)\n", fmt_name(f),
                                    width, rows, n1, ok ? "bytes differ" : err.c_str(), got.size(), want.size());
                    }
                }
    check(bad == 0, "the same bytes in " + std::to_string(cases) + " cases (3 formats x 2 widths x 5 row counts x 3 p1)");
}

// ---- 2.
std::atomic<int> g_alarms{0};
void on_alarm(int) { ++g_alarms; }

void test_short_writes() {
    std::printf("2. send_parts through short writes and a send timeout\n");
    struct sigaction sa {};
    sa.sa_handler = on_alarm;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;   // no SA_RESTART: a blocked sendmsg gives back what it has written (or EINTR)
    struct sigaction old {};
    sigaction(SIGALRM, &sa, &old);
    itimerval tv{};
    tv.it_interval.tv_usec = 1000;
    tv.it_value.tv_usec = 1000;

    const size_t n0 = 40, n1 = 3 << 20, n2 = 5 << 20;
    std::vector<unsigned char> a(n0), b(n1), c(n2);
    std::mt19937 rng(7);
    for (auto* v : {&a, &b, &c})
        for (auto& x : *v) x = (unsigned char) rng();
    std::vector<unsigned char> want;
    want.insert(want.end(), a.begin(), a.end());
    want.insert(want.end(), b.begin(), b.end());
    want.insert(want.end(), c.begin(), c.end());

    int sv[2];
    if (!pair_of(sv)) { check(false, "socketpair"); return; }
    no_sigpipe(sv[0]);
    int small = 4096;
    (void) setsockopt(sv[0], SOL_SOCKET, SO_SNDBUF, &small, sizeof small);
    (void) setsockopt(sv[1], SOL_SOCKET, SO_RCVBUF, &small, sizeof small);
    std::vector<unsigned char> got;
    std::thread rd([&] { read_n(sv[1], got, want.size(), 1 << 14, 100); });
    g_alarms = 0;
    setitimer(ITIMER_REAL, &tv, nullptr);
    std::string err;
    const co::Part ps[4] = {{a.data(), a.size()}, {nullptr, 0}, {b.data(), b.size()}, {c.data(), c.size()}};
    const bool ok = co::send_parts(sv[0], kFlags, ps, 4, err);
    itimerval off{};
    setitimer(ITIMER_REAL, &off, nullptr);
    sigaction(SIGALRM, &old, nullptr);
    ::close(sv[0]);
    rd.join();
    ::close(sv[1]);
    check(ok && got == want, "8 MiB in 3 parts (and an empty one) through a 4 KiB buffer, every byte in order");
    check(g_alarms.load() > 0, "signals arrived during it (" + std::to_string(g_alarms.load()) + ")");

    // a send timeout: nobody reads; the error is send_all's
    if (!pair_of(sv)) { check(false, "socketpair"); return; }
    no_sigpipe(sv[0]);
    (void) setsockopt(sv[0], SOL_SOCKET, SO_SNDBUF, &small, sizeof small);
    timeval to{};
    to.tv_usec = 150 * 1000;
    (void) setsockopt(sv[0], SOL_SOCKET, SO_SNDTIMEO, &to, sizeof to);
    const co::Part big[2] = {{b.data(), b.size()}, {c.data(), c.size()}};
    err.clear();
    const bool ok2 = co::send_parts(sv[0], kFlags, big, 2, err);
    check(!ok2 && err.rfind("remote stage: send failed: ", 0) == 0, "a send timeout: \"" + err + "\"");
    ::close(sv[0]);
    ::close(sv[1]);
    // the peer gone
    if (!pair_of(sv)) { check(false, "socketpair"); return; }
    no_sigpipe(sv[0]);
    ::close(sv[1]);
    err.clear();
    const co::Part one[1] = {{a.data(), a.size()}};
    const bool ok3 = co::send_parts(sv[0], kFlags, one, 1, err);
    check(!ok3 && err.rfind("remote stage: send failed: ", 0) == 0, "a closed peer: \"" + err + "\"");
    ::close(sv[0]);
}

// ---- 3.
void test_recv_parts() {
    std::printf("3. recv_parts\n");
    const size_t sizes[4] = {40, 1000, 70000, 3 << 20};
    size_t total = 0;
    for (size_t s : sizes) total += s;
    std::vector<unsigned char> msg(total);
    std::mt19937 rng(11);
    for (auto& x : msg) x = (unsigned char) rng();
    int sv[2];
    if (!pair_of(sv)) { check(false, "socketpair"); return; }
    no_sigpipe(sv[0]);
    std::thread wr([&] {
        std::mt19937 r(3);
        size_t o = 0;
        while (o < msg.size()) {
            const size_t k = std::min<size_t>(1 + r() % 40000, msg.size() - o);
            const ssize_t w = ::send(sv[0], msg.data() + o, k, kFlags);
            if (w <= 0) break;
            o += (size_t) w;
            if (r() % 8 == 0) std::this_thread::sleep_for(std::chrono::microseconds(200));
        }
    });
    std::vector<unsigned char> s0(sizes[0]), s1(sizes[1]), s2(sizes[2]), s3(sizes[3]);
    const co::Sink sk[5] = {{s0.data(), s0.size()}, {s1.data(), s1.size()}, {nullptr, 0}, {s2.data(), s2.size()},
                            {s3.data(), s3.size()}};
    std::string err;
    const bool ok = co::recv_parts(sv[1], sk, 5, err);
    wr.join();
    std::vector<unsigned char> back;
    for (auto* v : {&s0, &s1, &s2, &s3}) back.insert(back.end(), v->begin(), v->end());
    check(ok && back == msg, "a message in random pieces into 4 sinks (and an empty one) comes whole");

    // the peer closes in the middle
    std::thread wr2([&] {
        (void) ::send(sv[0], msg.data(), 100, kFlags);
        ::close(sv[0]);
    });
    err.clear();
    std::vector<unsigned char> t(1000);
    const co::Sink sk2[1] = {{t.data(), t.size()}};
    const bool ok2 = co::recv_parts(sv[1], sk2, 1, err);
    wr2.join();
    check(!ok2 && err == "remote stage: the peer closed the connection", "the peer closing: \"" + err + "\"");
    ::close(sv[1]);

    // a receive timeout
    if (!pair_of(sv)) { check(false, "socketpair"); return; }
    timeval to{};
    to.tv_usec = 100 * 1000;
    (void) setsockopt(sv[1], SOL_SOCKET, SO_RCVTIMEO, &to, sizeof to);
    err.clear();
    const bool ok3 = co::recv_parts(sv[1], sk2, 1, err);
    check(!ok3 && err == "remote stage: receive failed: timed out", "the receive timeout: \"" + err + "\"");
    ::close(sv[0]);
    ::close(sv[1]);
}

// ---- 4.
void test_ids_rows() {
    std::printf("4. ids + one slice of rows in one recvmsg, decoded after\n");
    const StageRows fmts[] = {StageRows::F32, StageRows::BF16, StageRows::FP8};
    const size_t width = 12804;
    int bad = 0, cases = 0;
    for (StageRows f : fmts)
        for (size_t rows : {(size_t) 1, (size_t) 3, (size_t) 6}) {
            std::vector<float> src = values(rows * width, (uint32_t) rows);
            std::vector<int32_t> slots(rows), toks(rows);
            std::vector<int64_t> pos(rows);
            for (size_t i = 0; i < rows; ++i) { slots[i] = (int32_t) i; toks[i] = (int32_t) (100 + i); pos[i] = (int64_t) i << 33; }
            StageHeader h;
            h.type = 15;
            std::vector<unsigned char> head(rows * 16);
            std::memcpy(head.data(), slots.data(), rows * 4);
            std::memcpy(head.data() + rows * 4, toks.data(), rows * 4);
            std::memcpy(head.data() + rows * 8, pos.data(), rows * 8);
            int sv[2];
            if (!pair_of(sv)) { check(false, "socketpair"); return; }
            no_sigpipe(sv[0]);
            std::vector<unsigned char> buf;
            double conv = 0;
            std::string err;
            std::thread wr([&] {
                (void) co::send_msg_rows(sv[0], kFlags, h, head.data(), head.size(), f, src.data(), rows, width, 256, buf,
                                         conv, err);
            });
            StageHeader gh;
            std::vector<int32_t> gs(rows), gt(rows);
            std::vector<int64_t> gp(rows);
            std::vector<float> dst(rows * width);
            std::vector<unsigned char> wire;
            std::string e2;
            bool ok = co::recv_parts(sv[1], std::vector<co::Sink>{{&gh, sizeof gh}}.data(), 1, e2);
            std::vector<co::Sink> sk = {{gs.data(), rows * 4}, {gt.data(), rows * 4}, {gp.data(), rows * 8}};
            if (f == StageRows::F32) {
                sk.push_back({dst.data(), rows * width * 4});
            } else {
                wire.resize(stage_rows_bytes(f, rows, width));
                sk.push_back({wire.data(), wire.size()});
            }
            ok = ok && co::recv_parts(sv[1], sk.data(), (int) sk.size(), e2);
            if (ok && f != StageRows::F32) stage_rows_decode(f, wire.data(), rows, width, dst.data());
            wr.join();
            ::close(sv[0]);
            ::close(sv[1]);
            // the rows as the old path decodes them: encode the whole, decode the whole
            std::vector<float> want(rows * width);
            if (f == StageRows::F32) {
                want = src;
            } else {
                std::vector<unsigned char> w(stage_rows_bytes(f, rows, width));
                stage_rows_encode(f, src.data(), rows, width, w.data());
                stage_rows_decode(f, w.data(), rows, width, want.data());
            }
            ++cases;
            const bool same = ok && gs == slots && gt == toks && gp == pos &&
                              std::memcmp(dst.data(), want.data(), dst.size() * 4) == 0;
            if (!same) {
                ++bad;
                std::printf("    %s rows %zu: %s\n", fmt_name(f), rows, ok ? "differs" : e2.c_str());
            }
        }
    check(bad == 0, "ids and rows arrive as sent in " + std::to_string(cases) + " cases (3 formats x 3 window sizes)");
}

// ---- 5. a worker and a client on loopback, the flag flipped at random
constexpr int64_t kW = 12804;      // a window row (as in production)
constexpr int64_t kMaxT = 6;
constexpr int64_t kD = 1024;       // a prompt row
constexpr int64_t kChunk = 600;    // 600 rows of 1024 values: three slices (256 rows each) in bf16 / fp8

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
    h.handoff_floats = kW;
    h.pack_hash = 0x5354524d;
    h.row_formats = kStageRowsAll;
    h.batch_slots = 4;
    h.batch_flags = kStageBatchKeepAll;
    return h;
}

struct Outputs {
    std::vector<std::vector<float>> runs, chunks, slot_runs;
    bool ok = true;
    std::string err;
};

Outputs drive(const std::string& where, StageRows f) {
    Outputs o;
    StageClient c;
    StageHello mine = hello_of();
    mine.row_formats = (1u << (uint32_t) StageRows::F32) | (1u << (uint32_t) f);
    StageHello peer;
    if (!c.connect(where, mine, peer, o.err)) { o.ok = false; return o; }
    std::vector<float> in = values((size_t) (kMaxT * kW), 99), out((size_t) (kMaxT * kW));
    std::vector<int32_t> tok = {1, 2, 3, 4, 5, 6};
    for (int rep = 0; rep < 8 && o.ok; ++rep)
        for (int T = 1; T <= (int) kMaxT && o.ok; ++T) {
            std::fill(out.begin(), out.end(), 0.0f);
            if (!c.run(T, tok.data(), rep * 10, in.data(), (size_t) (T * kW), out.data(), (size_t) (T * kW), o.err) ||
                !c.commit(T, o.err)) {
                o.ok = false;
                break;
            }
            o.runs.emplace_back(out.begin(), out.begin() + T * kW);
        }
    std::vector<float> pin = values((size_t) (kChunk * kD), 5), pout((size_t) (kChunk * kD));
    std::vector<int64_t> ptok(kChunk, 7);
    for (int rep = 0; rep < 3 && o.ok; ++rep) {
        std::fill(pout.begin(), pout.end(), 0.0f);
        if (!c.prefill_send(ptok.data(), kChunk, 0, 0, pin.data(), (size_t) kD, 0, o.err) ||
            !c.prefill_recv(pout.data(), (size_t) kD, kChunk, 0, o.err)) {
            o.ok = false;
            break;
        }
        o.chunks.push_back(pout);
    }
    std::vector<int32_t> sl = {0, 1, 2, 3}, st = {11, 12, 13, 14};
    std::vector<int64_t> sp = {5, 6, 7, 8};
    for (int rep = 0; rep < 8 && o.ok; ++rep)
        for (int S = 1; S <= 4 && o.ok; ++S) {
            std::fill(out.begin(), out.end(), 0.0f);
            if (!c.slot_run(S, sl.data(), st.data(), sp.data(), kStageSlotKeepAll, 1000 + rep * 10 + S, in.data(),
                            (size_t) (S * kW), out.data(), (size_t) (S * kW), o.err)) {
                o.ok = false;
                break;
            }
            o.slot_runs.emplace_back(out.begin(), out.begin() + S * kW);
        }
    c.close();
    return o;
}

bool same_bits(const std::vector<std::vector<float>>& a, const std::vector<std::vector<float>>& b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i)
        if (a[i].size() != b[i].size() || std::memcmp(a[i].data(), b[i].data(), a[i].size() * 4) != 0) return false;
    return true;
}

void test_link() {
    std::printf("5. a worker and a client on loopback, flag off / on / flipped at random\n");
    std::vector<float> run_in((size_t) (kMaxT * kW)), run_out(run_in.size());
    std::vector<float> pf_in0((size_t) (kChunk * kD)), pf_in1(pf_in0.size()), pf_out0(pf_in0.size()),
        pf_out1(pf_in0.size());
    StageBuffers buf;
    buf.run_in = run_in.data();
    buf.run_out = run_out.data();
    buf.run_floats = run_in.size();
    buf.pf_in[0] = pf_in0.data();
    buf.pf_in[1] = pf_in1.data();
    buf.pf_out[0] = pf_out0.data();
    buf.pf_out[1] = pf_out1.data();
    buf.pf_floats = pf_in0.size();
    buf.handoff_floats = kW;
    buf.pf_row_floats = kD;
    StageHandlers h;
    h.hello = [&](const StageHello&, StageHello& mine, std::string&) {
        mine = hello_of();
        return true;
    };
    h.run = [&](int T, const int32_t* tokens, int64_t, std::string&) {
        for (int64_t t = 0; t < T; ++t)
            for (int64_t j = 0; j < kW; ++j) run_out[t * kW + j] = run_in[t * kW + j] + (float) tokens[t];
        return true;
    };
    h.commit = [](int, std::string&) { return true; };
    h.prefill = [](const int64_t*, int64_t T, int64_t, int64_t, int64_t, const float* in, float* out, std::string&) {
        for (int64_t i = 0; i < T * kD; ++i) out[i] = in[i] * 2.0f;
        return true;
    };
    h.reset = [](const std::vector<int64_t>&, std::string&) { return true; };
    h.slot_run = [&](int S, const int32_t* slots, const int32_t* tokens, const int64_t* pos, int64_t, std::string&) {
        for (int t = 0; t < S; ++t)
            for (int64_t j = 0; j < kW; ++j)
                run_out[t * kW + j] = run_in[t * kW + j] + (float) slots[t] * 3.0f + (float) tokens[t] + (float) pos[t];
        return true;
    };
    h.slot_commit = [](const int32_t*, int, std::string&) { return true; };
    h.slot_save = [](int, int64_t, std::string&) { return true; };
    h.slot_load = [](int, int64_t, int64_t, std::string&) { return true; };
    std::atomic<bool> stop{false};
    std::atomic<int> served{-1};
    StageStats st;
    std::thread wt;
    std::string where;
    std::mt19937 prng(std::random_device{}());
    for (int attempt = 0; attempt < 8 && where.empty(); ++attempt) {
        const int port = 20000 + (int) (prng() % 40000);
        served = -1;
        wt = std::thread([&, port] { served = serve_stage("127.0.0.1", port, "", h, buf, st, &stop); });
        const std::string at = "127.0.0.1:" + std::to_string(port);
        for (int i = 0; i < 100 && served.load() < 0; ++i) {
            StageHello bad = hello_of();
            bad.protocol = 4;   // the worker's own refusal says it listens
            StageHello peer;
            std::string e;
            StageClient probe;
            if (probe.connect(at, bad, peer, e) || e.find("the worker refused:") != std::string::npos) { where = at; break; }
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
        if (where.empty()) {
            stop = true;
            wt.join();
            stop = false;
        }
    }
    if (where.empty()) {
        std::printf("  no loopback port to listen on\n");
        check(false, "a worker started");
        return;
    }
    for (StageRows f : {StageRows::F32, StageRows::BF16, StageRows::FP8}) {
        co::force_for_test(0);
        Outputs base = drive(where, f);
        co::force_for_test(1);
        Outputs on = drive(where, f);
        std::atomic<bool> flip{true};
        std::thread fl([&] {
            std::mt19937 r(1);
            while (flip.load()) {
                co::force_for_test((int) (r() & 1));
                std::this_thread::sleep_for(std::chrono::microseconds(r() % 200));
            }
        });
        Outputs mix = drive(where, f);
        flip = false;
        fl.join();
        co::force_for_test(-1);
        const std::string n = fmt_name(f);
        check(base.ok && !base.runs.empty() && !base.chunks.empty() && !base.slot_runs.empty(),
              n + ": the flag off: windows, chunks and batch windows all went through " + (base.ok ? "" : base.err));
        check(on.ok && same_bits(on.runs, base.runs) && same_bits(on.chunks, base.chunks) &&
                  same_bits(on.slot_runs, base.slot_runs),
              n + ": the flag on: the same rows as off " + (on.ok ? "" : on.err));
        check(mix.ok && same_bits(mix.runs, base.runs) && same_bits(mix.chunks, base.chunks) &&
                  same_bits(mix.slot_runs, base.slot_runs),
              n + ": flipped at random (old and new ends mixed): the same rows " + (mix.ok ? "" : mix.err));
    }
    stop = true;
    wt.join();
}

// ---- 6.
void test_on() {
    std::printf("6. on() and force_for_test\n");
    co::force_for_test(1);
    check(co::on(), "force 1: on");
    co::force_for_test(0);
    check(!co::on(), "force 0: off");
    co::force_for_test(-1);
    const char* e = std::getenv("STRATA_LINK_COALESCE");
    const bool want = e != nullptr && std::string(e).find_first_not_of(" ") != std::string::npos &&
                      std::string(e).substr(std::string(e).find_first_not_of(" "),
                                            std::string(e).find_last_not_of(" ") - std::string(e).find_first_not_of(" ") + 1) == "1";
    check(co::on() == want, std::string("the environment (") + (e ? e : "unset") + "): " + (want ? "on" : "off"));
}

}  // namespace

int main() {
    std::printf("stage_link_coalesce_test\n");
    test_on();
    test_bytes();
    test_short_writes();
    test_recv_parts();
    test_ids_rows();
    test_link();
    if (g_failed != 0) {
        std::printf("stage_link_coalesce_test: %d failed\n", g_failed);
        return 1;
    }
    std::printf("stage_link_coalesce_test: all passed\n");
    return 0;
}
#endif
