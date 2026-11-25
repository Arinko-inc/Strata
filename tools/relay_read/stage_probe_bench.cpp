// src/net/stage_probe.cpp (SYCL build, not a ctest: it needs a running worker) - a stand-in main process for one
// stage worker, to check a worker's image positions (--vision, StageMsg::Positions) without a main process.
//
//   STRATA_STAGE_TOKEN=... stage_probe HOST:PORT [--vision] [--batch] [--n N] [--decode D] [--seed S]
//   STRATA_STAGE_TOKEN=... stage_probe HOST:PORT --expect-refusal TEXT
//   STRATA_STAGE_TOKEN=... stage_probe HOST:PORT --slots [--expect-differ] [--windows K]
//
// It learns the worker's geometry from the refusal of an empty hello, then connects as a main process would and sends
// synthetic hand-off rows (normal noise, a fixed seed): a prompt of N tokens in 2048-token chunks, the same prompt to
// N - 4 followed by a 4-row verify window and D decode windows, and (--batch) a batch window on slot 0 over the same
// four rows.  Each part's output rows are hashed (FNV-1a over the fp32 bytes), so two runs, two executables or two
// position tables compare bitwise.  With --vision it runs these scenarios on one connection:
//   none      no Positions message at all
//   null      Positions(identity)                         == none, bitwise
//   image     two pictures in mtmd's layout                (and image2: the same again, bitwise - determinism)
//   offP      identity below cell 2048, +1000 from it on   the prompt's first chunk == null, its second differs
//   offW      identity below N - 4, +1000 from it on       the prompt to N - 4 == null; the window, the decode and the
//                                                          batch window differ (each path reads the table)
//   oversize  a table longer than the worker's context     refused at the next Reset; Positions(identity) recovers
// and then, on a new connection after an image table, a text run that must equal null (the table went back to the
// identity when its main process went away).  Without --vision it runs `none` only (for the hashes) and checks that
// a Positions message is refused.  --expect-refusal: a --vision hello must be refused with TEXT in the message.
//
// --slots (remote-batch-vision, strata-mesh IMPL-PLAN-remote-batch-vision.md 7): two batch slots with different
// position tables in one window.  Slot 0 holds a prompt of L0 cells, slot 1 one of L1 (L0 + K <= L1), and K windows
// run rows {0, 1} at cells {L0 + i, L1 + i}.  U is one table that is right for both (identity below L0, +1000 on
// [L0, L1), +3000 from L1).  `ref`: everything under U (main and slots: a one-table worker computes the right answer
// too).  `test`: each prompt read under U, then slot 0 saved under A (+1000 from L0: right for slot 0, wrong at slot
// 1's cells), slot 1 under B (identity below L1, +3000 from it: right for slot 1, wrong at slot 0's cells), and the
// windows run with the main table C (+5000 from L0: wrong for both).  With a table per slot test == ref bitwise, row
// by row; with one table (an old worker, STRATA_SLOT_POSITIONS=0) both rows differ (--expect-differ).  The hashes
// of `ref` compare across executables.
#include "strata/net/stage_link.hpp"

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using namespace strata::net;
using Clock = std::chrono::steady_clock;

namespace {

constexpr int64_t kImagePad = 248056;   // <|image_pad|> (the worker does not read it; the main process sends it)
int64_t kChunk = 2048;   // --chunk: the prompt chunk (the worker's --prefill must be at least this)
constexpr int kW = 4;                   // the verify window's rows

uint64_t fnv(const void* p, size_t n, uint64_t h = 1469598103934665603ull) {
    const auto* b = (const unsigned char*) p;
    for (size_t i = 0; i < n; ++i) h = (h ^ b[i]) * 1099511628211ull;
    return h;
}
double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

int g_bad = 0;
void verdict(bool ok, const std::string& what) {
    std::printf("  %s  %s\n", ok ? "PASS" : "FAIL", what.c_str());
    std::fflush(stdout);
    if (!ok) ++g_bad;
}

struct Geometry {
    StageHello peer;   // the worker's hello (HelloOk)
    int64_t D = 0, HB = 0;
};

// the worker's geometry from its refusal of an empty hello ("the two sides differ (protocol 8/8, n_embd 0/X, ...")
bool discover(const std::string& addr, const std::string& token, StageHello& h, std::string& err) {
    StageHello mine;
    std::snprintf(mine.token, sizeof mine.token, "%s", token.c_str());
    StageHello peer;
    StageClient c;
    if (c.connect(addr, mine, peer, err, 30)) {
        err = "the worker took an empty hello";
        return false;
    }
    const size_t at = err.find("the two sides differ (");
    if (at == std::string::npos) return false;
    unsigned pa, pb;
    unsigned long long ha, hb;
    long long fa, fb;
    int a[6];
    if (std::sscanf(err.c_str() + at,
                    "the two sides differ (protocol %u/%u, n_embd %d/%d, hc %d/%d, layers %d/%d, experts %d/%d, first "
                    "remote layer %d/%d, hand-off floats %lld/%lld, K/V type %d/%d, model pack %llx/%llx",
                    &pa, &pb, &a[0], &h.n_embd, &a[1], &h.hc, &a[2], &h.n_layers, &a[3], &h.n_expert, &a[4],
                    &h.layer_begin, &fa, &fb, &a[5], &h.kv_type, &ha, &hb) != 18) {
        err = "cannot read the worker's geometry from: " + err;
        return false;
    }
    h.handoff_floats = fb;
    h.pack_hash = hb;
    return true;
}

struct Inputs {
    int64_t N = 0;
    std::vector<int64_t> tok;          // [N]
    std::vector<float> rows;           // [N][D]
    std::vector<float> dec_rows;       // [decode][kW][HB] (the block layout: R rows, then bo, then inject)
    std::vector<int32_t> dec_tok;      // [decode][kW]
    int64_t img1 = 0, img1_nx = 0, img1_ny = 0, img2 = 0, img2_nx = 0, img2_ny = 0;
};

// mtmd's positions for `cells` cells: t = p, h = p + y, w = p + x on a picture, p + max(nx, ny) after it
std::vector<int32_t> image_table(const Inputs& in, int64_t cells) {
    std::vector<int32_t> t((size_t) cells * 3);
    int64_t p = 0;
    for (int64_t c = 0; c < cells;) {
        auto pic = [&](int64_t at, int64_t nx, int64_t ny) {
            if (c != at) return false;
            for (int64_t j = 0; j < nx * ny; ++j, ++c) {
                t[(size_t) c * 3] = (int32_t) p;
                t[(size_t) c * 3 + 1] = (int32_t) (p + j / nx);
                t[(size_t) c * 3 + 2] = (int32_t) (p + j % nx);
            }
            p += std::max(nx, ny);
            return true;
        };
        if (pic(in.img1, in.img1_nx, in.img1_ny) || pic(in.img2, in.img2_nx, in.img2_ny)) continue;
        t[(size_t) c * 3] = t[(size_t) c * 3 + 1] = t[(size_t) c * 3 + 2] = (int32_t) p;
        ++p;
        ++c;
    }
    return t;
}
// the identity below `from`, `shift` added from it on (text positions throughout)
std::vector<int32_t> offset_table(int64_t cells, int64_t from, int32_t shift) {
    std::vector<int32_t> t((size_t) cells * 3);
    for (int64_t c = 0; c < cells; ++c)
        t[(size_t) c * 3] = t[(size_t) c * 3 + 1] = t[(size_t) c * 3 + 2] = (int32_t) (c + (c >= from ? shift : 0));
    return t;
}

struct Result {
    bool ok = false;
    std::string err;
    std::vector<uint64_t> full;     // the whole prompt: one hash per chunk
    std::vector<uint64_t> part;     // the prompt to N - kW: one hash per chunk
    uint64_t window = 0, decode = 0, slot = 0;
    std::vector<float> full_tail;   // the whole prompt's rows [N - kW, N)
    std::vector<float> win_r;       // the window's R rows
    std::vector<float> slot_r;      // the batch window's R rows
    double chunk_ms = 0, win_ms = 0, win_worker_ms = 0;   // per full chunk (wall), per decode window (wall, worker)
};

struct Probe {
    std::string addr, token;
    Geometry g;
    bool vision = false, batch = false;
    int batch_slots = 1;
    int decode = 32;
    int64_t max_context = 8192;
    StageClient link;

    bool connect(std::string& err) {
        StageHello mine = g.peer;   // geometry, pack and K/V as the worker's
        mine.protocol = kStageProtocol;
        mine.max_t = kW;
        mine.max_context = max_context;
        mine.chunk = kChunk;
        mine.tail_skip = 0;
        mine.vision = vision ? 1 : 0;
        mine.ckpt_max = 0;
        mine.row_formats = 1u << (uint32_t) StageRows::F32;
        mine.batch_slots = batch ? batch_slots : 0;
        mine.batch_flags = batch ? kStageBatchKeepAll : 0;
        std::memset(mine.build, 0, sizeof mine.build);
        std::snprintf(mine.build, sizeof mine.build, "stage_probe");
        std::snprintf(mine.token, sizeof mine.token, "%s", token.c_str());
        link.close();
        StageHello peer;
        if (!link.connect(addr, mine, peer, err, 60)) return false;
        g.peer.max_context = peer.max_context;
        g.peer.vision = peer.vision;
        g.peer.batch_slots = peer.batch_slots;
        g.peer.batch_flags = peer.batch_flags;
        return true;
    }

    // the prompt [0, n) in chunks; `out` gets every row (N x D), hashes one per chunk
    bool prompt(const Inputs& in, int64_t n, std::vector<float>& out, std::vector<uint64_t>& hashes, double* chunk_ms,
                std::string& err) {
        out.assign((size_t) (n * g.D), 0.0f);
        hashes.clear();
        double full_ms = 0;
        int full = 0;
        for (int64_t c0 = 0; c0 < n; c0 += kChunk) {
            const int64_t T = std::min(kChunk, n - c0);
            const auto t0 = Clock::now();
            if (!link.prefill_send(in.tok.data() + c0, T, c0, n <= kChunk ? 1 : 0, in.rows.data() + c0 * g.D,
                                   (size_t) g.D, 0, err) ||
                !link.prefill_recv(out.data() + c0 * g.D, (size_t) g.D, T, 0, err))
                return false;
            if (T == kChunk) {
                full_ms += ms_since(t0);
                ++full;
            }
            hashes.push_back(fnv(out.data() + c0 * g.D, (size_t) (T * g.D) * 4));
        }
        if (chunk_ms) *chunk_ms = full > 0 ? full_ms / full : 0;
        return true;
    }

    // a window's hand-in rows in the verifier's block layout: T R rows (from `r`, D floats each), bo and inject zero
    std::vector<float> window_rows(const float* r, int T) const {
        std::vector<float> w((size_t) (T * g.HB), 0.0f);
        std::memcpy(w.data(), r, (size_t) (T * g.D) * 4);
        return w;
    }

    Result scenario(const Inputs& in, const char* name, int mode, const std::vector<int32_t>* table) {
        // mode: 0 no Positions, 1 Positions(identity), 2 Positions(table)
        Result res;
        std::string& err = res.err;
        const int64_t N = in.N;
        std::vector<float> out;
        if (mode == 1 && !link.positions(nullptr, 0, err)) return res;
        if (mode == 2 && !link.positions(table->data(), (int64_t) table->size() / 3, err)) return res;
        if (!link.reset({}, err) || !prompt(in, N, out, res.full, &res.chunk_ms, err)) return res;
        res.full_tail.assign(out.begin() + (N - kW) * g.D, out.begin() + N * g.D);
        if (!link.reset({}, err) || !prompt(in, N - kW, out, res.part, nullptr, err)) return res;
        if (batch && !link.slot_save(0, N - kW, err)) return res;
        const std::vector<float> win_in = window_rows(in.rows.data() + (N - kW) * g.D, kW);
        std::vector<float> win_out((size_t) (kW * g.HB));
        std::vector<int32_t> wtok(kW);
        for (int t = 0; t < kW; ++t) wtok[t] = (int32_t) in.tok[(size_t) (N - kW + t)];
        if (!link.run(kW, wtok.data(), N - kW, win_in.data(), win_in.size(), win_out.data(), win_out.size(), err))
            return res;
        res.window = fnv(win_out.data(), win_out.size() * 4);
        res.win_r.assign(win_out.begin(), win_out.begin() + kW * g.D);
        if (!link.commit(kW, err)) return res;
        // decode windows, one row kept each (the session grows by one per window)
        uint64_t h = 1469598103934665603ull;
        const double w0 = link.stats().ms_run_worker;
        double wall = 0;
        for (int i = 0; i < decode; ++i) {
            const float* r = in.dec_rows.data() + (size_t) i * kW * g.HB;
            const auto t0 = Clock::now();
            if (!link.run(kW, in.dec_tok.data() + (size_t) i * kW, N + i, r, (size_t) (kW * g.HB), win_out.data(),
                          win_out.size(), err) ||
                !link.commit(1, err))
                return res;
            wall += ms_since(t0);
            h = fnv(win_out.data(), win_out.size() * 4, h);
        }
        res.decode = h;
        res.win_ms = decode > 0 ? wall / decode : 0;
        res.win_worker_ms = decode > 0 ? (link.stats().ms_run_worker - w0) / decode : 0;
        if (batch) {
            const int32_t slots[kW] = {0, 0, 0, 0};
            int64_t pos[kW];
            for (int t = 0; t < kW; ++t) pos[t] = N - kW + t;
            if (!link.slot_run(kW, slots, wtok.data(), pos, kStageSlotKeepAll, 1, win_in.data(), win_in.size(),
                               win_out.data(), win_out.size(), err))
                return res;
            res.slot = fnv(win_out.data(), win_out.size() * 4);
            res.slot_r.assign(win_out.begin(), win_out.begin() + kW * g.D);
        }
        res.ok = true;
        std::printf("  %-9s chunks", name);
        for (uint64_t x : res.full) std::printf(" %016" PRIx64, x);
        std::printf(" | to N-%d", kW);
        for (uint64_t x : res.part) std::printf(" %016" PRIx64, x);
        std::printf(" | window %016" PRIx64 " decode %016" PRIx64, res.window, res.decode);
        if (batch) std::printf(" slot %016" PRIx64, res.slot);
        std::printf("\n            %.1f ms a 2048-token chunk (wall), %.2f ms a decode window (wall; the worker's own "
                    "%.2f)\n", res.chunk_ms, res.win_ms, res.win_worker_ms);
        std::fflush(stdout);
        return res;
    }

    // --slots: see the head of this file.  Hashes per row over the K windows' output rows (and the whole windows).
    struct SlotsResult {
        bool ok = false;
        std::string err;
        uint64_t row[2] = {1469598103934665603ull, 1469598103934665603ull};
        uint64_t all = 1469598103934665603ull;
        std::vector<float> r0, r1;   // the last window's rows
    };
    SlotsResult slots_run(const Inputs& in, int64_t L0, int64_t L1, int K, const std::vector<int32_t>& U,
                          const std::vector<int32_t>* A, const std::vector<int32_t>* B, const std::vector<int32_t>& main,
                          const std::vector<float>& win_rows, const std::vector<int32_t>& win_tok) {
        SlotsResult r;
        std::string& err = r.err;
        std::vector<float> out;
        std::vector<uint64_t> hs;
        auto pos_of = [&](const std::vector<int32_t>& t) { return link.positions(t.data(), (int64_t) t.size() / 3, err); };
        // slot 0: the prompt to L0 under U, saved under A (ref: U)
        if (!pos_of(U) || !link.reset({}, err) || !prompt(in, L0, out, hs, nullptr, err)) return r;
        if (A != nullptr && !pos_of(*A)) return r;
        if (!link.slot_save(0, L0, err)) return r;
        // slot 1: the prompt to L1 under U, saved under B (ref: U)
        if (!pos_of(U) || !link.reset({}, err) || !prompt(in, L1, out, hs, nullptr, err)) return r;
        if (B != nullptr && !pos_of(*B)) return r;
        if (!link.slot_save(1, L1, err)) return r;
        // the windows with the main table `main`
        if (!pos_of(main) || !link.reset({}, err)) return r;
        const int32_t slots[2] = {0, 1};
        std::vector<float> wo((size_t) (2 * g.HB));
        for (int i = 0; i < K; ++i) {
            const int64_t pos[2] = {L0 + i, L1 + i};
            const float* wi = win_rows.data() + (size_t) i * 2 * g.HB;
            if (!link.slot_run(2, slots, win_tok.data() + (size_t) i * 2, pos, kStageSlotKeepAll, 1 + i, wi,
                               (size_t) (2 * g.HB), wo.data(), wo.size(), err))
                return r;
            r.row[0] = fnv(wo.data(), (size_t) g.D * 4, r.row[0]);
            r.row[1] = fnv(wo.data() + g.D, (size_t) g.D * 4, r.row[1]);
            r.all = fnv(wo.data(), wo.size() * 4, r.all);
            if (i == K - 1) {
                r.r0.assign(wo.begin(), wo.begin() + g.D);
                r.r1.assign(wo.begin() + g.D, wo.begin() + 2 * g.D);
            }
        }
        r.ok = true;
        return r;
    }
};

std::string hx(uint64_t v) {
    char b[24];
    std::snprintf(b, sizeof b, "%016" PRIx64, v);
    return b;
}
// rms and max of a - b over R rows
std::string diff(const std::vector<float>& a, const std::vector<float>& b) {
    double ss = 0, mx = 0, ref = 0;
    const size_t n = std::min(a.size(), b.size());
    for (size_t i = 0; i < n; ++i) {
        const double d = (double) a[i] - b[i];
        ss += d * d;
        ref += (double) a[i] * a[i];
        mx = std::max(mx, std::fabs(d));
    }
    char buf[96];
    std::snprintf(buf, sizeof buf, "rms %.3e (of rms %.3e), max %.3e", n ? std::sqrt(ss / n) : 0.0,
                  n ? std::sqrt(ref / n) : 0.0, mx);
    return buf;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: stage_probe HOST:PORT [--vision] [--batch] [--n N] [--decode D] [--seed S] "
                             "[--expect-refusal TEXT]   (the token: STRATA_STAGE_TOKEN)\n");
        return 2;
    }
    Probe pr;
    pr.addr = argv[1];
    const char* tok = std::getenv("STRATA_STAGE_TOKEN");
    pr.token = tok ? tok : "";
    int64_t N = 2600;
    unsigned seed = 7;
    std::string expect_refusal;
    bool slots = false, expect_differ = false;
    int windows = 32;
    int64_t bench = 0;   // --bench N: a prompt of N tokens (noise rows), each chunk's time and hash; --reps R times
    int reps = 1;
    int skew = 0;            // --skew C: the bench rows are C cluster centres (Zipf weights) + half-size noise, not plain noise
    std::string dump;        // --dump FILE: the bench's first chunk's output rows (rep 0), raw fp32
    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> const char* {
            if (i + 1 >= argc) { std::fprintf(stderr, "%s needs a value\n", a.c_str()); std::exit(2); }
            return argv[++i];
        };
        if (a == "--vision") pr.vision = true;
        else if (a == "--batch") pr.batch = true;
        else if (a == "--n") N = std::atoll(next());
        else if (a == "--decode") pr.decode = std::atoi(next());
        else if (a == "--seed") seed = (unsigned) std::atoi(next());
        else if (a == "--expect-refusal") expect_refusal = next();
        else if (a == "--slots") slots = true;
        else if (a == "--expect-differ") expect_differ = true;
        else if (a == "--windows") windows = std::atoi(next());
        else if (a == "--bench") bench = std::atoll(next());
        else if (a == "--chunk") kChunk = std::atoll(next());
        else if (a == "--reps") reps = std::atoi(next());
        else if (a == "--skew") skew = std::atoi(next());
        else if (a == "--dump") dump = next();
        else { std::fprintf(stderr, "unknown option %s\n", a.c_str()); return 2; }
    }
    if (bench > 0) pr.max_context = (bench + 8191) / 8192 * 8192;
    if (bench > 0 && N < kChunk + 64) N = kChunk + 64;   // (--bench does not use N)
    if (N < kChunk + 64 || N > 8000) { std::fprintf(stderr, "--n: 2112..8000 (two prompt chunks)\n"); return 2; }
    std::string err;
    if (!discover(pr.addr, pr.token, pr.g.peer, err)) {
        std::fprintf(stderr, "stage_probe: %s\n", err.c_str());
        return 1;
    }
    pr.g.D = (int64_t) pr.g.peer.hc * pr.g.peer.n_embd;
    pr.g.HB = pr.g.peer.handoff_floats;
    std::printf("stage_probe: %s: n_embd %d, hc %d, layers %d-%d, hand-off %lld floats, K/V type %d, pack %016llx\n",
                pr.addr.c_str(), pr.g.peer.n_embd, pr.g.peer.hc, pr.g.peer.layer_begin, pr.g.peer.n_layers - 1,
                (long long) pr.g.HB, pr.g.peer.kv_type, (unsigned long long) pr.g.peer.pack_hash);
    if (!expect_refusal.empty()) {
        pr.vision = true;
        const bool took = pr.connect(err);
        verdict(!took && err.find(expect_refusal) != std::string::npos,
                "a --vision main process is refused: " + (took ? std::string("(it was taken)") : err));
        return g_bad == 0 ? 0 : 1;
    }
    if (slots) {
        pr.vision = pr.batch = true;
        pr.batch_slots = 2;
    }
    if (!pr.connect(err)) {
        std::fprintf(stderr, "stage_probe: %s\n", err.c_str());
        return 1;
    }
    std::printf("stage_probe: connected (vision %d, batch slots %d; the worker's vision %d, context %lld, batch slots "
                "%d, batch flags 0x%x: a position table per slot %s)\n",
                pr.vision ? 1 : 0, pr.batch ? pr.batch_slots : 0, pr.g.peer.vision,
                (long long) pr.g.peer.max_context, (int) pr.g.peer.batch_slots, (unsigned) pr.g.peer.batch_flags,
                (pr.g.peer.batch_flags & kStageBatchSlotPositions) ? "yes" : "no");
    if (bench > 0) {
        // four chunks of noise rows, taken in turn; each chunk's wall time (send, the worker's compute, the reply) and the
        // hash of its output rows.  The worker's own time per chunk is in its log (STRATA_REMOTE_TIMING=1,
        // STRATA_PREFILL_TIMING=1).
        const int kVar = 4;
        std::mt19937 rng(seed);
        std::normal_distribution<float> nd(0.0f, 1.0f);
        std::uniform_int_distribution<int> td(1000, 100000);
        std::vector<float> rows((size_t) (kVar * kChunk * pr.g.D));
        for (float& v : rows) v = nd(rng);
        if (skew > 0) {
            // routing with a skew, as text has: each row is one of `skew` centres (centre c with weight 1 / (c + 1))
            // plus the noise at half size, so the rows of a centre route alike and some experts take many rows
            std::vector<float> cen((size_t) skew * (size_t) pr.g.D);
            for (float& v : cen) v = nd(rng);
            std::vector<double> w((size_t) skew);
            for (int c = 0; c < skew; ++c) w[(size_t) c] = 1.0 / (c + 1);
            std::discrete_distribution<int> pick(w.begin(), w.end());
            for (int64_t r = 0; r < kVar * kChunk; ++r) {
                const float* c = cen.data() + (size_t) pick(rng) * (size_t) pr.g.D;
                float* x = rows.data() + (size_t) r * (size_t) pr.g.D;
                for (int64_t d = 0; d < pr.g.D; ++d) x[d] = c[d] + 0.5f * x[d];
            }
        }
        std::vector<int64_t> tok((size_t) (kVar * kChunk));
        for (auto& t : tok) t = td(rng);
        std::vector<float> out((size_t) (kChunk * pr.g.D));
        for (int rep = 0; rep < reps; ++rep) {
            if (!pr.link.reset({}, err)) { std::fprintf(stderr, "stage_probe: %s\n", err.c_str()); return 1; }
            uint64_t all = 1469598103934665603ull;
            const auto t_all = Clock::now();
            int i = 0;
            for (int64_t c0 = 0; c0 < bench; c0 += kChunk, ++i) {
                const int64_t T = std::min(kChunk, bench - c0);
                const int v = i % kVar;
                const auto t0 = Clock::now();
                if (!pr.link.prefill_send(tok.data() + v * kChunk, T, c0, bench <= kChunk ? 1 : 0,
                                          rows.data() + v * kChunk * pr.g.D, (size_t) pr.g.D, 0, err) ||
                    !pr.link.prefill_recv(out.data(), (size_t) pr.g.D, T, 0, err)) {
                    std::fprintf(stderr, "stage_probe: %s\n", err.c_str());
                    return 1;
                }
                const double ms = ms_since(t0);
                if (!dump.empty() && rep == 0 && c0 == 0) {
                    if (FILE* f = std::fopen(dump.c_str(), "wb")) {
                        std::fwrite(out.data(), 4, (size_t) (T * pr.g.D), f);
                        std::fclose(f);
                    }
                }
                const uint64_t hsh = fnv(out.data(), (size_t) (T * pr.g.D) * 4);
                all = fnv(&hsh, sizeof hsh, all);
                std::printf("  bench rep %d chunk T=%lld at %lld: %.1f ms %016" PRIx64 "\n", rep, (long long) T, (long long) c0, ms, hsh);
            }
            std::printf("  bench rep %d: %lld tokens in %.0f ms, all %016" PRIx64 "\n", rep, (long long) bench, ms_since(t_all), all);
            std::fflush(stdout);
        }
        return 0;
    }
    if (slots) {
        const int64_t L0 = 1200, L1 = 1600;   // multiples of the indexer block: each window's blocks start at L + 4j
        const int K = windows;
        if (K < 8 || L0 + K + 4 > L1) { std::fprintf(stderr, "--windows: 8..%lld\n", (long long) (L1 - L0 - 4)); return 2; }
        Inputs in;
        in.N = L1;
        std::mt19937 rng(seed);
        std::normal_distribution<float> nd(0.0f, 1.0f);
        std::uniform_int_distribution<int> td(1000, 100000);
        in.tok.resize((size_t) L1);
        for (auto& t : in.tok) t = td(rng);
        in.rows.resize((size_t) (L1 * pr.g.D));
        for (float& v : in.rows) v = nd(rng);
        std::vector<float> win_rows((size_t) K * 2 * pr.g.HB, 0.0f);   // per window: 2 R rows, then bo and inject 0
        std::vector<int32_t> win_tok((size_t) K * 2);
        for (int i = 0; i < K; ++i) {
            float* w = win_rows.data() + (size_t) i * 2 * pr.g.HB;
            for (int64_t j = 0; j < 2 * pr.g.D; ++j) w[j] = nd(rng);
            win_tok[(size_t) i * 2] = td(rng);
            win_tok[(size_t) i * 2 + 1] = td(rng);
        }
        const int64_t cells = L1 + K + 64;
        std::vector<int32_t> U = offset_table(cells, L0, 1000);
        for (int64_t c = L1; c < cells; ++c) U[(size_t) c * 3] = U[(size_t) c * 3 + 1] = U[(size_t) c * 3 + 2] = (int32_t) (c + 3000);
        const std::vector<int32_t> A = offset_table(cells, L0, 1000), B = offset_table(cells, L1, 3000),
                                   C = offset_table(cells, L0, 5000);
        const auto t0 = Clock::now();
        const Probe::SlotsResult ref = pr.slots_run(in, L0, L1, K, U, nullptr, nullptr, U, win_rows, win_tok);
        verdict(ref.ok, "ref (one table U for everything): " + std::to_string(K) + " windows of slots {0, 1}" +
                            (ref.ok ? "" : ": " + ref.err));
        const Probe::SlotsResult test = pr.slots_run(in, L0, L1, K, U, &A, &B, C, win_rows, win_tok);
        verdict(test.ok, "test (slot 0 saved under A, slot 1 under B, the windows under C)" +
                             (test.ok ? std::string() : ": " + test.err));
        const Probe::SlotsResult ref2 = pr.slots_run(in, L0, L1, K, U, nullptr, nullptr, U, win_rows, win_tok);
        if (!ref.ok || !test.ok || !ref2.ok) return 1;
        std::printf("  ref   row0 %s row1 %s all %s\n", hx(ref.row[0]).c_str(), hx(ref.row[1]).c_str(), hx(ref.all).c_str());
        std::printf("  test  row0 %s row1 %s all %s\n", hx(test.row[0]).c_str(), hx(test.row[1]).c_str(), hx(test.all).c_str());
        std::printf("  ref2  row0 %s row1 %s all %s   (%.1f s)\n", hx(ref2.row[0]).c_str(), hx(ref2.row[1]).c_str(),
                    hx(ref2.all).c_str(), ms_since(t0) / 1000.0);
        std::printf("  the last window, test against ref: row 0 %s; row 1 %s\n", diff(test.r0, ref.r0).c_str(),
                    diff(test.r1, ref.r1).c_str());
        verdict(ref.all == ref2.all, "ref twice: bitwise the same (determinism)");
        if (expect_differ)
            verdict(test.row[0] != ref.row[0] && test.row[1] != ref.row[1],
                    "one table: BOTH slots' rows differ from ref (each rotated by the main table C)");
        else
            verdict(test.row[0] == ref.row[0] && test.row[1] == ref.row[1],
                    "a table per slot: each slot's rows == ref, bitwise (each rotated by its own table)");
        std::printf("stage_probe: %s (%d failed)\n", g_bad == 0 ? "PASS" : "FAIL", g_bad);
        return g_bad == 0 ? 0 : 1;
    }

    // the inputs: a picture of 32 x 24 at cell 1000 and one of 6 x 3 that ends two cells before N, so the verify
    // window [N - 4, N) holds two picture cells and two text cells
    Inputs in;
    in.N = N;
    in.img1 = 1000; in.img1_nx = 32; in.img1_ny = 24;
    in.img2_nx = 6; in.img2_ny = 3; in.img2 = N - 2 - in.img2_nx * in.img2_ny;
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::uniform_int_distribution<int> td(1000, 100000);
    in.tok.resize((size_t) N);
    for (int64_t c = 0; c < N; ++c) {
        const bool pic = (c >= in.img1 && c < in.img1 + in.img1_nx * in.img1_ny) ||
                         (c >= in.img2 && c < in.img2 + in.img2_nx * in.img2_ny);
        in.tok[(size_t) c] = pic ? kImagePad : td(rng);
    }
    in.rows.resize((size_t) (N * pr.g.D));
    for (float& v : in.rows) v = nd(rng);
    in.dec_rows.assign((size_t) pr.decode * kW * pr.g.HB, 0.0f);
    in.dec_tok.resize((size_t) pr.decode * kW);
    for (int i = 0; i < pr.decode; ++i) {
        float* r = in.dec_rows.data() + (size_t) i * kW * pr.g.HB;
        for (int64_t j = 0; j < kW * pr.g.D; ++j) r[j] = nd(rng);
        for (int t = 0; t < kW; ++t) in.dec_tok[(size_t) i * kW + t] = td(rng);
    }
    const int64_t cells = N + pr.decode + 64;

    const Result none = pr.scenario(in, "none", 0, nullptr);
    verdict(none.ok, "none: the prompt, the window, " + std::to_string(pr.decode) + " decode windows" +
                         (pr.batch ? ", the batch window" : "") + (none.ok ? "" : ": " + none.err));
    if (!none.ok) return 1;
    std::printf("  the prompt's rows [N-4, N) against the window's: %s\n", diff(none.full_tail, none.win_r).c_str());
    if (pr.batch) std::printf("  the window's rows against the batch window's: %s\n", diff(none.win_r, none.slot_r).c_str());
    if (!pr.vision && pr.g.peer.vision != 0) {
        std::printf("stage_probe: the worker has --vision (a main process without it sends no Positions)\n");
        std::printf("stage_probe: %s (%d failed)\n", g_bad == 0 ? "PASS" : "FAIL", g_bad);
        return g_bad == 0 ? 0 : 1;
    }
    if (!pr.vision) {
        // a worker without --vision takes no Positions: the next reply says so
        std::string e;
        const bool sent = pr.link.positions(nullptr, 0, e);
        const bool reset = sent && pr.link.reset({}, e);
        verdict(sent && !reset && e.find("start it with --vision") != std::string::npos,
                "a worker without --vision refuses Positions: " + (reset ? std::string("(taken)") : e));
        std::printf("stage_probe: %s (%d failed)\n", g_bad == 0 ? "PASS" : "FAIL", g_bad);
        return g_bad == 0 ? 0 : 1;
    }

    const Result null = pr.scenario(in, "null", 1, nullptr);
    auto same_all = [&](const Result& a, const Result& b) {
        return a.ok && b.ok && a.full == b.full && a.part == b.part && a.window == b.window && a.decode == b.decode &&
               a.slot == b.slot;
    };
    verdict(same_all(none, null), "Positions(identity) == no Positions, bitwise (prompt, window, decode" +
                                      std::string(pr.batch ? ", batch window)" : ")") + (null.ok ? "" : ": " + null.err));

    const std::vector<int32_t> img = image_table(in, cells);
    const Result image = pr.scenario(in, "image", 2, &img);
    const Result image2 = pr.scenario(in, "image2", 2, &img);
    verdict(image.ok && same_all(image, image2), "the image table twice: bitwise the same (determinism)" +
                                                    (image.ok ? std::string() : ": " + image.err));
    verdict(image.ok && image.full[0] != null.full[0] && image.window != null.window && image.decode != null.decode,
            "the image table moves the prompt, the window and the decode");
    if (image.ok) {
        std::printf("  image: the prompt's rows [N-4, N) against the window's: %s\n",
                    diff(image.full_tail, image.win_r).c_str());
        std::printf("  image against null, the window's rows: %s\n", diff(image.win_r, null.win_r).c_str());
        if (pr.batch)
            std::printf("  image: the window's rows against the batch window's: %s\n",
                        diff(image.win_r, image.slot_r).c_str());
    }

    // each path reads the table: an offset from cell L on leaves everything before L as the identity's
    const std::vector<int32_t> offp = offset_table(cells, kChunk, 1000);
    const Result op = pr.scenario(in, "offP", 2, &offp);
    verdict(op.ok && op.full.size() >= 2 && op.full[0] == null.full[0] && op.full[1] != null.full[1] &&
                op.part[0] == null.part[0] && op.part[1] != null.part[1],
            "offset from cell 2048: the prompt's first chunk == null, its second chunk differs (the prompt path)" +
                (op.ok ? std::string() : ": " + op.err));
    const std::vector<int32_t> offw = offset_table(cells, N - kW, 1000);
    const Result ow = pr.scenario(in, "offW", 2, &offw);
    verdict(ow.ok && ow.part == null.part && ow.full[0] == null.full[0] && ow.window != null.window &&
                ow.decode != null.decode && (!pr.batch || ow.slot != null.slot),
            "offset from cell N-4: the prompt to N-4 == null; the window, the decode" +
                std::string(pr.batch ? " and the batch window" : "") + " differ (each reads the table)" +
                (ow.ok ? std::string() : ": " + ow.err));
    if (ow.ok) std::printf("  offW against null, the window's rows: %s\n", diff(ow.win_r, null.win_r).c_str());

    // a table longer than the worker's context: refused at the next Reset, then the identity recovers
    {
        const int64_t over = pr.g.peer.max_context + 64 + 1;
        std::vector<int32_t> big = offset_table(over, over, 0);
        std::string e;
        const bool sent = pr.link.positions(big.data(), over, e);
        const bool reset = sent && pr.link.reset({}, e);
        verdict(sent && !reset && e.find("longer than this worker's context") != std::string::npos,
                "a table of " + std::to_string(over) + " cells: " + (reset ? std::string("(taken)") : e));
        if (!pr.link.connected() && !pr.connect(e)) {
            verdict(false, "connect again: " + e);
            return 1;
        }
        const Result back = pr.scenario(in, "recover", 1, nullptr);
        verdict(same_all(back, null), "Positions(identity) after the refusal == null, bitwise" +
                                          (back.ok ? std::string() : ": " + back.err));
    }
    // a picture's table does not outlive its main process: a new connection's text is the identity's
    {
        std::string e;
        const bool sent = pr.link.positions(img.data(), (int64_t) img.size() / 3, e) && pr.link.reset({}, e);
        pr.link.close();
        const bool again = sent && pr.connect(e);
        const Result after = again ? pr.scenario(in, "reconnect", 0, nullptr) : Result{};
        verdict(again && same_all(after, null), "after an image table and a disconnect, a new main process's text == "
                                                "null, bitwise" + (after.ok ? std::string() : ": " + e + after.err));
    }
    std::printf("stage_probe: %s (%d failed)\n", g_bad == 0 ? "PASS" : "FAIL", g_bad);
    return g_bad == 0 ? 0 : 1;
}
