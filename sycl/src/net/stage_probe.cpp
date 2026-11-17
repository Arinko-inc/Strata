// src/net/stage_probe.cpp (SYCL build, not a ctest: it needs a running worker) - a stand-in main process for one
// stage worker, to check a worker's image positions (--vision, StageMsg::Positions) without a main process.
//
//   STRATA_STAGE_TOKEN=... stage_probe HOST:PORT [--vision] [--batch] [--n N] [--decode D] [--seed S]
//   STRATA_STAGE_TOKEN=... stage_probe HOST:PORT --expect-refusal TEXT
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
constexpr int64_t kChunk = 2048;
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
    int decode = 32;
    StageClient link;

    bool connect(std::string& err) {
        StageHello mine = g.peer;   // geometry, pack and K/V as the worker's
        mine.protocol = kStageProtocol;
        mine.max_t = kW;
        mine.max_context = 8192;
        mine.chunk = kChunk;
        mine.tail_skip = 0;
        mine.vision = vision ? 1 : 0;
        mine.ckpt_max = 0;
        mine.row_formats = 1u << (uint32_t) StageRows::F32;
        mine.batch_slots = batch ? 1 : 0;
        mine.batch_flags = batch ? kStageBatchKeepAll : 0;
        std::memset(mine.build, 0, sizeof mine.build);
        std::snprintf(mine.build, sizeof mine.build, "stage_probe");
        std::snprintf(mine.token, sizeof mine.token, "%s", token.c_str());
        link.close();
        StageHello peer;
        if (!link.connect(addr, mine, peer, err, 60)) return false;
        g.peer.max_context = peer.max_context;
        g.peer.vision = peer.vision;
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
        else { std::fprintf(stderr, "unknown option %s\n", a.c_str()); return 2; }
    }
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
    if (!pr.connect(err)) {
        std::fprintf(stderr, "stage_probe: %s\n", err.c_str());
        return 1;
    }
    std::printf("stage_probe: connected (vision %d, batch slots %d; the worker's vision %d, context %lld)\n",
                pr.vision ? 1 : 0, pr.batch ? 1 : 0, pr.g.peer.vision, (long long) pr.g.peer.max_context);

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
