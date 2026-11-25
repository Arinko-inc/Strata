// tools/pmtp_relay/stage_probe_dfr.cpp - a stand-in main process that checks a stage worker's deferred windows (L1,
// STRATA_WK_PIPED_MTP=1; strata-mesh IMPL-PLAN-2026-11-26-batch-drafts.md).  Not a ctest: it needs a running worker
// (a relay: with tools/pmtp_relay/stage_sink_dfr as its next worker).  The helpers are tools/relay_read/
// stage_probe_bench.cpp's.
//
//   STRATA_STAGE_TOKEN=... stage_probe_dfr HOST:PORT [--windows K] [--l0 L] [--seed S] [--vision] [--no-deferred]
//
// Four batch slots (two groups of two, as the main process's --batch-groups 2), each a prompt of noise rows read
// and saved, then K rounds of windows with two rows per slot (the token and one draft), in these runs - every run
// reads and saves the slots again, so each starts from the same state:
//   ka      kStageSlotKeepAll windows, group 0 then group 1 per round (every row kept)
//   c0-all  windows without a flag, each followed by its SlotCommit keeping every row (the worker's old waiting path)
//   df-all  kStageSlotDeferred windows as the main process pipelines them: group 0's window, the commit of group 1's
//           window before, group 1's window, group 0's commit ... - a window always runs while the other group's is
//           held; every row kept
//   c0-mix  as c0-all, the keeps mixed (1 or 2 rows per slot, by round and slot; a slot that kept 1 goes on from the
//           next cell)
//   df-mix  as df-all with c0-mix's keeps
// The output rows of every window are hashed.  With the same windows on the same state, a worker whose deferred commit
// (from the saved rows, after another window ran) writes what its commit graph writes gives df-all == c0-all == ka
// and df-mix == c0-mix, bitwise (an fp32 link; the next worker returns the rows unchanged).  Then the refusals: a
// deferred window on a held window's slot, a kStageSlotKeepAll window and a slot save while a window is held, a
// deferred commit that names no held window - and the link goes on after each.
// --no-deferred: ka, c0-all and c0-mix only (a worker without the flag: its hashes compare across executables).
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

constexpr int kSlots = 4;
constexpr uint64_t kFnv0 = 1469598103934665603ull;

uint64_t fnv(const void* p, size_t n, uint64_t h = kFnv0) {
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

struct Probe {
    std::string addr, token;
    StageHello peer;   // the worker's
    int64_t D = 0, HB = 0;
    bool vision = false;
    StageClient link;
    int64_t L[kSlots] = {};          // the slots' prompts
    std::vector<int64_t> ptok;       // the prompt's tokens [Lmax]
    std::vector<float> prow;         // its rows [Lmax][D]
    std::vector<float> wrow;         // the windows' hand-in R rows [kSlots][cells][D]
    std::vector<int32_t> wtok;       // [kSlots][cells]
    int cells = 0;                   // per slot
    int64_t tag = 0;

    bool connect(std::string& err) {
        StageHello mine = peer;
        mine.protocol = kStageProtocol;
        mine.max_t = 4;
        mine.max_context = 8192;
        mine.chunk = 2048;
        mine.tail_skip = 0;
        mine.vision = vision ? 1 : 0;
        mine.ckpt_max = 0;
        mine.row_formats = 1u << (uint32_t) StageRows::F32;
        mine.batch_slots = kSlots;
        mine.batch_flags = kStageBatchKeepAll;
        std::memset(mine.build, 0, sizeof mine.build);
        std::snprintf(mine.build, sizeof mine.build, "stage_probe_dfr");
        std::snprintf(mine.token, sizeof mine.token, "%s", token.c_str());
        link.close();
        StageHello got;
        if (!link.connect(addr, mine, got, err, 60)) return false;
        peer.max_context = got.max_context;
        peer.vision = got.vision;
        peer.batch_slots = got.batch_slots;
        peer.batch_flags = got.batch_flags;
        peer.max_t = got.max_t;
        tag = 0;
        return true;
    }

    // every slot's prompt read and saved (a run's start)
    bool setup(std::string& err) {
        std::vector<float> out;
        for (int s = 0; s < kSlots; ++s) {
            if (vision && !link.positions(nullptr, 0, err)) return false;
            if (!link.reset({}, err)) return false;
            out.assign((size_t) (L[s] * D), 0.0f);
            if (!link.prefill_send(ptok.data(), L[s], 0, 1, prow.data(), (size_t) D, 0, err) ||
                !link.prefill_recv(out.data(), (size_t) D, L[s], 0, err) || !link.slot_save(s, L[s], err))
                return false;
        }
        return link.reset({}, err);
    }

    struct Win {
        int S = 0;
        int32_t slots[8] = {};
        int32_t tok[8] = {};
        int64_t pos[8] = {};
        std::vector<float> in, out;
    };
    // group gi's window (slots 2 gi, 2 gi + 1), two rows per slot from each slot's cell at[s]
    // (ra / rb: the rows of the group's first / second slot, 0..2, not both 0)
    Win window(int gi, const int64_t* at, int ra = 2, int rb = 2) const {
        Win w;
        w.S = ra + rb;
        w.in.assign((size_t) (w.S * HB), 0.0f);   // the block layout: S R rows, then bo and inject (zero)
        w.out.assign((size_t) (w.S * HB), 0.0f);
        for (int t = 0; t < w.S; ++t) {
            const int s = 2 * gi + (t < ra ? 0 : 1);
            const int64_t c = at[s] + (t < ra ? t : t - ra);   // the cell after the slot's prompt
            w.slots[t] = s;
            w.pos[t] = L[s] + c;
            w.tok[t] = wtok[(size_t) (s * cells + c)];
            std::memcpy(w.in.data() + (size_t) t * D, wrow.data() + ((size_t) s * cells + (size_t) c) * D, (size_t) D * 4);
        }
        return w;
    }
    bool run(Win& w, int64_t flags, std::string& err) {
        return link.slot_run(w.S, w.slots, w.tok, w.pos, flags, ++tag, w.in.data(), w.in.size(), w.out.data(),
                             w.out.size(), err);
    }
    static int keep_of(bool mixed, int round, int s) { return mixed ? 1 + ((round + s + (s >> 1)) & 1) : 2; }

    struct RunResult {
        bool ok = false;
        std::string err;
        std::vector<uint64_t> win;   // per window, in (round, group) order
        uint64_t all = kFnv0;
        double ms = 0;               // per window, wall (the send to its rows)
        std::vector<float> rows;     // every window's output R rows (K <= 64), for the size of a difference
    };
    // mode 0: kStageSlotKeepAll; 1: no flag + SlotCommit; 2: kStageSlotDeferred, pipelined
    RunResult run_mode(int mode, bool mixed, int K) {
        RunResult r;
        std::string& err = r.err;
        if (!setup(err)) return r;
        int64_t at[kSlots] = {};
        r.win.assign((size_t) (2 * K), 0);
        double wall = 0;
        int64_t held_tag[2] = {0, 0};
        int held_round[2] = {-1, -1};
        auto commit = [&](int gi, int round, bool deferred) {
            int32_t keep[kSlots] = {};
            for (int s = 2 * gi; s < 2 * gi + 2; ++s) {
                keep[s] = keep_of(mixed, round, s);
                at[s] += keep[s];
            }
            return link.slot_commit(keep, kSlots, held_tag[gi], err, deferred);
        };
        for (int j = 0; j < K; ++j)
            for (int gi = 0; gi < 2; ++gi) {
                if (mode == 2 && held_round[gi] >= 0) {
                    // (cannot happen: a group's window is committed before its next one below)
                    err = "the probe's order";
                    return r;
                }
                Win w = window(gi, at);
                const auto t0 = Clock::now();
                if (!run(w, mode == 0 ? kStageSlotKeepAll : mode == 1 ? 0 : kStageSlotDeferred, err)) return r;
                wall += ms_since(t0);
                r.win[(size_t) (2 * j + gi)] = fnv(w.out.data(), w.out.size() * 4);
                if (K <= 64) r.rows.insert(r.rows.end(), w.out.begin(), w.out.begin() + (size_t) (w.S * D));
                held_tag[gi] = tag;
                if (mode == 0) {
                    for (int s = 2 * gi; s < 2 * gi + 2; ++s) at[s] += 2;
                } else if (mode == 1) {
                    if (!commit(gi, j, false)) return r;
                } else {
                    held_round[gi] = j;
                    // the other group's window (the one before this) is judged now: its commit, while this one is held
                    const int og = 1 - gi;
                    if (held_round[og] >= 0) {
                        if (!commit(og, held_round[og], true)) return r;
                        held_round[og] = -1;
                    }
                }
            }
        if (mode == 2)
            for (int gi = 0; gi < 2; ++gi)
                if (held_round[gi] >= 0) {
                    if (!commit(gi, held_round[gi], true)) return r;
                    held_round[gi] = -1;
                }
        // a commit is one way: a window after the last one brings its error back, if any (and shows the state it left)
        Win last = window(0, at);
        if (!run(last, kStageSlotKeepAll, err)) return r;
        for (uint64_t h : r.win) r.all = fnv(&h, sizeof h, r.all);
        const uint64_t lh = fnv(last.out.data(), last.out.size() * 4);
        r.all = fnv(&lh, sizeof lh, r.all);
        r.ms = wall / (2 * K);
        r.ok = true;
        return r;
    }
};

std::string hx(uint64_t v) {
    char b[24];
    std::snprintf(b, sizeof b, "%016" PRIx64, v);
    return b;
}
bool contains(const std::string& s, const char* t) { return s.find(t) != std::string::npos; }

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: stage_probe_dfr HOST:PORT [--windows K] [--l0 L] [--seed S] [--vision] "
                             "[--no-deferred]   (the token: STRATA_STAGE_TOKEN)\n");
        return 2;
    }
    Probe pr;
    pr.addr = argv[1];
    const char* tok = std::getenv("STRATA_STAGE_TOKEN");
    pr.token = tok ? tok : "";
    int K = 16;
    int64_t L0 = 192;
    unsigned seed = 11;
    bool no_deferred = false;
    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> const char* {
            if (i + 1 >= argc) { std::fprintf(stderr, "%s needs a value\n", a.c_str()); std::exit(2); }
            return argv[++i];
        };
        if (a == "--windows") K = std::atoi(next());
        else if (a == "--l0") L0 = std::atoll(next());
        else if (a == "--seed") seed = (unsigned) std::atoi(next());
        else if (a == "--vision") pr.vision = true;
        else if (a == "--no-deferred") no_deferred = true;
        else { std::fprintf(stderr, "unknown option %s\n", a.c_str()); return 2; }
    }
    if (K < 2 || K > 512 || L0 < 16 || L0 > 1500) { std::fprintf(stderr, "--windows 2..512, --l0 16..1500\n"); return 2; }
    std::string err;
    if (!discover(pr.addr, pr.token, pr.peer, err)) {
        std::fprintf(stderr, "stage_probe_dfr: %s\n", err.c_str());
        return 1;
    }
    pr.D = (int64_t) pr.peer.hc * pr.peer.n_embd;
    pr.HB = pr.peer.handoff_floats;
    if (!pr.connect(err)) {
        std::fprintf(stderr, "stage_probe_dfr: %s\n", err.c_str());
        return 1;
    }
    const bool has = (pr.peer.batch_flags & kStageBatchDeferred) != 0;
    std::printf("stage_probe_dfr: %s: layers from %d, hand-off %lld floats, batch slots %d, max_t %d, batch flags 0x%x "
                "(kStageBatchDeferred %s)\n", pr.addr.c_str(), pr.peer.layer_begin, (long long) pr.HB,
                (int) pr.peer.batch_slots, (int) pr.peer.max_t, (unsigned) pr.peer.batch_flags, has ? "yes" : "no");
    if (pr.peer.batch_slots < kSlots) {
        std::fprintf(stderr, "stage_probe_dfr: the worker has %d batch slots (4 needed)\n", (int) pr.peer.batch_slots);
        return 1;
    }
    // the inputs: normal noise, a fixed seed
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::uniform_int_distribution<int> td(1000, 100000);
    for (int s = 0; s < kSlots; ++s) pr.L[s] = L0 + 48 * s;
    const int64_t Lmax = pr.L[kSlots - 1];
    pr.ptok.resize((size_t) Lmax);
    for (auto& t : pr.ptok) t = td(rng);
    pr.prow.resize((size_t) (Lmax * pr.D));
    for (float& v : pr.prow) v = nd(rng);
    pr.cells = std::max(2 * K + 4, 16);   // (the layouts' part walks 9 cells of a slot)
    pr.wrow.resize((size_t) kSlots * (size_t) pr.cells * (size_t) pr.D);
    for (float& v : pr.wrow) v = nd(rng);
    pr.wtok.resize((size_t) kSlots * (size_t) pr.cells);
    for (auto& t : pr.wtok) t = td(rng);

    auto show = [&](const char* name, const Probe::RunResult& r) {
        if (!r.ok) {
            std::printf("  %-7s FAILED: %s\n", name, r.err.c_str());
            return;
        }
        std::printf("  %-7s all %s  first %s last %s  %.2f ms a window (wall)\n", name, hx(r.all).c_str(),
                    hx(r.win.front()).c_str(), hx(r.win.back()).c_str(), r.ms);
        std::fflush(stdout);
    };
    auto same = [&](const Probe::RunResult& a, const Probe::RunResult& b) {
        if (!a.ok || !b.ok || a.win.size() != b.win.size()) return std::string("(a run failed)");
        int diff = 0, first = -1;
        for (size_t i = 0; i < a.win.size(); ++i)
            if (a.win[i] != b.win[i]) {
                ++diff;
                if (first < 0) first = (int) i;
            }
        if (diff == 0 && a.all == b.all) return std::string();
        // how large: the relative L2 of the difference over every window's rows, and over the first that differs
        std::string size;
        if (!a.rows.empty() && a.rows.size() == b.rows.size()) {
            const size_t per = a.rows.size() / a.win.size();
            auto rel = [&](size_t lo, size_t hi) {
                double dd = 0, rr = 0;
                bool nan = false;
                for (size_t i = lo; i < hi; ++i) {
                    const double d = (double) a.rows[i] - b.rows[i];
                    dd += d * d;
                    rr += (double) a.rows[i] * a.rows[i];
                    nan = nan || !std::isfinite(a.rows[i]) || !std::isfinite(b.rows[i]);
                }
                char buf[64];
                std::snprintf(buf, sizeof buf, "%.3e%s", rr > 0 ? std::sqrt(dd / rr) : 0.0, nan ? " (non-finite!)" : "");
                return std::string(buf);
            };
            size = "; relative L2 " + rel(0, a.rows.size()) +
                   (first >= 0 ? ", window " + std::to_string(first) + " alone " +
                                     rel((size_t) first * per, (size_t) (first + 1) * per) +
                                     ", the last window " + rel(a.rows.size() - per, a.rows.size())
                               : "");
        }
        return size + "; " + std::to_string(diff) + " of " + std::to_string(a.win.size()) + " windows differ (the first: window " +
               std::to_string(first) + ")" + (a.all != b.all && diff == 0 ? ", the window after the last differs" : "");
        // (callers print it after their text: it starts with "; ")
    };
    // a failed run leaves the link as the error left it: connect again before the next
    auto run = [&](const char* name, int mode, bool mixed) {
        Probe::RunResult r = pr.run_mode(mode, mixed, K);
        show(name, r);
        if (!r.ok) {
            std::string ce;
            if (!pr.connect(ce)) std::printf("  (connecting again: %s)\n", ce.c_str());
        }
        return r;
    };
    std::printf("stage_probe_dfr: %d rounds of 2 windows (4 rows: 2 slots x 2 cells), slots at %lld / %lld / %lld / %lld\n",
                K, (long long) pr.L[0], (long long) pr.L[1], (long long) pr.L[2], (long long) pr.L[3]);
    const Probe::RunResult ka = run("ka", 0, false);
    const Probe::RunResult ka2 = run("ka", 0, false);
    const Probe::RunResult c0a = run("c0-all", 1, false);
    const Probe::RunResult c0m = run("c0-mix", 1, true);
    verdict(ka.ok && same(ka, ka2).empty(), "ka twice: the same rows (the worker is deterministic) " + same(ka, ka2));
    verdict(c0a.ok && same(ka, c0a).empty(), "c0-all == ka " + same(ka, c0a));
    verdict(c0m.ok && !same(c0a, c0m).empty(), "c0-mix differs from c0-all (the keeps reach the state)");
    if (no_deferred || !has) {
        if (!no_deferred) {
            Probe::Win w = pr.window(0, std::vector<int64_t>(kSlots, 0).data());
            std::string e;
            const bool took = pr.setup(e) && pr.run(w, kStageSlotDeferred, e);
            verdict(!took && contains(e, "bad slot window"),
                    "no kStageBatchDeferred in the hello: a deferred window is refused (" + e + ")");
        }
        std::printf("stage_probe_dfr: %s\n", g_bad == 0 ? "all passed (no deferred runs)" : "FAILED");
        return g_bad == 0 ? 0 : 1;
    }
    const Probe::RunResult dfa = run("df-all", 2, false);
    const Probe::RunResult dfm = run("df-mix", 2, true);
    const Probe::RunResult dfm2 = run("df-mix", 2, true);
    const Probe::RunResult c0a2 = run("c0-all", 1, false);
    verdict(dfa.ok && same(c0a, dfa).empty(), "df-all == c0-all (every row kept, from the saved rows) " + same(c0a, dfa));
    verdict(dfm.ok && same(c0m, dfm).empty(), "df-mix == c0-mix (mixed keeps, from the saved rows) " + same(c0m, dfm));
    verdict(dfm2.ok && same(dfm, dfm2).empty(), "df-mix twice: the same rows " + same(dfm, dfm2));
    verdict(c0a2.ok && same(c0a, c0a2).empty(), "c0-all after the deferred runs: the same rows " + same(c0a, c0a2));

    // the refusals
    {
        std::string e;
        int64_t at[kSlots] = {};
        bool ok = pr.setup(e);
        Probe::Win g0 = pr.window(0, at), g1 = pr.window(1, at);
        ok = ok && pr.run(g0, kStageSlotDeferred, e);
        const int64_t g0_tag = pr.tag;
        verdict(ok, "refusals: slots saved, group 0's deferred window held " + e);
        // (1) a deferred window with a row on a held slot
        Probe::Win bad = g1;
        bad.slots[0] = bad.slots[1] = 1;
        bad.pos[0] = pr.L[1];
        bad.pos[1] = pr.L[1] + 1;
        e.clear();
        bool took = pr.run(bad, kStageSlotDeferred, e);
        verdict(!took && contains(e, "waits for its SlotCommit") && pr.link.connected(),
                "a deferred window on a held window's slot is refused (" + e + ")");
        // (2) a kStageSlotKeepAll window while one is held
        e.clear();
        took = pr.run(g1, kStageSlotKeepAll, e);
        verdict(!took && contains(e, "deferred batch windows wait") && pr.link.connected(),
                "a kStageSlotKeepAll window while a deferred one is held is refused (" + e + ")");
        // (3) a slot save while one is held
        e.clear();
        took = pr.link.slot_save(3, 0, e);
        verdict(!took && pr.link.connected(), "a slot save while a deferred window is held is refused (" + e + ")");
        // (4) group 1's deferred window runs meanwhile, then a commit that names no held window (slot 0 alone keeps
        // rows: group 0's window needs slot 1 too - it is dropped uncommitted, the next window brings the error)
        e.clear();
        ok = pr.run(g1, kStageSlotDeferred, e);
        const int64_t g1_tag = pr.tag;
        verdict(ok, "group 1's deferred window runs while group 0's is held " + e);
        const int32_t k_bad[kSlots] = {1, 0, 0, 0};
        e.clear();
        ok = pr.link.slot_commit(k_bad, kSlots, g0_tag, e, true);
        const int32_t k1[kSlots] = {0, 0, 2, 2};
        std::string e2;
        const bool sent = ok && pr.link.slot_commit(k1, kSlots, g1_tag, e2, true);
        at[2] = at[3] = 2;
        Probe::Win g1b = pr.window(1, at);
        std::string e3;
        took = pr.run(g1b, kStageSlotDeferred, e3);
        verdict(sent && !took && contains(e3, "bad slot commit") && pr.link.connected(),
                "a deferred commit that keeps rows on one of its window's two slots: the next window's error (" + e3 + ")");
        // (5) the link goes on: group 1's commit above was skipped with the error (serve_stage reports one at a time),
        // so its window is still held - commit it again, then its next window runs
        e.clear();
        ok = pr.link.slot_commit(k1, kSlots, g1_tag, e, true) && pr.run(g1b, kStageSlotDeferred, e);
        const int64_t g1b_tag = pr.tag;
        verdict(ok, "... and the link goes on: group 1's commit, then its next deferred window " + e);
        // (6) a commit on slots no window holds (group 0's was dropped)
        const int32_t k0[kSlots] = {2, 2, 0, 0};
        e.clear();
        e2.clear();
        ok = pr.link.slot_commit(k0, kSlots, g0_tag, e, true) && pr.link.slot_commit(k1, kSlots, g1b_tag, e2, true);
        at[2] = at[3] = 4;
        Probe::Win g1c = pr.window(1, at);
        e3.clear();
        took = pr.run(g1c, kStageSlotKeepAll, e3);
        verdict(ok && !took && contains(e3, "no deferred window holds it") && pr.link.connected(),
                "a deferred commit on slots no window holds: the next window's error (" + e3 + ")");
        // (7) a reset with a window held (group 1's second window, whose commit was skipped with the error above)
        e.clear();
        took = pr.link.reset({}, e);
        verdict(!took && contains(e, "deferred batch windows wait"), "a reset while a deferred window is held is refused (" + e + ")");
    }
    // a new connection drops what was held, and the runs give the same rows again
    {
        std::string e;
        verdict(pr.connect(e), "a new connection after the refusals " + e);
        const Probe::RunResult dfm3 = run("df-mix", 2, true);
        verdict(dfm3.ok && same(c0m, dfm3).empty(), "df-mix on the new connection == c0-mix " + same(c0m, dfm3));
    }
    // every layout the main process's groups make with one draft per slot (2 groups x 8: one or both slots, 1 or 2
    // rows each), each window deferred and committed whole: the worker then holds every layout's graphs (its VRAM)
    {
        std::string e;
        bool ok = pr.setup(e);
        int64_t at[kSlots] = {};
        int n = 0;
        for (int gi = 0; gi < 2 && ok; ++gi)
            for (int ra = 0; ra <= 2 && ok; ++ra)
                for (int rb = 0; rb <= 2 && ok; ++rb) {
                    if (ra + rb == 0) continue;
                    Probe::Win w = pr.window(gi, at, ra, rb);
                    int32_t keep[kSlots] = {};
                    keep[2 * gi] = ra;
                    keep[2 * gi + 1] = rb;
                    ok = pr.run(w, kStageSlotDeferred, e) && pr.link.slot_commit(keep, kSlots, pr.tag, e, true);
                    at[2 * gi] += ra;
                    at[2 * gi + 1] += rb;
                    ++n;
                }
        Probe::Win last = pr.window(1, at);
        ok = ok && pr.run(last, kStageSlotKeepAll, e);
        verdict(ok && n == 16, "16 window layouts, each deferred and committed whole " + e);
    }
    std::printf("stage_probe_dfr: %s\n", g_bad == 0 ? "all passed" : "FAILED");
    return g_bad == 0 ? 0 : 1;
}
