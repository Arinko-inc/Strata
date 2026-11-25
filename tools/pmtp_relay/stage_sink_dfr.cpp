// tools/pmtp_relay/stage_sink_dfr.cpp - tools/relay_read/stage_sink.cpp with L1's deferred windows (its hello has
// kStageBatchDeferred; it logs every SlotRun with its number on this connection - the relay's tags count the same way -
// and every deferred commit's tag and keep[], so a script can check that each commit names a deferred window on the
// slots that keep rows).  A stand-in LAST stage for checking a relay worker alone (no GPU, no model).
// It answers a relay's hello with the relay's own geometry and returns every row it is sent unchanged, so the main
// process's stand-in (stage_probe_bench --bench) reads the relay's own output rows (fp32 link: bitwise) and its
// wall time is the relay's.  Not a ctest; strata-mesh IMPL-PLAN-2026-11-26-relay-read.md.
//
//   STRATA_STAGE_TOKEN=... stage_sink_dfr PORT [--bind ADDR] [--chunk 3072] [--n-embd 2560] [--hc 4]
#include "strata/net/stage_link.hpp"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace strata::net;

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: stage_sink_dfr PORT [--bind ADDR] [--chunk N] [--n-embd E] [--hc H]\n");
        return 2;
    }
    const int port = std::atoi(argv[1]);
    std::string bind = "127.0.0.1";
    int64_t chunk = 3072, n_embd = 2560, hc = 4;
    for (int i = 2; i + 1 < argc; i += 2) {
        const std::string a = argv[i];
        if (a == "--bind") bind = argv[i + 1];
        else if (a == "--chunk") chunk = std::atoll(argv[i + 1]);
        else if (a == "--n-embd") n_embd = std::atoll(argv[i + 1]);
        else if (a == "--hc") hc = std::atoll(argv[i + 1]);
    }
    const int64_t D = hc * n_embd, HBF = hc * n_embd + n_embd + hc;   // Verifier::handoff_floats
    const int kMaxT = 8;
    std::vector<float> run_in((size_t) (kMaxT * HBF)), run_out((size_t) (kMaxT * HBF));
    std::vector<float> pf[4];
    for (auto& v : pf) v.assign((size_t) (chunk * D), 0.0f);
    StageBuffers buf;
    buf.run_in = run_in.data();
    buf.run_out = run_out.data();
    buf.run_floats = run_in.size();
    buf.pf_in[0] = pf[0].data();
    buf.pf_in[1] = pf[1].data();
    buf.pf_out[0] = pf[2].data();
    buf.pf_out[1] = pf[3].data();
    buf.pf_floats = (size_t) (chunk * D);
    buf.handoff_floats = HBF;
    buf.pf_row_floats = D;

    long long n_run = 0;   // the SlotRuns of this connection (the relay tags them 1, 2, ...)
    StageHandlers h;
    h.hello = [&](const StageHello& theirs, StageHello& mine, std::string& err) {
        if (theirs.n_embd != n_embd || theirs.hc != hc || theirs.chunk > chunk) {
            err = "stage_sink: geometry (start it with --n-embd / --hc / --chunk as the relay's)";
            return false;
        }
        mine = theirs;
        std::memset(mine.token, 0, sizeof mine.token);
        std::memset(mine.build, 0, sizeof mine.build);
        std::snprintf(mine.build, sizeof mine.build, "stage_sink_dfr");
        mine.max_t = kMaxT;
        mine.ckpt_max = kStageCkptMax;
        mine.batch_flags = kStageBatchKeepAll | kStageBatchOverlap | kStageBatchSlotPositions | kStageBatchDeferred;
        n_run = 0;
        return true;
    };
    h.run = [&](int T, const int32_t*, int64_t, std::string&) {
        std::memcpy(run_out.data(), run_in.data(), (size_t) T * (size_t) HBF * 4);
        return true;
    };
    h.commit = [](int, std::string&) { return true; };
    h.prefill = [&](const int64_t*, int64_t T, int64_t, int64_t, int64_t, const float* rows_in, float* rows_out,
                    std::string&) {
        std::memcpy(rows_out, rows_in, (size_t) T * (size_t) D * 4);
        return true;
    };
    h.reset = [](const std::vector<int64_t>&, std::string&) { return true; };
    h.positions = [](const int32_t*, int64_t, std::string&) { return true; };
    h.ckpt_save = [](int64_t, int64_t, const std::vector<int64_t>&, std::string&) { return 1; };
    h.ckpt_restore = [](int64_t, int64_t, const std::vector<int64_t>&, std::string&) { return true; };
    h.disconnected = [] {};
    h.slot_run = [&](int S, const int32_t* slots, const int32_t*, const int64_t* pos, int64_t flags, std::string&) {
        std::memcpy(run_out.data(), run_in.data(), (size_t) S * (size_t) HBF * 4);
        std::string sl, ps;
        for (int t = 0; t < S; ++t) {
            sl += (t ? "," : "") + std::to_string(slots[t]);
            ps += (t ? "," : "") + std::to_string((long long) pos[t]);
        }
        std::fprintf(stderr, "sink: run n=%lld flags=%lld slots=%s pos=%s\n", (long long) ++n_run, (long long) flags,
                     sl.c_str(), ps.c_str());
        return true;
    };
    h.slot_commit_deferred = [&](const int32_t* keep, int n, int64_t tag, std::string&) {
        std::string k;
        for (int s = 0; s < n; ++s) k += (s ? "," : "") + std::to_string(keep[s]);
        std::fprintf(stderr, "sink: dcommit tag=%lld keep=%s\n", (long long) tag, k.c_str());
        return true;
    };
    h.slot_commit = [](const int32_t* keep, int n, std::string&) {
        std::string k;
        for (int s = 0; s < n; ++s) k += (s ? "," : "") + std::to_string(keep[s]);
        std::fprintf(stderr, "sink: commit keep=%s\n", k.c_str());
        return true;
    };
    h.slot_save = [](int, int64_t, std::string&) { return true; };
    h.slot_load = [](int, int64_t, int64_t, std::string&) { return true; };

    const char* tk = std::getenv("STRATA_STAGE_TOKEN");
    StageStats st;
    std::atomic<bool> stop{false};
    std::fprintf(stderr, "stage_sink_dfr: %s:%d, chunk %lld, D %lld\n", bind.c_str(), port, (long long) chunk, (long long) D);
    return serve_stage(bind, port, tk ? tk : "", h, buf, st, &stop);
}
