// src/program/remote_piped_mtp_test.cpp - L1: a remote stage's main process's decision to draft in its pipelined
// groups (include/strata/program/remote_piped_mtp.hpp), header-only, no GPU.  strata-mesh
// IMPL-PLAN-2026-11-26-batch-drafts.md 3.3.
//
// decide(): nothing asked (no STRATA_PIPED_MTP, or not a remote main process) is off with nothing to say; each missing
// condition is off with its reason (the measurement flag, --batch-mtp, the remote pipelined groups, the groups not
// dividing the slots, STRATA_BATCH_SELF_COMMIT, the worker's hello without kStageBatchDeferred, a worker whose max_t
// cannot hold a group's rows); everything met is on with the drafts clamped (to STRATA_PIPED_MTP and to what fits).
// drafts_that_fit(): 2 slots x 2 rows in 4, 1 slot up to 3 drafts in 4, 4 slots in 4: 0, and the edges.
#include "strata/program/remote_piped_mtp.hpp"

#include <cstdio>

namespace rp = strata::program::remote_piped_mtp;

namespace {
int g_fail = 0;
void check(bool ok, const char* what) {
    std::printf("  %-86s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}
/// every condition met: 4 slots in 2 groups, the worker takes 4 rows and has the deferred bit, STRATA_PIPED_MTP=1
rp::Input good() {
    rp::Input in;
    in.remote_main = true;
    in.remote_pipe = true;
    in.batch_mtp = true;
    in.pm_drafts = 1;
    in.batch = 4;
    in.groups = 2;
    in.peer_batch_flags = strata::net::kStageBatchDeferred;
    in.peer_max_t = 4;
    return in;
}
bool off_with_reason(const rp::Input& in) {
    const rp::Result r = rp::decide(in);
    return !r.on && r.drafts == 0 && r.why != nullptr && r.why[0] != '\0';
}
}  // namespace

int main() {
    std::printf("remote_piped_mtp_test\n");

    std::printf("decide: nothing asked\n");
    {
        rp::Input in;
        check(!rp::decide(in).on && rp::decide(in).why == nullptr, "nothing set: off, nothing to say");
        in = good();
        in.pm_drafts = 0;
        check(!rp::decide(in).on && rp::decide(in).why == nullptr, "STRATA_PIPED_MTP unset: off, nothing to say");
        in = good();
        in.remote_main = false;
        check(!rp::decide(in).on && rp::decide(in).why == nullptr, "not a remote main process: off, nothing to say");
    }

    std::printf("decide: every condition met\n");
    {
        rp::Input in = good();
        const rp::Result r = rp::decide(in);
        check(r.on && r.drafts == 1 && r.why == nullptr, "4 slots in 2 groups, worker 4 rows, N=1: on with 1 draft");
        in.pm_drafts = 3;
        check(rp::decide(in).on && rp::decide(in).drafts == 1, "STRATA_PIPED_MTP=3 with 2 slots per group in 4 rows: clamped to 1");
        in.batch = 2;
        in.groups = 2;
        check(rp::decide(in).on && rp::decide(in).drafts == 3, "1 slot per group in 4 rows, N=3: on with 3 drafts");
        in.peer_max_t = 8;
        in.pm_drafts = 7;
        check(rp::decide(in).on && rp::decide(in).drafts == 7, "1 slot per group in 8 rows, N=7: on with 7 drafts");
        in.peer_max_t = 64;
        in.pm_drafts = 20;
        check(rp::decide(in).on && rp::decide(in).drafts == 7, "a worker's max_t over 8: the window is 8 rows, N clamped to 7");
        in = good();
        in.batch = 8;
        in.groups = 2;
        in.peer_max_t = 8;
        check(rp::decide(in).on && rp::decide(in).drafts == 1, "8 slots in 2 groups (4 each) in 8 rows, N=1: on with 1 draft");
        in.batch = 6;
        in.groups = 2;
        in.peer_max_t = 6;
        check(rp::decide(in).on && rp::decide(in).drafts == 1, "3 slots per group in 6 rows: on with 1 draft");
    }

    std::printf("decide: each missing condition\n");
    {
        rp::Input in = good();
        in.probe_rows = 2;
        check(off_with_reason(in), "STRATA_PIPE_PROBE_ROWS together with it: off");
        in = good();
        in.batch_mtp = false;
        check(off_with_reason(in), "--batch-mtp off or refused: off");
        in = good();
        in.remote_pipe = false;
        check(off_with_reason(in), "no remote pipelined groups: off");
        in = good();
        in.groups = 1;
        check(off_with_reason(in), "one group: off");
        in = good();
        in.batch = 1;
        in.groups = 2;
        check(off_with_reason(in), "fewer than 2 slots: off");
        in = good();
        in.batch = 5;
        in.groups = 2;
        check(off_with_reason(in), "the groups do not divide the slots: off");
        in = good();
        in.self_commit = true;
        check(off_with_reason(in), "STRATA_BATCH_SELF_COMMIT: off");
        in = good();
        in.peer_batch_flags = 0;
        check(off_with_reason(in), "the worker's hello has no kStageBatchDeferred: off");
        in = good();
        in.peer_batch_flags = strata::net::kStageBatchKeepAll | strata::net::kStageBatchOverlap;
        check(off_with_reason(in), "the worker has other bits but not kStageBatchDeferred: off");
        in = good();
        in.peer_max_t = 3;
        check(off_with_reason(in), "the worker takes 3 rows, 2 slots x 2 rows = 4: off");
        in = good();
        in.batch = 8;
        in.groups = 2;
        check(off_with_reason(in), "4 slots per group in 4 rows (no room for a draft): off");
        in = good();
        in.peer_max_t = 0;
        check(off_with_reason(in), "the worker's max_t unknown (0): off");
    }

    std::printf("decide: the reasons differ by cause\n");
    {
        rp::Input a = good(), b = good();
        a.peer_batch_flags = 0;
        b.self_commit = true;
        check(rp::decide(a).why != rp::decide(b).why, "different causes give different reasons");
    }

    std::printf("drafts_that_fit\n");
    check(rp::drafts_that_fit(2, 1, 4) == 1, "2 slots, 1 draft, 4 rows: 1 (2 x 2 rows)");
    check(rp::drafts_that_fit(2, 3, 4) == 1, "2 slots, 3 drafts, 4 rows: cut to 1");
    check(rp::drafts_that_fit(1, 3, 4) == 3, "1 slot, 3 drafts, 4 rows: 3 (1 + 3 rows)");
    check(rp::drafts_that_fit(1, 5, 4) == 3, "1 slot, 5 drafts, 4 rows: cut to 3");
    check(rp::drafts_that_fit(1, 1, 4) == 1, "1 slot, 1 draft, 4 rows: 1 (asked for less than fits)");
    check(rp::drafts_that_fit(4, 1, 4) == 0, "4 slots, 4 rows: 0 (the tokens fill the window)");
    check(rp::drafts_that_fit(5, 1, 4) == 0, "5 slots, 4 rows: 0 (not even the tokens fit; never negative)");
    check(rp::drafts_that_fit(3, 2, 8) == 1, "3 slots, 2 drafts, 8 rows: 1 (3 x 2 = 6; 3 x 3 = 9 is over)");
    check(rp::drafts_that_fit(1, 20, 100) == 7, "a cap over 8 is 8 rows: 1 slot gets 7 drafts");
    check(rp::drafts_that_fit(0, 1, 4) == 0, "no active slot: 0");
    check(rp::drafts_that_fit(2, 0, 4) == 0, "no draft asked: 0");
    check(rp::drafts_that_fit(2, 1, 0) == 0, "no room: 0");
    check(rp::drafts_that_fit(2, -1, 4) == 0, "a negative draft count: 0");
    {   // 組の窓の行 = 枠の数 x (1 + 下書き) が cap を超えない
        bool ok = true;
        for (int s = 1; s <= 8; ++s)
            for (int d = 0; d <= 9; ++d)
                for (int cap = 1; cap <= 12; ++cap) {
                    const int n = rp::drafts_that_fit(s, d, cap);
                    const int c = cap < 8 ? cap : 8;
                    if (n < 0 || n > d || (n > 0 && s * (1 + n) > c) || (n < d && n >= 0 && s * (1 + n + 1) <= c && d > 0)) ok = false;
                }
        check(ok, "for every (slots, drafts, cap): the rows fit, and one more draft would not (or none asked)");
    }

    std::printf(g_fail == 0 ? "all passed (0 failed)\n" : "%d FAILED\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
