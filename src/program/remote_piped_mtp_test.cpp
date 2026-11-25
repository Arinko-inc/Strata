// src/program/remote_piped_mtp_test.cpp - L1: a remote stage's main process's decision to draft in its pipelined
// groups (include/strata/program/remote_piped_mtp.hpp), header-only, no GPU.  strata-mesh
// IMPL-PLAN-2026-11-26-batch-drafts.md 3.3.
//
// The skeleton (2026-11-26) checks only that the placeholders keep today's behaviour (off); 担当 M adds decide()'s
// cases (each missing condition off with its reason, the worker's hello without kStageBatchDeferred, a worker whose
// max_t cannot hold a group's rows, STRATA_BATCH_SELF_COMMIT, STRATA_PIPE_PROBE_ROWS together with it, on with the
// drafts clamped) and drafts_that_fit's (2 slots x 2 rows in 4, 1 slot up to 3 drafts in 4, 4 slots in 4: 0).
#include "strata/program/remote_piped_mtp.hpp"

#include <cstdio>

namespace rp = strata::program::remote_piped_mtp;

namespace {
int g_fail = 0;
void check(bool ok, const char* what) {
    std::printf("  %-78s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}
}  // namespace

int main() {
    std::printf("remote_piped_mtp_test\n");
    rp::Input in;
    check(!rp::decide(in).on && rp::decide(in).why == nullptr, "nothing asked: off, nothing to say");
    in.remote_main = true;
    in.pm_drafts = 1;
    check(!rp::decide(in).on && rp::decide(in).why != nullptr, "STRATA_PIPED_MTP on a remote main process: off with a reason");
    std::printf(g_fail == 0 ? "all passed (0 failed)\n" : "%d FAILED\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
