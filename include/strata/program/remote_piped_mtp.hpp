// include/strata/program/remote_piped_mtp.hpp - L1 (Arinko-inc): whether a remote stage's main process drafts in its
// pipelined groups (--batch-mtp + STRATA_PIPED_MTP=N over the link), and how many drafts a group's window carries
// (strata-mesh IMPL-PLAN-2026-11-26-batch-drafts.md 3.3).  Pure functions, no GPU (remote_piped_mtp_test.cpp).
//
// The skeleton (2026-11-26) fixes the declarations; the bodies at the end are 担当 M's (placeholders until then:
// decide() is always off, with a reason, which is what the main process does today - it turns STRATA_PIPED_MTP off
// over the link).  The public signatures may change only with a reason in the report.
#pragma once

#include <cstdint>

namespace strata::program::remote_piped_mtp {

struct Input {
    bool remote_main = false, remote_pipe = false;   ///< the main process of a remote stage, pipelined groups
    bool batch_mtp = false;                          ///< --batch-mtp / STRATA_BATCH_MTP after its own gates
    int pm_drafts = 0;                               ///< STRATA_PIPED_MTP=N (0: unset)
    int batch = 0, groups = 0;                       ///< the slots agreed, the groups
    uint32_t peer_batch_flags = 0;                   ///< the worker's hello (strata::net::kStageBatchDeferred, bit 3)
    int peer_max_t = 0;                              ///< the worker's hello max_t (the rows a window may hold there)
    bool self_commit = false;                        ///< STRATA_BATCH_SELF_COMMIT
    int probe_rows = 0;                              ///< STRATA_PIPE_PROBE_ROWS (stage 0's measurement; not with drafts)
};
/// on: the groups draft, `drafts` per slot (1..pm_drafts); off: `why` is the reason, logged once (nullptr when there is
/// nothing to say: STRATA_PIPED_MTP unset, or not a remote main process)
struct Result {
    bool on = false;
    int drafts = 0;
    const char* why = nullptr;
};
Result decide(const Input& in);
/// a group's rows: each active slot 1 + its drafts, the drafts cut (the same for every slot) so the window holds at
/// most `cap` rows; the drafts per slot (0: only the tokens fit)
int drafts_that_fit(int active_slots, int drafts, int cap);

// ==== the bodies: 担当 M (strata-mesh IMPL-PLAN-2026-11-26-batch-drafts.md, the section after the skeleton) ====
// (placeholders: off, as today)
inline Result decide(const Input& in) {
    Result r;
    if (in.remote_main && in.pm_drafts > 0)
        r.why = "the remote groups read a prompt between serial windows (L1 skeleton: not built yet)";
    return r;
}
inline int drafts_that_fit(int, int, int) { return 0; }
// ==== end of 担当 M's bodies ====

}  // namespace strata::program::remote_piped_mtp
