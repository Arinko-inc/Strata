// src/core/verify_deferred.cpp - L1 (Arinko-inc): a remote stage worker's deferred windows on CUDA / HIP
// (strata-mesh IMPL-PLAN-2026-11-26-batch-drafts.md 3.4).  Verifier::run_slot_rows_deferred: run_slot_rows with
// staging_deferred_ set (stage_batch then captures no commit graph), then capture_deferred(rows, S, save_base) and its
// save graph queued on cs_ behind the window.  The commit is upstream's batch_commit_rows (verify.cpp).
//
// The skeleton (2026-11-26) fixes the declaration (include/strata/core/verify.hpp); the body is 担当 F's (a placeholder
// until then: an error, and nothing calls it without STRATA_WK_PIPED_MTP=1).
#include "strata/core/verify.hpp"

#include <string>

namespace strata::core {

bool Verifier::run_slot_rows_deferred(const int* rows, int S, int save_base, const int32_t* tokens, const int64_t* pos,
                                      PoolMultiFn pool, void* user, int32_t* out, std::string& err) {
    (void) rows;
    (void) S;
    (void) save_base;
    (void) tokens;
    (void) pos;
    (void) pool;
    (void) user;
    (void) out;
    err = "verify: run_slot_rows_deferred is not built yet (L1 skeleton)";
    return false;
}

}  // namespace strata::core
