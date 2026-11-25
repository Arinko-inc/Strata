// sycl/src/core/verify_deferred.cpp - L1 (Arinko-inc): a remote stage worker's deferred windows on SYCL (the B70 end;
// strata-mesh IMPL-PLAN-2026-11-26-batch-drafts.md 3.5).  The CUDA tree's init_deferred_commit / capture_deferred /
// batch_commit_rows (src/core/verify.cpp 4013-4238) ported the SYCL way (dpct::experimental::begin_recording /
// end_recording, command_graph executables, copy_i32_from_mapped; the model is this tree's capture_commit_batch, with
// the saved rows dc_* in place of the window scratch), and run_slot_rows_deferred as the CUDA tree's
// (src/core/verify_deferred.cpp).  Only this stage's layers' rows are saved.
//
// The skeleton (2026-11-26) fixes the declarations (sycl/include/strata/core/verify.hpp) and the two hooks in
// verify.cpp (stage_batch skips capture_commit_batch while staging_deferred_; ~Verifier calls free_deferred); the
// bodies are 担当 S's (placeholders until then: errors, and nothing calls them without STRATA_WK_PIPED_MTP=1).
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "strata/sycl_queue.hpp"
#include "strata/core/verify.hpp"

#include <string>

namespace strata::core {

bool Verifier::init_deferred_commit(int rows, std::string& err) {
    (void) rows;
    err = "verify: init_deferred_commit is not built yet in the SYCL tree (L1 skeleton)";
    return false;
}

bool Verifier::capture_deferred(const int* rows, int S, int hbase, std::string& err) {
    (void) rows;
    (void) S;
    (void) hbase;
    err = "verify: capture_deferred is not built yet in the SYCL tree (L1 skeleton)";
    return false;
}

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
    err = "verify: run_slot_rows_deferred is not built yet in the SYCL tree (L1 skeleton)";
    return false;
}

bool Verifier::batch_commit_rows(const int* rows, int S, int hbase, const int* keep, const int32_t* tokens,
                                 const int64_t* pos, std::string& err) {
    (void) rows;
    (void) S;
    (void) hbase;
    (void) keep;
    (void) tokens;
    (void) pos;
    err = "verify: batch_commit_rows is not built yet in the SYCL tree (L1 skeleton)";
    return false;
}

void Verifier::free_deferred() {
    // (nothing is allocated yet: init_deferred_commit allocates nothing in the skeleton)
}

}  // namespace strata::core
