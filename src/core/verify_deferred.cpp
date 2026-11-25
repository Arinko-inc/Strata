// src/core/verify_deferred.cpp - L1 (Arinko-inc): a remote stage worker's deferred windows on CUDA / HIP
// (strata-mesh IMPL-PLAN-2026-11-26-batch-drafts.md 3.4).  Verifier::run_slot_rows_deferred: run_slot_rows with
// staging_deferred_ set (stage_batch then captures no commit graph), then capture_deferred(rows, S, save_base) and its
// save graph queued on cs_ behind the window.  The commit is upstream's batch_commit_rows (verify.cpp).
//
// The graphs' keys: the window's is run_slot_rows' (rows, S, 0) - shared with the same rows' kStageSlotKeepAll window;
// the save and the deferred commit are (rows, S, save_base).  A worker's save_base is its first row's slot * max_t, so
// a layout has one of each.
#include "strata/core/verify.hpp"

#include "strata/core/on_device.hpp"

#include <string>

namespace strata::core {

bool Verifier::run_slot_rows_deferred(const int* rows, int S, int save_base, const int32_t* tokens, const int64_t* pos,
                                      PoolMultiFn pool, void* user, int32_t* out, std::string& err) {
    const OnDevice on_device(device_);
    if (dc_rows_ < 1) { err = "verify: run_slot_rows_deferred without init_deferred_commit"; return false; }
    if (S < 1 || save_base < 0 || save_base + S > dc_rows_) {
        err = "verify: a deferred window outside the saved rows";
        return false;
    }
    // (a worker's layers are one stage; a chain's later stage would stage its own commit graph in run_slot_rows)
    if (next_ != nullptr) { err = "verify: run_slot_rows_deferred on a stage with a next one"; return false; }
    if (b_running_) { err = "verify: run_slot_rows_deferred while this stage is busy"; return false; }
    staging_deferred_ = true;
    const bool ran = run_slot_rows(rows, S, tokens, pos, pool, user, out, err);
    staging_deferred_ = false;
    // (the window is done and its rows are in the hand-off: cs_ is idle, and the scratch the save reads stays as it
    // is until this stage's next window, which the stream orders behind the save)
    if (!ran || !capture_deferred(rows, S, save_base, err)) return false;
    // bkey holds the doorbell variant (#871: ar_off_), which refresh_ar may flip between this window and its commit
    // (a prompt loan beside the windows) - batch_commit_rows would then look its commit graph up under the other key
    // and find none.  The save and the commit read no doorbell, so on a stage that can flip (all_resident_) both
    // variants' keys get their graphs now.
    if (all_resident_) {
        ar_off_ = !ar_off_;
        const bool both = capture_deferred(rows, S, save_base, err);
        ar_off_ = !ar_off_;
        if (!both) return false;
    }
    const auto it = save_bm_.find(bkey(rows, S, save_base));
    if (it == save_bm_.end() || it->second == nullptr) { err = "verify: a deferred window without its save graph"; return false; }
    const cudaError_t le = cudaGraphLaunch(it->second, cs_);
    if (le != cudaSuccess) { err = std::string("verify: deferred save launch: ") + cudaGetErrorString(le); return false; }
    (void) cudaStreamQuery(cs_);
    return true;
}

}  // namespace strata::core
