// include/strata/net/stage_deferred.hpp - L1 (Arinko-inc): a remote stage worker's deferred windows, the table of the
// windows it holds uncommitted (strata-mesh IMPL-PLAN-2026-11-26-batch-drafts.md 3.4).  Both workers use it: the
// CUDA/HIP one (src/program/stage_worker_deferred.inc, a relay) and the SYCL one (sycl/src/program/
// stage_worker_deferred.inc, the end).  No GPU: the GPU side is Verifier::run_slot_rows_deferred and
// Verifier::batch_commit_rows; this only checks and keeps what the link brought (stage_deferred_test.cpp).
//
// A deferred window (SlotRun with kStageSlotDeferred) runs on the worker's slots now and is committed later, by its tag
// (SlotCommit with kStageCommitDeferred), while other deferred windows on other slots run.  Each held window's rows
// that its commit replays are saved in the region of its first row's slot: saved rows [save_base, save_base + S),
// save_base = rows[0] * rows_per_region (the worker's max_t), so two held windows never share saved rows.
//
// The skeleton (2026-11-26) fixes the declarations below; the bodies at the end of this file are 担当 F's
// (placeholders until then: every check fails, so nothing is held - and nothing calls them without
// STRATA_WK_PIPED_MTP=1).  The private members may change; the public signatures are shared with 担当 S (change them
// only with a reason in the report).
#pragma once

#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

namespace strata::net {

/// the most rows a deferred window holds (= strata::kernels::kVerifyMaxT: a worker static_asserts it)
inline constexpr int kDeferredMaxRows = 8;

/// STRATA_WK_PIPED_MTP=1 on a worker (its value, std::getenv's): on for a value that is set, not empty and not "0"
inline bool wk_piped_mtp_on(const char* v) { return v != nullptr && v[0] != '\0' && !(v[0] == '0' && v[1] == '\0'); }

/// one deferred window, as the worker ran it
struct DeferredWindow {
    int64_t tag = 0;        ///< the SlotRun's tag (the main process's, or the relay's before this worker)
    int64_t next_tag = 0;   ///< a relay: the tag it sent the window on with (its next worker's commit names it)
    int S = 0;              ///< rows
    int rows[kDeferredMaxRows] = {};      ///< row t's slot (a slot's rows contiguous)
    int32_t tok[kDeferredMaxRows] = {};   ///< row t's token
    int64_t pos[kDeferredMaxRows] = {};   ///< row t's cell
    int save_base = 0;      ///< its saved rows [save_base, save_base + S): rows[0] * rows_per_region (add sets it)
};

/// the deferred windows a worker holds, per connection
class DeferredWindows {
public:
    /// The hello and a disconnect: none held; `n_slots` the slots the link agreed (<= what the worker carved),
    /// `rows_per_region` the saved rows of one slot's region (the worker's max_t).
    void reset(int n_slots, int rows_per_region);
    /// Holds `w`: S in 1..kDeferredMaxRows and <= rows_per_region, every slot in [0, n_slots), a slot's rows
    /// contiguous, and no slot in a window held already (so rows[0]'s region is free); a tag not held already.  Sets
    /// w.save_base.  False + err otherwise (nothing held then).
    bool add(DeferredWindow& w, std::string& err);
    /// The held window `tag` into `out`, no longer held.  False + err when no held window has that tag.
    bool take(int64_t tag, DeferredWindow& out, std::string& err);
    /// whether a held window has a row on `slot`
    bool slot_held(int slot) const;
    bool empty() const;
    int size() const;
    /// A SlotCommit's keep[s] (s < m, m <= n_slots) against `w`: each of w's slots keeps 1..its rows, every other
    /// slot 0.  keep_by_slot gets n_slots entries (its keep, zeros elsewhere: Verifier::batch_commit_rows's form).
    /// False + err otherwise.
    static bool keep_ok(const DeferredWindow& w, const int32_t* keep, int m, int n_slots, std::vector<int>& keep_by_slot,
                        std::string& err);

private:
    int n_slots_ = 0, rows_per_region_ = 0;
    std::vector<DeferredWindow> held_;
};

// ==== the bodies: 担当 F (strata-mesh IMPL-PLAN-2026-11-26-batch-drafts.md, the section after the skeleton) ====
// (placeholders: nothing is ever held)
inline void DeferredWindows::reset(int n_slots, int rows_per_region) {
    n_slots_ = n_slots;
    rows_per_region_ = rows_per_region;
    held_.clear();
}
inline bool DeferredWindows::add(DeferredWindow&, std::string& err) {
    err = "stage worker: deferred windows are not built yet (L1 skeleton)";
    return false;
}
inline bool DeferredWindows::take(int64_t tag, DeferredWindow&, std::string& err) {
    err = "stage worker: no deferred window has tag " + std::to_string((long long) tag) + " (L1 skeleton)";
    return false;
}
inline bool DeferredWindows::slot_held(int) const { return false; }
inline bool DeferredWindows::empty() const { return held_.empty(); }
inline int DeferredWindows::size() const { return (int) held_.size(); }
inline bool DeferredWindows::keep_ok(const DeferredWindow&, const int32_t*, int, int, std::vector<int>& keep_by_slot,
                                     std::string& err) {
    keep_by_slot.clear();
    err = "stage worker: deferred commits are not built yet (L1 skeleton)";
    return false;
}
// ==== end of 担当 F's bodies ====

}  // namespace strata::net
