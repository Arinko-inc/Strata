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
// The skeleton (2026-11-26) fixed the declarations below; the bodies at the end of this file are 担当 F's (nothing
// calls them without STRATA_WK_PIPED_MTP=1).  The public signatures are shared with 担当 S.
#pragma once

#include <cstddef>
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
inline void DeferredWindows::reset(int n_slots, int rows_per_region) {
    n_slots_ = n_slots;
    rows_per_region_ = rows_per_region;
    held_.clear();
}
inline bool DeferredWindows::add(DeferredWindow& w, std::string& err) {
    const std::string bad = "stage worker: bad slot window (deferred): ";
    if (n_slots_ < 1 || rows_per_region_ < 1) {
        err = bad + "this link keeps no deferred windows";
        return false;
    }
    if (w.S < 1 || w.S > kDeferredMaxRows || w.S > rows_per_region_) {
        err = bad + std::to_string(w.S) + " rows (a deferred window holds up to " +
              std::to_string(rows_per_region_ < kDeferredMaxRows ? rows_per_region_ : kDeferredMaxRows) + ")";
        return false;
    }
    for (int t = 0; t < w.S; ++t) {
        const int b = w.rows[t];
        if (b < 0 || b >= n_slots_) {
            err = bad + "slot " + std::to_string(b) + " is outside the " + std::to_string(n_slots_) + " batch slots";
            return false;
        }
        if (t > 0 && w.rows[t - 1] == b) continue;
        for (int u = 0; u < t; ++u)
            if (w.rows[u] == b) {
                err = bad + "slot " + std::to_string(b) + "'s rows are not together";
                return false;
            }
        if (slot_held(b)) {
            err = bad + "slot " + std::to_string(b) + " is in a deferred window that waits for its SlotCommit";
            return false;
        }
    }
    for (const DeferredWindow& h : held_)
        if (h.tag == w.tag) {
            err = bad + "tag " + std::to_string((long long) w.tag) + " is a deferred window held already";
            return false;
        }
    w.save_base = w.rows[0] * rows_per_region_;
    held_.push_back(w);
    return true;
}
inline bool DeferredWindows::take(int64_t tag, DeferredWindow& out, std::string& err) {
    for (size_t i = 0; i < held_.size(); ++i)
        if (held_[i].tag == tag) {
            out = held_[i];
            held_.erase(held_.begin() + (std::ptrdiff_t) i);
            return true;
        }
    err = "stage worker: bad slot commit: no deferred window has tag " + std::to_string((long long) tag);
    return false;
}
inline bool DeferredWindows::slot_held(int slot) const {
    for (const DeferredWindow& h : held_)
        for (int t = 0; t < h.S; ++t)
            if (h.rows[t] == slot) return true;
    return false;
}
inline bool DeferredWindows::empty() const { return held_.empty(); }
inline int DeferredWindows::size() const { return (int) held_.size(); }
inline bool DeferredWindows::keep_ok(const DeferredWindow& w, const int32_t* keep, int m, int n_slots,
                                     std::vector<int>& keep_by_slot, std::string& err) {
    const std::string bad = "stage worker: bad slot commit (deferred): ";
    keep_by_slot.assign((size_t) (n_slots > 0 ? n_slots : 0), 0);
    if (m < 1 || m > n_slots) {
        err = bad + std::to_string(m) + " slots, this worker has " + std::to_string(n_slots);
        return false;
    }
    if (w.S < 1 || w.S > kDeferredMaxRows) {
        err = bad + "the window has " + std::to_string(w.S) + " rows";
        return false;
    }
    int in_window[kDeferredMaxRows] = {};   // per slot of the window (in row order): its rows
    int slot_of[kDeferredMaxRows] = {};
    int n = 0;
    for (int t = 0; t < w.S; ++t) {
        if (t > 0 && w.rows[t] == w.rows[t - 1]) {
            ++in_window[n - 1];
            continue;
        }
        slot_of[n] = w.rows[t];
        in_window[n++] = 1;
    }
    for (int i = 0; i < n; ++i) {
        const int b = slot_of[i];
        const int k = b >= 0 && b < m ? (int) keep[b] : 0;
        if (k < 1 || k > in_window[i]) {
            err = bad + "slot " + std::to_string(b) + " keeps " + std::to_string(k) + " of its " +
                  std::to_string(in_window[i]) + " rows (1 to all of them)";
            keep_by_slot.assign(keep_by_slot.size(), 0);
            return false;
        }
        keep_by_slot[(size_t) b] = k;
    }
    for (int s = 0; s < m; ++s) {
        if (keep[s] == 0) continue;
        bool in = false;
        for (int i = 0; i < n; ++i) in = in || slot_of[i] == s;
        if (!in) {
            err = bad + "slot " + std::to_string(s) + " keeps " + std::to_string((int) keep[s]) +
                  " rows of a window it was not in";
            keep_by_slot.assign(keep_by_slot.size(), 0);
            return false;
        }
    }
    return true;
}
// ==== end of 担当 F's bodies ====

}  // namespace strata::net
