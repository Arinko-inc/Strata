// sycl/src/program/stage_worker_deferred.inc - L1 (Arinko-inc): the SYCL remote stage worker's deferred windows (the
// B70 end; STRATA_WK_PIPED_MTP=1; strata-mesh IMPL-PLAN-2026-11-26-batch-drafts.md 3.5, and its section after the
// skeleton).
//
// Not a translation unit: sycl/src/program/generate.cpp includes it in the stage worker's REMOTE-BATCH WORKER (C)
// block, after `if (o.batch > 0) { ... }`.  Every slot handler is set by then (hs.hello, hs.disconnected, hs.slot_run,
// hs.slot_commit, hs.slot_save, hs.slot_load; hs.reset and hs.prefill from before the block), and this runs once, at
// the worker's start, at that block's level: what it declares lives as long as the handlers do (capture it by
// reference, as the block does).  The C block's state and helpers are in scope (ver, bslot_ss, wk_cap, wk_slot_n,
// wk_slot_live, wk_slot_len, wk_win_S, wk_max_s, slot_ok, sync_q, refill, drive, win_pool_fn, wk_outv, wk_timing ...).
// The end has no next worker: a deferred window is answered at once (serve_stage has no slot_run_reply here).
//
// Without STRATA_WK_PIPED_MTP=1 (or without --batch) this sets nothing: no allocation, no handler wrapped, no hello
// bit - the worker runs as before.  With it: the saved rows (Verifier::init_deferred_commit: this stage's layers only,
// wk_cap * STRATA_WK_PIPED_MTP_ROWS rows, default 2 a slot: its token and one draft; 合流 D), the batch graph limit,
// and the handlers below.  A deferred window with more rows on one slot is refused (an error for that window; the
// link goes on).  STRATA_WK_PIPED_MTP_SELFTEST=1 saves wk_max_s rows a slot (its windows have 3 rows).
//
// A deferred window (SlotRun with kStageSlotDeferred) runs its layers now with no commit; right behind it the rows its
// commit replays go to the saved rows of its first slot; its rows are answered at once.  The main process's
// SlotCommit (kStageCommitDeferred) comes later, by tag, and replays the accepted prefixes from the saved rows - other
// deferred windows, on other slots, may run in between.
//
// A deferred window is keyed here by a count of this worker's own, not by the main process's tag: StageHandlers::slot_run
// does not carry the SlotRun's tag (only slot_commit_deferred gets the commit's).  So the commit finds its window by
// the slots it keeps (a held window's slots are in no other held window: DeferredWindows::add), and keep_ok checks that
// the commit keeps exactly that window's slots and no others.  The tag itself is not checked here.
static_assert(strata::net::kDeferredMaxRows == strata::kernels::kVerifyMaxT, "a deferred window's rows: kVerifyMaxT");
const bool wk_dfr_asked = strata::net::wk_piped_mtp_on(std::getenv("STRATA_WK_PIPED_MTP"));
// the saved rows of one slot (the most rows one slot may have in a deferred window; the self-test's windows: wk_max_s)
const int wk_dfr_slot_rows = strata::net::wk_piped_mtp_on(std::getenv("STRATA_WK_PIPED_MTP_SELFTEST"))
    ? std::min(wk_max_s, (int) strata::net::kDeferredMaxRows)
    : strata::net::wk_piped_mtp_rows(std::getenv("STRATA_WK_PIPED_MTP_ROWS"), wk_max_s);
strata::net::DeferredWindows wk_dfr;        // the windows held uncommitted
std::vector<int64_t> wk_dfr_of_slot;        // slot -> the count of the held window that has a row on it (0: none)
int64_t wk_dfr_seq = 0;                     // this worker's own count of deferred windows
double wk_dfr_ms = 0;
int64_t wk_dfr_windows = 0, wk_dfr_rows = 0;
bool wk_dfr_on = false;
if (wk_dfr_asked && o.batch > 0 && wk_cap > 0) {
    std::string de;
    if (!ver.init_deferred_commit(wk_cap * wk_dfr_slot_rows, de)) {
        std::fprintf(stderr, "strata serve: stage worker: STRATA_WK_PIPED_MTP: %s: off\n", de.c_str());
    } else {
        ver.set_batch_graph_limit(64);   // slot rotation makes many window layouts: LRU, as --batch-mtp does
        wk_dfr_on = true;
        wk_dfr_of_slot.assign((size_t) wk_cap, 0);
        std::fprintf(stderr, "strata serve: stage worker: STRATA_WK_PIPED_MTP: deferred windows on (%d slots x %d rows "
                             "saved (STRATA_WK_PIPED_MTP_ROWS), graph limit 64)\n", wk_cap, wk_dfr_slot_rows);
    }
} else if (wk_dfr_asked) {
    std::fprintf(stderr, "strata serve: stage worker: STRATA_WK_PIPED_MTP: needs --batch: off\n");
}
if (wk_dfr_on) {
    auto dfr_forget = [&] {   // the hello and a disconnect: nothing held (the saved rows are just not read again)
        wk_dfr.reset(wk_slot_n, wk_dfr_slot_rows);
        std::fill(wk_dfr_of_slot.begin(), wk_dfr_of_slot.end(), 0);
    };
    auto dfr_held_error = [&](const char* what, std::string& e) {
        e = std::string("stage worker: ") + what + " while " + std::to_string(wk_dfr.size()) +
            " deferred window(s) wait for their commit (the main process commits them first)";
    };
    auto hello_prev = hs.hello;
    hs.hello = [&, hello_prev, dfr_forget](const strata::net::StageHello& theirs, strata::net::StageHello& mine,
                               std::string& e) -> bool {
        if (!hello_prev(theirs, mine, e)) return false;
        mine.batch_flags |= strata::net::kStageBatchDeferred;
        dfr_forget();
        return true;
    };
    auto disconnected_prev = hs.disconnected;
    hs.disconnected = [&, disconnected_prev, dfr_forget] {
        if (disconnected_prev) disconnected_prev();
        dfr_forget();
    };
    auto slot_run_prev = hs.slot_run;
    hs.slot_run = [&, slot_run_prev, dfr_held_error](int S, const int32_t* slots, const int32_t* tokens, const int64_t* pos,
                                     int64_t flags, std::string& e) -> bool {
        if ((flags & strata::net::kStageSlotDeferred) == 0) {
            if (!wk_dfr.empty()) {   // (the main process commits its deferred windows before any other window)
                dfr_held_error("a batch window", e);
                return false;
            }
            return slot_run_prev(S, slots, tokens, pos, flags, e);
        }
        const WkBusy busy("stage worker deferred window");
        wk_win_S = 0;   // a window that fails leaves nothing for a SlotCommit
        if (flags != strata::net::kStageSlotDeferred) {
            e = "stage worker: a deferred batch window's flags " + std::to_string(flags) + " are not known here";
            return false;
        }
        if (S < 1 || S > wk_max_s) {
            e = "stage worker: a batch window of " + std::to_string(S) + " rows (this worker's windows hold 1-" +
                std::to_string(wk_max_s) + ")";
            return false;
        }
        // the slots: as an ordinary window's (agreed, saved into, each one's rows together, at its length on)
        for (int t = 0; t < S; ++t) {
            const int b = slots[t];
            if (!slot_ok(b, true, e)) return false;
            if (t > 0 && slots[t - 1] == b) {
                if (pos[t] != pos[t - 1] + 1) {
                    e = "stage worker: slot " + std::to_string(b) + "'s rows are not at consecutive positions";
                    return false;
                }
                continue;
            }
            for (int u = 0; u < t; ++u)
                if (slots[u] == b) {
                    e = "stage worker: slot " + std::to_string(b) + "'s rows are not together in the window";
                    return false;
                }
            if (pos[t] != wk_slot_len[(size_t) b]) {
                e = "stage worker: slot " + std::to_string(b) + " holds " + std::to_string(wk_slot_len[(size_t) b]) +
                    " cells, the window's row is at " + std::to_string(pos[t]);
                return false;
            }
        }
        const auto t0 = Clock::now();
        // held from here: a slot in another held window, or a window's saved rows already taken, is refused before it runs
        strata::net::DeferredWindow w;
        w.tag = ++wk_dfr_seq;
        w.S = S;
        for (int t = 0; t < S; ++t) {
            w.rows[t] = slots[t];
            w.tok[t] = tokens[t];
            w.pos[t] = pos[t];
        }
        if (!wk_dfr.add(w, e)) return false;
        auto drop = [&] {
            strata::net::DeferredWindow gone;
            std::string ignored;
            (void) wk_dfr.take(w.tag, gone, ignored);
        };
        // a prompt's loan goes back before a window reads the cache, as the ordinary window does
        if (!refill(e)) { drop(); return false; }
        int rows[strata::kernels::kVerifyMaxT] = {};
        for (int t = 0; t < S; ++t) rows[t] = slots[t];
        drive.d.layers = 0;
        drive.d.experts = 0;
        drive.d.failed = false;
        if (!ver.run_slot_rows_deferred(rows, S, w.save_base, tokens, pos, win_pool_fn, win_pool_user, wk_outv.data(),
                                        e) ||
            drive.d.failed) {
            if (drive.d.failed && drive.d.fail) e = drive.d.fail;
            drop();
            return false;
        }
        for (int t = 0; t < S; ++t) wk_dfr_of_slot[(size_t) slots[t]] = w.tag;
        if (wk_timing) {
            wk_dfr_ms += std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
            wk_dfr_rows += S;
            if (++wk_dfr_windows % 64 == 0) {
                std::fprintf(stderr, "strata stage worker: deferred batch windows: own layers %.2f ms each (64 windows, "
                                     "%.1f rows each, %d held now)\n", wk_dfr_ms / 64, (double) wk_dfr_rows / 64,
                             wk_dfr.size());
                wk_dfr_ms = 0;
                wk_dfr_rows = 0;
            }
        }
        return true;
    };
    hs.slot_commit_deferred = [&](const int32_t* keep, int m, int64_t tag, std::string& e) -> bool {
        const WkBusy busy("stage worker deferred commit");
        if (wk_slot_n <= 0) {
            e = "stage worker: this main process agreed no batch slots (its --batch or this worker's is 0)";
            return false;
        }
        if (m < 1 || m > wk_slot_n) {
            e = "stage worker: a slot commit of " + std::to_string(m) + " slots (" + std::to_string(wk_slot_n) +
                " agreed)";
            return false;
        }
        // the window: the one that has a row on the first slot the commit keeps (the commit's tag is the main
        // process's; this worker's windows are counted by itself, see the top of this file)
        int first = -1;
        for (int s = 0; s < m; ++s)
            if (keep[s] != 0) { first = s; break; }
        const int64_t mine_id = first >= 0 ? wk_dfr_of_slot[(size_t) first] : 0;
        strata::net::DeferredWindow w;
        if (first < 0 || mine_id == 0 || !wk_dfr.take(mine_id, w, e)) {
            e = "stage worker: a deferred commit (tag " + std::to_string((long long) tag) + ") for " +
                (first < 0 ? std::string("no slot") : "slot " + std::to_string(first)) +
                ", which no deferred window holds";
            return false;
        }
        for (int t = 0; t < w.S; ++t) wk_dfr_of_slot[(size_t) w.rows[t]] = 0;
        std::vector<int> keep_by_slot;
        if (!strata::net::DeferredWindows::keep_ok(w, keep, m, wk_slot_n, keep_by_slot, e)) return false;
        if (!ver.batch_commit_rows(w.rows, w.S, w.save_base, keep_by_slot.data(), w.tok, w.pos, e)) return false;
        // the accepted rows are in the slots' length now (the commit itself is queued behind the window on the GPU)
        for (int t = 0; t < w.S; ++t)
            if (t == 0 || w.rows[t] != w.rows[t - 1]) wk_slot_len[(size_t) w.rows[t]] += keep_by_slot[(size_t) w.rows[t]];
        return true;
    };
    // everything else that touches the slots' state or the window scratch is refused while a window is held
    auto slot_commit_prev = hs.slot_commit;
    hs.slot_commit = [&, slot_commit_prev, dfr_held_error](const int32_t* keep, int n, std::string& e) -> bool {
        if (!wk_dfr.empty()) { dfr_held_error("a slot commit (without kStageCommitDeferred)", e); return false; }
        return slot_commit_prev(keep, n, e);
    };
    auto slot_save_prev = hs.slot_save;
    hs.slot_save = [&, slot_save_prev, dfr_held_error](int slot, int64_t L, std::string& e) -> bool {
        if (!wk_dfr.empty()) { dfr_held_error("a slot save", e); return false; }
        return slot_save_prev(slot, L, e);
    };
    auto slot_load_prev = hs.slot_load;
    hs.slot_load = [&, slot_load_prev, dfr_held_error](int slot, int64_t L, int64_t ckpt_id, std::string& e) -> bool {
        if (!wk_dfr.empty()) { dfr_held_error("a slot load", e); return false; }
        return slot_load_prev(slot, L, ckpt_id, e);
    };
    auto reset_prev = hs.reset;
    hs.reset = [&, reset_prev, dfr_held_error](const std::vector<int64_t>& keep, std::string& e) -> bool {
        if (!wk_dfr.empty()) { dfr_held_error("a reset", e); return false; }
        return reset_prev(keep, e);
    };
    if (hs.prefill) {
        auto prefill_prev = hs.prefill;
        hs.prefill = [&, prefill_prev, dfr_held_error](const int64_t* tokens, int64_t T, int64_t p0, int64_t flags, int64_t skip,
                                       const float* rows_in, float* rows_out, std::string& e) -> bool {
            if (!wk_dfr.empty()) { dfr_held_error("a prompt chunk", e); return false; }
            return prefill_prev(tokens, T, p0, flags, skip, rows_in, rows_out, e);
        };
    }
    // STRATA_WK_PIPED_MTP_SELFTEST=1 (a test run: the process exits when it is done, it never serves)
    if (strata::net::wk_piped_mtp_on(std::getenv("STRATA_WK_PIPED_MTP_SELFTEST"))) {
#include "stage_worker_deferred_selftest.inc"
    }
}
