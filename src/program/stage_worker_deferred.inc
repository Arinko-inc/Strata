// src/program/stage_worker_deferred.inc - L1 (Arinko-inc): a CUDA / HIP remote stage worker's deferred windows
// (STRATA_WK_PIPED_MTP=1; strata-mesh IMPL-PLAN-2026-11-26-batch-drafts.md 3.4, and its section after the skeleton).
//
// Not a translation unit: src/program/generate.cpp includes it in the stage worker's REMOTE-BATCH WORKER (B) block,
// after `if (o.batch > 0) { ... }`.  Every slot handler is set by then (hs.hello, hs.disconnected, hs.slot_run,
// hs.slot_run_reply on a relay, hs.slot_commit, hs.slot_save, hs.slot_load; hs.reset and hs.prefill from before the
// block), and this runs once, at the worker's start, at that block's level: what it declares lives as long as the
// handlers do (capture it by reference, as the block does).  The B block's state and helpers are in scope (ver,
// bslot_ss, wk_slot_n, wk_slot_len, slot_ok, wk_after_commit, wk_late_q / wk_late_mu / WkLate, wk_next_tag, wk_overlap,
// next_link, relay, stage_relay, next_batch_flags, refill, apply_pending, drive, win_pool_fn, wk_outv, wk_relay, HBF,
// wk_timing ...).
//
// 担当 F's file (generate.cpp keeps only the #include).  Without the flag (or without --batch) it sets nothing: every
// handler stays the B block's and the worker runs as before.  With both it allocates the saved rows of this worker's
// layers (Verifier::init_deferred_commit, n_slots * max_t rows), wraps hs.hello / hs.disconnected / hs.slot_run /
// hs.slot_commit / hs.slot_save / hs.slot_load / hs.reset / hs.prefill and sets hs.slot_commit_deferred.
//
// Which held window a deferred SlotCommit commits: hs.slot_run is not given the SlotRun's tag (StageHandlers' signature
// has no tag), so this worker cannot match the commit's tag against the window's.  It finds the window from keep[]
// instead: the held windows are on separate slots (DeferredWindows::add), and a commit keeps 1.. rows on each of its
// window's slots and 0 on every other (DeferredWindows::keep_ok checks both), so the slots that keep rows name exactly
// one held window.  The table's own tags are this worker's numbers (wk_dfr_seq).  A relay sends the window on with its
// own tag (wk_next_tag) and names that tag in the commit it sends on, as the protocol says.
static_assert(strata::net::kDeferredMaxRows == strata::kernels::kVerifyMaxT, "a deferred window's rows: kVerifyMaxT");
const bool wk_dfr_asked = strata::net::wk_piped_mtp_on(std::getenv("STRATA_WK_PIPED_MTP"));
strata::net::DeferredWindows wk_dfr;   // per connection: the windows held uncommitted
bool wk_dfr_on = false;                // the saved rows are allocated: this worker can keep deferred windows
bool wk_dfr_link = false;              // this link's hello said kStageBatchDeferred
int64_t wk_dfr_seq = 0;                // the table's tags (this worker's own numbers)
std::vector<int64_t> wk_dfr_slot_tag((size_t) std::max(o.batch, 0), 0);   // per slot: its held window's, 0: none
int64_t wk_dfr_windows = 0, wk_dfr_commits = 0, wk_dfr_rows = 0, wk_dfr_kept = 0, wk_dfr_held_sum = 0;
int wk_dfr_held_max = 0;
double wk_dfr_run_ms = 0, wk_dfr_commit_ms = 0;
if (wk_dfr_asked && o.batch <= 0)
    std::fprintf(stderr, "strata serve: stage worker: STRATA_WK_PIPED_MTP needs --batch (the batch slots): off\n");
if (wk_dfr_asked && o.batch > 0) {
    if (stage_relay && !(wk_overlap && hs.slot_run_reply != nullptr)) {
        std::fprintf(stderr, "strata serve: stage worker: STRATA_WK_PIPED_MTP is off: a relay answers the deferred "
                             "windows late, which STRATA_RELAY_SLOT_OVERLAP=0 turns off\n");
    } else {
        size_t dfr_free0 = 0, dfr_free1 = 0, dfr_total = 0;
        (void) cudaMemGetInfo(&dfr_free0, &dfr_total);
        std::string de;
        if (!ver.init_deferred_commit(ver.n_slots() * ver.max_t(), de)) {
            (void) cudaGetLastError();
            std::fprintf(stderr, "strata serve: stage worker: STRATA_WK_PIPED_MTP is off: %s (VRAM free %.1f MiB)\n",
                         de.c_str(), (double) dfr_free0 / 1048576.0);
        } else {
            ver.set_batch_graph_limit(64);
            wk_dfr_on = true;
            (void) cudaMemGetInfo(&dfr_free1, &dfr_total);
            std::fprintf(stderr, "strata serve: stage worker: STRATA_WK_PIPED_MTP=1: deferred batch windows (%d slots x "
                                 "%d rows of saved rows, %.1f MiB of VRAM: free %.1f -> %.1f MiB; up to 64 window "
                                 "layouts keep their graphs)\n", ver.n_slots(), ver.max_t(),
                         (double) (dfr_free0 > dfr_free1 ? dfr_free0 - dfr_free1 : 0) / 1048576.0,
                         (double) dfr_free0 / 1048576.0, (double) dfr_free1 / 1048576.0);
        }
    }
}
if (wk_dfr_on) {
    auto dfr_forget = [&] {
        if (!wk_dfr.empty())
            std::fprintf(stderr, "strata stage worker: %d deferred batch windows were never committed (dropped with "
                                 "the link's slots)\n", wk_dfr.size());
        wk_dfr.reset(0, 0);
        wk_dfr_link = false;
        wk_dfr_seq = 0;
        std::fill(wk_dfr_slot_tag.begin(), wk_dfr_slot_tag.end(), 0);
    };
    // what a message that needs every window committed gets while some are held
    auto dfr_held = [&](const char* what, std::string& e) -> bool {
        if (wk_dfr.empty()) return false;
        e = std::string("stage worker: bad slot message: ") + what + " while " + std::to_string(wk_dfr.size()) +
            " deferred batch windows wait for their SlotCommit";
        return true;
    };
    // the hello: the bit (a relay: only when its next worker's hello has it), and an empty table of this link's slots
    auto dfr_hello = hs.hello;
    hs.hello = [&, dfr_hello, dfr_forget](const strata::net::StageHello& theirs, strata::net::StageHello& mine,
                                          std::string& e) -> bool {
        if (!dfr_hello(theirs, mine, e)) return false;
        dfr_forget();
        if (theirs.protocol != strata::net::kStageProtocol) return true;
        const bool next_has = !stage_relay || (next_batch_flags & strata::net::kStageBatchDeferred) != 0;
        wk_dfr_link = wk_slot_n > 0 && wk_slot_n <= ver.n_slots() && next_has;
        if (wk_dfr_link) {
            mine.batch_flags |= strata::net::kStageBatchDeferred;
            wk_dfr.reset(wk_slot_n, ver.max_t());
        }
        std::fprintf(stderr, "strata stage worker: deferred batch windows: %s\n",
                     wk_dfr_link ? "this link takes them (kStageBatchDeferred)"
                                 : !next_has ? "off on this link: the next worker's hello has no kStageBatchDeferred"
                                             : "off on this link: it agreed no batch slots");
        return true;
    };
    auto dfr_disconnected = hs.disconnected;
    hs.disconnected = [&, dfr_disconnected, dfr_forget] {
        if (dfr_disconnected) dfr_disconnected();
        dfr_forget();
    };
    // a reset: nothing held, and a relay's next worker (linked again meanwhile, maybe) still takes deferred windows
    auto dfr_reset = hs.reset;
    hs.reset = [&, dfr_reset, dfr_held](const std::vector<int64_t>& keep, std::string& e) -> bool {
        if (dfr_held("a reset", e)) return false;
        if (!dfr_reset(keep, e)) return false;
        if (stage_relay && wk_dfr_link && (next_batch_flags & strata::net::kStageBatchDeferred) == 0) {
            e = "relay: the next worker no longer keeps deferred batch windows (kStageBatchDeferred), which this link "
                "agreed (start the main process again)";
            return false;
        }
        return true;
    };
    auto dfr_prefill = hs.prefill;
    hs.prefill = [&, dfr_prefill, dfr_held](const int64_t* tokens, int64_t T, int64_t p0, int64_t flags, int64_t skip,
                                            const float* rows_in, float* rows_out, std::string& e) -> bool {
        if (dfr_held("a prompt chunk", e)) return false;
        return dfr_prefill(tokens, T, p0, flags, skip, rows_in, rows_out, e);
    };
    auto dfr_slot_commit = hs.slot_commit;
    hs.slot_commit = [&, dfr_slot_commit, dfr_held](const int32_t* keep, int m, std::string& e) -> bool {
        if (dfr_held("a SlotCommit without kStageCommitDeferred", e)) return false;
        return dfr_slot_commit(keep, m, e);
    };
    auto dfr_slot_save = hs.slot_save;
    hs.slot_save = [&, dfr_slot_save, dfr_held](int b, int64_t L, std::string& e) -> bool {
        if (dfr_held("a slot save", e)) return false;
        return dfr_slot_save(b, L, e);
    };
    auto dfr_slot_load = hs.slot_load;
    hs.slot_load = [&, dfr_slot_load, dfr_held](int b, int64_t L, int64_t ckpt_id, std::string& e) -> bool {
        if (dfr_held("a slot load", e)) return false;
        return dfr_slot_load(b, L, ckpt_id, e);
    };
    // a window: kStageSlotDeferred here; any other one only with nothing held (then the B block's handler, as before)
    auto dfr_slot_run = hs.slot_run;
    hs.slot_run = [&, dfr_slot_run, dfr_held](int S, const int32_t* slots, const int32_t* tokens, const int64_t* pos,
                                              int64_t flags, std::string& e) -> bool {
        if (flags != strata::net::kStageSlotDeferred) {
            if (dfr_held("a batch window without kStageSlotDeferred", e)) return false;
            return dfr_slot_run(S, slots, tokens, pos, flags, e);
        }
        const WkBusy busy("stage worker deferred batch window");
        const auto t0 = Clock::now();
        if (!wk_dfr_link) {
            e = "stage worker: bad slot window: this link agreed no deferred windows (kStageBatchDeferred)";
            return false;
        }
        if (wk_slot_wait) {
            e = "stage worker: bad slot window: it came before the last window's SlotCommit";
            return false;
        }
        // (the late replies' count, as the B block's: whether the reply before this window is still being read)
        bool late_early = false;
        if (stage_relay) {
            std::lock_guard<std::mutex> lk(wk_late_mu);
            late_early = !wk_late_q.empty();
        }
        if (S < 1 || S > ver.max_t() || S > (int) strata::kernels::kVerifyMaxT) {
            e = "stage worker: bad slot window: " + std::to_string(S) + " rows (this worker's windows hold " +
                std::to_string(ver.max_t()) + "), flags " + std::to_string((long long) flags);
            return false;
        }
        // each slot's rows together, at its cells: its first row where the slot is, the rest after it (the B block's
        // checks; a slot in a held window is refused by the table below, so its length is its committed one)
        strata::net::DeferredWindow w;
        w.S = S;
        for (int t = 0; t < S; ++t) {
            const int b = slots[t];
            if (!slot_ok(b, true, e)) return false;
            w.rows[t] = b;
            w.tok[t] = tokens[t];
            w.pos[t] = pos[t];
            if (t > 0 && slots[t - 1] == b) {
                if (pos[t] != pos[t - 1] + 1) {
                    e = "stage worker: bad slot window: slot " + std::to_string(b) + "'s rows are not at consecutive cells";
                    return false;
                }
                continue;
            }
            for (int u = 0; u < t; ++u)
                if (slots[u] == b) {
                    e = "stage worker: bad slot window: slot " + std::to_string(b) + "'s rows are not together";
                    return false;
                }
            if (wk_dfr.slot_held(b)) {
                e = "stage worker: bad slot window: slot " + std::to_string(b) +
                    " is in a deferred window that waits for its SlotCommit";
                return false;
            }
            if (pos[t] != wk_slot_len[(size_t) b]) {
                e = "stage worker: bad slot window: slot " + std::to_string(b) + " holds " +
                    std::to_string((long long) wk_slot_len[(size_t) b]) + " cells, its row is at " +
                    std::to_string((long long) pos[t]);
                return false;
            }
        }
        if (stage_relay && !relay.wait(e)) {
            e = "relay: the next worker: " + e;
            return false;
        }
        w.tag = ++wk_dfr_seq;
        if (stage_relay) w.next_tag = wk_next_tag + 1;   // (the tag it goes on with, below: the commit names it there)
        if (!wk_dfr.add(w, e)) return false;   // (sets w.save_base)
        // from here a failure lets the window go: nothing of it is committed, and nothing went on
        auto drop = [&] {
            strata::net::DeferredWindow gone;
            std::string ge;
            (void) wk_dfr.take(w.tag, gone, ge);
        };
        // as the B block's window: a prompt's loan goes back first, then the adaptive tier's swaps land
        if (!refill(e)) { drop(); return false; }
        if (adapt_nowait()) apply_pending(false);
        else if (aov && !pending.empty()) apply_pending(true);
        else if (pending.empty() || ++pending_age >= adapt_lag()) apply_pending(true);
        drive.d.layers = 0;
        drive.d.experts = 0;
        drive.d.failed = false;
        if (!ver.run_slot_rows_deferred(w.rows, S, w.save_base, tokens, pos, win_pool_fn, win_pool_user, wk_outv.data(),
                                        e) ||
            drive.d.failed) {
            if (drive.d.failed && drive.d.fail) e = drive.d.fail;
            drop();
            return false;
        }
        const auto t1 = Clock::now();
        if (stage_relay) {
            // the next worker runs the rest of the window and holds it too; its rows are this window's late reply
            // (hs.slot_run_reply, on the replying thread, in the order the windows came)
            const NextWorkerWait wait("stage worker: the next worker's batch window");
            const int64_t tag = ++wk_next_tag;
            if (!next_link.slot_run_send(S, slots, tokens, pos, strata::net::kStageSlotDeferred, tag, wk_relay,
                                         (size_t) S * HBF, e)) {
                e = "relay: the next worker: " + e;
                drop();
                return false;
            }
            WkLate lw;
            lw.tag = tag;
            lw.S = S;
            lw.sent = t1;
            lw.back = Clock::now();
            lw.own_ms = std::chrono::duration<double, std::milli>(lw.back - t0).count();
            lw.early = late_early;
            std::lock_guard<std::mutex> lk(wk_late_mu);
            wk_late_q.push_back(std::move(lw));
        }
        for (int t = 0; t < S; ++t) wk_dfr_slot_tag[(size_t) w.rows[t]] = w.tag;
        ++wk_dfr_windows;
        wk_dfr_rows += S;
        wk_dfr_held_sum += wk_dfr.size();
        wk_dfr_held_max = std::max(wk_dfr_held_max, wk_dfr.size());
        wk_dfr_run_ms += std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
        return true;
    };
    // the commit of one held window (the one on the slots that keep rows: see the top of this file).  A relay sends
    // it on over the next link's sending side only (the replying thread reads that link), then commits its own rows
    // from the saved ones - queued on the verifier's stream, not waited for.
    hs.slot_commit_deferred = [&](const int32_t* keep, int m, int64_t tag, std::string& e) -> bool {
        const WkBusy busy("stage worker deferred batch commit");
        const auto t0 = Clock::now();
        if (wk_slot_n <= 0) return slot_ok(0, false, e);
        if (!wk_dfr_link) {
            e = "stage worker: bad slot commit: this link agreed no deferred windows (kStageBatchDeferred)";
            return false;
        }
        if (m < 1 || m > wk_slot_n) {
            e = "stage worker: bad slot commit: " + std::to_string(m) + " slots, this link agreed " +
                std::to_string(wk_slot_n);
            return false;
        }
        int64_t held_tag = 0;
        for (int s = 0; s < m && held_tag == 0; ++s)
            if (keep[s] != 0) {
                held_tag = wk_dfr_slot_tag[(size_t) s];
                if (held_tag == 0) {
                    e = "stage worker: bad slot commit (deferred, tag " + std::to_string((long long) tag) + "): slot " +
                        std::to_string(s) + " keeps " + std::to_string((int) keep[s]) +
                        " rows, and no deferred window holds it";
                    return false;
                }
            }
        if (held_tag == 0) {
            e = "stage worker: bad slot commit (deferred, tag " + std::to_string((long long) tag) +
                "): no slot keeps a row";
            return false;
        }
        strata::net::DeferredWindow w;
        if (!wk_dfr.take(held_tag, w, e)) return false;
        for (int t = 0; t < w.S; ++t) wk_dfr_slot_tag[(size_t) w.rows[t]] = 0;
        std::vector<int> k;
        if (!strata::net::DeferredWindows::keep_ok(w, keep, m, ver.n_slots(), k, e)) {
            e += " (tag " + std::to_string((long long) tag) + ": the window is dropped uncommitted)";
            return false;
        }
        if (stage_relay) {   // the next worker commits meanwhile (one way; nothing is read from it here)
            const NextWorkerWait wait("stage worker: the next worker's batch commit");
            if (!next_link.slot_commit(keep, m, w.next_tag, e, true)) {
                e = "relay: the next worker: " + e;
                return false;
            }
        }
        if (!ver.batch_commit_rows(w.rows, w.S, w.save_base, k.data(), w.tok, w.pos, e)) return false;
        int kept = 0;
        for (int t = 0; t < w.S; ++t)
            if (t == 0 || w.rows[t] != w.rows[t - 1]) {
                wk_slot_len[(size_t) w.rows[t]] += k[(size_t) w.rows[t]];
                kept += k[(size_t) w.rows[t]];
            }
        if (!wk_after_commit(e)) return false;
        wk_dfr_kept += kept;
        wk_dfr_commit_ms += std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
        if (++wk_dfr_commits % 64 == 0 && wk_timing) {
            std::fprintf(stderr, "strata stage worker: deferred commits: 64 commits kept %lld rows; since the last line "
                                 "%lld deferred windows, %.2f rows each, own layers and the send %.2f ms each, a commit "
                                 "%.3f ms; %.2f windows held after each window (the most %d)\n",
                         (long long) wk_dfr_kept, (long long) wk_dfr_windows,
                         wk_dfr_windows > 0 ? (double) wk_dfr_rows / (double) wk_dfr_windows : 0.0,
                         wk_dfr_windows > 0 ? wk_dfr_run_ms / (double) wk_dfr_windows : 0.0, wk_dfr_commit_ms / 64,
                         wk_dfr_windows > 0 ? (double) wk_dfr_held_sum / (double) wk_dfr_windows : 0.0, wk_dfr_held_max);
            wk_dfr_windows = wk_dfr_rows = wk_dfr_kept = wk_dfr_held_sum = 0;
            wk_dfr_held_max = 0;
            wk_dfr_run_ms = wk_dfr_commit_ms = 0;
        }
        return true;
    };
}
