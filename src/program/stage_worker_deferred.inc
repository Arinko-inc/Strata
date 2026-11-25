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
// 担当 F writes this file (generate.cpp keeps only the #include).  Without the flag it must set nothing, so the worker
// runs as before; with it (and --batch) it wraps the handlers above and sets hs.slot_commit_deferred.
static_assert(strata::net::kDeferredMaxRows == strata::kernels::kVerifyMaxT, "a deferred window's rows: kVerifyMaxT");
const bool wk_dfr_asked = strata::net::wk_piped_mtp_on(std::getenv("STRATA_WK_PIPED_MTP"));
if (wk_dfr_asked && o.batch > 0)
    std::fprintf(stderr, "strata serve: stage worker: STRATA_WK_PIPED_MTP: this build keeps no deferred windows yet "
                         "(L1 skeleton): off\n");
