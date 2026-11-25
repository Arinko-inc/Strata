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
// 担当 S writes this file (generate.cpp keeps only the #include).  Without the flag it must set nothing, so the worker
// runs as before; with it (and --batch) it wraps the handlers above and sets hs.slot_commit_deferred.
static_assert(strata::net::kDeferredMaxRows == strata::kernels::kVerifyMaxT, "a deferred window's rows: kVerifyMaxT");
const bool wk_dfr_asked = strata::net::wk_piped_mtp_on(std::getenv("STRATA_WK_PIPED_MTP"));
if (wk_dfr_asked && o.batch > 0)
    std::fprintf(stderr, "strata serve: stage worker: STRATA_WK_PIPED_MTP: this build keeps no deferred windows yet "
                         "(L1 skeleton): off\n");
