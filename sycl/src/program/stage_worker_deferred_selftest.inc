// sycl/src/program/stage_worker_deferred_selftest.inc - L1 (Arinko-inc): the GPU check of the end's deferred windows.
// Not a translation unit: stage_worker_deferred.inc includes it inside `if (STRATA_WK_PIPED_MTP_SELFTEST=1)`, at the
// worker's start (the model loaded, the slot sessions carved, the saved rows allocated), and the process exits when it
// is done.  The same window - the same hand-off rows, the same positions - is run two ways on two slots that hold
// nothing:
//   reference: Verifier::run_slot_rows + commit_slot_prefixes (the commit graph right behind the window), slot R
//   deferred:  Verifier::run_slot_rows_deferred (save graph), another deferred window on another slot, then
//              Verifier::batch_commit_rows (the commit from the saved rows), slot D
// and the two slots' state (every GDN layer's recurrence and conv history, every QSA layer's indexer tail / dead /
// block positions, the PLE history) is read back through conversation_checkpoint_save and compared byte for byte.
// Two pairs: 3 rows keeping 2 (slots 0 and 1) and 2 rows keeping 1 (slots 2 and 3); the two deferred windows are
// committed in the order they ran, with the other one's window run between its save and its commit.  The state must
// not be all zero (a test that compares two zero states proves nothing).  Needs 4 slots and windows of 3 rows.
//
// It writes the slots' sessions (a worker whose slots are used afterwards is a worker that was served: SlotSave
// rewrites a slot completely), and nothing else: this process exits.  Run it only as a test: B70 is a production GPU.
{
    std::string st_e;
    const int st_ns = ver.n_slots(), st_mt = wk_max_s;
    if (st_ns < 4 || st_mt < 3) {
        std::fprintf(stderr, "strata stage worker: SELFTEST: needs 4 slots and windows of 3 rows (have %d and %d): "
                             "skipped\n", st_ns, st_mt);
        std::exit(2);
    }
    const size_t st_hf = (size_t) HBF;
    // one window's rows on `slot` (rows [0, S) of the hand-off, the same numbers every time)
    auto st_fill = [&](int S) {
        for (int t = 0; t < S; ++t)
            for (size_t i = 0; i < st_hf; ++i)
                wk_in[(size_t) t * st_hf + i] = 0.05f * std::sin(0.0007f * (float) (i + 1) * (float) (t + 2));
    };
    auto st_window = [&](bool deferred, int slot, int S, std::string& e) -> bool {
        int rows[strata::kernels::kVerifyMaxT] = {};
        int32_t tok[strata::kernels::kVerifyMaxT] = {};
        int64_t pos[strata::kernels::kVerifyMaxT] = {};
        for (int t = 0; t < S; ++t) {
            rows[t] = slot;
            tok[t] = 1000 + 7 * t;
            pos[t] = t;
        }
        st_fill(S);
        if (!refill(e)) return false;
        drive.d.layers = 0;
        drive.d.experts = 0;
        drive.d.failed = false;
        const bool ok = deferred ? ver.run_slot_rows_deferred(rows, S, slot * st_mt, tok, pos, win_pool_fn, win_pool_user,
                                                              wk_outv.data(), e)
                                 : ver.run_slot_rows(rows, S, tok, pos, win_pool_fn, win_pool_user, wk_outv.data(), e);
        if (drive.d.failed) {
            if (drive.d.fail) e = drive.d.fail;
            return false;
        }
        return ok;
    };
    auto st_commit_ref = [&](int slot, int keep_n, std::string& e) -> bool {
        std::vector<int> keep((size_t) st_ns, 0);
        keep[(size_t) slot] = keep_n;
        return ver.commit_slot_prefixes(keep.data(), e) && ver.wait_commit(e);
    };
    auto st_commit_def = [&](int slot, int S, int keep_n, std::string& e) -> bool {
        int rows[strata::kernels::kVerifyMaxT] = {};
        int32_t tok[strata::kernels::kVerifyMaxT] = {};
        int64_t pos[strata::kernels::kVerifyMaxT] = {};
        for (int t = 0; t < S; ++t) {
            rows[t] = slot;
            tok[t] = 1000 + 7 * t;
            pos[t] = t;
        }
        std::vector<int> keep((size_t) st_ns, 0);
        keep[(size_t) slot] = keep_n;
        return ver.batch_commit_rows(rows, S, slot * st_mt, keep.data(), tok, pos, e);
    };
    bool st_ok = st_window(false, 0, 3, st_e) && st_commit_ref(0, 2, st_e) &&   // reference, pair 1
                 st_window(false, 2, 2, st_e) && st_commit_ref(2, 1, st_e) &&   // reference, pair 2
                 st_window(true, 1, 3, st_e) && st_window(true, 3, 2, st_e) &&  // deferred: both windows, then
                 st_commit_def(1, 3, 2, st_e) && st_commit_def(3, 2, 1, st_e);  // both commits, in the order they ran
    if (st_ok) st_ok = sync_q("the self-test", st_e);
    if (!st_ok) {
        std::fprintf(stderr, "strata stage worker: SELFTEST: FAILED to run: %s\n", st_e.c_str());
        std::exit(1);
    }
    auto st_compare = [&](const char* what, int ref, int def, int64_t L) -> bool {
        strata::core::ConversationCheckpoint a, b;
        std::string ea, eb;
        if (!strata::core::conversation_checkpoint_save(a, *bslot_ss[0][(size_t) ref], g, (size_t) L, ea) ||
            !strata::core::conversation_checkpoint_save(b, *bslot_ss[0][(size_t) def], g, (size_t) L, eb)) {
            std::fprintf(stderr, "strata stage worker: SELFTEST: %s: the state was not read back (%s%s)\n", what,
                         ea.c_str(), eb.c_str());
            return false;
        }
        auto same = [](const char* part, const std::vector<uint8_t>& x, const std::vector<uint8_t>& y) {
            const bool eq = x == y;
            size_t first = 0, nz = 0, diff = 0;
            for (size_t i = 0; i < std::min(x.size(), y.size()); ++i) {
                if (x[i] != 0) ++nz;
                if (x[i] != y[i]) {
                    if (diff++ == 0) first = i;
                }
            }
            std::fprintf(stderr, "strata stage worker: SELFTEST:   %-9s %zu bytes (%zu vs %zu), %zu non-zero in the "
                                 "reference, %s", part, x.size(), x.size(), y.size(), nz, eq ? "identical\n" : "DIFFER");
            if (!eq) std::fprintf(stderr, " (%zu bytes differ, the first at %zu)\n", diff, first);
            return eq;
        };
        std::fprintf(stderr, "strata stage worker: SELFTEST: %s (slot %d against slot %d, L = %lld)\n", what, ref, def,
                     (long long) L);
        bool eq = same("gdn", a.gdn, b.gdn);
        eq = same("ple", a.ple, b.ple) && eq;
        eq = same("tails", a.tails, b.tails) && eq;
        eq = same("dead", a.dead, b.dead) && eq;
        eq = same("block_pos", a.block_pos, b.block_pos) && eq;
        size_t nz = 0;
        for (uint8_t v : a.gdn) nz += v != 0;
        if (nz == 0) {
            std::fprintf(stderr, "strata stage worker: SELFTEST:   the reference's GDN state is all zero: the test "
                                 "proves nothing\n");
            return false;
        }
        return eq;
    };
    const bool st_p1 = st_compare("3 rows, keep 2", 0, 1, 2);
    const bool st_p2 = st_compare("2 rows, keep 1", 2, 3, 1);
    std::fprintf(stderr, "strata stage worker: SELFTEST: %s\n", st_p1 && st_p2 ? "IDENTICAL" : "DIFFERENT");
    std::exit(st_p1 && st_p2 ? 0 : 1);
}
