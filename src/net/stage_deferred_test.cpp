// src/net/stage_deferred_test.cpp - L1: a remote stage worker's table of deferred windows (include/strata/net/
// stage_deferred.hpp), header-only, no GPU.  strata-mesh IMPL-PLAN-2026-11-26-batch-drafts.md 3.4.
//
// The skeleton (2026-11-26) checks only what it fixed (the flag's reading, the bound); 担当 F adds the table's tests:
// add (S bounds, slots in range, a slot's rows contiguous, windows on overlapping slots refused, a tag held twice
// refused, save_base = rows[0] * rows_per_region), take (by tag in any order, an unknown tag, a window taken twice),
// slot_held / empty / size, reset, and keep_ok (1..rows on the window's slots, 0 elsewhere, keep_by_slot padded).
#include "strata/net/stage_deferred.hpp"

#include <cstdio>

namespace sn = strata::net;

namespace {
int g_fail = 0;
void check(bool ok, const char* what) {
    std::printf("  %-78s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}
}  // namespace

int main() {
    std::printf("stage_deferred_test\n");
    check(sn::kDeferredMaxRows == 8, "a deferred window holds up to 8 rows (kVerifyMaxT)");
    check(!sn::wk_piped_mtp_on(nullptr) && !sn::wk_piped_mtp_on("") && !sn::wk_piped_mtp_on("0"),
          "STRATA_WK_PIPED_MTP: unset, empty and 0 are off");
    check(sn::wk_piped_mtp_on("1") && sn::wk_piped_mtp_on("yes") && sn::wk_piped_mtp_on("01"),
          "STRATA_WK_PIPED_MTP: any other value is on");
    sn::DeferredWindows d;
    d.reset(4, 4);
    check(d.empty() && d.size() == 0 && !d.slot_held(0), "reset: none held");
    std::printf(g_fail == 0 ? "all passed (0 failed)\n" : "%d FAILED\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
