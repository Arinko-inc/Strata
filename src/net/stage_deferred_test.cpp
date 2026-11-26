// src/net/stage_deferred_test.cpp - L1: a remote stage worker's table of deferred windows (include/strata/net/
// stage_deferred.hpp), header-only, no GPU.  strata-mesh IMPL-PLAN-2026-11-26-batch-drafts.md 3.4.
//
// The flag's reading and the bound (the skeleton's), and the table's tests (担当 F):
// add (S bounds, slots in range, a slot's rows contiguous and at most rows_per_region, windows on overlapping slots
// refused, a tag held twice refused, save_base = rows[0] * rows_per_region, saved rows past the end or over a held
// window's refused), STRATA_WK_PIPED_MTP_ROWS's reading (合流 D), take (by tag in any order, an unknown tag, a window taken twice),
// slot_held / empty / size, reset, and keep_ok (1..rows on the window's slots, 0 elsewhere, keep_by_slot padded).
#include "strata/net/stage_deferred.hpp"

#include <cstdio>
#include <initializer_list>
#include <string>
#include <vector>

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
    std::string e;
    d.reset(4, 4);
    check(d.empty() && d.size() == 0 && !d.slot_held(0), "reset: none held");
    // add: the bounds
    auto win = [](int64_t tag, std::initializer_list<int> rows) {
        sn::DeferredWindow w;
        w.tag = tag;
        for (int r : rows) {
            w.rows[w.S] = r;
            w.tok[w.S] = 100 + w.S;
            w.pos[w.S] = 10 + w.S;
            ++w.S;
        }
        return w;
    };
    {
        sn::DeferredWindow w = win(1, {});
        check(!d.add(w, e) && !e.empty() && d.empty(), "add: 0 rows refused");
        w = win(1, {0, 0, 0, 0, 0});
        check(!d.add(w, e) && d.empty() && e.find("more than 4 rows") != std::string::npos,
              "add: a slot with more rows than its region refused (5 of 4)");
        w = win(1, {4});
        check(!d.add(w, e) && d.empty(), "add: a slot outside the link's slots refused");
        w = win(1, {-1});
        check(!d.add(w, e) && d.empty(), "add: a negative slot refused");
        w = win(1, {0, 1, 0});
        check(!d.add(w, e) && d.empty(), "add: a slot's rows not together refused");
        sn::DeferredWindows z;
        w = win(1, {0});
        check(!z.add(w, e) && z.empty(), "add: before reset (no slots) refused");
        sn::DeferredWindows big;
        big.reset(4, 16);
        w = win(1, {0, 0, 0, 0, 0, 0, 0, 0});
        check(big.add(w, e) && big.size() == 1, "add: 8 rows (the most) held");
        sn::DeferredWindow w9 = win(2, {1, 1, 1, 1, 1, 1, 1, 1});
        w9.S = 9;
        check(!big.add(w9, e) && big.size() == 1, "add: 9 rows refused");
    }
    // two windows on separate slots, their regions, an overlap (IMPL-PLAN section 4 stage 1 (d))
    sn::DeferredWindow a = win(11, {0, 0, 1, 1}), b = win(12, {2, 3, 3});
    check(d.add(a, e) && a.save_base == 0 && d.size() == 1, "add: window 11 on slots 0,1 held, save_base 0");
    check(d.add(b, e) && b.save_base == 8 && d.size() == 2, "add: window 12 on slots 2,3 held, save_base 2 * 4");
    check(d.slot_held(0) && d.slot_held(1) && d.slot_held(2) && d.slot_held(3) && !d.slot_held(4) && !d.empty(),
          "slot_held: the held windows' slots");
    {
        sn::DeferredWindow c = win(13, {1});
        e.clear();
        check(!d.add(c, e) && e.find("slot 1") != std::string::npos && d.size() == 2,
              "add: a window on a held window's slot refused (d), nothing held of it");
        d.reset(4, 4);
        sn::DeferredWindow x = win(11, {0});
        check(d.add(x, e), "add: after reset the slot is free");
        sn::DeferredWindow y = win(11, {1});
        check(!d.add(y, e) && d.size() == 1 && !d.slot_held(1), "add: a tag held already refused");
        sn::DeferredWindow y2 = win(14, {3, 3});
        check(d.add(y2, e) && y2.save_base == 12, "add: save_base = rows[0] * rows_per_region (3 * 4)");
        d.reset(4, 4);
        check(d.add(a, e) && d.add(b, e), "the two windows again");
    }
    // STRATA_WK_PIPED_MTP_ROWS (合流 D): the saved rows of one slot, default 2, clamped to the window's rows
    check(sn::kDeferredSlotRowsDefault == 2 && sn::wk_piped_mtp_rows(nullptr, 6) == 2 && sn::wk_piped_mtp_rows("", 6) == 2 &&
              sn::wk_piped_mtp_rows("x", 6) == 2 && sn::wk_piped_mtp_rows("3x", 6) == 2,
          "STRATA_WK_PIPED_MTP_ROWS: unset, empty or not a number: 2");
    check(sn::wk_piped_mtp_rows("4", 6) == 4 && sn::wk_piped_mtp_rows("9", 6) == 6 && sn::wk_piped_mtp_rows("9", 12) == 8 &&
              sn::wk_piped_mtp_rows("0", 6) == 1 && sn::wk_piped_mtp_rows("-3", 6) == 1 && sn::wk_piped_mtp_rows("3", 2) == 2,
          "STRATA_WK_PIPED_MTP_ROWS: clamped to 1..min(the window's rows, 8)");
    {   // regions of 2 rows (the default): the main process's 2 groups x 2 slots x (a token + one draft)
        sn::DeferredWindows r2;
        r2.reset(4, 2);
        sn::DeferredWindow g0 = win(21, {0, 0, 1, 1}), g1 = win(22, {2, 2, 3, 3});
        check(r2.add(g0, e) && g0.save_base == 0 && r2.add(g1, e) && g1.save_base == 4 && r2.size() == 2,
              "rows 2: two groups of 2 slots x 2 rows held, saved rows [0,4) and [4,8)");
        sn::DeferredWindow o;
        check(r2.take(21, o, e), "rows 2: the first group committed");
        sn::DeferredWindow g0b = win(23, {1, 1});
        check(r2.add(g0b, e) && g0b.save_base == 2, "rows 2: the group with only its second slot live, save_base 2");
        check(r2.take(23, o, e) && r2.take(22, o, e) && r2.empty(), "rows 2: none held");
        sn::DeferredWindow three = win(24, {0, 0, 0});
        e.clear();
        check(!r2.add(three, e) && r2.empty() && e.find("STRATA_WK_PIPED_MTP_ROWS") != std::string::npos,
              "rows 2: a slot with 3 rows refused (a token + 2 drafts)");
        sn::DeferredWindow one = win(25, {1});
        check(r2.add(one, e) && one.save_base == 2, "rows 2: a window on slot 1 alone held");
        sn::DeferredWindow gap = win(26, {0, 0, 2, 2});
        e.clear();
        check(!r2.add(gap, e) && r2.size() == 1 && e.find("overlap") != std::string::npos,
              "rows 2: slots 0 and 2 (saved rows [0,4)) over slot 1's held [2,3) refused");
        check(r2.take(25, o, e) && r2.add(gap, e) && gap.save_base == 0, "rows 2: the same window with slot 1 free held");
        check(r2.take(26, o, e), "rows 2: taken");
        sn::DeferredWindow tail = win(27, {3, 3, 0});
        e.clear();
        check(!r2.add(tail, e) && r2.empty() && e.find("past the last") != std::string::npos,
              "rows 2: saved rows from slot 3's region past the end (6 + 3 > 8) refused");
    }
    // find (合流 D): the held window by tag, without taking it
    {
        sn::DeferredWindows f;
        f.reset(4, 2);
        sn::DeferredWindow x = win(51, {0, 0}), y = win(52, {2});
        check(f.add(x, e) && f.add(y, e), "find: two windows held");
        const sn::DeferredWindow* fx = f.find(51);
        check(fx != nullptr && fx->tag == 51 && fx->S == 2 && f.find(52) != nullptr && f.find(53) == nullptr &&
                  f.size() == 2,
              "find: by tag, nullptr for a tag not held, nothing taken");
    }
    // take: by tag, any order
    {
        sn::DeferredWindow o;
        check(!d.take(99, o, e) && e.find("99") != std::string::npos && d.size() == 2, "take: an unknown tag refused");
        check(d.take(12, o, e) && o.tag == 12 && o.S == 3 && o.rows[0] == 2 && o.rows[2] == 3 && o.save_base == 8 &&
                  o.tok[1] == 101 && o.pos[2] == 12,
              "take: the later window first, as it was held");
        check(d.size() == 1 && !d.slot_held(2) && !d.slot_held(3) && d.slot_held(0), "take: its slots are free");
        check(!d.take(12, o, e), "take: a window taken twice refused");
        sn::DeferredWindow c = win(15, {3});
        check(d.add(c, e) && c.save_base == 12, "add: a freed slot takes a new window while another is held");
        check(d.take(11, o, e) && o.tag == 11 && o.S == 4 && o.save_base == 0, "take: the earlier window");
        check(d.take(15, o, e) && d.empty(), "take: the last one, none held");
    }
    // keep_ok
    {
        std::vector<int> k;
        const int32_t k1[4] = {2, 1, 0, 0};
        check(sn::DeferredWindows::keep_ok(a, k1, 2, 4, k, e) && k.size() == 4 && k[0] == 2 && k[1] == 1 && k[2] == 0 && k[3] == 0,
              "keep_ok: 1..rows on the window's slots, padded to the worker's slots");
        check(sn::DeferredWindows::keep_ok(a, k1, 4, 6, k, e) && k.size() == 6 && k[0] == 2 && k[5] == 0,
              "keep_ok: m beyond the window's slots with zeros");
        const int32_t k2[4] = {3, 1, 0, 0};
        check(!sn::DeferredWindows::keep_ok(a, k2, 2, 4, k, e) && k.size() == 4 && k[0] == 0,
              "keep_ok: more than the slot's rows refused");
        const int32_t k3[4] = {0, 1, 0, 0};
        check(!sn::DeferredWindows::keep_ok(a, k3, 2, 4, k, e), "keep_ok: a window's slot keeping 0 refused");
        const int32_t k4[4] = {1, 1, 1, 0};
        check(!sn::DeferredWindows::keep_ok(a, k4, 3, 4, k, e), "keep_ok: a slot outside the window keeping rows refused");
        check(!sn::DeferredWindows::keep_ok(a, k1, 1, 4, k, e), "keep_ok: m short of a window's slot refused");
        check(!sn::DeferredWindows::keep_ok(a, k1, 0, 4, k, e) && !sn::DeferredWindows::keep_ok(a, k1, 5, 4, k, e),
              "keep_ok: m outside 1..n_slots refused");
        const int32_t k5[4] = {-1, 1, 0, 0};
        check(!sn::DeferredWindows::keep_ok(a, k5, 2, 4, k, e), "keep_ok: a negative keep refused");
        const int32_t k6[4] = {0, 0, 1, 3};
        check(!sn::DeferredWindows::keep_ok(b, k6, 4, 4, k, e), "keep_ok: window 12, slot 3 keeps 3 of 2 refused");
        const int32_t k7[4] = {0, 0, 1, 2};
        check(sn::DeferredWindows::keep_ok(b, k7, 4, 4, k, e) && k[2] == 1 && k[3] == 2 && k[0] == 0,
              "keep_ok: window 12 on slots 2,3");
    }
    std::printf(g_fail == 0 ? "all passed (0 failed)\n" : "%d FAILED\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
