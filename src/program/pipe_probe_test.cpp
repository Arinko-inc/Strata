// src/program/pipe_probe_test.cpp - STRATA_PIPE_PROBE_ROWS の行の並べ方(header-only、CPU)。
//
//   1. probe_rows: 2〜4 だけが通り、未設定・空・0・1・5・数字でない値は 0;
//   2. startup_error: 枠の数 × R が受け手の上限(4 行など)を超えるなら断る;
//   3. build_probe_group: 動いている枠ごとに R 行、1 行目は (x, p)・2 行目以降は (prev, p + j)、同じ枠の行は連続;
//   4. 文脈の上限の手前では行を減らし、cap を超えない。
#include "strata/program/pipe_probe.hpp"

#include <cstdio>

namespace pp = strata::program::pipe_probe;

namespace {
int g_fail = 0;
void check(bool ok, const char* what) {
    std::printf("  %-78s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}
}  // namespace

int main() {
    std::printf("pipe_probe_test\n");
    check(pp::probe_rows(nullptr) == 0 && pp::probe_rows("") == 0 && pp::probe_rows("0") == 0 && pp::probe_rows("1") == 0,
          "probe_rows: unset, empty, 0 and 1 are off");
    check(pp::probe_rows("2") == 2 && pp::probe_rows("3") == 3 && pp::probe_rows("4") == 4, "probe_rows: 2, 3, 4");
    check(pp::probe_rows("5") == 0 && pp::probe_rows("-2") == 0 && pp::probe_rows("2x") == 0 && pp::probe_rows("abc") == 0,
          "probe_rows: out of range or not a number are off");

    check(pp::startup_error(2, 2, 4) == nullptr, "startup: 2 slots x 2 rows in a 4-row stage is fine");
    check(pp::startup_error(2, 1, 4) == nullptr && pp::startup_error(4, 1, 4) == nullptr, "startup: 1 slot x 4 rows fits 4");
    check(pp::startup_error(3, 2, 4) != nullptr && pp::startup_error(4, 2, 4) != nullptr,
          "startup: 2 slots x 3 or 4 rows is over a 4-row stage");
    check(pp::startup_error(2, 4, 4) != nullptr, "startup: 4 slots x 2 rows is over a 4-row stage");
    check(pp::startup_error(4, 2, 8) == nullptr && pp::startup_error(4, 3, 8) != nullptr, "startup: a 8-row limit");
    check(pp::startup_error(2, 2, 99) == nullptr && pp::startup_error(4, 3, 99) != nullptr,
          "startup: limits above the array size (8) are cut to 8");
    check(pp::startup_error(1, 1, 4) != nullptr && pp::startup_error(2, 0, 4) != nullptr && pp::startup_error(2, 1, 0) != nullptr,
          "startup: R of 1, no slot, no limit");

    {   // 2 slots active of 3, R = 2
        pp::Slot s[3];
        s[0].active = true; s[0].x = 11; s[0].prev = 10; s[0].p = 100;
        s[1].active = false;
        s[2].active = true; s[2].x = 21; s[2].prev = 20; s[2].p = 50;
        pp::Group g;
        const int S = pp::build_probe_group(2, 4, 4096, s, 3, 6, g);
        check(S == 4 && g.S == 4 && g.active == 2, "group: 2 active slots x 2 rows = 4 rows");
        check(g.slot[0] == 6 && g.slot[1] == 6 && g.slot[2] == 8 && g.slot[3] == 8, "group: slot numbers are base + index, rows together");
        check(g.tok[0] == 11 && g.tok[1] == 10 && g.tok[2] == 21 && g.tok[3] == 20, "group: first row x, the rest prev");
        check(g.pos[0] == 100 && g.pos[1] == 101 && g.pos[2] == 50 && g.pos[3] == 51, "group: positions p, p+1, ...");
    }
    {   // 1 slot, R = 4
        pp::Slot s[1];
        s[0].active = true; s[0].x = 5; s[0].prev = 4; s[0].p = 7;
        pp::Group g;
        check(pp::build_probe_group(4, 4, 4096, s, 1, 0, g) == 4 && g.tok[3] == 4 && g.pos[3] == 10, "group: 1 slot x 4 rows");
        check(pp::build_probe_group(4, 3, 4096, s, 1, 0, g) == 3, "group: never over cap");
    }
    {   // a slot near the context's end carries fewer rows (its last position + 2 stays within max_context)
        pp::Slot s[1];
        s[0].active = true; s[0].x = 5; s[0].prev = 4; s[0].p = 98;
        pp::Group g;
        check(pp::build_probe_group(4, 4, 100, s, 1, 0, g) == 1, "context end: p + 2 == max_context leaves 1 row");
        s[0].p = 96;
        check(pp::build_probe_group(4, 4, 100, s, 1, 0, g) == 3 && g.pos[2] == 98, "context end: room for 3 rows");
        check(pp::rows_for_slot(2, 0, 100) == 2 && pp::rows_for_slot(4, 99, 100) == 1, "rows_for_slot");
    }
    {   // no active slot: an empty window
        pp::Slot s[2];
        pp::Group g;
        check(pp::build_probe_group(2, 4, 4096, s, 2, 0, g) == 0 && g.active == 0, "group: no active slot, no row");
    }
    {   // cap leaves a slot with fewer rows; one that gets none is left out
        pp::Slot s[2];
        s[0].active = true; s[0].x = 1; s[0].prev = 0; s[0].p = 10;
        s[1].active = true; s[1].x = 3; s[1].prev = 2; s[1].p = 20;
        pp::Group g;
        check(pp::build_probe_group(3, 4, 4096, s, 2, 0, g) == 4 && g.active == 2 && g.slot[3] == 1,
              "cap: 3 rows then 1 row");
        check(pp::build_probe_group(2, 2, 4096, s, 2, 0, g) == 2 && g.active == 1, "cap: the second slot is left out");
    }
    std::printf(g_fail == 0 ? "pipe_probe_test: ok\n" : "pipe_probe_test: %d FAILED\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
