#pragma once
// STRATA_PIPE_PROBE_ROWS=R (試験専用): 遠隔の組(remote_pipe)の窓で、動いている枠ごとに R 行を載せて 1 巡の時間を測る。
// 枠の 2 行目以降は下書きではなく、枠の直前のトークンの水増し行。全行を確定し(主機の段 0 も相手の段も KeepAll)、
// 判定は 1 行目の答えだけを出す。出る文章は使い物にならないので、停止回の計測だけに使う。旗なしなら何も変わらない。
// 行の並べ方だけを純粋な関数にして、GPU なしで試験できるようにしてある(src/program/pipe_probe_test.cpp)。
#include <algorithm>
#include <cstdint>
#include <cstdlib>

namespace strata::program::pipe_probe {

constexpr int kMinRows = 2;
constexpr int kMaxRows = 4;
constexpr int kMaxWindowRows = 8;   ///< 窓の行の配列の大きさ(kVerifyMaxT と同じ)

/// 環境変数の値から R を取り出す。2〜4 のとき R、それ以外(未設定・空・0・範囲外・数字でない)は 0。
inline int probe_rows(const char* env) {
    if (env == nullptr || env[0] == '\0') return 0;
    char* end = nullptr;
    const long v = std::strtol(env, &end, 10);
    if (end == env || *end != '\0') return 0;
    return v >= kMinRows && v <= kMaxRows ? (int) v : 0;
}

/// 起動時の検査。1 窓の行 = 1 組の枠の数 × R(全枠が動いたとき)が、主機・相手の段が受けられる行の上限を超えるなら理由、
/// 収まれば nullptr。`limit_rows`: 受け手のうち一番小さい 1 窓の行の上限。
inline const char* startup_error(int rows_per_slot, int slots_per_group, int limit_rows) {
    if (rows_per_slot < kMinRows || rows_per_slot > kMaxRows) return "R must be 2..4";
    if (slots_per_group < 1) return "no slot in a group";
    if (limit_rows < 1) return "no window row limit";
    if (limit_rows > kMaxWindowRows) limit_rows = kMaxWindowRows;
    if (slots_per_group * rows_per_slot > limit_rows)
        return "a group's window (slots per group x R rows) is over what a stage takes in one window";
    return nullptr;
}

/// 枠 1 つの現在の状態。
struct Slot {
    bool active = false;
    int32_t x = 0;       ///< 次の窓が食べるトークン
    int32_t prev = 0;    ///< 枠の直前のトークン(`ids` の末尾。無ければ x)
    int64_t p = 0;       ///< x の位置
};

/// 組の窓の行。`slot[t]` は t 行目の枠の番号(`slot_base` 込み)。同じ枠の行は連続し、位置は 1 ずつ増える。
struct Group {
    int S = 0;           ///< 行の数
    int active = 0;      ///< 載った枠の数
    int slot[kMaxWindowRows] = {};
    int32_t tok[kMaxWindowRows] = {};
    int64_t pos[kMaxWindowRows] = {};
};

/// 枠ごとの行の数: R 行。ただし位置が文脈の上限(`max_context`)を超えない範囲(最後の行の位置 + 2 が上限以内)で、
/// 1 行は必ず載せる(動いている枠は p + 2 <= max_context が保たれている)。
inline int rows_for_slot(int rows_per_slot, int64_t p, int64_t max_context) {
    const int64_t room = max_context - p - 1;   // 載せられる行の数(位置 p … p + room - 1)
    return (int) std::max<int64_t>(1, std::min<int64_t>(rows_per_slot, room));
}

/// 組の窓を組み立てる。動いている枠ごとに: 1 行目は (x, p)、2〜R 行目は (prev, p + j)。合計が `cap` 行を超えるときは、
/// 載らない分の行を減らす(1 行も載らない枠はこの窓に載せない。起動時の検査で通らない並びなので、通常は起きない)。
/// 戻り値は行の数 S。
inline int build_probe_group(int rows_per_slot, int cap, int64_t max_context, const Slot* slots, int n_slots,
                             int slot_base, Group& g) {
    g = Group{};
    cap = std::min(cap, kMaxWindowRows);
    for (int t = 0; t < n_slots; ++t) {
        const Slot& sl = slots[t];
        if (!sl.active) continue;
        const int nr = std::min(rows_for_slot(rows_per_slot, sl.p, max_context), cap - g.S);
        if (nr < 1) continue;
        ++g.active;
        for (int j = 0; j < nr; ++j) {
            g.slot[g.S] = slot_base + t;
            g.tok[g.S] = j == 0 ? sl.x : sl.prev;
            g.pos[g.S] = sl.p + j;
            ++g.S;
        }
    }
    return g.S;
}

}  // namespace strata::program::pipe_probe
