// include/strata/program/remote_piped_mtp.hpp - L1 (Arinko-inc): whether a remote stage's main process drafts in its
// pipelined groups (--batch-mtp + STRATA_PIPED_MTP=N over the link), and how many drafts a group's window carries
// (strata-mesh IMPL-PLAN-2026-11-26-batch-drafts.md 3.3).  Pure functions, no GPU (remote_piped_mtp_test.cpp).
//
// The skeleton (2026-11-26) fixed the declarations; the bodies at the end are 担当 M's.  Without a STRATA_PIPED_MTP
// decide() is off with nothing to say, so the main process runs the groups as it does today.  The public signatures
// may change only with a reason in the report.
#pragma once

#include <algorithm>
#include <cstdint>

#include "strata/net/stage_link.hpp"   // kStageBatchDeferred (a worker's hello bit)

namespace strata::program::remote_piped_mtp {

/// 窓の行の配列の大きさ(strata::kernels::kVerifyMaxT と同じ。generate.cpp は static_assert で突き合わせる)
inline constexpr int kMaxWindowRows = 8;

struct Input {
    bool remote_main = false, remote_pipe = false;   ///< the main process of a remote stage, pipelined groups
    bool batch_mtp = false;                          ///< --batch-mtp / STRATA_BATCH_MTP after its own gates
    int pm_drafts = 0;                               ///< STRATA_PIPED_MTP=N (0: unset)
    int batch = 0, groups = 0;                       ///< the slots agreed, the groups
    uint32_t peer_batch_flags = 0;                   ///< the worker's hello (strata::net::kStageBatchDeferred, bit 3)
    int peer_max_t = 0;                              ///< the rows a window may hold: the worker's hello max_t (generate.cpp passes the
                                                     ///< narrowest of it and this process's two verifiers)
    bool self_commit = false;                        ///< STRATA_BATCH_SELF_COMMIT
    int probe_rows = 0;                              ///< STRATA_PIPE_PROBE_ROWS (stage 0's measurement; not with drafts)
};
/// on: the groups draft, `drafts` per slot (1..pm_drafts); off: `why` is the reason, logged once (nullptr when there is
/// nothing to say: STRATA_PIPED_MTP unset, or not a remote main process)
struct Result {
    bool on = false;
    int drafts = 0;
    const char* why = nullptr;
};
Result decide(const Input& in);
/// a group's rows: each active slot 1 + its drafts, the drafts cut (the same for every slot) so the window holds at
/// most `cap` rows; the drafts per slot (0: only the tokens fit)
int drafts_that_fit(int active_slots, int drafts, int cap);

// ==== the bodies: 担当 M (strata-mesh IMPL-PLAN-2026-11-26-batch-drafts.md, the section after the skeleton) ====
// 切るときは理由を 1 つ返す(組の窓を今と同じ KeepAll で回す)。調べる順は、旗が無い → 計測旗 → --batch-mtp → 遠隔の組 →
// 自己確定 → 相手の hello → 窓の行の数。ログに出す文言は英語(既存の strata: の行と同じ)。
inline Result decide(const Input& in) {
    Result r;
    if (!in.remote_main || in.pm_drafts <= 0) return r;   // STRATA_PIPED_MTP が無い、または遠隔の主機ではない: 言うことは無い
    if (in.probe_rows > 0) {
        r.why = "STRATA_PIPE_PROBE_ROWS is a measurement of padded rows and does not run with drafts";
        return r;
    }
    if (!in.batch_mtp) {
        r.why = "it needs --batch-mtp (or STRATA_BATCH_MTP=1), which was off or refused";
        return r;
    }
    if (!in.remote_pipe || in.batch < 2 || in.groups < 2) {
        r.why = "it needs the remote pipelined groups (--batch-groups 2 or more, agreed with the worker)";
        return r;
    }
    if (in.batch % in.groups != 0) {
        r.why = "the groups do not divide the slots agreed";
        return r;
    }
    if (in.self_commit) {
        r.why = "it does not combine with STRATA_BATCH_SELF_COMMIT (the window would keep its rows itself)";
        return r;
    }
    if ((in.peer_batch_flags & strata::net::kStageBatchDeferred) == 0) {
        r.why = "the worker does not keep deferred windows (kStageBatchDeferred)";
        return r;
    }
    // 全枠が動いたときの 1 組の窓: 枠ごとにトークン 1 行 + 下書き n 行。受けられる行の数は一番狭い相手で決まる
    const int per_group = in.batch / in.groups;
    const int cap = std::min(kMaxWindowRows, in.peer_max_t);
    const int n = std::min(in.pm_drafts, cap / per_group - 1);
    if (n < 1) {
        r.why = "a group's window (its slots' tokens and one proposal each) does not fit in the rows a window takes "
                "(this process's verifiers and the worker's max_t)";
        return r;
    }
    r.on = true;
    r.drafts = n;
    return r;
}
inline int drafts_that_fit(int active_slots, int drafts, int cap) {
    if (active_slots < 1 || drafts < 1 || cap < 1) return 0;
    return std::max(0, std::min(drafts, std::min(cap, kMaxWindowRows) / active_slots - 1));
}
// ==== end of 担当 M's bodies ====

}  // namespace strata::program::remote_piped_mtp
