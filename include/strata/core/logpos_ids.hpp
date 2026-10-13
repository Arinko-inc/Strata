// include/strata/core/logpos_ids.hpp - measurement hooks beside STRATA_LOGPOS (strata-mesh stage A; off by default).
//
// Both are read once from the environment; unset, nothing here runs and the STRATA_LOGPOS output is byte-identical to
// a build without them.  Header-only and plain C++, so the CUDA/HIP tree and the SYCL port (which keeps its own
// verify.cpp and generate.cpp) share one implementation.
//
// STRATA_LOGPOS_SPLIT_AT=<n> (n > 0): a prompt's read is cut at token n - [0, n) goes through the batched prompt path
//   (prompt chunks) and [n, end) through the verify windows, which write the STRATA_LOGPOS rows.  The KL check of the
//   prompt-chunk hand-off (DESIGN.md 6.3): a long preamble read as chunks, the positions after it scored.
// STRATA_LOGPOS_IDS=<baseline dump>: every STRATA_LOGPOS row gets, after its own columns, a "\t|" and then
//   "\t<id>:<logprob>" for each id that the baseline dump lists at the same text and position (its STRATA_LOGPOS_TOPK
//   columns, in the baseline's order) - the log-probabilities this run gives the baseline's top ids, for a KL over
//   them.  Format and the text/position matching: ~/strata-mesh/IMPL-PLAN-stage-a.md, "STRATA_LOGPOS_IDS".
#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace strata::core::logpos {

/// STRATA_LOGPOS_SPLIT_AT: the token the prompt read is cut at; -1 when unset (or not positive).
inline int64_t split_at() {
    static const int64_t v = [] {
        const char* e = std::getenv("STRATA_LOGPOS_SPLIT_AT");
        return e != nullptr && *e != '\0' ? (int64_t) std::atoll(e) : (int64_t) -1;
    }();
    return v > 0 ? v : -1;
}

/// STRATA_LOGPOS_IDS: append the baseline's ids for the row at `pos` (`row`: the row's n_vocab logits, `lse`: their
/// log-sum-exp).  True and nothing written when unset; false with `err` when the baseline cannot be used.
/// SKELETON: not implemented yet (担当 C) - set, it refuses, so a measurement cannot silently miss its columns.
inline bool write_ids(std::FILE* out, int64_t pos, const float* row, int64_t n_vocab, double lse, std::string& err) {
    static const char* path = std::getenv("STRATA_LOGPOS_IDS");
    if (path == nullptr || *path == '\0') return true;
    (void) out; (void) pos; (void) row; (void) n_vocab; (void) lse;
    err = "STRATA_LOGPOS_IDS is not implemented in this build yet";
    return false;
}

}  // namespace strata::core::logpos
