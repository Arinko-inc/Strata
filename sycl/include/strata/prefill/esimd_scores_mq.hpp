// The prompt path's QSA block scores from prepared (de-interleaved) copies of the queries and keys, 1-3 queries a hardware thread
// (src/prefill/esimd_scores_mq.dp.cpp).  Opt-in: STRATA_SCORES_MQ=1, 2 or 3 (queries a thread; any other value: off).  The same
// bits as esimd_scores' variant 2 (STRATA_SCORES_ESIMD=2, the B70's production setting).
#pragma once
#include <cstdint>
#include "strata/prefill/esimd_scores.hpp"

namespace strata::prefill::i8 {

/// STRATA_SCORES_MQ (0: off).
int scores_mq_env();
/// Once per QSA layer and chunk, after the chunk's indexer appends: the chunk's `nq` queries (q_idx, [nq][heads][dim]) and the keys
/// of blocks [0, n_blocks) of `pooled` copied to this queue's scratch in the kernel's order.  False: off, or no room (the caller
/// keeps its other kernel).
bool scores_mq_prepare(const float* pooled, const float* q_idx, int64_t nq, int64_t n_blocks, const ScoreSteps& ss, void* stream);
/// block_scores for the queries [q_first, q_first + nq) of the prepared chunk (q_idx and steps point at query q_first).  False when
/// the prepared copies are not these (the caller keeps its other kernel).
bool block_scores_mq(const float* pooled, const float* dead, const float* q_idx, const int32_t* steps, const ScoreSteps& ss, int64_t q_first,
                     int64_t nq, int64_t max_blocks, float* scores, void* stream, int64_t active_blocks);

}  // namespace strata::prefill::i8
