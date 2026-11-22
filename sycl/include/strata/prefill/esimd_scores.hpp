// The prompt path's QSA block scores on ESIMD (Battlemage, the SYCL port): one hardware thread per query and 16 blocks instead of
// the warp kernel's 32 work-items per (query, block).  The same scores bit for bit (src/prefill/esimd_scores.dp.cpp: the same
// operations in the same order), so the selection is the same.  STRATA_SCORES_ESIMD=0 keeps the warp kernel.
#pragma once
#include <cstdint>

namespace strata::prefill::i8 {

/// Where a query's step record holds what the scorer reads (strata/kernels/qsa.hpp: kStepCount, kStepNKv, kStepNBid), and the
/// indexer's geometry (key dim 128, 4 query heads, blocks of `block` cells).
struct ScoreSteps {
    int count = 0, n_kv = 0, n_bid = 0;
    int dim = 0, heads = 0, block = 0;
};

/// scores[q][b] for b <= n_bid(q), as kernels::qsa_block_scores writes them (the tail block from `dead`, +1e9 while incomplete).
/// False when the device or the geometry is not the one the kernel is written for (the caller keeps the warp kernel).
bool block_scores(const float* pooled, const float* dead, const float* q_idx, const int32_t* steps, const ScoreSteps& ss, int64_t nq,
                  int64_t max_blocks, float* scores, void* stream, int64_t active_blocks, int variant);

}  // namespace strata::prefill::i8
