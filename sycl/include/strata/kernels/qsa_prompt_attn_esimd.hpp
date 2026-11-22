// SYCL port (Battlemage): the prompt attention of an INT8 K/V cache on ESIMD - the arithmetic of qsa_decode_attn_batch,
// operation for operation, from threads that keep their operands in registers (src/kernels/cuda/qsa_prompt_attn_esimd.dp.cpp).
#pragma once
#include <cstdint>
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"

namespace strata::kernels {

/// The contract of qsa_prompt_attn_batch.  False (nothing launched) unless STRATA_ATTN_ESIMD is set to a variant (1: the
/// one that gives the warp kernel's bits on the B70; +2: the score sums written unfused; +4: the last division by the
/// ESIMD `/` instead of div_ieee), the pools are INT8 K and V with scales, the page size is a power of two and the device
/// runs ESIMD; the caller then keeps its own kernel.
bool qsa_prompt_attn_esimd(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* steps, int64_t cap,
                           const QsaShapes& s, float* attn, int64_t n_q, void* stream);

}  // namespace strata::kernels
