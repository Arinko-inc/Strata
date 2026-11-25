// src/prefill/moe_f16_q50.hpp - gfx11: OrcaRouter Q4_K_S's Q5_0 down projection of the prompt experts on the F16 matrix
// cores (moe_f16_q50.cu), so its Q4_K / Q5_0 layers take the fused prompt path (experts read in their slots, one work
// item per 64-row tile) instead of MMQ.  OPT-IN: STRATA_PF_F16_Q50=1 on top of STRATA_PF_FUSED=1, STRATA_PF_FUSED_KQ=1
// and STRATA_PF_F16=1 (the output differs from MMQ's: int8 activations per 32 values and F16 weights).
// Internal to moe_fused_iq.cu.  strata-mesh IMPL-PLAN-2026-11-26-relay-read.md.
#pragma once

#include <cstdint>

#include <cuda_runtime.h>

#include "strata/prefill/moe_fused.hpp"
#include "strata/prefill/moe_fused_iq.hpp"

namespace strata::prefill::fused::f16q50 {

/// STRATA_PF_F16_Q50=1 and this build has gfx11 device code.
bool enabled();
/// Down for experts [b.e0, b.e1) and their tiles, as f16kq::down: H (int8, `h`) to the routed rows' FP32 outputs in `dm`.
void down(const Batch& b, const NativeGeom& g, const int32_t* ts, const int2* tiles, const int32_t* off, int64_t n, int n_expert,
          int64_t work_bound, const uint8_t* h, float* dm, cudaStream_t s);

}  // namespace strata::prefill::fused::f16q50
