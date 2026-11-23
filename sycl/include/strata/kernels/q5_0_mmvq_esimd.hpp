// sycl/include/strata/kernels/q5_0_mmvq_esimd.hpp - SYCL port only: STRATA_SHEXP_ESIMD=1's dense Q5_0 GEMV on ESIMD
// (Battlemage) for the shared expert's down projection (640 inputs, the decode window's 1-4 columns).  Every output is
// bit for bit native_mmvq(6, ...)'s (the small-K lane kernel); shexp_bench checks that on a model's own layers.
#pragma once

namespace strata::kernels {

/// True when the ESIMD kernel is in this build (not with STRATA_I8_NO_ESIMD).
bool q5_0_mmvq_esimd_built();

/// native_mmvq(6, weights, x_q8_1, y, 640, n_out, ncols, stream) on ESIMD.  rpt: weight rows per hardware thread (1, 2, 4,
/// 8); var < 0: the engine's variant.  Returns false (nothing launched) when it cannot serve the call: not built, n_in
/// != 640, ncols outside 1..4, n_out not a multiple of rpt, a pointer that is not a multiple of 4, an unknown variant.
bool q5_0_mmvq_esimd(const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, int ncols, void* stream,
                     int rpt, int var);

/// The shared expert's down projection (shared_expert_multi): 1 = through q5_0_mmvq_esimd where it serves the call, 0 =
/// native_mmvq.  The value is STRATA_SHEXP_ESIMD's (default 0) until shared_expert_set_esimd overrides it (the bench).
int shared_expert_esimd();
void shared_expert_set_esimd(int v);

}  // namespace strata::kernels
