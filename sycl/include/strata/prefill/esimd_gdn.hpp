// The prompt path's GDN recurrence on ESIMD (Battlemage, the SYCL port): variant 7 of gdn_recurrence_variant
// (src/prefill/esimd_gdn.dp.cpp).  The column kernel's arithmetic in the same order from threads that hold a block of
// columns of a head's state in registers: no local memory, no barriers.
#pragma once
#include <cstdint>

namespace strata::prefill {

/// The recurrence of `T` tokens (state updated in place, the un-normalized output columns to `out` [T][48][128]), as
/// gdn_rec_cols_kernel computes them.  `var`: 1 the form that gives the column kernel's bits on the B70; +2 the delta's
/// `v - g * kv` unfused; +4 1/sqrt(128) as a constant instead of the device's rsqrt; +8 eight columns a thread (16
/// without); +16 no software pipelining of the token inputs.  False (nothing launched): not an ESIMD device, or the
/// kernels are not in this build.
bool gdn_rec_esimd(float* state, const float* h, const float* gate, const float* beta, float* out, int64_t T, void* stream,
                   int var);

}  // namespace strata::prefill
