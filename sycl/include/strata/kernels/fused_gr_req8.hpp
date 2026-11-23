// sycl/include/strata/kernels/fused_gr_req8.hpp - SYCL port only: STRATA_HC_REQ8=1's hyper-connection read on ESIMD
// (Battlemage).  The projections' int8 codes are CUDA / HIP's STRATA_HC_REQ8 codes (Q8_0's rule per 32 contiguous
// values of a row: d = amax / 127, q = round-to-nearest-even(x / d), one fp32 scale a block); only their place in the
// device table differs (core/weights.hpp, hc_req8_pack).  The read is the RDNA read's int8 form (rdna_gr.cu, STRATA_GR_RDNA_A8):
// the activations are quantized to int8 per 32 too and meet the codes in dp4a.
#pragma once

#include <cstddef>
#include <cstdint>

#include "strata/kernels/fused_gr.hpp"

namespace strata::kernels {

/// True when the ESIMD kernels are in this build (not with STRATA_I8_NO_ESIMD).
bool hc_req8_built();

/// STRATA_HC_REQ8 is set to a non-zero value AND the kernels are built.
bool hc_req8_enabled();

/// fused_gr_read_multi for arguments whose w_down / w_up are hc_req8_pack's codes with s_down / s_up their scales
/// (w_inject stays BF16).  Returns false when it cannot serve the call (not built).
bool fused_gr_req8_read_multi(const FusedGrArgs* a, int n_tok, void* stream, unsigned long long* stamp_buf, int stamp_i0);

}  // namespace strata::kernels
