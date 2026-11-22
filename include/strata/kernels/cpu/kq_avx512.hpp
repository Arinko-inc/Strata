// include/strata/kernels/cpu/kq_avx512.hpp - 512-bit Q4_K . Q8_K dot product (AVX-512 F/BW), the same bits as
// ggml-cpu's AVX2 ggml_vec_dot_q4_K_q8_K.  See src/kernels/cpu/kq_avx512.cpp.
#pragma once

namespace strata::kernels::cpu {

/// One Q4_K row (n weights, n % 256 == 0) against one Q8_K activation row.  Same signature shape as ggml's vec_dot
/// minus the strides.  Only call when cpu_avx512_ok().
void kq512_dot_q4k_q8k(int n, float* s, const void* vx, const void* vy);

}  // namespace strata::kernels::cpu
