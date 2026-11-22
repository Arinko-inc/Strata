// src/kernels/cpu/kq_avx512.cpp - Q4_K . Q8_K in 512-bit lanes (AVX-512 F + BW; no VNNI/VBMI instruction is used).
//
// Ported from upstream Strata v0.2.19 engine/src/quant/x86/dot_avx512.cpp (dot_q4_K_q8_K_zmm, 34b82b540): one zmm
// holds the low and the high nibbles of 32 weight bytes (64 weights), so a 256-weight super-block is 4 iterations
// instead of ggml's 4 x 2 ymm halves.  The int32 lane sums are exact integers and the float accumulation
// (acc += d * sumi per super-block, acc_m += dmin * prod, the final horizontal sums) is ggml's AVX2 order, so the
// result is ggml_vec_dot_q4_K_q8_K's bits (AVX2 + FMA + F16C build).  native_expert.cpp checks that once against
// the ggml it is linked with before the first use (kq512_ok) and falls back to ggml's dot otherwise.
// Compiled with the AVX-512 flags of CMakeLists.txt; only called behind cpu_avx512_ok().
#include "strata/kernels/cpu/kq_avx512.hpp"

#include <cstdint>
#include <cstring>
#include <immintrin.h>

namespace strata::kernels::cpu {
namespace {

#pragma pack(push, 1)
struct BlockQ4K {
    uint16_t d, dmin;
    uint8_t scales[12];
    uint8_t qs[128];
};
struct BlockQ8K {
    float d;
    int8_t qs[256];
    int16_t bsums[16];
};
#pragma pack(pop)
static_assert(sizeof(BlockQ4K) == 144, "block_q4_K");
static_assert(sizeof(BlockQ8K) == 292, "block_q8_K");

inline float h2f(uint16_t h) { return _cvtsh_ss(h); }

inline float hsum_float_8(__m256 x) {   // ggml's order
    __m128 res = _mm256_extractf128_ps(x, 1);
    res = _mm_add_ps(res, _mm256_castps256_ps128(x));
    res = _mm_add_ps(res, _mm_movehl_ps(res, res));
    res = _mm_add_ss(res, _mm_movehdup_ps(res));
    return _mm_cvtss_f32(res);
}

// ggml's get_scale_shuffle_k4(i): 32 bytes of (2i, 2i+1); here k4(2j) in the low and k4(2j+1) in the high 256 bits
inline __m512i scale_shuffle(int j) {
    const __m256i lo = _mm256_set1_epi16((short) (((4 * j + 1) << 8) | (4 * j)));
    const __m256i hi = _mm256_set1_epi16((short) (((4 * j + 3) << 8) | (4 * j + 2)));
    return _mm512_inserti64x4(_mm512_castsi256_si512(lo), hi, 1);
}

}  // namespace

void kq512_dot_q4k_q8k(int n, float* s, const void* vx, const void* vy) {
    const BlockQ4K* x = (const BlockQ4K*) vx;
    const BlockQ8K* y = (const BlockQ8K*) vy;
    const int nb = n / 256;

    static constexpr uint32_t kmask1 = 0x3f3f3f3f;
    static constexpr uint32_t kmask2 = 0x0f0f0f0f;
    static constexpr uint32_t kmask3 = 0x03030303;
    uint32_t utmp[4];

    const __m512i m4 = _mm512_set1_epi8(0x0F);
    const __m512i shuf0 = scale_shuffle(0), shuf1 = scale_shuffle(1), shuf2 = scale_shuffle(2), shuf3 = scale_shuffle(3);
    __m256 acc = _mm256_setzero_ps();
    __m128 acc_m = _mm_setzero_ps();

    for (int i = 0; i < nb; ++i) {
        const float d = y[i].d * h2f(x[i].d);
        const float dmin = -y[i].d * h2f(x[i].dmin);

        std::memcpy(utmp, x[i].scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        const uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;

        const uint8_t* q4 = x[i].qs;
        const int8_t* q8 = y[i].qs;

        const __m256i mins_and_scales = _mm256_cvtepu8_epi16(_mm_set_epi32((int) utmp[3], (int) utmp[2], (int) utmp[1], (int) utmp[0]));
        const __m256i q8sums = _mm256_loadu_si256((const __m256i*) y[i].bsums);
        const __m128i q8s = _mm_hadd_epi16(_mm256_extracti128_si256(q8sums, 0), _mm256_extracti128_si256(q8sums, 1));
        const __m128i prod = _mm_madd_epi16(_mm256_extracti128_si256(mins_and_scales, 1), q8s);
        acc_m = _mm_fmadd_ps(_mm_set1_ps(dmin), _mm_cvtepi32_ps(prod), acc_m);

        const __m512i scales512 = _mm512_broadcast_i32x4(_mm256_castsi256_si128(mins_and_scales));
        __m512i sumi512;
        {
            const __m256i b0 = _mm256_loadu_si256((const __m256i*) (q4 + 0));
            const __m256i b1 = _mm256_loadu_si256((const __m256i*) (q4 + 32));
            const __m256i b2 = _mm256_loadu_si256((const __m256i*) (q4 + 64));
            const __m256i b3 = _mm256_loadu_si256((const __m256i*) (q4 + 96));
            const __m512i w0 = _mm512_and_si512(_mm512_inserti64x4(_mm512_castsi256_si512(b0), _mm256_srli_epi16(b0, 4), 1), m4);
            const __m512i w1 = _mm512_and_si512(_mm512_inserti64x4(_mm512_castsi256_si512(b1), _mm256_srli_epi16(b1, 4), 1), m4);
            const __m512i w2 = _mm512_and_si512(_mm512_inserti64x4(_mm512_castsi256_si512(b2), _mm256_srli_epi16(b2, 4), 1), m4);
            const __m512i w3 = _mm512_and_si512(_mm512_inserti64x4(_mm512_castsi256_si512(b3), _mm256_srli_epi16(b3, 4), 1), m4);
            const __m512i p0 = _mm512_maddubs_epi16(w0, _mm512_loadu_si512((const void*) (q8 + 0)));
            const __m512i p1 = _mm512_maddubs_epi16(w1, _mm512_loadu_si512((const void*) (q8 + 64)));
            const __m512i p2 = _mm512_maddubs_epi16(w2, _mm512_loadu_si512((const void*) (q8 + 128)));
            const __m512i p3 = _mm512_maddubs_epi16(w3, _mm512_loadu_si512((const void*) (q8 + 192)));
            const __m512i s0 = _mm512_madd_epi16(_mm512_shuffle_epi8(scales512, shuf0), p0);
            const __m512i s1 = _mm512_madd_epi16(_mm512_shuffle_epi8(scales512, shuf1), p1);
            const __m512i s2 = _mm512_madd_epi16(_mm512_shuffle_epi8(scales512, shuf2), p2);
            const __m512i s3 = _mm512_madd_epi16(_mm512_shuffle_epi8(scales512, shuf3), p3);
            sumi512 = _mm512_add_epi32(_mm512_add_epi32(s0, s1), _mm512_add_epi32(s2, s3));
        }
        const __m256i sumi = _mm256_add_epi32(_mm512_castsi512_si256(sumi512), _mm512_extracti64x4_epi64(sumi512, 1));
        acc = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi), acc);
    }

    acc_m = _mm_add_ps(acc_m, _mm_movehl_ps(acc_m, acc_m));
    acc_m = _mm_add_ss(acc_m, _mm_movehdup_ps(acc_m));
    *s = hsum_float_8(acc) + _mm_cvtss_f32(acc_m);
}

}  // namespace strata::kernels::cpu
