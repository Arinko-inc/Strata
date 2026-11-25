// src/prefill/moe_f16_q50.cu - see moe_f16_q50.hpp.  gfx11 only (opt-in STRATA_PF_F16_Q50=1).
//
// moe_f16_kq.cu's Q5_1 down kernel with Q5_0's block: {f16 d, qh (32 high bits), qs[16]} = 22 bytes, so a block does not
// start on a word (a row is 20 blocks = 440 bytes; the loads are unaligned), and the weight is d (q - 16) = q d + (-16 d):
// the Q5_1 kernel's packed FMA with m = -16 d (exact in F16), one rounding of the exact weight to F16.
//
// Adapted from gufo 97b8645f: src/models/qwen38_flash_next/kernels/rocm/kernels.hip.cpp
// (RoutedF16GEMMKernel and its helpers), through moe_f16_kq.cu.
// MIT License
// Copyright (c) 2026 gufo contributors
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
#include "moe_f16_q50.hpp"

#include "moe_f16_common.cuh"

#include <algorithm>
#include <cstdlib>

namespace strata::prefill::fused::f16q50 {
namespace {

using namespace f16;

constexpr int D_NS = 640 / 64;    // stages of 64 values (two Q5_0 blocks)
constexpr int BLOCK = 22;         // bytes of a block_q5_0

// Bits 0-3 of `bits` to bit 4 of each byte.
__device__ __forceinline__ uint32_t spread_high_bits(uint32_t bits) { return (__umul24(bits, 0x00204081U) & 0x01010101U) << 4U; }

constexpr uint32_t HALF_MAGIC = 0x64646464U;   // the high bytes of 1024.0 as F16
// Four code bytes (< 32) to four halves: 1024 + q exactly, minus 1024, then q * scale + bias.
__device__ __forceinline__ void codes_to_halves(uint32_t codes, __half2 magic, __half2 scale2, __half2 bias2, __half2& lo,
                                                __half2& hi) {
#if STRATA_F16_DEVICE
    const uint32_t p0 = __builtin_amdgcn_perm(codes, HALF_MAGIC, 0x01050004U);
    const uint32_t p1 = __builtin_amdgcn_perm(codes, HALF_MAGIC, 0x03070206U);
    lo = __hfma2(__hadd2(__builtin_bit_cast(__half2, p0), magic), scale2, bias2);
    hi = __hfma2(__hadd2(__builtin_bit_cast(__half2, p1), magic), scale2, bias2);
#else
    (void) codes; (void) magic; (void) scale2; (void) bias2; lo = hi = __half2{};
#endif
}

// Q5_0 down: 128 of the 2560 output rows against a tile's (or pair's) H rows, K = 640 in 10 stages; the routed rows'
// FP32 outputs in `dm`.
template <int BN>
__global__ void __launch_bounds__(THREADS) down_kernel(const Batch b, const NativeGeom geo, const int32_t* __restrict__ ts,
                                                       const int2* __restrict__ tiles, const int32_t* __restrict__ off,
                                                       const uint8_t* __restrict__ h, float* __restrict__ dm) {
#if STRATA_F16_DEVICE
    constexpr int CODE_BYTES = ROWS * 2 * 16, HIGH_BYTES = 2 * ROWS * 4, SCALE_BYTES = 2 * ROWS * 4;
    constexpr int STAGE_BYTES = CODE_BYTES + HIGH_BYTES + SCALE_BYTES + ActStage<BN>::BYTES;
    static_assert(D_EPI_BYTES <= STAGE_BYTES, "the epilogue fits the stage");
    __shared__ __attribute__((aligned(16))) uint8_t lds[STAGE_BYTES];
    __shared__ int srow[BN];
    auto* s_codes = reinterpret_cast<uint4*>(lds);
    auto* s_high = reinterpret_cast<uint32_t*>(lds + CODE_BYTES);
    auto* s_scale = reinterpret_cast<uint32_t*>(lds + CODE_BYTES + HIGH_BYTES);
    auto* s_act = reinterpret_cast<uint4*>(lds + CODE_BYTES + HIGH_BYTES + SCALE_BYTES);

    const int tid = threadIdx.x, wave = tid >> 5, sub_lane = tid & 15;
    const int f_c = tid & 1;   // this thread's K block of its weight row
    const auto swizzle = [](int row, int c) { return row * 2 + (c ^ ((row >> 2) & 1)); };
    const __half2 magic = __floats2half2_rn(-1024.0f, -1024.0f);

    int fb, e, row0, nrows;
    if (!work_item<BN, D_NFB>(b, ts, tiles, off, blockIdx.x, fb, e, row0, nrows)) return;
    const int live_tiles = (nrows + 15) / 16;
    if (tid < BN) srow[tid] = tid < nrows ? row0 + tid : -1;   // H is indexed by the routed row
    __syncthreads();

    const uint8_t* f_ptr = b.blob[e - b.e0] + geo.down_off + (size_t) (fb * ROWS + (tid >> 1)) * geo.d_row;
    uint4 f_codes;
    uint32_t f_high = 0, f_dm = 0;
    ActStage<BN> as;
    as.init(h, H_LD, srow, tid);
    const auto fetch_stage = [&](int s) {
        const uint8_t* blk = f_ptr + (2 * s + f_c) * BLOCK;   // block_q5_0: d, qh, qs[16] (unaligned)
        uint16_t d16;
        __builtin_memcpy(&d16, blk, 2);
        __builtin_memcpy(&f_high, blk + 2, 4);
        __builtin_memcpy(&f_codes, blk + 6, 16);
        // the half2 (d, -16 d): w = q d - 16 d
        const float d = __half2float(__builtin_bit_cast(__half, d16));
        f_dm = __builtin_bit_cast(uint32_t, __floats2half2_rn(d, -16.0f * d));
        as.fetch(s);
    };
    const auto commit_stage = [&]() {
        const int row = tid >> 1;
        s_codes[swizzle(row, f_c)] = f_codes;
        s_high[f_c * ROWS + row] = f_high;
        s_scale[f_c * ROWS + row] = f_dm;
        as.commit(s_act);
    };
    v8f acc[BN / 16];
#pragma unroll
    for (int j = 0; j < BN / 16; ++j) acc[j] = v8f{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    const auto compute_stage = [&]() {
        const int row = wave * 16 + sub_lane;
        const uint4 raw[2] = {s_codes[swizzle(row, 0)], s_codes[swizzle(row, 1)]};
#pragma unroll
        for (int kb = 0; kb < 2; ++kb) {
            const __half2 sb = __builtin_bit_cast(__half2, s_scale[kb * ROWS + row]);
            const __half2 scale2 = __low2half2(sb), bias2 = __high2half2(sb);
            // elements 0-15 are the low nibbles of the 16 code bytes, 16-31 the high, plus bit j of the high-bit word
            const uint32_t words[4] = {raw[kb].x, raw[kb].y, raw[kb].z, raw[kb].w};
            const uint32_t hb = s_high[kb * ROWS + row];
            __half2 hh[16];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const uint32_t lo = (words[i] & 0x0F0F0F0FU) | spread_high_bits((hb >> (4 * i)) & 0xFU);
                const uint32_t hi = ((words[i] >> 4U) & 0x0F0F0F0FU) | spread_high_bits((hb >> (16 + 4 * i)) & 0xFU);
                codes_to_halves(lo, magic, scale2, bias2, hh[2 * i], hh[2 * i + 1]);
                codes_to_halves(hi, magic, scale2, bias2, hh[8 + 2 * i], hh[8 + 2 * i + 1]);
            }
            v16h a_lo, a_hi;
            __builtin_memcpy(&a_lo, &hh[0], 32);
            __builtin_memcpy(&a_hi, &hh[8], 32);
            mma_tiles<BN>(acc, a_lo, a_hi, s_act, kb, sub_lane, live_tiles);
        }
    };
    fetch_stage(0);
    for (int s = 0; s < D_NS; ++s) {
        commit_stage();
        __syncthreads();
        if (s + 1 < D_NS) fetch_stage(s + 1);
        compute_stage();
        __syncthreads();
    }
    down_epilogue<BN>(acc, lds, tid, live_tiles, nrows, row0, fb, dm);
#else
    (void) b; (void) geo; (void) ts; (void) tiles; (void) off; (void) h; (void) dm;
    __builtin_trap();
#endif
}

}  // namespace

bool enabled() {
#if STRATA_F16_DEVICE
    static const bool on = [] {
        const char* v = std::getenv("STRATA_PF_F16_Q50");
        return v != nullptr && v[0] == '1';
    }();
    return on;
#else
    return false;
#endif
}

void down(const Batch& b, const NativeGeom& g, const int32_t* ts, const int2* tiles, const int32_t* off, int64_t n, int n_expert,
          int64_t work_bound, const uint8_t* h, float* dm, cudaStream_t s) {
    const unsigned grid = (unsigned) std::max<int64_t>(1, work_bound * D_NFB);
    if (pick_bn(n, n_expert) == 2 * kTileRows) down_kernel<2 * kTileRows><<<grid, THREADS, 0, s>>>(b, g, ts, tiles, off, h, dm);
    else down_kernel<kTileRows><<<grid, THREADS, 0, s>>>(b, g, ts, tiles, off, h, dm);
}

}  // namespace strata::prefill::fused::f16q50
