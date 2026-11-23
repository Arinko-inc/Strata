// sycl/src/kernels/cuda/fused_gr_req8.dp.cpp - SYCL port only: STRATA_HC_REQ8=1, the hyper-connection read of a verify
// window on int8 projections, in ESIMD (Battlemage).  See strata/kernels/fused_gr_req8.hpp.
//
// The default read (fused_gr.dp.cpp) is three launches of 32-wide sub-groups whose lanes gather BF16 chunks; on the B70
// it is latency bound (the up kernel: 6.55 MB in 23.7 us, 46% of the bandwidth).  Here a hardware thread owns 16 rows:
// lane l of a 16-wide register is row 16 g + l, and the table (hc_req8_pack) stores a block of 32 columns of those 16
// rows as 8 such registers, so a block is two contiguous 256-byte loads, eight dp4a against one broadcast dword of the
// activations each, and one fma with (the rows' scales * the activations' scale).  No gathers, no lane reductions.
//
// The maths is the RDNA read's (rdna_gr.cu with int8 activations), not the default kernels' order:
//   prep   (token, 128 columns): R' = R (+ the pending write, stored to R_out), v = R' * w_norm (NOT yet rms-scaled),
//          its int8 codes per 32 (d = amax / 127, round to nearest even) and scale, the slice's sum of R'^2, and the 4
//          BF16 inject rows on the fp32 v.
//   down   (16 rows, S blocks):  partial[token][row] = sum_b sw[row][b] * sa[token][b] * (codes . codes)
//   reduce (token, 32 rows):     rs[c] = rsqrt(ss[c] / N + eps);  lo = silu(sum_c rs[c] * partial[c] / HC), its int8
//          codes per 32; block 0 of a token also writes rs and inject_out.
//   up     (16 columns d):       for each stream c the 16 rows c * N + d of w_up on lo's codes, then the epilogue
//          mixed[d] = sum_c R'[c][d] * w_norm * rs[c] * sigmoid(.) / HC.
// The rms scale is per stream, so  w_down . (v * rs) = sum_c rs[c] * (w_down[:, c] . v[c]).
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>
#include "strata/sycl_queue.hpp"
#include "strata/kernels/fused_gr_req8.hpp"
#include "strata/kernels/verify_kernels.hpp"

namespace strata::kernels {

bool hc_req8_enabled() {
    static const bool on = [] {
        const char* v = std::getenv("STRATA_HC_REQ8");
        return v != nullptr && std::atoi(v) != 0 && hc_req8_built();
    }();
    return on;
}

}  // namespace strata::kernels

#ifdef STRATA_I8_NO_ESIMD
namespace strata::kernels {
bool hc_req8_built() { return false; }
bool fused_gr_req8_read_multi(const FusedGrArgs*, int, void*, unsigned long long*, int) { return false; }
}  // namespace strata::kernels
#else
namespace strata::kernels {
namespace {
using namespace sycl::ext::intel::esimd;

constexpr int N = 2560, HC = 4, D = N * HC, LR = 320;
constexpr int TM = kFusedGrMaxT;
constexpr int NBD = D / 32;          // 320 blocks a down row
constexpr int NBU = LR / 32;         // 10 blocks an up row
constexpr int RG = LR / 16;          // 20 row groups of down
constexpr int PP = 128;              // columns a prep thread
constexpr int NPC = N / PP;          // 20 prep threads a stream
constexpr int NSL_MAX = 40;          // down slices a row group at S = 8
constexpr int LOQ_ROW = 384;         // lo's codes a token, padded to whole loads

// the scratch of one queue (its launches are ordered; a graph capture records the pointers)
struct Scratch {
    uint8_t* qa = nullptr;    // [TM][D] codes of v
    float* sa = nullptr;      // [TM][NBD] their scales
    float* ss = nullptr;      // [TM][HC][32] sums of R'^2, 20 a stream
    float* inj = nullptr;     // [TM][HC][32][4] inject partials (4 rows) of a prep thread, 20 a stream
    float* part = nullptr;    // [TM][RG][NSL_MAX][16] down partials
    uint8_t* loq = nullptr;   // [TM][LOQ_ROW] codes of lo
    float* lod = nullptr;     // [TM][16] their scales
    float* rs = nullptr;      // [TM][HC]
};

Scratch& scratch_of(sycl::queue* q) {
    static std::mutex mu;
    static std::unordered_map<sycl::queue*, Scratch> all;
    std::lock_guard<std::mutex> lk(mu);
    auto it = all.find(q);
    if (it != all.end()) return it->second;
    Scratch s;
    auto get = [&](size_t bytes) {
        void* p = sycl::malloc_device(bytes, *q);
        if (p == nullptr) { std::fprintf(stderr, "strata: STRATA_HC_REQ8: no device memory for the read's scratch\n"); std::exit(1); }
        return p;   // not cleared (this can run while a graph is recorded): the kernels read only what they wrote
    };
    s.qa = (uint8_t*) get((size_t) TM * D);
    s.sa = (float*) get((size_t) TM * NBD * 4);
    s.ss = (float*) get((size_t) TM * HC * 32 * 4);
    s.inj = (float*) get((size_t) TM * HC * 32 * 4 * 4);
    s.part = (float*) get((size_t) TM * RG * NSL_MAX * 16 * 4);
    s.loq = (uint8_t*) get((size_t) TM * LOQ_ROW);
    s.lod = (float*) get((size_t) TM * 16 * 4);
    s.rs = (float*) get((size_t) TM * HC * 4 + 256);
    return all.emplace(q, s).first->second;
}

// what the kernels capture: the tokens' pointers by field, so a compile-time index names one kernel argument
struct M {
    const float* R[TM];
    float* R_out[TM];
    const float* bo[TM];
    const float* ip[TM];
    float* lo[TM];
    float* rs[TM];
    float* inj_out[TM];
    float* mixed[TM];
    int apply[TM];
    const float* w_norm;
    const uint16_t* w_inject;
    const uint8_t* wd;
    const float* sd;
    const uint8_t* wu;
    const float* su;
    float eps;
    int T;
    Scratch s;
};

// a run-time token index as a chain of selects over the 8 arguments (an indexed read would copy the struct to memory)
template <class P, std::size_t... J>
SYCL_ESIMD_FUNCTION inline P pick_impl(const P (&arr)[TM], int k, std::index_sequence<J...>) {
    P v = arr[0];
    ((v = (k == (int) J) ? arr[J] : v), ...);
    return v;
}
template <class P>
SYCL_ESIMD_FUNCTION inline P pick(const P (&arr)[TM], int k) { return pick_impl(arr, k, std::make_index_sequence<TM>{}); }

SYCL_ESIMD_FUNCTION inline simd<float, 16> fma16(simd<float, 16> a, simd<float, 16> b, simd<float, 16> c) {
    return simd<float, 16>(__builtin_elementwise_fma(a.data(), b.data(), c.data()));
}
SYCL_ESIMD_FUNCTION inline float hsum4(simd<float, 4> v) {
    simd<float, 2> c = v.template select<2, 1>(0) + v.template select<2, 1>(2);
    const float c0 = c[0], c1 = c[1];
    return c0 + c1;
}
SYCL_ESIMD_FUNCTION inline float hsum16(simd<float, 16> v) {
    simd<float, 8> a = v.template select<8, 1>(0) + v.template select<8, 1>(8);
    simd<float, 4> b = a.template select<4, 1>(0) + a.template select<4, 1>(4);
    simd<float, 2> c = b.template select<2, 1>(0) + b.template select<2, 1>(2);
    const float c0 = c[0], c1 = c[1];
    return c0 + c1;
}
// 32 fp32 values -> their int8 codes (d = amax / 127, round to nearest even) as 8 dwords, and d
SYCL_ESIMD_FUNCTION inline simd<uint32_t, 8> quant32(simd<float, 16> v0, simd<float, 16> v1, float& d) {
    simd<float, 16> m = (max)(abs(v0), abs(v1));
    const float amax = hmax<float, float, 16>(m);
    d = amax / 127.0f;
    const float id = amax > 0.0f ? 127.0f / amax : 0.0f;
    simd<int, 16> q0 = rnde<float, 16>(v0 * id), q1 = rnde<float, 16>(v1 * id);
    simd<int8_t, 32> b;
    b.template select<16, 1>(0) = simd<int8_t, 16>(q0);
    b.template select<16, 1>(16) = simd<int8_t, 16>(q1);
    simd<uint32_t, 8> r = b.template bit_cast_view<uint32_t>();
    return r;
}
// 16 bf16 (the low or high half of 32 words in 16 dwords would not be contiguous: take 16 uint16) -> fp32
SYCL_ESIMD_FUNCTION inline simd<float, 16> bf16_to_f32(simd<uint16_t, 16> h) {
    simd<uint32_t, 16> u = h;
    u = u << 16;
    simd<float, 16> f = u.template bit_cast_view<float>();
    return f;
}

struct PrepTag;
template <int S, int TT> struct DownTag;
template <int S> struct ReduceTag;
template <int TT> struct UpTag;

void launch_prep(sycl::queue& q, const M m) {
    q.parallel_for<PrepTag>(sycl::range<2>((size_t) m.T, (size_t) HC * NPC), [=](sycl::id<2> id) SYCL_ESIMD_KERNEL {
        const int k = (int) id[0], p = (int) id[1];
        const int c = p / NPC, pc = p - c * NPC, e0 = p * PP, d0 = e0 - c * N;
        const float* R = pick(m.R, k);
        const int apply = pick(m.apply, k);
        simd<float, 64> ra = block_load<float, 64>(R + e0), rb = block_load<float, 64>(R + e0 + 64);
        if (apply) {
            const float* bo = pick(m.bo, k);
            const float* ip = pick(m.ip, k);
            float* Ro = pick(m.R_out, k);
            const float ipc = ip[c] / (float) HC;
            simd<float, 1> ex = exp(simd<float, 1>(-ipc));
            const float e1 = ex[0];
            const float gw = 2.0f / (1.0f + e1);
            simd<float, 64> ba = block_load<float, 64>(bo + d0), bb = block_load<float, 64>(bo + d0 + 64);
            ra = simd<float, 64>(__builtin_elementwise_fma(ba.data(), simd<float, 64>(gw).data(), ra.data()));
            rb = simd<float, 64>(__builtin_elementwise_fma(bb.data(), simd<float, 64>(gw).data(), rb.data()));
            block_store(Ro + e0, ra);
            block_store(Ro + e0 + 64, rb);
        }
        simd<float, 64> na = block_load<float, 64>(m.w_norm + e0), nb = block_load<float, 64>(m.w_norm + e0 + 64);
        simd<float, 16> v[8], sq(0.0f);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            simd<float, 16> r0 = ra.template select<16, 1>(16 * j), r1 = rb.template select<16, 1>(16 * j);
            sq = fma16(r0, r0, sq);
            sq = fma16(r1, r1, sq);
            v[j] = r0 * na.template select<16, 1>(16 * j);
            v[4 + j] = r1 * nb.template select<16, 1>(16 * j);
        }
        m.s.ss[((size_t) k * HC + c) * 32 + pc] = hsum16(sq);
        simd<uint32_t, 32> codes;
        simd<float, 4> ds;
#pragma unroll
        for (int b = 0; b < 4; ++b) {
            float d;
            codes.template select<8, 1>(8 * b) = quant32(v[2 * b], v[2 * b + 1], d);
            ds[b] = d;
        }
        block_store((uint32_t*) (m.s.qa + (size_t) k * D + e0), codes);
        block_store(m.s.sa + (size_t) k * NBD + e0 / 32, ds);
        if (m.w_inject != nullptr) {
            simd<float, 4> pj;
#pragma unroll
            for (int r = 0; r < HC; ++r) {
                simd<uint32_t, 64> w = block_load<uint32_t, 64>((const uint32_t*) (m.w_inject + (size_t) r * D + e0));
                auto w16 = w.template bit_cast_view<uint16_t>();
                simd<float, 16> acc(0.0f);
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    simd<uint16_t, 16> h = w16.template select<16, 1>(16 * j);
                    acc = fma16(bf16_to_f32(h), v[j], acc);
                }
                pj[r] = hsum16(acc);
            }
            block_store(m.s.inj + (((size_t) k * HC + c) * 32 + pc) * 4, pj);
        }
    });
}

// S blocks (a multiple of 8) a thread: 320 / S slices a row group
template <int S, int TT>
void launch_down(sycl::queue& q, const M m) {
    constexpr int NSL = NBD / S;
    auto props = sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>};
    q.parallel_for<DownTag<S, TT>>(sycl::range<2>((size_t) RG, (size_t) NSL), props, [=](sycl::id<2> id) SYCL_ESIMD_KERNEL {
        const int rg = (int) id[0], sl = (int) id[1];
        const int T = m.T;
        simd<float, 16> facc[TT];
#pragma unroll
        for (int k = 0; k < TT; ++k) facc[k] = 0.0f;
#pragma unroll
        for (int ch = 0; ch < S / 8; ++ch) {
            const int b0 = sl * S + ch * 8;   // the chunk's first block
            simd<uint32_t, 64> A[TT];
            simd<float, 8> sA[TT];
#pragma unroll
            for (int k = 0; k < TT; ++k) {
                if (k < T) {
                    A[k] = block_load<uint32_t, 64>((const uint32_t*) (m.s.qa + (size_t) k * D + (size_t) b0 * 32));
                    sA[k] = block_load<float, 8>(m.s.sa + (size_t) k * NBD + b0);
                } else {
                    A[k] = 0u;
                    sA[k] = 0.0f;
                }
            }
            const uint32_t* wp = (const uint32_t*) (m.wd + ((size_t) rg * NBD + b0) * 512);
            const float* sp = m.sd + ((size_t) rg * NBD + b0) * 16;
#pragma unroll
            for (int b = 0; b < 8; ++b) {
                simd<uint32_t, 64> w0 = block_load<uint32_t, 64>(wp + b * 128);
                simd<uint32_t, 64> w1 = block_load<uint32_t, 64>(wp + b * 128 + 64);
                simd<float, 16> sw = block_load<float, 16>(sp + b * 16);
                simd<int, 16> W[8];
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    simd<uint32_t, 16> x0 = w0.template select<16, 1>(16 * j), x1 = w1.template select<16, 1>(16 * j);
                    W[j] = x0.template bit_cast_view<int>();
                    W[4 + j] = x1.template bit_cast_view<int>();
                }
#pragma unroll
                for (int k = 0; k < TT; ++k) {
                    simd<int, 16> ia(0), ib(0);
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        simd<uint32_t, 16> a0 = A[k].template replicate_w<16, 1>(b * 8 + j);
                        simd<uint32_t, 16> a1 = A[k].template replicate_w<16, 1>(b * 8 + 4 + j);
                        simd<int, 16> s0 = a0.template bit_cast_view<int>(), s1 = a1.template bit_cast_view<int>();
                        ia = dp4a<int>(ia, W[j], s0);
                        ib = dp4a<int>(ib, W[4 + j], s1);
                    }
                    simd<float, 16> f = ia + ib;
                    simd<float, 16> sc = sw * sA[k].template replicate_w<16, 1>(b);
                    facc[k] = fma16(f, sc, facc[k]);
                }
            }
        }
#pragma unroll
        for (int k = 0; k < TT; ++k)
            if (k < T) block_store(m.s.part + (((size_t) k * RG + rg) * NSL_MAX + sl) * 16, facc[k]);
    });
}

template <int S>
void launch_reduce(sycl::queue& q, const M m) {
    constexpr int NSL = NBD / S, NSC = NSL / HC;   // slices a row group, a stream
    q.parallel_for<ReduceTag<S>>(sycl::range<2>((size_t) m.T, (size_t) NBU), [=](sycl::id<2> id) SYCL_ESIMD_KERNEL {
        const int k = (int) id[0], b = (int) id[1];
        simd<float, 64> s0 = block_load<float, 64>(m.s.ss + (size_t) k * HC * 32);
        simd<float, 64> s1 = block_load<float, 64>(m.s.ss + (size_t) k * HC * 32 + 64);
        simd<float, 4> ssum;   // 20 partials a stream (the rows are padded to 32, the padding is never read)
        {
            simd<float, 16> a0 = s0.template select<16, 1>(0), a1 = s0.template select<16, 1>(32);
            simd<float, 16> a2 = s1.template select<16, 1>(0), a3 = s1.template select<16, 1>(32);
            simd<float, 4> t0 = s0.template select<4, 1>(16), t1 = s0.template select<4, 1>(48);
            simd<float, 4> t2 = s1.template select<4, 1>(16), t3 = s1.template select<4, 1>(48);
            ssum[0] = hsum16(a0) + hsum4(t0); ssum[1] = hsum16(a1) + hsum4(t1);
            ssum[2] = hsum16(a2) + hsum4(t2); ssum[3] = hsum16(a3) + hsum4(t3);
        }
        simd<float, 4> rs = simd<float, 4>(1.0f) / sqrt_ieee<float, 4>(ssum * (1.0f / (float) N) + m.eps);
        simd<float, 16> lo[2];
        float* alo = pick(m.lo, k);
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int rg = 2 * b + h;
            const float* pp = m.s.part + ((size_t) k * RG + rg) * NSL_MAX * 16;
            simd<float, 64> L[NSL / 4];   // the row group's slice partials, 4 slices a load
#pragma unroll
            for (int u = 0; u < NSL / 4; ++u) L[u] = block_load<float, 64>(pp + u * 64);
            simd<float, 16> x(0.0f);
#pragma unroll
            for (int c = 0; c < HC; ++c) {
                simd<float, 16> a(0.0f);
#pragma unroll
                for (int u = c * NSC; u < (c + 1) * NSC; ++u) a = a + L[u / 4].template select<16, 1>((u % 4) * 16);
                x = fma16(a, rs.template replicate_w<16, 1>(c), x);
            }
            x = x * (1.0f / (float) HC);
            simd<float, 16> e = exp(simd<float, 16>(0.0f) - x);
            lo[h] = x / (e + 1.0f);
            block_store(alo + rg * 16, lo[h]);
        }
        float d;
        simd<uint32_t, 8> qc = quant32(lo[0], lo[1], d);
        block_store((uint32_t*) (m.s.loq + (size_t) k * LOQ_ROW + b * 32), qc);
        m.s.lod[(size_t) k * 16 + b] = d;
        if (b == 0) {
            block_store(m.s.rs + (size_t) k * HC, rs);
            block_store(pick(m.rs, k), rs);
            if (m.w_inject != nullptr) {
                simd<float, 4> out(0.0f);
#pragma unroll
                for (int c = 0; c < HC; ++c) {
                    const float* ij = m.s.inj + ((size_t) k * HC + c) * 32 * 4;
                    simd<float, 64> i0 = block_load<float, 64>(ij), i1 = block_load<float, 64>(ij + 64);
                    // 20 partials of 4 rows (the padding past them is never read)
                    simd<float, 16> a = i0.template select<16, 1>(0) + i0.template select<16, 1>(16);
                    a = a + i0.template select<16, 1>(32);
                    a = a + i0.template select<16, 1>(48);
                    a = a + i1.template select<16, 1>(0);
                    simd<float, 4> r4 = a.template select<4, 1>(0) + a.template select<4, 1>(4);
                    r4 = r4 + a.template select<4, 1>(8);
                    r4 = r4 + a.template select<4, 1>(12);
                    out = out + r4 * rs.template replicate_w<4, 1>(c);
                }
                block_store(pick(m.inj_out, k), out);
            }
        }
    });
}

template <int TT>
void launch_up(sycl::queue& q, const M m) {
    auto props = sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>};
    q.parallel_for<UpTag<TT>>(sycl::range<1>((size_t) N / 16), props, [=](sycl::id<1> id) SYCL_ESIMD_KERNEL {
        const int d0 = (int) id[0] * 16;
        const int T = m.T;
        simd<float, 32> rsa = block_load<float, 32>(m.s.rs);
        simd<uint32_t, 64> Lq0[TT];
        simd<uint32_t, 16> Lq1[TT];
        simd<float, 16> ld[TT], g[TT];
#pragma unroll
        for (int k = 0; k < TT; ++k) {
            g[k] = 0.0f;
            if (k < T) {
                Lq0[k] = block_load<uint32_t, 64>((const uint32_t*) (m.s.loq + (size_t) k * LOQ_ROW));
                Lq1[k] = block_load<uint32_t, 16>((const uint32_t*) (m.s.loq + (size_t) k * LOQ_ROW + 256));
                ld[k] = block_load<float, 16>(m.s.lod + (size_t) k * 16);
            } else {
                Lq0[k] = 0u; Lq1[k] = 0u; ld[k] = 0.0f;
            }
        }
#pragma unroll
        for (int c = 0; c < HC; ++c) {
            const int i0 = c * N + d0;
            const size_t ig = (size_t) i0 / 16;
            const uint32_t* wp = (const uint32_t*) (m.wu + ig * NBU * 512);
            const float* sp = m.su + ig * NBU * 16;
            simd<float, 16> facc[TT];
#pragma unroll
            for (int k = 0; k < TT; ++k) facc[k] = 0.0f;
#pragma unroll
            for (int b = 0; b < NBU; ++b) {
                simd<uint32_t, 64> w0 = block_load<uint32_t, 64>(wp + b * 128);
                simd<uint32_t, 64> w1 = block_load<uint32_t, 64>(wp + b * 128 + 64);
                simd<float, 16> sw = block_load<float, 16>(sp + b * 16);
                simd<int, 16> W[8];
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    simd<uint32_t, 16> x0 = w0.template select<16, 1>(16 * j), x1 = w1.template select<16, 1>(16 * j);
                    W[j] = x0.template bit_cast_view<int>();
                    W[4 + j] = x1.template bit_cast_view<int>();
                }
#pragma unroll
                for (int k = 0; k < TT; ++k) {
                    simd<int, 16> ia(0), ib(0);
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        simd<uint32_t, 16> a0, a1;
                        if (b < 8) {
                            a0 = Lq0[k].template replicate_w<16, 1>(b * 8 + j);
                            a1 = Lq0[k].template replicate_w<16, 1>(b * 8 + 4 + j);
                        } else {
                            a0 = Lq1[k].template replicate_w<16, 1>((b - 8) * 8 + j);
                            a1 = Lq1[k].template replicate_w<16, 1>((b - 8) * 8 + 4 + j);
                        }
                        simd<int, 16> s0 = a0.template bit_cast_view<int>(), s1 = a1.template bit_cast_view<int>();
                        ia = dp4a<int>(ia, W[j], s0);
                        ib = dp4a<int>(ib, W[4 + j], s1);
                    }
                    simd<float, 16> f = ia + ib;
                    simd<float, 16> sc = sw * ld[k].template replicate_w<16, 1>(b);
                    facc[k] = fma16(f, sc, facc[k]);
                }
            }
            simd<float, 16> wn = block_load<float, 16>(m.w_norm + i0);
#pragma unroll
            for (int k = 0; k < TT; ++k) {
                if (k < T) {
                    const float* Rk = m.apply[k] ? m.R_out[k] : m.R[k];
                    simd<float, 16> rv = block_load<float, 16>(Rk + i0);
                    simd<float, 16> x = rv * wn * rsa.template replicate_w<16, 1>(k * HC + c);
                    simd<float, 16> e = exp(simd<float, 16>(0.0f) - facc[k]);
                    g[k] = g[k] + x / (e + 1.0f);
                }
            }
        }
#pragma unroll
        for (int k = 0; k < TT; ++k)
            if (k < T) block_store(m.mixed[k] + d0, g[k] * (1.0f / (float) HC));
    });
}

int slice_blocks() {   // STRATA_HC_REQ8_S: blocks of 32 columns a down thread (8 or 16)
    static const int v = [] {
        const char* e = std::getenv("STRATA_HC_REQ8_S");
        const int s = e ? std::atoi(e) : 8;
        return s == 16 ? 16 : 8;
    }();
    return v;
}

template <int S>
void launch_all(sycl::queue& q, const M& m, void* stream, unsigned long long* stamp_buf, int stamp_i0) {
    launch_prep(q, m);
    if (stamp_buf) gpu_stamp(stamp_buf, stamp_i0, stream);
    switch (m.T) {
        case 1: launch_down<S, 1>(q, m); break;
        case 2: launch_down<S, 2>(q, m); break;
        case 3: launch_down<S, 3>(q, m); break;
        case 4: launch_down<S, 4>(q, m); break;
        default: launch_down<S, TM>(q, m); break;
    }
    launch_reduce<S>(q, m);
    if (stamp_buf) gpu_stamp(stamp_buf, stamp_i0 + 1, stream);
    switch (m.T) {
        case 1: launch_up<1>(q, m); break;
        case 2: launch_up<2>(q, m); break;
        case 3: launch_up<3>(q, m); break;
        case 4: launch_up<4>(q, m); break;
        default: launch_up<TM>(q, m); break;
    }
}

}  // namespace

bool hc_req8_built() { return true; }

bool fused_gr_req8_read_multi(const FusedGrArgs* a, int n_tok, void* stream, unsigned long long* stamp_buf, int stamp_i0) {
    sycl::queue* q = strata::q_of(stream);
    M m{};
    for (int t = 0; t < TM; ++t) {
        const FusedGrArgs& x = a[t < n_tok ? t : 0];
        m.R[t] = x.R; m.R_out[t] = x.R_out; m.bo[t] = x.bo_prev; m.ip[t] = x.inj_prev;
        m.lo[t] = x.lo; m.rs[t] = x.rs; m.inj_out[t] = x.inject_out; m.mixed[t] = x.mixed;
        m.apply[t] = x.apply ? 1 : 0;
    }
    m.w_norm = a[0].w_norm;
    m.w_inject = a[0].w_inject;
    m.wd = reinterpret_cast<const uint8_t*>(a[0].w_down);
    m.sd = a[0].s_down;
    m.wu = reinterpret_cast<const uint8_t*>(a[0].w_up);
    m.su = a[0].s_up;
    m.eps = a[0].eps;
    m.T = n_tok;
    m.s = scratch_of(q);
    if (slice_blocks() == 16) launch_all<16>(*q, m, stream, stamp_buf, stamp_i0);
    else launch_all<8>(*q, m, stream, stamp_buf, stamp_i0);
    return true;
}

}  // namespace strata::kernels
#endif
