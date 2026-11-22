// The prompt path's GDN recurrence on ESIMD (Battlemage): see include/strata/prefill/esimd_gdn.hpp.
//
// The port's recurrence (kernels.dp.cpp: gdn_rec_cols_pipe_kernel) is the CUDA column kernel: a work-group of 128
// work-items per (value head, 32 columns), a work-item holding 32 rows of one column, the token's q and k rows passed
// through local memory, and five barriers per token (the Xe sub-group variant, one barrier, is slower than it on the
// B70: 19.21 against 18.76 s on a 57K prompt).  Here one hardware thread owns CW columns of a head's 128 x 128 state -
// 128 registers of CW floats - and walks the tokens alone:
//   kv_rg  = fma(s[r], k[r], kv_rg) over the 32 rows of row group rg, in order          (4 chains)
//   kv     = ((kv_0 + kv_1) + kv_2) + kv_3
//   delta  = (v - exp(gate) * kv) * beta
//   s[r]   = fma(exp(gate), s[r], k[r] * delta);   o_rg = fma(s[r], q[r], o_rg)          (in row order per group)
//   out    = (((o_0 + o_1) + o_2) + o_3) * rsqrt(128)
// operation for operation what the column kernel does per column.  The next token's q, k and v rows are loaded while
// this token computes.
//
// NOT IN USE (STRATA_GDN_ESIMD is off by default): on the B70 it gives the column kernel's bits (gdn_parity, state, output
// and fp16 output of 2,048 tokens, every form below) but takes 14.7 ms a layer against the column kernel's 3.2-4.2 ms.  Of
// the 14.7 ms the state updates (k * delta, fma into the 128 state registers) are 9.7, the output sums 4.7 and the kv
// sums - which only read the state - 0.4; eight columns a thread (768 threads instead of 384) takes 20 ms.  Why writing
// the state registers is slow was not found (the device compiler reports no spill; the loops are straight-line code;
// grouping the updates 16 rows at a time, nd_range instead of range, and software pipelining of the inputs change nothing).
#include <cstdint>
#include "strata/prefill/esimd_gdn.hpp"

#ifdef STRATA_I8_NO_ESIMD
namespace strata::prefill {
bool gdn_rec_esimd(float*, const float*, const float*, const float*, float*, int64_t, void*, int) { return false; }
}  // namespace strata::prefill
#else
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include "strata/sycl_queue.hpp"
#include <utility>

namespace strata::prefill {
namespace {
using namespace sycl::ext::intel::esimd;

constexpr int S = 128, HK = 16, HV = 48, C = 10240, RG = 4, RPG0 = S / RG;

template <int VAR, int CW> struct GdnRecTag;

// f(0), f(1), .. f(N - 1) as straight-line code with constant arguments (no reliance on `#pragma unroll` for 128 bodies)
template <class F, size_t... I>
SYCL_ESIMD_FUNCTION inline __attribute__((always_inline)) void each_impl(F&& f, std::index_sequence<I...>) { (f((int) I), ...); }
template <int N, class F>
SYCL_ESIMD_FUNCTION inline __attribute__((always_inline)) void each(F&& f) { each_impl(f, std::make_index_sequence<N>{}); }

template <int CW>
SYCL_ESIMD_FUNCTION inline simd<float, CW> fmaw(simd<float, CW> a, simd<float, CW> b, simd<float, CW> c) {
    return simd<float, CW>(__builtin_elementwise_fma(a.data(), b.data(), c.data()));
}

template <int VAR, int CW>
void launch(sycl::queue& q, float* state, const float* h, const float* gate, const float* beta, float* out, int64_t T) {
    auto props = sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>};
    q.submit([&](sycl::handler& hd) {
        hd.parallel_for<GdnRecTag<VAR, CW>>(sycl::nd_range<2>(sycl::range<2>((size_t) HV, (size_t) (S / CW)), sycl::range<2>(1, 1)), props,
                                            [=](sycl::nd_item<2> it) SYCL_ESIMD_KERNEL {
            const sycl::id<2> id = it.get_global_id();
            constexpr int RPG = RPG0;
            const size_t head = id[0], col0 = id[1] * CW;
            const size_t qh = head % HK;
            simd<float, CW> s[S];
            each<S>([&](int r) __attribute__((always_inline)) { s[r] = block_load<float, CW>(state + ((size_t) r * HV + head) * S + col0); });
            simd<float, CW> rs;
            if constexpr ((VAR & 4) != 0) rs = 0.088388346f;
            else rs = rsqrt(simd<float, CW>((float) S));
            // the token's rows: q and k (128 each, of the key head), v (the thread's columns)
            simd<float, 16> nq[S / 16], nk[S / 16];
            simd<float, CW> nv;
            const float* ht0 = h;
#pragma unroll
            for (int i = 0; i < S / 16; ++i) {
                nq[i] = block_load<float, 16>(ht0 + qh * S + 16 * i);
                nk[i] = block_load<float, 16>(ht0 + (size_t) HK * S + qh * S + 16 * i);
            }
            nv = block_load<float, CW>(ht0 + (size_t) 2 * HK * S + head * S + col0);
            for (int64_t t = 0; t < T; ++t) {
                simd<float, 16> cq[S / 16], ck[S / 16];
#pragma unroll
                for (int i = 0; i < S / 16; ++i) { cq[i] = nq[i]; ck[i] = nk[i]; }
                const simd<float, CW> cv = nv;
                if constexpr ((VAR & 16) == 0) {
                    if (t + 1 < T) {
                        const float* hn = h + (size_t) (t + 1) * C;
#pragma unroll
                        for (int i = 0; i < S / 16; ++i) {
                            nq[i] = block_load<float, 16>(hn + qh * S + 16 * i);
                            nk[i] = block_load<float, 16>(hn + (size_t) HK * S + qh * S + 16 * i);
                        }
                        nv = block_load<float, CW>(hn + (size_t) 2 * HK * S + head * S + col0);
                    }
                }
                const simd<float, CW> g = exp(simd<float, CW>(gate[(size_t) t * HV + head]));
                const simd<float, CW> bt(beta[(size_t) t * HV + head]);
                simd<float, CW> kv[RG];
#pragma unroll
                for (int rg = 0; rg < RG; ++rg) kv[rg] = 0.0f;
                if constexpr ((VAR & 64) == 0)   // (VAR 32 / 64: timing experiments, wrong output: no output sums / no kv sums)
                each<RG * RPG>([&](int n) __attribute__((always_inline)) {   // row r of the four groups, then row r + 1
                    const int rg = n % RG, i = rg * RPG0 + n / RG;
                    kv[rg] = fmaw<CW>(s[i], simd<float, CW>(ck[i / 16].template replicate_w<CW, 1>(i % 16)), kv[rg]);
                });
                const simd<float, CW> kvc = ((kv[0] + kv[1]) + kv[2]) + kv[3];
                simd<float, CW> delta;
                if constexpr ((VAR & 2) != 0) delta = (cv - g * kvc) * bt;
                else delta = fmaw<CW>(simd<float, CW>(0.0f) - g, kvc, cv) * bt;
                simd<float, CW> o[RG];
#pragma unroll
                for (int rg = 0; rg < RG; ++rg) o[rg] = 0.0f;
                // 16 rows at a time (4 rows of each group): their k * delta products, then their state updates, then their
                // output terms
                constexpr int NB = 16;
                each<RG * RPG / NB>([&](int blk) __attribute__((always_inline)) {
                    simd<float, CW> kd[NB];
                    each<NB>([&](int m) __attribute__((always_inline)) {
                        const int n = blk * NB + m, rg = n % RG, i = rg * RPG0 + n / RG;
                        kd[m] = simd<float, CW>(ck[i / 16].template replicate_w<CW, 1>(i % 16)) * delta;
                    });
                    each<NB>([&](int m) __attribute__((always_inline)) {
                        const int n = blk * NB + m, rg = n % RG, i = rg * RPG0 + n / RG;
                        s[i] = fmaw<CW>(g, s[i], kd[m]);
                    });
                    if constexpr ((VAR & 32) == 0)
                    each<NB>([&](int m) __attribute__((always_inline)) {
                        const int n = blk * NB + m, rg = n % RG, i = rg * RPG0 + n / RG;
                        o[rg] = fmaw<CW>(s[i], simd<float, CW>(cq[i / 16].template replicate_w<CW, 1>(i % 16)), o[rg]);
                    });
                });
                const simd<float, CW> oc = (((o[0] + o[1]) + o[2]) + o[3]) * rs;
                block_store<float, CW>(out + ((size_t) t * HV + head) * S + col0, oc);
                if constexpr ((VAR & 16) != 0) {
                    if (t + 1 < T) {
                        const float* hn = h + (size_t) (t + 1) * C;
#pragma unroll
                        for (int i = 0; i < S / 16; ++i) {
                            nq[i] = block_load<float, 16>(hn + qh * S + 16 * i);
                            nk[i] = block_load<float, 16>(hn + (size_t) HK * S + qh * S + 16 * i);
                        }
                        nv = block_load<float, CW>(hn + (size_t) 2 * HK * S + head * S + col0);
                    }
                }
            }
            each<S>([&](int r) __attribute__((always_inline)) { block_store<float, CW>(state + ((size_t) r * HV + head) * S + col0, s[r]); });
        });
    });
}
}  // namespace

bool gdn_rec_esimd(float* state, const float* h, const float* gate, const float* beta, float* out, int64_t T, void* stream, int var) {
    if (T <= 0 || (var & 1) == 0 || (var & ~127) != 0) return false;
    sycl::queue& q = *strata::q_of(stream);
    if (!q.get_device().has(sycl::aspect::ext_intel_esimd)) return false;
#define STRATA_GDN_L(V, W) launch<V, W>(q, state, h, gate, beta, out, T)
    switch (var) {
        case 1: STRATA_GDN_L(1, 16); break;
        case 3: STRATA_GDN_L(3, 16); break;
        case 5: STRATA_GDN_L(5, 16); break;
        case 7: STRATA_GDN_L(7, 16); break;
        case 9: STRATA_GDN_L(1, 8); break;
        case 11: STRATA_GDN_L(3, 8); break;
        case 17: STRATA_GDN_L(17, 16); break;
        case 25: STRATA_GDN_L(17, 8); break;
        case 33: STRATA_GDN_L(33, 16); break;
        case 65: STRATA_GDN_L(65, 16); break;
        case 97: STRATA_GDN_L(97, 16); break;
        default: return false;
    }
#undef STRATA_GDN_L
    return true;
}
}  // namespace strata::prefill
#endif
