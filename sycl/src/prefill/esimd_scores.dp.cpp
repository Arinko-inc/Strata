// The prompt path's QSA block scores on ESIMD (Battlemage): see include/strata/prefill/esimd_scores.hpp.
//
// The port's scorer (kernels/cuda/qsa_select: block_scores_kernel) is the CUDA warp kernel as it was written: 32 work-items per
// (query, block), each reading 4 of the key's 128 values and 16 of the query's, and five sub-group shuffles per indexer head.
// Here one hardware thread owns one query and a tile of NB blocks: the query's four heads are loaded once into registers (de-
// interleaved by 4), and each block's key is one 512-byte load.  The arithmetic is the warp kernel's, operation for operation:
//   lane L (0..31) of head h:  d_L = k[4L] q[4L] + k[4L+1] q[4L+1] + k[4L+2] q[4L+2] + k[4L+3] q[4L+3]   (left to right)
//   the xor-shuffle reduction at lane 0: s16[i] = d[i] + d[i+16], s8[i] = s16[i] + s16[i+8], s4, s2, s1   (halving tree)
//   score = sum over h (in order) of max(s1, 0);  the tail block reads `dead` and has 1e9 added while it is incomplete.
// `variant` picks how the four products of a lane are summed (the compiler decides this for the warp kernel; the one that gives
// the same bits on this GPU is the default, checked on real prompts with STRATA_SCORES_CHECK=1):
//   1: unfused       ((p0 + p1) + p2) + p3
//   2: fused         fma(k3, q3, fma(k2, q2, fma(k0, q0, p1)))     (clang's a*b + c*d: the left product is the fused one)
//   3: fused         fma(k3, q3, fma(k2, q2, fma(k1, q1, p0)))
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include <cstdint>
#include "strata/sycl_queue.hpp"
#include "strata/prefill/esimd_moe.hpp"
#include "strata/prefill/esimd_scores.hpp"

#ifdef STRATA_I8_NO_ESIMD
namespace strata::prefill::i8 {
bool block_scores(const float*, const float*, const float*, const int32_t*, const ScoreSteps&, int64_t, int64_t, float*, void*,
                  int64_t, int) { return false; }
}  // namespace strata::prefill::i8
#else
namespace strata::prefill::i8 {
namespace {
using namespace sycl::ext::intel::esimd;

constexpr int DIM = 128, HEADS = 4, BLOCK = 4, NB = 16;

template <int V, int GM> struct ScoresTag;

SYCL_ESIMD_FUNCTION inline simd<float, 16> fma16(simd<float, 16> a, simd<float, 16> b, simd<float, 16> c) {
    return simd<float, 16>(__builtin_elementwise_fma(a.data(), b.data(), c.data()));
}

// GM (variant + 8, + 16): the de-interleaved rows read by gathers.  A stride-4 `select` view of a loaded row kept in an array is not put
// in a register - every product re-reads the strided region (found on the prompt attention, qsa_prompt_attn_esimd.dp.cpp).  Bit 0:
// the query (once per thread); bit 1: the keys too (eight gathers per block instead of two loads).  The same values, the same sums.
template <int V, int GM>
void launch(sycl::queue& q, const float* pooled, const float* dead, const float* q_idx, const int32_t* steps, const ScoreSteps ss,
            int64_t nq, int64_t max_blocks, float* out, int64_t reach) {
    auto props = sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>};
    const int count = ss.count, o_nkv = ss.n_kv, o_nbid = ss.n_bid;
    q.submit([&](sycl::handler& h) {
        h.parallel_for<ScoresTag<V, GM>>(sycl::range<2>((size_t) nq, (size_t) ((reach + NB - 1) / NB)), props, [=](sycl::id<2> id) SYCL_ESIMD_KERNEL {
            const int64_t qi = (int64_t) id[0];
            const int64_t b0 = (int64_t) id[1] * NB;
            const int32_t* st = steps + qi * count;
            const int64_t n_kv = st[o_nkv], n_bid = st[o_nbid];
            if (b0 > n_bid || b0 >= max_blocks) return;
            // the query: head h, element 4 L + c -> Q[h][c] lane L (two halves of 16 lanes)
            simd<float, 16> Ql[HEADS][4], Qh[HEADS][4];
            const float* qp = q_idx + qi * (HEADS * DIM);
#pragma unroll
            for (int hh = 0; hh < HEADS; ++hh) {
                if constexpr ((GM & 1) != 0) {
                    const simd<uint32_t, 16> o4(0u, 16u);
#pragma unroll
                    for (int c = 0; c < 4; ++c) {
                        Ql[hh][c] = gather<float, 16>(qp + hh * DIM, o4 + (uint32_t) (4 * c));
                        Qh[hh][c] = gather<float, 16>(qp + hh * DIM + 64, o4 + (uint32_t) (4 * c));
                    }
                } else {
                simd<float, 64> a = block_load<float, 64>(qp + hh * DIM);
                simd<float, 64> b = block_load<float, 64>(qp + hh * DIM + 64);
#pragma unroll
                for (int c = 0; c < 4; ++c) {
                    Ql[hh][c] = a.template select<16, 4>(c);
                    Qh[hh][c] = b.template select<16, 4>(c);
                }
                }
            }
            for (int i = 0; i < NB; ++i) {
                const int64_t b = b0 + i;
                if (b > n_bid || b >= max_blocks) break;
                const float* key = (b == n_bid) ? dead : pooled + b * DIM;
                simd<float, 16> Kl[4], Kh[4];
                if constexpr ((GM & 2) != 0) {
                    const simd<uint32_t, 16> o4(0u, 16u);
#pragma unroll
                    for (int c = 0; c < 4; ++c) {
                        Kl[c] = gather<float, 16>(key, o4 + (uint32_t) (4 * c));
                        Kh[c] = gather<float, 16>(key + 64, o4 + (uint32_t) (4 * c));
                    }
                } else {
                simd<float, 64> ka = block_load<float, 64>(key);
                simd<float, 64> kb = block_load<float, 64>(key + 64);
#pragma unroll
                for (int c = 0; c < 4; ++c) {
                    Kl[c] = ka.template select<16, 4>(c);
                    Kh[c] = kb.template select<16, 4>(c);
                }
                }
                float score = 0.0f;
#pragma unroll
                for (int hh = 0; hh < HEADS; ++hh) {
                    simd<float, 16> dl, dh;
                    if constexpr (V == 1) {
                        dl = Kl[0] * Ql[hh][0];
                        dl = dl + Kl[1] * Ql[hh][1];
                        dl = dl + Kl[2] * Ql[hh][2];
                        dl = dl + Kl[3] * Ql[hh][3];
                        dh = Kh[0] * Qh[hh][0];
                        dh = dh + Kh[1] * Qh[hh][1];
                        dh = dh + Kh[2] * Qh[hh][2];
                        dh = dh + Kh[3] * Qh[hh][3];
                    } else if constexpr (V == 2) {
                        dl = fma16(Kl[0], Ql[hh][0], Kl[1] * Ql[hh][1]);
                        dl = fma16(Kl[2], Ql[hh][2], dl);
                        dl = fma16(Kl[3], Ql[hh][3], dl);
                        dh = fma16(Kh[0], Qh[hh][0], Kh[1] * Qh[hh][1]);
                        dh = fma16(Kh[2], Qh[hh][2], dh);
                        dh = fma16(Kh[3], Qh[hh][3], dh);
                    } else {
                        dl = fma16(Kl[1], Ql[hh][1], Kl[0] * Ql[hh][0]);
                        dl = fma16(Kl[2], Ql[hh][2], dl);
                        dl = fma16(Kl[3], Ql[hh][3], dl);
                        dh = fma16(Kh[1], Qh[hh][1], Kh[0] * Qh[hh][0]);
                        dh = fma16(Kh[2], Qh[hh][2], dh);
                        dh = fma16(Kh[3], Qh[hh][3], dh);
                    }
                    simd<float, 16> s16 = dl + dh;
                    simd<float, 8> s8 = s16.template select<8, 1>(0) + s16.template select<8, 1>(8);
                    simd<float, 4> s4 = s8.template select<4, 1>(0) + s8.template select<4, 1>(4);
                    simd<float, 2> s2 = s4.template select<2, 1>(0) + s4.template select<2, 1>(2);
                    const float s0 = s2[0], s1 = s2[1];
                    const float d = s0 + s1;
                    score += d > 0.0f ? d : 0.0f;
                }
                if (b == n_bid && (n_kv & (BLOCK - 1)) != 0) score += 1e9f;   // (a constant: `%` by a run-time value misbehaves in ESIMD here)
                out[qi * max_blocks + b] = score;
            }
        });
    });
}
}  // namespace

bool block_scores(const float* pooled, const float* dead, const float* q_idx, const int32_t* steps, const ScoreSteps& ss, int64_t nq,
                  int64_t max_blocks, float* scores, void* stream, int64_t active_blocks, int variant) {
    if (nq <= 0) return true;
    const int gm = variant >> 3;
    variant &= 7;
    if (gm < 0 || gm > 3 || variant < 1 || variant > 3 || ss.dim != DIM || ss.heads != HEADS || ss.block != BLOCK || !available(stream)) return false;
    sycl::queue& q = *strata::q_of(stream);
    const int64_t reach = active_blocks > 0 && active_blocks < max_blocks ? active_blocks : max_blocks;
#define STRATA_SC_L(V, G) launch<V, G>(q, pooled, dead, q_idx, steps, ss, nq, max_blocks, scores, reach)
    if (gm == 0) { if (variant == 1) STRATA_SC_L(1, 0); else if (variant == 2) STRATA_SC_L(2, 0); else STRATA_SC_L(3, 0); }
    else if (variant != 2) return false;   // the gathered forms exist for the variant in use only
    else if (gm == 1) STRATA_SC_L(2, 1);
    else if (gm == 2) STRATA_SC_L(2, 2);
    else STRATA_SC_L(2, 3);
#undef STRATA_SC_L
    return true;
}
}  // namespace strata::prefill::i8
#endif
