// SYCL port (Battlemage): the prompt attention of an INT8 K/V cache on ESIMD.
//
// The port's prompt attention is the decode kernel (qsa_decode_attn: attn_chunk_kernel + attn_merge_kernel) as CUDA wrote
// it: a work-group of 256 work-items per (query, KV head, 64 cells), the 12 query heads copied into local memory by every
// group and read back from it for every cell (12 KB of local-memory reads per cell against 256 bytes of K), five sub-group
// shuffles per head and cell, and every chunk's 12 x 256 partial sums written to a scratch buffer for a second kernel to
// merge (0.8 MB written and read per query).
//
// Here the same arithmetic, operation for operation, runs from threads that keep their operands in registers:
//   scores  - a thread per (query, KV head, 6 query heads): the heads de-interleaved by 8 in registers (lane L of 32 holds
//             dimensions 8L .. 8L+7, as the warp kernel's lanes do), a cell's K row two 128-byte loads.  Per cell and head:
//               d_L = k[8L] q[8L] + ... + k[8L+7] q[8L+7]      (clang's fused form: fma(k7, q7, .. fma(k2, q2, fma(k0, q0, k1 q1))))
//               the xor-shuffle reduction at lane 0: s16[i] = d[i] + d[i+16], s8, s4, s2, s1   (halving tree), times 1/16
//             then per chunk of 64 cells and head: the maximum, exp(score - max) (0 for a masked cell), and their sum as
//             the warp kernel adds them (e[i] + e[i+32], then the halving tree).  The probabilities, maxima and sums go
//             to a scratch buffer (3 KB per chunk and thread).
//   values  - a thread per (query, KV head, 64 of the 256 dimensions = one INT8 scale group): the 12 heads' accumulators
//             for its dimensions in registers, a cell's V one 64-byte load; per chunk acc_c = fma(p, v, acc_c) cell by cell,
//             then the merge kernel's acc = fma(acc_c, exp(m_c - M), acc) and L = fma(l_c, exp(m_c - M), L), and acc / L
//             (an IEEE division, as the warp kernel's is).
// No local memory, no shuffles, no partial accumulators in memory.  Whether the bits equal the warp kernel's depends on the
// device compiler fusing the warp kernel's sums as written above and on both using the same exp and division: checked by
// qsa_prompt_attn_parity (the count of differing outputs) and on real prompts by stage_probe --bench (the chunk hashes).
// Measured on the B70 (oneAPI 2026.1, 2,048 queries of 2,051 cells): STRATA_ATTN_ESIMD=1 gives the warp kernel's bits; +2
// (unfused sums) gives the same bits too (the compiler fuses them the same way); +4 (the ESIMD `/`) differs in the last
// bit of 28% of the outputs.
// What the first version got wrong, for the next kernel written this way: a stride-8 `select` view of a loaded row kept as
// a register array is NOT put in registers - every product re-reads the strided region (the score kernel took 67 ms a
// layer against 11 ms with the rows de-interleaved by gathers).
#include "strata/kernels/qsa_prompt_attn_esimd.hpp"
#include "strata/core/geometry.hpp"
#include "strata/sycl_queue.hpp"
#include <cfloat>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#ifdef STRATA_I8_NO_ESIMD
namespace strata::kernels {
bool qsa_prompt_attn_esimd(const float*, const QsaAttnPools&, const int32_t*, const int32_t*, int64_t, const QsaShapes&, float*,
                           int64_t, void*) { return false; }
}  // namespace strata::kernels
#else
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include <map>
#include <mutex>

namespace strata::kernels {
namespace {
using namespace sycl::ext::intel::esimd;

#ifndef STRATA_ATTN_ESIMD_HP
#define STRATA_ATTN_ESIMD_HP 6
#endif
constexpr int HD = 256, G = 12, HP = STRATA_ATTN_ESIMD_HP, NP = G / HP, CH = 64;   // HP query heads per score thread, NP threads per KV head
constexpr int QB = 256;            // queries per launch (the scratch is QB queries')
constexpr float NEG = -FLT_MAX;
static_assert(HD == strata::core::kCompiledGeometry.head_dim &&
              G == strata::core::kCompiledGeometry.n_head / strata::core::kCompiledGeometry.n_head_kv,
              "qsa_prompt_attn_esimd is compiled for core/geometry.hpp");

struct Args {
    const float* q;
    const int8_t* kq;
    const int8_t* vq;
    const uint16_t* ks;
    const uint16_t* vs;
    const int32_t* page;
    const int32_t* ids;
    const int32_t* steps;
    float* scr;        // per (query, KV head): NP planes of per_hg floats (HP heads each)
    float* out;
    size_t per_hg;     // nch * (CH * 8 + 16): [chunk][cell][8] probabilities, [chunk][8] maxima, [chunk][8] sums
    int nkv, psh, pmask, cap, nch;
    float scale;
};

template <int VAR> struct AttnScoreTag;
template <int VAR> struct AttnValueTag;

SYCL_ESIMD_FUNCTION inline simd<float, 16> fma16(simd<float, 16> a, simd<float, 16> b, simd<float, 16> c) {
    return simd<float, 16>(__builtin_elementwise_fma(a.data(), b.data(), c.data()));
}
SYCL_ESIMD_FUNCTION inline simd<float, 8> fma8(simd<float, 8> a, simd<float, 8> b, simd<float, 8> c) {
    return simd<float, 8>(__builtin_elementwise_fma(a.data(), b.data(), c.data()));
}

template <int VAR>
void launch_scores(sycl::queue& q, const Args a, int64_t nb) {
    auto props = sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>};
    q.submit([&](sycl::handler& h) {
        h.parallel_for<AttnScoreTag<VAR>>(sycl::range<3>((size_t) nb, (size_t) a.nkv, (size_t) (G / HP)), props,
                                          [=](sycl::id<3> id) SYCL_ESIMD_KERNEL {
            const size_t qi = id[0];
            const int kvh = (int) id[1], hg = (int) id[2];
            const int n_ids = a.steps[qi * kStepCount + kStepWidth];
            const int nused = (n_ids + CH - 1) >> 6;
            if (nused <= 0) return;
            // the thread's query heads: element 8 L + j of head h -> lane L of Ql[h][j] (L < 16) or Qh[h][j]
            const float* qp = a.q + (qi * (size_t) (a.nkv * G) + (size_t) (kvh * G + hg * HP)) * HD;
            simd<float, 16> Ql[HP][8], Qh[HP][8];
#pragma unroll
            for (int hh = 0; hh < HP; ++hh) {
                // (gathers: a stride-8 view of a loaded row is re-read by every product instead of being put in a register)
                const simd<uint32_t, 16> o8(0u, 32u);
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    Ql[hh][j] = gather<float, 16>(qp + hh * HD, o8 + (uint32_t) (4 * j));
                    Qh[hh][j] = gather<float, 16>(qp + hh * HD + 128, o8 + (uint32_t) (4 * j));
                }
            }
            float* scr = a.scr + (qi * (size_t) a.nkv + (size_t) kvh) * (NP * a.per_hg) + (size_t) hg * a.per_hg;
            float* mp = scr + (size_t) a.nch * (CH * 8);
            float* lp = mp + (size_t) a.nch * 8;
            const int32_t* idp = a.ids + qi * (size_t) a.cap;
            for (int ch = 0; ch < nused; ++ch) {
                const int c0 = ch * CH;
                const int n_here = (n_ids - c0) < CH ? (n_ids - c0) : CH;
                float* es = scr + (size_t) ch * (CH * 8);
                for (int t = 0; t < CH; ++t) {
                    simd<float, 8> sv = NEG;
                    int page = -1, cell = 0;
                    if (t < n_here) {
                        cell = idp[c0 + t];
                        page = a.page[cell >> a.psh];
                    }
                    if (page >= 0) {
                        const size_t r = ((size_t) (page * a.nkv + kvh) << a.psh) + (size_t) (cell & a.pmask);
                        const uint32_t* kp = reinterpret_cast<const uint32_t*>(a.kq + r * HD);
                        simd<uint32_t, 32> kw0 = block_load<uint32_t, 32>(kp);
                        simd<uint32_t, 32> kw1 = block_load<uint32_t, 32>(kp + 32);
                        // lane L's 8 codes are dwords 2L and 2L+1: the even and the odd dwords each fill one register, and code j
                        // of every lane is byte j % 4 of its dword (shifted out, sign extended)
                        simd<uint32_t, 16> ke0 = kw0.template select<16, 2>(0), ko0 = kw0.template select<16, 2>(1);
                        simd<uint32_t, 16> ke1 = kw1.template select<16, 2>(0), ko1 = kw1.template select<16, 2>(1);
                        simd<int32_t, 16> ie0 = ke0, io0 = ko0, ie1 = ke1, io1 = ko1;
                        simd<uint16_t, 4> sb = block_load<uint16_t, 4>(a.ks + r * 4);
                        simd<sycl::half, 4> sh4 = sb.template bit_cast_view<sycl::half>();
                        simd<float, 4> sf = sh4;
                        simd<float, 16> Sl, Sh;
                        Sl.template select<8, 1>(0) = simd<float, 8>((float) sf[0]);
                        Sl.template select<8, 1>(8) = simd<float, 8>((float) sf[1]);
                        Sh.template select<8, 1>(0) = simd<float, 8>((float) sf[2]);
                        Sh.template select<8, 1>(8) = simd<float, 8>((float) sf[3]);
                        simd<float, 16> Kl[8], Kh[8];
#pragma unroll
                        for (int j = 0; j < 8; ++j) {
                            const int sh = 24 - 8 * (j & 3);
                            simd<int32_t, 16> cl = ((j < 4 ? ie0 : io0) << sh) >> 24;
                            simd<int32_t, 16> chh = ((j < 4 ? ie1 : io1) << sh) >> 24;
                            simd<float, 16> fl = cl, fh = chh;
                            Kl[j] = fl * Sl;
                            Kh[j] = fh * Sh;
                        }
                        // (the 2 * HP sums advance together, one term at a time)
                        simd<float, 16> dl[HP], dh[HP];
                        if constexpr ((VAR & 2) == 0) {
#pragma unroll
                            for (int hh = 0; hh < HP; ++hh) {
                                dl[hh] = Kl[1] * Ql[hh][1];
                                dh[hh] = Kh[1] * Qh[hh][1];
                            }
#pragma unroll
                            for (int hh = 0; hh < HP; ++hh) {
                                dl[hh] = fma16(Kl[0], Ql[hh][0], dl[hh]);
                                dh[hh] = fma16(Kh[0], Qh[hh][0], dh[hh]);
                            }
#pragma unroll
                            for (int j = 2; j < 8; ++j) {
#pragma unroll
                                for (int hh = 0; hh < HP; ++hh) {
                                    dl[hh] = fma16(Kl[j], Ql[hh][j], dl[hh]);
                                    dh[hh] = fma16(Kh[j], Qh[hh][j], dh[hh]);
                                }
                            }
                        } else {
#pragma unroll
                            for (int hh = 0; hh < HP; ++hh) {
                                dl[hh] = Kl[0] * Ql[hh][0];
                                dh[hh] = Kh[0] * Qh[hh][0];
                            }
#pragma unroll
                            for (int j = 1; j < 8; ++j) {
#pragma unroll
                                for (int hh = 0; hh < HP; ++hh) {
                                    dl[hh] = dl[hh] + Kl[j] * Ql[hh][j];
                                    dh[hh] = dh[hh] + Kh[j] * Qh[hh][j];
                                }
                            }
                        }
                        simd<float, 16> s16[HP];
                        simd<float, 8> s8[HP];
                        simd<float, 4> s4[HP];
                        simd<float, 2> s2[HP];
#pragma unroll
                        for (int hh = 0; hh < HP; ++hh) s16[hh] = dl[hh] + dh[hh];
#pragma unroll
                        for (int hh = 0; hh < HP; ++hh) s8[hh] = s16[hh].template select<8, 1>(0) + s16[hh].template select<8, 1>(8);
#pragma unroll
                        for (int hh = 0; hh < HP; ++hh) s4[hh] = s8[hh].template select<4, 1>(0) + s8[hh].template select<4, 1>(4);
#pragma unroll
                        for (int hh = 0; hh < HP; ++hh) s2[hh] = s4[hh].template select<2, 1>(0) + s4[hh].template select<2, 1>(2);
#pragma unroll
                        for (int hh = 0; hh < HP; ++hh) {
                            const float s0 = s2[hh][0], s1 = s2[hh][1];
                            sv[hh] = (s0 + s1) * a.scale;
                        }
                    }
                    block_store<float, 8>(es + t * 8, sv);
                }
                // the chunk's 64 cells per head: maximum, exp(score - max), their sum
                simd<float, 512> SC;
#pragma unroll
                for (int i = 0; i < 8; ++i) SC.template select<64, 1>(64 * i) = block_load<float, 64>(es + 64 * i);
                simd<float, 8> mv = NEG, lv = 0.0f;
#pragma unroll
                for (int hh = 0; hh < HP; ++hh) {
                    simd<float, 64> S = SC.template select<64, 8>(hh);
                    const float m = hmax<float>(S);
                    simd<float, 64> e;
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        simd<float, 16> x = S.template select<16, 1>(16 * i);
                        simd<float, 16> ex = exp(x - m);
                        ex.merge(simd<float, 16>(0.0f), x == NEG);
                        e.template select<16, 1>(16 * i) = ex;
                    }
                    simd<float, 32> t32 = e.template select<32, 1>(0) + e.template select<32, 1>(32);
                    simd<float, 16> l16 = t32.template select<16, 1>(0) + t32.template select<16, 1>(16);
                    simd<float, 8> l8 = l16.template select<8, 1>(0) + l16.template select<8, 1>(8);
                    simd<float, 4> l4 = l8.template select<4, 1>(0) + l8.template select<4, 1>(4);
                    simd<float, 2> l2 = l4.template select<2, 1>(0) + l4.template select<2, 1>(2);
                    const float la = l2[0], lb = l2[1];
                    SC.template select<64, 8>(hh) = e;
                    mv[hh] = m;
                    lv[hh] = la + lb;
                }
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    simd<float, 64> o = SC.template select<64, 1>(64 * i);
                    block_store<float, 64>(es + 64 * i, o);
                }
                block_store<float, 8>(mp + ch * 8, mv);
                block_store<float, 8>(lp + ch * 8, lv);
            }
        });
    });
}

template <int VAR>
void launch_values(sycl::queue& q, const Args a, int64_t nb) {
    auto props = sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>};
    q.submit([&](sycl::handler& h) {
        h.parallel_for<AttnValueTag<VAR>>(sycl::range<3>((size_t) nb, (size_t) a.nkv, (size_t) (HD / 64)), props,
                                          [=](sycl::id<3> id) SYCL_ESIMD_KERNEL {
            const size_t qi = id[0];
            const int kvh = (int) id[1], sl = (int) id[2];
            const int n_ids = a.steps[qi * kStepCount + kStepWidth];
            const int nused = (n_ids + CH - 1) >> 6;
            float* o = a.out + (qi * (size_t) (a.nkv * G) + (size_t) (kvh * G)) * HD + (size_t) (sl * 64);
            const float* p0 = a.scr + (qi * (size_t) a.nkv + (size_t) kvh) * (NP * a.per_hg);
            const size_t moff = (size_t) a.nch * (CH * 8), loff = moff + (size_t) a.nch * 8;
            const int32_t* idp = a.ids + qi * (size_t) a.cap;
            simd<float, 8> Mv[NP], Lv[NP];
#pragma unroll
            for (int pl = 0; pl < NP; ++pl) { Mv[pl] = NEG; Lv[pl] = 0.0f; }
            for (int ch = 0; ch < nused; ++ch) {
#pragma unroll
                for (int pl = 0; pl < NP; ++pl) Mv[pl] = max<float, 8>(Mv[pl], block_load<float, 8>(p0 + pl * a.per_hg + moff + ch * 8));
            }
            simd<float, 16> ACC[G][4];
#pragma unroll
            for (int hh = 0; hh < G; ++hh)
#pragma unroll
                for (int i = 0; i < 4; ++i) ACC[hh][i] = 0.0f;
            for (int ch = 0; ch < nused; ++ch) {
                const int c0 = ch * CH;
                const int n_here = (n_ids - c0) < CH ? (n_ids - c0) : CH;
                simd<float, 8> mc[NP], wc[NP];
#pragma unroll
                for (int pl = 0; pl < NP; ++pl) {
                    mc[pl] = block_load<float, 8>(p0 + pl * a.per_hg + moff + ch * 8);
                    simd<float, 8> lc = block_load<float, 8>(p0 + pl * a.per_hg + loff + ch * 8);
                    wc[pl] = exp(mc[pl] - Mv[pl]);
                    Lv[pl].merge(fma8(lc, wc[pl], Lv[pl]), mc[pl] != NEG);
                }
                simd<float, 16> AC[G][4];
#pragma unroll
                for (int hh = 0; hh < G; ++hh)
#pragma unroll
                    for (int i = 0; i < 4; ++i) AC[hh][i] = 0.0f;
                const float* pc = p0 + (size_t) ch * (CH * 8);
                for (int t = 0; t < n_here; ++t) {
                    const int cell = idp[c0 + t];
                    const int page = a.page[cell >> a.psh];
                    if (page < 0) continue;
                    const size_t r = ((size_t) (page * a.nkv + kvh) << a.psh) + (size_t) (cell & a.pmask);
                    simd<uint32_t, 16> vw = block_load<uint32_t, 16>(reinterpret_cast<const uint32_t*>(a.vq + r * HD + (size_t) (sl * 64)));
                    auto vb = vw.template bit_cast_view<int8_t>();
                    const uint16_t sbits = a.vs[r * 4 + (size_t) sl];
                    simd<uint16_t, 16> sbv = sbits;
                    simd<sycl::half, 16> shv = sbv.template bit_cast_view<sycl::half>();
                    simd<float, 16> sc = shv;
                    simd<float, 16> v[4];
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        simd<int8_t, 16> c = vb.template select<16, 1>(16 * i);
                        simd<float, 16> f = c;
                        v[i] = f * sc;
                    }
#pragma unroll
                    for (int pl = 0; pl < NP; ++pl) {
                        simd<float, 8> pa = block_load<float, 8>(pc + pl * a.per_hg + t * 8);
#pragma unroll
                        for (int hh = 0; hh < HP; ++hh) {
                            simd<float, 16> ph((float) pa[hh]);
#pragma unroll
                            for (int i = 0; i < 4; ++i) AC[pl * HP + hh][i] = fma16(ph, v[i], AC[pl * HP + hh][i]);
                        }
                    }
                }
#pragma unroll
                for (int hh = 0; hh < G; ++hh) {
                    const float mh = (float) mc[hh / HP][hh % HP];
                    if (mh != NEG) {
                        simd<float, 16> wv((float) wc[hh / HP][hh % HP]);
#pragma unroll
                        for (int i = 0; i < 4; ++i) ACC[hh][i] = fma16(AC[hh][i], wv, ACC[hh][i]);
                    }
                }
            }
#pragma unroll
            for (int hh = 0; hh < G; ++hh) {
                const float Lh = (float) Lv[hh / HP][hh % HP];
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    simd<float, 16> ov = 0.0f;
                    if (Lh > 0.0f) {
                        if constexpr ((VAR & 4) == 0) ov = div_ieee<float, 16>(ACC[hh][i], simd<float, 16>(Lh));
                        else ov = ACC[hh][i] / simd<float, 16>(Lh);
                    }
                    block_store<float, 16>(o + hh * HD + 16 * i, ov);
                }
            }
        });
    });
}

struct Scratch { float* p = nullptr; size_t floats = 0; };
}  // namespace

bool qsa_prompt_attn_esimd(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* steps, int64_t cap,
                           const QsaShapes& s, float* attn, int64_t n_q, void* stream) {
    static const int var = [] { const char* v = std::getenv("STRATA_ATTN_ESIMD"); return v != nullptr && v[0] != 0 ? std::atoi(v) : 0; }();
    if ((var & 1) == 0 || (var & ~7) != 0 || n_q <= 0) return false;
    if (pools.k_q == nullptr || pools.v_q == nullptr || pools.k_scale == nullptr || pools.v_scale == nullptr || pools.k_q4 != nullptr ||
        pools.v_q4 != nullptr || pools.k_pool != nullptr || pools.page_table == nullptr || !ids || !steps || cap <= 0 || s.head_dim != HD ||
        s.n_head != (int64_t) G * s.n_head_kv || s.page_size <= 0 || (s.page_size & (s.page_size - 1)) != 0 || s.n_head_kv > 64)
        return false;
    sycl::queue& qu = *strata::q_of(stream);
    if (!qu.get_device().has(sycl::aspect::ext_intel_esimd)) return false;
    Args a{};
    a.q = q; a.kq = pools.k_q; a.vq = pools.v_q; a.ks = pools.k_scale; a.vs = pools.v_scale; a.page = pools.page_table;
    a.nkv = (int) s.n_head_kv;
    a.psh = 0;
    while (((int64_t) 1 << a.psh) < s.page_size) ++a.psh;
    a.pmask = (int) s.page_size - 1;
    a.cap = (int) cap;
    a.nch = (int) ((cap + CH - 1) / CH);
    a.per_hg = (size_t) a.nch * (CH * 8 + 16);
    a.scale = 1.0f / 16.0f;   // 1 / sqrt(HD)
    const size_t per_q = (size_t) a.nkv * NP * a.per_hg;
    float* scr = nullptr;
    {   // the scratch of this queue (an in-order queue runs one prompt chunk at a time), grown when the selection widens
        static std::mutex mu;
        static std::map<const void*, Scratch> per_q_scr;
        std::lock_guard<std::mutex> lk(mu);
        Scratch& sc = per_q_scr[(const void*) &qu];
        const size_t need = (size_t) QB * per_q;
        if (sc.floats < need) {
            if (sc.p != nullptr) { qu.wait(); sycl::free(sc.p, qu); sc.p = nullptr; sc.floats = 0; }
            sc.p = (float*) sycl::malloc_device(need * sizeof(float), qu);
            if (sc.p == nullptr) return false;
            sc.floats = need;
        }
        scr = sc.p;
    }
    if (static bool told = false; !told) {
        told = true;
        std::fprintf(stderr, "strata: prompt attention on ESIMD (STRATA_ATTN_ESIMD=%d): %d heads a score thread, %d queries a launch, %.1f MiB of scratch\n", var, HP,
                     (int) QB, (double) ((size_t) QB * per_q * 4) / 1048576.0);
    }
    a.scr = scr;
    for (int64_t t0 = 0; t0 < n_q; t0 += QB) {
        const int64_t nb = n_q - t0 < QB ? n_q - t0 : QB;
        a.q = q + t0 * s.n_head * HD;
        a.ids = ids + t0 * cap;
        a.steps = steps + t0 * kStepCount;
        a.out = attn + t0 * s.n_head * HD;
        // STRATA_ATTN_ESIMD_PART=1 / 2 (timing only, wrong output): the score kernel alone / the value kernel alone
        static const int part = [] { const char* v = std::getenv("STRATA_ATTN_ESIMD_PART"); return v ? std::atoi(v) : 0; }();
        if (part != 2) { if (var & 2) launch_scores<3>(qu, a, nb); else launch_scores<1>(qu, a, nb); }
        if (part != 1) { if (var & 4) launch_values<5>(qu, a, nb); else launch_values<1>(qu, a, nb); }
    }
    return true;
}
}  // namespace strata::kernels
#endif
