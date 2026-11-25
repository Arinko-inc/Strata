// The prompt path's QSA block scores, several queries a hardware thread (Battlemage, ESIMD): STRATA_SCORES_MQ=<queries>.
//
// esimd_scores.dp.cpp gives a thread one query and 16 blocks: per (query, block) it reads the block's key (two 256-byte loads, or
// eight gathers), de-interleaves it by 4 and writes one score.  The scores grow with the position (every query scores every
// completed block): at 55K tokens they are ~105 ms of a 3,072-token chunk's ~860 ms on the B70's 18 layers.
// Here:
//   * `prepare` copies the chunk's queries and the completed blocks' keys once per QSA layer into the order the products read them
//     (component c of lane L: [c][L], 4 x 32 floats a head / a key), so the kernel's loads land in the registers it multiplies;
//   * a thread holds NQ queries (32 registers each) and walks NT tiles of 16 blocks: one key is read once for NQ queries, a query
//     once for 16 NT blocks, and a query's 16 scores of a tile leave in one scatter.
// The arithmetic is esimd_scores' variant 2, operation for operation (so the warp kernel's, see that file): the same bits.
// A tile that holds any query's tail block (b == n_bid: `dead`, the +1e9), or lies past `max_blocks` / the prepared keys, or a
// thread with fewer than NQ queries, takes the per-(query, block) loop of the one-query kernel.
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>
#include "strata/sycl_queue.hpp"
#include "strata/prefill/esimd_moe.hpp"
#include "strata/prefill/esimd_scores.hpp"
#include "strata/prefill/esimd_scores_mq.hpp"

#ifdef STRATA_I8_NO_ESIMD
namespace strata::prefill::i8 {
int scores_mq_env() { return 0; }
bool scores_mq_prepare(const float*, const float*, int64_t, int64_t, const ScoreSteps&, void*) { return false; }
bool block_scores_mq(const float*, const float*, const float*, const int32_t*, const ScoreSteps&, int64_t, int64_t, int64_t, float*,
                     void*, int64_t) { return false; }
}  // namespace strata::prefill::i8
#else
namespace strata::prefill::i8 {
namespace {
using namespace sycl::ext::intel::esimd;

constexpr int DIM = 128, HEADS = 4, BLOCK = 4, NB = 16;

template <int NQ, int NT> struct ScoresMqTag;
struct ScoresMqPrepQ;
struct ScoresMqPrepK;

SYCL_ESIMD_FUNCTION inline simd<float, 16> fma16(simd<float, 16> a, simd<float, 16> b, simd<float, 16> c) {
    return simd<float, 16>(__builtin_elementwise_fma(a.data(), b.data(), c.data()));
}

// one head's dot product of a key and a query, both as [c] x (lanes 0..15, lanes 16..31): esimd_scores' variant 2 and its tree
SYCL_ESIMD_FUNCTION inline float head_dot(const simd<float, 16>* Kl, const simd<float, 16>* Kh, const simd<float, 16>* Ql,
                                          const simd<float, 16>* Qh) {
    simd<float, 16> dl = fma16(Kl[0], Ql[0], Kl[1] * Ql[1]);
    dl = fma16(Kl[2], Ql[2], dl);
    dl = fma16(Kl[3], Ql[3], dl);
    simd<float, 16> dh = fma16(Kh[0], Qh[0], Kh[1] * Qh[1]);
    dh = fma16(Kh[2], Qh[2], dh);
    dh = fma16(Kh[3], Qh[3], dh);
    simd<float, 16> s16 = dl + dh;
    simd<float, 8> s8 = s16.template select<8, 1>(0) + s16.template select<8, 1>(8);
    simd<float, 4> s4 = s8.template select<4, 1>(0) + s8.template select<4, 1>(4);
    simd<float, 2> s2 = s4.template select<2, 1>(0) + s4.template select<2, 1>(2);
    const float s0 = s2[0], s1 = s2[1];
    const float d = s0 + s1;
    return d > 0.0f ? d : 0.0f;
}

struct Scratch {
    float* qt = nullptr;   // [queries][HEADS][4][32]
    float* kt = nullptr;   // [blocks][4][32]
    int64_t q_cap = 0, k_cap = 0;
    // what the last `prepare` on this queue wrote
    const float* q_src = nullptr;
    const float* k_src = nullptr;
    int64_t nq = 0, nk = 0;
};
std::mutex g_mu;
std::map<const void*, Scratch> g_scr;

template <int NQ, int NT>
void launch(sycl::queue& q, const float* pooled, const float* dead, const float* q_idx, const float* qt, const float* kt, int64_t kt_n,
            const int32_t* steps, const ScoreSteps ss, int64_t nq, int64_t max_blocks, float* out, int64_t reach) {
    auto props = sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>};
    const int count = ss.count, o_nkv = ss.n_kv, o_nbid = ss.n_bid;
    q.submit([&](sycl::handler& h) {
        h.parallel_for<ScoresMqTag<NQ, NT>>(sycl::range<2>((size_t) ((nq + NQ - 1) / NQ), (size_t) ((reach + NB * NT - 1) / (NB * NT))), props,
                                            [=](sycl::id<2> id) SYCL_ESIMD_KERNEL {
            const int64_t q0 = (int64_t) id[0] * NQ;
            const int64_t bt = (int64_t) id[1] * (NB * NT);
            const int nqv = nq - q0 < NQ ? (int) (nq - q0) : NQ;
            int64_t lo = (int64_t) 1 << 40, hi = -1;   // the thread's queries' smallest and largest tail block
            for (int j = 0; j < nqv; ++j) {
                const int64_t nb = steps[(q0 + j) * count + o_nbid];
                lo = nb < lo ? nb : lo;
                hi = nb > hi ? nb : hi;
            }
            if (bt > hi || bt >= max_blocks) return;
            const int64_t fast_end = lo < max_blocks ? (lo < kt_n ? lo : kt_n) : (max_blocks < kt_n ? max_blocks : kt_n);   // blocks below it: every query reads `pooled`
            const bool any_fast = nqv == NQ && bt + NB <= fast_end;
            simd<float, 16> Ql[NQ][HEADS][4], Qh[NQ][HEADS][4];
            if (any_fast) {
#pragma unroll
                for (int j = 0; j < NQ; ++j) {
                    const float* qp = qt + (q0 + j) * (HEADS * DIM);
#pragma unroll
                    for (int hh = 0; hh < HEADS; ++hh) {
                        simd<float, 64> a = block_load<float, 64>(qp + hh * DIM);
                        simd<float, 64> b = block_load<float, 64>(qp + hh * DIM + 64);
                        Ql[j][hh][0] = a.template select<16, 1>(0);
                        Qh[j][hh][0] = a.template select<16, 1>(16);
                        Ql[j][hh][1] = a.template select<16, 1>(32);
                        Qh[j][hh][1] = a.template select<16, 1>(48);
                        Ql[j][hh][2] = b.template select<16, 1>(0);
                        Qh[j][hh][2] = b.template select<16, 1>(16);
                        Ql[j][hh][3] = b.template select<16, 1>(32);
                        Qh[j][hh][3] = b.template select<16, 1>(48);
                    }
                }
            }
            for (int t = 0; t < NT; ++t) {
                const int64_t b0 = bt + (int64_t) t * NB;
                if (b0 > hi || b0 >= max_blocks) break;
                if (any_fast && b0 + NB <= fast_end) {
                    simd<float, 16> acc[NQ];
#pragma unroll
                    for (int i = 0; i < NB; ++i) {
                        const float* key = kt + (b0 + i) * DIM;
                        simd<float, 64> ka = block_load<float, 64>(key);
                        simd<float, 64> kb = block_load<float, 64>(key + 64);
                        simd<float, 16> Kl[4], Kh[4];
                        Kl[0] = ka.template select<16, 1>(0);
                        Kh[0] = ka.template select<16, 1>(16);
                        Kl[1] = ka.template select<16, 1>(32);
                        Kh[1] = ka.template select<16, 1>(48);
                        Kl[2] = kb.template select<16, 1>(0);
                        Kh[2] = kb.template select<16, 1>(16);
                        Kl[3] = kb.template select<16, 1>(32);
                        Kh[3] = kb.template select<16, 1>(48);
#pragma unroll
                        for (int j = 0; j < NQ; ++j) {
                            float score = 0.0f;
#pragma unroll
                            for (int hh = 0; hh < HEADS; ++hh) score += head_dot(Kl, Kh, Ql[j][hh], Qh[j][hh]);
                            acc[j][i] = score;
                        }
                    }
                    const simd<uint32_t, 16> o4(0u, 4u);
#pragma unroll
                    for (int j = 0; j < NQ; ++j) scatter<float, 16>(out + (q0 + j) * max_blocks + b0, o4, acc[j]);
                    continue;
                }
                // the one-query kernel's loop (its gathered form): a tile with a tail block, an end, or a short thread
                for (int j = 0; j < nqv; ++j) {
                    const int64_t qi = q0 + j;
                    const int32_t* st = steps + qi * count;
                    const int64_t n_kv = st[o_nkv], n_bid = st[o_nbid];
                    if (b0 > n_bid) continue;
                    const float* qp = q_idx + qi * (HEADS * DIM);
                    const simd<uint32_t, 16> g4(0u, 16u);
                    simd<float, 16> Sl[HEADS][4], Sh[HEADS][4];
#pragma unroll
                    for (int hh = 0; hh < HEADS; ++hh) {
#pragma unroll
                        for (int c = 0; c < 4; ++c) {
                            Sl[hh][c] = gather<float, 16>(qp + hh * DIM, g4 + (uint32_t) (4 * c));
                            Sh[hh][c] = gather<float, 16>(qp + hh * DIM + 64, g4 + (uint32_t) (4 * c));
                        }
                    }
                    for (int i = 0; i < NB; ++i) {
                        const int64_t b = b0 + i;
                        if (b > n_bid || b >= max_blocks) break;
                        const float* key = (b == n_bid) ? dead : pooled + b * DIM;
                        simd<float, 16> Kl[4], Kh[4];
#pragma unroll
                        for (int c = 0; c < 4; ++c) {
                            Kl[c] = gather<float, 16>(key, g4 + (uint32_t) (4 * c));
                            Kh[c] = gather<float, 16>(key + 64, g4 + (uint32_t) (4 * c));
                        }
                        float score = 0.0f;
#pragma unroll
                        for (int hh = 0; hh < HEADS; ++hh) score += head_dot(Kl, Kh, Sl[hh], Sh[hh]);
                        if (b == n_bid && (n_kv & (BLOCK - 1)) != 0) score += 1e9f;
                        out[qi * max_blocks + b] = score;
                    }
                }
            }
        });
    });
}
}  // namespace

int scores_mq_env() {
    static const int v = [] {
        const char* e = std::getenv("STRATA_SCORES_MQ");
        return e != nullptr && e[0] != 0 ? std::atoi(e) : 0;
    }();
    return v;
}

bool scores_mq_prepare(const float* pooled, const float* q_idx, int64_t nq, int64_t n_blocks, const ScoreSteps& ss, void* stream) {
    if (scores_mq_env() <= 0 || nq <= 0 || ss.dim != DIM || ss.heads != HEADS || ss.block != BLOCK || !available(stream)) return false;
    sycl::queue& q = *strata::q_of(stream);
    if (n_blocks < 0) n_blocks = 0;
    float *qt = nullptr, *kt = nullptr;
    {
        std::lock_guard<std::mutex> lk(g_mu);
        Scratch& sc = g_scr[(const void*) &q];
        sc.nq = sc.nk = 0;
        if (sc.q_cap < nq) {
            if (sc.qt != nullptr) { q.wait(); strata::device_alloc_unregister(sc.qt); sycl::free(sc.qt, q); sc.qt = nullptr; sc.q_cap = 0; }
            sc.qt = (float*) strata::malloc_device_guarded((size_t) nq * HEADS * DIM * sizeof(float), q, "ESIMD multi-query scores: queries");
            if (sc.qt == nullptr) return false;
            sc.q_cap = nq;
        }
        if (sc.k_cap < n_blocks) {
            const int64_t want = (n_blocks + 8191) / 8192 * 8192;   // grown in steps of 32K tokens (4 MiB)
            if (sc.kt != nullptr) { q.wait(); strata::device_alloc_unregister(sc.kt); sycl::free(sc.kt, q); sc.kt = nullptr; sc.k_cap = 0; }
            sc.kt = (float*) strata::malloc_device_guarded((size_t) want * DIM * sizeof(float), q, "ESIMD multi-query scores: keys");
            if (sc.kt == nullptr) return false;
            sc.k_cap = want;
        }
        qt = sc.qt;
        kt = sc.kt;
        sc.q_src = q_idx;
        sc.k_src = pooled;
        sc.nq = nq;
        sc.nk = n_blocks;
    }
    if (static bool told = false; !told) {
        told = true;
        std::fprintf(stderr, "strata: prompt block scores on the multi-query ESIMD kernel (STRATA_SCORES_MQ=%d)\n", scores_mq_env());
    }
    // [row][4 L + c] -> [row][c][L] (rows: a query's head, a block's key)
    q.parallel_for<ScoresMqPrepQ>(sycl::range<1>((size_t) (nq * HEADS * DIM)), [=](sycl::id<1> id) {
        const size_t i = id[0], row = i / DIM, e = i % DIM;
        qt[row * DIM + (e % 4) * 32 + e / 4] = q_idx[i];
    });
    if (n_blocks > 0)
        q.parallel_for<ScoresMqPrepK>(sycl::range<1>((size_t) (n_blocks * DIM)), [=](sycl::id<1> id) {
            const size_t i = id[0], row = i / DIM, e = i % DIM;
            kt[row * DIM + (e % 4) * 32 + e / 4] = pooled[i];
        });
    return true;
}

bool block_scores_mq(const float* pooled, const float* dead, const float* q_idx, const int32_t* steps, const ScoreSteps& ss, int64_t q_first,
                     int64_t nq, int64_t max_blocks, float* scores, void* stream, int64_t active_blocks) {
    const int mq = scores_mq_env();
    if (mq <= 0 || nq <= 0 || ss.dim != DIM || ss.heads != HEADS || ss.block != BLOCK) return false;
    sycl::queue& q = *strata::q_of(stream);
    const float *qt = nullptr, *kt = nullptr;
    int64_t kt_n = 0;
    {
        std::lock_guard<std::mutex> lk(g_mu);
        auto it = g_scr.find((const void*) &q);
        if (it == g_scr.end()) return false;
        const Scratch& sc = it->second;
        // the prepared copies are this layer's: the queries [q_first, q_first + nq) of q_src and the keys of k_src
        if (sc.k_src != pooled || sc.q_src == nullptr || q_idx != sc.q_src + q_first * (HEADS * DIM) || q_first + nq > sc.nq) return false;
        qt = sc.qt + q_first * (HEADS * DIM);
        kt = sc.kt;
        kt_n = sc.nk;
    }
    const int64_t reach = active_blocks > 0 && active_blocks < max_blocks ? active_blocks : max_blocks;
    const int nt = mq / 100 > 0 ? mq / 100 : 4;   // STRATA_SCORES_MQ=<100 x tiles of 16 blocks a thread + queries a thread>, e.g. 404; tiles default 4
    const int nqt = mq % 100;
#define STRATA_SMQ(Q, T) launch<Q, T>(q, pooled, dead, q_idx, qt, kt, kt_n, steps, ss, nq, max_blocks, scores, reach)
    if (nqt == 1 && nt == 4) STRATA_SMQ(1, 4);
    else if (nqt == 2 && nt == 4) STRATA_SMQ(2, 4);
    else if (nqt == 3 && nt == 4) STRATA_SMQ(3, 4);
    else if (nqt == 4 && nt == 4) STRATA_SMQ(4, 4);
    else if (nqt == 4 && nt == 1) STRATA_SMQ(4, 1);
    else if (nqt == 4 && nt == 8) STRATA_SMQ(4, 8);
    else if (nqt == 2 && nt == 8) STRATA_SMQ(2, 8);
    else if (nqt == 6 && nt == 4) STRATA_SMQ(6, 4);
    else return false;
#undef STRATA_SMQ
    return true;
}
}  // namespace strata::prefill::i8
#endif
