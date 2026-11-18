// router_ngen_parity - pruned-experts (STRATA_ROUTER_NGEN=1): a pruned model's routing (384 / 320 experts, no native
// router) on the new path against the old one, bit for bit (GPU, synthetic, no model).
//
// The old path is what the engine ran before STRATA_ROUTER_NGEN: each verify-window row routed by itself - one
// `bf16_gemv_fp32_mmvf` and one portable top-k (`router_top10_variant` 0, router_top10_kernel) per row.  The new
// path is `moe_route_window`'s: one `bf16_gemv_fp32_mmvf_cols` over the window and one n-row `router_top10`, which
// for these expert counts takes the fast kernel (variant 1) with STRATA_ROUTER_NGEN=1.  Checked:
//   1. the top-k kernels on 4,096 rows of logits with ties, near-ties and a NaN row: router_top10 (the default
//      dispatch) and variants 1 / 2 against variant 0 - ids and weights bitwise;
//   2. a window of T = 1..8 rows (2560 -> E BF16 router, random): the new path's logits, ids and weights against the
//      old path's, bitwise.
// The test sets STRATA_ROUTER_NGEN=1 itself (unset or 0, the default dispatch is the portable kernel).
#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/bf16_gemv.hpp"
#include "strata/kernels/router_top10.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace {
void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e)); std::exit(2); }
}

struct Out {
    std::vector<int> ids;
    std::vector<float> w;
    bool operator==(const Out& o) const {
        return ids == o.ids && std::memcmp(w.data(), o.w.data(), w.size() * 4) == 0;
    }
};

// one top-k run on device logits `dl` (rows x E): variant -1 = router_top10 (the default dispatch), 0..2 =
// router_top10_variant; per_row: one launch per row.  The outputs start as the same bytes every time (a NaN row
// leaves ranks unwritten, and both kernels renormalise what the buffer held).
Out topk(const float* dl, int rows, int E, int K, int variant, bool per_row) {
    int* di; float* dw;
    ck(cudaMalloc(&di, (size_t) rows * K * 4), "ids"); ck(cudaMalloc(&dw, (size_t) rows * K * 4), "w");
    ck(cudaMemset(di, 0, (size_t) rows * K * 4), "ids0"); ck(cudaMemset(dw, 0, (size_t) rows * K * 4), "w0");
    for (int r0 = 0; r0 < rows; r0 += per_row ? 1 : rows) {
        const int n = per_row ? 1 : rows;
        const float* l = dl + (size_t) r0 * E;
        if (variant < 0) strata::kernels::router_top10(l, n, E, K, di + (size_t) r0 * K, dw + (size_t) r0 * K, nullptr);
        else if (!strata::kernels::router_top10_variant(l, n, E, K, di + (size_t) r0 * K, dw + (size_t) r0 * K, nullptr,
                                                         variant)) {
            std::fprintf(stderr, "router_top10_variant %d refused E %d\n", variant, E);
            std::exit(2);
        }
    }
    ck(cudaDeviceSynchronize(), "topk sync");
    Out o{std::vector<int>((size_t) rows * K), std::vector<float>((size_t) rows * K)};
    ck(cudaMemcpy(o.ids.data(), di, o.ids.size() * 4, cudaMemcpyDeviceToHost), "ids down");
    ck(cudaMemcpy(o.w.data(), dw, o.w.size() * 4, cudaMemcpyDeviceToHost), "w down");
    cudaFree(di); cudaFree(dw);
    return o;
}
}  // namespace

int main() {
    // the path under test is opt-in: turned on here, before the first router call reads the variable
#if defined(_WIN32)
    _putenv_s("STRATA_ROUTER_NGEN", "1");
#else
    setenv("STRATA_ROUTER_NGEN", "1", 1);
#endif
    const int N = 2560, K = 10, ROWS = 4096;
    int bad = 0;
    std::mt19937 rng(11);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::printf("router_ngen_parity: STRATA_ROUTER_NGEN %s\n", strata::kernels::router_ngen_enabled() ? "=1 (on)" : "unset or 0 (OFF: run it with STRATA_ROUTER_NGEN=1)");
    for (int E : {384, 320}) {
        if (!strata::kernels::router_ngen_applies(E, K))
            std::printf("  E %d: the default dispatch is the portable kernel (STRATA_ROUTER_NGEN is not 1)\n", E);
        // ---- 1. the top-k kernels
        std::vector<float> lg((size_t) ROWS * E);
        for (int r = 0; r < ROWS; ++r) {
            float* l = lg.data() + (size_t) r * E;
            for (int e = 0; e < E; ++e) l[e] = nd(rng) * 2.0f;
            if (r % 7 == 0) for (int e = 0; e < E; ++e) l[e] = 0.5f;                 // all equal
            if (r % 11 == 0) for (int e = 0; e < 12; ++e) l[e * 3] = 6.0f;           // a 12-way tie at the top
            if (r % 13 == 0) l[5] = std::nextafter(l[9] = 7.0f, 8.0f);               // a near-tie
            if (r == 999) l[77] = NAN;                                               // a NaN row
        }
        float* dl;
        ck(cudaMalloc(&dl, lg.size() * 4), "logits");
        ck(cudaMemcpy(dl, lg.data(), lg.size() * 4, cudaMemcpyHostToDevice), "logits up");
        const Out ref = topk(dl, ROWS, E, K, 0, true);
        const struct { const char* name; int variant; bool per_row; } runs[] = {
            {"variant 0, all rows in one launch", 0, false}, {"router_top10 (default), one launch", -1, false},
            {"router_top10 (default), per row", -1, true}, {"variant 1 (fast)", 1, false},
            {"variant 2 (fast, serial sum)", 2, false}};
        for (const auto& run : runs) {
            const bool same = topk(dl, ROWS, E, K, run.variant, run.per_row) == ref;
            std::printf("  E %d  top-k %-36s %s\n", E, run.name, same ? "bit-identical" : "*** DIFFERS ***");
            bad += same ? 0 : 1;
        }
        cudaFree(dl);
        // ---- 2. the window: the new path (cols projection + n-row top-k) against the old (per row, portable)
        for (int T = 1; T <= 8; ++T) {
            std::vector<uint16_t> w((size_t) E * N);
            std::vector<float> x((size_t) T * N);
            for (auto& v : w) v = strata::kernels::bf16_from_f32(nd(rng) * 0.05f);
            for (auto& v : x) v = nd(rng);
            uint16_t* dw; float *dx, *d_old, *d_new;
            ck(cudaMalloc(&dw, w.size() * 2), "w"); ck(cudaMalloc(&dx, x.size() * 4), "x");
            ck(cudaMalloc(&d_old, (size_t) T * E * 4), "old"); ck(cudaMalloc(&d_new, (size_t) T * E * 4), "new");
            ck(cudaMemcpy(dw, w.data(), w.size() * 2, cudaMemcpyHostToDevice), "up w");
            ck(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice), "up x");
            for (int t = 0; t < T; ++t)
                strata::kernels::bf16_gemv_fp32_mmvf(dx + (size_t) t * N, dw, d_old + (size_t) t * E, N, E, nullptr);
            strata::kernels::bf16_gemv_fp32_mmvf_cols(dx, dw, d_new, N, E, T, nullptr);
            ck(cudaDeviceSynchronize(), "sync");
            std::vector<float> lo((size_t) T * E), ln((size_t) T * E);
            ck(cudaMemcpy(lo.data(), d_old, lo.size() * 4, cudaMemcpyDeviceToHost), "down old");
            ck(cudaMemcpy(ln.data(), d_new, ln.size() * 4, cudaMemcpyDeviceToHost), "down new");
            const bool same_l = std::memcmp(lo.data(), ln.data(), lo.size() * 4) == 0;
            const bool same_k = topk(d_old, T, E, K, 0, true) == topk(d_new, T, E, K, -1, false);
            std::printf("  E %d  window T %d  logits %s  ids+weights %s\n", E, T,
                        same_l ? "bit-identical" : "*** DIFFER ***", same_k ? "bit-identical" : "*** DIFFER ***");
            bad += (same_l ? 0 : 1) + (same_k ? 0 : 1);
            cudaFree(dw); cudaFree(dx); cudaFree(d_old); cudaFree(d_new);
        }
    }
    std::printf(bad ? "router_ngen_parity: %d FAILURES\n" : "router_ngen_parity OK\n", bad);
    return bad ? 1 : 0;
}
