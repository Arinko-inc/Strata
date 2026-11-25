// tools/relay_read/relay_mmq_bench.cpp - the prompt path's MMQ expert products on HIP, timed by shape and type, to see
// what a launch's time follows (strata-mesh IMPL-PLAN-2026-11-26-relay-read.md: the Q5_0 down product has 0.3x the
// multiply-adds of the Q4_K gate/up product a block and takes as long).  One MMQ group as the prompt path launches it:
// 16 experts, R rows each.  Synthetic weights with valid scales, random activations.
//   STRATA_MMQ_KQ=1 relay_mmq_bench [R ...]      (default R: 16 48 64 96)
// Per case: the median of 25 launches (ms), and ns per (64 weight rows x R rows x 32 values) unit.
#include "strata/prefill/moe_mmq.hpp"

#include "ggml.h"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace mmq = strata::prefill::mmq;

namespace {

void ck(hipError_t e, const char* what) {
    if (e != hipSuccess) { std::fprintf(stderr, "%s: %s\n", what, hipGetErrorString(e)); std::exit(1); }
}

struct Case {
    const char* name;
    ggml_type t;
    int64_t rows, cols;
};

float run_case(const Case& c, int R, std::mt19937& rng) {
    constexpr int E = 16;
    const int T = E * R;
    const size_t per = mmq::matrix_bytes(c.t, c.rows, c.cols);
    std::vector<uint8_t> w((size_t) E * per + 4096, 0);
    std::uniform_int_distribution<int> byte(0, 255);
    for (size_t i = 0; i < (size_t) E * per; ++i) w[i] = (uint8_t) byte(rng);
    const size_t bs = ggml_type_size(c.t);
    for (size_t i = 0; i < (size_t) E * per / bs; ++i) {   // an f16 d (and dmin) leads the block
        const uint16_t d = 0x2000 + (uint16_t) byte(rng);
        std::memcpy(w.data() + i * bs, &d, 2);
        if (c.t == GGML_TYPE_Q4_K) std::memcpy(w.data() + i * bs + 2, &d, 2);
    }
    std::vector<float> x((size_t) T * c.cols);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (float& v : x) v = nd(rng);
    std::vector<int32_t> bounds(E + 1);
    for (int e = 0; e <= E; ++e) bounds[e] = e * R;
    uint8_t* dw; float *dx, *dd, *dys; void* dq; int32_t *dids, *db;
    ck(hipMalloc(&dw, w.size()), "w"); ck(hipMemcpy(dw, w.data(), w.size(), hipMemcpyHostToDevice), "w");
    ck(hipMalloc(&dx, x.size() * 4), "x"); ck(hipMemcpy(dx, x.data(), x.size() * 4, hipMemcpyHostToDevice), "x");
    ck(hipMalloc(&dq, mmq::q8_bytes(T, c.cols) + 4096), "xq");
    ck(hipMalloc(&dys, (size_t) T * 4), "ys");
    ck(hipMalloc(&dd, (size_t) T * c.rows * 4), "dst");
    ck(hipMalloc(&dids, (size_t) T * 4), "ids");
    ck(hipMalloc(&db, (E + 1) * 4), "bounds");
    ck(hipMemcpy(db, bounds.data(), (E + 1) * 4, hipMemcpyHostToDevice), "bounds");
    mmq::iota(dids, T, nullptr);
    mmq::quantize(dx, nullptr, dq, c.t, c.cols, c.cols, T, nullptr, dys);
    mmq::Context ctx;
    mmq::Product p;
    p.w = dw; p.type = c.t; p.w_rows = c.rows; p.w_cols = c.cols; p.expert_bytes = per; p.n = E;
    p.xq = dq; p.bounds = db; p.ids = dids; p.total_rows = T; p.max_rows = R; p.dst = dd; p.ld_dst = c.rows;
    hipEvent_t e0, e1;
    hipEventCreate(&e0); hipEventCreate(&e1);
    std::vector<float> ts;
    for (int i = 0; i < 28; ++i) {
        hipEventRecord(e0, nullptr);
        ctx.run(p, nullptr);
        hipEventRecord(e1, nullptr);
        ck(hipEventSynchronize(e1), "run");
        float ms; hipEventElapsedTime(&ms, e0, e1);
        if (i >= 3) ts.push_back(ms);
    }
    std::sort(ts.begin(), ts.end());
    hipFree(dw); hipFree(dx); hipFree(dq); hipFree(dys); hipFree(dd); hipFree(dids); hipFree(db);
    hipEventDestroy(e0); hipEventDestroy(e1);
    return ts[ts.size() / 2];
}

}  // namespace

int main(int argc, char** argv) {
    std::vector<int> Rs;
    for (int i = 1; i < argc; ++i) Rs.push_back(std::atoi(argv[i]));
    if (Rs.empty()) Rs = {16, 48, 64, 96};
    const Case cases[] = {{"q4_k 1280x2560 (gate/up)", GGML_TYPE_Q4_K, 1280, 2560}, {"q5_0 2560x640 (down)", GGML_TYPE_Q5_0, 2560, 640},
                          {"q5_0 1280x2560", GGML_TYPE_Q5_0, 1280, 2560},           {"q4_k 2560x768", GGML_TYPE_Q4_K, 2560, 768},
                          {"q5_0 2560x768", GGML_TYPE_Q5_0, 2560, 768},             {"q5_0 2560x512", GGML_TYPE_Q5_0, 2560, 512},
                          {"q5_0 1280x640", GGML_TYPE_Q5_0, 1280, 640}};
    std::mt19937 rng(7);
    for (const Case& c : cases) {
        if (!mmq::fits(c.t, c.rows)) { std::printf("%-26s not supported here (STRATA_MMQ_KQ=1?)\n", c.name); continue; }
        for (int R : Rs) {
            const float ms = run_case(c, R, rng);
            const double units = 16.0 * (double) (c.rows / 64) * R * (double) (c.cols / 32);
            std::printf("%-26s R %3d  %7.3f ms  %6.2f ns/unit  (%d blocks of 64 rows)\n", c.name, R, ms, 1e6 * ms / units,
                        (int) (16 * c.rows / 64));
        }
    }
    return 0;
}
