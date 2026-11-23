// sycl/src/kernels/gr_req8_bench.cpp - SYCL port: STRATA_HC_REQ8's ESIMD hyper-connection read (fused_gr_req8) against
// the default BF16 read (fused_gr_read_multi) on a model's own projections.
//   gr_req8_bench <any shard.gguf> <layer[,layer...]> [--t 1,2,3,4] [--calls 50] [--trials 7] [--graph 1|0]
// Per layer and half (hc_attn, hc_ffn) and T: the largest relative error of each output against the BF16 read
// (max |a - b| / max |b| over the output's values of all tokens) and the time of one read (--calls reads in ONE
// command graph, as the engine's window is; the median of --trials, the two reads alternating).  The inputs are
// random rows with the pending write folded (apply), as the engine's reads are.
#include "strata/artifact/gguf_reader.hpp"
#include "strata/core/weights.hpp"
#include "strata/kernels/fused_gr.hpp"
#include "strata/kernels/fused_gr_req8.hpp"
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace K = strata::kernels;
using GraphExec = sycl::ext::oneapi::experimental::command_graph<sycl::ext::oneapi::experimental::graph_state::executable>;

static dpct::queue_ptr g_q;
template <class T> static T* dalloc(size_t n) { return (T*) sycl::malloc_device(n * sizeof(T) + 256, *g_q); }
template <class T> static T* dput(const T* h, size_t n) {
    T* p = dalloc<T>(n);
    g_q->memcpy(p, h, n * sizeof(T)).wait();
    return p;
}
static std::vector<int> ints(const std::string& s) {
    std::vector<int> v;
    size_t i = 0;
    while (i < s.size()) { size_t j = s.find(',', i); if (j == std::string::npos) j = s.size(); v.push_back(std::atoi(s.substr(i, j - i).c_str())); i = j + 1; }
    return v;
}
static double median(std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; }

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: gr_req8_bench <shard.gguf> <layer[,layer]> [--t 1,2,3,4] [--calls 50] [--trials 7] [--graph 1|0]\n");
        return 2;
    }
    const std::vector<int> layers = ints(argv[2]);
    std::vector<int> Ts = {1, 2, 3, 4};
    int calls = 50, trials = 7, graph = 1;
    for (int i = 3; i + 1 < argc; i += 2) {
        const std::string a = argv[i];
        if (a == "--t") Ts = ints(argv[i + 1]);
        else if (a == "--calls") calls = std::atoi(argv[i + 1]);
        else if (a == "--trials") trials = std::atoi(argv[i + 1]);
        else if (a == "--graph") graph = std::atoi(argv[i + 1]);
        else { std::fprintf(stderr, "unknown option %s\n", a.c_str()); return 2; }
    }
    std::unique_ptr<strata::GgufModel> model;
    try {
        model = std::make_unique<strata::GgufModel>(strata::GgufModel::open(argv[1]));
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "gr_req8_bench: %s\n", ex.what());
        return 2;
    }
    g_q = dpct::get_current_device().create_queue(true);
    dpct::queue_ptr s = g_q;
    const char* es = std::getenv("STRATA_HC_REQ8_S");
    std::printf("device: %s; calls %d, trials %d, graph %d; built %d; STRATA_HC_REQ8_S %s\n",
                s->get_device().get_info<sycl::info::device::name>().c_str(), calls, trials, graph, (int) K::hc_req8_built(),
                es ? es : "-");
    if (!K::hc_req8_built()) return 2;
    constexpr int N = 2560, HC = 4, D = N * HC, LR = 320, TM = K::kFusedGrMaxT;
    constexpr int OUT = LR + HC + HC + N;   // lo, rs, inject, mixed a token
    std::mt19937 rng(20261123);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> hR((size_t) TM * D), hbo((size_t) TM * N), hip((size_t) TM * HC);
    for (auto& v : hR) v = nd(rng);
    for (auto& v : hbo) v = 0.3f * nd(rng);
    for (auto& v : hip) v = nd(rng);
    float* dR0 = dput(hR.data(), hR.size());
    float* dR[2] = {dalloc<float>(hR.size()), dalloc<float>(hR.size())};
    float* dbo = dput(hbo.data(), hbo.size());
    float* dip = dput(hip.data(), hip.size());
    float* dxn = dalloc<float>((size_t) TM * D);
    float* dout[2] = {dalloc<float>((size_t) TM * OUT), dalloc<float>((size_t) TM * OUT)};
    int failures = 0;
    double tot_us[2][5] = {};   // [read][T] summed over the layers and halves
    int tot_n = 0;
    double worst[5] = {};       // lo, rs, inject, mixed, R_out
    for (int l : layers) {
        for (int half = 0; half < 2; ++half) {
            const std::string pfx = "blk." + std::to_string(l) + (half == 0 ? ".hc_attn_" : ".hc_ffn_");
            const char* roles[4] = {"norm", "down", "up", "inject"};
            const strata::TensorInfo* t[4] = {};
            size_t sh[4] = {};
            bool ok = true;
            for (int r = 0; r < 4; ++r) {
                t[r] = model->find(pfx + roles[r] + ".weight", &sh[r]);
                if (!t[r]) ok = false;
            }
            if (!ok || t[0]->elements() != (uint64_t) D || t[1]->elements() != (uint64_t) LR * D || t[2]->elements() != (uint64_t) D * LR ||
                t[3]->elements() != (uint64_t) HC * D || std::string(t[1]->type_name()) != "BF16" ||
                std::string(t[2]->type_name()) != "BF16" || std::string(t[3]->type_name()) != "BF16" ||
                std::string(t[0]->type_name()) != "F32") {
                std::printf("layer %d %s: tensors missing or not F32 / BF16 of the compiled shapes\n", l, pfx.c_str());
                ++failures;
                continue;
            }
            const float* h_norm = (const float*) model->shard(sh[0]).tensor_data(*t[0]);
            const uint16_t* h_down = (const uint16_t*) model->shard(sh[1]).tensor_data(*t[1]);
            const uint16_t* h_up = (const uint16_t*) model->shard(sh[2]).tensor_data(*t[2]);
            const uint16_t* h_inj = (const uint16_t*) model->shard(sh[3]).tensor_data(*t[3]);
            float* d_norm = dput(h_norm, (size_t) D);
            uint16_t* d_down = dput(h_down, (size_t) LR * D);
            uint16_t* d_up = dput(h_up, (size_t) D * LR);
            uint16_t* d_inj = dput(h_inj, (size_t) HC * D);
            std::vector<int8_t> c_down, c_up;
            std::vector<float> s_down, s_up;
            strata::core::hc_req8_pack(h_down, LR, D, c_down, s_down);
            strata::core::hc_req8_pack(h_up, D, LR, c_up, s_up);
            int8_t* q_down = dput(c_down.data(), c_down.size());
            int8_t* q_up = dput(c_up.data(), c_up.size());
            float* qs_down = dput(s_down.data(), s_down.size());
            float* qs_up = dput(s_up.data(), s_up.size());
            // [0] the BF16 read, [1] the int8 read; each on its own copy of R (the pending write is in place)
            K::FusedGrArgs a[2][TM];
            for (int v = 0; v < 2; ++v)
                for (int k = 0; k < TM; ++k) {
                    K::FusedGrArgs& x = a[v][k];
                    float* o = dout[v] + (size_t) k * OUT;
                    x.R = dR[v] + (size_t) k * D; x.R_out = dR[v] + (size_t) k * D; x.apply = true;
                    x.bo_prev = dbo + (size_t) k * N; x.inj_prev = dip + (size_t) k * HC;
                    x.w_norm = d_norm; x.w_inject = d_inj;
                    x.w_down = v == 0 ? d_down : (const uint16_t*) q_down;
                    x.w_up = v == 0 ? d_up : (const uint16_t*) q_up;
                    if (v == 1) { x.s_down = qs_down; x.s_up = qs_up; }
                    x.lo = o; x.rs = o + LR; x.inject_out = o + LR + HC; x.mixed = o + LR + 2 * HC;
                }
            auto reset = [&] {
                s->memcpy(dR[0], dR0, hR.size() * 4);
                s->memcpy(dR[1], dR0, hR.size() * 4);
                s->wait();
            };
            std::printf("layer %2d %s\n", l, half == 0 ? "hc_attn (hc-read0)" : "hc_ffn (hc-read1)");
            for (int T : Ts) {
                if (T < 1 || T > 4) continue;
                // values: one read each from the same R
                reset();
                s->memset(dout[0], 0, (size_t) TM * OUT * 4);
                s->memset(dout[1], 0, (size_t) TM * OUT * 4);
                s->wait();
                K::fused_gr_read_multi(a[0], T, dxn, s);
                K::fused_gr_read_multi(a[1], T, dxn, s);
                s->wait();
                std::vector<float> o0((size_t) T * OUT), o1((size_t) T * OUT), r0((size_t) T * D), r1((size_t) T * D);
                s->memcpy(o0.data(), dout[0], o0.size() * 4);
                s->memcpy(o1.data(), dout[1], o1.size() * 4);
                s->memcpy(r0.data(), dR[0], r0.size() * 4);
                s->memcpy(r1.data(), dR[1], r1.size() * 4);
                s->wait();
                const int off[5] = {0, LR, LR + HC, LR + 2 * HC, OUT};
                double rel[5] = {};
                bool finite = true;
                for (int f = 0; f < 4; ++f) {
                    double md = 0, mr = 0;
                    for (int k = 0; k < T; ++k)
                        for (int i = off[f]; i < off[f + 1]; ++i) {
                            const float x = o0[(size_t) k * OUT + i], y = o1[(size_t) k * OUT + i];
                            if (!std::isfinite(y)) finite = false;
                            md = std::max(md, (double) std::fabs(x - y));
                            mr = std::max(mr, (double) std::fabs(x));
                        }
                    rel[f] = mr > 0 ? md / mr : md;
                }
                {
                    double md = 0, mr = 0;
                    for (size_t i = 0; i < r0.size(); ++i) { md = std::max(md, (double) std::fabs(r0[i] - r1[i])); mr = std::max(mr, (double) std::fabs(r0[i])); }
                    rel[4] = mr > 0 ? md / mr : md;
                }
                for (int f = 0; f < 5; ++f) worst[f] = std::max(worst[f], rel[f]);
                if (!finite || rel[3] > 0.1 || rel[1] > 1e-4 || rel[4] > 1e-5) ++failures;
                // times
                GraphExec* ge[2] = {nullptr, nullptr};
                if (graph)
                    for (int v = 0; v < 2; ++v) {
                        dpct::experimental::command_graph_ptr g;
                        dpct::experimental::begin_recording(s);
                        for (int i = 0; i < calls; ++i) K::fused_gr_read_multi(a[v], T, dxn, s);
                        dpct::experimental::end_recording(s, &g);
                        ge[v] = new GraphExec(g->finalize());
                        delete g;
                    }
                std::vector<double> us[2];
                for (int tr = 0; tr < trials + 1; ++tr)
                    for (int v = 0; v < 2; ++v) {
                        reset();
                        for (int i = 0; i < 4; ++i) K::fused_gr_read_multi(a[v], T, dxn, s);
                        s->wait();
                        const auto t0 = std::chrono::steady_clock::now();
                        if (graph) s->ext_oneapi_graph(*ge[v]);
                        else for (int i = 0; i < calls; ++i) K::fused_gr_read_multi(a[v], T, dxn, s);
                        s->wait();
                        const auto t1 = std::chrono::steady_clock::now();
                        if (tr > 0) us[v].push_back(std::chrono::duration<double, std::micro>(t1 - t0).count() / calls);
                    }
                for (int v = 0; v < 2; ++v) delete ge[v];
                const double m0 = median(us[0]), m1 = median(us[1]);
                tot_us[0][T] += m0; tot_us[1][T] += m1;
                std::printf("  T %d: bf16 %7.1f us  int8 %7.1f us  (x%.2f)   rel err: lo %.2e rs %.2e inject %.2e mixed %.2e R_out %.2e%s\n",
                            T, m0, m1, m0 / m1, rel[0], rel[1], rel[2], rel[3], rel[4], finite ? "" : "  NOT FINITE");
            }
            ++tot_n;
            for (void* p : {(void*) d_norm, (void*) d_down, (void*) d_up, (void*) d_inj, (void*) q_down, (void*) q_up, (void*) qs_down, (void*) qs_up})
                sycl::free(p, *g_q);
        }
    }
    std::printf("total over %d reads (layers x halves):\n", tot_n);
    for (int T : Ts)
        if (T >= 1 && T <= 4)
            std::printf("  T %d: bf16 %8.1f us  int8 %8.1f us  saved %8.1f us a window (%.3f ms)\n", T, tot_us[0][T], tot_us[1][T],
                        tot_us[0][T] - tot_us[1][T], (tot_us[0][T] - tot_us[1][T]) / 1e3);
    std::printf("largest rel err: lo %.2e rs %.2e inject %.2e mixed %.2e R_out %.2e\n", worst[0], worst[1], worst[2], worst[3], worst[4]);
    std::printf("%d failures\n", failures);
    return failures ? 1 : 0;
}
