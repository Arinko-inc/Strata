// sycl/src/kernels/mmvq_esimd_bench.cpp - STRATA_MMVQ_ESIMD's dense Q4_K kernel (q4k_mmvq_esimd.dp.cpp) against the wide
// kernel (native_mmvq with STRATA_MMVQ_ESIMD off) on a model's own dense matrices (SYCL port, the B70): every output bit
// for bit, and each one's time.
//
//     mmvq_esimd_bench <any shard.gguf> <layer[,layer...]> [--t 1,2,3,4] [--rpt 4] [--var -1] [--trials 9]
//                      [--names attn_qkv,...] [--cmp 1|0] [--time 1|0]
//
// --rpt: rows per hardware thread of the ESIMD kernel (1, 2, 4, 8) + 400 * the loop's shape (0: four blocks per
// iteration, 1: one block and rotating sums, 2: one block and a run-time offset) + 100: the 128-register file + 200: the
// chained dp4a.  0: the engine's choice by the shape (through native_mmvq; STRATA_MMVQ_ESIMD_RPT / _LOOP apply).
// --var: the float expression's shape (-1 the engine's; 0..9).  Every (rpt, var) pair is a configuration.  The engine's
// kernels (rows 1, 2, 4, 8 x LOOP 0, 2 and the lean ones, + 1600: LOOP 2 of rows 1-4 and LOOP 0 of rows 1-3, with var -1:
// rpt 1 2 4 8 801 802 804 808 2401-2404 1601-1603) run the library's own q4k_mmvq_esimd, the
// one the engine calls; the experiment's (any other rpt or var) run this program's own copy of the kernel file, built
// with STRATA_Q4KE_EXP (q4k_mmvq_esimd_exp).  The cmp line names which ("engine" / "exp").  The engine's way in
// (native_mmvq with native_mmvq_set_esimd(1)) is compared as well.
// The matrices: the layers' Q4_K dense tensors by role (attn_qkv, attn_gate, ssm_out, ssm_alpha, ssm_beta of a GDN layer;
// attn_q, attn_k, attn_v, attn_output of a QSA layer; ffn_gate_shexp, ffn_up_shexp of both).  Activations are normal
// random vectors through native_quantize_q8_1, T columns.
// Timing: one call per layer of a role in ONE command graph (as the engine's window reads every layer's matrix once),
// trials interleave the wide kernel and the configurations, the median is printed in microseconds per call.  The wide
// kernel reads STRATA_MMVQ_LARGE_GRF as the engine does: set it as the engine's start script does.
// Exit 1 when any output differs.
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "strata/artifact/gguf_reader.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace strata::kernels {
bool q4k_mmvq_esimd_exp(const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, int ncols, void* stream,
                        int rpt, int var);   // q4k_mmvq_esimd.dp.cpp with STRATA_Q4KE_EXP (this target's own copy)
}
namespace K = strata::kernels;

namespace {
std::vector<int> ints(const char* s) {
    std::vector<int> v;
    std::stringstream ss(s);
    std::string t;
    while (std::getline(ss, t, ',')) if (!t.empty()) v.push_back(std::atoi(t.c_str()));
    return v;
}
std::vector<std::string> strs(const char* s) {
    std::vector<std::string> v;
    std::stringstream ss(s);
    std::string t;
    while (std::getline(ss, t, ',')) if (!t.empty()) v.push_back(t);
    return v;
}
using GraphExec = sycl::ext::oneapi::experimental::command_graph<sycl::ext::oneapi::experimental::graph_state::executable>;
struct Cfg { int rpt, var; };
bool engine_cfg(const Cfg& c) {   // in the library's q4k_mmvq_esimd (q4k_mmvq_esimd.dp.cpp without STRATA_Q4KE_EXP)
    const int lean = c.rpt / 1600, rows = c.rpt % 400, loop = c.rpt % 1600 / 400;
    if (c.var >= 0 || lean > 1) return false;
    if (lean) return (loop == 2 && rows >= 1 && rows <= 4) || (loop == 0 && rows >= 1 && rows <= 3);
    return (rows == 1 || rows == 2 || rows == 4 || rows == 8) && (loop == 0 || loop == 2);
}
struct Mat { int layer; uint8_t* w; };
}  // namespace

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    if (argc < 3) {
        std::fprintf(stderr, "usage: mmvq_esimd_bench <shard.gguf> <layer[,layer]> [--t 1,2,3,4] [--rpt 4] [--var -1] [--trials 9] "
                             "[--names attn_qkv,...] [--cmp 1|0] [--time 1|0]\n");
        return 2;
    }
    const std::vector<int> layers = ints(argv[2]);
    std::vector<int> Ts = {1, 2, 3, 4}, rpts = {4}, vars = {-1};
    std::vector<std::string> names = {"attn_qkv", "attn_gate", "ssm_out", "ssm_alpha", "ssm_beta", "attn_q", "attn_k", "attn_v",
                                      "attn_output", "ffn_gate_shexp", "ffn_up_shexp"};
    int trials = 9, cmp_on = 1, time_on = 1;
    for (int i = 3; i + 1 < argc; i += 2) {
        const std::string a = argv[i];
        if (a == "--t") Ts = ints(argv[i + 1]);
        else if (a == "--rpt") rpts = ints(argv[i + 1]);
        else if (a == "--var") vars = ints(argv[i + 1]);
        else if (a == "--trials") trials = std::atoi(argv[i + 1]);
        else if (a == "--names") names = strs(argv[i + 1]);
        else if (a == "--cmp") cmp_on = std::atoi(argv[i + 1]);
        else if (a == "--time") time_on = std::atoi(argv[i + 1]);
        else { std::fprintf(stderr, "unknown option %s\n", a.c_str()); return 2; }
    }
    for (int t : Ts) if (t < 1 || t > 4) { std::fprintf(stderr, "--t: 1..4\n"); return 2; }
    if (trials < 1 || Ts.empty() || rpts.empty() || vars.empty()) { std::fprintf(stderr, "empty list or --trials < 1\n"); return 2; }
    if (!K::q4k_mmvq_esimd_built()) { std::fprintf(stderr, "mmvq_esimd_bench: the ESIMD kernel is not in this build\n"); return 2; }
    std::vector<Cfg> cfgs;
    for (int r : rpts) for (int v : vars) cfgs.push_back({r, v});
    std::unique_ptr<strata::GgufModel> model;
    try {
        model = std::make_unique<strata::GgufModel>(strata::GgufModel::open(argv[1]));
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "mmvq_esimd_bench: %s\n", ex.what());
        return 2;
    }
    dpct::queue_ptr s = dpct::get_current_device().create_queue(true);
    const char* lg = std::getenv("STRATA_MMVQ_LARGE_GRF");
    std::printf("device: %s; trials %d; STRATA_MMVQ_LARGE_GRF %s; STRATA_MMVQ_ESIMD of the environment %d\n",
                s->get_device().get_info<sycl::info::device::name>().c_str(), trials, lg ? lg : "-", K::native_mmvq_esimd());
    constexpr int TMAX = 4;
    int failures = 0;
    std::mt19937 rng(20261123);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<double> saved((size_t) (TMAX + 1) * cfgs.size(), 0.0), base(TMAX + 1, 0.0);   // sums over the roles: us per window
    for (const std::string& name : names) {
        std::vector<Mat> mats;
        int n_in = 0, n_out = 0;
        size_t wb = 0;
        for (int l : layers) {
            size_t sh = 0;
            const strata::TensorInfo* t = model->find("blk." + std::to_string(l) + "." + name + ".weight", &sh);
            if (!t) continue;
            if (t->type != 12 || t->shape.size() != 2 || t->shape[0] % 256 != 0) {
                std::printf("layer %d %s: %s, %zu dims: not a Q4_K matrix of 256-blocks, skipped\n", l, name.c_str(), t->type_name(),
                            t->shape.size());
                continue;
            }
            const int ni = (int) t->shape[0], no = (int) t->shape[1];
            if (n_in == 0) { n_in = ni; n_out = no; wb = K::native_mmvq_weight_bytes(12, ni, no); }
            if (ni != n_in || no != n_out) { std::printf("layer %d %s: another shape, skipped\n", l, name.c_str()); continue; }
            uint8_t* w = (uint8_t*) sycl::malloc_device(wb + 256, *s);
            if (!w) { std::fprintf(stderr, "malloc_device %zu bytes failed\n", wb); return 2; }
            s->memcpy(w, model->shard(sh).tensor_data(*t), wb).wait();
            mats.push_back({l, w});
        }
        if (mats.empty()) continue;
        const int M = (int) mats.size();
        std::vector<float> hx((size_t) TMAX * n_in);
        for (auto& v : hx) v = nd(rng);
        float* x = sycl::malloc_device<float>(hx.size(), *s);
        s->memcpy(x, hx.data(), hx.size() * 4).wait();
        void* xq = sycl::malloc_device(K::native_q8_1_bytes(n_in, TMAX) + 256, *s);
        const size_t ycap = (size_t) TMAX * n_out;
        float* y = sycl::malloc_device<float>(ycap * M, *s);
        std::vector<float> href(ycap * M), hy(ycap * M);
        for (int T : Ts) {
            K::native_quantize_q8_1(x, xq, n_in, T, s);
            s->wait();
            const size_t ny = (size_t) T * n_out;
            auto run = [&](int c) {   // c < 0: the wide kernel; c == cfgs.size(): configuration 0 through native_mmvq
                for (int m = 0; m < M; ++m) {
                    float* ym = y + ycap * m;
                    if (c < 0) {
                        K::native_mmvq_set_esimd(0);
                        K::native_mmvq(12, mats[m].w, xq, ym, n_in, n_out, T, s);
                    } else if (c == (int) cfgs.size() || cfgs[c].rpt == 0) {   // --rpt 0: the engine's choice by the shape
                        K::native_mmvq_set_esimd(1);
                        K::native_mmvq(12, mats[m].w, xq, ym, n_in, n_out, T, s);
                        K::native_mmvq_set_esimd(0);
                    } else if (!(engine_cfg(cfgs[c]) ? K::q4k_mmvq_esimd : K::q4k_mmvq_esimd_exp)(mats[m].w, xq, ym, n_in, n_out, T, s,
                                                                                                cfgs[c].rpt, cfgs[c].var)) {
                        std::fprintf(stderr, "%s declined rpt %d var %d\n", engine_cfg(cfgs[c]) ? "q4k_mmvq_esimd" : "q4k_mmvq_esimd_exp",
                                     cfgs[c].rpt, cfgs[c].var);
                        std::exit(2);
                    }
                }
            };
            if (cmp_on) {
                s->memset(y, 0xff, ycap * M * 4).wait();
                run(-1);
                s->wait();
                s->memcpy(href.data(), y, ycap * M * 4).wait();
                for (int c = 0; c <= (int) cfgs.size(); ++c) {
                    s->memset(y, 0xff, ycap * M * 4).wait();
                    run(c);
                    s->wait();
                    s->memcpy(hy.data(), y, ycap * M * 4).wait();
                    size_t bad = 0;
                    double maxrel = 0;
                    for (int m = 0; m < M; ++m)
                        for (size_t i = 0; i < ny; ++i) {
                            const float a = href[ycap * m + i], b = hy[ycap * m + i];
                            if (std::memcmp(&a, &b, 4) != 0) {
                                ++bad;
                                const double rel = std::fabs((double) a - b) / std::max(std::fabs((double) a), 1e-30);
                                if (!(rel <= maxrel)) maxrel = rel;
                            }
                        }
                    if (bad) ++failures;
                    if (c == (int) cfgs.size())
                        std::printf("cmp %-14s %5d x %5d T=%d %2d layers  via native_mmvq (STRATA_MMVQ_ESIMD=1): %zu of %zu differ, max rel %.3e\n",
                                    name.c_str(), n_in, n_out, T, M, bad, ny * M, maxrel);
                    else
                        std::printf("cmp %-14s %5d x %5d T=%d %2d layers  rpt %3d var %2d %-6s: %zu of %zu differ, max rel %.3e\n",
                                    name.c_str(), n_in, n_out, T, M, cfgs[c].rpt, cfgs[c].var, engine_cfg(cfgs[c]) ? "engine" : "exp", bad,
                                    ny * M, maxrel);
                }
            }
            if (time_on) {
                std::vector<std::unique_ptr<GraphExec>> graphs;
                for (int c = -1; c < (int) cfgs.size(); ++c) {
                    dpct::experimental::command_graph_ptr g;
                    dpct::experimental::begin_recording(s);
                    run(c);
                    dpct::experimental::end_recording(s, &g);
                    graphs.push_back(std::make_unique<GraphExec>(g->finalize()));
                    delete g;
                }
                std::vector<std::vector<double>> us(graphs.size());
                for (int i = 0; i < trials + 3; ++i)
                    for (size_t c = 0; c < graphs.size(); ++c) {
                        sycl::event e0 = s->ext_oneapi_submit_barrier();
                        s->ext_oneapi_graph(*graphs[c]);
                        sycl::event e1 = s->ext_oneapi_submit_barrier();
                        e1.wait();
                        const double ns = (double) (e1.get_profiling_info<sycl::info::event_profiling::command_end>() -
                                                    e0.get_profiling_info<sycl::info::event_profiling::command_start>());
                        if (i >= 3) us[c].push_back(ns / 1e3 / M);
                    }
                for (auto& v : us) std::sort(v.begin(), v.end());
                const double u0 = us[0][us[0].size() / 2];
                base[T] += u0 * M;
                std::printf("time %-14s %5d x %5d T=%d %2d layers  wide %8.2f us (%5.1f GB/s)", name.c_str(), n_in, n_out, T, M, u0,
                            wb / u0 / 1e3);
                for (size_t c = 1; c < graphs.size(); ++c) {
                    const double u = us[c][us[c].size() / 2];
                    saved[(size_t) T * cfgs.size() + c - 1] += (u0 - u) * M;
                    std::printf(" | rpt %d var %d %8.2f us (%5.1f GB/s) x%.2f", cfgs[c - 1].rpt, cfgs[c - 1].var, u, wb / u / 1e3, u0 / u);
                }
                std::printf("\n");
            }
        }
        for (auto& m : mats) sycl::free(m.w, *s);
        sycl::free(x, *s); sycl::free(xq, *s); sycl::free(y, *s);
    }
    if (time_on)
        for (int T : Ts) {
            std::printf("sum T=%d over the given layers' matrices (one call each): wide %.3f ms", T, base[T] / 1e3);
            for (size_t c = 0; c < cfgs.size(); ++c)
                std::printf(" | rpt %d var %d saves %.3f ms", cfgs[c].rpt, cfgs[c].var, saved[(size_t) T * cfgs.size() + c] / 1e3);
            std::printf("\n");
        }
    std::printf("mmvq_esimd_bench: %d failures\n", failures);
    dpct::get_current_device().destroy_queue(s);
    return failures ? 1 : 0;
}
