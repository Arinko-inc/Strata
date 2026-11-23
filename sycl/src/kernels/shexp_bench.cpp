// sycl/src/kernels/shexp_bench.cpp - the shared expert of a decode window (shared_expert_multi, SYCL port, the B70) launch
// by launch, and STRATA_SHEXP_ESIMD's Q5_0 down kernel (q5_0_mmvq_esimd.dp.cpp) against native_mmvq on a model's own
// layers: every output bit for bit, and each one's time.
//
//     shexp_bench <any shard.gguf> <layer[,layer...]> [--t 1,2,3,4] [--rpt 2] [--var -1] [--trials 9] [--reps 3]
//                 [--phases 1|0] [--cmp 1|0] [--time 1|0]
//
// --rpt: weight rows per hardware thread of the ESIMD kernel (2; 1, 4, 8 in a -DSTRATA_Q50E_EXP build).  --var: the kernel's
// variant (-1 the engine's; 0 / 10: the activations unpacked per row / once a thread; 1, 2 in an experiment build).  Every
// (rpt, var) pair is a configuration.
// The weights are the layers' ffn_gate_shexp / ffn_up_shexp / ffn_down_shexp; the scalar gate's vector is random (its
// values only scale the output).  The input is T normal random rows; the down kernels read the q8_1 image that the
// layer's own gate, up and SwiGLU make of it, as in the engine.
// cmp:   down by native_mmvq against each configuration, and shared_expert_multi with shared_expert_set_esimd(0)
//        against (1) (the engine's way in), over --reps inputs.
// phase: one call per layer in ONE command graph (as the engine's window runs every layer's shared expert once), the
//        median over --trials in microseconds per call: the whole of shared_expert_multi as the engine calls it without
//        STRATA_LFUSE / STRATA_VERIFY_QDEDUP, the same with the scalar gate left to the caller, and each launch alone.
// The Q4_K gate / up read STRATA_MMVQ_LARGE_GRF and STRATA_MMVQ_ESIMD as the engine does: set them as its start script does.
// Layers whose down projection is not Q5_0 of 640 inputs: no ESIMD configuration is run (cmp of shared_expert_multi only).
// Exit 1 when any output differs.
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "strata/artifact/gguf_reader.hpp"
#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/bf16_gemv.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/q5_0_mmvq_esimd.hpp"
#include "strata/kernels/shared_expert.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace K = strata::kernels;

namespace {
std::vector<int> ints(const char* s) {
    std::vector<int> v;
    std::stringstream ss(s);
    std::string t;
    while (std::getline(ss, t, ',')) if (!t.empty()) v.push_back(std::atoi(t.c_str()));
    return v;
}
using GraphExec = sycl::ext::oneapi::experimental::command_graph<sycl::ext::oneapi::experimental::graph_state::executable>;
struct Cfg { int rpt, var; };
struct Layer {
    int layer;
    uint8_t *wg, *wu, *wd;        // the three projections
    void *xq, *hq, *sq;           // q8_1 of the input, of the SwiGLU output, shared_expert_multi's scratch
    float *gate, *up, *out, *g;
};
}  // namespace

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    if (argc < 3) {
        std::fprintf(stderr, "usage: shexp_bench <shard.gguf> <layer[,layer]> [--t 1,2,3,4] [--rpt 2] [--var -1] [--trials 9] "
                             "[--reps 3] [--phases 1|0] [--cmp 1|0] [--time 1|0]\n");
        return 2;
    }
    const std::vector<int> layers = ints(argv[2]);
    std::vector<int> Ts = {1, 2, 3, 4}, rpts = {2}, vars = {-1};
    int trials = 9, reps = 3, phases_on = 1, cmp_on = 1, time_on = 1;
    for (int i = 3; i + 1 < argc; i += 2) {
        const std::string a = argv[i];
        if (a == "--t") Ts = ints(argv[i + 1]);
        else if (a == "--rpt") rpts = ints(argv[i + 1]);
        else if (a == "--var") vars = ints(argv[i + 1]);
        else if (a == "--trials") trials = std::atoi(argv[i + 1]);
        else if (a == "--reps") reps = std::atoi(argv[i + 1]);
        else if (a == "--phases") phases_on = std::atoi(argv[i + 1]);
        else if (a == "--cmp") cmp_on = std::atoi(argv[i + 1]);
        else if (a == "--time") time_on = std::atoi(argv[i + 1]);
        else { std::fprintf(stderr, "unknown option %s\n", a.c_str()); return 2; }
    }
    for (int t : Ts) if (t < 1 || t > 4) { std::fprintf(stderr, "--t: 1..4\n"); return 2; }
    if (trials < 1 || reps < 1 || Ts.empty() || rpts.empty() || vars.empty() || layers.empty()) {
        std::fprintf(stderr, "empty list, --trials < 1 or --reps < 1\n");
        return 2;
    }
    if (!K::q5_0_mmvq_esimd_built()) { std::fprintf(stderr, "shexp_bench: the ESIMD kernel is not in this build\n"); return 2; }
    std::vector<Cfg> cfgs;
    for (int r : rpts) for (int v : vars) cfgs.push_back({r, v});
    std::unique_ptr<strata::GgufModel> model;
    try {
        model = std::make_unique<strata::GgufModel>(strata::GgufModel::open(argv[1]));
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "shexp_bench: %s\n", ex.what());
        return 2;
    }
    dpct::queue_ptr s = dpct::get_current_device().create_queue(true);
    const char* lg = std::getenv("STRATA_MMVQ_LARGE_GRF");
    std::printf("device: %s; trials %d; reps %d; STRATA_MMVQ_LARGE_GRF %s; STRATA_MMVQ_ESIMD %d; STRATA_SHEXP_ESIMD of the environment %d\n",
                s->get_device().get_info<sycl::info::device::name>().c_str(), trials, reps, lg ? lg : "-", K::native_mmvq_esimd(),
                K::shared_expert_esimd());
    K::shared_expert_set_native_bf16(true);   // the engine's --native
    constexpr int TMAX = 4;
    int n_embd = 0, n_ff = 0, gt = -1, ut = -1, dt = -1;
    std::vector<Layer> L;
    for (int l : layers) {
        const std::string b = "blk." + std::to_string(l) + ".";
        size_t shg = 0, shu = 0, shd = 0;
        const strata::TensorInfo* tg = model->find(b + "ffn_gate_shexp.weight", &shg);
        const strata::TensorInfo* tu = model->find(b + "ffn_up_shexp.weight", &shu);
        const strata::TensorInfo* td = model->find(b + "ffn_down_shexp.weight", &shd);
        if (!tg || !tu || !td || tg->shape.size() != 2 || tu->shape.size() != 2 || td->shape.size() != 2) {
            std::printf("layer %d: no shared expert matrices, skipped\n", l);
            continue;
        }
        if (L.empty()) {
            n_embd = (int) tg->shape[0]; n_ff = (int) tg->shape[1];
            gt = (int) tg->type; ut = (int) tu->type; dt = (int) td->type;
            std::printf("layer %d: gate %s %d x %d, up %s %d x %d, down %s %d x %d\n", l, tg->type_name(), (int) tg->shape[0],
                        (int) tg->shape[1], tu->type_name(), (int) tu->shape[0], (int) tu->shape[1], td->type_name(),
                        (int) td->shape[0], (int) td->shape[1]);
        }
        if ((int) tg->shape[0] != n_embd || (int) tg->shape[1] != n_ff || (int) tu->shape[0] != n_embd || (int) tu->shape[1] != n_ff ||
            (int) td->shape[0] != n_ff || (int) td->shape[1] != n_embd || (int) tg->type != gt || (int) tu->type != ut ||
            (int) td->type != dt) {
            std::printf("layer %d: another shape or type, skipped\n", l);
            continue;
        }
        if (!K::native_mmvq_supported(gt) || !K::native_mmvq_supported(ut) || !K::native_mmvq_supported(dt)) {
            std::fprintf(stderr, "shexp_bench: a type native_mmvq does not read\n");
            return 2;
        }
        const size_t bg = K::native_mmvq_weight_bytes(gt, n_embd, n_ff), bu = K::native_mmvq_weight_bytes(ut, n_embd, n_ff),
                     bd = K::native_mmvq_weight_bytes(dt, n_ff, n_embd);
        // the copies below read the mapping: a truncated shard is an error, not a read past its end
        const strata::TensorInfo* tt[3] = {tg, tu, td};
        const size_t ts[3] = {shg, shu, shd}, tb[3] = {bg, bu, bd};
        const char* tn[3] = {"ffn_gate_shexp", "ffn_up_shexp", "ffn_down_shexp"};
        for (int k = 0; k < 3; ++k)
            if (!model->in_bounds(*tt[k], ts[k]) || strata::tensor_payload_bytes(*tt[k]) < tb[k]) {
                std::fprintf(stderr, "shexp_bench: layer %d %s: the payload is truncated or short of %zu B (%s)\n", l, tn[k], tb[k],
                             model->shard(ts[k]).path().c_str());
                return 2;
            }
        Layer y{};
        y.layer = l;
        y.wg = (uint8_t*) sycl::malloc_device(bg + 256, *s);
        y.wu = (uint8_t*) sycl::malloc_device(bu + 256, *s);
        y.wd = (uint8_t*) sycl::malloc_device(bd + 256, *s);
        y.xq = sycl::malloc_device(K::native_q8_1_bytes(n_embd, TMAX) + 256, *s);
        y.hq = sycl::malloc_device(K::native_q8_1_bytes(n_ff, TMAX) + 256, *s);
        y.sq = sycl::malloc_device(K::native_q8_1_bytes(std::max(n_embd, n_ff), TMAX) + 256, *s);   // the input's q8_1, then the SwiGLU's
        y.gate = sycl::malloc_device<float>((size_t) TMAX * n_ff, *s);
        y.up = sycl::malloc_device<float>((size_t) TMAX * n_ff, *s);
        y.out = sycl::malloc_device<float>((size_t) TMAX * n_embd, *s);
        y.g = sycl::malloc_device<float>(TMAX, *s);
        if (!y.wg || !y.wu || !y.wd || !y.xq || !y.hq || !y.sq || !y.gate || !y.up || !y.out || !y.g) {
            std::fprintf(stderr, "malloc_device failed\n");
            return 2;
        }
        s->memcpy(y.wg, model->shard(shg).tensor_data(*tg), bg);
        s->memcpy(y.wu, model->shard(shu).tensor_data(*tu), bu);
        s->memcpy(y.wd, model->shard(shd).tensor_data(*td), bd);
        s->wait();
        L.push_back(y);
    }
    if (L.empty()) { std::fprintf(stderr, "shexp_bench: no layer with a shared expert\n"); return 2; }
    const int M = (int) L.size();
    const bool esimd_shape = dt == 6 && n_ff == 640;   // the ESIMD kernel reads Q5_0 rows of 640 inputs only
    std::printf("%d layers; the down projection is type %d (%s the ESIMD kernel's Q5_0 of 640 inputs%s)\n", M, dt,
                esimd_shape ? "" : "NOT", esimd_shape ? "" : ": no configuration compared or timed, shared_expert_multi only");
    if (!esimd_shape) cfgs.clear();
    const int NCFG = (int) cfgs.size();
    std::mt19937 rng(20261124);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> hx((size_t) TMAX * n_embd);
    float* x = sycl::malloc_device<float>(hx.size(), *s);
    std::vector<uint16_t> hgi((size_t) n_embd);
    for (auto& v : hgi) v = K::bf16_from_f32(nd(rng) * 0.02f);
    uint16_t* gi = sycl::malloc_device<uint16_t>(hgi.size(), *s);
    s->memcpy(gi, hgi.data(), hgi.size() * 2).wait();
    const size_t ocap = (size_t) TMAX * n_embd;
    std::vector<float> href(ocap * M), hy(ocap * M);
    int failures = 0;
    size_t compared = 0;

    auto new_x = [&] {
        for (auto& v : hx) v = nd(rng);
        s->memcpy(x, hx.data(), hx.size() * 4).wait();
    };
    auto prep = [&](int T) {   // the launches before the down, layer by layer
        for (auto& y : L) {
            K::native_quantize_q8_1(x, y.xq, n_embd, T, s);
            K::native_mmvq(gt, y.wg, y.xq, y.gate, n_embd, n_ff, T, s);
            K::native_mmvq(ut, y.wu, y.xq, y.up, n_embd, n_ff, T, s);
            K::native_swiglu_quantize_q8_1(y.gate, y.up, y.hq, n_ff, T, s);
        }
        s->wait();
    };
    auto full = [&](int T, int esimd, int lfuse) {
        K::shared_expert_set_esimd(esimd);
        for (auto& y : L) {
            K::NativeSharedWeights nsw;
            nsw.gate_type = gt; nsw.gate_data = y.wg;
            nsw.up_type = ut; nsw.up_data = y.wu;
            nsw.down_type = dt; nsw.down_data = y.wd;
            nsw.q8_1 = y.sq;
            K::shared_expert_multi(T, x, nullptr, nsw, gi, y.gate, y.up, y.g, y.out, n_embd, n_ff, s, nullptr, lfuse);
        }
        K::shared_expert_set_esimd(0);
    };
    auto down = [&](int T, int c) {   // c < 0: native_mmvq
        for (auto& y : L) {
            if (c < 0) K::native_mmvq(dt, y.wd, y.hq, y.out, n_ff, n_embd, T, s);
            else if (!K::q5_0_mmvq_esimd(y.wd, y.hq, y.out, n_ff, n_embd, T, s, cfgs[c].rpt, cfgs[c].var)) {
                std::fprintf(stderr, "q5_0_mmvq_esimd declined rpt %d var %d\n", cfgs[c].rpt, cfgs[c].var);
                std::exit(2);
            }
        }
    };
    auto fetch = [&](std::vector<float>& h) {
        s->wait();
        for (int m = 0; m < M; ++m) s->memcpy(h.data() + ocap * m, L[m].out, ocap * 4);
        s->wait();
    };
    auto wipe = [&] {
        for (auto& y : L) s->memset(y.out, 0xff, ocap * 4);
        s->wait();
    };
    auto diff = [&](int T, size_t* bad, double* maxrel) {
        const size_t ny = (size_t) T * n_embd;
        for (int m = 0; m < M; ++m)
            for (size_t i = 0; i < ny; ++i) {
                const float a = href[ocap * m + i], b = hy[ocap * m + i];
                if (std::memcmp(&a, &b, 4) != 0) {
                    ++*bad;
                    const double rel = std::fabs((double) a - b) / std::max(std::fabs((double) a), 1e-30);
                    if (!(rel <= *maxrel)) *maxrel = rel;
                }
            }
        return ny * M;
    };
    auto timed = [&](const std::vector<std::function<void()>>& runs) {   // the median of each, us per call
        std::vector<std::unique_ptr<GraphExec>> graphs;
        for (const auto& r : runs) {
            dpct::experimental::command_graph_ptr g;
            dpct::experimental::begin_recording(s);
            r();
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
        std::vector<double> med;
        for (auto& v : us) { std::sort(v.begin(), v.end()); med.push_back(v[v.size() / 2]); }
        return med;
    };

    const size_t bd = K::native_mmvq_weight_bytes(dt, n_ff, n_embd);
    for (int T : Ts) {
        if (cmp_on) {
            std::vector<size_t> bad(NCFG + 1, 0), tot(NCFG + 1, 0);
            std::vector<double> maxrel(NCFG + 1, 0.0);
            for (int rep = 0; rep < reps; ++rep) {
                new_x();
                prep(T);
                wipe();
                down(T, -1);
                fetch(href);
                for (int c = 0; c < NCFG; ++c) {
                    wipe();
                    down(T, c);
                    fetch(hy);
                    tot[c] += diff(T, &bad[c], &maxrel[c]);
                }
                wipe();
                full(T, 0, 0);
                fetch(href);
                wipe();
                full(T, 1, 0);
                fetch(hy);
                tot[NCFG] += diff(T, &bad[NCFG], &maxrel[NCFG]);
            }
            for (int c = 0; c <= NCFG; ++c) {
                if (bad[c]) ++failures;
                compared += tot[c];
                if (c < NCFG)
                    std::printf("cmp down T=%d %2d layers  rpt %d var %2d: %zu of %zu differ, max rel %.3e\n", T, M, cfgs[c].rpt,
                                cfgs[c].var, bad[c], tot[c], maxrel[c]);
                else
                    std::printf("cmp shared_expert_multi T=%d %2d layers  STRATA_SHEXP_ESIMD 0 against 1: %zu of %zu differ, max rel %.3e\n",
                                T, M, bad[c], tot[c], maxrel[c]);
            }
        }
        if (!time_on) continue;
        new_x();
        prep(T);
        if (phases_on) {
            const std::vector<double> u = timed({
                [&] { full(T, 0, 0); },
                [&] { full(T, 1, 0); },
                [&] { full(T, 0, 1); },
                [&] { for (auto& y : L) K::native_quantize_q8_1(x, y.xq, n_embd, T, s); },
                [&] { for (auto& y : L) K::native_mmvq(gt, y.wg, y.xq, y.gate, n_embd, n_ff, T, s); },
                [&] { for (auto& y : L) K::native_mmvq(ut, y.wu, y.xq, y.up, n_embd, n_ff, T, s); },
                [&] { for (auto& y : L) K::native_swiglu_quantize_q8_1(y.gate, y.up, y.hq, n_ff, T, s); },
                [&] { down(T, -1); },
            });
            std::printf("phase T=%d %2d layers, us per call: whole %.2f | whole with STRATA_SHEXP_ESIMD=1 %.2f | whole without the scalar "
                        "gate %.2f (the gate's gemv + sigmoid-scale: %.2f) | quantize %.2f | gate %.2f | up %.2f | swiglu+quantize %.2f "
                        "| down %.2f | the five alone add to %.2f\n",
                        T, M, u[0], u[1], u[2], u[0] - u[2], u[3], u[4], u[5], u[6], u[7], u[3] + u[4] + u[5] + u[6] + u[7]);
            std::printf("window T=%d: the %d layers' shared experts %.3f ms, of it the down %.3f ms; STRATA_SHEXP_ESIMD=1 saves %.3f ms\n",
                        T, M, u[0] * M / 1e3, u[7] * M / 1e3, (u[0] - u[1]) * M / 1e3);
            prep(T);   // the whole calls rewrote gate / up with the same values; hq again for the down timing below
        }
        std::vector<std::function<void()>> runs;
        runs.push_back([&] { down(T, -1); });
        for (int c = 0; c < NCFG; ++c) runs.push_back([&, c] { down(T, c); });
        const std::vector<double> u = timed(runs);
        std::printf("time down T=%d %2d layers  native_mmvq %7.2f us (%5.1f GB/s)", T, M, u[0], bd / u[0] / 1e3);
        for (int c = 0; c < NCFG; ++c)
            std::printf(" | rpt %d var %2d %7.2f us (%5.1f GB/s) x%.2f saves %.3f ms", cfgs[c].rpt, cfgs[c].var, u[c + 1],
                        bd / u[c + 1] / 1e3, u[0] / u[c + 1], (u[0] - u[c + 1]) * M / 1e3);
        std::printf("\n");
    }
    if (cmp_on && compared == 0) { std::printf("shexp_bench: nothing was compared\n"); failures = 1; }
    std::printf("shexp_bench: %d failures\n", failures);
    dpct::get_current_device().destroy_queue(s);
    return failures ? 1 : 0;
}
