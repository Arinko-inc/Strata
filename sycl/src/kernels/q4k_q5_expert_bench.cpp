// sycl/src/kernels/q4k_q5_expert_bench.cpp - the grouped decode experts of a Q4_K gate/up + Q5_0 down layer on the
// model's own rows (SYCL port, the B70): STRATA_EXPERT_Q4KQ5's variants (iq_kernels.dp.cpp) against the kernels of
// variant 0, every output byte for byte, and each one's time.  The SYCL twin of src/kernels/q4k_q5_expert_bench.cpp.
//
//     q4k_q5_expert_bench <any shard.gguf> <layer[,layer...]> [--t 1,2,3,4] [--share 0,50] [--plans 100]
//                         [--trials 7] [--vs 2,6,...] [--phases 0|1] [--alias K] [--graph 1|0] [--cmp 1|0]
//
// --vs: STRATA_EXPERT_Q4KQ5 values (a sum of bits: 1 gate/up sub-group 16, 2 down split, 4 down sub-group 16, 8 no
// root sync, 16 fused gate/up + SwiGLU).
// A plan is a verify window's VRAM call: T rows of top-10 routing over the layer's 512 experts (`share` % of a row's
// experts are the previous row's), groups in first-appearance order, entries in row order.  A window of two slots
// with one row each is two T = 1 calls, which is what the T = 1 timing issues back to back.
// Timing: all plans of a (T, share) in ONE command graph (--graph 1, as the engine's window is; --graph 0 submits them
// one by one), launched and waited for; trials interleave the variants; the median is printed, microseconds per call.
// --phases 1 adds gate/up alone, gate/up + SwiGLU/q8_1, down alone, and "empty": the same launches with no group (the
// launches' fixed cost).  --alias K points every group at one of K blobs (they stay in the card's caches): what the
// kernels cost without the weight reads from memory.  The lane flags (STRATA_KQ_GU_LANES, STRATA_DOWN_LANES) are read
// at start as in the engine: variant 0 of a run with them is that run's reference, so compare times across runs.
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "strata/artifact/gguf_reader.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/iq_kernels.hpp"

#include "ggml.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace cpu = strata::kernels::cpu;
namespace K = strata::kernels;

namespace {

dpct::queue_ptr g_q;

template <typename T>
T* dalloc(size_t n) {
    T* p = (T*) sycl::malloc_device(n * sizeof(T) + 256, *g_q);
    if (!p) { std::fprintf(stderr, "malloc_device %zu bytes failed\n", n * sizeof(T)); std::exit(2); }
    return p;
}
template <typename T>
T* dput(const std::vector<T>& v) {
    T* p = dalloc<T>(std::max<size_t>(v.size(), 1));
    if (!v.empty()) g_q->memcpy(p, v.data(), v.size() * sizeof(T)).wait();
    return p;
}

struct Plan {
    int ng = 0, ne = 0, t = 0;
    std::vector<int> gid;
    std::vector<int32_t> start, tok, dst;
    unsigned long long* d_ptr = nullptr;
    int32_t *d_start = nullptr, *d_ng = nullptr, *d_tok = nullptr, *d_dst = nullptr;
};

Plan route(std::mt19937& rng, int t, int share) {
    std::vector<int> ids, prev, cur;
    for (int k = 0; k < t; ++k) {
        cur.clear();
        const bool sh = k > 0 && share > 0;
        if (sh) {
            std::vector<int> p = prev;
            std::shuffle(p.begin(), p.end(), rng);
            cur.assign(p.begin(), p.begin() + 10 * share / 100);
        }
        while ((int) cur.size() < 10) {
            const int e = (int) (rng() % 512u);
            if (std::find(cur.begin(), cur.end(), e) != cur.end()) continue;
            if (sh && std::find(prev.begin(), prev.end(), e) != prev.end()) continue;
            cur.push_back(e);
        }
        std::shuffle(cur.begin(), cur.end(), rng);
        ids.insert(ids.end(), cur.begin(), cur.end());
        prev = cur;
    }
    Plan p;
    p.t = t;
    std::map<int, std::vector<int>> by;
    std::vector<int> first;
    for (int j = 0; j < (int) ids.size(); ++j) {
        if (by.find(ids[j]) == by.end()) first.push_back(ids[j]);
        by[ids[j]].push_back(j);
    }
    for (int e : first) {
        p.gid.push_back(e);
        p.start.push_back(p.ne);
        for (int j : by[e]) {
            p.tok.push_back(j / 10);
            p.dst.push_back(j);
            ++p.ne;
        }
    }
    p.start.push_back(p.ne);
    p.ng = (int) p.gid.size();
    return p;
}

int g_alias = 0;

void upload(Plan& p, const uint8_t* base, size_t stride) {
    std::vector<unsigned long long> hp(p.ng);
    for (int g = 0; g < p.ng; ++g)
        hp[g] = (unsigned long long) (base + (size_t) (g_alias > 0 ? p.gid[g] % g_alias : p.gid[g]) * stride);
    p.d_ptr = dput(hp);
    p.d_start = dput(p.start);
    p.d_ng = dput(std::vector<int32_t>{p.ng});
    p.d_tok = dput(p.tok);
    p.d_dst = dput(p.dst);
}
void release(Plan& p) {
    sycl::free(p.d_ptr, *g_q); sycl::free(p.d_start, *g_q); sycl::free(p.d_ng, *g_q); sycl::free(p.d_tok, *g_q); sycl::free(p.d_dst, *g_q);
}

std::vector<int> ints(const std::string& s) {
    std::vector<int> v;
    std::stringstream ss(s);
    std::string t;
    while (std::getline(ss, t, ',')) v.push_back(std::atoi(t.c_str()));
    return v;
}

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

using GraphExec = sycl::ext::oneapi::experimental::command_graph<sycl::ext::oneapi::experimental::graph_state::executable>;

}  // namespace

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    if (argc < 3) {
        std::fprintf(stderr, "usage: q4k_q5_expert_bench <shard.gguf> <layer[,layer]> [--t 1,2,3,4] [--share 0,50] [--plans 100] "
                             "[--trials 7] [--vs 2,...] [--phases 0|1] [--alias K] [--graph 1|0] [--cmp 1|0]\n");
        return 2;
    }
    const std::vector<int> layers = ints(argv[2]);
    std::vector<int> Ts = {1, 2, 3, 4}, shares = {0, 50}, vs = {2};
    int plans = 100, trials = 7, phases = 0, graph = 1, cmp_on = 1, ref_v = 0;   // --ref V: compare with variant V's outputs
    for (int i = 3; i + 1 < argc; i += 2) {
        const std::string a = argv[i];
        if (a == "--t") Ts = ints(argv[i + 1]);
        else if (a == "--share") shares = ints(argv[i + 1]);
        else if (a == "--plans") plans = std::atoi(argv[i + 1]);
        else if (a == "--trials") trials = std::atoi(argv[i + 1]);
        else if (a == "--vs") vs = ints(argv[i + 1]);
        else if (a == "--phases") phases = std::atoi(argv[i + 1]);
        else if (a == "--alias") g_alias = std::atoi(argv[i + 1]);
        else if (a == "--graph") graph = std::atoi(argv[i + 1]);
        else if (a == "--cmp") cmp_on = std::atoi(argv[i + 1]);
        else if (a == "--ref") ref_v = std::atoi(argv[i + 1]);
        else { std::fprintf(stderr, "unknown option %s\n", a.c_str()); return 2; }
    }
    std::unique_ptr<strata::GgufModel> model;
    try {
        model = std::make_unique<strata::GgufModel>(strata::GgufModel::open(argv[1]));
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "q4k_q5_expert_bench: %s\n", ex.what());
        return 2;
    }
    g_q = dpct::get_current_device().create_queue(true);
    dpct::queue_ptr s = g_q;
    const char* e1 = std::getenv("STRATA_KQ_GU_LANES");
    const char* e2 = std::getenv("STRATA_DOWN_LANES");
    const char* e3 = std::getenv("STRATA_GU_LANES");
    std::printf("device: %s; plans %d, trials %d, graph %d, alias %d; STRATA_EXPERT_Q4KQ5 of the environment %d; "
                "STRATA_KQ_GU_LANES %s STRATA_GU_LANES %s STRATA_DOWN_LANES %s; local_mem %zu, max wg %zu\n",
                s->get_device().get_info<sycl::info::device::name>().c_str(), plans, trials, graph, g_alias,
                K::native_expert_q4kq5(), e1 ? e1 : "-", e3 ? e3 : "-", e2 ? e2 : "-",
                (size_t) s->get_device().get_info<sycl::info::device::local_mem_size>(),
                (size_t) s->get_device().get_info<sycl::info::device::max_work_group_size>());
    {
        const auto sgs = s->get_device().get_info<sycl::info::device::sub_group_sizes>();
        std::printf("sub-group sizes:");
        for (auto v : sgs) std::printf(" %zu", (size_t) v);
        std::printf("\n");
    }
    const int64_t H = 2560, FF = 640;
    constexpr int kTokMax = 8;
    int failures = 0;
    std::mt19937 rng(20261122);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> x((size_t) kTokMax * H);
    for (auto& v : x) v = nd(rng);
    const int64_t capmax = (int64_t) kTokMax * 10;
    float* dx = dput(x);
    uint8_t* dxq = dalloc<uint8_t>((size_t) kTokMax * H / 32 * 36);
    uint8_t* dscr = dalloc<uint8_t>(K::native_expert_scratch_bytes(capmax, FF));
    float* dout = dalloc<float>((size_t) capmax * H);
    K::quantize_q8_1_rows(dx, kTokMax, H, dxq, s);
    s->wait();
    for (int l : layers) {
        const char* roles[3] = {"gate", "up", "down"};
        const strata::TensorInfo* t[3] = {};
        size_t sh[3] = {};
        for (int r = 0; r < 3; ++r) t[r] = model->find("blk." + std::to_string(l) + ".ffn_" + roles[r] + "_exps.weight", &sh[r]);
        if (!t[0] || !t[1] || !t[2]) { std::printf("layer %d: no expert tensors\n", l); ++failures; continue; }
        cpu::NativeFmt f;
        std::string err;
        if (!cpu::native_fmt((int) t[0]->type, (int) t[2]->type, H, FF, f, err)) {
            std::printf("layer %d: %s\n", l, err.c_str()); ++failures; continue;
        }
        if (f.gu_type != 12 || f.d_type != 6) {
            std::printf("layer %d: %s / %s is not Q4_K / Q5_0\n", l, ggml_type_name((ggml_type) f.gu_type),
                        ggml_type_name((ggml_type) f.d_type));
            ++failures;
            continue;
        }
        const size_t stride = (f.bytes + 255) & ~(size_t) 255;
        std::vector<uint8_t> host((size_t) 512 * stride);
        const size_t dn_role = f.bytes - f.down_off;
        for (int e = 0; e < 512; ++e) {
            uint8_t* b = host.data() + (size_t) e * stride;
            std::memcpy(b, model->shard(sh[0]).tensor_data(*t[0]) + (size_t) e * f.up_off, f.up_off);
            std::memcpy(b + f.up_off, model->shard(sh[1]).tensor_data(*t[1]) + (size_t) e * f.up_off, f.up_off);
            std::memcpy(b + f.down_off, model->shard(sh[2]).tensor_data(*t[2]) + (size_t) e * dn_role, dn_role);
        }
        uint8_t* raw = dput(host);
        const auto L = K::native_expert_layout(f.gu_type, f.d_type, H, FF);
        std::printf("layer %2d %s/%s: 512 experts, blob %zu B\n", l, ggml_type_name((ggml_type) f.gu_type),
                    ggml_type_name((ggml_type) f.d_type), f.bytes);
        for (int T : Ts)
            for (int share : shares) {
                if (T == 1 && share != shares[0]) continue;
                std::mt19937 prng(1000u * (unsigned) l + 10u * (unsigned) T + (unsigned) share);
                std::vector<Plan> P;
                double ng = 0, ne = 0;
                for (int i = 0; i < plans; ++i) {
                    P.push_back(route(prng, T, share));
                    upload(P.back(), raw, stride);
                    ng += P.back().ng;
                    ne += P.back().ne;
                }
                int32_t* d_zero = dput(std::vector<int32_t>{0});
                const int64_t cap = (int64_t) T * 10;
                const size_t ob = (size_t) cap * H * 4;
                auto call = [&](const Plan& p, bool empty = false) {
                    K::native_expert_grouped(L, p.d_ptr, p.d_start, empty ? d_zero : p.d_ng, p.d_dst, p.d_tok, cap, cap, dxq, dscr,
                                             dout, s, 0);
                };
                std::string cmp;
                if (cmp_on) {
                    std::vector<std::vector<uint8_t>> ref(P.size(), std::vector<uint8_t>(ob));
                    K::native_expert_set_mode(-1, 0);
                    K::native_expert_set_q4kq5(ref_v);
                    for (size_t i = 0; i < P.size(); ++i) {
                        s->memset(dout, 0xff, ob);
                        call(P[i]);
                        s->memcpy(ref[i].data(), dout, ob).wait();
                    }
                    std::vector<uint8_t> got(ob);
                    for (int v : vs) {
                        K::native_expert_set_q4kq5(v);
                        size_t ndiff = 0, nval = 0;
                        double worst = 0, maxabs = 0, sumabs = 0;
                        for (size_t i = 0; i < P.size(); ++i) {
                            s->memset(dout, 0xff, ob);
                            s->memset(dscr, (int) (i & 1 ? 0x7f : 0), K::native_expert_scratch_bytes(cap, FF));   // stale scratch
                            call(P[i]);
                            s->memcpy(got.data(), dout, ob).wait();
                            const float* a = (const float*) ref[i].data();
                            const float* b = (const float*) got.data();
                            for (size_t j = 0; j < ob / 4; ++j, ++nval) {
                                if (std::isfinite(a[j])) sumabs += std::fabs((double) a[j]);
                                if (std::memcmp(a + j, b + j, 4) != 0) {
                                    ++ndiff;
                                    worst = std::fmax(worst, std::fabs((double) a[j] - b[j]) / (std::fabs((double) a[j]) + 1e-6));
                                    maxabs = std::fmax(maxabs, std::fabs((double) a[j] - b[j]));
                                }
                            }
                        }
                        char buf[240];
                        if (ndiff) {
                            // worst rel: of one value; max abs / mean |ref|: the largest difference against the outputs' typical size
                            std::snprintf(buf, sizeof buf, " %d:DIFFERS(%zu of %zu, worst rel %.2e, max abs %.2e, mean |ref| %.2e)", v, ndiff,
                                          nval, worst, maxabs, sumabs / (double) nval);
                            ++failures;
                        } else {
                            std::snprintf(buf, sizeof buf, " %d:bitwise", v);
                        }
                        cmp += buf;
                    }
                }
                // times
                std::vector<int> all = {0};
                all.insert(all.end(), vs.begin(), vs.end());
                const int nph = phases ? 5 : 1;
                const int ph_id[5] = {0, 3, 1, 2, 0};
                const char* ph_name[5] = {"all", "gu", "gu+sw", "down", "empty"};
                std::vector<std::vector<std::vector<double>>> us(all.size(), std::vector<std::vector<double>>(nph));
                // one executable graph per (variant, phase): every plan's call
                std::vector<std::vector<GraphExec*>> ge(all.size(), std::vector<GraphExec*>(nph, nullptr));
                auto issue = [&](int v, int ph) {
                    K::native_expert_set_q4kq5(v);
                    K::native_expert_set_mode(-1, ph_id[ph]);
                    for (const auto& p : P) call(p, ph == 4);
                    K::native_expert_set_mode(-1, 0);
                };
                if (graph)
                    for (size_t k = 0; k < all.size(); ++k)
                        for (int ph = 0; ph < nph; ++ph) {
                            dpct::experimental::command_graph_ptr g;
                            dpct::experimental::begin_recording(s);
                            issue(all[k], ph);
                            dpct::experimental::end_recording(s, &g);
                            ge[k][ph] = new GraphExec(g->finalize());
                            delete g;
                        }
                auto time = [&](size_t k, int ph) {
                    K::native_expert_set_q4kq5(all[k]);
                    K::native_expert_set_mode(-1, 0);
                    for (int i = 0; i < 8 && i < (int) P.size(); ++i) call(P[i]);   // warm, and hq for the down-only phase
                    s->wait();
                    const auto t0 = std::chrono::steady_clock::now();
                    if (graph) s->ext_oneapi_graph(*ge[k][ph]);
                    else issue(all[k], ph);
                    s->wait();
                    const auto t1 = std::chrono::steady_clock::now();
                    return std::chrono::duration<double, std::micro>(t1 - t0).count() / (double) P.size();
                };
                for (int tr = 0; tr < trials; ++tr)
                    for (size_t k = 0; k < all.size(); ++k)
                        for (int ph = 0; ph < nph; ++ph) us[k][ph].push_back(time(k, ph));
                K::native_expert_set_mode(-1, 0);
                std::printf("  T %d share %2d%% (groups %.1f, entries %.1f):%s\n", T, share, ng / plans, ne / plans, cmp.c_str());
                for (size_t k = 0; k < all.size(); ++k) {
                    std::printf("    v %3d:", all[k]);
                    for (int ph = 0; ph < nph; ++ph) {
                        const double m = median(us[k][ph]), m0 = median(us[0][ph]);
                        std::printf("  %s %7.1f us (%.3fx)", ph_name[ph], m, m0 / m);
                    }
                    std::printf("\n");
                }
                for (auto& row : ge)
                    for (auto* g : row) delete g;
                for (auto& p : P) release(p);
                sycl::free(d_zero, *g_q);
            }
        sycl::free(raw, *g_q);
    }
    K::native_expert_set_q4kq5(-1);
    std::printf("q4k_q5_expert_bench: %d differing\n", failures);
    return failures ? 1 : 0;
}
