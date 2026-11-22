// src/kernels/q4k_q5_expert_bench.cpp - the grouped decode experts of a Q4_K gate/up + Q5_0 down layer on the
// model's own rows: STRATA_EXPERT_Q4KQ5's kernels (src/kernels/cuda/q4k_q5_experts.cuh) against the old per-entry
// kernels (variant 0), every output byte for byte, and each one's time.
//
//     q4k_q5_expert_bench <any shard.gguf> <layer[,layer...]> [--t 1,2,3,4,5,6] [--share 0,50] [--plans 100]
//                         [--trials 7] [--vs 3,1,2,999] [--phases 0|1] [--alias K]
//
// --vs: STRATA_EXPERT_Q4KQ5 values (1 gate/up, 2 down, 3 both; 999 only reads the groups' blobs, the card's read
// rate - it "differs" by design).
// A plan is a verify window's VRAM call: T rows of top-10 routing over the layer's 512 experts (`share` % of a row's
// experts are the previous row's), groups in first-appearance order, entries in row order (resident_plan_kernel's).
// A window of two slots with one row each is two T = 1 calls (verify.cpp's post() runs per slot group), which is what
// the T = 1 timing loop issues back to back.  Timing: all plans of a (T, share) back to back, one stream
// synchronize; trials interleave the variants; the median is printed.  --phases 1 adds gate/up alone, gate/up +
// SwiGLU/q8_1 and down alone.  --alias K points every group at one of K blobs (they stay in the card's caches): what
// the kernels cost without the weight reads from memory.
#include "strata/artifact/gguf_reader.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/iq_kernels.hpp"

#include "ggml.h"

#include <cuda_runtime.h>

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

#define QB_CK(x)                                                                                                    \
    do {                                                                                                            \
        cudaError_t e_ = (x);                                                                                       \
        if (e_ != cudaSuccess) {                                                                                    \
            std::fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_));                 \
            std::exit(2);                                                                                           \
        }                                                                                                           \
    } while (0)

struct Plan {
    int ng = 0, ne = 0, t = 0;
    std::vector<int> gid, start, tok, dst;
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

int g_alias = 0;   // --alias K: group g reads blob (expert % K): K blobs stay in the caches, the kernels' compute time

void upload(Plan& p, const uint8_t* base, size_t stride) {
    std::vector<unsigned long long> hp(p.ng);
    for (int g = 0; g < p.ng; ++g)
        hp[g] = (unsigned long long) (base + (size_t) (g_alias > 0 ? p.gid[g] % g_alias : p.gid[g]) * stride);
    QB_CK(cudaMalloc((void**) &p.d_ptr, hp.size() * 8));
    QB_CK(cudaMalloc((void**) &p.d_start, (size_t) (p.ng + 1) * 4));
    QB_CK(cudaMalloc((void**) &p.d_ng, 4));
    QB_CK(cudaMalloc((void**) &p.d_tok, (size_t) p.ne * 4));
    QB_CK(cudaMalloc((void**) &p.d_dst, (size_t) p.ne * 4));
    QB_CK(cudaMemcpy(p.d_ptr, hp.data(), hp.size() * 8, cudaMemcpyHostToDevice));
    QB_CK(cudaMemcpy(p.d_start, p.start.data(), (size_t) (p.ng + 1) * 4, cudaMemcpyHostToDevice));
    QB_CK(cudaMemcpy(p.d_ng, &p.ng, 4, cudaMemcpyHostToDevice));
    QB_CK(cudaMemcpy(p.d_tok, p.tok.data(), (size_t) p.ne * 4, cudaMemcpyHostToDevice));
    QB_CK(cudaMemcpy(p.d_dst, p.dst.data(), (size_t) p.ne * 4, cudaMemcpyHostToDevice));
}
void release(Plan& p) {
    cudaFree(p.d_ptr); cudaFree(p.d_start); cudaFree(p.d_ng); cudaFree(p.d_tok); cudaFree(p.d_dst);
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

}  // namespace

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    if (argc < 3) {
        std::fprintf(stderr, "usage: q4k_q5_expert_bench <shard.gguf> <layer[,layer]> [--t 1,2,3,4,5,6] [--share 0,50] "
                             "[--plans 100] [--trials 7] [--vs 3,...] [--phases 0|1] [--alias K]\n");
        return 2;
    }
    const std::vector<int> layers = ints(argv[2]);
    std::vector<int> Ts = { 1, 2, 3, 4, 5, 6 }, shares = { 0, 50 }, vs = { 3 };
    int plans = 100, trials = 7, phases = 0;
    for (int i = 3; i + 1 < argc; i += 2) {
        const std::string a = argv[i];
        if (a == "--t") Ts = ints(argv[i + 1]);
        else if (a == "--share") shares = ints(argv[i + 1]);
        else if (a == "--plans") plans = std::atoi(argv[i + 1]);
        else if (a == "--trials") trials = std::atoi(argv[i + 1]);
        else if (a == "--vs") vs = ints(argv[i + 1]);
        else if (a == "--phases") phases = std::atoi(argv[i + 1]);
        else if (a == "--alias") g_alias = std::atoi(argv[i + 1]);
        else { std::fprintf(stderr, "unknown option %s\n", a.c_str()); return 2; }
    }
    std::unique_ptr<strata::GgufModel> model;
    try {
        model = std::make_unique<strata::GgufModel>(strata::GgufModel::open(argv[1]));
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "q4k_q5_expert_bench: %s\n", ex.what());
        return 2;
    }
    cudaDeviceProp prop{};
    QB_CK(cudaGetDeviceProperties(&prop, 0));
    std::printf("device 0: %s; plans %d, trials %d\n", prop.name, plans, trials);
    const int64_t H = 2560, FF = 640;
    constexpr int kTokMax = 8;
    int failures = 0;
    cudaStream_t s;
    QB_CK(cudaStreamCreate(&s));
    std::mt19937 rng(20261122);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> x((size_t) kTokMax * H);
    for (auto& v : x) v = nd(rng);
    void *dx = nullptr, *dxq = nullptr, *dscr = nullptr;
    float* dout = nullptr;
    const int64_t capmax = (int64_t) kTokMax * 10;
    QB_CK(cudaMalloc(&dx, x.size() * 4));
    QB_CK(cudaMalloc(&dxq, (size_t) kTokMax * H / 32 * 36));
    QB_CK(cudaMalloc(&dscr, K::native_expert_scratch_bytes(capmax, FF)));
    QB_CK(cudaMalloc((void**) &dout, (size_t) capmax * H * 4));
    QB_CK(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice));
    K::quantize_q8_1_rows((const float*) dx, kTokMax, H, dxq, s);
    for (int l : layers) {
        const char* roles[3] = { "gate", "up", "down" };
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
        std::vector<uint8_t> host((size_t) 512 * f.bytes);
        const size_t dn_role = f.bytes - f.down_off;
        for (int e = 0; e < 512; ++e) {
            uint8_t* b = host.data() + (size_t) e * f.bytes;
            std::memcpy(b, model->shard(sh[0]).tensor_data(*t[0]) + (size_t) e * f.up_off, f.up_off);
            std::memcpy(b + f.up_off, model->shard(sh[1]).tensor_data(*t[1]) + (size_t) e * f.up_off, f.up_off);
            std::memcpy(b + f.down_off, model->shard(sh[2]).tensor_data(*t[2]) + (size_t) e * dn_role, dn_role);
        }
        uint8_t* raw = nullptr;
        QB_CK(cudaMalloc((void**) &raw, host.size()));
        QB_CK(cudaMemcpy(raw, host.data(), host.size(), cudaMemcpyHostToDevice));
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
                    upload(P.back(), raw, f.bytes);
                    ng += P.back().ng;
                    ne += P.back().ne;
                }
                const int64_t cap = (int64_t) T * 10;
                const size_t ob = (size_t) cap * H * 4;
                auto call = [&](const Plan& p) {
                    K::native_expert_grouped(L, p.d_ptr, p.d_start, p.d_ng, p.d_dst, p.d_tok, cap, cap, dxq, dscr, dout, s, 0);
                };
                // every plan's outputs: variant 0, then each variant
                std::vector<std::vector<uint8_t>> ref(P.size(), std::vector<uint8_t>(ob));
                K::native_expert_set_mode(-1, 0);
                K::native_expert_set_q4kq5(0);
                for (size_t i = 0; i < P.size(); ++i) {
                    QB_CK(cudaMemsetAsync(dout, 0xff, ob, s));
                    call(P[i]);
                    QB_CK(cudaMemcpyAsync(ref[i].data(), dout, ob, cudaMemcpyDeviceToHost, s));
                    QB_CK(cudaStreamSynchronize(s));
                }
                std::vector<uint8_t> got(ob);
                std::string cmp;
                for (int v : vs) {
                    K::native_expert_set_q4kq5(v);
                    size_t ndiff = 0, nval = 0;
                    double worst = 0;
                    for (size_t i = 0; i < P.size(); ++i) {
                        QB_CK(cudaMemsetAsync(dout, 0xff, ob, s));
                        call(P[i]);
                        QB_CK(cudaMemcpyAsync(got.data(), dout, ob, cudaMemcpyDeviceToHost, s));
                        QB_CK(cudaStreamSynchronize(s));
                        const float* a = (const float*) ref[i].data();
                        const float* b = (const float*) got.data();
                        for (size_t j = 0; j < ob / 4; ++j, ++nval)
                            if (std::memcmp(a + j, b + j, 4) != 0) {
                                ++ndiff;
                                worst = std::fmax(worst, std::fabs((double) a[j] - b[j]) / (std::fabs((double) a[j]) + 1e-6));
                            }
                    }
                    char buf[160];
                    if (ndiff) {
                        std::snprintf(buf, sizeof buf, " %d:DIFFERS(%zu of %zu, worst rel %.2e)", v, ndiff, nval, worst);
                        ++failures;
                    } else {
                        std::snprintf(buf, sizeof buf, " %d:bitwise", v);
                    }
                    cmp += buf;
                }
                // times
                std::vector<int> all = { 0 };
                all.insert(all.end(), vs.begin(), vs.end());
                const int nph = phases ? 4 : 1;
                const int ph_id[4] = { 0, 3, 1, 2 };
                const char* ph_name[4] = { "all", "gu", "gu+sw", "down" };
                std::vector<std::vector<std::vector<double>>> us(all.size(), std::vector<std::vector<double>>(nph));
                auto time = [&](int v, int ph) {
                    K::native_expert_set_q4kq5(v);
                    K::native_expert_set_mode(-1, 0);
                    for (int i = 0; i < 8 && i < (int) P.size(); ++i) call(P[i]);   // warm, and hq for the down-only phase
                    K::native_expert_set_mode(-1, ph);
                    QB_CK(cudaStreamSynchronize(s));
                    const auto t0 = std::chrono::steady_clock::now();
                    for (const auto& p : P) call(p);
                    QB_CK(cudaStreamSynchronize(s));
                    const auto t1 = std::chrono::steady_clock::now();
                    return std::chrono::duration<double, std::micro>(t1 - t0).count() / (double) P.size();
                };
                for (int tr = 0; tr < trials; ++tr)
                    for (size_t k = 0; k < all.size(); ++k)
                        for (int ph = 0; ph < nph; ++ph) us[k][ph].push_back(time(all[k], ph_id[ph]));
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
                for (auto& p : P) release(p);
            }
        cudaFree(raw);
    }
    K::native_expert_set_q4kq5(-1);
    std::printf("q4k_q5_expert_bench: %d differing\n", failures);
    return failures ? 1 : 0;
}
