// src/kernels/cpu/pool_dep_bench.cpp - the CPU pool's native layer call (run_split_multi_native) with and without
// STRATA_POOL_DEP, on the decode shape: a few CPU experts per call, each a fresh read from DRAM.  No GPU, no model:
// --blobs expert blobs of ggml-quantized random weights (gate/up and down types as --pair), walked in order so a call's
// experts are cold.  Prints, per mode, the time per call and an FNV-1a hash of every output of the run: equal hashes =
// the same bits.  STRATA_KQ512 (read once per process by native_expert.cpp) is compared across two runs of this tool.
//
//     pool_dep_bench [--pair q4_K/q5_0] [--workers 15] [--experts 4] [--nt 1,1,2,1] [--blobs 512] [--calls 2000]
//                    [--reps 3] [--pin 0|1]
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/cpu/pool.hpp"

#include "ggml.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace cpu = strata::kernels::cpu;

namespace {
constexpr int64_t kH = 2560, kFF = 640;

int type_of(const std::string& s) {
    for (int t = 0; t < GGML_TYPE_COUNT; ++t) {
        const char* n = ggml_type_name((ggml_type) t);
        if (n != nullptr && s == n) return t;
    }
    return -1;
}
std::vector<int> ints(const std::string& s) {
    std::vector<int> v;
    size_t from = 0;
    for (;;) {
        const size_t c = s.find(',', from);
        v.push_back(std::atoi(s.substr(from, c == std::string::npos ? std::string::npos : c - from).c_str()));
        if (c == std::string::npos) break;
        from = c + 1;
    }
    return v;
}
void make_blob(const cpu::NativeFmt& f, uint8_t* blob, uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.f, 0.02f);
    std::vector<float> row((size_t) kH);
    auto quant = [&](int type, uint8_t* dst, int64_t n) { ggml_quantize_chunk((ggml_type) type, row.data(), dst, 0, 1, n, nullptr); };
    for (int64_t r = 0; r < 2 * kFF; ++r) {
        for (int64_t i = 0; i < kH; ++i) row[(size_t) i] = nd(rng);
        quant(f.gu_type, blob + (size_t) r * f.gu_row, kH);
    }
    for (int64_t r = 0; r < kH; ++r) {
        for (int64_t i = 0; i < kFF; ++i) row[(size_t) i] = nd(rng);
        quant(f.d_type, blob + f.down_off + (size_t) r * f.d_row, kFF);
    }
}
uint64_t fnv(uint64_t h, const void* p, size_t n) {
    const uint8_t* b = (const uint8_t*) p;
    for (size_t i = 0; i < n; ++i) { h ^= b[i]; h *= 1099511628211ull; }
    return h;
}
}  // namespace

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    std::string pair = "q4_K/q5_0", nts = "1,1,2,1";
    int workers = 15, experts = 4, blobs = 512, calls = 2000, reps = 3, pin = 1;
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string a = argv[i];
        if (a == "--pair") pair = argv[i + 1];
        else if (a == "--workers") workers = std::atoi(argv[i + 1]);
        else if (a == "--experts") experts = std::atoi(argv[i + 1]);
        else if (a == "--nt") nts = argv[i + 1];
        else if (a == "--blobs") blobs = std::atoi(argv[i + 1]);
        else if (a == "--calls") calls = std::atoi(argv[i + 1]);
        else if (a == "--reps") reps = std::atoi(argv[i + 1]);
        else if (a == "--pin") pin = std::atoi(argv[i + 1]);
        else { std::fprintf(stderr, "unknown %s\n", a.c_str()); return 2; }
    }
    const size_t sl = pair.find('/');
    cpu::NativeFmt f;
    std::string err;
    if (sl == std::string::npos || !cpu::native_fmt(type_of(pair.substr(0, sl)), type_of(pair.substr(sl + 1)), kH, kFF, f, err)) {
        std::fprintf(stderr, "pair %s: %s\n", pair.c_str(), err.c_str());
        return 2;
    }
    const std::vector<int> nt = ints(nts);
    experts = std::min(experts, (int) cpu::ExpertPool::kMaxSplitMulti);
    std::vector<uint8_t> store((size_t) blobs * f.bytes);
    {   // quantize in parallel (ggml's quantizer is the slow part of the setup)
        const int th = std::max(1, std::min(8, (int) std::thread::hardware_concurrency()));
        std::vector<std::thread> ts;
        for (int k = 0; k < th; ++k)
            ts.emplace_back([&, k] { for (int b = k; b < blobs; b += th) make_blob(f, store.data() + (size_t) b * f.bytes, 1000u + (uint32_t) b); });
        for (auto& t : ts) t.join();
    }
    std::vector<std::vector<uint8_t>> act(cpu::MAXT, std::vector<uint8_t>(cpu::kNativeActBytes));
    {
        std::mt19937 rng(7);
        std::normal_distribution<float> nd(0.f, 1.f);
        std::vector<float> x((size_t) kH);
        for (int t = 0; t < cpu::MAXT; ++t) {
            for (auto& v : x) v = nd(rng);
            cpu::native_quant_act(f, x.data(), act[(size_t) t].data());
        }
    }
    std::vector<float> out((size_t) experts * cpu::MAXT * kH);
    std::vector<cpu::ExpertJobMulti> jobs((size_t) experts);
    std::printf("pool_dep_bench: %s, %d workers (pin %d), %d experts a call, nt %s, %d blobs of %zu bytes, %d calls x %d reps\n",
                pair.c_str(), workers, pin, experts, nts.c_str(), blobs, (size_t) f.bytes, calls, reps);
    uint64_t hashes[2] = {0, 0};
    for (int rep = 0; rep < reps; ++rep)
        for (int dep = 0; dep < 2; ++dep) {
            setenv("STRATA_POOL_DEP", dep ? "1" : "0", 1);
            cpu::ExpertPool pool(workers, pin != 0);
            uint64_t h = 1469598103934665603ull;
            double ms = 0.0;
            int next = 0;
            for (int c = 0; c < calls; ++c) {
                for (int e = 0; e < experts; ++e) {
                    cpu::ExpertJobMulti& j = jobs[(size_t) e];
                    j.blob = store.data() + (size_t) next * f.bytes;
                    next = (next + 1) % blobs;
                    j.nt = std::max(1, std::min((int) cpu::MAXT, nt[(size_t) (e + c) % nt.size()]));
                    for (int t = 0; t < j.nt; ++t) {
                        j.nact[t] = act[(size_t) ((t + e + c) % cpu::MAXT)].data();
                        j.out[t] = out.data() + ((size_t) e * cpu::MAXT + (size_t) t) * kH;
                    }
                }
                const auto a = std::chrono::steady_clock::now();
                pool.run_split_multi_native(f, jobs.data(), experts);
                ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - a).count();
                for (int e = 0; e < experts; ++e)
                    for (int t = 0; t < jobs[(size_t) e].nt; ++t) h = fnv(h, jobs[(size_t) e].out[t], (size_t) kH * sizeof(float));
            }
            std::printf("  rep %d  STRATA_POOL_DEP=%d  %.1f us/call  hash %016llx\n", rep, dep, 1000.0 * ms / calls, (unsigned long long) h);
            if (rep == 0) hashes[dep] = h;
            else if (hashes[dep] != h) { std::printf("FAIL: hash changed between reps\n"); return 1; }
        }
    if (hashes[0] != hashes[1]) { std::printf("FAIL: STRATA_POOL_DEP changes the outputs\n"); return 1; }
    std::printf("PASS: the same bits with and without STRATA_POOL_DEP (hash %016llx)\n", (unsigned long long) hashes[0]);
    return 0;
}
