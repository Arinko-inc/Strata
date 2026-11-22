// rdna_bench: the gfx11 (RDNA3) decode kernels ported from the community fork (rdna_gr.cu, rdna_mmvq.cu) against the
// default kernels, at this model's shapes: time per call (a rotating working set larger than the card's cache, inside
// one graph) and the difference of the outputs.  After the fork's src/kernels/claude_bench.cpp.
//
//   rdna_bench gr   [--pack <pack dir>] [--layers 18-29] [--t 1,2,3,4]
//       the hyper-connection read: fused_gr_read_multi (the variant fused_gr_check picks, as the engine) against
//       rdna_gr_read_multi, on the BF16 weights and on STRATA_HC_REQ8's int8 image of them (STRATA_GR_RDNA_A8=0 in the
//       environment: the rdna int8 read keeps FP32 activations).
//       --pack: the real norm / down / up / inject matrices of the layers' two halves (dense.bin through index.txt);
//       without it random weights.
//   rdna_bench mmvq [--gguf <any shard>] [--layers 18-29] [--t 1,2,3,4] [--dump <file>]
//       the dense GEMVs: native_mmvq against rdna_kq_mmvq / rdna_q6k_mmvq (every column count: the bench sets
//       STRATA_RDNA_KQ_ALL).  --gguf: every 2-D K-quant / Q5_0 / Q8_0 matrix of the layers the engine reads from the
//       GGUF; without it random bytes at the GDN shapes.  --dump: native_mmvq's outputs (first replica) appended to a
//       file, to compare two processes (STRATA_RDNA_MMVQ_WAVE=3 against unset) with cmp.
// Run with STRATA_GR_RDNA and STRATA_RDNA_Q6K unset: set, they redirect the default calls as in the engine, and the
// bench then checks that redirection (every rdna column bitwise its default).
#include "strata/artifact/gguf_reader.hpp"
#include "strata/kernels/fused_gr.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <vector>
namespace K = strata::kernels;
static void ck(cudaError_t e, const char* at) { if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", at, cudaGetErrorString(e)); std::exit(2); } }
static cudaStream_t st;
static void* dupload(const void* h, size_t bytes) {
    void* p; ck(cudaMalloc(&p, bytes), "malloc"); ck(cudaMemcpy(p, h, bytes, cudaMemcpyHostToDevice), "upload"); return p;
}
static void* dalloc_rand(size_t bytes, unsigned seed, bool sane_fp16 = true) {
    std::vector<uint8_t> h(bytes); std::mt19937 r(seed);
    for (size_t i = 0; i < bytes; ++i) h[i] = (uint8_t) r();
    if (sane_fp16) for (size_t i = 1; i < bytes; i += 2) h[i] &= 0x3b;   // keep fp16/bf16 high bytes finite and small
    return dupload(h.data(), bytes);
}
static std::vector<float> rand_floats(size_t n, unsigned seed) {
    std::vector<float> v(n); std::mt19937 r(seed); std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    for (auto& x : v) x = u(r);
    return v;
}
// time `calls` launches rotating over `reps` replicas inside one graph; us per call (best of `trials`)
static double timeit(int reps, int calls, const std::function<void(int)>& f, int trials = 7) {
    for (int i = 0; i < reps; ++i) f(i);
    ck(cudaStreamSynchronize(st), "warm");
    cudaGraph_t g; cudaGraphExec_t e;
    ck(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal), "cap");
    for (int i = 0; i < calls; ++i) f(i % reps);
    ck(cudaStreamEndCapture(st, &g), "capend"); ck(cudaGraphInstantiate(&e, g, nullptr, nullptr, 0), "inst");
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    ck(cudaGraphLaunch(e, st), "l0"); ck(cudaStreamSynchronize(st), "s0");
    double best = 1e30;
    for (int r = 0; r < trials; ++r) {
        cudaEventRecord(a, st); cudaGraphLaunch(e, st); cudaEventRecord(b, st); cudaEventSynchronize(b);
        float ms; cudaEventElapsedTime(&ms, a, b); best = std::min(best, ms * 1000.0 / calls);
    }
    cudaEventDestroy(a); cudaEventDestroy(b);
    cudaGraphExecDestroy(e); cudaGraphDestroy(g);
    return best;
}
struct Diff { size_t n = 0, bits = 0; double maxabs = 0, maxref = 0; bool nonfinite = false; };
static Diff diff(const float* a, const float* b, size_t n) {
    Diff d; d.n = n;
    for (size_t i = 0; i < n; ++i) {
        uint32_t x, y; std::memcpy(&x, a + i, 4); std::memcpy(&y, b + i, 4);
        if (x != y) ++d.bits;
        if (!std::isfinite(b[i])) d.nonfinite = true;
        d.maxabs = std::max(d.maxabs, (double) std::fabs(a[i] - b[i])); d.maxref = std::max(d.maxref, (double) std::fabs(a[i]));
    }
    return d;
}
static std::string dstr(const Diff& d) {
    char b[128];
    if (d.bits == 0) std::snprintf(b, sizeof b, "bitwise");
    else std::snprintf(b, sizeof b, "%zu/%zu differ, max|d|/max|ref| %.2e%s", d.bits, d.n, d.maxabs / (d.maxref + 1e-30), d.nonfinite ? " NON-FINITE" : "");
    return b;
}
static std::vector<int> int_list(const std::string& s) {   // "18-29" or "1,2,4"
    std::vector<int> o;
    const size_t dash = s.find('-');
    if (dash != std::string::npos && s.find(',') == std::string::npos) {
        for (int i = std::atoi(s.substr(0, dash).c_str()); i <= std::atoi(s.substr(dash + 1).c_str()); ++i) o.push_back(i);
        return o;
    }
    std::stringstream ss(s); std::string t;
    while (std::getline(ss, t, ',')) if (!t.empty()) o.push_back(std::atoi(t.c_str()));
    return o;
}

// ---- the hyper-connection read
constexpr int N = 2560, HC = 4, D = N * HC, LR = 320;
struct HcW { float* norm; uint16_t* down; uint16_t* up; uint16_t* inject; int8_t* qd; float* sd; int8_t* qu; float* su; };
static float bf16f(uint16_t h) { const uint32_t u = (uint32_t) h << 16; float f; std::memcpy(&f, &u, 4); return f; }
// STRATA_HC_REQ8's int8 image of a BF16 matrix (WeightTable::load_hc_q8's rule): per 32 contiguous values
// d = amax / 127 (fp32), q = round-to-nearest-even(x / d)
static void make_q8(const uint16_t* bf, size_t n, int8_t** q, float** sc) {
    std::vector<int8_t> hq(n); std::vector<float> hs(n / 32);
    for (size_t b = 0; b < n / 32; ++b) {
        float amax = 0.f;
        for (int j = 0; j < 32; ++j) amax = std::max(amax, std::fabs(bf16f(bf[b * 32 + j])));
        const float d = amax / 127.f, id = d > 0.f ? 1.f / d : 0.f;
        hs[b] = d;
        for (int j = 0; j < 32; ++j) hq[b * 32 + j] = (int8_t) std::nearbyint(bf16f(bf[b * 32 + j]) * id);
    }
    *q = (int8_t*) dupload(hq.data(), n); *sc = (float*) dupload(hs.data(), n / 32 * 4);
}
static int run_gr(const std::string& pack, const std::vector<int>& layers, const std::vector<int>& ts) {
    K::fused_gr_check();
    std::vector<HcW> w;
    const size_t nm = (size_t) D * LR;
    if (!pack.empty()) {
        std::map<std::string, std::pair<uint64_t, uint64_t>> idx;
        std::ifstream in(pack + "/index.txt"); std::string line;
        while (std::getline(in, line)) {
            if (line.empty() || line[0] == '#') continue;
            std::stringstream ss(line); std::string name; long long a, ty; uint64_t off, bytes;
            if (ss >> name >> a >> ty >> off >> bytes) idx[name] = {off, bytes};
        }
        FILE* f = std::fopen((pack + "/dense.bin").c_str(), "rb");
        if (!f) { std::fprintf(stderr, "cannot open %s/dense.bin\n", pack.c_str()); return 2; }
        auto rd = [&](const std::string& name, size_t want) {
            const auto it = idx.find(name);
            if (it == idx.end() || it->second.second != want) { std::fprintf(stderr, "pack: %s missing or not %zu bytes\n", name.c_str(), want); std::exit(2); }
            std::vector<uint8_t> h(want);
            if (fseeko(f, (off_t) it->second.first, SEEK_SET) != 0 || std::fread(h.data(), 1, want, f) != want) { std::fprintf(stderr, "pack: read %s\n", name.c_str()); std::exit(2); }
            return h;
        };
        for (int l : layers)
            for (const char* half : {"attn", "ffn"}) {
                const std::string p = "blk." + std::to_string(l) + ".hc_" + half + "_";
                const auto n = rd(p + "norm.weight", (size_t) D * 4), d = rd(p + "down.weight", nm * 2), u = rd(p + "up.weight", nm * 2),
                           i = rd(p + "inject.weight", (size_t) HC * D * 2);
                HcW x;
                x.norm = (float*) dupload(n.data(), n.size()); x.down = (uint16_t*) dupload(d.data(), d.size());
                x.up = (uint16_t*) dupload(u.data(), u.size()); x.inject = (uint16_t*) dupload(i.data(), i.size());
                make_q8((const uint16_t*) d.data(), nm, &x.qd, &x.sd); make_q8((const uint16_t*) u.data(), nm, &x.qu, &x.su);
                w.push_back(x);
            }
        std::fclose(f);
        std::printf("gr: real weights, %zu reads (layers x 2 halves) from %s\n", w.size(), pack.c_str());
    } else {
        for (int i = 0; i < 24; ++i) {
            HcW x;
            const auto n = rand_floats(D, 11 + i);
            x.norm = (float*) dupload(n.data(), (size_t) D * 4);
            x.down = (uint16_t*) dalloc_rand(nm * 2, 100 + i); x.up = (uint16_t*) dalloc_rand(nm * 2, 200 + i);
            x.inject = (uint16_t*) dalloc_rand((size_t) HC * D * 2, 300 + i);
            // random BF16 bytes are not a sane matrix (huge exponents): the int8 image is made from values in [-1, 1]
            const auto fd = rand_floats(nm, 500 + i), fu = rand_floats(nm, 600 + i);
            std::vector<uint16_t> bd(nm), bu(nm);
            for (size_t k = 0; k < nm; ++k) { uint32_t u; std::memcpy(&u, &fd[k], 4); bd[k] = (uint16_t) (u >> 16); std::memcpy(&u, &fu[k], 4); bu[k] = (uint16_t) (u >> 16); }
            cudaFree(x.down); cudaFree(x.up);
            x.down = (uint16_t*) dupload(bd.data(), nm * 2); x.up = (uint16_t*) dupload(bu.data(), nm * 2);
            make_q8(bd.data(), nm, &x.qd, &x.sd); make_q8(bu.data(), nm, &x.qu, &x.su);
            w.push_back(x);
        }
        std::printf("gr: random weights in [-1, 1], %zu replicas\n", w.size());
    }
    const int reps = (int) w.size();
    const auto hR = rand_floats(8 * D, 3), hbo = rand_floats(8 * N, 4), hinj = rand_floats(8 * HC, 5);
    float *R = (float*) dupload(hR.data(), hR.size() * 4), *bo = (float*) dupload(hbo.data(), hbo.size() * 4),
          *inj = (float*) dupload(hinj.data(), hinj.size() * 4);
    float *xn, *lo, *rs, *io, *mix;
    ck(cudaMalloc(&xn, 8 * D * 4 * 2), "xn"); ck(cudaMalloc(&lo, 8 * LR * 4), "lo"); ck(cudaMalloc(&rs, 8 * HC * 4), "rs");
    ck(cudaMalloc(&io, 8 * HC * 4), "io"); ck(cudaMalloc(&mix, 8 * N * 4), "mix");
    const char* a8 = std::getenv("STRATA_GR_RDNA_A8");
    std::printf("gr: int8 reads: default = fused_gr's STRATA_HC_REQ8 read, rdna = %s\n", a8 && a8[0] == '0' ? "int8 codes x FP32 activations (STRATA_GR_RDNA_A8=0)" : "int8 codes x int8 activations (A8)");
    for (int T : ts) {
        // mode 0: the default read, 1: rdna, both on BF16; 2: the default read, 3: rdna, both on the int8 codes
        auto call = [&](int r, int mode) {
            K::FusedGrArgs a[8];
            for (int t = 0; t < T; ++t) {
                auto& x = a[t]; const HcW& p = w[r];
                x.R = R + t * D; x.R_out = R + t * D; x.apply = true; x.bo_prev = bo + t * N; x.inj_prev = inj + t * HC;
                x.w_norm = p.norm; x.w_inject = p.inject;
                if (mode >= 2) { x.w_down = (const uint16_t*) p.qd; x.w_up = (const uint16_t*) p.qu; x.s_down = p.sd; x.s_up = p.su; }
                else { x.w_down = p.down; x.w_up = p.up; }
                x.lo = lo + t * LR; x.rs = rs + t * HC; x.inject_out = io + t * HC; x.mixed = mix + t * N;
            }
            if (mode == 0 || mode == 2) K::fused_gr_read_multi(a, T, xn, st); else K::rdna_gr_read_multi(a, T, xn, st);
        };
        // outputs of one read (the pending write applied: R restored before each); R' itself is compared too
        auto grab = [&](int r, int mode) {
            ck(cudaMemcpy(R, hR.data(), hR.size() * 4, cudaMemcpyHostToDevice), "R"); ck(cudaMemcpy(bo, hbo.data(), hbo.size() * 4, cudaMemcpyHostToDevice), "bo");
            call(r, mode); ck(cudaStreamSynchronize(st), "s");
            std::vector<float> o((size_t) T * (LR + HC + HC + N + D));
            float* p = o.data();
            ck(cudaMemcpy(p, lo, T * LR * 4, cudaMemcpyDeviceToHost), "c1"); p += T * LR;
            ck(cudaMemcpy(p, rs, T * HC * 4, cudaMemcpyDeviceToHost), "c2"); p += T * HC;
            ck(cudaMemcpy(p, io, T * HC * 4, cudaMemcpyDeviceToHost), "c3"); p += T * HC;
            ck(cudaMemcpy(p, mix, T * N * 4, cudaMemcpyDeviceToHost), "c4"); p += T * N;
            ck(cudaMemcpy(p, R, (size_t) T * D * 4, cudaMemcpyDeviceToHost), "c5");
            return o;
        };
        // lo, rs, inject, mixed, R' - the worst over the replicas compared.  Against the default BF16 read (the engine
        // today): rdna BF16, the default int8 read, rdna int8
        Diff dd[3][5];
        const int ncmp = std::min(reps, 4);
        for (int r = 0; r < ncmp; ++r) {
            const auto o0 = grab(r, 0);
            const size_t off[6] = {0, (size_t) T * LR, (size_t) T * (LR + HC), (size_t) T * (LR + 2 * HC), (size_t) T * (LR + 2 * HC + N), o0.size()};
            for (int v = 0; v < 3; ++v) {
                const auto o = grab(r, v + 1);
                for (int k = 0; k < 5; ++k) {
                    const Diff n = diff(o0.data() + off[k], o.data() + off[k], off[k + 1] - off[k]);
                    if (n.maxabs / (n.maxref + 1e-30) >= dd[v][k].maxabs / (dd[v][k].maxref + 1e-30)) dd[v][k] = n;
                }
            }
        }
        {   // the int8 pair against each other (the hook check: bitwise when STRATA_GR_RDNA=1 redirects the default read)
            Diff w0;
            for (int r = 0; r < ncmp; ++r) {
                const auto a2 = grab(r, 2), a3 = grab(r, 3);
                const Diff n = diff(a2.data(), a3.data(), a2.size());
                if (n.maxabs / (n.maxref + 1e-30) >= w0.maxabs / (w0.maxref + 1e-30)) w0 = n;
            }
            std::printf("gr T%d: int8 rdna vs int8 default, every output: %s\n", T, dstr(w0).c_str());
        }
        ck(cudaMemset(bo, 0, 8 * N * 4), "bo0");   // timing: the in-place R update idempotent
        ck(cudaMemcpy(R, hR.data(), hR.size() * 4, cudaMemcpyHostToDevice), "R");
        double tm[4];
        for (int v = 0; v < 4; ++v) tm[v] = timeit(reps, 480, [&](int r) { call(r, v); });
        std::printf("gr T%d: bf16 default %7.2f us | bf16 rdna %7.2f us (%.3fx) | int8 default %7.2f us (%.3fx) | int8 rdna %7.2f us (%.3fx)\n", T, tm[0], tm[1],
                    tm[0] / tm[1], tm[2], tm[0] / tm[2], tm[3], tm[0] / tm[3]);
        const char* nmv[5] = {"lo", "rs", "inject", "mixed", "R'"};
        const char* vn[3] = {"bf16 rdna   ", "int8 default", "int8 rdna   "};
        for (int v = 0; v < 3; ++v)
            for (int k = 0; k < 5; ++k) std::printf("    %s vs bf16 default: %-6s %s\n", vn[v], nmv[k], dstr(dd[v][k]).c_str());
        std::fflush(stdout);
    }
    return 0;
}

// ---- the dense GEMVs
static int run_mmvq(const std::string& gguf, const std::vector<int>& layers, const std::vector<int>& ts, const std::string& dump) {
    setenv("STRATA_RDNA_KQ_ALL", "1", 1);   // the Q4_K / Q5_K kernels at every column count (the engine: from 4)
    struct M { std::string name; int type, nin, nout; std::vector<void*> w; size_t bytes; };
    std::vector<M> ms;
    std::unique_ptr<strata::GgufModel> model;
    if (!gguf.empty()) {
        model = std::make_unique<strata::GgufModel>(strata::GgufModel::open(gguf));
        std::map<std::string, size_t> at;
        for (int l : layers) {
            const std::string pre = "blk." + std::to_string(l) + ".";
            for (size_t s = 0; s < model->size(); ++s)
                for (const auto& t : model->shard(s).tensors()) {
                    if (t.name.rfind(pre, 0) != 0 || t.shape.size() != 2) continue;
                    const std::string suf = t.name.substr(pre.size());
                    if (suf.rfind("hc_", 0) == 0 || suf.rfind("ssm_alpha", 0) == 0 || suf.rfind("ssm_beta", 0) == 0) continue;   // BF16 in the pack
                    if (t.type != 12 && t.type != 13 && t.type != 14 && t.type != 6 && t.type != 8) continue;
                    const int nin = (int) t.shape[0], nout = (int) t.shape[1];
                    if (nin % ((t.type == 6 || t.type == 8) ? 32 : 256) != 0) continue;   // K-quants: whole superblocks
                    const std::string key = suf + " " + t.type_name() + " " + std::to_string(nin) + "->" + std::to_string(nout);
                    if (!at.count(key)) { at[key] = ms.size(); ms.push_back({key, (int) t.type, nin, nout, {}, K::native_mmvq_weight_bytes((int) t.type, nin, nout)}); }
                    M& m = ms[at[key]];
                    m.w.push_back(dupload(model->shard(s).tensor_data(t), m.bytes));
                }
        }
        std::printf("mmvq: real weights from %s\n", gguf.c_str());
    } else {
        const struct { int type, nin, nout; const char* name; } sh[] = {{12, 2560, 10240, "qkv Q4_K 2560->10240"}, {12, 2560, 6144, "gate Q4_K 2560->6144"},
                                                                       {12, 6144, 2560, "out Q4_K 6144->2560"}, {12, 2560, 640, "shexp Q4_K 2560->640"}};
        for (const auto& s : sh) {
            M m{s.name, s.type, s.nin, s.nout, {}, K::native_mmvq_weight_bytes(s.type, s.nin, s.nout)};
            const int reps = (int) std::min<size_t>(64, (384ull << 20) / m.bytes + 1);
            for (int i = 0; i < reps; ++i) m.w.push_back(dalloc_rand(m.bytes, 100 + i));
            ms.push_back(m);
        }
        std::printf("mmvq: random weight bytes\n");
    }
    constexpr int MAXIN = 12288, MAXT = 8;
    void* xq; ck(cudaMalloc(&xq, (size_t) MAXT * MAXIN / 32 * 36 + 4096), "xq");
    const auto hx = rand_floats((size_t) MAXT * MAXIN, 9);
    float* x = (float*) dupload(hx.data(), hx.size() * 4);
    FILE* df = dump.empty() ? nullptr : std::fopen(dump.c_str(), "wb");
    for (M& m : ms) {
        if (m.nin > MAXIN) continue;
        const int reps = (int) m.w.size();
        float *y, *y1, *y2;
        ck(cudaMalloc(&y, (size_t) MAXT * m.nout * 4), "y"); ck(cudaMalloc(&y1, (size_t) MAXT * m.nout * 4), "y1"); ck(cudaMalloc(&y2, (size_t) MAXT * m.nout * 4), "y2");
        // T columns: column j is the input x[j * n_in ..]
        const bool has = m.type == 12 || m.type == 13 || m.type == 14;
        for (int T : ts) {
            K::native_quantize_q8_1(x, xq, m.nin, T, st);
            Diff worst;
            const int ncmp = std::min(reps, 4);
            for (int r = 0; r < ncmp; ++r) {
                ck(cudaMemset(y1, 0, (size_t) T * m.nout * 4), "z1"); ck(cudaMemset(y2, 0, (size_t) T * m.nout * 4), "z2");
                K::native_mmvq(m.type, m.w[r], xq, y1, m.nin, m.nout, T, st);
                bool ok = false;
                if (has) ok = m.type == 14 ? K::rdna_q6k_mmvq(m.w[r], xq, y2, m.nin, m.nout, T, st) : K::rdna_kq_mmvq(m.type, m.w[r], xq, y2, m.nin, m.nout, T, st);
                ck(cudaStreamSynchronize(st), "sync");
                std::vector<float> a((size_t) T * m.nout), b((size_t) T * m.nout);
                ck(cudaMemcpy(a.data(), y1, a.size() * 4, cudaMemcpyDeviceToHost), "a"); ck(cudaMemcpy(b.data(), y2, b.size() * 4, cudaMemcpyDeviceToHost), "b");
                if (df && r == 0) std::fwrite(a.data(), 4, a.size(), df);
                if (ok) { const Diff d = diff(a.data(), b.data(), a.size()); if (d.maxabs / (d.maxref + 1e-30) >= worst.maxabs / (worst.maxref + 1e-30)) worst = d; }
            }
            const double t0 = timeit(reps, 300, [&](int r) { K::native_mmvq(m.type, m.w[r], xq, y, m.nin, m.nout, T, st); });
            if (has) {
                const double t1 = timeit(reps, 300, [&](int r) { if (m.type == 14) K::rdna_q6k_mmvq(m.w[r], xq, y, m.nin, m.nout, T, st); else K::rdna_kq_mmvq(m.type, m.w[r], xq, y, m.nin, m.nout, T, st); });
                std::printf("%-44s T%d (%2d reps %5.1f MB): native %7.2f us | rdna %7.2f us (%.3fx) | %s\n", m.name.c_str(), T, reps, m.bytes / 1e6, t0, t1, t0 / t1, dstr(worst).c_str());
            } else
                std::printf("%-44s T%d (%2d reps %5.1f MB): native %7.2f us\n", m.name.c_str(), T, reps, m.bytes / 1e6, t0);
            std::fflush(stdout);
        }
        cudaFree(y); cudaFree(y1); cudaFree(y2);
        for (void* p : m.w) cudaFree(p);
    }
    if (df) std::fclose(df);
    for (int T : ts) std::printf("quantize_q8_1 2560 T%d: %7.2f us\n", T, timeit(1, 500, [&](int) { K::native_quantize_q8_1(x, xq, 2560, T, st); }));
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: rdna_bench gr|mmvq [--pack dir] [--gguf shard] [--layers 18-29] [--t 1,2,3,4] [--dump file]\n"); return 2; }
    const std::string which = argv[1];
    std::string pack, gguf, dump; std::vector<int> layers = int_list("18-29"), ts = {1, 2, 3, 4};
    for (int i = 2; i + 1 < argc; i += 2) {
        const std::string k = argv[i], v = argv[i + 1];
        if (k == "--pack") pack = v; else if (k == "--gguf") gguf = v; else if (k == "--layers") layers = int_list(v);
        else if (k == "--t") ts = int_list(v); else if (k == "--dump") dump = v;
        else { std::fprintf(stderr, "unknown option %s\n", k.c_str()); return 2; }
    }
    if (std::getenv("STRATA_GR_RDNA") || std::getenv("STRATA_RDNA_Q6K"))
        std::printf("HOOK CHECK: STRATA_GR_RDNA / STRATA_RDNA_Q6K is set, so the default calls are redirected to the rdna kernels as in the engine: "
                    "every 'rdna' column must read bitwise against its default (and the times are the same kernel twice)\n");
    ck(cudaStreamCreate(&st), "stream");
    int dev = 0; cudaGetDevice(&dev); cudaDeviceProp pr; ck(cudaGetDeviceProperties(&pr, dev), "props");
    std::printf("device %d: %s\n", dev, pr.name);
    if (which == "gr") return run_gr(pack, layers, ts);
    if (which == "mmvq") return run_mmvq(gguf, layers, ts, dump);
    std::fprintf(stderr, "unknown bench %s\n", which.c_str());
    return 2;
}
