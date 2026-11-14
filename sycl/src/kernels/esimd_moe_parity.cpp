// esimd_moe_parity (151-b70): the fused int8 DPAS expert products (prefill/esimd_moe.dp.cpp) against
//   (a) the engine's dequantize + oneMKL path (FP16 weights, FP16 activations), the path they replace,
//   (b) an FP64 host reference from the blobs with the unquantized activations,
//   (c) an exact host reference on the SAME int8 activations the device produced (isolates the kernel arithmetic from the
//       activation quantization: only the float accumulation order differs),
//   (d) the whole expert (gate/up, SwiGLU, down) in FP64 from the FP16 activations, for both chains: sum |got - ref| / sum |ref|,
//       the measure native_expert_parity prints (CUDA's int8 path on the Q4_K_S pack: 2.1e-02 to 2.8e-02),
// for every gate/up x down type pair the kernels handle, and times both paths per expert.
//   esimd_moe_parity [mean_rows=80] [experts=16] [reps=10] [seed=1] [gate_up_type down_type]
//   esimd_moe_parity --gguf <shard.gguf>[,<shard.gguf>...] [--mean R] [--experts N] [--reps N] [--seed S] [--sweep R,R,...] layer ...
//   (--sweep: timing only, at each mean rows per expert on the engine's crossover row counts, no host references)
//   types: gate/up 21 IQ3_S, 18 IQ3_XXS, 22 IQ2_S, 12 Q4_K; down 20 IQ4_NL, 42 Q2_0, 6 Q5_0.
// Synthetic blobs are random bytes with sane block scales; --gguf takes N experts of each layer (ids spread over the layer) from the
// model's own tensors (the shards are searched in turn).  In a build with ggml (STRATA_NATIVE_EXPERTS) the host decoder below is
// checked against ggml's to_float on the blobs used.
// Exit status 1 when (c) is off by more than 2e-5 of the largest value, (b) exceeds the int8 activation error budget or (d) for the
// int8 chain exceeds 5e-2.
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/iq_dequant_expert.hpp"
#include "strata/prefill/esimd_moe.hpp"
#include "strata/prefill/kernels.hpp"
#include "strata/sycl_queue.hpp"
#include "strata/artifact/gguf_reader.hpp"
#ifdef STRATA_NATIVE_EXPERTS
#include "ggml.h"
#endif
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <dpct/blas_utils.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <vector>

#define GGML_TABLE_BEGIN(type, name, size) static const type name[size] = {
#define GGML_TABLE_END() };
#include "../prefill/iq_tables.inc"

namespace i8 = strata::prefill::i8;
static uint16_t f2h(float f) { return sycl::bit_cast<uint16_t>(sycl::half(f)); }
static float h2f(uint16_t h) { return (float) sycl::bit_cast<sycl::half>(h); }

static int block_bytes(int t) { return i8::block_bytes(t); }
static int block_vals(int t) { return (t == 21 || t == 18 || t == 22 || t == 12) ? 256 : t == 42 ? 64 : 32; }

// host decode of one row (K values) -> double
static void dec_row(int t, const uint8_t* row, int K, double* out) {
    const int bb = block_bytes(t), bv = block_vals(t);
    for (int b = 0; b < K / bv; ++b) {
        const uint8_t* blk = row + (size_t) b * bb;
        uint16_t dh; std::memcpy(&dh, blk, 2);
        const double d = h2f(dh);
        double* o = out + (size_t) b * bv;
        if (t == 21) {
            const uint8_t *qs = blk + 2, *qh = blk + 66, *sgn = blk + 74, *scl = blk + 106;
            for (int ib = 0; ib < 8; ++ib) {
                const double dd = d * (1 + 2 * ((scl[ib / 2] >> (4 * (ib % 2))) & 0xf));
                for (int r = 0; r < 8; ++r) {
                    const int idx = qs[8 * ib + r] | (((qh[ib] >> r) & 1) << 8);
                    const uint8_t* g = (const uint8_t*) &iq3s_grid[idx];
                    const uint8_t sb = sgn[4 * ib + r / 2];
                    for (int j = 0; j < 4; ++j) o[ib * 32 + 4 * r + j] = dd * ((sb >> (4 * (r & 1) + j) & 1) ? -(int) g[j] : (int) g[j]);
                }
            }
        } else if (t == 18) {
            const uint8_t* qs = blk + 2;
            for (int ib = 0; ib < 8; ++ib) {
                uint32_t aux; std::memcpy(&aux, blk + 66 + 4 * ib, 4);
                const double dd = d * (0.5 + (aux >> 28)) * 0.5;
                for (int l = 0; l < 4; ++l) {
                    const uint8_t sb = ksigns_iq2xs[(aux >> (7 * l)) & 127];
                    const uint8_t *g1 = (const uint8_t*) &iq3xxs_grid[qs[8 * ib + 2 * l]], *g2 = (const uint8_t*) &iq3xxs_grid[qs[8 * ib + 2 * l + 1]];
                    for (int j = 0; j < 4; ++j) {
                        o[ib * 32 + 8 * l + j] = dd * ((sb >> j & 1) ? -(int) g1[j] : (int) g1[j]);
                        o[ib * 32 + 8 * l + 4 + j] = dd * ((sb >> (4 + j) & 1) ? -(int) g2[j] : (int) g2[j]);
                    }
                }
            }
        } else if (t == 22) {
            const uint8_t *qs = blk + 2, *qh = blk + 66, *scl = blk + 74, *sgn = blk + 34;
            for (int ib = 0; ib < 8; ++ib)
                for (int l = 0; l < 4; ++l) {
                    const int idx = qs[4 * ib + l] | ((qh[ib] << (8 - 2 * l)) & 0x300);
                    const uint8_t* g = (const uint8_t*) &iq2s_grid[idx];
                    const double dd = d * (0.5 + ((scl[ib] >> (4 * (l / 2))) & 0xf)) * 0.25;
                    const uint8_t sb = sgn[4 * ib + l];
                    for (int j = 0; j < 8; ++j) o[ib * 32 + 8 * l + j] = dd * ((sb >> j & 1) ? -(int) g[j] : (int) g[j]);
                }
        } else if (t == 12) {   // Q4_K: d, dmin, scales[12], qs[128] (ggml dequantize_row_q4_K, get_scale_min_k4)
            uint16_t mh; std::memcpy(&mh, blk + 2, 2);
            const double dmin = h2f(mh);
            const uint8_t *sc = blk + 4, *qs = blk + 16;
            for (int j = 0; j < 8; ++j) {
                int s, m;
                if (j < 4) { s = sc[j] & 63; m = sc[j + 4] & 63; }
                else { s = (sc[j + 4] & 0xF) | ((sc[j - 4] >> 6) << 4); m = (sc[j + 4] >> 4) | ((sc[j] >> 6) << 4); }
                for (int l = 0; l < 32; ++l) o[32 * j + l] = d * s * ((qs[32 * (j / 2) + l] >> (4 * (j & 1))) & 0xF) - dmin * m;
            }
        } else if (t == 6) {    // Q5_0: d, qh[4], qs[16] (ggml dequantize_row_q5_0)
            uint32_t qh; std::memcpy(&qh, blk + 2, 4);
            for (int j = 0; j < 16; ++j) {
                o[j] = d * ((int) ((blk[6 + j] & 0xF) | (((qh >> j) & 1) << 4)) - 16);   // (int): qh makes the OR unsigned
                o[16 + j] = d * ((int) ((blk[6 + j] >> 4) | (((qh >> (j + 16)) & 1) << 4)) - 16);
            }
        } else if (t == 20) {
            for (int j = 0; j < 16; ++j) {
                o[j] = d * kvalues_iq4nl[blk[2 + j] & 15];
                o[16 + j] = d * kvalues_iq4nl[blk[2 + j] >> 4];
            }
        } else {   // 42 Q2_0
            for (int i = 0; i < 64; ++i) o[i] = d * (((blk[2 + i / 4] >> ((i % 4) * 2)) & 3) - 1);
        }
    }
}

// The decoder above against ggml's to_float on `rows` rows of a matrix (largest |difference| over the largest |value|); -1 without ggml.
static double check_decoder(int t, const uint8_t* m, size_t rowb, int K, int rows, int stride) {
#ifdef STRATA_NATIVE_EXPERTS
    if (t == 42) return -1;   // Q2_0 is Strata's type: not in an upstream ggml
    const auto* tr = ggml_get_type_traits((ggml_type) t);
    if (tr == nullptr || tr->to_float == nullptr) return -1;
    std::vector<float> f(K);
    std::vector<double> mine(K);
    double err = 0, mx = 0;
    for (int r = 0; r < rows; ++r) {
        const uint8_t* row = m + (size_t) r * stride * rowb;
        tr->to_float(row, f.data(), K);
        dec_row(t, row, K, mine.data());
        for (int k = 0; k < K; ++k) { err = std::max(err, std::fabs(mine[k] - f[k])); mx = std::max(mx, std::fabs((double) f[k])); }
    }
    return err / (mx + 1e-30);
#else
    (void) t; (void) m; (void) rowb; (void) K; (void) rows; (void) stride;
    return -1;
#endif
}

// Real experts: one layer's gate / up / down tensors, from whichever shard holds them.
struct RealLayer {
    const uint8_t* t[3] = {};
    int gt = -1, dt = -1;
    int64_t n_expert = 0;
    int layer = -1;
};
static bool find_layer(const std::vector<std::unique_ptr<strata::GgufFile>>& files, int l, RealLayer& out) {
    const char* roles[3] = {"gate", "up", "down"};
    const strata::TensorInfo* ti[3] = {};
    for (int r = 0; r < 3; ++r)
        for (const auto& f : files) {
            const strata::TensorInfo* x = f->find("blk." + std::to_string(l) + ".ffn_" + roles[r] + "_exps.weight");
            if (x) { ti[r] = x; out.t[r] = f->tensor_data(*x); break; }
        }
    if (!ti[0] || !ti[1] || !ti[2]) return false;
    if (ti[0]->type != ti[1]->type || ti[0]->shape.size() != 3 || ti[1]->shape.size() != 3 || ti[2]->shape.size() != 3) return false;
    // the shape run_case is written for: gate / up [2560 (K), 640, experts], down [640, 2560, experts] (GGUF order)
    if (ti[0]->shape[0] != 2560 || ti[0]->shape[1] != 640 || ti[1]->shape != ti[0]->shape || ti[2]->shape[0] != 640 ||
        ti[2]->shape[1] != 2560 || ti[2]->shape[2] != ti[0]->shape[2]) {
        std::printf("layer %d: expert shape %llux%llu / %llux%llu, not the 2560 x 640 expert\n", l, (unsigned long long) ti[0]->shape[0],
                    (unsigned long long) ti[0]->shape[1], (unsigned long long) ti[2]->shape[0], (unsigned long long) ti[2]->shape[1]);
        return false;
    }
    out.gt = (int) ti[0]->type;
    out.dt = (int) ti[2]->type;
    out.n_expert = (int64_t) ti[0]->shape[2];
    out.layer = l;
    return true;
}

struct Result { bool ok; double t_ref, t_fused; };

// `check` false (--sweep): timing only, on the row counts the engine's crossover measurement uses (i8_calibrate: uneven factors
// of the mean, mean 1), no host references.
static Result run_case(sycl::queue* q, dpct::blas::descriptor_ptr hd, int gt, int dt, int mean, int n, int reps, int seed,
                       const RealLayer* real, bool check = true) {
    const int K = 2560, NFF = 640, NE = 2560;
    std::mt19937 rng(seed);
    std::vector<int32_t> cnt(n);
    int tot = 0;
    if (check) {
        std::gamma_distribution<double> gd(1.5, mean / 1.5);
        for (auto& c : cnt) { c = std::max(1, (int) gd(rng)); tot += c; }
        if (n >= 3) { cnt[0] = 1; cnt[1] = 7; cnt[2] = 9; tot = 0; for (int c : cnt) tot += c; }   // pad edge cases
    } else {
        const double facs[8] = {0.15, 0.35, 0.6, 0.85, 1.1, 1.4, 1.8, 2.45};
        for (int i = 0; i < n; ++i) { cnt[i] = std::max(1, (int) (mean * facs[i % 8] / 1.0875 + 0.5)); tot += cnt[i]; }
    }
    i8::Layout lo;
    i8::make_layout(lo, cnt.data(), n);

    const size_t rowb_gu = (size_t) (K / block_vals(gt)) * block_bytes(gt), rowb_dn = (size_t) (NFF / block_vals(dt)) * block_bytes(dt);
    const size_t gate_b = NFF * rowb_gu, down_b = NE * rowb_dn, blob_b = 2 * gate_b + down_b;
    std::vector<std::vector<uint8_t>> hb(n);
    std::vector<uint8_t*> db(n);
    std::vector<int64_t> ids(n, -1);
    for (int i = 0; i < n; ++i) {
        hb[i].resize(blob_b);
        if (real) {
            // ids spread over the layer, shifted by the seed (in 64 bits, and into [0, n_expert) for a negative seed)
            const int64_t ne = real->n_expert;
            ids[i] = ((int64_t) i * ne / n + (int64_t) seed * 7) % ne;
            if (ids[i] < 0) ids[i] += ne;
            std::memcpy(hb[i].data(), real->t[0] + (size_t) ids[i] * gate_b, gate_b);
            std::memcpy(hb[i].data() + gate_b, real->t[1] + (size_t) ids[i] * gate_b, gate_b);
            std::memcpy(hb[i].data() + 2 * gate_b, real->t[2] + (size_t) ids[i] * down_b, down_b);
        } else {
            for (auto& x : hb[i]) x = (uint8_t) rng();
            for (size_t o = 0; o < 2 * gate_b; o += block_bytes(gt)) {
                const uint16_t d = f2h(0.0004f + 0.00002f * (rng() % 50)); std::memcpy(&hb[i][o], &d, 2);
                if (gt == 12) { const uint16_t m = f2h(0.0004f + 0.00002f * (rng() % 50)); std::memcpy(&hb[i][o + 2], &m, 2); }   // Q4_K dmin
            }
            for (size_t o = 2 * gate_b; o < blob_b; o += block_bytes(dt)) { const uint16_t d = f2h(0.0004f + 0.00002f * (rng() % 50)); std::memcpy(&hb[i][o], &d, 2); }
        }
        db[i] = sycl::malloc_device<uint8_t>(blob_b + 256, *q);
        q->memcpy(db[i], hb[i].data(), blob_b);
    }
    if (real && check) {
        std::printf("  layer %d experts", real->layer);
        for (int i = 0; i < n; ++i) std::printf(" %lld", (long long) ids[i]);
        std::printf("\n");
    }
    bool ok = true;
    if (check) {   // the host decoder against ggml's (gate, up and down rows of the first and last expert)
        double e = -1;
        for (int i : {0, n - 1}) {
            e = std::max(e, check_decoder(gt, hb[i].data(), rowb_gu, K, 8, NFF / 8));
            e = std::max(e, check_decoder(gt, hb[i].data() + gate_b, rowb_gu, K, 8, NFF / 8));
            e = std::max(e, check_decoder(dt, hb[i].data() + 2 * gate_b, rowb_dn, NFF, 8, NE / 8));
        }
        if (e >= 0) {
            const bool okd = e < 1e-6;
            ok = ok && okd;
            std::printf("  host decoder vs ggml to_float: max err %.1e of the largest value  %s\n", e, okd ? "OK" : "FAIL");
        }
    }
    std::vector<uint16_t> hx((size_t) tot * K);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (auto& x : hx) { float v = nd(rng) * 0.5f; if (rng() % 97 == 0) v *= 12.f; x = f2h(v); }   // a few outliers
    uint16_t* dxh = sycl::malloc_device<uint16_t>(hx.size(), *q);
    q->memcpy(dxh, hx.data(), hx.size() * 2);
    int8_t* Xq = sycl::malloc_device<int8_t>(i8::xq_bytes(lo, K), *q);
    float* dX = sycl::malloc_device<float>(i8::dx_bytes(lo, K) / 4, *q);
    float* sX = sycl::malloc_device<float>(i8::dx_bytes(lo, K) / 4, *q);
    int8_t* Hq = sycl::malloc_device<int8_t>(i8::xq_bytes(lo, NFF), *q);
    float* dH = sycl::malloc_device<float>(i8::dx_bytes(lo, NFF) / 4, *q);
    float *GU = sycl::malloc_device<float>((size_t) tot * 2 * NFF, *q), *Dm = sycl::malloc_device<float>((size_t) tot * NE, *q);
    float *GUr = sycl::malloc_device<float>((size_t) tot * 2 * NFF, *q), *Dmr = sycl::malloc_device<float>((size_t) tot * NE, *q);
    uint16_t* Hh = sycl::malloc_device<uint16_t>((size_t) tot * NFF, *q);
    uint16_t* dwgu = sycl::malloc_device<uint16_t>((size_t) 2 * NFF * K, *q);
    uint16_t* dwd = sycl::malloc_device<uint16_t>((size_t) NE * NFF, *q);
    q->wait();
    std::vector<const uint8_t*> pb(n);
    for (int i = 0; i < n; ++i) pb[i] = db[i];

    auto fused = [&]() {
        i8::quantize_rows(dxh, lo, K, Xq, dX, q, sX);
        if (!i8::gate_up(gt, pb.data(), gate_b, NFF, lo, K, Xq, dX, GU, q, sX)) { std::printf("gate_up refused (type %d)\n", gt); std::exit(2); }
        i8::swiglu_quantize(GU, lo, NFF, Hq, dH, q);
        if (!i8::down(dt, pb.data(), 2 * gate_b, NE, lo, NFF, Hq, dH, Dm, q)) { std::printf("down refused (type %d)\n", dt); std::exit(2); }
    };
    auto ref = [&]() {
        int64_t o = 0;
        for (int i = 0; i < n; ++i) {
            strata::kernels::iq_dequant_expert_f16(gt, dt, db[i], db[i] + gate_b, db[i] + 2 * gate_b, NFF, K, dwgu, dwd, q);
            const float a = 1.f, b = 0.f;
            dpct::blas::gemm(hd, oneapi::mkl::transpose::trans, oneapi::mkl::transpose::nontrans, 2 * NFF, cnt[i], K, &a, dwgu,
                             dpct::library_data_t::real_half, K, dxh + o * K, dpct::library_data_t::real_half, K, &b,
                             GUr + o * 2 * NFF, dpct::library_data_t::real_float, 2 * NFF, dpct::compute_type::f32);
            strata::prefill::swiglu_interleaved(GUr + o * 2 * NFF, Hh + o * NFF, cnt[i], q);
            dpct::blas::gemm(hd, oneapi::mkl::transpose::trans, oneapi::mkl::transpose::nontrans, NE, cnt[i], NFF, &a, dwd,
                             dpct::library_data_t::real_half, NFF, Hh + o * NFF, dpct::library_data_t::real_half, NFF, &b,
                             Dmr + o * NE, dpct::library_data_t::real_float, NE, dpct::compute_type::f32);
            o += cnt[i];
        }
    };
    auto tm = [&](auto f) {
        f(); q->wait();
        const auto t0 = std::chrono::steady_clock::now();
        while (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < 0.5) { f(); q->wait(); }   // GPU clocks up
        const auto a = std::chrono::steady_clock::now();
        for (int r = 0; r < reps; ++r) f();
        q->wait();
        return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - a).count() / reps;
    };

    fused(); q->wait();
    std::vector<int8_t> hXq(i8::xq_bytes(lo, K));
    std::vector<float> hdX(i8::dx_bytes(lo, K) / 4), hsX(i8::dx_bytes(lo, K) / 4), hGU((size_t) tot * 2 * NFF);
    q->memcpy(hXq.data(), Xq, hXq.size());
    q->memcpy(hdX.data(), dX, hdX.size() * 4);
    q->memcpy(hsX.data(), sX, hsX.size() * 4);
    q->memcpy(hGU.data(), GU, hGU.size() * 4).wait();
    const int nkb = K / 32;
    auto xq_at = [&](int tile, int r, int k) { return (int) hXq[(((size_t) tile * nkb + k / 32) * 8 + r) * 32 + (k % 32)]; };
    auto dx_at = [&](int tile, int r, int kb) { return (double) hdX[((size_t) tile * nkb + kb) * 8 + r]; };
    if (check) {   // the scaled sums: scale x the sum of the 32 int8 values, for every real row (float rounding of the product only)
        double se = 0;
        for (int e = 0; e < n; ++e)
            for (int rl = 0; rl < cnt[e]; ++rl) {
                const int tile = lo.tile0[e] + rl / 8, r = rl % 8;
                for (int kb = 0; kb < nkb; ++kb) {
                    long s = 0;
                    for (int k = 0; k < 32; ++k) s += xq_at(tile, r, kb * 32 + k);
                    const double want = dx_at(tile, r, kb) * (double) s;
                    se = std::max(se, std::fabs(hsX[((size_t) tile * nkb + kb) * 8 + r] - want) / (std::fabs(want) + dx_at(tile, r, kb)));
                }
            }
        const bool oks = se < 1e-6;
        ok = ok && oks;
        std::printf("  activation sums: max rel err %.1e  %s\n", se, oks ? "OK" : "FAIL");
    }
    if (check) {   // gate/up against (c) and (b), on sampled rows and outputs
        std::vector<double> wrow(K);
        double err_c = 0, mx_c = 0, err_b = 0, rms_bref = 0;
        int sampled = 0;
        for (int e = 0; e < n; ++e)
            for (int rl = 0; rl < cnt[e]; rl += (cnt[e] > 12 ? 5 : 1)) {
                const int tile = lo.tile0[e] + rl / 8, r = rl % 8, row = lo.row0[e] + rl;
                for (int v = 0; v < 2 * NFF; v += 37) {
                    dec_row(gt, &hb[e][(size_t) (v & 1 ? gate_b : 0) + (size_t) (v >> 1) * rowb_gu], K, wrow.data());
                    double refc = 0, refb = 0;
                    for (int kb = 0; kb < nkb; ++kb) {
                        double s = 0;
                        for (int k = 0; k < 32; ++k) s += (double) xq_at(tile, r, kb * 32 + k) * wrow[kb * 32 + k];
                        refc += s * dx_at(tile, r, kb);
                    }
                    for (int k = 0; k < K; ++k) refb += (double) h2f(hx[(size_t) row * K + k]) * wrow[k];
                    const double got = hGU[(size_t) row * 2 * NFF + v];
                    err_c = std::max(err_c, std::fabs(got - refc)); mx_c = std::max(mx_c, std::fabs(refc));
                    err_b += (got - refb) * (got - refb); rms_bref += refb * refb;
                    ++sampled;
                }
            }
        const bool ok1 = err_c <= 2e-5 * mx_c && std::sqrt(err_b / rms_bref) < 0.02;
        ok = ok && ok1;
        std::printf("  gate/up type %2d: exact-int8 ref max err %.2e of %.2e (%.1e)  vs FP64 true rms rel err %.2e (%d samples)  %s\n", gt,
                    err_c, mx_c, err_c / mx_c, std::sqrt(err_b / rms_bref), sampled, ok1 ? "OK" : "FAIL");
    }
    std::vector<float> hDm((size_t) tot * NE);
    q->memcpy(hDm.data(), Dm, hDm.size() * 4).wait();
    if (check) {
        std::vector<int8_t> hHq(i8::xq_bytes(lo, NFF));
        std::vector<float> hdH(i8::dx_bytes(lo, NFF) / 4);
        q->memcpy(hHq.data(), Hq, hHq.size());
        q->memcpy(hdH.data(), dH, hdH.size() * 4).wait();
        const int nkh = NFF / 32;
        double ec = 0, mc = 0, eb = 0, rb = 0, hq_bad = 0;
        std::vector<double> drow(NFF);
        int smp = 0;
        for (int e = 0; e < n; ++e)
            for (int rl = 0; rl < cnt[e]; rl += (cnt[e] > 12 ? 5 : 1)) {
                const int tile = lo.tile0[e] + rl / 8, r = rl % 8, row = lo.row0[e] + rl;
                std::vector<double> H(NFF);
                for (int j = 0; j < NFF; ++j) {
                    const double g = hGU[(size_t) row * 2 * NFF + 2 * j], u = hGU[(size_t) row * 2 * NFF + 2 * j + 1];
                    H[j] = g / (1.0 + std::exp(-g)) * u;
                }
                for (int kb = 0; kb < nkh; ++kb) {
                    double amax = 0; for (int k = 0; k < 32; ++k) amax = std::max(amax, std::fabs(H[kb * 32 + k]));
                    const double dd = hdH[((size_t) tile * nkh + kb) * 8 + r];
                    if (std::fabs(dd - amax / 127.0) > 1e-5 * amax + 1e-30) hq_bad += 1;
                }
                for (int v = 0; v < NE; v += 41) {
                    dec_row(dt, &hb[e][2 * gate_b + (size_t) v * rowb_dn], NFF, drow.data());
                    double refc = 0, refb = 0;
                    for (int kb = 0; kb < nkh; ++kb) {
                        double s = 0;
                        for (int k = 0; k < 32; ++k) s += (double) hHq[(((size_t) tile * nkh + kb) * 8 + r) * 32 + k] * drow[kb * 32 + k];
                        refc += s * hdH[((size_t) tile * nkh + kb) * 8 + r];
                    }
                    for (int k = 0; k < NFF; ++k) refb += H[k] * drow[k];
                    const double got = hDm[(size_t) row * NE + v];
                    ec = std::max(ec, std::fabs(got - refc)); mc = std::max(mc, std::fabs(refc));
                    eb += (got - refb) * (got - refb); rb += refb * refb;
                    ++smp;
                }
            }
        const bool ok2 = ec <= 2e-5 * mc && std::sqrt(eb / rb) < 0.03 && hq_bad == 0;
        ok = ok && ok2;
        std::printf("  down type %2d   : exact-int8 ref max err %.2e of %.2e (%.1e)  vs FP64 true rms rel err %.2e, swiglu scale mismatches %.0f (%d samples)  %s\n",
                    dt, ec, mc, ec / mc, std::sqrt(eb / rb), hq_bad, smp, ok2 ? "OK" : "FAIL");
    }
    ref(); q->wait();
    std::vector<float> hDmr((size_t) tot * NE);
    q->memcpy(hDmr.data(), Dmr, hDmr.size() * 4).wait();
    {
        double e2 = 0, r2 = 0;
        for (size_t i = 0; i < hDmr.size(); ++i) { e2 += (double) (hDmr[i] - hDm[i]) * (hDmr[i] - hDm[i]); r2 += (double) hDmr[i] * hDmr[i]; }
        std::printf("  chain vs dequantize + oneMKL (FP16 path): rms rel diff %.2e (rows %d)\n", std::sqrt(e2 / r2), tot);
    }
    if (check) {   // (d) the whole expert in FP64 from the FP16 activations (sampled rows, every output), both chains against it
        std::vector<double> G((size_t) 2 * NFF * K), D((size_t) NE * NFF), h(NFF), x(K);
        double s8 = 0, s16 = 0, sref = 0;
        int rows = 0;
        for (int e = 0; e < n; ++e) {
            for (int v = 0; v < 2 * NFF; ++v) dec_row(gt, &hb[e][(size_t) (v & 1 ? gate_b : 0) + (size_t) (v >> 1) * rowb_gu], K, &G[(size_t) v * K]);
            for (int v = 0; v < NE; ++v) dec_row(dt, &hb[e][2 * gate_b + (size_t) v * rowb_dn], NFF, &D[(size_t) v * NFF]);
            for (int rl = 0; rl < cnt[e]; rl += (cnt[e] > 12 ? 7 : 1)) {
                const int row = lo.row0[e] + rl;
                for (int k = 0; k < K; ++k) x[k] = h2f(hx[(size_t) row * K + k]);
                for (int j = 0; j < NFF; ++j) {
                    double g = 0, u = 0;
                    const double *wg = &G[(size_t) (2 * j) * K], *wu = &G[(size_t) (2 * j + 1) * K];
                    for (int k = 0; k < K; ++k) { g += wg[k] * x[k]; u += wu[k] * x[k]; }
                    h[j] = g / (1.0 + std::exp(-g)) * u;
                }
                for (int v = 0; v < NE; ++v) {
                    double o = 0;
                    for (int k = 0; k < NFF; ++k) o += D[(size_t) v * NFF + k] * h[k];
                    s8 += std::fabs(hDm[(size_t) row * NE + v] - o);
                    s16 += std::fabs(hDmr[(size_t) row * NE + v] - o);
                    sref += std::fabs(o);
                }
                ++rows;
            }
        }
        const bool ok3 = s8 / sref < 5e-2;
        ok = ok && ok3;
        std::printf("  whole expert vs FP64 (%d rows): int8 chain rel %.2e, FP16 chain rel %.2e  %s\n", rows, s8 / sref, s16 / sref,
                    ok3 ? "OK" : "FAIL");
    }
    Result res{ok, tm(ref) / n, tm(fused) / n};
    const double t_q = tm([&] { i8::quantize_rows(dxh, lo, K, Xq, dX, q, sX); }) / n;
    const double t_gu = tm([&] { i8::gate_up(gt, pb.data(), gate_b, NFF, lo, K, Xq, dX, GU, q, sX); }) / n;
    const double t_sw = tm([&] { i8::swiglu_quantize(GU, lo, NFF, Hq, dH, q); }) / n;
    const double t_dn = tm([&] { i8::down(dt, pb.data(), 2 * gate_b, NE, lo, NFF, Hq, dH, Dm, q); }) / n;
    std::printf("  rows %d (%d experts, mean %.0f): dequant+oneMKL %.1f us/expert | fused int8 %.1f us/expert (%.2fx)  [quantize %.1f, gate/up %.1f, swiglu+q %.1f, down %.1f]\n",
                tot, n, (double) tot / n, res.t_ref, res.t_fused, res.t_ref / res.t_fused, t_q, t_gu, t_sw, t_dn);
    for (int i = 0; i < n; ++i) sycl::free(db[i], *q);
    sycl::free(dxh, *q); sycl::free(Xq, *q); sycl::free(dX, *q); sycl::free(sX, *q); sycl::free(Hq, *q); sycl::free(dH, *q); sycl::free(GU, *q);
    sycl::free(Dm, *q); sycl::free(GUr, *q); sycl::free(Dmr, *q); sycl::free(Hh, *q); sycl::free(dwgu, *q); sycl::free(dwd, *q);
    return res;
}

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    int mean = 80, n = 16, reps = 10, seed = 1;
    std::vector<std::pair<int, int>> cases = {{21, 20}, {18, 20}, {22, 20}, {21, 42}, {18, 42}, {22, 42}, {12, 6}, {12, 20}, {21, 6}};
    std::vector<std::unique_ptr<strata::GgufFile>> files;
    std::vector<int> layers, sweep;
    if (argc > 1 && std::strcmp(argv[1], "--gguf") == 0) {
        if (argc < 3) { std::fprintf(stderr, "usage: esimd_moe_parity --gguf <shard>[,<shard>...] [--mean R] [--experts N] [--reps N] [--seed S] [--sweep R,R,...] layer ...\n"); return 2; }
        mean = 40;
        std::string list = argv[2];
        for (size_t p = 0; p <= list.size();) {
            size_t c = list.find(',', p);
            if (c == std::string::npos) c = list.size();
            if (c > p) files.emplace_back(new strata::GgufFile(list.substr(p, c - p)));
            p = c + 1;
        }
        for (int i = 3; i < argc; ++i) {
            if (std::strcmp(argv[i], "--mean") == 0 && i + 1 < argc) mean = std::atoi(argv[++i]);
            else if (std::strcmp(argv[i], "--experts") == 0 && i + 1 < argc) n = std::atoi(argv[++i]);
            else if (std::strcmp(argv[i], "--reps") == 0 && i + 1 < argc) reps = std::atoi(argv[++i]);
            else if (std::strcmp(argv[i], "--seed") == 0 && i + 1 < argc) seed = std::atoi(argv[++i]);
            else if (std::strcmp(argv[i], "--sweep") == 0 && i + 1 < argc) {
                for (const char* p = argv[++i]; *p;) {
                    char* e = nullptr;
                    const long r = std::strtol(p, &e, 10);
                    if (e == p) break;
                    if (r > 0) sweep.push_back((int) r);
                    p = *e == ',' ? e + 1 : e;
                }
            }
            else layers.push_back(std::atoi(argv[i]));
        }
        if (layers.empty()) layers = {30, 40, 47};
    } else {
        mean = argc > 1 ? std::atoi(argv[1]) : 80;
        n = argc > 2 ? std::atoi(argv[2]) : 16;
        reps = argc > 3 ? std::atoi(argv[3]) : 10;
        seed = argc > 4 ? std::atoi(argv[4]) : 1;
        if (argc > 6) cases = {{std::atoi(argv[5]), std::atoi(argv[6])}};
    }
    if (n < 1 || n > i8::kMaxGroup) { std::fprintf(stderr, "experts must be 1..%d\n", i8::kMaxGroup); return 2; }
    sycl::queue* q = &dpct::get_in_order_queue();
    if (!i8::available(q)) { std::printf("esimd_moe_parity: SKIP (no ESIMD device with the s8 8x16x32 DPAS)\n"); return 0; }
    dpct::blas::descriptor_ptr hd = new dpct::blas::descriptor();
    hd->set_queue(q);
    int fails = 0;
    if (!files.empty()) {
        for (int l : layers) {
            RealLayer rl;
            if (!find_layer(files, l, rl)) { std::printf("layer %d: no expert tensors in the given shards\n", l); ++fails; continue; }
            std::printf("layer %d (%lld experts): gate/up %d, down %d, mean %d rows, %d experts\n", l, (long long) rl.n_expert, rl.gt, rl.dt,
                        mean, n);
            if (!i8::gu_type_ok(rl.gt) || !i8::d_type_ok(rl.dt)) { std::printf("  formats not handled by the int8 kernels\n"); ++fails; continue; }
            if (sweep.empty()) fails += !run_case(q, hd, rl.gt, rl.dt, mean, n, reps, seed, &rl).ok;
            else
                for (int r : sweep) run_case(q, hd, rl.gt, rl.dt, r, n, reps, seed, &rl, false);
        }
    } else {
        for (auto [gt, dt] : cases) {
            std::printf("gate/up %d, down %d, mean %d rows, %d experts\n", gt, dt, mean, n);
            fails += !run_case(q, hd, gt, dt, mean, n, reps, seed, nullptr).ok;
        }
    }
    std::printf("%s\n", fails ? "FAILED" : "all OK");
    return fails ? 1 : 0;
}
