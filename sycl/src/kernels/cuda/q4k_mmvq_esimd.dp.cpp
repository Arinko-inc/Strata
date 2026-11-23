// STRATA_MMVQ_ESIMD=1's dense Q4_K GEMV for the decode window's 1-4 columns (Battlemage): native_mmvq_wide_kernel<WideQ4K>
// (native_mmvq.dp.cpp) with one hardware thread per RPT weight rows instead of a 32-wide sub-group per row.
//
// The wide kernel's work-item (g, sub) of a row sums, over the row's 256-blocks kb = sub, sub + 4, ..., the dot of qs bytes
// [16g, 16g + 16) with the column's activations, each read with per-lane gathers (the weights' 16 bytes, the scales, two
// 16-byte runs and two scales of the column's Q8_1 blocks); the 32 sums then meet in the sub-group's xor butterfly.  Here a
// thread takes a whole block per step (a 16-byte and a 128-byte load) and a column's 8 Q8_1 blocks (three 128-byte loads,
// shared by the thread's rows): the 8 dots of a block are the terms of lanes (0..7, kb % 4), so four 8-wide float registers
// per (row, column) are the 32 lane sums.  The arithmetic is WideQ4K::apply's as the device compiler builds it there:
//   A, C (int32, exact in any order) = the dots of the low / high nibbles;  B, D = the sums of the activations' bytes
//   I1   = fma(dl, A, -(ml * B)),  I2 = fma(dh, C, -(mh * D))        (`dl * A - ml * B` contracted, ml * B on its own)
//   term = fma(da, I1, dc * I2)                                       (dc * I2 rounded on its own);  lane += term
//   out  = the butterfly's tree: sub ^ 2, sub ^ 1, then g ^ 4, g ^ 2, g ^ 1
// The ESIMD backend fuses a product into the add that follows it, so the products that the lane kernel rounds on their own
// go through an xor with a run-time zero.  mmvq_esimd_bench checks the bits against the wide kernel on a model's own
// matrices, and has to be rerun after a compiler update.
// The loads take dword-aligned addresses: the weights and the Q8_1 buffer must sit at multiples of 4 (the launch checks).
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include <cstdint>
#include "strata/sycl_queue.hpp"

namespace strata::kernels {
bool q4k_mmvq_esimd_built();
bool q4k_mmvq_esimd(const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, int ncols, void* stream,
                    int rpt, int var);
}

#ifdef STRATA_I8_NO_ESIMD
namespace strata::kernels {
bool q4k_mmvq_esimd_built() { return false; }
bool q4k_mmvq_esimd(const void*, const void*, float*, int, int, int, void*, int, int) { return false; }
}  // namespace strata::kernels
#else
namespace strata::kernels {
namespace {
using namespace sycl::ext::intel::esimd;
using sycl::half;

template <int NC, int RPT, int VAR, int GRF> struct Q4kMmvqTag;

SYCL_ESIMD_FUNCTION inline simd<float, 8> fma8(simd<float, 8> a, simd<float, 8> b, simd<float, 8> c) {
    return simd<float, 8>(__builtin_elementwise_fma(a.data(), b.data(), c.data()));
}
SYCL_ESIMD_FUNCTION inline simd<float, 8> twice(simd<float, 4> v) {   // 4 values, each twice: a sub-block pair's two lanes
    simd<float, 8> r;
    r.select<4, 2>(0) = v;
    r.select<4, 2>(1) = v;
    return r;
}
SYCL_ESIMD_FUNCTION inline simd<float, 8> xorf(simd<float, 8> v, simd<uint32_t, 8> m) {
    simd<uint32_t, 8> b = v.template bit_cast_view<uint32_t>();
    b = b ^ m;
    simd<float, 8> r = b.template bit_cast_view<float>();
    return r;
}
// the 8 lanes' dots: 16 bytes (4 dwords) each
SYCL_ESIMD_FUNCTION inline simd<float, 8> dot8(simd<int, 16> w0, simd<int, 16> w1, simd<int, 16> u0, simd<int, 16> u1) {
    simd<int, 16> p0 = dp4a<int>(simd<int, 16>(0), w0, u0);
    simd<int, 16> p1 = dp4a<int>(simd<int, 16>(0), w1, u1);
    simd<int, 8> r;
    r.select<4, 1>(0) = p0.template select<4, 4>(0) + p0.template select<4, 4>(1) + p0.template select<4, 4>(2) + p0.template select<4, 4>(3);
    r.select<4, 1>(4) = p1.template select<4, 4>(0) + p1.template select<4, 4>(1) + p1.template select<4, 4>(2) + p1.template select<4, 4>(3);
    simd<float, 8> f = r;
    return f;
}

struct Act {              // a column's 8 Q8_1 blocks of one 256-block
    simd<int, 16> a0, a1;   // blocks 0, 2 / 4, 6: the low nibbles' activations
    simd<int, 16> c0, c1;   // blocks 1, 3 / 5, 7: the high nibbles'
    simd<float, 8> da, dc;  // their scales, per lane
    simd<float, 8> sa, sc;  // the sums of each lane's 16 bytes
};
struct Wt {               // a row's 256-block
    simd<int, 16> l0, l1, h0, h1;
    simd<float, 8> dl, dh, ml, mh;
};

// VAR: how the float expression is built (see the top).  inner = VAR % 3, outer = VAR / 3 % 3; 9: the expression of 0
// without the xors.
template <int VAR>
SYCL_ESIMD_FUNCTION inline simd<float, 8> term(const Wt& w, const Act& x, simd<uint32_t, 8> zz) {
    const simd<uint32_t, 8> sgn = zz ^ 0x80000000u;
    const simd<float, 8> A = dot8(w.l0, w.l1, x.a0, x.a1);
    const simd<float, 8> C = dot8(w.h0, w.h1, x.c0, x.c1);
    if constexpr (VAR == 9) {
        simd<float, 8> t1 = w.ml * x.sa, t2 = w.mh * x.sc;
        const simd<float, 8> i1 = fma8(w.dl, A, xorf(t1, simd<uint32_t, 8>(0x80000000u)));
        const simd<float, 8> i2 = fma8(w.dh, C, xorf(t2, simd<uint32_t, 8>(0x80000000u)));
        simd<float, 8> p2 = x.dc * i2;
        return fma8(x.da, i1, p2);
    } else {
        constexpr int inner = VAR % 3, outer = VAR / 3;
        simd<float, 8> i1, i2;
        if constexpr (inner == 0) {
            i1 = fma8(w.dl, A, xorf(w.ml * x.sa, sgn));
            i2 = fma8(w.dh, C, xorf(w.mh * x.sc, sgn));
        } else if constexpr (inner == 1) {
            i1 = fma8(xorf(w.ml, sgn), x.sa, xorf(w.dl * A, zz));
            i2 = fma8(xorf(w.mh, sgn), x.sc, xorf(w.dh * C, zz));
        } else {
            i1 = xorf(xorf(w.dl * A, zz) - xorf(w.ml * x.sa, zz), zz);
            i2 = xorf(xorf(w.dh * C, zz) - xorf(w.mh * x.sc, zz), zz);
        }
        if constexpr (outer == 0) return fma8(x.da, i1, xorf(x.dc * i2, zz));
        else if constexpr (outer == 1) return fma8(x.dc, i2, xorf(x.da * i1, zz));
        else return xorf(xorf(x.da * i1, zz) + xorf(x.dc * i2, zz), zz);
    }
}

template <int NC, int RPT, int VAR, int GRF>
void launch(sycl::queue& q, const uint8_t* w, const uint32_t* x, float* y, int n_in, int n_out) {
    const size_t threads = ((size_t) n_out + RPT - 1) / RPT;
    auto kernel = [=](sycl::id<1> id) SYCL_ESIMD_KERNEL {
        const int bpr = n_in / 256;            // 256-blocks per row
        const size_t xcol = (size_t) (n_in / 32) * 9;   // dwords of a column's Q8_1 blocks
        const int row0 = (int) id[0] * RPT;
        const simd<uint32_t, 8> zz((uint32_t) (n_out >> 31));   // 0, unknown to the compiler
        simd<float, 8> acc[RPT][4][NC];
#pragma unroll
        for (int r = 0; r < RPT; ++r)
#pragma unroll
            for (int s = 0; s < 4; ++s)
#pragma unroll
                for (int j = 0; j < NC; ++j) acc[r][s][j] = simd<float, 8>(0.0f);
        const uint8_t* wr[RPT];
#pragma unroll
        for (int r = 0; r < RPT; ++r) wr[r] = w + (size_t) (row0 + r < n_out ? row0 + r : row0) * bpr * 144;
        for (int kb4 = 0; kb4 < bpr; kb4 += 4) {
#pragma unroll
            for (int s = 0; s < 4; ++s) {
                const int kb = kb4 + s;
                if (kb >= bpr) break;
                Act X[NC];
#pragma unroll
                for (int j = 0; j < NC; ++j) {
                    // Q8_1 blocks of 9 dwords (ds, 8 of qs), 72 dwords: L0 = dwords 0..31, L1 = 28..59, L2 = 40..71
                    const uint32_t* xp = x + (size_t) j * xcol + (size_t) kb * 72;
                    simd<uint32_t, 32> L0 = block_load<uint32_t, 32>(xp);
                    simd<uint32_t, 32> L1 = block_load<uint32_t, 32>(xp + 28);
                    simd<uint32_t, 32> L2 = block_load<uint32_t, 32>(xp + 40);
                    simd<uint32_t, 16> a0, a1, c0, c1;
                    a0.select<8, 1>(0) = L0.template select<8, 1>(1);     // block 0
                    c0.select<8, 1>(0) = L0.template select<8, 1>(10);    // 1
                    a0.select<8, 1>(8) = L0.template select<8, 1>(19);    // 2
                    c0.select<8, 1>(8) = L1.template select<8, 1>(0);     // 3
                    a1.select<8, 1>(0) = L1.template select<8, 1>(9);     // 4
                    c1.select<8, 1>(0) = L1.template select<8, 1>(18);    // 5
                    a1.select<8, 1>(8) = L2.template select<8, 1>(15);    // 6
                    c1.select<8, 1>(8) = L2.template select<8, 1>(24);    // 7
                    simd<uint32_t, 8> ds;   // d of blocks 0, 2, 4, 6, then 1, 3, 5, 7
                    ds.select<2, 1>(0) = L0.template select<2, 18>(0);
                    ds.select<1, 1>(2) = L1.template select<1, 1>(8);
                    ds.select<1, 1>(3) = L2.template select<1, 1>(14);
                    ds.select<2, 1>(4) = L0.template select<2, 18>(9);
                    ds.select<1, 1>(6) = L1.template select<1, 1>(17);
                    ds.select<1, 1>(7) = L2.template select<1, 1>(23);
                    auto dsh = ds.template bit_cast_view<half>();
                    simd<half, 8> dh8 = dsh.template select<8, 2>(0);
                    simd<float, 8> df = dh8;
                    simd<float, 4> da4 = df.template select<4, 1>(0), dc4 = df.template select<4, 1>(4);
                    Act& A = X[j];
                    A.a0 = a0.template bit_cast_view<int>();
                    A.a1 = a1.template bit_cast_view<int>();
                    A.c0 = c0.template bit_cast_view<int>();
                    A.c1 = c1.template bit_cast_view<int>();
                    A.da = twice(da4);
                    A.dc = twice(dc4);
                    const simd<int, 16> ones(0x01010101);
                    A.sa = dot8(ones, ones, A.a0, A.a1);
                    A.sc = dot8(ones, ones, A.c0, A.c1);
                }
#pragma unroll
                for (int r = 0; r < RPT; ++r) {
                    const uint32_t* wp = (const uint32_t*) (wr[r] + (size_t) kb * 144);
                    simd<uint32_t, 4> H = block_load<uint32_t, 4>(wp);          // d, dmin (fp16), scales[12]
                    simd<uint32_t, 32> Q = block_load<uint32_t, 32>(wp + 4);    // qs[128]
                    simd<uint32_t, 16> q0 = Q.template select<16, 1>(0), q1 = Q.template select<16, 1>(16);
                    simd<uint32_t, 16> l0 = q0 & 0x0F0F0F0Fu, l1 = q1 & 0x0F0F0F0Fu;
                    simd<uint32_t, 16> h0 = (q0 >> 4) & 0x0F0F0F0Fu, h1 = (q1 >> 4) & 0x0F0F0F0Fu;
                    Wt W;
                    W.l0 = l0.template bit_cast_view<int>();
                    W.l1 = l1.template bit_cast_view<int>();
                    W.h0 = h0.template bit_cast_view<int>();
                    W.h1 = h1.template bit_cast_view<int>();
                    // ggml's get_scale_min_k4 for the 8 sub-blocks: scales j < 4 in bytes j (6 bits), mins in bytes j + 4;
                    // j >= 4: the low / high nibble of byte j + 4 with the top 2 bits of byte j - 4 / byte j
                    auto H8 = H.template bit_cast_view<uint8_t>();
                    simd<uint8_t, 4> b0 = H8.template select<4, 1>(4), b1 = H8.template select<4, 1>(8), b2 = H8.template select<4, 1>(12);
                    simd<uint32_t, 4> s0 = b0, s1 = b1, s2 = b2;
                    simd<uint32_t, 4> sl = s0 & 63u, mlo = s1 & 63u;
                    simd<uint32_t, 4> sh = (s2 & 15u) | ((s0 >> 6) << 4), mhi = (s2 >> 4) | ((s1 >> 6) << 4);
                    simd<uint32_t, 4> se, so, me, mo;   // sub-blocks 0, 2, 4, 6 and 1, 3, 5, 7
                    se.select<2, 1>(0) = sl.template select<2, 2>(0);
                    se.select<2, 1>(2) = sh.template select<2, 2>(0);
                    so.select<2, 1>(0) = sl.template select<2, 2>(1);
                    so.select<2, 1>(2) = sh.template select<2, 2>(1);
                    me.select<2, 1>(0) = mlo.template select<2, 2>(0);
                    me.select<2, 1>(2) = mhi.template select<2, 2>(0);
                    mo.select<2, 1>(0) = mlo.template select<2, 2>(1);
                    mo.select<2, 1>(2) = mhi.template select<2, 2>(1);
                    auto Hh = H.template bit_cast_view<half>();
                    simd<half, 2> dmh = Hh.template select<2, 1>(0);
                    simd<float, 2> dmf = dmh;
                    const float dx = dmf[0], dy = dmf[1];
                    simd<float, 4> fse = se, fso = so, fme = me, fmo = mo;
                    simd<float, 4> dl4 = fse * dx, dh4 = fso * dx, ml4 = fme * dy, mh4 = fmo * dy;
                    W.dl = twice(dl4);
                    W.dh = twice(dh4);
                    W.ml = twice(ml4);
                    W.mh = twice(mh4);
#pragma unroll
                    for (int j = 0; j < NC; ++j) acc[r][s][j] = acc[r][s][j] + term<VAR>(W, X[j], zz);
                }
            }
        }
#pragma unroll
        for (int r = 0; r < RPT; ++r) {
            if (row0 + r >= n_out) break;
#pragma unroll
            for (int j = 0; j < NC; ++j) {
                simd<float, 8> u0 = acc[r][0][j] + acc[r][2][j];
                simd<float, 8> u1 = acc[r][1][j] + acc[r][3][j];
                simd<float, 8> u = u0 + u1;
                simd<float, 4> t4 = u.template select<4, 1>(0) + u.template select<4, 1>(4);
                simd<float, 2> t2 = t4.template select<2, 1>(0) + t4.template select<2, 1>(2);
                const float v0 = t2[0], v1 = t2[1];
                y[(size_t) j * n_out + row0 + r] = v0 + v1;
            }
        }
    };
    if constexpr (GRF == 256) {
        auto props = sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>};
        q.parallel_for<Q4kMmvqTag<NC, RPT, VAR, GRF>>(sycl::range<1>(threads), props, kernel);
    } else {
        q.parallel_for<Q4kMmvqTag<NC, RPT, VAR, GRF>>(sycl::range<1>(threads), kernel);
    }
}

#define STRATA_Q4KE_EXP 1   // the experiment's variants (the float expression's shapes, the register file)
constexpr int kVar = 0;

template <int RPT, int VAR, int GRF>
bool launch_nc(sycl::queue& q, const uint8_t* w, const uint32_t* x, float* y, int n_in, int n_out, int ncols) {
    switch (ncols) {
        case 1: launch<1, RPT, VAR, GRF>(q, w, x, y, n_in, n_out); return true;
        case 2: launch<2, RPT, VAR, GRF>(q, w, x, y, n_in, n_out); return true;
        case 3: launch<3, RPT, VAR, GRF>(q, w, x, y, n_in, n_out); return true;
        case 4: launch<4, RPT, VAR, GRF>(q, w, x, y, n_in, n_out); return true;
        default: return false;
    }
}
}  // namespace

bool q4k_mmvq_esimd_built() { return true; }

// rpt: rows per hardware thread (1, 2, 4, 8); + 100: the 128-register file.  var < 0: the engine's expression.
bool q4k_mmvq_esimd(const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, int ncols, void* stream,
                    int rpt, int var) {
    if (ncols < 1 || ncols > 4 || n_in <= 0 || n_in % 256 != 0 || n_out <= 0) return false;
    if (((uintptr_t) weights | (uintptr_t) x_q8_1) & 3) return false;
    sycl::queue& q = *strata::q_of(stream);
    const auto* w = (const uint8_t*) weights;
    const auto* x = (const uint32_t*) x_q8_1;
    if (var < 0) var = kVar;
#define STRATA_Q4KE(RP, VA, GR) if (rpt == RP + (GR == 128 ? 100 : 0) && var == VA) return launch_nc<RP, VA, GR>(q, w, x, y, n_in, n_out, ncols);
    STRATA_Q4KE(1, kVar, 256) STRATA_Q4KE(2, kVar, 256) STRATA_Q4KE(4, kVar, 256) STRATA_Q4KE(8, kVar, 256)
#ifdef STRATA_Q4KE_EXP
    STRATA_Q4KE(2, 1, 256) STRATA_Q4KE(2, 2, 256) STRATA_Q4KE(2, 3, 256) STRATA_Q4KE(2, 4, 256) STRATA_Q4KE(2, 5, 256)
    STRATA_Q4KE(2, 6, 256) STRATA_Q4KE(2, 7, 256) STRATA_Q4KE(2, 8, 256) STRATA_Q4KE(2, 9, 256)
    STRATA_Q4KE(1, kVar, 128) STRATA_Q4KE(2, kVar, 128) STRATA_Q4KE(4, kVar, 128)
#endif
#undef STRATA_Q4KE
    return false;
}
}  // namespace strata::kernels
#endif
