// STRATA_SHEXP_ESIMD=1's dense Q5_0 GEMV for the shared expert's down projection (Battlemage): native_mmvq(6, ...) for
// 640 inputs and the decode window's 1-4 columns, with one hardware thread per RPT weight rows instead of a 128-item
// work-group (88 of them idle, a barrier and a pass through shared local memory) per 4 rows.
//
// The lane kernel (native_small_mmvq_kernel<Q50Block, 4, true>, and native_mmvq_multi_kernel's 4-rows layout, which is
// bitwise the same per column): 640 inputs are 20 blocks of 32, under the 64 blocks of one iteration, so work-item tid
// (0..39) computes ONE dot, half `tid % 2` of block `tid / 2`, and the rest add nothing.  Lane l of the 32-wide sub-group
// then holds (warp 0's term l) + (warp 1's term l: blocks 16..19 in lanes 0..7, zero elsewhere) + 0 + 0, and the 32 lanes
// meet in the xor butterfly (offsets 16, 8, 4, 2, 1).
// Here step j (0..4) takes the row's blocks 4j..4j+3 (88 bytes, two 64-byte loads) and the column's q8_1 blocks 4j..4j+3
// (144 bytes, two 128-byte loads): its 8 dots t_j are lanes 8j..8j+7 (j < 4) and warp 1's lanes 0..7 (j = 4), so
//   out = fold8(((t0 + t4) + t2) + (t1 + t3)),   fold8(u) = ((u0 + u4) + (u2 + u6)) + ((u1 + u5) + (u3 + u7))
// The arithmetic of a dot is vec_dot_q5_0_q8_1's as the device compiler builds it in the lane kernel (the same as
// q4kq5_down_esimd.dp.cpp's, whose row decode this is):
//   sumi (int32, exact in any order) = the four dp4a of a dot
//   v    = fma(sumi, d8, -(8 * s8))      (the compiler contracts `sumi * d8 - 8 * s8`; 8 * s8 is exact)
//   term = d5 * v, rounded on its own
// The ESIMD backend fuses a product into the add that follows it, so `term` goes through an xor with a run-time zero.
// shexp_bench checks the bits against native_mmvq on a model's own layers, and has to be rerun after a compiler update.
// The loads take dword-aligned addresses: the weights and the q8_1 buffer must sit at multiples of 4 (the launch checks).
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include <cstdint>
#include "strata/sycl_queue.hpp"
#include "strata/kernels/q5_0_mmvq_esimd.hpp"

#ifdef STRATA_I8_NO_ESIMD
namespace strata::kernels {
bool q5_0_mmvq_esimd_built() { return false; }
bool q5_0_mmvq_esimd(const void*, const void*, float*, int, int, int, void*, int, int) { return false; }
}  // namespace strata::kernels
#else
namespace strata::kernels {
namespace {
using namespace sycl::ext::intel::esimd;
using sycl::half;

template <int NC, int RPT, int VAR, int HOIST> struct Q50MmvqTag;

SYCL_ESIMD_FUNCTION inline simd<float, 8> fma8(simd<float, 8> a, simd<float, 8> b, simd<float, 8> c) {
    return simd<float, 8>(__builtin_elementwise_fma(a.data(), b.data(), c.data()));
}
SYCL_ESIMD_FUNCTION inline simd<float, 8> twice(simd<float, 4> v) {   // 4 values, each twice: the two dots of a block
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

constexpr int kNin = 640, kRow = 440, kStep = 88, kXRow = 180;   // a Q5_0 row of 640, 4 of its blocks; a q8_1 column in dwords

// 4 blocks of a weight row: the 16 low-nibble and 16 high-nibble weight dwords (5 bits a byte) and the 4 scales
struct Wt {
    simd<int, 16> lw, hw;
    simd<float, 8> d5;
};
SYCL_ESIMD_FUNCTION inline Wt load_w(const uint8_t* p, simd<uint32_t, 16> sh) {
    // blocks at bytes 0, 22 (in A) and 44, 66 (20, 42 of B); a block: fp16 d, 4 bytes qh, 16 bytes qs
    simd<uint32_t, 16> A = block_load<uint32_t, 16>((const uint32_t*) p);
    simd<uint32_t, 16> B = block_load<uint32_t, 16>((const uint32_t*) (p + 24));
    auto A8 = A.template bit_cast_view<uint8_t>();
    auto B8 = B.template bit_cast_view<uint8_t>();
    simd<uint8_t, 64> qs;
    qs.select<16, 1>(0) = A8.template select<16, 1>(6);
    qs.select<16, 1>(16) = A8.template select<16, 1>(28);
    qs.select<16, 1>(32) = B8.template select<16, 1>(26);
    qs.select<16, 1>(48) = B8.template select<16, 1>(48);
    simd<uint32_t, 16> vl = qs.template bit_cast_view<uint32_t>();
    simd<uint8_t, 64> qhb;   // each block's qh in its 4 dword lanes
#pragma unroll
    for (int m = 0; m < 4; ++m) {
        qhb.select<4, 1>(4 * m) = A8.template select<4, 1>(2);
        qhb.select<4, 1>(16 + 4 * m) = A8.template select<4, 1>(24);
        qhb.select<4, 1>(32 + 4 * m) = B8.template select<4, 1>(22);
        qhb.select<4, 1>(48 + 4 * m) = B8.template select<4, 1>(44);
    }
    simd<uint32_t, 16> qh = qhb.template bit_cast_view<uint32_t>();
    simd<uint32_t, 16> nl = (qh >> sh) & 0xFu;
    simd<uint32_t, 16> nh = (qh >> (sh + 16u)) & 0xFu;
    // bit j of a nibble to bit 4 of byte j (the partial products land on distinct bits: no carry)
    simd<uint32_t, 16> lw = (vl & 0x0F0F0F0Fu) | (((nl * 0x00204081u) & 0x01010101u) << 4);
    simd<uint32_t, 16> hw = ((vl >> 4) & 0x0F0F0F0Fu) | (((nh * 0x00204081u) & 0x01010101u) << 4);
    Wt w;
    w.lw = lw.template bit_cast_view<int>();
    w.hw = hw.template bit_cast_view<int>();
    auto A16 = A.template bit_cast_view<uint16_t>();
    auto B16 = B.template bit_cast_view<uint16_t>();
    simd<uint16_t, 4> dw;
    dw.select<1, 1>(0) = A16.template select<1, 1>(0);
    dw.select<1, 1>(1) = A16.template select<1, 1>(11);
    dw.select<1, 1>(2) = B16.template select<1, 1>(10);
    dw.select<1, 1>(3) = B16.template select<1, 1>(21);
    simd<half, 4> dh = dw.template bit_cast_view<half>();
    simd<float, 4> d4 = dh;
    w.d5 = twice(d4);
    return w;
}

// 4 q8_1 blocks of a column: the dwords the low / high nibbles meet, the scales and -(8 * the sums)
struct Act {
    simd<int, 16> ul, uh;
    simd<float, 8> d8, nu;
};
SYCL_ESIMD_FUNCTION inline Act load_x(const uint32_t* xe) {
    // q8_1 blocks of 9 dwords (ds, 8 of qs): blocks 0..2 in X0, block 3 at dword 23 of X1
    simd<uint32_t, 32> X0 = block_load<uint32_t, 32>(xe);
    simd<uint32_t, 32> X1 = block_load<uint32_t, 32>(xe + 4);
    simd<uint32_t, 16> ulo, uhi;
    ulo.select<4, 1>(0) = X0.template select<4, 1>(1);
    uhi.select<4, 1>(0) = X0.template select<4, 1>(5);
    ulo.select<4, 1>(4) = X0.template select<4, 1>(10);
    uhi.select<4, 1>(4) = X0.template select<4, 1>(14);
    ulo.select<4, 1>(8) = X0.template select<4, 1>(19);
    uhi.select<4, 1>(8) = X0.template select<4, 1>(23);
    ulo.select<4, 1>(12) = X1.template select<4, 1>(24);
    uhi.select<4, 1>(12) = X1.template select<4, 1>(28);
    simd<uint32_t, 4> ds;
    ds.select<3, 1>(0) = X0.template select<3, 9>(0);
    ds.select<1, 1>(3) = X1.template select<1, 1>(23);
    Act a;
    a.ul = ulo.template bit_cast_view<int>();
    a.uh = uhi.template bit_cast_view<int>();
    auto dsh = ds.template bit_cast_view<half>();
    simd<half, 4> d8h = dsh.template select<4, 2>(0);
    simd<half, 4> s8h = dsh.template select<4, 2>(1);
    simd<float, 4> d8q = d8h, s8q = s8h;
    a.d8 = twice(d8q);
    const simd<float, 8> u = twice(s8q) * 8.0f;
    a.nu = -u;
    return a;
}

// the 8 dots of 4 blocks.  VAR 0: the lane kernel's bits; 1 and 2: the other shapes (the experiment build)
template <int VAR>
SYCL_ESIMD_FUNCTION inline simd<float, 8> term(const Wt& w, const Act& a, simd<uint32_t, 8> zz) {
    simd<int, 16> p = dp4a<int>(simd<int, 16>(0), w.lw, a.ul);
    p = dp4a<int>(p, w.hw, a.uh);
    simd<int, 8> si = p.template select<8, 2>(0) + p.template select<8, 2>(1);
    simd<float, 8> f = si;
    if constexpr (VAR == 0) {
        const simd<float, 8> v = fma8(f, a.d8, a.nu);
        return xorf(w.d5 * v, zz);   // d5 * v rounded on its own: the xor with a run-time zero keeps it out of the add
    } else if constexpr (VAR == 1) {
        const simd<float, 8> v = xorf(f * a.d8, zz) + a.nu;
        return xorf(w.d5 * v, zz);
    } else {
        const simd<float, 8> v = fma8(f, a.d8, a.nu);
        return w.d5 * v;
    }
}

SYCL_ESIMD_FUNCTION inline float fold(simd<float, 8> t0, simd<float, 8> t1, simd<float, 8> t2, simd<float, 8> t3,
                                      simd<float, 8> t4) {
    const simd<float, 8> a = t0 + t4;
    const simd<float, 8> p0 = a + t2;
    const simd<float, 8> p1 = t1 + t3;
    const simd<float, 8> u = p0 + p1;
    simd<float, 4> t4s = u.template select<4, 1>(0) + u.template select<4, 1>(4);
    simd<float, 2> t2s = t4s.template select<2, 1>(0) + t4s.template select<2, 1>(2);
    const float s0 = t2s[0], s1 = t2s[1];
    return s0 + s1;
}

// HOIST 1: the columns' activations are unpacked once a thread (registers); 0: per row (as q4kq5_down_esimd does per entry)
template <int NC, int RPT, int VAR, int HOIST>
void launch(sycl::queue& q, const uint8_t* w, const uint32_t* x, float* y, int n_out) {
    const size_t threads = (size_t) (n_out / RPT);
    auto props = sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>};
    q.parallel_for<Q50MmvqTag<NC, RPT, VAR, HOIST>>(sycl::range<1>(threads), props, [=](sycl::id<1> id) SYCL_ESIMD_KERNEL {
        const int row0 = (int) id[0] * RPT;
        const simd<uint32_t, 8> zz((uint32_t) (n_out >> 31));   // 0 (n_out > 0), unknown to the compiler
        const simd<uint32_t, 16> sh = simd<uint32_t, 16>(0u, 4u) & 15u;   // 0, 4, 8, 12 for the 4 dwords of each block
        if constexpr (HOIST == 1) {
            Act X[NC][5];
#pragma unroll
            for (int c = 0; c < NC; ++c)
#pragma unroll
                for (int j = 0; j < 5; ++j) X[c][j] = load_x(x + (size_t) c * kXRow + 36 * j);
            for (int rr = 0; rr < RPT; ++rr) {
                const int r = row0 + rr;
                const uint8_t* wr = w + (size_t) r * kRow;
                Wt W[5];
#pragma unroll
                for (int j = 0; j < 5; ++j) W[j] = load_w(wr + j * kStep, sh);
#pragma unroll
                for (int c = 0; c < NC; ++c) {
                    simd<float, 8> t[5];
#pragma unroll
                    for (int j = 0; j < 5; ++j) t[j] = term<VAR>(W[j], X[c][j], zz);
                    y[(size_t) c * n_out + r] = fold(t[0], t[1], t[2], t[3], t[4]);
                }
            }
        } else {
            for (int rr = 0; rr < RPT; ++rr) {
                const int r = row0 + rr;
                const uint8_t* wr = w + (size_t) r * kRow;
                Wt W[5];
#pragma unroll
                for (int j = 0; j < 5; ++j) W[j] = load_w(wr + j * kStep, sh);
                for (int c = 0; c < NC; ++c) {
                    const uint32_t* xe = x + (size_t) c * kXRow;
                    simd<float, 8> t[5];
#pragma unroll
                    for (int j = 0; j < 5; ++j) t[j] = term<VAR>(W[j], load_x(xe + 36 * j), zz);
                    y[(size_t) c * n_out + r] = fold(t[0], t[1], t[2], t[3], t[4]);
                }
            }
        }
    });
}

#define STRATA_Q50E_EXP 1   // the experiment's variants (rows per thread, the float expression's shapes, the hoist)
constexpr int kVar = 0;     // the engine's variant: var = the expression's shape + 10 * HOIST

template <int RPT, int VAR, int HOIST>
bool launch_nc(sycl::queue& q, const uint8_t* w, const uint32_t* x, float* y, int n_out, int ncols) {
    switch (ncols) {
        case 1: launch<1, RPT, VAR, HOIST>(q, w, x, y, n_out); return true;
        case 2: launch<2, RPT, VAR, HOIST>(q, w, x, y, n_out); return true;
        case 3: launch<3, RPT, VAR, HOIST>(q, w, x, y, n_out); return true;
        case 4: launch<4, RPT, VAR, HOIST>(q, w, x, y, n_out); return true;
        default: return false;
    }
}
}  // namespace

bool q5_0_mmvq_esimd_built() { return true; }

bool q5_0_mmvq_esimd(const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, int ncols, void* stream,
                     int rpt, int var) {
    if (ncols < 1 || ncols > 4 || n_in != kNin || n_out <= 0 || rpt <= 0 || n_out % rpt != 0) return false;
    if (((uintptr_t) weights | (uintptr_t) x_q8_1) & 3) return false;
    sycl::queue& q = *strata::q_of(stream);
    const auto* w = (const uint8_t*) weights;
    const auto* x = (const uint32_t*) x_q8_1;
    if (var < 0) var = kVar;
#define STRATA_Q50E(RP, VA, HO) if (rpt == RP && var == VA + 10 * HO) return launch_nc<RP, VA, HO>(q, w, x, y, n_out, ncols);
#ifdef STRATA_Q50E_EXP
    STRATA_Q50E(1, 0, 0) STRATA_Q50E(2, 0, 0) STRATA_Q50E(4, 0, 0) STRATA_Q50E(8, 0, 0)
    STRATA_Q50E(1, 0, 1) STRATA_Q50E(2, 0, 1) STRATA_Q50E(4, 0, 1) STRATA_Q50E(8, 0, 1)
    STRATA_Q50E(4, 1, 0) STRATA_Q50E(4, 2, 0)
#else
    STRATA_Q50E(4, 0, 0)
#endif
#undef STRATA_Q50E
    return false;
}
}  // namespace strata::kernels
#endif
