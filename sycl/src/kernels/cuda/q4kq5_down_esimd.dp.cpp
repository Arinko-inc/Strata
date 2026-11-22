// STRATA_EXPERT_Q4KQ5's ESIMD down kernel (Battlemage): native_down_kernel<6, 8> (the decode window's Q5_0 down rows,
// n_ff 640) with one hardware thread per weight row instead of 8 work-items of a 32-wide sub-group.
//
// The lane kernel's work-item `sub` of a row sums the dots k = sub, sub + 8, ... (k = 2 * block + half; 40 a row), each
// read with per-lane gathers; the 8 sums then meet in lanes_sum<8>'s tree.  Here step j (0..4) takes the row's blocks
// 4j..4j+3 (88 bytes, two 64-byte loads) and the entry's q8_1 blocks 4j..4j+3 (144 bytes, two 128-byte loads): its 8 dots
// are the 8 lanes' j-th terms, so an 8-wide float register is the 8 lane sums.  The arithmetic is vec_dot_q5_0_q8_1's:
//   sumi (int32, exact in any order) = the four dp4a of a dot;  term = d5 * (sumi * d8 - 8 * s8);  lane += term
//   out = ((l0 + l4) + (l2 + l6)) + ((l1 + l5) + (l3 + l7))                                      (lanes_sum<8>)
// `FMA` is how the compiler built `sumi * d8 - 8 * s8` in the lane kernel: fused (fma(sumi, d8, -(8 * s8)); 8 * s8 is
// exact) or not; q4k_q5_expert_bench says which one gives the lane kernel's bits on this GPU.
// The loads take dword-aligned addresses: a blob's base must be a multiple of 4 (the rows' offsets are).
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include <cstdint>
#include "strata/sycl_queue.hpp"
#include "strata/kernels/iq_kernels.hpp"

namespace strata::kernels {
bool q4kq5_down_esimd(int variant, unsigned groups, void* stream, const unsigned long long* grp_ptr, const int32_t* grp_start,
                      const int32_t* n_groups, const int32_t* ent_dst, const void* hq, const NativeExpertLayout& L, float* out);
}

#ifdef STRATA_I8_NO_ESIMD
namespace strata::kernels {
bool q4kq5_down_esimd(int, unsigned, void*, const unsigned long long*, const int32_t*, const int32_t*, const int32_t*, const void*,
                      const NativeExpertLayout&, float*) { return false; }
}  // namespace strata::kernels
#else
namespace strata::kernels {
namespace {
using namespace sycl::ext::intel::esimd;
using sycl::half;

template <int IN, int RPT, int OUT> struct Q4kq5DownTag;

SYCL_ESIMD_FUNCTION inline simd<float, 8> fma8(simd<float, 8> a, simd<float, 8> b, simd<float, 8> c) {
    return simd<float, 8>(__builtin_elementwise_fma(a.data(), b.data(), c.data()));
}
// 4 values, each twice: the two dots of a block
SYCL_ESIMD_FUNCTION inline simd<float, 8> twice(simd<float, 4> v) {
    simd<float, 8> r;
    r.select<4, 2>(0) = v;
    r.select<4, 2>(1) = v;
    return r;
}

constexpr int kRow = 440, kStep = 88, kHRow = 720;   // a Q5_0 row of 640, 4 of its blocks; an entry's q8_1 row

// IN / OUT: how `sumi * d8 - 8 * s8` and `lane + d5 * (...)` are built: 0 as written (the compiler's choice), 1 fma,
// 2 the product rounded on its own (it goes through an xor with a run-time zero, so no later pass can fuse it)
template <int IN, int RPT, int OUT>
void launch(sycl::queue& q, unsigned groups, const unsigned long long* grp_ptr, const int32_t* grp_start,
            const int32_t* n_groups, const int32_t* ent_dst, const uint8_t* hq, size_t down_off, int n_embd, float* out) {
    auto props = sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>};
    q.submit([&](sycl::handler& h) {
        h.parallel_for<Q4kq5DownTag<IN, RPT, OUT>>(sycl::range<2>((size_t) groups, (size_t) (n_embd / RPT)), props, [=](sycl::id<2> id) SYCL_ESIMD_KERNEL {
            const int ng = *n_groups;
            const int gy = (int) id[0];
            if (gy >= ng) return;
            const simd<uint32_t, 8> zz((uint32_t) (ng >> 31));   // 0 (ng >= 0), unknown to the compiler
            const simd<uint32_t, 16> sh = simd<uint32_t, 16>(0u, 4u) & 15u;   // 0, 4, 8, 12 for the 4 dwords of each block
            for (int rr = 0; rr < RPT; ++rr) {
                const int r = (int) id[1] * RPT + rr;
                for (int g = gy; g < ng; g += (int) groups) {
                    const uint8_t* wr = (const uint8_t*) grp_ptr[g] + down_off + (size_t) r * kRow;
                    // the row: per step the 16 low-nibble and 16 high-nibble weight dwords (5 bits a byte) and the 4 scales
                    simd<int, 16> Lw[5], Hw[5];
                    simd<float, 8> D5[5];
#pragma unroll
                    for (int j = 0; j < 5; ++j) {
                        // blocks at bytes 0, 22 (in A) and 44, 66 (20, 42 of B); a block: fp16 d, 4 bytes qh, 16 bytes qs
                        simd<uint32_t, 16> A = block_load<uint32_t, 16>((const uint32_t*) (wr + j * kStep));
                        simd<uint32_t, 16> B = block_load<uint32_t, 16>((const uint32_t*) (wr + j * kStep + 24));
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
                        Lw[j] = lw.template bit_cast_view<int>();
                        Hw[j] = hw.template bit_cast_view<int>();
                        auto A16 = A.template bit_cast_view<uint16_t>();
                        auto B16 = B.template bit_cast_view<uint16_t>();
                        simd<uint16_t, 4> dw;
                        dw.select<1, 1>(0) = A16.template select<1, 1>(0);
                        dw.select<1, 1>(1) = A16.template select<1, 1>(11);
                        dw.select<1, 1>(2) = B16.template select<1, 1>(10);
                        dw.select<1, 1>(3) = B16.template select<1, 1>(21);
                        simd<half, 4> dh = dw.template bit_cast_view<half>();
                        simd<float, 4> d4 = dh;
                        D5[j] = twice(d4);
                    }
                    const int e0 = grp_start[g], e1 = grp_start[g + 1];
                    for (int e = e0; e < e1; ++e) {
                        const uint32_t* xe = (const uint32_t*) (hq + (size_t) e * kHRow);
                        simd<float, 8> acc(0.0f);
#pragma unroll
                        for (int j = 0; j < 5; ++j) {
                            // q8_1 blocks of 9 dwords (ds, 8 of qs): blocks 0..2 in X0, block 3 at dword 23 of X1
                            simd<uint32_t, 32> X0 = block_load<uint32_t, 32>(xe + 36 * j);
                            simd<uint32_t, 32> X1 = block_load<uint32_t, 32>(xe + 36 * j + 4);
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
                            simd<int, 16> ul = ulo.template bit_cast_view<int>(), uh = uhi.template bit_cast_view<int>();
                            simd<int, 16> p = dp4a<int>(simd<int, 16>(0), Lw[j], ul);
                            p = dp4a<int>(p, Hw[j], uh);
                            simd<int, 8> si = p.template select<8, 2>(0) + p.template select<8, 2>(1);
                            simd<float, 8> f = si;
                            auto dsh = ds.template bit_cast_view<half>();
                            simd<half, 4> d8h = dsh.template select<4, 2>(0);
                            simd<half, 4> s8h = dsh.template select<4, 2>(1);
                            simd<float, 4> d8q = d8h, s8q = s8h;
                            const simd<float, 8> d8 = twice(d8q), s8 = twice(s8q);
                            const simd<float, 8> u = s8 * 8.0f;
                            simd<float, 8> v;
                            if constexpr (IN == 1) v = fma8(f, d8, -u);
                            else if constexpr (IN == 2) {
                                simd<float, 8> pr = f * d8;
                                simd<uint32_t, 8> pb = pr.template bit_cast_view<uint32_t>();
                                pb = pb ^ zz;
                                simd<float, 8> pp = pb.template bit_cast_view<float>();
                                v = pp - u;
                            } else v = f * d8 - u;
                            if constexpr (OUT == 1) acc = fma8(D5[j], v, acc);
                            else if constexpr (OUT == 2) {
                                simd<float, 8> pr = D5[j] * v;
                                simd<uint32_t, 8> pb = pr.template bit_cast_view<uint32_t>();
                                pb = pb ^ zz;
                                simd<float, 8> pp = pb.template bit_cast_view<float>();
                                acc = acc + pp;
                            } else {
                                const simd<float, 8> term = D5[j] * v;
                                acc = acc + term;
                            }
                        }
                        simd<float, 4> t4 = acc.template select<4, 1>(0) + acc.template select<4, 1>(4);
                        simd<float, 2> t2 = t4.template select<2, 1>(0) + t4.template select<2, 1>(2);
                        const float s0 = t2[0], s1 = t2[1];
                        out[(size_t) ent_dst[e] * n_embd + r] = s0 + s1;
                    }
                }
            }
        });
    });
}
}  // namespace

bool q4kq5_down_esimd(int variant, unsigned groups, void* stream, const unsigned long long* grp_ptr, const int32_t* grp_start,
                      const int32_t* n_groups, const int32_t* ent_dst, const void* hq, const NativeExpertLayout& L, float* out) {
    if (L.d_type != 6 || L.n_ff != 640 || L.d_row != kRow || L.n_embd % 16 != 0 || variant < 1 || variant > 7) return false;
    sycl::queue& q = *strata::q_of(stream);
    const uint8_t* h = (const uint8_t*) hq;
#define STRATA_Q4KQ5_E(IN, RPT, OUT) launch<IN, RPT, OUT>(q, groups, grp_ptr, grp_start, n_groups, ent_dst, h, L.down_off, (int) L.n_embd, out); break;
    switch (variant) {
        case 1: STRATA_Q4KQ5_E(1, 1, 2)
        case 2: STRATA_Q4KQ5_E(1, 4, 2)
        case 3: STRATA_Q4KQ5_E(2, 4, 1)
        case 4: STRATA_Q4KQ5_E(2, 4, 2)
        case 5: STRATA_Q4KQ5_E(1, 8, 2)
        case 6: STRATA_Q4KQ5_E(1, 16, 2)
        default: STRATA_Q4KQ5_E(1, 2, 2)
    }
#undef STRATA_Q4KQ5_E
    return true;
}
}  // namespace strata::kernels
#endif
