// src/kernels/cuda/q4k_q5_experts.cuh - the grouped decode experts of a Q4_K gate/up + Q5_0 down layer (orca
// Q4_K_S: every layer 6..47; n_embd 2560, n_ff 640), included by iq_kernels.cu just above native_expert_grouped.
//
// STRATA_EXPERT_V2K's kernels cover Q4_K gate/up with a Q5_1 / Q8_0 down only, so this pair ran native_gu_kernel<12>,
// swiglu_q8_1_entries_kernel and native_down_kernel<6>: one entry at a time, one warp a row.  Here (RX 7900 XT,
// gfx1100, layer 20 of orca Q4_K_S, q4k_q5_expert_bench; the card reads the same blobs at ~740 GB/s with nothing
// computed, and the old kernels were at 555 GB/s for a T = 4 window):
//   down     8 lanes a row (q4kq5_down8_kernel): a Q5_0 row of n_ff 640 is 40 calls, of which a full warp's second
//            pass has 8 lanes working; a group's entries GRP_NC at a time, the weight words built once
//   gate/up  the old grid; a group of 2+ entries takes them GRP_NC at a time (q4kq5_gu_h_kernel)
// Every output is the old kernels', bit for bit (the same calls into the same sums in the same order).
// STRATA_EXPERT_Q4KQ5: 0 the old kernels, 1 gate/up only, 2 down only, 3 both; unset: 3 on a gfx11 card (HIP), else 0.
namespace {

constexpr int Q4KQ5_XB = 2560 / 32, Q4KQ5_HB = 640 / 32;

// Q5_0 as a Split (load / apply) for the decode-once row dots.  load builds the four 5-bit weight words of the call
// (the 5th bits spread with one multiply: bits 0..3 of x to bit 0 of each byte is (x * 0x204081) & 0x01010101; the
// terms x, x << 7, x << 14, x << 21 do not overlap) - the words vec_dot_q5_0_q8_1 builds - and apply runs its dp4a
// chain and float expression.  Not registered in kSplit: only the kernels below use it.
template<> struct Split<6> {
    struct W { int vi[2 * VDR_Q5_0]; float d; };
    template<bool STAGE_GRID = false>
    __device__ static W load(const void* __restrict__ vbq, int kbx, int iqs, const uint32_t* __restrict__ = nullptr) {
        const block_q5_0* b = (const block_q5_0*) vbq + kbx;
        const uint32_t qh = (uint32_t) get_int_b2(b->qh, 0);
        W r;
#pragma unroll
        for (int i = 0; i < VDR_Q5_0; ++i) {
            const uint32_t vl = (uint32_t) get_int_b2(b->qs, iqs + i);
            const uint32_t vh = qh >> (4 * (iqs + i));
            r.vi[2 * i + 0] = (int) ((vl & 0x0F0F0F0Fu) | ((((vh & 0xFu) * 0x00204081u) & 0x01010101u) << 4));
            r.vi[2 * i + 1] = (int) (((vl >> 4) & 0x0F0F0F0Fu) | (((((vh >> 16) & 0xFu) * 0x00204081u) & 0x01010101u) << 4));
        }
        r.d = __half2float(b->d);
        return r;
    }
    __device__ static float apply(const W& r, const block_q8_1* __restrict__ bq8_1, int iqs) {
        int sumi = 0;
#pragma unroll
        for (int i = 0; i < VDR_Q5_0; ++i) {
            const int u0 = get_int_b4(bq8_1->qs, iqs + i), u1 = get_int_b4(bq8_1->qs, iqs + i + QI5_0);
            sumi = ggml_cuda_dp4a(r.vi[2 * i + 0], u0, sumi);
            sumi = ggml_cuda_dp4a(r.vi[2 * i + 1], u1, sumi);
        }
        const float d5 = r.d;
        const float d8 = __low2float(bq8_1->ds);
        const float s8 = __high2float(bq8_1->ds);
        return d5 * (sumi * d8 - (16.0f * VDR_Q5_0 / QI5_0) * s8);
    }
};

// The down rows as 8-lane rows.  A Q5_0 row of n_ff 640 is 40 calls (20 blocks x 2), so native_down_kernel's warp
// has lanes 0..7 make two calls and lanes 8..31 one: 40 of the 64 lane slots of its two passes work.  As IQ4_NL's
// rows (the same 40 calls): 8 lanes x 5 calls, 4 rows a warp, 32 rows a block, row_dot_40_sub8 (the full warp's
// addition tree, bit for bit), a group's entries GRP_NC at a time with each weight word built once.
// The chunk's h rows are read where they are (staged in shared memory as native_down_multi_kernel does: no faster).
__global__ void __launch_bounds__(256) q4kq5_down8_kernel(const unsigned long long* __restrict__ grp_ptr,
                                                          const int32_t* __restrict__ grp_start,
                                                          const int32_t* __restrict__ n_groups,
                                                          const int32_t* __restrict__ ent_dst,
                                                          const block_q8_1* __restrict__ hq, NativeExpertLayout L,
                                                          float* __restrict__ out) {
    const int g = blockIdx.y;
    if (g >= *n_groups) return;
    const int r = blockIdx.x * 32 + (threadIdx.x >> 3), t = threadIdx.x & 7;
    const uint8_t* wr = (const uint8_t*) grp_ptr[g] + L.down_off + (size_t) r * 440;
    const int e0 = grp_start[g], e1 = grp_start[g + 1];
    for (int e = e0; e < e1; e += GRP_NC) {
        const int n = min(GRP_NC, e1 - e);
        const block_q8_1* hx = hq + (size_t) e * Q4KQ5_HB;
        float* o = out + r;
        if (n == 1) {
            const int off1[1] = { 0 };
            float s1[1];
            row_dot_40_sub8<6, 1, true>(wr, hx, off1, 1, t, s1);
            if (t == 0) o[(size_t) ent_dst[e] * 2560] = s1[0];
        } else if (n == 2) {
            const int off2[2] = { 0, 20 };
            float s2[2];
            row_dot_40_sub8<6, 2, true>(wr, hx, off2, 2, t, s2);
            if (t < 2) o[(size_t) ent_dst[e + t] * 2560] = s2[t];
        } else if (n == 3) {
            const int off3[3] = { 0, 20, 40 };
            float s3[3];
            row_dot_40_sub8<6, 3, true>(wr, hx, off3, 3, t, s3);
            if (t < 3) o[(size_t) ent_dst[e + t] * 2560] = s3[t];
        } else {
            const int off4[4] = { 0, 20, 40, 60 };
            float s4[4];
            row_dot_40_sub8<6, 4, true>(wr, hx, off4, 4, t, s4);
            if (t < 4) o[(size_t) ent_dst[e + t] * 2560] = s4[t];
        }
    }
}

// gate/up: native_gu_kernel's grid (8 rows a block, one warp a row; a Q4_K row of n_embd 2560 is 160 calls, 5 a
// lane).  A group of one entry runs row_dot (that kernel's code); a larger one takes its entries GRP_NC at a time
// with the weight side of a call done once (row_dot_multi with SplitK<12>: the same calls, order and warp_sum).
__global__ void __launch_bounds__(256) q4kq5_gu_h_kernel(const unsigned long long* __restrict__ grp_ptr,
                                                         const int32_t* __restrict__ grp_start,
                                                         const int32_t* __restrict__ n_groups,
                                                         const int32_t* __restrict__ ent_tok,
                                                         const block_q8_1* __restrict__ xq, NativeExpertLayout L,
                                                         float* __restrict__ gate, float* __restrict__ up) {
    const int g = blockIdx.y;
    if (g >= *n_groups) return;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row = blockIdx.x * GU_ROWS + warp;
    const bool is_up = row >= 640;
    const int r = is_up ? row - 640 : row;
    const uint8_t* wr = (const uint8_t*) grp_ptr[g] + (is_up ? L.up_off : 0) + (size_t) r * L.gu_row;
    float* dst = (is_up ? up : gate) + r;
    const int nb = (int) (L.n_embd / Fmt<12>::qk);
    const int e0 = grp_start[g], e1 = grp_start[g + 1];
    if (e1 - e0 == 1) {
        const float v = row_dot<12>(wr, xq + (size_t) ent_tok[e0] * Q4KQ5_XB, nb, lane);
        if (lane == 0) dst[(size_t) e0 * 640] = v;
        return;
    }
    for (int e = e0; e < e1; e += GRP_NC) {
        const int n = min(GRP_NC, e1 - e);
        if (n == 1) {
            const int off1[1] = { ent_tok[e] * Q4KQ5_XB };
            float s1[1];
            row_dot_multi<12, 1, true>(wr, xq, off1, 1, nb, lane, s1);
            if (lane == 0) dst[(size_t) e * 640] = s1[0];
        } else if (n == 2) {
            const int off2[2] = { ent_tok[e] * Q4KQ5_XB, ent_tok[e + 1] * Q4KQ5_XB };
            float s2[2];
            row_dot_multi<12, 2, true>(wr, xq, off2, 2, nb, lane, s2);
            if (lane < 2) dst[(size_t) (e + lane) * 640] = s2[lane];
        } else if (n == 3) {
            const int off3[3] = { ent_tok[e] * Q4KQ5_XB, ent_tok[e + 1] * Q4KQ5_XB, ent_tok[e + 2] * Q4KQ5_XB };
            float s3[3];
            row_dot_multi<12, 3, true>(wr, xq, off3, 3, nb, lane, s3);
            if (lane < 3) dst[(size_t) (e + lane) * 640] = s3[lane];
        } else {
            const int off4[4] = { ent_tok[e] * Q4KQ5_XB, ent_tok[e + 1] * Q4KQ5_XB, ent_tok[e + 2] * Q4KQ5_XB,
                                  ent_tok[e + 3] * Q4KQ5_XB };
            float s4[4];
            row_dot_multi<12, 4, true>(wr, xq, off4, 4, nb, lane, s4);
            if (lane < 4) dst[(size_t) (e + lane) * 640] = s4[lane];
        }
    }
}

// the bench only (variant 999): every byte of the groups' blobs read with 16-byte loads, nothing computed - the card's
// read rate on this access pattern, the bound for any kernel that reads each weight once
__global__ void __launch_bounds__(256) q4kq5_read_kernel(const unsigned long long* __restrict__ grp_ptr,
                                                         const int32_t* __restrict__ n_groups, size_t bytes, int per,
                                                         float* __restrict__ sink) {
    const int g = blockIdx.y;
    if (g >= *n_groups) return;
    const uint4* p = (const uint4*) grp_ptr[g] + (size_t) blockIdx.x * per * 256;
    const size_t left = bytes / 16 - (size_t) blockIdx.x * per * 256;
    unsigned acc = 0;
    for (int i = 0; i < per; ++i) {
        const size_t j = (size_t) i * 256 + threadIdx.x;
        if (j < left) {
            const uint4 v = p[j];
            acc ^= v.x ^ v.y ^ v.z ^ v.w;
        }
    }
    if (acc == 0x12345678u) sink[0] = 1.0f;
}

int g_q4kq5 = -2;   // -2: not read yet; native_expert_set_q4kq5 overrides
int q4kq5_default() {
    const char* v = std::getenv("STRATA_EXPERT_Q4KQ5");
    if (v != nullptr && v[0] != '\0') return std::atoi(v);
#if defined(__HIPCC__) && !defined(STRATA_HIP_GFX906)
    // gfx11 (measured on gfx1100, RX 7900 XT): on.  Other cards: not measured, the old kernels.
    int dev = 0;
    cudaDeviceProp p;
    const bool g11 = cudaGetDevice(&dev) == cudaSuccess && cudaGetDeviceProperties(&p, dev) == cudaSuccess &&
                     strata::kernels::gfx_arch_is_gfx11_wmma(p.gcnArchName);
    (void) cudaGetLastError();
    return g11 ? 3 : 0;
#else
    return 0;
#endif
}
int q4kq5_variant() {
    if (g_q4kq5 == -2) {
        g_q4kq5 = q4kq5_default();
        std::fprintf(stderr, "strata: Q4_K / Q5_0 decode experts: STRATA_EXPERT_Q4KQ5=%d (%s)\n", g_q4kq5,
                     g_q4kq5 <= 0 ? "the per-entry kernels" : g_q4kq5 == 3 ? "8-lane down rows, grouped gate/up" :
                     g_q4kq5 == 2 ? "8-lane down rows" : g_q4kq5 == 1 ? "grouped gate/up" : "bench");
    }
    return g_q4kq5;
}
bool q4kq5_layout(const NativeExpertLayout& L) {
    return !L.gu_nib && L.gu_type == 12 && L.d_type == 6 && L.n_embd == 2560 && L.n_ff == 640 && L.had_block == 0 &&
           L.tail_off == 0;
}

// hx: STRATA_EXPERT_Q4KQ5's value (> 0); phase: native_expert_set_mode's (0 all, 1 gate/up + SwiGLU + q8_1, 2 down,
// 3 gate/up).  One block row per possible group (every kernel here returns for g >= *n_groups).
void q4kq5_launch(int hx, int phase, const NativeExpertLayout& L, int64_t cap_groups, int64_t cap_entries, cudaStream_t s,
                  const unsigned long long* grp_ptr, const int32_t* grp_start, const int32_t* n_groups,
                  const int32_t* ent_dst, const int32_t* ent_tok, const block_q8_1* X, float* gate, float* up,
                  block_q8_1* hq, float* out) {
    const unsigned cg = (unsigned) cap_groups;
    if (hx >= 997) {   // the bench: the read rate, 999 = 8 loads a lane, 998 = 2, 997 = 32
        const int per = hx == 999 ? 8 : hx == 998 ? 2 : 32;
        const unsigned nbx = (unsigned) ((L.bytes / 16 + (size_t) per * 256 - 1) / ((size_t) per * 256));
        q4kq5_read_kernel<<<dim3(nbx, cg), 256, 0, s>>>(grp_ptr, n_groups, L.bytes, per, gate);
        return;
    }
    if (phase != 2) {
        const dim3 ggu((unsigned) (2 * L.n_ff / GU_ROWS), cg);
        if (hx & 1) q4kq5_gu_h_kernel<<<ggu, 256, 0, s>>>(grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up);
        else launch_gu<12>(ggu, s, grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up);
        if (phase == 3) return;
        const long long nh = (long long) cap_entries * L.n_ff;
        swiglu_q8_1_entries_kernel<<<(unsigned) ((nh + 255) / 256), 256, 0, s>>>(gate, up, grp_start, n_groups, (int) L.n_ff, hq);
    }
    if (phase == 1) return;
    if (hx & 2) q4kq5_down8_kernel<<<dim3((unsigned) (L.n_embd / 32), cg), 256, 0, s>>>(grp_ptr, grp_start, n_groups, ent_dst, hq, L, out);
    else launch_down<6>(dim3((unsigned) (L.n_embd / 8), cg), s, grp_ptr, grp_start, n_groups, ent_dst, hq, L, out);
}

}  // namespace

void native_expert_set_q4kq5(int v) { g_q4kq5 = v < 0 ? q4kq5_default() : v; }
int native_expert_q4kq5() { return q4kq5_variant(); }
