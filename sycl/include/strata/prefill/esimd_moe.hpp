// 151-b70: fused int8 DPAS expert products for the prompt path of the SYCL port (Battlemage, ESIMD).
//
// A group of experts' gate/up and down products straight from their quantized blobs: the weights are decoded in registers into
// the DPAS B operand (int8, one decode per 16 output rows and 32 of K), the activations are int8 with a float scale per 32 of K
// (llama.cpp's q8_1 grid, rounding half away from zero), each 32-block's int32 dot product is converted and scaled in float.
// The FP16 weights never exist in memory.  The bits differ from the dequantize + oneMKL path (activations are quantized): the
// caller keeps it opt-in until a quality check (docs/INTEL.md, "int8 expert path").
//
// Row layout.  A group's rows are the experts' rows back to back (cnt[0], cnt[1], ... rows, as the FP16 path has them).  The
// int8 activations are stored per expert padded to a multiple of 8 rows, in DPAS A-operand order:
//   Xq [padded rows / 8][K / 32][8 rows x 8 dwords]   dx [padded rows / 8][K / 32][8 floats]
// so one 256-byte block load is an 8x32 A tile.  Pad rows are zero.
#pragma once
#include <cstddef>
#include <cstdint>

namespace strata::prefill::i8 {

constexpr int kMaxGroup = 16;

/// Per group: the experts' row counts and where their (8-row padded) tiles start.
struct Layout {
    int n = 0;
    int cnt[kMaxGroup] = {};
    int row0[kMaxGroup + 1] = {};    // first row of expert i (unpadded), row0[n] = total rows
    int tile0[kMaxGroup + 1] = {};   // first 8-row tile of expert i in the padded layout, tile0[n] = total tiles
};
void make_layout(Layout& lo, const int32_t* cnt, int n);

/// True when this device can run the kernels (a Battlemage-class device: sub-group 16 and the s8 DPAS 8x16x32).
bool available(void* stream);
/// True when this device is a Battlemage GPU by its architecture id (bmg-g21: B580 / B570, bmg-g31: B70), not by its name.  The default
/// choice below uses it: Lunar Lake and the other Xe2 parts can run the kernels (`available`) but are not defaulted.  False in a build
/// without the ESIMD kernels (STRATA_I8_NO_ESIMD).
bool battlemage(void* stream);
/// True when any GPU on this machine is a Battlemage one (sizes the staging ring before a queue exists).
bool any_battlemage();

/// Which chain the prompt experts take.  `env` is STRATA_PREFILL_MMQ (else STRATA_PF_I8), null or empty when unset.  Returns 0 = the
/// FP16 chain (dequantize + oneMKL), 1 = the int8 chain everywhere, 2 = auto (int8 per group while the group's rows per expert are under
/// the crossover measured on this GPU).  Set: "0" -> 0, "auto" -> 2, any other number -> 1.  Unset: auto on a Battlemage device whose
/// kernels are in the binary and run, else 0 (Alchemist and every other GPU keep the FP16 chain).
inline int choose_mode(const char* env, bool battlemage_device, bool kernels_available) {
    if (env != nullptr && env[0] != 0) {
        if (env[0] == 'a' || env[0] == 'A') return 2;
        long v = 0;
        for (const char* c = env; *c >= '0' && *c <= '9' && v < 1000000; ++c) v = v * 10 + (*c - '0');
        return v != 0 ? 1 : 0;
    }
    return (battlemage_device && kernels_available) ? 2 : 0;
}

/// The expert formats (ggml types) the kernels take.  gate/up: IQ3_S 21, IQ3_XXS 18, IQ2_S 22 and Q4_K 12 (K a multiple of 256);
/// down: IQ4_NL 20, Q2_0 42 and Q5_0 6.  Q4_K and Q5_0 (the "K-quant" pair of the Q4_K_S packs) are behind their own switch:
/// STRATA_I8_KQ=0 sends a layer with either of them to the FP16 chain, whatever STRATA_PREFILL_MMQ says.
constexpr int kTypeQ4K = 12, kTypeQ5_0 = 6;
inline bool kq_type(int t) { return t == kTypeQ4K || t == kTypeQ5_0; }
/// STRATA_I8_KQ (null or empty when unset): on unless it starts with '0'.
inline bool kq_on(const char* env) { return !(env != nullptr && env[0] == '0'); }
inline bool gu_type_ok(int t) { return t == 21 || t == 18 || t == 22 || t == kTypeQ4K; }
inline bool d_type_ok(int t) { return t == 20 || t == 42 || t == kTypeQ5_0; }
/// A layer's pair of formats takes the int8 chain (`kq`: kq_on(STRATA_I8_KQ)).
inline bool formats_ok(int gu_type, int d_type, bool kq) {
    if (!gu_type_ok(gu_type) || !d_type_ok(d_type)) return false;
    return kq || (!kq_type(gu_type) && !kq_type(d_type));
}
/// Bytes of one ggml block of a handled type (0 for any other type).
inline int block_bytes(int t) {
    return t == 21 ? 110 : t == 18 ? 98 : t == 22 ? 82 : t == kTypeQ4K ? 144 : t == 20 || t == 42 ? 18 : t == kTypeQ5_0 ? 22 : 0;
}

size_t xq_bytes(const Layout& lo, int K);   // int8 activations, padded tiles
size_t dx_bytes(const Layout& lo, int K);   // their scales (floats); the activation sums (sx) take as many

/// Scratch for one group: int8 activation tiles + scales for the gate/up input (K) and for the down input (n_ff).  Grow-only,
/// owned by the library (device USM on the queue's device).  `sx`: per row and 32 of K, the scale times the sum of the 32 int8
/// values (q8_1's `s`), for the formats with a per-block minimum (Q4_K: w = d sc q - dmin m).
struct Workspace {
    int8_t* xq = nullptr;
    float* dx = nullptr;
    int8_t* hq = nullptr;
    float* dh = nullptr;
    float* sx = nullptr;
};
bool workspace(const Layout& lo, int K, int n_ff, Workspace& ws, void* stream);

/// FP16 rows X[row0[n]][K] -> int8 tiles + scales, and with `sx` non-null the scaled sums (same layout as dx).
void quantize_rows(const uint16_t* X, const Layout& lo, int K, int8_t* Xq, float* dx, void* stream, float* sx = nullptr);

/// GU[row][v] (FP32, row stride 2 * n_ff), v = 2 r gate row r, 2 r + 1 up row r (the interleave swiglu_interleaved reads), for the
/// rows of the group.  blob[i] holds expert i's gate rows at 0 and its up rows at up_off (ggml rows, K a multiple of 256).
/// Q4_K needs `sx` (quantize_rows' sums).  Returns false when the type / shape is not handled (the caller keeps its other path).
bool gate_up(int gu_type, const uint8_t* const* blob, size_t up_off, int n_ff, const Layout& lo, int K, const int8_t* Xq,
             const float* dx, float* GU, void* stream, const float* sx = nullptr);

/// Hq / dh: silu(gate) * up from GU (FP32, interleaved) quantized per 32 of n_ff into the same tile layout (K = n_ff).
void swiglu_quantize(const float* GU, const Layout& lo, int n_ff, int8_t* Hq, float* dh, void* stream);

/// Dm[row][n_embd] (FP32) = Hq . down^T, down rows (n_embd of them, K = n_ff) in ggml type d_type at blob[i] + down_off
/// (Q5_0: n_ff a multiple of 64, so that every row starts dword aligned).
bool down(int d_type, const uint8_t* const* blob, size_t down_off, int n_embd, const Layout& lo, int n_ff, const int8_t* Hq,
          const float* dh, float* Dm, void* stream);

}  // namespace strata::prefill::i8
