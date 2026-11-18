// include/strata/kernels/router_top10.hpp - the MoE router, host-callable (P2.S2).
//
// logits (n_tokens, n_expert) -> the top-k experts per token with renormalised weights.  Semantics are
// `ref/moe.py::router`'s: softmax over ALL experts, stable descending argsort (ties break by index), gather,
// then renormalise with ggml's 2**-14 lower clamp.
//
// The BF16 GEMV that PRODUCES the logits is a separate kernel and is not here; this handles the routing only,
// which is why `logits` is taken as an f32 buffer the caller has already filled.
#pragma once

#include <cstdint>
#include <cstdlib>

namespace strata::kernels {

// `stream` may be null, in which case the call synchronises before returning.  `ids` is (n_tokens, k) int32 and
// `weights` is (n_tokens, k) f32, both device pointers.  k must be <= 64.
void router_top10(const float* logits, int n_tokens, int n_expert, int k, int* ids, float* weights,
                  void* stream);

/// Tests (S6): one router kernel by name - 0 the portable kernel, 1 the fast kernel (HIP's default for 64 < n_expert
/// <= 512, k <= 32, and every backend's for router_ngen_applies; bitwise the same results), 2 the fast kernel forced
/// onto its serial-sum path.  HIP, CUDA (since pruned-experts) and SYCL; false (nothing launched) outside that geometry.
bool router_top10_variant(const float* logits, int n_tokens, int n_expert, int k, int* ids, float* weights,
                          void* stream, int variant);

/// pruned-experts: STRATA_ROUTER_NGEN (on unless set to 0).  A model whose expert count is neither 512 nor 256 (a
/// pruned Flash-Next, 384 or 320 of 512) has no native router: without this its verify window routed token by token
/// (two launches per row and layer instead of two per layer) and CUDA / SYCL ran the portable top-k kernel.  With it
/// the window routes in one multi-column projection and one n-token top-k (verify.cpp, moe_route_window) and the
/// top-k is the fast kernel - every result bitwise the old path's (router_ngen_parity).  512 and 256 are untouched.
inline bool router_ngen_enabled() {
    static const bool on = [] {
        const char* v = std::getenv("STRATA_ROUTER_NGEN");
        return v == nullptr || *v == '\0' || std::atoi(v) != 0;
    }();
    return on;
}
/// The router shapes STRATA_ROUTER_NGEN covers: n_expert not 512 / 256 (their own kernels), 64 < n_expert <= 512 and
/// 1 <= k <= 32 (the fast kernel's range).
constexpr bool router_ngen_shape(int n_expert, int k) {
    return n_expert != 512 && n_expert != 256 && n_expert > 64 && n_expert <= 512 && k >= 1 && k <= 32;
}
// the canonical model and the Coder keep their paths; the pruned Flash-Next variants take the new one
static_assert(!router_ngen_shape(512, 10) && !router_ngen_shape(256, 8) && !router_ngen_shape(256, 10),
              "STRATA_ROUTER_NGEN must not touch the 512 / 256 expert routers");
static_assert(router_ngen_shape(384, 10) && router_ngen_shape(320, 10), "STRATA_ROUTER_NGEN covers 384 and 320");
/// Whether STRATA_ROUTER_NGEN is on and covers this router shape.
inline bool router_ngen_applies(int n_expert, int k) { return router_ngen_enabled() && router_ngen_shape(n_expert, k); }

}  // namespace strata::kernels
