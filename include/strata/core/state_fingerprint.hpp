#pragma once

// STRATA_CKPT_FP=1 (diagnostic, off by default): one stderr line per conversation checkpoint saved and per batch slot
// copy, with 64-bit FNV-1a fingerprints of the running state (each part apart) and of the K/V of the cells below it.
// Two runs, or two paths of one run that reach the same tokens (a solo next turn and the same turn from a batch slot,
// both with a turn checkpoint at the same length), can then be compared bit for bit per process from the log lines
// alone: a remote stage's main process, relay and last worker each print their own layers'.
//
// The K/V covers the whole pages below the length only: the cells of the last, partial page past the length still
// hold whatever an earlier conversation wrote there (they differ between two paths that agree on every cell below the
// length), and the pooled index rows likewise only the completed ones.  Computing it copies the K/V to host memory:
// a diagnostic only.  Header-only (the CUDA, HIP and SYCL builds use it alike).

#include "strata/core/conversation_cache.hpp"

#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace strata::core {

inline bool state_fp_on() {
    static const bool on = [] {
        const char* v = std::getenv("STRATA_CKPT_FP");
        return v != nullptr && v[0] == '1';
    }();
    return on;
}

inline constexpr uint64_t kStateFpBasis = 14695981039346656037ull;

inline uint64_t state_fp_bytes(uint64_t h, const uint8_t* p, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        h ^= p[i];
        h *= 1099511628211ull;
    }
    return h;
}

inline uint64_t state_fp_vec(const std::vector<uint8_t>& v) { return state_fp_bytes(kStateFpBasis, v.data(), v.size()); }

/// Folds one layer's K/V image (conversation_kv_save at >= `tokens`) into `h`: the whole pages below `tokens` of each
/// part, and the completed pooled index rows.  `cells` gets the cells hashed (the same for every layer).
inline uint64_t state_fp_kv(uint64_t h, const ConversationKv& kv, int64_t tokens, int64_t& cells) {
    cells = kv.page_size > 0 ? (tokens / kv.page_size) * kv.page_size : 0;
    if (kv.cells <= 0 || cells > kv.cells) cells = 0;
    auto part = [&](const ConversationBuffer& b, size_t n) {
        n = std::min(n, b.size());
        b.visit(0, n, [&](const uint8_t* p, size_t k, size_t) {
            h = state_fp_bytes(h, p, k);
            return true;
        });
    };
    for (const ConversationBuffer* b : {&kv.k, &kv.v, &kv.k_scale, &kv.v_scale})
        if (kv.cells > 0) part(*b, b->size() / (size_t) kv.cells * (size_t) cells);
    if (kv.pooled_rows > 1) part(kv.pooled, kv.pooled.size() / (size_t) kv.pooled_rows * (size_t) (kv.pooled_rows - 1));
    return h;
}

/// The line: `strata ckpt fp: <who> at <tokens>: gdn .. ple .. tails .. dead .. bpos .. kv .. (<cells> cells)`.
/// `kv` = kStateFpBasis with `cells` 0 when the K/V was not read.
inline void state_fp_print(const char* who, int64_t tokens, const ConversationCheckpoint& c, uint64_t kv, int64_t cells) {
    std::fprintf(stderr,
                 "strata ckpt fp: %s at %lld: gdn %016" PRIx64 " ple %016" PRIx64 " tails %016" PRIx64
                 " dead %016" PRIx64 " bpos %016" PRIx64 " kv %016" PRIx64 " (%lld cells)\n",
                 who, (long long) tokens, state_fp_vec(c.gdn), state_fp_vec(c.ple), state_fp_vec(c.tails),
                 state_fp_vec(c.dead), state_fp_vec(c.block_pos), kv, (long long) cells);
    std::fflush(stderr);
}

}  // namespace strata::core
