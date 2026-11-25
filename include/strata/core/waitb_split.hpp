// include/strata/core/waitb_split.hpp - STRATA_WAITB_SPLIT: the verify window's waitB split in two, per layer.
//
// Measurement only (plan: IMPL-PLAN-2026-11-26-main-engine2, M2).  The GPU profile's "waitB" (stamp 20 -> 21) is the
// spin on flag B ("the plan's PCIe share is ready") plus the copy kernels that stage the PCIe experts (fetch_blobs,
// rebase_ptrs).  With STRATA_VERIFY_PROFILE=1 and STRATA_WAITB_SPLIT=1 one more stamp sits right after the flag-B
// wait, so the two parts are told apart, and for each layer the time that overlapping the copy with the VRAM experts'
// compute could save at most is added up ("overlap bound").  Unset: nothing is allocated or launched.
//
// Header only, no CUDA: the rules are tested without a GPU (tests/core/waitb_split_test.cpp).
#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <string>

namespace strata::core::waitb_split {

/// STRATA_WAITB_SPLIT's value: a value starting with '1' -> true; nullptr, "" or anything else -> false
inline bool on(const char* value) { return value != nullptr && value[0] == '1'; }

/// One layer of one window, token group 0 (GPU clock, ns; 0 = not stamped): the profile's stamps 19 (the VRAM
/// experts start), 20 (they end), the new stamp after the flag-B wait, 21 (the PCIe share is staged), 22 (the PCIe
/// groups end), 23 (the wait for the CPU rows ends); the host's count of experts the plan sent over PCIe.
struct Layer {
    unsigned long long hits0 = 0, hits1 = 0, flagb = 0, copy1 = 0, pcie1 = 0, cpu1 = 0;
    int n_fetch = 0;
    uint64_t blob_bytes = 0;   ///< one expert of this layer (expert_layout().blob_bytes(l))
};

struct Sums {
    double flag_ns = 0, copy_ns = 0, empty_copy_ns = 0, hits_ns = 0, cpu_wait_ns = 0, bound_ns = 0;
    double fetch_bytes = 0;
    long long windows = 0, layers = 0, skipped = 0, fetches = 0;

    /// Adds one layer.  Skipped (++skipped) when a stamp is 0 or hits0 <= hits1 <= flagb <= copy1 <= pcie1 <= cpu1 fails.
    /// flag = flagb - hits1, copy = copy1 - flagb, hits = hits1 - hits0, cpu_wait = cpu1 - pcie1.
    /// copy goes to copy_ns when n_fetch > 0 (fetch_bytes += n_fetch * blob_bytes), else to empty_copy_ns.
    /// bound += max(0, s - cpu_wait), s = min(copy, hits) when flag <= flag_free_ns, else 0
    /// (the copy started with the VRAM experts, flag B already up: an upper bound of what the overlap saves).
    void add(const Layer& x, double flag_free_ns = 10000.0) {
        const bool stamped = x.hits0 != 0 && x.hits1 != 0 && x.flagb != 0 && x.copy1 != 0 && x.pcie1 != 0 && x.cpu1 != 0;
        if (!stamped || !(x.hits0 <= x.hits1 && x.hits1 <= x.flagb && x.flagb <= x.copy1 && x.copy1 <= x.pcie1 &&
                          x.pcie1 <= x.cpu1)) {
            ++skipped;
            return;
        }
        const double flag = (double) (x.flagb - x.hits1);
        const double copy = (double) (x.copy1 - x.flagb);
        const double hits = (double) (x.hits1 - x.hits0);
        const double cpu_wait = (double) (x.cpu1 - x.pcie1);
        ++layers;
        flag_ns += flag;
        hits_ns += hits;
        cpu_wait_ns += cpu_wait;
        if (x.n_fetch > 0) {
            copy_ns += copy;
            fetch_bytes += (double) x.n_fetch * (double) x.blob_bytes;
            fetches += x.n_fetch;
        } else {
            empty_copy_ns += copy;
        }
        const double s = flag <= flag_free_ns ? std::min(copy, hits) : 0.0;
        bound_ns += std::max(0.0, s - cpu_wait);
    }

    void end_window() { ++windows; }

    /// "" when windows == 0, else
    /// " | waitB split over N windows (ms/window): flag B %.3f, copy %.3f (%.2f experts, %.1f GB/s; empty copies %.3f),
    ///  VRAM hits %.3f, CPU wait %.3f, overlap bound %.3f; %lld layers skipped"
    /// (GB/s = fetch_bytes / copy_ns of the layers with a fetch; "-" when there was none)
    std::string report() const {
        if (windows == 0) return std::string();
        const double w = (double) windows;
        char gbs[32];
        if (copy_ns > 0 && fetches > 0) std::snprintf(gbs, sizeof gbs, "%.1f", fetch_bytes / copy_ns);   // bytes/ns = GB/s
        else std::snprintf(gbs, sizeof gbs, "-");
        char b[512];
        std::snprintf(b, sizeof b,
                      " | waitB split over %lld windows (ms/window): flag B %.3f, copy %.3f (%.2f experts, %s GB/s; "
                      "empty copies %.3f), VRAM hits %.3f, CPU wait %.3f, overlap bound %.3f; %lld layers skipped",
                      windows, flag_ns / 1e6 / w, copy_ns / 1e6 / w, (double) fetches / w, gbs, empty_copy_ns / 1e6 / w,
                      hits_ns / 1e6 / w, cpu_wait_ns / 1e6 / w, bound_ns / 1e6 / w, skipped);
        return std::string(b);
    }

    void reset() { *this = Sums{}; }
};

}  // namespace strata::core::waitb_split
