// include/strata/core/waitb_overlap.hpp - STRATA_WAITB_OVERLAP: the PCIe share's copy beside the VRAM experts.
//
// Plan: IMPL-PLAN-2026-11-26-main-engine2, A1 (section 2.3).  In a verify window's layer the GPU waits for flag B and
// stages the plan's PCIe experts (fetch_blobs, rebase_ptrs) only after the VRAM experts' compute has ended (the GPU
// profile's "waitB", stamp 20 -> 21).  With STRATA_WAITB_OVERLAP=1 that wait and copy run on a stream of their own,
// forked right after the plan is on the device (stamp 19) and joined before the PCIe groups' compute: the same
// kernels on the same inputs, only their order against the VRAM experts' kernels is freer.  The two branches touch
// disjoint memory (the VRAM branch reads the plan's VRAM half and writes hit_scratch / hit_out; the copy branch reads
// the plan's PCIe half and writes staging and the PCIe pointers), so the window's output is bit-identical.
// Unset: no stream, no event, nothing added to the window's graph.
//
// With STRATA_VERIFY_PROFILE=1 and STRATA_WAITB_SPLIT=1 the overlapped window reports what is left of the wait
// ("waitB overlap ..." in place of "waitB split ..."): two stamps on the copy's stream (after the flag-B wait, after
// the copy) and the profile's stamps 19..23.
//
// Header only, no CUDA: the rules are tested without a GPU (tests/core/waitb_overlap_test.cpp).
#pragma once

#include <cstdint>
#include <cstdio>
#include <string>

namespace strata::core::waitb_overlap {

/// STRATA_WAITB_OVERLAP's value: a value starting with '1' -> true; nullptr, "" or anything else -> false
inline bool on(const char* value) { return value != nullptr && value[0] == '1'; }

/// What a window's capture knows when it decides (Verifier::record_window).
struct Cond {
    bool flag = false;           ///< on(getenv("STRATA_WAITB_OVERLAP"))
    bool built = false;          ///< a CUDA build (not HIP)
    bool streams = false;        ///< the copy's stream and the fork / join events exist
    int pcie_mode = 0;           ///< GpuPlanSink::pcie_mode (2 = staged by the copy kernel)
    bool device_plan = false;    ///< E-6: the device plans resident-only groups (the waits can be skipped per group)
    bool pcie_balance = false;   ///< set_pcie_balance: the flag waits stamp the balance's clock on the window's stream
};

/// nullptr when the copy is overlapped; else why not (the first reason that applies; a static string).
inline const char* off_reason(const Cond& c) {
    if (!c.flag) return "STRATA_WAITB_OVERLAP is not 1";
    if (!c.built) return "not built for this backend (CUDA only)";
    if (!c.streams) return "the copy's stream or its events could not be created";
    if (c.pcie_mode != 2) return "--pcie-mode is not the staged copy (mode 2)";
    if (c.device_plan) return "STRATA_VERIFY_DEVICE_PLAN plans groups on the device";
    if (c.pcie_balance) return "the PCIe balance stamps the flag waits";
    return nullptr;
}

/// One layer of one overlapped window, token group 0 (GPU clock, ns; 0 = not stamped).  On the window's stream: the
/// profile's stamps 19 (the fork: the VRAM experts and the copy's branch start), 20 (the VRAM experts end), 21 (the
/// join: the copy's branch has ended too), 22 (the PCIe groups end), 23 (the wait for the CPU rows ends).  On the
/// copy's stream: after the flag-B wait, after the copy (fetch_blobs + rebase_ptrs).
struct Layer {
    unsigned long long fork = 0, hits1 = 0, join = 0, pcie1 = 0, cpu1 = 0;
    unsigned long long flagb = 0, copy1 = 0;
    int n_fetch = 0;             ///< the host's count of experts the plan sent over PCIe
    uint64_t blob_bytes = 0;     ///< one expert of this layer
};

struct Sums {
    double join_ns = 0, flag_ns = 0, copy_ns = 0, empty_copy_ns = 0, hits_ns = 0, cpu_wait_ns = 0, late_ns = 0;
    double fetch_bytes = 0;
    long long windows = 0, layers = 0, skipped = 0, fetches = 0;

    /// Adds one layer.  Skipped (++skipped) when a stamp is 0 or fork <= hits1 <= join <= pcie1 <= cpu1 or
    /// fork <= flagb <= copy1 fails.
    /// join = join - hits1 (what is left of waitB: the window's stream waiting for the copy's branch),
    /// flag = flagb - fork, copy = copy1 - flagb (to copy_ns when n_fetch > 0, else to empty_copy_ns),
    /// hits = hits1 - fork, cpu_wait = cpu1 - pcie1, late = max(0, copy1 - hits1) (the copy's branch past the VRAM
    /// experts' end; join >= late, the rest of join is the stamps' and the join's own launches).
    void add(const Layer& x) {
        const bool stamped = x.fork != 0 && x.hits1 != 0 && x.join != 0 && x.pcie1 != 0 && x.cpu1 != 0 && x.flagb != 0 &&
                             x.copy1 != 0;
        if (!stamped || !(x.fork <= x.hits1 && x.hits1 <= x.join && x.join <= x.pcie1 && x.pcie1 <= x.cpu1) ||
            !(x.fork <= x.flagb && x.flagb <= x.copy1)) {
            ++skipped;
            return;
        }
        const double copy = (double) (x.copy1 - x.flagb);
        ++layers;
        join_ns += (double) (x.join - x.hits1);
        flag_ns += (double) (x.flagb - x.fork);
        hits_ns += (double) (x.hits1 - x.fork);
        cpu_wait_ns += (double) (x.cpu1 - x.pcie1);
        if (x.copy1 > x.hits1) late_ns += (double) (x.copy1 - x.hits1);
        if (x.n_fetch > 0) {
            copy_ns += copy;
            fetch_bytes += (double) x.n_fetch * (double) x.blob_bytes;
            fetches += x.n_fetch;
        } else {
            empty_copy_ns += copy;
        }
    }
    void end_window() { ++windows; }

    /// "" when windows == 0, else
    /// " | waitB overlap over N windows (ms/window): join wait %.3f (copy past the VRAM hits %.3f), flag B %.3f,
    ///  copy %.3f (%.2f experts, %.1f GB/s; empty copies %.3f), VRAM hits %.3f, CPU wait %.3f; %lld layers skipped"
    /// (flag B and copy are on the copy's stream, beside the VRAM hits; GB/s = fetch_bytes / copy_ns of the layers
    /// with a fetch, "-" when there was none)
    std::string report() const {
        if (windows == 0) return "";
        const double w = (double) windows;
        char gbs[32];
        if (copy_ns > 0 && fetch_bytes > 0) std::snprintf(gbs, sizeof gbs, "%.1f", fetch_bytes / copy_ns);   // B/ns = GB/s
        else std::snprintf(gbs, sizeof gbs, "-");
        char b[512];
        std::snprintf(b, sizeof b,
                      " | waitB overlap over %lld windows (ms/window): join wait %.3f (copy past the VRAM hits %.3f), "
                      "flag B %.3f, copy %.3f (%.2f experts, %s GB/s; empty copies %.3f), VRAM hits %.3f, CPU wait %.3f; "
                      "%lld layers skipped",
                      windows, join_ns / 1e6 / w, late_ns / 1e6 / w, flag_ns / 1e6 / w, copy_ns / 1e6 / w,
                      (double) fetches / w, gbs, empty_copy_ns / 1e6 / w, hits_ns / 1e6 / w, cpu_wait_ns / 1e6 / w, skipped);
        return b;
    }
    void reset() { *this = Sums{}; }
};

}  // namespace strata::core::waitb_overlap
