// STRATA_WAITB_OVERLAP (A1): the flag's reading, the capture's condition and the overlapped window's sums.  No GPU.
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>

#include "strata/core/waitb_overlap.hpp"

using namespace strata::core::waitb_overlap;

static int failed = 0;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); ++failed; } } while (0)
static bool near(double a, double b, double eps = 1e-6) { return std::fabs(a - b) <= eps; }
static bool has(const char* s, const char* part) { return s != nullptr && std::strstr(s, part) != nullptr; }

static Cond good() {
    Cond c;
    c.flag = true; c.built = true; c.streams = true; c.pcie_mode = 2;
    return c;
}

// a layer on the clock (us -> ns): the VRAM experts `hits` us from the fork; on the copy's stream flag B after `flag`
// us and the copy `copy` us; the join `extra` us after the later of the two branches; the PCIe groups 40 us; the CPU
// wait `cpu_wait` us
static Layer layer(double flag, double copy, double hits, double cpu_wait, int n_fetch = 1, double extra = 2) {
    Layer x;
    x.fork = 1000000;
    x.hits1 = x.fork + (unsigned long long) (hits * 1000);
    x.flagb = x.fork + (unsigned long long) (flag * 1000);
    x.copy1 = x.flagb + (unsigned long long) (copy * 1000);
    x.join = (x.hits1 > x.copy1 ? x.hits1 : x.copy1) + (unsigned long long) (extra * 1000);
    x.pcie1 = x.join + 40000;
    x.cpu1 = x.pcie1 + (unsigned long long) (cpu_wait * 1000);
    x.n_fetch = n_fetch;
    x.blob_bytes = 2970000;
    return x;
}

int main() {
    // 1. the flag's reading
    CHECK(!on(nullptr));
    CHECK(!on(""));
    CHECK(!on("0"));
    CHECK(!on("2"));
    CHECK(!on("on"));
    CHECK(on("1"));
    CHECK(on("1 "));

    // 2. the capture's condition: every reason alone, and their order
    CHECK(off_reason(good()) == nullptr);
    { Cond c = good(); c.flag = false; CHECK(has(off_reason(c), "not 1")); }
    { Cond c = good(); c.built = false; CHECK(has(off_reason(c), "CUDA only")); }
    { Cond c = good(); c.streams = false; CHECK(has(off_reason(c), "stream")); }
    for (int m : {0, 1, 3}) { Cond c = good(); c.pcie_mode = m; CHECK(has(off_reason(c), "mode 2")); }
    { Cond c = good(); c.device_plan = true; CHECK(has(off_reason(c), "DEVICE_PLAN")); }
    { Cond c = good(); c.pcie_balance = true; CHECK(has(off_reason(c), "balance")); }
    { Cond c; CHECK(has(off_reason(c), "not 1")); }                                    // the default: off by the flag
    { Cond c = good(); c.pcie_mode = 0; c.device_plan = true; CHECK(has(off_reason(c), "mode 2")); }

    // 3. the copy ends inside the VRAM experts: the join waits only for its own launches
    {
        Sums s;
        s.add(layer(3, 50, 100, 30));
        s.end_window();
        CHECK(s.layers == 1 && s.skipped == 0 && s.windows == 1);
        CHECK(near(s.join_ns, 2000));
        CHECK(near(s.late_ns, 0));
        CHECK(near(s.flag_ns, 3000));
        CHECK(near(s.copy_ns, 50000));
        CHECK(near(s.empty_copy_ns, 0));
        CHECK(near(s.hits_ns, 100000));
        CHECK(near(s.cpu_wait_ns, 30000));
        CHECK(s.fetches == 1 && near(s.fetch_bytes, 2970000));
    }
    // 4. the copy runs past the VRAM experts: the rest is the join wait
    {
        Sums s;
        s.add(layer(5, 150, 100, 0, 3));
        CHECK(near(s.late_ns, 55000));
        CHECK(near(s.join_ns, 57000));
        CHECK(s.fetches == 3);
    }
    // 5. a layer with no fetch: the copy's time is an empty copy
    {
        Sums s;
        s.add(layer(2, 8, 100, 10, 0));
        CHECK(near(s.copy_ns, 0) && near(s.empty_copy_ns, 8000));
        CHECK(s.fetches == 0 && near(s.fetch_bytes, 0));
    }
    // 6. skipped: a missing stamp, the window's stream out of order, the copy's stream out of order
    {
        Sums s;
        Layer x = layer(3, 50, 100, 30);
        x.copy1 = 0;
        s.add(x);
        x = layer(3, 50, 100, 30); x.flagb = 0; s.add(x);
        x = layer(3, 50, 100, 30); x.join = x.hits1 - 1; s.add(x);
        x = layer(3, 50, 100, 30); x.flagb = x.fork - 1; s.add(x);
        x = layer(3, 50, 100, 30); x.copy1 = x.flagb - 1; s.add(x);
        x = layer(3, 50, 100, 30); x.cpu1 = x.pcie1 - 1; s.add(x);
        CHECK(s.skipped == 6 && s.layers == 0);
        CHECK(near(s.join_ns + s.flag_ns + s.copy_ns + s.hits_ns + s.cpu_wait_ns, 0));
    }
    // 7. the report: ms/window, GB/s (3 experts x 2.97 MB in 0.198 ms = 45.0 GB/s), "-" without a fetch, "" when empty
    {
        Sums s;
        CHECK(s.report().empty());
        s.add(layer(6, 198, 100, 20, 3));
        s.add(layer(2, 8, 100, 10, 0));
        s.end_window();
        s.end_window();
        const std::string r = s.report();
        CHECK(r.find(" | waitB overlap over 2 windows (ms/window): ") == 0);
        CHECK(r.find("join wait 0.054 (copy past the VRAM hits 0.052)") != std::string::npos);   // (106 + 2) / 2, 104 / 2 us
        CHECK(r.find("flag B 0.004, copy 0.099 (1.50 experts, 45.0 GB/s; empty copies 0.004)") != std::string::npos);
        CHECK(r.find("VRAM hits 0.100, CPU wait 0.015; 0 layers skipped") != std::string::npos);
        Sums e;
        e.add(layer(2, 8, 100, 10, 0));
        e.end_window();
        CHECK(e.report().find("- GB/s") != std::string::npos);
        s.reset();
        CHECK(s.windows == 0 && s.layers == 0 && s.report().empty());
    }
    if (failed) std::fprintf(stderr, "%d checks failed\n", failed);
    else std::printf("waitb_overlap_test: ok\n");
    return failed;
}
