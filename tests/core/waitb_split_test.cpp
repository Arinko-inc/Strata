// STRATA_WAITB_SPLIT (M2): the flag's reading, the per-layer sums and the overlap bound.  No GPU, no model.
#include <cmath>
#include <cstdio>
#include <string>

#include "strata/core/waitb_split.hpp"

using namespace strata::core::waitb_split;

static int failed = 0;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); ++failed; } } while (0)
static bool near(double a, double b, double eps = 1e-6) { return std::fabs(a - b) <= eps; }

// a layer on the clock: the VRAM experts 100 us, then flag B after `flag` us, the copy `copy` us, the PCIe groups
// 40 us, the CPU wait `cpu_wait` us; times in us -> ns
static Layer layer(double flag, double copy, double hits, double cpu_wait, int n_fetch = 1) {
    Layer x;
    x.hits0 = 1000000;
    x.hits1 = x.hits0 + (unsigned long long) (hits * 1000);
    x.flagb = x.hits1 + (unsigned long long) (flag * 1000);
    x.copy1 = x.flagb + (unsigned long long) (copy * 1000);
    x.pcie1 = x.copy1 + 40000;
    x.cpu1 = x.pcie1 + (unsigned long long) (cpu_wait * 1000);
    x.n_fetch = n_fetch;
    x.blob_bytes = 3000000;
    return x;
}

int main() {
    // 1. the flag's reading
    CHECK(!on(nullptr));
    CHECK(!on(""));
    CHECK(!on("0"));
    CHECK(!on("x"));
    CHECK(on("1"));
    CHECK(on("1abc"));

    // 2. the overlap bound
    {
        Sums s;
        s.add(layer(0, 50, 100, 0));
        CHECK(s.skipped == 0 && s.layers == 1);
        CHECK(near(s.bound_ns, 50000));
    }
    {
        Sums s;
        s.add(layer(0, 50, 100, 80));   // the CPU rows come later than the copy: the saving is absorbed
        CHECK(near(s.bound_ns, 0));
    }
    {
        Sums s;
        s.add(layer(0, 50, 100, 30));
        CHECK(near(s.bound_ns, 20000));
    }
    {
        Sums s;
        s.add(layer(5, 50, 100, 0));    // flag B up within the launch cost: the copy can start with the VRAM experts
        CHECK(near(s.bound_ns, 50000));
        CHECK(near(s.flag_ns, 5000));
    }
    {
        Sums s;
        s.add(layer(20, 50, 100, 0));   // flag B late: the copy cannot start early
        CHECK(near(s.bound_ns, 0));
        CHECK(near(s.flag_ns, 20000));
    }
    {
        Sums s;   // min(copy, hits): a copy longer than the VRAM experts
        s.add(layer(0, 150, 100, 0));
        CHECK(near(s.bound_ns, 100000));
    }

    // 3. skipped layers
    {
        Sums s;
        Layer a = layer(0, 50, 100, 0);
        a.pcie1 = 0;                     // a stamp is 0
        s.add(a);
        Layer b = layer(0, 50, 100, 0);
        b.flagb = b.hits1 - 1;           // flag B before the VRAM experts ended: the order is reversed
        s.add(b);
        Layer c = layer(0, 50, 100, 0);
        c.cpu1 = c.pcie1 - 1;
        s.add(c);
        CHECK(s.skipped == 3 && s.layers == 0);
        CHECK(near(s.flag_ns + s.copy_ns + s.empty_copy_ns + s.hits_ns + s.cpu_wait_ns + s.bound_ns, 0));
    }

    // 4. a layer without a fetch goes to the empty copies
    {
        Sums s;
        s.add(layer(0, 8, 100, 0, 0));
        CHECK(near(s.empty_copy_ns, 8000));
        CHECK(near(s.copy_ns, 0));
        CHECK(s.fetches == 0 && near(s.fetch_bytes, 0));
        CHECK(near(s.bound_ns, 8000));   // the empty copy is overlapped too
    }

    // 5. GB/s: 3 experts x 2.97 MB in 0.2 ms -> 44.55 GB/s (printed with one decimal: a binary 44.55 shows as 44.5)
    {
        Sums s;
        Layer x = layer(0, 200, 100, 0, 3);
        x.blob_bytes = 2970000;
        s.add(x);
        s.end_window();
        CHECK(s.fetches == 3);
        CHECK(near(s.fetch_bytes / s.copy_ns, 44.55));
        const std::string r = s.report();
        CHECK(r.find("44.5 GB/s") != std::string::npos || r.find("44.6 GB/s") != std::string::npos);
        CHECK(r.find("3.00 experts") != std::string::npos);
        Sums t;   // 3 x 3.0 MB in 0.2 ms -> 45.0 exactly
        Layer y = layer(0, 200, 100, 0, 3);
        y.blob_bytes = 3000000;
        t.add(y);
        t.end_window();
        CHECK(t.report().find("45.0 GB/s") != std::string::npos);
    }

    // 6. report(): ms per window; "-" without a fetch; empty before the first window
    {
        Sums s;
        CHECK(s.report().empty());
        s.add(layer(2, 500, 1000, 100, 2));
        s.add(layer(2, 500, 1000, 100, 2));
        s.end_window();
        s.end_window();
        const std::string r = s.report();
        CHECK(r.find(" | waitB split over 2 windows (ms/window): flag B 0.002, copy 0.500") == 0);
        CHECK(r.find("VRAM hits 1.000, CPU wait 0.100") != std::string::npos);
        CHECK(r.find("0 layers skipped") != std::string::npos);
        Sums e;
        e.add(layer(0, 8, 100, 0, 0));
        e.end_window();
        CHECK(e.report().find("- GB/s") != std::string::npos);
    }

    // 7. reset()
    {
        Sums s;
        s.add(layer(0, 50, 100, 0));
        s.add(layer(0, 50, 100, 0));
        s.end_window();
        s.reset();
        CHECK(s.windows == 0 && s.layers == 0 && s.skipped == 0 && s.fetches == 0);
        CHECK(near(s.bound_ns, 0) && near(s.copy_ns, 0) && near(s.fetch_bytes, 0));
        CHECK(s.report().empty());
    }

    if (failed) {
        std::fprintf(stderr, "waitb_split_test: %d failed\n", failed);
        return 1;
    }
    std::printf("waitb_split_test: ok\n");
    return 0;
}
