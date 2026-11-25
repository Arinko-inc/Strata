// tools/relay_read/rows_cmp.cpp - two raw fp32 dumps of the same shape (stage_probe_bench --dump): how far B is from A.
//   rows_cmp A B [row floats, default 10240]
// Prints the values that differ in bits, the largest |a - b|, the largest |a - b| / max|a| of its row, the relative
// L2 error of the whole (|A - B| / |A|) and the worst row's.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static std::vector<float> rd(const char* p) {
    std::vector<float> v;
    FILE* f = std::fopen(p, "rb");
    if (!f) { std::fprintf(stderr, "rows_cmp: cannot open %s\n", p); std::exit(2); }
    std::fseek(f, 0, SEEK_END);
    const long n = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    v.resize((size_t) n / 4);
    if (std::fread(v.data(), 4, v.size(), f) != v.size()) { std::fprintf(stderr, "rows_cmp: short read\n"); std::exit(2); }
    std::fclose(f);
    return v;
}

int main(int argc, char** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: rows_cmp A B [row floats]\n"); return 2; }
    const std::vector<float> a = rd(argv[1]), b = rd(argv[2]);
    const size_t D = argc > 3 ? (size_t) std::atoll(argv[3]) : 10240;
    if (a.size() != b.size() || a.empty() || a.size() % D != 0) { std::fprintf(stderr, "rows_cmp: shapes differ\n"); return 2; }
    size_t bits = 0, nonfinite = 0;
    double max_abs = 0, max_rel_row = 0, num = 0, den = 0, worst_row = 0;
    for (size_t r = 0; r < a.size() / D; ++r) {
        double rmax = 0, rnum = 0, rden = 0, rdiff = 0;
        for (size_t d = 0; d < D; ++d) {
            const float x = a[r * D + d], y = b[r * D + d];
            if (std::memcmp(&x, &y, 4) != 0) ++bits;
            if (!std::isfinite(x) || !std::isfinite(y)) { ++nonfinite; continue; }
            const double e = std::fabs((double) x - (double) y);
            rmax = std::fmax(rmax, std::fabs((double) x));
            rdiff = std::fmax(rdiff, e);
            rnum += e * e;
            rden += (double) x * (double) x;
        }
        max_abs = std::fmax(max_abs, rdiff);
        if (rmax > 0) max_rel_row = std::fmax(max_rel_row, rdiff / rmax);
        if (rden > 0) worst_row = std::fmax(worst_row, std::sqrt(rnum / rden));
        num += rnum;
        den += rden;
    }
    std::printf("rows %zu x %zu: %zu values differ in bits (%.2f%%), %zu not finite; max |a-b| %.4g; max |a-b| / max|a| of its row %.4g; "
                "relative L2 %.4g (worst row %.4g)\n", a.size() / D, D, bits, 100.0 * (double) bits / (double) a.size(), nonfinite,
                max_abs, max_rel_row, den > 0 ? std::sqrt(num / den) : 0.0, worst_row);
    return 0;
}
