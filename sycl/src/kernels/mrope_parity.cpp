// src/kernels/mrope_parity.cpp (SYCL build only) - the image path's M-RoPE table (strata/kernels/mrope.hpp) in every
// rotation a SYCL stage worker runs.
//
// A worker started with --vision rotates by the table its main process sends (StageMsg::Positions).  The table is
// read by cell index, and pair i of the 32 rotated pairs takes its position from stream i % 3 (t, h, w), as the
// CUDA/HIP kernels and ggml's rope_multi (imrope, sections 11/11/10/0) do.  The entry points checked here are every
// rotation of a QSA layer: the prompt path's rope (prefill/kernels), the window's native rope and fused norm+rope,
// the table-driven rope_neox_apply, and the indexer's pooled keys - the native single append, its window steps and
// the prompt path's batch, and the non-native indexer_key_append.  For each:
//
//   1. no table (null) against the identity table: BITWISE equal (the vision worker's text is the plain worker's);
//   2. an image table against a host rotation written from the definition (pair % 3), on the kernel's own unrotated
//      values (its output under the all-zero table, where every angle is 0 and the rotation is exact);
//   3. the same image table against three text tables, one per stream (t = h = w = that stream's position): every
//      pair must equal, BITWISE, the run of its own stream - so the table is read per pair and nothing else moves;
//   4. that the image table is observable at all: its output differs from the identity's, and the cells scored
//      have t, h, w pairwise different (the scrambled table: all of them).
// The native indexer's three appends (batch, single, steps) must also agree with each other bitwise under a table.
// Two tables: mtmd's layout (t = p, h = p + y, w = p + x, the text after a picture at p + max(nx, ny)) and a
// scrambled one.  STRATA_ROPE_TABLE=1 (ctest mrope_parity_rope_table) runs the same checks on the float64 angle table.
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "strata/kernels/mrope.hpp"
#include "strata/kernels/native_qsa_indexer.hpp"
#include "strata/kernels/native_rope.hpp"
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/rope.hpp"
#include "strata/kernels/rope_scaling.hpp"
#include "strata/prefill/kernels.hpp"

#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <vector>

namespace sk = strata::kernels;

namespace {

constexpr int kCells = 64;     // cells of the sequence (16 indexer blocks)
constexpr int kMaxPos = 384;   // rows of the cos/sin tables (above every position the tables below hold)
constexpr int kRot = 64, kHalf = 32;

sycl::queue& Q() { return dpct::get_in_order_queue(); }

template <class T> struct Dev {
    T* p = nullptr;
    size_t n = 0;
    explicit Dev(size_t count) : n(count) {
        p = sycl::malloc_device<T>(n, Q());
        if (p == nullptr) {
            std::fprintf(stderr, "mrope_parity: device allocation failed\n");
            std::exit(1);
        }
        zero();
    }
    ~Dev() { sycl::free(p, Q()); }
    Dev(const Dev&) = delete;
    Dev& operator=(const Dev&) = delete;
    void zero() { Q().memset(p, 0, n * sizeof(T)).wait(); }
    void put(const std::vector<T>& h) { Q().memcpy(p, h.data(), std::min(n, h.size()) * sizeof(T)).wait(); }
    std::vector<T> get() const {
        std::vector<T> h(n);
        Q().wait();
        Q().memcpy(h.data(), p, n * sizeof(T)).wait();
        return h;
    }
};

using Table = std::vector<int32_t>;   // [kCells][3] (t, h, w)

Table identity_table() {
    Table t((size_t) kCells * 3);
    for (int c = 0; c < kCells; ++c) t[c * 3] = t[c * 3 + 1] = t[c * 3 + 2] = c;
    return t;
}
// mtmd's positions: text, a 6 x 4 picture, text, a 3 x 5 picture, text
Table mtmd_table() {
    Table t((size_t) kCells * 3);
    int c = 0, p = 0;
    auto text = [&](int n) {
        for (int i = 0; i < n; ++i, ++c, ++p) t[c * 3] = t[c * 3 + 1] = t[c * 3 + 2] = p;
    };
    auto image = [&](int nx, int ny) {
        for (int y = 0; y < ny; ++y)
            for (int x = 0; x < nx; ++x, ++c) {
                t[c * 3] = p;
                t[c * 3 + 1] = p + y;
                t[c * 3 + 2] = p + x;
            }
        p += std::max(nx, ny);
    };
    text(6);
    image(6, 4);
    text(10);
    image(3, 5);
    text(kCells - c);
    return t;
}
// every cell's t < h < w, no two cells alike
Table scrambled_table() {
    Table t((size_t) kCells * 3);
    for (int c = 0; c < kCells; ++c) {
        t[c * 3] = (c * 7 + 3) % 211;
        t[c * 3 + 1] = t[c * 3] + 1 + (c * 13) % 50;
        t[c * 3 + 2] = t[c * 3 + 1] + 1 + (c * 29) % 60;
    }
    return t;
}
Table zero_table() { return Table((size_t) kCells * 3, 0); }
// text positions from one stream of `t`: t = h = w = that stream's
Table stream_table(const Table& t, int s) {
    Table o(t.size());
    for (int c = 0; c < kCells; ++c) o[c * 3] = o[c * 3 + 1] = o[c * 3 + 2] = t[c * 3 + s];
    return o;
}

// The rows of an entry point's output: each starts at `at` and is rotated (its first 64 floats, NEOX pairs i and
// i + 32) by cell `cell`'s positions; cell -1: not rotated by position (its values must not depend on the table).
struct Row {
    size_t at;
    int cell;
};
struct Case {
    std::string name;
    std::function<std::vector<float>(const int32_t* dev_table)> run;   // nullptr: no table
    std::vector<Row> rows;
};

int g_bad = 0;
void verdict(bool ok, const char* fmt, ...) __attribute__((format(printf, 2, 3)));
void verdict(bool ok, const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    std::printf("  %s  ", ok ? "PASS" : "FAIL");
    std::vprintf(fmt, ap);
    std::printf("\n");
    va_end(ap);
    if (!ok) ++g_bad;
}

bool same_bits(const std::vector<float>& a, const std::vector<float>& b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

void check_case(const Case& k, const char* table_name, const Table& tab, Dev<int32_t>& dtab, double freq_base) {
    auto run_with = [&](const Table& t) {
        dtab.put(t);
        return k.run(dtab.p);
    };
    const std::vector<float> none = k.run(nullptr);
    const std::vector<float> ident = run_with(identity_table());
    const std::vector<float> zero = run_with(zero_table());
    const std::vector<float> img = run_with(tab);
    const std::vector<float> img2 = run_with(tab);
    std::vector<float> per_stream[3];
    for (int s = 0; s < 3; ++s) per_stream[s] = run_with(stream_table(tab, s));

    bool finite = true;
    for (float v : img) finite = finite && std::isfinite(v);
    // 1. null and identity, bitwise
    if (std::strcmp(table_name, "mtmd") == 0)
        verdict(same_bits(none, ident), "%-34s no table == identity table, bitwise (%zu floats)", k.name.c_str(),
                none.size());
    // determinism (the bitwise checks below need it)
    verdict(same_bits(img, img2) && finite, "%-34s %s: the same table twice, bitwise; finite", k.name.c_str(),
            table_name);
    // 2. host rotation from the definition, 3. per-stream decomposition, 4. observability
    double worst = 0, ref_mag = 0, moved = 0;
    long long stream_bad = 0, fixed_bad = 0, distinct = 0, scored = 0;
    for (const Row& r : k.rows) {
        if (r.cell < 0) {
            for (int d = 0; d < kRot; ++d)
                if (std::memcmp(&img[r.at + d], &ident[r.at + d], 4) != 0) ++fixed_bad;
            continue;
        }
        ++scored;
        const int32_t* p = &tab[(size_t) r.cell * 3];
        if (p[0] != p[1] && p[1] != p[2] && p[0] != p[2]) ++distinct;
        for (int i = 0; i < kHalf; ++i) {
            const size_t lo = r.at + i, hi = r.at + i + kHalf;
            const double pos = p[i % 3];   // THE DEFINITION: pair i takes stream i % 3 (t, h, w)
            const double ang = pos * std::pow(freq_base, -2.0 * i / kRot);
            const double c = std::cos(ang), s = std::sin(ang);
            const double a = zero[lo], b = zero[hi];
            const double rlo = a * c - b * s, rhi = a * s + b * c;
            worst = std::max({worst, std::fabs(rlo - img[lo]), std::fabs(rhi - img[hi])});
            ref_mag = std::max({ref_mag, std::fabs(rlo), std::fabs(rhi)});
            moved = std::max({moved, std::fabs((double) img[lo] - ident[lo]), std::fabs((double) img[hi] - ident[hi])});
            const std::vector<float>& own = per_stream[i % 3];
            if (std::memcmp(&img[lo], &own[lo], 4) != 0 || std::memcmp(&img[hi], &own[hi], 4) != 0) ++stream_bad;
        }
    }
    verdict(worst <= 5e-3 * std::max(1.0, ref_mag), "%-34s %s: against the host rotation (pair %% 3): worst %.2e "
            "(largest value %.2f)", k.name.c_str(), table_name, worst, ref_mag);
    verdict(stream_bad == 0 && fixed_bad == 0, "%-34s %s: each pair == its own stream's text run, bitwise (%lld "
            "differ; unrotated rows %lld differ)", k.name.c_str(), table_name, stream_bad, fixed_bad);
    const bool all_distinct = std::strcmp(table_name, "scrambled") != 0 || distinct == scored;
    verdict(moved > 0.05 && distinct > 0 && all_distinct, "%-34s %s: the table is observable: it moves the output "
            "by up to %.2f; %lld of %lld cells scored have t, h, w pairwise different", k.name.c_str(), table_name,
            moved, distinct, scored);
}

}  // namespace

int main(int argc, char** argv) {
    for (int i = 1; i < argc; ++i)
        if (std::string(argv[i]) != "--selftest") {
            std::fprintf(stderr, "usage: mrope_parity [--selftest]\n");
            return 2;
        }
    const sk::RopeScaling sc = sk::rope_scaling();   // none: freq_base 1e7, the artifact's
    const bool rope_tab = [] { const char* v = std::getenv("STRATA_ROPE_TABLE"); return v && v[0] == '1'; }();
    std::printf("mrope_parity: %s, %s\n", Q().get_device().get_info<sycl::info::device::name>().c_str(),
                rope_tab ? "STRATA_ROPE_TABLE=1 (the float64 angle table)" : "the computed angles");
    void* stream = &Q();

    // the float64-built cos/sin tables: rope_neox_apply's and indexer_key_append's argument, and the
    // STRATA_ROPE_TABLE=1 registration the native and prompt-path kernels read
    std::vector<float> hcos((size_t) kMaxPos * kHalf), hsin((size_t) kMaxPos * kHalf);
    sk::build_rope_table(kRot, sc, kMaxPos, hcos.data(), hsin.data());
    Dev<float> dcos(hcos.size()), dsin(hsin.size());
    dcos.put(hcos);
    dsin.put(hsin);
    sk::rope_table_set(dcos.p, dsin.p, kMaxPos, sc);

    std::mt19937 rng(1234);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::uniform_real_distribution<float> ud(0.5f, 1.5f);
    auto randn = [&](size_t n) {
        std::vector<float> v(n);
        for (float& x : v) x = nd(rng);
        return v;
    };
    Dev<int32_t> dtab((size_t) kCells * 3);

    // ---- the rope kernels: kHeads rows per cell, cell = row / kHeads
    constexpr int kHeads = 3, kHD = 256;
    const int rows = kCells * kHeads;
    const std::vector<float> x = randn((size_t) rows * kHD), xq = randn((size_t) rows * 2 * kHD);
    std::vector<float> gamma(kHD);
    for (float& g : gamma) g = ud(rng);
    std::vector<int32_t> hpos(rows);
    for (int r = 0; r < rows; ++r) hpos[r] = r / kHeads;
    Dev<float> dx(x.size()), dxq(xq.size()), dout(x.size()), dgamma(gamma.size());
    Dev<int32_t> dpos(hpos.size());
    dx.put(x);
    dxq.put(xq);
    dgamma.put(gamma);
    dpos.put(hpos);
    std::vector<Row> head_rows;
    for (int r = 0; r < rows; ++r) head_rows.push_back({(size_t) r * kHD, r / kHeads});

    std::vector<Case> cases;
    cases.push_back({"rope_neox_apply (table)", [&](const int32_t* t) {
                         sk::mrope_table_set(t);
                         sk::rope_neox_apply(dx.p, dout.p, rows, kHD, kRot, dcos.p, dsin.p, dpos.p, stream);
                         return dout.get();
                     }, head_rows});
    cases.push_back({"native_rope_apply", [&](const int32_t* t) {
                         sk::mrope_table_set(t);
                         sk::native_rope_apply(dx.p, dout.p, rows, kHD, kRot, sc, dpos.p, stream);
                         return dout.get();
                     }, head_rows});
    cases.push_back({"native_rope_apply (in place)", [&](const int32_t* t) {
                         sk::mrope_table_set(t);
                         dout.put(x);
                         sk::native_rope_apply(dout.p, dout.p, rows, kHD, kRot, sc, dpos.p, stream);
                         return dout.get();
                     }, head_rows});
    cases.push_back({"native_qsa_rms_norm_rope (q|gate)", [&](const int32_t* t) {
                         sk::mrope_table_set(t);
                         sk::native_qsa_rms_norm_rope(dxq.p, 2 * kHD, dgamma.p, dout.p, rows, kHD, kRot, 1e-6f, sc,
                                                      dpos.p, stream);
                         return dout.get();
                     }, head_rows});
    cases.push_back({"native_qsa_rms_norm_rope (in place)", [&](const int32_t* t) {
                         sk::mrope_table_set(t);
                         dout.put(x);
                         sk::native_qsa_rms_norm_rope(dout.p, kHD, dgamma.p, dout.p, rows, kHD, kRot, 1e-6f, sc,
                                                      dpos.p, stream);
                         return dout.get();
                     }, head_rows});
    // the prompt path's rope: x [T][ld], heads of `dim` from offset 0, at cells pos0 + t
    constexpr int kP0 = 8, kPT = kCells - kP0, kPHeads = 2, kPLd = kPHeads * kHD + 64;   // (a gap after the heads)
    const std::vector<float> xp = randn((size_t) kPT * kPLd);
    Dev<float> dxp(xp.size());
    std::vector<Row> prefill_rows;
    for (int t = 0; t < kPT; ++t) {
        for (int h = 0; h < kPHeads; ++h) prefill_rows.push_back({(size_t) t * kPLd + (size_t) h * kHD, kP0 + t});
        prefill_rows.push_back({(size_t) t * kPLd + (size_t) kPHeads * kHD, -1});   // the gap: never touched
    }
    cases.push_back({"prefill rope (prompt path)", [&](const int32_t* t) {
                         sk::mrope_table_set(t);
                         dxp.put(xp);
                         strata::prefill::rope(dxp.p, kPT, kPHeads, kHD, kPLd, kP0, sc, stream);
                         Q().wait();
                         return dxp.get();
                     }, prefill_rows});

    // ---- the indexer's pooled keys: block b rotates at its first cell 4b; the spare row (n_bid) at 0
    const sk::QsaShapes S = sk::qsa_real_shapes();
    const int ID = (int) S.idx_dim, R = (int) S.idx_block, NB = kCells / R;
    const std::vector<float> raw = randn((size_t) kCells * ID);
    std::vector<float> ikn(ID);
    for (float& g : ikn) g = ud(rng);
    Dev<float> draw(raw.size()), dikn(ikn.size()), dtail((size_t) (R - 1) * ID), ddead(ID),
        dpooled((size_t) (NB + 1) * ID);
    Dev<int32_t> dbpos(1), dcell(kCells);
    draw.put(raw);
    dikn.put(ikn);
    std::vector<int32_t> hcell(kCells);
    for (int c = 0; c < kCells; ++c) hcell[c] = c;
    dcell.put(hcell);
    const sk::QsaIndexerBuffers ib{dtail.p, ddead.p, dpooled.p, dbpos.p};
    auto clear = [&] {
        dtail.zero();
        ddead.zero();
        dpooled.zero();
        dbpos.zero();
    };
    std::vector<Row> pooled_rows;
    for (int b = 0; b < NB; ++b) pooled_rows.push_back({(size_t) b * ID, b * R});
    pooled_rows.push_back({(size_t) NB * ID, -1});
    const float EPS = sk::qsa_rms_eps();
    cases.push_back({"indexer append_batch (prompt path)", [&](const int32_t* t) {
                         sk::mrope_table_set(t);
                         clear();
                         sk::native_qsa_indexer_append_batch(draw.p, 24, 0, 0, dikn.p, EPS, ib, S, kCells, sc, stream);
                         sk::native_qsa_indexer_append_batch(draw.p + 24 * ID, kCells - 24, 24, 0, dikn.p, EPS, ib, S,
                                                             kCells, sc, stream);
                         return dpooled.get();
                     }, pooled_rows});
    cases.push_back({"indexer append (one cell)", [&](const int32_t* t) {
                         sk::mrope_table_set(t);
                         clear();
                         for (int c = 0; c < kCells; ++c)
                             sk::native_qsa_indexer_append(draw.p + (size_t) c * ID, dcell.p + c, 0, dikn.p, EPS, ib,
                                                           S, kCells, sc, stream);
                         return dpooled.get();
                     }, pooled_rows});
    cases.push_back({"indexer append_steps (window)", [&](const int32_t* t) {
                         sk::mrope_table_set(t);
                         clear();
                         sk::native_qsa_indexer_append(draw.p, dcell.p, 0, dikn.p, EPS, ib, S, kCells, sc, stream);
                         for (int c = 1; c < kCells; c += 3) {
                             const int n = std::min(3, kCells - c);   // windows that straddle block ends
                             sk::native_qsa_indexer_append_steps(draw.p + (size_t) c * ID, dcell.p + c, 1, n, 0,
                                                                 dikn.p, EPS, ib, S, kCells, sc, stream);
                         }
                         return dpooled.get();
                     }, pooled_rows});
    cases.push_back({"indexer_key_append (not native)", [&](const int32_t* t) {
                         sk::mrope_table_set(t);
                         clear();
                         for (int c = 0; c < kCells; ++c)
                             sk::indexer_key_append(draw.p + (size_t) c * ID, dcell.p + c, 0, dikn.p, EPS, ib, S,
                                                    dcos.p, dsin.p, stream);
                         Q().wait();
                         return dpooled.get();
                     }, pooled_rows});

    const struct {
        const char* name;
        Table t;
    } tables[] = {{"mtmd", mtmd_table()}, {"scrambled", scrambled_table()}};
    for (const auto& tb : tables) {
        std::printf("table %s\n", tb.name);
        for (const Case& k : cases) check_case(k, tb.name, tb.t, dtab, sc.freq_base);
        // the native indexer's three appends agree under the table (the prompt path's batch, a window's steps)
        dtab.put(tb.t);
        const std::vector<float> a = cases[6].run(dtab.p), b = cases[7].run(dtab.p), c = cases[8].run(dtab.p);
        verdict(same_bits(a, b) && same_bits(a, c), "%-34s %s: batch == one cell == steps, bitwise",
                "indexer (native) appends", tb.name);
    }
    sk::mrope_table_set(nullptr);
    std::printf("mrope_parity: %s (%d failed)\n", g_bad == 0 ? "PASS" : "FAIL", g_bad);
    return g_bad == 0 ? 0 : 1;
}
