// src/core/logpos_ids_test.cpp - include/strata/core/logpos_ids.hpp on a baseline dump written here (CPU only).
// Cases: ~/strata-mesh/IMPL-PLAN-stage-a.md, "STRATA_LOGPOS_IDS" (5.3).
#include "strata/core/logpos_ids.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

namespace lp = strata::core::logpos;

static int g_fail = 0;
#define CHECK(cond)                                                                       \
    do {                                                                                  \
        if (!(cond)) {                                                                    \
            std::fprintf(stderr, "logpos_ids_test: FAILED %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            ++g_fail;                                                                     \
        }                                                                                 \
    } while (0)

// one baseline row: pos, target, lp_target, top, lp_top, hit, lp_extra, lp_target_without_extra, then the tail
static std::string row(int pos, const std::string& tail) {
    return std::to_string(pos) + "\t7\t-1.000000000\t7\t-0.500000000\t1\tnan\tnan" + tail + "\n";
}

// a path in the temp directory that no other run or call uses (time + counter; no POSIX-only calls)
static std::string temp_path() {
    static int seq = 0;
    const auto t = std::chrono::steady_clock::now().time_since_epoch().count();
    return (std::filesystem::temp_directory_path() /
            ("strata_logpos_ids_test_" + std::to_string((long long) t) + "_" + std::to_string(seq++)))
        .string();
}

static std::string write_file(const std::string& text) {
    const std::string name = temp_path();
    std::FILE* f = std::fopen(name.c_str(), "wb");
    CHECK(f != nullptr);
    if (f == nullptr) return name;
    std::fputs(text.c_str(), f);
    std::fclose(f);
    return name;
}

// write_ids_with into a tmpfile, read it back
static std::string written(lp::BaselineIds& b, int64_t pos, const std::vector<float>& logits, double lse) {
    const std::string name = temp_path();
    std::FILE* f = std::fopen(name.c_str(), "w+b");
    CHECK(f != nullptr);
    if (f == nullptr) return "";
    CHECK(lp::write_ids_with(b, f, pos, logits.data(), (int64_t) logits.size(), lse));
    std::fflush(f);
    std::rewind(f);
    std::string s;
    char buf[256];
    size_t n;
    while ((n = std::fread(buf, 1, sizeof buf, f)) > 0) s.append(buf, n);
    std::fclose(f);
    std::remove(name.c_str());
    return s;
}

int main() {
    // text 0: pos 0..3, text 1: pos 0..2
    const std::string base = write_file(row(0, "\t5:-0.1\t6:-2.5\t1:-3") + row(1, "\t2:-0.2") + row(2, "\t3:-0.3\t4:-0.4") +
                                        row(3, "") + row(0, "\t9:-0.9") + row(1, "\t8:-0.8\t7:-0.7") + row(2, "\t1:-1"));
    std::string err;
    lp::BaselineIds b;
    // 1: counts
    CHECK(b.load(base, err));
    CHECK(b.texts() == 2);
    CHECK(b.rows() == 7);

    // 2: next_row in writing order; the second text starts at the pos that goes back
    const std::vector<int32_t>* r = b.next_row(0);
    CHECK(r != nullptr && *r == (std::vector<int32_t>{5, 6, 1}));
    r = b.next_row(1);
    CHECK(r != nullptr && *r == (std::vector<int32_t>{2}));
    r = b.next_row(2);
    CHECK(r != nullptr && *r == (std::vector<int32_t>{3, 4}));
    r = b.next_row(3);
    CHECK(r != nullptr && r->empty());
    r = b.next_row(0);  // text 1
    CHECK(r != nullptr && *r == (std::vector<int32_t>{9}));
    r = b.next_row(1);
    CHECK(r != nullptr && *r == (std::vector<int32_t>{8, 7}));
    r = b.next_row(2);
    CHECK(r != nullptr && *r == (std::vector<int32_t>{1}));

    // 3: a pos the baseline lacks (and a third text) -> nullptr, and only "\t|" is written
    CHECK(b.next_row(99) == nullptr);  // pos 99 is above 2: still text 1, no such row
    CHECK(b.next_row(0) == nullptr);   // text 2: the baseline has only two
    {
        lp::BaselineIds c;
        CHECK(c.load(base, err));
        const std::vector<float> logits(10, 0.0f);
        CHECK(written(c, 50, logits, 0.0) == "\t|");
    }

    // 4: columns after a '|' are not read (a baseline that itself ran with STRATA_LOGPOS_IDS)
    {
        const std::string p = write_file(row(0, "\t5:-0.1\t6:-2.5\t|\t5:-0.2\t6:-2.6") + row(1, "\t|\t3:-0.3"));
        lp::BaselineIds c;
        CHECK(c.load(p, err));
        CHECK(c.rows() == 2 && c.texts() == 1);
        const std::vector<int32_t>* a = c.next_row(0);
        CHECK(a != nullptr && *a == (std::vector<int32_t>{5, 6}));
        a = c.next_row(1);
        CHECK(a != nullptr && a->empty());
        std::remove(p.c_str());
    }

    // 5: malformed rows fail with the line number
    {
        const std::string p = write_file(row(0, "\t5:-0.1") + row(1, "\t12:abc"));
        lp::BaselineIds c;
        err.clear();
        CHECK(!c.load(p, err));
        CHECK(err.find(":2:") != std::string::npos);
        std::remove(p.c_str());
        const std::string q = write_file(row(0, "") + "1\t7\t-1.0\t7\n");
        err.clear();
        CHECK(!c.load(q, err));
        CHECK(err.find(":2:") != std::string::npos);
        std::remove(q.c_str());
        const std::string s = write_file(row(0, "\t5-0.1"));
        err.clear();
        CHECK(!c.load(s, err));
        CHECK(err.find(":1:") != std::string::npos);
        std::remove(s.c_str());
        err.clear();
        CHECK(!c.load("/nonexistent/logpos_ids_test", err));
        CHECK(!err.empty());
    }

    // 6: the written columns are row[id] - lse at %.6f; an id outside the vocabulary is nan
    {
        const std::string p = write_file(row(0, "\t5:-1\t3:-1\t99:-1\t-4:-1"));
        lp::BaselineIds c;
        CHECK(c.load(p, err));
        std::vector<float> logits(10, 0.0f);
        logits[5] = 1.5f;
        logits[3] = -2.25f;
        const double lse = 3.0;
        char want[128];
        std::snprintf(want, sizeof want, "\t|\t5:%.6f\t3:%.6f\t99:nan\t-4:nan", (double) logits[5] - lse,
                      (double) logits[3] - lse);
        const std::string got = written(c, 0, logits, lse);
        CHECK(got == want);
        CHECK(got == "\t|\t5:-1.500000\t3:-5.250000\t99:nan\t-4:nan");
        std::remove(p.c_str());
    }
    std::remove(base.c_str());

    // 7, 8: the environment variables are not set here
    if (std::getenv("STRATA_LOGPOS_IDS") == nullptr && std::getenv("STRATA_LOGPOS_SPLIT_AT") == nullptr) {
        const std::string name = temp_path();
        std::FILE* f = std::fopen(name.c_str(), "w+b");
        CHECK(f != nullptr);
        err.clear();
        if (f != nullptr) {
            CHECK(lp::write_ids(f, 0, nullptr, 0, 0.0, err));
            CHECK(std::ftell(f) == 0);
            std::fclose(f);
            std::remove(name.c_str());
        }
        CHECK(lp::split_at() == -1);
    } else {
        std::printf("logpos_ids_test: STRATA_LOGPOS_IDS / STRATA_LOGPOS_SPLIT_AT is set; cases 7 and 8 skipped\n");
    }

    if (g_fail != 0) {
        std::fprintf(stderr, "logpos_ids_test: %d check(s) failed\n", g_fail);
        return 1;
    }
    std::printf("logpos_ids_test: all passed\n");
    return 0;
}
