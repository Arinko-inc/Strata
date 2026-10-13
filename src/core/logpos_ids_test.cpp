// src/core/logpos_ids_test.cpp - include/strata/core/logpos_ids.hpp on a baseline dump written here (CPU only).
// SKELETON (strata-mesh stage A, 担当 C): the cases are listed in ~/strata-mesh/IMPL-PLAN-stage-a.md.
#include "strata/core/logpos_ids.hpp"

#include <cstdio>

int main() {
    std::string err;
    // unset: nothing is written and it succeeds
    if (!strata::core::logpos::write_ids(stdout, 0, nullptr, 0, 0.0, err)) {
        std::fprintf(stderr, "logpos_ids_test: write_ids failed with STRATA_LOGPOS_IDS unset: %s\n", err.c_str());
        return 1;
    }
    std::printf("logpos_ids_test: skeleton only\n");
    return 0;
}
