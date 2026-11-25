// tools/relay_read/mmq_wb_bench.cu - is the MMQ write-back's store pattern what a relay prompt chunk's expert
// products wait for?  (strata-mesh IMPL-PLAN-2026-11-26-relay-read.md: per MMQ block the time is ~0.008 us x J for
// gate/up and down alike, whatever K.)  One launch per "layer" with the down product's grid at a 3072-token chunk:
// 512 experts x 40 tiles of 64 outputs, 128 threads a block, J columns a block (rows 10 KB apart, as Dm's).
//   mode 0: each block stores its 64 x J sums as MMQ does: 64 floats in each of J rows of a [rows x N] matrix
//   mode 1: each block stores the same 64 x J floats in one piece
//   mode 2: mode 0 with N = 1280 and 20 tiles (the gate/up product's)
// Prints ms per launch (median of 12).  hipcc: clang++ -x hip --offload-arch=gfx1100 -O3 mmq_wb_bench.cu -lamdhip64
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

__global__ void wb(float* dst, int mode, int N, int J, int rows_per_expert, long rows) {
    const int it = blockIdx.x, z = blockIdx.z;
    const int tid = threadIdx.y * 32 + threadIdx.x;
    const long row0 = (long) z * rows_per_expert;
    const long blk = (long) z * gridDim.x + it;
    for (int n = 0; n < J / 2; ++n) {
        const int e = n * 128 + tid, i = e % 64, j = e / 64;
        const float v = (float) (e + it) * 0.5f;
        if (mode == 1) {
            dst[blk * 64 * J + e] = v;
        } else {
            const long r = row0 + j;
            if (r < rows) dst[r * N + it * 64 + i] = v;
        }
    }
}

int main(int argc, char** argv) {
    const int J = argc > 1 ? std::atoi(argv[1]) : 96, rpe = argc > 2 ? std::atoi(argv[2]) : 60;
    const long rows = 512L * rpe + 128;
    float* dst = nullptr;
    const size_t bytes = std::max((size_t) rows * 2560 * 4, (size_t) 512 * 40 * 64 * (size_t) J * 4);
    if (hipMalloc(&dst, bytes) != hipSuccess) { std::fprintf(stderr, "hipMalloc failed\n"); return 1; }
    hipMemset(dst, 0, bytes);
    hipEvent_t a, b;
    hipEventCreate(&a);
    hipEventCreate(&b);
    for (int mode = 0; mode < 3; ++mode) {
        const int N = mode == 2 ? 1280 : 2560, nty = N / 64;
        std::vector<float> ms;
        for (int rep = 0; rep < 14; ++rep) {
            hipEventRecord(a, 0);
            wb<<<dim3(nty, 1, 512), dim3(32, 4, 1), 0, 0>>>(dst, mode, N, J, rpe, rows);
            hipEventRecord(b, 0);
            hipEventSynchronize(b);
            float t = 0;
            hipEventElapsedTime(&t, a, b);
            if (rep >= 2) ms.push_back(t);
        }
        std::sort(ms.begin(), ms.end());
        std::printf("mode %d (N %d, %d blocks, J %d): %.3f ms a launch (min %.3f)\n", mode, N, nty * 512, J, ms[ms.size() / 2], ms[0]);
    }
    hipFree(dst);
    return 0;
}
