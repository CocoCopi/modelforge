// Micro-benchmark for the forward projection kernel. Not a ctest: it prints
// timings so a speed claim can be checked instead of asserted.
#include "core/gpt.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

using mf::Tensor;

namespace {

// The pre-optimisation kernel: one dot product per (row, output) pair.
void matmul_reference(const Tensor& x, const float* w, int out, Tensor& y) {
    const int n = x.rows;
    const int in = x.cols;
    y.ensure(n, out);
    constexpr int kOutTile = 8;
    constexpr int kRowTile = 16;
    const int outTiles = (out + kOutTile - 1) / kOutTile;
    const int rowTiles = (n + kRowTile - 1) / kRowTile;
    for (int tile = 0; tile < outTiles * rowTiles; ++tile) {
        const int ob = tile / rowTiles;
        const int rb = tile % rowTiles;
        const int o0 = ob * kOutTile;
        const int o1 = std::min(out, o0 + kOutTile);
        const int r0 = rb * kRowTile;
        const int r1 = std::min(n, r0 + kRowTile);
        for (int o = o0; o < o1; ++o) {
            const float* wr = w + static_cast<size_t>(o) * in;
            for (int r = r0; r < r1; ++r) {
                const float* xr = x.row(r);
                float sum = 0.0f;
                for (int i = 0; i < in; ++i) sum += xr[i] * wr[i];
                y.row(r)[o] = sum;
            }
        }
    }
}

double time_ms(void (*fn)(const Tensor&, const float*, int, Tensor&),
               const Tensor& x, const float* w, int out, Tensor& y, int reps) {
    // Warm up once so the first-touch allocation is not timed.
    fn(x, w, out, y);
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < reps; ++i) fn(x, w, out, y);
    const auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count() / reps;
}

void run_case(int n, int out, int in) {
    Tensor x(n, in);
    Tensor w(out, in);
    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    for (auto& v : x.d) v = dist(rng);
    for (auto& v : w.d) v = dist(rng);

    Tensor a, b;
    matmul_reference(x, w.d.data(), out, a);
    mf::matmul_nt(x, w.d.data(), out, b);
    float worst = 0.0f;
    for (size_t i = 0; i < a.d.size(); ++i) {
        worst = std::max(worst, std::fabs(a.d[i] - b.d[i]));
    }

    const int reps = 12;
    const double oldMs = time_ms(matmul_reference, x, w.d.data(), out, a, reps);
    const double newMs = time_ms(mf::matmul_nt, x, w.d.data(), out, b, reps);
    const double flops = 2.0 * n * out * in;
    std::printf(
        "n=%3d out=%4d in=%4d | scalar %7.2f ms | tiled %7.2f ms | speedup %5.2fx"
        " | %.2f GFLOP/s | max diff %.3g\n",
        n, out, in, oldMs, newMs, oldMs / newMs, flops / (newMs * 1e6), worst
    );
}

} // namespace

int main() {
    run_case(1, 512, 256);
    run_case(8, 512, 256);
    run_case(32, 512, 256);
    run_case(64, 512, 256);
    run_case(64, 1024, 512);
    run_case(128, 2048, 512);
    return 0;
}
