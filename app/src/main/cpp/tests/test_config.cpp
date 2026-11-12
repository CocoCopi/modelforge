#include "core/config.h"

#include <cstdio>
#include <cstdlib>

using namespace mf;

static int failures = 0;

static void check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++failures;
    } else {
        std::printf("ok  : %s\n", what);
    }
}

int main() {
    ModelConfig cfg;
    cfg.vocab = 1024;
    cfg.context = 128;
    cfg.d_model = 64;
    cfg.layers = 2;
    cfg.heads = 4;
    cfg.d_ff = 256;

    const int64_t expected = 1024LL * 64 + 128LL * 64 +
        2LL * (4LL * 64 * 64 + 2LL * 64 * 256 + 9LL * 64 + 256) + 2LL * 64;
    check(cfg.param_count() == expected, "param count matches Kotlin formula");
    check(cfg.valid(), "config is valid");
    check(cfg.head_dim() == 16, "head dim is 16");
    check(cfg.linear_weight_count() > 0, "linear weights counted");

    ModelConfig bad;
    bad.heads = 5;
    check(!bad.valid(), "invalid head split rejected");

    if (failures != 0) {
        std::fprintf(stderr, "%d native test failure(s)\n", failures);
        return 1;
    }
    std::printf("all native config tests passed\n");
    return 0;
}
