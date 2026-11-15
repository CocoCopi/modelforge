#include "core/config.h"
#include "core/gpt.h"

#include <cmath>
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

    const int64_t expected = 1024LL * 64 +
        2LL * (4LL * 64 * 64 + 2LL * 64 * 256 + 9LL * 64 + 256) + 2LL * 64;
    check(cfg.param_count() == expected, "param count matches Kotlin formula");
    check(cfg.valid(), "config is valid");
    check(cfg.head_dim() == 16, "head dim is 16");
    check(cfg.linear_weight_count() > 0, "linear weights counted");

    ModelConfig longCfg = cfg;
    longCfg.context = 1'000'000;
    check(longCfg.param_count() == cfg.param_count(),
          "context length adds no trainable parameters");

    GPT model;
    check(model.init(longCfg, 1), "million-token context initializes");
    int tokens[4] = {1, 2, 3, 4};
    int targets[4] = {2, 3, 4, 5};
    const float loss = model.forward(tokens, targets, 1, 4);
    check(std::isfinite(loss) && loss > 0.0f,
          "forward pass works with a million-token capacity");

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
