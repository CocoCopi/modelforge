#include "core/gpt.h"

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

using namespace mf;

int main() {
    ModelConfig cfg;
    cfg.vocab = 16;
    cfg.context = 8;
    cfg.d_model = 16;
    cfg.layers = 1;
    cfg.heads = 2;
    cfg.d_ff = 32;
    cfg.tied_embeddings = true;

    GPT model;
    if (!model.init(cfg, 1234)) {
        std::fprintf(stderr, "FAIL: model init\n");
        return 1;
    }

    const int B = 1;
    const int T = 6;
    std::mt19937 rng(99);
    std::uniform_int_distribution<int> dist(0, cfg.vocab - 1);
    std::vector<int> tokens(B * T), targets(B * T);
    for (int i = 0; i < B * T; ++i) {
        tokens[i] = dist(rng);
        targets[i] = dist(rng);
    }

    model.zero_grads();
    const float loss = model.forward(tokens.data(), targets.data(), B, T);
    model.backward();

    struct Check {
        Param* p;
        int idx;
        float analytic;
        float maxRel;
    };
    std::vector<Check> checks;
    auto params = model.params();
    for (size_t pi = 0; pi < params.size(); ++pi) {
        Param* p = params[pi];
        if (p->w.empty()) continue;
        const int idx1 = static_cast<int>((pi * 7 + 3) % p->w.size());
        const int idx2 = static_cast<int>((pi * 13 + 1) % p->w.size());
        checks.push_back({p, idx1, p->g[idx1], 0.0f});
        if (idx2 != idx1) checks.push_back({p, idx2, p->g[idx2], 0.0f});
    }

    float worst = 0.0f;
    int worstParam = -1;
    const float eps = 5e-3f;
    for (auto& c : checks) {
        const float orig = c.p->w[c.idx];
        c.p->w[c.idx] = orig + eps;
        const float lp = model.forward(tokens.data(), targets.data(), B, T);
        c.p->w[c.idx] = orig - eps;
        const float lm = model.forward(tokens.data(), targets.data(), B, T);
        c.p->w[c.idx] = orig;

        const float numeric = (lp - lm) / (2.0f * eps);
        const float denom = std::max(1e-3f, std::fabs(numeric) + std::fabs(c.analytic));
        const float rel = std::fabs(numeric - c.analytic) / denom;
        c.maxRel = rel;
        if (rel > worst) {
            worst = rel;
            worstParam = static_cast<int>(&c - checks.data());
        }
        if (rel > 0.05f) {
            std::fprintf(stderr,
                         "FAIL: grad mismatch check=%d idx=%d analytic=%g numeric=%g rel=%g\n",
                         static_cast<int>(&c - checks.data()), c.idx,
                         c.analytic, numeric, rel);
        }
    }

    std::printf("loss=%f, checked %zu gradients, worst relative error=%.5f (check %d)\n",
                loss, checks.size(), worst, worstParam);
    if (worst > 0.05f) {
        std::fprintf(stderr, "FAIL: analytic gradients do not match numerical gradients\n");
        return 1;
    }
    std::printf("gradient check passed\n");
    return 0;
}
