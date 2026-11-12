#include "core/gpt.h"

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

using namespace mf;

int main() {
    ModelConfig cfg;
    cfg.vocab = 32;
    cfg.context = 16;
    cfg.d_model = 64;
    cfg.layers = 2;
    cfg.heads = 4;
    cfg.d_ff = 256;
    cfg.tied_embeddings = true;

    GPT model;
    if (!model.init(cfg, 7)) {
        std::fprintf(stderr, "FAIL: model init\n");
        return 1;
    }

    const int B = 2;
    const int T = 16;
    std::mt19937 rng(2024);
    std::uniform_int_distribution<int> dist(0, cfg.vocab - 1);
    std::vector<int> tokens(B * T), targets(B * T);
    for (int i = 0; i < B * T; ++i) {
        tokens[i] = dist(rng);
        targets[i] = dist(rng);
    }

    const int steps = 600;
    const float lr = 0.003f;
    float initial = 0.0f;
    float loss = 0.0f;
    for (int step = 1; step <= steps; ++step) {
        loss = model.train_step(tokens.data(), targets.data(), B, T, lr, 0.0f);
        if (step == 1) initial = loss;
        if (step % 100 == 0 || step == 1) {
            std::printf("step %d loss %.6f\n", step, loss);
        }
    }

    const bool dropped = loss < 0.1f && loss < initial * 0.2f;
    std::printf("initial loss %.4f -> final loss %.6f after %d steps\n", initial, loss, steps);
    if (!dropped) {
        std::fprintf(stderr, "FAIL: single-batch overfit did not reduce the loss enough\n");
        return 1;
    }

    // Checkpoint round trip (weights only) must reproduce greedy generation.
    const std::string path = "/tmp/modelforge_overfit_test.mfck";
    if (!model.save_checkpoint(path, false)) {
        std::fprintf(stderr, "FAIL: checkpoint save\n");
        return 1;
    }
    GPT loaded;
    if (!loaded.load_checkpoint(path, false)) {
        std::remove(path.c_str());
        std::fprintf(stderr, "FAIL: checkpoint load\n");
        return 1;
    }
    std::remove(path.c_str());

    std::vector<int> prompt = {1, 2, 3};
    const auto a = model.generate(prompt, 8, 0.0f, 0, 1.0f, 123);
    const auto b = loaded.generate(prompt, 8, 0.0f, 0, 1.0f, 123);
    if (a != b || a.empty()) {
        std::fprintf(stderr, "FAIL: loaded checkpoint does not reproduce generation\n");
        return 1;
    }
    std::printf("generated tokens:");
    for (int t : a) std::printf(" %d", t);
    std::printf("\noverfit + checkpoint test passed\n");
    return 0;
}
