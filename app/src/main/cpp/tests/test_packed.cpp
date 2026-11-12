#include "core/gpt.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <random>
#include <string>
#include <vector>

using namespace mf;

namespace {

size_t expected_file_bytes(const ModelConfig& cfg, int bits, int group) {
    const size_t d = static_cast<size_t>(cfg.d_model);
    const size_t df = static_cast<size_t>(cfg.d_ff);
    const size_t gs = static_cast<size_t>(group);
    const size_t layers = static_cast<size_t>(cfg.layers);
    const size_t linear = cfg.linear_weight_count();
    const size_t non_linear = static_cast<size_t>(cfg.param_count()) - linear;
    const size_t packed = (linear * static_cast<size_t>(bits) + 7) / 8;
    const size_t per_layer_scales =
        4 * d * ((d + gs - 1) / gs) +
        df * ((d + gs - 1) / gs) +
        d * ((df + gs - 1) / gs);
    const size_t scales = layers * per_layer_scales;
    const size_t header = 11 * sizeof(int32_t) + 3 * sizeof(int64_t);
    return header + non_linear * sizeof(float) + scales * sizeof(float) + packed;
}

size_t file_size(const std::string& path) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) return 0;
    return static_cast<size_t>(in.tellg());
}

} // namespace

int main() {
    ModelConfig cfg;
    cfg.vocab = 64;
    cfg.context = 16;
    cfg.d_model = 64;
    cfg.layers = 2;
    cfg.heads = 4;
    cfg.d_ff = 128;
    cfg.tied_embeddings = true;

    GPT model;
    if (!model.init(cfg, 99)) {
        std::fprintf(stderr, "FAIL: model init\n");
        return 1;
    }

    const int T = 8;
    std::vector<int> tokens(T);
    std::mt19937 rng(5);
    std::uniform_int_distribution<int> dist(0, cfg.vocab - 1);
    for (int& t : tokens) t = dist(rng);

    for (int bits : {8, 4, 2}) {
        model.set_qat(bits, 32);
        model.forward(tokens.data(), nullptr, 1, T);
        const Tensor reference = model.cache().logits;

        const std::string path = "/tmp/modelforge_packed_" + std::to_string(bits) + ".mfpk";
        if (!model.save_packed(path, bits, 32)) {
            std::fprintf(stderr, "FAIL: save_packed %d-bit\n", bits);
            return 1;
        }
        PackedModel packed;
        if (!packed.load(path)) {
            std::remove(path.c_str());
            std::fprintf(stderr, "FAIL: packed load %d-bit\n", bits);
            return 1;
        }
        if (packed.bits() != bits || packed.group() != 32 ||
            packed.config().vocab != cfg.vocab || packed.config().layers != cfg.layers) {
            std::remove(path.c_str());
            std::fprintf(stderr, "FAIL: packed metadata mismatch %d-bit\n", bits);
            return 1;
        }

        const size_t actual_size = file_size(path);
        const size_t expected_size = expected_file_bytes(cfg, bits, 32);
        if (actual_size != expected_size) {
            std::remove(path.c_str());
            std::fprintf(stderr, "FAIL: %d-bit file is %zu bytes, expected %zu\n",
                         bits, actual_size, expected_size);
            return 1;
        }

        if (model.save_packed(path, 5, 32)) {
            std::remove(path.c_str());
            std::fprintf(stderr, "FAIL: unsupported bit width was accepted\n");
            return 1;
        }

        packed.forward(tokens.data(), 1, T);
        const Tensor& got = packed.logits();
        if (got.rows != reference.rows || got.cols != reference.cols) {
            std::remove(path.c_str());
            std::fprintf(stderr, "FAIL: packed logits shape mismatch\n");
            return 1;
        }

        float max_diff = 0.0f;
        int mismatched_argmax = 0;
        for (int r = 0; r < reference.rows; ++r) {
            const float* a = reference.row(r);
            const float* b = got.row(r);
            int ai = 0, bi = 0;
            for (int v = 0; v < reference.cols; ++v) {
                max_diff = std::max(max_diff, std::fabs(a[v] - b[v]));
                if (a[v] > a[ai]) ai = v;
                if (b[v] > b[bi]) bi = v;
            }
            if (ai != bi) ++mismatched_argmax;
        }

        std::printf("%d-bit packed: %zu bytes, max logit diff %.7f, argmax mismatches %d\n",
                    bits, actual_size, max_diff, mismatched_argmax);
        if (max_diff > 1e-2f || mismatched_argmax != 0) {
            std::remove(path.c_str());
            std::fprintf(stderr, "FAIL: packed inference differs from QAT reference\n");
            return 1;
        }

        const std::vector<int> prompt = {1, 2, 3, 4};
        if (bits == 8) {
            const auto reference_tokens = model.generate(prompt, 6, 0.0f, 0, 1.0f, 77);
            const auto packed_tokens = packed.generate(prompt, 6, 0.0f, 0, 1.0f, 77);
            if (reference_tokens != packed_tokens || packed_tokens.empty()) {
                std::remove(path.c_str());
                std::fprintf(stderr, "FAIL: greedy generation mismatch after packed load\n");
                return 1;
            }
        }
        std::remove(path.c_str());
    }

    std::printf("packed export + inference test passed\n");
    return 0;
}
