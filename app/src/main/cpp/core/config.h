#pragma once

#include <cstdint>

namespace mf {

struct ModelConfig {
    int vocab = 1024;
    int context = 128;
    int d_model = 64;
    int layers = 2;
    int heads = 4;
    int d_ff = 256;
    bool tied_embeddings = true;

    int head_dim() const { return heads > 0 ? d_model / heads : 0; }
    // The engine has no separate output head: logits come from the tied
    // embedding matrix. Rejecting !tied_embeddings is deliberate so the
    // parameter count, checkpoint header and memory estimate cannot describe a
    // model that is never actually built.
    bool valid() const {
        return tied_embeddings && vocab > 0 && context > 0 && d_model > 0 &&
               layers > 0 && heads > 0 && d_ff > 0 && (d_model % heads) == 0;
    }

    // Exact same formula as ModelConfig.paramCount() in Kotlin.
    int64_t param_count() const {
        if (!valid()) return 0;
        const int64_t d = d_model;
        const int64_t df = d_ff;
        const int64_t v = vocab;
        const int64_t l = layers;
        const int64_t per_layer = 4 * d * d + 2 * d * df + 9 * d + df;
        // Positions are fixed sinusoidal values, so the context length adds no
        // trainable parameters and no context-sized position table.
        return v * d + l * per_layer + 2 * d;
    }

    int64_t linear_weight_count() const {
        const int64_t d = d_model;
        const int64_t df = d_ff;
        return int64_t(layers) * (4 * d * d + 2 * d * df);
    }
};

} // namespace mf
