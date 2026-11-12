#pragma once

#include "core/config.h"

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace mf {

// ---------------------------------------------------------------------------
// Small dense tensor helpers
// ---------------------------------------------------------------------------

struct Tensor {
    int rows = 0;
    int cols = 0;
    std::vector<float> d;

    Tensor() = default;
    Tensor(int r, int c) { init(r, c); }

    void init(int r, int c) {
        rows = r;
        cols = c;
        d.assign(static_cast<size_t>(r) * static_cast<size_t>(c), 0.0f);
    }

    void ensure(int r, int c) {
        if (rows != r || cols != c) {
            init(r, c);
        } else {
            std::fill(d.begin(), d.end(), 0.0f);
        }
    }

    size_t size() const { return d.size(); }
    float* data() { return d.data(); }
    const float* data() const { return d.data(); }
    float* row(int r) { return d.data() + static_cast<size_t>(r) * cols; }
    const float* row(int r) const { return d.data() + static_cast<size_t>(r) * cols; }
};

// A single trainable tensor together with its gradient and AdamW moments.
struct Param {
    int rows = 0;
    int cols = 0;
    std::vector<float> w;
    std::vector<float> g;
    std::vector<float> m;
    std::vector<float> v;
    bool weight_decay = true;

    void init(int r, int c, bool decay = true) {
        rows = r;
        cols = c;
        const size_t n = static_cast<size_t>(r) * static_cast<size_t>(c);
        w.assign(n, 0.0f);
        g.assign(n, 0.0f);
        m.assign(n, 0.0f);
        v.assign(n, 0.0f);
        weight_decay = decay;
    }

    size_t size() const { return w.size(); }
    float* data() { return w.data(); }
    const float* data() const { return w.data(); }
    float* row(int r) { return w.data() + static_cast<size_t>(r) * cols; }
    const float* row(int r) const { return w.data() + static_cast<size_t>(r) * cols; }
};

struct Linear {
    Param weight;  // [out, in]
    Param bias;    // [1, out]
    int qat_bits = 0;
    int qat_group = 32;
    Tensor fake_weight;  // dequantized copy used by QAT forward

    int in() const { return weight.cols; }
    int out() const { return weight.rows; }
};

struct LayerNorm {
    Param gamma;  // [1, d]
    Param beta;   // [1, d]
};

struct Layer {
    LayerNorm ln1;
    LayerNorm ln2;
    Linear wq, wk, wv, wo;  // attention
    Linear w1, w2;          // MLP
};

// ---------------------------------------------------------------------------
// Caches
// ---------------------------------------------------------------------------

struct LayerCache {
    Tensor xin;       // [n, d]
    Tensor ln1;       // [n, d]
    Tensor q, k, v;   // [n, d]
    Tensor probs;     // [B*H*T*T]
    Tensor attn_out;  // [n, d]
    Tensor proj;      // [n, d]
    Tensor xmid;      // [n, d]
    Tensor ln2;       // [n, d]
    Tensor h1pre;     // [n, df]
    Tensor h1;        // [n, df]
    Tensor h2;        // [n, d]
    Tensor inv1;      // [n]
    Tensor inv2;      // [n]
};

struct ForwardCache {
    int B = 0;
    int T = 0;
    Tensor x0;         // [n, d]
    Tensor final_in;   // [n, d]
    Tensor lnf;        // [n, d]
    Tensor invf;       // [n]
    Tensor logits;     // [n, V]
    Tensor dlogits;    // [n, V], filled by the loss
    std::vector<int> tokens; // [n]
    std::vector<LayerCache> layers;
};

// ---------------------------------------------------------------------------
// GPT model
// ---------------------------------------------------------------------------

class GPT {
public:
    GPT() = default;

    bool init(const ModelConfig& cfg, uint64_t seed = 1337);

    const ModelConfig& config() const { return cfg_; }
    int64_t param_count() const { return cfg_.param_count(); }

    // Trains one step on a flat batch: tokens/targets have B*T elements.
    // Returns the measured mean cross-entropy loss for this batch.
    float train_step(const int* tokens, const int* targets, int B, int T,
                     float lr, float weight_decay = 0.01f);

    // Forward pass. If targets is null, no loss is computed and 0 is returned.
    float forward(const int* tokens, const int* targets, int B, int T);

    void backward();
    void zero_grads();
    void adam_update(float lr, float weight_decay = 0.01f);

    std::vector<int> generate(const std::vector<int>& prompt, int max_new,
                              float temperature, int top_k, float top_p,
                              uint64_t seed = 42);

    bool save_checkpoint(const std::string& path, bool with_optimizer) const;
    bool load_checkpoint(const std::string& path, bool with_optimizer);

    // Exports a deployment file with linear weights quantized to bits (2/3/4/8)
    // and everything else kept in fp32. PackedModel can load and run it.
    bool save_packed(const std::string& path, int bits, int group = 32) const;

    // QAT is applied to linear layers only. bits==0 disables it.
    void set_qat(int bits, int group = 32);
    int qat_bits() const { return qat_bits_; }
    int qat_group() const { return qat_group_; }

    // Used by the quantizer (Phase 5) and tests.
    std::vector<Linear*> linears();
    std::vector<const Linear*> linears() const;
    std::vector<Param*> params();

    long long step() const { return step_; }
    void set_step(long long s) { step_ = s; }

    ForwardCache& cache() { return cache_; }

private:
    void forward_logits(const int* tokens, int B, int T);
    void init_params();

    ModelConfig cfg_;
    Param wte_, wpe_, lnf_g_, lnf_b_;
    std::vector<Layer> layers_;
    int qat_bits_ = 0;
    int qat_group_ = 32;
    long long step_ = 0;
    ForwardCache cache_;

    friend struct PackedModel;
};

// ---------------------------------------------------------------------------
// Packed (quantized) inference model
// ---------------------------------------------------------------------------

struct PackedLinear {
    int out = 0;
    int in = 0;
    int bits = 0;
    int group = 32;
    std::vector<uint8_t> q;      // packed two's-complement values, row-major
    std::vector<float> scales;   // [out, groups]
    std::vector<float> bias;     // [out]

    int groups() const {
        if (group <= 0 || in <= 0) return 0;
        return (in + group - 1) / group;
    }
    size_t packed_bytes() const {
        if (bits <= 0 || out <= 0 || in <= 0) return 0;
        return (static_cast<size_t>(out) * in * bits + 7) / 8;
    }
};

struct PackedLayer {
    Tensor ln1_gamma, ln1_beta, ln2_gamma, ln2_beta;
    PackedLinear wq, wk, wv, wo, w1, w2;
};

class PackedModel {
public:
    // Loads a file written by GPT::save_packed.
    bool load(const std::string& path);

    const ModelConfig& config() const { return cfg_; }
    int bits() const { return bits_; }
    int group() const { return group_; }
    int64_t param_count() const { return cfg_.param_count(); }

    // Forward pass used by generation and tests. Fills logits().
    void forward(const int* tokens, int B, int T) const;
    const Tensor& logits() const { return cache_.logits; }

    std::vector<int> generate(const std::vector<int>& prompt, int max_new,
                              float temperature, int top_k, float top_p,
                              uint64_t seed = 42) const;

private:
    ModelConfig cfg_;
    int bits_ = 0;
    int group_ = 32;
    Tensor wte_, wpe_, lnf_g_, lnf_b_;
    std::vector<PackedLayer> layers_;
    mutable ForwardCache cache_;
};

// ---------------------------------------------------------------------------
// Utilities shared with tests and the quantizer
// ---------------------------------------------------------------------------

void matmul_nt(const Tensor& x, const float* w, int out, Tensor& y);
void matmul_nt(const Tensor& x, const Tensor& w, Tensor& y);
void gelu_forward(const Tensor& in, Tensor& out);
void layernorm_forward(const Tensor& x, const float* gamma, const float* beta,
                       Tensor& y, Tensor& inv, float eps = 1e-5f);

double cross_entropy_loss(const Tensor& logits, const int* targets, Tensor& dlogits);

} // namespace mf
