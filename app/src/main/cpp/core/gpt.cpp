#include "core/gpt.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <fstream>
#include <functional>
#include <mutex>
#include <random>
#include <thread>

#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace mf {

namespace {

constexpr float kLnEps = 1e-5f;
constexpr float kGeluC = 0.7978845608028654f;  // sqrt(2/pi)
constexpr float kGeluK = 0.044715f;

int hardware_threads() {
    static int cached = [] {
        int hc = static_cast<int>(std::thread::hardware_concurrency());
        if (hc < 1) hc = 1;
        if (hc > 8) hc = 8;
        return hc;
    }();
    return cached;
}

// Persistent worker pool. Training is single-threaded from the caller's point of
// view, so one shared pool is enough and avoids paying thread creation cost for
// every small matrix multiply.
class ThreadPool {
public:
    static ThreadPool& instance() {
        static ThreadPool* pool = new ThreadPool();
        return *pool;
    }

    void run(int n, const std::function<void(int)>& fn, int min_parallel) {
        if (n <= 0) return;
        const int workers = static_cast<int>(threads_.size());
        if (workers <= 0 || n < min_parallel) {
            for (int i = 0; i < n; ++i) fn(i);
            return;
        }
        const int w = std::min(workers, n);
        std::atomic<int> next{0};
        std::atomic<int> done{0};
        {
            std::lock_guard<std::mutex> lock(mu_);
            all_done_ = false;
            for (int t = 0; t < w; ++t) {
                tasks_.emplace_back([&] {
                    for (;;) {
                        const int i = next.fetch_add(1);
                        if (i >= n) break;
                        fn(i);
                    }
                    if (done.fetch_add(1) + 1 == w) {
                        std::lock_guard<std::mutex> lk(done_mu_);
                        all_done_ = true;
                        done_cv_.notify_all();
                    }
                });
            }
        }
        cv_.notify_all();
        std::unique_lock<std::mutex> lk(done_mu_);
        done_cv_.wait(lk, [&] { return all_done_; });
    }

private:
    ThreadPool() {
        const int nt = hardware_threads();
        for (int i = 0; i < nt; ++i) {
            threads_.emplace_back([this] { worker(); });
        }
    }

    void worker() {
        for (;;) {
            std::function<void()> task;
            {
                std::unique_lock<std::mutex> lk(mu_);
                cv_.wait(lk, [&] { return !tasks_.empty(); });
                task = std::move(tasks_.front());
                tasks_.pop_front();
            }
            task();
        }
    }

    std::vector<std::thread> threads_;
    std::deque<std::function<void()>> tasks_;
    std::mutex mu_;
    std::condition_variable cv_;
    std::mutex done_mu_;
    std::condition_variable done_cv_;
    bool all_done_ = false;
};

template <typename Fn>
void parallel_for(int n, Fn&& fn) {
    ThreadPool::instance().run(n, std::function<void(int)>(fn), 64);
}

template <typename Fn>
void parallel_for_any(int n, Fn&& fn) {
    ThreadPool::instance().run(n, std::function<void(int)>(fn), 1);
}

inline float dot_f32(const float* a, const float* b, int n) {
#if defined(__aarch64__)
    float32x4_t acc = vdupq_n_f32(0.0f);
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        acc = vmlaq_f32(acc, vld1q_f32(a + i), vld1q_f32(b + i));
    }
    float s = vaddvq_f32(acc);
    for (; i < n; ++i) s += a[i] * b[i];
    return s;
#elif defined(__ARM_NEON)
    float32x4_t acc = vdupq_n_f32(0.0f);
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        acc = vmlaq_f32(acc, vld1q_f32(a + i), vld1q_f32(b + i));
    }
    float32x2_t lo = vget_low_f32(acc);
    float32x2_t hi = vget_high_f32(acc);
    float32x2_t sum = vadd_f32(lo, hi);
    sum = vpadd_f32(sum, sum);
    float s = vget_lane_f32(sum, 0);
    for (; i < n; ++i) s += a[i] * b[i];
    return s;
#else
    float s = 0.0f;
    for (int i = 0; i < n; ++i) s += a[i] * b[i];
    return s;
#endif
}

void matmul_nt_raw(const Tensor& x, const float* w, int out, Tensor& y) {
    const int n = x.rows;
    const int in = x.cols;
    y.ensure(n, out);
    const int64_t work = static_cast<int64_t>(n) * out * in;
    auto body = [&](int r) {
        const float* xr = x.row(r);
        float* yr = y.row(r);
        for (int o = 0; o < out; ++o) {
            yr[o] = dot_f32(xr, w + static_cast<size_t>(o) * in, in);
        }
    };
    if (work < 300000) {
        for (int r = 0; r < n; ++r) body(r);
    } else {
        parallel_for(n, body);
    }
}

void add_bias(Tensor& y, const float* bias) {
    for (int r = 0; r < y.rows; ++r) {
        float* yr = y.row(r);
        for (int c = 0; c < y.cols; ++c) yr[c] += bias[c];
    }
}

inline float gelu_scalar(float x) {
    const float u = kGeluC * (x + kGeluK * x * x * x);
    return 0.5f * x * (1.0f + std::tanh(u));
}

inline float gelu_grad_scalar(float x) {
    const float u = kGeluC * (x + kGeluK * x * x * x);
    const float t = std::tanh(u);
    const float du = kGeluC * (1.0f + 3.0f * kGeluK * x * x);
    return 0.5f * (1.0f + t) + 0.5f * x * (1.0f - t * t) * du;
}

void matmul_dw_add(Param& p, const Tensor& dout, const Tensor& x) {
    const int out = p.rows;
    const int in = p.cols;
    const int n = dout.rows;
    const int64_t work = static_cast<int64_t>(out) * in * n;
    auto body = [&](int o) {
        float* gp = p.g.data() + static_cast<size_t>(o) * in;
        for (int i = 0; i < in; ++i) {
            float s = 0.0f;
            for (int r = 0; r < n; ++r) {
                s += dout.row(r)[o] * x.row(r)[i];
            }
            gp[i] += s;
        }
    };
    if (work < 300000) {
        for (int o = 0; o < out; ++o) body(o);
    } else {
        parallel_for(out, body);
    }
}

void matmul_dx_add(const Tensor& dout, const float* w, int out, int in, Tensor& din) {
    const int n = dout.rows;
    const int64_t work = static_cast<int64_t>(n) * out * in;
    auto body = [&](int r) {
        const float* dr = dout.row(r);
        float* di = din.row(r);
        for (int o = 0; o < out; ++o) {
            const float g = dr[o];
            if (g == 0.0f) continue;
            const float* wr = w + static_cast<size_t>(o) * in;
            for (int k = 0; k < in; ++k) di[k] += g * wr[k];
        }
    };
    if (work < 300000) {
        for (int r = 0; r < n; ++r) body(r);
    } else {
        parallel_for(n, body);
    }
}

void bias_grad_add(Param& b, const Tensor& dout) {
    const int out = b.cols;
    auto body = [&](int o) {
        float s = 0.0f;
        for (int r = 0; r < dout.rows; ++r) s += dout.row(r)[o];
        b.g[o] += s;
    };
    if (static_cast<int64_t>(out) * dout.rows < 300000) {
        for (int o = 0; o < out; ++o) body(o);
    } else {
        parallel_for(out, body);
    }
}

void layernorm_backward(const Tensor& x, Param& gamma, Param& beta,
                        const Tensor& dy, Tensor& dx, const Tensor& inv) {
    const int n = x.rows;
    const int d = x.cols;
    dx.ensure(n, d);
    std::vector<float> dg(d, 0.0f);
    std::vector<float> db(d, 0.0f);
    for (int r = 0; r < n; ++r) {
        const float* xr = x.row(r);
        const float* dyr = dy.row(r);
        float mean = 0.0f;
        for (int j = 0; j < d; ++j) mean += xr[j];
        mean /= d;
        const float iv = inv.row(r)[0];
        float m1 = 0.0f;
        float m2 = 0.0f;
        for (int j = 0; j < d; ++j) {
            const float xhat = (xr[j] - mean) * iv;
            const float g = gamma.w[j];
            dg[j] += dyr[j] * xhat;
            db[j] += dyr[j];
            const float t = dyr[j] * g;
            m1 += t;
            m2 += t * xhat;
        }
        m1 /= d;
        m2 /= d;
        float* dxr = dx.row(r);
        for (int j = 0; j < d; ++j) {
            const float xhat = (xr[j] - mean) * iv;
            dxr[j] = iv * (dyr[j] * gamma.w[j] - m1 - xhat * m2);
        }
    }
    for (int j = 0; j < d; ++j) {
        gamma.g[j] += dg[j];
        beta.g[j] += db[j];
    }
}

void attention_forward(const ModelConfig& cfg, LayerCache& lc, int B, int T) {
    const int d = cfg.d_model;
    const int H = cfg.heads;
    const int hd = cfg.head_dim();
    const int n = B * T;
    const float inv_sqrt_hd = 1.0f / std::sqrt(static_cast<float>(hd));

    lc.attn_out.ensure(n, d);
    lc.probs.ensure(B * H * T, T);

    auto body = [&](int bh) {
        const int b = bh / H;
        const int h = bh % H;
        for (int t = 0; t < T; ++t) {
            const float* q = lc.q.row(b * T + t) + h * hd;
            float* prow = lc.probs.row(bh * T + t);
            float maxv = -1e30f;
            for (int s = 0; s <= t; ++s) {
                const float* k = lc.k.row(b * T + s) + h * hd;
                const float sc = dot_f32(q, k, hd) * inv_sqrt_hd;
                prow[s] = sc;
                if (sc > maxv) maxv = sc;
            }
            float sum = 0.0f;
            for (int s = 0; s <= t; ++s) {
                const float e = std::exp(prow[s] - maxv);
                prow[s] = e;
                sum += e;
            }
            const float invsum = 1.0f / sum;
            for (int s = 0; s <= t; ++s) prow[s] *= invsum;

            float* out = lc.attn_out.row(b * T + t) + h * hd;
            for (int j = 0; j < hd; ++j) out[j] = 0.0f;
            for (int s = 0; s <= t; ++s) {
                const float p = prow[s];
                const float* v = lc.v.row(b * T + s) + h * hd;
                for (int j = 0; j < hd; ++j) out[j] += p * v[j];
            }
        }
    };

    const int64_t work = static_cast<int64_t>(B) * H * T * T * hd;
    if (work < 300000) {
        for (int bh = 0; bh < B * H; ++bh) body(bh);
    } else {
        parallel_for_any(B * H, body);
    }
}

void attention_backward(const ModelConfig& cfg, const LayerCache& lc,
                        const Tensor& dattn, Tensor& dq, Tensor& dk, Tensor& dv,
                        int B, int T) {
    const int d = cfg.d_model;
    const int H = cfg.heads;
    const int hd = cfg.head_dim();
    const int n = B * T;
    const float inv_sqrt_hd = 1.0f / std::sqrt(static_cast<float>(hd));
    dq.ensure(n, d);
    dk.ensure(n, d);
    dv.ensure(n, d);

    auto body = [&](int bh) {
        const int b = bh / H;
        const int h = bh % H;
        std::vector<float> dp(T);
        for (int t = 0; t < T; ++t) {
            const float* datt = dattn.row(b * T + t) + h * hd;
            const float* prow = lc.probs.row(bh * T + t);
            float dot = 0.0f;
            for (int s = 0; s <= t; ++s) {
                const float* v = lc.v.row(b * T + s) + h * hd;
                float sdp = 0.0f;
                for (int j = 0; j < hd; ++j) sdp += datt[j] * v[j];
                dp[s] = sdp;
                dot += prow[s] * sdp;
            }
            float* dqr = dq.row(b * T + t) + h * hd;
            const float* q = lc.q.row(b * T + t) + h * hd;
            for (int s = 0; s <= t; ++s) {
                const float ds = prow[s] * (dp[s] - dot) * inv_sqrt_hd;
                const float* k = lc.k.row(b * T + s) + h * hd;
                float* dkr = dk.row(b * T + s) + h * hd;
                float* dvr = dv.row(b * T + s) + h * hd;
                const float p = prow[s];
                for (int j = 0; j < hd; ++j) {
                    dqr[j] += ds * k[j];
                    dkr[j] += ds * q[j];
                    dvr[j] += p * datt[j];
                }
            }
        }
    };

    const int64_t work = static_cast<int64_t>(B) * H * T * T * hd * 2;
    if (work < 300000) {
        for (int bh = 0; bh < B * H; ++bh) body(bh);
    } else {
        parallel_for_any(B * H, body);
    }
}

// Symmetric per-group fake quantization used by QAT (straight-through in backward).
void fake_quant_into(const Param& src, Tensor& dst, int bits, int group) {
    const int out = src.rows;
    const int in = src.cols;
    dst.init(out, in);
    if (bits <= 0 || bits >= 32) {
        std::copy(src.w.begin(), src.w.end(), dst.d.begin());
        return;
    }
    const int qmax = (1 << (bits - 1)) - 1;
    if (group <= 0) group = 32;
    for (int r = 0; r < out; ++r) {
        const float* wr = src.row(r);
        float* dr = dst.row(r);
        for (int g0 = 0; g0 < in; g0 += group) {
            const int g1 = std::min(in, g0 + group);
            float maxabs = 0.0f;
            for (int j = g0; j < g1; ++j) maxabs = std::max(maxabs, std::fabs(wr[j]));
            if (maxabs == 0.0f) {
                for (int j = g0; j < g1; ++j) dr[j] = 0.0f;
                continue;
            }
            const float scale = maxabs / static_cast<float>(qmax);
            for (int j = g0; j < g1; ++j) {
                float q = std::round(wr[j] / scale);
                q = std::max(-static_cast<float>(qmax), std::min(static_cast<float>(qmax), q));
                dr[j] = q * scale;
            }
        }
    }
}

const float* lin_w(const Linear& lin) {
    if (lin.qat_bits > 0 && !lin.fake_weight.d.empty()) return lin.fake_weight.data();
    return lin.weight.data();
}

} // namespace

void matmul_nt(const Tensor& x, const float* w, int out, Tensor& y) {
    matmul_nt_raw(x, w, out, y);
}

void matmul_nt(const Tensor& x, const Tensor& w, Tensor& y) {
    matmul_nt_raw(x, w.data(), w.rows, y);
}

void gelu_forward(const Tensor& in, Tensor& out) {
    out.ensure(in.rows, in.cols);
    for (size_t i = 0; i < in.d.size(); ++i) out.d[i] = gelu_scalar(in.d[i]);
}

void layernorm_forward(const Tensor& x, const float* gamma, const float* beta,
                       Tensor& y, Tensor& inv, float eps) {
    const int n = x.rows;
    const int d = x.cols;
    y.ensure(n, d);
    inv.ensure(n, 1);
    parallel_for(n, [&](int r) {
        const float* xr = x.row(r);
        float mean = 0.0f;
        for (int j = 0; j < d; ++j) mean += xr[j];
        mean /= d;
        float var = 0.0f;
        for (int j = 0; j < d; ++j) {
            const float t = xr[j] - mean;
            var += t * t;
        }
        var /= d;
        const float iv = 1.0f / std::sqrt(var + eps);
        inv.row(r)[0] = iv;
        float* yr = y.row(r);
        for (int j = 0; j < d; ++j) {
            yr[j] = (xr[j] - mean) * iv * gamma[j] + beta[j];
        }
    });
}

double cross_entropy_loss(const Tensor& logits, const int* targets, Tensor& dlogits) {
    const int n = logits.rows;
    const int V = logits.cols;
    dlogits.ensure(n, V);
    double total = 0.0;
    for (int r = 0; r < n; ++r) {
        const float* lr = logits.row(r);
        float mx = -1e30f;
        for (int v = 0; v < V; ++v) mx = std::max(mx, lr[v]);
        float sum = 0.0f;
        for (int v = 0; v < V; ++v) sum += std::exp(lr[v] - mx);
        const int tgt = targets[r];
        const float p = std::exp(lr[tgt] - mx) / sum;
        total += -std::log(std::max(p, 1e-30f));
        float* dr = dlogits.row(r);
        const float invn = 1.0f / static_cast<float>(n);
        for (int v = 0; v < V; ++v) {
            float prob = std::exp(lr[v] - mx) / sum;
            dr[v] = (prob - (v == tgt ? 1.0f : 0.0f)) * invn;
        }
    }
    return total / static_cast<double>(n);
}

// ---------------------------------------------------------------------------
// GPT
// ---------------------------------------------------------------------------

bool GPT::init(const ModelConfig& cfg, uint64_t seed) {
    if (!cfg.valid()) return false;
    cfg_ = cfg;
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> normal(0.0f, 0.02f);

    wte_.init(cfg_.vocab, cfg_.d_model);
    wpe_.init(cfg_.context, cfg_.d_model);
    lnf_g_.init(1, cfg_.d_model, false);
    lnf_b_.init(1, cfg_.d_model, false);

    for (auto* p : {&wte_, &wpe_}) {
        for (auto& x : p->w) x = normal(rng);
    }
    std::fill(lnf_g_.w.begin(), lnf_g_.w.end(), 1.0f);

    const float residual_std = 0.02f / std::sqrt(2.0f * static_cast<float>(cfg_.layers));
    layers_.assign(cfg_.layers, Layer{});
    for (auto& layer : layers_) {
        layer.ln1.gamma.init(1, cfg_.d_model, false);
        layer.ln1.beta.init(1, cfg_.d_model, false);
        layer.ln2.gamma.init(1, cfg_.d_model, false);
        layer.ln2.beta.init(1, cfg_.d_model, false);
        std::fill(layer.ln1.gamma.w.begin(), layer.ln1.gamma.w.end(), 1.0f);
        std::fill(layer.ln2.gamma.w.begin(), layer.ln2.gamma.w.end(), 1.0f);

        auto init_linear = [&](Linear& lin, int out, int in) {
            lin.weight.init(out, in, true);
            lin.bias.init(1, out, false);
            for (auto& x : lin.weight.w) x = normal(rng);
        };
        init_linear(layer.wq, cfg_.d_model, cfg_.d_model);
        init_linear(layer.wk, cfg_.d_model, cfg_.d_model);
        init_linear(layer.wv, cfg_.d_model, cfg_.d_model);
        init_linear(layer.wo, cfg_.d_model, cfg_.d_model);
        init_linear(layer.w1, cfg_.d_ff, cfg_.d_model);
        init_linear(layer.w2, cfg_.d_model, cfg_.d_ff);

        for (auto* lin : {&layer.wo, &layer.w2}) {
            const float scale = residual_std / 0.02f;
            for (auto& x : lin->weight.w) x *= scale;
        }
        for (auto* lin : {&layer.wq, &layer.wk, &layer.wv, &layer.wo, &layer.w1, &layer.w2}) {
            lin->qat_bits = 0;
            lin->qat_group = 32;
        }
    }
    qat_bits_ = 0;
    qat_group_ = 32;
    step_ = 0;
    cache_ = ForwardCache{};
    return true;
}

std::vector<Param*> GPT::params() {
    std::vector<Param*> out = {&wte_, &wpe_, &lnf_g_, &lnf_b_};
    for (auto& l : layers_) {
        out.push_back(&l.ln1.gamma);
        out.push_back(&l.ln1.beta);
        out.push_back(&l.ln2.gamma);
        out.push_back(&l.ln2.beta);
        out.push_back(&l.wq.weight);
        out.push_back(&l.wq.bias);
        out.push_back(&l.wk.weight);
        out.push_back(&l.wk.bias);
        out.push_back(&l.wv.weight);
        out.push_back(&l.wv.bias);
        out.push_back(&l.wo.weight);
        out.push_back(&l.wo.bias);
        out.push_back(&l.w1.weight);
        out.push_back(&l.w1.bias);
        out.push_back(&l.w2.weight);
        out.push_back(&l.w2.bias);
    }
    return out;
}

std::vector<Linear*> GPT::linears() {
    std::vector<Linear*> out;
    for (auto& l : layers_) {
        out.push_back(&l.wq);
        out.push_back(&l.wk);
        out.push_back(&l.wv);
        out.push_back(&l.wo);
        out.push_back(&l.w1);
        out.push_back(&l.w2);
    }
    return out;
}

void GPT::set_qat(int bits, int group) {
    qat_bits_ = bits;
    qat_group_ = group;
    for (auto* lin : linears()) {
        lin->qat_bits = bits;
        lin->qat_group = group;
        if (bits > 0) {
            fake_quant_into(lin->weight, lin->fake_weight, bits, group);
        } else {
            lin->fake_weight = Tensor{};
        }
    }
}

void GPT::zero_grads() {
    for (auto* p : params()) std::fill(p->g.begin(), p->g.end(), 0.0f);
}

void GPT::forward_logits(const int* tokens, int B, int T) {
    const int d = cfg_.d_model;
    const int V = cfg_.vocab;
    const int n = B * T;
    const int H = cfg_.heads;
    const int df = cfg_.d_ff;

    for (auto* lin : linears()) {
        if (lin->qat_bits > 0) {
            fake_quant_into(lin->weight, lin->fake_weight, lin->qat_bits, lin->qat_group);
        }
    }

    cache_.B = B;
    cache_.T = T;
    cache_.tokens.assign(tokens, tokens + n);
    cache_.layers.resize(cfg_.layers);

    Tensor cur(n, d);
    for (int i = 0; i < n; ++i) {
        const int tok = tokens[i];
        const int pos = i % T;
        const float* te = wte_.row(tok);
        const float* pe = wpe_.row(pos);
        float* xr = cur.row(i);
        for (int j = 0; j < d; ++j) xr[j] = te[j] + pe[j];
    }
    cache_.x0 = cur;

    for (int l = 0; l < cfg_.layers; ++l) {
        Layer& layer = layers_[l];
        LayerCache& lc = cache_.layers[l];

        lc.xin = cur;
        layernorm_forward(cur, layer.ln1.gamma.w.data(), layer.ln1.beta.w.data(),
                          lc.ln1, lc.inv1, kLnEps);

        matmul_nt_raw(lc.ln1, lin_w(layer.wq), d, lc.q);
        add_bias(lc.q, layer.wq.bias.w.data());
        matmul_nt_raw(lc.ln1, lin_w(layer.wk), d, lc.k);
        add_bias(lc.k, layer.wk.bias.w.data());
        matmul_nt_raw(lc.ln1, lin_w(layer.wv), d, lc.v);
        add_bias(lc.v, layer.wv.bias.w.data());

        attention_forward(cfg_, lc, B, T);

        matmul_nt_raw(lc.attn_out, lin_w(layer.wo), d, lc.proj);
        add_bias(lc.proj, layer.wo.bias.w.data());
        lc.xmid.ensure(n, d);
        for (int i = 0; i < n; ++i) {
            const float* a = cur.row(i);
            const float* b = lc.proj.row(i);
            float* o = lc.xmid.row(i);
            for (int j = 0; j < d; ++j) o[j] = a[j] + b[j];
        }

        layernorm_forward(lc.xmid, layer.ln2.gamma.w.data(), layer.ln2.beta.w.data(),
                          lc.ln2, lc.inv2, kLnEps);

        matmul_nt_raw(lc.ln2, lin_w(layer.w1), df, lc.h1pre);
        add_bias(lc.h1pre, layer.w1.bias.w.data());
        gelu_forward(lc.h1pre, lc.h1);

        matmul_nt_raw(lc.h1, lin_w(layer.w2), d, lc.h2);
        add_bias(lc.h2, layer.w2.bias.w.data());

        Tensor next(n, d);
        for (int i = 0; i < n; ++i) {
            const float* a = lc.xmid.row(i);
            const float* b = lc.h2.row(i);
            float* o = next.row(i);
            for (int j = 0; j < d; ++j) o[j] = a[j] + b[j];
        }
        cur = std::move(next);
    }

    cache_.final_in = cur;
    layernorm_forward(cur, lnf_g_.w.data(), lnf_b_.w.data(), cache_.lnf, cache_.invf, kLnEps);
    matmul_nt_raw(cache_.lnf, wte_.data(), V, cache_.logits);
}

float GPT::forward(const int* tokens, const int* targets, int B, int T) {
    if (!cfg_.valid() || B <= 0 || T <= 0 || T > cfg_.context) return 0.0f;
    forward_logits(tokens, B, T);
    if (targets == nullptr) return 0.0f;
    return static_cast<float>(cross_entropy_loss(cache_.logits, targets, cache_.dlogits));
}

void GPT::backward() {
    if (cache_.logits.d.empty() || cache_.dlogits.d.empty()) return;
    const int d = cfg_.d_model;
    const int df = cfg_.d_ff;
    const int V = cfg_.vocab;
    const int B = cache_.B;
    const int T = cache_.T;
    const int n = B * T;

    Tensor dcur(n, d);

    // Tied output head: logits = lnf @ wte^T
    Tensor dlnf(n, d);
    matmul_dx_add(cache_.dlogits, wte_.data(), V, d, dlnf);
    matmul_dw_add(wte_, cache_.dlogits, cache_.lnf);
    layernorm_backward(cache_.final_in, lnf_g_, lnf_b_, dlnf, dcur, cache_.invf);

    for (int l = cfg_.layers - 1; l >= 0; --l) {
        Layer& layer = layers_[l];
        LayerCache& lc = cache_.layers[l];

        // MLP: xout = xmid + h2
        matmul_dw_add(layer.w2.weight, dcur, lc.h1);
        bias_grad_add(layer.w2.bias, dcur);
        Tensor dh1(n, df);
        matmul_dx_add(dcur, lin_w(layer.w2), d, df, dh1);
        Tensor dh1pre(n, df);
        for (int i = 0; i < n; ++i) {
            const float* a = dh1.row(i);
            const float* pre = lc.h1pre.row(i);
            float* o = dh1pre.row(i);
            for (int j = 0; j < df; ++j) o[j] = a[j] * gelu_grad_scalar(pre[j]);
        }
        matmul_dw_add(layer.w1.weight, dh1pre, lc.ln2);
        bias_grad_add(layer.w1.bias, dh1pre);
        Tensor dln2(n, d);
        matmul_dx_add(dh1pre, lin_w(layer.w1), df, d, dln2);
        Tensor dxmid_ln(n, d);
        layernorm_backward(lc.xmid, layer.ln2.gamma, layer.ln2.beta,
                           dln2, dxmid_ln, lc.inv2);

        Tensor dxmid(n, d);
        for (int i = 0; i < n; ++i) {
            const float* a = dcur.row(i);
            const float* b = dxmid_ln.row(i);
            float* o = dxmid.row(i);
            for (int j = 0; j < d; ++j) o[j] = a[j] + b[j];
        }

        // Attention block: xmid = xin + proj
        matmul_dw_add(layer.wo.weight, dxmid, lc.attn_out);
        bias_grad_add(layer.wo.bias, dxmid);
        Tensor dattn(n, d);
        matmul_dx_add(dxmid, lin_w(layer.wo), d, d, dattn);

        Tensor dq(n, d), dk(n, d), dv(n, d);
        attention_backward(cfg_, lc, dattn, dq, dk, dv, B, T);

        matmul_dw_add(layer.wq.weight, dq, lc.ln1);
        bias_grad_add(layer.wq.bias, dq);
        matmul_dw_add(layer.wk.weight, dk, lc.ln1);
        bias_grad_add(layer.wk.bias, dk);
        matmul_dw_add(layer.wv.weight, dv, lc.ln1);
        bias_grad_add(layer.wv.bias, dv);

        Tensor dln1(n, d);
        matmul_dx_add(dq, lin_w(layer.wq), d, d, dln1);
        matmul_dx_add(dk, lin_w(layer.wk), d, d, dln1);
        matmul_dx_add(dv, lin_w(layer.wv), d, d, dln1);

        Tensor dxin_ln(n, d);
        layernorm_backward(lc.xin, layer.ln1.gamma, layer.ln1.beta,
                           dln1, dxin_ln, lc.inv1);

        for (int i = 0; i < n; ++i) {
            const float* a = dxmid.row(i);
            const float* b = dxin_ln.row(i);
            float* o = dcur.row(i);
            for (int j = 0; j < d; ++j) o[j] = a[j] + b[j];
        }
    }

    // Embedding gradients (weight tying: wte also received the head gradient).
    for (int i = 0; i < n; ++i) {
        const int tok = cache_.tokens[i];
        const int pos = i % T;
        const float* dc = dcur.row(i);
        float* gte = wte_.g.data() + static_cast<size_t>(tok) * d;
        float* gpe = wpe_.g.data() + static_cast<size_t>(pos) * d;
        for (int j = 0; j < d; ++j) {
            gte[j] += dc[j];
            gpe[j] += dc[j];
        }
    }
}

void GPT::adam_update(float lr, float weight_decay) {
    ++step_;
    const float b1 = 0.9f;
    const float b2 = 0.999f;
    const float eps = 1e-8f;
    const double bc1 = 1.0 - std::pow(static_cast<double>(b1), static_cast<double>(step_));
    const double bc2 = 1.0 - std::pow(static_cast<double>(b2), static_cast<double>(step_));
    const float inv_bc1 = static_cast<float>(1.0 / bc1);
    const float inv_bc2 = static_cast<float>(1.0 / bc2);

    for (auto* p : params()) {
        const size_t count = p->w.size();
        float* w = p->w.data();
        const float* g = p->g.data();
        float* m = p->m.data();
        float* v = p->v.data();
        const float wd = p->weight_decay ? weight_decay : 0.0f;
        for (size_t i = 0; i < count; ++i) {
            m[i] = b1 * m[i] + (1.0f - b1) * g[i];
            v[i] = b2 * v[i] + (1.0f - b2) * g[i] * g[i];
            const float mh = m[i] * inv_bc1;
            const float vh = v[i] * inv_bc2;
            w[i] -= lr * (mh / (std::sqrt(vh) + eps) + wd * w[i]);
        }
    }
}

float GPT::train_step(const int* tokens, const int* targets, int B, int T,
                      float lr, float weight_decay) {
    zero_grads();
    const float loss = forward(tokens, targets, B, T);
    backward();
    adam_update(lr, weight_decay);
    zero_grads();
    return loss;
}

std::vector<int> GPT::generate(const std::vector<int>& prompt, int max_new,
                               float temperature, int top_k, float top_p,
                               uint64_t seed) {
    std::vector<int> seq = prompt;
    if (seq.empty()) seq.push_back(0);
    if (max_new <= 0) return {};
    const int V = cfg_.vocab;
    std::mt19937_64 rng(seed);
    std::vector<int> generated;
    generated.reserve(max_new);

    for (int step = 0; step < max_new; ++step) {
        const int start = std::max(0, static_cast<int>(seq.size()) - cfg_.context);
        const int T = static_cast<int>(seq.size()) - start;
        forward(seq.data() + start, nullptr, 1, T);
        const float* logits = cache_.logits.row(T - 1);

        int next = 0;
        if (temperature <= 0.0f) {
            next = static_cast<int>(std::max_element(logits, logits + V) - logits);
        } else {
            std::vector<int> idx(V);
            for (int i = 0; i < V; ++i) idx[i] = i;
            std::vector<float> scaled(V);
            for (int i = 0; i < V; ++i) scaled[i] = logits[i] / std::max(1e-5f, temperature);

            int keep = V;
            if (top_k > 0 && top_k < V) {
                keep = top_k;
                std::partial_sort(idx.begin(), idx.begin() + keep, idx.end(),
                                  [&](int a, int b) { return scaled[a] > scaled[b]; });
            } else {
                std::sort(idx.begin(), idx.end(),
                          [&](int a, int b) { return scaled[a] > scaled[b]; });
            }

            std::vector<float> probs(keep);
            float mx = -1e30f;
            for (int i = 0; i < keep; ++i) mx = std::max(mx, scaled[idx[i]]);
            float sum = 0.0f;
            for (int i = 0; i < keep; ++i) {
                probs[i] = std::exp(scaled[idx[i]] - mx);
                sum += probs[i];
            }
            for (int i = 0; i < keep; ++i) probs[i] /= sum;

            int cutoff = keep;
            if (top_p > 0.0f && top_p < 1.0f) {
                float cumulative = 0.0f;
                for (int i = 0; i < keep; ++i) {
                    cumulative += probs[i];
                    if (cumulative >= top_p) {
                        cutoff = i + 1;
                        break;
                    }
                }
            }
            float renorm = 0.0f;
            for (int i = 0; i < cutoff; ++i) renorm += probs[i];
            std::uniform_real_distribution<float> uni(0.0f, renorm);
            const float target = uni(rng);
            float acc = 0.0f;
            next = idx[cutoff - 1];
            for (int i = 0; i < cutoff; ++i) {
                acc += probs[i];
                if (target <= acc) {
                    next = idx[i];
                    break;
                }
            }
        }
        seq.push_back(next);
        generated.push_back(next);
    }
    return generated;
}

namespace {

bool write_param(FILE* f, const Param& p, bool with_optimizer) {
    const int32_t rows = p.rows;
    const int32_t cols = p.cols;
    if (std::fwrite(&rows, sizeof(rows), 1, f) != 1) return false;
    if (std::fwrite(&cols, sizeof(cols), 1, f) != 1) return false;
    if (!p.w.empty() && std::fwrite(p.w.data(), sizeof(float), p.w.size(), f) != p.w.size()) {
        return false;
    }
    if (with_optimizer) {
        if (!p.m.empty() && std::fwrite(p.m.data(), sizeof(float), p.m.size(), f) != p.m.size()) {
            return false;
        }
        if (!p.v.empty() && std::fwrite(p.v.data(), sizeof(float), p.v.size(), f) != p.v.size()) {
            return false;
        }
    }
    return true;
}

bool read_param(FILE* f, Param& p, bool file_has_optimizer, bool load_optimizer) {
    int32_t rows = 0;
    int32_t cols = 0;
    if (std::fread(&rows, sizeof(rows), 1, f) != 1) return false;
    if (std::fread(&cols, sizeof(cols), 1, f) != 1) return false;
    if (rows != p.rows || cols != p.cols) return false;
    if (!p.w.empty() && std::fread(p.w.data(), sizeof(float), p.w.size(), f) != p.w.size()) {
        return false;
    }
    if (file_has_optimizer) {
        if (load_optimizer) {
            if (!p.m.empty() && std::fread(p.m.data(), sizeof(float), p.m.size(), f) != p.m.size()) {
                return false;
            }
            if (!p.v.empty() && std::fread(p.v.data(), sizeof(float), p.v.size(), f) != p.v.size()) {
                return false;
            }
        } else {
            std::vector<float> scratch(p.w.size());
            if (!scratch.empty() &&
                std::fread(scratch.data(), sizeof(float), scratch.size(), f) != scratch.size()) {
                return false;
            }
            if (!scratch.empty() &&
                std::fread(scratch.data(), sizeof(float), scratch.size(), f) != scratch.size()) {
                return false;
            }
        }
    }
    return true;
}

} // namespace

bool GPT::save_checkpoint(const std::string& path, bool with_optimizer) const {
    const std::string tmp = path + ".tmp";
    FILE* f = std::fopen(tmp.c_str(), "wb");
    if (!f) return false;

    int32_t magic = 0x4D46434B;  // "MFCK"
    int32_t version = 1;
    int32_t vocab = cfg_.vocab;
    int32_t context = cfg_.context;
    int32_t d_model = cfg_.d_model;
    int32_t layers = cfg_.layers;
    int32_t heads = cfg_.heads;
    int32_t d_ff = cfg_.d_ff;
    int32_t tied = cfg_.tied_embeddings ? 1 : 0;
    int64_t step = step_;
    int32_t has_opt = with_optimizer ? 1 : 0;
    int32_t qat_bits = qat_bits_;
    int32_t qat_group = qat_group_;
    int64_t param_count = cfg_.param_count();

    bool ok = true;
    ok &= std::fwrite(&magic, sizeof(magic), 1, f) == 1;
    ok &= std::fwrite(&version, sizeof(version), 1, f) == 1;
    ok &= std::fwrite(&vocab, sizeof(vocab), 1, f) == 1;
    ok &= std::fwrite(&context, sizeof(context), 1, f) == 1;
    ok &= std::fwrite(&d_model, sizeof(d_model), 1, f) == 1;
    ok &= std::fwrite(&layers, sizeof(layers), 1, f) == 1;
    ok &= std::fwrite(&heads, sizeof(heads), 1, f) == 1;
    ok &= std::fwrite(&d_ff, sizeof(d_ff), 1, f) == 1;
    ok &= std::fwrite(&tied, sizeof(tied), 1, f) == 1;
    ok &= std::fwrite(&step, sizeof(step), 1, f) == 1;
    ok &= std::fwrite(&has_opt, sizeof(has_opt), 1, f) == 1;
    ok &= std::fwrite(&qat_bits, sizeof(qat_bits), 1, f) == 1;
    ok &= std::fwrite(&qat_group, sizeof(qat_group), 1, f) == 1;
    ok &= std::fwrite(&param_count, sizeof(param_count), 1, f) == 1;

    ok &= write_param(f, wte_, with_optimizer);
    ok &= write_param(f, wpe_, with_optimizer);
    ok &= write_param(f, lnf_g_, with_optimizer);
    ok &= write_param(f, lnf_b_, with_optimizer);
    for (const auto& l : layers_) {
        ok &= write_param(f, l.ln1.gamma, with_optimizer);
        ok &= write_param(f, l.ln1.beta, with_optimizer);
        ok &= write_param(f, l.ln2.gamma, with_optimizer);
        ok &= write_param(f, l.ln2.beta, with_optimizer);
        ok &= write_param(f, l.wq.weight, with_optimizer);
        ok &= write_param(f, l.wq.bias, with_optimizer);
        ok &= write_param(f, l.wk.weight, with_optimizer);
        ok &= write_param(f, l.wk.bias, with_optimizer);
        ok &= write_param(f, l.wv.weight, with_optimizer);
        ok &= write_param(f, l.wv.bias, with_optimizer);
        ok &= write_param(f, l.wo.weight, with_optimizer);
        ok &= write_param(f, l.wo.bias, with_optimizer);
        ok &= write_param(f, l.w1.weight, with_optimizer);
        ok &= write_param(f, l.w1.bias, with_optimizer);
        ok &= write_param(f, l.w2.weight, with_optimizer);
        ok &= write_param(f, l.w2.bias, with_optimizer);
    }
    std::fclose(f);
    if (!ok) {
        std::remove(tmp.c_str());
        return false;
    }
    if (std::rename(tmp.c_str(), path.c_str()) != 0) {
        std::remove(tmp.c_str());
        return false;
    }
    return true;
}

bool GPT::load_checkpoint(const std::string& path, bool with_optimizer) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;

    int32_t magic = 0, version = 0;
    int32_t vocab = 0, context = 0, d_model = 0, layers = 0, heads = 0, d_ff = 0, tied = 0;
    int64_t step = 0;
    int32_t has_opt = 0, qat_bits = 0, qat_group = 32;
    int64_t param_count = 0;

    bool ok = true;
    ok &= std::fread(&magic, sizeof(magic), 1, f) == 1;
    ok &= std::fread(&version, sizeof(version), 1, f) == 1;
    ok &= std::fread(&vocab, sizeof(vocab), 1, f) == 1;
    ok &= std::fread(&context, sizeof(context), 1, f) == 1;
    ok &= std::fread(&d_model, sizeof(d_model), 1, f) == 1;
    ok &= std::fread(&layers, sizeof(layers), 1, f) == 1;
    ok &= std::fread(&heads, sizeof(heads), 1, f) == 1;
    ok &= std::fread(&d_ff, sizeof(d_ff), 1, f) == 1;
    ok &= std::fread(&tied, sizeof(tied), 1, f) == 1;
    ok &= std::fread(&step, sizeof(step), 1, f) == 1;
    ok &= std::fread(&has_opt, sizeof(has_opt), 1, f) == 1;
    ok &= std::fread(&qat_bits, sizeof(qat_bits), 1, f) == 1;
    ok &= std::fread(&qat_group, sizeof(qat_group), 1, f) == 1;
    ok &= std::fread(&param_count, sizeof(param_count), 1, f) == 1;

    if (!ok || magic != 0x4D46434B || version != 1) {
        std::fclose(f);
        return false;
    }

    ModelConfig cfg;
    cfg.vocab = vocab;
    cfg.context = context;
    cfg.d_model = d_model;
    cfg.layers = layers;
    cfg.heads = heads;
    cfg.d_ff = d_ff;
    cfg.tied_embeddings = tied != 0;
    if (!cfg.valid() || cfg.param_count() != param_count) {
        std::fclose(f);
        return false;
    }
    if (!init(cfg, 1)) {
        std::fclose(f);
        return false;
    }

    const bool file_has_opt = has_opt != 0;
    const bool load_opt = with_optimizer && file_has_opt;
    ok &= read_param(f, wte_, file_has_opt, load_opt);
    ok &= read_param(f, wpe_, file_has_opt, load_opt);
    ok &= read_param(f, lnf_g_, file_has_opt, load_opt);
    ok &= read_param(f, lnf_b_, file_has_opt, load_opt);
    for (auto& l : layers_) {
        ok &= read_param(f, l.ln1.gamma, file_has_opt, load_opt);
        ok &= read_param(f, l.ln1.beta, file_has_opt, load_opt);
        ok &= read_param(f, l.ln2.gamma, file_has_opt, load_opt);
        ok &= read_param(f, l.ln2.beta, file_has_opt, load_opt);
        ok &= read_param(f, l.wq.weight, file_has_opt, load_opt);
        ok &= read_param(f, l.wq.bias, file_has_opt, load_opt);
        ok &= read_param(f, l.wk.weight, file_has_opt, load_opt);
        ok &= read_param(f, l.wk.bias, file_has_opt, load_opt);
        ok &= read_param(f, l.wv.weight, file_has_opt, load_opt);
        ok &= read_param(f, l.wv.bias, file_has_opt, load_opt);
        ok &= read_param(f, l.wo.weight, file_has_opt, load_opt);
        ok &= read_param(f, l.wo.bias, file_has_opt, load_opt);
        ok &= read_param(f, l.w1.weight, file_has_opt, load_opt);
        ok &= read_param(f, l.w1.bias, file_has_opt, load_opt);
        ok &= read_param(f, l.w2.weight, file_has_opt, load_opt);
        ok &= read_param(f, l.w2.bias, file_has_opt, load_opt);
    }
    std::fclose(f);
    if (!ok) return false;

    step_ = step;
    set_qat(qat_bits, qat_group);
    return true;
}

} // namespace mf
