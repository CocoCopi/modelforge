#include "core/gpt.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

using mf::GPT;
using mf::ModelConfig;

namespace {

int failures = 0;

void expect(bool ok, const char* what) {
    if (!ok) {
        std::printf("FAIL: %s\n", what);
        ++failures;
    }
}

// Reference decoding: recompute the whole window for every token, which is
// what the pre-KV-cache generator did. Any divergence here means the fast path
// changed the model's answers rather than just its speed.
std::vector<int> reference_greedy(GPT& model, std::vector<int> seq, int max_new,
                                  int context) {
    std::vector<int> out;
    for (int step = 0; step < max_new; ++step) {
        const int start = std::max(0, static_cast<int>(seq.size()) - context);
        const int T = static_cast<int>(seq.size()) - start;
        model.forward(seq.data() + start, nullptr, 1, T);
        const float* logits = model.cache().logits.row(T - 1);
        const int V = model.config().vocab;
        const int next = static_cast<int>(std::max_element(logits, logits + V) - logits);
        seq.push_back(next);
        out.push_back(next);
    }
    return out;
}

void init_model(GPT& model, int context, uint64_t seed) {
    ModelConfig cfg;
    cfg.vocab = 64;
    cfg.context = context;
    cfg.d_model = 32;
    cfg.layers = 2;
    cfg.heads = 4;
    cfg.d_ff = 64;
    cfg.tied_embeddings = true;
    if (!model.init(cfg, seed)) {
        std::printf("FAIL: model init\n");
        ++failures;
    }
}

} // namespace

int main() {
    const std::vector<int> prompt = {3, 7, 11, 19, 23, 31};

    // Greedy parity inside the context window.
    {
        GPT fast;
        init_model(fast, 16, 7);
        GPT slow;
        init_model(slow, 16, 7);
        const auto a = fast.generate(prompt, 12, 0.0f, 0, 1.0f, 42);
        const auto b = reference_greedy(slow, prompt, 12, 16);
        expect(a == b, "greedy decode matches full recompute");
        expect(a.size() == 12, "generated the requested number of tokens");
    }

    // Greedy parity across the context boundary, where the window slides.
    {
        GPT fast;
        init_model(fast, 8, 11);
        GPT slow;
        init_model(slow, 8, 11);
        const auto a = fast.generate(prompt, 14, 0.0f, 0, 1.0f, 42);
        const auto b = reference_greedy(slow, prompt, 14, 8);
        expect(a == b, "greedy decode matches full recompute after re-windowing");
    }

    // Sampling stays deterministic per seed, and seeds actually differ.
    {
        GPT a;
        init_model(a, 16, 5);
        GPT b;
        init_model(b, 16, 5);
        GPT c;
        init_model(c, 16, 5);
        const auto one = a.generate(prompt, 16, 0.9f, 12, 0.9f, 1234);
        const auto two = b.generate(prompt, 16, 0.9f, 12, 0.9f, 1234);
        const auto three = c.generate(prompt, 16, 0.9f, 12, 0.9f, 98765);
        expect(one == two, "same seed reproduces the same sampled sequence");
        expect(one != three, "a different seed produces a different sequence");
    }

    // The streaming callback sees every token and can stop the run early.
    {
        GPT model;
        init_model(model, 16, 9);
        std::vector<int> streamed;
        const auto out = model.generate(prompt, 20, 0.0f, 0, 1.0f, 42,
                                        [&](int token) {
                                            streamed.push_back(token);
                                            return streamed.size() < 3;
                                        });
        expect(out.size() == 3, "callback stopped generation after three tokens");
        expect(streamed == out, "callback received exactly the returned tokens");
    }

    // A long prompt is truncated to the context window in both paths.
    {
        GPT fast;
        init_model(fast, 8, 13);
        GPT slow;
        init_model(slow, 8, 13);
        const std::vector<int> long_prompt = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
        const auto a = fast.generate(long_prompt, 6, 0.0f, 0, 1.0f, 42);
        const auto b = reference_greedy(slow, long_prompt, 6, 8);
        expect(a == b, "long prompts decode identically");
    }

    // Empty prompts and zero-length requests must not crash or invent tokens.
    {
        GPT model;
        init_model(model, 16, 17);
        expect(model.generate({}, 4, 0.0f, 0, 1.0f, 42).size() == 4,
               "empty prompt still decodes from the fallback token");
        expect(model.generate(prompt, 0, 0.0f, 0, 1.0f, 42).empty(),
               "max_new = 0 returns no tokens");
    }

    if (failures == 0) {
        std::printf("test_decode: all assertions passed\n");
        return 0;
    }
    std::printf("test_decode: %d assertion(s) failed\n", failures);
    return 1;
}
