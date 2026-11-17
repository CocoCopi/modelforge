// Concurrency and input-validation regression tests.
//
// Before the thread-pool rewrite, two callers entering parallel_for at the
// same time shared the pool's global completion flag and captured caller
// stack state by reference. Two threads training different models reliably
// segfaulted. This test hammers that path and also checks that invalid tokens
// and shapes fail closed instead of corrupting the cache or the heap.

#include "core/gpt.h"

#include <atomic>
#include <cmath>
#include <cstdio>
#include <thread>
#include <vector>

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

static ModelConfig small_config() {
    ModelConfig cfg;
    cfg.vocab = 256;
    cfg.context = 64;
    cfg.d_model = 64;
    cfg.layers = 2;
    cfg.heads = 4;
    cfg.d_ff = 256;
    return cfg;
}

static void fill_batch(int* tokens, int* targets, int n, int vocab) {
    for (int i = 0; i < n; ++i) {
        tokens[i] = (i * 37 + 11) % vocab;
        targets[i] = (i * 53 + 7) % vocab;
    }
}

int main() {
    constexpr int kB = 4;
    constexpr int kT = 32;
    constexpr int kN = kB * kT;

    // 1. Two models training at the same time must not corrupt each other.
    {
        GPT a, b;
        check(a.init(small_config(), 7), "model A init");
        check(b.init(small_config(), 9), "model B init");

        std::vector<int> ta(kN), ya(kN), tb(kN), yb(kN);
        fill_batch(ta.data(), ya.data(), kN, 256);
        fill_batch(tb.data(), yb.data(), kN, 256);

        std::atomic<int> bad{0};
        std::atomic<int> done{0};
        auto worker = [&](GPT& model, std::vector<int>& tokens, std::vector<int>& targets) {
            for (int step = 0; step < 12; ++step) {
                const float loss =
                    model.train_step(tokens.data(), targets.data(), kB, kT, 1e-3f);
                if (!(loss >= 0.0f) || !std::isfinite(loss)) ++bad;
            }
            ++done;
        };

        std::thread t1(worker, std::ref(a), std::ref(ta), std::ref(ya));
        std::thread t2(worker, std::ref(b), std::ref(tb), std::ref(yb));
        t1.join();
        t2.join();

        check(done.load() == 2, "both concurrent trainers finished");
        check(bad.load() == 0, "concurrent training produced finite losses");
        check(a.step() == 12 && b.step() == 12, "each model counted its own steps");
    }

    // 2. Concurrent forward passes on separate instances.
    {
        GPT a, b;
        check(a.init(small_config(), 11), "forward model A init");
        check(b.init(small_config(), 13), "forward model B init");
        std::vector<int> tokens(kN);
        fill_batch(tokens.data(), tokens.data(), kN, 256);

        std::atomic<int> bad{0};
        auto worker = [&](GPT& model) {
            for (int i = 0; i < 40; ++i) {
                const float loss = model.forward(tokens.data(), nullptr, kB, kT);
                if (loss != 0.0f) ++bad;
            }
        };
        std::thread t1(worker, std::ref(a));
        std::thread t2(worker, std::ref(b));
        t1.join();
        t2.join();
        check(bad.load() == 0, "concurrent forwards returned cleanly");
    }

    // 3. Out-of-range tokens and oversized sequences fail closed.
    {
        GPT model;
        check(model.init(small_config(), 17), "validation model init");
        std::vector<int> tokens(kN), targets(kN);
        fill_batch(tokens.data(), targets.data(), kN, 256);

        tokens[5] = 256;  // vocab is 256, so this is one past the end
        check(model.forward(tokens.data(), targets.data(), kB, kT) < 0.0f,
              "out-of-range token rejected by forward");
        check(model.train_step(tokens.data(), targets.data(), kB, kT, 1e-3f) < 0.0f,
              "out-of-range token rejected by train_step");

        fill_batch(tokens.data(), targets.data(), kN, 256);
        targets[0] = -1;
        check(model.train_step(tokens.data(), targets.data(), kB, kT, 1e-3f) < 0.0f,
              "out-of-range target rejected");

        fill_batch(tokens.data(), targets.data(), kN, 256);
        check(model.forward(tokens.data(), targets.data(), kB, 65) < 0.0f,
              "sequence longer than context rejected");
        check(model.forward(tokens.data(), targets.data(), 0, kT) < 0.0f,
              "empty batch rejected");

        // A rejected batch must not poison the next real step.
        const float loss =
            model.train_step(tokens.data(), targets.data(), kB, kT, 1e-3f);
        check(loss >= 0.0f && std::isfinite(loss),
              "valid step still works after rejected input");
        check(model.step() == 1, "rejected batches do not advance the step");
    }

    // 4. Weights-only resume keeps progress but restarts AdamW bias
    //    correction, because the moments were not saved.
    {
        GPT model;
        check(model.init(small_config(), 19), "resume model init");
        std::vector<int> tokens(kN), targets(kN);
        fill_batch(tokens.data(), targets.data(), kN, 256);
        for (int i = 0; i < 5; ++i) {
            model.train_step(tokens.data(), targets.data(), kB, kT, 1e-3f);
        }
        check(model.step() == 5 && model.optimizer_step() == 5,
              "optimizer step tracks applied updates");

        const char* path = "/tmp/mf_test_weights_only.mfck";
        check(model.save_checkpoint(path, false), "weights-only checkpoint saved");

        GPT restored;
        check(restored.load_checkpoint(path, true), "weights-only checkpoint loaded");
        check(restored.step() == 5, "weights-only resume keeps training progress");
        check(restored.optimizer_step() == 0,
              "weights-only resume restarts bias correction");
        std::remove(path);
    }

    if (failures == 0) std::printf("all concurrency/validation tests passed\n");
    return failures == 0 ? 0 : 1;
}
