// Host training driver. This is the binary the generated Colab notebook
// builds and runs, and it is also useful on a desktop: same engine, same
// checkpoint format, same AdamW update as the Android app.
#include "core/gpt.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace {

volatile std::sig_atomic_t g_stop = 0;

void on_signal(int) { g_stop = 1; }

struct Options {
    std::string tokens;
    std::string config;
    std::string init;
    std::string out = "cloud_out";
    int steps = 500;
    int batch = 4;
    int seq = 128;
    int warmup = 25;
    int every = 50;
    float lr = 6e-4f;
    float weight_decay = 0.01f;
    uint64_t seed = 42;
};

void usage() {
    std::printf(
        "mf_train --tokens tokens.bin --config config.json --out out_dir [options]\n"
        "  --steps N       optimizer steps (default 500)\n"
        "  --batch B       sequences per step (default 4)\n"
        "  --seq T         tokens per sequence (default 128)\n"
        "  --lr X          peak learning rate (default 6e-4)\n"
        "  --wd X          weight decay (default 0.01)\n"
        "  --warmup N      warmup steps (default 25)\n"
        "  --every N       checkpoint/progress interval (default 50)\n"
        "  --init FILE     .mfck to initialise weights from (fresh run)\n"
        "  --seed N        sampling seed (default 42)\n"
    );
}

std::string read_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

// Minimal flat-field reader: the config the app exports contains only simple
// "name": number pairs, so a full JSON dependency would be dead weight.
int json_int(const std::string& text, const std::string& key, int fallback) {
    const std::string needle = "\"" + key + "\"";
    const size_t at = text.find(needle);
    if (at == std::string::npos) return fallback;
    const size_t colon = text.find(':', at + needle.size());
    if (colon == std::string::npos) return fallback;
    return static_cast<int>(std::strtol(text.c_str() + colon + 1, nullptr, 10));
}

// TokenCache.writeInt is big-endian (Java DataOutputStream).
std::vector<int> read_tokens(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return {};
    auto read_be32 = [&](int32_t& out) -> bool {
        unsigned char b[4];
        if (!in.read(reinterpret_cast<char*>(b), 4)) return false;
        out = (static_cast<int32_t>(b[0]) << 24) | (static_cast<int32_t>(b[1]) << 16) |
            (static_cast<int32_t>(b[2]) << 8) | static_cast<int32_t>(b[3]);
        return true;
    };
    int32_t magic = 0, version = 0, count = 0;
    if (!read_be32(magic) || !read_be32(version) || !read_be32(count)) return {};
    if (magic != 0x4D46544B || version != 1 || count <= 0 || count > 64000000) return {};
    std::vector<int> tokens(static_cast<size_t>(count));
    for (int i = 0; i < count; ++i) {
        int32_t value = 0;
        if (!read_be32(value)) return {};
        tokens[static_cast<size_t>(i)] = value;
    }
    return tokens;
}

void write_progress(const std::string& dir, int step, int total, float loss,
                    float best, double tokens_per_sec, double eta_seconds,
                    const std::string& status) {
    std::ofstream out(dir + "/progress.json", std::ios::trunc);
    out << "{\n"
        << "  \"step\": " << step << ",\n"
        << "  \"totalSteps\": " << total << ",\n"
        << "  \"loss\": " << loss << ",\n"
        << "  \"bestLoss\": " << best << ",\n"
        << "  \"tokensPerSec\": " << tokens_per_sec << ",\n"
        << "  \"etaSeconds\": " << static_cast<long long>(eta_seconds) << ",\n"
        << "  \"status\": \"" << status << "\",\n"
        << "  \"updatedAt\": "
        << std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::system_clock::now().time_since_epoch()
           ).count()
        << "\n}\n";
}

} // namespace

int main(int argc, char** argv) {
    Options opt;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto next = [&](const char* name) -> std::string {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing value for %s\n", name);
                std::exit(2);
            }
            return argv[++i];
        };
        if (arg == "--tokens") opt.tokens = next("--tokens");
        else if (arg == "--config") opt.config = next("--config");
        else if (arg == "--init") opt.init = next("--init");
        else if (arg == "--out") opt.out = next("--out");
        else if (arg == "--steps") opt.steps = std::atoi(next("--steps").c_str());
        else if (arg == "--batch") opt.batch = std::atoi(next("--batch").c_str());
        else if (arg == "--seq") opt.seq = std::atoi(next("--seq").c_str());
        else if (arg == "--lr") opt.lr = std::atof(next("--lr").c_str());
        else if (arg == "--wd") opt.weight_decay = std::atof(next("--wd").c_str());
        else if (arg == "--warmup") opt.warmup = std::atoi(next("--warmup").c_str());
        else if (arg == "--every") opt.every = std::atoi(next("--every").c_str());
        else if (arg == "--seed") opt.seed = std::strtoull(next("--seed").c_str(), nullptr, 10);
        else if (arg == "--help") { usage(); return 0; }
        else {
            std::fprintf(stderr, "unknown argument: %s\n", arg.c_str());
            usage();
            return 2;
        }
    }
    if (opt.tokens.empty() || opt.config.empty()) {
        usage();
        return 2;
    }

    const std::string config_text = read_file(opt.config);
    mf::ModelConfig cfg;
    cfg.vocab = json_int(config_text, "vocabSize", 0);
    cfg.context = json_int(config_text, "contextLength", 0);
    cfg.d_model = json_int(config_text, "dModel", 0);
    cfg.layers = json_int(config_text, "layers", 0);
    cfg.heads = json_int(config_text, "heads", 0);
    cfg.d_ff = json_int(config_text, "dFf", 0);
    if (!cfg.valid()) {
        std::fprintf(stderr, "config.json is missing fields or is invalid.\n");
        return 1;
    }

    const auto tokens = read_tokens(opt.tokens);
    if (tokens.empty()) {
        std::fprintf(stderr, "tokens.bin is missing, truncated or not an MFTK cache.\n");
        return 1;
    }
    if (static_cast<int>(tokens.size()) <= opt.seq + 1) {
        std::fprintf(stderr, "not enough tokens (%zu) for sequence length %d.\n",
                     tokens.size(), opt.seq);
        return 1;
    }

    std::string command = "mkdir -p \"" + opt.out + "\"";
    if (std::system(command.c_str()) != 0) {
        std::fprintf(stderr, "could not create %s\n", opt.out.c_str());
        return 1;
    }

    mf::GPT model;
    bool resumed = false;
    if (!opt.init.empty()) {
        resumed = model.load_checkpoint(opt.init, false);
        if (!resumed) {
            std::fprintf(stderr, "could not load %s; starting from scratch.\n", opt.init.c_str());
        } else if (model.config().vocab != cfg.vocab || model.config().d_model != cfg.d_model ||
                   model.config().layers != cfg.layers || model.config().d_ff != cfg.d_ff) {
            std::fprintf(stderr, "init checkpoint does not match config.json.\n");
            return 1;
        }
    }
    if (!resumed && !model.init(cfg, opt.seed)) {
        std::fprintf(stderr, "could not allocate the model.\n");
        return 1;
    }
    model.set_step(0);
    model.reset_optimizer();

    std::printf("params=%lld tokens=%zu steps=%d batch=%d seq=%d init=%s\n",
                static_cast<long long>(model.param_count()), tokens.size(), opt.steps,
                opt.batch, opt.seq, resumed ? opt.init.c_str() : "random");

    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);

    std::mt19937 rng(opt.seed);
    std::vector<int> x(static_cast<size_t>(opt.batch) * opt.seq);
    std::vector<int> y(x.size());
    float best = 1e30f;
    const auto started = std::chrono::steady_clock::now();

    for (int step = 1; step <= opt.steps; ++step) {
        if (g_stop) break;
        const int max_start = static_cast<int>(tokens.size()) - opt.seq - 1;
        for (int b = 0; b < opt.batch; ++b) {
            std::uniform_int_distribution<int> pick(0, max_start);
            const int start = pick(rng);
            for (int t = 0; t < opt.seq; ++t) {
                const size_t at = static_cast<size_t>(b) * opt.seq + t;
                x[at] = tokens[static_cast<size_t>(start + t)];
                y[at] = tokens[static_cast<size_t>(start + t + 1)];
            }
        }
        // Warmup then cosine decay, matching LrSchedule in the app.
        const float progress = static_cast<float>(step) / std::max(1, opt.steps);
        const float lr = step < opt.warmup
            ? opt.lr * static_cast<float>(step) / std::max(1, opt.warmup)
            : opt.lr * (0.1f + 0.9f * 0.5f *
                (1.0f + std::cos(3.14159265358979f * progress)));
        const float loss = model.train_step(x.data(), y.data(), opt.batch, opt.seq,
                                            lr, opt.weight_decay);
        if (loss == mf::GPT::kStopped) break;
        if (loss < 0.0f || !std::isfinite(loss)) {
            std::fprintf(stderr, "non-finite loss at step %d; stopping.\n", step);
            break;
        }
        best = std::min(best, loss);
        const double elapsed = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - started).count();
        const double tps = elapsed > 0.0 ? step * opt.batch * opt.seq / elapsed : 0.0;

        if (step == 1 || step % 10 == 0) {
            std::printf("step %4d/%d loss %.4f best %.4f lr %.2e %.0f tok/s\n",
                        step, opt.steps, loss, best, lr, tps);
            std::fflush(stdout);
        }
        if (step % opt.every == 0 || step == opt.steps) {
            model.save_checkpoint(opt.out + "/last.mfck", true);
            if (loss <= best) model.save_checkpoint(opt.out + "/best.mfck", false);
            write_progress(opt.out, step, opt.steps, loss, best, tps,
                           (opt.steps - step) * elapsed / step, "running");
        }
    }

    model.save_checkpoint(opt.out + "/last.mfck", true);
    model.save_checkpoint(opt.out + "/best.mfck", false);
    write_progress(opt.out, static_cast<int>(model.step()), opt.steps, best, best, 0.0, 0.0,
                   g_stop ? "stopped" : "completed");
    std::printf("done. checkpoints in %s\n", opt.out.c_str());
    return 0;
}
