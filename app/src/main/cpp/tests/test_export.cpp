// Structural tests for the GGUF v3 and safetensors exporters.

#include "core/gpt.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
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

static std::vector<uint8_t> read_all(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    return std::vector<uint8_t>(std::istreambuf_iterator<char>(in),
                                std::istreambuf_iterator<char>());
}

struct Reader {
    const std::vector<uint8_t>& b;
    size_t p = 0;
    bool ok = true;

    uint16_t u16() {
        if (p + 2 > b.size()) { ok = false; return 0; }
        const uint16_t v = static_cast<uint16_t>(b[p] | (b[p + 1] << 8));
        p += 2;
        return v;
    }
    uint32_t u32() {
        if (p + 4 > b.size()) { ok = false; return 0; }
        uint32_t v = 0;
        for (int i = 0; i < 4; ++i) v |= static_cast<uint32_t>(b[p + i]) << (8 * i);
        p += 4;
        return v;
    }
    uint64_t u64() {
        if (p + 8 > b.size()) { ok = false; return 0; }
        uint64_t v = 0;
        for (int i = 0; i < 8; ++i) v |= static_cast<uint64_t>(b[p + i]) << (8 * i);
        p += 8;
        return v;
    }
    void skip(uint64_t n) {
        if (p + n > b.size()) { ok = false; p = b.size(); return; }
        p += static_cast<size_t>(n);
    }
    std::string str() {
        const uint64_t n = u64();
        if (!ok || p + n > b.size()) { ok = false; return {}; }
        std::string s(reinterpret_cast<const char*>(b.data() + p), static_cast<size_t>(n));
        p += static_cast<size_t>(n);
        return s;
    }
};

static void skip_gguf_value(Reader& r, uint32_t type) {
    switch (type) {
        case 0: case 1: r.skip(1); break;             // u8 / i8
        case 2: case 3: r.skip(2); break;             // u16 / i16
        case 4: case 5: case 6: r.skip(4); break;     // u32 / i32 / f32
        case 7: r.skip(1); break;                     // bool
        case 8: r.str(); break;                       // string
        case 10: case 11: case 12: r.skip(8); break;  // u64 / i64 / f64
        case 9: {                                     // array
            const uint32_t elem = r.u32();
            const uint64_t count = r.u64();
            for (uint64_t i = 0; i < count && r.ok; ++i) skip_gguf_value(r, elem);
            break;
        }
        default: r.ok = false;
    }
}

int main() {
    ModelConfig cfg;
    cfg.vocab = 128;
    cfg.context = 32;
    cfg.d_model = 32;
    cfg.layers = 1;
    cfg.heads = 4;
    cfg.d_ff = 64;

    GPT model;
    check(model.init(cfg, 5), "export model init");
    std::vector<int> tokens(16), targets(16);
    for (int i = 0; i < 16; ++i) {
        tokens[i] = i % cfg.vocab;
        targets[i] = (i + 1) % cfg.vocab;
    }
    check(model.train_step(tokens.data(), targets.data(), 2, 8, 1e-3f) >= 0.0f,
          "export model trains");

    const uint64_t expected_tensors = 3 + 16ull * cfg.layers;

    for (int type : {0, 1, 2, 8}) {
        const std::string path = "/tmp/mf_test_export_" + std::to_string(type) + ".gguf";
        check(model.save_gguf(path, type), "GGUF export succeeds");
        const auto bytes = read_all(path);
        check(bytes.size() > 24, "GGUF file has content");

        Reader r{bytes};
        char magic[4];
        std::memcpy(magic, bytes.data(), 4);
        check(std::memcmp(magic, "GGUF", 4) == 0, "GGUF magic");
        r.p = 4;
        check(r.u32() == 3, "GGUF version 3");
        check(r.u64() == expected_tensors, "GGUF tensor count");
        const uint64_t kv = r.u64();
        check(kv >= 10, "GGUF metadata present");
        for (uint64_t i = 0; i < kv && r.ok; ++i) {
            r.str();
            const uint32_t t = r.u32();
            skip_gguf_value(r, t);
        }
        check(r.ok, "GGUF metadata parses");

        uint64_t last_end = 0;
        bool offsets_ok = true;
        uint64_t previous = 0;
        for (uint64_t i = 0; i < expected_tensors && r.ok; ++i) {
            r.str();
            const uint32_t dims = r.u32();
            for (uint32_t d = 0; d < dims; ++d) r.u64();
            const uint32_t tensor_type = r.u32();
            const uint64_t offset = r.u64();
            if (i > 0 && offset <= previous) offsets_ok = false;
            previous = offset;
            last_end = offset;
            check(tensor_type == static_cast<uint32_t>(type) ||
                      (type >= 2 && tensor_type == 1),
                  "GGUF tensor type is requested type or F16 fallback");
        }
        check(r.ok && offsets_ok, "GGUF tensor infos parse with ordered offsets");
        const uint64_t data_start = (r.p + 31) & ~uint64_t{31};
        check(data_start < bytes.size(), "GGUF data section starts in file");
        check(last_end < bytes.size(), "GGUF tensor data fits in file");
        std::remove(path.c_str());
    }

    // Safetensors: header is an 8-byte little-endian length plus JSON.
    for (bool fp16 : {false, true}) {
        const std::string path = std::string("/tmp/mf_test_export_") +
                                 (fp16 ? "f16" : "f32") + ".safetensors";
        check(model.save_safetensors(path, fp16), "safetensors export succeeds");
        const auto bytes = read_all(path);
        check(bytes.size() > 16, "safetensors file has content");
        Reader r{bytes};
        const uint64_t header = r.u64();
        check(header > 0 && header + 8 < bytes.size(), "safetensors header length");
        const std::string json(reinterpret_cast<const char*>(bytes.data() + 8),
                               static_cast<size_t>(header));
        check(json.find("\"model.embed_tokens.weight\"") != std::string::npos,
              "safetensors contains embedding tensor");
        check(json.find("\"model.layers.0.self_attn.q_proj.weight\"") != std::string::npos,
              "safetensors contains attention tensor");
        check(json.find(fp16 ? "\"F16\"" : "\"F32\"") != std::string::npos,
              "safetensors dtype matches request");
        check(json.front() == '{' && json.back() == '}', "safetensors JSON is an object");

        const uint64_t data_bytes = bytes.size() - 8 - header;
        uint64_t params = static_cast<uint64_t>(cfg.param_count());
        check(data_bytes == params * (fp16 ? 2 : 4),
              "safetensors data length matches parameter count");
        std::remove(path.c_str());
    }

    if (failures == 0) std::printf("all export tests passed\n");
    return failures == 0 ? 0 : 1;
}
