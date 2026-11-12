# ModelForge progress log

This file records, phase by phase, what actually works, what does not, and the
state of the CI run. Only measured facts belong here.

## Environment note

- The `GITHUB_TOKEN` environment variable was **not set** in the build shell.
- The pre-authenticated `gh` CLI was available for account **CocoCopi** with the
  scopes `repo` and `workflow`. Its OAuth token was used through git's credential
  helper and the REST API. The token was never printed, written to a file, or
  committed.
- Repository: https://github.com/CocoCopi/modelforge

## Phase 1 - skeleton, CI, hardware detection, config derivation

Status: DONE - CI run 43147652432 green (unit tests + host native tests + debug APK artifact).

- Gradle project with AGP 8.5.2, Kotlin 1.9.24, Compose BOM 2024.06.00,
  compileSdk 34, minSdk 26, arm64-v8a only.
- CI workflow `.github/workflows/build.yml`: JDK 17, Android SDK + NDK
  26.1.10909125 + CMake 3.22.1, host C++ tests, Kotlin unit tests, debug APK
  artifact `modelforge-debug-apk`, Gradle/NDK/CMake caching.
- Hardware detection (`HardwareDetector`): RAM, memory class, low-RAM flag,
  CPU cores, max frequency, battery, thermal status, and a RAM/CPU-derived
  maximum trainable parameter count with a human-readable reason.
- Config derivation (`ConfigDeriver`): searches real architectures, counts
  parameters with the same formula as C++, ranks sensible shapes, targets
  within 5% for 1M, 10M and 50M (unit tested).
- Native scaffold: host-buildable C++ config + test, JNI library target.

CI fixes made during this phase: the removed `tools` SDK package was no longer
installable (setup-android now only installs `platform-tools`), and
kotlinx.serialization 1.7.1 was downgraded to 1.6.3 because it requires Kotlin 2.0.

Nothing in this phase claims to train or generate text yet.

## Phase 2 - C++ engine + JNI bridge

Status: implemented, awaiting CI green.

- From-scratch GPT: token/position embeddings, causal multi-head attention,
  MLP with tanh-GELU, LayerNorm, tied output head, manual backward pass,
  AdamW with bias correction, weight decay on matrices/embeddings only.
- ARM NEON dot products on arm64, persistent worker pool with work-size
  thresholds (small matrices stay single-threaded to avoid thread overhead).
- QAT hook: symmetric per-group fake quantization with a straight-through
  estimator on linear layers. Embeddings, norms, biases and the tied head stay
  high precision. Packed file export/inference lands in Phase 5.
- Checkpoints: atomic write, fp32 weights plus optional AdamW m/v and step;
  architecture is validated on load. Weights-only load reproduces greedy output.
- JNI bridge with create/free/train/generate/checkpoint/QAT/config calls.
- Native tests added to CTest: gradient check and single-batch overfit.

Measured locally (aarch64 host build, g++ -O2):

- gradient check: 37 parameter entries, worst relative error 0.0183 (tolerance 0.05).
- overfit: batch of 2x16 tokens, loss 3.445281 -> 0.000637 in 600 AdamW steps
  (lr 0.003, no weight decay). Greedy generation after checkpoint reload is
  identical to the in-memory model.
- The overfit test took about 30 s on this proot after adding a persistent
  thread pool and small-work serial thresholds.

## Phase 3 - tokenizer, Hugging Face search and dataset download

Status: implemented, awaiting CI green.

- Byte-level BPE tokenizer in Kotlin: 256 byte tokens plus learned merges,
  whitespace-preserving chunk training with frequency weighting, deterministic
  merges, JSON save/load, exact decode round trip. Training work is bounded by a
  symbol budget so huge vocabularies do not run forever on a phone.
- Hugging Face Hub search and details via Retrofit + OkHttp +
  kotlinx.serialization, with an optional Bearer token interceptor that never
  logs the token.
- datasets-server info/splits/rows: discovers configs, splits and string
  columns, pages through rows, extracts selected text columns, writes a UTF-8
  cache file and reports rows/bytes/preview. Readable errors for 401/403/404/429
  and timeouts.
- License extraction from cardData/tags and a non-commercial warning helper.
- EncryptedSharedPreferences token store (no plaintext fallback).
- JVM tests cover BPE round trips/determinism plus a live integration test
  against roneneldan/TinyStories (search, license, structure, 3-row download).
