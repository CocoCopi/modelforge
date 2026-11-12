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

Status: DONE - CI run 43150152809 green (host CTest + unit tests + debug APK).

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

Status: DONE - CI run 43150773713 green.

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

## Phase 4 - on-device training service, checkpoints and resume

Status: DONE - CI run 43152727996 green (host native tests incl. optimizer resume,
JVM tests and debug APK).

- Foreground service (`TrainingService`, `dataSync` type) with a low-importance
  notification showing step progress, measured loss, tokens/second, RAM, battery,
  ETA and thermal state. Notification actions pause/resume and stop training.
- Partial wake lock only while actually training. The service pauses itself when
  battery drops below 15% or the thermal status reaches SEVERE, releases the wake
  lock, and resumes automatically when conditions are safe again.
- Durable resume: `active_training.json`, live state (`training_state.json`),
  `START_STICKY`, and a boot receiver (also handles package updates) restart an
  interrupted run when `autoResume` is enabled.
- Training pipeline: weighted dataset mixing, deterministic BPE training cache,
  token-id cache (`tokens.bin` + signature manifest), linear warmup + cosine LR
  schedule, automatic batch lowering from a RAM estimate, and a sample generation
  every N steps.
- Checkpoints every N steps: `last.mfck` includes AdamW state for exact resume;
  `best.mfck` is weights-only for export. Save is atomic. If a native training
  step fails (JNI now returns a negative error sentinel), the batch is halved and
  training reloads the last checkpoint; repeated failure produces a readable error
  with a suggested fix.
- Persistence models (`ModelManifest`, `ModelStore`) and JVM tests for LR schedule,
  memory guard, dataset mixer, token cache, JSON round trips and the live state bus.

Measured locally with the host C++ build: the new optimizer-resume check loads
`last.mfck`, continues at step 601, and reproduces the reference loss exactly
(0.000635 vs 0.000635). Gradient check and overfit numbers are unchanged.

Not yet covered here: the Compose training screen (Phase 6) is what will start
this service and render the live state; this phase verifies the service, native
resume path and support logic.

## Phase 5 - quantized packed export and inference

Status: implemented, awaiting CI green.

- `GPT::save_packed` writes a deployment file (`MFPK`): config header, fp32
  embeddings/norms/biases/tied embedding head, and 2/3/4/8-bit symmetric
  per-group quantized linear weights with fp32 scales. Layout matches
  `ModelConfig.packedExportBytes` exactly (unit tested).
- `PackedModel` loads the packed file and runs forward/greedy/sampling inference
  without dequantizing whole weight matrices to fp32 at rest. Linear products
  decode the packed values per group and apply the fp32 scale.
- JNI and Kotlin wrappers: `NativeModel.savePacked`, `NativeBridge` packed
  functions, and `PackedModel` (load/config/generate/close). `ModelStore` owns
  `packed/model_q<bits>.mfpk`. The export UI itself lands in Phase 7.
- Invalid bit widths are rejected; the C++ test also checks the exact file size
  and that packed logits match QAT reference logits.

Measured locally (host g++ build, same 64-wide 2-layer test model):

- 8-bit: 100,420 bytes, max logit difference 0.0000002, 0 argmax mismatches.
- 4-bit: 67,652 bytes, max logit difference 0.0000002, 0 argmax mismatches.
- 2-bit: 51,268 bytes, max logit difference 0.0000001, 0 argmax mismatches.
- Greedy generation after a packed 8-bit load matched the QAT model exactly.
- The fp32 weights-only checkpoint for the same model would be about 289 KB, so
  the 4-bit export is roughly 4.3x smaller.
