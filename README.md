# ModelForge

Train, quantize and test a small language model entirely on an Android phone.

ModelForge is a from-scratch GPT implementation (C++ engine, Kotlin/Compose UI).
There is no cloud training and no downloaded model weights: the app builds a
byte-level BPE tokenizer, trains the network on data from Hugging Face, runs
quantization-aware training, writes checkpoints, and exports packed 2/3/4/8-bit
models. The playground generates text from either the full-precision checkpoint
or a packed export and can share the packed file.

Status: v2.0.1 production build. The repository builds and tests on GitHub
Actions and publishes a signed release APK/AAB. See
[Honest limitations](#honest-limitations) before relying on model quality.

## Features

- **On-device architecture search** - `ConfigDeriver` finds a GPT shape within
  5% of a requested parameter target, capped by measured RAM/CPU limits.
- **From-scratch C++ engine** - LayerNorm, causal multi-head attention,
  tanh-GELU MLP, tied embeddings, manual backprop, AdamW, NEON dot products and
  a persistent thread pool. JNI handles are serialized and always freed.
- **Real training service** - foreground service with progress notification,
  pause/resume/stop actions, partial wake lock, battery/thermal auto-pause and
  auto-resume, boot resume, and atomic checkpoints. `last.mfck` keeps AdamW
  state for exact resume; `best.mfck` is weights-only for export.
- **Hugging Face datasets** - search, config/split/text-column discovery,
  license warnings, row/byte progress, weighted mixing, tokenizer training and a
  token-id cache.
- **Quantization-aware training (QAT)** - symmetric per-group 32 fake
  quantization with straight-through estimation on linear layers only.
- **Packed deployment format** - `MFPK` file with fp32 embeddings/norms/biases/
  tied head and packed linear weights plus fp32 scales; packed inference decodes
  groups on the fly instead of materializing fp32 matrices.
- **Compose UI** - interactive model wizard, live training dashboard with a
  loss chart and samples, model details, an identity check and a playground with
  prompt/temperature/top-k/top-p controls, measured tokens/second, export and
  share.
- **NVIDIA AI architect** - optional hosted planning using
  `minimaxai/minimax-m3.1-flash`. It proposes an overview, datasets, rows, mix
  weights and training settings; every dataset is checked against Hugging Face
  before download. The NVIDIA key is encrypted on device and never logged.
- **Three architecture paths** - traditional transformer (trainable), NCM
  (planning-only) and AI-designed architecture (planning-only), with honest
  gating so the wizard never claims to train an engine that does not exist.
- **Modalities and languages** - text, image, audio, video and documents can be
  planned; native training is still text-to-text. Tamil is supported in the
  language list and identity templates.
- **Encrypted credentials** - optional Hugging Face and NVIDIA keys in
  `EncryptedSharedPreferences`; no telemetry and no key logging.

## Honest limitations

- These are tiny models trained from scratch on a phone. Expect short, simple
  text with repetition and grammar mistakes, not chatbot-level answers.
- Only `arm64-v8a` is built. There is no x86 emulator or 32-bit build.
- 2-bit and 3-bit exports are experimental; 4-bit is the recommended default.
- Training time and memory vary a lot by phone. The app reports measured
  tokens/second, RAM and an ETA instead of guessing.
- The Compose screens are compiled into the release APK but have not yet been
  exercised on a physical device in this repository's CI. Native engine
  behaviour, checkpoint resume and packed inference are covered by host C++
  tests.
- NCM and the AI-designed architecture are planning paths only; they have no
  native trainer yet.

## Install (v2.0.1)

1. Open [Releases](https://github.com/CocoCopi/modelforge/releases) and download
   the latest `ModelForge-2.0.1-arm64.apk` directly.
2. Install it on an arm64 Android 8.0+ phone, for example
   `adb install ModelForge-2.0.1-arm64.apk`.
3. If an older development build is installed, uninstall it once. Older CI
   builds used an ephemeral debug certificate.
4. Allow the notification permission when prompted so training controls are
   visible in the notification shade.

The release APK is signed with the project's stable release key. The matching
AAB is attached for store distribution. Updating from v2.0.0 keeps app data.

### AI architect troubleshooting

The NVIDIA model may reason silently before it emits the plan, so the first
token can take 10-60 seconds. v2.0.1 shows elapsed time, stream-event count and a
**Cancel request** button. A 401/403 means the key is invalid or lacks access to
the configured model; a 404 means that model is not enabled for the account. If
the request finishes without plan text, retry; the app now explains this instead
of staying on "Designing..." indefinitely.

## Using the app

1. **New model** - choose a parameter target and quantization width. The
   architecture and counted parameter total update live.
2. **Data** - search Hugging Face (optionally save a read token first), pick a
   config, split and text columns, set rows and mix weight, then download.
3. **Training** - adjust steps if desired and start. Training runs in the
   foreground service; you can leave the app and use the notification actions.
4. **Watch** - the dashboard shows step progress, loss, best loss, tokens/second,
   RAM, ETA, battery/thermal state, a loss curve and generated samples.
5. **Playground** - generate from the best checkpoint, or export a packed model
   and switch the source to it. Use **Share file** to send the `.mfpk` file.

Models, datasets and checkpoints stay in the app's private storage and are
removed by **Delete model**.

## Build and test

Requirements: JDK 17, Android SDK platform 34, build-tools 34.0.0, NDK
`26.1.10909125`, CMake `3.22.1`.

```bash
# JVM unit tests (tokenizer, config derivation, training support, JSON, HF live test)
./gradlew testDebugUnitTest

# ARM64 debug APK
./gradlew assembleDebug

# Signed production APK and AAB (requires MF_KEYSTORE_PATH and passwords)
./gradlew assembleRelease bundleRelease

# Host C++ tests: config, gradient check, overfit, optimizer resume, packed export
cmake -S app/src/main/cpp -B build-native -DCMAKE_BUILD_TYPE=Release
cmake --build build-native -j"$(nproc)"
ctest --test-dir build-native --output-on-failure
```

CI (`.github/workflows/build.yml`) runs the host C++ tests, JVM tests,
`assembleDebug`, `assembleRelease` and `bundleRelease` on every push, verifies
the release signature, and uploads the debug APK, release APK, AAB and R8
mapping. The same workflow is the source of truth for "green".

## Measured results

Recorded from host builds in `PROGRESS.md`; they are not marketing estimates.

- Gradient check: 37 entries, worst relative error `0.0183` (tolerance 0.05).
- Overfit test (2x16 tokens, 600 AdamW steps): loss `3.445281 -> 0.000637`;
  greedy generation after checkpoint reload is identical.
- Optimizer resume: continues at step 601 and reproduces the reference loss
  exactly (`0.000635` vs `0.000635`).
- Packed test model (64-wide, 2 layers):

| Bits | Packed size | Max logit difference | Argmax mismatches |
| ---: | ----------: | -------------------: | ----------------: |
| 8 | 100,420 B | 0.0000002 | 0 |
| 4 | 67,652 B | 0.0000002 | 0 |
| 2 | 51,268 B | 0.0000001 | 0 |

The fp32 weights-only checkpoint for that model is about 289 KB, so the 4-bit
export is roughly 4.3x smaller.

## Architecture

```
app/src/main/
  cpp/
    core/gpt.{h,cpp}       GPT, backprop, AdamW, QAT, checkpoints, packed model
    jni.cpp                JNI surface (model, training, packed, config)
    tests/                 native config/gradient/overfit/packed tests
  java/com/modelforge/app/
    core/                  hardware detection, config derivation, manifest shape
    data/                  model store, manifests, tokenizer, Hugging Face client
    training/              foreground service, scheduler, mixer, token cache
    nativebridge/          NativeBridge, NativeModel, PackedModel wrappers
    ui/                    Compose screens and ModelForgeViewModel
```

Training state is persisted as JSON manifest/live-state files and binary
checkpoints. The packed file layout is mirrored exactly by
`ModelConfig.packedExportBytes(bits, group = 32)` and is unit tested.

## Datasets, licensing and privacy

- Dataset licenses are shown during selection. Licenses that look
  non-commercial or unknown trigger a warning; you are responsible for checking
  the dataset card and respecting its terms.
- The only network traffic is to Hugging Face for search, metadata, dataset rows
  and optional gated access. There is no analytics or crash upload.
- The Hugging Face token and NVIDIA API key, if saved, are encrypted on device
  and never written to logs or committed. All models and datasets remain in
  app-private storage. See `PRIVACY.md`.

## Known gaps

- Physical-device UI verification and timing measurements are still pending.
- No Compose instrumentation UI tests yet; native, JVM and Paparazzi logic have
  automated coverage.
- Export currently targets the best checkpoint; there is no separate
  fine-tune/merge workflow.
- NCM and AI-designed architectures remain planning-only until native engines
  and tests exist.

See `PROGRESS.md` for the phase-by-phase record and the exact CI run IDs.
