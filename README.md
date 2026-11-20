# ModelForge

Train, quantize and test a small language model entirely on an Android phone.

ModelForge is a from-scratch GPT implementation (C++ engine, Kotlin/Compose UI).
There is no cloud training and no downloaded model weights: the app builds a
byte-level BPE tokenizer, trains the network on data from Hugging Face, runs
quantization-aware training, writes checkpoints, and exports packed 2/3/4/8-bit
models. The playground generates text from either the full-precision checkpoint
or a packed export and can share the packed file. Models can also be exported as
**GGUF** (F32/F16/Q4_0/Q8_0) and **safetensors** (F32/F16) for tooling and
archival.

Status: v2.4.0 production build. The repository builds and tests on GitHub
Actions and publishes a signed release APK/AAB. See
[Honest limitations](#honest-limitations) before relying on model quality.

Licence: Apache License 2.0 (see `LICENSE` and `NOTICE`).

## Features

- **On-device architecture search** - `ConfigDeriver` finds a GPT shape within
  5% of a requested parameter target, capped by measured RAM/CPU limits.
- **From-scratch C++ engine** - LayerNorm, causal multi-head attention,
  tanh-GELU MLP, tied embeddings, manual backprop, AdamW, NEON dot products and
  a persistent thread pool. The three matrix-multiply kernels are tiled and
  vectorized: the forward pass blocks output and row dimensions, and both
  gradient kernels keep their accumulators in vector registers (an 8x8 tile
  for the weight gradients, a 4x8 tile for the input gradients) instead of
  streaming a scratch buffer for every row. The NEON dot product uses four
  independent accumulators so the FMA pipeline is not latency-bound. Measured
  on the development phone: about 1.3x faster per step for a 1M-parameter
  model. JNI handles are serialized and always freed.
- **Real training service** - foreground service with progress notification,
  pause/resume/stop actions, partial wake lock, battery/thermal auto-pause and
  auto-resume, boot resume, and atomic checkpoints. `last.mfck` keeps AdamW
  state for exact resume; `best.mfck` is weights-only for export. Training
  keeps running in the background until you stop it, and the service serializes
  every command so a stale run can never clobber a new one.
- **Stop that actually stops** - the stop action reaches into the running
  native step and aborts it at the next layer boundary instead of waiting for
  the step to finish, so a large model stops in seconds rather than minutes.
  The aborted step never reaches AdamW, so the saved weights stay consistent
  with the last checkpoint. Stopping also clears the active marker immediately,
  so a stopped run is never auto-resumed when the app is reopened.
- **A stopped model is immediately testable** - stopping always leaves a
  loadable checkpoint: if the weights-only `best.mfck` does not exist yet (a
  run stopped before its first periodic checkpoint), it is written from the
  final weights. The playground, the identity check and every export fall back
  to `last.mfck` when `best.mfck` is missing, so models trained with v2.3.0 can
  be opened and tested without retraining. The stop message now says whether
  the checkpoint was really written instead of always claiming it was.
- **Hugging Face datasets** - search, config/split/text-column discovery,
  exact split sizes, license warnings, row/byte progress, weighted mixing,
  tokenizer training and a token-id cache. The row field has no 5,000-row cap:
  enter any count or use **Entire split**. Downloads use the repository's own
  JSONL/CSV/TXT/JSON data files through resolver URLs, so a whole split is one
  or two requests instead of one request per 100 rows. The dataset viewer is
  only a fallback for parquet-only repositories.
- **Inside your rate limit** - every Hugging Face request passes through a
  shared, adaptive pacer: requests are spaced to fit the plan's 5-minute quota,
  a 429 pauses every worker together and honours `Retry-After` or the
  `RateLimit` reset header, and the pace recovers after sustained success. The
  app stops bursting into 429s and tells you when it is deliberately slowing
  down instead of failing the download.
- **Corpus sized to the run** - the amount of text loaded into memory is
  derived from the step count and the free RAM (up to 128M characters) instead
  of a fixed 4M-character clamp, so a long run no longer replays the same few
  thousand rows every epoch.
- **Up to 1,000,000-token context** - a logarithmic slider spans 128 to 1M
  tokens with quick presets. Positions use fixed sinusoidal values, so context
  length adds no parameters and no context-sized position table.
- **User-set training steps** - enter an exact optimizer step count, choose a
  preset, or let the AI architect propose one. There is no maximum step count,
  and the review screen shows the estimated seconds per step and the total
  on-device compute for the chosen size before training starts.
- **AI dataset diversity** - the architect is told to pick distinct,
  complementary repositories, and the app enforces it: repeated dataset ids are
  collapsed, fewer than three distinct sources are topped up with verified
  complementary datasets, and each AI-chosen dataset is retried three times
  automatically before a visible **Retry N failed dataset(s)** button appears.
- **Quantization-aware training (QAT)** - symmetric per-group 32 fake
  quantization with straight-through estimation on linear layers only.
- **Packed deployment format** - `MFPK` file with fp32 embeddings/norms/biases/
  tied head and packed linear weights plus fp32 scales; packed inference decodes
  groups on the fly instead of materializing fp32 matrices.
- **Interchange exports** - GGUF v3 (`F32`, `F16`, `Q4_0`, `Q8_0`) and
  safetensors (`F32`, `F16`) written by the native engine from the best
  checkpoint. GGUF carries an explicit `modelforge` architecture tag and
  metadata describing the fixed sinusoidal positions and LayerNorm, so the
  files are honest about what they are.
- **Compose UI** - interactive model wizard, live training dashboard with a
  loss chart and samples, model details, an identity check and a playground with
  prompt/temperature/top-k/top-p controls, measured tokens/second, export and
  share.
- **NVIDIA AI autopilot** - optional hosted planning using
  `minimaxai/minimax-m3.1-flash`. The AI runs a bounded tool loop: it searches
  Hugging Face, inspects real configs/splits/text columns, then applies the
  training recipe and downloads verified datasets into the mix automatically.
  Unavailable or invented dataset configs are rejected with a visible reason; if
  every AI suggestion fails, the app tries a small list of verified starter
  datasets. Dataset metadata is cached, row pages are fetched in parallel and
  rate limits are retried with backoff, and a **Retry dataset downloads** action
  resumes without re-running the AI. NVIDIA `429`/`5xx` responses are also retried automatically.
  The NVIDIA key is encrypted on device and never logged. Tool replies are parsed
  tolerantly, and a direct plan is retried if the model ignores the tool protocol.
- **Three architecture paths** - traditional transformer (trainable), NCM
  (planning-only) and AI-designed architecture (planning-only), with honest
  gating so the wizard never claims to train an engine that does not exist.
- **Modalities and languages** - text, image, audio, video and documents can be
  planned; native training is still text-to-text. Tamil is supported in the
  language list and identity templates.
- **Encrypted credentials** - optional Hugging Face and NVIDIA keys in
  `EncryptedSharedPreferences`; no telemetry and no key logging.
- **Diagnostics** - explicit Hugging Face/NVIDIA credential checks, device and
  storage facts, redacted structured logs, a local last-crash report, and
  copy/share actions for support. Reports are only shared when you choose to.

## Honest limitations

- These are tiny models trained from scratch on a phone. Expect short, simple
  text with repetition and grammar mistakes, not chatbot-level answers.
- Release APKs are `arm64-v8a` only. Debug/CI builds also include `x86_64` so the emulator test job can run; no 32-bit build is produced.
- 2-bit and 3-bit exports are experimental; 4-bit is the recommended default.
  16-bit and 32-bit are **not** packed-training widths; use the GGUF F16/F32 or
  safetensors F16/F32 exports for those precisions.
- GGUF exports are archival/tooling containers, not llama.cpp-compatible
  models: ModelForge uses fixed sinusoidal positions and LayerNorm, while
  llama.cpp expects RoPE and RMSNorm. The file records this in its metadata
  rather than pretending the two are equivalent.
- ONNX, TensorFlow Lite and CoreML exports are **not** implemented. The app
  does not pretend to write them.
- Training time and memory vary a lot by phone. The app reports measured
  tokens/second, RAM and an ETA instead of guessing, and the review screen
  shows an approximate per-step and total compute estimate derived from the
  model size and the device's core count.
- The Compose screens are compiled into the release APK. CI now launches them
  on an x86_64 emulator and opens Diagnostics; physical-device coverage is still
  needed for phone-specific performance and thermal behaviour. Native engine
  behaviour, checkpoint resume and packed inference are covered by host C++
  tests.
- NCM and the AI-designed architecture are planning paths only; they have no
  native trainer yet.
- 1,000,000 tokens is a positional-capacity ceiling. Training still uses the
  sequence length chosen for the device (usually 128-256 tokens), and full
  attention costs O(tokens²), so practical inference on a phone is far shorter.
- Checkpoints and packed files written by older versions still load. Learned
  position tables from those files are ignored in favour of the new sinusoidal
  positions.

## Install (v2.4.0)

1. Open [Releases](https://github.com/CocoCopi/modelforge/releases) and download
   the latest `ModelForge-2.4.0-arm64.apk` directly.
2. Install it on an arm64 Android 8.0+ phone, for example
   `adb install ModelForge-2.4.0-arm64.apk`.
3. If an older development build is installed, uninstall it once. Older CI
   builds used an ephemeral debug certificate.
4. Allow the notification permission when prompted so training controls are
   visible in the notification shade.

The release APK is signed with the project's stable release key, so v2.4.0
updates over v2.3.0 in place and keeps your models, datasets and checkpoints.
The matching AAB is attached for store distribution.

### AI architect troubleshooting

v2.1.0 uses `reasoning_effort: none`, shows elapsed time, stream-event count and
the current dataset being checked/downloaded, and offers **Cancel request**. A
timeout with 0 events usually means the phone cannot reach
`integrate.api.nvidia.com`; try switching Wi-Fi/mobile data or VPN. A 401/403
means the key is invalid or lacks access to the configured model; a 404 means
that model is not enabled for the account. If the AI suggests datasets that
cannot be verified, they are skipped and listed with the reason. When every
suggestion fails, the app retries with verified starter datasets; if those also
fail, check the connection or add data manually from the dataset step.

An NVIDIA **HTTP 503** with `content safety screening is temporarily
unavailable` is a server-side outage, not a bad key or prompt. v2.1.0 retries
`429` and all `5xx` responses automatically, honoring `Retry-After` and
otherwise waiting 5s, 15s and 30s. The AI screen shows the wait. If all four
attempts fail, wait a minute and tap **Ask AI to design** again.

If downloads hit **HTTP 429**, v2.1.0 retries with `Retry-After`/exponential
backoff and caches dataset metadata. When the limit persists it stops instead of
hammering the endpoint, keeps the AI plan, and shows **Retry dataset
downloads**. Wait a minute, add a Hugging Face token on Home for higher limits,
or switch network, then retry. Page requests run in parallel, so downloads are
bounded by the network and the split size rather than a fixed row cap.

## Using the app

1. **New model** - choose a parameter target and quantization width. The
   architecture and counted parameter total update live.
2. **AI design** - add an NVIDIA key and tap **Ask AI to design**. The architect
   searches and inspects Hugging Face through bounded tools, chooses the training
   recipe and verified datasets, applies the settings, downloads the text data
   and takes you to review. Manual dataset search is still
   available if you want to override its choices.
3. **Training** - review the AI-chosen settings and start. Training runs in the
   foreground service; you can leave the app and use the notification actions.
4. **Watch** - the dashboard shows step progress, loss, best loss, tokens/second,
   RAM, ETA, battery/thermal state, a loss curve and generated samples.
5. **Playground** - generate from the best checkpoint, or export a packed model
   and switch the source to it. Use **Share file** to send the `.mfpk` file.
   **Interchange export** writes GGUF (`F32`/`F16`/`Q4_0`/`Q8_0`) and
   safetensors (`F32`/`F16`) files from the best checkpoint and shares them the
   same way.

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
mapping. The same workflow is the source of truth for "green". The release job
is skipped for pull requests (forks have no signing secret) and fails loudly if
the signing secret is missing, so an unsigned release can never be published.

## Measured results

Recorded from host builds in `PROGRESS.md`; they are not marketing estimates.

- Gradient check: 37 entries, worst relative error `0.0183` (tolerance 0.05).
- Overfit test (2x16 tokens, 600 AdamW steps): loss `3.445281 -> 0.000637`;
  greedy generation after checkpoint reload is identical.
- Optimizer resume: continues at step 601 and reproduces the reference loss
  exactly (`0.000635` vs `0.000635`).
- Training step time on an aarch64 Cortex-A55 host (4 usable cores, 256
  tokens per step, `batch 2 x seq 128`):

  | Model | Parameters | Before | After |
  | --- | ---: | ---: | ---: |
  | 4 layers, d=128 | 1.06M | 1.15 s/step | 0.54 s/step |
  | 15 layers, d=768 | 112.6M | 205 s/step | 14.6 s/step |

  The forward and backward matrix multiplies were the bottleneck: the old
  gradient kernels walked both operands with a stride and could not vectorize.
  They are now blocked and contiguous, and the NEON dot product uses four
  independent accumulators.
- Packed test model (64-wide, 2 layers):

| Bits | Packed size | Max logit difference | Argmax mismatches |
| ---: | ----------: | -------------------: | ----------------: |
| 8 | 100,420 B | 0.0000002 | 0 |
| 4 | 67,652 B | 0.0000002 | 0 |
| 2 | 51,268 B | 0.0000001 | 0 |

The fp32 weights-only checkpoint for that model is about 289 KB, so the 4-bit
export is roughly 4.3x smaller.

## Release v2.3.0

- `ModelForge-2.3.0-arm64.apk` (arm64-v8a, minSdk 26, 3,574,627 bytes)
  SHA-256 `0a9cd688c08c4c5343e0c958984ff5e3bd09421339330f01fbc7c4f0122b6185`
- `ModelForge-2.3.0.aab` (5,409,147 bytes)
  SHA-256 `3ce4c7891a33e8d13d7bfcd08027c14e08fa158ffba148f1fedc3ced9d928fee`
- Signed with the stable release key (certificate unchanged from v2.2.0), so it
  updates in place.
- CI run 44587657299 green: host C++ tests, JVM tests, Paparazzi, debug APK,
  signed release APK/AAB and the x86_64 emulator job.

## Architecture

```
app/src/main/
  cpp/
    core/gpt.{h,cpp}       GPT, backprop, AdamW, QAT, checkpoints, packed model,
                           GGUF and safetensors exporters
    jni.cpp                JNI surface (model, training, packed, exports, config)
    tests/                 native config/gradient/overfit/packed/concurrency/export
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

### Why an authenticated download can still be rate limited

Hugging Face limits Hub API calls over a fixed 5-minute window: 500 for an
anonymous IP, 1,000 for a signed-in free account, 2,500 for PRO. That is 3.3
requests per second for a free account.

The dataset viewer's `/rows` endpoint returns at most 100 rows per request, so a
132,924-row download is about 1,330 requests. Sending 24 of those at once takes
roughly 20 seconds on a normal connection - around 20,000 requests per 5-minute
window, or 20x the quota - so the server correctly answers with 429s. Retrying
each in-flight request on its own then produced a visible 429 every five
seconds. The token was never the problem.

ModelForge now does two things about that:

1. **It downloads the data files directly.** Repositories that ship JSONL, CSV,
   TXT or JSON are read through resolver URLs, which have their own much larger
   budget (5,000 per 5 minutes). A whole split is one or two requests instead of
   1,330, and long cells are no longer truncated by the preview API. The viewer
   remains the fallback for parquet-only repositories.
2. **It paces what is left.** A single adaptive gate per host spaces requests to
   the plan's quota, ramps up slowly after sustained success, and on a 429 pauses
   every worker together using `Retry-After` or the `RateLimit` reset header.
   Nothing retries in parallel behind your back.

If you still see a rate-limit message, it is because the quota is shared with
anything else using the same token or IP in that 5-minute window. The app waits
and retries instead of failing the download.

## Known gaps

- Physical-device UI verification and timing measurements are still pending.
- No Compose instrumentation UI tests yet; native, JVM and Paparazzi logic have
  automated coverage.
- Export currently targets the best checkpoint; there is no separate
  fine-tune/merge workflow.
- No ONNX, TensorFlow Lite or CoreML export; the implemented set is MFPK,
  GGUF and safetensors.
- The AI dataset agent uses a bounded tool loop (up to 12 model calls and a
  15-minute budget) so a runaway plan cannot hang the app.
- NCM and AI-designed architectures remain planning-only until native engines
  and tests exist.

See `PROGRESS.md` for the phase-by-phase record and the exact CI run IDs.
