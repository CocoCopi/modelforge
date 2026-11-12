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

Status: DONE - CI run 43155091694 green. The first attempt failed only because
jni.cpp needed `using mf::PackedModel`; host tests had already passed.

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

## Phase 6 - Compose wizard, training dashboard and model details

Status: DONE - CI run 43156679474 green (host native tests, JVM tests and debug
APK). The first attempt failed on a Kotlin delegated-state smart cast in the
training screen; that was fixed without changing behaviour.

- Real Compose app: `MainActivity` now hosts `ModelForgeRoot`, a manual screen
  state machine (Home, Wizard, Training, ModelDetail) with system-back handling;
  no placeholder Phase 1 screen remains.
- Home lists saved models with live status/progress/loss, the detected hardware
  limit, battery/thermal state and an encrypted Hugging Face token dialog.
- Four-step new-model wizard: target-size chips with derived architecture and
  counted parameters, 4/8/32-bit selection, Hugging Face dataset search,
  config/split/column selection, row and mix-weight sliders, download progress,
  dataset licence warnings, training-plan sliders and a review screen that
  writes the manifest and starts the foreground service.
- Training dashboard polls the live state once per second: progress, elapsed
  time, ETA, loss/best loss/tokens per second/RAM, loss history chart, sample
  output, device safety state and pause/resume/stop controls. Training continues
  in the background via the existing notification actions.
- Model detail shows architecture, training metrics, datasets, full-precision
  and packed-size estimates, checkpoint sizes, a confirmation-guarded delete,
  and links back into live training. Deleting an actively training model is
  blocked.
- UI numbers stay honest: sizes are exact layout estimates, quality text comes
  from measured training state, and the app states that these tiny from-scratch
  models produce simple text.

Not yet covered here: device-level Compose UI tests. The wizard and dashboards
were verified by compilation and the ARM64 debug APK build; an emulator/device
pass is still useful before release.

## Phase 7 - playground and packed export

Status: DONE - CI run 43157495082 green (host native tests, JVM tests and debug
APK).

- Playground screen opened from a model's details. It can generate from the
  full-precision `best.mfck` checkpoint or from a packed `MFPK` file, with a
  clear source toggle and 2/3/4/8-bit selection.
- Export runs on a background dispatcher: loads the best checkpoint, calls
  `NativeModel.savePacked` into `packed/model_q<bits>.mfpk`, reports the real
  file size, and switches the playground to the packed source. The packed file
  can be sent through Android's share sheet via the existing FileProvider.
- Generation controls: prompt (multi-line), max new tokens, temperature, top-k
  and top-p. Prompt tokens are trimmed to the model context, generation is timed,
  and the screen shows the decoded text, generated token count and measured
  tokens/second.
- Readable guards for missing tokenizer, missing checkpoint or missing packed
  file; native handles are closed in `finally` blocks after each operation.
- The model detail screen links into the playground and no longer claims the
  feature is future work.

Measured facts reused here: packed export sizes/logit agreement are the Phase 5
numbers. This phase adds the UI and the export/generation wiring; the packed
inference path itself is the same tested native code.

Not yet covered here: a device-level playground run. The screen compiles into the
ARM64 debug APK, but actual on-device generation timing and memory will vary by
phone and should be recorded when a physical device is available.

## Phase 8 - documentation and development release

Status: DONE - CI run 43157828229 green. Prerelease `v1.0.0` published:
https://github.com/CocoCopi/modelforge/releases/tag/v1.0.0

- README rewritten with the real feature set, install steps, build and test
  commands, measured results, architecture, dataset/licence rules, privacy and
  an explicit limitations section. It states that the current APK is debug
  signed and that the Compose UI has not yet been run on a physical device.
- Android 13+ now requests `POST_NOTIFICATIONS` at startup so foreground
  training controls are actually visible when the user grants it.
- The CI debug APK (19,250,797 bytes, arm64-v8a only) was downloaded from the
  green run and attached to the prerelease. Release notes repeat the debug
  signing, device-testing and tiny-model caveats.
- No release keystore or Play Store metadata is configured. The prerelease is a
  development build, not a signed production release.
- Remaining gaps are documented rather than hidden: physical-device UI
  verification, release signing/proguard, an app icon, and Compose UI tests.

## v2 Phase 1 - design system and screenshot gate

Status: DONE - CI run 43165048507 green (host native tests, JVM tests,
`verifyPaparazziDebug`, debug APK and screenshot artifacts).

- Replaced the generic dark Material look with the light "maker's notebook"
  system: paper/surface/ink/hairline tokens, one ember accent, verdigris and
  ochre annotations, 4-6 dp corners, 1 dp rules and an 8 dp spacing rhythm.
- Bundled Fraunces (display and live numbers), Instrument Sans (body/UI) and
  JetBrains Mono (ledger data) as static TTFs with OFL licences in `res/raw`.
  Typography, shapes and the full light Material colour scheme live in
  `ui/theme/Theme.kt` and `ui/theme/Type.kt`.
- Added signature components in `ui/design/Signature.kt`: parameter ruler with
  decade haptics, bit-grain selector with real sizes, wizard ruler, ledger
  chart, typographic empty state, rectangular buttons, status tags, ledger stats
  and notebook cards.
- Restyled Home, Wizard, Training, Model detail and Playground to the new
  components. Removed the old `ForgeAccent`/`ForgeBlue` colour aliases so screens
  use the token object directly.
- Added a scrollable design proof screen reachable from Home, and a Paparazzi
  JVM screenshot harness pinned to 1.3.4 to match Kotlin 1.9.24. The checked-in
  golden is verified with `verifyPaparazziDebug` in CI; snapshots and failure
  output are uploaded as artifacts.
- AA decisions are enforced in `scripts/check_contrast.py`: body ember text uses
  `emberText`, warning text uses `ochreInk`, and `ember` is reserved for large
  type, icons and marks. All documented pairs pass.
- The screenshot is layoutlib/JVM output, not a physical-device render. Device
  UI verification remains an explicit gap.

Remaining v2 work starts with the architecture registry, model config and the
14-step one-question-per-screen wizard, then identity, speed/reasoning and the
new architecture/modality paths.

## v2 Phase 2 - registry, 14-step wizard and identity data

Status: DONE - CI run 43173763360 green (host native tests, JVM tests,
`verifyPaparazziDebug`, debug APK and artifacts).

- Added `core/ArchitectureRegistry.kt`: a plugin interface with name,
  modalities, parameter derivation, train/infer hooks and dataset format, plus
  Stable/Experimental/Planned statuses. GPT text is Stable; NCM, image, audio,
  CTC speech, captioning, text-to-image, TTS and video are listed as Planned
  with honest reasons. `ModelConfig.architectureId` selects the plugin.
- Added `ModelManifest.creator`, `languages` and `BehaviorConfig`, and a
  `WizardDraft` SharedPreferences store so the wizard resumes after process
  death or navigation. Starting training and discarding clear the draft.
- Rebuilt the wizard as 14 one-question-per-screen steps with Back/Next, a ruler
  progress mark, per-step validation, architecture cards, compression bit-grain
  selector, dataset search/mix and auto-added Identity/Reasoning/Tool-call rows.
  Unsupported choices are disabled with reasons instead of hidden. "Start
  training" exists only on the review step.
- Added `IdentityDataGenerator`: deterministic, offline identity conversations
  in all offered languages, with openers and two-turn variants for variety,
  a system prompt, and a 3-5% character budget. Training mixes this corpus near
  the end of the dataset, and the token-cache signature now includes identity
  and behaviour so stale caches are rejected. `IdentityDataGeneratorTest` checks
  300+ conversations, variety, language coverage, model/creator naming, no
  other-model claims, budgets and the system prompt.

Still not wired: the 20-question post-training identity check and one-tap
reinforce, preset save/load, runtime speed/reasoning switches, and the C++
architecture plugins behind the new registry entries.

## v2 Phase 3 - response speed and reasoning modes

Status: DONE - CI run 43177818802 green (host native tests, JVM tests,
`verifyPaparazziDebug`, debug APK and artifacts).

- Added `core/ResponseModes.kt`: Instant, Normal and Extra time budgets (32,
  requested and at least 512 tokens), a reasoning prompt only when thinking is
  allowed, and think-tag splitting for collapsible output. Instant explicitly
  skips thinking.
- The playground now reads speed and reasoning defaults from the manifest,
  exposes both as inference-time switches, shows the effective token budget, and
  warns below ~50M parameters that reasoning is shallow.
- Generation measures and reports latency and tokens/second; the output no
  longer includes the prompt text, so the displayed answer is the generated
  passage. Reasoning is shown in a collapsed block when a `thinking` section is
  present.
- `ResponseModesTest` covers mode normalization, budgets, Instant skipping
  thinking, prompt construction, tag splitting and plain text passthrough.
