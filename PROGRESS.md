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

## v2 Phase 4 - post-training identity check

Status: DONE - CI run 43181598407 green (host native tests, JVM tests,
`verifyPaparazziDebug`, debug APK and artifacts).

- `core/IdentityCheck.kt` generates 20 questions across the selected languages
  and scores an answer only when it contains both the model name and creator
  (case-insensitive). The pass threshold is 80%.
- The model detail screen runs the check locally against the best checkpoint
  with low-temperature, short generations, shows the live question count and
  score, and lists a few failed answers rather than hiding them.
- Below 80%, "Reinforce identity" raises the training total by 100 steps and
  sets the identity fraction to at least 8%, then resumes from the existing
  checkpoint. Training settings now record `identityFraction` and the token
  cache includes it in its signature.
- `IdentityCheckTest` covers case-insensitive name+creator matching, the 80%
  boundary, reinforcement flag and multilingual question generation.

The check measures real generated text; on tiny from-scratch models a low score
is expected and is reported honestly rather than tuned into a fake pass.

## v2 Phase 5 - reusable presets

Status: DONE - CI run 43182363626 green (host native tests, JVM tests,
`verifyPaparazziDebug`, debug APK and artifacts).

- `data/Preset.kt` stores up to 20 named `ModelPreset` records in
  SharedPreferences, including architecture, size, context, languages, speed,
  reasoning, tools, persona, compression and the whole training budget.
- The review step can save the current configuration under a name, apply a
  preset to the current draft, or delete one. Applying re-derives the model
  config and reports that datasets are intentionally not part of a preset.
- Presets never include checkpoints or cached datasets; each model still picks
  its own data before training. `PresetTest` covers JSON round-tripping and the
  summary text.

## v2 Phase 6 - automatic model card

Status: DONE - CI run 43183528487 green (host native tests, JVM tests,
`verifyPaparazziDebug`, debug APK and artifacts).

- `data/ModelCard.kt` builds a Markdown card from the manifest: creator,
  architecture and status, exact parameter layout, languages, training budget,
  identity fraction, behaviour, datasets with licences/non-commercial flags,
  intended use, limitations and a licence-responsibility note.
- The training service writes `MODEL_CARD.md` automatically when a run finishes.
  Model details can also regenerate and share the card through the existing
  FileProvider.
- The card never claims benchmarks or a production quality level; it reports
  the attached data and the fact that samples are tiny-model samples.
- `ModelCardTest` checks names, creator, licence, languages, identity fraction
  and the responsibility note.

## v2 Phase 7 - background training and live window

Status: DONE - CI run 43210808386 green (host native tests, JVM tests,
`verifyPaparazziDebug`, debug APK and artifacts).

- The live training screen and model detail now subscribe to `TrainingBus`, so
  progress, loss, speed, RAM, device state and samples update as the service
  publishes them instead of polling the disk only at checkpoints. Disk state
  remains the fallback after a process restart.
- The service returns `START_REDELIVER_INTENT` for start requests, uses a
  `specialUse` foreground service type on Android 14+ (data sync below that),
  and schedules a restart through `TrainingRestartReceiver` from
  `onTaskRemoved`/`onDestroy` when a model is still active. Boot resume stays.
- Crash recovery checkpoints are written weights-only every 5 steps up to 5M
  parameters, 10 steps up to 20M and 20 steps above. They are selected on resume
  when newer than the full optimizer checkpoint, so a background kill no longer
  falls back to the last 50-step checkpoint. The UI states honestly when weights
  were recovered and optimizer state may have restarted.
- Live state is persisted about once per second, and checkpoint writes run on
  `Dispatchers.IO` so the training loop is not blocked by file writes.
- The training screen offers Android's battery-optimisation exemption dialog
  because some vendors still kill foreground work without it.
- `RecoveryPolicyTest` covers interval scaling, due checks and little-endian
  checkpoint step reading.

## v2 Phase 8 - NVIDIA AI architect and the three-path wizard

Status: DONE - CI run 43260935632 green (host native tests, JVM tests,
`verifyPaparazziDebug`, debug APK and artifacts).

- `data/ai/NvidiaClient.kt` streams chat completions from
  `https://integrate.api.nvidia.com/v1/chat/completions` with
  `minimaxai/minimax-m3.1-flash`, parses `data:` SSE frames and `[DONE]`, and
  returns the accumulated answer. The API key is read from
  `SecureTokenStore.nvidiaToken`; it is never logged, written to disk in
  plaintext, or included in model cards.
- A key dialog is reachable from Home and the AI design step. Saving fails
  loudly if encrypted storage is unavailable instead of silently dropping the
  key.
- `data/ai/AiPlanner.kt` builds an explicit architect prompt from the draft:
  name, creator, architecture, parameter count, input/output modalities,
  languages (including Tamil), training minutes, context, speed, reasoning,
  tools, persona, compression and current datasets. The model must return one
  JSON object with an overview, analogy, training settings, per-purpose dataset
  suggestions (general, identity, reasoning, tool calls, languages, modalities)
  with rows, mix weight, licence and commercial-use flag, plus notes/warnings.
- Parsing accepts bare or fenced JSON and sanitizes every bound: steps 100-2000,
  batch 1-32, sequence 32-1024, learning rate 1e-5..1e-2, rows 100-5000 and
  weights 0.1-5. Non-commercial or unconfirmed suggestions get a visible
  warning. Applying a plan updates the draft's steps, batch, sequence, learning
  rate and warmup; dataset suggestions are inspected against the real Hugging
  Face API before use, so hallucinated ids fail visibly.
- The registry now exposes exactly three choices: traditional transformer
  (stable and trainable), NCM (planning-only) and a new AI-designed
  architecture (planning-only). Both experimental paths are selectable but
  return no native config, so the wizard cannot claim to train an engine that
  does not exist.
- The wizard adds an interactive input/output modality step (text, image,
  audio, video, documents), a 15-480 minute training-time slider, an AI design
  step with streaming output and dataset inspection, and Tamil in the language
  list and identity templates. Native training remains honestly gated to
  text-to-text on the traditional transformer.
- Tests cover the planner prompt, fenced/plain JSON parsing, clamping,
  commercial-licence warnings, SSE delta parsing, the three-architecture
  registry, and Tamil identity generation.

Limitations: the real NVIDIA endpoint was not called in CI because no API key
is stored there. NCM and AI-designed architectures still need native engines;
the planner designs their data and settings but cannot train them yet.

## Production release engineering

Status: IN PROGRESS - release signing, R8, launcher icon and release artifacts.

- CI debug and release builds use a stable project signing key stored only in
  GitHub Actions secrets. Existing v1.0.0/debug installs must be uninstalled
  once because their ephemeral certificate differs.
- Version advanced to `2.0.0` (`versionCode 2`).
- Added adaptive launcher icon and light splash/window colours.
- Enabled R8 code shrinking, resource shrinking and conservative keep rules for
  JNI, persisted JSON models, Retrofit and kotlinx.serialization.

## Production release v2.0.0

Status: DONE - CI run 43269569997 green.

- Stable project signing key is injected only from GitHub Actions secrets; the
  debug and release APKs share a consistent certificate for future updates.
- `versionCode 2`, `versionName 2.0.0`; adaptive launcher icon and light
  splash/window colours added.
- Release build uses R8 code shrinking and resource shrinking with keep rules
  for JNI, persisted JSON, Retrofit and kotlinx.serialization. The release APK
  is about 3.5 MB versus about 20 MB for the debug build.
- CI now builds and verifies `app-release.apk`, `app-release.aab`, and uploads
  the R8 mapping alongside the debug APK.
- Release APK verified: package `com.modelforge.app`, version 2.0.0, minSdk 26,
  arm64-v8a, v2 signature, debuggable false, cleartext disabled.

## Production release v2.0.1 - AI request feedback

Status: DONE - CI runs 43274899079 and 43276120628 green.

- Fixed the "Designing..." dead-end in the AI architect. The network call had a
  five-minute read timeout with no visible progress when the model reasoned
  before emitting content or did not stream.
- The AI step now displays elapsed seconds and the number of NVIDIA stream
  events from the first moment, and provides **Cancel request**, which cancels
  the coroutine and the active OkHttp call immediately.
- Streaming now has a 90-second read timeout, a 180-second total call timeout,
  and a non-streaming JSON fallback. HTTP 401/403/404/429 responses explain the
  likely cause. If no plan text arrives, the app reports that instead of hanging.
- Request tuning: `max_tokens` 8192, `temperature` 0.4, compact prompt with at
  most 8 dataset suggestions. Streaming still uses seed 0, top-p 0.95 and the
  same NVIDIA model and endpoint.
- Parser now handles `message.content`, `text`, JSON-array content and
  `reasoning_content`; tests cover non-streaming content and reasoning frames.
- Version `2.0.1` (`versionCode 3`), signed with the stable key so it updates
  over v2.0.0 without uninstalling. Release:
  https://github.com/CocoCopi/modelforge/releases/tag/v2.0.1
- APK SHA-256:
  `cd2cf605d29e169b92319df09d08799545983db21f47a5f6bd4caf7bf2a64643`
- AAB SHA-256:
  `1958ec421de70931ae85eead3b38a418e3bd562b616ff214f46133b65571fe9d`

Limitations: the NVIDIA endpoint is still not called in CI because no API key
is available there. If the new UI reports a specific HTTP error or a no-text
completion, that is the next diagnostic to investigate.

## Production release v2.0.2 - disable hidden reasoning

Status: DONE - CI run 43289520212 green.

- Live NVIDIA testing showed MiniMax M3.1 Flash streamed `reasoning_content`
  before any plan `content`. With a large output budget this could exceed the
  request timeout and surface as "Could not reach NVIDIA: timeout".
- The request now sends `reasoning_effort: "none"`, which was accepted by the
  live endpoint and produced plan content directly. `max_tokens` is 4096 and
  the overall non-stall cap is 300 seconds; the 90-second read timeout still
  catches silent connections.
- Verified with the app's actual prompt against the live endpoint: first stream
  event about 6 seconds, complete plan about 47 seconds, valid JSON with 6-7
  dataset suggestions.
- Timeout errors now say whether zero stream events arrived (phone cannot reach
  `integrate.api.nvidia.com`; try another network/VPN) or events arrived but
  the model ran too long. DNS, connect and TLS failures have separate messages.
- Version `2.0.2` (`versionCode 4`), signed with the stable key so it updates
  over 2.0.0/2.0.1. Release:
  https://github.com/CocoCopi/modelforge/releases/tag/v2.0.2
- APK SHA-256:
  `401440084462f6fae035ba286a9fd021ca32d9caae84a11a26352d0f37506e7d`
- AAB SHA-256:
  `76201563ccf754d47fea5dd55d1221bd4680871f47ae9f62f971bd43f1bc9527`

## Production release v2.0.3 - AI autopilot downloads datasets

Status: DONE - CI run 43360697894 green.

- The AI architect now applies its chosen training recipe automatically:
  steps, batch size, sequence length, learning rate and warmup are written to
  the model draft before the user reaches Review.
- Up to 8 AI-selected Hugging Face datasets are resolved, downloaded and added
  to the dataset mix automatically. Existing datasets are retained on redesign.
- `AiDatasetResolver` now rejects explicitly requested configs that do not
  exist, uses real splits, prefers instruction/response/text columns, and
  rejects datasets with no usable text columns.
- Dataset dedupe now normalizes blank config/split to `default`/`train`, so
  repeated suggestions collapse while the same id with different real configs
  is preserved.
- Hugging Face page downloads retry transient `429/5xx` and I/O failures before
  reporting a dataset as skipped. The integration test uses Dolly-15k because
  the TinyStories viewer returned repeated 503s.
- Added AI stage/elapsed/event progress, per-dataset status, an auto-download
  flow that opens Review when ready, and a Review "AI autopilot" summary card.
- Version `2.0.3` (`versionCode 5`), signed with the stable key so it updates
  over 2.0.0-2.0.2. Release:
  https://github.com/CocoCopi/modelforge/releases/tag/v2.0.3
- APK SHA-256:
  `f96914f8b683627bc49f1d37a7af2ce6251005fc8684a5efc75e06f8cd0b929d`
- AAB SHA-256:
  `e4a747a6bf9ebf74a710f7575b460c0240cf2c33b80e2fb028e8c9cb67b56f52`

Limitations: native training remains text-to-text only; NCM and AI-designed
architectures remain planning paths. AI output quality still depends on the
NVIDIA model and the datasets it selects.

## Production release v2.0.4 - recover when AI datasets fail

Status: DONE - CI run 43367803126 green.

- Reported issue: the AI chose datasets, but none were verified or downloaded,
  leaving only a generic error and a manual-dataset fallback.
- Added `AiDatasetFallbacks` with verified starter datasets for Tamil,
  tool-calling, reasoning, instruction and general English. When every AI
  suggestion fails, up to two fallback datasets are downloaded automatically.
- Skip reasons are now stored in `AiPlannerState.skippedDatasetReasons` and
  shown per dataset in the AI design screen and Review. The final error includes
  the underlying HTTP/network reasons.
- Empty AI plans also try the verified fallbacks instead of stopping.
- `HfRepository.structure` now falls through to `/splits` plus sample rows when
  `/info` returns configs without text features, and checks all splits instead
  of only the first. Added `HfRepositoryStructureTest`.
- Added `AiDatasetFallbacksTest` for Tamil/tool/reasoning ordering and the
  general two-dataset fallback.
- Version `2.0.4` (`versionCode 6`), signed with the stable key so it updates
  over 2.0.0-2.0.3. Release:
  https://github.com/CocoCopi/modelforge/releases/tag/v2.0.4
- APK SHA-256:
  `cce105aa7487b4d0931081620244e2e875806d62d2a6feeb635523fd6b328539`
- AAB SHA-256:
  `6bf18323669a8456953426c0c90ac22a0b1a6a34fedd10fd5fc17ec0e8d0ddd4`
