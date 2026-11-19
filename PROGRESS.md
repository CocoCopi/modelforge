# ModelForge progress log

This file records, phase by phase, what actually works, what does not, and the
state of the CI run. Only measured facts belong here.

## Environment note

- The `GITHUB_TOKEN` environment variable was **not set** in the build shell.
- The pre-authenticated `gh` CLI was available for account **CocoCopi**. Its
  OAuth token was used through git's credential helper and the REST API. The
  token was never printed, written to a file, or committed.
- Scope note: the token currently carries `repo`, `read:org` and `gist`. GitHub
  refuses pushes that modify `.github/workflows/**` without the `workflow`
  scope, so the hardened workflow is committed locally and pushed once the
  scope is granted; the branch was first pushed with the previous workflow
  file so CI could start immediately.
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

## Production release v2.0.5 - live Hugging Face tool loop

Status: DONE - CI runs 43371588988 and 43372121583 green.

- Replaced the one-shot AI plan request with `AiDatasetAgent`, a bounded tool
  loop (maximum 6 model turns, 7-minute budget). The AI can call:
  - `{"action":"search","query":"...","limit":8}` for real Hugging Face search
    results, download counts and licences;
  - `{"action":"inspect","dataset_id":"owner/name","config":"..."}` for real
    configs, splits, text columns and licence;
  - `{"action":"final","plan":{...}}` only after using a tool.
- Final plans are verified against actual metadata through `AiDatasetResolver`.
  Real config/split/columns are copied into the plan; unverifiable datasets are
  dropped with warnings. Unverified finals are sent back to the AI for another
  tool turn.
- Added `AiChat` and `AiDatasetSource` interfaces plus `AiDatasetAgentTest` with
  scripted search/inspect/final flows and a rejected-unverified-plan retry.
- The agent cannot run arbitrary shell commands. It only gets typed Hugging Face
  search/metadata tools; downloads remain in the app's private model directory.
- Version `2.0.5` (`versionCode 7`), signed with the stable key so it updates
  over 2.0.0-2.0.4. Release:
  https://github.com/CocoCopi/modelforge/releases/tag/v2.0.5
- APK SHA-256:
  `e0cc9de00dbb38b7046a8600ef91fd2268e8dc5444f1b1cc7a1f9efdd47fe0e3`
- AAB SHA-256:
  `07505971ccf52ea03ce1b58618f01262e36262cbc751969b5409b12ab37e39e6`

## Production release v2.0.6 - tolerant AI action parsing

Status: DONE - CI run 43374392558 green.

- Reported issue: "The AI could not produce a verified dataset plan. The AI
  sent an unreadable tool action." The agent gave up after repeated unreadable
  responses.
- Reworked `AiPlanner.parseAgentAction` to scan the whole response and try the
  first valid JSON value. Handles prose around JSON, multiple objects, fenced
  blocks, arrays, `action` objects, OpenAI-style `tool_calls`, and a final plan
  stored as a JSON string.
- Added action aliases for search/inspect/final names and fields
  (`query`/`q`/`term`, `dataset_id`/`id`/`repo_id`, `limit`/`count`, and so on).
- Added a direct-plan retry after the tool loop. Readable direct plans are
  accepted and verified instead of being rejected for missing tool syntax.
- Unreadable responses now include a 220-character sanitized snippet in the
  error. Tool calls rethrow cancellation instead of swallowing it.
- Added parser tests for prose/multiple objects, OpenAI tool calls, action
  objects, direct plans and string-encoded plans, plus a direct-plan fallback
  agent test.
- Version `2.0.6` (`versionCode 8`), signed with the stable key so it updates
  over 2.0.0-2.0.5. Release:
  https://github.com/CocoCopi/modelforge/releases/tag/v2.0.6
- APK SHA-256:
  `d4fab67250bc259d75fbbb5e862961606074139ee0283b968a18e7a54c5530db`
- AAB SHA-256:
  `6a7354be90baed8ca3f8b1a48ae0e734c5f344339abfd5a15de603d6481476ad`

## Production release v2.0.7 - Hugging Face rate-limit recovery

Status: DONE - CI run 43381804014 green.

- Reported issue: all six AI dataset suggestions and the verified fallbacks
  failed with "Hugging Face rate limit reached (HTTP 429)".
- `HfRepository` now retries `429` four times, honoring `Retry-After` and
  otherwise waiting 5s, 20s and 60s. The current wait is shown in the AI stage.
- Added a 10-minute in-memory cache for dataset details and structures. The AI
  inspection, final verification and download steps reuse metadata instead of
  repeatedly calling `/info`, `/splits` and `/rows`.
- Row pages are paced with a 200ms delay. Automatic downloads are capped at six
  suggestions with 100-1000 rows each to reduce request pressure.
- A persistent `429` now stops the download loop instead of hammering the
  endpoint. The AI plan is kept and the error explains how to recover.
- Added **Retry dataset downloads**, which reuses the existing plan and cached
  metadata without another NVIDIA request.
- Search, detail, info and splits calls now share the same retry path; a `429`
  from `/info` no longer falls through to `/splits` and doubles the pressure.
- Added a cache test in `HfRepositoryStructureTest`.
- Version `2.0.7` (`versionCode 9`), signed with the stable key so it updates
  over 2.0.0-2.0.6. Release:
  https://github.com/CocoCopi/modelforge/releases/tag/v2.0.7
- APK SHA-256:
  `b09d62743c8961cb43b4e01e3dd0408b86f380b4f2917a50b8bda4b7999a13c0`
- AAB SHA-256:
  `40d0587d778b345a849d243842d935efb1943105a762f20eb98f189a3c40816a`

Limitations: Hugging Face still controls the rate limit. A Hugging Face token
on Home is the most reliable way to raise it; without a token, waiting or
switching networks may be required even with backoff and pacing.

## Production release v2.0.8 - NVIDIA transient-error retry

Status: DONE - CI run 43401231533 green after fixing a compile error from run
43400728327.

- Reported issue: `NVIDIA HTTP 503: {"error":{"message":"content safety
  screening is temporarily unavailable"}}`. This is a temporary hosted-service
  failure, not a bad API key or prompt.
- Added `NvidiaRetryPolicy`: `429` and `5xx` responses retry up to four
  attempts. Server errors wait 5s, 15s and 30s; rate limits wait 20s, 45s and
  60s. A valid `Retry-After` header takes priority and is capped at 120s.
- `NvidiaClient` retries before any plan text is emitted, so a retry cannot
  duplicate streamed content. The AI stage shows
  "NVIDIA is temporarily unavailable; retrying in Xs".
- Non-retryable errors remain immediate: `401/403`, `404`, malformed responses
  and cancellation.
- The final `503` message states that NVIDIA is temporarily unavailable and
  asks the user to wait a minute and retry.
- Added `NvidiaRetryPolicyTest` for retryable status codes, `Retry-After`
  handling and backoff values.
- Version `2.0.8` (`versionCode 10`), signed with the stable key so it updates
  over 2.0.0-2.0.7. Release:
  https://github.com/CocoCopi/modelforge/releases/tag/v2.0.8
- APK SHA-256:
  `4f74e072268fe814664aff2e7a2761ba23fd5f8b9d8e3ffa088a0bd98c7736d7`
- AAB SHA-256:
  `a9ac1b810cfe0924dee431270af8d0312e654c42c7d9d88d3ac9f0ccdd988e37`

Limitations: a persistent 503 still depends on NVIDIA restoring the content
safety screening service. Retrying cannot bypass a server-side outage.

## Full-spec audit (v2.0.9)

Status: IN PROGRESS - audit complete, implementation has started with diagnostics.
This section is the working checklist for the larger request. Nothing below is
claimed as complete unless it says so.

### Verified gaps that still block the full specification

- **Native engine is still text-only.** NCM and the AI-designed architecture are
  planning paths, not runnable trainers. `ArchitectureRegistry` still labels
  them `Experimental`/`Planned` and blocks training honestly.
- **Artificial native limits remain.** The worker pool caps at 8 threads
  (`app/src/main/cpp/core/gpt.cpp:31`). Context is bounded by a learned absolute
  position table and `T > cfg_.context` is rejected or truncated
  (`gpt.cpp:694`, `:1422`, `:845`, `:1499`). Packed export accepts only 2/3/4/8
  bits (`gpt.cpp:1120-1122`). The quantization group is hardcoded to 32
  (`gpt.cpp:1124-1126`, `gpt.h:77`, `TrainingService.kt:659`). Attention still
  materializes `B*H*T*T` scores (`gpt.cpp:291`), which limits long contexts.
- **Artificial product limits remain.** `HardwareDetector` has an uncompromising
  `ABSOLUTE_MAX_PARAMS = 200_000_000` (`HardwareDetector.kt:42`). The default
  target is clamped to 5M (`ArchitectureRegistry.kt`). Training time is clamped
  to 15-480 minutes, steps to 100-2000, corpus characters to 2M, and the wizard
  still exposes only 128/256/512 context presets. These must become warnings,
  not silent caps.
- **Wizard/spec mismatch.** Production has 15 steps including `AI_PLAN` and
  `TRAINING_TIME` (`WizardSteps.kt:65-70`), but the design gallery still encodes
  14 (`DesignGalleryScreen.kt:113-114`). The training-budget step is
  under-built (`WizardScreen.kt:286-301`). Input/output modalities are stacked
  instead of the required two-column layout (`WizardScreen.kt:235-240`,
  `:261-277`). Architecture cards and non-text modalities still dead-end
  (`WizardScreen.kt:191-228`, `:1118-1128`, `:252-254`, `:1122-1124`, `:727-728`).
- **Hugging Face pipeline is incomplete.** Search/detail/structure use the
  datasets-server APIs, but there is no listing of repository Parquet files,
  no raw `resolve/main` fallback, no Parquet reader, no resumable HTTP range
  download, and only scalar string feature decoding
  (`HfApi.kt:35-42`, `HfRepository.kt:135-206`, `:147`, `:153-154`, `:177-181`,
  `:243-258`). `downloadText` restarts at offset 0 and truncates the destination
  on resume. There is no Arrow/Parquet dependency
  (`app/build.gradle.kts:109-127`).
- **AI architect is narrower than requested.** It has bounded `search` and
  `inspect` tools plus `final`, but no `plan`, `files`, `file`, `test_load`, or
  general service fallback chain. The existing fallback is a fixed verified
  starter list (`AiDatasetFallbacks.kt:3-89`), not a runtime HF fallback chain
  (`raw resolve` -> `list tree` -> `SEQ` -> parquet).
- **Diagnostics coverage is partial.** 2.0.9 adds the screen, global crash
  handler, redacted logs, HF `whoami` validation and NVIDIA model-list
  validation. Still missing: automated emulator flows for training/export,
  mocked 401/403/404/429/503 tests, and physical-device thermal runs.
- **Export/quality gates are not yet complete.** Model card and every export mode
  still need explicit tests. Full build requires arm64 release, x86_64 debug,
  unit tests, native tests, emulator smoke/e2e and screenshot verification.

### Changed in v2.0.9

- Added `DiagnosticsScreen`, `CrashGuard`, `ForgeLog`, `SecretRedactor`,
  `DeviceReport` and `DiagnosticsReport`.
- Added explicit HF/NVIDIA credential checks and secure-storage status.
- Added x86_64 to debug builds only and an emulator CI job; release remains
  arm64-v8a.
- Added `SecretRedactorTest` and an emulator smoke test that opens Diagnostics.
- No architecture status labels or hard training caps were removed in this step;
  removing them without the replacement engines would make the UI dishonest.

### Execution order for the remaining work

1. Finish the design/wizard contract and remove silent caps only together with
   warnings, estimates and graceful failure.
2. Expand HF data: tree/Parquet/raw fallback, resumable downloads and non-text
   decoding with tests.
3. Extend the native transformer to long context and the full bit-width/thread
   matrix, then add measured quality gates.
4. Implement NCM training, the AI architecture DSL/runtime, and all five input/
   output modality paths.
5. Add full emulator end-to-end flows, mocked-service tests, diagnostics export
   and release verification for every architecture and modality.

## Production release v2.0.9 - diagnostics and emulator CI

Status: DONE - CI run 43411445495 green (build + x86_64 emulator UI tests).
Earlier attempts failed on a `DiagnosticsScreen` argument type, the HF fake API
not implementing `whoAmI`, the notification permission dialog stealing focus,
an off-screen LazyColumn target, and an invalid Compose test import; each was
fixed and rerun.

- Added the Diagnostics screen with app/device/ABI/heap/storage/battery/thermal
  facts, HF and NVIDIA checks, secure-storage state, redacted logs, last crash,
  copy and share.
- Added a global uncaught-exception handler that writes a redacted local crash
  report before delegating to the platform handler.
- Added bounded structured logs and redaction for HF tokens, NVIDIA keys,
  Bearer headers and key/token/password assignments.
- Added HF `whoami-v2` validation and NVIDIA `/v1/models` validation.
- Debug APK now contains `arm64-v8a` and `x86_64`; release APK/AAB remain arm64.
- CI added an x86_64 Pixel 6 emulator job with notification permission granted
  and two Compose smoke tests (Home and Diagnostics).
- Version `2.0.9` (`versionCode 11`), signed with the stable key so it updates
  over 2.0.0-2.0.8. Release:
  https://github.com/CocoCopi/modelforge/releases/tag/v2.0.9
- APK SHA-256:
  `9a8f9f80e45ebb9735c83052b81b5467c598728a0caeb2a6daea98447a291449`
- AAB SHA-256:
  `95019ef6d608812405e123a8caa47df23aa1c2f2c4b57ac15e57345856ef1a50`

Limitations: the full architecture/modality engine is still not implemented.
Cap removal, HF Parquet/resume fallback, NCM, AI architecture runtime, non-text
training and full end-to-end quality gates remain open as listed above.

## Production release v2.1.0 - full rows, 1M context, custom steps

Status: DONE - CI run 43692528303 green (build + x86_64 emulator). Release:
https://github.com/CocoCopi/modelforge/releases/tag/v2.1.0

- Version `2.1.0` (`versionCode 12`), signed with the stable release key.
- APK SHA-256:
  `b33fb70111b41c0b03754856fc5467dae07b361725f41f7b3c80ccb7085a9d0d`
- AAB SHA-256:
  `7aaf10ab6e124373e3c8d1cefd310192583fcc2084a4cf802c29e7b8b6a1df39`

- Dataset rows are no longer capped at 5,000. The wizard has an exact row field,
  quick presets and an **Entire split** action. Split sizes come from
  `dataset_info.splits[].num_examples`; when absent, one lightweight `/rows`
  request supplies `num_rows_total`.
- Text downloads fetch six 100-row pages concurrently, write pages in order,
  remove the artificial page delay, and raise OkHttp's dispatcher limits. The
  free-space guard is `availableBytes - 64 MiB`, minimum 1 MB.
- Context spans 128 to 1,000,000 tokens with a logarithmic slider and presets.
  The learned position table was replaced by fixed sinusoidal positions, so
  context length adds no parameters and no context-sized allocation.
- The parameter formula (Kotlin and C++) no longer includes `context * d_model`.
  Legacy v1 checkpoints and packed files still load; their position table is
  skipped and sinusoidal positions are used.
- Training steps are user-editable at every entry point. The 100-2,000 clamps in
  `AiPlanner`, the ViewModel and the UI were removed, and AI row suggestions are
  no longer capped at 5,000.
- Native tests cover a 1,000,000-token config, independent parameter count and
  forward pass. Kotlin tests cover context-independent parameter counts, split
  sizes and ordered concurrent page downloads.
- Compile fixes after CI run 43691525503: `tryApplyDataset` now receives the
  model id for the free-space guard, and `selectSplit` copies from `state`.
  Run 43691953680 then failed only on a unit test that referenced the wrong
  derivation field; run 43692528303 passed.

Honest limitation: 1,000,000 tokens is a positional-capacity ceiling. Training
still uses the device sequence length (usually 128-256), and attention cost grows
with tokens², so practical inference remains RAM- and time-limited. Audio,
video, document and image trainers, NCM, the AI architecture runtime and full
end-to-end quality gates remain open.

## Production hardening - correctness, safety and interchange exports

Status: DONE - branch `production-hardening`, CI run 44119123600 green (host
C++ tests, 95 JVM tests, Paparazzi, debug APK, signed release APK/AAB and the
x86_64 emulator job). Released as v2.2.0 (`versionCode 13`).

Ten previously identified defects were addressed, and the app gained GGUF and
safetensors exports.

### Native engine (C++)

- **Thread-pool race (P0).** The shared pool used one global completion flag and
  captured the caller's stack frame by reference. Two threads entering
  `parallel_for` at the same time reliably segfaulted; the repro harness is
  `/tmp/opencode/thread_race.cpp`. `ThreadPool::run` now keeps all per-call
  state in a heap `Call` object owned by `shared_ptr`, serializes whole
  regions with `region_mu_` (a worker must never start a nested region),
  captures and rethrows the first exception, and never lets a task escape the
  worker loop. The original harness now reports `failures=0`.
- **Input validation.** `tokens_in_range` / `targets_in_range` guard
  `forward`, `train_step` and `PackedModel::forward`; invalid ids, empty
  batches, oversized batches and sequences longer than the context fail closed
  with `-1.0f` and clear the cache. `cross_entropy_loss` guards bad targets,
  `backward` verifies the `dlogits` shape, and an inference-only forward clears
  stale `dlogits`.
- **Optimizer resume.** `adam_step_` is separate from `step_`. A weights-only
  recovery file keeps the progress counter but restarts AdamW bias correction
  from zero, because the moments were not saved.
- **Configuration.** Untied embeddings are rejected in `ModelConfig::valid()`
  and in the Kotlin `ModelConfig.isValid()`; the untied `v*d` head term was
  removed from both parameter formulas.
- **QAT/export alignment.** `set_qat` disables itself for widths it cannot
  quantize instead of allocating a full fp32 copy, and `save_packed` exports
  the QAT `fake_weight` when the model trained with quantization.
- **Exports.** `GPT::save_gguf` writes GGUF v3 with `F32`, `F16`, `Q4_0` or
  `Q8_0` tensors (1-D and non-32-divisible tensors fall back to F16) plus
  `modelforge` metadata describing sinusoidal positions and LayerNorm.
  `GPT::save_safetensors` writes a standard safetensors file with a contiguous
  data section, F32 or F16, using the common Llama/PyTorch tensor names. Both
  are exposed through JNI (`nativeSaveGguf`, `nativeSaveSafetensors`) and
  `NativeModel`.
- Host C++ suite is now 6/6 green: `test_config`, `test_grad`, `test_overfit`,
  `test_packed`, `test_concurrent`, `test_export`.

### Training service

- All service commands are serialized through a `Channel<Command>` and one
  command coroutine, so a stale run can never clobber a new run's flags.
- Per-run `RunControl` holds `stopRequested`, `paused`, `autoPaused`,
  `checkpointRequested` and `resumedWithRecovery`.
- Starting a different model cancels and joins the previous job; cancellation
  saves a checkpoint and is not reported as a failure.
- Checkpoint recovery tries `recovery -> last -> best -> fresh`, skipping
  unreadable files and files whose architecture does not match the manifest.
- `TrainingRestartReceiver` honours `manifest.settings.autoResume`.
- `Train again` on a finished model extends the run instead of doing nothing.

### Data and UI

- Row counts are `Long` end to end (manifest, download stats, AI suggestions,
  wizard state). The wizard accepts any practical row count and the **Entire
  split** action uses the real split size.
- Mix weights accept 0.01x-1000x. The slider is a quick control; an exact text
  field allows any positive multiplier.
- Dataset cache files use UUIDs, so two downloads of the same dataset can never
  collide. Partial files are deleted on failure, duplicate
  dataset/config/split pairs are rejected, and a stale dataset response cannot
  overwrite a newer selection.
- Hugging Face page downloads run 12 pages (max 16) in parallel, still write
  pages in order, and stop at the row count, split end or free-space guard.
- The packed-estimate UI only reports sizes for 2/3/4/8-bit, the widths the
  exporter actually implements. F16/F32 are export choices in the GGUF and
  safetensors menus.
- Home model cards and the training dashboard subscribe to `TrainingBus`; the
  disk fallback is read on `Dispatchers.IO` instead of during composition.
- Secure token writes use `commit()` and report failure. The UI no longer says
  "Token saved securely" when the keystore was unavailable.

### Repository and CI

- Apache License 2.0 `LICENSE` and `NOTICE` added; dependency licence and
  notice files are no longer stripped from the APK.
- README updated for the new exports, download concurrency, weight range and
  the honest GGUF/llama.cpp limitation.
- **Unsigned releases are now impossible from the build itself.** The Gradle
  script refuses any release task when the keystore or its passwords are
  missing, so the guarantee does not depend on one workflow file. CI run
  44129900318 caught a real defect in the first version of this check: inside
  `create("ci") { ... }` the receiver's `keyPassword` property shadowed the
  outer value, so the signing config received a null key password. The outer
  values were renamed and run 44132783459 is green.
- The workflow gains a `pull_request` trigger, read-only default permissions,
  a separate release job whose secrets are scoped to that job, and a release
  job that fails instead of publishing an unsigned APK. **This file is still
  waiting to be pushed:** GitHub refuses workflow changes from an OAuth token
  without the `workflow` scope, and the token in this environment carries
  `repo`, `read:org` and `gist`. Authorising the scope (`gh auth refresh -h
  github.com -s workflow`) or pushing from an account that has it completes the
  change; it is prepared and committed locally.
- ONNX/TFLite/CoreML exports remain unimplemented and are documented as such.

### Release

- v2.2.0 (`versionCode 13`), signed with the stable release key:
  https://github.com/CocoCopi/modelforge/releases/tag/v2.2.0
- `ModelForge-2.2.0-arm64.apk` (arm64-v8a, minSdk 26)
- `ModelForge-2.2.0.aab`
- APK SHA-256:
  `e0dae4f37ff24b4c87dd2a1abd9223ace20bfe6455c23db18cb4edec3a59a9df`
- AAB SHA-256:
  `d7fda84cda3acfdffbeb7485ce9e44a718f5d8001e2637d4798b03bbd03d875f`
- CI runs 44124950596 and 44132783459: host C++ tests, 95 JVM tests, Paparazzi,
  debug APK, signed release APK/AAB and the x86_64 emulator job all green.

Honest limitation: GGUF files produced here are archival/tooling containers,
not llama.cpp-compatible models - ModelForge uses fixed sinusoidal positions
and LayerNorm while llama.cpp expects RoPE and RMSNorm. The container records
this in its metadata instead of pretending the two are equivalent.

## Production release v2.3.0 - training speed, a stop that stops, dataset diversity

Status: IN PROGRESS - host C++ suite 6/6 green; Kotlin/JVM tests and the signed
build run in GitHub Actions.

### Reported problems

- A 50M-parameter model trained at roughly one hour per step on v2.2.0.
- Pressing **Stop training** showed "stopping and saving checkpoint" forever
  and the run restarted by itself when the app was reopened.

### Native engine (C++)

The matrix multiplies, not the configuration, were the bottleneck. The three
kernels were rewritten:

- `matmul_nt_raw` now tiles the output and row dimensions, so a block of
  weights stays resident while it is reused instead of re-reading the whole
  weight matrix for every token.
- `matmul_dw_add` accumulates a block of output rows in a contiguous
  thread-local buffer. The old triple loop walked both operands with a stride
  and could not vectorize at all.
- `matmul_dx_add` processes a block of rows at a time and keeps the inner loop
  over the input dimension contiguous.
- `dot_f32` uses four independent NEON accumulators. A single FMA chain stalls
  on its own latency on an A55, which cost about 1.8x on the dot product alone.

Measured on the aarch64 Cortex-A55 host (4 usable cores, `batch 2 x seq 128`,
256 tokens per step), old kernels versus new kernels in the same session:

| Model | Parameters | Before | After | Speed-up |
| --- | ---: | ---: | ---: | ---: |
| 4 layers, d=128 | 1.06M | 1.15 s/step | 0.54 s/step | 2.1x |
| 15 layers, d=768 | 112.6M | 205 s/step | 14.6 s/step | 14x |

The 112M model previously exceeded a 280-second timeout in this harness. The
profile that guided the rewrite showed backward was nine times slower than
forward (3.2 s vs 0.35 s on the 1M model) before the gradient kernels were
fixed.

### Stop that stops

- `GPT` gained an atomic stop flag. `forward_logits` and `backward` check it
  at every layer boundary, and `train_step` returns the new `kStopped`
  sentinel (`-2.0f`) instead of a loss.
- The abort happens before AdamW, so the weights still match the last
  checkpoint. A native test asserts this: the parameter vectors are
  byte-for-byte identical after a stopped step and the step counter does not
  advance.
- `NativeBridge.nativeRequestStop` / `nativeClearStop` and
  `NativeModel.requestStop()` expose it to Kotlin. `requestStop` deliberately
  does not take the per-model monitor, because the monitor is held by the
  running step; a read/write lock only protects the native handle from being
  freed while the stop request is in flight.
- `TrainingService` keeps a `stopTarget` for the current run. The stop command
  calls `requestStop()` immediately, so a 15-second step is abandoned within a
  layer or two instead of after the full step.
- The stop command also clears the active marker (`store.setActive(id,
  false)`) at once. Previously the marker was only cleared in `finish()`, so a
  process kill during the final checkpoint write left the model active and the
  next launch auto-resumed the run the user had just stopped.
- `onStartCommand(null)` now refuses to restart a run whose persisted state is
  `STOPPED`, `COMPLETED` or `ERROR`, even if an active marker survived a
  crash.
- Stopping during preparation (corpus build, tokenizer training, tokenization)
  is also detected and reports a clean `STOPPED` state instead of continuing
  into the first step.

### Corpus budget

The training corpus was clamped to 4,000,000 characters, which silently threw
away most of a 132k-row dataset and made every epoch replay the same few
thousand rows. `CorpusBudget` now derives the budget from the step count and
the free RAM (ceiling 128M characters, minimum 200k) and is used by the
training service, the wizard, the AI plan and `TrainingSettings.autoFor`. The
token cache limit was raised from 20M to 64M tokens to match.

### Datasets

- `HfRepository.downloadText` is now a sliding window: up to 24 page requests
  are in flight at all times (64 maximum), the next page is requested while the
  current one is written to disk, and pages are still written in order. The
  OkHttp dispatcher allows 64 requests per host.
- The known split size is passed in, so the download stops at the real end of
  the split instead of probing past it.
- Each AI-chosen dataset is retried three times automatically. Whatever still
  fails is listed with a visible **Retry N failed dataset(s)** button that
  retries only the failed ids.
- The AI prompt now requires distinct, complementary dataset ids, and
  `AiPlanner.sanitize` collapses repeated ids instead of treating two configs
  of one repository as separate sources. When fewer than three distinct
  sources were downloaded, verified complementary datasets are added.

### Tests

- `test_concurrent` gained a stop test: a requested stop returns `kStopped`,
  does not advance the step counter and leaves every parameter unchanged.
- `AiPlannerTest` covers long runs (50,000 steps) and one-entry-per-dataset-id
  deduplication.
- `CorpusBudgetTest` covers step-driven growth, the RAM ceiling and the
  minimum.

Honest limitation: the step-time numbers above come from this aarch64 host,
not from the phone. The review screen now shows an approximate per-step and
total compute estimate derived from the parameter count and the device's core
count, and the app keeps reporting measured tokens/second and ETA during the
run. A stopped run keeps the weights of the last completed step; the step that
was interrupted is repeated when training is resumed.

