# ModelForge privacy

ModelForge has no analytics, telemetry, advertising or crash-upload service.

## Data stored on the phone

- Models, checkpoints, tokenizer caches and downloaded datasets stay in the
  app's private storage.
- The optional Hugging Face token and optional NVIDIA API key are stored with
  Android `EncryptedSharedPreferences`. They are not written to logs, model
  cards or exported model files.
- Structured diagnostics logs are held in memory and a last-crash report is
  stored in the app's private files. Both are redacted and never uploaded
  automatically. A diagnostics report leaves the device only when you tap
  **Copy report** or **Share**.
- Android backup is disabled for the app.

## Network requests

- Hugging Face (`huggingface.co`) is contacted for dataset search, metadata,
  rows and optional gated access. A saved Hugging Face token is sent only there.
- NVIDIA (`integrate.api.nvidia.com`) is contacted when you press the AI design
  button or run the NVIDIA diagnostic check. The prompt includes the model name, creator, architecture,
  modalities, languages, training-time budget and other wizard choices. The
  NVIDIA key is sent only to NVIDIA.
- No other network endpoints are used.

## Your responsibilities

- Dataset licences are shown in the app. You are responsible for checking and
  following each dataset licence, especially for commercial use.
- The AI plan and dataset suggestions are proposals. The app checks suggested
  dataset ids against Hugging Face before download, but you must review the
  dataset card, contents and licence yourself.
- NVIDIA and Hugging Face process requests under their own terms and privacy
  policies.
