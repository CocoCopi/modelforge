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
