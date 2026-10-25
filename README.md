**English** | [简体中文](README.zh-CN.md)

# Yuke

An offline voice-command tool for Windows. Say a short word and it sends a keystroke, a key combination, or a macro. Built for users who need an alternative input method.

Recognition runs entirely on-device with no network requests; commands, recordings and templates stay local and are stored encrypted.

## Features

- **Custom recordings** — record a few takes per command and it starts triggering
- **Single-word triggering** — character-level discrimination, with multi-character commands and multiple pronunciations per command
- **Continuous triggering** — keep speaking commands back-to-back without restarting recognition
- **Flexible key behaviour** — tap, hold-to-toggle, macro sequences, mutually exclusive groups, command groups, global hotkey or voice toggle

## Build

Building requires Rust stable (MSVC). Running requires Windows 10/11 and administrator privileges.

```bat
cargo build --release
powershell -ExecutionPolicy Bypass -File packaging/package_dist.ps1
dist\Yuke-<version>\yuke.exe
```

- The packaging script builds `yuke.pak` (standard and noise-robust encoders; Silero VAD is bundled) and assembles the distribution directory. Runtime dependencies such as `onnxruntime.dll` are taken from `packaging/redist/`.
- Tests must run serially, as they use the audio device and temporary directories: `cargo test --all-targets -- --test-threads=1`.
- For development the app can also run unpackaged: `xcopy /E /I assets\model target\release\model` and `copy packaging\redist\*.dll target\release\` (the pak takes precedence over the loose model files).
- Pushing a tag such as `1.2.6` triggers CI to build and publish a release.

## Getting started

1. Settings → Audio: select a microphone.
2. Commands → "+ New command": give it a name, add three takes, regenerate the pronunciation template, then capture the key.
3. Back on the home page, press "Start recognition" and say the word.

## Documentation

- Architecture, data flow and CLI diagnostics: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- Data handling: [PRIVACY.md](PRIVACY.md)

## License

MIT — see [LICENSE](LICENSE). Sources and licences of models and assets: [ATTRIBUTIONS.md](ATTRIBUTIONS.md).
