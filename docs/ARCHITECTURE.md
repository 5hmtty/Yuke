# Yuke architecture and code guide

Offline speech-to-keyboard for Windows: register a spoken word, bind it to a key, and saying it fires the key. All recognition happens on the local machine.

## Build and run

```bat
cargo build --release
xcopy /E /I assets\model target\release\model
copy packaging\redist\*.dll target\release\
cargo test --all-targets -- --test-threads=1
target\release\yuke.exe
```

- Rust stable (MSVC); tests must run serially.
- Models are not compiled into the binary: at runtime they are looked up in `yuke.pak` first, then in `<exe>/model/`. If neither exists the app reports "no v2 encoder found". They can also be packed with `yuke pak-pack yuke.pak model`.
- Administrator rights are required (key injection must pass UIPI; the app relaunches itself through UAC).
- The default encoder is `encoder-v2-chunk-8.int8.onnx`. `encoder-v2-noise.int8.onnx` is the noise-robust variant, switchable under Settings → Recognition → Noise-robust model (all pronunciation templates are rebuilt automatically after restart).
- `silero_vad.onnx` (non-speech veto) is bundled with the repository and packed into `yuke.pak`. The optional `sv_cnceleb_resnet34.onnx` (voiceprint gate) is not committed; place it under `<exe>/assets/` when needed. When a model is missing, the corresponding feature is disabled automatically.

## Main data flow

```
microphone --50 ms blocks--> cpal capture --> echo cancellation (optional) --> log-mel frames
                                                                                   |
                                                                   activity gate -> burst segmentation
                                                                                   |
                                                                                   v
game <-- key injection <-- dispatch (tap/toggle/macro) <-- emission <-- syllable-nucleus counting
                                                                                   ^
                                                                  identity scoring (embedding cosine >= threshold)
```

1. **Capture / features**: 16 kHz mono; log-mel with a 25 ms window and 10 ms hop; optional per-frame mean removal in the front end.
2. **Segmentation / nucleus counting**: the activity gate carves out continuous speech regions, and peak detection inside each region counts syllable nuclei.
3. **Identity scoring**: a window is taken around each nucleus peak and compared with the registered templates by cosine similarity; the best-scoring command wins.
4. **Emission / dispatch**: once the expected number of nuclei is reached the trigger fires; the key is sent as tap / toggle / macro, with repeat protection and mutex release.

## Directory layout

```
src/
├── main.rs        entry point, CLI, data directory, settings→env bridge
├── config.rs      commands.json (commands / macros / mutex / multi-pronunciation / gate)
├── crypto.rs      AES-256-CTR encryption container
├── live.rs        capture -> processor -> key output
├── pak.rs         yuke.pak container
├── ui.rs + ui/    egui three-tab UI, dialogs, logs
├── engine/        decision layer: udkws_v2 / gate / auto_gate / features / nuclei_cache
├── audio/         capture, AEC, recordings, Silero VAD
└── platform/      key injection, elevation, single instance, update check
```

Modules are declared in groups at the top of `main.rs` and re-exported at the crate root, so internal paths do not change when directories move.

## Key mechanisms

- **Activity gate**: the only absolute threshold in the pipeline, applied to the band-passed envelope; it decides whether a burst opens and where its edges are. In automatic mode it follows a low percentile of the noise floor.
- **Syllable nuclei**: relative criteria on the curve (high-percentile margin + valley depth + minimum peak distance); the nucleus count decides how many must be collected before emitting.
- **Pronunciation templates**: each recording is encoded over its active region and averaged into a centroid; low recording levels do not get chopped up.
- **Veto layer (off by default)**: VAD decides whether the sound is speech, the voiceprint gate whether it is the enrolled speaker; both are disabled automatically when their models are missing.

## User data

`<exe>/UserData/` contains `settings.json`, `commands.json`, `groups.json`, `command_recordings/` (encrypted), `recordings/` (session recordings, off by default), `logs/`, `embed_proto_v2.json`, `nuclei_templates_v2.json.enc`, and `embed_proto_v2.meta.json` (the template input fingerprint, which decides whether to rebuild on startup). Backups and migrations only need to copy this directory.

Command fields: `enabled` / `phrase` / `key` / `mode` / `macro` + `macro_reentry` / `mutex` / `multi_nuclei`. The engine's command set is the intersection of the `embed_proto_v2.json` keys and the enabled commands; the syllable-nucleus templates are generated from `command_recordings/` and cached.

## Command-line tools

| Subcommand | Purpose |
|---|---|
| `gen-embed-proto-v2 <model-dir>` | Regenerate the pronunciation templates |
| `pak-pack <out.pak> <dir>...` / `pak-list` | Pack / list the resource package |

## Debugging environment variables

| Variable | Effect |
|---|---|
| `VIQ_DEBUG_LOG=1` | Per-line diagnostics |
| `VIQ_ACT_GATE_DB` / `VIQ_ACT_GATE_AUTO=1` | Override the activity gate |
| `VIQ_V2_FRAME_MEAN_NORM=0/1` | Per-frame mean removal in the front end (rebuild templates afterwards) |
| `VIQ_DIAG_IDENT=1` | Identity-layer best / second scores |
| `VIQ_SV_GATE_THR` | Voiceprint-gate threshold |
| `VIQ_NO_ELEVATE=1` | Skip UAC (for testing) |

## Notes

- Key injection requires administrator rights; a second launch activates the existing window instead of starting a second instance.
- After changing the microphone or any front-end setting, the templates must be regenerated, otherwise the feature definition no longer matches.
- Release builds enable fat LTO and `panic=abort`: slow to build, and a crash produces no backtrace.
