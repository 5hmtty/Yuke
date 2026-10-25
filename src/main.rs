//! Yuke: offline speech-to-keyboard for Windows. With no arguments it starts the egui UI; the remaining
//! arguments are template-generation / packaging subcommands (see the usage line printed by `yuke` with no arguments).
//!
//! Release builds use the Windows GUI subsystem (no console window); debug keeps the console for CLI debugging.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod audio;
mod engine;
mod platform;

pub use audio::{comm, mic_monitor, recordings, silero, sonora_aec};
pub use engine::{auto_gate, features, gate, nuclei_cache, udkws_v2};
pub use platform::{elevate, keys, single_instance, update};

mod config;
mod crypto;
mod i18n_keys;
mod lang;
mod live;
mod pak;
mod sv;
mod ui;
mod window_geom;

/// Export the engine parameters from settings.json as env vars (the engine only reads env and knows nothing about the UI configuration).
fn apply_settings_env() {
    let Ok(txt) = std::fs::read_to_string(config_path("settings.json")) else {
        // First run has no settings.json: keys whose default is non-zero or non-false must be written before the early return,
        // otherwise the engine falls back to its built-in defaults and disagrees with what the settings page shows.
        // The rule is "never touch what was set explicitly outside", which preserves the env injection path used by replay-live and friends.
        let set_default = |k: &str, v: &str| {
            if std::env::var_os(k).is_none() {
                unsafe { std::env::set_var(k, v) };
            }
        };
        set_default("VIQ_V2_FRAME_MEAN_NORM", "1");
        set_default("VIQ_UDKWS_V2_NUCLEI_REFR", "15");
        set_default("VIQ_ACT_GATE_AUTO", "1");
        set_default("VIQ_UDKWS_V2_MINCONF", "0.6");
        set_default("VIQ_ACT_GATE_DB", "-50");
        set_default("VIQ_IDENT_FWD_SEC", "0.13");
        return;
    };
    let Ok(v) = serde_json::from_str::<serde_json::Value>(&txt) else {
        return;
    };
    let gb = |k: &str, d: bool| v.get(k).and_then(|x| x.as_bool()).unwrap_or(d);
    let gf = |k: &str, d: f64| v.get(k).and_then(|x| x.as_f64()).unwrap_or(d);
    unsafe {
        std::env::set_var(
            "VIQ_UDKWS_V2_MINCONF",
            format!("{}", gf("v2_min_conf", 0.6)),
        );
        std::env::set_var(
            "VIQ_UDKWS_V2_NUCLEI_REFR",
            format!(
                "{:.0}",
                (gf("refr_s", 0.15).clamp(0.0, 2.0) * 100.0).round()
            ),
        );
        std::env::set_var("VIQ_ACT_GATE_DB", format!("{}", gf("act_gate_db", -50.0)));
        std::env::set_var(
            "VIQ_ACT_GATE_AUTO",
            if gb("act_gate_auto", true) { "1" } else { "0" },
        );
        std::env::set_var(
            "VIQ_V2_FRAME_MEAN_NORM",
            if gb("frame_mean_norm", true) {
                "1"
            } else {
                "0"
            },
        );
        let aec_mode = v
            .get("aec_mode")
            .and_then(|x| x.as_u64())
            .map(|n| match n as u8 {
                1 => 1,
                3 => 3,
                2 => 3,
                _ => 0,
            })
            .unwrap_or(0);
        std::env::set_var("VIQ_AEC_MODE", format!("{aec_mode}"));
        std::env::set_var(
            "VIQ_IDENT_FWD_SEC",
            format!("{}", gf("ident_fwd_sec", 0.13)),
        );
        std::env::set_var(
            "VIQ_ENCODER_NOISE",
            if gb("noise_model", false) { "1" } else { "0" },
        );
        std::env::set_var(
            "VIQ_VAD_GATE",
            if gb("vad_gate", false) { "1" } else { "0" },
        );
        std::env::set_var("VIQ_SV_GATE", if gb("sv_gate", false) { "1" } else { "0" });
        std::env::set_var(
            "VIQ_SV_GATE_THR",
            format!("{}", gf("sv_gate_threshold", 0.35)),
        );
    }
}

pub fn default_model_dir() -> String {
    "model".to_string()
}

pub fn exe_dir() -> Option<std::path::PathBuf> {
    std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|d| d.to_path_buf()))
}

/// Writable data root: the directory containing the exe (portable -- data travels with the program, unpack and run).
pub fn data_root() -> std::path::PathBuf {
    exe_dir().unwrap_or_else(|| std::path::PathBuf::from("."))
}

pub fn config_dir() -> std::path::PathBuf {
    user_data_dir()
}

pub fn user_data_dir() -> std::path::PathBuf {
    data_root().join("UserData")
}

pub fn ensure_user_data() {
    let _ = std::fs::create_dir_all(user_data_dir());
}

pub fn config_path(file: &str) -> std::path::PathBuf {
    config_dir().join(file)
}

/// Absolutize and strip the Win32 extended-length prefix (the canonicalize `\\?\\` prefix makes mixed-slash paths from `format!("{dir}/x")`
/// unusable under extended-length path semantics, and looks bad in the UI).
/// When the path does not exist, join against the cwd (display/fallback only; normal resolution always yields an existing path).
pub fn abs_norm(p: &std::path::Path) -> std::path::PathBuf {
    if let Ok(c) = std::fs::canonicalize(p) {
        let s = c.to_string_lossy().to_string();
        return std::path::PathBuf::from(s.strip_prefix(r"\\?\").unwrap_or(&s));
    }
    if p.is_absolute() {
        p.to_path_buf()
    } else {
        std::env::current_dir()
            .map(|d| d.join(p))
            .unwrap_or_else(|_| p.to_path_buf())
    }
}

/// Data-directory resolution result (model directory; used for UI logs, the status bar and WARN passthrough)
pub struct DirResolution {
    /// Resolved absolute path (callers join their own sub-paths onto it)
    pub path: std::path::PathBuf,
    /// How it resolved ("relative to exe" / "absolute path from settings")
    pub source: &'static str,
    /// Resolution warnings (settings absolute path deviating from the principle / model directory missing)
    pub warnings: Vec<String>,
}

pub fn resolve_data_dir(name: &str) -> DirResolution {
    let raw = std::path::Path::new(name);
    let mut warnings: Vec<String> = Vec::new();
    let (path, source) = if raw.is_absolute() {
        warnings.push(format!(
            "settings.model_dir is an absolute path ({name}), which departs from the \"relative to exe\" principle\
             (prefer a bare directory name such as model)"
        ));
        (raw.to_path_buf(), "absolute path from settings")
    } else {
        let p = data_root().join(name);
        (p, "relative to exe")
    };
    let path = abs_norm(&path);
    if !path.is_dir() {
        warnings.push(format!(
            "model directory missing: {} (expected <exe dir>/{name}) -- check that the installation is complete;\
             the engine cannot start if <model_dir>/commands.json is missing as well",
            path.display()
        ));
    }
    DirResolution {
        path,
        source,
        warnings,
    }
}

/// Pin cwd to the exe directory: config.rs and live.rs still read a few paths relative to the cwd, which after alignment
/// equals `<exe dir>/...`; after an elevation restart the cwd is often System32 and would drift into the system directory.
fn align_cwd_to_exe() {
    let d = data_root();
    let _ = std::fs::create_dir_all(&d);
    let already = std::env::current_dir()
        .ok()
        .and_then(|c| Some(std::fs::canonicalize(c).ok() == std::fs::canonicalize(&d).ok()))
        .unwrap_or(false);
    if already {
        return;
    }
    match std::env::set_current_dir(&d) {
        Ok(()) => eprintln!("[cwd] aligned to the exe directory: {}", d.display()),
        Err(e) => eprintln!(
            "[cwd] WARN: failed to align to the exe directory {} ({e}); cwd-relative reads may drift",
            d.display()
        ),
    }
}

fn init_ort() {
    if std::env::var_os("ORT_DYLIB_PATH").is_none() {
        if let Some(dll) = std::env::current_exe()
            .ok()
            .and_then(|p| p.parent().map(|d| d.join("onnxruntime.dll")))
        {
            if dll.exists() {
                unsafe { std::env::set_var("ORT_DYLIB_PATH", dll) };
            }
        }
    }
}

fn install_crash_hook() {
    use std::sync::atomic::{AtomicBool, Ordering};
    static INSTALLED: AtomicBool = AtomicBool::new(false);
    if INSTALLED.swap(true, Ordering::Relaxed) {
        return;
    }
    unsafe extern "system" {
        fn SetUnhandledExceptionFilter(filter: Option<unsafe extern "system" fn(*mut i32) -> i32>);
    }
    unsafe extern "system" fn crash_filter(pep: *mut i32) -> i32 {
        let code = if pep.is_null() { 0 } else { unsafe { *pep } };
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let dir = user_data_dir().join("logs");
        let _ = std::fs::create_dir_all(&dir);
        let _ = std::fs::write(
            dir.join(format!("crash_{ts}.log")),
            format!(
                "crash: exception 0x{code:08x} at {ts}
app: Yuke v{}
os: {} {}",
                env!("CARGO_PKG_VERSION"),
                std::env::consts::OS,
                std::env::consts::ARCH
            ),
        );
        0 // EXCEPTION_CONTINUE_SEARCH: hand back to the default system handler
    }
    unsafe { SetUnhandledExceptionFilter(Some(crash_filter)) };
}

fn install_panic_hook() {
    std::panic::set_hook(Box::new(|info| {
        let msg = info
            .payload()
            .downcast_ref::<&str>()
            .map(|s| (*s).to_string())
            .or_else(|| info.payload().downcast_ref::<String>().cloned())
            .unwrap_or_else(|| "unknown panic".into());
        let loc = info
            .location()
            .map(|l| format!("{}:{}", l.file(), l.line()))
            .unwrap_or_default();
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis())
            .unwrap_or(0);
        let dir = user_data_dir().join("logs");
        let _ = std::fs::create_dir_all(&dir);
        let _ = std::fs::write(
            dir.join(format!("crash_panic_{ts}.log")),
            format!(
                "Yuke v{}\npanic: {msg}\nat {loc}\nos: {} {}",
                env!("CARGO_PKG_VERSION"),
                std::env::consts::OS,
                std::env::consts::ARCH
            ),
        );
    }));
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    install_crash_hook();
    install_panic_hook();
    init_ort();
    let args: Vec<String> = std::env::args().collect();
    // UI mode (no arguments or `ui`) elevates automatically: games often run as administrator, and SendInput from a normal process is blocked by UIPI
    if args.get(1).map(|s| s == "ui").unwrap_or(true) {
        if std::env::var_os("VIQ_NO_ELEVATE").is_none() {
            elevate::elevate_if_needed();
        }
        if let crate::single_instance::Startup::AlreadyRunning =
            crate::single_instance::acquire_or_signal()
        {
            return Ok(());
        }
        align_cwd_to_exe(); // Pin cwd to the exe directory (cwd-relative reads no longer drift)
        ensure_user_data();
        apply_settings_env();
        let ud = user_data_dir();
        let ud_s = ud.to_string_lossy().to_string();
        let model_dir_raw = args
            .get(2)
            .cloned()
            .or_else(|| {
                std::fs::read_to_string(config_path("settings.json"))
                    .ok()
                    .and_then(|t| serde_json::from_str::<serde_json::Value>(&t).ok())
                    .and_then(|v| {
                        v.get("model_dir")
                            .and_then(|x| x.as_str())
                            .map(String::from)
                    })
            })
            .unwrap_or_else(default_model_dir);
        let mut warnings = Vec::new();
        if crate::pak::available() {
            eprintln!(
                "[pak] model assets served from {} (a loose model/ directory is only a fallback)",
                crate::pak::PACK_NAME
            );
        } else {
            let res = resolve_data_dir(&model_dir_raw);
            for w in &res.warnings {
                eprintln!("[model_dir] WARN: {w}");
            }
            eprintln!(
                "[model_dir] using {} (source: {})",
                res.path.display(),
                res.source
            );
            warnings = res.warnings;
        }
        eprintln!("[userdata] user data directory: {}", ud.display());
        ui::run(&ud_s, &warnings);
        return Ok(());
    }
    match args.get(1).map(String::as_str) {
        Some("pak-pack") => {
            let out = std::path::PathBuf::from(
                args.get(2)
                    .ok_or_else(|| anyhow::anyhow!("usage: pak-pack <out.pak> <dir>..."))?
                    .as_str(),
            );
            let dirs: Vec<std::path::PathBuf> =
                args[3..].iter().map(std::path::PathBuf::from).collect();
            if dirs.is_empty() {
                return Err(anyhow::anyhow!(
                    "usage: pak-pack <out.pak> <dir>... (at least one input directory)"
                )
                .into());
            }
            let (n, total) = pak::build(&out, &dirs)?;
            println!(
                "[pak] {} entries / {:.1} MiB -> {}",
                n,
                total as f64 / 1048576.0,
                out.display()
            );
        }
        Some("pak-list") => {
            pak::print_list()?;
        }
        Some("gen-embed-proto-v2") => {
            // Usage: gen-embed-proto-v2 <model-dir> [encoder.onnx] -- generates the v2 template marker embed_proto_v2.json
            // and the syllable-nucleus template cache from command_recordings
            let model_dir = args.get(2).cloned().unwrap_or_else(default_model_dir);
            // Template gate = the hand-set value bridged through env (bit-identical to the old EngineParams::default behaviour)
            udkws_v2::gen_embed_proto(
                &model_dir,
                args.get(3).map(String::as_str),
                udkws_v2::EngineParams::default().act_db,
            )?;
        }
        // Unknown argument: print the usage
        _ => {
            println!(
                "usage: yuke [gen-embed-proto-v2|pak-pack|pak-list] ...\nrun without arguments to start the UI (egui)"
            );
        }
    }
    Ok(())
}

#[cfg(test)]
mod pathunify_tests {

    use super::{abs_norm, config_path, exe_dir, resolve_data_dir};

    const PROBE: &str = "viq_probe_model_rel";

    #[test]
    fn relative_name_resolves_under_exe_dir_only() {
        let exe = exe_dir().expect("the test process can also get the current_exe parent");
        let res = resolve_data_dir(PROBE);
        assert_eq!(res.source, "relative to exe");
        assert_eq!(res.path, abs_norm(&exe.join(PROBE)));
        assert_eq!(
            res.path.parent(),
            Some(exe.as_path()),
            "the parent must be the exe directory"
        );
        assert!(
            res.warnings.iter().any(|w| w.contains("model directory missing")),
            "a missing model directory must be reported: {:?}",
            res.warnings
        );
    }

    /// Absolute paths are used as-is (no drift), but a WARN must flag the deviation from the "relative to exe" principle
    #[test]
    fn absolute_name_used_as_is_with_deviation_warn() {
        let d = std::env::temp_dir().join(format!("viq_pathunify_abs_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        let res = resolve_data_dir(d.to_str().unwrap());
        assert_eq!(res.source, "absolute path from settings");
        assert_eq!(res.path, abs_norm(&d));
        assert!(
            res.warnings
                .iter()
                .any(|w| w.contains("absolute path") && w.contains("departs")),
            "an absolute path must warn about departing from the principle: {:?}",
            res.warnings
        );
        assert!(
            !res.warnings.iter().any(|w| w.contains("model directory missing")),
            "an existing directory must not be reported missing: {:?}",
            res.warnings
        );
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn config_paths_are_userdata_relative() {
        let exe = exe_dir().unwrap();
        let ud = exe.join("UserData");
        assert_eq!(config_path("settings.json"), ud.join("settings.json"));
        assert_eq!(config_path("groups.json"), ud.join("groups.json"));
        assert_eq!(config_path("commands.json"), ud.join("commands.json"));
    }
}
