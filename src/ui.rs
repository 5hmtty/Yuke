//! egui UI: a three-tab console (Home / Commands / Settings).

//! Threading model: eframe renders on the main thread; the audio thread (cpal) keeps calling add_audio;
//! both share `Arc<Mutex<LiveProcessor>>` and each frame locks it to read a state snapshot. CaptureContext /
//! MicMonitor hold the cpal::Stream (which is !Send), which is legal on the eframe main thread.

//! Capture lifecycle: startup opens only the UI; the microphone is opened by "Start recognition" on the home
//! tab (CaptureContext owns the stream, dropping it stops capture). Config changes are written immediately and rebuild LiveProcessor.

use crate::gate::LiveProcessor;
use crate::live::{self, CaptureContext};
use eframe::egui;
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

/// Load a system Chinese font as fallback (egui's default fonts carry no CJK glyphs, so they render as blank boxes).
/// Priority: SimHei (single-face TTF, the most reliably parsed) -> Microsoft YaHei (TTC, index 0) -> SimSun
/// Missing-Chinese-font flag (set true at the start of setup_chinese_fonts, cleared once a font loads)
pub static FONT_MISSING: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);

pub fn setup_chinese_fonts(ctx: &egui::Context) {
    FONT_MISSING.store(true, std::sync::atomic::Ordering::Relaxed);
    let candidates = [
        "C:\\Windows\\Fonts\\simhei.ttf", // SimHei
        "C:\\Windows\\Fonts\\msyh.ttc",   // Microsoft YaHei (bundled since Windows 7)
        "C:\\Windows\\Fonts\\simsun.ttc", // SimSun
    ];
    for path in candidates {
        if let Ok(bytes) = std::fs::read(path) {
            let mut fonts = egui::FontDefinitions::default();
            fonts.font_data.insert(
                "chinese".to_owned(),
                Arc::new(egui::FontData::from_owned(bytes)),
            );
            // Symbol fallback: SimHei (GBK) lacks circled symbols such as U+24D8, so Segoe UI Symbol fills the gap
            if let Ok(sym) = std::fs::read("C:/Windows/Fonts/seguisym.ttf") {
                fonts.font_data.insert(
                    "symbol".to_owned(),
                    Arc::new(egui::FontData::from_owned(sym)),
                );
            }
            for fam in [egui::FontFamily::Proportional, egui::FontFamily::Monospace] {
                fonts
                    .families
                    .entry(fam.clone())
                    .or_default()
                    .push("chinese".to_owned());
                fonts
                    .families
                    .entry(fam)
                    .or_default()
                    .push("symbol".to_owned());
            }
            ctx.set_fonts(fonts);
            FONT_MISSING.store(false, std::sync::atomic::Ordering::Relaxed);
            return;
        }
    }
    eprintln!("warning: no Chinese font found (checked simhei.ttf/msyh.ttc/simsun.ttc)");
}

fn window_title() -> String {
    t(k::UI_APP_NAME).to_string()
}

const WARN_COLOR: egui::Color32 = egui::Color32::from_rgb(230, 90, 90);

const LABEL_VOICE_TOGGLE: &str = k::UI_LBL_VOICE_TOGGLE;
/// Home: key toggle (paired with the voice toggle)
const LABEL_KEY_TOGGLE: &str = k::UI_LBL_KEY_TOGGLE;


fn settings_path() -> PathBuf {
    crate::config_path("settings.json")
}

/// Pronunciation group file: **the only rule is `<exe dir>/config/groups.json`** (same path rule as settings_path).
fn groups_path() -> PathBuf {
    crate::config_path("groups.json")
}

mod dialogs;
mod logs;
mod pages;

// -- Tabs --

#[derive(PartialEq, Clone, Copy)]
enum Tab {
    Home,     // Tab 1: Home
    Pron,     // Tab 2: Commands
    Settings, // Tab 3: Settings
}

#[derive(Clone)]
struct Settings {
    hotkey: String,             // Key toggle hotkey (e.g. "CTRL+F8"; empty = unset)
    voice_toggle_cmd: String,   // Voice toggle command id: saying that pronunciation flips the global switch; empty = disabled
    input_mode: u8,             // Key injection mode (keys::MODE_*)
    target_window: String,      // Game window title keyword (empty = unspecified)
    device: String,             // Microphone name (empty = system auto-select)
    record_enabled: bool,       // Save recordings (session audio.wav)
    auto_start_recognize: bool, // Start recognition automatically at launch (off by default)
    v2_min_conf: f64,
    refr_s: f64, // Repeat guard (emission-level refractory, seconds; default 0.15; 0 = off)
    act_gate_db: f64,
    act_gate_auto: bool,
    frame_mean_norm: bool,
    aec_mode: u8,
    /// Response speed: identity-window right lookahead (seconds). 0 = purely causal and fastest, 0.13 = most accurate; applies after a restart.
    ident_fwd_sec: f64,
    /// Noise-robust model: uses encoder-v2-noise-*.onnx; switching requires a restart and rebuilding every template.
    noise_model: bool,
    vad_gate: bool,
    sv_gate: bool,
    hotkey_mode: u8,
    // default) 1 = hold to pause (hold pauses, release resumes) 2 = hold to input (hold enables, release pauses)
    update_check_enabled: bool,
    language: String, // UI language ID (lang/<ID>.txt; CN by default; hot-switched from the settings dropdown)
}

impl Default for Settings {
    fn default() -> Self {
        Settings {
            hotkey: String::new(),
            voice_toggle_cmd: String::new(),
            input_mode: crate::keys::MODE_STANDARD,
            target_window: String::new(),
            device: String::new(),
            record_enabled: false,
            auto_start_recognize: false,
            v2_min_conf: 0.6,
            refr_s: 0.15,
            act_gate_db: -50.0,
            act_gate_auto: true,
            frame_mean_norm: true,
            aec_mode: 0,
            ident_fwd_sec: 0.13,
            noise_model: false,
            vad_gate: false,
            sv_gate: false,
            hotkey_mode: 0,
            update_check_enabled: true,
            language: crate::lang::DEFAULT_LANG.to_string(),
        }
    }
}

fn voice_toggle_from_raw(raw: &serde_json::Value) -> String {
    raw.get("voice_toggle_cmd")
        .and_then(|x| x.as_str())
        .unwrap_or("")
        .to_string()
}

fn mute_passcode_key(
    commands: &mut std::collections::HashMap<String, crate::config::Command>,
    passcode_cmd: &str,
) {
    if passcode_cmd.is_empty() {
        return;
    }
    if let Some(c) = commands.get_mut(passcode_cmd) {
        c.key.clear();
    }
}

fn load_settings() -> (Settings, serde_json::Value) {
    let raw: serde_json::Value = std::fs::read_to_string(settings_path())
        .ok()
        .and_then(|t| serde_json::from_str(&t).ok())
        .unwrap_or_else(|| serde_json::json!({}));
    (settings_from_raw(&raw), raw)
}

fn settings_from_raw(raw: &serde_json::Value) -> Settings {
    let d = Settings::default();
    let g = |k: &str| raw.get(k);
    Settings {
        hotkey: g("hotkey")
            .and_then(|x| x.as_str())
            .unwrap_or("")
            .to_string(),
        voice_toggle_cmd: voice_toggle_from_raw(raw),
        input_mode: g("input_mode")
            .and_then(|x| x.as_u64())
            .map(|n| n as u8)
            .unwrap_or(d.input_mode),
        target_window: g("target_window")
            .and_then(|x| x.as_str())
            .unwrap_or("")
            .to_string(),
        device: g("device")
            .and_then(|x| x.as_str())
            .unwrap_or("")
            .to_string(),
        record_enabled: g("record_enabled")
            .and_then(|x| x.as_bool())
            .unwrap_or(false),
        auto_start_recognize: g("auto_start_recognize")
            .and_then(|x| x.as_bool())
            .unwrap_or(false),
        v2_min_conf: g("v2_min_conf")
            .and_then(|x| x.as_f64())
            .unwrap_or(d.v2_min_conf),
        refr_s: g("refr_s")
            .and_then(|x| x.as_f64())
            .unwrap_or(d.refr_s),
        act_gate_db: g("act_gate_db")
            .and_then(|x| x.as_f64())
            .unwrap_or(d.act_gate_db),
        act_gate_auto: g("act_gate_auto")
            .and_then(|x| x.as_bool())
            .unwrap_or(d.act_gate_auto),
        frame_mean_norm: g("frame_mean_norm")
            .and_then(|x| x.as_bool())
            .unwrap_or(true),
        aec_mode: g("aec_mode")
            .and_then(|x| x.as_u64())
            .map(|n| match n as u8 {
                1 => 1,
                3 => 3,
                2 => 3,
                _ => 0,
            })
            .unwrap_or(0),
        vad_gate: g("vad_gate").and_then(|x| x.as_bool()).unwrap_or(false),
        sv_gate: g("sv_gate").and_then(|x| x.as_bool()).unwrap_or(false),
        ident_fwd_sec: g("ident_fwd_sec")
            .and_then(|x| x.as_f64())
            .filter(|v| (0.0..=0.13).contains(v))
            .unwrap_or(0.13),
        noise_model: g("noise_model").and_then(|x| x.as_bool()).unwrap_or(false),
        hotkey_mode: g("hotkey_mode")
            .and_then(|x| x.as_u64())
            .map(|n| n as u8)
            .filter(|&m| m <= 2)
            .unwrap_or(0),
        update_check_enabled: g("update_check_enabled")
            .and_then(|x| x.as_bool())
            .unwrap_or(true),
        language: g("language")
            .and_then(|x| x.as_str())
            .filter(|s| !s.is_empty())
            .map(|s| {
                match s {
                    "CN" => "zh",
                    "EN" => "en",
                    other => other,
                }
                .to_string()
            })
            .unwrap_or_else(crate::lang::detect_lang), // First run with no key: derive from the system locale (zh -> Chinese, otherwise English)
    }
}

#[derive(Clone)]
enum Confirm {
    /// Regenerate every pronunciation template (heavy but non-destructive: old templates remain until generation succeeds)
    RegenAll { cmds: usize, recs: usize },
    /// "Optimise recognition": switching this switches the front end, so existing templates were built from a different
    /// source and must be regenerated before it takes effect -- in both directions. Enabling and disabling both ask for
    /// confirmation; confirm = persist + prompt for restart (the rebuild happens automatically at load); cancel = the checkbox rolls back.
    BoostRecognition { enable: bool },
    /// Noise-robust model: switching changes the embedding space, so every template must be rebuilt; confirmation persists the setting and the rebuild happens automatically at load after restart.
    NoiseModel { enable: bool },
}

/// Double-click delete arming state: the first click arms it (the button turns red with "confirm delete"), the second executes;
/// it disarms automatically after 5 s. Confirmation dialogs are not used.
#[derive(Clone, PartialEq)]
enum ArmedDelete {
    Group(String),
    Command(String),
    Rec { cmd: String, file: String },
}

/// Sortable columns (one enum per table; the "actions" column and the select-all header are deliberately excluded, i.e. unsortable).
#[derive(Clone, Copy, PartialEq, Eq)]
enum MutexCol {
    Name,
    Mode,
    Key,
}
#[derive(Clone, Copy, PartialEq, Eq)]
enum CmdCol {
    Name,
    Mode,
    Key,
    Recs,
}
#[derive(Clone, Copy, PartialEq, Eq)]
enum RecCol {
    File,
    Source,
    Length,
    Time,
}

/// Header click: writes (sort key + direction) into `state` (`None` = the table's original order).
/// First click on a column -> ascending; clicking the **already active** column flips the direction.
pub(super) fn sort_header<K: Copy + PartialEq>(
    ui: &mut egui::Ui,
    text: &str,
    key: K,
    state: &mut Option<(K, bool)>,
) {
    let resp = ui
        .add(egui::Label::new(egui::RichText::new(text).weak()).sense(egui::Sense::click()))
        .on_hover_cursor(egui::CursorIcon::PointingHand);
    if resp.clicked() {
        *state = match *state {
            Some((k, desc)) if k == key => Some((key, !desc)),
            _ => Some((key, false)),
        };
    }
}

/// Generic table sorting: `key_of` extracts the comparison key, `tie_of` a stable tiebreaker (usually the row id).

/// Ascending/descending applies **only to the primary key**; within equal keys the order is always ascending by the tiebreaker,
/// otherwise rows with equal keys would drift every frame as the sort implementation changes (rows that move by themselves are worse than a wrong order).
pub(super) fn sort_table_rows<T, K: Ord>(
    rows: &mut Vec<T>,
    desc: bool,
    key_of: impl Fn(&T) -> K,
    tie_of: impl Fn(&T) -> String,
) {
    let mut keyed: Vec<(K, String, T)> = Vec::with_capacity(rows.len());
    for r in rows.drain(..) {
        let k = key_of(&r);
        let t = tie_of(&r);
        keyed.push((k, t, r));
    }
    keyed.sort_by(|a, b| {
        let o = a.0.cmp(&b.0);
        let o = if desc { o.reverse() } else { o };
        o.then_with(|| a.1.cmp(&b.1))
    });
    rows.extend(keyed.into_iter().map(|(_, _, r)| r));
}

/// Rename dialog (group name / command display name)
#[derive(Clone)]
enum Rename {
    Group(String),
    CommandPhrase(String),
}

/// Macro edit row card: action-type dropdown index plus an argument buffer (key name or milliseconds as text)
#[derive(Clone)]
struct MacroRow {
    kind: usize, // Action kind: 0=down 1=up 2=press 3=wait
    param: String,
}

/// Text export convention: the initial that corresponds one-to-one with the kind index
const MACRO_KIND_ABBR: [&str; 4] = ["d", "u", "p", "w"];
const MK_PRESS: usize = 2;
const MK_WAIT: usize = 3;

/// Row argument validity (the red-border criterion): key actions must be non-empty and at most 32 characters; wait must be an integer in 0-10000
fn macro_row_valid(r: &MacroRow) -> bool {
    let p = r.param.trim();
    match r.kind {
        MK_WAIT => p
            .parse::<u32>()
            .map(|ms| ms <= crate::config::MACRO_MAX_WAIT_MS)
            .unwrap_or(false),
        _ => !p.is_empty() && p.chars().count() <= 32,
    }
}

fn macro_row_to_action(r: &MacroRow) -> Option<crate::config::MacroAction> {
    use crate::config::MacroAction;
    let p = r.param.trim().to_uppercase();
    match r.kind {
        0 if !p.is_empty() => Some(MacroAction::Down(p)),
        1 if !p.is_empty() => Some(MacroAction::Up(p)),
        MK_PRESS if !p.is_empty() => Some(MacroAction::Press(p)),
        MK_WAIT => p.parse::<u32>().ok().map(MacroAction::Wait),
        _ => None,
    }
}

/// The command's macro array in cfg -> row-card buffers (lenient: a corrupt entry becomes a default row and the UI flags it in red)
fn macro_rows_from_cfg(cfg: &serde_json::Value, cmd: &str) -> Vec<MacroRow> {
    let mut rows = Vec::new();
    let arr = cfg
        .get("commands")
        .and_then(|v| v.get(cmd))
        .and_then(|c| c.get("macro"))
        .and_then(|v| v.as_array());
    for it in arr.into_iter().flatten() {
        let action = it.get("action").and_then(|x| x.as_str()).unwrap_or("down");
        let kind = match action {
            "up" => 1,
            "press" => MK_PRESS,
            "wait" => MK_WAIT,
            _ => 0,
        };
        let param = if kind == MK_WAIT {
            it.get("ms")
                .and_then(|x| x.as_f64())
                .map(|ms| ms as u64)
                .unwrap_or(0)
                .to_string()
        } else {
            it.get("key")
                .and_then(|x| x.as_str())
                .unwrap_or("")
                .to_string()
        };
        rows.push(MacroRow { kind, param });
    }
    rows
}

/// Row-card buffers -> multi-line text (for export; the inverse of config::parse_macro_text)
fn macro_rows_to_text(rows: &[MacroRow]) -> String {
    rows.iter()
        .map(|r| {
            let kind = MACRO_KIND_ABBR.get(r.kind).copied().unwrap_or("d");
            format!("{kind} {}", r.param.trim())
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Display name of a trigger mode (commands.json mode value -> localized text)
fn mode_label(mode: &str) -> String {
    match mode {
        "toggle" => t(k::UI_OPT_TOGGLE_PRESS),
        "macro" => t(k::UI_OPT_MACRO),
        _ => t(k::UI_OPT_TAP),
    }
}

fn reentry_label(policy: &str) -> String {
    match policy {
        "concurrent" => t(k::UI_OPT_REENTRY_CONCURRENT),
        "interrupt" => t(k::UI_OPT_REENTRY_INTERRUPT),
        _ => t(k::UI_OPT_REENTRY_IGNORE),
    }
}

#[derive(PartialEq, Clone, Copy)]
enum CaptureJob {
    None,
    /// Started from a stopped state: the engine is loading in the background
    Starting,
    Restarting,
}

/// Result channel from the engine-loading thread to the UI thread
type ProcRx = std::sync::mpsc::Receiver<anyhow::Result<(Arc<Mutex<LiveProcessor>>, u64)>>;

fn engine_input_fingerprint_of(model_dir: &str, voice_toggle_cmd: &str) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    voice_toggle_cmd.hash(&mut h);
    let md = std::path::Path::new(model_dir);
    let mut files: Vec<std::path::PathBuf> = [
        "commands.json",
        "embed_proto_v2.json",
        "nuclei_templates_v2.json.enc",
    ]
    .iter()
    .map(|f| md.join(f))
    .collect();
    let mut stack = vec![md.join("command_recordings")];
    while let Some(d) = stack.pop() {
        if let Ok(es) = std::fs::read_dir(&d) {
            for e in es.flatten() {
                let p = e.path();
                if p.is_dir() {
                    stack.push(p);
                } else {
                    files.push(p);
                }
            }
        }
    }
    files.sort();
    for f in files {
        f.hash(&mut h);
        if let Ok(m) = std::fs::metadata(&f) {
            m.len().hash(&mut h);
            if let Ok(t) = m.modified() {
                if let Ok(d) = t.duration_since(std::time::UNIX_EPOCH) {
                    d.as_secs().hash(&mut h);
                }
            }
        }
    }
    h.finish()
}

enum ProtoKind {
    /// Single-command update: re-encode only that command's recordings and replace only its entry in embed_proto_v2.json
    Single(String),
    Full,
}

struct ProtoJob {
    kind: ProtoKind,
    rx: std::sync::mpsc::Receiver<anyhow::Result<String>>,
}

use crate::i18n_keys::k;
use crate::lang::t;

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum LogKind {
    Trigger,
    Note,
    Warn,
}

/// One trigger-log line: rendered text plus colour category (mutex-release footnotes are drawn grey).
#[derive(Clone)]
pub(super) struct LogRow {
    pub text: String,
    pub kind: LogKind,
}

pub struct ViqApp {
    tab: Tab,
    model_dir: String,

    // Capture (home-page master switch = ctx Some/None; enabled_flag = pause/resume of triggering, shared with the hotkey)
    ctx: Option<CaptureContext>,
    enabled_flag: Arc<std::sync::atomic::AtomicBool>,
    capture_job: CaptureJob,
    proc_rx: Option<ProcRx>,
    warming: bool,
    warm_rx: Option<ProcRx>,
    warm_proc: Option<Arc<Mutex<LiveProcessor>>>,
    warm_fingerprint: u64,
    /// Session recording held back while the engine restarts, plus the stop-time counter (saved if the rebuild fails, reused if it succeeds)
    pending_rec: Option<(Arc<crate::live::Recorder>, usize, String)>,
    /// A configuration save arrives while the engine is loading -> rebuild once more from the latest on-disk configuration
    reload_pending: bool,
    /// v2 pronunciation-template generation job (None = idle; buttons are disabled while generating to debounce)
    proto_job: Option<ProtoJob>,
    /// Completion text shown after a reload triggered by a template update (overrides the generic "configuration applied")
    proto_note: Option<String>,
    hotkey_manager: Option<global_hotkey::GlobalHotKeyManager>,
    hotkey_thread: Option<std::thread::JoinHandle<()>>,
    hotkey_stop: Option<Arc<std::sync::atomic::AtomicBool>>,
    capture_hotkey: bool,
    /// Last switch state the UI saw: the hotkey only flips enabled_flag, and once a frame detects the flip it
    /// routes through proc.set_enabled (gate.rs releases toggle-held keys while paused)
    last_enabled_seen: bool,
    /// Last trigger count the UI saw (voice passcode dispatch): compared against keypress_count every frame
    /// to detect a "new trigger", which together with last_command decides whether the passcode matched (see the dispatch block in update).
    /// Reset together with the new engine when it is rebuilt (poll_capture_job).
    last_seen_trigger_count: usize,

    // Trigger log + status
    log_lines: Vec<LogRow>,
    logged_events: usize,
    logged_swallowed: usize,
    logged_debug: usize,
    logged_energy: usize,
    status: String,
    log_file: Option<std::fs::File>,

    // Command configuration (commands.json edit buffer; changes are written immediately and rebuild the pipeline)
    cfg: serde_json::Value,
    cfg_path: PathBuf,

    // Devices
    devices: Vec<String>,
    selected_device: String,

    // Commands tab: list / edit popup
    pron_open: Option<String>, // Some(cmd) = the edit popup is open (no secondary page any more)
    rec_counts: BTreeMap<String, usize>, // Command -> recording count
    rec_list: Vec<crate::recordings::CmdRecording>, // Edit popup: recordings of the current command
    capture_key_for: Option<String>, // Command targeted by key capture
    key_edit: String,          // Key edit buffer for the edit popup
    key_mod_sel: String,
    min_conf_buf: String,
    act_gate_buf: String,
    refr_buf: String,
    aec_probe_rx: Option<std::sync::mpsc::Receiver<crate::comm::CommProbe>>, // Echo-cancellation availability probe (background thread)
    aec_probe: Option<crate::comm::CommProbe>,
    macro_edit: Vec<MacroRow>,
    macro_edit_for: Option<String>,
    /// Standalone macro editor window is open (Some(cmd) = that command's macro window; runtime state, never persisted)
    macro_editor_open: Option<String>,
    mutex_editor_open: Option<String>,
    mutex_edit: Vec<String>,
    mutex_edit_for: Option<String>,
    mutex_sort: Option<(MutexCol, bool)>,
    cmd_sort: Option<(CmdCol, bool)>,
    rec_sort: Option<(RecCol, bool)>,
    /// Mutex panel: checked rows pinned on top (on by default). Turning it off sorts purely by the sort key, which helps when hunting for a command.
    mutex_pin_checked: bool,
    key_edit_bufs: std::collections::BTreeMap<String, String>, // Inline key edit buffers in the command list (cmd -> value being typed)
    macro_import_open: bool,
    macro_import_buf: String,
    macro_import_errs: Vec<(usize, crate::config::MacroLineErr)>,
    win_geom_saved_ppp: Option<f32>,
    /// First-frame geometry/title self-check has run (monitor_size may be unavailable on the first frame; if so it stays unset and retries next frame)
    win_geom_checked: bool,
    /// Last window title sent (sent whenever it differs from window_title(), covering both startup and language switching)
    win_title: String,
    /// Current window geometry cached every frame (on_exit has no egui Context, so persisting depends on this)
    win_geom_live: Option<crate::window_geom::SavedGeometry>,
    /// Double-click delete arming: Some = first click done, the button shows a red "confirm delete"
    armed_delete: Option<(ArmedDelete, std::time::Instant)>,
    /// Commands covered by v2 pronunciation templates (cache of the embed_proto_v2.json keys; data source for the status badge)
    proto_cover: std::collections::BTreeSet<String>,
    /// "Multi-pronunciation" template-derived state (exp_nuclei >= 2 -> checked automatically)
    cmd_multi_auto: std::collections::HashMap<String, bool>,

    // Mic monitor for extra takes (16 kHz mono; dropping it stops capture)
    rec_mic: Option<crate::mic_monitor::MicMonitor>,
    rec_start_len: usize,
    rec_start_at: Option<std::time::Instant>,

    pron_new: bool,       // The edit dialog is in create mode (pron_open is None and the command is not yet persisted)
    pron_new_focus: bool, // In create mode, request focus for the name field on the first frame only
    group_switch_armed: Option<String>, // Armed target for the double-click confirmation while the group state is dirty
    about_open: bool,     // About dialog flag (runtime only, never persisted)
    add_phrase: String,

    // Groups (`<exe dir>/config/groups.json`: {"active","groups"})
    groups: BTreeMap<String, BTreeMap<String, bool>>,
    active_group: Option<String>, // Active group (the group the current checkboxes came from)
    group_sel: String,
    new_group_name: String,
    group_new_open: bool,
    confirm: Option<Confirm>, // Destructive-action confirmation dialog
    rename: Option<Rename>,   // Rename dialog
    rename_buf: String,
    fatal_error: Option<String>,

    // Settings tab
    settings: Settings,
    settings_raw: serde_json::Value, // Raw settings.json (used for merged saves)
    window_list: Vec<String>,
    last_target_window: String,
    /// "Start recognition automatically at launch" pending flag (settings-page general group; consumed on the first frame to avoid
    /// starting the engine-loading thread inside ViqApp::new, since many tests construct ViqApp directly)
    auto_start_pending: bool,

    update_check_tx: std::sync::mpsc::Sender<anyhow::Result<Option<String>>>,
    update_check_rx: std::sync::mpsc::Receiver<anyhow::Result<Option<String>>>,
    latest_update: Option<String>,
    update_check_inflight: bool, // A manual update check is in flight (button greyed out against double clicks)
    auto_update_launched: bool,  // The startup automatic check has been triggered (once per process)
    auto_update_at: Option<std::time::Instant>, // Fires the startup automatic check after a few seconds
}

impl ViqApp {
    pub fn new(model_dir: &str) -> Self {
        Self::new_inner(model_dir)
    }

    pub(crate) fn spawn_aec_probe(&mut self) {
        let want = self.settings.device.clone();
        let (ptx, prx) = std::sync::mpsc::channel();
        self.aec_probe_rx = Some(prx);
        self.aec_probe = None;
        std::thread::Builder::new()
            .name("aec-probe".into())
            .spawn(move || {
                let _ = ptx.send(crate::comm::probe(Some(want.as_str())));
            })
            .ok();
    }

    fn new_inner(model_dir: &str) -> Self {
        let (settings, settings_raw) = load_settings();
        crate::udkws_v2::set_debug_log(
            settings_raw
                .get("debug_log")
                .and_then(|x| x.as_bool())
                .unwrap_or(false),
        );
        let lang_dir = crate::exe_dir()
            .map(|d| d.join("lang"))
            .unwrap_or_else(|| std::path::PathBuf::from("lang"));
        crate::lang::set_language(&settings.language, &lang_dir);
        let (cfg, cfg_path) = load_cfg(model_dir);
        let (groups, active_group) = load_groups();
        #[cfg(test)]
        let devices: Vec<String> = Vec::new();
        #[cfg(not(test))]
        let devices = live::list_input_devices();
        let selected_device = if settings.device.is_empty() {
            devices
                .iter()
                .find(|d| d.contains("Microphone") || d.contains("麦克风"))
                .or_else(|| devices.first())
                .cloned()
                .unwrap_or_default()
        } else {
            settings.device.clone()
        };
        // Start recognition automatically at launch (settings-page general group, off by default): taken first, because the struct literal moves settings
        let auto_start_pending = settings.auto_start_recognize;
        let (update_check_tx, update_check_rx) = std::sync::mpsc::channel();
        let mut app = ViqApp {
            tab: Tab::Home,
            model_dir: model_dir.to_string(),
            ctx: None,
            enabled_flag: Arc::new(std::sync::atomic::AtomicBool::new(true)),
            capture_job: CaptureJob::None,
            proc_rx: None,
            warming: false,
            warm_rx: None,
            warm_proc: None,
            warm_fingerprint: 0,
            pending_rec: None,
            reload_pending: false,
            proto_job: None,
            proto_note: None,
            hotkey_manager: None,
            hotkey_thread: None,
            hotkey_stop: None,
            capture_hotkey: false,
            last_enabled_seen: true,
            last_seen_trigger_count: 0,
            log_lines: Vec::new(),
            logged_events: 0,
            logged_swallowed: 0,
            logged_debug: 0,
            logged_energy: 0,
            status: String::new(),
            log_file: None,
            cfg,
            cfg_path,
            devices,
            selected_device,
            pron_open: None,
            rec_counts: BTreeMap::new(),
            rec_list: Vec::new(),
            capture_key_for: None,
            key_edit: String::new(),
            key_mod_sel: String::new(),
            min_conf_buf: format!("{}", settings.v2_min_conf),
            act_gate_buf: format!("{}", settings.act_gate_db),
            refr_buf: format!("{}", settings.refr_s),
            aec_probe_rx: None,
            aec_probe: None,
            macro_edit: Vec::new(),
            macro_edit_for: None,
            macro_editor_open: None,
            mutex_editor_open: None,
            mutex_edit: Vec::new(),
            mutex_edit_for: None,
            mutex_sort: Some((MutexCol::Name, false)),
            cmd_sort: None,
            rec_sort: None,
            mutex_pin_checked: true,
            key_edit_bufs: std::collections::BTreeMap::new(),
            macro_import_open: false,
            macro_import_buf: String::new(),
            macro_import_errs: Vec::new(),
            win_geom_saved_ppp: None,
            win_geom_checked: false,
            win_title: String::new(),
            win_geom_live: None,
            armed_delete: None,
            proto_cover: std::collections::BTreeSet::new(),
            cmd_multi_auto: std::collections::HashMap::new(),
            rec_mic: None,
            rec_start_len: 0,
            rec_start_at: None,
            pron_new: false,
            pron_new_focus: false,
            group_switch_armed: None,
            about_open: false,
            add_phrase: String::new(),
            groups,
            active_group,
            group_sel: String::new(),
            new_group_name: String::new(),
            group_new_open: false,
            confirm: None,
            rename: None,
            rename_buf: String::new(),
            fatal_error: None,
            settings,
            settings_raw,
            window_list: Vec::new(),
            last_target_window: String::new(),
            // "Start recognition automatically" pending flag: consumed on the first frame (update); new does not start the engine
            auto_start_pending,
            update_check_tx,
            update_check_rx,
            latest_update: None,
            update_check_inflight: false,
            auto_update_launched: false,
            auto_update_at: None,
        };
        app.log_file = {
            let logs = crate::user_data_dir().join("logs");
            let _ = std::fs::create_dir_all(&logs);
            let ts = chrono::Local::now().format("%Y%m%d_%H%M%S");
            std::fs::File::create(logs.join(format!("ui_{ts}.log"))).ok()
        };
        let _ = std::fs::create_dir_all(crate::config_dir());
        app.reset_key_edit_bufs();
        app.refresh_rec_counts();
        app.refresh_proto_cover(); // Data source for the status badge (the embed_proto_v2.json command keys)
        app.group_sel = app
            .active_group
            .clone()
            .or_else(|| app.groups.keys().next().cloned())
            .unwrap_or_default();
        let judge_lines = {
            let mut enabled: Vec<String> = app
                .cfg
                .get("commands")
                .and_then(|v| v.as_object())
                .map(|o| {
                    o.iter()
                        .filter(|(_, c)| c.get("enabled").and_then(|v| v.as_bool()) != Some(false))
                        .map(|(n, _)| n.clone())
                        .collect()
                })
                .unwrap_or_default();
            enabled.sort();
            crate::config::judge_path_report(
                &enabled,
                &crate::config::v2_proto_commands(&app.model_dir),
                true,
            )
        };
        for line in &judge_lines {
            if line.starts_with('⚠') {
                app.log(line);
            }
        }
        if let Some(warn) = judge_lines.iter().find(|l| l.starts_with('⚠')) {
            app.status = warn.clone();
        }
        // Apply input mode / target window / key toggle at startup
        crate::keys::set_input_mode(app.settings.input_mode);
        crate::keys::set_target_window(&app.settings.target_window);
        app.window_list = crate::keys::list_window_titles();
        app.last_target_window = app.settings.target_window.clone();
        app.apply_hotkey();
        app
    }

    /// Write the file log (logs/ui_*.log); silent when it cannot be opened or written
    fn log(&mut self, msg: &str) {
        if let Some(f) = self.log_file.as_mut() {
            let line = format!("{} {msg}\n", chrono::Local::now().format("%H:%M:%S"));
            use std::io::Write;
            let _ = f.write_all(line.as_bytes());
            let _ = f.flush();
        }
    }

    /// Drop the inline key-edit buffer after a config reload: an external key change should be reflected rather than shown stale
    fn reset_key_edit_bufs(&mut self) {
        self.key_edit_bufs.clear();
    }

    /// Refresh command -> recording count (the command_recordings index)
    fn refresh_rec_counts(&mut self) {
        let mut m: BTreeMap<String, usize> = BTreeMap::new();
        for r in crate::recordings::list(&self.model_dir) {
            *m.entry(r.cmd).or_insert(0) += 1;
        }
        self.rec_counts = m;
    }

    /// Refresh the secondary-page recording list
    fn refresh_rec_list(&mut self, cmd: &str) {
        self.rec_list = crate::recordings::list_cmd(&self.model_dir, cmd);
    }

    /// Refresh the v2 template coverage set (data source for the status badge): the command keys in embed_proto_v2.json.
    /// Reading and decrypting has a cost, so only refresh when opening the edit popup, when a template job finishes, or at startup.
    fn refresh_proto_cover(&mut self) {
        self.proto_cover = crate::config::v2_proto_commands(&self.model_dir)
            .into_iter()
            .collect();
        self.cmd_multi_auto = crate::nuclei_cache::exp_nuclei_map(&self.model_dir)
            .into_iter()
            .map(|(k, n)| (k, n >= 2))
            .collect();
    }

    /// Command display name (phrase first; empty or missing falls back to the id)
    fn cmd_display(&self, id: &str) -> String {
        self.cfg
            .get("commands")
            .and_then(|v| v.get(id))
            .and_then(|c| c.get("phrase"))
            .and_then(|v| v.as_str())
            .filter(|s| !s.is_empty())
            .map(String::from)
            .unwrap_or_else(|| id.to_string())
    }

    /// Voice passcode dispatch (called every frame from update; probes test it directly): when the trigger count advances and
    /// last_command is the passcode, only enabled_flag is flipped (toggle, same as "start input").
    /// The key side is guaranteed elsewhere: the passcode's key was cleared at engine construction (mute_passcode_key), and
    /// gate neither presses nor records an event for an empty key, so this function need not (and must not) touch keys.
    fn voice_toggle_dispatch(&mut self, count: usize, last_cmd: &str) {
        if count == self.last_seen_trigger_count {
            return; // No new trigger (baseline already consumed)
        }
        self.last_seen_trigger_count = count;
        if last_cmd == self.settings.voice_toggle_cmd {
            let cur = self.enabled_flag.load(std::sync::atomic::Ordering::Relaxed);
            self.enabled_flag
                .store(!cur, std::sync::atomic::Ordering::Relaxed);
        }
    }

    fn close_edit_popup(&mut self) {
        if self.rec_mic.is_some() {
            self.stop_rec(); // Reads pron_open internally to locate the command, so it must run before Noneing it
        }
        let cmd = self.pron_open.clone();
        if self.pron_new {
            if let Some(c) = cmd.as_deref() {
                self.finalize_new_command(c);
            }
        }
        if let Some(cmd) = cmd.as_deref() {
            if self.rec_counts.get(cmd).copied().unwrap_or(0) > 0 && !self.proto_cover.contains(cmd)
            {
                self.start_proto_single(cmd);
            }
        }
        self.pron_open = None;
        // Clear the new-command marker and the name buffer together, leaving no residue
        self.pron_new = false;
        self.pron_new_focus = false;
        self.add_phrase.clear();
        self.capture_key_for = None;
        self.armed_delete = None;
        self.refresh_rec_counts();
    }

    /// Whether a delete target is currently armed (the second click must land within 5 s of the first).
    /// update() clears ArmedDelete every frame as a backstop.
    fn armed_is(&self, kind: &ArmedDelete) -> bool {
        matches!(&self.armed_delete,
            Some((a, t)) if a == kind && t.elapsed() < std::time::Duration::from_secs(5))
    }

    fn launch_osk(&mut self, capture_target: Option<&str>) {
        match spawn_osk() {
            Ok(_) => {
                if let Some(target) = capture_target {
                    // Enter capture mode automatically (osk key presses go through the capture path, recording chords in full)
                    self.capture_key_for = Some(target.to_string());
                    self.status = t(k::UI_HINT_PRESS_KEY);
                } else {
                    self.status = t(k::MSG_OSK_OPENED);
                }
                self.log(&t(k::MSG_LAUNCHING_OSK));
            }
            Err(e) => {
                self.status = crate::tfmt!(k::ERR_OSK_OPEN, e);
                self.log(&self.status.clone());
            }
        }
    }

    // -- Config persistence + pipeline rebuild --

    fn persist_cfg(&mut self) {
        let pretty = match serde_json::to_string_pretty(&self.cfg) {
            Ok(s) => s,
            Err(e) => {
                self.status = crate::tfmt!(k::ERR_CONFIG_SERIALIZE, e);
                return;
            }
        };
        if let Err(e) = write_json_atomic(&self.cfg_path, &pretty) {
            self.status = crate::tfmt!(k::ERR_CONFIG_WRITE, e);
            return;
        }
        self.reset_key_edit_bufs();
        if self.ctx.is_some() {
            self.restart_capture_async(&t(k::MSG_CONFIG_CHANGED));
        } else if self.capture_job != CaptureJob::None {
            self.reload_pending = true; // Config changed while the engine is loading -> rebuild from the latest on-disk config once loading finishes
            self.status = t(k::MSG_SAVED_APPLY_AFTER_LOAD);
        } else {
            self.status = t(k::MSG_SAVED_NEXT_START);
        }
        self.log(&crate::tfmt!("config changed: {}", self.status));
    }

    fn restart_capture_async(&mut self, why: &str) {
        // Engine-level change: the warm-up result is void (including an in-flight build -- later loading reflects the newest inputs anyway)
        self.warming = false;
        self.warm_rx = None;
        self.warm_proc = None;
        if self.capture_job != CaptureJob::None {
            self.reload_pending = true;
            return;
        }
        let Some(c) = self.ctx.take() else { return };
        let (count, last) = {
            let mut p = c.proc.lock().unwrap_or_else(|e| e.into_inner());
            p.release_all_toggles();
            p.cancel_all_macros();
            (p.keypress_count(), p.last_command())
        };
        self.pending_rec = Some((c.rec.clone(), count, last));
        drop(c); // Stream stopped -> the audio thread winds down on its alive flag (live.rs)
        self.capture_job = CaptureJob::Restarting;
        self.status = crate::tfmt!(k::MSG_SAVED_REBUILDING_ENGINE, why);
        self.log(&crate::tfmt!(k::MSG_CONFIG_APPLIED_RESTART_CAPTURE, why));
        self.spawn_engine_load();
    }

    fn spawn_engine_thread(&mut self) -> Option<ProcRx> {
        let (tx, rx) = std::sync::mpsc::channel();
        let model_dir = self.model_dir.clone();
        // Reuse the global switch Arc: the paused state survives a restart, and hotkey and checkbox stay consistent across engine instances
        let flag = self.enabled_flag.clone();
        // Voice passcode: the passcode pronunciation's key mapping is cleared on the engine side (mute_passcode_key) --
        // recognition continues, a match presses nothing, and the dispatch layer (UI frame) flips the switch from keypress_count/last_command
        let mute_cmd = self.settings.voice_toggle_cmd.clone();
        let ov = crate::udkws_v2::EngineOverrides {
            refr_s: Some(self.settings.refr_s),
            min_conf: Some(self.settings.v2_min_conf),
            act_db: Some(self.settings.act_gate_db),
            auto_gate: Some(self.settings.act_gate_auto),
        };
        let spawned = std::thread::Builder::new()
            .name("engine-load".into())
            .spawn(move || {
                let t0 = std::time::Instant::now();
                let res = crate::config::CommandConfig::load_for_model(&model_dir)
                    .map(|mut cfg| {
                        mute_passcode_key(&mut cfg.commands, &mute_cmd);
                        cfg
                    })
                    .and_then(|cfg| {
                        live::prepare_processor_with_overrides(&model_dir, cfg, flag, ov)
                    })
                    .map(|proc| {
                        let fp = engine_input_fingerprint_of(&model_dir, &mute_cmd);
                        (proc, fp)
                    });
                println!("engine load took {:.1}s", t0.elapsed().as_secs_f32());
                let _ = tx.send(res);
            });
        if spawned.is_err() {
            return None;
        }
        Some(rx)
    }

    fn spawn_engine_load(&mut self) {
        match self.spawn_engine_thread() {
            Some(rx) => self.proc_rx = Some(rx),
            None => {
                self.capture_job = CaptureJob::None;
                self.status = t(k::ERR_ENGINE_THREAD_SPAWN);
            }
        }
    }

    fn spawn_engine_warmup(&mut self) {
        if self.warming || self.warm_proc.is_some() || self.ctx.is_some() {
            return;
        }
        match self.spawn_engine_thread() {
            Some(rx) => {
                self.warm_rx = Some(rx);
                self.warming = true;
            }
            None => {}
        }
    }

    /// Engine build-input fingerprint: the passcode key plus, for every file the engine consumes in the model directory (command config,
    /// pronunciation templates, learning data, nucleus cache, recording tree), path + size + mtime. If any of them changes during
    /// warm-up the fingerprint no longer matches, so the warm instance is dropped and the existing background load rebuilds from the newest inputs.
    fn engine_input_fingerprint(&self) -> u64 {
        engine_input_fingerprint_of(&self.model_dir, &self.settings.voice_toggle_cmd)
    }

    /// Warm-up result polling (UI frame): stage a successful build, silently drop a failed one (starting recognition then reports through the normal path)
    fn poll_warmup(&mut self) {
        if !self.warming {
            return;
        }
        let got = match self.warm_rx.as_ref().unwrap().try_recv() {
            Ok(r) => Some(r),
            Err(std::sync::mpsc::TryRecvError::Empty) => None,
            Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                Some(Err(anyhow::Error::msg(t(k::ERR_WARMUP_THREAD_EXIT))))
            }
        };
        if let Some(res) = got {
            self.warming = false;
            self.warm_rx = None;
            match res {
                Ok((proc, fp)) => {
                    self.warm_proc = Some(proc); // Silent: losing the home page's loading state is the signal
                    self.warm_fingerprint = fp;
                }
                Err(e) => self.log(&crate::tfmt!("engine warm-up failed (harmless): {e}", e)),
            }
        }
    }

    /// Current activity gate: use the live value while an engine exists (including the auto-gate's evolved value), otherwise the configured one.
    /// Consumed by template regeneration so the template gate and the runtime gate share one source (symmetric, preventing double fires in auto mode).
    fn current_gate(&self) -> f64 {
        self.ctx
            .as_ref()
            .and_then(|c| c.proc.lock().ok())
            .and_then(|p| p.v2.as_ref().map(|v| v.gate()))
            .unwrap_or(self.settings.act_gate_db)
    }

    /// Poll the background engine load every frame; once ready, build the stream on the UI thread (cpal::Stream is not Send)
    fn poll_capture_job(&mut self) {
        if self.proc_rx.is_none() {
            return;
        }
        let rx = self.proc_rx.as_ref().unwrap();
        let got = match rx.try_recv() {
            Ok(r) => Some(r),
            Err(std::sync::mpsc::TryRecvError::Empty) => None,
            Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                Some(Err(anyhow::Error::msg(t(k::ERR_ENGINE_THREAD_EXIT))))
            }
        };
        let Some(res) = got else { return };
        self.proc_rx = None;
        let restarting = self.capture_job == CaptureJob::Restarting;
        self.capture_job = CaptureJob::None;
        match res {
            Ok((proc, _)) => self.activate_proc(proc, restarting),
            Err(e) => {
                if let Some((r, c, l)) = self.pending_rec.take() {
                    r.save(c, &l);
                }
                self.proto_note = None;
                self.status = crate::tfmt!(k::ERR_ENGINE_LOAD, e);
                self.log(&crate::tfmt!(k::ERR_ENGINE_LOAD, e));
                // Report loudly (error box + log with the real reason) -- v2-only: a construction failure must never fall back silently
                self.fatal_error = Some(crate::tfmt!(k::ERR_ENGINE_LOAD, e));
            }
        }
        if self.reload_pending {
            self.reload_pending = false;
            self.restart_capture_async(&t(k::MSG_CONFIG_CHANGED_WHILE_LOADING));
        }
    }

    /// Engine built -> start capture (on the UI thread; shared by poll_capture_job and the warm-up fast path).
    /// cpal::Stream is not Send, so stream creation must stay on the UI thread -- engine loading already finished in the background.
    fn activate_proc(&mut self, proc: Arc<Mutex<LiveProcessor>>, restarting: bool) {
        {
            let mut p = proc.lock().unwrap_or_else(|e| e.into_inner());
            p.set_passcode_cmd(&self.settings.voice_toggle_cmd);
        }
        let pend = self.pending_rec.take();
        let rec = pend.as_ref().map(|(r, _, _)| r.clone());
        match live::CaptureContext::start_with_proc(
            &self.model_dir,
            if self.selected_device.is_empty() {
                None
            } else {
                Some(&self.selected_device)
            },
            proc,
            rec,
        ) {
            Ok(ctx) => {
                ctx.rec.log(&crate::tfmt!(
                    "recognition started (device {})",
                    if self.selected_device.is_empty() {
                        "default".to_string()
                    } else {
                        self.selected_device.clone()
                    }
                ));
                self.logged_events = 0; // Count the new engine's event stream from the start
                self.logged_swallowed = 0;
                self.logged_debug = 0; // Count the new engine's diagnostic stream from the start ([heartbeat] / decision detail)
                self.logged_energy = 0; // Count the new engine's energy timeline from the start ([E] RMS)
                self.last_enabled_seen =
                    self.enabled_flag.load(std::sync::atomic::Ordering::Relaxed);
                self.last_seen_trigger_count = 0; // Count the new engine's triggers from the start (passcode dispatch baseline)
                self.ctx = Some(ctx);
                self.status = if restarting {
                    t(k::MSG_CONFIG_APPLIED)
                } else {
                    t(k::MSG_RECOGNITION_STARTED)
                };
                self.log(if restarting {
                    "configuration hot-applied"
                } else {
                    "recognition started"
                });
                // Reload triggered by a template update: overridden with a dedicated completion message (including the recording count)
                if restarting {
                    if let Some(note) = self.proto_note.take() {
                        self.status = note.clone();
                        self.log(&note);
                    }
                }
            }
            Err(e) => {
                // Stream creation failed: in the restart case, save the held-back session so no Recorder is left dangling
                if let Some((r, c, l)) = pend {
                    r.save(c, &l);
                }
                self.proto_note = None;
                self.status = crate::tfmt!(k::ERR_STARTUP, e);
                self.log(&crate::tfmt!(k::ERR_STARTUP, e));
                // Report loudly (error box + log with the real reason) -- v2-only: a construction failure must never fall back silently
                self.fatal_error = Some(crate::tfmt!(k::ERR_STARTUP, e));
            }
        }
    }

    // -- v2 pronunciation template (embed_proto_v2.json) generation --

    fn proto_busy(&self) -> bool {
        self.proto_job.is_some()
    }

    /// Regenerate the v2 template for one command (button on the secondary page of tab 2): encode that command's recordings on a
    /// background thread and replace only its entry in embed_proto_v2.json; then reload the engine automatically.
    /// The front-end switch changed but this process has not restarted yet (settings disagree with the value locked in-process). Pure predicate.
    fn template_params_pending_restart(&self) -> bool {
        self.settings.frame_mean_norm != crate::engine::udkws_v2::frame_mean_norm()
            || (self.settings.ident_fwd_sec - crate::engine::udkws_v2::ident_fwd_sec_env()).abs()
                > 1e-9
            || self.settings.noise_model != crate::engine::udkws_v2::noise_encoder_on()
    }

    /// Whether the current settings and recordings already differ from those that built the templates (decides the automatic rebuild at startup).
    fn templates_need_rebuild(&self) -> bool {
        let cmds: Vec<String> = self
            .cfg
            .get("commands")
            .and_then(|v| v.as_object())
            .map(|o| o.keys().cloned().collect())
            .unwrap_or_default();
        crate::engine::proto_meta::needs_rebuild(&self.model_dir, &cmds)
    }

    fn start_proto_single(&mut self, cmd: &str) {
        // Same as start_proto_full: never recompute while the front end awaits a restart (it would use the old one)
        if self.template_params_pending_restart() {
            self.status = t(k::MSG_REGEN_NEED_RESTART);
            return;
        }
        if self.proto_busy() {
            self.status = t(k::MSG_TEMPLATE_BUSY);
            return;
        }
        if self.capture_job != CaptureJob::None || self.warming {
            self.status = if self.warming {
                t(k::MSG_ENGINE_WARMING_WAIT)
            } else {
                t(k::MSG_ENGINE_LOADING_BUILD)
            };
            return;
        }
        let n = self.rec_counts.get(cmd).copied().unwrap_or(0);
        if n == 0 {
            self.status = crate::tfmt!(k::MSG_NO_RECS_BUILD_TEMPLATE, cmd);
            return;
        }
        let (tx, rx) = std::sync::mpsc::channel();
        self.proto_job = Some(ProtoJob {
            kind: ProtoKind::Single(cmd.to_string()),
            rx,
        });
        let model_dir = self.model_dir.clone();
        let owned = cmd.to_string();
        let msg = crate::tfmt!(k::MSG_TEMPLATE_UPDATED, owned, n);
        let gate = self.current_gate(); // Template gate and runtime gate share one source (symmetric, preventing double fires in auto mode)
        let spawned = std::thread::Builder::new()
            .name("proto-gen".into())
            .spawn(move || {
                let _ = tx.send(
                    crate::udkws_v2::gen_embed_proto_single(&model_dir, None, &owned, gate)
                        .map(|_| msg),
                );
            });
        if spawned.is_err() {
            self.proto_job = None;
            self.status = t(k::ERR_TEMPLATE_THREAD_SPAWN);
            return;
        }
        self.status = crate::tfmt!(k::MSG_TEMPLATE_BUILDING, cmd, n);
        self.log(&crate::tfmt!(k::MSG_TEMPLATE_BUILD_START, cmd, n));
    }

    fn start_proto_full(&mut self) {
        // Template parameters changed without a restart means a recompute would use the old ones (wasted, and still rebuilt at restart)
        if self.template_params_pending_restart() {
            self.status = t(k::MSG_REGEN_NEED_RESTART);
            return;
        }
        if self.proto_busy() {
            self.status = t(k::MSG_TEMPLATE_BUSY);
            return;
        }
        if self.capture_job != CaptureJob::None || self.warming {
            self.status = if self.warming {
                t(k::MSG_ENGINE_WARMING_WAIT)
            } else {
                t(k::MSG_ENGINE_LOADING_BUILD)
            };
            return;
        }
        let n_cmds = self.rec_counts.len();
        let n_recs: usize = self.rec_counts.values().sum();
        if n_cmds == 0 {
            self.status = t(k::ERR_NO_RECS_NO_TEMPLATE);
            return;
        }
        let (tx, rx) = std::sync::mpsc::channel();
        self.proto_job = Some(ProtoJob {
            kind: ProtoKind::Full,
            rx,
        });
        let model_dir = self.model_dir.clone();
        let msg = crate::tfmt!(k::MSG_TEMPLATE_UPDATED_ALL, n_cmds, n_recs);
        let gate = self.current_gate(); // Drift pinning and auto mode: the template gate uses the gate in force (including a handed-over one)
        let spawned = std::thread::Builder::new()
            .name("proto-gen".into())
            .spawn(move || {
                let _ =
                    tx.send(crate::udkws_v2::gen_embed_proto(&model_dir, None, gate).map(|_| msg));
            });
        if spawned.is_err() {
            self.proto_job = None;
            self.status = t(k::ERR_TEMPLATE_THREAD_SPAWN);
            return;
        }
        self.status = crate::tfmt!(k::MSG_TEMPLATE_BUILDING_ALL, n_cmds, n_recs);
        self.log(&crate::tfmt!(
            "generating all pronunciation templates ({n_cmds} pronunciations, {n_recs} recordings)",
            n_cmds,
            n_recs
        ));
    }

    /// Poll the template generation result every frame: on success, hook into the existing reload chain according to engine
    /// state (running or loading both go through restart_capture_async, which converts to reload_pending while loading).
    fn poll_proto_job(&mut self) {
        let Some(job) = &self.proto_job else { return };
        let got = match job.rx.try_recv() {
            Ok(r) => Some(r),
            Err(std::sync::mpsc::TryRecvError::Empty) => None,
            Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                Some(Err(anyhow::Error::msg(t(k::ERR_TEMPLATE_THREAD_EXIT))))
            }
        };
        let Some(res) = got else { return };
        self.proto_job = None;
        match res {
            Ok(msg) => {
                self.log(&msg);
                self.refresh_proto_cover(); // The template coverage set changed, so refresh the badge immediately
                if self.ctx.is_some() {
                    // Engine running: show the "engine reloading" text for the 1-3 s gap, and
                    // poll_capture_job overwrites it with the final result via proto_note
                    self.proto_note = Some(crate::tfmt!(k::MSG_ENGINE_RELOADED, msg));
                    self.restart_capture_async(&t(k::MSG_TEMPLATE_UPDATE_TITLE));
                } else if self.capture_job != CaptureJob::None {
                    self.proto_note = Some(crate::tfmt!(k::MSG_ENGINE_RELOADED, msg));
                    self.status = crate::tfmt!(k::MSG_TEMPLATE_RELOAD_PENDING, msg);
                    self.restart_capture_async(&t(k::MSG_TEMPLATE_UPDATE_TITLE)); // While loading this becomes reload_pending automatically
                } else {
                    self.status = crate::tfmt!(k::MSG_APPLIED_NEXT_START, msg);
                    if !self.auto_start_pending && !self.warming && self.warm_proc.is_none() {
                        self.spawn_engine_warmup();
                    }
                }
            }
            Err(e) => {
                self.status = crate::tfmt!(k::ERR_TEMPLATE_BUILD, e);
                self.log(&crate::tfmt!(k::ERR_TEMPLATE_BUILD, e));
                if self.ctx.is_none()
                    && self.capture_job == CaptureJob::None
                    && !self.warming
                    && self.warm_proc.is_none()
                    && !self.auto_start_pending
                {
                    // The app must stay usable even when the template build failed: bring the engine up with the old on-disk templates
                    self.spawn_engine_warmup();
                }
            }
        }
    }

    /// Modify one field of commands[name]
    fn set_cmd_field(&mut self, name: &str, field: &str, value: serde_json::Value) {
        if let Some(cmds) = self.cfg.get_mut("commands").and_then(|v| v.as_object_mut()) {
            if let Some(cmd) = cmds.get_mut(name) {
                cmd[field] = value;
            }
        }
    }

    fn mutex_set_of(&self, cmd: &str) -> Vec<String> {
        let Some(cmds) = self.cfg.get("commands").and_then(|v| v.as_object()) else {
            return Vec::new();
        };
        let Some(this) = cmds.get(cmd) else {
            return Vec::new();
        };
        let mine = mutex_list_of(this);
        let mut out: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
        for (other, c) in cmds {
            if other == cmd {
                continue;
            }
            if mine.iter().any(|x| x == other) || mutex_list_of(c).iter().any(|x| x == cmd) {
                out.insert(other.clone());
            }
        }
        out.into_iter().collect()
    }

    pub(super) fn apply_mutex_edit(&mut self, cmd: &str) {
        let sel: std::collections::BTreeSet<String> = self
            .mutex_edit
            .iter()
            .filter(|s| s.as_str() != cmd)
            .cloned()
            .collect();
        let others: Vec<String> = self
            .cfg
            .get("commands")
            .and_then(|v| v.as_object())
            .map(|m| m.keys().cloned().collect())
            .unwrap_or_default();
        for other in others {
            if other == cmd {
                continue;
            }
            set_mutex_pair(&mut self.cfg, cmd, &other, sel.contains(&other));
        }
        set_mutex_list_of(&mut self.cfg, cmd, sel.iter().cloned().collect());
        self.persist_cfg();
        self.status = crate::lang::tf(k::MSG_MUTEX_SAVED, &[("{n}", &sel.len().to_string())]);
        let line = self.status.clone();
        self.log(&line);
    }

    /// Whether the panel buffer differs from cfg (the criterion for "apply on switch" / "save on close").

    /// It exists to **avoid pointless writes**: apply_mutex_edit ends with persist_cfg(), which while running triggers an
    /// engine rebuild, and switching commands is frequent -- writing with no change is pure waste (and flashes "mutex saved" in the status bar).
    fn mutex_dirty(&self, cmd: &str) -> bool {
        let mut buf: Vec<&str> = self
            .mutex_edit
            .iter()
            .map(String::as_str)
            .filter(|s| *s != cmd)
            .collect();
        buf.sort_unstable();
        let cur = self.mutex_set_of(cmd);
        let mut cur: Vec<&str> = cur.iter().map(String::as_str).collect();
        cur.sort_unstable();
        buf != cur
    }

    pub(super) fn switch_mutex_editor_to(&mut self, new_cmd: &str) {
        let Some(old) = self.mutex_editor_open.clone() else {
            return;
        };
        if old == new_cmd {
            return;
        }
        if self.mutex_dirty(&old) {
            self.apply_mutex_edit(&old);
        }
        self.mutex_editor_open = Some(new_cmd.to_string());
        self.mutex_edit_for = None;
    }

    // -- Capture control (home page and settings share the master switch) --

    fn toggle_capture(&mut self) {
        if self.capture_job != CaptureJob::None {
            self.status = t(k::MSG_ENGINE_LOADING);
            return;
        }
        if self.ctx.is_none() && self.proto_busy() {
            self.status = t(k::MSG_TEMPLATE_BUILDING_WAIT_START);
            return;
        }
        if self.ctx.is_some() {
            // Stop: release toggle-held keys + cancel running macros + drop the stream (stop capture) + save the session recording
            let mut saved = false;
            if let Some(c) = self.ctx.take() {
                let (count, last) = {
                    let mut p = c.proc.lock().unwrap_or_else(|e| e.into_inner());
                    p.release_all_toggles();
                    p.cancel_all_macros();
                    (p.keypress_count(), p.last_command())
                };
                saved = c.rec.save(count, &last);
            }
            self.status = if saved {
                t(k::MSG_RECOGNITION_STOPPED_SAVED)
            } else {
                t(k::MSG_RECOGNITION_STOPPED)
            };
            self.log(&t(k::MSG_RECOGNITION_STOPPED));
            return;
        }
        if self.rec_mic.is_some() {
            self.status = t(k::MSG_REC_BUSY_BEFORE_START);
            return;
        }
        let enabled_n = self
            .cfg
            .get("commands")
            .and_then(|v| v.as_object())
            .map(|o| {
                o.values()
                    .filter(|c| c.get("enabled").and_then(|x| x.as_bool()).unwrap_or(false))
                    .count()
            })
            .unwrap_or(0);
        if enabled_n == 0 {
            self.status = t(k::MSG_WARN_NO_ENABLED);
            self.log(&t(k::MSG_WARN_ZERO_ENABLED));
        }
        if self.proto_busy() {
            self.status = t(k::MSG_TEMPLATE_BUILDING_WAIT_START);
            return;
        }
        if let Some(proc) = self.warm_proc.take() {
            if self.engine_input_fingerprint() == self.warm_fingerprint {
                self.activate_proc(proc, false);
                return;
            }
        }
        if self.warming {
            if let Some(rx) = self.warm_rx.take() {
                self.warming = false;
                self.proc_rx = Some(rx);
                self.capture_job = CaptureJob::Starting;
                self.status = t(k::MSG_ENGINE_LOADING_BG);
                return;
            }
        }
        self.capture_job = CaptureJob::Starting;
        self.status = t(k::MSG_ENGINE_LOADING_BG);
        self.spawn_engine_load();
    }

    // -- Groups (full CRUD) --

    /// Snapshot of the enabled state of every pronunciation
    fn current_enabled_map(&self) -> BTreeMap<String, bool> {
        let mut map = BTreeMap::new();
        if let Some(cmds) = self.cfg.get("commands").and_then(|v| v.as_object()) {
            for (cname, c) in cmds {
                let on = c.get("enabled").and_then(|v| v.as_bool()).unwrap_or(true);
                map.insert(cname.clone(), on);
            }
        }
        map
    }

    /// Whether the current checkboxes deviate from the active group (None = no active group)
    fn group_dirty(&self) -> Option<bool> {
        let name = self.active_group.as_ref()?;
        let Some(map) = self.groups.get(name) else {
            return Some(false);
        };
        let cur = self.current_enabled_map();
        Some(
            cur.iter()
                .any(|(k, v)| map.get(k).map(|g| g != v).unwrap_or(false))
                || map
                    .iter()
                    .any(|(k, v)| cur.get(k).map(|c| c != v).unwrap_or(false)),
        )
    }

    /// Apply the selected group: switch every pronunciation's enabled state to that group (missing commands stay unchanged);
    /// the group then becomes the active one
    fn apply_group(&mut self, name: &str) {
        let Some(map) = self.groups.get(name).cloned() else {
            return;
        };
        let mut changed = 0;
        if let Some(cmds) = self.cfg.get_mut("commands").and_then(|v| v.as_object_mut()) {
            for (cname, c) in cmds.iter_mut() {
                if let Some(on) = map.get(cname) {
                    if c.get("enabled").and_then(|v| v.as_bool()).unwrap_or(true) != *on {
                        c["enabled"] = serde_json::Value::Bool(*on);
                        changed += 1;
                    }
                }
            }
        }
        self.group_sel = name.to_string();
        self.set_active_group(name);
        if changed > 0 {
            self.persist_cfg();
            self.status = crate::tfmt!(k::MSG_GROUP_APPLIED, name, changed);
        } else {
            self.status = crate::tfmt!(k::MSG_GROUP_ALREADY_ACTIVE, name);
        }
        self.log(&crate::tfmt!(
            "applied group {name} ({changed} changes)",
            name,
            changed
        ));
    }

    fn set_active_group(&mut self, name: &str) {
        self.active_group = Some(name.to_string());
        if let Err(e) = save_groups(&self.groups, self.active_group.as_deref()) {
            self.status = crate::tfmt!(k::ERR_GROUP_SAVE, e);
        }
    }

    /// Overwrite the selected group with the current checkboxes (that group becomes the active one)
    fn overwrite_group(&mut self, name: &str) {
        if !self.groups.contains_key(name) {
            self.status = crate::tfmt!(k::MSG_GROUP_MISSING, name);
            return;
        }
        let map = self.current_enabled_map();
        self.groups.insert(name.to_string(), map);
        self.group_sel = name.to_string();
        self.set_active_group(name); // Persists internally (including error state)
        self.status = crate::tfmt!(k::MSG_GROUP_OVERWRITTEN, name);
        self.log(&crate::tfmt!("overwrote group {name}", name));
    }

    /// Save the current checkboxes as a new group (the new group becomes the active one)
    fn save_new_group(&mut self, name: &str) {
        let name = name.trim();
        if name.is_empty() {
            self.status = t(k::ERR_GROUP_NAME_REQUIRED);
            return;
        }
        let map = self.current_enabled_map();
        self.groups.insert(name.to_string(), map);
        self.group_sel = name.to_string();
        self.new_group_name.clear();
        self.set_active_group(name); // Persists internally (including error state)
        self.status = crate::tfmt!(k::MSG_GROUP_SAVED, self.groups[name].len());
        self.log(&crate::tfmt!("created group {name}", name));
    }

    /// Delete a group (after confirmation). Deleting the active group only unbinds it (active = empty) and leaves the checkboxes alone.
    fn delete_group(&mut self, name: &str) {
        if let Some(map) = self.groups.remove(name) {
            if map.is_empty() {
                self.status = crate::tfmt!(k::MSG_GROUP_DELETED_EMPTY, name);
            } else {
                self.status = crate::tfmt!(k::MSG_GROUP_DELETED, name);
            }
        }
        if self.active_group.as_deref() == Some(name) {
            self.active_group = None; // Only unbinds the group; the current checkboxes stay untouched
        }
        if self.group_sel == name {
            self.group_sel = self.groups.keys().next().cloned().unwrap_or_default();
        }
        if let Err(e) = save_groups(&self.groups, self.active_group.as_deref()) {
            self.status = crate::tfmt!(k::ERR_GROUP_SAVE, e);
        }
        self.log(&crate::tfmt!("deleted group {name}", name));
    }

    /// Rename a group (after confirmation): keeps the mapping, the active group and the selected name in sync
    fn rename_group(&mut self, old: &str, new: &str) {
        let new = new.trim();
        if new.is_empty() {
            self.status = t(k::ERR_GROUP_NAME_EMPTY);
            return;
        }
        if self.groups.contains_key(new) {
            self.status = crate::tfmt!(k::MSG_GROUP_EXISTS, new);
            return;
        }
        if let Some(map) = self.groups.remove(old) {
            self.groups.insert(new.to_string(), map);
        }
        if self.active_group.as_deref() == Some(old) {
            self.active_group = Some(new.to_string());
        }
        if self.group_sel == old {
            self.group_sel = new.to_string();
        }
        if let Err(e) = save_groups(&self.groups, self.active_group.as_deref()) {
            self.status = crate::tfmt!(k::ERR_GROUP_SAVE, e);
            return;
        }
        self.status = crate::tfmt!(k::MSG_GROUP_RENAMED, old, new);
        self.log(&crate::tfmt!(k::MSG_GROUP_RENAME_LOG, old, new));
    }

    fn delete_command(&mut self, name: &str) {
        if self.rec_mic.is_some() && self.pron_open.as_deref() == Some(name) {
            self.rec_mic = None;
            self.rec_start_at = None;
            self.log(&format!("deleted pronunciation {name}: the extra take was cancelled"));
        }
        if let Some(c) = &self.ctx {
            let mut p = c.proc.lock().unwrap_or_else(|e| e.into_inner());
            p.release_toggle_cmd(name);
            p.cancel_macro(name);
        }
        if let Some(cmds) = self.cfg.get_mut("commands").and_then(|v| v.as_object_mut()) {
            cmds.remove(name);
        }
        drop_mutex_refs(&mut self.cfg, name);
        if self.pron_open.as_deref() == Some(name) {
            self.pron_open = None;
            self.rec_list.clear();
        }
        // If the standalone macro editor is editing the deleted command, close it too (draw_macro_editor also self-guards
        // on a missing cfg entry; this explicit cleanup prevents leftovers while the window is open)
        if self.macro_editor_open.as_deref() == Some(name) {
            self.macro_editor_open = None;
            self.macro_edit_for = None;
        }
        // The mutex panel closes the same way (draw_mutex_editor also self-guards)
        if self.mutex_editor_open.as_deref() == Some(name) {
            self.mutex_editor_open = None;
            self.mutex_edit_for = None;
        }
        if self.settings.voice_toggle_cmd == name {
            self.settings.voice_toggle_cmd.clear();
            self.persist_settings();
        }
        if let Err(e) = crate::recordings::remove_cmd_all(&self.model_dir, name) {
            self.status = crate::tfmt!(k::MSG_CMD_DELETED_REC_FAILED, name, e);
            self.log(&crate::tfmt!("failed to delete the recording of pronunciation {name}: {e}", name, e));
        }
        if let Err(e) = crate::udkws_v2::remove_embed_proto_entry(&self.model_dir, name) {
            eprintln!("[ui] failed to remove the template entry of pronunciation {name}: {e}");
            self.log(&crate::tfmt!(
                "failed to delete the template entry of pronunciation {name}: {e}",
                name,
                e
            ));
        }
        self.persist_cfg();
        self.refresh_rec_counts();
        self.refresh_proto_cover();
        self.status = crate::tfmt!(k::MSG_CMD_DELETED, name);
        self.log(&crate::tfmt!("deleted pronunciation {name}", name));
    }

    fn start_new_command(&mut self) {
        let id = next_command_id(&self.cfg);
        self.cfg["commands"][id.as_str()] = serde_json::json!({
            "enabled": true,
            "key": "",
            "mode": "tap",
            "phrase": "",
        });
        self.persist_cfg();
        self.add_phrase.clear();
        self.key_edit.clear();
        self.refresh_rec_counts();
        self.refresh_rec_list(&id);
        self.refresh_proto_cover();
        self.pron_open = Some(id);
        self.pron_new = true;
        self.pron_new_focus = true;
        self.status = t(k::MSG_CMD_ADDED);
    }

    /// Default name when none is given: "New phrase N" (the format comes from the language-table key for new phrases).
    /// N starts at 1 and takes the **first free** number (compared against every non-empty phrase), so no existing name is reused.
    fn default_phrase_name(&self) -> String {
        let used: std::collections::BTreeSet<&str> = self
            .cfg
            .get("commands")
            .and_then(|v| v.as_object())
            .map(|o| {
                o.values()
                    .filter_map(|c| c.get("phrase").and_then(|v| v.as_str()))
                    .filter(|s| !s.is_empty())
                    .collect()
            })
            .unwrap_or_default();
        for n in 1usize.. {
            let cand = crate::lang::tf(k::MSG_DEFAULT_CMD_NAME, &[("{n}", &n.to_string())]);
            if !used.contains(cand.as_str()) {
                return cand;
            }
        }
        unreachable!("default name numbering increases without bound, so a free slot always exists")
    }

    fn finalize_new_command(&mut self, cmd: &str) {
        let typed = self.add_phrase.trim().to_string();
        if !typed.is_empty() {
            self.set_cmd_field(cmd, "phrase", serde_json::Value::String(typed));
            self.persist_cfg();
        }
        // 2) The "nothing was entered" criterion: empty name + empty key + trigger mode still the default tap + no recordings.
        // (`enabled` can only be changed in the list, not in this dialog, so it does not participate.)
        let phrase = self.cmd_field_str(cmd, "phrase");
        let key = self.cmd_field_str(cmd, "key");
        let mode = self.cmd_field_str(cmd, "mode");
        let recs = self.rec_counts.get(cmd).copied().unwrap_or(0);
        if phrase.is_empty() && key.is_empty() && (mode.is_empty() || mode == "tap") && recs == 0 {
            self.delete_command(cmd);
            self.status = t(k::MSG_CMD_DISCARDED);
            return;
        }
        // 3) Content present but no name -> assign the default name
        if phrase.is_empty() {
            let name = self.default_phrase_name();
            self.set_cmd_field(cmd, "phrase", serde_json::Value::String(name.clone()));
            self.persist_cfg();
            self.status = crate::lang::tf(k::MSG_CMD_AUTO_NAMED, &[("{name}", &name)]);
            let line = self.status.clone();
            self.log(&line);
        }
    }

    /// Read a string field of a command (missing or non-string -> empty string).
    fn cmd_field_str(&self, cmd: &str, field: &str) -> String {
        self.cfg
            .get("commands")
            .and_then(|v| v.get(cmd))
            .and_then(|c| c.get(field))
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string()
    }

    // -- Add takes --

    fn start_rec(&mut self, cmd: &str) {
        if self.rec_mic.is_some() {
            self.status = t(k::MSG_ALREADY_RECORDING);
            return;
        }
        match crate::mic_monitor::MicMonitor::start(if self.selected_device.is_empty() {
            None
        } else {
            Some(&self.selected_device)
        }) {
            Ok(mic) => {
                self.rec_start_len = mic.len();
                self.rec_start_at = Some(std::time::Instant::now());
                self.rec_mic = Some(mic);
                self.status = crate::tfmt!(k::MSG_RECORDING_IN_PROGRESS, cmd);
                self.log(&crate::tfmt!(
                    "recording an extra take for {cmd} (capture running: {})",
                    self.ctx.is_some()
                ));
            }
            Err(e) => self.status = crate::tfmt!(k::ERR_REC_START, e),
        }
    }

    fn stop_rec(&mut self) {
        let Some(mic) = self.rec_mic.take() else {
            return;
        };
        let Some(cmd) = self.pron_open.clone() else {
            return;
        };
        self.rec_start_at = None;
        let samples = mic.snapshot(self.rec_start_len);
        drop(mic);
        if samples.len() < 1600 {
            self.status = t(k::ERR_REC_TOO_SHORT);
            return;
        }
        let rec_dir = crate::recordings::record_dir(&self.model_dir);
        let _ = std::fs::create_dir_all(&rec_dir);
        let ts = chrono::Local::now().format("%Y%m%d_%H%M%S");
        let file = format!("{cmd}_act_{ts}.wav");
        let path = rec_dir.join(&file);
        if let Err(e) = crate::features::write_wav(
            path.to_str().unwrap_or(""),
            &samples,
            crate::features::TARGET_SAMPLE_RATE,
        ) {
            self.status = crate::tfmt!(k::ERR_REC_WRITE, e);
            return;
        }
        let rec = crate::recordings::CmdRecording {
            cmd: cmd.clone(),
            file: file.clone(),
            kind: crate::recordings::RecKind::Active,
            ts: ts.to_string(),
            dur_s: samples.len() as f32 / crate::features::TARGET_SAMPLE_RATE as f32,
        };
        if let Err(e) = crate::recordings::append(&self.model_dir, rec) {
            self.status = crate::tfmt!(k::ERR_REC_INDEX_WRITE, e);
            return;
        }
        self.refresh_rec_counts();
        self.refresh_rec_list(&cmd);
        let sec = format!(
            "{:.1}",
            samples.len() as f32 / crate::features::TARGET_SAMPLE_RATE as f32
        );
        self.status = crate::tfmt!(k::MSG_REC_DONE, file, sec);
        self.log(&crate::tfmt!("extra take recorded for {cmd}", cmd)); // Redacted: no local paths in the log (the location is fixed)
    }

    fn play_rec(&mut self, file: &str) {
        let orig = crate::recordings::record_dir(&self.model_dir).join(file);
        let Some(path) = crate::recordings::decrypt_to_temp(&self.model_dir, file) else {
            self.log(&crate::tfmt!(k::ERR_PLAY_MISSING, file));
            return;
        };
        let is_tmp = path != orig;
        let cleanup = |p: &std::path::Path| {
            if is_tmp {
                let _ = std::fs::remove_file(p);
            }
        };
        let Some(full) = path.to_str() else {
            cleanup(&path);
            return;
        };
        let full_win = full.replace('/', "\\");
        match std::process::Command::new("powershell")
            .args([
                "-NoProfile",
                "-Command",
                &format!("(New-Object Media.SoundPlayer '{full_win}').PlaySync()"),
            ])
            .spawn()
        {
            Ok(mut child) if is_tmp => {
                std::thread::spawn(move || {
                    let _ = child.wait();
                    let _ = std::fs::remove_file(&path);
                });
            }
            Ok(_) => {}
            Err(e) => {
                cleanup(&path);
                self.log(&crate::tfmt!(k::ERR_PLAY_POWERSHELL, e));
            }
        }
    }

    // -- Key toggle (pause/resume triggering) --

    fn stop_hotkey_thread(&mut self) {
        if let Some(stop) = self.hotkey_stop.take() {
            stop.store(true, std::sync::atomic::Ordering::Relaxed);
        }
        if let Some(h) = self.hotkey_thread.take() {
            let _ = h.join(); // <=30 ms (one poll period), ensuring the old thread exits before re-registering the hotkey
        }
    }

    fn apply_hotkey(&mut self) {
        self.stop_hotkey_thread();
        self.hotkey_manager = None;
        if self.settings.hotkey.is_empty() {
            self.status = t(k::MSG_KEY_TOGGLE_UNSET);
            return;
        }
        let Some((mods, code)) = parse_hotkey(&self.settings.hotkey) else {
            self.status = t(k::ERR_HOTKEY_FORMAT);
            return;
        };
        let manager = match global_hotkey::GlobalHotKeyManager::new() {
            Ok(m) => m,
            Err(e) => {
                self.status = crate::tfmt!(k::ERR_HOTKEY_INIT, e);
                return;
            }
        };
        let hotkey = global_hotkey::hotkey::HotKey::new(Some(mods), code);
        if let Err(e) = manager.register(hotkey) {
            self.status = crate::tfmt!(k::ERR_HOTKEY_REGISTER, e);
            return;
        }
        let hotkey_id = hotkey.id();
        let hk_mode = self.settings.hotkey_mode;
        let flag = self.enabled_flag.clone();
        let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let stop2 = stop.clone();
        self.hotkey_thread = Some(std::thread::spawn(move || {
            while !stop2.load(std::sync::atomic::Ordering::Relaxed) {
                if let Ok(ev) = global_hotkey::GlobalHotKeyEvent::receiver().try_recv() {
                    if ev.id() == hotkey_id {
                        match ev.state() {
                            global_hotkey::HotKeyState::Pressed => {
                                // The key toggle equals switching recognition on and off (emitting triggers is the job of the "voice toggle";
                                // the hotkey only flips enabled_flag; key release and status hints are applied by the UI frame through
                                // proc.set_enabled, and gate.rs releases toggle-held keys while paused).
                                // 0 = key toggle (one press flips); 1 = hold to pause (hold = pause, releasing
                                // on Released = resume); 2 = hold to input (press = enable, release = pause).
                                match hk_mode {
                                    1 => flag.store(false, std::sync::atomic::Ordering::Relaxed),
                                    2 => flag.store(true, std::sync::atomic::Ordering::Relaxed),
                                    _ => {
                                        let cur = flag.load(std::sync::atomic::Ordering::Relaxed);
                                        flag.store(!cur, std::sync::atomic::Ordering::Relaxed);
                                    }
                                }
                            }
                            // global-hotkey 0.6 on Windows polls GetAsyncKeyState after WM_HOTKEY to synthesise
                            // the Released event (crate source platform_impl/windows/mod.rs).
                            global_hotkey::HotKeyState::Released => {
                                if hk_mode == 1 {
                                    flag.store(true, std::sync::atomic::Ordering::Relaxed);
                                } else if hk_mode == 2 {
                                    flag.store(false, std::sync::atomic::Ordering::Relaxed);
                                }
                            }
                        }
                    }
                }
                std::thread::sleep(std::time::Duration::from_millis(30));
            }
        }));
        self.hotkey_stop = Some(stop);
        self.hotkey_manager = Some(manager);
    }

    /// Key capture (mapped key / key toggle)
    fn handle_key_capture(&mut self, ctx: &egui::Context) {
        let capturing = self
            .capture_key_for
            .clone()
            .or_else(|| self.capture_hotkey.then(|| "__hotkey__".to_string()));
        let Some(target) = capturing else { return };
        let key_event = ctx.input(|i| {
            i.events.iter().find_map(|e| match e {
                egui::Event::Key {
                    key,
                    pressed: true,
                    modifiers,
                    ..
                } => Some((*key, *modifiers)),
                _ => None,
            })
        });
        let Some((key, mods)) = key_event else { return };
        if key == egui::Key::Escape {
            self.capture_key_for = None;
            self.capture_hotkey = false;
            self.status = t(k::MSG_KEY_CAPTURE_CANCELLED);
            return;
        }
        let Some(main) = egui_key_name(key) else {
            self.status = t(k::MSG_KEY_UNSUPPORTED);
            return;
        };
        // Recognise modifier keys during capture (CTRL/SHIFT/ALT are folded into the chord, e.g. holding CTRL and pressing F1
        // gives "CTRL+F1"); modifiers are no longer mandatory either -- the key toggle accepts single keys (the WIN key is
        // swallowed by Windows itself and is invisible to egui capture).
        let mut parts: Vec<&str> = Vec::new();
        if mods.ctrl {
            parts.push("CTRL");
        }
        if mods.shift {
            parts.push("SHIFT");
        }
        if mods.alt {
            parts.push("ALT");
        }
        parts.push(main);
        if target == "__hotkey__" {
            let key_str = parts.join("+");
            self.settings.hotkey = key_str.clone();
            self.capture_hotkey = false;
            self.apply_hotkey();
            self.persist_settings();
            self.status = crate::tfmt!(k::MSG_KEY_TOGGLE_SET, key_str);
        } else if let Some(idx) = target
            .strip_prefix("__macro_")
            .and_then(|s| s.strip_suffix("__"))
            .and_then(|s| s.parse::<usize>().ok())
        {
            let key_str = parts.join("+");
            if let Some(row) = self.macro_edit.get_mut(idx) {
                row.param = key_str.clone();
            }
            self.capture_key_for = None;
            self.status = crate::tfmt!(k::MSG_MACRO_KEY, key_str);
        } else {
            let key_str = parts.join("+");
            let cmd = target.clone();
            self.set_cmd_field(&cmd, "key", serde_json::Value::String(key_str.clone()));
            self.key_edit = key_str.clone();
            self.capture_key_for = None;
            self.persist_cfg();
            self.status = format!("{cmd} → {key_str}");
        }
    }

    // -- Settings persistence (merged write-back, preserving other settings.json keys such as ai_*) --

    /// Effective immediately while running: write the settings into the running engine (no rebuild, no restart).
    /// When recognition is not running this does nothing -- the next start carries them through EngineOverrides.
    fn apply_live_tuning(&self, ov: crate::udkws_v2::EngineOverrides) {
        if let Some(c) = self.ctx.as_ref() {
            let mut p = c.proc.lock().unwrap_or_else(|e| e.into_inner());
            p.apply_live_tuning(&ov);
        }
    }

    fn persist_settings(&mut self) {
        let v = &mut self.settings_raw;
        if !v.is_object() {
            *v = serde_json::json!({});
        }
        let o = v.as_object_mut().unwrap();
        let s = &self.settings;
        o.insert("hotkey".into(), serde_json::json!(s.hotkey));
        o.insert(
            "voice_toggle_cmd".into(),
            serde_json::json!(s.voice_toggle_cmd),
        );
        o.insert("input_mode".into(), serde_json::json!(s.input_mode));
        o.insert("target_window".into(), serde_json::json!(s.target_window));
        o.insert("device".into(), serde_json::json!(s.device));
        o.insert("record_enabled".into(), serde_json::json!(s.record_enabled));
        o.insert(
            "auto_start_recognize".into(),
            serde_json::json!(s.auto_start_recognize),
        );
        o.insert("v2_min_conf".into(), serde_json::json!(s.v2_min_conf));
        o.insert("act_gate_db".into(), serde_json::json!(s.act_gate_db));
        o.insert("refr_s".into(), serde_json::json!(s.refr_s));
        o.insert("act_gate_auto".into(), serde_json::json!(s.act_gate_auto));
        o.insert(
            "frame_mean_norm".into(),
            serde_json::json!(s.frame_mean_norm),
        );
        o.insert("aec_mode".into(), serde_json::json!(s.aec_mode));
        o.insert("vad_gate".into(), serde_json::json!(s.vad_gate));
        o.insert("sv_gate".into(), serde_json::json!(s.sv_gate));
        o.insert("ident_fwd_sec".into(), serde_json::json!(s.ident_fwd_sec));
        o.insert("hotkey_mode".into(), serde_json::json!(s.hotkey_mode));
        o.insert(
            "update_check_enabled".into(),
            serde_json::json!(s.update_check_enabled),
        );
        o.insert("language".into(), serde_json::json!(s.language));
        o.remove("listen_passcode_when_off");
        o.remove("ident_fast");
        if let Err(e) = serde_json::to_string_pretty(v)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))
            .and_then(|s| write_json_atomic(&settings_path(), &s))
        {
            self.status = crate::tfmt!(k::ERR_SETTINGS_WRITE, e);
            return;
        }
        // The warm instance was built with the old settings: any settings change voids it (including an in-flight build), and the next start rebuilds with the new parameters
        self.warming = false;
        self.warm_rx = None;
        self.warm_proc = None;
        self.warm_fingerprint = 0;
        self.status = t(k::MSG_SETTINGS_SAVED);
    }
}

impl ViqApp {
    fn window_frame_housekeeping(&mut self, ctx: &egui::Context) {
        // -- Title: compared against the language table and sent when different (so hot language switching needs no hooks at each entry point) --
        let want = window_title();
        if self.win_title != want {
            self.win_title = want.clone();
            ctx.send_viewport_cmd(egui::ViewportCommand::Title(want));
        }
        let vp = ctx.input(|i| i.viewport().clone());
        // -- Cache the geometry every frame so it can be persisted on exit --
        // Maximised and fullscreen states are not recorded: that is a different state, and recording it would pollute the "restored size" (common practice)
        if vp.maximized != Some(true) && vp.fullscreen != Some(true) {
            if let (Some(outer), Some(inner)) = (vp.outer_rect, vp.inner_rect) {
                self.win_geom_live = Some(crate::window_geom::SavedGeometry {
                    x: outer.min.x,
                    y: outer.min.y,
                    w: inner.width(),
                    h: inner.height(),
                    ppp: ctx.pixels_per_point(),
                });
            }
        }
        // -- First-frame geometry self-check (monitor_size may be unavailable on the first frame; if so, retry next frame) --
        if self.win_geom_checked {
            return;
        }
        let Some(mon) = vp.monitor_size else { return };
        self.win_geom_checked = true;
        let ppp = ctx.pixels_per_point();
        let ppp_changed = matches!(self.win_geom_saved_ppp, Some(p) if (p - ppp).abs() > 0.01);
        // egui does not expose the taskbar height, so reserve 48pt of usable height (slightly tight beats off-screen)
        let area = crate::window_geom::WorkArea {
            left: 0.0,
            top: 0.0,
            w: mon.x,
            h: (mon.y - 48.0).max(320.0),
            ppp,
        };
        let oversized = vp.inner_rect.map_or(false, |r| {
            r.width() > mon.x + 0.5 || r.height() > area.h + 0.5
        });
        if !ppp_changed && !oversized {
            return;
        }
        let (w, h) = crate::window_geom::default_size(&area);
        let min = crate::window_geom::min_size((w, h));
        ctx.send_viewport_cmd(egui::ViewportCommand::MinInnerSize(egui::vec2(
            min.0, min.1,
        )));
        ctx.send_viewport_cmd(egui::ViewportCommand::InnerSize(egui::vec2(w, h)));
        // Centre at the same time, otherwise the result is "correct size but still off-screen" (e.g. x stays at 3000 after an external monitor is unplugged)
        ctx.send_viewport_cmd(egui::ViewportCommand::OuterPosition(egui::pos2(
            (mon.x - w) / 2.0,
            ((mon.y - h) / 2.0).max(0.0),
        )));
        if let Some(o) = self.settings_raw.as_object_mut() {
            o.remove("window_rect");
        }
        self.log(if ppp_changed {
            "scale changed: window position and size were reset"
        } else {
            "the saved window size exceeded the screen: reset automatically"
        });
    }

    /// Persist the window geometry on exit (on_exit receives no egui Context, hence the per-frame cache in update()).
    /// A write failure only goes to stderr and does not disturb the exit.
    fn persist_window_geometry(&mut self) {
        let Some(g) = self.win_geom_live else { return };
        if let Some(o) = self.settings_raw.as_object_mut() {
            o.insert("window_rect".into(), g.to_json());
        }
        if let Err(e) = serde_json::to_string_pretty(&self.settings_raw)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))
            .and_then(|s| write_json_atomic(&settings_path(), &s))
        {
            eprintln!("failed to persist the window geometry (exit unaffected): {e}");
        }
    }

    fn auto_update_frame(&mut self) {
        while let Ok(outcome) = self.update_check_rx.try_recv() {
            let manual = std::mem::take(&mut self.update_check_inflight);
            match outcome {
                Ok(Some(ver)) => {
                    self.latest_update = Some(ver.clone());
                    self.status = crate::tfmt!(k::MSG_UPDATE_AVAILABLE, ver);
                    let line = self.status.clone();
                    self.log(&line);
                }
                Ok(None) => {
                    if manual {
                        self.status = t(k::MSG_UPDATE_LATEST);
                    }
                }
                Err(_) => {
                    if manual {
                        self.status = t(k::ERR_UPDATE_CHECK);
                    }
                }
            }
        }
        // 2) Startup automatic check (on by default): only runs when the setting is enabled.
        if self.auto_update_launched {
            return;
        }
        if !self.settings.update_check_enabled {
            self.auto_update_launched = true; // Marked even when the conditions are unmet, so it is not re-evaluated every frame
            return;
        }
        match self.auto_update_at {
            None => self.auto_update_at = Some(std::time::Instant::now()),
            Some(t) if t.elapsed() >= std::time::Duration::from_secs(3) => {
                self.auto_update_launched = true;
                spawn_update_check(&self.update_check_tx);
            }
            Some(_) => {}
        }
    }
}

impl eframe::App for ViqApp {
    fn on_exit(&mut self, _gl: Option<&eframe::glow::Context>) {
        // On exit: stop the hotkey thread + release every held key + save the recording + stop the mic monitor
        self.stop_hotkey_thread();
        if self.rec_mic.take().is_some() {
            self.log(&t(k::MSG_REC_ABORTED_ON_EXIT));
        }
        if let Some(c) = &self.ctx {
            let (count, last) = {
                let mut p = c.proc.lock().unwrap_or_else(|e| e.into_inner());
                p.release_all_toggles();
                p.cancel_all_macros();
                (p.keypress_count(), p.last_command())
            };
            c.rec.save(count, &last);
        }
        // Exiting mid-hot-apply: save the held-back session recording
        if let Some((r, count, last)) = self.pending_rec.take() {
            r.save(count, &last);
        }
        self.persist_window_geometry();
    }

    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.window_frame_housekeeping(ctx);
        if crate::single_instance::poll_show_request() {
            ctx.send_viewport_cmd(egui::ViewportCommand::Minimized(false)); // Restore a minimised window first
            ctx.send_viewport_cmd(egui::ViewportCommand::Visible(true));
            ctx.send_viewport_cmd(egui::ViewportCommand::Focus); // Foreground rights were already granted by the second instance via ASFW_ANY
            self.log(&t(k::MSG_SECOND_LAUNCH));
        }
        self.auto_update_frame();
        // "Start recognition automatically at launch" (settings-page general group, off by default): fires once on the first frame,
        // along exactly the same path as the home page's "Start recognition" (loads the engine in the background, never freezing the UI)
        if self.auto_start_pending && !self.proto_busy() && !self.warming {
            self.auto_start_pending = false;
            if self.ctx.is_none() && self.capture_job == CaptureJob::None {
                self.log(&t(k::MSG_AUTO_STARTED));
                self.toggle_capture();
            }
        }
        self.handle_key_capture(ctx);
        self.poll_warmup();
        self.poll_proto_job();
        self.poll_capture_job();
        if !self.settings.voice_toggle_cmd.is_empty() {
            if let Some(c) = &self.ctx {
                let (count, cmd) = {
                    let p = c.proc.lock().unwrap_or_else(|e| e.into_inner());
                    (p.keypress_count(), p.last_command())
                };
                self.voice_toggle_dispatch(count, &cmd);
            }
        }
        // Double-click delete arms for 5 s then disarms automatically (a first click outside that window counts as a misclick)
        if let Some((_, t)) = &self.armed_delete {
            if t.elapsed() >= std::time::Duration::from_secs(5) {
                self.armed_delete = None;
            }
        }
        let enabled_now = self.enabled_flag.load(std::sync::atomic::Ordering::Relaxed);
        if enabled_now != self.last_enabled_seen {
            self.last_enabled_seen = enabled_now;
            if let Some(c) = &self.ctx {
                c.proc
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .set_enabled(enabled_now);
            }
            self.status = if enabled_now {
                t(k::MSG_KEYS_RESUMED)
            } else {
                t(k::MSG_KEYS_PAUSED)
            };
            let line = self.status.clone();
            self.log(&line);
        }
        self.update_logs();
        self.draw_modals(ctx);
        self.draw_edit_popup(ctx);
        self.draw_macro_editor(ctx);
        self.draw_mutex_editor(ctx);

        egui::CentralPanel::default().show(ctx, |ui| {
            self.tab_bar(ui);
            match self.tab {
                Tab::Home => self.tab_home(ui),
                Tab::Pron => self.tab_pron(ui),
                Tab::Settings => self.tab_settings(ui),
            }
        });

        // Refresh the encryption state while the engine loads (stream creation on the UI thread is snappier once ready); 100 ms is enough normally
        let refresh = if self.capture_job != CaptureJob::None {
            50
        } else {
            100
        };
        ctx.request_repaint_after(std::time::Duration::from_millis(refresh));
    }
}

// -- Utilities --

/// Launch the Windows on-screen keyboard osk.exe (System32, resolvable via the default search path).
/// Returns the child handle (probes assert the process exists after spawn; the UI drops it, since Windows does not kill
/// osk when the parent exits). If the direct spawn fails (path, permissions), fall back to `cmd /C start`,
/// with CREATE_NO_WINDOW to avoid a flashing console.
fn spawn_osk() -> std::io::Result<std::process::Child> {
    use std::os::windows::process::CommandExt;
    const CREATE_NO_WINDOW: u32 = 0x0800_0000;
    match std::process::Command::new("osk.exe").spawn() {
        Ok(child) => Ok(child),
        Err(_) => std::process::Command::new("cmd")
            .args(["/C", "start", "", "osk.exe"])
            .creation_flags(CREATE_NO_WINDOW)
            .spawn(),
    }
}

fn write_json_atomic(path: &std::path::Path, content: &str) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            let _ = std::fs::create_dir_all(parent);
        }
    }
    let tmp = path.with_extension("tmp");
    std::fs::write(&tmp, content)?;
    match std::fs::rename(&tmp, path) {
        Ok(()) => Ok(()),
        Err(e) => {
            let _ = std::fs::remove_file(&tmp);
            Err(e)
        }
    }
}

fn spawn_update_check(tx: &std::sync::mpsc::Sender<anyhow::Result<Option<String>>>) {
    let tx = tx.clone();
    std::thread::spawn(move || {
        let out = crate::update::check_for_update(env!("CARGO_PKG_VERSION"));
        let _ = tx.send(out);
    });
}

/// Read a command's mutex list (missing field / non-array / non-string entries are ignored accordingly).
fn mutex_list_of(cmd_entry: &serde_json::Value) -> Vec<String> {
    cmd_entry
        .get("mutex")
        .and_then(|v| v.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|x| x.as_str())
                .map(|s| s.to_string())
                .collect()
        })
        .unwrap_or_default()
}

/// Write a command's mutex list (an empty list removes the field, keeping commands.json free of empty-array noise).
fn set_mutex_list_of(cfg: &mut serde_json::Value, id: &str, list: Vec<String>) {
    let Some(cmds) = cfg.get_mut("commands").and_then(|v| v.as_object_mut()) else {
        return;
    };
    let Some(entry) = cmds.get_mut(id) else {
        return;
    };
    if list.is_empty() {
        if let Some(obj) = entry.as_object_mut() {
            obj.remove("mutex");
        }
        return;
    }
    let arr: Vec<serde_json::Value> = list.into_iter().map(serde_json::Value::String).collect();
    entry["mutex"] = serde_json::Value::Array(arr);
}

fn set_mutex_pair(cfg: &mut serde_json::Value, a: &str, b: &str, on: bool) {
    for (x, y) in [(a, b), (b, a)] {
        let mut cur = cfg
            .get("commands")
            .and_then(|v| v.get(x))
            .map(mutex_list_of)
            .unwrap_or_default();
        cur.retain(|p| p != y);
        if on {
            cur.push(y.to_string());
        }
        cur.sort();
        cur.dedup();
        set_mutex_list_of(cfg, x, cur);
    }
}

/// Remove every mutex reference to `id` from all commands (called when a command is deleted, so no entry keeps pointing at a ghost id).
fn drop_mutex_refs(cfg: &mut serde_json::Value, id: &str) {
    let others: Vec<String> = cfg
        .get("commands")
        .and_then(|v| v.as_object())
        .map(|m| m.keys().cloned().collect())
        .unwrap_or_default();
    for other in others {
        if other == id {
            continue;
        }
        let mut cur = cfg
            .get("commands")
            .and_then(|v| v.get(&other))
            .map(mutex_list_of)
            .unwrap_or_default();
        let before = cur.len();
        cur.retain(|p| p != id);
        if cur.len() != before {
            set_mutex_list_of(cfg, &other, cur);
        }
    }
}

/// Generate a non-conflicting new command id: an incrementing `cmdN` (cmd1, cmd2, ... skipping used numbers).
/// Rationale: existing ids in commands.json are hand-written pinyin names (chong/gong/shan ...), and Chinese-to-pinyin would
/// need a transliteration table (not feasible without adding a crate), so new commands use a short incrementing prefix that does not
/// clash with the existing style. New ids avoid '_' for consistency; existing ids may contain '_' and are grouped by longest
/// matching prefix (classify_recording).
fn next_command_id(cfg: &serde_json::Value) -> String {
    let used: std::collections::BTreeSet<&str> = cfg
        .get("commands")
        .and_then(|v| v.as_object())
        .map(|o| o.keys().map(|k| k.as_str()).collect())
        .unwrap_or_default();
    let mut n = 1usize;
    while used.contains(format!("cmd{n}").as_str()) {
        n += 1;
    }
    format!("cmd{n}")
}

fn backup_corrupt_json(path: &std::path::Path, text: &str) {
    let ts = chrono::Local::now().format("%Y%m%d_%H%M%S");
    let bak = PathBuf::from(format!("{}.bak.{}", path.display(), ts));
    match std::fs::write(&bak, text) {
        Ok(()) => eprintln!(
            "warning: failed to parse {}; the corrupt file was backed up to {}",
            path.display(),
            bak.display()
        ),
        Err(e) => eprintln!("warning: failed to parse {} and the backup failed too ({e})", path.display()),
    }
}

fn load_cfg(model_dir: &str) -> (serde_json::Value, PathBuf) {
    let path = std::path::PathBuf::from(model_dir).join("commands.json");
    match std::fs::read_to_string(&path) {
        Ok(t) => match serde_json::from_str(&t) {
            Ok(v) => (v, path),
            Err(e) => {
                backup_corrupt_json(&path, &t);
                eprintln!(
                    "  ({} failed to parse: {e}; started with an empty configuration -- reconfigure or restore from the .bak)",
                    path.display()
                );
                (serde_json::json!({}), path)
            }
        },
        // Missing or unreadable: fresh defaults (nothing to back up)
        Err(_) => (serde_json::json!({}), path),
    }
}

/// Load pronunciation groups (`<exe dir>/config/groups.json`).
/// v2 format: {"active": "active group name or null", "groups": {group: {command: enabled}}};
/// the old flat format ({group: {command: enabled}}, no active) is accepted -> active = None.
fn load_groups() -> (BTreeMap<String, BTreeMap<String, bool>>, Option<String>) {
    let v: serde_json::Value = std::fs::read_to_string(groups_path())
        .ok()
        .and_then(|t| serde_json::from_str(&t).ok())
        .unwrap_or(serde_json::Value::Null);
    if let Some(obj) = v.as_object() {
        if let Some(groups_v) = obj.get("groups") {
            let groups: BTreeMap<String, BTreeMap<String, bool>> =
                serde_json::from_value(groups_v.clone()).unwrap_or_default();
            let active = obj.get("active").and_then(|x| x.as_str()).map(String::from);
            return (groups, active);
        }
    }
    (BTreeMap::new(), None)
}

fn save_groups(
    groups: &BTreeMap<String, BTreeMap<String, bool>>,
    active: Option<&str>,
) -> std::io::Result<()> {
    let v = serde_json::json!({
        "active": active,
        "groups": groups,
    });
    let s = serde_json::to_string_pretty(&v)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
    write_json_atomic(&groups_path(), &s)
}

fn parse_hotkey(
    spec: &str,
) -> Option<(
    global_hotkey::hotkey::Modifiers,
    global_hotkey::hotkey::Code,
)> {
    use global_hotkey::hotkey::{Code, Modifiers};
    let mut mods = Modifiers::empty();
    let mut main = String::new();
    for part in spec.split('+') {
        let part = part.trim().to_uppercase();
        match part.as_str() {
            "CTRL" => mods |= Modifiers::CONTROL,
            "SHIFT" => mods |= Modifiers::SHIFT,
            "ALT" => mods |= Modifiers::ALT,
            "" => {}
            p => main = p.to_string(),
        }
    }
    if main.is_empty() {
        return None;
    }
    let code = match main.as_str() {
        "SPACE" => Code::Space,
        "ENTER" => Code::Enter,
        "TAB" => Code::Tab,
        // Short names match the capture side (egui_key_name); long-name aliases remain valid for hand-typed and legacy configs
        "ESC" | "ESCAPE" => Code::Escape,
        "BACKSPACE" => Code::Backspace,
        "DELETE" => Code::Delete,
        "INSERT" => Code::Insert,
        "HOME" => Code::Home,
        "END" => Code::End,
        "PAGEUP" => Code::PageUp,
        "PAGEDOWN" => Code::PageDown,
        "UP" | "ARROWUP" => Code::ArrowUp,
        "DOWN" | "ARROWDOWN" => Code::ArrowDown,
        "LEFT" | "ARROWLEFT" => Code::ArrowLeft,
        "RIGHT" | "ARROWRIGHT" => Code::ArrowRight,
        "MINUS" => Code::Minus,
        "PLUS" => Code::Equal,
        "COMMA" => Code::Comma,
        "PERIOD" => Code::Period,
        "SLASH" => Code::Slash,
        "BACKSLASH" => Code::Backslash,
        "OPENBRACKET" => Code::BracketLeft,
        "CLOSEBRACKET" => Code::BracketRight,
        _ => {
            // A-Z / 0-9 / F1-F12
            match main.as_str() {
                "A" => Code::KeyA,
                "B" => Code::KeyB,
                "C" => Code::KeyC,
                "D" => Code::KeyD,
                "E" => Code::KeyE,
                "F" => Code::KeyF,
                "G" => Code::KeyG,
                "H" => Code::KeyH,
                "I" => Code::KeyI,
                "J" => Code::KeyJ,
                "K" => Code::KeyK,
                "L" => Code::KeyL,
                "M" => Code::KeyM,
                "N" => Code::KeyN,
                "O" => Code::KeyO,
                "P" => Code::KeyP,
                "Q" => Code::KeyQ,
                "R" => Code::KeyR,
                "S" => Code::KeyS,
                "T" => Code::KeyT,
                "U" => Code::KeyU,
                "V" => Code::KeyV,
                "W" => Code::KeyW,
                "X" => Code::KeyX,
                "Y" => Code::KeyY,
                "Z" => Code::KeyZ,
                "0" => Code::Digit0,
                "1" => Code::Digit1,
                "2" => Code::Digit2,
                "3" => Code::Digit3,
                "4" => Code::Digit4,
                "5" => Code::Digit5,
                "6" => Code::Digit6,
                "7" => Code::Digit7,
                "8" => Code::Digit8,
                "9" => Code::Digit9,
                "F1" => Code::F1,
                "F2" => Code::F2,
                "F3" => Code::F3,
                "F4" => Code::F4,
                "F5" => Code::F5,
                "F6" => Code::F6,
                "F7" => Code::F7,
                "F8" => Code::F8,
                "F9" => Code::F9,
                "F10" => Code::F10,
                "F11" => Code::F11,
                "F12" => Code::F12,
                _ => return None,
            }
        }
    };
    Some((mods, code))
}

fn egui_key_name(key: egui::Key) -> Option<&'static str> {
    use egui::Key::*;
    let name = match key {
        Space => "SPACE",
        Enter => "ENTER",
        Tab => "TAB",
        Escape => "ESC",
        Backspace => "BACKSPACE",
        Delete => "DELETE",
        Insert => "INSERT",
        Home => "HOME",
        End => "END",
        PageUp => "PAGEUP",
        PageDown => "PAGEDOWN",
        ArrowUp => "UP",
        ArrowDown => "DOWN",
        ArrowLeft => "LEFT",
        ArrowRight => "RIGHT",
        Minus => "MINUS",
        Plus => "PLUS",
        Comma => "COMMA",
        Period => "PERIOD",
        Slash => "SLASH",
        Backslash => "BACKSLASH",
        OpenBracket => "OPENBRACKET",
        CloseBracket => "CLOSEBRACKET",
        A => "A",
        B => "B",
        C => "C",
        D => "D",
        E => "E",
        F => "F",
        G => "G",
        H => "H",
        I => "I",
        J => "J",
        K => "K",
        L => "L",
        M => "M",
        N => "N",
        O => "O",
        P => "P",
        Q => "Q",
        R => "R",
        S => "S",
        T => "T",
        U => "U",
        V => "V",
        W => "W",
        X => "X",
        Y => "Y",
        Z => "Z",
        Num0 => "0",
        Num1 => "1",
        Num2 => "2",
        Num3 => "3",
        Num4 => "4",
        Num5 => "5",
        Num6 => "6",
        Num7 => "7",
        Num8 => "8",
        Num9 => "9",
        F1 => "F1",
        F2 => "F2",
        F3 => "F3",
        F4 => "F4",
        F5 => "F5",
        F6 => "F6",
        F7 => "F7",
        F8 => "F8",
        F9 => "F9",
        F10 => "F10",
        F11 => "F11",
        F12 => "F12",
        _ => return None,
    };
    Some(name)
}

pub fn run(model_dir: &str, warnings: &[String]) {
    let mut app = ViqApp::new(model_dir);
    app.spawn_aec_probe();
    if let Some(first) = warnings.first() {
        app.status = first.clone();
    }
    let stale = app.templates_need_rebuild() && app.rec_counts.values().any(|&n| n > 0);
    if stale {
        app.start_proto_full();
        if app.proto_job.is_none() {
            app.spawn_engine_warmup();
        }
    } else if !app.auto_start_pending {
        app.spawn_engine_warmup();
    }
    let area =
        crate::window_geom::primary_work_area().unwrap_or(crate::window_geom::WORK_AREA_FALLBACK);
    let def = crate::window_geom::default_size(&area);
    let min = crate::window_geom::min_size(def);
    let saved = crate::window_geom::parse_saved(&app.settings_raw);
    let (size, pos) = match saved.filter(|g| crate::window_geom::saved_ok(g, &area, min)) {
        Some(g) => {
            let c = crate::window_geom::clamp_into(g, &area, min);
            ([c.w, c.h], Some(eframe::egui::pos2(c.x, c.y)))
        }
        None => ([def.0, def.1], None),
    };
    if saved.is_some() && pos.is_none() {
        app.log("window position and size reset (the saved record was off-screen or the scale changed)");
    }
    app.win_geom_saved_ppp = saved.map(|g| g.ppp);
    let title = window_title();
    app.win_title = title.clone();
    let mut viewport = eframe::egui::ViewportBuilder::default()
        .with_inner_size(size)
        .with_min_inner_size([min.0, min.1])
        .with_resizable(true)
        // Backstop: if the computed size still exceeds the screen, let the platform clamp it once more against the monitor size
        // (egui docs: true by default on Linux, false elsewhere)
        .with_clamp_size_to_monitor_size(true)
        .with_icon(std::sync::Arc::new(eframe::egui::IconData {
            width: 256,
            height: 256,
            rgba: include_bytes!("../assets/icon.rgba").to_vec(),
        }))
        .with_title(title);
    if let Some(p) = pos {
        viewport = viewport.with_position(p);
    }
    let options = eframe::NativeOptions {
        viewport,
        ..Default::default()
    };
    if let Err(e) = eframe::run_native(
        "Yuke",
        options,
        Box::new(move |cc| {
            setup_chinese_fonts(&cc.egui_ctx);
            Ok(Box::new(app))
        }),
    ) {
        eprintln!("egui UI failed: {e}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn engine_input_fingerprint_tracks_inputs() {
        let dir = std::env::temp_dir().join(format!("viq_warmfp_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("command_recordings")).unwrap();
        std::fs::write(dir.join("commands.json"), "{\"commands\":{}}").unwrap();
        let mut app = ViqApp::new("PROBE_TEST_DIR_SETTINGS");
        app.stop_hotkey_thread();
        app.hotkey_manager = None;
        app.model_dir = dir.to_string_lossy().to_string();

        let fp1 = app.engine_input_fingerprint();
        assert_eq!(fp1, app.engine_input_fingerprint(), "the fingerprint must be stable while inputs are unchanged");

        // New recordings change nucleus-cache hits and the voiceprint centroid -> the fingerprint changes
        std::fs::write(dir.join("command_recordings").join("a.wav"), b"x").unwrap();
        let fp2 = app.engine_input_fingerprint();
        assert_ne!(fp1, fp2, "adding or removing recordings must change the fingerprint");

        // A command-config change -> the fingerprint changes (the length differs, dodging equal mtimes at second granularity)
        std::fs::write(dir.join("commands.json"), "{\"commands\":{\"a\":{}}}").unwrap();
        let fp3 = app.engine_input_fingerprint();
        assert_ne!(fp2, fp3, "a command-config change must change the fingerprint");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Capture side (egui_key_name) and parse_hotkey key names form a closed loop: capture emits short names,
    /// which parsing must accept, while long-name aliases keep hand-typed keys working.
    #[test]
    fn hotkey_parse_accepts_captured_names() {
        use global_hotkey::hotkey::Code;
        assert_eq!(parse_hotkey("CTRL+ESC").unwrap().1, Code::Escape);
        assert_eq!(parse_hotkey("CTRL+ESCAPE").unwrap().1, Code::Escape);
        assert_eq!(parse_hotkey("SHIFT+UP").unwrap().1, Code::ArrowUp);
        assert_eq!(parse_hotkey("SHIFT+ARROWUP").unwrap().1, Code::ArrowUp);
        assert_eq!(parse_hotkey("CTRL+DOWN").unwrap().1, Code::ArrowDown);
        assert_eq!(parse_hotkey("CTRL+LEFT").unwrap().1, Code::ArrowLeft);
        assert_eq!(parse_hotkey("CTRL+RIGHT").unwrap().1, Code::ArrowRight);
        assert_eq!(parse_hotkey("CTRL+F8").unwrap().1, Code::F8);
        // egui Key -> capture name (short form)
        assert_eq!(egui_key_name(egui::Key::ArrowUp), Some("UP"));
        assert_eq!(egui_key_name(egui::Key::Escape), Some("ESC"));
        assert_eq!(egui_key_name(egui::Key::Num0), Some("0"));
        assert_eq!(egui_key_name(egui::Key::E), Some("E"));
    }

    #[test]
    fn load_cfg_backs_up_corrupt_and_degrades() {
        let dir = "PROBE_TEST_DIR_CFG";
        let _ = std::fs::remove_dir_all(dir);
        std::fs::create_dir_all(dir).unwrap();
        let p = std::path::Path::new(dir).join("commands.json");
        std::fs::write(&p, "{ is not valid json").unwrap();
        let (v, path) = load_cfg(dir);
        assert_eq!(v, serde_json::json!({}), "a corrupt file must degrade to an empty configuration");
        assert!(path.ends_with("commands.json"));
        let has_bak = std::fs::read_dir(dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .any(|e| e.file_name().to_string_lossy().contains(".bak."));
        assert!(has_bak, "the corrupt file must be backed up before degrading");
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn write_json_atomic_replaces_without_tmp() {
        let dir = "PROBE_TEST_DIR_ATOMIC";
        let _ = std::fs::remove_dir_all(dir);
        std::fs::create_dir_all(dir).unwrap();
        let p = std::path::Path::new(dir).join("x.json");
        write_json_atomic(&p, "v1").unwrap();
        write_json_atomic(&p, "v2").unwrap();
        assert_eq!(std::fs::read_to_string(&p).unwrap(), "v2");
        assert_eq!(
            std::fs::read_dir(dir).unwrap().count(),
            1,
            "no temporary files may be left behind"
        );
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn delete_command_removes_and_persists_atomically() {
        let mut app = ViqApp::new("PROBE_TEST_DIR_APP");
        app.stop_hotkey_thread(); // Do not leave global hotkey polling in the test process
        app.hotkey_manager = None; // Release the global hotkey registration
        std::fs::create_dir_all("PROBE_TEST_DIR_APP").unwrap();
        // Now that paths are unified, cfg_path is derived directly from model_dir (<model_dir>/commands.json),
        // with no manual redirection; this also asserts that rule
        assert_eq!(
            app.cfg_path,
            PathBuf::from("PROBE_TEST_DIR_APP/commands.json")
        );
        app.cfg = serde_json::json!({
            "commands": {
                "__probe_a__": {"enabled": true,  "key": "E"},
                "__probe_b__": {"enabled": false, "key": "Q"},
            }
        });
        app.reset_key_edit_bufs();
        app.delete_command("__probe_a__");
        let cmds = app.cfg.get("commands").unwrap();
        assert!(cmds.get("__probe_a__").is_none(), "the command must be removed after confirmation");
        assert!(cmds.get("__probe_b__").is_some(), "other commands must be kept");
        let disk = std::fs::read_to_string("PROBE_TEST_DIR_APP/commands.json").unwrap();
        assert!(
            disk.contains("__probe_b__") && !disk.contains("__probe_a__"),
            "the deletion must be persisted atomically"
        );
        std::fs::remove_dir_all("PROBE_TEST_DIR_APP").ok();
    }

    #[test]
    fn delete_command_removes_recordings_and_proto_entry() {
        let dir = "PROBE_TEST_DIR_DELCMD";
        let _ = std::fs::remove_dir_all(dir);
        std::fs::create_dir_all(dir).unwrap();
        let mut app = ViqApp::new(dir);
        app.stop_hotkey_thread();
        app.hotkey_manager = None;
        app.cfg = serde_json::json!({
            "commands": {"cmd1": {"enabled": true, "key": "E", "phrase": "闪现"}}
        });
        // cmd1 with two real recordings (wav on disk plus the index) and a proto entry (another command's entry coexists)
        std::fs::create_dir_all(format!("{dir}/command_recordings")).unwrap();
        for f in [
            "cmd1_act_20990101_000000.wav",
            "cmd1_act_20990101_000001.wav",
        ] {
            crate::features::write_wav(
                &format!("{dir}/command_recordings/{f}"),
                &[0.0f32; 160],
                16000,
            )
            .unwrap();
            crate::recordings::append(
                dir,
                crate::recordings::CmdRecording {
                    cmd: "cmd1".into(),
                    file: f.into(),
                    kind: crate::recordings::RecKind::Active,
                    ts: "20990101_000000".into(),
                    dur_s: 1.0,
                },
            )
            .unwrap();
        }
        std::fs::write(
            format!("{dir}/embed_proto_v2.json"),
            serde_json::to_string_pretty(&serde_json::json!({
                "cmd1": {"0": vec![0.5f64; 4]},
                "other": {"0": vec![0.25f64; 4]},
            }))
            .unwrap(),
        )
        .unwrap();
        app.refresh_rec_counts();
        app.refresh_proto_cover();
        assert_eq!(app.rec_counts.get("cmd1").copied(), Some(2));
        assert!(app.proto_cover.contains("cmd1"));

        app.delete_command("cmd1");

        // Delete a command; its wav, index entry and proto entry all disappear, and other commands' entries remain
        assert!(app.cfg["commands"].get("cmd1").is_none());
        assert_eq!(app.rec_counts.get("cmd1").copied().unwrap_or(0), 0);
        assert!(!app.proto_cover.contains("cmd1"));
        assert!(
            !std::path::PathBuf::from(format!(
                "{dir}/command_recordings/cmd1_act_20990101_000000.wav"
            ))
            .exists()
        );
        assert!(
            crate::recordings::list(dir).is_empty(),
            "the index must not keep a cmd1 entry"
        );
        let v: serde_json::Value = serde_json::from_str(
            &crate::crypto::read_text_auto(&std::path::PathBuf::from(format!(
                "{dir}/embed_proto_v2.json"
            )))
            .unwrap(),
        )
        .unwrap();
        assert!(v.get("cmd1").is_none(), "the template entry must be gone");
        assert_eq!(
            v["other"]["0"],
            serde_json::json!(vec![0.25f64; 4]),
            "other commands' entries must be kept"
        );
        // A new command reusing an id: 0 recordings, no template (guards against "matching the wrong pronunciation")
        app.cfg["commands"]["cmd1"] =
            serde_json::json!({"enabled": true, "key": "W", "phrase": "上"});
        app.refresh_rec_counts();
        app.refresh_proto_cover();
        assert_eq!(app.rec_counts.get("cmd1").copied().unwrap_or(0), 0);
        assert!(!app.proto_cover.contains("cmd1"));
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn add_command_creates_unique_id_and_persists() {
        // 1) Id scheme: no cmdN taken -> cmd1; cmd1 and cmd2 taken -> cmd3; no clash with pinyin ids
        assert_eq!(next_command_id(&serde_json::json!({})), "cmd1");
        assert_eq!(
            next_command_id(&serde_json::json!({"commands": {"chong": {}, "e": {}}})),
            "cmd1"
        );
        assert_eq!(
            next_command_id(&serde_json::json!({"commands": {"cmd1": {}, "cmd2": {}, "gong": {}}})),
            "cmd3"
        );

        let mut app = ViqApp::new("PROBE_TEST_DIR_ADD");
        app.stop_hotkey_thread(); // Do not leave global hotkey polling in the test process
        app.hotkey_manager = None;
        std::fs::create_dir_all("PROBE_TEST_DIR_ADD").unwrap();
        assert_eq!(
            app.cfg_path,
            PathBuf::from("PROBE_TEST_DIR_ADD/commands.json")
        );
        app.cfg = serde_json::json!({
            "commands": {
                "__probe_a__": {"enabled": true, "key": "E", "mode": "tap", "phrase": "攻"},
            }
        });
        app.reset_key_edit_bufs();

        // 2) Clicking "+ New pronunciation": the command is persisted immediately and the edit dialog opens in create mode
        app.start_new_command();
        let id1 = app.pron_open.clone().expect("the edit dialog opens after adding");
        assert!(
            app.pron_new && app.pron_new_focus,
            "must be in create mode (the first row draws a name field)"
        );
        assert_ne!(id1, "__probe_a__", "the new id must not clash with existing commands");
        let c1 = app.cfg.get("commands").unwrap().get(&id1).unwrap().clone();
        assert_eq!(
            c1.get("phrase").and_then(|v| v.as_str()),
            Some(""),
            "the name must be empty in create mode"
        );
        assert_eq!(
            c1.get("enabled").and_then(|v| v.as_bool()),
            Some(true),
            "enabled defaults to on"
        );
        assert_eq!(
            c1.get("key").and_then(|v| v.as_str()),
            Some(""),
            "the key is empty by default (configure it in the dialog)"
        );
        assert_eq!(c1.get("mode").and_then(|v| v.as_str()), Some("tap"));
        let disk = std::fs::read_to_string("PROBE_TEST_DIR_ADD/commands.json").unwrap();
        assert!(disk.contains(&id1), "clicking add must persist atomically right away");

        // 3) Closing with nothing entered -> the empty command is deleted (add followed by an immediate close leaves no junk)
        let n_before = app.cfg["commands"].as_object().unwrap().len();
        app.close_edit_popup();
        assert_eq!(
            app.cfg["commands"].as_object().unwrap().len(),
            n_before - 1,
            "an empty temporary command must be deleted"
        );
        assert!(
            !app.pron_new && app.pron_open.is_none(),
            "closing the dialog must reset create mode"
        );
        let disk = std::fs::read_to_string("PROBE_TEST_DIR_ADD/commands.json").unwrap();
        assert!(!disk.contains(&id1), "the deletion must be persisted too");

        // 4) Information entered but no name -> named "New phrase 1" automatically
        app.start_new_command();
        let id2 = app.pron_open.clone().unwrap();
        app.set_cmd_field(&id2, "key", serde_json::Value::String("F8".into()));
        app.persist_cfg();
        app.close_edit_popup();
        assert_eq!(
            app.cmd_field_str(&id2, "phrase"),
            crate::lang::tf(k::MSG_DEFAULT_CMD_NAME, &[("{n}", "1")]),
            "a key but no name -> auto-named \"New phrase 1\""
        );

        // 5) Two in a row without names: the number increments and never collides with an existing name
        app.start_new_command();
        let id3 = app.pron_open.clone().unwrap();
        app.set_cmd_field(&id3, "key", serde_json::Value::String("F9".into()));
        app.persist_cfg();
        app.close_edit_popup();
        assert_eq!(
            app.cmd_field_str(&id3, "phrase"),
            crate::lang::tf(k::MSG_DEFAULT_CMD_NAME, &[("{n}", "2")]),
            "the second unnamed command gets \"New phrase 2\""
        );

        app.start_new_command();
        let id4 = app.pron_open.clone().unwrap();
        app.add_phrase = "闪现".into();
        app.close_edit_popup();
        assert_eq!(
            app.cmd_field_str(&id4, "phrase"),
            "闪现",
            "closing must commit the name buffer (typed and closed without losing focus)"
        );
        std::fs::remove_dir_all("PROBE_TEST_DIR_ADD").ok();
    }

    #[test]
    fn proto_job_poll_success_and_failure_paths() {
        let mut app = ViqApp::new("PROBE_TEST_DIR_APP_PROTO");
        app.stop_hotkey_thread(); // Do not leave global hotkey polling in the test process
        app.hotkey_manager = None;
        // 1) Success while the engine is stopped -> show the completion message immediately (noting it applies on the next start)
        let (tx, rx) = std::sync::mpsc::channel();
        tx.send(Ok("pronunciation templates updated (__probe__, 2 recordings)".into()))
            .unwrap();
        app.proto_job = Some(ProtoJob {
            kind: ProtoKind::Single("__probe__".into()),
            rx,
        });
        app.poll_proto_job();
        assert!(app.proto_job.is_none(), "the job must be cleared once finished");
        assert!(
            app.status.contains("pronunciation templates updated"),
            "status={}",
            app.status
        );
        assert!(
            app.status.contains("下次开始识别时生效"),
            "status={}",
            app.status
        );
        // 2) Failure -> error message, no engine operation
        let (tx, rx) = std::sync::mpsc::channel();
        tx.send(Err(anyhow::anyhow!("encoding failed"))).unwrap();
        app.proto_job = Some(ProtoJob {
            kind: ProtoKind::Full,
            rx,
        });
        app.poll_proto_job();
        assert!(app.proto_job.is_none());
        assert!(
            app.status.contains("发音模板生成失败"), // i18n template text (lang/zh.json)
            "status={}",
            app.status
        );
        assert!(app.status.contains("encoding failed"), "status={}", app.status);
        // 3) An idle task (None) must not change any state
        let keep = app.status.clone();
        app.poll_proto_job();
        assert_eq!(app.status, keep);
        std::fs::remove_dir_all("PROBE_TEST_DIR_APP_PROTO").ok();
    }

    /// Recording-path probe (exactly the chain used by stop_rec in ui.rs):
    /// MicMonitor capture -> snapshot -> write_wav -> recordings::append -> list -> remove.
    /// Uses a dedicated PROBE directory so the real model index is untouched. Requires a working microphone.
    #[test]
    #[ignore = "requires a microphone"]
    fn rerecord_writes_wav_and_index() {
        let mic = match crate::mic_monitor::MicMonitor::start(None) {
            Ok(m) => m,
            Err(e) => {
                println!("skipped (no usable microphone): {e}");
                return;
            }
        };
        let start_len = mic.len();
        std::thread::sleep(std::time::Duration::from_millis(1200));
        let samples = mic.snapshot(start_len);
        drop(mic);
        assert!(samples.len() >= 1600, "recording too short: {} samples", samples.len());

        let model_dir = "PROBE_TEST_DIR";
        let rec_dir = crate::recordings::record_dir(model_dir);
        std::fs::create_dir_all(&rec_dir).unwrap();
        let ts = "20990101_000000";
        let file = format!("__probe___act_{ts}.wav");
        let path = rec_dir.join(&file);
        crate::features::write_wav(
            path.to_str().unwrap(),
            &samples,
            crate::features::TARGET_SAMPLE_RATE,
        )
        .expect("write_wav failed");
        crate::recordings::append(
            model_dir,
            crate::recordings::CmdRecording {
                cmd: "__probe__".into(),
                file: file.clone(),
                kind: crate::recordings::RecKind::Active,
                ts: ts.into(),
                dur_s: samples.len() as f32 / crate::features::TARGET_SAMPLE_RATE as f32,
            },
        )
        .expect("append failed");

        let listed = crate::recordings::list_cmd(model_dir, "__probe__");
        assert_eq!(listed.len(), 1, "the index must contain the probe recording");
        assert!(path.exists(), "the wav must be on disk");

        // Delete path (same as the secondary page's delete button)
        crate::recordings::remove(model_dir, &file).expect("remove failed");
        assert!(!path.exists());
        assert!(crate::recordings::list_cmd(model_dir, "__probe__").is_empty());
        std::fs::remove_dir_all(model_dir).ok();
        println!(
            "extra-take chain OK: {} samples -> {} -> index/delete passed",
            samples.len(),
            file
        );
    }

    #[test]
    fn new_command_hits_authoritative_file_and_engine_count_plus_one() {
        let model =
            std::env::temp_dir().join(format!("viq_pathunify_new_cmd_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&model);
        std::fs::create_dir_all(&model).unwrap();
        let auth = model.join("commands.json");
        // Preset three commands as the authoritative factory file (2 with enabled=false, so the engine loads only cmd1)
        let base = serde_json::json!({
            "commands": {
                "1":    {"enabled": false, "key": "1", "mode": "tap", "phrase": "一"},
                "c":    {"enabled": false, "key": "R", "mode": "tap", "phrase": "唱"},
                "cmd1": {"enabled": true,  "key": "G", "mode": "tap", "phrase": "攻"},
            }
        });
        std::fs::write(&auth, serde_json::to_string_pretty(&base).unwrap()).unwrap();
        let md = model.to_str().unwrap().to_string();

        // Engine-side baseline (engine reload counter)
        let before = crate::config::CommandConfig::load_for_model(&md)
            .expect("engine loads the authoritative config")
            .commands
            .len();
        assert_eq!(
            before, 1,
            "the engine only loads commands with enabled != false (config.rs:733)"
        );

        // UI side: cfg_path must be the authoritative file and must see every command (including disabled ones)
        let mut app = ViqApp::new(&md);
        app.stop_hotkey_thread(); // Do not leave global hotkey polling in the test process
        app.hotkey_manager = None;
        assert_eq!(
            app.cfg_path, auth,
            "the UI must read and write <model_dir>/commands.json"
        );
        assert_eq!(
            app.cfg_path.parent(),
            Some(model.as_path()),
            "it must no longer point at a cwd-relative config/"
        );
        let ui_total_before = app
            .cfg
            .get("commands")
            .and_then(|v| v.as_object())
            .unwrap()
            .len();
        assert_eq!(ui_total_before, 3, "the UI must see all three commands from the authoritative file");

        app.start_new_command();
        let id = app.pron_open.clone().expect("the edit dialog opens after adding");
        assert_eq!(id, "cmd2", "id generation skips the taken cmd1");
        app.set_cmd_field(&id, "phrase", serde_json::Value::String("闪现".into()));
        app.persist_cfg();

        // Persistence assertion: the new command appears in the authoritative file
        let disk = std::fs::read_to_string(&auth).unwrap();
        assert!(
            disk.contains(&id) && disk.contains("闪现"),
            "the new command must land in the authoritative file"
        );
        let disk_json: serde_json::Value = serde_json::from_str(&disk).unwrap();
        assert_eq!(
            disk_json["commands"].as_object().unwrap().len(),
            ui_total_before + 1
        );
        // and the persisted object is the in-memory cfg (UI and engine read the same file)
        assert_eq!(disk_json["commands"], app.cfg["commands"]);

        // Engine reload -> counter +1
        let after = crate::config::CommandConfig::load_for_model(&md)
            .expect("engine reloads the authoritative config")
            .commands
            .len();
        assert_eq!(
            after,
            before + 1,
            "the engine must see one more command after the reload ({before} -> {after})"
        );
        assert!(app.cfg_path.exists(), "the authoritative file must exist (no fallback to config/)");

        let _ = std::fs::remove_dir_all(&model);
        println!(
            "path unification closed loop OK: UI added {id} -> {} | engine count {before} -> {after}",
            auth.display()
        );
    }

    /// Settings persistence: unknown keys are preserved verbatim, known keys are written back normally.
    #[test]
    fn settings_preserves_unknown_keys_and_writes_known_fields() {
        let mut app = ViqApp::new("PROBE_TEST_DIR_SETTINGS");
        app.stop_hotkey_thread();
        app.hotkey_manager = None;
        app.settings_raw = serde_json::json!({
            "udkws_v2": false,
            "max_active_paths": 9,
            "cmn": true, "cmn_min": 55.0, "multiband": true,
            "listen_passcode_when_off": false,
            "record_enabled": true, "device": "",
        });
        app.settings.record_enabled = false;
        app.settings.auto_start_recognize = true;
        app.settings.voice_toggle_cmd = "__probe_cmd__".into(); // The voice toggle is written back by persist
        app.persist_settings();
        let disk: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(crate::config_path("settings.json")).unwrap(),
        )
        .unwrap();
        // 1) Unknown keys are not deleted (kept verbatim in the json), and a UI change does not overwrite them with new defaults
        assert_eq!(
            disk["udkws_v2"].as_bool(),
            Some(false),
            "unknown udkws_v2 keys must be preserved verbatim"
        );
        assert_eq!(
            disk["max_active_paths"].as_u64(),
            Some(9),
            "unknown beam-width keys must be preserved verbatim"
        );
        assert_eq!(
            disk["cmn"].as_bool(),
            Some(true),
            "unknown experiment keys must be preserved verbatim"
        );
        assert_eq!(disk["cmn_min"].as_f64(), Some(55.0));
        // 2) Keys still present in the UI are written back normally (auto_start_recognize explicitly on)
        assert_eq!(disk["record_enabled"].as_bool(), Some(false));
        assert_eq!(disk["auto_start_recognize"].as_bool(), Some(true));
        assert_eq!(
            disk["voice_toggle_cmd"].as_str(),
            Some("__probe_cmd__"),
            "the voice-global toggle command id is persisted"
        );
        // 4) Retired keys: even if an old json still carries them, persist does not write them back (listening semantics moved into
        // voice_toggle_cmd; configuring it on the gate side enables listening)
        assert!(
            disk.get("listen_passcode_when_off").is_none(),
            "the retired key listen_passcode_when_off must not be written back: {}",
            disk
        );
        let (reloaded, _) = load_settings();
        assert_eq!(reloaded.record_enabled, false);
        assert_eq!(reloaded.auto_start_recognize, true);
        assert_eq!(reloaded.device, "");
        assert_eq!(
            reloaded.voice_toggle_cmd, "__probe_cmd__",
            "the voice-global toggle is restored after a restart"
        );
        // Cleanup: remove the settings.json written by the test (under the test exe directory's config/)
        let _ = std::fs::remove_file(crate::config_path("settings.json"));
        let _ = std::fs::remove_dir_all("PROBE_TEST_DIR_SETTINGS");
    }

    /// The "optimise recognition" confirmation dialog: enabling and disabling take the same path -- persist + prompt for a restart,
    /// neither regenerating templates on the spot (this process's front end is locked by a OnceLock, so recomputing would use the old one;
    /// both directions invalidate the templates, and the rebuild happens automatically at load).
    #[test]
    fn boost_confirm_persists_both_directions_without_regenerating() {
        std::fs::create_dir_all("PROBE_TEST_DIR_BOOST").unwrap();
        let mut app = ViqApp::new("PROBE_TEST_DIR_BOOST");
        app.stop_hotkey_thread();
        app.hotkey_manager = None;

        // on -> off
        app.settings.frame_mean_norm = true;
        app.exec_confirm(Confirm::BoostRecognition { enable: false });
        assert!(!app.settings.frame_mean_norm, "the off direction must persist");
        assert!(app.proto_job.is_none(), "the off direction must not regenerate in place");
        assert_eq!(
            app.status,
            t(k::MSG_BOOST_SWITCH_SAVED),
            "both directions show the same message"
        );
        assert_eq!(load_settings().0.frame_mean_norm, false, "off must be persisted");

        // off -> on: likewise no regeneration
        app.exec_confirm(Confirm::BoostRecognition { enable: true });
        assert!(app.settings.frame_mean_norm, "the on direction must persist");
        assert!(app.proto_job.is_none(), "the on direction must likewise not regenerate in place");
        assert_eq!(load_settings().0.frame_mean_norm, true, "on must be persisted");

        let _ = std::fs::remove_file(crate::config_path("settings.json"));
        let _ = std::fs::remove_dir_all("PROBE_TEST_DIR_BOOST");
    }

    /// Guard: while template parameters have changed and no restart has happened, both template-generation entry points must block and prompt for a restart.
    #[test]
    fn regen_blocked_when_template_params_pending_restart() {
        std::fs::create_dir_all("PROBE_TEST_DIR_FEGUARD").unwrap();
        let mut app = ViqApp::new("PROBE_TEST_DIR_FEGUARD");
        app.stop_hotkey_thread();
        app.hotkey_manager = None;
        app.settings.frame_mean_norm = !crate::engine::udkws_v2::frame_mean_norm();
        assert!(app.template_params_pending_restart(), "this must count as a pending-restart state");

        app.status.clear();
        app.start_proto_full();
        assert_eq!(app.status, t(k::MSG_REGEN_NEED_RESTART), "the regenerate-all entry must be blocked");
        assert!(app.proto_job.is_none(), "no job may start while blocked");

        app.status.clear();
        app.start_proto_single("cmd_probe");
        assert_eq!(app.status, t(k::MSG_REGEN_NEED_RESTART), "the single-entry must be blocked");
        assert!(app.proto_job.is_none(), "no job may start while blocked");

        let _ = std::fs::remove_dir_all("PROBE_TEST_DIR_FEGUARD");
    }

    #[test]
    fn voice_toggle_setting_reads_cmd_key() {
        assert_eq!(
            voice_toggle_from_raw(&serde_json::json!({"voice_toggle_cmd": "cmd9"})),
            "cmd9"
        );
        assert_eq!(voice_toggle_from_raw(&serde_json::json!({})), "");
    }

    /// Probe: a passcode hit dispatches through toggle rather than a key mapping -- voice_toggle_dispatch
    /// only flips enabled_flag when the passcode pronunciation matches (the same state source as "start input"),
    /// does not flip for other commands, does not flip twice for the same trigger, and toggles back and forth across two utterances. The key side is
    /// covered by the mute_passcode_key probe (the engine-side passcode key is empty, and gate's existing semantics for an empty key
    /// are to press no key and record no event while still updating count/last_command).
    #[test]
    fn voice_toggle_passcode_hit_toggles_global_switch() {
        let mut app = ViqApp::new("PROBE_TEST_DIR_VTOGGLE");
        app.stop_hotkey_thread(); // Do not leave global hotkey polling in the test process
        app.hotkey_manager = None;
        app.settings.voice_toggle_cmd = "cmd1".into();
        let flag = app.enabled_flag.clone();
        flag.store(true, std::sync::atomic::Ordering::Relaxed);
        app.last_enabled_seen = true;
        // 1) A non-passcode trigger: only advances the trigger baseline, does not flip the switch
        app.voice_toggle_dispatch(1, "cmd2");
        assert!(
            flag.load(std::sync::atomic::Ordering::Relaxed),
            "a non-passcode hit must not flip the toggle"
        );
        // 2) Passcode hit -> toggle (on -> off)
        app.voice_toggle_dispatch(2, "cmd1");
        assert!(
            !flag.load(std::sync::atomic::Ordering::Relaxed),
            "a passcode hit should switch the toggle off"
        );
        // 3) The same trigger reported twice (re-read in the same frame): must not flip again
        app.voice_toggle_dispatch(2, "cmd1");
        assert!(
            !flag.load(std::sync::atomic::Ordering::Relaxed),
            "the same trigger must not flip twice"
        );
        // 4) Saying the passcode again -> toggles back (off -> on)
        app.voice_toggle_dispatch(3, "cmd1");
        assert!(
            flag.load(std::sync::atomic::Ordering::Relaxed),
            "saying the passcode again should switch it back on"
        );
        // 5) After the passcode is retired, a normal command: no flip (update also short-circuits on an empty passcode before dispatch)
        app.settings.voice_toggle_cmd.clear();
        app.voice_toggle_dispatch(4, "cmd2");
        assert!(
            flag.load(std::sync::atomic::Ordering::Relaxed),
            "with the passcode disabled the toggle must not flip"
        );
        std::fs::remove_dir_all("PROBE_TEST_DIR_VTOGGLE").ok();
    }

    #[test]
    fn mute_passcode_key_clears_only_passcode() {
        let cmd = |k: &str| crate::config::Command {
            key: k.to_string(),
            mode: "tap".to_string(),
        };
        let mut commands = std::collections::HashMap::from([
            ("pass".to_string(), cmd("E")),
            ("other".to_string(), cmd("Q")),
        ]);
        // No passcode configured (empty id): no operation
        mute_passcode_key(&mut commands, "");
        assert_eq!(commands["pass"].key, "E");
        // Clearing the passcode: only `pass` changes and `other` is untouched; a nonexistent id is a no-op
        mute_passcode_key(&mut commands, "pass");
        mute_passcode_key(&mut commands, "gone");
        assert_eq!(commands["pass"].key, "", "the passcode key must be empty (a hit presses nothing)");
        assert_eq!(commands["other"].key, "Q", "non-passcode command keys must not be cleared");
    }

    fn probe_listen_config() -> crate::config::CommandConfig {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let raw = serde_json::json!({
            "commands": {
                "pass": {"key": "", "mode": "tap"},
                "other": {"key": "K9", "mode": "tap"},
            },
        });
        let dir = std::env::temp_dir().join(format!(
            "viq_ui_listen_{}_{}",
            std::process::id(),
            COUNTER.fetch_add(1, Ordering::SeqCst)
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let cfg_path = dir.join("commands.json");
        std::fs::write(&cfg_path, raw.to_string()).unwrap();
        crate::config::CommandConfig::load(cfg_path.to_str().unwrap()).unwrap()
    }

    #[test]
    fn passcode_listen_full_chain_on_off_on() {
        let mut app = ViqApp::new("PROBE_TEST_DIR_VLISTEN");
        app.stop_hotkey_thread(); // Do not leave global hotkey polling in the test process
        app.hotkey_manager = None;
        app.settings.voice_toggle_cmd = "pass".into();
        // Same construction and injection as an engine-ready path (the mock model takes no part in recognition; triggers all go through emit_v2_triggers)
        let flag = app.enabled_flag.clone();
        let mut proc = LiveProcessor::new(probe_listen_config(), flag.clone(), None);
        proc.set_replay_mode();
        proc.set_passcode_cmd(&app.settings.voice_toggle_cmd);
        app.last_seen_trigger_count = 0;
        app.last_enabled_seen = true;
        let trig = |sec: f64, cmd: &str| crate::udkws_v2::Trigger {
            start_sec: sec as f32,
            end_sec: (sec + 0.08) as f32,
            cmd: cmd.to_string(),
            score: 0.9,
            t: (sec * 100.0) as usize,
        };
        let on = || flag.load(std::sync::atomic::Ordering::Relaxed);
        proc.emit_v2_triggers(vec![trig(1.0, "pass")]);
        assert_eq!(proc.keypress_count(), 1);
        assert!(proc.key_events.is_empty(), "the passcode key is empty: no keypress, no recorded events");
        app.voice_toggle_dispatch(proc.keypress_count(), &proc.last_command());
        assert!(!on(), "saying the passcode while on should turn it off");
        proc.set_enabled(false); // the same key-release path the update-frame check uses
        app.last_enabled_seen = false;
        // Phase 2: with the switch off, say the passcode (the core of listening: a configured passcode listens automatically) -> gate only advances
        // the count and presses nothing, and dispatch flips it back on
        proc.emit_v2_triggers(vec![trig(2.0, "pass")]);
        assert_eq!(proc.keypress_count(), 2, "a passcode hit in listening state must advance the counter");
        assert!(
            proc.key_events.is_empty(),
            "a listening-state hit produces zero key output (mute semantics)"
        );
        app.voice_toggle_dispatch(proc.keypress_count(), &proc.last_command());
        assert!(on(), "saying the passcode while off turns it back on (listening semantics)");
        // saying the passcode while off turns it back on (listening semantics)
        // -> gate swallows everything (count unchanged) -> dispatch sees no new trigger and the switch stays off
        proc.emit_v2_triggers(vec![trig(3.0, "pass")]);
        app.voice_toggle_dispatch(proc.keypress_count(), &proc.last_command());
        assert!(!on(), "saying the passcode while on turns it off again (round trip)");
        proc.emit_v2_triggers(vec![trig(4.0, "other")]);
        assert_eq!(proc.keypress_count(), 3, "a non-passcode hit must not advance the count");
        assert!(proc.key_events.is_empty(), "a non-passcode hit produces no key output");
        app.voice_toggle_dispatch(proc.keypress_count(), &proc.last_command());
        assert!(!on(), "a non-passcode hit while off must not flip the toggle");
        // a non-passcode hit while off must not flip the switch
        // passcode hits are swallowed too (listening is derived from the passcode configuration, not an independent switch)
        proc.set_passcode_cmd("");
        proc.emit_v2_triggers(vec![trig(5.0, "pass")]);
        assert_eq!(proc.keypress_count(), 3, "passcode disabled: a hit while off does not advance either");
        app.voice_toggle_dispatch(proc.keypress_count(), &proc.last_command());
        assert!(!on(), "passcode disabled: the passcode cannot restore it while off (fully swallowed)");
        std::fs::remove_dir_all("PROBE_TEST_DIR_VLISTEN").ok();
    }

    #[test]
    fn macro_editor_save_and_reentry_policy_probe() {
        let dir = "PROBE_TEST_DIR_MEDIT";
        let _ = std::fs::remove_dir_all(dir);
        std::fs::create_dir_all(dir).unwrap();
        let mut app = ViqApp::new(dir);
        app.stop_hotkey_thread(); // Do not leave global hotkey polling in the test process
        app.hotkey_manager = None;
        app.cfg = serde_json::json!({"commands": {
            "cmd2": {"enabled": true, "key": "", "mode": "macro", "phrase": "回城补给"},
        }});
        app.macro_edit = vec![
            MacroRow {
                kind: 0,
                param: "B".into(),
            }, // down B
            MacroRow {
                kind: MK_WAIT,
                param: "250".into(),
            }, // wait 250
            MacroRow {
                kind: 0,
                param: String::new(),
            }, // down (empty key name, invalid)
        ];
        app.macro_save_from_edit("cmd2");
        let saved = app
            .cfg
            .get("commands")
            .and_then(|v| v.get("cmd2"))
            .and_then(|c| c.get("macro"))
            .and_then(|v| v.as_array())
            .expect("after saving cfg must have the cmd2.macro array");
        assert_eq!(saved.len(), 2, "invalid lines must not persist: {saved:?}");
        // invalid lines must not be persisted: {saved:?}
        let cur = app
            .cfg
            .get("commands")
            .and_then(|v| v.get("cmd2"))
            .and_then(|c| c.get("macro_reentry"))
            .and_then(|v| v.as_str())
            .unwrap_or(crate::config::MACRO_REENTRY_IGNORE);
        assert_eq!(
            cur, "ignore",
            "a missing macro_reentry field reads as ignore (backward compatible)"
        );
        app.set_cmd_field(
            "cmd2",
            "macro_reentry",
            serde_json::Value::String("interrupt".into()),
        );
        app.persist_cfg();
        assert_eq!(app.cfg["commands"]["cmd2"]["macro_reentry"], "interrupt");
        // 3) The macro window draws its own frame loop (re-entry dropdown, row cards, toolbar, the whole UI) without panicking
        app.macro_editor_open = Some("cmd2".into());
        app.macro_edit_for = None; // force a buffer reload from cfg (macro-window open mechanism)
        let ctx = egui::Context::default();
        let frame = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(900.0, 660.0),
            )),
            ..Default::default()
        };
        let _ = ctx.run(frame, |_c| {
            app.draw_macro_editor(&ctx);
        });
        assert_eq!(
            app.macro_edit.len(),
            2,
            "after reopening the buffer must reload from cfg (2 valid actions)"
        );
        let on_disk = std::fs::read_to_string(std::path::Path::new(dir).join("commands.json"))
            .expect("persist_cfg should have written to disk");
        assert!(
            on_disk.contains("macro_reentry"),
            "macro_reentry must be written into commands.json"
        );
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn mutex_editor_symmetric_save_and_probe() {
        let dir = "PROBE_TEST_DIR_MUTEX";
        let _ = std::fs::remove_dir_all(dir);
        std::fs::create_dir_all(dir).unwrap();
        let mut app = ViqApp::new(dir);
        app.stop_hotkey_thread();
        app.hotkey_manager = None;
        app.cfg = serde_json::json!({"commands": {
            "cmd1": {"enabled": true, "key": "W", "phrase": "前"},
            "cmd2": {"enabled": true, "key": "A", "phrase": "左"},
            "cmd3": {"enabled": true, "key": "D", "phrase": "右"},
        }});
        app.reset_key_edit_bufs();
        // 1) The panel checks cmd2 -> written on both sides (this command shows up in the other list = mutually visible)
        app.mutex_edit = vec!["cmd2".into()];
        app.apply_mutex_edit("cmd1");
        assert_eq!(
            app.cfg["commands"]["cmd1"]["mutex"],
            serde_json::json!(["cmd2"])
        );
        assert_eq!(
            app.cfg["commands"]["cmd2"]["mutex"],
            serde_json::json!(["cmd1"]),
            "mutex is mutual: this command must also appear in the other side's list"
        );
        assert_eq!(
            app.mutex_set_of("cmd2"),
            vec!["cmd1".to_string()],
            "the cmd2 editor must show cmd1 as selected (union read on both directions)"
        );
        assert_eq!(app.mutex_set_of("cmd1"), vec!["cmd2".to_string()]);
        assert!(
            app.mutex_set_of("cmd3").is_empty(),
            "the uninvolved cmd3 must not be affected"
        );
        // 2) Unchecking -> removed on both sides (an empty list leaves no field, keeping the config clean)
        app.mutex_edit.clear();
        app.apply_mutex_edit("cmd1");
        assert!(
            app.cfg["commands"]["cmd1"].get("mutex").is_none(),
            "clearing must not leave an empty array"
        );
        assert!(
            app.cfg["commands"]["cmd2"].get("mutex").is_none(),
            "the reference on the other side is removed too"
        );
        // 3) A multi-selection is persisted, and the panel/edit-popup frame loops draw without panicking (including the mutex button row)
        app.mutex_edit = vec!["cmd3".into(), "cmd2".into()];
        app.apply_mutex_edit("cmd1");
        assert_eq!(
            app.mutex_set_of("cmd1"),
            vec!["cmd2".to_string(), "cmd3".to_string()]
        );
        let ctx = egui::Context::default();
        let frame = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(900.0, 660.0),
            )),
            ..Default::default()
        };
        app.mutex_editor_open = Some("cmd1".into());
        app.mutex_edit_for = None; // force a buffer reload from cfg (open mechanism)
        let _ = ctx.run(frame, |_c| {
            app.draw_mutex_editor(&ctx);
        });
        assert_eq!(
            app.mutex_edit,
            vec!["cmd2".to_string(), "cmd3".to_string()],
            "after reopening the buffer must reload from cfg as the two-direction union"
        );
        app.pron_open = Some("cmd1".into());
        let frame2 = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(900.0, 660.0),
            )),
            ..Default::default()
        };
        let _ = ctx.run(frame2, |_c| {
            app.draw_edit_popup(&ctx);
        });
        // 4) Deleting cmd2 -> leftover references elsewhere (cmd1's list) are cleaned up; if the panel is editing it, the panel closes
        app.mutex_editor_open = Some("cmd2".into());
        app.delete_command("cmd2");
        assert!(app.mutex_editor_open.is_none(), "the panel of a deleted command must auto-close");
        assert_eq!(
            app.mutex_set_of("cmd1"),
            vec!["cmd3".to_string()],
            "after cmd2 is deleted cmd1's mutex list must not keep entries pointing at the ghost id"
        );
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn mutex_switch_saves_old_buffer_and_reloads_new() {
        let dir = "PROBE_TEST_DIR_MUTEX_SWITCH";
        let _ = std::fs::remove_dir_all(dir);
        std::fs::create_dir_all(dir).unwrap();
        let mut app = ViqApp::new(dir);
        app.stop_hotkey_thread();
        app.hotkey_manager = None;
        app.cfg = serde_json::json!({"commands": {
            "cmd1": {"enabled": true, "key": "W", "phrase": "前"},
            "cmd2": {"enabled": true, "key": "A", "phrase": "左"},
            "cmd3": {"enabled": true, "key": "D", "phrase": "右"},
        }});
        app.reset_key_edit_bufs();
        // Open cmd1's panel and check cmd2 (buffer only, not yet saved)
        app.mutex_editor_open = Some("cmd1".into());
        app.mutex_edit_for = Some("cmd1".into());
        app.mutex_edit = vec!["cmd2".into()];
        assert!(app.mutex_dirty("cmd1"), "buffer differs from cfg -> must count as dirty");
        // buffer differs from cfg -> must be considered dirty
        app.switch_mutex_editor_to("cmd3");
        assert_eq!(
            app.cfg["commands"]["cmd1"]["mutex"],
            serde_json::json!(["cmd2"]),
            "switching saves: the previous command's buffer must be on disk"
        );
        assert_eq!(
            app.cfg["commands"]["cmd2"]["mutex"],
            serde_json::json!(["cmd1"]),
            "symmetric write: the old command must appear on the other side too"
        );
        // 2) The target changed, so the buffer awaits a reload
        assert_eq!(app.mutex_editor_open.as_deref(), Some("cmd3"));
        assert!(
            app.mutex_edit_for.is_none(),
            "after switching it must be None -> the next frame reloads the buffer for the new command"
        );
        // 3) Switching with no changes writes nothing
        app.mutex_edit_for = Some("cmd3".into());
        app.mutex_edit = app.mutex_set_of("cmd3");
        assert!(!app.mutex_dirty("cmd3"), "buffer matches cfg -> not dirty");
        app.status = "SENTINEL".to_string();
        app.switch_mutex_editor_to("cmd1");
        assert_eq!(
            app.status, "SENTINEL",
            "switching without changes must not trigger apply_mutex_edit (no disk write, no status refresh)"
        );
        assert!(
            app.mutex_set_of("cmd3").is_empty(),
            "cmd3 has nothing checked; no mutex should appear out of nowhere"
        );
        std::fs::remove_dir_all(dir).ok();
    }
}
