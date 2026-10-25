//! Command configuration: parses commands.json (command -> key / mode / macro / mutex) and gate parameters.
//!
//! Recognition is performed by the v2 engine (pronunciation templates + syllable nuclei). This module takes no
//! part in acoustic scoring; it only decides what to do when a command matches, plus the input level gate.

use anyhow::Context;
use std::collections::HashMap;

pub fn persisted_record_enabled() -> bool {
    std::fs::read_to_string(crate::config_path("settings.json"))
        .ok()
        .and_then(|t| serde_json::from_str::<serde_json::Value>(&t).ok())
        .and_then(|v| v.get("record_enabled").and_then(|x| x.as_bool()))
        .unwrap_or(false)
}

pub fn ensure_default_commands_file(model_dir: &str) -> std::io::Result<bool> {
    let path = std::path::Path::new(model_dir).join("commands.json");
    if path.exists() {
        return Ok(false);
    }
    std::fs::create_dir_all(std::path::Path::new(model_dir))?;
    std::fs::write(&path, "{\n  \"commands\": {}\n}")?;
    Ok(true)
}

pub fn aec_mode() -> u8 {
    std::env::var("VIQ_AEC_MODE")
        .ok()
        .and_then(|v| v.trim().parse::<u8>().ok())
        .map(|m| match m { 1 => 1, 3 => 3, 2 => 3, _ => 0 })
        .unwrap_or(0)
}

pub fn v2_proto_commands(model_dir: &str) -> Vec<String> {
    let p = std::path::Path::new(model_dir).join("embed_proto_v2.json");
    let Ok(t) = crate::crypto::read_text_auto(&p) else {
        return Vec::new();
    };
    let Ok(v) = serde_json::from_str::<serde_json::Value>(&t) else {
        return Vec::new();
    };
    let mut names: Vec<String> = v
        .as_object()
        .map(|o| o.keys().cloned().collect())
        .unwrap_or_default();
    names.sort();
    names
}

/// "Multi-pronunciation" explicit override: `multi_nuclei` in commands.json takes effect when it is a bool;
/// missing or non-boolean -> None = automatic (derived from the per-recording nucleus counts in the templates).
pub fn multi_override(model_dir: &str, cmd: &str) -> Option<bool> {
    let p = std::path::Path::new(model_dir).join("commands.json");
    let txt = std::fs::read_to_string(&p).ok()?;
    let v = serde_json::from_str::<serde_json::Value>(&txt).ok()?;
    v.get("commands")?.get(cmd)?.get("multi_nuclei")?.as_bool()
}

/// Expected nucleus count after the override: explicitly disabling multi-pronunciation -> always 1; otherwise the template-derived value (minimum 1).
pub fn apply_multi_override(exp_from_templates: usize, ov: Option<bool>) -> usize {
    match ov {
        Some(false) => 1,
        _ => exp_from_templates.max(1),
    }
}

/// Decision-path report: written to the status bar and log at startup; answers "why does this command never trigger".
///
/// `enabled`: enabled command names; `v2_cmds`: commands that already have a pronunciation template in `embed_proto_v2.json`;
/// `engine_built`: whether the v2 engine was constructed successfully.
pub fn judge_path_report(
    enabled: &[String],
    v2_cmds: &[String],
    engine_built: bool,
) -> Vec<String> {
    let has_proto = |n: &str| v2_cmds.iter().any(|c| c == n);
    let mut missing: Vec<&str> = enabled
        .iter()
        .map(|s| s.as_str())
        .filter(|n| !has_proto(n))
        .collect();
    missing.sort();
    let mut out = vec![format!(
        "decision path: recognition={} | enabled commands {} | templates cover {}{}",
        if engine_built { "ready" } else { "unavailable" },
        enabled.len(),
        enabled.len() - missing.len(),
        if enabled.is_empty() {
            " -- no commands are enabled, so recognition produces no output"
        } else {
            ""
        }
    )];
    if !engine_built {
        out.push("  ⚠ engine construction failed; recognition is unavailable this run (see the [engine] lines on stderr for the cause)".to_string());
    }
    if !missing.is_empty() {
        out.push(format!(
            "⚠ these enabled commands have no pronunciation template yet (they never trigger, though the level meter still moves) -- record them or regenerate their templates on the Commands tab: {}",
            missing.join(", ")
        ));
    }
    out
}

/// Commands (how a match responds: key + mode)
#[derive(Debug, Clone)]
pub struct Command {
    pub key: String,
    pub mode: String,
}

/// Maximum number of macro actions (guards against absurdly long sequences from a bad configuration)
pub const MACRO_MAX_ACTIONS: usize = 64;
/// Maximum duration of a single wait action (ms)
pub const MACRO_MAX_WAIT_MS: u32 = 10_000;
/// Maximum total macro duration (ms): sum of waits + presses × PRESS duration
pub const MACRO_MAX_TOTAL_MS: u64 = 30_000;
/// Hold duration of a press action (ms)
pub const MACRO_PRESS_MS: u32 = 60;

/// Macro action primitives (four): down hold / up release / press tap (down + 60 ms + up) / wait
#[derive(Debug, Clone, PartialEq)]
pub enum MacroAction {
    Down(String),
    Up(String),
    Press(String),
    Wait(u32),
}

pub const MACRO_REENTRY_IGNORE: &str = "ignore";
pub const MACRO_REENTRY_CONCURRENT: &str = "concurrent";
pub const MACRO_REENTRY_INTERRUPT: &str = "interrupt";

/// Normalize the macro re-entry policy: only the three literals are accepted; a missing field or unknown value -> ignore (backward compatible:
/// old commands.json files without the field behave as before, and a typo is not fatal but silently falls back to the default, like mode).
pub fn normalize_macro_reentry(v: Option<&str>) -> &'static str {
    match v {
        Some(MACRO_REENTRY_CONCURRENT) => MACRO_REENTRY_CONCURRENT,
        Some(MACRO_REENTRY_INTERRUPT) => MACRO_REENTRY_INTERRUPT,
        _ => MACRO_REENTRY_IGNORE,
    }
}

#[derive(Debug, Clone)]
pub struct MacroSpec {
    pub actions: Vec<MacroAction>,
    /// Normalized re-entry policy (produced by normalize_macro_reentry; always one of three states)
    pub reentry: String,
}

fn macro_key_from_obj(
    obj: &serde_json::Map<String, serde_json::Value>,
    idx: usize,
) -> Result<String, String> {
    let k = obj.get("key").and_then(|x| x.as_str()).unwrap_or("").trim();
    if k.is_empty() {
        return Err(format!("macro action {} is missing a key", idx + 1));
    }
    if k.len() > 32 {
        return Err(format!("macro action {} has a key longer than 32 characters", idx + 1));
    }
    Ok(k.to_uppercase())
}

pub fn macro_action_ms(a: &MacroAction) -> u64 {
    match a {
        MacroAction::Wait(ms) => *ms as u64,
        MacroAction::Press(_) => MACRO_PRESS_MS as u64,
        _ => 0,
    }
}

pub fn parse_macro_json(v: &serde_json::Value) -> Result<Vec<MacroAction>, String> {
    let arr = v.as_array().ok_or("macro is not an array")?;
    if arr.len() > MACRO_MAX_ACTIONS {
        return Err(format!("action count {} exceeds the limit {MACRO_MAX_ACTIONS}", arr.len()));
    }
    let mut total_ms = 0u64;
    let mut out = Vec::with_capacity(arr.len());
    for (i, item) in arr.iter().enumerate() {
        let obj = item.as_object().ok_or(format!("item {} is not an object", i + 1))?;
        let action = obj
            .get("action")
            .and_then(|x| x.as_str())
            .ok_or(format!("item {} has no action field", i + 1))?;
        let a = match action {
            "down" => MacroAction::Down(macro_key_from_obj(obj, i)?),
            "up" => MacroAction::Up(macro_key_from_obj(obj, i)?),
            "press" => {
                let k = macro_key_from_obj(obj, i)?;
                total_ms += MACRO_PRESS_MS as u64;
                MacroAction::Press(k)
            }
            "wait" => {
                let ms = obj
                    .get("ms")
                    .and_then(|x| x.as_f64())
                    .ok_or(format!("wait in item {} has no ms (number)", i + 1))?;
                if ms.fract() != 0.0 || ms < 0.0 || ms > MACRO_MAX_WAIT_MS as f64 {
                    return Err(format!(
                        "wait ms={ms} in action {} is invalid (must be an integer 0..={MACRO_MAX_WAIT_MS})",
                        i + 1
                    ));
                }
                total_ms += ms as u64;
                MacroAction::Wait(ms as u32)
            }
            other => {
                return Err(format!(
                    "unknown action {other:?} in action {} (available: down/up/press/wait)",
                    i + 1
                ));
            }
        };
        out.push(a);
    }
    if total_ms > MACRO_MAX_TOTAL_MS {
        return Err(format!("total duration {total_ms} ms exceeds the limit {MACRO_MAX_TOTAL_MS} ms"));
    }
    Ok(out)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MacroLineErr {
    /// Key action without a key name (act = the original action word, e.g. down)
    MissingKey { act: String },
    /// wait argument is not a u32 (empty arg = no milliseconds given)
    BadWait { arg: String },
    /// wait exceeds the limit
    WaitOverLimit { ms: u32, max: u32 },
    /// Unknown action word
    UnknownAction { act: String },
}

fn parse_macro_line(line: &str) -> Result<MacroAction, MacroLineErr> {
    let mut it = line.split_whitespace();
    let act = it
        .next()
        .ok_or(MacroLineErr::UnknownAction { act: String::new() })?;
    let arg = it.collect::<Vec<&str>>().join(" ");
    match act.to_ascii_lowercase().as_str() {
        "down" | "d" | "按下" => {
            if arg.is_empty() {
                return Err(MacroLineErr::MissingKey {
                    act: act.to_string(),
                });
            }
            Ok(MacroAction::Down(arg.to_uppercase()))
        }
        "up" | "u" | "释放" => {
            if arg.is_empty() {
                return Err(MacroLineErr::MissingKey {
                    act: act.to_string(),
                });
            }
            Ok(MacroAction::Up(arg.to_uppercase()))
        }
        "press" | "p" | "点击" => {
            if arg.is_empty() {
                return Err(MacroLineErr::MissingKey {
                    act: act.to_string(),
                });
            }
            Ok(MacroAction::Press(arg.to_uppercase()))
        }
        "wait" | "w" | "等待" => {
            let ms: u32 = arg
                .parse()
                .map_err(|_| MacroLineErr::BadWait { arg: arg.clone() })?;
            if ms > MACRO_MAX_WAIT_MS {
                return Err(MacroLineErr::WaitOverLimit {
                    ms,
                    max: MACRO_MAX_WAIT_MS,
                });
            }
            Ok(MacroAction::Wait(ms))
        }
        other => Err(MacroLineErr::UnknownAction {
            act: other.to_string(),
        }),
    }
}

/// Parse multi-line macro text (UI "import from text"): one action per line; blank lines and `#` comments are skipped.
/// Returns (valid actions, [(1-based line number, error)]).
pub fn parse_macro_text(text: &str) -> (Vec<MacroAction>, Vec<(usize, MacroLineErr)>) {
    let mut ok = Vec::new();
    let mut bad = Vec::new();
    for (i, raw) in text.lines().enumerate() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        match parse_macro_line(line) {
            Ok(a) => ok.push(a),
            Err(e) => bad.push((i + 1, e)),
        }
    }
    (ok, bad)
}

/// Gating parameters (the top-level `gate` section of commands.json).
///
/// The recognition criteria are not here -- v2's activity gate and syllable-nucleus parameters are engine-internal relative values;
/// this only keeps the energy threshold behind the "someone is speaking" indicator.
#[derive(Debug, Clone)]
pub struct GateConfig {
    /// RMS threshold: below this counts as silence (drives the UI and log speaking indicator)
    pub speech_energy_threshold: f32,
}

impl Default for GateConfig {
    fn default() -> Self {
        GateConfig {
            speech_energy_threshold: 0.008,
        }
    }
}

impl GateConfig {
    fn from_value(v: Option<&serde_json::Value>) -> Self {
        let mut g = GateConfig::default();
        let Some(v) = v else { return g };
        if let Some(x) = v.get("speech_energy_threshold").and_then(|x| x.as_f64()) {
            g.speech_energy_threshold = x as f32;
        }
        g
    }
}

#[derive(Debug, Clone)]
pub struct CommandConfig {
    pub commands: HashMap<String, Command>,
    /// Gating and onset-detection parameters (top-level `gate` section; defaults when absent)
    pub gate: GateConfig,
    pub macro_seqs: HashMap<String, MacroSpec>,
    pub mutex_sets: HashMap<String, Vec<String>>,
}

impl CommandConfig {
    /// Load the command configuration ({model_dir}/commands.json; writes an empty command table first when missing).
    pub fn load_for_model(model_dir: &str) -> anyhow::Result<Self> {
        let _ = ensure_default_commands_file(model_dir);
        let cfg_path = std::path::Path::new(model_dir).join("commands.json");
        Self::load(cfg_path.to_str().unwrap_or(""))
    }

    /// Load and parse from a file.
    pub fn load(config_path: &str) -> anyhow::Result<Self> {
        let raw: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(config_path)
                .with_context(|| format!("failed to read the configuration {config_path}"))?,
        )
        .with_context(|| "failed to parse the configuration JSON")?;
        Self::from_values(raw)
    }

    /// Build from in-memory JSON (UI edit buffer -> takes effect immediately, without writing to disk).
    pub fn from_values(raw: serde_json::Value) -> anyhow::Result<Self> {
        let mut commands = HashMap::new();
        let mut macro_seqs: HashMap<String, MacroSpec> = HashMap::new();
        if let Some(cmds) = raw.get("commands").and_then(|v| v.as_object()) {
            for (name, cfg) in cmds {
                if cfg.get("enabled").and_then(|v| v.as_bool()) == Some(false) {
                    continue; // Hidden: kept in the configuration but excluded from recognition
                }
                let key = cfg
                    .get("key")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let mut mode = cfg
                    .get("mode")
                    .and_then(|v| v.as_str())
                    .unwrap_or("tap")
                    .to_string();
                let reentry =
                    normalize_macro_reentry(cfg.get("macro_reentry").and_then(|v| v.as_str()));
                let mut macro_actions: Vec<MacroAction> = Vec::new();
                if mode == "macro" {
                    match cfg.get("macro") {
                        Some(mv) => match parse_macro_json(mv) {
                            Ok(a) if !a.is_empty() => macro_actions = a,
                            Ok(_) => {
                                eprintln!(
                                    "[config][WARN] command {name:?} has an empty macro array -- falling back to tap"
                                );
                                mode = "tap".to_string();
                            }
                            Err(e) => {
                                eprintln!(
                                    "[config][WARN] command {name:?} has an invalid macro configuration ({e}) -- falling back to tap"
                                );
                                mode = "tap".to_string();
                            }
                        },
                        None => {
                            eprintln!(
                                "[config][WARN] command {name:?} has mode=macro but no macro array -- falling back to tap"
                            );
                            mode = "tap".to_string();
                        }
                    }
                }
                commands.insert(name.clone(), Command { key, mode });
                if !macro_actions.is_empty() {
                    macro_seqs.insert(
                        name.clone(),
                        MacroSpec {
                            actions: macro_actions,
                            reentry: reentry.to_string(),
                        },
                    );
                }
            }
        }

        let mut mutex_sets: HashMap<String, Vec<String>> = HashMap::new();
        if let Some(cmds) = raw.get("commands").and_then(|v| v.as_object()) {
            for (name, c) in cmds {
                let Some(arr) = c.get("mutex").and_then(|v| v.as_array()) else {
                    continue;
                };
                let mut ids: Vec<String> = arr
                    .iter()
                    .filter_map(|x| x.as_str())
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty() && s.as_str() != name.as_str())
                    .collect();
                ids.sort();
                ids.dedup();
                if !ids.is_empty() {
                    mutex_sets.insert(name.clone(), ids);
                }
            }
        }

        Ok(CommandConfig {
            commands,
            gate: GateConfig::from_value(raw.get("gate")),
            macro_seqs,
            mutex_sets,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base_cfg() -> serde_json::Value {
        serde_json::json!({
            "commands": {
                "ok": { "enabled": true, "key": "K", "mode": "tap" }
            }
        })
    }

    /// Decision-path report: names enabled commands without a pronunciation template; reports unavailable honestly when the engine was not built.
    #[test]
    fn judge_path_report_names_commands_without_templates() {
        let enabled = vec!["c".to_string(), "1".to_string(), "cmd1".to_string()];
        let proto = vec!["1".to_string(), "c".to_string()];

        let on = judge_path_report(&enabled, &proto, true).join("\n");
        assert!(on.contains("recognition=ready"), "{on}");
        assert!(on.contains("enabled commands 3"), "{on}");
        assert!(on.contains("templates cover 2"), "{on}");
        assert!(on.contains("cmd1"), "should name the command without a template: {on}");
        assert!(!on.contains("1, cmd1"), "1 has a template and must not be named: {on}");
        assert!(on.contains("regenerate their templates"), "{on}");

        let failed = judge_path_report(&enabled, &proto, false).join("\n");
        assert!(failed.contains("recognition=unavailable"), "{failed}");
        assert!(failed.contains("construction failed"), "{failed}");

        let ok = judge_path_report(&enabled[..1], &proto, true).join("\n");
        assert!(!ok.contains("⚠"), "no warning when everything is covered: {ok}");

        let none = judge_path_report(&[], &proto, true).join("\n");
        assert!(none.contains("no commands are enabled"), "{none}");
    }

    #[test]
    fn ensure_default_commands_file_creates_once_and_never_overwrites() {
        let dir = std::env::temp_dir().join(format!("yuke_ensure_cfg_{}", std::process::id()));
        let md = dir.join("model");
        let _ = std::fs::remove_dir_all(&dir);
        // First run: missing -> create it; an empty command table is a legal state
        assert!(ensure_default_commands_file(md.to_str().unwrap()).unwrap());
        let txt = std::fs::read_to_string(md.join("commands.json")).unwrap();
        let v: serde_json::Value = serde_json::from_str(&txt).unwrap();
        assert!(
            v["commands"].as_object().unwrap().is_empty(),
            "the factory file is an empty command set"
        );
        std::fs::write(
            md.join("commands.json"),
            r#"{"commands":{"cmd1":{"phrase":"闪现"}}}"#,
        )
        .unwrap();
        assert!(!ensure_default_commands_file(md.to_str().unwrap()).unwrap());
        let txt2 = std::fs::read_to_string(md.join("commands.json")).unwrap();
        assert!(txt2.contains("闪现"), "existing user data must never be overwritten by the factory file");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Invalid macros are rejected: unknown action, missing key, wait out of range, more than 64 actions, total duration over 30 s
    #[test]
    fn macro_json_rejects_invalid() {
        let mk = |n: usize| {
            serde_json::Value::Array(
                (0..n)
                    .map(|_| serde_json::json!({"action": "wait", "ms": 100}))
                    .collect(),
            )
        };
        // Unknown action name
        let e = parse_macro_json(&serde_json::json!([{"action": "fly", "key": "B"}])).unwrap_err();
        assert!(e.contains("unknown action"), "{e}");
        // Key action without a key
        let e = parse_macro_json(&serde_json::json!([{"action": "down"}])).unwrap_err();
        assert!(e.contains("missing a key"), "{e}");
        // wait without ms / not an integer / negative / over 10 s
        assert!(parse_macro_json(&serde_json::json!([{"action": "wait"}])).is_err());
        assert!(parse_macro_json(&serde_json::json!([{"action": "wait", "ms": 1.5}])).is_err());
        assert!(parse_macro_json(&serde_json::json!([{"action": "wait", "ms": -1}])).is_err());
        let e =
            parse_macro_json(&serde_json::json!([{"action": "wait", "ms": 10001}])).unwrap_err();
        assert!(e.contains("is invalid"), "{e}");
        // More than 64 actions
        let e = parse_macro_json(&mk(65)).unwrap_err();
        assert!(e.contains("exceeds the limit"), "{e}");
        assert!(parse_macro_json(&mk(64)).is_ok(), "exactly 64 actions must be valid");
        // Total duration over 30 s (accumulated waits; a press counts as 60 ms)
        let heavy = serde_json::json!([{"action": "wait", "ms": 10000}, {"action": "wait", "ms": 10000},
            {"action": "wait", "ms": 10000}, {"action": "wait", "ms": 1}]);
        let e = parse_macro_json(&heavy).unwrap_err();
        assert!(e.contains("total duration"), "{e}");
        // press counts toward the total: 500 × 60 ms = exactly 30 s (legal), 501 × 60 ms is over (rejected)
        let press = |n: usize| {
            serde_json::Value::Array(
                (0..n)
                    .map(|_| serde_json::json!({"action": "press", "key": "B"}))
                    .collect(),
            )
        };
        assert!(
            parse_macro_json(&press(64)).is_ok(),
            "64 presses = 3.84 s < 30 s"
        );
        let e = parse_macro_json(
            &serde_json::json!([{"action":"wait","ms":10000},{"action":"wait","ms":10000},
            {"action":"wait","ms":9999},{"action":"press","key":"B"}]),
        )
        .unwrap_err();
        assert!(e.contains("total duration"), "a press of 60 ms must count toward the total: {e}");
        // Non-macro top-level structure
        assert!(parse_macro_json(&serde_json::json!({"action": "wait", "ms": 1})).is_err());
    }

    /// mode == "macro" only takes effect together with a macro array: missing, empty or invalid array ->
    /// WARN and the command falls back to tap (not registered in macro_seqs); valid -> the mode is kept and macro_seqs registers it.
    #[test]
    fn macro_mode_requires_valid_array_else_fallback_tap() {
        let mk_raw = |macro_v: Option<serde_json::Value>| {
            let mut cmd = serde_json::json!({
            "enabled": true, "key": "B", "mode": "macro",
                        });
            match macro_v {
                Some(v) => cmd["macro"] = v,
                None => {}
            }
            let mut raw = base_cfg();
            raw["commands"]["m"] = cmd;
            raw
        };
        let good = serde_json::json!([{"action":"down","key":"B"},{"action":"wait","ms":100},
            {"action":"up","key":"B"}]);
        // Valid: mode stays macro and the actions are registered in macro_seqs
        let cfg = CommandConfig::from_values(mk_raw(Some(good))).unwrap();
        let c = cfg.commands.get("m").unwrap();
        assert_eq!(c.mode, "macro");
        assert_eq!(cfg.macro_seqs["m"].actions.len(), 3);
        // Missing array: falls back to tap, not registered
        let cfg = CommandConfig::from_values(mk_raw(None)).unwrap();
        assert_eq!(cfg.commands["m"].mode, "tap");
        assert!(cfg.macro_seqs.get("m").is_none());
        // Empty array: falls back to tap
        let cfg = CommandConfig::from_values(mk_raw(Some(serde_json::json!([])))).unwrap();
        assert_eq!(cfg.commands["m"].mode, "tap");
        // Invalid (unknown action): falls back to tap
        let cfg = CommandConfig::from_values(mk_raw(Some(
            serde_json::json!([{"action":"boom","key":"B"}]),
        )))
        .unwrap();
        assert_eq!(cfg.commands["m"].mode, "tap");
        assert!(cfg.macro_seqs.get("m").is_none());
        // A tap command carrying a macro array: takes no effect (not registered, mode unchanged)
        let mut raw = base_cfg();
        raw["commands"]["t"] = serde_json::json!({
            "enabled": true, "key": "C", "mode": "tap",
            "macro": [{"action":"down","key":"B"}],         });
        let cfg = CommandConfig::from_values(raw).unwrap();
        assert_eq!(cfg.commands["t"].mode, "tap");
        assert!(cfg.macro_seqs.get("t").is_none());
    }

    #[test]
    fn mutex_sets_parse_sorts_dedups_and_drops_self() {
        let mut raw = base_cfg();
        raw["commands"]["a"] = serde_json::json!({
            "enabled": true, "key": "K1", "mode": "toggle",
                        "mutex": ["b", "b", "", "a", 7],
        });
        raw["commands"]["b"] = serde_json::json!({
        "enabled": true, "key": "K2", "mode": "toggle",
                });
        raw["commands"]["c"] = serde_json::json!({
            "enabled": false, "key": "K3",
                        "mutex": ["a"],
        });
        let cfg = CommandConfig::from_values(raw).unwrap();
        assert_eq!(
            cfg.mutex_sets.get("a").map(Vec::as_slice),
            Some(&["b".to_string()][..]),
            "after dedup and dropping self-refs/empty/non-string entries only b remains"
        );
        assert!(
            cfg.mutex_sets.get("b").is_none(),
            "commands without a mutex field do not enter the table"
        );
        assert_eq!(
            cfg.mutex_sets.get("c").map(Vec::as_slice),
            Some(&["a".to_string()][..]),
            "mutex declarations of disabled commands are still collected"
        );
        // No mutex field anywhere in the file -> empty table (an old commands.json behaves exactly as before)
        let cfg2 = CommandConfig::from_values(base_cfg()).unwrap();
        assert!(cfg2.mutex_sets.is_empty(), "a missing field means zero mutex sets");
    }

    #[test]
    fn macro_reentry_parse_defaults_and_tri_state() {
        let mk_raw = |reentry: Option<&str>| {
            let mut cmd = serde_json::json!({
            "enabled": true, "key": "B", "mode": "macro",
            "macro": [{"action": "press", "key": "B"}],
                        });
            if let Some(r) = reentry {
                cmd["macro_reentry"] = serde_json::Value::String(r.to_string());
            }
            let mut raw = base_cfg();
            raw["commands"]["m"] = cmd;
            raw
        };
        // Missing field (old commands.json) = ignore, backward compatible
        let cfg = CommandConfig::from_values(mk_raw(None)).unwrap();
        assert_eq!(cfg.macro_seqs["m"].reentry, "ignore");
        assert_eq!(cfg.macro_seqs["m"].actions.len(), 1);
        // The three states are registered as-is
        for (raw_v, want) in [
            ("concurrent", "concurrent"),
            ("interrupt", "interrupt"),
            ("ignore", "ignore"),
        ] {
            let cfg = CommandConfig::from_values(mk_raw(Some(raw_v))).unwrap();
            assert_eq!(cfg.macro_seqs["m"].reentry, want, "macro_reentry={raw_v:?}");
        }
        // Invalid values normalize to ignore (not fatal, no fallback to tap)
        let cfg = CommandConfig::from_values(mk_raw(Some("boom"))).unwrap();
        assert_eq!(cfg.macro_seqs["m"].reentry, "ignore");
        assert_eq!(cfg.commands["m"].mode, "macro");
        // The normalize function covers this case directly: None -> ignore
        assert_eq!(normalize_macro_reentry(None), "ignore");
    }

    /// Macro text import validates line by line: valid lines parse, invalid lines come back with line number and reason (without aborting the import)
    #[test]
    fn macro_text_import_reports_bad_lines() {
        let (ok, bad) =
            parse_macro_text("down B\n\n# 注释\nwait 300\nup b\nboom X\nwait 99999\npress ctrl+h");
        assert_eq!(
            ok,
            vec![
                MacroAction::Down("B".into()),
                MacroAction::Wait(300),
                MacroAction::Up("B".into()),
                MacroAction::Press("CTRL+H".into()),
            ]
        );
        assert_eq!(bad.len(), 2, "{bad:?}");
        assert_eq!(bad[0].0, 6, "boom is on line 6 (blank/comment lines also count toward the line number)");
        assert_eq!(bad[0].1, MacroLineErr::UnknownAction { act: "boom".into() });
        assert_eq!(bad[1].0, 7, "wait 99999 is on line 7");
        assert_eq!(
            bad[1].1,
            MacroLineErr::WaitOverLimit {
                ms: 99999,
                max: MACRO_MAX_WAIT_MS
            }
        );
    }

    #[test]
    fn macro_line_err_variants() {
        let err = |s: &str| parse_macro_text(s).1;
        assert_eq!(
            err("down").remove(0).1,
            MacroLineErr::MissingKey { act: "down".into() }
        );
        assert_eq!(
            err("按下").remove(0).1,
            MacroLineErr::MissingKey {
                act: "按下".into()
            }
        );
        assert_eq!(
            err("wait abc").remove(0).1,
            MacroLineErr::BadWait { arg: "abc".into() }
        );
        assert_eq!(
            err("wait").remove(0).1,
            MacroLineErr::BadWait { arg: String::new() }
        );
        assert_eq!(
            err("wait -1").remove(0).1,
            MacroLineErr::BadWait { arg: "-1".into() }
        );
        assert!(err("down B\nwait 100").is_empty(), "valid lines must not produce errors");
    }

    /// Initials: `d/u/p/w` are equivalent to the full names, Chinese aliases are kept; export always uses the initial.
    #[test]
    fn macro_text_accepts_single_letter_abbreviations() {
        let (ok, bad) = parse_macro_text("d B\nu b\np ctrl+h\nw 300\nD SHIFT\nP F1\nW 0");
        assert!(bad.is_empty(), "the initial must not produce invalid lines: {bad:?}");
        assert_eq!(
            ok,
            vec![
                MacroAction::Down("B".into()),
                MacroAction::Up("B".into()),
                MacroAction::Press("CTRL+H".into()),
                MacroAction::Wait(300),
                MacroAction::Down("SHIFT".into()),
                MacroAction::Press("F1".into()),
                MacroAction::Wait(0),
            ]
        );
        let (ok2, bad2) = parse_macro_text("down B\nwait 50\n按下 C\n等待 10");
        assert!(bad2.is_empty(), "full names and aliases must keep working: {bad2:?}");
        assert_eq!(ok2.len(), 4);
        assert_eq!(
            parse_macro_text("w").1.remove(0).1,
            MacroLineErr::BadWait { arg: String::new() }
        );
        assert_eq!(
            parse_macro_text("d").1.remove(0).1,
            MacroLineErr::MissingKey { act: "d".into() }
        );
    }

    #[test]
    fn multi_override_applies_single_nucleus() {
        assert_eq!(apply_multi_override(2, Some(false)), 1, "unchecked -> force single nucleus");
        assert_eq!(apply_multi_override(2, None), 2, "auto -> use the template value");
        assert_eq!(apply_multi_override(2, Some(true)), 2, "checked -> use the template value");
        assert_eq!(
            apply_multi_override(1, Some(true)),
            1,
            "the template already has 1 nucleus -> stays 1"
        );
        assert_eq!(apply_multi_override(0, None), 1, "the lower bound is 1");
    }

    #[test]
    fn multi_override_reads_commands_json() {
        let d = std::env::temp_dir().join(format!("viq_multi_override_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        std::fs::write(
            d.join("commands.json"),
            r#"{"commands":{"c1":{"multi_nuclei":false},"c2":{"multi_nuclei":true},"c3":{},"c4":{"multi_nuclei":"no"}}}"#,
        )
        .unwrap();
        let dir = d.to_str().unwrap();
        assert_eq!(multi_override(dir, "c1"), Some(false));
        assert_eq!(multi_override(dir, "c2"), Some(true));
        assert_eq!(multi_override(dir, "c3"), None);
        assert_eq!(multi_override(dir, "c4"), None, "a non-bool value does not count as an override");
        assert_eq!(multi_override(dir, "nope"), None);
        std::fs::remove_dir_all(&d).unwrap();
    }
}
