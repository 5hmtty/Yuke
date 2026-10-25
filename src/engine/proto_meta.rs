//! Template freshness: records which inputs produced the template, so startup can decide whether to rebuild it.
//!
//! Input fingerprint = encoder choice (standard / noise-robust) + front-end mean removal + identity-window lookahead + per-command recordings / multi-pronunciation.
//! A successful build writes `embed_proto_v2.meta.json`; once settings or recordings change, the fingerprint stops matching.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

const META_VERSION: u32 = 1;
const META_FILE: &str = "embed_proto_v2.meta.json";

#[derive(Serialize, Deserialize)]
struct Meta {
    version: u32,
    encoder: String,
    frame_mean_norm: bool,
    ident_fwd_sec: f64,
    commands: BTreeMap<String, String>,
}

fn meta_path(model_dir: &str) -> PathBuf {
    Path::new(model_dir).join(META_FILE)
}

fn encoder_choice() -> &'static str {
    if crate::udkws_v2::noise_encoder_on() {
        "noise"
    } else {
        "standard"
    }
}

fn global_ok(m: &Meta) -> bool {
    m.version == META_VERSION
        && m.encoder == encoder_choice()
        && m.frame_mean_norm == crate::udkws_v2::frame_mean_norm()
        && (m.ident_fwd_sec - crate::udkws_v2::ident_fwd_sec_env()).abs() < 1e-9
}

fn load(model_dir: &str) -> Option<Meta> {
    let text = crate::crypto::read_text_auto(&meta_path(model_dir)).ok()?;
    serde_json::from_str(&text).ok()
}

fn save(model_dir: &str, meta: &Meta) -> anyhow::Result<()> {
    let json = serde_json::to_string_pretty(meta)?;
    crate::udkws_v2::write_json_atomic_v2(&meta_path(model_dir), &json)
}

/// Per-command template fingerprint: the recording files (name / size / mtime).
/// Multi-pronunciation is part of the fingerprint: toggling it changes the expected trigger count, so it forces a rebuild.
fn command_hash(model_dir: &str, cmd: &str) -> String {
    let mut s = String::new();
    s.push_str(cmd);
    s.push('\n');
    s.push_str(&format!(
        "multi={:?}\n",
        crate::config::multi_override(model_dir, cmd)
    ));
    let mut recs = crate::recordings::list_cmd(model_dir, cmd);
    recs.sort_by(|a, b| a.file.cmp(&b.file));
    let dir = crate::recordings::record_dir(model_dir);
    for r in recs {
        s.push_str(&r.file);
        s.push('\n');
        if let Ok(md) = std::fs::metadata(dir.join(&r.file)) {
            let mtime = md
                .modified()
                .ok()
                .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                .map(|d| d.as_secs())
                .unwrap_or(0);
            s.push_str(&format!("{} {mtime}\n", md.len()));
        }
    }
    crate::nuclei_cache::md5_hex(s.as_bytes())
}

fn build_meta(model_dir: &str, commands: &[String]) -> Meta {
    let commands = commands
        .iter()
        .map(|c| (c.clone(), command_hash(model_dir, c)))
        .collect();
    Meta {
        version: META_VERSION,
        encoder: encoder_choice().to_string(),
        frame_mean_norm: crate::udkws_v2::frame_mean_norm(),
        ident_fwd_sec: crate::udkws_v2::ident_fwd_sec_env(),
        commands,
    }
}

/// Written after a full template build succeeds (overwrites the previous metadata).
pub fn write_meta(model_dir: &str, commands: &[String]) {
    if let Err(e) = save(model_dir, &build_meta(model_dir, commands)) {
        eprintln!("[templates] WARN failed to write the template metadata: {e}");
    }
}

/// After a single-command build only that command's fingerprint is updated; if global parameters changed, the old metadata is kept and a rebuild happens on restart.
pub fn update_command_meta(model_dir: &str, cmd: &str) {
    let Some(mut m) = load(model_dir) else { return };
    if !global_ok(&m) {
        return;
    }
    m.commands
        .insert(cmd.to_string(), command_hash(model_dir, cmd));
    if let Err(e) = save(model_dir, &m) {
        eprintln!("[templates] WARN failed to write the template metadata: {e}");
    }
}

/// Whether the current inputs differ from those that built the templates (i.e. a rebuild is needed). Always false when there are no recordings.
pub fn needs_rebuild(model_dir: &str, commands: &[String]) -> bool {
    let has_recs = |c: &str| !crate::recordings::list_cmd(model_dir, c).is_empty();
    let proto = Path::new(model_dir).join("embed_proto_v2.json");
    if !proto.exists() {
        return commands.iter().any(|c| has_recs(c));
    }
    let Some(m) = load(model_dir) else {
        return commands.iter().any(|c| has_recs(c));
    };
    if !global_ok(&m) {
        return true;
    }
    commands
        .iter()
        .any(|c| has_recs(c) && m.commands.get(c) != Some(&command_hash(model_dir, c)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn meta_detects_recording_change_and_clears_after_write() {
        let d = std::env::temp_dir().join(format!("yuke_proto_meta_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(d.join("command_recordings")).unwrap();
        let dir = d.to_str().unwrap();
        let file = "cmd1_act_20990101_000000.wav";
        let wav = d.join("command_recordings").join(file);
        std::fs::write(&wav, b"stub").unwrap();
        std::fs::write(d.join("embed_proto_v2.json"), b"{}").unwrap();
        crate::recordings::append(
            dir,
            crate::recordings::CmdRecording {
                cmd: "cmd1".into(),
                file: file.into(),
                kind: crate::recordings::RecKind::Active,
                ts: "20261026_000000".into(),
                dur_s: 1.0,
            },
        )
        .unwrap();

        let cmds = vec!["cmd1".to_string()];
        assert!(needs_rebuild(dir, &cmds), "no metadata -> rebuild needed");
        write_meta(dir, &cmds);
        assert!(!needs_rebuild(dir, &cmds), "after writing metadata it must hit");
        std::fs::write(&wav, b"stub-changed").unwrap();
        assert!(needs_rebuild(dir, &cmds), "recordings changed -> rebuild needed");

        let _ = std::fs::remove_dir_all(&d);
    }
}
