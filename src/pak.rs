//! Container format (plaintext index, plaintext payload -- this is the open-source distribution):
//! ```text
//! b"VIQPAK" | u8 ver=1 | u32 count (le)
//! x count  : u16 name_len(le) | name(utf-8, '/'-separated) | u64 offset(le) | u64 len(le)
//! payload  : files stored verbatim in plaintext
//! ```

//! Lookup discipline: **pack first, loose files as fallback**. A dev machine or the repository has no yuke.pak, so
//! everything goes through the loose path (the CLI toolchain is unaffected); a production package ships only the pack,
//! so no loose model files exist. Lookups are keyed by entry name (`model/<file>`, `assets/<file>`), which is
//! determined by the input directory names passed to pak-pack.

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use anyhow::Context;

pub const PACK_NAME: &str = "yuke.pak";
const MAGIC: &[u8; 6] = b"VIQPAK";
const VERSION: u8 = 1;

/// Directory prefix for model role files inside the pack (pak-pack input directory naming convention)
pub const MODEL_PREFIX: &str = "model";
/// Directory prefix for veto-layer models inside the pack
pub const ASSETS_PREFIX: &str = "assets";

struct Pack {
    raw: Vec<u8>,
    /// (name, offset, len), sorted by name (deterministic layout)
    entries: Vec<(String, u64, u64)>,
}

fn pack_path() -> Option<PathBuf> {
    let p = crate::exe_dir().map(|d| d.join(PACK_NAME))?;
    p.is_file().then_some(p)
}

static PACK: OnceLock<Option<Pack>> = OnceLock::new();

fn get() -> &'static Option<Pack> {
    PACK.get_or_init(|| {
        let path = pack_path()?;
        let raw = match std::fs::read(&path) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("[pak] failed to read {PACK_NAME} (continuing without a pack): {e}");
                return None;
            }
        };
        match parse(&raw) {
            Ok(p) => {
                eprintln!("[pak] loaded {} ({} entries)", path.display(), p.entries.len());
                Some(p)
            }
            Err(e) => {
                eprintln!("[pak] failed to parse {PACK_NAME} (continuing without a pack, using loose files): {e}");
                None
            }
        }
    })
}

fn parse(raw: &[u8]) -> anyhow::Result<Pack> {
    anyhow::ensure!(raw.len() >= 11 && &raw[..6] == MAGIC, "bad magic");
    anyhow::ensure!(raw[6] == VERSION, "unsupported container version {}", raw[6]);
    let count = u32::from_le_bytes(raw[7..11].try_into()?) as usize;
    // Hardening: cap `count` at one entry per 2+1+16 bytes so a corrupt header cannot blow up into OOM;
    // use checked_add for off+len so u64 wraparound cannot bypass the bounds check
    anyhow::ensure!(
        count <= raw.len().saturating_sub(11) / 19,
        "invalid entry count: {count}"
    );
    let mut i = 11usize;
    let mut entries = Vec::with_capacity(count);
    for _ in 0..count {
        anyhow::ensure!(i + 2 <= raw.len(), "truncated index (name_len)");
        let nlen = u16::from_le_bytes(raw[i..i + 2].try_into()?) as usize;
        i += 2;
        anyhow::ensure!(i + nlen + 16 <= raw.len(), "truncated index (entry)");
        let name = String::from_utf8(raw[i..i + nlen].to_vec())?;
        i += nlen;
        let off = u64::from_le_bytes(raw[i..i + 8].try_into()?);
        i += 8;
        let len = u64::from_le_bytes(raw[i..i + 8].try_into()?);
        i += 8;
        let end = off.checked_add(len).context("payload length overflow")?;
        anyhow::ensure!(end <= raw.len() as u64, "payload out of range: {name}");
        entries.push((name, off, len));
    }
    Ok(Pack {
        raw: raw.to_vec(),
        entries,
    })
}

/// A parseable yuke.pak exists in the install directory (true = production packaged form)
pub fn available() -> bool {
    get().is_some()
}

/// Pack file path (used for the encoder identity stamp: models inside the pack key the cache on the whole pack's md5)
pub fn path() -> Option<PathBuf> {
    pack_path()
}

/// Raw payload (plaintext for the open-source distribution)
pub fn read_raw(name: &str) -> Option<Vec<u8>> {
    let p = get().as_ref()?;
    let (_, off, len) = p.entries.iter().find(|(n, _, _)| n == name)?;
    let off = *off as usize;
    let len = *len as usize;
    Some(p.raw[off..off + len].to_vec())
}

/// Whether the pack contains an entry (without reading its payload)
pub fn contains(name: &str) -> bool {
    get().as_ref().is_some_and(|p| p.entries.iter().any(|(n, _, _)| n == name))
}

/// Read entry bytes (plaintext)
pub fn read(name: &str) -> Option<Vec<u8>> {
    read_raw(name)
}

/// Resolve a model role file inside the pack by filename priority:
/// exact {name}.onnx -> {name}-*.onnx (int8 preferred).
/// Returns (entry name, raw payload).
/// Standard versus noise-robust encoder is chosen by whether the name contains `noise` (pass false for other models).
pub fn find_model_raw_pref(name: &str, want_noise: bool) -> Option<(String, Vec<u8>)> {
    let p = get().as_ref()?;
    let prefix = format!("{MODEL_PREFIX}/");
    let names: Vec<&str> = p
        .entries
        .iter()
        .filter(|(n, _, _)| n.starts_with(&prefix))
        .map(|(n, _, _)| &n[prefix.len()..])
        .collect();
    let hit = best_model_entry(&names, name, want_noise)?;
    let full = format!("{prefix}{hit}");
    Some((full.clone(), read_raw(&full)?))
}

/// Priority as a pure function (applies equally to pack entries and loose directory names, so it is unit-testable):
/// exact plaintext -> plaintext variant (int8 preferred).
fn best_model_entry<'a>(names: &[&'a str], name: &str, want_noise: bool) -> Option<&'a str> {
    let noise_ok = |n: &str| n.contains("noise") == want_noise;
    let exact_plain = format!("{name}.onnx");
    if let Some(n) = names
        .iter()
        .copied()
        .find(|n| *n == exact_plain && noise_ok(n))
    {
        return Some(n);
    }
    let pref = format!("{name}-");
    let pick = |names: &[&'a str], suffix: &str| -> Option<&'a str> {
        let mut best: Option<&'a str> = None;
        for &n in names {
            if n.starts_with(&pref) && n.ends_with(suffix) && noise_ok(n) {
                let better = match best {
                    None => true,
                    Some(b) => n.contains(".int8.") && !b.contains(".int8."),
                };
                if better {
                    best = Some(n);
                }
            }
        }
        best
    };
    pick(names, ".onnx")
}

// Packing (CLI pak-pack)

/// Pack several directories into yuke.pak. Entry names are `<dir>/<relative path>` ('/'-separated).
/// Returns (entry count, total payload bytes).
pub fn build(out: &Path, dirs: &[PathBuf]) -> anyhow::Result<(usize, u64)> {
    let mut items: Vec<(String, Vec<u8>)> = Vec::new();
    for dir in dirs {
        let dir = abs_norm(dir);
        let Some(dir_name) = dir.file_name().and_then(|s| s.to_str()) else {
            anyhow::bail!("failed to resolve the directory name: {}", dir.display());
        };
        let mut stack = vec![(dir.clone(), String::new())];
        while let Some((d, rel)) = stack.pop() {
            for e in std::fs::read_dir(&d)?.flatten() {
                let p = e.path();
                let fname = e.file_name().to_string_lossy().to_string();
                let child_rel = if rel.is_empty() {
                    fname
                } else {
                    format!("{rel}/{fname}")
                };
                if p.is_dir() {
                    stack.push((p, child_rel));
                } else {
                    let bytes =
                        std::fs::read(&p).with_context(|| format!("failed to read: {}", p.display()))?;
                    anyhow::ensure!(
                        !crate::crypto::is_encrypted(&bytes),
                        "encrypted payload not supported: {}",
                        p.display()
                    );
                    items.push((format!("{dir_name}/{child_rel}"), bytes));
                }
            }
        }
    }
    items.sort_by(|a, b| a.0.cmp(&b.0));
    // Layout: header + index + payloads in order
    let mut header: Vec<u8> = Vec::new();
    header.extend_from_slice(MAGIC);
    header.push(VERSION);
    header.extend_from_slice(&(items.len() as u32).to_le_bytes());
    let index_len: usize = items.iter().map(|(n, _)| 2 + n.len() + 16).sum();
    let mut payload_off = (header.len() + index_len) as u64;
    let mut index: Vec<u8> = Vec::new();
    let mut body: Vec<u8> = Vec::new();
    for (name, payload) in &items {
        let len = payload.len() as u64;
        index.extend_from_slice(&(name.len() as u16).to_le_bytes());
        index.extend_from_slice(name.as_bytes());
        index.extend_from_slice(&payload_off.to_le_bytes());
        index.extend_from_slice(&len.to_le_bytes());
        body.extend_from_slice(payload);
        payload_off += len;
    }
    header.extend_from_slice(&index);
    header.extend_from_slice(&body);
    if let Some(parent) = out.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    std::fs::write(out, &header).with_context(|| format!("failed to write: {}", out.display()))?;
    let total: u64 = items.iter().map(|(_, p)| p.len() as u64).sum();
    Ok((items.len(), total))
}

/// List pack entries (CLI pak-list)
pub fn print_list() -> anyhow::Result<()> {
    let p = get().as_ref().context("no usable yuke.pak in the exe directory")?;
    for (n, _, len) in &p.entries {
        println!("{len:>10}  {n}");
    }
    Ok(())
}

fn abs_norm(p: &Path) -> PathBuf {
    crate::abs_norm(p)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Container round-trip: write two entries (plaintext into the pack) and read the same bytes back after parse.
    /// Tests the parse/read layer directly (the process-wide OnceLock points at the exe directory's yuke.pak, and the test
    /// process has no installed pack -- available() must be false, so this exercises the standalone parsing path).
    #[test]
    fn container_roundtrip_plaintext() {
        let dir = std::env::temp_dir().join(format!("viqpak_test_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let model = dir.join("model");
        let assets = dir.join("assets");
        std::fs::create_dir_all(&model).unwrap();
        std::fs::create_dir_all(&assets).unwrap();
        std::fs::write(model.join("labels.txt"), "音素 1\n测试 2\n").unwrap();
        std::fs::write(assets.join("x.bin"), vec![7u8; 1000]).unwrap();
        let out = dir.join("out.pak");
        let (n, total) = build(&out, &[model, assets]).unwrap();
        assert_eq!(n, 2);
        assert!(total >= 1000);

        let raw = std::fs::read(&out).unwrap();
        let pack = parse(&raw).unwrap();
        assert_eq!(pack.entries.len(), 2);
        // Plaintext entries go in verbatim (the open-source distribution does not encrypt)
        let (_, off, len) = pack
            .entries
            .iter()
            .find(|(n, _, _)| n == "model/labels.txt")
            .unwrap();
        assert_eq!(
            &raw[*off as usize..(*off + *len) as usize],
            "音素 1\n测试 2\n".as_bytes()
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Model-name priority scan (pure function): exact plaintext > plaintext int8 variant.
    #[test]
    fn best_model_entry_precedence() {
        fn mk<'a>(names: &[&'a str]) -> &'a str {
            best_model_entry(names, "encoder", false).unwrap()
        }
        assert_eq!(mk(&["encoder.onnx", "encoder-a.int8.onnx"]), "encoder.onnx");
        assert_eq!(
            mk(&["encoder-a.onnx", "encoder-b.int8.onnx"]),
            "encoder-b.int8.onnx"
        );
        assert_eq!(mk(&["encoder-a.onnx", "encoder-b.onnx"]), "encoder-a.onnx");
        assert!(best_model_entry(&["decoder.onnx"], "encoder", false).is_none());
    }

    #[test]
    fn best_model_entry_noise_filter() {
        let names = [
            "encoder-v2-chunk-8.int8.onnx",
            "encoder-v2-noise.int8.onnx",
        ];
        assert_eq!(
            best_model_entry(&names, "encoder-v2", false).unwrap(),
            "encoder-v2-chunk-8.int8.onnx"
        );
        assert_eq!(
            best_model_entry(&names, "encoder-v2", true).unwrap(),
            "encoder-v2-noise.int8.onnx"
        );
    }
}
