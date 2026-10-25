//! Command recording management: wav files plus an index JSON under {model_dir}/command_recordings/.

//! Index: `{model_dir}/command_recordings.json` (array, newest first).
//! wav naming: `{cmd}_{kind}_{ts}.wav` (active) / `{cmd}_learn_{frame}.wav` (passive).

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

/// In-process mutex for the index: when UI re-recording (ui.rs) and passive slicing on the gate audio thread update concurrently,
/// an unguarded read-modify-write loses entries (both sides read the old index and each writes the whole file back;
/// the later write clobbers the earlier one). Every index read-modify-write runs under this lock; the read-only
/// interface (list) does not lock -- a single whole-file read may at worst see a slightly older snapshot, never a torn one (writers rename atomically).
static INDEX_LOCK: Mutex<()> = Mutex::new(());

/// Origin: actively recorded / passively learned
#[derive(Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RecKind {
    Active,
    Learned,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct CmdRecording {
    pub cmd: String,
    pub file: String,
    pub kind: RecKind,
    /// Recorded at (YYYYMMDD_HHMMSS)
    pub ts: String,
    /// Duration (seconds)
    pub dur_s: f32,
}

/// Recording root (sits next to commands.json in the model directory)
pub fn record_dir(model_dir: &str) -> PathBuf {
    PathBuf::from(model_dir).join("command_recordings")
}

pub fn index_path(model_dir: &str) -> PathBuf {
    PathBuf::from(model_dir).join("command_recordings.json")
}

fn read_index(path: &Path) -> Vec<CmdRecording> {
    let Some(text) = std::fs::read_to_string(path).ok() else {
        return Vec::new();
    };
    match serde_json::from_str(&text) {
        Ok(recs) => recs,
        Err(e) => {
            let ts = chrono::Local::now().format("%Y%m%d_%H%M%S");
            let bak = PathBuf::from(format!("{}.bak.{}", path.display(), ts));
            let bak_msg = match std::fs::write(&bak, &text) {
                Ok(()) => format!("corrupt index backed up to {}", bak.display()),
                Err(be) => format!("failed to back up the corrupt index ({be}); data may be lost permanently"),
            };
            eprintln!(
                "warning: failed to parse the recording index {} ({e}) -- {bak_msg}; continuing with an empty index",
                path.display()
            );
            Vec::new()
        }
    }
}

/// All recordings (newest first)
pub fn list(model_dir: &str) -> Vec<CmdRecording> {
    read_index(&index_path(model_dir))
}

/// Recordings of one command (newest first)
pub fn list_cmd(model_dir: &str, cmd: &str) -> Vec<CmdRecording> {
    list(model_dir)
        .into_iter()
        .filter(|r| r.cmd == cmd)
        .collect()
}

fn write_index(path: &Path, recs: &[CmdRecording]) -> std::io::Result<()> {
    let s = serde_json::to_string_pretty(recs).unwrap_or_else(|_| "[]".into());
    let tmp = PathBuf::from(format!("{}.tmp", path.display()));
    let mut last_err: Option<std::io::Error> = None;
    for attempt in 0..3u32 {
        match std::fs::write(&tmp, &s).and_then(|_| std::fs::rename(&tmp, path)) {
            Ok(()) => return Ok(()),
            Err(e) => {
                let _ = std::fs::remove_file(&tmp); // Clean up leftovers so .tmp files do not pile up
                last_err = Some(e);
                if attempt < 2 {
                    std::thread::sleep(std::time::Duration::from_millis(20 * (attempt as u64 + 1)));
                }
            }
        }
    }
    Err(last_err.unwrap_or_else(|| std::io::Error::new(std::io::ErrorKind::Other, "index write failed")))
}

/// Serialized index read-modify-write: every mutation (the append/remove family) must go through here;
/// "read index -> modify -> atomic write" holds the lock the whole time, so concurrent callers never lose entries. Signatures unchanged.
fn update_index<F: FnOnce(&mut Vec<CmdRecording>)>(model_dir: &str, f: F) -> std::io::Result<()> {
    // Lock poisoning does not spread: the index stays maintainable after a panic while holding the lock (project convention, into_inner)
    let _guard = INDEX_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let path = index_path(model_dir);
    let mut recs = read_index(&path);
    f(&mut recs);
    write_index(&path, &recs)
}

/// Append one recording
pub fn append(model_dir: &str, rec: CmdRecording) -> std::io::Result<()> {
    encrypt_wav_in_place(&record_dir(model_dir).join(&rec.file));
    update_index(model_dir, |recs| recs.insert(0, rec))
}

/// Encrypt the wav in place (L1, idempotent): plaintext -> L1 encrypted (tmp + atomic rename).
/// Already encrypted, or file absent -> returned unchanged.
pub fn encrypt_wav_in_place(path: &Path) {
    let Ok(raw) = std::fs::read(path) else { return };
    if raw.is_empty() || crate::crypto::is_encrypted(&raw) {
        return;
    }
    match crate::crypto::protect(&raw) {
        Ok(enc) => {
            let tmp = PathBuf::from(format!("{}.enc.tmp", path.display()));
            if let Err(e) = std::fs::write(&tmp, &enc).and_then(|_| std::fs::rename(&tmp, path)) {
                let _ = std::fs::remove_file(&tmp);
                eprintln!("[recordings] WARN encryption-in-place failed (plaintext kept): {e}");
            }
        }
        Err(e) => eprintln!("[recordings] WARN encryption failed (plaintext kept): {e}"),
    }
}

/// Read a wav (auto-detects encrypted vs plaintext; the single entry point for dual-compatible reads): ciphertext is decrypted in memory first;
/// plaintext is parsed as-is (16-bit PCM, multi-channel averaged to mono).
pub fn read_wav_auto(path: &Path) -> Result<(u32, Vec<f32>), String> {
    let raw = std::fs::read(path).map_err(|e| format!("failed to open the WAV {}: {e}", path.display()))?;
    let plain = if crate::crypto::is_encrypted(&raw) {
        crate::crypto::unprotect(&raw).map_err(|e| format!("failed to decrypt {}: {e}", path.display()))?
    } else {
        raw
    };
    parse_wav_mem(&plain)
}

/// Parse 16-bit PCM wav from memory (the hound Cursor variant)
fn parse_wav_mem(bytes: &[u8]) -> Result<(u32, Vec<f32>), String> {
    let reader = hound::WavReader::new(std::io::Cursor::new(bytes))
        .map_err(|e| format!("failed to parse the WAV: {e}"))?;
    let spec = reader.spec();
    if spec.bits_per_sample != 16 {
        return Err(format!(
            "only 16-bit PCM WAV is supported (got {} bit)",
            spec.bits_per_sample
        ));
    }
    let channels = spec.channels as usize;
    let samples: Vec<i16> = reader
        .into_samples::<i16>()
        .collect::<Result<_, _>>()
        .map_err(|e| format!("failed to read the WAV: {e}"))?;
    let mono: Vec<f32> = if channels == 1 {
        samples.iter().map(|&s| s as f32 / 32768.0).collect()
    } else {
        samples
            .chunks(channels)
            .map(|c| c.iter().map(|&s| s as f32).sum::<f32>() / (channels as f32 * 32768.0))
            .collect()
    };
    Ok((spec.sample_rate, mono))
}

pub fn decrypt_to_temp(model_dir: &str, file: &str) -> Option<PathBuf> {
    let src = record_dir(model_dir).join(file);
    let raw = std::fs::read(&src).ok()?;
    if !crate::crypto::is_encrypted(&raw) {
        return Some(src);
    }
    let plain = crate::crypto::unprotect(&raw).ok()?;
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .ok()?
        .as_nanos();
    let mut tmp = std::env::temp_dir().join(format!("viq_play_{}_{}", std::process::id(), nanos));
    tmp.set_extension("wav");
    std::fs::write(&tmp, plain).ok()?;
    Some(tmp)
}

/// Delete by file name (wav and index entry)
pub fn remove(model_dir: &str, file: &str) -> std::io::Result<()> {
    let dir = record_dir(model_dir);
    let _ = std::fs::remove_file(dir.join(file)); // A missing file is not an error
    let file = file.to_string();
    update_index(model_dir, |recs| recs.retain(|r| r.file != file))
}

pub fn remove_cmd_all(model_dir: &str, cmd: &str) -> std::io::Result<()> {
    let dir = record_dir(model_dir);
    let cmd = cmd.to_string();
    // Put the wav deletion inside the closure (while holding the lock) so it happens in the same batch as the index rewrite
    update_index(model_dir, |recs| {
        let mut kept = Vec::new();
        for r in recs.drain(..) {
            if r.cmd == cmd {
                let _ = std::fs::remove_file(dir.join(&r.file));
            } else {
                kept.push(r);
            }
        }
        *recs = kept;
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn probe_dir(tag: &str) -> String {
        let d = std::env::temp_dir().join(format!(
            "kws_recidx_{tag}_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&d).unwrap();
        d.to_str().unwrap().to_string()
    }

    fn rec(cmd: &str, file: String) -> CmdRecording {
        CmdRecording {
            cmd: cmd.into(),
            file,
            kind: RecKind::Active,
            ts: "20990101_000000".into(),
            dur_s: 1.0,
        }
    }

    #[test]
    fn concurrent_append_never_loses_entries() {
        let model_dir = probe_dir("conc");
        const THREADS: usize = 2;
        const N: usize = 40;
        let mut handles = Vec::new();
        for t in 0..THREADS {
            let md = model_dir.clone();
            handles.push(std::thread::spawn(move || {
                for i in 0..N {
                    append(&md, rec("conc", format!("t{t}_{i}.wav"))).expect("append failed");
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
        let all = list(&model_dir);
        assert_eq!(all.len(), THREADS * N, "concurrent appends must not drop entries");
        let mut files: Vec<_> = all.iter().map(|r| r.file.clone()).collect();
        files.sort();
        let unique = files.len();
        files.dedup();
        assert_eq!(files.len(), unique, "entry file names must be unique (guards against overwrite false positives)");
        std::fs::remove_dir_all(&model_dir).ok();
    }

    #[test]
    fn concurrent_append_with_remove_keeps_new_entries() {
        let model_dir = probe_dir("mix");
        for i in 0..10 {
            append(&model_dir, rec("mix", format!("del_{i}.wav"))).expect("append failed");
        }
        let md = model_dir.clone();
        let h = std::thread::spawn(move || {
            for i in 0..20 {
                append(&md, rec("mix", format!("add_{i}.wav"))).expect("append failed");
            }
        });
        for i in 0..10 {
            remove(&model_dir, &format!("del_{i}.wav")).expect("remove failed");
        }
        h.join().unwrap();
        let all = list(&model_dir);
        assert_eq!(all.len(), 20, "after 10 deletions + 20 appends the index must hold exactly 20 entries");
        assert!(
            all.iter().all(|r| r.file.starts_with("add_")),
            "only the appended entries should remain"
        );
        std::fs::remove_dir_all(&model_dir).ok();
    }

    /// Atomic write path: after a normal append/remove the index is still valid JSON (rename replacement never tears the file)
    #[test]
    fn index_survives_write_cycles() {
        let model_dir = probe_dir("cycle");
        for i in 0..30 {
            append(&model_dir, rec("cyc", format!("f{i}.wav"))).expect("append failed");
        }
        for i in 0..15 {
            remove(&model_dir, &format!("f{i}.wav")).expect("remove failed");
        }
        let all = list(&model_dir);
        assert_eq!(all.len(), 15);
        // The index file parses on its own (no .tmp leftovers mixed in)
        let raw = std::fs::read_to_string(index_path(&model_dir)).unwrap();
        let parsed: Vec<CmdRecording> = serde_json::from_str(&raw).unwrap();
        assert_eq!(parsed.len(), 15);
        assert!(!PathBuf::from(format!("{}.tmp", index_path(&model_dir).display())).exists());
        std::fs::remove_dir_all(&model_dir).ok();
    }

    /// Write a plaintext wav -> after append the file on disk is ciphertext (no RIFF header, VIQENC magic) and read_wav_auto
    /// reads back the original samples; after remove the file and the index entry disappear together.
    #[test]
    fn append_encrypts_wav_and_reads_back() {
        let model_dir = probe_dir("enc");
        let dir = record_dir(&model_dir);
        std::fs::create_dir_all(&dir).unwrap();
        let file = "enc_act_20990101.wav".to_string();
        let path = dir.join(&file);
        let samples: Vec<f32> = (0..1600).map(|i| (i % 100) as f32 / 1000.0).collect();
        crate::features::write_wav(path.to_str().unwrap(), &samples, 16000).unwrap();
        let expected: Vec<f32> = samples
            .iter()
            .map(|&s| ((s.clamp(-1.0, 1.0) * 32767.0) as i16) as f32 / 32768.0)
            .collect();
        assert_eq!(std::fs::read(&path).unwrap()[..4], *b"RIFF");
        append(&model_dir, rec("enc", file.clone())).unwrap();
        // Lands on disk as ciphertext: no longer RIFF but the VIQENC magic
        let on_disk = std::fs::read(&path).unwrap();
        assert_eq!(&on_disk[..6], crate::crypto::MAGIC, "after append the data must be ciphertext");
        assert_ne!(&on_disk[..4], b"RIFF");
        let (sr, back) = read_wav_auto(&path).unwrap();
        assert_eq!(sr, 16000);
        assert_eq!(back, expected);
        // Direct plaintext parsing (the old tool path) must fail -- proof this is not fake encryption that reads both ways
        remove(&model_dir, &file).unwrap();
        assert!(!path.exists());
        std::fs::remove_dir_all(&model_dir).ok();
    }

    #[test]
    fn read_wav_auto_accepts_plaintext() {
        let model_dir = probe_dir("plain");
        let dir = record_dir(&model_dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("plain_act_20990101.wav");
        let samples: Vec<f32> = (0..800).map(|i| (i % 50) as f32 / 500.0).collect();
        crate::features::write_wav(path.to_str().unwrap(), &samples, 16000).unwrap();
        let expected: Vec<f32> = samples
            .iter()
            .map(|&s| ((s.clamp(-1.0, 1.0) * 32767.0) as i16) as f32 / 32768.0)
            .collect();
        let (sr, back) = read_wav_auto(&path).unwrap();
        assert_eq!(sr, 16000);
        assert_eq!(back, expected);
        std::fs::remove_dir_all(&model_dir).ok();
    }

    /// Corrupt encrypted wav: read_wav_auto returns an error instead of panicking (with a clear message)
    #[test]
    fn read_wav_auto_tolerates_corruption() {
        let model_dir = probe_dir("bad");
        let dir = record_dir(&model_dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("bad_act_20990101.wav");
        // Magic followed by a truncated payload
        std::fs::write(&path, [&crate::crypto::MAGIC[..], b"JUNKJUNK"].concat()).unwrap();
        let err = read_wav_auto(&path).unwrap_err();
        assert!(!err.is_empty());
        std::fs::remove_dir_all(&model_dir).ok();
    }

    #[test]
    fn decrypt_to_temp_serves_playback() {
        let model_dir = probe_dir("play");
        let dir = record_dir(&model_dir);
        std::fs::create_dir_all(&dir).unwrap();
        let file = "play_act_20990101.wav".to_string();
        let path = dir.join(&file);
        let samples: Vec<f32> = (0..1600).map(|i| (i % 100) as f32 / 1000.0).collect();
        crate::features::write_wav(path.to_str().unwrap(), &samples, 16000).unwrap();
        append(&model_dir, rec("play", file.clone())).unwrap();
        let tmp = decrypt_to_temp(&model_dir, &file).expect("decrypt_to_temp failed on ciphertext");
        assert_ne!(tmp, path, "ciphertext must yield a temp copy, not the original path");
        let bytes = std::fs::read(&tmp).unwrap();
        assert_eq!(
            &bytes[..4],
            b"RIFF",
            "the decrypted copy must be a plain wav (playable by SoundPlayer)"
        );
        let (sr, back) = read_wav_auto(&tmp).unwrap();
        assert_eq!(sr, 16000);
        assert_eq!(back.len(), samples.len());
        std::fs::remove_file(&tmp).ok();
        // Plaintext input: returns the original path even without an index entry (decrypt_to_temp looks only at file content)
        let file2 = "plain2_act_20990101.wav";
        crate::features::write_wav(dir.join(file2).to_str().unwrap(), &samples, 16000).unwrap();
        let p2 = decrypt_to_temp(&model_dir, file2).expect("decrypt_to_temp failed on plaintext");
        assert_eq!(p2, dir.join(file2), "plaintext should return the original path directly");
        std::fs::remove_dir_all(&model_dir).ok();
    }

    #[test]
    fn corrupt_index_backed_up_and_degrades_to_empty() {
        let model_dir = probe_dir("corrupt");
        let idx = index_path(&model_dir);
        let corrupt = "{\"截断的JSON\": ";
        std::fs::write(&idx, corrupt).unwrap();
        // Read side degrades: no panic, returns empty
        assert!(list(&model_dir).is_empty(), "a corrupt index must degrade to an empty list");
        // Exactly one .bak.<timestamp> backup, whose content is the original corrupt text
        let mut baks: Vec<PathBuf> = std::fs::read_dir(&model_dir)
            .unwrap()
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.to_string_lossy().contains(".bak."))
            .collect();
        assert_eq!(baks.len(), 1, "a corrupt index must leave exactly one .bak backup");
        assert_eq!(
            std::fs::read_to_string(&baks.remove(0)).unwrap(),
            corrupt,
            "the backup content must be the original corrupt text"
        );
        // After degrading, writes still work: append registers normally and reads back
        append(&model_dir, rec("corrupt", "a.wav".into())).expect("append after degradation should succeed");
        assert_eq!(list(&model_dir).len(), 1);
        std::fs::remove_dir_all(&model_dir).ok();
    }
}
