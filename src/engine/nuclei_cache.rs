//! **Problem**: `Engine::from_model_dir` re-encoded every recording in command_recordings on every startup
//! to build the syllable-nucleus templates (80 ms row grid); with hundreds of recordings that is seconds
//! of model inference -- the bulk of startup time.

//! Wiring contract: this module owns all cache logic; udkws_v2.rs only needs three pub(crate) symbols
//! (classify_recording, build_nuclei_template_for_cmd, write_json_atomic_v2),
//! and from_model_dir has exactly one wiring point, the `load_or_build` call.

use anyhow::Context;
use std::collections::{BTreeMap, HashMap};

const CACHE_VERSION: u32 = 6; // 6: template row times unified on the 80 ms grid (rows8)

fn cache_version() -> u32 {
    CACHE_VERSION
        + if crate::udkws_v2::frame_mean_norm() {
            100
        } else {
            0
        }
        + crate::udkws_v2::ident_win_fingerprint()
}

// md5 (RFC 1321, self-contained, zero dependencies -- used as the invalidation-key hash)

const MD5_S: [u32; 64] = [
    7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 5, 9, 14, 20, 5, 9, 14, 20, 5, 9,
    14, 20, 5, 9, 14, 20, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 6, 10, 15,
    21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21,
];
const MD5_K: [u32; 64] = [
    0xd76aa478, 0xe8c7b756, 0x242070db, 0xc1bdceee, 0xf57c0faf, 0x4787c62a, 0xa8304613, 0xfd469501,
    0x698098d8, 0x8b44f7af, 0xffff5bb1, 0x895cd7be, 0x6b901122, 0xfd987193, 0xa679438e, 0x49b40821,
    0xf61e2562, 0xc040b340, 0x265e5a51, 0xe9b6c7aa, 0xd62f105d, 0x02441453, 0xd8a1e681, 0xe7d3fbc8,
    0x21e1cde6, 0xc33707d6, 0xf4d50d87, 0x455a14ed, 0xa9e3e905, 0xfcefa3f8, 0x676f02d9, 0x8d2a4c8a,
    0xfffa3942, 0x8771f681, 0x6d9d6122, 0xfde5380c, 0xa4beea44, 0x4bdecfa9, 0xf6bb4b60, 0xbebfbc70,
    0x289b7ec6, 0xeaa127fa, 0xd4ef3085, 0x04881d05, 0xd9d4d039, 0xe6db99e5, 0x1fa27cf8, 0xc4ac5665,
    0xf4292244, 0x432aff97, 0xab9423a7, 0xfc93a039, 0x655b59c3, 0x8f0ccc92, 0xffeff47d, 0x85845dd1,
    0x6fa87e4f, 0xfe2ce6e0, 0xa3014314, 0x4e0811a1, 0xf7537e82, 0xbd3af235, 0x2ad7d2bb, 0xeb86d391,
];

/// md5 digest (16 bytes). md5 is enough for key integrity (guarding against accidental drift); this is not a security boundary.
fn md5(data: &[u8]) -> [u8; 16] {
    let mut msg = data.to_vec();
    let bitlen = (data.len() as u64).wrapping_mul(8);
    msg.push(0x80);
    while msg.len() % 64 != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&bitlen.to_le_bytes());
    let (mut a0, mut b0, mut c0, mut d0) =
        (0x67452301u32, 0xefcdab89u32, 0x98badcfeu32, 0x10325476u32);
    for chunk in msg.chunks_exact(64) {
        let mut m = [0u32; 16];
        for (i, w) in m.iter_mut().enumerate() {
            *w = u32::from_le_bytes(chunk[i * 4..i * 4 + 4].try_into().unwrap());
        }
        let (mut a, mut b, mut c, mut d) = (a0, b0, c0, d0);
        for i in 0..64 {
            let (f, g) = match i / 16 {
                0 => ((b & c) | (!b & d), i),
                1 => ((d & b) | (!d & c), (5 * i + 1) % 16),
                2 => (b ^ c ^ d, (3 * i + 5) % 16),
                _ => (c ^ (b | !d), (7 * i) % 16),
            };
            let f2 = f.wrapping_add(a).wrapping_add(MD5_K[i]).wrapping_add(m[g]);
            a = d;
            d = c;
            c = b;
            b = b.wrapping_add(f2.rotate_left(MD5_S[i]));
        }
        a0 = a0.wrapping_add(a);
        b0 = b0.wrapping_add(b);
        c0 = c0.wrapping_add(c);
        d0 = d0.wrapping_add(d);
    }
    let mut out = [0u8; 16];
    for (i, w) in [a0, b0, c0, d0].iter().enumerate() {
        out[i * 4..i * 4 + 4].copy_from_slice(&w.to_le_bytes());
    }
    out
}

pub(crate) fn md5_hex(data: &[u8]) -> String {
    md5(data).iter().map(|b| format!("{b:02x}")).collect()
}

// Recording-set fingerprint (invalidation key)

fn fingerprints_for(
    cr_dir: &std::path::Path,
    targets: &[String],
    cand: &[String],
) -> anyhow::Result<HashMap<String, String>> {
    let mut acc: HashMap<String, Vec<u8>> =
        targets.iter().map(|c| (c.clone(), Vec::new())).collect();
    let mut entries: Vec<std::path::PathBuf> = std::fs::read_dir(cr_dir)?
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("wav"))
        .collect();
    entries.sort();
    for p in entries {
        let name = p
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_string();
        let cmd = crate::udkws_v2::classify_recording(&name, cand);
        let Some(buf) = acc.get_mut(&cmd) else {
            continue;
        };
        let (ms, ns, sz) = match p.metadata() {
            Ok(md) => {
                let (ms, ns) = md
                    .modified()
                    .ok()
                    .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                    .map(|d| (d.as_secs(), d.subsec_nanos()))
                    .unwrap_or((0, 0));
                (ms, ns, md.len())
            }
            Err(_) => (0, 0, 0),
        };
        buf.extend_from_slice(name.as_bytes());
        buf.push(0);
        buf.extend_from_slice(format!("{ms}:{ns}:{sz}\0").as_bytes());
    }
    Ok(acc.into_iter().map(|(k, v)| (k, md5_hex(&v))).collect())
}

// Cache document (serde; f32/f64 round-trip is lossless)

#[derive(serde::Serialize, serde::Deserialize, Clone, Debug)]
struct CmdEntry {
    rec_fingerprint: String,
    #[serde(default)]
    centroid_bits: Vec<u32>,
    /// Median row count for this command's duration prior (median_of(rec_cnt).max(3.0); part of the syllable-length prior), stored as f64 bits
    #[serde(default)]
    med_rows_bits: u64,
    /// Sequence templates (cache v2): per recording -> per character -> centroid bits. Only recordings with >=2 nuclei and a complete character split qualify;
    /// single-character commands (expected nuclei 1) stay empty and never use sequence mode.
    #[serde(default)]
    seq_bits: Vec<Vec<Vec<u32>>>,
    /// Sequence template (cache v2): expected nuclei = median nucleus count across recordings (>=1)
    #[serde(default)]
    exp_nuclei: usize,
}

#[derive(serde::Serialize, serde::Deserialize, Clone, Debug)]
struct CacheDoc {
    version: u32,
    encoder_md5: String,
    k: usize,
    #[serde(default)]
    commands: BTreeMap<String, CmdEntry>,
}

impl CacheDoc {
    fn empty() -> Self {
        CacheDoc {
            version: 0,
            encoder_md5: String::new(),
            k: 0,
            commands: BTreeMap::new(),
        }
    }
    fn structural_ok(&self, enc_md5: &str, k: usize) -> bool {
        self.version == cache_version() && self.k == k && self.encoder_md5 == enc_md5
    }
}

fn cache_path(model_dir: &str) -> std::path::PathBuf {
    std::path::Path::new(model_dir).join("nuclei_templates_v2.json.enc")
}

/// Command -> expected nucleus count derived from templates (feeds the UI's multi-pronunciation state; missing cache -> empty map).
pub fn exp_nuclei_map(model_dir: &str) -> std::collections::HashMap<String, usize> {
    match read_cache(model_dir) {
        Ok(doc) if doc.version == cache_version() => doc
            .commands
            .iter()
            .map(|(k, v)| (k.clone(), v.exp_nuclei))
            .collect(),
        _ => Default::default(),
    }
}

/// Read the cache (missing / corrupt / decryption failure / missing field -> Err; the caller logs one WARN and rebuilds from an empty document)
fn read_cache(model_dir: &str) -> anyhow::Result<CacheDoc> {
    let path = cache_path(model_dir);
    let text = crate::crypto::read_text_auto(&path).context("failed to read or decrypt the cache")?;
    let doc: CacheDoc = serde_json::from_str(&text).context("failed to parse the cache JSON")?;
    Ok(doc)
}

/// Atomic cache write (reuses the same L1 encrypted atomic path as embed_proto_v2)
fn write_cache(model_dir: &str, doc: &CacheDoc) -> anyhow::Result<()> {
    let json = serde_json::to_string(doc)?;
    crate::udkws_v2::write_json_atomic_v2(&cache_path(model_dir), &json)
}

/// Syllable-length prior (same expression as the end of build_nuclei_templates: sum/len*0.08 -- floating point is not associative,
/// so the sum must run in the same order to match the on-the-fly value bit for bit)
fn prior_from_med_rows(med_rows: &[f64]) -> f64 {
    med_rows.iter().sum::<f64>() / med_rows.len().max(1) as f64 * 0.08
}

// Engine entry: template loading from from_model_dir

#[derive(Debug, Default)]
pub struct NucleiCacheStats {
    /// Cache file read and parsed successfully (corrupt / missing = false)
    pub cache_loaded: bool,
    pub hit_cmds: usize,
    pub rebuilt_cmds: usize,
    pub rebuilt_names: Vec<String>,
}

/// Load syllable-nucleus templates at startup: try the cache first -- a full hit skips encoding; if an invalidation key changed,
/// only the affected commands are recomputed and the rest come from the cache; the document is written back afterwards with
/// other commands' entries merged in. Returns (cents, prior, med_rows, seq_recs, exp_nuclei, stats); cents/prior match a full
/// on-the-fly build_nuclei_templates bit for bit (same med_rows order, same prior formula); seq_recs/exp_nuclei carry the
/// sequence templates (old caches are invalidated and rebuilt naturally).

/// Encoding failure (e.g. a command with no recordings) propagates as Err -- same semantics as the direct path (from_model_dir degrades).
#[allow(clippy::type_complexity)]
pub(crate) fn load_or_build(
    enc: &mut crate::udkws_v2::Encoder256,
    cr_dir: &std::path::Path,
    cmds: &[String],
    model_dir: &str,
    k: usize,
    gate: f64,
) -> anyhow::Result<(
    Vec<Vec<f32>>,
    f64,
    Vec<f64>,
    Vec<crate::udkws_v2::CmdSeqTemplate>,
    Vec<usize>,
    NucleiCacheStats,
)> {
    let t0 = std::time::Instant::now();
    let enc_md5 = enc.src_md5.clone();
    let fps = fingerprints_for(cr_dir, cmds, cmds)?;
    let (mut doc, cache_loaded) = match read_cache(model_dir) {
        Ok(d) => (d, true),
        Err(e) => {
            eprintln!("[engine][WARN] syllable-nucleus template cache unavailable ({e:#}); recomputing this time");
            (CacheDoc::empty(), false)
        }
    };
    let structural_ok = doc.structural_ok(&enc_md5, k);
    let mut cents: Vec<Vec<f32>> = Vec::with_capacity(cmds.len());
    let mut med_rows: Vec<f64> = Vec::with_capacity(cmds.len());
    let mut seqs: Vec<crate::udkws_v2::CmdSeqTemplate> = Vec::with_capacity(cmds.len());
    let mut exps: Vec<usize> = Vec::with_capacity(cmds.len());
    let mut stats = NucleiCacheStats {
        cache_loaded,
        ..Default::default()
    };
    for cmd in cmds {
        let hit = structural_ok
            && doc.commands.get(cmd).is_some_and(|e| {
                e.rec_fingerprint == fps.get(cmd).map(String::as_str).unwrap_or("")
                    && e.centroid_bits.len() == enc.out_dim
                    && e.centroid_bits.iter().all(|&b| f32::from_bits(b).is_finite())
                    // med_rows is always >= 3.0 (median_of.max(3.0)); a missing field defaults to 0 and is rejected
                    && f64::from_bits(e.med_rows_bits) >= 3.0
                    // Sequence-template v2 fields: expected nuclei >= 1; sequence entries are checked for dimension and finiteness;
                    // consistency: n == 1 <=> no sequence (single-character commands never build one, so a half-stale document cannot match)
                    && e.exp_nuclei >= 1
                    && ((e.exp_nuclei < 2 && e.seq_bits.is_empty())
                        || (e.exp_nuclei >= 2 && !e.seq_bits.is_empty()))
                    && e.seq_bits.iter().all(|r| {
                        !r.is_empty()
                            && r.iter().all(|c| {
                                c.len() == enc.out_dim
                                    && c.iter().all(|&b| f32::from_bits(b).is_finite())
                            })
                    })
            });
        if hit {
            let e = &doc.commands[cmd.as_str()];
            cents.push(e.centroid_bits.iter().map(|&b| f32::from_bits(b)).collect());
            med_rows.push(f64::from_bits(e.med_rows_bits));
            seqs.push(crate::udkws_v2::CmdSeqTemplate {
                recs: e
                    .seq_bits
                    .iter()
                    .map(|r| {
                        r.iter()
                            .map(|c| c.iter().map(|&b| f32::from_bits(b)).collect())
                            .collect()
                    })
                    .collect(),
                n: e.exp_nuclei,
            });
            exps.push(e.exp_nuclei);
            stats.hit_cmds += 1;
        } else {
            let t = crate::udkws_v2::build_nuclei_template_for_cmd(enc, cr_dir, cmd, cmds, gate)
                .with_context(|| format!("syllable-nucleus template build failed (command {cmd})"))?;
            cents.push(t.centroid);
            med_rows.push(t.med_rows);
            seqs.push(t.seq);
            exps.push(t.exp_nuclei);
            stats.rebuilt_cmds += 1;
            stats.rebuilt_names.push(cmd.clone());
        }
    }
    let prior = prior_from_med_rows(&med_rows);
    let ms = t0.elapsed().as_secs_f64() * 1000.0;
    if stats.rebuilt_cmds == 0 {
        eprintln!(
            "[engine] syllable-nucleus template cache fully hit: {}/{} commands need no re-encoding ({ms:.0} ms)",
            stats.hit_cmds,
            cmds.len()
        );
    } else {
        eprintln!(
            "[engine] syllable-nucleus template cache (previous cache read: {}): {} hits, recomputed {:?} (of {} commands, {ms:.0} ms)",
            if stats.cache_loaded {
                "ok"
            } else {
                "failed/missing"
            },
            stats.hit_cmds,
            stats.rebuilt_names,
            cmds.len()
        );
    }
    if stats.rebuilt_cmds > 0 || !structural_ok {
        if !structural_ok {
            doc.commands.clear();
        }
        for (i, cmd) in cmds.iter().enumerate() {
            doc.commands.insert(
                cmd.clone(),
                CmdEntry {
                    rec_fingerprint: fps.get(cmd).cloned().unwrap_or_default(),
                    centroid_bits: cents[i].iter().map(|x| x.to_bits()).collect(),
                    med_rows_bits: med_rows[i].to_bits(),
                    seq_bits: seqs[i]
                        .recs
                        .iter()
                        .map(|r| {
                            r.iter()
                                .map(|c| c.iter().map(|x| x.to_bits()).collect())
                                .collect()
                        })
                        .collect(),
                    exp_nuclei: exps[i],
                },
            );
        }
        doc.version = cache_version();
        doc.k = k;
        doc.encoder_md5 = enc_md5;
        if let Err(e) = write_cache(model_dir, &doc) {
            eprintln!("[engine][WARN] failed to write the syllable-nucleus template cache (recomputed on the next start): {e:#}");
        }
    }
    Ok((cents, prior, med_rows, seqs, exps, stats))
}

// Generation writes: gen_embed_proto / gen_embed_proto_single

/// The existing document is the merge base; if version / encoder / k differ from the current ones it is discarded wholesale
/// (old entries were built under the old encoder or k, and mixing them would let unrecomputed commands match stale centroids).
fn load_doc_for_merge(model_dir: &str, enc_md5: &str, k: usize) -> CacheDoc {
    match read_cache(model_dir) {
        Ok(d) if d.structural_ok(enc_md5, k) => d,
        Ok(_) => CacheDoc::empty(),
        Err(_) => CacheDoc::empty(),
    }
}

pub(crate) fn store_full(
    model_dir: &str,
    enc_md5: &str,
    k: usize,
    cr_dir: &std::path::Path,
    cmds: &[String],
    cents: &[Vec<f32>],
    med_rows: &[f64],
    seqs: &[crate::udkws_v2::CmdSeqTemplate],
    exps: &[usize],
) {
    if let Err(e) = store_full_impl(
        model_dir, enc_md5, k, cr_dir, cmds, cents, med_rows, seqs, exps,
    ) {
        eprintln!("[engine][WARN] failed to update the syllable-nucleus template cache (result unaffected): {e:#}");
    }
}

fn store_full_impl(
    model_dir: &str,
    enc_md5: &str,
    k: usize,
    cr_dir: &std::path::Path,
    cmds: &[String],
    cents: &[Vec<f32>],
    med_rows: &[f64],
    seqs: &[crate::udkws_v2::CmdSeqTemplate],
    exps: &[usize],
) -> anyhow::Result<()> {
    anyhow::ensure!(
        cmds.len() == cents.len()
            && cmds.len() == med_rows.len()
            && cmds.len() == seqs.len()
            && cmds.len() == exps.len(),
        "cmds/cents/med_rows/seqs/exps length mismatch: {} {} {} {} {}",
        cmds.len(),
        cents.len(),
        med_rows.len(),
        seqs.len(),
        exps.len()
    );
    // Structural key supplied by the caller (= enc.src_md5, the md5 of the decrypted encoder contents; see the load_or_build note)
    let fps = fingerprints_for(cr_dir, cmds, cmds)?;
    let mut doc = load_doc_for_merge(model_dir, enc_md5, k);
    for (i, cmd) in cmds.iter().enumerate() {
        doc.commands.insert(
            cmd.clone(),
            CmdEntry {
                rec_fingerprint: fps.get(cmd).cloned().unwrap_or_default(),
                centroid_bits: cents[i].iter().map(|x| x.to_bits()).collect(),
                med_rows_bits: med_rows[i].to_bits(),
                seq_bits: seqs[i]
                    .recs
                    .iter()
                    .map(|r| {
                        r.iter()
                            .map(|c| c.iter().map(|x| x.to_bits()).collect())
                            .collect()
                    })
                    .collect(),
                exp_nuclei: exps[i],
            },
        );
    }
    doc.version = cache_version();
    doc.k = k;
    doc.encoder_md5 = enc_md5.to_string();
    write_cache(model_dir, &doc)
}

pub fn store_single(
    model_dir: &str,
    k: usize,
    cr_dir: &std::path::Path,
    cand: &[String],
    cmd: &str,
    enc: &mut crate::udkws_v2::Encoder256,
    gate: f64,
) {
    if let Err(e) = store_single_impl(model_dir, k, cr_dir, cand, cmd, enc, gate) {
        eprintln!("[engine][WARN] failed to update a single command's node-template cache entry (result unaffected): {e:#}");
    }
}

fn store_single_impl(
    model_dir: &str,
    k: usize,
    cr_dir: &std::path::Path,
    cand: &[String],
    cmd: &str,
    enc: &mut crate::udkws_v2::Encoder256,
    gate: f64,
) -> anyhow::Result<()> {
    let t = crate::udkws_v2::build_nuclei_template_for_cmd(enc, cr_dir, cmd, cand, gate)?;
    let targets = vec![cmd.to_string()];
    let fps = fingerprints_for(cr_dir, &targets, cand)?;
    // Structural key = md5 of the decrypted encoder contents (same source as load_or_build; see that function's note)
    let enc_md5 = enc.src_md5.clone();
    let mut doc = load_doc_for_merge(model_dir, &enc_md5, k);
    doc.commands.insert(
        cmd.to_string(),
        CmdEntry {
            rec_fingerprint: fps.get(cmd).cloned().unwrap_or_default(),
            centroid_bits: t.centroid.iter().map(|x| x.to_bits()).collect(),
            med_rows_bits: t.med_rows.to_bits(),
            seq_bits: t
                .seq
                .recs
                .iter()
                .map(|r| {
                    r.iter()
                        .map(|c| c.iter().map(|x| x.to_bits()).collect())
                        .collect()
                })
                .collect(),
            exp_nuclei: t.exp_nuclei,
        },
    );
    doc.version = cache_version();
    doc.k = k;
    doc.encoder_md5 = enc_md5;
    write_cache(model_dir, &doc)
}

// Unit tests

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn temp_dir(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!(
            "kws_nuclei_cache_test_{tag}_{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    // -- Pure logic --

    /// md5 matches every RFC 1321 test vector (locks down the self-contained implementation)
    #[test]
    fn md5_rfc1321_vectors() {
        let cases: &[(&str, &str)] = &[
            ("", "d41d8cd98f00b204e9800998ecf8427e"),
            ("a", "0cc175b9c0f1b6a831c399e269772661"),
            ("abc", "900150983cd24fb0d6963f7d28e17f72"),
            ("message digest", "f96b697d7cb7938d525a2f31aaf161d0"),
            (
                "abcdefghijklmnopqrstuvwxyz",
                "c3fcd3d76192e4007dfb496cca67e13b",
            ),
            (
                "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789",
                "d174ab98d277d9f5a5611c2c9f419d9f",
            ),
            (
                "12345678901234567890123456789012345678901234567890123456789012345678901234567890",
                "57edf4a22be3c955ac49da2e2107b67a",
            ),
        ];
        for (msg, want) in cases {
            assert_eq!(md5_hex(msg.as_bytes()), *want, "md5({msg:?})");
        }
        // Message longer than 64 bytes, exercising the multi-chunk path
        let long = "The quick brown fox jumps over the lazy dog. ".repeat(8);
        assert_eq!(md5_hex(long.as_bytes()), md5_hex(long.as_bytes()));
        assert_eq!(md5_hex(long.as_bytes()).len(), 32);
    }

    #[test]
    fn cache_number_roundtrip_bitexact() {
        let mut x = 0x123456789abcdef0u64;
        let mut u64s = vec![0u64, u64::MAX, 1 << 53, (1 << 53) + 1];
        let mut u32s = vec![
            0u32,
            u32::MAX,
            0x7fc00000, /*NaN bit pattern round-trips unchanged*/
        ];
        for _ in 0..2000 {
            x = x
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            u64s.push(x);
            u32s.push(x as u32);
        }
        for v in &u64s {
            let back: u64 = serde_json::from_str(&serde_json::to_string(v).unwrap()).unwrap();
            assert_eq!(&back, v, "u64 {v:#x}");
        }
        for v in &u32s {
            let back: u32 = serde_json::from_str(&serde_json::to_string(v).unwrap()).unwrap();
            assert_eq!(&back, v, "u32 {v:#x}");
        }
        // Whole-document round-trip: bit patterns survive f32/f64 exactly and BTreeMap key order is deterministic
        let doc = CacheDoc {
            version: cache_version(),
            encoder_md5: "deadbeef".into(),
            k: 2,
            commands: BTreeMap::from([(
                "c".into(),
                CmdEntry {
                    rec_fingerprint: "fp".into(),
                    centroid_bits: u32s[..8].to_vec(),
                    med_rows_bits: u64s[3],
                    seq_bits: vec![vec![u32s[..4].to_vec(), u32s[4..8].to_vec()]],
                    exp_nuclei: 2,
                },
            )]),
        };
        let back: CacheDoc = serde_json::from_str(&serde_json::to_string(&doc).unwrap()).unwrap();
        assert_eq!(
            back.commands["c"].centroid_bits,
            doc.commands["c"].centroid_bits
        );
        assert_eq!(
            back.commands["c"].med_rows_bits,
            doc.commands["c"].med_rows_bits
        );
        assert_eq!(back.commands["c"].seq_bits, doc.commands["c"].seq_bits);
        assert_eq!(back.commands["c"].exp_nuclei, 2);
        assert_eq!(
            f64::from_bits(back.commands["c"].med_rows_bits).to_bits(),
            u64s[3]
        );
    }

    /// The prior formula matches the end of build_nuclei_templates bit for bit (same summation order; guards against formula drift)
    #[test]
    fn prior_formula_matches_build() {
        for med in [
            vec![4.0],
            vec![3.0, 5.5],
            vec![3.0, 4.0, 7.25, 2.0],
            vec![3.1, 4.4, 5.9, 6.2, 3.0, 9.75],
        ] {
            // Same expression as the tail of build_nuclei_templates in udkws_v2.rs
            let want = med.iter().sum::<f64>() / med.len().max(1) as f64 * 0.08;
            assert_eq!(prior_from_med_rows(&med).to_bits(), want.to_bits());
        }
    }

    /// Recording-set fingerprint: unchanged contents -> stable; mtime change -> changed; size change -> changed; add or remove -> changed
    #[test]
    fn fingerprint_stability_and_change() {
        let d = temp_dir("fp");
        let f = d.join("c_act_x.wav");
        std::fs::write(&f, b"RIFF____WAVEfmt data").unwrap();
        let cmds = vec!["c".to_string()];
        let fp1 = fingerprints_for(&d, &cmds, &cmds).unwrap()["c"].clone();
        let fp2 = fingerprints_for(&d, &cmds, &cmds).unwrap()["c"].clone();
        assert_eq!(fp1, fp2, "unchanged inputs must give the same template fingerprint");
        // mtime change (std::fs::FileTimes, contents unchanged)
        let future = std::time::SystemTime::now() + std::time::Duration::from_secs(3600);
        std::fs::File::options()
            .write(true)
            .open(&f)
            .unwrap()
            .set_times(std::fs::FileTimes::new().set_modified(future))
            .unwrap();
        let fp3 = fingerprints_for(&d, &cmds, &cmds).unwrap()["c"].clone();
        assert_ne!(fp3, fp1, "an mtime change must invalidate");
        // size change (appended content)
        let mut cur = std::fs::read(&f).unwrap();
        cur.extend_from_slice(b"x");
        std::fs::write(&f, &cur).unwrap();
        let fp4 = fingerprints_for(&d, &cmds, &cmds).unwrap()["c"].clone();
        assert_ne!(fp4, fp3, "a size change must invalidate");
        // A new file appears -> changed
        std::fs::write(d.join("c_act_y.wav"), b"RIFF____WAVEfmt data2").unwrap();
        let fp5 = fingerprints_for(&d, &cmds, &cmds).unwrap()["c"].clone();
        assert_ne!(fp5, fp4, "a recording-set change must invalidate");
        // Empty set yields a deterministic fingerprint (the no-recordings case)
        let none = vec!["zz".to_string()];
        assert_eq!(
            fingerprints_for(&d, &none, &none).unwrap()["zz"],
            md5_hex(b"")
        );
        std::fs::remove_dir_all(&d).ok();
    }
}
