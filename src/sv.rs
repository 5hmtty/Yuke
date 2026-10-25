use anyhow::Context;
use ndarray::{Array3, Axis};
use std::path::Path;

const ONNX_NAME: &str = "sv_cnceleb_resnet34.onnx";

/// Whether the speaker model is available from the pack, `<exe>/assets/`, or `assets/`.
pub fn model_available() -> bool {
    let packed = format!("{}/{}", crate::pak::ASSETS_PREFIX, ONNX_NAME);
    if crate::pak::contains(&packed) {
        return true;
    }
    if let Some(dir) = crate::exe_dir() {
        if dir.join("assets").join(ONNX_NAME).is_file() {
            return true;
        }
    }
    Path::new("assets").join(ONNX_NAME).is_file()
}

pub struct SvGate {
    sess: ort::session::Session,
    input: String,
    /// Enrolled centroid (L2-normalized, 256-dim)
    centroid: Vec<f32>,
}

fn sv_onnx_bytes() -> anyhow::Result<Vec<u8>> {
    crate::pak::read(&format!("{}/{}", crate::pak::ASSETS_PREFIX, ONNX_NAME))
        .or_else(|| {
            crate::exe_dir()
                .map(|d| d.join("assets").join(ONNX_NAME))
                .filter(|p| p.is_file())
                .and_then(|p| std::fs::read(p).ok())
        })
        .or_else(|| std::fs::read(Path::new("assets").join(ONNX_NAME)).ok())
        .context("speaker model not found (sv_cnceleb_resnet34.onnx is in neither the pack nor assets/)")
}

fn make_session() -> anyhow::Result<ort::session::Session> {
    let raw = sv_onnx_bytes()?;
    ort::session::Session::builder()
        .map_err(|e| anyhow::anyhow!("sv builder: {e}"))?
        .with_intra_threads(2)
        .map_err(|e| anyhow::anyhow!("sv thread setup: {e}"))?
        .commit_from_memory(&raw)
        .context("failed to load the speaker model (in memory)")
}

fn l2(v: &mut [f32]) {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if n > 0.0 {
        for x in v.iter_mut() {
            *x /= n;
        }
    }
}

pub fn trim_active(samples: &[f32], sr: usize) -> &[f32] {
    const FRAME_MS: usize = 10;
    const PAD_MS: usize = 50;
    const REL_DB: f32 = 20.0;
    const ABS_FLOOR_DB: f32 = -55.0;
    const MIN_KEEP_MS: usize = 200;
    let frame = (sr * FRAME_MS / 1000).max(1);
    let n_frames = samples.len() / frame;
    if n_frames < 4 {
        return samples;
    }
    let mut db = Vec::with_capacity(n_frames);
    for i in 0..n_frames {
        let s = &samples[i * frame..(i + 1) * frame];
        let e = s.iter().map(|x| x * x).sum::<f32>() / frame as f32;
        db.push(10.0 * e.max(1e-12).log10());
    }
    let peak = db.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !peak.is_finite() || peak < ABS_FLOOR_DB {
        return samples;
    }
    let gate = peak - REL_DB;
    // Longest run above threshold (frame index range [best_l, best_l+best_len))
    let (mut best_l, mut best_len, mut cur_l, mut cur_len) = (0usize, 0usize, 0usize, 0usize);
    for (i, &v) in db.iter().enumerate() {
        if v >= gate {
            if cur_len == 0 {
                cur_l = i;
            }
            cur_len += 1;
            if cur_len > best_len {
                best_len = cur_len;
                best_l = cur_l;
            }
        } else {
            cur_len = 0;
        }
    }
    if best_len == 0 || best_len * FRAME_MS < MIN_KEEP_MS {
        return samples;
    }
    let pad = (sr * PAD_MS / 1000).max(1);
    let a = (best_l * frame).saturating_sub(pad);
    let b = ((best_l + best_len) * frame + pad).min(samples.len());
    if b <= a {
        return samples;
    }
    &samples[a..b]
}

fn embed_with(
    sess: &mut ort::session::Session,
    input: &str,
    samples_16k: &[f32],
) -> anyhow::Result<Vec<f32>> {
    let samples_16k = trim_active(samples_16k, 16000);
    let feats = crate::features::log_mel_fbank(samples_16k, 16000, 80);
    let (t, d) = (feats.nrows(), feats.ncols());
    if t == 0 {
        anyhow::bail!("audio clip too short");
    }
    let mut mean = vec![0.0f32; d];
    for i in 0..t {
        for b in 0..d {
            mean[b] += feats[[i, b]];
        }
    }
    for m in mean.iter_mut() {
        *m /= t as f32;
    }
    let mut arr = feats.view().insert_axis(Axis(0)).to_owned();
    for i in 0..t {
        for b in 0..d {
            arr[[0, i, b]] -= mean[b];
        }
    }
    let x3: Array3<f32> = arr;
    let inputs: Vec<(String, ort::session::SessionInputValue)> =
        vec![(input.to_string(), ort::value::Value::from_array(x3)?.into())];
    let outputs = sess.run(inputs).context("sv inference failed")?;
    // wespeaker export convention: input `feats`, output `embs` (confirmed with an onnx probe)
    let mut e: Vec<f32> = outputs["embs"]
        .try_extract_array::<f32>()
        .context("failed to extract sv output")?
        .iter()
        .copied()
        .collect();
    l2(&mut e);
    Ok(e)
}

impl SvGate {
    /// Load the gate from the model directory: sv model (resolved via pack/assets) plus the centroid json (computed by the caller and passed in).
    /// A missing piece returns Err (the caller degrades to disabled). `model_dir` is kept only for signature compatibility.
    pub fn load(model_dir: &Path, centroid_json: &str) -> anyhow::Result<SvGate> {
        let _ = model_dir;
        let sess = make_session()?;
        let input = sess
            .inputs()
            .first()
            .map(|i| i.name().to_string())
            .context("the sv model has no inputs")?;
        let centroid: Vec<f32> =
            serde_json::from_str(centroid_json).context("failed to parse sv_voiceprint")?;
        if centroid.len() != 256 {
            anyhow::bail!("sv_voiceprint dimension {} != 256", centroid.len());
        }
        Ok(SvGate {
            sess,
            input,
            centroid,
        })
    }

    /// Voiceprint similarity (trigger candidate window vs the enrolled centroid).
    pub fn similarity(&mut self, samples_16k: &[f32]) -> anyhow::Result<f32> {
        let e = embed_with(&mut self.sess, &self.input, samples_16k)?;
        Ok(self.centroid.iter().zip(&e).map(|(a, b)| a * b).sum())
    }
}

/// Only enabled when n >= 5 -- too few samples make the statistics unstable, so filtering is skipped; if fewer than 3 remain after
/// pruning, filtering is abandoned entirely so the enrolled set never becomes unusable.
fn trim_outliers(embs: Vec<Vec<f32>>) -> Vec<Vec<f32>> {
    /// Allowed pruning margin: the maximum drop below the median. Same-speaker similarity naturally fluctuates by about ±0.15
    /// (measured 0.5-0.73 for the same speaker), so only 0.15 or more below the median counts as an outlier.
    const OUTLIER_MARGIN: f32 = 0.15;
    /// Absolute floor for pruning: anything at or above 0.3 is not an outlier (below 0.3 is genuinely unlike the speaker)
    const MIN_CUT: f32 = 0.3;
    if embs.len() < 5 {
        return embs;
    }
    let dim = embs[0].len();
    let mut c = vec![0.0f32; dim];
    for e in &embs {
        for (a, x) in c.iter_mut().zip(e) {
            *a += x;
        }
    }
    let inv = 1.0 / embs.len() as f32;
    for a in c.iter_mut() {
        *a *= inv;
    }
    l2(&mut c);
    let cos: Vec<f32> = embs
        .iter()
        .map(|e| c.iter().zip(e).map(|(a, b)| a * b).sum())
        .collect();
    let mut s = cos.clone();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let med = s[s.len() / 2];
    let cut = (med - OUTLIER_MARGIN).max(MIN_CUT);
    let mut kept: Vec<Vec<f32>> = Vec::new();
    for (e, c) in embs.iter().zip(&cos) {
        if *c >= cut {
            kept.push(e.clone());
        }
    }
    if kept.len() >= 3 { kept } else { embs }
}

fn embed_all(
    sess: &mut ort::session::Session,
    input: &str,
    paths: impl Iterator<Item = std::path::PathBuf>,
) -> (Vec<Vec<f32>>, usize) {
    let mut embs = Vec::new();
    let mut skipped = 0usize;
    for p in paths {
        let Ok((sr, samples)) = crate::recordings::read_wav_auto(&p) else {
            skipped += 1;
            continue;
        };
        let mut f32s: Vec<f32> = samples;
        if f32s.len() < sr as usize / 4 {
            skipped += 1;
            continue;
        }
        if sr != 16000 {
            f32s = crate::features::resample_linear(&f32s, sr, 16000);
        }
        match embed_with(sess, input, &f32s) {
            Ok(e) => embs.push(e),
            Err(_) => skipped += 1,
        }
    }
    (embs, skipped)
}

/// Shared: embedding list -> (centroid, number actually used). Outlier pruning -> mean -> L2 normalize.
fn centroid_of(embs: Vec<Vec<f32>>) -> (Vec<f32>, usize) {
    let n0 = embs.len();
    let kept = trim_outliers(embs);
    let n = kept.len();
    let mut acc = vec![0.0f64; 256];
    for e in &kept {
        for (a, x) in acc.iter_mut().zip(e) {
            *a += *x as f64;
        }
    }
    for a in acc.iter_mut() {
        *a /= n as f64;
    }
    let mut cen: Vec<f32> = acc.iter().map(|&x| x as f32).collect();
    l2(&mut cen);
    if n0 != n {
        eprintln!("[sv] centroid: {n0} embeddings in -> {n} after outlier rejection");
    }
    (cen, n)
}

/// **live path**: build the enrolled centroid from the model directory. Returns (centroid, number actually used).
pub fn build_centroid(model_dir: &str) -> anyhow::Result<(Vec<f32>, usize)> {
    use crate::recordings::RecKind;
    let mut sess = make_session()?;
    let input = sess
        .inputs()
        .first()
        .map(|i| i.name().to_string())
        .context("the sv model has no inputs")?;
    let dir = crate::recordings::record_dir(model_dir);
    let all = crate::recordings::list(model_dir);
    let total = all.len();
    let active: Vec<std::path::PathBuf> = all
        .into_iter()
        .filter(|r| r.kind == RecKind::Active)
        .map(|r| dir.join(&r.file))
        .collect();
    let skipped_learned = total - active.len();
    let (embs, skipped) = embed_all(&mut sess, &input, active.into_iter());
    if embs.is_empty() {
        anyhow::bail!(
            "no actively recorded takes available for voiceprint enrolment ({skipped_learned} learned takes excluded, {skipped} unreadable): {}",
            dir.display()
        );
    }
    if skipped_learned > 0 {
        eprintln!("[sv] voiceprint enrolment: ignoring {skipped_learned} passively learned slices");
    }
    Ok(centroid_of(embs))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tone(n: usize, amp: f32) -> Vec<f32> {
        (0..n).map(|i| amp * ((i as f32) * 0.1).sin()).collect()
    }

    /// Trim to the effective speech region: silence-speech-silence -> keep only the middle part (with 50 ms of margin)
    #[test]
    fn trim_active_keeps_speech_and_drops_silence() {
        let sr = 16000usize;
        let q = sr / 4; // 0.25s
        let mut x = vec![0.0f32; q];
        x.extend(tone(q * 2, 0.5)); // 0.5 s of high energy
        x.extend(vec![0.0f32; q]);
        let t = trim_active(&x, sr);
        assert!(t.len() < x.len(), "silence should be trimmed: {} -> {}", x.len(), t.len());
        assert!(t.len() >= sr / 2, "the whole utterance must survive (with margin): {}", t.len());
        assert!(
            t.len() <= (q * 2 + sr / 10),
            "the margin must not exceed 50 ms on each side: {}",
            t.len()
        );
    }

    /// Outlier pruning: a highly consistent group must not be trimmed
    #[test]
    fn trim_outliers_keeps_coherent_set() {
        let base = vec![1.0f32, 0.0, 0.0];
        let embs: Vec<Vec<f32>> = (0..6)
            .map(|i| {
                let mut v = base.clone();
                v[1] = 0.02 * i as f32; // slight perturbation
                l2(&mut v);
                v
            })
            .collect();
        assert_eq!(trim_outliers(embs).len(), 6, "a consistent group must not be trimmed");
    }

    /// Outlier pruning: a clearly deviating sample must be dropped (and with fewer than 5 samples nothing is filtered)
    #[test]
    fn trim_outliers_drops_outlier() {
        let mk = |v: [f32; 3]| {
            let mut x = v.to_vec();
            l2(&mut x);
            x
        };
        let mut embs = vec![
            mk([1.0, 0.0, 0.0]),
            mk([1.0, 0.05, 0.0]),
            mk([1.0, 0.0, 0.05]),
            mk([1.0, -0.05, 0.0]),
            mk([1.0, 0.0, -0.05]),
        ];
        embs.push(mk([0.0, 1.0, 0.0])); // a fully opposite vector
        let kept = trim_outliers(embs);
        assert_eq!(kept.len(), 5, "the opposite outlier must be removed: {}", kept.len());
        // Fewer than 5 samples -> no filtering (statistics are unstable)
        assert_eq!(
            trim_outliers(vec![mk([1.0, 0.0, 0.0]), mk([0.0, 1.0, 0.0])]).len(),
            2
        );
    }
}
