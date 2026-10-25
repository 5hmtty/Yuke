//! UDKWS v2 decision pipeline: streaming features -> activity gate and burst segmentation -> syllable-nucleus counting -> identity matching -> emission.

//! Template marker: `embed_proto_v2.json` ({cmd:{}}) only records which commands have a template;
//! the acoustic templates are built by nuclei_cache from command_recordings (80 ms row grid).

use crate::features::{MEL_BINS, TARGET_SAMPLE_RATE, resample_linear};
use anyhow::Context;
use ndarray::{Array, Array1, Array2, IxDyn};
use std::collections::HashMap;

/// Curve-invalid sentinel (energy-gate position)
pub const INVALID: f64 = -9.0e9;

/// Decision parameters (calibrated for the int8 encoder).
#[derive(Clone, Debug)]
pub struct EngineParams {
    /// Legacy cache layout key (fixed at 2 so old cache invalidation keys stay compatible); rows are no longer split into slots.
    pub k: usize,
    pub min_conf: f64,
    /// Repeat guard (emission-level refractory period, seconds)
    pub refr_s: f64,
    pub act_db: f64,
    pub act_gate_floor: f64, // Gate lower clamp (default −70)
    pub act_gate_ceiling: f64,
    pub act_gate_margin: f64,      // Gate = noise-floor estimate + margin (default 6.0)
    pub act_floor_alpha: f64,      // Noise-floor EMA coefficient per update (default 0.05)
    pub act_gate_up_alpha: f64,    // Grey-band upward-chase EMA per frame (default 0.001; 0 = disabled,
    pub act_gate_up_guard_db: f64, // Speech guard band: frames at or above gate + this never raise the gate (default 12 dB;
    pub act_gate_fixed: bool,
}

fn min_conf_default() -> f64 {
    std::env::var("VIQ_UDKWS_V2_MINCONF")
        .ok()
        .and_then(|v| v.trim().parse::<f64>().ok())
        .unwrap_or(0.6)
}

impl Default for EngineParams {
    fn default() -> Self {
        EngineParams {
            k: 2,
            min_conf: min_conf_default(),
            // Same source as NucleiParams.refr_pf: the env value is in 10 ms grid units (÷100 gives seconds)
            refr_s: std::env::var("VIQ_UDKWS_V2_NUCLEI_REFR")
                .ok()
                .and_then(|v| v.trim().parse::<usize>().ok())
                .map(|pf| pf as f64 / 100.0)
                .filter(|&s| (0.0..=2.0).contains(&s))
                .unwrap_or(0.57),
            act_db: std::env::var("VIQ_ACT_GATE_DB")
                .ok()
                .and_then(|v| v.trim().parse::<f64>().ok())
                .filter(|&v| (-75.0..=-15.0).contains(&v))
                .unwrap_or(-50.0),
            act_gate_floor: -70.0,
            act_gate_ceiling: -30.0,
            act_gate_margin: 6.0,
            act_floor_alpha: 0.05,
            act_gate_up_alpha: 0.001,
            act_gate_up_guard_db: 12.0,
            act_gate_fixed: true,
        }
    }
}

/// Below-gate hysteresis margin: a frame counts as a noise-floor sample only below (gate − this), so speech and transition frames never pollute the floor estimate
const ACT_GATE_HYST_DB: f64 = 6.0;

fn act_gate_step(nf: &mut Option<f64>, db: f64, p: &EngineParams) -> f64 {
    if p.act_gate_fixed {
        return p.act_db;
    }
    let gate = match *nf {
        None => p.act_db,
        Some(v) => (v + p.act_gate_margin).clamp(p.act_gate_floor, p.act_gate_ceiling),
    };
    if db < gate - ACT_GATE_HYST_DB {
        *nf = Some(match *nf {
            None => db,
            Some(v) => v + p.act_floor_alpha * (db - v),
        });
    } else if let Some(v) = *nf {
        if db < gate + p.act_gate_up_guard_db && p.act_gate_up_alpha > 0.0 {
            *nf = Some(v + p.act_gate_up_alpha * (db - v));
        }
    }
    gate
}

/// Banker's rounding (round half to even) -- refr_rows = round(refr/period)
fn round_half_even(x: f64) -> i64 {
    let f = x.floor();
    let diff = x - f;
    if diff > 0.5 {
        f as i64 + 1
    } else if diff < 0.5 {
        f as i64
    } else if (f as i64) % 2 == 0 {
        f as i64
    } else {
        f as i64 + 1
    }
}

// -- Pure functions --

fn l2_normalize(v: &mut [f32]) {
    let nrm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if nrm > 0.0 {
        for x in v.iter_mut() {
            *x /= nrm;
        }
    }
}

use std::sync::OnceLock;

fn v2_window() -> &'static Vec<f32> {
    static W: OnceLock<Vec<f32>> = OnceLock::new();
    W.get_or_init(|| crate::features::hamming_window(400)) // 25 ms at 16 kHz; symmetric Hamming window
}

fn v2_fbank() -> &'static ndarray::Array2<f32> {
    static F: OnceLock<ndarray::Array2<f32>> = OnceLock::new();
    F.get_or_init(|| {
        // Triangular filterbank (linear mel, no slaney norm)
        crate::features::mel_filterbank(16000, 512, MEL_BINS)
    })
}

/// One frame (400 samples) -> 80-bin log-magnitude mel spectrum
fn v2_amp_frame(frame: &[f32]) -> Vec<f32> {
    static FFT: OnceLock<std::sync::Arc<dyn rustfft::Fft<f32>>> = OnceLock::new();
    let fft = FFT.get_or_init(|| rustfft::FftPlanner::<f32>::new().plan_fft_forward(512));
    let win = v2_window();
    let mut spec = vec![rustfft::num_complex::Complex::new(0.0f32, 0.0f32); 512];
    for k in 0..frame.len() {
        spec[k] = rustfft::num_complex::Complex::new(frame[k] * win[k], 0.0);
    }
    fft.process(&mut spec);
    let fbank = v2_fbank();
    let mut out = Vec::with_capacity(MEL_BINS);
    for m in 0..MEL_BINS {
        let mut mel = 0.0f32;
        for d in 0..fbank.ncols() {
            mel += spec[d].norm() * fbank[[m, d]];
        }
        out.push(mel.max(1e-10).ln());
    }
    if frame_mean_norm() {
        let mean = out.iter().sum::<f32>() / out.len() as f32;
        for v in out.iter_mut() {
            *v -= mean;
        }
    }
    out
}

/// Current identity-window right lookahead (env; template, cache and runtime share one source).
pub(crate) fn ident_fwd_sec_env() -> f64 {
    std::env::var("VIQ_IDENT_FWD_SEC")
        .ok()
        .and_then(|v| v.trim().parse::<f64>().ok())
        .filter(|&v| (0.0..=0.40).contains(&v))
        .unwrap_or(0.16)
}

/// Identity-window fingerprint: part of the template cache invalidation key (changing the window changes the per-character centroid cropping).
/// 0 for the default 0.16 (old caches stay valid); a stable non-zero value once response speed is adjusted in the UI.
pub(crate) fn ident_win_fingerprint() -> u32 {
    let fwd = ident_fwd_sec_env();
    if (fwd - 0.16).abs() < 1e-9 {
        0
    } else {
        1 + (fwd * 1000.0).round() as u32
    }
}

/// "Optimise recognition" switch (env `VIQ_V2_FRAME_MEAN_NORM`, bridged from settings.frame_mean_norm).
/// Called on every hot path, so it caches in a OnceLock and reads once per process (**changing it requires restarting Yuke**).
pub(crate) fn frame_mean_norm() -> bool {
    static F: OnceLock<bool> = OnceLock::new();
    *F.get_or_init(|| {
        std::env::var("VIQ_V2_FRAME_MEAN_NORM")
            .map(|v| v == "1")
            .unwrap_or(false)
    })
}

/// Whole signal -> log-magnitude mel frame sequence (offline enrolment and comparison; 400-sample frames, 160-sample hop)
pub fn log_mel_amp_fbank(samples: &[f32]) -> Vec<Vec<f32>> {
    let mut out = Vec::new();
    if samples.len() < 400 {
        return out;
    }
    let n = (samples.len() - 400) / 160 + 1;
    for i in 0..n {
        out.push(v2_amp_frame(&samples[i * 160..i * 160 + 400]));
    }
    out
}

// -- 256-dim encoder session (v2 only) --

/// "Noise-robust model" switch (env `VIQ_ENCODER_NOISE`, bridged from settings): selects the encoder whose name contains `noise`.
pub(crate) fn noise_encoder_on() -> bool {
    std::env::var("VIQ_ENCODER_NOISE")
        .map(|v| v == "1")
        .unwrap_or(false)
}

/// Locate the v2 encoder in the model directory: `encoder-v2*.onnx`, int8 preferred.
/// Standard and noise-robust models are told apart by whether the name contains `noise`.
/// Falls back to `<exe>/model/` when `model_dir` (UserData) has none -- the loose-file distribution without a pak.
fn resolve_encoder_onnx(model_dir: &str) -> Option<std::path::PathBuf> {
    resolve_encoder_in(model_dir).or_else(|| {
        let exe_model = crate::exe_dir().map(|d| d.join("model"))?;
        resolve_encoder_in(exe_model.to_str()?)
    })
}

fn resolve_encoder_in(model_dir: &str) -> Option<std::path::PathBuf> {
    let want_noise = noise_encoder_on();
    let entries = std::fs::read_dir(model_dir).ok()?;
    // (path, int8)
    let mut best: Option<(std::path::PathBuf, bool)> = None;
    for e in entries.flatten() {
        let p = e.path();
        let fname = p.file_name().and_then(|s| s.to_str()).unwrap_or("");
        let ok = fname.starts_with("encoder-v2") && fname.ends_with(".onnx");
        if !ok || fname.contains("noise") != want_noise {
            continue;
        }
        let is_int8 = fname.contains(".int8.");
        if best.as_ref().is_none_or(|(_, b8)| is_int8 && !*b8) {
            best = Some((p, is_int8));
        }
    }
    best.map(|(p, _)| p)
}

/// v2 encoder (256-dim per-frame pronunciation templates). Session options: graph optimisation All, intra 1, inter 1.
/// Cache-chain protocol x/processed_lens/cached_* -> encoder_out/new_cached_*, output [1,4,256].
pub struct Encoder256 {
    sess: ort::session::Session,
    cache_inputs: Vec<(String, Vec<usize>, bool)>,
    proc_lens: Option<Array1<i64>>,
    cache: HashMap<String, Array<f32, IxDyn>>,
    pub chunk_frames: usize,
    pub out_dim: usize,
    /// Identity path (for logs): loose = model file; pack = the pack file
    pub src_path: std::path::PathBuf,
    pub src_md5: String,
}

impl Encoder256 {
    pub fn load(path: &std::path::Path) -> anyhow::Result<Self> {
        let raw = std::fs::read(path)
            .with_context(|| format!("failed to read the v2 encoder: {}", path.display()))?;
        Self::load_bytes(&raw, path)
    }

    pub fn load_bytes(raw: &[u8], src: &std::path::Path) -> anyhow::Result<Self> {
        use ort::session::builder::GraphOptimizationLevel;
        let label = src.display().to_string();
        let mut b = ort::session::Session::builder()
            .map_err(|e| anyhow::anyhow!("failed to create the session builder: {e}"))?
            .with_optimization_level(GraphOptimizationLevel::All)
            .map_err(|e| anyhow::anyhow!("failed to set the optimisation level: {e}"))?
            .with_intra_threads(1)
            .map_err(|e| anyhow::anyhow!("failed to set the thread count: {e}"))?
            .with_inter_threads(1)
            .map_err(|e| anyhow::anyhow!("failed to set the inter-op thread count: {e}"))?;
        let sess = b
            .commit_from_memory(raw)
            .with_context(|| format!("failed to load the v2 encoder (in memory): {label}"))?;
        let mut cache_inputs = Vec::new();
        let mut chunk_frames = 29usize;
        for inp in sess.inputs() {
            if let ort::value::ValueType::Tensor { ty, shape, .. } = inp.dtype() {
                let dims: Vec<usize> = shape
                    .iter()
                    .map(|&d| if d > 0 { d as usize } else { 1 })
                    .collect();
                if inp.name() == "x" {
                    chunk_frames = dims.get(1).copied().unwrap_or(29);
                } else {
                    cache_inputs.push((
                        inp.name().to_string(),
                        dims,
                        ty.eq(&ort::value::TensorElementType::Int64),
                    ));
                }
            }
        }
        let mut enc = Encoder256 {
            sess,
            cache_inputs,
            proc_lens: None,
            cache: HashMap::new(),
            chunk_frames,
            out_dim: 0,
            src_path: src.to_path_buf(),
            // Structural key = the decrypted content (not the file bytes of src) -- see the field comment
            src_md5: crate::nuclei_cache::md5_hex(raw),
        };
        // Self-check with one all-zero feature block to determine the output shape [1, rows, out_dim]
        // and clear the cache chain afterwards, so a real stream starts from a fresh cache state
        let probe = enc.run_chunk(
            &Array2::<f32>::zeros((enc.chunk_frames, MEL_BINS)),
            enc.chunk_frames as i64,
        )?;
        enc.out_dim = probe.first().map(|r| r.len()).unwrap_or(0);
        enc.reset_state();
        if enc.out_dim == 0 {
            anyhow::bail!("the v2 encoder produced no output: {label}");
        }
        Ok(enc)
    }

    pub fn load_best(model_dir: &str) -> anyhow::Result<Self> {
        let want_noise = noise_encoder_on();
        if let Some((entry, raw)) = crate::pak::find_model_raw_pref("encoder-v2", want_noise) {
            let pp = crate::pak::path().unwrap_or_default();
            eprintln!("[engine] encoder (VIQPAK: {entry})");
            return Self::load_bytes(&raw, &pp);
        }
        let Some(p) = resolve_encoder_onnx(model_dir) else {
            if want_noise {
                anyhow::bail!(
                    "noise-robust encoder onnx not found (model_dir={model_dir}, expected encoder-v2-noise-*.onnx)"
                );
            }
            anyhow::bail!("v2 encoder onnx not found (model_dir={model_dir})");
        };
        eprintln!("[engine] encoder (model directory {model_dir}): {}", p.display());
        Self::load(&p)
    }

    /// Clear the cache-chain state (called before encoding each independent audio stream so results are reproducible)
    pub fn reset_state(&mut self) {
        self.cache.clear();
        self.proc_lens = None;
    }

    /// One block of inference: x [chunk,80] + processed_lens -> output rows (256-dim).
    fn run_chunk(&mut self, x: &Array2<f32>, proc_len: i64) -> anyhow::Result<Vec<Vec<f32>>> {
        let mut inputs: Vec<(String, ort::session::SessionInputValue)> = Vec::new();
        let x3 = x.view().insert_axis(ndarray::Axis(0)).to_owned();
        inputs.push(("x".into(), ort::value::Value::from_array(x3)?.into()));
        for (name, shape, is_i64) in &self.cache_inputs {
            if name == "processed_lens" {
                inputs.push((
                    "processed_lens".into(),
                    ort::value::Value::from_array(Array1::from_elem([1], proc_len))?.into(),
                ));
                continue;
            }
            // Borrows directly from ort (TensorRef::from_array_view, a zero-copy view) -- the previous .cloned() deep-copied every
            // cache tensor per chunk (36 of them for c8 v2), which was pure waste, since new_cached_* overwrite
            // self.cache wholesale after run. Error handling also leaves the cache in place (run_chunk_store's
            // `let Ok(..) else return`), so retry semantics match the copying version bit for bit (mem::take could lose the cache on failure, hence borrowing).
            match self.cache.get(name) {
                Some(v) => inputs.push((
                    name.clone(),
                    ort::value::TensorRef::from_array_view(v)?.into(),
                )),
                None => {
                    let dims = IxDyn(shape.as_slice());
                    if *is_i64 {
                        inputs.push((
                            name.clone(),
                            ort::value::Value::from_array(Array::<i64, _>::zeros(dims))?.into(),
                        ));
                    } else {
                        inputs.push((
                            name.clone(),
                            ort::value::Value::from_array(Array::<f32, _>::zeros(dims))?.into(),
                        ));
                    }
                }
            }
        }
        let outputs = self
            .sess
            .run(inputs)
            .with_context(|| "v2 encoder inference failed")?;
        // Cache write-back (new_cached_* -> cached_*; new_processed_lens -> proc_lens)
        let mut rows: Vec<Vec<f32>> = Vec::new();
        for (name, val) in outputs.iter() {
            if name.starts_with("new_cached_") {
                let mapped = name.replacen("new_cached_", "cached_", 1);
                let v = val
                    .try_extract_array::<f32>()
                    .with_context(|| format!("failed to extract v2 cache {name}"))?
                    .into_owned();
                self.cache.insert(mapped, v);
            } else if name == "new_processed_lens" {
                let pl = val
                    .try_extract_array::<i64>()
                    .with_context(|| "failed to extract v2 processed_lens")?
                    .into_owned();
                self.proc_lens = Some(Array1::from_iter(pl.iter().copied()));
            } else {
                // Main output (encoder_out) [1, T, D] -> T rows
                let v = val
                    .try_extract_array::<f32>()
                    .with_context(|| "failed to extract v2 encoder_out")?
                    .into_owned();
                let shape = v.shape().to_vec();
                if shape.len() == 3 {
                    for t in 0..shape[1] {
                        let row: Vec<f32> = (0..shape[2]).map(|d| v[[0, t, d]]).collect();
                        rows.push(row);
                    }
                }
            }
        }
        Ok(rows)
    }
}

// -- v2 streaming/offline engine --

/// One v2 trigger (a local peak accepted by the decision layer)
#[derive(Clone, Debug)]
pub struct Trigger {
    pub start_sec: f32,
    pub end_sec: f32,
    pub cmd: String,
    pub score: f64,
    /// 10 ms frame number of the peak (pf)
    pub t: usize,
}

/// Accepted candidates (including report-segment time; sec is snapshotted on acceptance so row trimming cannot invalidate it)
type Accepted = (f64, usize, usize, f32, f32);

/// Judged identity of one syllable nucleus. With sequence templates there is also emb = the L2-normalized mean embedding of the
/// identity window (same row-mean as pooled_cos), used for word-instance sequence scoring (shared by replay finish_nuclei and live staging).
struct NucJudged {
    pf: usize,  // Peak frame (absolute frame number on the 10 ms grid)
    cmd: usize, // Command chosen by identity scoring (index into cents/cmds)
    score: f64, // Identity-window pooled_cos (already above min_conf)
    start: f32, // Report segment start (τ−0.04)
    end: f32,   // Report segment end (τ+0.04)
    emb: Vec<f32>,
}

/// Start index of the trailing same-command run (the list preserves order; returns 0 when empty).

/// Grouping of a single-command run within the cap bound (the per-run body of merge_burst_groups; the emit_word_instance
/// fallback passes the cap straight in -- **never pass `&[cap]` as the per-cmd table**: indexing by cmd would pick the wrong entry).
/// Run precondition: non-empty and all the same command. Group score = max, start/t = first nucleus, end = last nucleus.
fn merge_run_groups(run: &[NucJudged], cap: usize) -> Vec<Accepted> {
    let mut out = Vec::new();
    let cmd = run[0].cmd;
    let mut start = 0usize;
    while start < run.len() {
        // Greedy grouping: each group takes exactly min(remaining, cap) consecutive nuclei (same semantics as element-wise advancement)
        let end = (start + cap).min(run.len());
        // Group score = max (fold form; equivalent to element-wise comparison)
        let best = run[start..end]
            .iter()
            .fold(f64::NEG_INFINITY, |acc, n| acc.max(n.score));
        out.push((best, run[start].pf, cmd, run[start].start, run[end - 1].end));
        start = end;
    }
    out
}

/// Cosine of two L2-normalized f32 vectors (defensive re-normalization: embeddings come from l2_normalize, this is a backstop).
fn cosine_f32(a: &[f32], b: &[f32]) -> f64 {
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += *x as f64 * *y as f64;
        na += *x as f64 * *x as f64;
        nb += *y as f64 * *y as f64;
    }
    let d = na.sqrt() * nb.sqrt();
    if d > 0.0 { dot / d } else { -1.0 }
}

/// band-DTW (Sakoe-Chiba |i−j| ≤ band) alignment score: mean cosine along the optimal path (per-frame cosine of f32 embeddings,
/// 2-6 rows per nucleus or character, negligible cost). No feasible path (length gap exceeds the band) -> None.
fn dtw_align_score(a: &[Vec<f32>], b: &[Vec<f32>], band: usize) -> Option<f64> {
    let (n, m) = (a.len(), b.len());
    if n == 0 || m == 0 || n.max(m).saturating_sub(n.min(m)) > band {
        return None;
    }
    // dp[i][j] = (path cosine sum, path length), NEGINF = unreachable; the optimal mean compares sum/length
    let mut sum = vec![vec![f64::NEG_INFINITY; m]; n];
    let mut len = vec![vec![0usize; m]; n];
    for i in 0..n {
        for j in 0..m {
            if i.abs_diff(j) > band {
                continue;
            }
            let c = cosine_f32(&a[i], &b[j]);
            if i == 0 && j == 0 {
                sum[i][j] = c;
                len[i][j] = 1;
                continue;
            }
            let mut bs = 0.0f64;
            let mut bl = 0usize;
            for (pi, pj) in [
                (i.checked_sub(1), j.checked_sub(1)),
                (i.checked_sub(1), Some(j)),
                (Some(i), j.checked_sub(1)),
            ] {
                let (Some(pi), Some(pj)) = (pi, pj) else {
                    continue;
                };
                if sum[pi][pj] == f64::NEG_INFINITY {
                    continue;
                }
                if bl == 0 || sum[pi][pj] / len[pi][pj] as f64 > bs / bl as f64 {
                    bs = sum[pi][pj];
                    bl = len[pi][pj];
                }
            }
            if bl > 0 {
                sum[i][j] = bs + c;
                len[i][j] = bl + 1;
            }
        }
    }
    let (s, l) = (sum[n - 1][m - 1], len[n - 1][m - 1]);
    (l > 0).then_some(s / l as f64)
}

fn seq_score_instance(
    emb: &[Vec<f32>],
    seq: &CmdSeqTemplate,
    min_conf: f64,
    force_dtw: bool,
) -> Option<(f64, &'static str, Option<f64>, Option<f64>)> {
    let mut fast: Option<f64> = None;
    let mut dtw: Option<f64> = None;
    for r in &seq.recs {
        if r.len() == emb.len() && emb.len() >= 2 {
            let mut sum = 0.0f64;
            let mut ok = true;
            for (e, t) in emb.iter().zip(r.iter()) {
                let c = cosine_f32(e, t);
                if c < min_conf {
                    ok = false;
                    break;
                }
                sum += c;
            }
            if ok {
                let s = sum / emb.len() as f64;
                if fast.map_or(true, |x| s > x) {
                    fast = Some(s);
                }
            }
        }
        if let Some(s) = dtw_align_score(emb, r, 1) {
            if dtw.map_or(true, |x| s > x) {
                dtw = Some(s);
            }
        }
    }
    let chosen = if force_dtw {
        dtw.map(|s| (s, "dtw"))
    } else {
        fast.map(|s| (s, "fast"))
            .or_else(|| dtw.map(|s| (s, "dtw")))
    };
    chosen.map(|(s, mode)| (s, mode, fast, dtw))
}

/// Emission context (shared by live staging and close-flush; replay calls emit_word_instance directly).
struct EmitCtx<'a> {
    exp: &'a [usize],
    seq: &'a [Option<CmdSeqTemplate>],
    min_conf: f64,
    force_dtw: bool,
    diag: bool,
    cmds: &'a [String],
}

fn emit_word_instance(
    run: &[NucJudged],
    seq: Option<&CmdSeqTemplate>,
    exp: usize,
    min_conf: f64,
    force_dtw: bool,
    diag: bool,
    cmd_name: &str,
) -> Vec<Accepted> {
    if let Some(seq) = seq {
        let emb: Vec<Vec<f32>> = run.iter().map(|n| n.emb.clone()).collect();
        if let Some((s, mode, f, d)) = seq_score_instance(&emb, seq, min_conf, force_dtw) {
            if diag {
                eprintln!(
                    "[v2-seq] {cmd_name} k={} n={} fast={:?} dtw={:?} → {} {s:.4}",
                    run.len(),
                    seq.n,
                    f,
                    d,
                    mode
                );
            }
            if s >= min_conf {
                return vec![(
                    s,
                    run[0].pf,
                    run[0].cmd,
                    run[0].start,
                    run[run.len() - 1].end,
                )];
            }
            if diag {
                eprintln!(
                    "[v2-seq] {cmd_name} k={} n={} below threshold -> falling back to grouping",
                    run.len(),
                    seq.n
                );
            }
        } else if diag {
            eprintln!(
                "[v2-seq] {cmd_name} k={} n={} no feasible alignment -> falling back to grouping",
                run.len(),
                seq.n
            );
        }
    }
    merge_run_groups(run, exp)
}

fn flush_stage_run(
    ctx: &EmitCtx,
    run: &[NucJudged],
    accepted: &mut Vec<Accepted>,
    refr_pf: usize,
    refr_until_pf: &mut usize,
) {
    if run.is_empty() {
        return;
    }
    let cmd = run[0].cmd;
    for a in emit_word_instance(
        run,
        ctx.seq.get(cmd).and_then(|s| s.as_ref()),
        ctx.exp.get(cmd).copied().unwrap_or(1),
        ctx.min_conf,
        ctx.force_dtw,
        ctx.diag,
        ctx.cmds.get(cmd).map(String::as_str).unwrap_or("?"),
    ) {
        let pf = a.1;
        if pf >= *refr_until_pf {
            *refr_until_pf = pf + refr_pf;
            accepted.push(a);
        } else {
            *refr_until_pf = (*refr_until_pf).max(pf + refr_pf);
            if ctx.diag {
                eprintln!(
                    "[nlive-refr] pf={pf} cmd={} swallowed within watermark {}",
                    ctx.cmds.get(a.2).map(String::as_str).unwrap_or("?"),
                    *refr_until_pf
                );
            }
        }
    }
}

/// Staging append for the grouped path: merge a judged nucleus into the (burst_seq, nucleus list) staging buffer and evaluate
/// emission in place at three moments (**count-equivalent** to replay's full grouping, but emitted earlier thanks to causality):
/// a) the new nucleus has a different command than the trailing run -> the earlier segment cannot grow -> evaluate and emit;
/// b) the trailing run reaches the expected nucleus count (the evaluation point) -> evaluate and emit (later nuclei open a new group);
/// c) the burst is closed and all its nuclei are judged -> the caller emits (end of advance_nuclei_live).
/// Invariant: the staging list is always a single same-command run (moments a and b guarantee it).
/// Evaluation = emit_word_instance (a sequence template passing its threshold wins; otherwise the grouped fallback) -- for a
/// single-nucleus or single-character command with exp=1 every nucleus fires immediately, matching the old per-nucleus behaviour.
fn stage_append(
    ctx: &EmitCtx,
    stage: &mut Option<(u64, Vec<NucJudged>)>,
    bseq: u64,
    n: NucJudged,
    accepted: &mut Vec<Accepted>,
    refr_pf: usize,
    refr_until_pf: &mut usize,
) {
    let list = match stage {
        Some((sid, list)) if *sid == bseq => list,
        Some((sid, list)) => {
            // First nucleus of a new burst arrives: the old burst's leftovers must already be judged (pending_ident preserves order) -> evaluate and emit
            flush_stage_run(ctx, list, accepted, refr_pf, refr_until_pf);
            *sid = bseq;
            list.clear();
            list
        }
        None => {
            *stage = Some((bseq, Vec::new()));
            &mut stage.as_mut().unwrap().1
        }
    };
    if !list.is_empty() && list.last().map(|x| x.cmd) != Some(n.cmd) {
        // a) command switch: the previous run is cut off -> evaluate and emit immediately
        let taken = std::mem::take(list);
        flush_stage_run(ctx, &taken, accepted, refr_pf, refr_until_pf);
    }
    list.push(n);
    let cmd = list.last().unwrap().cmd;
    let exp = ctx.exp.get(cmd).copied().unwrap_or(1).max(1);
    if list.len() >= exp {
        // b) the expected nucleus count is reached -> evaluate and emit
        let taken = std::mem::take(list);
        flush_stage_run(ctx, &taken, accepted, refr_pf, refr_until_pf);
    }
}

fn phrase_syllable_upper(model_dir: &str, cmd: &str) -> usize {
    let p = std::path::Path::new(model_dir).join("commands.json");
    let n = std::fs::read_to_string(&p)
        .ok()
        .and_then(|txt| serde_json::from_str::<serde_json::Value>(&txt).ok())
        .and_then(|v| v.get("commands").and_then(|c| c.get(cmd)).cloned())
        .and_then(|c| c.get("phrase").and_then(|x| x.as_str()).map(String::from))
        .map(|s| {
            s.split_whitespace()
                .map(|w| w.chars().count())
                .sum::<usize>()
        })
        .unwrap_or(0);
    n.max(1)
}

pub struct Engine {
    enc: Encoder256,
    p: EngineParams,
    cmds: Vec<String>,
    streaming: bool,

    // -- mel frame stream (encoder input) --
    mel: Vec<Vec<f32>>, // Unconsumed mel frames (80-dim); origin = mel_base
    mel_base: usize,    // Absolute frame number of mel[0]
    mel_total: usize,   // Total absolute frames fed in
    next_chunk: usize,  // Absolute frame where the next block starts (step hop=8)

    // -- energy activity (10 ms frame grid) --
    act_frames: Vec<bool>,
    /// Absolute frame number of act_frames[0] (re-based after head trimming on the nuclei live path).
    act_base: usize,
    /// Raw sample buffer (shared by the magnitude-spectrum front end and activity detection; origin = sample_base)
    sample_buf: Vec<f32>,
    sample_base: usize,
    act_fed: usize, // Frames whose activity flag is computed
    mel_fed: usize, // mel frames extracted (= mel_total)
    act_noise_floor: Option<f64>,

    accepted: Vec<Accepted>,
    emitted: usize, // Number of accepted entries already emitted
    /// Nuclei rejected by the identity layer for min_conf (consumed by the home-page summary row)
    rejected: u64,

    /// Syllable-nucleus counter state (None for a zero-command engine)
    nuc: Option<NucleiState>,

    auto_gate_on: bool,
    /// Frame number of the next tick (10 ms grid, driven by the audio clock; replay-live reproduces it offline).
    auto_next_frame: usize,
    auto_adjusted: bool,
}

fn nuclei_diag_on() -> bool {
    debug_log_on()
}

/// Diagnostics: `VIQ_DIAG_IDENT=1` enables per-nucleus identity logging on the live path ([nlive-ident] best/second
/// against min_conf), which demonstrates the "nucleus counted but identity rejects it" class of misses. Off by default.
fn diag_ident_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("VIQ_DIAG_IDENT")
            .map(|v| v == "1")
            .unwrap_or(false)
    })
}

static DEBUG_LOG: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

pub fn set_debug_log(on: bool) {
    DEBUG_LOG.store(on, std::sync::atomic::Ordering::Relaxed);
}

pub fn debug_log_on() -> bool {
    DEBUG_LOG.load(std::sync::atomic::Ordering::Relaxed)
        || std::env::var("VIQ_DEBUG_LOG")
            .map(|v| v == "1")
            .unwrap_or(false)
}

/// Start-up parameter override: the four settings-page values are passed explicitly into the engine instead of relying on process env.
/// `None` = keep the env / built-in default (internal construction paths).
#[derive(Clone, Copy, Default, Debug)]
pub struct EngineOverrides {
    pub refr_s: Option<f64>,
    pub min_conf: Option<f64>,
    pub act_db: Option<f64>,
    pub auto_gate: Option<bool>,
}

impl EngineOverrides {
    /// Write into an already constructed EngineParams (out-of-range values are clamped the same way as the settings page).
    pub(crate) fn apply_to(&self, p: &mut EngineParams) {
        if let Some(v) = self.refr_s {
            p.refr_s = v.clamp(0.0, 2.0);
        }
        if let Some(v) = self.min_conf {
            p.min_conf = v.clamp(0.0, 0.9);
        }
        if let Some(v) = self.act_db {
            p.act_db = v.clamp(-75.0, -15.0);
        }
    }
}

impl Engine {
    /// Build from the model directory: read the embed_proto_v2.json template marker and load the encoder.
    /// `cmd_names`: enabled command names from commands.json (intersected with the marker keys);
    /// an empty slice means all marker keys (used by internal construction).
    pub fn from_model_dir(model_dir: &str, cmd_names: &[String]) -> anyhow::Result<Self> {
        Self::from_model_dir_with(model_dir, cmd_names, &EngineOverrides::default())
    }

    /// Production entry: overrides the settings-page parameters explicitly, so rebuilding the engine applies them.
    pub fn from_model_dir_with(
        model_dir: &str,
        cmd_names: &[String],
        ov: &EngineOverrides,
    ) -> anyhow::Result<Self> {
        let path = std::path::Path::new(model_dir).join("embed_proto_v2.json");
        let raw: HashMap<String, HashMap<String, Vec<f32>>> = if !path.exists() {
            eprintln!("[engine] no pronunciation templates yet; recognition starts once pronunciations are registered");
            HashMap::new()
        } else {
            let content = crate::crypto::read_text_auto(&path).with_context(|| {
                format!(
                    "failed to read embed_proto_v2.json: {} (run gen-embed-proto-v2 first)",
                    path.display()
                )
            })?;
            serde_json::from_str(&content).with_context(|| "failed to parse embed_proto_v2.json")?
        };
        let mut enc = Encoder256::load_best(model_dir)
            .with_context(|| format!("failed to load the v2 encoder (model_dir={model_dir})"))?;
        // Template commands = file keys ∩ enabled commands (empty = whole file, name-sorted for determinism)
        let wanted: Vec<String> = if cmd_names.is_empty() {
            let mut all: Vec<String> = raw.keys().cloned().collect();
            all.sort();
            all
        } else {
            cmd_names.to_vec()
        };
        let mut cmds: Vec<String> = Vec::new();
        for name in &wanted {
            if raw.contains_key(name) {
                cmds.push(name.clone());
            }
        }
        if cmds.is_empty() && raw.is_empty() {
            // v2-only first run with no commands: a missing template file or an empty object {} still constructs a 0-command engine
            // safely (gate and UI handle zero commands naturally); registering pronunciations (gen_embed_proto*) applies on reload.
        } else if cmds.is_empty() {
            let mut have: Vec<String> = raw.keys().cloned().collect();
            have.sort();
            anyhow::bail!(
                "embed_proto_v2.json has no usable command template (none of the enabled commands {wanted:?} appears in the covered set {have:?};\
                 usually the recordings are not classified per command or the templates were never generated -- regenerate that command's template on the Commands tab,\
                 or run gen-embed-proto-v2 for a full rebuild)"
            );
        }
        let mut p = EngineParams::default();
        ov.apply_to(&mut p);
        let nuc = if cmds.is_empty() {
            None
        } else {
            let cr_dir = std::path::Path::new(model_dir).join("command_recordings");
            // Expected nucleus count for sequence templates: with n >= 2 use the
            // median nucleus count per recording (closer to real segmentation); n == 1 or no sequence -> fall back to the phrase syllable count
            // (phrase_syllable_upper, missing -> 1 = no merging, the conservative direction).
            let phrase_cap: Vec<usize> = cmds
                .iter()
                .map(|c| phrase_syllable_upper(model_dir, c))
                .collect();
            // "Multi-pronunciation" explicit override: None = derive from templates, Some(false) = force a single nucleus
            let multi_ov: Vec<Option<bool>> = cmds
                .iter()
                .map(|c| crate::config::multi_override(model_dir, c))
                .collect();
            let built = crate::nuclei_cache::load_or_build(
                &mut enc, &cr_dir, &cmds, model_dir, p.k, p.act_db,
            )
            .map(|(cents, prior, _med_rows, seqs, exps, _stats)| {
                let exp_nuclei = cmds
                    .iter()
                    .enumerate()
                    .map(|(i, _)| {
                        let ph = phrase_cap[i];
                        let auto = if exps[i] >= 2 && !seqs[i].recs.is_empty() {
                            ph.max(exps[i])
                        } else if exps[i] >= 1 {
                            exps[i]
                        } else {
                            ph.max(1)
                        };
                        crate::config::apply_multi_override(auto, multi_ov[i])
                    })
                    .collect();
                let seq_tmpl = seqs
                    .into_iter()
                    .enumerate()
                    .map(|(i, s)| {
                        (multi_ov[i] != Some(false)
                            && phrase_cap[i] >= 2
                            && s.n >= 2
                            && !s.recs.is_empty())
                        .then(|| s)
                    })
                    .collect();
                NucleiState {
                    p: NucleiParams::from_v2(&p),
                    cents,
                    st8: St8Accum::new(),
                    samples: Vec::new(),
                    char_dur_prior: prior,
                    single_durs: Vec::new(),
                    live: NucleiLiveState::new(cmds.len()),
                    exp_nuclei,
                    seq_tmpl,
                }
            });
            enc.reset_state();
            let n = built.with_context(|| {
                format!("failed to build the syllable-nucleus templates (model_dir={model_dir})")
            })?;
            eprintln!(
                "[engine] syllable-nucleus counting enabled: commands {:?} syllable-length prior {:.2}s (offline finish is a full pass; live is causal)",
                cmds, n.char_dur_prior
            );
            Some(n)
        };
        eprintln!(
            "[engine] engine enabled: encoder={} out_dim={} K={} min_conf={} refr={}s gate>={} commands {:?}",
            enc.src_path
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("?"),
            enc.out_dim,
            p.k,
            p.min_conf,
            p.refr_s,
            p.act_db,
            cmds
        );
        let auto_gate_on = ov.auto_gate.unwrap_or_else(|| {
            std::env::var("VIQ_ACT_GATE_AUTO")
                .map(|v| v == "1")
                .unwrap_or(false)
        });
        let auto_adjusted = false;
        Ok(Engine {
            enc,
            p,
            cmds,
            streaming: true,
            mel: Vec::new(),
            mel_base: 0,
            mel_total: 0,
            next_chunk: 0,
            act_frames: Vec::new(),
            act_base: 0,
            act_noise_floor: None,
            sample_buf: Vec::new(),
            sample_base: 0,
            act_fed: 0,
            mel_fed: 0,
            accepted: Vec::new(),
            emitted: 0,
            rejected: 0,
            nuc,
            auto_gate_on,
            auto_next_frame: 0,
            auto_adjusted,
        })
    }

    /// Hot-update settings-page parameters at runtime: written straight into the running engine, no rebuild, no interrupted keypresses.
    pub fn apply_live_tuning(&mut self, ov: &EngineOverrides) {
        ov.apply_to(&mut self.p);
        if let Some(nuc) = self.nuc.as_mut() {
            if let Some(v) = ov.min_conf {
                nuc.p.min_conf = v.clamp(0.0, 0.9);
            }
            if let Some(v) = ov.act_db {
                nuc.p.act_db = v.clamp(-75.0, -15.0);
            }
            if let Some(v) = ov.refr_s {
                nuc.p.refr_pf = refr_pf_from_secs(v);
            }
        }
        if let Some(b) = ov.auto_gate {
            self.auto_gate_on = b;
        }
    }

    /// live (true) / offline (false) mode; tests set offline mode via set_replay_mode.
    pub fn set_streaming(&mut self, v: bool) {
        self.streaming = v;
    }

    pub fn grid(&self) -> (usize, usize) {
        (self.enc.chunk_frames, 8)
    }
    pub fn is_streaming(&self) -> bool {
        self.streaming
    }
    pub fn cmds(&self) -> &[String] {
        &self.cmds
    }

    /// Nuclei rejected by the identity layer (consumed by the home-page summary row).
    pub fn rejected_nuclei(&self) -> u64 {
        self.rejected
    }

    /// Current activity gate (consumed by the settings page).
    pub fn gate(&self) -> f64 {
        self.p.act_db
    }

    pub fn diag_summary(&self) -> String {
        let Some(nuc) = &self.nuc else {
            return "no nuclei".into();
        };
        let live = &nuc.live;
        let nf = live
            .noise_floor
            .map(|v| format!("{:.1}", v))
            .unwrap_or("uninitialised".into());
        let burst = live
            .burst
            .as_ref()
            .map(|b| {
                format!(
                    "open(a10={} n={} counted={})",
                    b.a10, b.n_nuclei, b.counted_upto
                )
            })
            .unwrap_or("none".into());
        let gate_val = live
            .noise_floor
            .map(|v| format!("{:.1}", v + 6.0))
            .unwrap_or_else(|| format!("{:.1}", self.p.act_db)); // Fixed or automatic gate: show the gate currently in force
        format!(
            "frame={} gate={} nf={} burst={} nuclei={} last_active={}",
            live.st8_seen, gate_val, nf, burst, live.burst_seq, live.last_act_frame
        )
    }

    pub fn push_samples(&mut self, samples: &[f32]) {
        // nuclei path: offline accumulates all samples (filtfilt is non-causal, computed once at finish);
        // live uses a forward causal SOS filter -> a 10 ms RMS dB envelope ring (same frame grid as act_frames)
        if let Some(nuc) = self.nuc.as_mut() {
            if self.streaming {
                for &s in samples {
                    let y = nuc.live.sos.push(s as f64) as f32;
                    nuc.live.acc += y * y;
                    nuc.live.acc_n += 1;
                    if nuc.live.acc_n == 160 {
                        let rms = (nuc.live.acc / 160.0f32).sqrt() + 1e-9;
                        let db = 20.0 * (rms as f64).log10();
                        nuc.live.env.push(db);
                        let f = nuc.live.env_base + nuc.live.env.len() - 1;
                        let gate = act_gate_step(&mut nuc.live.noise_floor, db, &self.p);
                        let active = db > gate;
                        if active {
                            nuc.live.last_act_frame = f;
                        }
                        nuc.live.acc = 0.0;
                        nuc.live.acc_n = 0;
                        // Ring trimming: only the part the counter has consumed is trimmed (keeping 4 frames of smoothing lookback), capped at ~32 s
                        if nuc.live.env.len() > 3200 {
                            let consumed = nuc
                                .live
                                .burst
                                .as_ref()
                                .map(|b| b.counted_upto)
                                .unwrap_or(nuc.live.env_base + nuc.live.env.len());
                            let drop = nuc.live.env.len() - 3000;
                            let drop = drop.min(consumed.saturating_sub(nuc.live.env_base + 4));
                            if drop > 0 {
                                nuc.live.env.drain(..drop);
                                nuc.live.env_base += drop;
                            }
                        }
                    }
                }
                if self.auto_gate_on {
                    let end = nuc.live.env_base + nuc.live.env.len();
                    if end >= self.auto_next_frame {
                        self.auto_next_frame = end + crate::auto_gate::CHUNK;
                        let aggressive = !self.auto_adjusted;
                        let tail_len = nuc.live.env.len().min(crate::auto_gate::LOOKBACK_MAX);
                        let tail = &nuc.live.env[nuc.live.env.len() - tail_len..];
                        let d = crate::auto_gate::tick(
                            tail,
                            self.p.act_db,
                            aggressive,
                        );
                        if debug_log_on() {
                            eprintln!(
                                "[auto-gate] end={end} gate={:.1} -> {:?} nf={:?} reason={}",
                                self.p.act_db, d.new_gate, d.nf_est, d.reason
                            );
                        }
                        if let Some(g) = d.new_gate {
                            self.p.act_db = g;
                            self.auto_adjusted = true;
                        }
                    }
                }
            } else {
                nuc.samples.extend_from_slice(samples);
            }
        }
        self.sample_buf.extend_from_slice(samples);
        let total = self.sample_base + self.sample_buf.len();
        // Activity frames (10 ms grid; matching window_active_frac's RMS dB criterion)
        while (self.act_fed + 1) * 160 <= total {
            let off = self.act_fed * 160 - self.sample_base;
            let seg = &self.sample_buf[off..off + 160];
            let rms = (seg.iter().map(|s| s * s).sum::<f32>() / 160.0).sqrt() + 1e-9;
            let db = 20.0 * (rms as f64).log10();
            let gate = act_gate_step(&mut self.act_noise_floor, db, &self.p);
            let active = db > gate;
            self.act_frames.push(active);
            self.act_fed += 1;
        }
        // mel frames (25 ms window, 10 ms hop; same frame order as activity)
        while self.mel_fed * 160 + 400 <= total {
            let off = self.mel_fed * 160 - self.sample_base;
            let frame = self.sample_buf[off..off + 400].to_vec();
            self.mel.push(v2_amp_frame(&frame));
            self.mel_fed += 1;
        }
        self.mel_total = self.mel_fed;
        // Trim the consumed sample head (up to the slower of the act/mel cursors)
        let keep = self.act_fed.min(self.mel_fed) * 160 - self.sample_base;
        if keep > 48000 {
            self.sample_buf.drain(..keep);
            self.sample_base += keep;
        }
        self.pump(false);
    }

    /// Encoder pump: push one block once a full 29-frame chunk is available (step 8); at final, zero-pad the tail (processed_lens).
    fn pump(&mut self, final_flush: bool) {
        loop {
            let avail = self.mel_total;
            if self.next_chunk + self.enc.chunk_frames <= avail {
                let s = self.next_chunk - self.mel_base;
                let mut chunk = Array2::<f32>::zeros((self.enc.chunk_frames, MEL_BINS));
                for i in 0..self.enc.chunk_frames {
                    let row = &self.mel[s + i];
                    for b in 0..MEL_BINS {
                        chunk[[i, b]] = row[b];
                    }
                }
                let start = self.next_chunk;
                self.run_chunk_store(&chunk, self.enc.chunk_frames as i64, start);
                self.next_chunk += 8; // fixed hop = 8
            } else if final_flush && self.next_chunk < avail && avail >= 10 {
                while self.next_chunk < avail {
                    let n = (avail - self.next_chunk).min(self.enc.chunk_frames);
                    let s = self.next_chunk - self.mel_base;
                    let mut chunk = Array2::<f32>::zeros((self.enc.chunk_frames, MEL_BINS));
                    for i in 0..n {
                        let row = &self.mel[s + i];
                        for b in 0..MEL_BINS {
                            chunk[[i, b]] = row[b];
                        }
                    }
                    let start = self.next_chunk;
                    self.run_chunk_store(&chunk, n as i64, start);
                    self.next_chunk += 8; // fixed hop = 8
                }
            } else {
                break;
            }
            // mel trimming: the step (8) exceeds the in-block coverage (4), so consumed frames are never needed again
            let drop = self.next_chunk - self.mel_base;
            if drop > 4096 {
                self.mel.drain(..drop.min(self.mel.len()));
                self.mel_base += drop;
            }
        }
    }

    /// One block of inference plus row storage (rows whose idx exceeds the fed frame count are dropped)
    fn run_chunk_store(
        &mut self,
        chunk: &Array2<f32>,
        proc_len: i64,
        chunk_start: usize,
    ) {
        let rows = match self.enc.run_chunk(chunk, proc_len) {
            Ok(rows) => rows,
            Err(e) => {
                static RUN_CHUNK_FAILS: std::sync::atomic::AtomicUsize =
                    std::sync::atomic::AtomicUsize::new(0);
                let n = RUN_CHUNK_FAILS.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                if n == 1 || n % 100 == 0 {
                    eprintln!(
                        "[engine] encoder inference failed (attempt {n}): {e:#} -- this block's rows are missing, affecting syllable-nucleus counting"
                    );
                }
                return;
            }
        };
        if let Some(nuc) = self.nuc.as_mut()
            && chunk_start % 8 == 0
        {
            let end = chunk_start + (proc_len.max(0) as usize);
            nuc.st8.feed_chunk(chunk_start / 8, rows, end);
        }
    }

    fn housekeeping_nuclei_live(&mut self) {
        const DROP_ST8_ROWS: usize = 4000; // 80 ms rows ≈ 320 s (trimmed roughly every 3 min)
        const KEEP_ACT_FRAMES: usize = 3000; // 30 s at 100 Hz (60x headroom over the gating window and the burst walk-back)
        // 1) st8 (80 ms rows)
        if let Some(nuc) = self.nuc.as_mut() {
            let w = nuc.p.curve_win_rows;
            // Left edge of the curve window for the next unconsumed row g = st8_seen (absolute row)
            let curve_need = (nuc.live.st8_seen + 1).saturating_sub(w);
            // Left edge of the window for the oldest unjudged nucleus (absolute row; the queue head is oldest, preserving order)
            let ident_need = nuc.live.pending_ident.first().map(|&(pf, _, _)| {
                round_half_even((pf as f64 - nuc.p.ident_back_sec - 0.04) / 0.08).max(0) as usize
            });
            let keep_abs = ident_need.unwrap_or(curve_need).min(curve_need);
            let drop = keep_abs.saturating_sub(nuc.st8.head_base);
            if drop > DROP_ST8_ROWS {
                nuc.st8.trim_head(drop);
            }
        }
        // 2) act_frames head trimming
        trim_act_head(&mut self.act_frames, &mut self.act_base, KEEP_ACT_FRAMES);
    }

    pub fn finish(&mut self) {
        self.pump(true);
        if self.nuc.is_none() {
            return;
        }
        if self.streaming {
            self.advance_nuclei_live(true);
            return;
        }
        if let Err(e) = self.finish_nuclei() {
            eprintln!("[engine] syllable-nucleus counting finalisation failed: {e:#}");
        }
    }

    /// Syllable-nucleus counting, final form: 80 ms plateau -> energy burst -> per-nucleus F1 envelope counting + per-nucleus identity -> trigger.
    fn finish_nuclei(&mut self) -> anyhow::Result<()> {
        // Snapshot the read-only self data first (avoiding a conflict with the mutable borrow of nuc)
        let act_frames: Vec<bool> = self.act_frames.clone();
        let cmds: Vec<String> = self.cmds.clone();
        let nuc = self.nuc.as_mut().context("nuclei state not initialised")?;
        nuc.st8.finish();
        let n = nuc.st8.rows.len();
        let w = nuc.p.curve_win_rows;
        anyhow::ensure!(n >= w && !nuc.samples.is_empty(), "not enough rows8/samples (n={n})");
        anyhow::ensure!(nuc.cents.len() == cmds.len(), "centroid count does not match the command count");
        // Activity fraction per group (an 80 ms group = mean of 8 ten-millisecond activity frames)
        let act_row: Vec<f64> = (0..n)
            .map(|g| {
                let f0 = g * 8;
                let mut s = 0.0;
                for j in 0..8 {
                    let f = f0 + j;
                    if f < act_frames.len() && act_frames[f] {
                        s += 1.0;
                    }
                }
                s / 8.0
            })
            .collect();
        // Per-command curves plus floors (floor_track_init: first segment −Δ plus silence decay)
        let mut curves: Vec<Vec<f64>> = Vec::with_capacity(nuc.cents.len());
        let mut floors: Vec<Vec<f64>> = Vec::with_capacity(nuc.cents.len());
        for cent in nuc.cents.iter() {
            let curve = plateau_curve(&nuc.st8.cs, &act_row, w, nuc.p.gate_frac, cent);
            let valid: Vec<bool> = curve.iter().map(|&v| v > -9.0e8).collect();
            floors.push(floor_track_nuclei(
                &curve,
                &valid,
                nuc.p.floor_p,
                nuc.p.tau_rows,
                nuc.p.delta_plateau,
                nuc.p.sil_drop,
                nuc.p.a_sil,
            ));
            curves.push(curve);
        }
        // F1 envelope (zero-phase band-pass -> RMS dB -> smoothing)
        let f1: Vec<f32> = sos_filt_filt(&nuc.samples)
            .iter()
            .map(|&v| v as f32)
            .collect();
        let env_db = energy_db_env(&f1);
        // Speaking-rate adaptation: rolling median duration of confirmed single-character plateaus (1-nucleus bursts),
        // effective from >= 2 samples; a same-command plateau gap below 1× the median character length counts as the same
        // plateau continuing. Cold start or too little evidence -> keep the validated 0.30 s.
        // Two passes: pass A merges the baseline and counts nuclei -> estimate the character length -> pass B recomputes
        // the bursts with the same-command merge rule (skipped when the threshold is unchanged).
        let mut single_durs: Vec<f64> = nuc.single_durs.clone();
        let compute_bursts = |gap_same_cmd: f64| -> Vec<(usize, usize)> {
            let mut union = vec![false; n];
            for ci in 0..cmds.len() {
                let mask: Vec<bool> = (0..n)
                    .map(|t| {
                        curves[ci][t] > floors[ci][t] + nuc.p.delta_plateau
                            && curves[ci][t] >= nuc.p.min_conf
                    })
                    .collect();
                let regs =
                    regions_from_mask(&mask, 0.08, nuc.p.merge_gap_sec, nuc.p.min_plateau_sec);
                // Same-command continuation merge (in rows: gap_sec / 0.08 s per row)
                let max_hole_rows = (gap_same_cmd / 0.08).floor() as usize;
                let mut merged: Vec<(usize, usize)> = Vec::new();
                for (a, b) in regs {
                    if let Some(last) = merged.last_mut() {
                        if a - last.1 <= max_hole_rows {
                            last.1 = b;
                            continue;
                        }
                    }
                    merged.push((a, b));
                }
                for (a, b) in merged {
                    for m in union.iter_mut().skip(a).take(b - a) {
                        *m = true;
                    }
                }
            }
            let regs = regions_from_mask(&union, 0.08, nuc.p.merge_gap_sec, nuc.p.min_plateau_sec);
            // Plateau midpoint -> energy burst bounds (the sliding-window curve lags ~0.3 s, so using plateau bounds directly would cut off the first nucleus) -> merge overlaps
            let mut bursts: Vec<(usize, usize)> = Vec::new();
            for (a, b) in regs {
                let mid_frame = (a + b) * 4; // (a+b)/2 rows ×0.08 s ÷ 0.01 s per frame
                let (a10, b10) = burst_bounds(&act_frames, mid_frame, nuc.p.burst_gap_frames);
                if let Some(last) = bursts.last_mut() {
                    if a10 <= last.1 {
                        last.1 = last.1.max(b10);
                        continue;
                    }
                }
                bursts.push((a10, b10));
            }
            bursts
        };
        let count_peaks = |bursts: &[(usize, usize)], sds: &mut Vec<f64>| -> Vec<Vec<usize>> {
            let mut all = Vec::with_capacity(bursts.len());
            for &(a10, b10) in bursts {
                let (_, peaks) = count_nuclei(
                    &env_db,
                    a10,
                    b10,
                    nuc.p.thr_q,
                    nuc.p.rel_db,
                    nuc.p.valley_db,
                    nuc.p.min_dist_frames,
                    nuc.p.smooth_win,
                );
                if peaks.len() == 1 {
                    sds.push((b10 - a10) as f64 * 0.01);
                }
                all.push(peaks);
            }
            all
        };
        let gap0 = nuc.p.merge_gap_sec;
        let mut bursts = compute_bursts(gap0);
        let mut peaks_all = count_peaks(&bursts, &mut single_durs);
        let gap1 = median_of(&single_durs)
            .filter(|_| single_durs.len() >= 2)
            .map(|m| m.max(nuc.p.merge_gap_sec))
            .unwrap_or(gap0);
        if (gap1 - gap0).abs() > 1e-9 {
            // Speaking-rate adaptation active: same-command plateaus are merged by 1× the median character length and recomputed
            bursts = compute_bursts(gap1);
            peaks_all = count_peaks(&bursts, &mut single_durs);
            eprintln!("[nuclei] speaking-rate adaptation: median character length {gap1:.2}s -> recomputing with same-command plateau merging");
        }
        let diag = nuclei_diag_on();
        let force_dtw = false;
        let dim = nuc.cents.first().map(|c| c.len()).unwrap_or(0);
        let mut accepted: Vec<Accepted> = Vec::new();
        let mut seq_dbg = String::new();
        let mut n_nuc_pre = 0usize;
        for (bi, &(a10, b10)) in bursts.iter().enumerate() {
            let mut judged: Vec<NucJudged> = Vec::new();
            for &pf in &peaks_all[bi] {
                let tau = pf as f64 * 0.01;
                let g0 = round_half_even((tau - nuc.p.ident_back_sec - 0.04) / 0.08).max(0) as usize;
                let g1 = round_half_even((tau + nuc.p.ident_fwd_sec - 0.04) / 0.08).max(0) as usize + 1;
                let mut best = (-1.0f64, 0usize);
                let mut second = -1.0f64; // L0-b: second-best score (same form as the live path)
                for (ci, cent) in nuc.cents.iter().enumerate() {
                    let c = pooled_cos(&nuc.st8.cs, g0, g1, cent);
                    if c > best.0 {
                        second = best.0;
                        best = (c, ci);
                    } else if c > second {
                        second = c;
                    }
                }
                if best.0 < nuc.p.min_conf {
                    continue;
                }
                let emb = window_embedding(&nuc.st8.cs, g0, g1, dim).unwrap_or_default();
                judged.push(NucJudged {
                    pf,
                    cmd: best.1,
                    score: best.0,
                    start: (tau - 0.04) as f32,
                    end: (tau + 0.04) as f32,
                    emb,
                });
                if diag {
                    eprintln!(
                        "[v2-nuclei-diag] burst {bi} {:.2}~{:.2} peak {:.2}s id {} cos {:.4} second {:.4} margin {:.4}",
                        a10 as f64 * 0.01,
                        b10 as f64 * 0.01,
                        tau,
                        cmds[best.1],
                        best.0,
                        second,
                        best.0 - second
                    );
                }
            }
            n_nuc_pre += judged.len();
            // Within a burst, adjacent same-command nuclei are cut into runs -> per-run word-instance decision (sequence template first, grouped fallback)
            let mut k = 0usize;
            while k < judged.len() {
                let cmd = judged[k].cmd;
                let mut j = k;
                while j + 1 < judged.len() && judged[j + 1].cmd == cmd {
                    j += 1;
                }
                let before = accepted.len();
                // Emission-level refractory watermark (same rule as flush_stage_run): live and replay-live agree bit for bit
                for a in emit_word_instance(
                    &judged[k..=j],
                    nuc.seq_tmpl.get(cmd).and_then(|s| s.as_ref()),
                    nuc.exp_nuclei.get(cmd).copied().unwrap_or(1),
                    nuc.p.min_conf,
                    force_dtw,
                    diag,
                    &cmds[cmd],
                ) {
                    let pf = a.1;
                    if pf >= nuc.live.refr_until_pf {
                        nuc.live.refr_until_pf = pf + nuc.p.refr_pf;
                        accepted.push(a);
                    } else {
                        nuc.live.refr_until_pf = nuc.live.refr_until_pf.max(pf + nuc.p.refr_pf);
                    }
                }
                for a in &accepted[before..] {
                    seq_dbg.push(cmds[a.2].as_str().chars().next().unwrap_or('?'));
                }
                k = j + 1;
            }
        }
        if diag {
            for (bi, &(a10, b10)) in bursts.iter().enumerate() {
                eprintln!(
                    "[v2-nuclei-diag] burst {bi} {:.2}~{:.2} n={}",
                    a10 as f64 * 0.01,
                    b10 as f64 * 0.01,
                    peaks_all[bi].len()
                );
            }
        }
        eprintln!(
            "[nuclei] bursts {} -> nuclei {} ({} triggers after group merging) sequence {}",
            bursts.len(),
            n_nuc_pre,
            accepted.len(),
            seq_dbg
        );
        // Rolling single-character duration evidence (online rolling median)
        nuc.single_durs = single_durs;
        self.accepted = accepted;
        self.emitted = 0;
        Ok(())
    }

    pub fn poll(&mut self) -> Vec<Trigger> {
        if !self.streaming {
            return Vec::new();
        }
        if self.nuc.is_none() {
            // A zero-command engine still receives audio, so act_frames must still be trimmed to avoid unbounded growth.
            self.housekeeping_nuclei_live();
            return Vec::new();
        }
        self.advance_nuclei_live(false);
        self.housekeeping_nuclei_live();
        self.take_ready()
    }

    /// Drain ready triggers (ascending by t)
    pub fn take_ready(&mut self) -> Vec<Trigger> {
        let mut out = Vec::new();
        for &(s, t, c, st, en) in &self.accepted[self.emitted..] {
            out.push(Trigger {
                start_sec: st,
                end_sec: en,
                cmd: self.cmds[c].clone(),
                score: s,
                t,
            });
        }
        self.emitted = self.accepted.len();
        // accepted is only used for incremental output; on long sessions with many triggers, fully drained old entries are cleared.
        if self.accepted.len() > 1024 {
            self.accepted.clear();
            self.emitted = 0;
        }
        out
    }

    /// Live causal advance (streaming + nuclei only): incremental plateau rows (curves, gating, floor and mask are all
    /// computable row by row) -> burst opening (plateau start, expanded back by energy) -> envelope counter (rolling Q99 + valley
    /// confirmation, emitting as soon as the peak distance is exceeded) -> per-nucleus identity -> accepted. final_flush=true (finish) appends the tail rows and forces the burst closed.
    fn advance_nuclei_live(&mut self, final_flush: bool) {
        let Some(nuc) = self.nuc.as_mut() else { return };
        let act = &self.act_frames; // Disjoint field borrow with respect to nuc
        if final_flush {
            nuc.st8.finish();
        }
        let a_up = 1.0 - (-1.0 / nuc.p.tau_rows.max(1.0)).exp();
        let a_dn = (a_up * (1.0 - nuc.p.floor_p) / nuc.p.floor_p.max(1e-6)).min(1.0);
        let min_rows = ((nuc.p.min_plateau_sec / 0.08).ceil() as usize).max(2);
        // -- 1) new rows -> curves / gating / floor / mask -> runs -> burst opening --
        while nuc.live.st8_seen < nuc.st8.abs_len() {
            let g = nuc.live.st8_seen;
            // Re-base after head trimming: available absolute frames = act_base + act.len()
            if self.act_base + act.len() < 8 * (g + 1) {
                break; // Activity frames not ready yet (they advance before mel, so this does not happen in practice)
            }
            nuc.live.st8_seen += 1;
            // Same-command adaptive merge hole (rows): effective once the rolling median character length has >= 2 samples (same rule as the offline slow path)
            let gap = median_of(&nuc.single_durs)
                .filter(|_| nuc.single_durs.len() >= 2)
                .map(|m| m.max(nuc.p.merge_gap_sec))
                .unwrap_or(nuc.p.merge_gap_sec);
            let max_hole = (gap / 0.08).floor() as usize;
            for ci in 0..nuc.cents.len() {
                let w = nuc.p.curve_win_rows;
                let (mask, _cosv) = if g + 1 < w {
                    (false, INVALID)
                } else {
                    let hb = nuc.st8.head_base;
                    let cosv = pooled_cos(
                        &nuc.st8.cs,
                        (g + 1 - w).saturating_sub(hb),
                        (g + 1) - hb,
                        &nuc.cents[ci],
                    );
                    let mut s = 0.0f64;
                    for j in (g + 1 - w)..(g + 1) {
                        for f in (8 * j)..(8 * j + 8) {
                            // Re-base after head trimming: absolute frame number -> relative index
                            // (the trim base is always <= the gating window's left edge 8×(st8_seen+1−w), so this cannot go out of bounds)
                            if act
                                .get(f.saturating_sub(self.act_base))
                                .copied()
                                .unwrap_or(false)
                            {
                                s += 1.0;
                            }
                        }
                    }
                    let gate = s / (8 * w) as f64;
                    let valid = gate >= nuc.p.gate_frac;
                    // Floor increment (step-equivalent to floor_track_nuclei: valid-frame updates and silence decay)
                    let q = if valid {
                        let x = cosv;
                        let v = match nuc.live.floor_q[ci] {
                            None => x - nuc.p.delta_plateau,
                            Some(qv) => {
                                let a = if x >= qv { a_up } else { a_dn };
                                qv + a * (x - qv)
                            }
                        };
                        nuc.live.floor_last[ci] = Some(v);
                        Some(v)
                    } else {
                        nuc.live.floor_q[ci].map(|qv| {
                            let target = nuc.live.floor_last[ci].unwrap_or(qv) - nuc.p.sil_drop;
                            qv + nuc.p.a_sil * (target - qv)
                        })
                    };
                    nuc.live.floor_q[ci] = q;
                    let fl = q.unwrap_or(0.0);
                    let vmask = valid && cosv > fl + nuc.p.delta_plateau && cosv >= nuc.p.min_conf;
                    if nuclei_diag_on() {
                        eprintln!(
                            "[nlive-row] g={g} cmd={ci} cos={cosv:.4} gate={gate:.3} valid={valid} floor={fl:.4} mask={vmask}"
                        );
                    }
                    (vmask, cosv)
                };
                run_update(&mut nuc.live.runs[ci], g, mask, max_hole, min_rows);
            }
            // Burst opening: any command run reaches the minimum plateau duration (opening lags by one row so short runs never form a burst)
            if nuc.live.burst.is_none() {
                for r in nuc.live.runs.iter() {
                    if let Some((a, b)) = r.open {
                        if b - a + 1 >= min_rows {
                            // Re-base after head trimming: convert the burst start walk-back to a relative index
                            // and add the base back to restore the absolute frame number (an out-of-range walk-back is clamped to the trim base, and
                            // the actual counting start is further clamped by the envelope ring's env_base, so the semantics are unchanged)
                            let a10 = burst_start_causal(
                                act,
                                (a * 8).saturating_sub(self.act_base),
                                nuc.p.burst_gap_frames,
                            ) + self.act_base;
                            let diag = nuclei_diag_on();
                            if diag {
                                eprintln!(
                                    "[nlive] burst open a10={a10} (row {a}/{b})",
                                    a = a,
                                    b = b
                                );
                            }
                            let mut ctr = LiveCounter::new(
                                nuc.p.valley_db,
                                2.0,
                                nuc.p.min_dist_frames,
                                nuc.p.thr_q,
                                nuc.p.rel_db,
                                diag,
                            );
                            // 2) Counter stall-release parameters (see the NucleiParams comment)
                            ctr.absorb_max_frames = nuc.p.absorb_max_frames;
                            ctr.start();
                            // Session-level quantiles move into the counter with the burst (and back out on close); nucleus peak prominence takes over the configured value
                            ctr.sess_q = std::mem::take(&mut nuc.live.sess_q);
                            if nuc.p.hold_frames > 0 {
                                ctr.hold =
                                    Some((nuc.p.hold_frames, nuc.p.sup_db, nuc.p.hold_max_depth));
                            }
                            let start = a10.max(nuc.live.consumed_watermark);
                            nuc.live.burst_seq = nuc.live.burst_seq.wrapping_add(1);
                            let bseq = nuc.live.burst_seq;
                            nuc.live.burst = Some(LiveBurst {
                                counter: ctr,
                                a10: start,
                                counted_upto: start,
                                n_nuclei: 0,
                                seq: bseq,
                            });
                            break;
                        }
                    }
                }
            }
        }
        let env_end = nuc.live.env_base + nuc.live.env.len();
        let mut new_nuclei: Vec<(usize, f64)> = Vec::new();
        if let Some(bst) = nuc.live.burst.as_mut() {
            let lo = bst.a10.max(nuc.live.env_base);
            if bst.counted_upto < lo {
                bst.counted_upto = lo;
            }
            while bst.counted_upto + 2 <= env_end.saturating_sub(1) {
                let i = bst.counted_upto;
                let (a, b) = smooth_window_bounds(i, nuc.live.env_base);
                let x = nuc.live.env[a - nuc.live.env_base..=b - nuc.live.env_base]
                    .iter()
                    .sum::<f64>()
                    / ((b - a + 1) as f64);
                bst.counted_upto += 1;
                bst.counter.feed(i, x);
                new_nuclei.extend(bst.counter.emitted.drain(..));
            }
            bst.n_nuclei += new_nuclei.len();
        }
        // Burst close: all runs closed + activity silent for > 0.3 s (or finish / 2-stall timeout) -> emit pending candidates plus single-character duration evidence
        let any_open = nuc.live.runs.iter().any(|r| r.open.is_some());
        let burst_expired = nuc.live.burst.as_ref().map_or(false, |b| {
            nuc.p.burst_max_frames > 0
                && env_end >= 31
                && env_end - 1 >= b.a10 + nuc.p.burst_max_frames
        });
        let close_now = if nuc.live.burst.is_some() {
            if burst_expired {
                true
            } else {
                !any_open && {
                    let quiet = env_end >= 31
                        && nuc.live.last_act_frame + nuc.p.burst_gap_frames < env_end - 1;
                    final_flush || quiet
                }
            }
        } else {
            false
        };
        if close_now {
            let mut bst = nuc.live.burst.take().unwrap();
            if nuclei_diag_on() {
                eprintln!(
                    "[nlive] burst close a10={} n={}{}",
                    bst.a10,
                    bst.n_nuclei,
                    if burst_expired { " (closed by timeout)" } else { "" }
                );
            }
            nuc.live.consumed_watermark = nuc.live.consumed_watermark.max(bst.counted_upto);
            nuc.live.sess_q = std::mem::take(&mut bst.counter.sess_q); // Quantiles move back, persisting across bursts
            bst.counter.close();
            let flushed = bst.counter.emitted.drain(..).collect::<Vec<_>>();
            if bst.n_nuclei + flushed.len() == 1 && nuc.live.last_act_frame > bst.a10 {
                // Single-character plateau evidence (burst energy duration) -> rolling median (same source as the slow merge rule)
                nuc.single_durs
                    .push((nuc.live.last_act_frame - bst.a10) as f64 * 0.01);
                if nuc.single_durs.len() > 24 {
                    nuc.single_durs.remove(0);
                }
            }
            new_nuclei.extend(flushed);
        }
        // -- 3) per-nucleus identity (centred-window raw cosine argmax across all commands; same as offline finish_nuclei) --
        // Nuclei whose window tail rows are not flushed are held first (judged only once g1 <= flushed rows; finish forces it),
        // keeping the identity window as long as replay's -- otherwise pooled_cos returns −1 on an empty window and argmax degenerates to cmd0.
        if !new_nuclei.is_empty() {
            // Nuclei are queued with their burst sequence number (the grouping id; they belong to the current or just-closed burst)
            let bseq = nuc.live.burst_seq;
            nuc.live
                .pending_ident
                .extend(new_nuclei.drain(..).map(|(pf, h)| (pf, h, bseq)));
        }
        if !nuc.live.pending_ident.is_empty() {
            let rows_flushed = nuc.st8.abs_len();
            let hb = nuc.st8.head_base;
            let mut ready_until = 0usize;
            for (k, (pf, _h, _s)) in nuc.live.pending_ident.iter().enumerate() {
                let tau = *pf as f64 * 0.01;
                let g1 = round_half_even((tau + nuc.p.ident_fwd_sec - 0.04) / 0.08).max(0) as usize + 1;
                if !final_flush && g1 > rows_flushed {
                    break; // Preserve order: while the head is incomplete, everything waits
                }
                ready_until = k + 1;
            }
            let emit_frame = env_end.saturating_sub(1);
            let emit_ctx = EmitCtx {
                exp: &nuc.exp_nuclei,
                seq: &nuc.seq_tmpl,
                min_conf: nuc.p.min_conf,
                force_dtw: false,
                diag: nuclei_diag_on(),
                cmds: &self.cmds,
            };
            let dim = nuc.cents.first().map(|c| c.len()).unwrap_or(0);
            let diag_ident = diag_ident_on();
            for (pf, _h, bseq) in nuc.live.pending_ident[..ready_until].iter() {
                let tau = *pf as f64 * 0.01;
                let g0 = round_half_even((tau - nuc.p.ident_back_sec - 0.04) / 0.08).max(0) as usize;
                let mut g1 =
                    round_half_even((tau + nuc.p.ident_fwd_sec - 0.04) / 0.08).max(0) as usize + 1;
                if final_flush {
                    g1 = g1.min(rows_flushed).max(g0 + 1); // Forced decision on a truncated window (no more rows arrive)
                }
                let mut best = (-1.0f64, 0usize);
                let mut second = -1.0f64;
                for (ci, cent) in nuc.cents.iter().enumerate() {
                    let c = pooled_cos(
                        &nuc.st8.cs,
                        g0.saturating_sub(hb),
                        g1.saturating_sub(hb),
                        cent,
                    );
                    if c > best.0 {
                        second = best.0;
                        best = (c, ci);
                    } else if c > second {
                        second = c;
                    }
                }
                if diag_ident {
                    eprintln!(
                        "[nlive-ident] pf={pf} bseq={bseq} cmd={cmd_i}({name}) best={best:.4} second={second:.4} min_conf={mc:.4} {verdict}",
                        pf = pf,
                        bseq = bseq,
                        cmd_i = best.1,
                        name = self.cmds.get(best.1).map(String::as_str).unwrap_or("?"),
                        best = best.0,
                        second = second,
                        mc = nuc.p.min_conf,
                        verdict = if best.0 < nuc.p.min_conf {
                            "reject-minconf"
                        } else {
                            "full-ok"
                        },
                    );
                }
                if best.0 < nuc.p.min_conf {
                    self.rejected = self.rejected.saturating_add(1);
                    continue;
                }
                push_diag_capped(&mut nuc.live.diag, (*pf, emit_frame, best.1, best.0));
                let nuc_j = NucJudged {
                    pf: *pf,
                    cmd: best.1,
                    score: best.0,
                    start: (tau - 0.04) as f32,
                    end: (tau + 0.04) as f32,
                    emb: window_embedding(
                        &nuc.st8.cs,
                        g0.saturating_sub(hb),
                        g1.saturating_sub(hb),
                        dim,
                    )
                    .unwrap_or_default(),
                };
                stage_append(
                    &emit_ctx,
                    &mut nuc.live.stage,
                    *bseq,
                    nuc_j,
                    &mut self.accepted,
                    nuc.p.refr_pf,
                    &mut nuc.live.refr_until_pf,
                );
            }
            nuc.live.pending_ident.drain(..ready_until);
        }
        // Grouped-path moment c: the staged burst is closed and all its nuclei are judged -> emit the remainder (including the last group).
        // Checked separately (not inside the ident block): after a burst closes pending_ident may already be empty,
        // so that poll sees no identity decision and a missed check would delay the last group until the next burst.
        // (a new burst not yet open, or a different sequence number, both count as "closed"; pending_ident's ordering guarantees "judged")
        let stage_closed_all_judged = match &nuc.live.stage {
            Some((sid, list)) => {
                !list.is_empty()
                    && nuc.live.burst.as_ref().map(|b| b.seq) != Some(*sid)
                    && !nuc.live.pending_ident.iter().any(|(_, _, s)| s == sid)
            }
            None => false,
        };
        if stage_closed_all_judged {
            if let Some((_, list)) = nuc.live.stage.take() {
                let emit_ctx = EmitCtx {
                    exp: &nuc.exp_nuclei,
                    seq: &nuc.seq_tmpl,
                    min_conf: nuc.p.min_conf,
                    force_dtw: false,
                    diag: nuclei_diag_on(),
                    cmds: &self.cmds,
                };
                flush_stage_run(
                    &emit_ctx,
                    &list,
                    &mut self.accepted,
                    nuc.p.refr_pf,
                    &mut nuc.live.refr_until_pf,
                );
            }
        }
    }
}

// -- Syllable-nucleus counting (nuclei path) --

/// SOS coefficients of the 4th-order Butterworth F1 band-pass (300-900 Hz at 16 kHz), matching scipy
/// `butter(4,[300,900],btype='band',fs=16000,output='sos')` ([b0,b1,b2,a0,a1,a2], generated once and embedded verbatim).
const F1_SOS_300_900_16K: [[f64; 6]; 4] = [
    [
        1.4412022408e-04,
        2.8824044815e-04,
        1.4412022408e-04,
        1.0,
        -1.6994300445,
        0.76051891682,
    ],
    [1.0, 2.0, 1.0, 1.0, -1.8252606306, 0.84833657789],
    [1.0, -2.0, 1.0, 1.0, -1.7669374379, 0.87649371345],
    [1.0, -2.0, 1.0, 1.0, -1.9394195491, 0.95377908112],
];

/// Nuclei-path parameters (calibrated values plus two env-adjustable ones)
#[derive(Clone, Debug)]
pub struct NucleiParams {
    pub delta_plateau: f64,      // Plateau opening threshold Δ=0.09
    pub curve_win_rows: usize,   // Curve sliding window, 4 rows (0.32 s)
    pub gate_frac: f64,          // Required activity fraction inside the window >= 0.50
    pub floor_p: f64,            // Floor percentile p=0.20
    pub tau_rows: f64,           // τ=3 s / 0.08 s row pitch = 37.5 rows
    pub sil_drop: f64,           // Floor drop target during silence 0.15
    pub a_sil: f64,              // Silence decay coefficient 1−exp(−0.08/0.4)
    pub merge_gap_sec: f64,      // Plateau fragment merge 0.30 s
    pub min_plateau_sec: f64,    // Minimum plateau 0.12 s
    pub burst_gap_frames: usize, // Burst boundary energy hole 0.30 s = 30 frames
    pub smooth_win: usize,       // Envelope smoothing 50 ms = 5 frames
    pub thr_q: f64,              // Q99
    pub rel_db: f64,             // −25dB
    pub valley_db: f64,          // Valley depth 6 dB (slow lever)
    pub min_dist_frames: usize,  // Peak distance 0.12 s = 12 frames
    pub refr_pf: usize,
    pub hold_frames: usize,  // Hold window length (frames)
    pub sup_db: f64,         // Superseding peak height difference
    pub hold_max_depth: f64, // Only shallow-valley confirmed nuclei are held (a deep valley is a real boundary and emits immediately)
    /// Absolute cosine floor (same as EngineParams.min_conf; the hit criterion on the production nuclei path)
    pub min_conf: f64,
    pub ident_back_sec: f64, // Identity window [τ − ident_back_sec, τ + ident_fwd_sec]
    pub ident_fwd_sec: f64,
    pub act_db: f64,
    pub absorb_max_frames: usize,
    /// Maximum burst duration in frames (default 800 = 8 s; 0 = off).
    /// While a run lingers open (the cosine plateau never closes) the quiet-close condition is never met, and a single
    /// counter's rolling Q99 gets pinned by the early loud speech; forcing a close rebuilds the counter and advances the floor, giving later weak speech a local threshold.
    pub burst_max_frames: usize,
}

/// Seconds -> 10 ms frames (refr_pf); shared by `from_v2` and `apply_live_tuning`.
pub(crate) fn refr_pf_from_secs(s: f64) -> usize {
    (s.clamp(0.0, 2.0) * 100.0).round().max(0.0) as usize
}

impl NucleiParams {
    /// Derives the settings-affected fields from `EngineParams` (single source of truth).
    fn from_v2(v: &EngineParams) -> Self {
        let mut np = NucleiParams::default();
        np.refr_pf = refr_pf_from_secs(v.refr_s);
        np.min_conf = v.min_conf;
        np.act_db = v.act_db;
        np
    }
}

impl Default for NucleiParams {
    fn default() -> Self {
        NucleiParams {
            delta_plateau: 0.09,
            gate_frac: 0.50,
            floor_p: 0.20,
            tau_rows: 3.0 / 0.08,
            sil_drop: 0.15,
            a_sil: 1.0 - (-0.08f64 / 0.4).exp(),
            merge_gap_sec: 0.30,
            min_plateau_sec: 0.12,
            // Curve sliding-window rows (= the length of identity evidence, 4 rows = 0.32 s)
            curve_win_rows: 4,
            burst_gap_frames: 30,
            smooth_win: 5,
            thr_q: 99.0,
            rel_db: 25.0,
            valley_db: 6.0,
            min_dist_frames: 12,
            refr_pf: std::env::var("VIQ_UDKWS_V2_NUCLEI_REFR")
                .ok()
                .and_then(|v| v.trim().parse::<usize>().ok())
                .unwrap_or(57), // = refr_s 0.57 s / 0.01 s pf grid
            hold_frames: 25,
            absorb_max_frames: 150,
            burst_max_frames: 800,
            sup_db: 8.0,
            hold_max_depth: 16.0,
            ident_back_sec: 0.12,
            ident_fwd_sec: ident_fwd_sec_env(),
            act_db: -40.0,
            min_conf: min_conf_default(),
        }
    }
}

fn percentile_sorted(sorted: &[f64], q: f64) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return 0.0;
    }
    if n == 1 {
        return sorted[0];
    }
    let rank = (n as f64 - 1.0) * q / 100.0;
    let lo = rank.floor().max(0.0) as usize;
    let hi = (rank.ceil() as usize).min(n - 1);
    let frac = rank - lo as f64;
    sorted[lo] * (1.0 - frac) + sorted[hi] * frac
}

fn percentile_linear(v: &[f64], q: f64) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    percentile_sorted(&s, q)
}

/// Direct-form I biquad forward filtering (one SOS section), zero initial state
fn biquad_forward(x: &[f64], s: &[f64; 6]) -> Vec<f64> {
    let (b0, b1, b2) = (s[0], s[1], s[2]);
    let (a1, a2) = (s[4] / s[3], s[5] / s[3]);
    let mut y = vec![0.0f64; x.len()];
    let (mut x1, mut x2, mut y1, mut y2) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for (i, &xi) in x.iter().enumerate() {
        let yi = b0 * xi + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2;
        y[i] = yi;
        x2 = x1;
        x1 = xi;
        y2 = y1;
        y1 = yi;
    }
    y
}

/// Merged entry score = max within the group, start/t = first nucleus, end = last nucleus (a single-nucleus group passes through unchanged).
/// Non-streaming reference implementation, used only by tests to check the streaming `merge_run_groups`.
#[cfg(test)]
fn merge_burst_groups(nucs: &[NucJudged], exp: &[usize]) -> Vec<Accepted> {
    let mut out = Vec::new();
    let mut i = 0usize;
    while i < nucs.len() {
        let cmd = nucs[i].cmd;
        let cap = exp.get(cmd).copied().unwrap_or(1).max(1);
        let mut j = i;
        while j + 1 < nucs.len() && nucs[j + 1].cmd == cmd {
            j += 1;
        }
        out.extend(merge_run_groups(&nucs[i..=j], cap));
        i = j + 1;
    }
    out
}

/// Zero-phase filtfilt (SOS cascade forward -> backward -> forward; odd-reflection padding of 64 samples ≈ 4 ms at both ends).
/// Known difference: scipy's sosfiltfilt initializes with the lfilter_zi steady state while this starts from zero -- interior
/// segments (further than ~64 samples from the edge) match to 1e-5 (see the unit test), and the difference is confined to the ~4 ms transient at each end;
/// counted bursts always sit inside the session (earliest at 0.9 s), so it does not matter.
fn sos_filt_filt(x: &[f32]) -> Vec<f64> {
    const PAD: usize = 64;
    let n = x.len();
    if n < 2 * PAD + 8 {
        return vec![0.0; n];
    }
    let mut v = vec![0.0f64; n + 2 * PAD];
    for i in 0..n {
        v[PAD + i] = x[i] as f64;
    }
    for i in 0..PAD {
        v[PAD - 1 - i] = 2.0 * x[0] as f64 - x[i + 1] as f64;
        v[PAD + n + i] = 2.0 * x[n - 1] as f64 - x[n - 2 - i] as f64;
    }
    let mut y = v;
    for s in F1_SOS_300_900_16K.iter() {
        y = biquad_forward(&y, s);
    }
    y.reverse();
    for s in F1_SOS_300_900_16K.iter() {
        y = biquad_forward(&y, s);
    }
    y.reverse();
    y[PAD..PAD + n].to_vec()
}

#[derive(Clone)]
struct SosForward {
    z: [[f64; 4]; 8], // Two passes × 4 DF1 sections of state [x1, x2, y1, y2]
}

impl SosForward {
    fn new() -> Self {
        SosForward { z: [[0.0; 4]; 8] }
    }

    /// Feed one sample, return the filter output
    fn push(&mut self, x: f64) -> f64 {
        let mut x0 = x;
        for (s, st) in F1_SOS_300_900_16K
            .iter()
            .cycle()
            .take(8)
            .zip(self.z.iter_mut())
        {
            let (b0, b1, b2) = (s[0], s[1], s[2]);
            let (a1, a2) = (s[4] / s[3], s[5] / s[3]);
            let y = b0 * x0 + b1 * st[0] + b2 * st[1] - a1 * st[2] - a2 * st[3];
            st[1] = st[0];
            st[0] = x0;
            st[3] = st[2];
            st[2] = y;
            x0 = y;
        }
        x0
    }
}

struct RollingQuantile {
    ring: std::collections::VecDeque<f64>, // Insertion order (eviction always drops the oldest sample)
    sorted: Vec<f64>,                      // Sorted mirror of the same contents
    cap: usize,
    n_total: usize,
}

const LIVE_Q_WINDOW_FRAMES: usize = 3000; // 30s @10ms

impl RollingQuantile {
    fn new(cap: usize) -> Self {
        RollingQuantile {
            ring: Default::default(),
            sorted: Vec::new(),
            cap,
            n_total: 0,
        }
    }

    fn push(&mut self, x: f64) {
        if self.ring.len() == self.cap {
            let old = self.ring.pop_front().unwrap();
            let pos = self
                .sorted
                .binary_search_by(|v| v.partial_cmp(&old).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap_or_else(|p| p);
            self.sorted.remove(pos);
        }
        let pos = self
            .sorted
            .binary_search_by(|v| v.partial_cmp(&x).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or_else(|p| p);
        self.sorted.insert(pos, x);
        self.ring.push_back(x);
        self.n_total += 1;
    }

    /// Cumulative frame count within the burst (including frames already pushed out of the window; not the current window size)
    fn total(&self) -> usize {
        self.n_total
    }

    fn percentile(&self, q: f64) -> f64 {
        percentile_sorted(&self.sorted, q)
    }

    fn clear(&mut self) {
        self.ring.clear();
        self.sorted.clear();
        self.n_total = 0;
    }
}

impl Default for RollingQuantile {
    fn default() -> Self {
        RollingQuantile::new(LIVE_Q_WINDOW_FRAMES)
    }
}

struct LiveCounter {
    valley_db: f64,
    early_db: f64,
    min_dist: usize,
    thr_q: f64,
    rel_db: f64,
    diag: bool,
    hist: RollingQuantile, // Rolling high-percentile window over the burst's consumed smoothed frames (capped at 30 s, see RollingQuantile)
    sess_q: RollingQuantile, // Session-level rolling high percentile (persists across bursts; the session side of thr = max(burst, session) - rel).
    hold: Option<(usize, f64, f64)>,
    held: Option<(usize, f64)>, // Confirmed nucleus held for release (shallow valley; superseded by a peak at least sup_db higher)
    hold_until: usize,
    cand: Option<(usize, f64, f64)>, // (peak frame, peak height, v_at_peak snapshot) -- the snapshot travels with the candidate:
    vmin: f64,
    absorbing: bool,
    baseline: f64, // Absorb-domain baseline = height of the last emitted nucleus
    cluster_min: f64,
    /// Absorb-cluster entry frame (used by the 2-stall collection); None = not inside the absorb domain
    absorbing_start: Option<usize>,
    /// Maximum duration of the absorb domain in frames (2; production overrides it from NucleiParams.absorb_max_frames, 0 = off)
    absorb_max_frames: usize,
    last_emit: Option<(usize, f64)>,
    v_since_emit: f64,
    last_x: Option<f64>, // Previous smoothed value (a candidate opens only on a rising edge, suppressing falling-edge phantom peaks)
    emitted: Vec<(usize, f64)>,
}

impl LiveCounter {
    fn new(
        valley_db: f64,
        early_db: f64,
        min_dist: usize,
        thr_q: f64,
        rel_db: f64,
        diag: bool,
    ) -> Self {
        LiveCounter {
            valley_db,
            early_db,
            min_dist,
            thr_q,
            rel_db,
            diag,
            hist: RollingQuantile::new(LIVE_Q_WINDOW_FRAMES),
            sess_q: RollingQuantile::new(LIVE_Q_WINDOW_FRAMES),
            hold: None,
            held: None,
            hold_until: 0,
            cand: None,
            vmin: 0.0,
            absorbing: false,
            baseline: 0.0,
            cluster_min: f64::INFINITY,
            absorbing_start: None,
            absorb_max_frames: 150,
            last_emit: None,
            v_since_emit: f64::INFINITY,
            last_x: None,
            emitted: Vec::new(),
        }
    }

    fn start(&mut self) {
        self.hist.clear();
        self.cand = None;
        self.absorbing = false;
        self.cluster_min = f64::INFINITY;
        self.last_emit = None;
        self.v_since_emit = f64::INFINITY;
        self.last_x = None;
        self.emitted.clear();
        self.held = None;
        self.hold_until = 0;
        // sess_q is not cleared (it persists across bursts); the caller mem::take's it in after start
    }

    /// A held nucleus is emitted as usual once it expires (the caller guarantees holding only while hold is on).
    /// It does not go through emit(): v_since_emit is not reset -- a held nucleus's "emission" is anchored at its confirmation frame
    /// (already reset once when held), and the valley between the held nucleus and the next one must survive in the next nucleus's valley snapshot.
    /// Resetting on release would erase that inter-nucleus valley and the next nucleus would be dropped as a tail ripple.
    fn release_held(&mut self) {
        if let Some((pf, h)) = self.held.take() {
            if self.diag {
                eprintln!("[nlive] release pf={pf} h={h:.1}");
            }
            self.emitted.push((pf, h));
            self.last_emit = Some((pf, h));
        }
    }

    /// A held nucleus is superseded by a clearly higher new peak (an intra-character pulse merging into the later peak): the hold is withdrawn and the absorb domain reset
    fn cancel_held(&mut self) {
        if self.diag {
            eprintln!(
                "[nlive] supersede-cancel pf={}",
                self.held.map(|h| h.0).unwrap_or(0)
            );
        }
        self.held = None;
        self.absorbing = false;
        self.cluster_min = f64::INFINITY;
    }

    /// Candidate confirmation (shared by the deep-valley, early and close paths): first the valley-snapshot depth check (tail ripples merge back into the previous nucleus),
    /// then per the hold setting: a shallow valley (below hold_max_depth) is held pending supersession, a deep valley is a real boundary and emits immediately.
    /// Returns whether it was confirmed (emitted or held).
    fn confirm(&mut self, ci: usize, ch: f64, i: usize, cap: f64, conf_depth: f64) -> bool {
        if ch - cap < self.valley_db {
            if self.diag {
                eprintln!("[nlive] tail-ripple drop pf={ci} h={ch:.1} v={cap:.1}");
            }
            return false;
        }
        self.release_held(); // Preserve order: release the older held nucleus first
        match self.hold {
            Some((hf, _, hmd)) if conf_depth < hmd => {
                self.held = Some((ci, ch));
                self.hold_until = i + hf;
                self.last_emit = Some((ci, ch));
                self.v_since_emit = f64::INFINITY;
                if self.diag {
                    eprintln!("[nlive] hold pf={ci} h={ch:.1} until={}", i + hf);
                }
            }
            _ => self.emit((ci, ch)),
        }
        true
    }

    fn feed(&mut self, i: usize, x: f64) {
        self.hist.push(x);
        self.sess_q.push(x);
        let mut thr = if self.hist.total() >= 3 {
            self.hist.percentile(self.thr_q) - self.rel_db
        } else {
            f64::INFINITY
        };
        if self.sess_q.total() >= 3 {
            thr = thr.max(self.sess_q.percentile(self.thr_q) - self.rel_db);
        }
        let rising = self.last_x.map_or(true, |px| x > px);
        self.last_x = Some(x);
        self.v_since_emit = self.v_since_emit.min(x);
        // Hold expires -> emit as usual; a new peak more than sup_db higher within the window supersedes it (an intra-letter pulse merges into the vowel nucleus)
        if self.held.is_some() && i >= self.hold_until {
            self.release_held();
        }
        if let Some((_, sup_db, _)) = self.hold {
            if let Some((_, hh)) = self.held {
                if x >= hh + sup_db && hh - self.v_since_emit < self.hold_max_depth() {
                    self.cancel_held();
                }
            }
        }
        if x < thr {
            if let Some((ci, ch, cap)) = self.cand {
                self.vmin = self.vmin.min(x);
                if ch - self.vmin >= self.valley_db {
                    self.cand = None;
                    self.absorbing = false;
                    self.absorbing_start = None;
                    self.confirm(ci, ch, i, cap, ch - self.vmin);
                }
            }
            return;
        }
        if self.absorbing {
            self.cluster_min = self.cluster_min.min(x);
            if self.baseline - self.cluster_min >= self.valley_db {
                self.absorbing = false; // The cluster closed; fall through to the normal logic below
            } else if self.absorb_max_frames > 0
                && self
                    .absorbing_start
                    .map_or(false, |a0| i - a0 >= self.absorb_max_frames)
            {
                self.absorbing = false;
                self.absorbing_start = None;
            } else {
                return; // Absorb a shallow-valley peak of the same cluster (serial equivalent of the offline merge rule)
            }
        }
        match self.cand {
            Some((ci, ch, cap)) => {
                self.vmin = self.vmin.min(x);
                if ch - self.vmin >= self.valley_db {
                    // Deep-valley confirmation of the previous peak (the pre-emission valley-snapshot check happens inside confirm)
                    let conf_depth = ch - self.vmin;
                    self.cand = None;
                    self.absorbing = false;
                    self.absorbing_start = None;
                    if self.confirm(ci, ch, i, cap, conf_depth)
                        && i.abs_diff(ci) >= self.min_dist
                        && rising
                    {
                        self.cand = Some((i, x, self.v_since_emit));
                        self.vmin = x;
                    }
                } else if x > ch {
                    self.cand = Some((i, x, self.v_since_emit));
                    self.vmin = x;
                } else if i - ci >= self.min_dist && ch - self.vmin >= self.early_db {
                    // Shallow valley >= 2 dB with the peak distance exceeded -> early confirmation (same depth check), entering the absorb domain
                    let conf_depth = ch - self.vmin;
                    if self.diag {
                        eprintln!(
                            "[nlive] early-check i={i} ch={ch:.1} cap={cap:.1} depth={:.1}",
                            ch - cap
                        );
                    }
                    if self.confirm(ci, ch, i, cap, conf_depth) {
                        self.absorbing = true;
                        self.absorbing_start = Some(i);
                        self.baseline = ch;
                        self.cluster_min = self.vmin;
                        self.cand = None;
                    } else {
                        self.cand = None;
                    }
                }
            }
            None => {
                // Open a new candidate: at least min_dist from the last emitted peak and on a rising edge (never on a falling edge --
                // that suppresses the phantom "tail" of peaks rejected by the distance rule; past 12 frames on the rising edge it is picked up, so at fast speech rates
                // the next character's onset is not swallowed by the whole run)
                let ok = match self.last_emit {
                    Some((li, _)) => i.abs_diff(li) >= self.min_dist,
                    None => true,
                };
                if ok && rising {
                    if self.diag {
                        eprintln!("[nlive] cand-open i={i} x={x:.1} thr={thr:.1}");
                    }
                    self.cand = Some((i, x, self.v_since_emit));
                    self.vmin = x;
                }
            }
        }
    }

    fn hold_max_depth(&self) -> f64 {
        self.hold.map(|(_, _, d)| d).unwrap_or(f64::INFINITY)
    }

    fn emit(&mut self, p: (usize, f64)) {
        self.v_since_emit = f64::INFINITY;
        if self.diag {
            eprintln!(
                "[nlive] emit pf={} h={:.1} (cand={:?} absorbing={})",
                p.0,
                p.1,
                self.cand.map(|c| c.0),
                self.absorbing
            );
        }
        self.emitted.push(p);
        self.last_emit = Some(p);
    }

    /// Burst close / stream end: pending candidates run the depth check before emitting (tail ripples merge back into the previous nucleus; same merge rule as the full version);
    /// a held nucleus has no future frames to be superseded by, so it is emitted too (in order: whichever was held first goes first)
    fn close(&mut self) {
        if let Some((pf, h, cap)) = self.cand.take() {
            if h - cap >= self.valley_db {
                self.release_held();
                self.emit((pf, h));
            } else if self.diag {
                eprintln!("[nlive] tail-ripple drop-close pf={pf} h={h:.1} v={cap:.1}");
            }
        }
        self.release_held();
    }
}

fn smooth_window_bounds(i: usize, env_base: usize) -> (usize, usize) {
    (i.saturating_sub(2).max(env_base), i + 2)
}

/// live burst start: from the plateau start frame, expand left to an activity hole wider than gap frames (the left half of burst_bounds, causal version;
/// the plateau's start and midpoint lie in the same activity connected component, so the left edge agrees).
fn burst_start_causal(act: &[bool], from: usize, gap: usize) -> usize {
    if act.is_empty() {
        return 0;
    }
    let mut a = from.min(act.len() - 1) as i64;
    let mut silence = 0i64;
    while a > 0 && silence <= gap as i64 {
        a -= 1;
        silence = if act[a as usize] { 0 } else { silence + 1 };
    }
    (a + silence).max(0) as usize
}

/// Single-command plateau runs (incremental regions_from_mask + same-command adaptive merge)
#[derive(Clone, Default)]
struct RunState {
    open: Option<(usize, usize)>, // (start, last_true) absolute rows
    hole: usize,
    prev: Option<(usize, usize)>, // Closed segment (a,b) (b half-open; short segments are dropped, merged ones take the union)
}

fn run_update(r: &mut RunState, g: usize, mask: bool, max_hole: usize, min_rows: usize) {
    if mask {
        match r.open.as_mut() {
            Some(o) => o.1 = g,
            None => {
                let merged = matches!(r.prev, Some(p) if g.saturating_sub(p.1) <= max_hole);
                let start = if merged { r.prev.unwrap().0 } else { g };
                r.open = Some((start, g));
                if merged {
                    r.prev = None;
                }
            }
        }
        r.hole = 0;
    } else if r.open.is_some() {
        r.hole += 1;
        if r.hole > max_hole {
            let (a, b) = r.open.take().unwrap();
            if b + 1 - a >= min_rows {
                r.prev = Some((a, b + 1));
            }
        }
    }
}

/// Live causal path state (advances only when streaming=true; the offline full path never touches it)
struct NucleiLiveState {
    sos: SosForward,
    acc: f32,
    acc_n: usize,
    env: Vec<f64>, // Raw dB envelope ring; absolute frame numbers start at env_base (10 ms grid, same as act_frames)
    env_base: usize,
    last_act_frame: usize,
    noise_floor: Option<f64>,
    st8_seen: usize,              // Rows already pushed through plateau handling (= the st8.rows consumption cursor)
    floor_q: Vec<Option<f64>>,    // [cmd] incremental floor state
    floor_last: Vec<Option<f64>>, // [cmd] last valid floor (the silence-decay target)
    runs: Vec<RunState>,          // [cmd] plateau runs
    burst: Option<LiveBurst>,
    sess_q: RollingQuantile, // Session-level nucleus-count quantile (persists across bursts; moved into the counter on open, back on close)
    consumed_watermark: usize, // Highest frame consumed by a closed burst (the lower bound for a new burst's back-expansion --
    // Guards against activity-frame noise pushing the back-expansion out of bounds and re-consuming an already counted region of the previous burst)
    pending_ident: Vec<(usize, f64, u64)>, // Emitted nuclei awaiting identity (peak frame, hash, burst_seq; judged after the window tail flushes, in order)
    burst_seq: u64, // Opened-burst counter (the grouping id used by pending_ident and stage)
    // Live staging for the grouped path: (burst_seq, judged nuclei of that burst). The list is always a single
    // same-command run (guaranteed by moments a/b in stage_append); once the burst is closed and judged, advance_nuclei_live
    // checks for emission at the end. None = no unemitted staging.
    stage: Option<(u64, Vec<NucJudged>)>,
    refr_until_pf: usize,
    diag: Vec<(usize, usize, usize, f64)>, // (peak frame, emit frame, cmd_idx, cos)
}

struct LiveBurst {
    counter: LiveCounter,
    a10: usize,
    counted_upto: usize,
    n_nuclei: usize,
    seq: u64, // Group sequence number (= NucleiLiveState::burst_seq at open time)
}

impl NucleiLiveState {
    fn new(n_cmds: usize) -> Self {
        NucleiLiveState {
            sos: SosForward::new(),
            acc: 0.0,
            acc_n: 0,
            env: Vec::new(),
            env_base: 0,
            last_act_frame: 0,
            noise_floor: None,
            st8_seen: 0,
            floor_q: vec![None; n_cmds],
            floor_last: vec![None; n_cmds],
            runs: vec![RunState::default(); n_cmds],
            burst: None,
            sess_q: RollingQuantile::new(LIVE_Q_WINDOW_FRAMES),
            consumed_watermark: 0,
            pending_ident: Vec::new(),
            burst_seq: 0,
            stage: None,
            refr_until_pf: 0,
            diag: Vec::new(),
        }
    }
}

fn trim_act_head(act_frames: &mut Vec<bool>, act_base: &mut usize, keep: usize) {
    if act_frames.len() > keep {
        let drop = act_frames.len() - keep;
        act_frames.drain(..drop);
        *act_base += drop;
    }
}

fn push_diag_capped(diag: &mut Vec<(usize, usize, usize, f64)>, e: (usize, usize, usize, f64)) {
    diag.push(e);
    if diag.len() > 20000 + 2000 {
        let excess = diag.len() - 20000;
        diag.drain(..excess);
    }
}

/// 10 ms-grid RMS dB envelope
fn energy_db_env(samples: &[f32]) -> Vec<f64> {
    let hop = 160usize;
    let mut out = Vec::new();
    let mut i = 0usize;
    while i + hop <= samples.len() {
        let rms =
            (samples[i..i + hop].iter().map(|s| s * s).sum::<f32>() / hop as f32).sqrt() + 1e-9;
        out.push(20.0 * (rms as f64).log10());
        i += hop;
    }
    if out.is_empty() {
        out.push(0.0);
    }
    out
}

/// Moving average in the dB domain (mode='same' semantics: out-of-window counts as 0, denominator = window length)
fn smooth_env(env: &[f64], win: usize) -> Vec<f64> {
    if win <= 1 || env.is_empty() {
        return env.to_vec();
    }
    let half = win / 2;
    let mut out = Vec::with_capacity(env.len());
    for i in 0..env.len() {
        let a = i.saturating_sub(half);
        let b = (i + half).min(env.len() - 1);
        let s: f64 = env[a..=b].iter().sum();
        out.push(s / win as f64);
    }
    out
}

fn find_peaks_height_distance(seg: &[f64], thr: f64, dist: usize) -> Vec<usize> {
    let mut cands: Vec<(usize, f64)> = Vec::new();
    for i in 1..seg.len().saturating_sub(1) {
        if seg[i] >= seg[i - 1] && seg[i] > seg[i + 1] && seg[i] >= thr {
            cands.push((i, seg[i]));
        }
    }
    cands.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.0.cmp(&b.0))
    });
    let mut pk: Vec<usize> = Vec::new();
    for &(i, _) in &cands {
        if pk.iter().all(|&j| i.abs_diff(j) >= dist) {
            pk.push(i);
        }
    }
    pk.sort_unstable();
    pk
}

/// Syllable-nucleus counting: Q{thr_q} − rel_db relative threshold + minimum peak distance + valley-depth merging (iterated to a fixed point, keeping the higher peak).
/// Returns (nucleus count, list of absolute peak frames).
fn count_nuclei(
    env: &[f64],
    a: usize,
    b: usize,
    thr_q: f64,
    rel_db: f64,
    valley_db: f64,
    dist_frames: usize,
    smooth_win: usize,
) -> (usize, Vec<usize>) {
    let smoothed = smooth_env(env, smooth_win);
    let ai = a.min(smoothed.len());
    let bi = b.min(smoothed.len());
    if bi <= ai {
        return (0, Vec::new());
    }
    let seg = &smoothed[ai..bi];
    if seg.len() < 3 {
        return (0, Vec::new());
    }
    let thr = percentile_linear(seg, thr_q) - rel_db;
    let mut pk = find_peaks_height_distance(seg, thr, dist_frames);
    loop {
        if pk.len() <= 1 {
            break;
        }
        let mut changed = false;
        for i in 0..pk.len() - 1 {
            let (p1, p2) = (pk[i], pk[i + 1]);
            let val = seg[p1..=p2].iter().cloned().fold(f64::INFINITY, f64::min);
            if seg[p1].min(seg[p2]) - val < valley_db {
                pk.remove(if seg[p1] <= seg[p2] { i } else { i + 1 });
                changed = true;
                break;
            }
        }
        if !changed {
            break;
        }
    }
    (pk.len(), pk.iter().map(|&p| p + ai).collect())
}

/// Energy activity burst [a,b) containing the mid frame (expanded both ways to a hole wider than gap frames)
fn burst_bounds(act: &[bool], mid: usize, gap: usize) -> (usize, usize) {
    if act.is_empty() {
        return (0, 0);
    }
    let mut i = mid.min(act.len() - 1);
    if !act[i] {
        match (0..act.len())
            .filter(|&j| act[j])
            .min_by_key(|&j| (j.abs_diff(i), j))
        {
            Some(j) => i = j,
            None => return (i, i),
        }
    }
    let mut a = i as i64;
    let mut silence: i64 = 0;
    while a > 0 && silence <= gap as i64 {
        a -= 1;
        silence = if act[a as usize] { 0 } else { silence + 1 };
    }
    a += silence;
    let mut b = i as i64;
    silence = 0;
    while (b as usize) < act.len() - 1 && silence <= gap as i64 {
        b += 1;
        silence = if act[b as usize] { 0 } else { silence + 1 };
    }
    b += 1 - silence;
    (a.max(0) as usize, (b.max(0) as usize).min(act.len()))
}

/// Boolean mask -> contiguous intervals [(a,b)] (half-open row indices);
/// gaps longer than merge_gap/period split, intervals shorter than min_dur are dropped.
fn regions_from_mask(
    mask: &[bool],
    period: f64,
    merge_gap: f64,
    min_dur: f64,
) -> Vec<(usize, usize)> {
    let idx: Vec<usize> = mask
        .iter()
        .enumerate()
        .filter(|p| *p.1)
        .map(|p| p.0)
        .collect();
    if idx.is_empty() {
        return Vec::new();
    }
    let max_hole = (merge_gap / period).floor() as usize;
    let mut regs: Vec<(usize, usize)> = Vec::new();
    let mut a = idx[0];
    let mut p = idx[0];
    for &i in &idx[1..] {
        if i - p > max_hole {
            regs.push((a, p + 1));
            a = i;
        }
        p = i;
    }
    regs.push((a, p + 1));
    regs.into_iter()
        .filter(|(a, b)| (b - a) as f64 * period >= min_dur)
        .collect()
}

/// floor_track's first-segment −Δ correction plus silence decay
/// (the first valid value starts at q = x−Δ; invalid rows decay towards q_last − sil_drop at rate a_sil).
fn floor_track_nuclei(
    curve: &[f64],
    valid: &[bool],
    p: f64,
    tau_rows: f64,
    delta: f64,
    sil_drop: f64,
    a_sil: f64,
) -> Vec<f64> {
    let a_up = 1.0 - (-1.0 / tau_rows.max(1.0)).exp();
    let a_dn = (a_up * (1.0 - p) / p.max(1e-6)).min(1.0);
    let mut q: Option<f64> = None;
    let mut q_last: Option<f64> = None;
    let mut out = Vec::with_capacity(curve.len());
    for i in 0..curve.len() {
        if valid[i] {
            let x = curve[i];
            let nq = match q {
                None => x - delta,
                Some(qv) => {
                    let a = if x >= qv { a_up } else { a_dn };
                    qv + a * (x - qv)
                }
            };
            q = Some(nq);
            q_last = Some(nq);
        } else if let Some(qv) = q {
            let target = q_last.unwrap_or(qv) - sil_drop;
            q = Some(qv + a_sil * (target - qv));
        }
        out.push(q.unwrap_or(0.0));
    }
    out
}

/// Single-command cosine curve core: slide the window [t−W+1, t], mean-pool the template rows and take the cosine against the centroid;
/// an activity fraction below gate_frac -> INVALID. Returns an N-length curve (the first W−1 entries are INVALID).
fn plateau_curve(
    cs: &[Vec<f64>],
    act_row: &[f64],
    w: usize,
    gate_frac: f64,
    centroid: &[f32],
) -> Vec<f64> {
    let n = act_row.len();
    let mut out = vec![INVALID; n];
    if n < w || cs.len() < n + 1 {
        return out;
    }
    for t in w - 1..n {
        let hi = &cs[t + 1];
        let lo = &cs[t + 1 - w];
        let inv = 1.0 / w as f64;
        let mut dot = 0.0f64;
        let mut nrm = 0.0f64;
        for d in 0..centroid.len() {
            let m = (hi[d] - lo[d]) * inv;
            dot += m * centroid[d] as f64;
            nrm += m * m;
        }
        let gate: f64 = act_row[t + 1 - w..t + 1].iter().sum::<f64>() / w as f64;
        if gate >= gate_frac {
            let nrm = nrm.sqrt();
            out[t] = if nrm > 0.0 { dot / nrm } else { 0.0 };
        }
    }
    out
}

/// Per-nucleus identity: row mean over [g0,g1) using the prefix sums cs, raw cosine against the centroid (no gating)
fn pooled_cos(cs: &[Vec<f64>], g0: usize, g1: usize, centroid: &[f32]) -> f64 {
    let g0 = g0.min(cs.len().saturating_sub(1));
    let g1 = g1.min(cs.len().saturating_sub(1)).max(g0);
    if g1 <= g0 {
        return -1.0;
    }
    let hi = &cs[g1];
    let lo = &cs[g0];
    let inv = 1.0 / (g1 - g0) as f64;
    let mut dot = 0.0f64;
    let mut nrm = 0.0f64;
    for d in 0..centroid.len() {
        let m = (hi[d] - lo[d]) * inv;
        dot += m * centroid[d] as f64;
        nrm += m * m;
    }
    let nrm = nrm.sqrt();
    if nrm > 0.0 { dot / nrm } else { -1.0 }
}

/// Identity-window mean embedding (the same [g0,g1) row mean as pooled_cos, L2-normalized, f32 precision).
/// Used for word-instance sequence alignment (nucleus-window embedding vs per-character centroids); sharing the window with
/// identity scoring keeps the sequence alignment window bit-identical to the "one vote per nucleus" decision window.
fn window_embedding(cs: &[Vec<f64>], g0: usize, g1: usize, dim: usize) -> Option<Vec<f32>> {
    let g0 = g0.min(cs.len().saturating_sub(1));
    let g1 = g1.min(cs.len().saturating_sub(1)).max(g0);
    if g1 <= g0 || dim == 0 {
        return None;
    }
    let hi = &cs[g1];
    let lo = &cs[g0];
    let inv = 1.0 / (g1 - g0) as f64;
    let mut m = vec![0.0f32; dim];
    for d in 0..dim {
        m[d] = ((hi[d] - lo[d]) * inv) as f32;
    }
    l2_normalize(&mut m);
    Some(m)
}

/// Rolling median (for adaptive group merging)
fn median_of(v: &[f64]) -> Option<f64> {
    if v.is_empty() {
        return None;
    }
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = s.len();
    Some(if n % 2 == 1 {
        s[n / 2]
    } else {
        (s[n / 2 - 1] + s[n / 2]) / 2.0
    })
}

/// 80 ms grid row accumulator for run_stream8: dedupe by frame group of 8 and keep the row from the latest chunk (fullest context).
/// Equivalent to v1's "audio queue + model output queue": model rows are keyed by **audio group number g**, not by
/// production time; audio frames [8g, 8g+8) correspond to that row, and the consumer takes rows and activity frames by the same g.
/// Group g can only be updated by blocks k <= g, so once block k is processed, every g < k is final and can be flushed.
#[derive(Default)]
struct St8Accum {
    pending: std::collections::BTreeMap<usize, Vec<f32>>, // g -> row (later blocks overwrite)
    next_flush: usize,
    rows: Vec<Vec<f32>>,
    cs: Vec<Vec<f64>>, // Row prefix sums; cs[0] = zeros
    head_base: usize,
}

impl St8Accum {
    fn new() -> Self {
        St8Accum {
            pending: Default::default(),
            next_flush: 0,
            rows: Vec::new(),
            cs: vec![vec![0.0; 256]],
            head_base: 0,
        }
    }

    /// Absolute row count (including the trimmed head) -- every live consumer cursor (st8_seen, pending g1) works in absolute row numbers
    fn abs_len(&self) -> usize {
        self.head_base + self.rows.len()
    }

    /// Trim the head by dropping rows: rows and cs shift forward and cs is rebased by subtracting the new base (absolute row numbers are unchanged,
    /// and pooled_cos reads absolute row − head_base). Callers guarantee only consumed ranges are trimmed
    /// (housekeeping_nuclei_live: the smaller of the plateau curve window's left edge and the pending identity window's left edge).
    fn trim_head(&mut self, drop: usize) {
        let drop = drop.min(self.rows.len());
        if drop == 0 {
            return;
        }
        let base = self.cs[drop].clone();
        self.cs = self.cs[drop..]
            .iter()
            .map(|v| v.iter().zip(&base).map(|(a, b)| a - b).collect())
            .collect();
        self.rows.drain(..drop);
        self.head_base += drop;
    }

    /// One block of output: k = chunk_start/8 (chunk_start is a multiple of 8), rows holds every output row of the block,
    /// end = the last valid mel frame in the block (min(start+chunk_frames, T)).
    /// Output row j belongs to audio group k+j (model look-ahead), not to the current streaming instant; later chunks overwrite the same group.
    fn feed_chunk(&mut self, k: usize, rows: Vec<Vec<f32>>, end: usize) {
        for (j, row) in rows.into_iter().enumerate() {
            let g = k + j;
            if 8 * (g + 1) > end {
                continue;
            }
            self.pending.insert(g, row); // A later block has a larger k, so it overwrites the earlier result
        }
        while self.next_flush < k {
            let g = self.next_flush;
            if let Some(row) = self.pending.remove(&g) {
                let mut last = self.cs.last().unwrap().clone();
                for (n, x) in last.iter_mut().zip(row.iter()) {
                    *n += *x as f64;
                }
                self.cs.push(last);
                self.rows.push(row);
            }
            self.next_flush += 1;
        }
    }

    fn finish(&mut self) {
        while let Some(row) = self.pending.remove(&self.next_flush) {
            let mut last = self.cs.last().unwrap().clone();
            for (n, x) in last.iter_mut().zip(row.iter()) {
                *n += *x as f64;
            }
            self.cs.push(last);
            self.rows.push(row);
            self.next_flush += 1;
        }
    }
}

/// Offline run_stream8 (used for enrolment and template encoding): feats -> (rows8, secs8); secs = 0.08·g + 0.04.
fn encode_rows8_offline(
    enc: &mut Encoder256,
    feats: &[Vec<f32>],
) -> anyhow::Result<(Vec<Vec<f32>>, Vec<f64>)> {
    let t = feats.len();
    let mut best: std::collections::BTreeMap<usize, Vec<f32>> = Default::default();
    if t >= 8 {
        let mut start = 0usize;
        while start < t {
            let end = (start + enc.chunk_frames).min(t);
            let n = end - start;
            let mut chunk = Array2::<f32>::zeros((enc.chunk_frames, MEL_BINS));
            for i in 0..n {
                let row = &feats[start + i];
                for b in 0..MEL_BINS {
                    chunk[[i, b]] = row[b];
                }
            }
            let rows = enc.run_chunk(&chunk, n as i64)?;
            let k = start / 8;
            for (j, r) in rows.into_iter().enumerate() {
                let g = k + j;
                if 8 * (g + 1) <= end {
                    best.insert(g, r); // Later blocks overwrite
                }
            }
            start += 8;
        }
    }
    let rows: Vec<Vec<f32>> = best.values().cloned().collect();
    let secs: Vec<f64> = best.keys().map(|&g| 0.08 * g as f64 + 0.04).collect();
    Ok((rows, secs))
}

/// 10 ms activity frame flag (RMS dB > thr)
#[cfg(test)]
fn act_flags_10ms(samples: &[f32], thr_db: f64) -> Vec<bool> {
    energy_db_env(samples).iter().map(|&d| d > thr_db).collect()
}

const TPL_REL_FLOOR_DB: f64 = 5.0;

fn tpl_act_flags(samples: &[f32]) -> Vec<bool> {
    act_flags_10ms_rel(samples)
}

/// The relative criterion itself (pure function, unit-testable): midpoint gate plus a relative floor.
fn act_flags_10ms_rel(samples: &[f32]) -> Vec<bool> {
    let env = energy_db_env(samples);
    let p20 = percentile_linear(&env, 20.0);
    let p99 = percentile_linear(&env, 99.0);
    let thr = (p20 + 0.5 * (p99 - p20)).max(p20 + TPL_REL_FLOOR_DB);
    env.iter().map(|&d| d > thr).collect()
}

fn act_flags_10ms_adaptive(samples: &[f32], p: &EngineParams) -> Vec<bool> {
    let mut nf: Option<f64> = None;
    energy_db_env(samples)
        .iter()
        .map(|&d| {
            let gate = act_gate_step(&mut nf, d, p);
            d > gate
        })
        .collect()
}

fn active_range_secs(act: &[bool]) -> (f64, f64) {
    let idx: Vec<usize> = act
        .iter()
        .enumerate()
        .filter(|p| *p.1)
        .map(|p| p.0)
        .collect();
    if idx.is_empty() {
        return (0.0, 0.0);
    }
    let a = idx[0].saturating_sub(3);
    let b = (idx[idx.len() - 1] + 3).min(act.len() - 1);
    (a as f64 * 0.01, b as f64 * 0.01)
}

pub(crate) fn build_nuclei_templates(
    enc: &mut Encoder256,
    cr_dir: &std::path::Path,
    cmds: &[String],
    gate: f64,
) -> anyhow::Result<(
    Vec<Vec<f32>>,
    f64,
    Vec<f64>,
    Vec<CmdSeqTemplate>,
    Vec<usize>,
)> {
    let mut cents = Vec::with_capacity(cmds.len());
    let mut med_rows_all: Vec<f64> = Vec::with_capacity(cmds.len());
    let mut seq_all: Vec<CmdSeqTemplate> = Vec::with_capacity(cmds.len());
    let mut exp_all: Vec<usize> = Vec::with_capacity(cmds.len());
    for cmd in cmds {
        let t = build_nuclei_template_for_cmd(enc, cr_dir, cmd, cmds, gate)?;
        cents.push(t.centroid);
        med_rows_all.push(t.med_rows);
        seq_all.push(t.seq);
        exp_all.push(t.exp_nuclei);
    }
    let prior = med_rows_all.iter().sum::<f64>() / med_rows_all.len().max(1) as f64 * 0.08;
    Ok((cents, prior, med_rows_all, seq_all, exp_all))
}

/// Complete single-command syllable-nucleus template (returned by build_nuclei_template_for_cmd):
/// whole-word mean centroid (the current identity template) + duration prior + sequence templates.
#[derive(Debug, Clone)]
pub(crate) struct CmdNucleiTemplate {
    pub centroid: Vec<f32>,
    pub med_rows: f64,
    pub seq: CmdSeqTemplate,
    pub exp_nuclei: usize,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct CmdSeqTemplate {
    /// Per recording -> per character -> centroid (each L2-normalized, dim=256). Only recordings with >= 2 nuclei and a complete character split qualify.
    pub recs: Vec<Vec<Vec<f32>>>,
    /// Expected nucleus count = median nucleus count per recording (rounded, >= 1)
    pub n: usize,
}

pub(crate) fn build_nuclei_template_for_cmd(
    enc: &mut Encoder256,
    cr_dir: &std::path::Path,
    cmd: &str,
    cand: &[String],
    gate: f64,
) -> anyhow::Result<CmdNucleiTemplate> {
    let mut files: Vec<std::path::PathBuf> = std::fs::read_dir(cr_dir)
        .with_context(|| format!("failed to read command_recordings: {}", cr_dir.display()))?
        .flatten()
        .map(|e| e.path())
        .filter(|p| {
            p.extension().and_then(|s| s.to_str()) == Some("wav")
                && p.file_name()
                    .and_then(|s| s.to_str())
                    .map_or(false, |n| classify_recording(n, cand) == cmd)
        })
        .collect();
    files.sort();
    anyhow::ensure!(
        !files.is_empty(),
        "command {cmd} has no template recordings in command_recordings"
    );
    let p = NucleiParams::default();
    let dim = enc.out_dim;
    let vp = EngineParams {
        act_db: gate,
        ..EngineParams::default()
    };
    let mut all_rows: Vec<Vec<f32>> = Vec::new();
    let mut rec_cnt: Vec<f64> = Vec::new();
    let mut seq_recs: Vec<Vec<Vec<f32>>> = Vec::new();
    let mut nuclei_counts: Vec<f64> = Vec::new();
    for f in &files {
        // L1 dual-compatible read: recordings are ciphertext in the release form (recordings::read_wav_auto detects the magic)
        let (sr, samples) = crate::recordings::read_wav_auto(f)
            .map_err(|e| anyhow::anyhow!("failed to read {f:?}: {e}"))?;
        let samples = resample_linear(&samples, sr, TARGET_SAMPLE_RATE);
        let feats = log_mel_amp_fbank(&samples);
        let (a_s, b_s) = active_range_secs(&tpl_act_flags(&samples));
        enc.reset_state();
        let (rows8, secs8) = encode_rows8_offline(enc, &feats)?;
        let keep8: Vec<Vec<f32>> = rows8
            .iter()
            .zip(secs8.iter())
            .filter(|p| a_s <= *p.1 && *p.1 <= b_s)
            .map(|(r, _)| r.clone())
            .collect();
        let keep8_n = keep8.len();
        if keep8_n > 0 {
            all_rows.extend(keep8);
        }
        let act = act_flags_10ms_adaptive(&samples, &vp);
        let n_i = if let (Some(a10), Some(b10)) =
            (act.iter().position(|&x| x), act.iter().rposition(|&x| x))
        {
            let f1: Vec<f32> = sos_filt_filt(&samples).iter().map(|&v| v as f32).collect();
            let env_db = energy_db_env(&f1);
            let (_, peaks) = count_nuclei(
                &env_db,
                a10,
                b10,
                p.thr_q,
                p.rel_db,
                p.valley_db,
                p.min_dist_frames,
                p.smooth_win,
            );
            let mut chars: Vec<Vec<f32>> = Vec::with_capacity(peaks.len());
            for &pf in &peaks {
                let tau = pf as f64 * 0.01;
                let (w0, w1) = (tau - p.ident_back_sec, tau + p.ident_fwd_sec);
                let mut mean = vec![0.0f32; dim];
                let mut cnt = 0usize;
                for (r, s) in rows8.iter().zip(secs8.iter()) {
                    if w0 <= *s && *s <= w1 && a_s <= *s && *s <= b_s {
                        for (m, x) in mean.iter_mut().zip(r.iter()) {
                            *m += x;
                        }
                        cnt += 1;
                    }
                }
                if cnt > 0 {
                    for m in mean.iter_mut() {
                        *m /= cnt as f32;
                    }
                    l2_normalize(&mut mean);
                    chars.push(mean);
                }
            }
            let n_i = peaks.len();
            // Incomplete character split (empty window) -> that recording stays out of the sequence templates (its count still feeds the median, reflecting the real split)
            if n_i >= 2 && chars.len() == n_i {
                seq_recs.push(chars);
            }
            n_i
        } else {
            0
        };
        nuclei_counts.push(n_i as f64);
        if keep8_n > 0 {
            rec_cnt.push(keep8_n as f64);
        }
    }
    anyhow::ensure!(!all_rows.is_empty(), "the template recordings of command {cmd} have no active rows");
    let d = all_rows[0].len();
    let mut mean = vec![0.0f32; d];
    for r in &all_rows {
        for (m, x) in mean.iter_mut().zip(r.iter()) {
            *m += x;
        }
    }
    for m in mean.iter_mut() {
        *m /= all_rows.len() as f32;
    }
    l2_normalize(&mut mean);
    // Expected nucleus count = median nucleus count per recording (rounded, >= 1). n == 1 -> clear the sequence (single-character commands use
    // the current path entirely); n >= 2 but every character split failed (empty sequence) -> the runtime falls back to grouping as if there were no template.
    let exp_nuclei = (median_of(&nuclei_counts).unwrap_or(1.0).round() as usize).max(1);
    if exp_nuclei < 2 {
        seq_recs.clear();
    }
    eprintln!(
        "[seq] {cmd}: expected nuclei {exp_nuclei}, {} sequence templates (per-recording nucleus counts {:?})",
        seq_recs.len(),
        nuclei_counts
            .iter()
            .map(|&x| x as usize)
            .collect::<Vec<_>>()
    );
    Ok(CmdNucleiTemplate {
        centroid: mean,
        med_rows: median_of(&rec_cnt).unwrap_or(4.0).max(3.0),
        seq: CmdSeqTemplate {
            recs: seq_recs,
            n: exp_nuclei,
        },
        exp_nuclei,
    })
}

/// nuclei path state (replay: streaming=false accumulates all samples and computes once at finish; live: streaming=true
/// runs the live causal path -- forward-filtered rolling envelope + rolling Q99 + serial counter, see NucleiLiveState)
struct NucleiState {
    p: NucleiParams,
    cents: Vec<Vec<f32>>, // Aligned with self.cmds
    st8: St8Accum,
    samples: Vec<f32>,   // All samples (filtfilt is non-causal; used by the offline/test finish path)
    char_dur_prior: f64, // Template syllable-length prior
    single_durs: Vec<f64>, // Online single-character plateau durations (rolling-median evidence)
    live: NucleiLiveState, // Live causal-path state
    exp_nuclei: Vec<usize>,
    // Command sequence template (None = not a sequence command: n == 1 builds none / every character split failed -> grouping).
    seq_tmpl: Vec<Option<CmdSeqTemplate>>,
}

pub(crate) fn classify_recording(fname: &str, ids: &[String]) -> String {
    let mut best: Option<&str> = None;
    for id in ids {
        let pref = format!("{id}_");
        if fname.starts_with(&pref) && best.is_none_or(|b| id.len() > b.len()) {
            best = Some(id);
        }
    }
    match best {
        Some(id) => id.to_string(),
        None => fname.split('_').next().unwrap_or("").to_string(),
    }
}

/// Candidate command ids for classification: every command id in the model directory's commands.json ∪ the existing
/// embed_proto_v2.json keys (neither has any -> empty set and classify_recording falls back to the first segment).
/// A full regeneration therefore self-heals: a corrected classification writes the right id back, and the old wrong key
/// (e.g. "new") disappears during the rewrite because it has no recordings of its own.
fn load_candidate_cmd_ids(model_dir: &str) -> Vec<String> {
    let mut ids: Vec<String> = Vec::new();
    let cj = std::path::Path::new(model_dir).join("commands.json");
    if let Ok(t) = std::fs::read_to_string(&cj) {
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(&t) {
            let obj = v.get("commands").and_then(|x| x.as_object());
            ids.extend(obj.into_iter().flat_map(|o| o.keys().cloned()));
        }
    }
    let pj = std::path::Path::new(model_dir).join("embed_proto_v2.json");
    // L1 dual-compatible read (plaintext/ciphertext magic detection; unreadable -> no keys, so classification falls back to commands.json)
    if let Ok(t) = crate::crypto::read_text_auto(&pj) {
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(&t) {
            ids.extend(v.as_object().into_iter().flat_map(|o| o.keys().cloned()));
        }
    }
    ids.sort();
    ids.dedup();
    ids
}

pub fn gen_embed_proto(model_dir: &str, encoder: Option<&str>, gate: f64) -> anyhow::Result<()> {
    let mut enc = match encoder {
        Some(e) => Encoder256::load(std::path::Path::new(e))?,
        None => Encoder256::load_best(model_dir)?,
    };
    let cr_dir = std::path::Path::new(model_dir).join("command_recordings");
    let mut per_cmd: HashMap<String, usize> = HashMap::new();
    // Classification candidates = commands.json ∪ existing template keys (supports command ids containing '_', see classify_recording)
    let cand = load_candidate_cmd_ids(model_dir);
    let mut files: Vec<std::path::PathBuf> = std::fs::read_dir(&cr_dir)
        .with_context(|| format!("failed to read {}", cr_dir.display()))?
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("wav"))
        .collect();
    files.sort();
    for f in &files {
        let name = f.file_name().and_then(|s| s.to_str()).unwrap_or("");
        let cmd = classify_recording(name, &cand);
        if !cmd.is_empty() {
            *per_cmd.entry(cmd).or_insert(0) += 1;
        }
    }
    // embed_proto_v2.json only records which commands have a template key; the acoustic templates come from nuclei_cache.
    let mut doc: HashMap<String, HashMap<String, Vec<f32>>> = HashMap::new();
    let mut cmd_names: Vec<&String> = per_cmd.keys().collect();
    cmd_names.sort();
    for cmd in cmd_names {
        doc.insert(cmd.clone(), HashMap::new());
        println!("  {cmd}: {} recordings", per_cmd[cmd]);
    }
    let out_path = std::path::Path::new(model_dir).join("embed_proto_v2.json");
    // L1: the same encrypted atomic write path as single-command updates
    write_json_atomic_v2(&out_path, &serde_json::to_string_pretty(&doc)?)?;
    // Record the input fingerprint of this template build; startup uses it to decide whether to rebuild automatically
    crate::engine::proto_meta::write_meta(model_dir, &cand);
    println!(
        "saved {} commands -> {} (encoder {})",
        doc.len(),
        out_path.display(),
        enc.src_path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("?")
    );
    // At the end of the same flow: write or update the syllable-nucleus template cache (nuclei_templates_v2.json.enc) --
    // recording and encoder changes are regenerated here, and startup's from_model_dir hits the templates to avoid ~4.5 s of re-encoding.
    // Centroids and priors are computed in the same encoder session (the build resets state per recording);
    // a failure only logs a WARN and does not affect the embed_proto_v2.json result.
    let k = EngineParams::default().k;
    let cmds_all: Vec<String> = per_cmd.keys().cloned().collect();
    match build_nuclei_templates(&mut enc, &cr_dir, &cmds_all, gate) {
        Ok((cents, _prior, med_rows, seqs, exps)) => {
            crate::nuclei_cache::store_full(
                model_dir,
                &enc.src_md5,
                k,
                &cr_dir,
                &cmds_all,
                &cents,
                &med_rows,
                &seqs,
                &exps,
            );
        }
        Err(e) => eprintln!("[engine][WARN] syllable-nucleus template cache skipped (on-the-fly build failed): {e:#}"),
    }
    Ok(())
}

/// Regenerate the v2 template for a single command (the UI's "regenerate pronunciation template" entry):
/// update that command's marker in embed_proto_v2.json and recompute the nuclei template cache;
/// other commands' entries are preserved byte for byte (text-level entry replacement plus atomic write-back).
pub fn gen_embed_proto_single(
    model_dir: &str,
    encoder: Option<&str>,
    command: &str,
    gate: f64,
) -> anyhow::Result<()> {
    anyhow::ensure!(!command.is_empty(), "empty command name");
    let mut enc = match encoder {
        Some(e) => Encoder256::load(std::path::Path::new(e))?,
        None => Encoder256::load_best(model_dir)?,
    };
    let cr_dir = std::path::Path::new(model_dir).join("command_recordings");
    let mut cand = load_candidate_cmd_ids(model_dir);
    if !cand.iter().any(|c| c == command) {
        cand.push(command.to_string());
        cand.sort();
    }
    let mut files: Vec<std::path::PathBuf> = std::fs::read_dir(&cr_dir)
        .with_context(|| format!("failed to read {} (record some takes first)", cr_dir.display()))?
        .flatten()
        .map(|e| e.path())
        .filter(|p| {
            p.extension().and_then(|s| s.to_str()) == Some("wav")
                && p.file_name()
                    .and_then(|s| s.to_str())
                    .map_or(false, |n| classify_recording(n, &cand) == command)
        })
        .collect();
    files.sort();
    anyhow::ensure!(
        !files.is_empty(),
        "command {command} has no recordings in command_recordings"
    );
    update_proto_entry(model_dir, command, "{}")
        .with_context(|| format!("failed to write the template marker for {command}"))?;
    crate::engine::proto_meta::update_command_meta(model_dir, command);
    println!(
        "  {command}: {} recordings -> template marker updated (other entries untouched)",
        files.len()
    );
    let k = EngineParams::default().k;
    {
        crate::nuclei_cache::store_single(model_dir, k, &cr_dir, &cand, command, &mut enc, gate);
    }
    Ok(())
}

/// Replace `command`'s entry with `entry_pretty` (an already serialized entry json) and write it back to
/// embed_proto_v2.json (atomic replacement). If the file exists, replace the entry at text level (other bytes unchanged);
/// otherwise create a document containing only that command.
fn update_proto_entry(model_dir: &str, command: &str, entry_pretty: &str) -> anyhow::Result<()> {
    let path = std::path::Path::new(model_dir).join("embed_proto_v2.json");
    // L1 dual-compatible read (ciphertext or plaintext both serve as the base for a text-level entry replacement)
    let text = crate::crypto::read_text_auto(&path).ok();
    let new_text = splice_proto_entry_text(text.as_deref(), command, entry_pretty)?;
    write_json_atomic_v2(&path, &new_text)
}

pub fn remove_embed_proto_entry(model_dir: &str, command: &str) -> anyhow::Result<()> {
    let path = std::path::Path::new(model_dir).join("embed_proto_v2.json");
    let Ok(text) = crate::crypto::read_text_auto(&path) else {
        return Ok(());
    };
    let new_text = splice_remove_entry_text(&text, command)?;
    if new_text == text {
        return Ok(());
    }
    write_json_atomic_v2(&path, &new_text)
}

/// Deletion variant of splice_proto_entry_text (pure function, unit-testable): remove `command`'s whole entry line
/// (from the start of the entry's line through the end of its value) and fix the seam -- a non-final entry also drops its trailing comma;
/// a final entry drops the trailing comma left by the previous one. A missing entry -> the input comes back unchanged (idempotent).
fn splice_remove_entry_text(text: &str, command: &str) -> anyhow::Result<String> {
    anyhow::ensure!(
        !command.is_empty() && !command.contains('"') && !command.contains('\\'),
        "invalid character in the command name: {command:?}"
    );
    let Some((key_start, _val_start, val_end)) = find_top_level_entry(text, command) else {
        return Ok(text.to_string());
    };
    let line_start = text[..key_start].rfind('\n').map(|i| i + 1).unwrap_or(0);
    let rest = &text[val_end..];
    if let Some(p) = rest.find(',') {
        if rest[..p].chars().all(|c| c.is_ascii_whitespace()) {
            // Non-final entry: drop the trailing comma together with the entry, leaving every other byte as it was
            return Ok(format!("{}{}", &text[..line_start], &rest[p + 1..]));
        }
    }
    // Final entry (value followed directly by the closing '}'): drop the previous entry's trailing comma and re-attach the closing brace
    let close_in_rest = rest
        .find('}')
        .ok_or_else(|| anyhow::anyhow!("embed_proto_v2.json has no closing root brace"))?;
    let before = text[..line_start].trim_end();
    let before = before.strip_suffix(',').unwrap_or(before);
    Ok(format!("{}\n{}", before, &rest[close_in_rest..]))
}

/// Core of entry-level text replacement in embed_proto_v2.json (pure function, unit-testable):
/// `doc_text` = the existing file content (None = create); `entry_pretty` = the serialized text of the new entry
/// (indent base 0). Returns the whole document after replacing or inserting `command`'s entry, **with every byte
/// outside the target entry preserved**.
fn splice_proto_entry_text(
    doc_text: Option<&str>,
    command: &str,
    entry_pretty: &str,
) -> anyhow::Result<String> {
    anyhow::ensure!(
        !command.is_empty() && !command.contains('"') && !command.contains('\\'),
        "invalid character in the command name: {command:?}"
    );
    let Some(text) = doc_text else {
        // New file: only that command (the value gains one level of indentation)
        return Ok(format!(
            "{{\n  \"{command}\": {}\n}}",
            reindent_pretty(entry_pretty, "  ")
        ));
    };
    // The existing document must be a valid json object
    let parsed: serde_json::Value = serde_json::from_str(text).map_err(|e| {
        anyhow::anyhow!("failed to parse the existing embed_proto_v2.json (run a full generation to repair it): {e}")
    })?;
    anyhow::ensure!(parsed.is_object(), "the top level of embed_proto_v2.json is not an object");
    // Find the span of the target key's entry (a text-level scan, avoiding a full re-serialization)
    if let Some((key_start, val_start, val_end)) = find_top_level_entry(text, command) {
        // Indentation of the line holding the target entry (the value starts on the same line as the key)
        let line_start = text[..key_start].rfind('\n').map(|i| i + 1).unwrap_or(0);
        let indent: String = text[line_start..]
            .chars()
            .take_while(|c| c.is_whitespace())
            .collect();
        return Ok(format!(
            "{}{}{}",
            &text[..val_start],
            reindent_pretty(entry_pretty, &indent),
            &text[val_end..]
        ));
    }
    // Missing -> insert at the end of the root object (keeping other bytes; matches the existing trailing newline and indent style)
    let close = text
        .rfind('}')
        .with_context(|| "embed_proto_v2.json: trailing bytes after the root object")?;
    let before = &text[..close];
    let trim_len = before.trim_end().len();
    anyhow::ensure!(trim_len > 0, "embed_proto_v2.json is an empty document");
    let last_line_start = before[..trim_len].rfind('\n').map(|i| i + 1).unwrap_or(0);
    let indent: String = before[last_line_start..trim_len]
        .chars()
        .take_while(|c| c.is_whitespace())
        .collect();
    if before.as_bytes()[trim_len - 1] == b'{' {
        // Empty object {} or a single newline between the braces
        return Ok(format!(
            "{{\n  \"{command}\": {}{}",
            reindent_pretty(entry_pretty, "  "),
            &text[close..]
        ));
    }
    Ok(format!(
        "{},\n{}\"{}\": {}{}",
        &before[..trim_len],
        indent,
        command,
        reindent_pretty(entry_pretty, &indent),
        &text[close..]
    ))
}

/// Deepen the whole to_string_pretty output (indent base 0) by `indent` (not the first line -- the value shares the key's line)
fn reindent_pretty(val_pretty: &str, indent: &str) -> String {
    let mut out = String::with_capacity(val_pretty.len() + indent.len() * 4);
    for (n, line) in val_pretty.split('\n').enumerate() {
        if n > 0 {
            out.push('\n');
            out.push_str(indent);
        }
        out.push_str(line);
    }
    out
}

/// Top-level key scan: returns (key start, value start, value end) -- the value span excludes the trailing comma and whitespace.
/// Only recognizes `"k": v` top-level keys (the embed_proto_v2.json shape; a minimal scanner, no json-span library needed).
fn find_top_level_entry(text: &str, key: &str) -> Option<(usize, usize, usize)> {
    let b = text.as_bytes();
    let mut i = 0usize;
    macro_rules! skip_ws {
        () => {
            while i < b.len() && b[i].is_ascii_whitespace() {
                i += 1;
            }
        };
    }
    skip_ws!();
    if i >= b.len() || b[i] != b'{' {
        return None;
    }
    i += 1;
    loop {
        skip_ws!();
        if i >= b.len() || b[i] == b'}' {
            return None; // Key missing / early end
        }
        if b[i] != b'"' {
            return None; // Invalid structure
        }
        let key_start = i;
        i += 1;
        let ks = i;
        while i < b.len() && b[i] != b'"' {
            if b[i] == b'\\' {
                i += 1;
            }
            i += 1;
        }
        if i >= b.len() {
            return None;
        }
        let k = &text[ks..i];
        i += 1; // Skip past the key's closing quote
        skip_ws!();
        if i >= b.len() || b[i] != b':' {
            return None;
        }
        i += 1;
        skip_ws!();
        let val_start = i;
        // Value span: bracket-depth counting plus string awareness
        let mut depth = 0i32;
        let mut in_str = false;
        while i < b.len() {
            let c = b[i];
            if in_str {
                if c == b'\\' {
                    i += 2;
                    continue;
                }
                if c == b'"' {
                    in_str = false;
                }
            } else {
                match c {
                    b'"' => in_str = true,
                    b'{' | b'[' => depth += 1,
                    b'}' | b']' => {
                        depth -= 1;
                        if depth == 0 {
                            i += 1;
                            if k == key {
                                return Some((key_start, val_start, i));
                            }
                            break;
                        }
                    }
                    _ => {}
                }
            }
            i += 1;
        }
        if i >= b.len() {
            return None;
        }
        skip_ws!();
        if i < b.len() && b[i] == b',' {
            i += 1;
            continue;
        }
        if i < b.len() && b[i] == b'}' {
            return None; // Key missing (closing root object)
        }
        return None; // Invalid structure
    }
}

pub(crate) fn write_json_atomic_v2(path: &std::path::Path, content: &str) -> anyhow::Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            let _ = std::fs::create_dir_all(parent);
        }
    }
    let tmp = path.with_extension("tmp");
    let payload = crate::crypto::protect(content.as_bytes())
        .map_err(|e| anyhow::anyhow!("failed to encrypt {}: {e}", path.display()))?;
    std::fs::write(&tmp, &payload)?;
    match std::fs::rename(&tmp, path) {
        Ok(()) => Ok(()),
        Err(e) => {
            let _ = std::fs::remove_file(&tmp);
            Err(e).with_context(|| format!("atomic replace of {} failed", path.display()))
        }
    }
}

// -- Unit tests --

/// The repository's v2 encoder (tests only; release binaries never embed local paths).
#[cfg(test)]
fn repo_encoder() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("assets/model/encoder-v2-chunk-8.int8.onnx")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tpl_act_rel_is_gain_invariant_and_lands_on_speech() {
        // 1 s: 0.30-0.60 s is "speech" (amplitude 0.20) and the rest is "silence" (amplitude 0.002) -- roughly 40 dB apart
        let synth = |scale: f32| -> Vec<f32> {
            (0..16000)
                .map(|i| {
                    let t = i as f32 / 16000.0;
                    let amp = if (0.30..0.60).contains(&t) {
                        0.20
                    } else {
                        0.002
                    };
                    amp * (i as f32 * 0.3).sin() * scale
                })
                .collect()
        };
        let a = act_flags_10ms_rel(&synth(1.0));
        let b = act_flags_10ms_rel(&synth(0.1)); // Overall −20 dB
        assert_eq!(a, b, "the relative gate must be invariant to overall gain (per-frame flags identical)");
        let (a_s, b_s) = active_range_secs(&a);
        assert!((0.20..0.40).contains(&a_s), "the window start must land in the speech region: a_s={a_s}");
        assert!((0.50..0.70).contains(&b_s), "the window end must land in the speech region: b_s={b_s}");
        // Contrast: the old hard-coded −40 dB gate necessarily fails at an overall −30 dB level -- speech peaks drop to about −47 dB and are all cut.
        // (−20 dB speech is still −37 dB and barely clears −40, which is why −30 dB is used to reproduce the collapse explicitly.)
        let (a_s_abs, b_s_abs) = active_range_secs(&act_flags_10ms(&synth(0.03), -40.0));
        assert!(
            (b_s_abs - a_s_abs) + 0.10 < (b_s - a_s),
            "the old absolute gate must collapse clearly: abs={a_s_abs}..{b_s_abs} rel={a_s}..{b_s}"
        );
    }

    /// Relative floor: an almost constant low-level noise (range → 0) must not be judged as a large activity region.
    #[test]
    fn tpl_act_rel_floor_rejects_flat_noise() {
        let s: Vec<f32> = (0..16000).map(|i| 0.001 * (i as f32 * 0.7).sin()).collect();
        let act = act_flags_10ms_rel(&s);
        let n = act.iter().filter(|&&x| x).count();
        assert!(n < act.len() / 5, "pure noise must not produce a large active region: {n}/{}", act.len());
    }

    /// Convenience constructor: (peak frame, cmd, score); start/end are synthesised as peak ±0.04; emb = a zero vector
    /// (merge and staging semantics do not depend on the embedding; sequence-template tests build nuclei with emb_nuc)
    fn nuc(pf: usize, cmd: usize, score: f64) -> NucJudged {
        NucJudged {
            pf,
            cmd,
            score,
            start: (pf as f64 * 0.01 - 0.04) as f32,
            end: (pf as f64 * 0.01 + 0.04) as f32,
            emb: Vec::new(),
        }
    }

    /// Sequence-template test constructor: with embeddings (dim=4 one-hot directions, the index picks the direction, for exact cosine assertions)
    fn emb_nuc(pf: usize, cmd: usize, score: f64, dirs: &[usize]) -> NucJudged {
        assert_eq!(dirs.len(), 1);
        let mut e = vec![0.0f32; 4];
        e[dirs[0] % 4] = 1.0;
        NucJudged {
            pf,
            cmd,
            score,
            start: (pf as f64 * 0.01 - 0.04) as f32,
            end: (pf as f64 * 0.01 + 0.04) as f32,
            emb: e,
        }
    }

    /// Build an emission context (exp/seq/min_conf; no diag output)
    fn ctx<'a>(exp: &'a [usize], seq: &'a [Option<CmdSeqTemplate>], min_conf: f64) -> EmitCtx<'a> {
        EmitCtx {
            exp,
            seq,
            min_conf,
            force_dtw: false,
            diag: false,
            cmds: &[],
        }
    }

    #[test]
    fn merge_burst_groups_adjacent_same_cmd_merges() {
        // Adjacent same-command [c,c] with cap=2 -> one entry: score = max, t/start = first nucleus, end = last nucleus
        let out = merge_burst_groups(&[nuc(30, 0, 0.51), nuc(62, 0, 0.72)], &[2]);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0], (0.72, 30, 0, 0.26, 0.66));
    }

    #[test]
    fn merge_burst_groups_different_cmd_never_merges() {
        let nucs = [nuc(30, 0, 0.7), nuc(90, 1, 0.6), nuc(150, 0, 0.5)];
        let out = merge_burst_groups(&nucs, &[4, 4]);
        assert_eq!(out.len(), 3);
        assert_eq!(out[0].2, 0);
        assert_eq!(out[1].2, 1);
        assert_eq!(out[2].2, 0);
    }

    #[test]
    fn merge_burst_groups_per_cmd_caps_independent() {
        // Per-command cap: cmd0=2, cmd1=4 -- [c1×4] merges into one, [c0×3] splits into two
        let nucs = [
            nuc(10, 1, 0.5),
            nuc(30, 1, 0.6),
            nuc(50, 1, 0.55),
            nuc(70, 1, 0.65),
            nuc(200, 0, 0.5),
            nuc(220, 0, 0.6),
            nuc(240, 0, 0.7),
        ];
        let out = merge_burst_groups(&nucs, &[2, 4]);
        assert_eq!(out.len(), 3);
        assert_eq!(out[0], (0.65, 10, 1, 0.06, 0.74));
        assert_eq!(out[1].1, 200);
        assert_eq!(out[2].1, 240);
    }

    #[test]
    fn stage_append_matches_offline_grouping() {
        // Live staging grouping is count-equivalent to offline full grouping: [c0,c0,c1,c1,c0] with cap=[2,4] -> 3 triggers
        let mut stage: Option<(u64, Vec<NucJudged>)> = None;
        let mut acc: Vec<Accepted> = Vec::new();
        let mut refr = 0usize;
        let c = ctx(&[2, 4], &[], 0.45);
        let seq = 7u64;
        for n in [
            nuc(30, 0, 0.5),
            nuc(60, 0, 0.7),
            nuc(90, 1, 0.6),
            nuc(120, 1, 0.65),
            nuc(150, 0, 0.55),
        ] {
            stage_append(&c, &mut stage, seq, n, &mut acc, 0, &mut refr);
        }
        // The last group [c0] has not reached its cap and the burst is still open -> staging is kept
        assert_eq!(acc.len(), 2);
        assert_eq!(acc[0], (0.7, 30, 0, 0.26, 0.64));
        assert_eq!(acc[1], (0.65, 90, 1, 0.86, 1.24));
        // Burst closed and everything judged -> the last group emits
        if let Some((_, list)) = &mut stage {
            let l = std::mem::take(list);
            flush_stage_run(&c, &l, &mut acc, 0, &mut refr);
        }
        assert_eq!(acc.len(), 3);
        assert_eq!(acc[2], (0.55, 150, 0, 1.46, 1.54));
    }

    #[test]
    fn stage_append_new_burst_flushes_residual() {
        // First nucleus of a new burst arrives: every leftover of the old burst (closed, last group under cap) emits
        let mut stage: Option<(u64, Vec<NucJudged>)> = None;
        let mut acc: Vec<Accepted> = Vec::new();
        let mut refr = 0usize;
        let c = ctx(&[2], &[], 0.45);
        stage_append(&c, &mut stage, 1, nuc(30, 0, 0.6), &mut acc, 0, &mut refr);
        assert_eq!(acc.len(), 0); // Under the cap, keep it staged
        stage_append(&c, &mut stage, 2, nuc(200, 0, 0.7), &mut acc, 0, &mut refr);
        assert_eq!(acc.len(), 1); // The old burst's leftover emits one entry; the new burst stays staged under the cap
        assert_eq!(acc[0], (0.6, 30, 0, 0.26, 0.34));
        // The new burst closes -> the held single-nucleus group emits
        if let Some((_, list)) = &mut stage {
            let l = std::mem::take(list);
            flush_stage_run(&c, &l, &mut acc, 0, &mut refr);
        }
        assert_eq!(acc.len(), 2);
        assert_eq!(acc[1], (0.7, 200, 0, 1.96, 2.04));
    }

    #[test]
    fn refr_chain_swallows_burst_echo_and_extends_window() {
        // Echo chain within a burst: pf100 really fires -> watermark 157; echo pf150 < 157 -> swallowed, watermark extends to 207
        // (the suppression chain -- a swallowed echo also extends the window); echo pf160 < 207 -> swallowed, extends to 217;
        // a legitimate follow-up at pf300 >= 217 -> fires.
        let mut stage: Option<(u64, Vec<NucJudged>)> = None;
        let mut acc: Vec<Accepted> = Vec::new();
        let mut refr = 0usize;
        let c = ctx(&[1], &[], 0.45);
        for n in [
            nuc(100, 0, 0.7),
            nuc(150, 0, 0.8),
            nuc(160, 0, 0.6),
            nuc(300, 0, 0.7),
        ] {
            stage_append(&c, &mut stage, 1, n, &mut acc, 57, &mut refr);
        }
        assert_eq!(acc.len(), 2, "both echo nuclei must be swallowed: {acc:?}");
        assert_eq!(acc[0].1, 100);
        assert_eq!(acc[1].1, 300);
        assert_eq!(refr, 357);
    }

    #[test]
    fn refr_persists_across_bursts() {
        // Across bursts: burst1 really fires at pf100 (watermark 157); burst2's first nucleus pf150 is an echo ->
        // the leftovers flush but must still clear the watermark -> swallowed (reopening a burst is not a back door); burst3 pf400 -> fires.
        let mut stage: Option<(u64, Vec<NucJudged>)> = None;
        let mut acc: Vec<Accepted> = Vec::new();
        let mut refr = 0usize;
        let c = ctx(&[1], &[], 0.45);
        stage_append(&c, &mut stage, 1, nuc(100, 0, 0.7), &mut acc, 57, &mut refr);
        assert_eq!(acc.len(), 1);
        stage_append(&c, &mut stage, 2, nuc(150, 0, 0.8), &mut acc, 57, &mut refr);
        assert_eq!(acc.len(), 1, "the cross-burst echo must be swallowed: {acc:?}");
        stage_append(&c, &mut stage, 3, nuc(400, 0, 0.7), &mut acc, 57, &mut refr);
        assert_eq!(acc.len(), 2);
        assert_eq!(acc[1].1, 400);
    }

    #[test]
    fn refr_does_not_clip_exp2_word_instances() {
        // exp=2 command: a legitimate sequence with a 0.30 s nucleus gap merges into a single emission at the word-instance layer (recording the first nucleus's pf);
        // two utterances whose first nuclei are 1.7 s apart (> 0.57 s) -> both fire -- the case nucleus-level suppression would kill.
        let mut stage: Option<(u64, Vec<NucJudged>)> = None;
        let mut acc: Vec<Accepted> = Vec::new();
        let mut refr = 0usize;
        let c = ctx(&[2], &[], 0.45);
        for n in [
            nuc(100, 0, 0.7),
            nuc(130, 0, 0.8),
            nuc(270, 0, 0.6),
            nuc(300, 0, 0.7),
        ] {
            stage_append(&c, &mut stage, 1, n, &mut acc, 57, &mut refr);
        }
        assert_eq!(acc.len(), 2, "both word instances of the exp=2 command must emit: {acc:?}");
        assert_eq!((acc[0].1, acc[1].1), (100, 270));
    }

    /// 4-dim one-hot direction embeddings and centroids (for exact cosine assertions)
    fn dirv(idx: usize) -> Vec<f32> {
        let mut v = vec![0.0f32; 4];
        v[idx % 4] = 1.0;
        v
    }

    #[test]
    fn dtw_align_score_known_small_cases() {
        // Identical sequences (k=n=3) -> the diagonal path, mean 1.0
        let a = vec![dirv(0), dirv(1), dirv(2)];
        let b = vec![dirv(0), dirv(1), dirv(2)];
        assert_eq!(dtw_align_score(&a, &b, 1), Some(1.0));
        // One extra nucleus (fragmented, k=4 vs n=3): absorbable within ±1 -> still at or above some high value
        let a4 = vec![dirv(0), dirv(0), dirv(1), dirv(2)];
        let s = dtw_align_score(&a4, &b, 1).unwrap();
        assert!(s > 0.8, "the mean must stay high after absorbing the fragmented nucleus (got {s})");
        // Length gap beyond the band (k=1, n=3) -> no feasible path
        let a1 = vec![dirv(0)];
        assert_eq!(dtw_align_score(&a1, &b, 1), None);
        // Orthogonal sequences -> mean 0
        let orth = vec![dirv(3), dirv(3), dirv(3)];
        let s0 = dtw_align_score(&a, &orth, 1).unwrap();
        assert!(s0.abs() < 1e-6, "the mean cosine of orthogonal sequences must be 0 (got {s0})");
    }

    #[test]
    fn seq_score_positional_gates_per_position() {
        // Template: a 2-character sequence [direction 0, direction 1]
        let seq = CmdSeqTemplate {
            recs: vec![vec![dirv(0), dirv(1)]],
            n: 2,
        };
        // Embeddings identical to the template -> fast path Some(1.0)
        let emb = vec![dirv(0), dirv(1)];
        let (s, mode, f, d) = seq_score_instance(&emb, &seq, 0.45, false).unwrap();
        assert_eq!(mode, "fast");
        assert_eq!(s, 1.0);
        assert_eq!(f, Some(1.0));
        assert!(d.is_some(), "control criterion: the DTW score is always computed");
        // Second position swapped to a mixed direction (cos 0.8 < the 0.85 threshold) -> the fast path fails and DTW takes over
        let mut weak = vec![0.6f32, 0.8, 0.0, 0.0]; // Cosine against dirv(1) = 0.8
        let n = (weak.iter().map(|x| *x as f64 * *x as f64).sum::<f64>()).sqrt();
        for x in weak.iter_mut() {
            *x = (*x as f64 / n) as f32;
        }
        let emb2 = vec![dirv(0), weak];
        let (s2, mode2, f2, _) = seq_score_instance(&emb2, &seq, 0.85, false).unwrap();
        assert_eq!(f2, None, "per-position threshold: any position below min_conf disables the fast path");
        assert_eq!(mode2, "dtw");
        assert!(s2 >= 0.85, "the DTW fallback score must pass the threshold (got {s2})");
    }

    #[test]
    fn emit_word_instance_one_vote_and_fallback() {
        // 2-character template; a run of two nuclei matching it exactly -> one-vote emission (start/t = first nucleus, end = last nucleus)
        let seq = CmdSeqTemplate {
            recs: vec![vec![dirv(0), dirv(1)]],
            n: 2,
        };
        let run = vec![emb_nuc(30, 0, 0.5, &[0]), emb_nuc(62, 0, 0.7, &[1])];
        let out = emit_word_instance(&run, Some(&seq), 2, 0.45, false, false, "cmdX");
        assert_eq!(out.len(), 1);
        assert_eq!(out[0], (1.0, 30, 0, 0.26, 0.66));
        // The run is orthogonal to the template -> below threshold -> falls back to grouping (cap=2 merges into one entry, score = group max)
        let run_bad = vec![emb_nuc(30, 0, 0.5, &[2]), emb_nuc(62, 0, 0.7, &[3])];
        let out2 = emit_word_instance(&run_bad, Some(&seq), 2, 0.45, false, false, "cmdX");
        assert_eq!(out2.len(), 1);
        assert_eq!(out2[0], (0.7, 30, 0, 0.26, 0.66));
        // No template -> pure A-grouping (single-character semantics)
        let out3 = emit_word_instance(&run, None, 2, 0.45, false, false, "cmdX");
        assert_eq!(out3.len(), 1);
        assert_eq!(out3[0], (0.7, 30, 0, 0.26, 0.66));
    }

    #[test]
    fn emit_word_instance_dtw_tolerates_extra_nucleus() {
        // Fragmented case: a 3-nucleus run (double nucleus in the first character) against a 2-character template -> DTW aligns above threshold -> one vote
        let seq = CmdSeqTemplate {
            recs: vec![vec![dirv(0), dirv(1)]],
            n: 2,
        };
        let run = vec![
            emb_nuc(30, 0, 0.5, &[0]),
            emb_nuc(50, 0, 0.5, &[0]),
            emb_nuc(90, 0, 0.6, &[1]),
        ];
        let out = emit_word_instance(&run, Some(&seq), 2, 0.45, false, false, "cmdX");
        assert_eq!(out.len(), 1, "a fragmented nucleus aligned by DTW must emit on one vote");
        assert_eq!(out[0], (out[0].0, 30, 0, 0.26, 0.94));
    }

    #[test]
    fn from_model_dir_empty_proto_builds_empty_engine_zero_trigger() {
        if !repo_encoder().exists() {
            println!("skipped (the repository has no encoder under assets/model)");
            return;
        }
        let dir = std::env::temp_dir().join(format!(
            "viq_v2_empty_proto_{}_{}",
            std::process::id(),
            line!()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::copy(repo_encoder(), dir.join("encoder-v2-chunk-8.int8.onnx")).unwrap();

        // 1) The proto file is entirely missing -> construction succeeds with 0 commands
        let mut eng = Engine::from_model_dir(dir.to_str().unwrap(), &[] as &[String]).unwrap();
        assert!(eng.cmds().is_empty());
        assert!(eng.nuc.is_none());
        synthetic_session_zero_trigger(&mut eng);
        drop(eng);

        // 2) proto = empty object {} -> likewise 0 commands; and a non-empty enabled-command set (commands enabled before registering)
        // must not bail either (the common first-run order: enable commands, then register pronunciations)
        std::fs::write(dir.join("embed_proto_v2.json"), "{}").unwrap();
        let names = vec!["one".to_string(), "two".to_string()];
        let mut eng = Engine::from_model_dir(dir.to_str().unwrap(), &names).unwrap();
        assert!(eng.cmds().is_empty());
        assert!(eng.nuc.is_none());
        synthetic_session_zero_trigger(&mut eng);

        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// Synthetic "session": 0.5 s silence + 3× (0.5 s 440 Hz sine) + 0.5 s silence, streamed in 10 ms steps;
    /// with an empty command set, poll/finish/take_ready all yield zero triggers.
    fn synthetic_session_zero_trigger(eng: &mut Engine) {
        eng.set_streaming(true);
        let mut samples: Vec<f32> = Vec::new();
        for _ in 0..3 {
            samples.extend(std::iter::repeat(0.0f32).take(8000));
            for i in 0..8000 {
                samples.push(0.3 * (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 16000.0).sin());
            }
        }
        samples.extend(std::iter::repeat(0.0f32).take(8000));
        let mut pos = 0usize;
        while pos < samples.len() {
            let end = (pos + 160).min(samples.len());
            eng.push_samples(&samples[pos..end]);
            pos = end;
            assert!(eng.poll().is_empty(), "an empty command set must not trigger");
        }
        eng.finish();
        assert!(eng.take_ready().is_empty(), "finishing with an empty command set must not emit triggers");
    }

    #[test]
    fn p2_memory_caps_diag_and_act_frames() {
        // 1) diag cap
        let mut diag = Vec::new();
        for i in 0..25000usize {
            push_diag_capped(&mut diag, (i, i, 0, i as f64));
        }
        assert!(diag.len() <= 22000, "diag must stay bounded, got {}", diag.len());
        assert_eq!(
            diag.first().unwrap().0,
            25000 - diag.len(),
            "the head must be dropped (the first entry equals the dropped count)"
        );
        assert_eq!(diag.last().unwrap().0, 24999, "the most recent entries in the tail are kept");

        // 2) act_frames head trimming and rebasing (pure functions)
        let mut act = vec![false; 5000];
        let mut base = 0usize;
        trim_act_head(&mut act, &mut base, 3000);
        assert_eq!(act.len(), 3000, "act_frames must be trimmed to the 30 s retention window");
        assert_eq!(base, 2000, "the trimmed head count must be reflected in act_base");
        assert_eq!(base + act.len(), 5000, "base+len identity");
        trim_act_head(&mut act, &mut base, 3000);
        assert_eq!((act.len(), base), (3000, 2000), "no trimming below the limit (idempotent)");
    }

    fn e2_gate_params_adaptive() -> EngineParams {
        EngineParams {
            act_gate_fixed: false,
            ..EngineParams::default()
        }
    }

    /// Noise-floor EMA convergence: nf converges exponentially from −50 towards −60, closed form nf_n = −60 + 10·0.95^n
    /// (α=0.05; the gate follows towards nf+6). Checked against the analytic form, in the same style as floor_track_asymmetric_rates.
    #[test]
    fn act_gate_noise_floor_ema_converges() {
        let p = e2_gate_params_adaptive();
        let mut nf: Option<f64> = Some(-50.0);
        for n in 1..=100usize {
            let g = act_gate_step(&mut nf, -60.0, &p);
            let expect_nf = -60.0 + 10.0 * 0.95f64.powi(n as i32);
            assert!(
                (nf.unwrap() - expect_nf).abs() < 1e-9,
                "the floor at step {n} must equal the closed form {expect_nf:.4}, got {}",
                nf.unwrap()
            );
            // A −60 dB frame stays below gate−6 (true throughout while the gate <= −44) -> updated every frame
            // The gate uses the pre-update floor: g_n = nf_{n−1} + margin
            let expect_prev = -60.0 + 10.0 * 0.95f64.powi(n as i32 - 1);
            assert!(
                (g - (expect_prev + 6.0)).abs() < 1e-9,
                "gate = previous floor + margin"
            );
        }
        assert!(
            (nf.unwrap() + 60.0).abs() < 0.1,
            "the floor must converge within ±0.1 dB after 100 steps"
        );
    }

    /// Gate clamping: the floor (very low noise) and ceiling (very high noise) bounds take effect,
    /// and loud frames fail the floor-sample test, so the estimate is never polluted by speech.
    #[test]
    fn act_gate_clamped_to_floor_and_ceiling() {
        let p = e2_gate_params_adaptive();
        // floor: a −120 dB noise floor (digital silence) -> the gate clamps at −70
        let mut nf = Some(-120.0);
        assert_eq!(act_gate_step(&mut nf, -120.0, &p), p.act_gate_floor);
        let mut nf = Some(-20.0);
        assert_eq!(act_gate_step(&mut nf, -15.0, &p), p.act_gate_ceiling);
        assert_eq!(nf, Some(-20.0), "speech-level frames must not update the floor");
        // Grey-band frames at the ceiling (db=−19, between gate−30 and gate+12=−18) do creep upward (for comparison)
        let mut nf2 = Some(-20.0);
        let _ = act_gate_step(&mut nf2, -19.0, &p);
        assert!(nf2.unwrap() > -20.0, "grey-band frames still creep the floor up while clamped by the ceiling");
    }

    /// Explicit setting override: takes effect, clamps out-of-range values and leaves None fields alone (the "no restart needed after changing settings" contract).
    #[test]
    fn engine_overrides_apply_and_clamp() {
        let mut p = EngineParams::default();
        let base_refr = p.refr_s;
        let base_min = p.min_conf;
        let base_gate = p.act_db;
        EngineOverrides {
            refr_s: Some(0.15),
            min_conf: Some(0.8),
            act_db: Some(-58.0),
            auto_gate: Some(true),
        }
        .apply_to(&mut p);
        assert_eq!(p.refr_s, 0.15);
        assert_eq!(p.min_conf, 0.8);
        assert_eq!(p.act_db, -58.0);
        // Out-of-range clamping (the same rule as the settings-page input validation)
        EngineOverrides {
            refr_s: Some(9.0),
            min_conf: Some(5.0),
            act_db: Some(-200.0),
            auto_gate: None,
        }
        .apply_to(&mut p);
        assert_eq!(p.refr_s, 2.0, "refractory cap is 2 s");
        assert_eq!(p.min_conf, 0.9, "min_conf cap is 0.9");
        assert_eq!(p.act_db, -75.0, "act gate floor is -75 dB");
        // None fields do not overwrite existing values
        let mut q = EngineParams::default();
        EngineOverrides::default().apply_to(&mut q);
        assert_eq!(q.refr_s, base_refr);
        assert_eq!(q.min_conf, base_min);
        assert_eq!(q.act_db, base_gate);
    }

    #[test]
    fn act_gate_gray_band_lifts_toward_ambient() {
        let p = e2_gate_params_adaptive();
        let mut nf: Option<f64> = Some(-64.6);
        for n in 1..=2000usize {
            let g = act_gate_step(&mut nf, -50.0, &p);
            let expect_nf = -50.0 - 14.6 * 0.999f64.powi(n as i32);
            if matches!(n, 1 | 100 | 1000) {
                assert!(
                    (nf.unwrap() - expect_nf).abs() < 1e-9,
                    "the floor at step {n} must equal the closed form {expect_nf:.4}, got {}",
                    nf.unwrap()
                );
            }
            let _ = g;
        }
        assert!(
            nf.unwrap() >= -56.0,
            "after 2000 frames (20 s) the noise floor should rise to >= -56 (gate >= -50: ambient counts as silence)"
        );
        // The last frame's gate uses the previous floor: gate = nf_1999 + 6 >= −50 -> that frame at −50 is no longer active
        let g = act_gate_step(&mut nf, -50.0, &p);
        assert!(g >= -50.0, "after convergence an ambient frame must be judged silent (gate {g} >= -50)");
    }

    #[test]
    fn template_act_flags_adaptive_includes_weak_second_syllable() {
        let sr = 16000usize;
        let seg = |samples: &mut Vec<f32>, secs: f64, amp: f64, freq: f64| {
            for i in 0..(secs * sr as f64) as usize {
                let t = i as f64 / sr as f64;
                samples.push((amp * (2.0 * std::f64::consts::PI * freq * t).sin()) as f32);
            }
        };
        let mut samples = Vec::new();
        // Layout: strong 0.4 s (−13 dB) -> gap 0.2 s (pure floor) -> weak 0.4 s (−53 dB) -> floor 0.2 s
        // A 50 Hz @ −70 dB noise floor runs throughout (RMS = a/√2 -> a = 10^(−70/20)·√2 ≈ 4.47e−4)
        let bed = 4.47e-4f32;
        let n = (1.2 * sr as f64) as usize;
        for i in 0..n {
            let t = i as f64 / sr as f64;
            samples.push(bed * (2.0 * std::f64::consts::PI * 50.0 * t).sin() as f32);
        }
        for (off, secs, amp, freq) in [
            (0.0, 0.4, 0.2236, 300.0), // strong syllable
            (0.6, 0.4, 0.0022, 300.0), // weak syllable (below -40, above the adaptive gate)
        ] {
            let s0 = (off * sr as f64) as usize;
            let mut syl = Vec::new();
            seg(&mut syl, secs, amp, freq);
            for (k, x) in syl.into_iter().enumerate() {
                samples[s0 + k] += x;
            }
        }
        let fixed = act_flags_10ms(&samples, -40.0);
        assert!(EngineParams::default().act_gate_fixed, "the default must be a fixed gate");
        let n = |v: &[bool]| v.iter().filter(|&&x| x).count();
        assert_eq!(n(&fixed), 40, "fixed -40: only the strong syllable is seen");
        assert_eq!(
            n(&act_flags_10ms_adaptive(&samples, &EngineParams::default())),
            40,
            "default fixed gate: template side uses the same rule; the weak syllable (-53 < -40) is not counted"
        );
        assert_eq!(
            n(&act_flags_10ms_adaptive(
                &samples,
                &EngineParams {
                    act_gate_fixed: false,
                    ..EngineParams::default()
                }
            )),
            80,
            "explicit adaptive gate (opt-in): both syllables counted (template exp=2 after symmetry)"
        );
    }

    #[test]
    fn act_gate_low_amplitude_attenuation_adaptive_detects() {
        let sr = 16000usize;
        let mut samples: Vec<f32> = Vec::new();
        // 100 Hz noise floor (whole cycles per frame -> no frame-RMS ripple): A=8e−4 -> RMS −64.9 dB
        let (noise_a, tone_a) = (8e-4f64, 0.1f64);
        let seg = |samples: &mut Vec<f32>, secs: f64, amp: f64, freq: f64| {
            for i in 0..(secs * sr as f64) as usize {
                let t = i as f64 / sr as f64;
                samples.push((amp * (2.0 * std::f64::consts::PI * freq * t).sin()) as f32);
            }
        };
        seg(&mut samples, 0.5, noise_a, 100.0);
        for _ in 0..3 {
            seg(&mut samples, 0.3, tone_a, 400.0);
            seg(&mut samples, 0.3, noise_a, 100.0);
        }
        let frame_n = samples.len() / 160;

        // Contrast: |att| × gate mode -> (active frame count, number of active runs, final gate)
        let run = |fixed: bool, att_db: f64| -> (usize, usize, f64) {
            let scale = 10f64.powf(att_db / 20.0);
            let scaled: Vec<f32> = samples.iter().map(|&s| (s as f64 * scale) as f32).collect();
            let env = energy_db_env(&scaled);
            assert_eq!(env.len(), frame_n);
            let p = EngineParams {
                act_gate_fixed: fixed,
                act_db: -40.0,
                ..EngineParams::default()
            };
            let mut nf: Option<f64> = None;
            let mut act = 0usize;
            let mut runs = 0usize;
            let mut prev = false;
            let mut last_gate = f64::NAN;
            for &db in &env {
                let gate = act_gate_step(&mut nf, db, &p);
                let a = db > gate;
                if a {
                    act += 1;
                }
                if a && !prev {
                    runs += 1;
                }
                prev = a;
                last_gate = gate;
            }
            (act, runs, last_gate)
        };

        // Expected values (analytic): a syllable region is 3×0.30 s = 90 frames; the adaptive gate is floor+6, and once the floor decays below
        // −71 dB the gate hits the floor clamp (−20/−30 dB levels give a −84.9/−94.9 floor -> gate = −70)
        let tone_frames = 90usize;
        let pd = e2_gate_params_adaptive();
        let noise_db0 = 20.0 * (noise_a / 2f64.sqrt()).log10();
        for &att in &[0.0, -20.0, -30.0] {
            let (fa, fr, _) = run(true, att);
            let (aa, ar, ag) = run(false, att);
            if att == 0.0 {
                assert_eq!(fa, tone_frames, "fixed gate 0 dB: the syllable segment is active as usual");
            } else {
                assert_eq!(fa, 0, "fixed gate @{att} dB: all frames blocked (reproduces the zero-recognition root cause)");
                assert_eq!(fr, 0);
            }
            assert_eq!(aa, tone_frames, "adaptive gate @{att} dB: the syllable segment is fully detected");
            assert_eq!(ar, 3, "adaptive gate @{att} dB: 3 contiguous active segments");
            let expect_gate = ((noise_db0 + att) + pd.act_gate_margin)
                .clamp(pd.act_gate_floor, pd.act_gate_ceiling);
            assert!(
                (ag - expect_gate).abs() < 0.1,
                "adaptive gate @{att} dB should be {expect_gate:.2} (noise floor + 6, floor clamp in effect); got {ag:.2}"
            );
        }
    }

    // -- Unit tests for the nucleus-counting pure functions (synthetic envelopes with known counts) --

    /// Synthetic dB envelope: bumps = (centre seconds, peak dB, half-width seconds); a parabolic envelope in the dB domain over a −60 dB floor
    fn synthetic_env(bumps: &[(f64, f64, f64)], dur_sec: f64) -> Vec<f64> {
        let n = (dur_sec / 0.01) as usize;
        let mut env = vec![-60.0f64; n];
        for &(c, amp, w) in bumps {
            for (i, e) in env.iter_mut().enumerate() {
                let t = i as f64 * 0.01;
                let v = amp - 60.0 * (t - c) * (t - c) / (w * w);
                if v > *e {
                    *e = v;
                }
            }
        }
        env
    }

    /// count_nuclei: four deep-valley bumps -> 4 nuclei (identical at valley depths 2 and 6 dB)
    #[test]
    fn count_nuclei_deep_valleys() {
        let env = synthetic_env(
            &[
                (0.20, 0.0, 0.05),
                (0.50, -1.0, 0.05),
                (0.80, -2.0, 0.05),
                (1.10, -3.0, 0.05),
            ],
            1.5,
        );
        for valley in [2.0f64, 6.0] {
            let (n, peaks) = count_nuclei(&env, 0, env.len(), 99.0, 25.0, valley, 12, 5);
            assert_eq!(n, 4, "valley depth {valley} should yield 4 nuclei, peaks={peaks:?}");
            // Peak positions should land within ±30 ms of each bump centre
            for (p, c) in peaks.iter().zip([0.20, 0.50, 0.80, 1.10]) {
                assert!((*p as f64 * 0.01 - c).abs() < 0.03);
            }
        }
    }

    #[test]
    fn smooth_window_bounds_no_underflow_at_session_start() {
        assert_eq!(smooth_window_bounds(0, 0), (0, 2));
        assert_eq!(smooth_window_bounds(1, 0), (0, 3));
        assert_eq!(smooth_window_bounds(2, 0), (0, 4));
        // The normal path (i >= 2) matches the original expression bit for bit; the env_base > 0 clamp is unchanged
        assert_eq!(smooth_window_bounds(105, 0), (103, 107));
        assert_eq!(smooth_window_bounds(10, 8), (8, 12));
        let env = vec![-40.0f64; 105];
        let env_base = 0usize;
        let env_end = env_base + env.len();
        let mut counted_upto = 0usize; // = a10.max(consumed_watermark) = 0
        let mut n = 0usize;
        while counted_upto + 2 <= env_end.saturating_sub(1) {
            let i = counted_upto;
            let (a, b) = smooth_window_bounds(i, env_base);
            let x = env[a - env_base..=b - env_base].iter().sum::<f64>() / ((b - a + 1) as f64);
            counted_upto += 1;
            let _ = x;
            n += 1;
        }
        assert_eq!(counted_upto, env_end - 2, "consumption should advance to the start of the last window");
        assert_eq!(n, env_end - 2, "each frame is consumed exactly once");
        // Short-window semantics: i=0 uses a window of 3 (0..=2), i=1 uses 4, and i >= 2 uses 5
        assert_eq!(
            smooth_window_bounds(0, 0).1 - smooth_window_bounds(0, 0).0 + 1,
            3
        );
        assert_eq!(
            smooth_window_bounds(1, 0).1 - smooth_window_bounds(1, 0).0 + 1,
            4
        );
    }

    /// count_nuclei: a shallow saddle (0.2 dB valley) merges two peaks into 1 nucleus under a 2 dB valley rule; a deep valley keeps them apart
    #[test]
    fn count_nuclei_shallow_saddle_merges() {
        // Hand-built envelope: peak 0 dB@40, peak −0.3 dB@50, saddle −0.5 dB@45 (valley depth 0.2 dB)
        let mut env = vec![-40.0f64; 100];
        env[39] = -2.0;
        env[40] = 0.0;
        env[41] = -2.0;
        for (i, e) in env.iter_mut().enumerate().skip(42).take(8) {
            *e = -0.5 + (i as f64 - 45.0).abs() * 0.06; // 42..50 linear saddle (45 is lowest at −0.5)
        }
        env[45] = -0.5;
        env[50] = -0.3;
        env[51] = -0.4;
        let (n, peaks) = count_nuclei(&env, 0, env.len(), 99.0, 25.0, 2.0, 1, 1);
        assert_eq!(n, 1, "a shallow-saddle double peak should merge into 1 nucleus (keep the higher one @40), peaks={peaks:?}");
        // Deep-valley contrast: saddle −10 dB (depth 9.7 dB >= 6) -> separate
        let mut env2 = env.clone();
        env2[45] = -10.0;
        let (n2, _) = count_nuclei(&env2, 0, env2.len(), 99.0, 25.0, 6.0, 1, 1);
        assert_eq!(n2, 2, "a deep-valley double peak must stay split");
    }

    /// count_nuclei: the relative threshold (Q99−25 dB) keeps a −20 dB weak nucleus (an absolute threshold would miss it) and rejects −30 dB
    #[test]
    fn count_nuclei_relative_threshold_keeps_weak() {
        let env = synthetic_env(
            &[(0.20, 0.0, 0.05), (0.60, -20.0, 0.05), (1.00, -30.0, 0.05)],
            1.3,
        );
        let (n, peaks) = count_nuclei(&env, 0, env.len(), 99.0, 25.0, 2.0, 12, 5);
        assert_eq!(
            n, 2,
            "a strong nucleus plus a -20 dB weak nucleus should both count (-30 dB is below the threshold), peaks={peaks:?}"
        );
    }

    /// count_nuclei: peak distance 0.12 s (12 frames) -- two spikes 0.06 s apart merge, keeping the higher one
    #[test]
    fn count_nuclei_min_distance_merges_close_peaks() {
        let env = synthetic_env(
            &[(0.30, 0.0, 0.02), (0.36, -1.0, 0.02), (0.80, -2.0, 0.05)],
            1.2,
        );
        let (n, peaks) = count_nuclei(&env, 0, env.len(), 99.0, 25.0, 2.0, 12, 1);
        assert_eq!(n, 2, "closely spaced peaks should merge (keep the higher) plus the far peak, peaks={peaks:?}");
        assert!(
            (peaks[0] as f64 * 0.01 - 0.30).abs() < 0.011,
            "the retained one should be the higher peak @0.30 s"
        );
    }

    /// regions_from_mask: gap splitting and merging plus minimum-duration rejection
    #[test]
    fn regions_merge_and_min_dur() {
        // 10 rows at 0.08 s each: segment 1 = [0,3) (3 rows = 0.24 s); the 2-row hole (0.16 s) < 0.30 s -> merged with segment 2;
        // segment 2 = [5,8); then a large hole -> segment 3 = [20,21) (1 row = 0.08 s < 0.12 s) is dropped
        let mut mask = vec![false; 30];
        for i in 0..3 {
            mask[i] = true;
        }
        for i in 5..8 {
            mask[i] = true;
        }
        mask[20] = true;
        let regs = regions_from_mask(&mask, 0.08, 0.30, 0.12);
        assert_eq!(regs, vec![(0, 8)], "a 0.16 s hole (< 0.30 s) should merge, short segment dropped");
    }

    /// burst_bounds: a 0.4 s activity hole splits bursts; a hole midpoint snaps to the nearest active frame
    #[test]
    fn burst_bounds_split_at_hole() {
        let mut act = vec![false; 200];
        for i in 10..30 {
            act[i] = true;
        } // 0.10~0.30s
        for i in 70..90 {
            act[i] = true;
        } // 0.70-0.90 s (hole 0.40 s > 0.30 s)
        let (a, b) = burst_bounds(&act, 20, 30);
        assert_eq!((a, b), (10, 30), "a 0.4 s hole must split the burst");
        let (a2, b2) = burst_bounds(&act, 80, 30);
        assert_eq!((a2, b2), (70, 90));
        let (a3, b3) = burst_bounds(&act, 50, 30);
        assert_eq!((a3, b3), (70, 90), "the midpoint of a hole should snap to the burst containing the nearest active frame");
    }

    /// floor_track_nuclei: the first valid value starts at −Δ; during silence it decays towards q_last − sil_drop
    #[test]
    fn floor_nuclei_first_segment_and_silence() {
        let mut curve = vec![INVALID; 10];
        curve.extend(vec![0.5; 40]);
        curve.extend(vec![INVALID; 100]);
        let mut valid = vec![false; 10];
        valid.extend(vec![true; 40]);
        valid.extend(vec![false; 100]);
        let q = floor_track_nuclei(
            &curve,
            &valid,
            0.20,
            37.5,
            0.09,
            0.15,
            1.0 - (-0.2f64).exp(),
        );
        assert!(
            (q[10] - (0.5 - 0.09)).abs() < 1e-12,
            "at the first valid frame the floor should be x - delta"
        );
        // On a constant valid segment the floor creeps towards x at rate a_up (asymmetric): 40 rows -> 0.41+0.09·0.645 ≈ 0.468
        assert!(
            (q[49] - 0.4682).abs() < 5e-3,
            "during a constant segment the floor climbs slowly, q[49]={}",
            q[49]
        );
        // During silence it decays towards q_last−0.15 = 0.468−0.15 ≈ 0.318 (essentially there after 100 rows)
        assert!(
            q[149] < q[49] - 0.14,
            "during silence the floor should decay toward q_last - 0.15, q[149]={}",
            q[149]
        );
        assert!(q[149] > q[49] - 0.16);
    }

    /// St8Accum: later blocks overwrite the same group (higher k wins), groups with 8(g+1) > end are dropped, and flushing goes in ascending g
    #[test]
    fn st8_accum_dedup_keeps_latest_chunk() {
        let mut st = St8Accum::new();
        // Block k=0 (start=0), end=29: g0,g1,g2 accepted (8*3=24 <= 29), g3 rejected (8*4=32 > 29)
        st.feed_chunk(
            0,
            vec![
                vec![0.0; 256],
                vec![1.0; 256],
                vec![2.0; 256],
                vec![9.0; 256],
            ],
            29,
        );
        assert_eq!(st.rows.len(), 0);
        // Block k=1 (start=8), end=37: g1,g2,g3 accepted (8*4=32 <= 37, g4 rejected) -> g1/g2 are overwritten;
        // after block k=1, every g < 1 is final -> g0 flushes
        st.feed_chunk(
            1,
            vec![vec![11.0; 256], vec![12.0; 256], vec![13.0; 256]],
            37,
        );
        assert_eq!(st.rows.len(), 1);
        assert_eq!(st.rows[0][0], 0.0, "g0 only has rows from block 0");
        // Block k=2: g2 is overwritten by the later block; g1 flushes
        st.feed_chunk(2, vec![vec![22.0; 256]], 45);
        assert_eq!(st.rows.len(), 2);
        assert_eq!(st.rows[1][0], 11.0, "g1 should be overwritten by the later block k=1");
        st.finish();
        assert_eq!(st.rows.len(), 4, "g0..g3 all produced");
        assert_eq!(st.rows[2][0], 22.0, "g2 should hold rows from the last covering block k=2");
        assert_eq!(st.rows[3][0], 13.0);
        // Prefix-sum consistency
        let sum: f64 = st.rows.iter().map(|r| r[0] as f64).sum();
        assert!((st.cs[4][0] - sum).abs() < 1e-9);
    }

    #[test]
    fn st8_trim_head_keeps_prefix_semantics() {
        let mut st = St8Accum::new();
        // Row g = [g, g+1, g+2, g+3] (the direction varies with g, since pooled_cos is sensitive to rebasing)
        let rows: Vec<Vec<f32>> = (0..10u32)
            .map(|i| (i..i + 4).map(|v| v as f32).collect())
            .collect();
        for (g, r) in rows.iter().enumerate() {
            st.pending.insert(g, r.clone());
        }
        st.finish();
        assert_eq!(st.abs_len(), 10);
        assert_eq!(st.head_base, 0);
        let cent = [1.0f32; 4];
        // Untrimmed: absolute window [5, 9)
        let c0 = pooled_cos(&st.cs, 5, 9, &cent);
        st.trim_head(4);
        assert_eq!(st.head_base, 4);
        assert_eq!(st.abs_len(), 10, "the absolute row count is unchanged");
        assert_eq!(st.rows.len(), 6);
        assert_eq!(st.cs.len(), 7, "cs = rows+1");
        // After trimming, the same absolute window (relative [1, 5)) matches bit for bit
        let c1 = pooled_cos(&st.cs, 1, 5, &cent);
        assert!(
            (c0 - c1).abs() < 1e-12,
            "the same window must give the same cosine after rebasing: {c0} vs {c1}"
        );
        // Trim down to 1 remaining row, and an over-trim is truncated by min
        st.trim_head(5);
        assert_eq!(st.abs_len(), 10);
        st.trim_head(3);
        assert_eq!(st.rows.len(), 0, "over-trim is clamped by min(rows.len())");
        assert_eq!(st.head_base, 10);
    }

    /// pooled_cos: a parallel centroid gives 1, an orthogonal one gives 0 (cs holds row prefix sums)
    #[test]
    fn pooled_cos_identity() {
        // rows: r0=[1,0], r1=[0,1], r2=[0,1], r3=[0,1] -> cs prefix sums
        let cs = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![1.0, 1.0],
            vec![1.0, 2.0],
            vec![1.0, 3.0],
        ];
        assert!(
            (pooled_cos(&cs, 0, 1, &[1.0, 0.0]) - 1.0).abs() < 1e-12,
            "window=r0, same direction -> 1"
        );
        assert!(
            pooled_cos(&cs, 0, 1, &[0.0, 1.0]).abs() < 1e-12,
            "window=r0, orthogonal -> 0"
        );
        assert!(
            (pooled_cos(&cs, 1, 3, &[0.0, 1.0]) - 1.0).abs() < 1e-12,
            "window=r1,r2, same direction -> 1"
        );
        assert!(pooled_cos(&cs, 1, 3, &[1.0, 0.0]).abs() < 1e-12);
        assert!(
            (pooled_cos(&cs, 99, 100, &[1.0, 0.0]) + 1.0).abs() < 1e-12,
            "empty window -> -1"
        );
    }

    /// Speaking-rate adaptive group merging: a same-command plateau gap below 1× the median character length merges, above it stays separate
    #[test]
    fn adaptive_same_cmd_merge_rule() {
        let mut mask = vec![false; 100];
        for i in 10..14 {
            mask[i] = true;
        }
        for i in 20..24 {
            mask[i] = true;
        } // gap of 6 rows = 0.48 s
        let regs = regions_from_mask(&mask, 0.08, 0.30, 0.12);
        assert_eq!(regs.len(), 2, "baseline 0.30 s: no merge");
        let mut merged: Vec<(usize, usize)> = Vec::new();
        let max_hole = (0.50f64 / 0.08).floor() as usize; // median char length 0.50 s -> 1x rule
        for (a, b) in regs {
            if let Some(last) = merged.last_mut() {
                if a - last.1 <= max_hole {
                    last.1 = b;
                    continue;
                }
            }
            merged.push((a, b));
        }
        assert_eq!(
            merged,
            vec![(10, 24)],
            "a 0.48 s gap < 1x char length 0.50 s -> merge as the same command continuing"
        );
    }

    /// sos_filt_filt: verified against fixed scipy sosfiltfilt values (same input signal, interior points and tail)
    #[test]
    fn sos_filt_filt_matches_scipy_reference() {
        let n = 1600usize;
        let mut x = vec![0.0f32; n];
        for (i, v) in x.iter_mut().enumerate() {
            let t = i as f64 / 16000.0;
            *v = (0.5 * (2.0 * std::f64::consts::PI * 600.0 * t).sin()
                + 0.3 * (2.0 * std::f64::consts::PI * 3000.0 * t).sin()
                + 0.4 * (2.0 * std::f64::consts::PI * 80.0 * t).sin()) as f32;
        }
        x[800] += 1.0;
        let y = sos_filt_filt(&x);
        let ref_inner = [
            0.35357478f64,
            0.42635375,
            0.47557500,
            0.49851867,
            0.49391699,
            0.46202486,
            0.40460560,
            0.32483291,
            0.22711513,
            0.11685200,
        ];
        for (k, &r) in ref_inner.iter().enumerate() {
            assert!(
                (y[590 + k] - r).abs() < 1e-5,
                "interior point y[{}]={} vs scipy {r}",
                590 + k,
                y[590 + k]
            );
        }
        // Known boundary: the tail transient differs from scipy by ~0.05 (scipy's sosfiltfilt initializes with the lfilter_zi steady state
        // while this starts from zero; see the sos_filt_filt comment). Counted bursts always sit inside the session, so only boundedness of the tail is asserted.
        for (k, &v) in y[1595..].iter().enumerate() {
            assert!(v.abs() < 1.0, "tail |y[{}]|={v} out of range", 1595 + k);
        }
    }

    /// F1 band-pass selectivity: the 600 Hz passband is ≈1, while 80 Hz and 3 kHz are suppressed by >= 30 dB
    #[test]
    fn f1_bandpass_selectivity() {
        let n = 8000usize;
        let gain_at = |f: f64| -> f64 {
            let mut x = vec![0.0f32; n];
            for (i, v) in x.iter_mut().enumerate() {
                let t = i as f64 / 16000.0;
                *v = (2.0 * std::f64::consts::PI * f * t).sin() as f32;
            }
            let y = sos_filt_filt(&x);
            let start = n / 4; // skip the edge transient
            let rms: f64 =
                y[start..].iter().map(|v| v * v).sum::<f64>().sqrt() / ((n - start) as f64).sqrt();
            rms / (std::f64::consts::FRAC_1_SQRT_2)
        };
        let g600 = gain_at(600.0);
        assert!((g600 - 1.0).abs() < 0.05, "600 Hz passband gain should be ~1; got {g600}");
        let g80 = gain_at(80.0);
        assert!(g80 < 0.03, "80 Hz should be attenuated by >= 30 dB; got {g80}");
        let g3000 = gain_at(3000.0);
        assert!(g3000 < 0.03, "3 kHz should be attenuated by >= 30 dB; got {g3000}");
    }

    // -- Live causal counting (LiveCounter fed frame by frame vs full count_nuclei) --

    #[test]
    fn live_counter_absorb_timeout_releases_cluster() {
        let piecewise = |pts: &[(f64, f64)]| -> Vec<f64> {
            let n = pts.last().map(|p| p.0).unwrap_or(0.0) as usize + 30;
            (0..n)
                .map(|i| {
                    let t = i as f64;
                    let seg = pts.windows(2).find(|w| t >= w[0].0 && t < w[1].0);
                    match seg {
                        Some(w) => w[0].1 + (w[1].1 - w[0].1) * (t - w[0].0) / (w[1].0 - w[0].0),
                        None => pts.last().map(|p| p.1).unwrap_or(-40.0),
                    }
                })
                .collect()
        };
        // Peak@40 (0 dB) -> shallow valley@65 (−4.7) confirms early and enters the absorb domain -> a long in-band plateau (−2 dB, the valley never closes)
        // -> strong peak@170 (+8 dB, 12.7 dB above the inter-cluster valley, >= 6, so it passes the tail-ripple check) (a second nucleus must be emitted after the timeout)
        let env = piecewise(&[
            (0.0, -40.0),
            (40.0, 0.0),
            (65.0, -4.7),
            (80.0, -2.0),
            (150.0, -3.0),
            (170.0, 8.0),
            (200.0, -40.0),
            (260.0, -40.0),
        ]);
        // Timeout 30 frames (confirmation lands around 62-68, so it is released before frame 100)
        let mut ctr = LiveCounter::new(6.0, 2.0, 12, 99.0, 25.0, false);
        ctr.absorb_max_frames = 30;
        feed_env_causal(&mut ctr, &env);
        assert_eq!(
            ctr.emitted.len(),
            2,
            "after cluster-close timeout a strong peak should emit a second nucleus, emitted={:?}",
            ctr.emitted
        );
        // Contrast: timeout off (0 = off) -> the old behaviour holds, the absorb domain swallows the strong peak and only 1 nucleus is produced
        let mut ctr0 = LiveCounter::new(6.0, 2.0, 12, 99.0, 25.0, false);
        ctr0.absorb_max_frames = 0;
        feed_env_causal(&mut ctr0, &env);
        assert_eq!(
            ctr0.emitted.len(),
            1,
            "with the timeout off the old absorb behavior should keep 1 nucleus, emitted={:?}",
            ctr0.emitted
        );
    }

    /// The same 5-frame centred smoothing as NucleiLive, delayed by 2 frames (causal): smoothed[i] = mean(raw[i-2..=i+2])
    fn feed_env_causal(ctr: &mut LiveCounter, env: &[f64]) {
        for i in 0..env.len() {
            if i + 2 > env.len() - 1 {
                break;
            }
            let s0 = i.saturating_sub(2);
            // 5-frame centred smoothing: 'same' semantics pad the edges with zeros and the denominator is always 5
            let x = env[s0..=i + 2].iter().sum::<f64>() / 5.0;
            ctr.feed(i, x);
        }
        ctr.close();
    }

    /// Synthetic envelope fed frame by frame: the nucleus count matches the full version and peak positions differ by <= 3 frames (the maximum phase shift of trailing vs centred smoothing)
    #[test]
    fn live_counter_matches_full_count_nuclei() {
        for valley in [2.0f64, 6.0] {
            // Four deep-valley bumps (valley depth >= 15 dB, spacing >= 0.30 s > the peak distance)
            let env = synthetic_env(
                &[
                    (0.20, 0.0, 0.05),
                    (0.50, -1.0, 0.05),
                    (0.80, -2.0, 0.05),
                    (1.10, -3.0, 0.05),
                ],
                1.5,
            );
            let (n_full, pk_full) = count_nuclei(&env, 0, env.len(), 99.0, 25.0, valley, 12, 5);
            let mut ctr = LiveCounter::new(valley, 2.0, 12, 99.0, 25.0, false);
            feed_env_causal(&mut ctr, &env);
            assert_eq!(ctr.emitted.len(), n_full, "deep valley 4 packets, valley={valley}");
            for (p, q) in ctr.emitted.iter().zip(pk_full.iter()) {
                assert!(
                    (p.0 as i64 - *q as i64).abs() <= 3,
                    "peak positions differ too much, valley={valley}: live={} full={q}",
                    p.0
                );
            }
            // The relative threshold keeps a −20 dB weak nucleus
            let env2 = synthetic_env(
                &[(0.20, 0.0, 0.05), (0.60, -20.0, 0.05), (1.00, -30.0, 0.05)],
                1.3,
            );
            let (n2, _) = count_nuclei(&env2, 0, env2.len(), 99.0, 25.0, valley, 12, 5);
            let mut ctr2 = LiveCounter::new(valley, 2.0, 12, 99.0, 25.0, false);
            feed_env_causal(&mut ctr2, &env2);
            assert_eq!(ctr2.emitted.len(), n2, "weak nucleus, valley={valley}");
            // Two peaks close together (0.06 s < the 0.12 s peak distance): the full version's distance suppression keeps the higher one -> the serial counter also gives 1 plus the distant peak
            let env3 = synthetic_env(
                &[(0.30, 0.0, 0.02), (0.36, -1.0, 0.02), (0.80, -2.0, 0.05)],
                1.2,
            );
            let (n3, pk3) = count_nuclei(&env3, 0, env3.len(), 99.0, 25.0, valley, 12, 1);
            let mut ctr3 = LiveCounter::new(valley, 2.0, 12, 99.0, 25.0, false);
            feed_env_causal(&mut ctr3, &env3);
            assert_eq!(ctr3.emitted.len(), n3, "closely spaced peaks, valley={valley}");
            if n3 == 2 {
                assert!(
                    (ctr3.emitted[0].0 as i64 - pk3[0] as i64).abs() <= 3,
                    "the retained one should be the higher peak @0.30 s"
                );
            }
        }
    }

    /// Two peaks with a shallow valley (2-6 dB) plus early emission (absorb domain): valley ~2.7 dB, emitting early once the peak distance is exceeded ->
    /// the following shallow-valley peak is absorbed and the final count matches the full version (valley 6 merges, keeping the higher) = 1 (the peak position
    /// is the earlier peak; differing from the full version's kept-higher position is the accepted cost of early emission); deep-valley contrast (valley 7 dB) = 2, matching
    #[test]
    fn live_counter_shallow_valley_early_emit_absorbs() {
        // Piecewise-linear envelope builder
        let piecewise = |pts: &[(f64, f64)]| -> Vec<f64> {
            let n = 220usize;
            (0..n)
                .map(|i| {
                    let t = i as f64;
                    for w in pts.windows(2) {
                        let ((ta, va), (tb, vb)) = ((w[0].0, w[0].1), (w[1].0, w[1].1));
                        if t >= ta && t <= tb {
                            return va + (vb - va) * (t - ta) / (tb - ta);
                        }
                    }
                    pts[pts.len() - 1].1
                })
                .collect()
        };
        // Shallow valley: [(0,−40),(40,0),(65,−4.7),(85,−2),(100,−40),(219,−40)]
        let env = piecewise(&[
            (0.0, -40.0),
            (40.0, 0.0),
            (65.0, -4.7),
            (85.0, -2.0),
            (100.0, -40.0),
            (219.0, -40.0),
        ]);
        let (n_full, _pk) = count_nuclei(&env, 0, env.len(), 99.0, 25.0, 6.0, 12, 5);
        assert_eq!(n_full, 1, "batch version: valley depth 2.7 < 6 should merge into 1 nucleus");
        let mut ctr = LiveCounter::new(6.0, 2.0, 12, 99.0, 25.0, false);
        feed_env_causal(&mut ctr, &env);
        assert_eq!(
            ctr.emitted.len(),
            n_full,
            "streaming machine: a shallow-valley double peak should be 1 nucleus (early emit + absorb), peaks={:?}",
            ctr.emitted
        );
        assert!(
            (ctr.emitted[0].0 as i64 - 40).abs() <= 4,
            "the early-emitted nucleus should take the first peak @~40 (valley confirmation timing); got {:?}",
            ctr.emitted[0]
        );
        // Deep-valley contrast: peak 2 at −12 dB, saddle −22 dB (smoothed depth ~8 dB >= 6) -> both the full version and the serial counter give 2
        let env2 = piecewise(&[
            (0.0, -40.0),
            (40.0, 0.0),
            (65.0, -22.0),
            (85.0, -12.0),
            (100.0, -40.0),
            (219.0, -40.0),
        ]);
        let (n2, pk2) = count_nuclei(&env2, 0, env2.len(), 99.0, 25.0, 6.0, 12, 5);
        let mut ctr2 = LiveCounter::new(6.0, 2.0, 12, 99.0, 25.0, false);
        feed_env_causal(&mut ctr2, &env2);
        assert_eq!(n2, 2, "batch version: a deep valley should give 2 nuclei");
        assert_eq!(ctr2.emitted.len(), 2, "streaming machine: deep-valley control should give 2 nuclei");
        for (p, q) in ctr2.emitted.iter().zip(pk2.iter()) {
            assert!(
                (p.0 as i64 - *q as i64).abs() <= 4,
                "peak positions differ too much: live={} full={q}",
                p.0
            );
        }
    }

    #[test]
    fn live_counter_q_double_pulse_counts_one() {
        let piecewise = |pts: &[(f64, f64)]| -> Vec<f64> {
            (0..140usize)
                .map(|i| {
                    let t = i as f64;
                    for w in pts.windows(2) {
                        let ((ta, va), (tb, vb)) = ((w[0].0, w[0].1), (w[1].0, w[1].1));
                        if t >= ta && t <= tb {
                            return va + (vb - va) * (t - ta) / (tb - ta);
                        }
                    }
                    pts[pts.len() - 1].1
                })
                .collect()
        };
        // Frames: 0..30 silence at −60; a k peak (smoothed −36) around 36; the k->u valley at −46 around 45 (smoothed depth 8.7 dB);
        // the vowel peak at −14.9 around 58 (smoothed −15.4 around 60); a deep drop to −55 around 82. Matches the trial Q: peak height difference ~20 dB, valleys 6.5-15 dB.
        let env = piecewise(&[
            (0.0, -60.0),
            (30.0, -60.0),
            (34.0, -33.7),
            (40.0, -40.5),
            (45.0, -46.0),
            (50.0, -40.5),
            (58.0, -14.9),
            (64.0, -16.5),
            (82.0, -55.0),
            (139.0, -55.0),
        ]);
        let mut base = LiveCounter::new(6.0, 2.0, 12, 99.0, 25.0, false);
        feed_env_causal(&mut base, &env);
        assert_eq!(
            base.emitted.len(),
            2,
            "hold off = pre-fix behavior: Q splits into 2 nuclei, {:?}",
            base.emitted
        );
        // After the fix (production config hold=25 frames, sup=8 dB, hold_max_depth=16 dB): k merges into the vowel, counting 1
        let mut ctr = LiveCounter::new(6.0, 2.0, 12, 99.0, 25.0, false);
        ctr.hold = Some((25, 8.0, 16.0));
        feed_env_causal(&mut ctr, &env);
        assert_eq!(
            ctr.emitted.len(),
            1,
            "Q double pulse should merge into 1 nucleus (vowel), {:?}",
            ctr.emitted
        );
        assert!(
            (ctr.emitted[0].0 as i64 - 60).abs() <= 4,
            "the retained nucleus should be the vowel peak @~60; got {:?}",
            ctr.emitted[0]
        );
    }

    #[test]
    fn live_counter_noise_burst_after_speech_zero_nuclei() {
        // Noise burst: three bumps on a −60 floor (−46/−48/−52, 0.3 s apart, valley depth >= 8 dB)
        let env = synthetic_env(
            &[
                (0.20, -46.0, 0.10),
                (0.50, -48.0, 0.10),
                (0.80, -52.0, 0.10),
            ],
            1.1,
        );
        let mut base = LiveCounter::new(6.0, 2.0, 12, 99.0, 25.0, false);
        feed_env_causal(&mut base, &env);
        assert!(
            base.emitted.len() >= 2,
            "no session anchor = pre-fix behavior: a noise burst yields a nucleus, {:?}",
            base.emitted
        );
        let mut ctr = LiveCounter::new(6.0, 2.0, 12, 99.0, 25.0, false);
        for _ in 0..350 {
            ctr.sess_q.push(-55.0);
        }
        for i in 0..50 {
            ctr.sess_q.push(if i % 2 == 0 { -17.0 } else { -40.0 });
        }
        feed_env_causal(&mut ctr, &env);
        assert_eq!(
            ctr.emitted.len(),
            0,
            "with a session anchor a noise burst should yield 0 nuclei, {:?}",
            ctr.emitted
        );
    }

    #[test]
    fn live_counter_shuo_shuo_shuo_counts_three() {
        let env = synthetic_env(
            &[(0.30, 0.0, 0.08), (0.70, -1.0, 0.08), (1.10, -0.5, 0.08)],
            1.4,
        );
        let mut ctr = LiveCounter::new(6.0, 2.0, 12, 99.0, 25.0, false);
        ctr.hold = Some((25, 8.0, 16.0));
        feed_env_causal(&mut ctr, &env);
        assert_eq!(ctr.emitted.len(), 3, "shuo-shuo-shuo should count 3 nuclei, {:?}", ctr.emitted);
        for (got, want) in ctr.emitted.iter().zip([30usize, 70, 110]) {
            assert!(
                (got.0 as i64 - want as i64).abs() <= 4,
                "peak positions {got:?} should be ~{want}"
            );
        }
    }

    #[test]
    fn live_counter_decay_tail_ripples_not_counted() {
        let piecewise = |pts: &[(f64, f64)]| -> Vec<f64> {
            (0..320usize)
                .map(|i| {
                    let t = i as f64;
                    for w in pts.windows(2) {
                        let ((ta, va), (tb, vb)) = ((w[0].0, w[0].1), (w[1].0, w[1].1));
                        if t >= ta && t <= tb {
                            return va + (vb - va) * (t - ta) / (tb - ta);
                        }
                    }
                    pts[pts.len() - 1].1
                })
                .collect()
        };
        // Peak 0 dB@40 decaying to −30@130 (with two 1-2 dB ripple plateaus at −26@80 and −28@105),
        // a deep drop to −55@145, and a real peak at −10 dB@160
        let mut env = piecewise(&[
            (0.0, -40.0),
            (40.0, 0.0),
            (80.0, -26.0),
            (84.0, -27.5),
            (105.0, -28.0),
            (109.0, -30.0),
            (130.0, -30.0),
            (145.0, -55.0),
            (160.0, -10.0),
            (180.0, -55.0),
            (319.0, -55.0),
        ]);
        // Adds a tiny ripple (±0.8 dB, peak distance < 12 frames so it cannot form its own peak but does trigger the shallow-valley path)
        for i in 45..130 {
            env[i] += if (i / 7) % 2 == 0 { 0.8 } else { -0.8 };
        }
        let (n_full, pk_full) = count_nuclei(&env, 0, env.len(), 99.0, 25.0, 6.0, 12, 5);
        let mut ctr = LiveCounter::new(6.0, 2.0, 12, 99.0, 25.0, false);
        feed_env_causal(&mut ctr, &env);
        assert_eq!(
            n_full, 2,
            "batch version should give 2 nuclei (main peak + the real peak after the deep valley), peaks={pk_full:?}"
        );
        assert_eq!(
            ctr.emitted.len(),
            2,
            "the streaming machine should also give 2 nuclei (trailing ripple not counted), peaks={:?}",
            ctr.emitted
        );
        for (p, q) in ctr.emitted.iter().zip(pk_full.iter()) {
            assert!(
                (p.0 as i64 - *q as i64).abs() <= 4,
                "peak positions differ too much: live={} full={q}",
                p.0
            );
        }
    }

    #[test]
    fn rolling_quantile_long_burst_within_half_db_of_full_sort() {
        // 200 s envelope: one 8 dB short peak every 0.5 s (syllable nuclei) + a −28 dB noise floor + ±1.5 dB slow drift
        let n = 20000usize;
        let env: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64;
                let syll = if i % 50 < 8 { 8.0 } else { 0.0 };
                -28.0 + syll + 1.5 * (t * 0.7 * std::f64::consts::PI / 180.0).sin()
            })
            .collect();
        let mut h = RollingQuantile::new(LIVE_Q_WINDOW_FRAMES);
        let mut all: Vec<f64> = Vec::new();
        let mut max_diff = 0.0f64;
        // The reference (full sort) is computed every 100 frames -- computing it per frame is exactly the O(n²logn) slow path that was fixed, and the test
        // would never finish; sampled comparison is enough to verify the ±0.5 dB bound
        for (k, &x) in env.iter().enumerate() {
            h.push(x);
            all.push(x);
            if k % 100 == 99 {
                let d = (h.percentile(99.0) - percentile_linear(&all, 99.0)).abs();
                max_diff = max_diff.max(d);
            }
        }
        assert!(max_diff <= 0.5, "the percentile difference over a long burst should be <= 0.5 dB; got {max_diff:.4}");
        // Eviction correctness: feed 4000 ascending frames only, and the window should hold just the most recent 3000 (min = 1001)
        let mut h2 = RollingQuantile::new(LIVE_Q_WINDOW_FRAMES);
        for i in 0..4000i64 {
            h2.push(i as f64);
        }
        assert_eq!(h2.total(), 4000);
        assert_eq!(h2.percentile(0.0), 1000.0, "the oldest 1000 frames should be evicted");
        assert_eq!(h2.percentile(100.0), 3999.0);
    }

    /// run_update: opening runs, closing holes, same-command merging and dropping short runs (the incremental regions_from_mask)
    #[test]
    fn run_update_open_close_merge() {
        let mut r = RunState::default();
        // Run [10,13): 4 valid rows
        for g in 10..14 {
            run_update(&mut r, g, true, 3, 2);
        }
        assert_eq!(r.open, Some((10, 13)));
        // A 3-row hole (<= max_hole): no close
        for g in 14..17 {
            run_update(&mut r, g, false, 3, 2);
        }
        assert_eq!(r.open, Some((10, 13)));
        // The 4th hole row > max_hole -> close as prev
        run_update(&mut r, 17, false, 3, 2);
        assert_eq!(r.open, None);
        assert_eq!(r.prev, Some((10, 14)));
        // A gap <= max_hole opens again -> merge, keeping the original start; a larger gap -> a new start
        run_update(&mut r, 17, true, 3, 2);
        assert_eq!(r.open, Some((10, 17)));
        assert_eq!(r.prev, None);
        run_update(&mut r, 30, false, 3, 2);
        run_update(&mut r, 31, false, 3, 2);
        run_update(&mut r, 32, false, 3, 2);
        run_update(&mut r, 33, false, 3, 2);
        assert_eq!(r.prev, Some((10, 18)));
        run_update(&mut r, 40, true, 3, 2);
        assert_eq!(r.open, Some((40, 40)), "a gap > max_hole should open a new run");
        // A short run (1 row) closes -> dropped (prev keeps the old segment)
        run_update(&mut r, 41, false, 3, 2);
        run_update(&mut r, 42, false, 3, 2);
        run_update(&mut r, 43, false, 3, 2);
        run_update(&mut r, 44, false, 3, 2);
        assert_eq!(r.prev, Some((10, 18)), "a 1-row run should be dropped");
    }

}

#[cfg(test)]
mod encoder_resolve_tests {
    use super::*;

    fn temp_dir(tag: &str) -> std::path::PathBuf {
        static N: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let d = std::env::temp_dir().join(format!(
            "kws_v2_enc_{}_{tag}_{}",
            std::process::id(),
            N.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn touch(p: &std::path::Path) {
        std::fs::write(p, b"stub").unwrap();
    }

    #[test]
    fn model_dir_hit_prefers_int8() {
        let d = temp_dir("hit");
        touch(&d.join("encoder-v2-latest.onnx")); // fp32 (distractor)
        touch(&d.join("encoder-v2-latest.int8.onnx")); // int8 (expected)
        touch(&d.join("encoder-other-chunk-8-left-64.int8.onnx")); // non-v2 distractor
        let p = resolve_encoder_onnx(d.to_str().unwrap());
        assert_eq!(
            p.unwrap().file_name().unwrap().to_str().unwrap(),
            "encoder-v2-latest.int8.onnx"
        );
        std::fs::remove_dir_all(&d).unwrap();
    }

    #[test]
    fn noise_encoder_selected_by_env() {
        let d = temp_dir("noise");
        touch(&d.join("encoder-v2-chunk-8.int8.onnx"));
        touch(&d.join("encoder-v2-noise.int8.onnx"));
        unsafe { std::env::set_var("VIQ_ENCODER_NOISE", "1") };
        let p = resolve_encoder_onnx(d.to_str().unwrap()).unwrap();
        assert!(
            p.file_name().unwrap().to_string_lossy().contains("noise"),
            "the noise-robust model should be selected"
        );
        unsafe { std::env::remove_var("VIQ_ENCODER_NOISE") };
        let p = resolve_encoder_onnx(d.to_str().unwrap()).unwrap();
        assert!(
            !p.file_name().unwrap().to_string_lossy().contains("noise"),
            "it should fall back to the standard model"
        );
        std::fs::remove_dir_all(&d).unwrap();
    }

    /// Noise-robust model file self-check: it loads and the output dimension and block parameters are sane (catches a misplaced file).
    #[test]
    fn noise_encoder_loads() {
        let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("assets/model/encoder-v2-noise.int8.onnx");
        if !p.exists() {
            eprintln!("noise-robust model missing, skipping");
            return;
        }
        let enc = Encoder256::load(&p).unwrap();
        assert_eq!(enc.chunk_frames, 29);
        assert_eq!(enc.out_dim, 256);
    }

    #[test]
    fn load_candidate_cmd_ids_merges_both_sources() {
        let d = std::env::temp_dir().join(format!("kws_v2_cls_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        // Both missing -> empty
        assert!(load_candidate_cmd_ids(d.to_str().unwrap()).is_empty());
        // commands.json's top-level shape is mainly {"commands":{...}}
        std::fs::write(
            d.join("commands.json"),
            r#"{"commands": {"new_cmd_1": {}, "s": {}}, "decoder": {}}"#,
        )
        .unwrap();
        std::fs::write(
            d.join("embed_proto_v2.json"),
            r#"{"s": {"0": []}, "jump": {"0": []}}"#,
        )
        .unwrap();
        let got = load_candidate_cmd_ids(d.to_str().unwrap());
        assert_eq!(
            got,
            vec!["jump", "new_cmd_1", "s"],
            "union, deduped and sorted; non-commands keys not collected: {got:?}"
        );
        std::fs::remove_dir_all(d).ok();
    }
}
