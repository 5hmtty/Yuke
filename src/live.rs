//! Real-time microphone capture -> gated processor -> key output.

//! - cpal capture (explicit 50 ms blocks, device native sample rate / channels)
//! - Callback: lock-free copy; the processing thread mixes to mono, resamples to 16 kHz and calls LiveProcessor.add_audio
//! - Recordings are saved automatically to recordings/<timestamp>/audio.wav (for replay and diagnosis)
//! - Logs go to logs/live_<timestamp>.log (trigger events / audio stream errors / blocking warnings)

use crate::config::CommandConfig;
use crate::features::{TARGET_SAMPLE_RATE, resample_linear};
use crate::gate::LiveProcessor;
use chrono::Local;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use std::fs::File;
use std::io::Write as _;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

/// Capture and processing context
pub struct CaptureContext {
    pub proc: Arc<Mutex<LiveProcessor>>,
    pub stream: Option<cpal::Stream>,
    pub rec: Arc<Recorder>,
    pub(crate) aec_alive: Option<StreamAlive>,
    pub(crate) aec_fallback_rx: Option<std::sync::mpsc::Receiver<()>>,
}

/// Recording archive cap (16 kHz mono samples): 30 minutes by default = 28_800_000 samples ≈ 115 MB.
/// Without a cap the buffer grows linearly with session length (16 kHz mono f32 ≈ 230 MB/hour), so long sessions blow up memory.
/// Deliberately not a ring buffer: wrapping would destroy the "log timestamp ↔ audio offset" alignment
/// (frame events in logs/live_*.log are located by offset into audio.wav; overwriting breaks that workflow).
/// Recording stops at the cap: the stored prefix stays complete (alignment intact) and later audio only goes to the engine.
const RECORDER_MAX_SAMPLES: usize = 30 * 60 * TARGET_SAMPLE_RATE as usize;

/// Recording + logging
pub struct Recorder {
    log_file: Mutex<Option<File>>,
    audio: Mutex<Vec<f32>>, // 16 kHz mono accumulation (stops at RECORDER_MAX_SAMPLES)
    rec_dir: PathBuf,
    /// Recording-to-disk switch (controlled on the recognition page). false = do not accumulate audio and do not write audio.wav on stop (logs are still written).
    pub record_enabled: std::sync::atomic::AtomicBool,
    /// Set to true once RECORDER_MAX_SAMPLES is reached: recording stops for this session (only a new Recorder on capture restart resets it).
    /// Nothing accumulates after the stop -- resuming would punch a hole in the audio and break the same timestamp/offset alignment.
    rec_stopped: std::sync::atomic::AtomicBool,
}

impl Recorder {
    fn new() -> anyhow::Result<Arc<Self>> {
        let ts = Local::now().format("%Y%m%d_%H%M%S").to_string();
        let data_logs = crate::user_data_dir().join("logs");
        let rec_dir = crate::user_data_dir().join("recordings").join(&ts);
        std::fs::create_dir_all(&data_logs)?;
        let log_path = data_logs.join(format!("live_{ts}.log"));
        let log_file = File::create(&log_path)?;
        // Initial value read from settings.json (the control-page switch is persisted; defaults to off)
        let record_enabled = crate::config::persisted_record_enabled();
        if record_enabled {
            std::fs::create_dir_all(&rec_dir)?;
        }
        let r = Arc::new(Recorder {
            log_file: Mutex::new(Some(log_file)),
            audio: Mutex::new(Vec::new()),
            rec_dir,
            record_enabled: std::sync::atomic::AtomicBool::new(record_enabled),
            rec_stopped: std::sync::atomic::AtomicBool::new(false),
        });
        r.log(&format!(
            "=== Yuke session started {} ===",
            Local::now().format("%H:%M:%S")
        ));
        Ok(r)
    }

    /// Recognition page: whether to record to disk
    pub fn set_record_enabled(&self, on: bool) {
        self.record_enabled
            .store(on, std::sync::atomic::Ordering::Relaxed);
    }

    /// Write one log line (local time HH:MM:SS.mmm + message)
    pub fn log(&self, msg: &str) {
        let ts = Local::now().format("%H:%M:%S%.3f");
        let line = format!("{ts} {msg}\n");
        if let Ok(mut f) = self.log_file.lock() {
            if let Some(f) = f.as_mut() {
                let _ = f.write_all(line.as_bytes());
                let _ = f.flush();
            }
        }
    }

    /// Append 16 kHz mono audio in the callback (recording archive). Nothing accumulates while the switch is off.
    /// At RECORDER_MAX_SAMPLES recording stops and a WARN is logged once: the stored prefix stays complete and the
    /// "log timestamp ↔ audio offset" alignment holds; the excess is dropped (no ring buffer -- overwriting old audio would misalign it).
    fn record(&self, samples: &[f32]) {
        self.record_with_cap(samples, RECORDER_MAX_SAMPLES);
    }

    /// Configurable core of `record` (production uses RECORDER_MAX_SAMPLES; tests inject a small cap to exercise the stop logic).
    /// Called only from the audio processing thread (no concurrency); the stop flag guarantees the WARN is logged once.
    fn record_with_cap(&self, samples: &[f32], cap: usize) {
        if self.rec_stopped.load(std::sync::atomic::Ordering::Relaxed) {
            return;
        }
        if !self
            .record_enabled
            .load(std::sync::atomic::Ordering::Relaxed)
        {
            return;
        }
        let mut overflow = false;
        if let Ok(mut buf) = self.audio.lock() {
            let room = cap.saturating_sub(buf.len());
            if room >= samples.len() {
                buf.extend_from_slice(samples);
            } else {
                // Only the prefix that fits is kept: audio[i] ↔ the session's i/16k second stays aligned through the last sample
                buf.extend_from_slice(&samples[..room]);
                overflow = true;
            }
        }
        if overflow {
            self.rec_stopped
                .store(true, std::sync::atomic::Ordering::Relaxed);
            self.log(&format!(
                "warning: the recording archive reached its {:.0}-minute cap; recording stopped (later audio is not written to audio.wav; the stored prefix stays complete and log-timestamp/audio-offset alignment is unaffected)",
                cap as f32 / TARGET_SAMPLE_RATE as f32 / 60.0,
            ));
        }
    }

    /// Save the recording wav (16 kHz mono 16-bit PCM) and write the session summary.
    /// When the recording switch is off no audio.wav is written (avoiding frequent disk writes) and false is returned;
    /// true means it was written (the UI picks its message accordingly instead of always claiming "recording saved").
    pub fn save(&self, keypress_count: usize, last_command: &str) -> bool {
        let audio = self.audio.lock().map(|b| b.clone()).unwrap_or_default();
        if self
            .record_enabled
            .load(std::sync::atomic::Ordering::Relaxed)
        {
            // The directory does not exist yet if recording was switched on mid-session, so create it on demand
            if let Err(e) = std::fs::create_dir_all(&self.rec_dir) {
                self.log(&format!("failed to save the recording: {e}"));
                return false;
            }
            let wav_path = self.rec_dir.join("audio.wav");
            let spec = hound::WavSpec {
                channels: 1,
                sample_rate: TARGET_SAMPLE_RATE,
                bits_per_sample: 16,
                sample_format: hound::SampleFormat::Int,
            };
            match hound::WavWriter::create(&wav_path, spec) {
                Ok(mut w) => {
                    for &s in &audio {
                        let _ = w.write_sample((s.clamp(-1.0, 1.0) * 32767.0) as i16);
                    }
                    let _ = w.finalize();
                    let dur = audio.len() as f32 / TARGET_SAMPLE_RATE as f32;
                    let capped = if self.rec_stopped.load(std::sync::atomic::Ordering::Relaxed) {
                        " (recording stopped at the cap; the excess was not stored)"
                    } else {
                        ""
                    };
                    // Relative directory names only: absolute paths contain the local user name and end up in public issues
                    let sess = self
                        .rec_dir
                        .file_name()
                        .and_then(|s| s.to_str())
                        .unwrap_or("?");
                    self.log(&format!(
                        "session ended: {keypress_count} triggers, last command = {last_command}, recording recordings/{sess}/audio.wav ({dur:.1}s){capped}"
                    ));
                    println!("\nrecording saved: {} ({dur:.1}s){capped}", wav_path.display());
                    return true;
                }
                Err(e) => {
                    self.log(&format!("failed to save the recording: {e}"));
                    eprintln!("failed to save the recording: {e}");
                    return false;
                }
            }
        }
        self.log("recording to disk is disabled; logging only");
        false
    }
}

/// List available input device names (for the UI selector)
pub fn list_input_devices() -> Vec<String> {
    let host = cpal::default_host();
    host.input_devices()
        .ok()
        .into_iter()
        .flatten()
        .filter_map(|d| d.name().ok())
        .collect()
}

/// UI path: pass the settings-page parameters explicitly into engine construction (None fields fall back to env or built-in defaults).
pub fn prepare_processor_with_overrides(
    model_dir: &str,
    cfg: CommandConfig,
    enabled_flag: Arc<AtomicBool>,
    ov: crate::udkws_v2::EngineOverrides,
) -> anyhow::Result<Arc<Mutex<LiveProcessor>>> {
    let proc = std::sync::Arc::new(std::sync::Mutex::new(LiveProcessor::new_with_overrides(
        cfg,
        enabled_flag,
        Some(model_dir),
        ov,
    )));
    {
        let p = proc.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(err) = &p.v2_error {
            return Err(anyhow::anyhow!(
                "{err} (fix: regenerate that command's template on the Commands tab,\
                or make sure <model dir>/embed_proto_v2.json covers every enabled command)"
            ));
        }
    }
    {
        let mut p = proc.lock().unwrap_or_else(|e| e.into_inner());
        p.init_veto(model_dir);
    }
    Ok(proc)
}

impl CaptureContext {
    pub fn start_with_proc(
        _model_dir: &str,
        device_name: Option<&str>,
        proc: Arc<Mutex<LiveProcessor>>,
        rec: Option<Arc<Recorder>>,
    ) -> anyhow::Result<CaptureContext> {
        let rec = match rec {
            Some(r) => r,
            None => Recorder::new()?,
        };
        {
            let p = proc.lock().unwrap_or_else(|e| e.into_inner());
            rec.log(&format!("model loaded OK | commands: {:?}", p.commands()));
            for line in p.judge_report() {
                rec.log(&line);
                eprintln!("[decision] {line}");
            }
        }

        let aec_mode = crate::config::aec_mode();
        if aec_mode > 0 {
            let (tx_fb, rx_fb) = std::sync::mpsc::channel::<()>();
            let started = if aec_mode == 1 {
                start_comm_capture(&proc, &rec, device_name.map(|s| s.to_string()), tx_fb)
            } else {
                start_sonora_capture(&proc, &rec, device_name.map(|s| s.to_string()), tx_fb)
                    .map(|a| (a, "sonora-aec3"))
            };
            match started {
                Ok((alive, via)) => {
                    rec.log(&format!(
                        "echo-cancellation capture thread started (mode {aec_mode}, via {via})"
                    ));
                    return Ok(CaptureContext {
                        proc,
                        stream: None,
                        rec,
                        aec_alive: Some(alive),
                        aec_fallback_rx: Some(rx_fb),
                    });
                }
                Err(e) => {
                    let msg = format!(
                        "⚠ echo-cancellation initialisation failed (mode {aec_mode}); fell back to plain microphone capture (no echo cancellation this session): {e:#}"
                    );
                    rec.log(&msg);
                    eprintln!("[aec] {msg}");
                }
            }
        }

        // -- cpal capture --
        let stream = open_cpal_stream(device_name, proc.clone(), rec.clone())?;
        Ok(CaptureContext {
            proc,
            stream: Some(stream),
            rec,
            aec_alive: None,
            aec_fallback_rx: None,
        })
    }
}

pub(crate) fn open_cpal_stream(
    device_name: Option<&str>,
    proc: Arc<Mutex<LiveProcessor>>,
    rec: Arc<Recorder>,
) -> anyhow::Result<cpal::Stream> {
    let host = cpal::default_host();
    let device = match device_name {
        Some(name) => host
            .input_devices()?
            .find(|d| d.name().map(|n| n == name).unwrap_or(false))
            .ok_or_else(|| anyhow::anyhow!("input device not found: {name}"))?,
        None => host
            .default_input_device()
            .ok_or_else(|| anyhow::anyhow!("no input device (microphone) found"))?,
    };
    let device_name = device.name()?;
    let config = device.default_input_config()?;
    println!(
        "device in use: {}  ({} Hz, {} ch)",
        device_name,
        config.sample_rate().0,
        config.channels()
    );
    rec.log(&format!(
        "device: {device_name} ({} Hz, {} ch)",
        config.sample_rate().0,
        config.channels()
    ));

    let sample_format = config.sample_format();
    let channels = config.channels() as usize;
    let native_sr = config.sample_rate().0;

    // Windows WASAPI defaults to f32; other formats are not supported yet (a FromSample conversion could be added)
    if sample_format != cpal::SampleFormat::F32 {
        anyhow::bail!("only the f32 sample format is supported (currently {sample_format:?}); check the microphone driver");
    }

    let block_len = (0.05f32 * native_sr as f32) as u32;
    let stream_config = cpal::StreamConfig {
        channels: config.channels(),
        sample_rate: config.sample_rate(),
        buffer_size: cpal::BufferSize::Fixed(block_len),
    };
    rec.log("audio stream started (50 ms blocks)");
    let stream = build_stream(&device, &stream_config, channels, native_sr, proc, rec)?;
    stream.play()?;
    Ok(stream)
}

/// Capture-thread liveness guard: clearing the flag on drop makes the thread exit on its own.
pub(crate) struct StreamAlive(Arc<AtomicBool>);
impl Drop for StreamAlive {
    fn drop(&mut self) {
        self.0.store(false, Ordering::SeqCst);
    }
}

/// Build the input stream plus the processing thread.
fn start_comm_capture(
    proc: &Arc<Mutex<LiveProcessor>>,
    rec: &Arc<Recorder>,
    device_name: Option<String>,
    tx_fallback: std::sync::mpsc::Sender<()>,
) -> anyhow::Result<(StreamAlive, &'static str)> {
    rec.log(
        "echo cancellation enabled: Windows Communications mode (system-level AEC APO, the same one Teams uses; the configured microphone)",
    );
    let p2 = proc.clone();
    let r2 = rec.clone();
    let alive_flag = Arc::new(AtomicBool::new(true));
    let alive = StreamAlive(alive_flag.clone());
    const BLOCK: usize = 800; // 50ms @16k
    let (tx, rx) = std::sync::mpsc::channel::<anyhow::Result<()>>();
    std::thread::Builder::new()
        .name("comm-capture".into())
        .spawn(move || {
            // Re-probe at runtime (the UI probe and the boot state may differ): probe the **same** microphone
            let probe = crate::comm::probe(device_name.as_deref());
            if !probe.aec_selectable() {
                let msg = format!(
                    "this device does not support system echo cancellation (no AEC APO): the stream reports no effects and the reference endpoint cannot be controlled. Device \"{}\"",
                    if probe.device_label.is_empty() { probe.device_name.clone() } else { probe.device_label.clone() }
                );
                let _ = tx.send(Err(anyhow::anyhow!("{msg}")));
                r2.log(&format!("⚠ {msg}; falling back to plain microphone capture"));
                let _ = tx_fallback.send(());
                return;
            }
            if probe.aec_verdict() == crate::comm::AecVerdict::Unknown {
                r2.log(&format!(
                    "note: system AEC availability cannot be determined here ({}); trying the selected mode anyway -- switch to sonora if it does not work",
                    probe.effects_note.as_deref().unwrap_or("effect enumeration unavailable")
                ));
            }
            let mut cap = match crate::comm::CommCapture::open(device_name.as_deref()) {
                Ok(c) => {
                    let _ = tx.send(Ok(()));
                    c
                }
                Err(e) => {
                    let _ = tx.send(Err(e));
                    return;
                }
            };
            let mix_sr = cap.sample_rate();
            r2.log(&format!(
                "system AEC active (mic {} @{} Hz -> 16 kHz resampling; AEC reference stream {})",
                cap.device_label(),
                mix_sr,
                if cap.ref_endpoint_set() {
                    "explicitly pointed at the system default output device"
                } else {
                    "could not be set (the endpoint cannot be controlled; system defaults apply)"
                }
            ));
            let mut pending: Vec<f32> = Vec::with_capacity(BLOCK * 2);
            while alive_flag.load(Ordering::SeqCst) {
                std::thread::sleep(std::time::Duration::from_millis(5));
                let samples = match cap.poll() {
                    Ok(s) => s,
                    Err(e) => {
                        let msg = format!("⚠ echo-cancellation capture failed ({e:#}); falling back to plain microphone capture");
                        r2.log(&msg);
                        eprintln!("[aec] {msg}");
                        let _ = tx_fallback.send(());
                        break;
                    }
                };
                if samples.is_empty() {
                    continue;
                }
                let s16 = if mix_sr != 16000 {
                    crate::features::resample_linear(&samples, mix_sr, 16000)
                } else {
                    samples
                };
                pending.extend_from_slice(&s16);
                while pending.len() >= BLOCK {
                    let block: Vec<f32> = pending.drain(..BLOCK).collect();
                    r2.record(&block);
                    let mut p = match p2.lock() {
                        Ok(p) => p,
                        Err(poisoned) => {
                            r2.log("warning: proc Mutex was poisoned; recovered");
                            poisoned.into_inner()
                        }
                    };
                    if let Err(e) = p.add_audio(&block) {
                        r2.log(&format!("add_audio error: {e}"));
                    }
                }
            }
            cap.stop();
            r2.log("echo-cancellation capture thread exited (capture stopped)");
        })?;
    match rx.recv_timeout(std::time::Duration::from_secs(5)) {
        Ok(Ok(())) => Ok((alive, "system-aec")),
        Ok(Err(e)) => Err(e),
        Err(_) => anyhow::bail!("echo cancellation initialisation timed out (no response in 5 s)"),
    }
}

fn start_sonora_capture(
    proc: &Arc<Mutex<LiveProcessor>>,
    rec: &Arc<Recorder>,
    device_name: Option<String>,
    tx_fallback: std::sync::mpsc::Sender<()>,
) -> anyhow::Result<StreamAlive> {
    rec.log("echo cancellation enabled: sonora-aec3 (pure-Rust WebRTC AEC3, full 16 kHz band; the configured microphone plus a loopback reference from the system default output)");
    rec.log("sonora known boundary: a completely silent render device produces no loopback packets (reference = silence, AEC passes through), which is expected");
    let p2 = proc.clone();
    let r2 = rec.clone();
    let alive_flag = Arc::new(AtomicBool::new(true));
    let alive = StreamAlive(alive_flag.clone());
    const BLOCK: usize = 800; // 50 ms at 16 kHz -- the same block granularity as the cpal and system-AEC paths
    let (tx, rx) = std::sync::mpsc::channel::<anyhow::Result<()>>();
    std::thread::Builder::new()
        .name("sonora-capture".into())
        .spawn(move || {
            let mut blocks: usize = 0; // Telemetry cadence (every 100 blocks = 5 s)
            // Dual-path open (COM and device opening happen on this thread -- the same thread-affinity semantics as the comm/aec threads)
            let mut mic = match crate::sonora_aec::MicCapture::open(device_name.as_deref()) {
                Ok(c) => c,
                Err(e) => {
                    let _ = tx.send(Err(e.context("failed to open the microphone")));
                    return;
                }
            };
            let mut lp = match crate::sonora_aec::LoopbackCapture::open() {
                Ok(c) => {
                    let _ = tx.send(Ok(()));
                    c
                }
                Err(e) => {
                    let _ = tx.send(Err(e.context("failed to open the render loopback")));
                    return;
                }
            };
            let mut aec =
                match crate::sonora_aec::SonoraAec::new(mic.sample_rate(), lp.sample_rate()) {
                    Ok(a) => a,
                    Err(e) => {
                        let _ = tx.send(Err(e));
                        return;
                    }
                };
            r2.log(&format!(
                "sonora AEC ready (mic {} @{} Hz + reference {} @{} Hz -> AEC3 full 16 kHz band -> 16 kHz resampled blocks)",
                mic.device_label(),
                mic.sample_rate(),
                lp.device_label(),
                lp.sample_rate()
            ));
            if mic.sample_rate() != lp.sample_rate() {
                r2.log(&format!(
                    "sonora paths run at different rates (mic {} Hz / reference {} Hz); each is aligned and resampled per its own 10 ms frame",
                    mic.sample_rate(),
                    lp.sample_rate()
                ));
            }
            let mut pending: Vec<f32> = Vec::with_capacity(BLOCK * 2);
            while alive_flag.load(Ordering::SeqCst) {
                std::thread::sleep(std::time::Duration::from_millis(5));
                // Dual-path polling: a failure on either path counts as a capture fault (fall back to cpal, leaving a log trail)
                let m = match mic.poll() {
                    Ok(s) => s,
                    Err(e) => {
                        let msg = format!("⚠ sonora microphone capture failed ({e:#}); falling back to plain microphone capture");
                        r2.log(&msg);
                        eprintln!("[sonora] {msg}");
                        let _ = tx_fallback.send(());
                        break;
                    }
                };
                let r = match lp.poll() {
                    Ok(s) => s,
                    Err(e) => {
                        let msg = format!("⚠ sonora loopback reference stream failed ({e:#}); falling back to plain microphone capture");
                        r2.log(&msg);
                        eprintln!("[sonora] {msg}");
                        let _ = tx_fallback.send(());
                        break;
                    }
                };
                // AEC processing (alignment, resampling and blocking live inside SonoraAec and are unit-testable)
                let cleaned = aec.feed(&m, &r);
                // Telemetry (every 5 s): ERLE < 10 dB ≈ not converged; delay should stay between tens of ms and 200 ms
                blocks += 1;
                if blocks % 100 == 0 {
                    if let (Some(erle), Some(erl), Some(delay)) = aec.stats_snapshot() {
                        r2.log(&format!("[sonora-stats] ERLE={:.1}dB ERL={:.1}dB delay={:?}ms",
                            erle, erl, delay));
                    }
                }
                pending.extend_from_slice(&cleaned);
                while pending.len() >= BLOCK {
                    let block: Vec<f32> = pending.drain(..BLOCK).collect();
                    r2.record(&block);
                    let mut p = match p2.lock() {
                        Ok(p) => p,
                        Err(poisoned) => {
                            r2.log("warning: proc Mutex was poisoned; recovered");
                            poisoned.into_inner()
                        }
                    };
                    if let Err(e) = p.add_audio(&block) {
                        r2.log(&format!("add_audio error: {e}"));
                    }
                }
            }
            mic.stop();
            lp.stop();
            r2.log("sonora AEC capture thread exited (capture stopped)");
        })?;
    match rx.recv_timeout(std::time::Duration::from_secs(5)) {
        Ok(Ok(())) => Ok(alive),
        Ok(Err(e)) => Err(e),
        Err(_) => anyhow::bail!("sonora AEC initialisation timed out (no response in 5 s)"),
    }
}

fn build_stream(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    channels: usize,
    native_sr: u32,
    proc: Arc<Mutex<LiveProcessor>>,
    rec: Arc<Recorder>,
) -> anyhow::Result<cpal::Stream> {
    // Buffer capacity: about 1 s of audio (2 ch × 48 k × 1 s = 96 k f32, with headroom for transient backlog)
    let (producer, consumer) = rtrb::RingBuffer::<f32>::new(131072);
    let mut producer = producer;
    let alive_flag = Arc::new(AtomicBool::new(true));
    let alive = StreamAlive(alive_flag.clone()); // Lives inside the callback closure; dropping the Stream sets it to false
    let r_err = rec.clone();
    let err_fn = move |e| {
        r_err.log(&format!("audio stream error: {e}"));
        eprintln!("audio stream error: {e}");
    };
    let stream = device.build_input_stream(
        config,
        move |data: &[f32], _| {
            let _alive: &StreamAlive = &alive; // The guard lives with the closure (see the StreamAlive comment)
            // Hard real-time callback: lock-free copy only, never block; drop when full (better to drop than to stall the audio engine)
            let _ = producer.push_entire_slice(data); // Drop when full (hard real time, never block)
        },
        err_fn,
        None,
    )?;

    // Processing thread: consumes continuously (no throttling, same as the replay path); it winds down after the Stream is dropped
    let mut consumer = consumer;
    let p2 = proc.clone();
    let r2 = rec.clone();
    std::thread::Builder::new()
        .name("audio-process".into())
        .spawn(move || {
            let mut pending: Vec<f32> = Vec::with_capacity(8192);
            let block_src = native_sr as usize / 20; // 50 ms (48k=2400, 44.1k=2205)
            let mut buf: Vec<f32> = Vec::with_capacity(block_src * 2);
            loop {
                // Drain the buffer in one go (batching reduces loop overhead)
                let mut n = 0;
                while n < 65536 {
                    match consumer.pop() {
                        Ok(s) => {
                            pending.push(s);
                            n += 1;
                        }
                        Err(_) => break, // Empty buffer
                    }
                }
                if pending.is_empty() {
                    // Stream dropped with no residual data -> the thread exits (freeing the proc/rec Arcs)
                    if !alive_flag.load(Ordering::SeqCst) {
                        r2.log("processing thread exited (capture stopped)");
                        break;
                    }
                    std::thread::sleep(std::time::Duration::from_millis(2));
                    continue;
                }
                for ch in pending.chunks(channels) {
                    if ch.len() == channels {
                        let mut sum = 0.0f32;
                        for s in ch {
                            sum += *s;
                        }
                        buf.push(sum / channels as f32);
                    }
                }
                pending.clear();
                // Whole-block consumption: resample only once 50 ms has accumulated (exactly 800 samples at 16 kHz)
                while buf.len() >= block_src {
                    let resampled =
                        resample_linear(&buf[..block_src], native_sr, TARGET_SAMPLE_RATE);
                    // Recording archive (16 kHz mono)
                    r2.record(&resampled);
                    // Poison recovery: a poisoned Mutex after a panic would silently drop all audio --
                    // recover with into_inner and log it (keep running rather than stall)
                    let mut p = match p2.lock() {
                        Ok(p) => p,
                        Err(poisoned) => {
                            r2.log("warning: proc Mutex was poisoned; recovered");
                            poisoned.into_inner()
                        }
                    };
                    if let Err(e) = p.add_audio(&resampled) {
                        r2.log(&format!("add_audio error: {e}"));
                    }
                    buf.drain(..block_src);
                }
                // Stream dropped -> discard the sub-50 ms tail and wind down
                if !alive_flag.load(Ordering::SeqCst) {
                    r2.log("processing thread exited (capture stopped)");
                    break;
                }
            }
        })?;
    Ok(stream)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test Recorder: creates no directories and writes no log file (log() is a no-op when log_file is None)
    fn test_recorder() -> Recorder {
        Recorder {
            log_file: Mutex::new(None),
            audio: Mutex::new(Vec::new()),
            rec_dir: PathBuf::from("target/test-recorder"),
            record_enabled: AtomicBool::new(true),
            rec_stopped: AtomicBool::new(false),
        }
    }

    fn audio(r: &Recorder) -> Vec<f32> {
        r.audio.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }

    /// At the cap recording stops: only the prefix within the limit is kept, the rest is dropped, and further pushes do not accumulate (bounded memory)
    #[test]
    fn record_stops_at_cap_and_ignores_later_audio() {
        let r = test_recorder();
        let cap = 10_000usize;
        // Push 20 blocks × 800 = 16000 samples in order (value = global sample index, for alignment assertions)
        let mut pushed = 0usize;
        for _ in 0..20 {
            let block: Vec<f32> = (pushed..pushed + 800).map(|x| x as f32).collect();
            r.record_with_cap(&block, cap);
            pushed += 800;
        }
        let buf = audio(&r);
        assert_eq!(buf.len(), cap, "the buffer must stop exactly at the cap (prefix fully preserved)");
        assert!(r.rec_stopped.load(Ordering::Relaxed), "reaching the cap must set the stop-recording flag");
        // Push after the stop (simulating a long session that keeps talking): nothing grows
        r.record_with_cap(&vec![9.9f32; 50_000], cap);
        assert_eq!(audio(&r).len(), cap, "nothing may accumulate after recording stops");
    }

    /// Timestamp alignment is preserved: audio[i] must equal the session's i-th 16 kHz sample
    /// (a ring buffer would overwrite old samples and shift the indices; this implementation keeps the prefix in order)
    #[test]
    fn record_keeps_timestamp_alignment() {
        let r = test_recorder();
        let cap = 16_000usize; // exactly 1 s at 16k
        for round in 0..3u32 {
            let block: Vec<f32> = (0..800).map(|i| (round * 800 + i) as f32).collect();
            r.record_with_cap(&block, cap);
        }
        let buf = audio(&r);
        for (idx, &s) in buf.iter().enumerate() {
            assert_eq!(
                s, idx as f32,
                "audio[{idx}] offset corrupted (timestamp<->audio alignment broken)"
            );
        }
    }

    /// One WARN line at the cap (session log); repeated pushes after the stop do not warn again
    #[test]
    fn record_warns_once_at_cap() {
        std::fs::create_dir_all("target/test-recorder").unwrap();
        let log_path = "target/test-recorder/warn.log";
        let f = File::create(log_path).unwrap();
        let mut r = test_recorder();
        r.log_file = Mutex::new(Some(f));
        let cap = 1_000usize;
        r.record_with_cap(&vec![1.0f32; 600], cap);
        r.record_with_cap(&vec![2.0f32; 600], cap); // 400 accepted + stop-recording WARN
        r.record_with_cap(&vec![3.0f32; 600], cap); // already stopped: no new WARN
        assert_eq!(audio(&r).len(), cap);
        let txt = std::fs::read_to_string(log_path).unwrap();
        assert_eq!(
            txt.matches("warning: the recording archive reached its").count(),
            1,
            "the WARN must appear exactly once"
        );
    }

    /// Nothing accumulates while the recording switch is off (the off→on path is unaffected by the stop logic)
    #[test]
    fn record_disabled_accumulates_nothing() {
        let r = test_recorder();
        r.set_record_enabled(false);
        r.record_with_cap(&vec![1.0f32; 500], RECORDER_MAX_SAMPLES);
        assert!(audio(&r).is_empty());
        assert!(!r.rec_stopped.load(Ordering::Relaxed));
    }
}
