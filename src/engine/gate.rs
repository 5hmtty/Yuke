use crate::config::CommandConfig;
use crate::features::TARGET_SAMPLE_RATE;
use crate::keys;
use std::collections::HashMap;

// encoder parameters (read from the model at runtime; owned here: chunk-16=45/16, chunk-8=29/8)
const FRAME_LEN: usize = 400; // Fbank frame length (25 ms = 400 samples at 16 kHz)
const HOP_LEN: usize = 160; // Fbank frame hop (10 ms)

const AUDIBLE_RMS_THRESHOLD: f32 = 0.001;

const VAD_MIN_SPEECH_FRAMES: usize = 8;
const VAD_MAX_GAP_FRAMES: usize = 3;

/// Per-frame speech probability -> **longest run of consecutive frames above threshold**, tolerating a short gap of up to `max_gap` frames.
fn longest_speech_run(probs: &[f32], thr: f32, max_gap: usize) -> usize {
    let (mut best, mut cur, mut gap) = (0usize, 0usize, 0usize);
    for &p in probs {
        if p >= thr {
            cur += gap + 1;
            gap = 0;
            best = best.max(cur);
        } else if cur > 0 {
            gap += 1;
            if gap > max_gap {
                cur = 0;
                gap = 0;
            }
        }
    }
    best
}

pub struct LiveProcessor {
    command_to_key: HashMap<String, String>,
    judge_report: Vec<String>,
    command_to_mode: HashMap<String, String>,
    command_to_mutex: HashMap<String, Vec<String>>,

    // encoder block parameters (decided by the model)
    chunk_frames: usize, // Input frames per block (chunk-16=45 / chunk-8=29)
    hop_frames: usize,   // Frame step between blocks (chunk-16=16 / chunk-8=8)

    // state
    pub global_frame: i64, // Global frame counter (used by decoder.step)
    audio_buffer: Vec<f32>,
    consumed_frames: usize, // Consumed Fbank frames

    // VAD (voice activity detection)
    speech_energy_threshold: f32, // RMS threshold; below this counts as silence
    is_speaking: bool,
    rms_ema: f32, // Exponential moving average of the level (smoothing)

    // key state
    pub emit_keys: bool, // false = replay mode (record only, no real key presses)
    pub enabled: std::sync::Arc<std::sync::atomic::AtomicBool>, // Global switch (shared by the settings page and the hotkey)
    pub key_events: Vec<(i64, &'static str, String, String, f64)>,
    vad_gate: bool,     // Silero non-speech veto (no speech in the trigger window -> veto)
    sv_gate: bool,      // Speaker-verification veto (voiceprint in the trigger window != enrolled speaker -> veto)
    sv_threshold: f32,  // Voiceprint veto threshold (default 0.35; env VIQ_SV_GATE_THR)
    vad_threshold: f32, // VAD per-frame speech probability threshold (default 0.5)
    vad: Option<crate::silero::SileroVad>,
    sv: Option<crate::sv::SvGate>,
    veto_ring: Vec<f32>,       // 16 kHz audio ring (<=2 s) read by the veto checks
    ring_base_sec: f64,        // Stream time in seconds of veto_ring[0]
    pub keypress_count: usize, // Number of key presses executed (tap = press)
    pub last_command: String,  // Most recently triggered command name
    pub last_key: String,      // Most recently triggered key (shown in the UI)
    pub last_volume: f32,      // RMS level of the latest audio block (shown in the UI)
    last_audible_frame: i64,

    pub swallowed_events: Vec<(i64, String)>, // Path completed but gating did not let it through (miss and debounce diagnostics); the cap truncates the head so it stays bounded (compact_event_logs)
    pub swallowed_count: u64,
    pub debug_events: Vec<String>, // Ring buffer of decision detail (per-frame gating decisions, at most 20000 entries)

    // Energy timeline: per-block RMS + VAD state + onset rise, used by gating and to align triggers with the energy curve
    block_counter: u64,            // Audio block counter (50 ms per block, used to derive the timeline)
    pub energy_trace: Vec<String>, // Ring buffer of energy log lines (capped at 40000)

    buf_base_frame: usize, // Absolute frame number at the start of audio_buffer (updated on trim)
    /// v2 decision engine (the only decision path). None when construction failed; see `v2_error`.
    /// 256-dim encoder + syllable-nuclei / activity gate + emission-level refractory period.
    pub v2: Option<crate::udkws_v2::Engine>,
    pub v2_error: Option<String>,
    passcode_cmd: String,

    toggle_held: HashMap<String, String>,

    command_to_macro: HashMap<String, Vec<crate::config::MacroAction>>,
    command_to_reentry: HashMap<String, String>,
    macro_tokens: HashMap<String, std::sync::Arc<std::sync::atomic::AtomicBool>>,
    macro_running: HashMap<String, std::sync::Arc<std::sync::atomic::AtomicBool>>,
    diag_heartbeat: std::time::Instant,
    mute_beep_last: HashMap<String, i64>,
    macro_instances: Vec<MacroInstance>,
    macro_active: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    macro_notices: Vec<(String, &'static str)>,
    live_streaming: bool,
}

/// Macro execution report (tests and logs): cancelled = whether the cancel token interrupted it;
/// released = keys that were down but not up, released in reverse order on cancel (empty on a normal run).
#[derive(Debug, PartialEq)]
pub struct MacroRunReport {
    pub cancelled: bool,
    pub released: Vec<String>,
}

struct MacroInstance {
    cmd: String,
    running: std::sync::Arc<std::sync::atomic::AtomicBool>,
    cancel: std::sync::Arc<std::sync::atomic::AtomicBool>,
}

pub(crate) const MACRO_MAX_CONCURRENT: usize = 4;

/// Executes macro actions in order. Semantics:
/// - keeps a stack of keys that are down but not up; on cancel it **releases them in reverse order** (safe for pause and exit);
/// - `emit` = false sends no real keys (replay/tests) while the stack is maintained as usual;
/// - `honor_waits` = false makes wait/press instantaneous (inline test path, deterministic);
/// - once `cancel` is set: waits sleep in 50 ms steps and every action checks first -- cancel stops and releases immediately.
pub fn run_macro(
    actions: &[crate::config::MacroAction],
    emit: bool,
    cancel: &std::sync::atomic::AtomicBool,
    honor_waits: bool,
) -> MacroRunReport {
    use crate::config::MacroAction;
    use std::sync::atomic::Ordering;
    let mut stack: Vec<String> = Vec::new();
    for a in actions {
        if cancel.load(Ordering::Relaxed) {
            break;
        }
        match a {
            MacroAction::Down(k) => {
                if emit {
                    keys::key_down(k);
                }
                stack.push(k.clone());
            }
            MacroAction::Up(k) => {
                if emit {
                    keys::key_up(k);
                }
                // Explicit up: removed from the down stack (if absent, only the release is sent; not treated as an error)
                if let Some(pos) = stack.iter().rposition(|x| x == k) {
                    stack.remove(pos);
                }
            }
            MacroAction::Press(k) => {
                if emit {
                    keys::key_down(k);
                }
                macro_wait(crate::config::MACRO_PRESS_MS, cancel, honor_waits);
                if emit {
                    keys::key_up(k);
                }
            }
            MacroAction::Wait(ms) => macro_wait(*ms, cancel, honor_waits),
        }
    }
    let cancelled = cancel.load(Ordering::Relaxed);
    let mut released = Vec::new();
    if cancelled {
        // Reverse release: whatever went down last comes up first (same semantics as the reverse release in keys::key_up)
        for k in stack.iter().rev() {
            if emit {
                keys::key_up(k);
            }
            released.push(k.clone());
        }
    }
    MacroRunReport {
        cancelled,
        released,
    }
}

/// Macro wait: sleep in 50 ms steps, checking the cancel token each step, so a long wait never pins the thread --
/// a cancel from stopping recognition, exiting, or deleting the command is observed within 50 ms
fn macro_wait(total_ms: u32, cancel: &std::sync::atomic::AtomicBool, honor: bool) {
    use std::sync::atomic::Ordering;
    if !honor {
        return;
    }
    let mut left = total_ms as u64;
    while left > 0 {
        if cancel.load(Ordering::Relaxed) {
            return;
        }
        let step = left.min(50);
        std::thread::sleep(std::time::Duration::from_millis(step));
        left -= step;
    }
}

impl LiveProcessor {
    /// `chunk_frames`: encoder block parameter
    pub fn new(
        cfg: CommandConfig,
        enabled_flag: std::sync::Arc<std::sync::atomic::AtomicBool>,
        model_dir: Option<&str>,
    ) -> Self {
        Self::new_with_overrides(
            cfg,
            enabled_flag,
            model_dir,
            crate::udkws_v2::EngineOverrides::default(),
        )
    }

    /// UI path: constructed with the settings-page parameters (rebuilding after a settings change takes effect without restarting the process).
    pub fn new_with_overrides(
        cfg: CommandConfig,
        enabled_flag: std::sync::Arc<std::sync::atomic::AtomicBool>,
        model_dir: Option<&str>,
        ov: crate::udkws_v2::EngineOverrides,
    ) -> Self {
        let gate = cfg.gate.clone();
        let command_to_key = cfg
            .commands
            .iter()
            .map(|(name, cmd)| (name.clone(), cmd.key.clone()))
            .collect();
        let command_to_mode: std::collections::HashMap<String, String> = cfg
            .commands
            .iter()
            .map(|(name, cmd)| (name.clone(), cmd.mode.clone()))
            .collect();
        let command_to_macro: HashMap<String, Vec<crate::config::MacroAction>> = cfg
            .macro_seqs
            .iter()
            .map(|(name, s)| (name.clone(), s.actions.clone()))
            .collect();
        let command_to_reentry: HashMap<String, String> = cfg
            .macro_seqs
            .iter()
            .map(|(name, s)| (name.clone(), s.reentry.clone()))
            .collect();
        let mut command_to_mutex: HashMap<String, Vec<String>> = HashMap::new();
        for (name, peers) in cfg.mutex_sets.iter() {
            for p in peers {
                if p == name {
                    continue;
                }
                command_to_mutex
                    .entry(name.clone())
                    .or_default()
                    .push(p.clone());
                command_to_mutex
                    .entry(p.clone())
                    .or_default()
                    .push(name.clone());
            }
        }
        for v in command_to_mutex.values_mut() {
            v.sort();
            v.dedup();
        }
        let mut v2_error: Option<String> = None;
        let v2 = match model_dir {
            Some(md) => {
                let names: Vec<String> = cfg.commands.keys().cloned().collect();
                match crate::udkws_v2::Engine::from_model_dir_with(md, &names, &ov) {
                    Ok(e) => Some(e),
                    Err(e) => {
                        eprintln!("[engine] engine construction failed: {e:#}");
                        v2_error = Some(format!("engine construction failed; recognition is unavailable. Cause: {e:#}"));
                        None
                    }
                }
            }
            None => {
                eprintln!("[engine] no model directory; cannot construct the engine");
                v2_error = Some("no model directory; cannot construct the engine".to_string());
                None
            }
        };
        // The consumption grid follows the v2 engine itself (no audio is consumed while v2 is disabled)
        let (chunk_frames, hop_frames) = match &v2 {
            Some(e) => e.grid(),
            None => (29, 8),
        };
        let judge_report = {
            let mut enabled: Vec<String> = cfg.commands.keys().cloned().collect();
            enabled.sort();
            let v2_cmds = model_dir
                .map(crate::config::v2_proto_commands)
                .unwrap_or_default();
            crate::config::judge_path_report(&enabled, &v2_cmds, v2.is_some())
        };
        LiveProcessor {
            command_to_key,
            command_to_mode,
            command_to_mutex,
            judge_report,
            chunk_frames,
            hop_frames,
            global_frame: 0,
            audio_buffer: Vec::new(),
            consumed_frames: 0,
            speech_energy_threshold: gate.speech_energy_threshold,
            is_speaking: false,
            rms_ema: 0.0,
            emit_keys: true,
            enabled: enabled_flag,
            key_events: Vec::new(),
            vad_gate: std::env::var("VIQ_VAD_GATE")
                .map(|v| v == "1")
                .unwrap_or(false),
            sv_gate: std::env::var("VIQ_SV_GATE")
                .map(|v| v == "1")
                .unwrap_or(false),
            sv_threshold: std::env::var("VIQ_SV_GATE_THR")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0.35),
            vad_threshold: 0.5,
            vad: None,
            sv: None,
            veto_ring: Vec::new(),
            ring_base_sec: 0.0,
            keypress_count: 0,
            last_command: String::new(),
            last_key: String::new(),
            last_volume: 0.0,
            last_audible_frame: -1,
            swallowed_events: Vec::new(),
            swallowed_count: 0,
            debug_events: Vec::new(),
            block_counter: 0,
            energy_trace: Vec::new(),
            buf_base_frame: 0,
            v2,
            v2_error,
            passcode_cmd: String::new(),
            toggle_held: HashMap::new(),
            diag_heartbeat: std::time::Instant::now(),
            command_to_macro,
            command_to_reentry,
            macro_tokens: HashMap::new(),
            macro_running: HashMap::new(),
            mute_beep_last: HashMap::new(),
            macro_instances: Vec::new(),
            macro_active: std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            macro_notices: Vec::new(),
            live_streaming: true,
        }
    }

    /// Whether the level EMA crossed the speech threshold (UI indication and the "last audible frame" diagnostic only).
    fn is_speech(&mut self, rms: f32) -> bool {
        let alpha = 0.3f32;
        self.rms_ema = alpha * rms + (1.0 - alpha) * self.rms_ema;
        self.is_speaking = self.rms_ema > self.speech_energy_threshold;
        self.is_speaking
    }

    /// Appends decision detail (diagnostic log ring buffer: the most recent 20000 entries, dropping the head beyond that;
    /// the consumer writes increments to the log file with a cursor, see live.rs). 20000 covers a full session,
    /// so detail from fast repeated bursts is not truncated)
    fn push_debug(&mut self, msg: String) {
        self.debug_events.push(msg);
        if self.debug_events.len() > 20000 {
            let excess = self.debug_events.len() - 20000;
            self.debug_events.drain(..excess);
        }
    }

    /// Appends to the energy timeline log ([E] per-block RMS / [onset] rise). Independent of debug_events,
    /// for aligning the energy curve with trigger events. The 40000 cap covers long sessions.
    fn push_energy(&mut self, msg: String) {
        self.energy_trace.push(msg);
        if self.energy_trace.len() > 40000 {
            let excess = self.energy_trace.len() - 40000;
            self.energy_trace.drain(..excess);
        }
    }

    fn compact_event_logs(&mut self) {
        const KEEP: usize = 10000; // Entries kept after truncation (caps memory at a few hundred KB)
        const HYSTERESIS: usize = 2000; // Hysteresis: truncate only above KEEP+2000, avoiding a memmove per entry
        if self.key_events.len() > KEEP + HYSTERESIS {
            let excess = self.key_events.len() - KEEP;
            self.key_events.drain(..excess);
        }
        if self.swallowed_events.len() > KEEP + HYSTERESIS {
            let excess = self.swallowed_events.len() - KEEP;
            self.swallowed_events.drain(..excess);
        }
    }

    /// Releases every key held by a toggle and clears the table. Called on: stop recognition / engine reload / Drop backstop.
    /// Reuses the reverse-release semantics of keys::key_up (records "up" in key_events).
    pub fn release_all_toggles(&mut self) {
        if self.toggle_held.is_empty() {
            return;
        }
        let held: Vec<(String, String)> = self.toggle_held.drain().collect();
        for (cmd, key) in held {
            self.key_events
                .push((self.global_frame, "up", key.clone(), cmd.to_string(), 0.0));
            if self.emit_keys {
                keys::key_up(&key);
            }
            self.push_debug(format!(
                "[toggle] frame={} cmd={cmd} up {key}(released in bulk on stop or reload)",
                self.global_frame
            ));
        }
    }

    /// Releases one command's toggle key (if held). Called when the UI deletes a command (boundary 2:
    /// release before deleting -- deletion touches only that command's keys, not the global state).
    pub fn release_toggle_cmd(&mut self, cmd: &str) {
        let Some(key) = self.toggle_held.remove(cmd) else {
            return;
        };
        self.key_events
            .push((self.global_frame, "up", key.clone(), cmd.to_string(), 0.0));
        if self.emit_keys {
            keys::key_up(&key);
        }
        self.push_debug(format!(
            "[toggle] frame={} cmd={cmd} up {key}(released when the command is deleted)",
            self.global_frame
        ));
    }

    fn release_mutex_peers(&mut self, cmd: &str, frame: i64) {
        let peers: Vec<String> = match self.command_to_mutex.get(cmd) {
            Some(v) if !v.is_empty() => v.clone(),
            _ => return,
        };
        for p in peers {
            if p == cmd {
                continue;
            }
            // 1) toggle is holding -> release (reverse-release semantics live in keys::key_up)
            if let Some(held) = self.toggle_held.remove(&p) {
                // upx: a release triggered by a mutex, rendered in the UI as a footnote rather than a separate trigger
                self.key_events
                    .push((frame, "upx", held.clone(), p.clone(), 0.0));
                if self.emit_keys {
                    keys::key_up(&held);
                }
                self.push_debug(format!(
                    "[mutex] frame={frame} cmd={cmd} -> releasing held key {held} of mutex command {p}"
                ));
            }
            // 2) a macro is running -> cancel (live instance: a hit in either the per-command table or the concurrent registry)
            let running =
                self.macro_running
                    .get(&p)
                    .map(|f| f.load(std::sync::atomic::Ordering::Relaxed))
                    .unwrap_or(false)
                    || self.macro_instances.iter().any(|i| {
                        i.cmd == p && i.running.load(std::sync::atomic::Ordering::Relaxed)
                    });
            if running {
                self.cancel_macro(&p);
                self.push_debug(format!(
                    "[mutex] frame={frame} cmd={cmd} -> cancelling the running macro of mutex command {p}"
                ));
            }
        }
    }

    /// Cancels every running macro (sets the cancel token and clears the tables; execution threads exit at their next
    /// <=50 ms wait step and release their own down stacks in reverse order). Called on: stop recognition / engine reload / Drop backstop.
    /// A reload is a stop-recognition-level reset: macros cancelled, all toggles released (design boundaries 1 and 3).
    /// Instances under the concurrent policy (not in the per-command table) are cancelled one by one and the registry is cleared;
    /// macro_active is not reset here -- each finishing thread does exactly one fetch_sub, so it returns to zero naturally.
    pub fn cancel_all_macros(&mut self) {
        for (_, tok) in self.macro_tokens.drain() {
            tok.store(true, std::sync::atomic::Ordering::Relaxed);
        }
        self.macro_running.clear();
        for inst in self.macro_instances.drain(..) {
            inst.cancel
                .store(true, std::sync::atomic::Ordering::Relaxed);
        }
    }

    /// Cancels the running macros of a single command (including all concurrent-policy instances of it).
    /// Called when the UI deletes a command (alongside boundary 2).
    pub fn cancel_macro(&mut self, cmd: &str) {
        if let Some(tok) = self.macro_tokens.remove(cmd) {
            tok.store(true, std::sync::atomic::Ordering::Relaxed);
        }
        self.macro_running.remove(cmd);
        for inst in self.macro_instances.iter() {
            if inst.cmd == cmd {
                inst.cancel
                    .store(true, std::sync::atomic::Ordering::Relaxed);
            }
        }
        self.macro_instances.retain(|inst| inst.cmd != cmd);
    }

    /// Drains pending macro notices ((command id, kind) pairs; the UI calls this every frame -> status bar
    /// and log. Kinds: "reentry" = re-entry ignored / "cap" = concurrent instance limit reached).
    pub fn drain_macro_notices(&mut self) -> Vec<(String, &'static str)> {
        std::mem::take(&mut self.macro_notices)
    }

    /// Whether a command's macro is running (tests and diagnostics). Concurrent-policy instances are not in the
    /// per-command table and are invisible here (tests rely on the macro_active count instead).
    #[cfg(test)]
    #[cfg(test)]
    pub fn macro_is_running(&self, cmd: &str) -> bool {
        self.macro_running
            .get(cmd)
            .map(|f| f.load(std::sync::atomic::Ordering::Relaxed))
            .unwrap_or(false)
    }

    /// Number of active macro instances globally (the concurrent cap criterion; tests and diagnostics)
    #[cfg(test)]
    #[cfg(test)]
    pub fn macro_active_count(&self) -> usize {
        self.macro_active.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Silence beyond the threshold -> clear gating state (the encoder cache keeps its continuity)
    /// Appends a batch of new audio samples and tries to process them
    pub fn add_audio(&mut self, samples_16khz: &[f32]) -> anyhow::Result<()> {
        // Compute the level
        if !samples_16khz.is_empty() {
            let mut sum_sq = 0.0f32;
            for &s in samples_16khz {
                sum_sq += s * s;
            }
            let rms = (sum_sq / samples_16khz.len() as f32).sqrt();
            self.last_volume = rms;
            if rms > AUDIBLE_RMS_THRESHOLD {
                self.last_audible_frame = self.global_frame;
            }
            self.is_speech(rms);
            // The frame counter advances on 10 ms mel frames (v2 keeps its own frame numbering; this one feeds diagnostics
            // such as frames_since_audible)
            self.global_frame += (samples_16khz.len() / 160) as i64;
            if crate::udkws_v2::debug_log_on() {
                let spk = if self.is_speaking { "SPK" } else { "---" };
                let t_ms = self.block_counter;
                self.push_energy(format!("[E] t={t_ms}ms rms={rms:.4} spk={spk}"));
            }
            self.block_counter += (samples_16khz.len() as u64) * 1000 / TARGET_SAMPLE_RATE as u64;
        }

        // v2: feed raw samples (10 ms frame activity flags, used by the activity gate).
        if let Some(v2) = self.v2.as_mut() {
            v2.push_samples(samples_16khz);
        }
        // Feed the veto layer's audio ring (rolling, <=2 s; the voiceprint and VAD checks read the trigger window from it)
        if self.vad_gate || self.sv_gate {
            self.veto_ring.extend_from_slice(samples_16khz);
            const RING_CAP: usize = 48000;
            if self.veto_ring.len() > RING_CAP {
                let over = self.veto_ring.len() - RING_CAP;
                self.veto_ring.drain(..over);
                self.ring_base_sec += over as f64 / 16000.0;
            }
        }

        self.audio_buffer.extend_from_slice(samples_16khz);
        self.process()
    }

    /// Advances v2 decision-making on the chunk grid (a no-op while v2 is disabled).
    fn process(&mut self) -> anyhow::Result<()> {
        if self.v2.is_none() {
            return Ok(());
        }
        loop {
            // Number of audio samples needed to consume the block from consumed_frames to consumed_frames + chunk_frames
            let last_frame = self.consumed_frames + self.chunk_frames - 1;
            let needed = last_frame * HOP_LEN + FRAME_LEN;
            if self.audio_buffer.len() < needed {
                break; // Not enough samples; wait for the next batch
            }
            self.consumed_frames += self.hop_frames;
            // Trim the buffer: keep samples starting at consumed_frames - hop_frames*3
            if self.consumed_frames >= self.hop_frames * 4 {
                let keep_frame = self.consumed_frames - self.hop_frames * 3;
                let keep_sample = keep_frame * HOP_LEN;
                self.audio_buffer
                    .drain(..keep_sample.min(self.audio_buffer.len()));
                self.consumed_frames -= keep_frame;
                self.buf_base_frame += keep_frame;
            }
            // Only the live (streaming) path polls streaming triggers; replay finalisation goes through the CLI's Engine::finish
            let trigs = self
                .v2
                .as_mut()
                .filter(|v| v.is_streaming())
                .map(|v| v.poll())
                .unwrap_or_default();
            self.emit_v2_triggers(trigs);
        }
        self.compact_event_logs();
        Ok(())
    }

    /// Count of nuclei rejected by the identity layer (shown in the home-page summary row).
    pub fn rejected_nuclei(&self) -> u64 {
        self.v2.as_ref().map(|e| e.rejected_nuclei()).unwrap_or(0)
    }

    /// Runtime hot update: parameters are written straight into the running V2 engine (no rebuild, no interrupted keypresses).
    pub fn apply_live_tuning(&mut self, ov: &crate::udkws_v2::EngineOverrides) {
        if let Some(v2) = self.v2.as_mut() {
            v2.apply_live_tuning(ov);
        }
    }

    /// Replay mode: no real keys; triggers are recorded into key_events
    /// Global switch (settings page or hotkey: pause/resume key output)
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled
            .store(enabled, std::sync::atomic::Ordering::Relaxed);
        if !enabled {
            self.release_all_toggles();
        }
    }

    #[cfg(test)]
    pub fn set_replay_mode(&mut self) {
        self.emit_keys = false;
        self.vad_gate = false;
        self.sv_gate = false;
        // Macros: replay runs inline and instantaneously (no spawned thread, no real keys, no sleeping -- deterministic replay)
        self.live_streaming = false;
        if let Some(v2) = self.v2.as_mut() {
            v2.set_streaming(false);
        }
    }

    pub fn init_veto(&mut self, model_dir: &str) {
        let assets = crate::exe_dir()
            .map(|d| d.join("assets"))
            .unwrap_or_else(|| std::path::PathBuf::from("assets"));
        if self.vad_gate {
            match crate::silero::SileroVad::load(&assets.join("silero_vad.onnx"), 16000) {
                Ok(v) => {
                    self.vad = Some(v);
                    self.push_debug("[veto] Silero VAD loaded (non-speech triggers will be blocked)".into());
                }
                Err(e) => {
                    self.vad_gate = false;
                    eprintln!("[veto] failed to load Silero VAD; disabled: {e:#}");
                    self.push_debug(format!("[veto] failed to load Silero VAD; disabled: {e:#}"));
                }
            }
        }
        if self.sv_gate {
            match crate::sv::build_centroid(model_dir) {
                Ok((cen, n)) => {
                    let json = serde_json::to_string(&cen).unwrap_or_else(|_| "[]".into());
                    match crate::sv::SvGate::load(std::path::Path::new(model_dir), &json) {
                        Ok(g) => {
                            self.sv = Some(g);
                            self.push_debug(format!(
                                "[veto] voiceprint gate ready ({n} actively recorded centroids, threshold {:.2})",
                                self.sv_threshold
                            ));
                            eprintln!("[veto] voiceprint gate ready ({n} actively recorded takes)");
                        }
                        Err(e) => {
                            self.sv_gate = false;
                            eprintln!("[veto] failed to load the voiceprint gate; disabled: {e:#}");
                            self.push_debug(format!("[veto] failed to load the voiceprint gate; disabled: {e:#}"));
                        }
                    }
                }
                Err(e) => {
                    self.sv_gate = false;
                    eprintln!("[veto] failed to build the voiceprint centroid; disabled: {e:#}");
                    self.push_debug(format!(
                        "[veto] failed to build the voiceprint centroid (model_dir={model_dir}); disabled: {e:#}"
                    ));
                }
            }
        }
    }

    /// Trigger-window veto: VAD (not speech) and voiceprint (not the enrolled speaker) chained. Returns true = allow.
    /// The audio window is trigger start -0.9 s to trigger end +0.1 s, taken from veto_ring.
    fn passed_veto(&mut self, tg: &crate::udkws_v2::Trigger) -> bool {
        const SR: f64 = 16000.0;
        let i0 = (((tg.start_sec as f64 - 0.9) - self.ring_base_sec) * SR).max(0.0) as usize;
        let i1 = (((tg.end_sec as f64 + 0.1) - self.ring_base_sec) * SR)
            .min(self.veto_ring.len() as f64) as usize;
        if i1 <= i0 {
            return true; // No matching audio in the ring (should not happen) -> allow, never a false veto
        }
        let win = self.veto_ring[i0..i1].to_vec();
        if self.vad_gate {
            if let Some(v) = &mut self.vad {
                let run = v
                    .prob_sequence(&win)
                    .map(|p| longest_speech_run(&p, self.vad_threshold, VAD_MAX_GAP_FRAMES))
                    .unwrap_or(usize::MAX); // A VAD inference failure counts as speech, so nothing is vetoed by mistake
                if run < VAD_MIN_SPEECH_FRAMES {
                    let msg = format!(
                        "[vad-veto] t={:.2}s cmd={} non-speech (longest speech run {run} frames < {VAD_MIN_SPEECH_FRAMES}, threshold {:.2}); trigger vetoed",
                        tg.start_sec, tg.cmd, self.vad_threshold
                    );
                    self.push_debug(msg.clone());
                    eprintln!("{msg}");
                    return false;
                }
            }
        }
        if self.sv_gate {
            if let Some(g) = &mut self.sv {
                match g.similarity(&win) {
                    Ok(c) if c < self.sv_threshold => {
                        let msg = format!(
                            "[sv-veto] t={:.2}s cmd={} voiceprint={c:.3} (not the enrolled speaker); trigger vetoed",
                            tg.start_sec, tg.cmd
                        );
                        self.push_debug(msg.clone());
                        eprintln!("{msg}");
                        return false;
                    }
                    Ok(_) => {}
                    Err(e) => eprintln!("[sv-veto] voiceprint inference failed (allowing): {e:#}"),
                }
            }
        }
        true
    }

    pub(crate) fn emit_v2_triggers(&mut self, trigs: Vec<crate::udkws_v2::Trigger>) {
        if self.diag_heartbeat.elapsed() >= std::time::Duration::from_secs(5) {
            self.diag_heartbeat = std::time::Instant::now();
            if let Some(v2) = &self.v2 {
                if crate::udkws_v2::debug_log_on() {
                    let summary = v2.diag_summary();
                    self.push_debug(format!("[heartbeat] level={:.4} {}", self.last_volume, summary));
                }
            }
        }
        for tg in trigs {
            // Global switch: while paused nothing is keyed or recorded (same as the main path -- recognition continues, keys/events/slices are skipped)
            if !self.enabled.load(std::sync::atomic::Ordering::Relaxed) {
                if !self.passcode_cmd.is_empty() && tg.cmd == self.passcode_cmd {
                    self.keypress_count += 1;
                    self.last_command = tg.cmd.clone();
                    self.last_key.clear();
                    eprintln!(
                        "[passcode-listen] {} @t={:.2}s score={:.4} (hit while disabled -> flip the switch only, no key press)",
                        tg.cmd, tg.start_sec, tg.score
                    );
                    self.push_debug(format!(
                        "[passcode-hit] t={:.2}s cmd={} src=engine (listening hit while disabled; the dispatch layer flips the switch)",
                        tg.start_sec, tg.cmd
                    ));
                } else if !self.passcode_cmd.is_empty() {
                    self.push_debug(format!(
                        "[swallow] t={:.2}s cmd={} not the passcode while disabled (still swallowed while listening, zero key output)",
                        tg.start_sec, tg.cmd
                    ));
                    let now_ms = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .map(|d| d.as_millis() as i64)
                        .unwrap_or(0);
                    let last = self
                        .mute_beep_last
                        .get(&tg.cmd)
                        .copied()
                        .unwrap_or(i64::MIN);
                    // saturating_sub: subtracting the i64::MIN sentinel would overflow (debug panic; in release it wraps negative
                    // and the cue tone never plays)
                    if now_ms.saturating_sub(last) > 2000 {
                        self.mute_beep_last.insert(tg.cmd.clone(), now_ms);
                        crate::keys::message_beep(0x30); // MB_ICONASTERISK (the system beep)
                    }
                }
                continue;
            }
            let toggle_release = self.toggle_held.contains_key(&tg.cmd);
            if (self.vad_gate || self.sv_gate) && !toggle_release && !self.passed_veto(&tg) {
                continue;
            }
            let key = self
                .command_to_key
                .get(&tg.cmd)
                .cloned()
                .unwrap_or_default();
            let frame = ((tg.start_sec as f64) * 100.0 / self.hop_frames as f64).round() as i64;
            let mut mode = self
                .command_to_mode
                .get(&tg.cmd)
                .cloned()
                .unwrap_or_default();
            if mode == "toggle" && key.is_empty() {
                mode = "tap".into();
            }
            if mode == "macro"
                && self
                    .command_to_macro
                    .get(&tg.cmd)
                    .map_or(true, |a| a.is_empty())
            {
                mode = "tap".into();
            }
            // Trigger log display string (tap/toggle = key name; macro = one line for the whole {macro} of N actions, so action-by-action does not flood it)
            let mut disp_key = key.clone();
            match mode.as_str() {
                "toggle" => {
                    self.release_mutex_peers(&tg.cmd, frame);
                    if let Some(held) = self.toggle_held.remove(&tg.cmd) {
                        // Second trigger -> release (reverse-release semantics live in keys::key_up)
                        self.key_events
                            .push((frame, "up", held.clone(), tg.cmd.clone(), tg.score));
                        if self.emit_keys {
                            keys::key_up(&held);
                        }
                        disp_key = held.clone();
                        self.push_debug(format!(
                            "[toggle] frame={} cmd={} up {held} (released by the second trigger)",
                            frame, tg.cmd
                        ));
                    } else {
                        self.key_events.push((
                            frame,
                            "down",
                            key.clone(),
                            tg.cmd.clone(),
                            tg.score,
                        ));
                        if self.emit_keys {
                            keys::key_down(&key);
                        }
                        self.toggle_held.insert(tg.cmd.clone(), key.clone());
                        self.push_debug(format!(
                            "[toggle] frame={} cmd={} down {key} (pressed by the first trigger)",
                            frame, tg.cmd
                        ));
                    }
                    // Both count as one trigger (keypress_count is incremented once in the shared tail)
                }
                "macro" => {
                    let policy = crate::config::normalize_macro_reentry(
                        self.command_to_reentry.get(&tg.cmd).map(String::as_str),
                    );
                    let running_now = self
                        .macro_running
                        .get(&tg.cmd)
                        .map(|f| f.load(std::sync::atomic::Ordering::Relaxed))
                        .unwrap_or(false);
                    match policy {
                        "concurrent" => {
                            // Global concurrency cap: first reclaim dead entries (keeping instances with running=true),
                            // then compare the active count against the cap (over the cap is treated as ignored)
                            self.macro_instances.retain(|inst| {
                                inst.running.load(std::sync::atomic::Ordering::Relaxed)
                            });
                            if self.macro_active.load(std::sync::atomic::Ordering::Relaxed)
                                >= MACRO_MAX_CONCURRENT
                            {
                                self.macro_notices.push((tg.cmd.clone(), "cap"));
                                if self.macro_notices.len() > 32 {
                                    let excess = self.macro_notices.len() - 32;
                                    self.macro_notices.drain(..excess);
                                }
                                self.push_debug(format!(
                                    "[macro] frame={} cmd={} ignored, concurrency cap reached (active instances >= {MACRO_MAX_CONCURRENT})",
                                    frame, tg.cmd));
                                continue;
                            }
                        }
                        "interrupt" => {
                            if running_now {
                                if let Some(tok) = self.macro_tokens.get(&tg.cmd) {
                                    tok.store(true, std::sync::atomic::Ordering::Relaxed);
                                }
                                self.push_debug(format!(
                                    "[macro] frame={} cmd={} interrupting the old instance (it releases its held keys in reverse order)",
                                    frame, tg.cmd
                                ));
                            }
                            // Unconditionally install a fresh token and flag: old instances keep writing to their old Arc without interference,
                            // and the new instance's cancel.store(false) cannot revive an old thread still holding the old token
                            self.macro_running.insert(
                                tg.cmd.clone(),
                                std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
                            );
                            self.macro_tokens.insert(
                                tg.cmd.clone(),
                                std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
                            );
                        }
                        _ => {
                            // ignore (default): a new trigger while the previous instance is running is ignored
                            if running_now {
                                self.macro_notices.push((tg.cmd.clone(), "reentry"));
                                if self.macro_notices.len() > 32 {
                                    let excess = self.macro_notices.len() - 32;
                                    self.macro_notices.drain(..excess);
                                }
                                self.push_debug(format!(
                                    "[macro] frame={} cmd={} re-entry ignored (the previous instance is still running)",
                                    frame, tg.cmd
                                ));
                                continue;
                            }
                        }
                    }
                    self.release_mutex_peers(&tg.cmd, frame);
                    let actions = self
                        .command_to_macro
                        .get(&tg.cmd)
                        .cloned()
                        .unwrap_or_default();
                    // One line for the whole trigger (the execution thread does not push per-action key_events, avoiding log floods)
                    self.key_events.push((
                        frame,
                        "macro",
                        format!("{} actions", actions.len()),
                        tg.cmd.clone(),
                        tg.score,
                    ));
                    if self.live_streaming {
                        let (running, cancel) = if policy == "concurrent" {
                            let inst = MacroInstance {
                                cmd: tg.cmd.clone(),
                                running: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(
                                    false,
                                )),
                                cancel: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(
                                    false,
                                )),
                            };
                            let (r, c) = (inst.running.clone(), inst.cancel.clone());
                            self.macro_instances.push(inst);
                            (r, c)
                        } else {
                            let r = self
                                .macro_running
                                .entry(tg.cmd.clone())
                                .or_insert_with(|| {
                                    std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false))
                                })
                                .clone();
                            let c = self
                                .macro_tokens
                                .entry(tg.cmd.clone())
                                .or_insert_with(|| {
                                    std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false))
                                })
                                .clone();
                            (r, c)
                        };
                        running.store(true, std::sync::atomic::Ordering::Relaxed);
                        cancel.store(false, std::sync::atomic::Ordering::Relaxed);
                        // Global active-instance count: incremented before spawn, decremented when the thread finishes (rolled back if spawn fails)
                        self.macro_active
                            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        let emit = self.emit_keys;
                        let running_thread = running.clone();
                        let active_counter = self.macro_active.clone();
                        // The thread owns copies of the action list and command name (actions are still needed on the inline replay path)
                        let actions_t = actions.clone();
                        let name_t = tg.cmd.clone();
                        let spawned = std::thread::Builder::new()
                            .name(format!("macro-{name_t}"))
                            .spawn(move || {
                                let rep = run_macro(&actions_t, emit, &cancel, true);
                                if rep.cancelled {
                                    eprintln!("[macro] {name_t} cancelled; released in reverse: {:?}", rep.released);
                                }
                                running_thread.store(false, std::sync::atomic::Ordering::Relaxed);
                                active_counter.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
                            });
                        if let Err(e) = spawned {
                            running.store(false, std::sync::atomic::Ordering::Relaxed);
                            self.macro_active
                                .fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
                            eprintln!("[macro] failed to spawn the execution thread for {} ({e}); the macro did not run", tg.cmd);
                        }
                    } else {
                        // replay (playback/offline): inline instantaneous execution (no keys, no sleeping) -- deterministic replay takes priority
                        let never = std::sync::atomic::AtomicBool::new(false);
                        let _ = run_macro(&actions, false, &never, false);
                    }
                    self.last_key.clear(); // For a macro command the "key" is the whole sequence (the key field is intentionally empty)
                    self.push_debug(format!(
                        "[key] frame={} cmd={} macro with {} actions src=engine",
                        frame,
                        tg.cmd,
                        actions.len()
                    ));
                    self.keypress_count += 1;
                    self.last_command = tg.cmd.clone();
                    eprintln!(
                        "[trigger] {} -> {{macro}}x{} @frame {} [engine t={:.2}s-{:.2}s score={:.4}]",
                        tg.cmd,
                        actions.len(),
                        frame,
                        tg.start_sec,
                        tg.end_sec,
                        tg.score
                    );
                    continue;
                }
                _ => {
                    self.release_mutex_peers(&tg.cmd, frame);
                    if !key.is_empty() {
                        self.key_events.push((
                            frame,
                            "press",
                            key.clone(),
                            tg.cmd.clone(),
                            tg.score,
                        ));
                        if self.emit_keys {
                            if self.live_streaming {
                                let k = key.clone();
                                std::thread::Builder::new()
                                    .name("tap-key".into())
                                    .spawn(move || {
                                        keys::press_key_hold(&k, 30);
                                    })
                                    .map_err(|e| eprintln!("[tap] spawn failed: {e}"))
                                    .ok();
                            } else {
                                keys::press_key(&key);
                            }
                        }
                    }
                }
            }
            // Passive recording slices: v2 triggers are dumped according to learn_enabled;
            // frame uses v2's own frame number (in v2 mode global_frame does not advance and would collide at 0)
            self.keypress_count += 1;
            self.last_command = tg.cmd.clone();
            self.last_key = key;
            eprintln!(
                "[trigger] {} -> {} @frame {} [engine t={:.2}s-{:.2}s score={:.4}]",
                tg.cmd, disp_key, frame, tg.start_sec, tg.end_sec, tg.score
            );
            let msg = format!(
                "[key] frame={} cmd={} key={} total={} src=engine",
                frame, tg.cmd, disp_key, self.keypress_count
            );
            self.push_debug(msg);
        }
        // The event list keeps only the tail above the watermark: prevents unbounded growth of key_events under heavy triggering
        self.compact_event_logs();
    }

    pub fn set_passcode_cmd(&mut self, cmd: &str) {
        self.passcode_cmd = cmd.to_string();
    }

    /// Current speaking state (tests and UI)
    pub fn is_speaking(&self) -> bool {
        self.is_speaking
    }

    /// Command name list (UI)
    pub fn commands(&self) -> Vec<String> {
        self.command_to_key.keys().cloned().collect()
    }

    /// Decision-path diagnostics (UI and live log; see config::judge_path_report)
    pub fn judge_report(&self) -> Vec<String> {
        self.judge_report.clone()
    }

    /// Latest level (UI)
    pub fn last_volume(&self) -> f32 {
        self.last_volume
    }

    pub fn frames_since_audible(&self) -> Option<i64> {
        if self.last_audible_frame < 0 {
            None
        } else {
            Some(self.global_frame - self.last_audible_frame)
        }
    }

    /// Most recent triggered command (UI)
    pub fn last_command(&self) -> String {
        self.last_command.clone()
    }

    /// Key-press count (UI)
    pub fn keypress_count(&self) -> usize {
        self.keypress_count
    }

    pub fn swallowed_count(&self) -> u64 {
        self.swallowed_count
    }

    pub fn frame_sec(&self) -> f64 {
        self.hop_frames as f64 / 100.0
    }
}

impl Drop for LiveProcessor {
    fn drop(&mut self) {
        self.release_all_toggles();
        self.cancel_all_macros();
    }
}

// -- Tests: gating design invariants --

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::CommandConfig;

    #[test]
    fn vad_longest_speech_run_hysteresis() {
        let thr = 0.5;
        let gap = VAD_MAX_GAP_FRAMES;
        // Single-frame spike (a noise transient) -> run length 1, far below 8 -> must be vetoed
        assert_eq!(longest_speech_run(&[0.9, 0.0, 0.0, 0.0, 0.0], thr, gap), 1);
        // 8 consecutive frames (~250 ms) -> just counts as one speech segment
        assert_eq!(longest_speech_run(&[0.9; 8], thr, gap), 8);
        // 7 frames -> not enough (boundary)
        assert_eq!(longest_speech_run(&[0.9; 7], thr, gap), 7);
        // Gaps of up to 3 frames are tolerated: 6 speech frames split by 3 silent ones still count as continuous (3+gap+3)
        assert_eq!(
            longest_speech_run(&[1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0], thr, gap),
            9
        );
        // A 4-frame gap exceeds the tolerance -> two segments of 3 frames each -> the longest (3) wins
        assert_eq!(
            longest_speech_run(
                &[1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                thr,
                gap
            ),
            3
        );
        // Threshold boundary: exactly at the threshold passes (>= semantics)
        assert_eq!(longest_speech_run(&[thr, thr], thr, gap), 2);
        // All silence / empty input
        assert_eq!(longest_speech_run(&[0.0; 8], thr, gap), 0);
        assert_eq!(longest_speech_run(&[], thr, gap), 0);
        // Takes the longest run rather than the sum: the long middle run wins
        assert_eq!(
            longest_speech_run(
                &[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                thr,
                gap
            ),
            6
        );
    }

    fn make_config(mode: &str) -> CommandConfig {
        // alpha: A+B -> K1 (abstract phoneme families, decoupled from real command words)
        let raw = serde_json::json!({
            "commands": {
                "alpha": {"key": "K1", "mode": mode},
            },
        });
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let dir = std::env::temp_dir().join(format!(
            "viq_gate_{}_{}",
            std::process::id(),
            COUNTER.fetch_add(1, Ordering::SeqCst)
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let cfg_path = dir.join("commands.json");
        std::fs::write(&cfg_path, raw.to_string()).unwrap();
        CommandConfig::load(cfg_path.to_str().unwrap()).unwrap()
    }

    fn make_proc(mode: &str) -> LiveProcessor {
        let cfg = make_config(mode);
        let mut proc = LiveProcessor::new(
            cfg,
            std::sync::Arc::new(std::sync::atomic::AtomicBool::new(true)),
            None,
        );
        proc.set_replay_mode();
        proc
    }

    // -- v2 triggers (passive recording slices) --

    fn make_v2_proc() -> LiveProcessor {
        let proc = make_proc("tap");
        proc
    }

    fn make_config_same_key(mode: &str) -> CommandConfig {
        let raw = serde_json::json!({
            "commands": {
                "alpha": {"key": "K1", "mode": mode},
                "beta": {"key": "K1", "mode": mode},
            },
        });
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let dir = std::env::temp_dir().join(format!(
            "viq_gate_{}_{}",
            std::process::id(),
            COUNTER.fetch_add(1, Ordering::SeqCst)
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let cfg_path = dir.join("commands.json");
        std::fs::write(&cfg_path, raw.to_string()).unwrap();
        CommandConfig::load(cfg_path.to_str().unwrap()).unwrap()
    }

    fn make_v2_proc_same_key() -> LiveProcessor {
        let cfg = make_config_same_key("tap");
        let mut proc = LiveProcessor::new(
            cfg,
            std::sync::Arc::new(std::sync::atomic::AtomicBool::new(true)),
            None,
        );
        proc.set_replay_mode();
        proc
    }

    fn v2_trig(start_sec: f64, cmd: &str) -> crate::udkws_v2::Trigger {
        crate::udkws_v2::Trigger {
            start_sec: start_sec as f32,
            end_sec: (start_sec + 0.08) as f32,
            cmd: cmd.to_string(),
            score: 0.9,
            t: (start_sec * 100.0) as usize,
        }
    }

    /// Two commands on the same key: one event each, and slices and templates stay under their own command.
    #[test]
    fn v2_same_key_two_commands_record_own_cmd() {
        let mut proc = make_v2_proc_same_key();
        proc.emit_v2_triggers(vec![v2_trig(1.0, "alpha"), v2_trig(2.0, "beta")]);
        assert_eq!(proc.key_events.len(), 2, "one event recorded per command");
        assert_eq!(proc.key_events[0].2, "K1", "same key K1");
        assert_eq!(proc.key_events[1].2, "K1", "same key K1");
        assert_eq!(proc.key_events[0].3, "alpha", "the first nucleus must carry its own command");
        assert_eq!(proc.key_events[1].3, "beta", "the second nucleus must carry its own command");
    }

    /// Listening state (enabled=false with a passcode configured): a passcode hit only advances
    /// keypress_count/last_command (zero keys, zero events),
    /// while non-passcode hits are swallowed with no output -- the dispatch layer (UI voice_toggle_dispatch) flips the switch on that.
    #[test]
    fn disabled_listening_passcode_hits_but_others_swallowed() {
        let mut proc = make_v2_proc();
        proc.set_passcode_cmd("pass");
        proc.set_enabled(false);
        // A non-passcode command first: still swallowed, zero output and zero count
        proc.emit_v2_triggers(vec![v2_trig(1.0, "alpha")]);
        assert_eq!(proc.keypress_count, 0, "a non-passcode hit must not advance the count");
        assert!(proc.key_events.is_empty(), "a non-passcode hit produces no key output");
        assert!(proc.last_command.is_empty(), "a non-passcode hit must not touch the last command");
        // Passcode hit: advances the count and last command, but zero keys and zero events (the UI frame flips the switch on that)
        proc.emit_v2_triggers(vec![v2_trig(2.0, "pass")]);
        assert_eq!(proc.keypress_count, 1, "a passcode hit must advance the count (the dispatch criterion)");
        assert_eq!(proc.last_command, "pass", "the last command must be the passcode");
        assert!(
            proc.key_events.is_empty(),
            "a passcode hit produces no key output (mute semantics)"
        );
        // Another non-passcode command: still swallowed
        proc.emit_v2_triggers(vec![v2_trig(3.0, "alpha")]);
        assert_eq!(proc.keypress_count, 1);
        assert_eq!(proc.last_command, "pass", "a non-passcode command must not overwrite the last command");
        assert!(proc.key_events.is_empty());
    }

    /// No passcode configured (empty passcode_cmd, CLI replay, or settings unset): the disabled state swallows everything,
    /// command by command -- there is no listening without a passcode.
    #[test]
    fn disabled_no_passcode_keeps_full_swallow() {
        let mut proc = make_v2_proc();
        assert!(proc.passcode_cmd.is_empty(), "no passcode configured by default");
        proc.set_enabled(false);
        proc.emit_v2_triggers(vec![v2_trig(1.0, "pass"), v2_trig(2.0, "alpha")]);
        assert_eq!(proc.keypress_count, 0, "no passcode: the disabled state swallows everything (legacy behaviour)");
        assert!(proc.last_command.is_empty(), "no passcode: the last command must not change");
        assert!(proc.key_events.is_empty(), "no passcode: zero key output");
    }

    /// Passcode cleared midway (set_passcode_cmd("")): listening turns off as well -- disabled swallows everything
    /// (listening is a derived state of the passcode configuration, not an independent switch).
    #[test]
    fn disabled_listening_stops_when_passcode_cleared() {
        let mut proc = make_v2_proc();
        proc.set_passcode_cmd("pass");
        proc.set_enabled(false);
        proc.set_passcode_cmd("");
        proc.emit_v2_triggers(vec![v2_trig(1.0, "pass")]);
        assert_eq!(proc.keypress_count, 0, "after clearing the passcode the disabled state swallows everything");
        assert!(proc.last_command.is_empty(), "after clearing the passcode the last command must not change");
    }

    #[test]
    fn event_log_cap_truncates_keeps_counts_tail_and_cursor_safe() {
        let mut proc = make_v2_proc();
        for i in 0..15000i64 {
            proc.key_events
                .push((i, "press", "K1".into(), "C".into(), 0.0));
            proc.swallowed_events.push((i, "cmd".into()));
            proc.keypress_count += 1;
            proc.swallowed_count += 1;
        }
        proc.emit_v2_triggers(vec![]); // An empty trigger list still runs the watermark check (one of the real wiring points)
        assert!(
            proc.key_events.len() <= 12000,
            "key_events must stay bounded after head truncation, got {}",
            proc.key_events.len()
        );
        assert!(
            proc.swallowed_events.len() <= 12000,
            "swallowed_events must stay bounded after head truncation"
        );
        assert_eq!(
            proc.keypress_count, 15000,
            "the trigger count must be unaffected by truncation"
        );
        assert_eq!(proc.swallowed_count, 15000, "the swallowed total must be unaffected by truncation");
        assert_eq!(proc.key_events.last().unwrap().0, 14999, "the tail window is complete");
        assert_eq!(
            proc.swallowed_events.last().unwrap().0,
            14999,
            "the tail window is complete"
        );
        let stale_cursor = 15000usize; // A stale cursor that already consumed all 15000 entries
        proc.key_events
            .push((15000, "press", "K1".into(), "C".into(), 0.0));
        let n = proc.key_events.len();
        let fresh = proc.key_events[stale_cursor.min(n)..n].to_vec();
        assert!(
            fresh.is_empty(),
            "a stale cursor in an oversized session hides new events (known trade-off), got {fresh:?}"
        );
    }

    fn make_config_raw(
        mode: &str,
        macro_v: Option<serde_json::Value>,
        reentry: &str,
    ) -> CommandConfig {
        let mut raw = serde_json::json!({
            "commands": {
                "alpha": {"key": "K1", "mode": mode},
            },
        });
        if let Some(mv) = macro_v {
            raw["commands"]["alpha"]["macro"] = mv;
        }
        if !reentry.is_empty() {
            raw["commands"]["alpha"]["macro_reentry"] = serde_json::Value::String(reentry.into());
        }
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let dir = std::env::temp_dir().join(format!(
            "viq_gate_{}_{}",
            std::process::id(),
            COUNTER.fetch_add(1, Ordering::SeqCst)
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let cfg_path = dir.join("commands.json");
        std::fs::write(&cfg_path, raw.to_string()).unwrap();
        CommandConfig::load(cfg_path.to_str().unwrap()).unwrap()
    }

    /// Macro command proc: alpha = down B -> wait `wait_ms` -> up B.
    /// `reentry`: "" = field absent (ignore); `streaming`=true reopens live-thread mode
    /// (make_proc already turned replay mode off); emit_keys is always false (a replay side effect),
    /// so tests never send real keys.
    fn make_macro_proc_with(reentry: &str, streaming: bool, wait_ms: u64) -> LiveProcessor {
        let macro_v = serde_json::json!([
            {"action": "down", "key": "B"},
            {"action": "wait", "ms": wait_ms},
            {"action": "up", "key": "B"},
        ]);
        let cfg = make_config_raw("macro", Some(macro_v), reentry);
        let mut proc = LiveProcessor::new(
            cfg,
            std::sync::Arc::new(std::sync::atomic::AtomicBool::new(true)),
            None,
        );
        proc.set_replay_mode();
        proc.live_streaming = streaming;
        proc
    }

    fn make_macro_proc(streaming: bool) -> LiveProcessor {
        make_macro_proc_with("", streaming, 400)
    }

    /// Macro executor: a complete run clears the stack; a mid-run cancel releases the down stack in reverse; honor_waits=false is instantaneous
    #[test]
    fn macro_run_cancel_releases_down_stack_reverse() {
        use crate::config::MacroAction;
        use std::sync::atomic::{AtomicBool, Ordering};
        // Complete run: no cancel, stack cleared
        let actions = vec![
            MacroAction::Down("B".into()),
            MacroAction::Wait(30),
            MacroAction::Up("B".into()),
        ];
        let rep = run_macro(&actions, false, &AtomicBool::new(false), true);
        assert!(!rep.cancelled);
        assert!(rep.released.is_empty(), "a complete run must leave nothing on the stack");
        // Mid-run cancel: exits at a wait step and releases, in reverse, every key that went down but not up
        let actions = vec![
            MacroAction::Down("B".into()),
            MacroAction::Down("C".into()),
            MacroAction::Wait(5000),
            MacroAction::Up("B".into()),
        ];
        let cancel = std::sync::Arc::new(AtomicBool::new(false));
        let c2 = cancel.clone();
        let h = std::thread::spawn(move || run_macro(&actions, false, &c2, true));
        std::thread::sleep(std::time::Duration::from_millis(200));
        cancel.store(true, Ordering::Relaxed);
        let rep = h.join().unwrap();
        assert!(rep.cancelled, "the cancel token must interrupt execution");
        assert_eq!(
            rep.released,
            vec!["C".to_string(), "B".to_string()],
            "released in reverse (last pressed, first released); up B is handled by the release path"
        );
        let actions = vec![MacroAction::Wait(5000)];
        let t0 = std::time::Instant::now();
        let rep = run_macro(&actions, false, &AtomicBool::new(false), false);
        assert!(
            !rep.cancelled && t0.elapsed() < std::time::Duration::from_millis(500),
            "the inline path must not actually sleep"
        );
    }

    /// Live macro trigger: spawns a dedicated thread, one key_events line for the whole macro, re-entry ignored (notified once),
    /// triggerable again once the run finishes; cancel_all_macros cancels running instances
    #[test]
    fn macro_trigger_spawns_thread_ignores_reentry_and_cancels() {
        let mut proc = make_macro_proc(true);
        proc.emit_v2_triggers(vec![v2_trig(1.0, "alpha")]);
        assert_eq!(
            proc.key_events.len(),
            1,
            "a macro must be one event (not one per action)"
        );
        assert_eq!(proc.key_events[0].1, "macro");
        assert_eq!(proc.key_events[0].2, "3 actions");
        assert_eq!(proc.keypress_count, 1);
        assert!(proc.macro_is_running("alpha"), "live mode must spawn an execution thread");
        // Re-entry: the new trigger is ignored -- no event, no count, one status-bar notice
        proc.emit_v2_triggers(vec![v2_trig(2.0, "alpha")]);
        assert_eq!(proc.key_events.len(), 1, "re-entry must not record another event");
        assert_eq!(proc.keypress_count, 1, "re-entry must not advance the count");
        assert_eq!(proc.macro_notices.len(), 1, "re-entry must raise one UI notice");
        // Wait for the first run to finish (400 ms wait plus margin) -> the running flag clears itself -> triggering is allowed again
        for _ in 0..40 {
            if !proc.macro_is_running("alpha") {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(50));
        }
        assert!(
            !proc.macro_is_running("alpha"),
            "the running flag must clear itself when the macro thread ends"
        );
        proc.emit_v2_triggers(vec![v2_trig(3.0, "alpha")]);
        assert_eq!(proc.key_events.len(), 2, "a trigger after the run completes must be allowed");
        assert_eq!(proc.keypress_count, 2);
        assert_eq!(proc.macro_notices.len(), 1, "a non-re-entry trigger must not add a notice");
        // Cancel a running instance (the same entry point as stop-recognition and delete-command): token set -> thread exits
        proc.cancel_all_macros();
        for _ in 0..40 {
            if !proc.macro_is_running("alpha") {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(50));
        }
        assert!(!proc.macro_is_running("alpha"), "the running flag must be cleared after cancelling");
    }

    /// replay (playback/offline): the macro runs inline and instantaneously -- records the event and advances the count, but spawns no thread
    #[test]
    fn macro_replay_mode_inline_records_event_no_thread() {
        let mut proc = make_macro_proc(false);
        assert!(!proc.live_streaming, "set_replay_mode must turn off live thread mode");
        let t0 = std::time::Instant::now();
        proc.emit_v2_triggers(vec![v2_trig(1.0, "alpha")]);
        assert!(
            t0.elapsed() < std::time::Duration::from_millis(500),
            "inline execution does not sleep"
        );
        assert_eq!(proc.key_events.len(), 1, "replay also records one event for the whole macro");
        assert_eq!(proc.key_events[0].1, "macro");
        assert_eq!(proc.keypress_count, 1);
        assert!(!proc.macro_is_running("alpha"), "replay spawns no thread");
    }

    #[test]
    fn macro_concurrent_policy_spawns_parallel_with_global_cap() {
        let mut proc = make_macro_proc_with("concurrent", true, 8000);
        // Four triggers in a row: all spawn concurrently (a running instance does not block them)
        for i in 0..MACRO_MAX_CONCURRENT {
            proc.emit_v2_triggers(vec![v2_trig(1.0 + i as f64 * 0.1, "alpha")]);
        }
        assert_eq!(proc.key_events.len(), 4, "under the concurrent policy a trigger while running is not ignored");
        assert_eq!(proc.keypress_count, 4);
        assert_eq!(proc.macro_active_count(), 4, "four instances must stay alive concurrently");
        assert!(proc.macro_notices.is_empty(), "no notice below the cap");
        // Fifth trigger: the global cap is reached -> treated as ignored (no event, no count, one notice)
        proc.emit_v2_triggers(vec![v2_trig(2.0, "alpha")]);
        assert_eq!(proc.key_events.len(), 4, "a trigger over the cap must not record another event");
        assert_eq!(proc.keypress_count, 4, "a trigger over the cap must not advance the count");
        assert_eq!(proc.macro_notices.len(), 1, "reaching the cap must raise one notice");
        assert_eq!(proc.macro_notices[0].1, "cap", "the notice kind must be the concurrency cap");
        assert_eq!(proc.macro_active_count(), 4, "a trigger over the cap must not add an instance");
        // cancel_all_macros cancels everything (including registry instances) -> the active count returns to zero
        proc.cancel_all_macros();
        for _ in 0..60 {
            if proc.macro_active_count() == 0 {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(50));
        }
        assert_eq!(proc.macro_active_count(), 0, "a full cancel must reset the active count to zero");
    }

    #[test]
    fn macro_interrupt_policy_cancels_old_and_spawns_new() {
        let mut proc = make_macro_proc_with("interrupt", true, 400);
        proc.emit_v2_triggers(vec![v2_trig(1.0, "alpha")]);
        assert!(proc.macro_is_running("alpha"), "the first trigger must spawn an execution thread");
        // Triggering while running: interrupts the old instance and spawns a new one (records the event, advances the count, no ignore notice)
        proc.emit_v2_triggers(vec![v2_trig(2.0, "alpha")]);
        assert_eq!(proc.key_events.len(), 2, "an interrupt-policy trigger is not ignored");
        assert_eq!(proc.keypress_count, 2, "an interrupt-policy trigger advances the count normally");
        assert!(
            proc.macro_notices.is_empty(),
            "interrupt raises no ignore or cap notice"
        );
        // The old instance is cancelled (exits within 50 ms) -> the active count converges to 1 (the new instance is still in its 400 ms wait)
        for _ in 0..20 {
            if proc.macro_active_count() <= 1 {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(50));
        }
        assert_eq!(
            proc.macro_active_count(),
            1,
            "the interrupted instance must exit and the new one stay alive"
        );
        assert!(
            proc.macro_is_running("alpha"),
            "the new instance must be visible in the running table"
        );
        // Cleanup: cancel the new instance (the stop-recognition entry point), active count back to zero
        proc.cancel_all_macros();
        for _ in 0..20 {
            if proc.macro_active_count() == 0 {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(50));
        }
        assert_eq!(proc.macro_active_count(), 0, "the active count must return to zero after cancelling");
    }
}
