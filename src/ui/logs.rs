use super::*;
use crate::i18n_keys::k;
use crate::lang::t;

/// Draw one trigger-log line: one entry per line, colour keyed by category only.
pub(super) fn draw_log_row(ui: &mut egui::Ui, row: &LogRow) {
    let color = match row.kind {
        LogKind::Warn => egui::Color32::from_rgb(196, 118, 40),
        LogKind::Note => egui::Color32::from_rgb(150, 150, 143),
        LogKind::Trigger => ui.visuals().text_color(),
    };
    ui.label(egui::RichText::new(&row.text).monospace().color(color));
}

impl ViqApp {
    /// Trigger-log increment: new key_events / swallowed_events -> timestamped log lines
    pub(super) fn update_logs(&mut self) {
        let mut aec_fallback: Option<Result<(), String>> = None;
        if let Some(c) = &mut self.ctx {
            let mut fallback = false;
            if let Some(rx) = &c.aec_fallback_rx {
                if rx.try_recv().is_ok() {
                    fallback = true;
                }
            }
            if fallback {
                c.aec_fallback_rx = None;
                c.aec_alive = None; // The capture thread exited on its own, so the guard lapses
                let dev = if self.settings.device.is_empty() {
                    None
                } else {
                    Some(self.settings.device.clone())
                };
                aec_fallback = Some(
                    match crate::live::open_cpal_stream(
                        dev.as_deref(),
                        c.proc.clone(),
                        c.rec.clone(),
                    ) {
                        Ok(stream) => {
                            c.stream = Some(stream);
                            Ok(())
                        }
                        Err(e) => Err(format!("{e:#}")),
                    },
                );
            }
        }
        if let Some(r) = aec_fallback {
            match r {
                Ok(()) => {
                    self.log("echo cancellation unavailable (capture fault); fell back to plain microphone capture");
                    self.status = t(k::MSG_AEC_FALLBACK_STATUS);
                }
                Err(e) => {
                    self.status = t(k::ERR_AEC_FALLBACK);
                    self.log(&crate::lang::tf(k::ERR_AEC_FALLBACK_DETAIL, &[("{e}", &e)]));
                }
            }
        }
        // Collect the echo-cancellation availability probe (one-shot on a background thread)
        if self.aec_probe.is_none() {
            if let Some(rx) = &self.aec_probe_rx {
                if let Ok(p) = rx.try_recv() {
                    self.aec_probe = Some(p);
                    self.aec_probe_rx = None;
                }
            }
        }
        for msg in crate::keys::drain_send_log() {
            let ts = chrono::Local::now().format("%H:%M:%S");
            self.log_lines.push(LogRow {
                text: format!("[{ts}] ⚠ {msg}"),
                kind: LogKind::Warn,
            });
        }
        let Some(c) = &self.ctx else { return };
        let snap = c
            .proc
            .lock()
            .map(|p| {
                let n = p.key_events.len();
                let count = p.keypress_count();
                let events: Vec<(i64, &'static str, String, String, f64)> =
                    p.key_events[self.logged_events.min(n)..n].to_vec();
                let sn = p.swallowed_events.len();
                let sw: Vec<(i64, String)> =
                    p.swallowed_events[self.logged_swallowed.min(sn)..sn].to_vec();
                (n, count, events, sn, sw)
            })
            .ok();
        let Some((n, count, events, sn, sw)) = snap else {
            return;
        };
        for (f, act, key, cmd, score) in &events {
            let ts = chrono::Local::now().format("%H:%M:%S");
            let sec = *f as f32 * 0.08;
            let conf = if (*act == "press" || *act == "down" || *act == "up" || *act == "macro")
                && *score > 0.0
            {
                format!(" score={:.2}", score)
            } else {
                String::new()
            };
            let cmd = self.cmd_display(cmd);
            // A macro trigger is a single line (the gate side emits key_events action="macro" with key="N actions", not one per action);
            // upx = a release caused by a mutex, shown as a separate grey footnote rather than as a trigger
            let (line, kind) = if *act == "upx" {
                (
                    crate::lang::tf(k::MSG_LOG_NOTE_MUTEX, &[("{key}", key), ("{cmd}", &cmd)]),
                    LogKind::Note,
                )
            } else if *act == "macro" {
                (
                    crate::lang::tf(
                        k::MSG_LOG_MACRO_TRIGGER,
                        &[
                            ("{ts}", &ts.to_string()),
                            ("{cmd}", &cmd),
                            ("{key}", key),
                            ("{count}", &count.to_string()),
                            ("{sec}", &format!("{sec:.1}")),
                            ("{conf}", &conf),
                        ],
                    ),
                    LogKind::Trigger,
                )
            } else {
                (
                    crate::lang::tf(
                        k::MSG_LOG_KEY_PRESS,
                        &[
                            ("{ts}", &ts.to_string()),
                            ("{cmd}", &cmd),
                            ("{key}", key),
                            ("{conf}", &conf),
                            ("{count}", &count.to_string()),
                            ("{sec}", &format!("{sec:.1}")),
                        ],
                    ),
                    LogKind::Trigger,
                )
            };
            self.log_lines.push(LogRow {
                text: line.clone(),
                kind,
            });
            if let Some(c) = &self.ctx {
                c.rec.log(&line);
            }
        }
        for (f, cmd) in &sw {
            let ts = chrono::Local::now().format("%H:%M:%S");
            let sec = *f as f32 * 0.02;
            let line = crate::lang::tf(
                k::MSG_LOG_SWALLOWED,
                &[
                    ("{ts}", &ts.to_string()),
                    ("{cmd}", &cmd),
                    ("{sec}", &format!("{sec:.1}")),
                ],
            );
            self.log_lines.push(LogRow {
                text: line.clone(),
                kind: LogKind::Trigger,
            });
            if let Some(c) = &self.ctx {
                c.rec.log(&line);
            }
        }
        if let Some(c) = &self.ctx {
            let mut new_debug: Vec<String> = Vec::new();
            let mut new_energy: Vec<String> = Vec::new();
            {
                let p = c.proc.lock().unwrap_or_else(|e| e.into_inner());
                let dn = p.debug_events.len();
                let dstart = self.logged_debug.min(dn);
                new_debug.extend(p.debug_events[dstart..dn].iter().cloned());
                self.logged_debug = dn;
                let en = p.energy_trace.len();
                let estart = self.logged_energy.min(en);
                new_energy.extend(p.energy_trace[estart..en].iter().cloned());
                self.logged_energy = en;
            }
            for e in &new_debug {
                c.rec.log(e);
            }
            for e in &new_energy {
                c.rec.log(e);
            }
        }
        self.logged_events = n;
        self.logged_swallowed = sn;
        if let Some(c) = &self.ctx {
            let notices = c
                .proc
                .lock()
                .map(|mut p| p.drain_macro_notices())
                .unwrap_or_default();
            for (cmd, kind) in notices {
                let line = if kind == "cap" {
                    crate::lang::tf(
                        k::MSG_MACRO_CONCURRENCY_IGNORED,
                        &[("{cmd}", &self.cmd_display(&cmd))],
                    )
                } else {
                    crate::lang::tf(
                        k::MSG_MACRO_RUNNING_IGNORED,
                        &[("{cmd}", &self.cmd_display(&cmd))],
                    )
                };
                self.status = line.clone();
                self.log(&line);
            }
        }
        // Keep only the most recent 200 entries
        if self.log_lines.len() > 200 {
            let excess = self.log_lines.len() - 200;
            self.log_lines.drain(..excess);
        }
    }
}
