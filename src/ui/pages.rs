use super::*;
use crate::i18n_keys::k;
use crate::lang::t;
use crate::live;
use eframe::egui;

const CONTENT_W: f32 = 640.0;

/// Reserved height of the fixed bottom area on the home page (logical points): the disclaimer line (long text wraps to two lines at the container width) plus the status line.
/// The trigger-log area then takes the remaining height, so these bottom lines are never pushed out.
const HOME_FOOTER_H: f32 = 76.0;

/// Content column: constrain the content to `max_w` and **centre the whole column**.
fn content_column<R>(ui: &mut egui::Ui, max_w: f32, body: impl FnOnce(&mut egui::Ui) -> R) -> R {
    let avail = ui.available_width();
    let inner = if max_w > 0.0 { avail.min(max_w) } else { avail };
    let pad = ((avail - inner) / 2.0).max(0.0);
    let mut out = None;
    let rect = egui::Rect::from_min_size(
        ui.cursor().min + egui::vec2(pad, 0.0),
        egui::vec2(inner, ui.available_height()),
    );
    ui.scope_builder(
        egui::UiBuilder::new()
            .max_rect(rect)
            .layout(egui::Layout::top_down(egui::Align::Min)),
        |ui| {
            out = Some(body(ui));
        },
    );
    out.expect("the content_column closure is always called exactly once")
}

pub(super) fn text_button(ui: &mut egui::Ui, text: egui::RichText) -> egui::Response {
    ui.scope(|ui| {
        let st = ui.style_mut();
        st.visuals.widgets.inactive.bg_fill = egui::Color32::TRANSPARENT;
        st.visuals.widgets.inactive.weak_bg_fill = egui::Color32::TRANSPARENT;
        st.visuals.widgets.inactive.bg_stroke = egui::Stroke::NONE;
        ui.button(text)
    })
    .inner
}

pub(super) fn count_badge(ui: &mut egui::Ui, n: usize) {
    let bg = ui.visuals().widgets.noninteractive.bg_fill;
    let fg = ui.visuals().weak_text_color();
    egui::Frame::NONE
        .fill(bg)
        .corner_radius(6.0)
        .inner_margin(egui::Margin::symmetric(6, 1))
        .show(ui, |ui| {
            ui.label(egui::RichText::new(n.to_string()).small().color(fg));
        });
}

fn settings_grid(id: &str, ui: &mut egui::Ui, body: impl FnOnce(&mut egui::Ui)) {
    egui::Grid::new(id)
        .num_columns(2)
        .min_col_width(128.0)
        .spacing([12.0, 6.0])
        .show(ui, body);
}

pub(super) struct CmdRow {
    pub name: String,
    pub display: String,
    pub key: String,
    pub enabled: bool,
    pub recs: usize,
    pub mode: String,
    /// "Multi-pronunciation" display value: explicit checkbox > template-derived (exp_nuclei >= 2)
    pub multi: bool,
}

/// Sort key for the pronunciation list (only one variant matches at a time, so cross-variant comparison never happens -- enum order is irrelevant).
#[derive(PartialEq, Eq, PartialOrd, Ord)]
enum CmdKey {
    Name(String),
    Mode(String),
    Key(String),
    Recs(usize),
}

/// Name and mapped key are compared via `to_lowercase` (otherwise `B` would sort before `a`); the recording count is **numeric**
/// (not lexicographic: `10` must sort after `9`); ties fall back to id.
pub(super) fn sort_cmd_rows(rows: &mut Vec<CmdRow>, col: CmdCol, desc: bool) {
    let key_of = |r: &CmdRow| match col {
        CmdCol::Name => CmdKey::Name(r.display.to_lowercase()),
        CmdCol::Mode => CmdKey::Mode(r.mode.clone()),
        CmdCol::Key => CmdKey::Key(r.key.to_lowercase()),
        CmdCol::Recs => CmdKey::Recs(r.recs),
    };
    sort_table_rows(rows, desc, key_of, |r| r.name.clone());
}

impl ViqApp {
    pub(super) fn tab_bar(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            let tabs = [
                (Tab::Home, t(k::UI_TAB_HOME)),
                (Tab::Pron, t(k::UI_TAB_CMDS)),
                (Tab::Settings, t(k::UI_TAB_SETTINGS)),
            ];
            for (tab, label) in tabs {
                if ui
                    .selectable_label(self.tab == tab, egui::RichText::new(label).size(16.0))
                    .clicked()
                {
                    self.tab = tab;
                    if self.tab == Tab::Pron {
                        self.refresh_rec_counts();
                    }
                }
                ui.add_space(24.0); // Navigation item spacing ~32 px (yuke-ui-spec Layout Spacing)
            }
        });
        ui.separator();
        ui.add_space(8.0); // Navigation to body content ~16 px (yuke-ui-spec Layout Spacing)
    }

    /// Tab 1, Home: master switch + start input + save recordings + voice toggle + key toggle
    /// + level meter (always visible) + trigger log
    pub(super) fn tab_home(&mut self, ui: &mut egui::Ui) {
        content_column(ui, CONTENT_W, |ui| {
            ui.horizontal(|ui| {
                let running = self.ctx.is_some();
                let loading = self.capture_job != CaptureJob::None || self.warming;
                let paused = !self.enabled_flag.load(std::sync::atomic::Ordering::Relaxed);
                let btn = if running {
                    t(k::UI_BTN_STOP_RECOGNITION)
                } else {
                    t(k::UI_BTN_START_RECOGNITION)
                };
                let resp = ui.add_enabled(
                    !loading,
                    egui::Button::new(egui::RichText::new(btn).size(16.0))
                        .min_size(egui::vec2(150.0, 36.0)),
                );
                if resp.clicked() {
                    self.toggle_capture();
                }
                let (color, state) = if !running {
                    (egui::Color32::GRAY, t(k::UI_LBL_STATE_NOT_RUNNING))
                } else if paused {
                    (WARN_COLOR, t(k::UI_LBL_STATE_PAUSED))
                } else {
                    (
                        egui::Color32::from_rgb(76, 175, 80),
                        t(k::UI_LBL_STATE_RECOGNIZING),
                    )
                };
                ui.colored_label(color, format!("● {state}"));
            });
            // "Start input" (renamed from "automatic keying"; moved here from the settings page's global switch, and
            // shares the state source enabled_flag with the key toggle and the "paused" status text)
            ui.horizontal(|ui| {
                let mut enabled = self.enabled_flag.load(std::sync::atomic::Ordering::Relaxed);
                if ui
                    .checkbox(&mut enabled, t(k::UI_BTN_ENABLE_INPUT))
                    .changed()
                {
                    self.enabled_flag
                        .store(enabled, std::sync::atomic::Ordering::Relaxed);
                    if let Some(c) = &self.ctx {
                        // Unified key-release path: set_enabled(false) releases toggle-held keys internally (gate.rs)
                        c.proc
                            .lock()
                            .unwrap_or_else(|e| e.into_inner())
                            .set_enabled(enabled);
                    }
                    self.last_enabled_seen = enabled; // The key-release path already ran, so the frame check skips it
                    self.status = if enabled {
                        t(k::MSG_KEYS_RESUMED)
                    } else {
                        t(k::MSG_KEYS_PAUSED)
                    };
                    self.log(&self.status.clone());
                }
                ui.separator();
                let mut rec = self.settings.record_enabled;
                if ui
                    .checkbox(&mut rec, t(k::UI_BTN_SAVE_RECORDINGS))
                    .changed()
                {
                    self.settings.record_enabled = rec;
                    self.persist_settings();
                    if let Some(c) = &self.ctx {
                        c.rec.set_record_enabled(rec);
                    }
                }
            });
            ui.horizontal(|ui| {
                ui.label(t(LABEL_VOICE_TOGGLE));
                let prev = self.settings.voice_toggle_cmd.clone();
                egui::ComboBox::from_id_salt("voice_toggle_cmd")
                    .selected_text(if prev.is_empty() {
                        t(k::UI_OPT_DISABLED)
                    } else {
                        format!("「{}」", self.cmd_display(&prev))
                    })
                    .width(160.0 + 2.0 * ui.spacing().button_padding.x) // ComboBox::width is the minimum of (total width - 2x padding); compensate to an actual width of 160
                    .show_ui(ui, |ui| {
                        ui.selectable_value(
                            &mut self.settings.voice_toggle_cmd,
                            String::new(),
                            t(k::UI_OPT_DISABLED),
                        );
                        let mut entries: Vec<(String, String)> = self
                            .cfg
                            .get("commands")
                            .and_then(|v| v.as_object())
                            .map(|o| {
                                o.iter()
                                    .map(|(id, c)| {
                                        let d = c
                                            .get("phrase")
                                            .and_then(|v| v.as_str())
                                            .filter(|s| !s.is_empty())
                                            .unwrap_or(id)
                                            .to_string();
                                        (id.clone(), d)
                                    })
                                    .collect()
                            })
                            .unwrap_or_default();
                        entries.sort_by(|a, b| a.1.cmp(&b.1));
                        for (id, display) in entries {
                            ui.selectable_value(
                                &mut self.settings.voice_toggle_cmd,
                                id.clone(),
                                format!("「{display}」"),
                            );
                        }
                    })
                    .response
                    .on_hover_text(t(k::UI_HINT_VOICE_TOGGLE));
                if self.settings.voice_toggle_cmd != prev {
                    self.persist_settings();
                    if self.ctx.is_some() {
                        // Clearing the passcode key takes effect at engine construction, so hot-applying the configuration rebuilds through the same path
                        self.restart_capture_async(&t(k::MSG_VOICE_TOGGLE_CHANGED));
                    } else if self.proc_rx.is_some() {
                        // The engine is loading in the background: once it finishes, rebuild again with the latest passcode (the existing reload_pending mechanism)
                        self.reload_pending = true;
                    }
                    self.log(&crate::tfmt!(
                        "voice toggle -> {}",
                        if self.settings.voice_toggle_cmd.is_empty() {
                            t(k::UI_OPT_DISABLED)
                        } else {
                            format!("「{}」", self.cmd_display(&self.settings.voice_toggle_cmd))
                        }
                    ));
                }
            });
            ui.horizontal(|ui| {
                ui.label(t(LABEL_KEY_TOGGLE));
                if self.capture_hotkey {
                    ui.colored_label(WARN_COLOR, t(k::UI_HINT_PRESS_KEY));
                } else {
                    let hk = if self.settings.hotkey.is_empty() {
                        t(k::UI_OPT_NOT_SET)
                    } else {
                        self.settings.hotkey.clone()
                    };
                    if ui.button(hk).clicked() {
                        self.capture_hotkey = true;
                    }
                }
                if self.capture_hotkey {
                    if ui.button(t(k::UI_BTN_CANCEL)).clicked() {
                        self.capture_hotkey = false;
                        self.status = t(k::MSG_KEY_CAPTURE_CANCELLED);
                    }
                } else if ui.button(t(k::UI_BTN_CLEAR)).clicked() {
                    self.settings.hotkey.clear();
                    self.apply_hotkey();
                    self.persist_settings();
                }
                ui.label(t(k::UI_LBL_TRIGGER_MODE));
                let mut mode = self.settings.hotkey_mode;
                egui::ComboBox::from_id_salt("hotkey_mode_select")
                    .selected_text(match mode {
                        1 => t(k::UI_OPT_HOLD_PAUSE),
                        2 => t(k::UI_OPT_HOLD_TALK),
                        _ => t(k::UI_OPT_PRESS_TOGGLE),
                    })
                    .width(160.0 + 2.0 * ui.spacing().button_padding.x) // ComboBox::width is the minimum of (total width - 2x padding); compensate to an actual width of 160
                    .show_ui(ui, |ui| {
                        ui.selectable_value(&mut mode, 0, t(k::UI_OPT_PRESS_TOGGLE));
                        ui.selectable_value(&mut mode, 1, t(k::UI_OPT_HOLD_PAUSE));
                        ui.selectable_value(&mut mode, 2, t(k::UI_OPT_HOLD_TALK));
                    });
                if mode != self.settings.hotkey_mode {
                    self.settings.hotkey_mode = mode;
                    self.persist_settings();
                    self.apply_hotkey();
                }
            });
            ui.separator();

            // Live level (meter, always visible -- it no longer "appears only after starting recognition") + speaking state + trigger statistics
            let mut snap = None;
            if let Some(c) = &self.ctx {
                snap = c.proc.lock().ok().map(|p| {
                    (
                        p.last_volume(),
                        p.is_speaking(),
                        p.keypress_count(),
                        p.rejected_nuclei(),
                        p.frames_since_audible().unwrap_or(0),
                    )
                });
            }
            let (vol, speaking, count, rejected, silent_frames) =
                snap.unwrap_or((0.0, false, 0, 0, 0i64));
            let enabled_n = self
                .cfg
                .get("commands")
                .and_then(|v| v.as_object())
                .map(|o| {
                    o.values()
                        .filter(|c| c.get("enabled").and_then(|x| x.as_bool()).unwrap_or(false))
                        .count()
                })
                .unwrap_or(0);
            ui.horizontal(|ui| {
                ui.label(t(k::UI_LBL_LIVE_LEVEL));
                let frac = (vol / 0.02).clamp(0.0, 1.0); // RMS 0.02 is roughly full scale
                ui.add(
                    egui::ProgressBar::new(frac)
                        .desired_width(320.0)
                        .text(if vol > 0.001 {
                            format!("{:.3}", vol)
                        } else {
                            String::new()
                        }),
                );
                ui.label(if speaking {
                    t(k::UI_LBL_SPEAKING)
                } else {
                    t(k::UI_LBL_MUTED)
                });
                if self.ctx.is_some() && silent_frames > 600 {
                    ui.colored_label(egui::Color32::from_rgb(230, 90, 90), t(k::MSG_WARN_NO_MIC));
                }
                ui.separator();
                ui.label(crate::lang::tf(
                    k::UI_LBL_SUMMARY,
                    &[
                        ("{n}", &count.to_string()),
                        ("{k}", &enabled_n.to_string()),
                        ("{r}", &rejected.to_string()),
                    ],
                ));
            });

            ui.separator();
            ui.label(t(k::UI_LBL_TRIGGER_LOG));
            let log_h = (ui.available_height() - HOME_FOOTER_H).max(120.0);
            egui::ScrollArea::vertical()
                .id_salt("home_log_scroll")
                .auto_shrink([false, false])
                .max_height(log_h)
                .stick_to_bottom(true)
                .show(ui, |ui| {
                    for row in &self.log_lines {
                        crate::ui::logs::draw_log_row(ui, row);
                    }
                });

            ui.with_layout(egui::Layout::bottom_up(egui::Align::LEFT), |ui| {
                ui.weak(t(k::UI_HINT_DISCLAIMER_KEYS));
                if crate::ui::FONT_MISSING.load(std::sync::atomic::Ordering::Relaxed) {
                    ui.colored_label(
                        egui::Color32::from_rgb(230, 90, 90),
                        t(k::MSG_WARN_FONT_MISSING),
                    );
                }
                let enabled_n = self
                    .cfg
                    .get("commands")
                    .and_then(|v| v.as_object())
                    .map(|o| {
                        o.values()
                            .filter(|c| c.get("enabled").and_then(|x| x.as_bool()).unwrap_or(false))
                            .count()
                    })
                    .unwrap_or(0);
                if enabled_n == 0 {
                    ui.colored_label(
                        egui::Color32::from_rgb(76, 175, 80),
                        t(k::UI_HINT_NO_ENABLED_CMDS),
                    );
                }
                // Recognition stopped / status text (top of the bottom area, visually just below the log)
                if !self.status.is_empty() {
                    ui.label(egui::RichText::new(&self.status).color(WARN_COLOR));
                }
            });
        });
    }

    /// Tab 2, Commands: group area (pinned on top, see pron_list_ui) + pronunciation list (checkbox / name / key / recording count);
    /// "Edit" opens a floating dialog (pron_open only toggles the dialog, it no longer swaps page content)
    pub(super) fn tab_pron(&mut self, ui: &mut egui::Ui) {
        content_column(ui, CONTENT_W, |ui| {
            self.pron_list_ui(ui);
        });
    }

    fn pron_list_ui(&mut self, ui: &mut egui::Ui) {
        enum Action {
            Toggle(String, bool),
            Edit(String),
            SetKey(String, String),
            OpenMacroEditor(String), // The table's macro action opens the macro editor directly
            DeleteCommand(String),
            ApplyGroup(String),
            OverwriteGroup(String),
            DeleteGroup(String),
            RenameGroup,
            GenProtoFull,
            AddCommand,
            ToggleAll(bool), // Header select-all: one log entry per save (unlike the per-row toggle)
            ToggleMulti(String, bool), // Multi-pronunciation checkbox: persists multi_nuclei
        }
        let mut actions: Vec<Action> = Vec::new();

        // -- Group area (pinned; simplified interaction: a checkbox applies and saves immediately, a dropdown applies on selection) --
        // Active group status line
        match (&self.active_group, self.group_dirty()) {
            (Some(name), Some(true)) => {
                ui.colored_label(
                    WARN_COLOR,
                    crate::tfmt!(k::UI_LBL_ACTIVE_GROUP_UNSAVED, name),
                );
            }
            (Some(name), _) => {
                ui.colored_label(
                    egui::Color32::from_rgb(76, 175, 80),
                    crate::tfmt!(k::UI_LBL_ACTIVE_GROUP, name),
                );
            }
            (None, _) => {
                ui.weak(t(k::UI_LBL_ACTIVE_GROUP_NONE));
            }
        }
        // Group row: the dropdown **applies on selection** (switches every pronunciation's enabled state to that group); overwrite-save,
        // rename and delete (double-click confirmation) keep their semantics
        ui.horizontal(|ui| {
            ui.label(t(k::UI_LBL_GROUP));
            let names: Vec<String> = self.groups.keys().cloned().collect();
            let prev_sel = self.group_sel.clone();
            let sel_text = if self.group_sel.is_empty() {
                t(k::UI_OPT_NO_GROUP)
            } else {
                self.group_sel.clone()
            };
            egui::ComboBox::from_id_salt("group_select")
                .selected_text(sel_text)
                .width(160.0 + 2.0 * ui.spacing().button_padding.x) // ComboBox::width is the minimum of (total width - 2x padding); compensate to an actual width of 160
                .show_ui(ui, |ui| {
                    for n in &names {
                        ui.selectable_value(&mut self.group_sel, n.clone(), n);
                    }
                });
            if self.group_sel != prev_sel && !self.group_sel.is_empty() {
                let dirty = matches!(self.group_dirty(), Some(true))
                    && Some(self.group_sel.clone()) == self.active_group;
                let armed = self.group_switch_armed.as_deref() == Some(self.group_sel.as_str());
                if dirty && !armed {
                    // First click: arms the confirmation for that target (status bar hint), no action yet
                    self.group_switch_armed = Some(self.group_sel.clone());
                    self.status = t(k::MSG_CONFIRM_SWITCH_GROUP);
                } else {
                    self.group_switch_armed = None;
                    actions.push(Action::ApplyGroup(self.group_sel.clone()));
                }
            } else if self.group_sel != prev_sel {
                self.group_switch_armed = None;
            }
            if ui.button(t(k::UI_BTN_OVERWRITE_SAVE)).clicked() && !self.group_sel.is_empty() {
                actions.push(Action::OverwriteGroup(self.group_sel.clone()));
            }
            if ui.button(t(k::UI_BTN_RENAME)).clicked() && !self.group_sel.is_empty() {
                actions.push(Action::RenameGroup);
            }
            // Double-click delete -- the first click turns the button red ("confirm delete"), the second executes (valid for 5 s)
            let armed_group = !self.group_sel.is_empty()
                && self.armed_is(&ArmedDelete::Group(self.group_sel.clone()));
            let del_btn = text_button(
                ui,
                if armed_group {
                    egui::RichText::new(t(k::UI_BTN_CONFIRM_DELETE)).color(WARN_COLOR)
                } else {
                    egui::RichText::new(t(k::UI_BTN_DELETE_GROUP))
                },
            );
            if del_btn.clicked() && !self.group_sel.is_empty() {
                actions.push(Action::DeleteGroup(self.group_sel.clone()));
            }
            if ui.button(t(k::UI_BTN_NEW_GROUP)).clicked() {
                self.group_new_open = true;
                self.new_group_name.clear();
            }
        });
        ui.separator();

        // -- Toolbar ("+ New pronunciation" is the first button in the list) --
        ui.horizontal(|ui| {
            // -- New-pronunciation entry (dialog: name + optional key -> opens the edit dialog afterwards for recording) --
            if ui
                .add(egui::Button::new(t(k::UI_BTN_NEW_CMD)).min_size(egui::vec2(110.0, 26.0)))
                .clicked()
            {
                actions.push(Action::AddCommand);
            }
            let busy = self.proto_busy() || self.capture_job != CaptureJob::None || self.warming;
            let gen_full = matches!(
                &self.proto_job,
                Some(ProtoJob {
                    kind: ProtoKind::Full,
                    ..
                })
            );
            let n_cmds = self.rec_counts.len();
            let n_recs: usize = self.rec_counts.values().sum();
            let resp = ui.add_enabled(!busy, egui::Button::new(t(k::UI_BTN_REBUILD_ALL)));
            if resp.clicked() {
                if n_cmds > 0 {
                    self.confirm = Some(Confirm::RegenAll {
                        cmds: n_cmds,
                        recs: n_recs,
                    });
                } else {
                    actions.push(Action::GenProtoFull);
                }
            }
            if gen_full {
                ui.colored_label(WARN_COLOR, t(k::MSG_TEMPLATE_BUILDING_SHORT));
            } else if self.proto_busy() {
                ui.colored_label(WARN_COLOR, t(k::MSG_TEMPLATE_BUILDING_SHORT));
            } else if self.capture_job != CaptureJob::None || self.warming {
                ui.weak(t(k::MSG_ENGINE_WARMING_NO_BUILD));
            }
        });

        let mut rows: Vec<CmdRow> = Vec::new();
        if let Some(cmds) = self.cfg.get("commands").and_then(|v| v.as_object()) {
            for (name, c) in cmds {
                let display = c
                    .get("phrase")
                    .and_then(|v| v.as_str())
                    .filter(|s| !s.is_empty())
                    .unwrap_or(name);
                let key = c
                    .get("key")
                    .and_then(|v| v.as_str())
                    .unwrap_or("-")
                    .to_string();
                rows.push(CmdRow {
                    name: name.clone(),
                    display: display.to_string(),
                    key,
                    enabled: c.get("enabled").and_then(|v| v.as_bool()).unwrap_or(true),
                    recs: *self.rec_counts.get(name).unwrap_or(&0),
                    mode: c
                        .get("mode")
                        .and_then(|v| v.as_str())
                        .unwrap_or("tap")
                        .to_string(),
                    multi: c
                        .get("multi_nuclei")
                        .and_then(|v| v.as_bool())
                        .unwrap_or_else(|| self.cmd_multi_auto.get(name).copied().unwrap_or(false)),
                });
            }
        }
        match self.cmd_sort {
            Some((col, desc)) => sort_cmd_rows(&mut rows, col, desc),
            None => rows.sort_by(|a, b| a.name.cmp(&b.name)),
        }
        ui.horizontal(|ui| {
            ui.label(t(k::UI_TAB_CMDS));
            count_badge(ui, rows.len());
        });
        let list_h = (ui.available_height() - 8.0).max(60.0);
        let rh = ui.spacing().interact_size.y;
        ui.scope(|ui| {
            // With resizable on, egui_extras draws full-height column separators using `noninteractive.bg_stroke`;
            // clearing it shows them only on hover or drag (same treatment as the edit-pronunciation table).
            ui.style_mut().visuals.widgets.noninteractive.bg_stroke = egui::Stroke::NONE;
            egui_extras::TableBuilder::new(ui)
                .id_salt("pron_table")
                .striped(true)
                .resizable(true)
                .min_scrolled_height(60.0)
                .max_scroll_height(list_h)
                .cell_layout(egui::Layout::left_to_right(egui::Align::Center))
                // Enabled / name / trigger mode / mapped key / recording count / actions (the last column takes the remaining width)
                .column(egui_extras::Column::initial(54.0).at_least(44.0))
                .column(
                    egui_extras::Column::initial(110.0)
                        .at_least(70.0)
                        .clip(true),
                )
                .column(egui_extras::Column::initial(96.0).at_least(70.0))
                .column(egui_extras::Column::initial(120.0).at_least(80.0))
                .column(egui_extras::Column::initial(70.0).at_least(50.0))
                .column(egui_extras::Column::initial(64.0).at_least(52.0))
                .column(egui_extras::Column::remainder().at_least(132.0))
                .header(rh, |mut header| {
                    // Header select-all checkbox: checked = select all, unchecked = select none (a mixed state renders unchecked; one click selects all)
                    let all_on = rows.iter().all(|r| r.enabled);
                    header.col(|ui| {
                        let mut header_all = all_on;
                        if ui.checkbox(&mut header_all, t(k::UI_TBL_ENABLED)).changed() {
                            actions.push(Action::ToggleAll(header_all));
                        }
                    });
                    header.col(|ui| {
                        sort_header(ui, &t(k::UI_TBL_NAME), CmdCol::Name, &mut self.cmd_sort);
                    });
                    header.col(|ui| {
                        sort_header(
                            ui,
                            &t(k::UI_TBL_TRIGGER_MODE),
                            CmdCol::Mode,
                            &mut self.cmd_sort,
                        );
                    });
                    header.col(|ui| {
                        sort_header(
                            ui,
                            &t(k::UI_TBL_KEY_MAPPING),
                            CmdCol::Key,
                            &mut self.cmd_sort,
                        );
                    });
                    header.col(|ui| {
                        sort_header(
                            ui,
                            &t(k::UI_TBL_REC_COUNT),
                            CmdCol::Recs,
                            &mut self.cmd_sort,
                        );
                    });
                    header.col(|ui| {
                        ui.weak(t(k::UI_TBL_MULTI))
                            .on_hover_text(t(k::UI_HINT_MULTI));
                    });
                    header.col(|ui| {
                        ui.weak(t(k::UI_TBL_ACTIONS));
                    });
                })
                .body(|mut body| {
                    for r in &rows {
                        body.row(rh, |mut row| {
                            row.col(|ui| {
                                let mut on = r.enabled;
                                if ui.checkbox(&mut on, "").changed() && on != r.enabled {
                                    actions.push(Action::Toggle(r.name.clone(), on));
                                }
                            });
                            row.col(|ui| {
                                ui.add(egui::Label::new(&r.display).truncate());
                            });
                            row.col(|ui| {
                                let mode_text = mode_label(&r.mode);
                                if r.mode == "macro" {
                                    if ui.button(t(k::UI_BTN_MACRO_EDIT)).clicked() {
                                        actions.push(Action::OpenMacroEditor(r.name.clone()));
                                    }
                                } else {
                                    ui.weak(mode_text);
                                }
                            });
                            row.col(|ui| {
                                if r.mode == "macro" {
                                    ui.weak("—");
                                } else {
                                    let cfg_key = r.key.clone();
                                    let buf = self
                                        .key_edit_bufs
                                        .entry(r.name.clone())
                                        .or_insert_with(|| cfg_key.clone());
                                    let kresp = ui.add(
                                        egui::TextEdit::singleline(buf)
                                            .id_salt(format!("pron_key_{}", r.name))
                                            .desired_width(70.0),
                                    );
                                    let committed = (kresp.lost_focus()
                                        || ui.input(|i| i.key_pressed(egui::Key::Enter)))
                                        && buf.trim() != r.key
                                        && !buf.trim().is_empty();
                                    if committed {
                                        let v = buf.trim().to_uppercase();
                                        *buf = v.clone();
                                        actions.push(Action::SetKey(r.name.clone(), v));
                                    }
                                }
                            });
                            row.col(|ui| {
                                if r.recs == 0 {
                                    // 0 recordings = not yet recognisable; the red highlight itself is the hint
                                    ui.colored_label(WARN_COLOR, "0");
                                } else {
                                    ui.label(format!("{}", r.recs));
                                }
                            });
                            row.col(|ui| {
                                let mut m = r.multi;
                                if ui.checkbox(&mut m, "").changed() && m != r.multi {
                                    actions.push(Action::ToggleMulti(r.name.clone(), m));
                                }
                            });
                            row.col(|ui| {
                                if ui.button(t(k::UI_BTN_EDIT)).clicked() {
                                    actions.push(Action::Edit(r.name.clone()));
                                }
                                // Double-click delete -- the first click turns red ("confirm delete"), the second executes
                                let armed_cmd =
                                    self.armed_is(&ArmedDelete::Command(r.name.clone()));
                                let del_btn = text_button(
                                    ui,
                                    if armed_cmd {
                                        egui::RichText::new(t(k::UI_BTN_CONFIRM_DELETE))
                                            .color(WARN_COLOR)
                                    } else {
                                        egui::RichText::new(t(k::UI_BTN_DELETE))
                                    },
                                );
                                if del_btn.clicked() {
                                    actions.push(Action::DeleteCommand(r.name.clone()));
                                }
                            });
                        });
                    }
                });
        });

        // Perform the action (outside the closure, to avoid borrow conflicts)
        for a in actions {
            match a {
                Action::Toggle(name, on) => {
                    self.set_cmd_field(&name, "enabled", serde_json::Value::Bool(on));
                    self.persist_cfg();
                    self.log(&crate::tfmt!("pronunciation {name} enabled -> {on}", name, on));
                }
                Action::ToggleMulti(name, on) => {
                    self.set_cmd_field(&name, "multi_nuclei", serde_json::Value::Bool(on));
                    self.persist_cfg();
                    self.log(&crate::tfmt!(
                        "pronunciation {name} multi-pronunciation -> {on} (applies the next time recognition starts)",
                        name,
                        on
                    ));
                }
                Action::ToggleAll(on) => {
                    let names: Vec<String> = self
                        .cfg
                        .get("commands")
                        .and_then(|v| v.as_object())
                        .map(|o| o.keys().cloned().collect())
                        .unwrap_or_default();
                    let mut changed = 0usize;
                    for name in names {
                        let cur = self
                            .cfg
                            .get("commands")
                            .and_then(|c| c.get(&name))
                            .and_then(|c| c.get("enabled"))
                            .and_then(|v| v.as_bool())
                            .unwrap_or(true);
                        if cur != on {
                            self.set_cmd_field(&name, "enabled", serde_json::Value::Bool(on));
                            changed += 1;
                        }
                    }
                    if changed > 0 {
                        self.persist_cfg();
                        self.log(&crate::tfmt!(
                            "all pronunciations enabled -> {on} ({changed} changes)",
                            on,
                            changed
                        ));
                    }
                }
                Action::Edit(name) => {
                    if self.pron_open.is_some() && self.pron_open.as_deref() != Some(name.as_str())
                    {
                        self.close_edit_popup();
                    }
                    // Entering from the list always starts in edit mode (the first row shows the pronunciation name rather than a name field)
                    self.pron_new = false;
                    self.pron_new_focus = false;
                    self.add_phrase.clear();
                    self.key_edit = self
                        .cfg
                        .get("commands")
                        .and_then(|v| v.get(&name))
                        .and_then(|c| c.get("key"))
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    self.refresh_rec_list(&name);
                    self.refresh_proto_cover();
                    self.pron_open = Some(name);
                }
                Action::DeleteCommand(name) => {
                    // Double-click delete -- the first click arms it (button turns red), the second executes
                    let kind = ArmedDelete::Command(name.clone());
                    if self.armed_is(&kind) {
                        self.armed_delete = None;
                        self.delete_command(&name);
                    } else {
                        self.armed_delete = Some((kind, std::time::Instant::now()));
                    }
                }
                Action::SetKey(name, key) => {
                    self.set_cmd_field(&name, "key", serde_json::Value::String(key.clone()));
                    self.persist_cfg();
                    self.log(&crate::tfmt!("pronunciation {name} key -> {key}", name, key));
                }
                Action::OpenMacroEditor(name) => {
                    self.macro_editor_open = Some(name);
                }
                Action::ApplyGroup(name) => self.apply_group(&name),
                Action::OverwriteGroup(name) => self.overwrite_group(&name),
                Action::DeleteGroup(name) => {
                    // Double-click delete (as above)
                    let kind = ArmedDelete::Group(name.clone());
                    if self.armed_is(&kind) {
                        self.armed_delete = None;
                        self.delete_group(&name);
                    } else {
                        self.armed_delete = Some((kind, std::time::Instant::now()));
                    }
                }
                Action::RenameGroup => {
                    self.rename_buf = self.group_sel.clone();
                    self.rename = Some(Rename::Group(self.group_sel.clone()));
                }
                Action::GenProtoFull => self.start_proto_full(),
                Action::AddCommand => {
                    self.start_new_command();
                }
            }
        }
    }

    pub(super) fn tab_settings(&mut self, ui: &mut egui::Ui) {
        content_column(ui, CONTENT_W, |ui| {
            let status_h = if self.status.is_empty() {
                0.0
            } else {
                ui.text_style_height(&egui::TextStyle::Body) + 8.0
            };
            egui::ScrollArea::vertical()
                .id_salt("settings_scroll")
                .auto_shrink([false, false])
                .max_height((ui.available_height() - status_h).max(60.0))
                .show(ui, |ui| {
                    ui.add_space(4.0);
                    settings_grid("set_grid_all", ui, |ui| {
                        // -- Audio --
                        ui.label(egui::RichText::new(t(k::UI_HDR_AUDIO)).strong());
                        ui.end_row();
                        // Microphone (renamed from "input device"; empty = system auto-select, takes effect the next time recognition starts)
                        ui.label(t(k::UI_LBL_MICROPHONE));
                        ui.horizontal(|ui| {
                            let sel = if self.selected_device.is_empty() {
                                t(k::UI_OPT_SYSTEM_DEFAULT)
                            } else {
                                self.selected_device.clone()
                            };
                            egui::ComboBox::from_id_salt("device_select")
                                .selected_text(sel)
                                .width(160.0 + 2.0 * ui.spacing().button_padding.x) // ComboBox::width is the minimum of (total width - 2x padding); compensate to an actual width of 160
                                .show_ui(ui, |ui| {
                                    ui.selectable_value(
                                        &mut self.selected_device,
                                        String::new(),
                                        t(k::UI_OPT_SYSTEM_DEFAULT),
                                    );
                                    for d in &self.devices {
                                        ui.selectable_value(
                                            &mut self.selected_device,
                                            d.clone(),
                                            d,
                                        );
                                    }
                                });
                            if ui.button(t(k::UI_BTN_REFRESH_DEVICES)).clicked() {
                                self.devices = live::list_input_devices();
                            }
                        });
                        ui.end_row();
                        if self.selected_device != self.settings.device {
                            self.settings.device = self.selected_device.clone();
                            self.persist_settings();
                            // Changing the microphone re-probes system AEC availability (the verdict is **per device**; skipping the re-probe would
                            // attribute the "system" mode's availability to the wrong device)
                            self.spawn_aec_probe();
                        }

                        // -- Recognition --
                        ui.vertical(|ui| {
                            ui.add_space(8.0);
                            ui.label(egui::RichText::new(t(k::UI_HDR_RECOGNITION)).strong());
                            ui.add_space(2.0);
                        });
                        ui.end_row();
                        ui.label(t(k::UI_LBL_MIN_CONFIDENCE));
                        ui.horizontal(|ui| {
                            let resp = ui.add(
                                egui::TextEdit::singleline(&mut self.min_conf_buf)
                                    .hint_text("0.6")
                                    .desired_width(160.0),
                            );
                            let entered =
                                resp.lost_focus() || ui.input(|i| i.key_pressed(egui::Key::Enter));
                            if entered {
                                let entered_text = self.min_conf_buf.trim().to_string();
                                let parsed = if entered_text.is_empty() {
                                    Ok(0.6)
                                } else {
                                    entered_text.parse::<f64>()
                                };
                                match parsed {
                                    Ok(v) if (0.0..=0.9).contains(&v) => {
                                        self.settings.v2_min_conf = v;
                                        self.persist_settings();
                                        self.apply_live_tuning(crate::udkws_v2::EngineOverrides {
                                            min_conf: Some(v),
                                            ..Default::default()
                                        });
                                        self.status =
                                            crate::tfmt!(k::MSG_MIN_CONFIDENCE_APPLIED, v);
                                    }
                                    _ => {
                                        self.min_conf_buf =
                                            format!("{}", self.settings.v2_min_conf);
                                        self.status = t(k::UI_HINT_MIN_CONFIDENCE_RANGE);
                                    }
                                }
                            }
                            ui.label("ⓘ").on_hover_text(t(k::UI_HINT_MIN_CONFIDENCE));
                        });
                        ui.end_row();
                        ui.label(t(k::UI_LBL_REFRACTORY));
                        ui.horizontal(|ui| {
                            let resp = ui.add(
                                egui::TextEdit::singleline(&mut self.refr_buf)
                                    .desired_width(160.0),
                            );
                            let entered =
                                resp.lost_focus() || ui.input(|i| i.key_pressed(egui::Key::Enter));
                            if entered {
                                let text = self.refr_buf.trim().to_string();
                                let parsed = if text.is_empty() {
                                    Ok(0.15)
                                } else {
                                    text.parse::<f64>()
                                };
                                match parsed {
                                    Ok(v) if (0.0..=2.0).contains(&v) => {
                                        self.settings.refr_s = v;
                                        self.refr_buf = format!("{}", v);
                                        self.persist_settings();
                                        self.apply_live_tuning(crate::udkws_v2::EngineOverrides {
                                            refr_s: Some(v),
                                            ..Default::default()
                                        });
                                        self.status = crate::tfmt!(k::MSG_REFRACTORY_APPLIED, v);
                                    }
                                    _ => {
                                        self.refr_buf = format!("{}", self.settings.refr_s);
                                        self.status = t(k::UI_HINT_REFRACTORY);
                                    }
                                }
                            }
                        });
                        ui.end_row();
                        ui.label(t(k::UI_LBL_GATE));
                        ui.horizontal(|ui| {
                            let resp = ui.add(
                                egui::TextEdit::singleline(&mut self.act_gate_buf)
                                    .desired_width(160.0),
                            );
                            let entered =
                                resp.lost_focus() || ui.input(|i| i.key_pressed(egui::Key::Enter));
                            if entered {
                                let text = self.act_gate_buf.trim().to_string();
                                let parsed = if text.is_empty() {
                                    Ok(-50.0)
                                } else {
                                    text.parse::<f64>()
                                };
                                match parsed {
                                    Ok(v) if (-75.0..=-15.0).contains(&v) => {
                                        self.settings.act_gate_db = v;
                                        self.act_gate_buf = format!("{}", v);
                                        self.persist_settings();
                                        self.apply_live_tuning(crate::udkws_v2::EngineOverrides {
                                            act_db: Some(v),
                                            ..Default::default()
                                        });
                                        self.status = crate::tfmt!(k::MSG_GATE_APPLIED, v);
                                    }
                                    _ => {
                                        self.act_gate_buf =
                                            format!("{}", self.settings.act_gate_db);
                                        self.status = t(k::UI_HINT_GATE_RANGE);
                                    }
                                }
                            }
                        });
                        ui.end_row();
                        ui.label(t(k::UI_CHK_AUTO_GATE));
                        ui.horizontal(|ui| {
                            ui.add_space(160.0 - ui.spacing().icon_width);
                            let mut auto = self.settings.act_gate_auto;
                            if ui.checkbox(&mut auto, "").changed() {
                                self.settings.act_gate_auto = auto;
                                self.persist_settings();
                                self.apply_live_tuning(crate::udkws_v2::EngineOverrides {
                                    auto_gate: Some(auto),
                                    ..Default::default()
                                });
                                self.status = if self.ctx.is_some() {
                                    t(k::MSG_SAVED_LIVE)
                                } else {
                                    t(k::MSG_SAVED_NEXT_START)
                                };
                            }
                            if auto {
                                ui.weak(t(k::UI_HINT_GATE_MANUAL_START));
                            }
                        });
                        ui.end_row();
                        ui.label(t(k::UI_CHK_NOISE_MODEL))
                            .on_hover_text(t(k::UI_HINT_NOISE_MODEL));
                        ui.horizontal(|ui| {
                            ui.add_space(160.0 - ui.spacing().icon_width);
                            let mut n = self.settings.noise_model;
                            if ui.checkbox(&mut n, "").changed() {
                                self.confirm = Some(Confirm::NoiseModel { enable: n });
                            }
                        });
                        ui.end_row();
                        // "Optimise recognition" has moved to the "Experimental" section at the end of the page.

                        // -- General --
                        ui.vertical(|ui| {
                            ui.add_space(8.0);
                            ui.label(egui::RichText::new(t(k::UI_HDR_GENERAL)).strong());
                            ui.add_space(2.0);
                        });
                        ui.end_row();
                        ui.label(t(k::UI_LBL_LANGUAGE));
                        ui.horizontal(|ui| {
                            let lang_dir = crate::exe_dir()
                                .map(|d| d.join("lang"))
                                .unwrap_or_else(|| std::path::PathBuf::from("lang"));
                            let langs = crate::lang::available(&lang_dir);
                            let mut sel = self.settings.language.clone();
                            egui::ComboBox::from_id_salt("language_select")
                                .selected_text(sel.clone())
                                .width(160.0 + 2.0 * ui.spacing().button_padding.x) // ComboBox::width is the minimum of (total width - 2x padding); compensate to an actual width of 160
                                .show_ui(ui, |ui| {
                                    for l in &langs {
                                        ui.selectable_value(&mut sel, l.clone(), l);
                                    }
                                });
                            if sel != self.settings.language {
                                self.settings.language = sel.clone();
                                crate::lang::set_language(&sel, &lang_dir);
                                self.min_conf_buf = format!("{}", self.settings.v2_min_conf);
                                self.persist_settings();
                            }
                        });
                        ui.end_row();

                        ui.label(t(k::UI_BTN_CHECK_UPDATE));
                        ui.horizontal(|ui| {
                            ui.add_space(160.0 - ui.spacing().icon_width);
                            let mut on = self.settings.update_check_enabled;
                            if ui.checkbox(&mut on, "").changed() {
                                self.settings.update_check_enabled = on;
                                self.persist_settings();
                            }
                        });
                        ui.end_row();

                        // -- Keys --
                        ui.vertical(|ui| {
                            ui.add_space(8.0);
                            ui.label(egui::RichText::new(t(k::UI_HDR_KEYS)).strong());
                            ui.add_space(2.0);
                        });
                        ui.end_row();
                        ui.label(t(k::UI_LBL_KEY_MODE));
                        ui.horizontal(|ui| {
                            let mut v = self.settings.input_mode;
                            let label = match v {
                                crate::keys::MODE_STANDARD => t(k::UI_OPT_STANDARD),
                                crate::keys::MODE_BACKGROUND => t(k::UI_OPT_BACKGROUND),
                                crate::keys::MODE_SCANCODE => t(k::UI_OPT_COMPAT1_SCANCODE),
                                crate::keys::MODE_VIRTUALKEY => t(k::UI_OPT_COMPAT2_VK),
                                _ => t(k::UI_OPT_COMPAT3_LEGACY),
                            };
                            egui::ComboBox::from_id_salt("input_mode")
                                .selected_text(label)
                                .width(160.0 + 2.0 * ui.spacing().button_padding.x) // ComboBox::width is the minimum of (total width - 2x padding); compensate to an actual width of 160
                                .show_ui(ui, |ui| {
                                    ui.selectable_value(
                                        &mut v,
                                        crate::keys::MODE_STANDARD,
                                        t(k::UI_OPT_STANDARD),
                                    );
                                    ui.selectable_value(
                                        &mut v,
                                        crate::keys::MODE_BACKGROUND,
                                        t(k::UI_OPT_BACKGROUND),
                                    );
                                    ui.selectable_value(
                                        &mut v,
                                        crate::keys::MODE_SCANCODE,
                                        t(k::UI_OPT_COMPAT1_SCANCODE),
                                    );
                                    ui.selectable_value(
                                        &mut v,
                                        crate::keys::MODE_VIRTUALKEY,
                                        t(k::UI_OPT_COMPAT2_VK),
                                    );
                                    ui.selectable_value(
                                        &mut v,
                                        crate::keys::MODE_LEGACY,
                                        t(k::UI_OPT_COMPAT3_LEGACY),
                                    );
                                });
                            if v != self.settings.input_mode {
                                self.settings.input_mode = v;
                                crate::keys::set_input_mode(v);
                                self.persist_settings();
                                self.status = t(k::MSG_KEY_MODE_SWITCHED);
                            }
                        });
                        ui.end_row();

                        ui.label(t(k::UI_LBL_GAME_WINDOW));
                        ui.horizontal(|ui| {
                            let sel = if self.settings.target_window.is_empty() {
                                t(k::UI_OPT_ANY_WINDOW)
                            } else {
                                self.settings.target_window.clone()
                            };
                            egui::ComboBox::from_id_salt("target_window")
                                .selected_text(sel)
                                .width(160.0 + 2.0 * ui.spacing().button_padding.x) // ComboBox::width is the minimum of (total width - 2x padding); compensate to an actual width of 160
                                .show_ui(ui, |ui| {
                                    ui.selectable_value(
                                        &mut self.settings.target_window,
                                        String::new(),
                                        t(k::UI_OPT_ANY_WINDOW),
                                    );
                                    for t in &self.window_list {
                                        ui.selectable_value(
                                            &mut self.settings.target_window,
                                            t.clone(),
                                            t,
                                        );
                                    }
                                });
                            if ui.button(t(k::UI_BTN_REFRESH_WINDOWS)).clicked() {
                                self.window_list = crate::keys::list_window_titles();
                            }
                            if self.settings.target_window != self.last_target_window {
                                crate::keys::set_target_window(&self.settings.target_window);
                                self.last_target_window = self.settings.target_window.clone();
                                self.persist_settings();
                            }
                        });
                        ui.end_row();
                        // The pause hotkey and the voice toggle both live on the home page ("key toggle" and
                        // "voice toggle"); the forced-modifier requirement is gone (single keys are allowed). Not repeated here.

                        ui.vertical(|ui| {
                            ui.add_space(8.0);
                            ui.label(egui::RichText::new(t(k::UI_HDR_EXPERIMENTAL)).strong());
                            ui.add_space(2.0);
                        });
                        ui.end_row();
                        // Optimise recognition: per-frame cross-dimension mean removal in the front-end log-mel (dropping c0),
                        // cancelling the additive constant that gain and distance introduce in the log domain.
                        // ⚠ Both enabling and disabling require regenerating the templates: the cache version includes this switch,
                        // so the two states never share a cache and a full rebuild happens at load time in either direction.
                        // The rebuild is deliberately not done on the spot: frame_mean_norm() is a OnceLock with no setter, so the
                        // front end cannot change within a process -- recomputing here would only reuse the old one (wasted work).
                        ui.label(t(k::UI_CHK_BOOST_RECOGNITION));
                        ui.horizontal(|ui| {
                            ui.add_space(160.0 - ui.spacing().icon_width);
                            // The checkbox renders from a local copy: cancelling the confirmation rolls the settings value back on the next frame.
                            let mut b = self.settings.frame_mean_norm;
                            if ui.checkbox(&mut b, "").changed() {
                                self.confirm = Some(Confirm::BoostRecognition { enable: b });
                            }
                        });
                        ui.end_row();
                        // Response speed: identity-window right lookahead from 0 to 0.13 s, labelled only at the two ends (fast / slow).
                        ui.label(t(k::UI_LBL_RESPONSIVENESS));
                        ui.horizontal(|ui| {
                            let font = egui::FontId::proportional(13.0);
                            let txt_fast = t(k::UI_LBL_RESP_FAST);
                            let txt_slow = t(k::UI_LBL_RESP_SLOW);
                            let gap = 12.0_f32;
                            let (tw, th) = (160.0_f32, 20.0_f32);
                            let hpad = 5.0_f32;
                            let (rect, resp) = ui.allocate_exact_size(
                                egui::vec2(tw, th),
                                egui::Sense::click_and_drag(),
                            );
                            let (tl, tr) = (rect.left(), rect.right());
                            let x0 = tl + hpad;
                            let x1 = tr - hpad;
                            let mid = rect.center().y;
                            let v = (self.settings.ident_fwd_sec / 0.13).clamp(0.0, 1.0) as f32;
                            let hx = x0 + v * (x1 - x0);
                            {
                                let vis = ui.visuals();
                                let p = ui.painter();
                                let track = |l: f32, r: f32| {
                                    egui::Rect::from_min_max(
                                        egui::pos2(l, mid - 2.0),
                                        egui::pos2(r, mid + 2.0),
                                    )
                                };
                                p.rect_filled(
                                    track(tl, tr),
                                    egui::CornerRadius::same(2),
                                    vis.widgets.inactive.bg_fill,
                                );
                                p.rect_filled(
                                    track(tl, hx),
                                    egui::CornerRadius::same(2),
                                    vis.selection.bg_fill,
                                );
                                let hw = if resp.hovered() || resp.dragged() {
                                    10.0
                                } else {
                                    8.0
                                };
                                p.rect_filled(
                                    egui::Rect::from_center_size(
                                        egui::pos2(hx, mid),
                                        egui::vec2(hw, 16.0),
                                    ),
                                    egui::CornerRadius::same(2),
                                    vis.widgets.active.bg_fill,
                                );
                                p.text(
                                    egui::pos2(rect.left() - gap, mid),
                                    egui::Align2::RIGHT_CENTER,
                                    txt_fast.clone(),
                                    font.clone(),
                                    vis.weak_text_color(),
                                );
                                p.text(
                                    egui::pos2(rect.right() + gap, mid),
                                    egui::Align2::LEFT_CENTER,
                                    txt_slow.clone(),
                                    font.clone(),
                                    vis.weak_text_color(),
                                );
                            }
                            if resp.dragged() || resp.clicked() {
                                if let Some(pos) = resp.interact_pointer_pos() {
                                    let f = ((pos.x - x0) / (x1 - x0)).clamp(0.0, 1.0) as f64;
                                    self.settings.ident_fwd_sec =
                                        ((f * 0.13 * 100.0).round() / 100.0).clamp(0.0, 0.13);
                                }
                            }
                            if resp.drag_stopped() || resp.clicked() {
                                self.persist_settings();
                                self.status = t(k::MSG_SAVED_RESTART_REBUILD);
                            }
                            resp.on_hover_text(t(k::UI_HINT_RESPONSIVENESS));
                        });
                        ui.end_row();
                        ui.label(t(k::UI_CHK_NONVOICE_REJECT));
                        ui.horizontal(|ui| {
                            ui.add_space(160.0 - ui.spacing().icon_width);
                            let mut vad = self.settings.vad_gate;
                            if ui.checkbox(&mut vad, "").changed() {
                                self.settings.vad_gate = vad;
                                self.persist_settings();
                                self.status = t(k::MSG_SAVED_RESTART);
                            }
                        });
                        ui.end_row();
                        ui.label(t(k::UI_CHK_SPEAKER_VERIFY));
                        ui.horizontal(|ui| {
                            ui.add_space(160.0 - ui.spacing().icon_width);
                            let available = crate::sv::model_available();
                            let mut sv = self.settings.sv_gate;
                            if ui
                                .add_enabled(available, egui::Checkbox::new(&mut sv, ""))
                                .changed()
                            {
                                self.settings.sv_gate = sv;
                                self.persist_settings();
                                self.status = t(k::MSG_SAVED_RESTART);
                            }
                            if !available {
                                ui.weak(t(k::UI_HINT_SV_MODEL_MISSING));
                            }
                        });
                        ui.end_row();

                        ui.label(t(k::UI_LBL_AEC));
                        ui.horizontal(|ui| {
                            let mut mode = self.settings.aec_mode;
                            let sys_ok = self
                                .aec_probe
                                .as_ref()
                                .map(|p| p.aec_selectable())
                                .unwrap_or(true);
                            egui::ComboBox::from_id_salt("aec_mode_select")
                                .selected_text(match mode {
                                    1 => t(k::UI_OPT_SYSTEM),
                                    3 => t(k::UI_OPT_SONORA),
                                    _ => t(k::UI_OPT_OFF),
                                })
                                .width(160.0 + 2.0 * ui.spacing().button_padding.x) // ComboBox::width is the minimum of (total width - 2x padding); compensate to an actual width of 160
                                .show_ui(ui, |ui| {
                                    ui.selectable_value(&mut mode, 0, t(k::UI_OPT_OFF));
                                    let lab = ui.add_enabled(
                                        sys_ok,
                                        egui::SelectableLabel::new(mode == 1, t(k::UI_OPT_SYSTEM)),
                                    );
                                    if lab.clicked() {
                                        mode = 1;
                                    }
                                    ui.selectable_value(&mut mode, 3, t(k::UI_OPT_SONORA));
                                });
                            if mode != self.settings.aec_mode {
                                self.settings.aec_mode = mode;
                                self.persist_settings();
                                self.status = t(k::MSG_AEC_SAVED);
                            }
                            if let Some(crate::comm::AecVerdict::Unknown) =
                                self.aec_probe.as_ref().map(|p| p.aec_verdict())
                            {
                                ui.weak(t(k::MSG_AEC_UNKNOWN));
                            }
                        });
                        ui.end_row();
                    });

                    // -- Bottom helper (clearly separated from the settings groups) --
                    ui.add_space(16.0);
                    if ui.button(t(k::UI_TTL_ABOUT)).clicked() {
                        self.about_open = true;
                    }
                });
            if !self.status.is_empty() {
                ui.colored_label(WARN_COLOR, &self.status);
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cmd_rows_sort_by_name_and_numeric_recs() {
        let mk = |id: &str, disp: &str, recs: usize| CmdRow {
            name: id.into(),
            display: disp.into(),
            key: "W".into(),
            enabled: true,
            recs,
            mode: "tap".into(),
            multi: false,
        };
        let ids = |r: &[CmdRow]| r.iter().map(|x| x.name.clone()).collect::<Vec<_>>();
        let mut rows = vec![
            mk("cmd3", "Banana", 2),
            mk("cmd1", "apple", 10),
            mk("cmd2", "Cherry", 9),
        ];
        sort_cmd_rows(&mut rows, CmdCol::Name, false);
        assert_eq!(
            ids(&rows),
            vec!["cmd1", "cmd3", "cmd2"],
            "name ascending, case-insensitive"
        );
        sort_cmd_rows(&mut rows, CmdCol::Recs, false);
        assert_eq!(
            ids(&rows),
            vec!["cmd3", "cmd2", "cmd1"],
            "recording counts: 2 < 9 < 10 (numeric order)"
        );
        sort_cmd_rows(&mut rows, CmdCol::Recs, true);
        assert_eq!(ids(&rows), vec!["cmd1", "cmd2", "cmd3"]);
    }
}
