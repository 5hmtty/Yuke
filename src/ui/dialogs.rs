use super::*;
use crate::i18n_keys::k;
use crate::lang::t;
use eframe::egui;

/// Height to reserve below the mutex panel's list area (logical points): one hairline plus the "Save mutex / Close" row.
/// The list then takes the remaining height and never pushes that button row out of the window.
const MUTEX_FOOTER_H: f32 = 52.0;

/// Remove first, then insert, and **decrement the insert index by 1 when `from < insert`**: removing the source row shifts every later index down by one;
/// without that correction the row lands one slot past the target (the classic off-by-one in drag reordering).
/// Out-of-range payloads are ignored (defensive against illegal payloads).
pub(super) fn reorder_rows<T>(rows: &mut Vec<T>, from: usize, insert: usize) {
    if from >= rows.len() {
        return;
    }
    let moved = rows.remove(from);
    let ins = if from < insert { insert - 1 } else { insert };
    rows.insert(ins.min(rows.len()), moved);
}

pub(super) fn macro_line_err_text(e: &crate::config::MacroLineErr) -> String {
    use crate::config::MacroLineErr as E;
    match e {
        E::MissingKey { act } => crate::lang::tf(k::ERR_MACRO_MISSING_KEY, &[("{act}", act)]),
        E::BadWait { arg } if arg.is_empty() => t(k::ERR_MACRO_MISSING_MS).to_string(),
        E::BadWait { arg } => crate::lang::tf(k::ERR_MACRO_BAD_MS, &[("{arg}", arg)]),
        E::WaitOverLimit { max, .. } => {
            crate::lang::tf(k::ERR_MACRO_MS_OVER, &[("{max}", &max.to_string())])
        }
        E::UnknownAction { act } => crate::lang::tf(k::ERR_MACRO_UNKNOWN_ACTION, &[("{act}", act)]),
    }
}

/// Append a run of hints **to the right of the last button** (pass them in visual left-to-right order).

/// Overlong text is truncated via `Label::truncate`, so it never pushes the buttons out of the window.
fn hints_after_buttons(ui: &mut egui::Ui, hints: &[(String, bool)]) {
    for (text, warn) in hints {
        let rt = egui::RichText::new(text);
        let rt = if *warn {
            rt.color(WARN_COLOR)
        } else {
            rt.weak()
        };
        ui.add(egui::Label::new(rt).truncate());
    }
}

/// One candidate row in the mutex panel (hoisted to module level so the sorting pure function is directly testable).
pub(super) struct MutexRow {
    pub id: String,
    pub display: String,
    pub mode: String,
    pub key: String,
}

/// Sort key for mutex-panel candidate rows (only one variant matches at a time, so cross-variant comparison never happens).
#[derive(PartialEq, Eq, PartialOrd, Ord)]
enum MutexKey {
    Name(String),
    Mode(String),
    Key(String),
}

/// Pinning is **a group**, not the first field of the key: folding it into the key would flip pinned rows to the bottom under a descending sort.
/// Implementation: split into two passes by checked state, sort each, then concatenate.

/// ⚠ Chinese names compare by **Unicode code point** (UTF-8 byte order), **not** by pinyin: the project has no pinyin table and
/// will not add a dependency for sorting. When mixed, Latin letters sort before CJK characters.
fn sort_mutex_rows(
    rows: &mut Vec<MutexRow>,
    checked: &std::collections::BTreeSet<String>,
    pin_checked: bool,
    col: MutexCol,
    desc: bool,
) {
    let key_of = |r: &MutexRow| match col {
        MutexCol::Name => MutexKey::Name(r.display.to_lowercase()),
        MutexCol::Mode => MutexKey::Mode(r.mode.clone()),
        MutexCol::Key => MutexKey::Key(r.key.to_lowercase()),
    };
    let tie_of = |r: &MutexRow| r.id.clone();
    if !pin_checked {
        sort_table_rows(rows, desc, key_of, tie_of);
        return;
    }
    let (mut on, mut off): (Vec<MutexRow>, Vec<MutexRow>) =
        rows.drain(..).partition(|r| checked.contains(&r.id));
    sort_table_rows(&mut on, desc, &key_of, &tie_of);
    sort_table_rows(&mut off, desc, &key_of, &tie_of);
    on.extend(off);
    *rows = on;
}

/// Sort key for the recording table. ⚠ Duration uses **integer milliseconds**: `dur_s` is `f32` and `f32` does not implement `Ord`
/// (NaN), so using it directly does not compile; converting to an integer also settles precision.
#[derive(PartialEq, Eq, PartialOrd, Ord)]
enum RecKey {
    File(String),
    Source(u8),
    Length(u64),
    Time(String),
}

fn sort_rec_rows(rows: &mut Vec<crate::recordings::CmdRecording>, col: RecCol, desc: bool) {
    let key_of = |r: &crate::recordings::CmdRecording| match col {
        RecCol::File => RecKey::File(r.file.to_lowercase()),
        RecCol::Source => RecKey::Source(match r.kind {
            crate::recordings::RecKind::Active => 0,
            crate::recordings::RecKind::Learned => 1,
        }),
        RecCol::Length => RecKey::Length((r.dur_s * 1000.0).round().max(0.0) as u64),
        RecCol::Time => RecKey::Time(r.ts.clone()),
    };
    sort_table_rows(rows, desc, key_of, |r| r.file.clone());
}

impl ViqApp {
    fn pron_edit_ui(&mut self, ui: &mut egui::Ui, cmd: &str) {
        enum Action {
            Play(String),
            DelRec(String),
            StartRec,
            StopRec,
            CaptureKey,
            SaveKey(String),
            SetModKey(String),
            ClearKey,
            GenProtoSingle,
            LaunchOsk(String),
            SetMode(String), // Trigger mode, three states (tap/toggle/macro)
        }
        let mut actions: Vec<Action> = Vec::new();
        let phrase = self
            .cfg
            .get("commands")
            .and_then(|v| v.get(cmd))
            .and_then(|c| c.get("phrase"))
            .and_then(|v| v.as_str())
            .filter(|s| !s.is_empty())
            .unwrap_or(cmd)
            .to_string();
        let cur_mode = self
            .cfg
            .get("commands")
            .and_then(|v| v.get(cmd))
            .and_then(|c| c.get("mode"))
            .and_then(|v| v.as_str())
            .unwrap_or("tap")
            .to_string();
        let is_passcode = self.settings.voice_toggle_cmd == cmd;

        let is_new = self.pron_new;
        let cfg_phrase = self
            .cfg
            .get("commands")
            .and_then(|v| v.get(cmd))
            .and_then(|c| c.get("phrase"))
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let mut save_name = false;
        if is_new {
            ui.horizontal(|ui| {
                ui.label(t(k::UI_LBL_NAME));
                let resp = ui.add(
                    egui::TextEdit::singleline(&mut self.add_phrase)
                        .desired_width(220.0)
                        .hint_text(t(k::UI_HINT_NAME_EXAMPLE)),
                );
                if self.pron_new_focus {
                    resp.request_focus();
                    self.pron_new_focus = false;
                }
                let typed = self.add_phrase.trim().to_string();
                if !typed.is_empty()
                    && typed != cfg_phrase
                    && (resp.lost_focus() || ui.input(|i| i.key_pressed(egui::Key::Enter)))
                {
                    save_name = true;
                }
                ui.weak(t(k::UI_HINT_NAME_ENTER));
            });
        } else {
            // -- Status badges: command name plus whether the pronunciation template exists (prominent) --
            ui.horizontal(|ui| {
                ui.heading(crate::tfmt!(k::UI_TTL_CMD, phrase));
                ui.weak(format!("（id: {cmd}）"));
                let covered = self.proto_cover.contains(cmd);
                if covered {
                    ui.colored_label(
                        egui::Color32::from_rgb(76, 175, 80),
                        t(k::UI_BADGE_TEMPLATE_READY),
                    );
                } else {
                    ui.colored_label(WARN_COLOR, t(k::UI_BADGE_TEMPLATE_MISSING));
                }
            });
        }
        ui.separator();

        if self.rec_counts.get(cmd).copied().unwrap_or(0) == 0 {
            ui.colored_label(WARN_COLOR, t(k::UI_HINT_NO_RECS));
        }

        ui.horizontal(|ui| {
            ui.label(t(k::UI_LBL_TRIGGER_MODE));
            let mut mode_sel = cur_mode.clone();
            egui::ComboBox::from_id_salt(format!("mode_select_{cmd}"))
                .selected_text(mode_label(&mode_sel))
                .width(110.0)
                .show_ui(ui, |ui| {
                    // Tap (default)
                    ui.selectable_value(&mut mode_sel, "tap".to_string(), t(k::UI_OPT_TAP));
                    // Hold-to-toggle: not offered for passcode pronunciations (mute semantics conflict with toggle; boundary 4)
                    if !is_passcode {
                        ui.selectable_value(
                            &mut mode_sel,
                            "toggle".to_string(),
                            t(k::UI_OPT_TOGGLE_PRESS),
                        );
                    }
                    // Macro: also not offered for passcode pronunciations (a passcode means zero key output, which contradicts a macro)
                    if !is_passcode {
                        ui.selectable_value(&mut mode_sel, "macro".to_string(), t(k::UI_OPT_MACRO));
                    }
                });
            if cur_mode == "macro" {
                if ui.button(t(k::UI_TTL_MACRO_EDITOR)).clicked()
                    && self.macro_editor_open.as_deref() != Some(cmd)
                {
                    self.macro_editor_open = Some(cmd.to_string());
                    self.macro_edit_for = None; // Force the buffer to reload for this command
                }
            }
            if mode_sel != cur_mode {
                actions.push(Action::SetMode(mode_sel));
            }
            ui.weak(match cur_mode.as_str() {
                "toggle" => t(k::UI_HINT_TOGGLE_PRESS),
                "macro" => t(k::UI_HINT_MACRO_MODE),
                _ => t(k::UI_HINT_TAP),
            });
        });
        if cur_mode == "macro" {
            ui.horizontal(|ui| {
                ui.label(t(k::UI_LBL_KEY_MAPPING));
                ui.weak(t(k::UI_HINT_KEY_BY_MACRO));
                self.mutex_button_ui(ui, cmd);
            });
        } else {
            ui.horizontal(|ui| {
                ui.label(t(k::UI_LBL_KEY_MAPPING));
                let armed = self.capture_key_for == Some(cmd.to_string());
                let mut edit = self.key_edit.clone();
                let mut do_capture = false;
                let resp = ui.add_enabled(
                    !armed,
                    egui::TextEdit::singleline(&mut edit).desired_width(150.0),
                );
                if resp.changed() {
                    self.key_edit = edit.clone();
                }
                if (resp.lost_focus() || ui.input(|i| i.key_pressed(egui::Key::Enter)))
                    && !self.key_edit.trim().is_empty()
                {
                    actions.push(Action::SaveKey(self.key_edit.trim().to_uppercase()));
                }
                if !armed
                    && ui
                        .button("⌨")
                        .on_hover_text(t(k::UI_BTN_CAPTURE_KEY))
                        .clicked()
                {
                    do_capture = true;
                }
                if do_capture {
                    actions.push(Action::CaptureKey);
                }
                if armed {
                    if ui.button(t(k::UI_BTN_CANCEL)).clicked() {
                        self.capture_key_for = None;
                        self.status = t(k::MSG_KEY_CAPTURE_CANCELLED);
                    }
                } else if ui.button(t(k::UI_BTN_CLEAR_KEY)).clicked() {
                    actions.push(Action::ClearKey);
                }
                // Visible hint while capturing (ESC to cancel is the expectation; the exit is the "Cancel" button at the left of this row)
                if armed {
                    ui.colored_label(WARN_COLOR, t(k::UI_HINT_PRESS_KEY));
                }
                if ui.button(t(k::UI_BTN_OSK)).clicked() {
                    // Opens the on-screen keyboard and enters capture mode: osk presses travel through the capture path, which records the full chord
                    // (SHIFT+A etc.); typing straight into the field would lose the modifiers (only the uppercase character remains).
                    actions.push(Action::LaunchOsk(cmd.to_string()));
                }
                let mod_cur = if self.key_mod_sel.is_empty() {
                    t(k::UI_LBL_MODIFIER)
                } else {
                    self.key_mod_sel.clone()
                };
                egui::ComboBox::from_id_salt(format!("key_mod_{cmd}"))
                    .selected_text(mod_cur)
                    .width(90.0)
                    .show_ui(ui, |ui| {
                        ui.selectable_value(&mut self.key_mod_sel, "CTRL".to_string(), "Ctrl");
                        ui.selectable_value(&mut self.key_mod_sel, "SHIFT".to_string(), "Shift");
                        ui.selectable_value(&mut self.key_mod_sel, "ALT".to_string(), "Alt");
                    });
                if !self.key_mod_sel.is_empty() {
                    let picked = self.key_mod_sel.clone();
                    self.key_mod_sel.clear();
                    actions.push(Action::SetModKey(picked));
                }
                self.mutex_button_ui(ui, cmd);
            });
        }

        ui.separator();
        // -- Recording actions (add takes / regenerate template / rename, placed next to the existing recordings) --
        ui.horizontal(|ui| {
            if self.rec_mic.is_some() {
                if ui.button(t(k::UI_BTN_STOP_RECORDING)).clicked() {
                    actions.push(Action::StopRec);
                }
            } else if ui.button(t(k::UI_BTN_RECORD)).clicked() {
                actions.push(Action::StartRec);
            }
            let loading = self.capture_job != CaptureJob::None;
            let busy = self.proto_busy();
            let gen_this = matches!(
                &self.proto_job,
                Some(ProtoJob { kind: ProtoKind::Single(c), .. }) if c == cmd
            );
            let resp = ui.add_enabled(
                !busy && !loading,
                egui::Button::new(t(k::UI_BTN_REBUILD_TEMPLATE)),
            );
            if resp.clicked() {
                actions.push(Action::GenProtoSingle);
            }
            if ui.button(t(k::UI_BTN_RENAME_CMD)).clicked() {
                self.rename_buf = phrase.clone();
                self.rename = Some(Rename::CommandPhrase(cmd.to_string()));
            }

            // Hint area: appended to the right of "Rename command" (array order = visual left to right; the most important red text goes last)
            let mut hints: Vec<(String, bool)> = Vec::new();
            if self.ctx.is_some() {
                hints.push((t(k::MSG_REC_SHARES_MIC), false));
            }
            if busy {
                hints.push(if gen_this {
                    (t(k::MSG_TEMPLATE_BUILDING_SHORT), true)
                } else {
                    (t(k::MSG_TEMPLATE_BUSY_OTHER), false)
                });
            } else if loading {
                hints.push((t(k::MSG_ENGINE_LOADING_NO_BUILD), false));
            }
            if self.rec_mic.is_some() {
                let elapsed = self
                    .rec_start_at
                    .map(|t| t.elapsed().as_secs_f32())
                    .unwrap_or(0.0);
                hints.push((
                    crate::lang::tf(
                        k::MSG_RECORDING_ELAPSED,
                        &[
                            ("{elapsed}", &format!("{elapsed:.1}")),
                            ("{phrase}", &phrase),
                        ],
                    ),
                    true,
                ));
            }
            hints_after_buttons(ui, &hints);
        });
        // -- Recording list --
        ui.horizontal(|ui| {
            ui.label(t(k::UI_LBL_RECORDINGS));
            super::pages::count_badge(ui, self.rec_list.len());
        });
        let list_h = (ui.available_height() - 44.0).max(60.0);
        let rh = ui.spacing().interact_size.y;
        if let Some((col, desc)) = self.rec_sort {
            sort_rec_rows(&mut self.rec_list, col, desc);
        }
        ui.scope(|ui| {
            ui.style_mut().visuals.widgets.noninteractive.bg_stroke = egui::Stroke::NONE;
            egui_extras::TableBuilder::new(ui)
                .id_salt("rec_table")
                .striped(true)
                .resizable(true)
                .min_scrolled_height(60.0)
                .max_scroll_height(list_h)
                .cell_layout(egui::Layout::left_to_right(egui::Align::Center))
                // File / origin / duration / time / actions (the last column takes the remaining width)
                .column(
                    egui_extras::Column::initial(170.0)
                        .at_least(80.0)
                        .clip(true),
                )
                .column(egui_extras::Column::initial(52.0).at_least(40.0))
                .column(egui_extras::Column::initial(52.0).at_least(40.0))
                .column(
                    egui_extras::Column::initial(150.0)
                        .at_least(80.0)
                        .clip(true),
                )
                .column(egui_extras::Column::remainder().at_least(112.0))
                .header(rh, |mut header| {
                    let cols = [
                        (t(k::UI_TBL_FILE), RecCol::File),
                        (t(k::UI_TBL_SOURCE), RecCol::Source),
                        (t(k::UI_TBL_LENGTH), RecCol::Length),
                        (t(k::UI_TBL_TIME), RecCol::Time),
                    ];
                    for (label, col) in cols {
                        header.col(|ui| {
                            sort_header(ui, &label, col, &mut self.rec_sort);
                        });
                    }
                    header.col(|ui| {
                        ui.weak(t(k::UI_TBL_ACTIONS));
                    });
                })
                .body(|mut body| {
                    for r in &self.rec_list {
                        body.row(rh, |mut row| {
                            row.col(|ui| {
                                // The longest part of a file name is the trailing timestamp; after truncation, hovering shows the full name
                                ui.add(
                                    egui::Label::new(egui::RichText::new(&r.file).monospace())
                                        .truncate(),
                                )
                                .on_hover_text(&r.file);
                            });
                            row.col(|ui| {
                                ui.add(
                                    egui::Label::new(match r.kind {
                                        crate::recordings::RecKind::Active => t(k::UI_OPT_ENROLLED),
                                        crate::recordings::RecKind::Learned => t(k::UI_OPT_LEARNED),
                                    })
                                    .truncate(),
                                );
                            });
                            row.col(|ui| {
                                ui.add(egui::Label::new(format!("{:.1}s", r.dur_s)).truncate());
                            });
                            row.col(|ui| {
                                ui.add(egui::Label::new(&r.ts).truncate());
                            });
                            row.col(|ui| {
                                if ui.button(t(k::UI_BTN_PLAY)).clicked() {
                                    actions.push(Action::Play(r.file.clone()));
                                }
                                // Double-click delete -- the first click turns red ("confirm delete"), the second executes
                                let armed_rec = self.armed_is(&ArmedDelete::Rec {
                                    cmd: cmd.to_string(),
                                    file: r.file.clone(),
                                });
                                let del_btn = ui.button(if armed_rec {
                                    egui::RichText::new(t(k::UI_BTN_CONFIRM_DELETE))
                                        .color(WARN_COLOR)
                                } else {
                                    egui::RichText::new(t(k::UI_BTN_DELETE))
                                });
                                if del_btn.clicked() {
                                    actions.push(Action::DelRec(r.file.clone()));
                                }
                            });
                        });
                    }
                });
        });

        // -- Bottom close (the explicit exit inside the dialog) --
        // Unified for both the new and edit modes. Finishing a new command (no name -> a default name; nothing filled in -> delete the command)
        // is handled uniformly by `close_edit_popup` -> `finalize_new_command`.
        ui.separator();
        ui.horizontal(|ui| {
            if ui.button(t(k::UI_BTN_CLOSE)).clicked() {
                self.close_edit_popup();
            }
        });

        if save_name {
            let name = self.add_phrase.trim().to_string();
            self.set_cmd_field(cmd, "phrase", serde_json::Value::String(name.clone()));
            self.persist_cfg();
            self.status = crate::tfmt!(k::MSG_CMD_NAMED, name);
        }

        for a in actions {
            match a {
                Action::Play(file) => self.play_rec(&file),
                Action::DelRec(file) => {
                    // Double-click delete -- the first click arms it, the second executes (the last recording goes through the same path)
                    let kind = ArmedDelete::Rec {
                        cmd: cmd.to_string(),
                        file: file.clone(),
                    };
                    if self.armed_is(&kind) {
                        self.armed_delete = None;
                        match crate::recordings::remove(&self.model_dir, &file) {
                            Ok(_) => {
                                self.log(&crate::tfmt!(k::MSG_REC_DELETED_LOG, file));
                                self.status = crate::tfmt!(k::MSG_REC_DELETED, file);
                            }
                            Err(e) => self.status = crate::tfmt!(k::ERR_DELETE, e),
                        }
                        self.refresh_rec_counts();
                        self.refresh_rec_list(cmd);
                    } else {
                        self.armed_delete = Some((kind, std::time::Instant::now()));
                    }
                }
                Action::StartRec => self.start_rec(cmd),
                Action::StopRec => self.stop_rec(),
                Action::CaptureKey => {
                    self.capture_key_for = Some(cmd.to_string());
                }
                Action::SaveKey(key) => {
                    self.set_cmd_field(cmd, "key", serde_json::Value::String(key.clone()));
                    self.key_edit = key.clone();
                    self.persist_cfg();
                    self.status = format!("{cmd} → {key}");
                }
                Action::SetModKey(m) => {
                    let cur = self.key_edit.trim().to_uppercase();
                    let merged = if cur.is_empty() {
                        m.clone()
                    } else {
                        format!("{m}+{cur}")
                    };
                    self.key_edit = merged.clone();
                    self.key_mod_sel.clear();
                    self.set_cmd_field(cmd, "key", serde_json::Value::String(merged.clone()));
                    self.persist_cfg();
                    self.status = format!("{cmd} → {merged}");
                }
                Action::ClearKey => {
                    self.key_edit.clear();
                    self.set_cmd_field(cmd, "key", serde_json::Value::String(String::new()));
                    self.persist_cfg();
                    self.status = format!("key of {cmd} cleared");
                }
                Action::SetMode(m) => {
                    self.set_cmd_field(cmd, "mode", serde_json::Value::String(m.clone()));
                    self.persist_cfg();
                    self.status =
                        crate::lang::tf(k::MSG_TRIGGER_MODE_SET, &[("{mode}", &mode_label(&m))]);
                    let line = self.status.clone();
                    self.log(&line);
                }
                Action::GenProtoSingle => self.start_proto_single(cmd),
                Action::LaunchOsk(target) => self.launch_osk(Some(&target)),
            }
        }
    }

    /// Edit popup shell (drawn every frame while pron_open = Some(cmd); content in pron_edit_ui).
    pub(super) fn draw_edit_popup(&mut self, ctx: &egui::Context) {
        let Some(cmd) = self.pron_open.clone() else {
            return;
        };
        const W: f32 = 560.0;
        const H0: f32 = 480.0;
        let mut open = true;
        let center = ctx.screen_rect().center();
        // The title varies by mode but the **id is fixed** (`.id(Id::new(..))`): both states share one window's
        // position and size memory, so the create dialog grows into the edit dialog in place without jumping or resizing.
        let title = if self.pron_new {
            t(k::UI_TTL_NEW_CMD)
        } else {
            t(k::UI_TTL_EDIT_CMD)
        };
        egui::Window::new(title)
            .id(egui::Id::new("pron_edit_win"))
            .open(&mut open)
            .default_pos(center - egui::vec2(W, H0) / 2.0)
            .default_size(egui::vec2(W, H0))
            .min_size(egui::vec2(W, H0))
            .max_size(egui::vec2(W, 900.0))
            .collapsible(false)
            .resizable([false, true])
            .show(ctx, |ui| {
                self.pron_edit_ui(ui, &cmd);
            });
        if !open {
            self.close_edit_popup();
        }
    }

    pub(super) fn macro_save_from_edit(&mut self, cmd: &str) {
        let actions_v: Vec<crate::config::MacroAction> = self
            .macro_edit
            .iter()
            .filter_map(macro_row_to_action)
            .collect();
        let total_ms: u64 = actions_v.iter().map(crate::config::macro_action_ms).sum();
        if actions_v.len() > crate::config::MACRO_MAX_ACTIONS
            || total_ms > crate::config::MACRO_MAX_TOTAL_MS
        {
            self.status = t(k::ERR_MACRO_OVER_LIMIT_DETAIL);
            return;
        }
        use crate::config::MacroAction;
        let arr: Vec<serde_json::Value> = actions_v
            .iter()
            .map(|a| match a {
                MacroAction::Down(k) => serde_json::json!({"action": "down", "key": k}),
                MacroAction::Up(k) => serde_json::json!({"action": "up", "key": k}),
                MacroAction::Press(k) => serde_json::json!({"action": "press", "key": k}),
                MacroAction::Wait(ms) => serde_json::json!({"action": "wait", "ms": ms}),
            })
            .collect();
        self.set_cmd_field(cmd, "macro", serde_json::Value::Array(arr));
        self.persist_cfg();
        self.status = crate::lang::tf(k::MSG_MACRO_SAVED, &[("{n}", &actions_v.len().to_string())]);
        let line = self.status.clone();
        self.log(&line);
    }

    pub(super) fn draw_macro_editor(&mut self, ctx: &egui::Context) {
        let Some(cmd) = self.macro_editor_open.clone() else {
            return;
        };
        // The command was deleted (not found in cfg) -> close the window automatically (no editing or saving a ghost command)
        if self.cfg.get("commands").and_then(|v| v.get(&cmd)).is_none() {
            self.macro_editor_open = None;
            self.macro_edit_for = None;
            return;
        }
        // Macro buffer sync (same popup mechanism): reload from cfg on first open and when switching to another command
        if self.macro_edit_for.as_deref() != Some(cmd.as_str()) {
            self.macro_edit_for = Some(cmd.clone());
            self.macro_edit = macro_rows_from_cfg(&self.cfg, &cmd);
            self.macro_import_open = false;
            self.macro_import_errs.clear();
        }
        let phrase = self
            .cfg
            .get("commands")
            .and_then(|v| v.get(&cmd))
            .and_then(|c| c.get("phrase"))
            .and_then(|v| v.as_str())
            .filter(|s| !s.is_empty())
            .unwrap_or(&cmd)
            .to_string();
        let mut open = true; // Window close button (egui::Window::open)
        let mut do_save = false;
        let mut do_close = false;
        const MW: f32 = 440.0;
        const MH0: f32 = 420.0;
        egui::Window::new(crate::tfmt!(k::UI_TTL_MACRO_EDITOR_CMD, phrase))
            .open(&mut open)
            .collapsible(false)
            .default_size(egui::vec2(MW, MH0))
            .min_size(egui::vec2(MW, MH0))
            .max_size(egui::vec2(MW, 1000.0))
            .resizable([false, true])
            .show(ctx, |ui| {
                let mut rows = std::mem::take(&mut self.macro_edit);
                let est_ms: u64 = rows
                    .iter()
                    .map(|r| match r.kind {
                        MK_WAIT => r.param.trim().parse::<u64>().unwrap_or(0),
                        MK_PRESS => crate::config::MACRO_PRESS_MS as u64,
                        _ => 0,
                    })
                    .sum();
                let over_count = rows.len() > crate::config::MACRO_MAX_ACTIONS;
                let over_time = est_ms > crate::config::MACRO_MAX_TOTAL_MS;
                ui.horizontal(|ui| {
                    ui.label(t(k::UI_LBL_MACRO_ACTIONS));
                    let summary = crate::lang::tf(
                        k::MSG_MACRO_SUMMARY,
                        &[
                            ("{n}", &rows.len().to_string()),
                            ("{sec}", &format!("{:.1}", est_ms as f32 / 1000.0)),
                        ],
                    );
                    if over_time || over_count {
                        ui.colored_label(
                            WARN_COLOR,
                            egui::RichText::new(format!(
                                "{summary} {}",
                                t(k::ERR_MACRO_OVER_LIMIT)
                            )),
                        );
                    } else {
                        ui.weak(summary);
                    }
                });
                // Row cards: type dropdown + argument field + up/down + remove; clicks are collected first and applied by row index
                // in descending order outside the row loop (so multiple operations in one frame do not invalidate indices -- same technique as the original actions block)
                let mut row_ops: Vec<(usize, Option<isize>)> = Vec::new(); // (row index, Some(±1) = move / None = delete)
                let mut add_clicked = false;
                let mut drag_insert: Option<usize> = None;
                let mut drag_move: Option<(usize, usize)> = None; // (from, insert position)
                let dragging = egui::DragAndDrop::has_payload_of_type::<usize>(ui.ctx());
                ui.horizontal(|ui| {
                    if ui.button(t(k::UI_BTN_ADD_ACTION)).clicked() {
                        add_clicked = true;
                    }
                    if ui.button(t(k::UI_BTN_TEXT_EDIT)).clicked() {
                        // Pre-fill the current sequence: entering this mode means "edit these actions as text"
                        self.macro_import_buf = macro_rows_to_text(&rows);
                        self.macro_import_errs.clear();
                        self.macro_import_open = true;
                    }
                    // Save macro: invalid lines or over-limit greys out and writes nothing to disk
                    let can_save = !rows.is_empty()
                        && !over_count
                        && !over_time
                        && rows.iter().all(macro_row_valid);
                    let resp = ui.add_enabled(
                        can_save,
                        egui::Button::new(
                            egui::RichText::new(t(k::UI_BTN_SAVE_MACRO))
                                .color(egui::Color32::from_rgb(76, 175, 80)),
                        ),
                    );
                    if resp.clicked() {
                        do_save = true;
                    }
                    if !rows.is_empty() && rows.iter().any(|r| !macro_row_valid(r)) {
                        ui.colored_label(WARN_COLOR, t(k::ERR_MACRO_ROW_INVALID));
                    }
                    if ui.button(t(k::UI_BTN_CLOSE)).clicked() {
                        do_close = true;
                    }
                });
                ui.separator();
                // The action list takes the remaining height: together with the window's min_size it makes "0 rows and 64 rows occupy the same space",
                // otherwise content height drives window auto-sizing (per the egui docs, even resizable(false) can still auto-resize,
                // so a size cap and content adaptation are both needed to box it in).
                // Reserve ~34pt for the re-entry policy row below.
                let rows_h = (ui.available_height() - 34.0).max(90.0);
                let mut row_bounds: Vec<egui::Rect> = Vec::new();
                egui::ScrollArea::vertical()
                    .id_salt("macro_rows_scroll")
                    .max_height(rows_h)
                    .auto_shrink([false, false])
                    // Disable stick_to_bottom while dragging: otherwise it keeps pulling the list and fights the drag near the bottom
                    .stick_to_bottom(!dragging)
                    .show(ui, |ui| {
                        let scroll_rect = ui.clip_rect();
                        for i in 0..rows.len() {
                            let row_resp = ui
                                .horizontal(|ui| {
                                    // Drag handle: the **only** control in this row that senses drag.
                                    // `dnd_set_drag_payload` writes the global payload once the drag actually starts,
                                    // using payload type usize so it stays distinguishable from other DnD uses.
                                    ui.add(egui::Label::new("⠿").sense(egui::Sense::drag()))
                                        .on_hover_cursor(egui::CursorIcon::Grab)
                                        .dnd_set_drag_payload(i);
                                    // Action type dropdown (down / up / press / wait)
                                    let kind_names = [
                                        t(k::UI_OPT_MACRO_DOWN),
                                        t(k::UI_OPT_MACRO_UP),
                                        t(k::UI_OPT_MACRO_CLICK),
                                        t(k::UI_OPT_MACRO_WAIT),
                                    ];
                                    egui::ComboBox::from_id_salt(format!("macro_kind_{i}"))
                                        .selected_text(kind_names[rows[i].kind].clone())
                                        .width(64.0)
                                        .show_ui(ui, |ui| {
                                            for (k, name) in kind_names.iter().enumerate() {
                                                ui.selectable_value(
                                                    &mut rows[i].kind,
                                                    k,
                                                    name.clone(),
                                                );
                                            }
                                        });
                                    // Argument field (key name / milliseconds); invalid rows get a red border (without an extra text marker)
                                    let param_hint = if rows[i].kind == MK_WAIT {
                                        t(k::UI_HINT_MACRO_WAIT_MS)
                                    } else {
                                        t(k::UI_HINT_MACRO_KEY)
                                    };
                                    let mut param = rows[i].param.clone();
                                    let resp = ui.add(
                                        egui::TextEdit::singleline(&mut param)
                                            .hint_text(param_hint)
                                            .desired_width(120.0),
                                    );
                                    rows[i].param = param;
                                    let cap_target = format!("__macro_{i}__");
                                    let capturing_this = self.capture_key_for.as_deref()
                                        == Some(cap_target.as_str());
                                    if rows[i].kind == MK_WAIT {
                                        ui.add_enabled(false, egui::Button::new("⌨"));
                                    } else if capturing_this {
                                        if ui.small_button(t(k::UI_BTN_CANCEL)).clicked() {
                                            self.capture_key_for = None;
                                        }
                                        ui.spinner();
                                    } else if ui.small_button("⌨").clicked() {
                                        self.capture_key_for = Some(cap_target);
                                    }
                                    if !macro_row_valid(&rows[i]) {
                                        ui.painter().rect_stroke(
                                            resp.rect,
                                            3.0,
                                            egui::Stroke::new(1.5, WARN_COLOR),
                                            egui::StrokeKind::Inside,
                                        );
                                    }
                                    // Up/down move + remove (edit the buffer directly; collect then apply)
                                    ui.add_enabled_ui(i > 0, |ui| {
                                        if ui.small_button(t(k::UI_BTN_MOVE_UP)).clicked() {
                                            row_ops.push((i, Some(-1)));
                                        }
                                    });
                                    ui.add_enabled_ui(i + 1 < rows.len(), |ui| {
                                        if ui.small_button(t(k::UI_BTN_MOVE_DOWN)).clicked() {
                                            row_ops.push((i, Some(1)));
                                        }
                                    });
                                    if ui.small_button(t(k::UI_BTN_REMOVE_ROW)).clicked() {
                                        row_ops.push((i, None));
                                    }
                                })
                                .response;
                            row_bounds.push(row_resp.rect);
                            // Drop detection is **pure geometry**, not `contains_pointer`, because child controls such as dropdowns and text fields
                            // would sit on top and make it miss (the official dnd_drop_zone uses contains_pointer, but there the drop area is empty).
                            //
                            if dragging {
                                if let Some(p) = ui.ctx().pointer_interact_pos() {
                                    if row_resp.rect.contains(p) {
                                        drag_insert = Some(if p.y < row_resp.rect.center().y {
                                            i
                                        } else {
                                            i + 1
                                        });
                                    }
                                }
                            }
                        }
                        if dragging {
                            // Blank area past the end of the list -> insert at the end
                            if let Some(p) = ui.ctx().pointer_interact_pos() {
                                let last = row_bounds
                                    .last()
                                    .map(|r| r.bottom())
                                    .unwrap_or(scroll_rect.top());
                                if scroll_rect.contains(p) && p.y > last {
                                    drag_insert = Some(rows.len());
                                }
                            }
                            // Insertion line (a 2px marker between rows): more intuitive than swapping with the hovered row, and it does not
                            // jitter back and forth on long lists
                            if let Some(k) = drag_insert {
                                let y = row_bounds.get(k).map(|r| r.top()).unwrap_or_else(|| {
                                    row_bounds
                                        .last()
                                        .map(|r| r.bottom())
                                        .unwrap_or(scroll_rect.top())
                                });
                                ui.painter().hline(
                                    scroll_rect.x_range(),
                                    y,
                                    egui::Stroke::new(2.0, ui.visuals().selection.bg_fill),
                                );
                            }
                            // Edge auto-scroll: egui has **no** built-in support (a crate-wide search for auto_scroll finds nothing),
                            // so the scroll amount is fed every frame from pointer proximity to the edge. Long macros (up to 64 actions) rely on it.
                            if let Some(p) = ui.ctx().pointer_interact_pos() {
                                if p.y < scroll_rect.top() + 22.0 {
                                    ui.scroll_with_delta(egui::vec2(0.0, 14.0));
                                } else if p.y > scroll_rect.bottom() - 22.0 {
                                    ui.scroll_with_delta(egui::vec2(0.0, -14.0));
                                }
                            }
                            // Release: take payload and drop point -> record one reorder (the buffer mutation happens outside the loop, see below)
                            if ui.input(|i| i.pointer.any_released()) {
                                if let (Some(from), Some(k)) = (
                                    egui::DragAndDrop::take_payload::<usize>(ui.ctx()),
                                    drag_insert,
                                ) {
                                    drag_move = Some((*from, k)); // take_payload returns the Arc; take the inner index
                                }
                            }
                        }
                    });
                // Row deletions and moves are applied in descending index order (multiple operations in one frame do not invalidate indices)
                row_ops.sort_by(|x, y| y.0.cmp(&x.0));
                for (i, op) in row_ops {
                    match op {
                        None => {
                            if i < rows.len() {
                                rows.remove(i);
                            }
                        }
                        Some(d) => {
                            let j = if d < 0 { i.saturating_sub(1) } else { i + 1 };
                            if j < rows.len() {
                                rows.swap(i, j);
                            }
                        }
                    }
                }
                // Drag reorder applied on the release frame.
                // **Never reorder during the drag** -- row control ids are index-based (`macro_kind_{i}`), so reordering mid-drag
                // would apply the in-progress drag state to a different row.
                if let Some((from, insert)) = drag_move {
                    reorder_rows(&mut rows, from, insert);
                }
                if add_clicked {
                    rows.push(MacroRow {
                        kind: 0,
                        param: String::new(),
                    });
                }
                self.macro_edit = rows;
                ui.horizontal(|ui| {
                    ui.label(t(k::UI_LBL_REENTRY));
                    let cur = self
                        .cfg
                        .get("commands")
                        .and_then(|v| v.get(&cmd))
                        .and_then(|c| c.get("macro_reentry"))
                        .and_then(|v| v.as_str())
                        .unwrap_or(crate::config::MACRO_REENTRY_IGNORE)
                        .to_string();
                    let mut sel = cur.clone();
                    egui::ComboBox::from_id_salt(format!("macro_reentry_{cmd}"))
                        .selected_text(reentry_label(&sel))
                        .width(110.0)
                        .show_ui(ui, |ui| {
                            ui.selectable_value(
                                &mut sel,
                                "ignore".to_string(),
                                t(k::UI_OPT_REENTRY_IGNORE),
                            );
                            ui.selectable_value(
                                &mut sel,
                                "concurrent".to_string(),
                                t(k::UI_OPT_REENTRY_CONCURRENT),
                            );
                            ui.selectable_value(
                                &mut sel,
                                "interrupt".to_string(),
                                t(k::UI_OPT_REENTRY_INTERRUPT),
                            );
                        });
                    ui.weak(match sel.as_str() {
                        "concurrent" => t(k::UI_HINT_REENTRY_CONCURRENT),
                        "interrupt" => t(k::UI_HINT_REENTRY_INTERRUPT),
                        _ => t(k::UI_HINT_REENTRY_IGNORE),
                    });
                    if sel != cur {
                        self.set_cmd_field(
                            &cmd,
                            "macro_reentry",
                            serde_json::Value::String(sel.clone()),
                        );
                        self.persist_cfg();
                        self.status = crate::lang::tf(
                            k::MSG_REENTRY_SET,
                            &[("{mode}", &reentry_label(&sel))],
                        );
                        let line = self.status.clone();
                        self.log(&line);
                    }
                });
            });
        // Save runs after the buffer is written back (macro_save_from_edit reads self.macro_edit)
        if do_save {
            self.macro_save_from_edit(&cmd);
        }
        // Close: the X or the Close button (the buffer stays in macro_edit and is reloaded via macro_edit_for on reopen)
        if do_close {
            open = false;
        }
        if !open {
            self.macro_editor_open = None;
            self.macro_edit_for = None;
        }
        // ⚠ This window is nested (the parent already owns a name), so the title carries the phrase to tell multiple open editors apart.
        if self.macro_import_open {
            let mut buf = std::mem::take(&mut self.macro_import_buf);
            let buf_before = buf.clone();
            let errs = self.macro_import_errs.clone();
            let mut do_apply = false;
            let mut do_copy = false;
            let mut do_close_import = false;
            const TW: f32 = 420.0;
            egui::Window::new(crate::tfmt!(k::UI_TTL_TEXT_EDIT, phrase))
                .collapsible(false)
                .default_size(egui::vec2(TW, 300.0))
                .min_size(egui::vec2(TW, 240.0))
                .max_size(egui::vec2(TW, 700.0))
                .resizable([false, true])
                .show(ctx, |w| {
                    w.label(t(k::UI_HINT_MACRO_TEXT_RULES));
                    let text_h = (w.available_height() - 70.0).max(90.0);
                    w.add_sized(
                        egui::vec2(w.available_width(), text_h),
                        egui::TextEdit::multiline(&mut buf),
                    );
                    if !errs.is_empty() {
                        w.colored_label(
                            WARN_COLOR,
                            crate::lang::tf(
                                k::ERR_MACRO_LINES_INVALID,
                                &[("{n}", &errs.len().to_string())],
                            ),
                        );
                        for (ln, e) in &errs {
                            w.colored_label(
                                WARN_COLOR,
                                crate::lang::tf(
                                    k::ERR_MACRO_LINE_DETAIL,
                                    &[
                                        ("{line}", &ln.to_string()),
                                        ("{reason}", &macro_line_err_text(e)),
                                    ],
                                ),
                            );
                        }
                    }
                    w.horizontal(|w| {
                        if w.button(t(k::UI_BTN_SAVE)).clicked() {
                            do_apply = true;
                        }
                        if w.button(t(k::UI_BTN_COPY_TEXT)).clicked() {
                            do_copy = true;
                        }
                        if w.button(t(k::UI_BTN_CANCEL)).clicked() {
                            do_close_import = true;
                        }
                    });
                });
            if buf != buf_before {
                self.macro_import_errs.clear();
            }
            self.macro_import_buf = buf;
            if do_close_import {
                self.macro_import_open = false;
                self.macro_import_errs.clear();
            }
            if do_copy {
                ctx.copy_text(self.macro_import_buf.clone());
                self.status = t(k::MSG_MACRO_COPIED);
                let line = self.status.clone();
                self.log(&line);
            }
            if do_apply {
                // "Save" = parse + replace the action list + write to disk (all in one step). With an invalid line,
                // apply returns false, the window stays open and reports per-line errors, and **nothing is written**.
                if self.apply_macro_import() {
                    self.macro_save_from_edit(&cmd);
                }
            }
        }
    }

    /// The actual execution of "import and replace" (a separate function: drawing only collects clicks, probes test the chain directly -- same pattern as exec_confirm).
    pub(super) fn apply_macro_import(&mut self) -> bool {
        use crate::config::MacroAction;
        let text = self.macro_import_buf.clone();
        let (ok, bad) = crate::config::parse_macro_text(&text);
        if !bad.is_empty() {
            self.macro_import_errs = bad;
            return false;
        }
        self.macro_edit = ok
            .iter()
            .map(|a| match a {
                MacroAction::Down(k) => MacroRow {
                    kind: 0,
                    param: k.clone(),
                },
                MacroAction::Up(k) => MacroRow {
                    kind: 1,
                    param: k.clone(),
                },
                MacroAction::Press(k) => MacroRow {
                    kind: MK_PRESS,
                    param: k.clone(),
                },
                MacroAction::Wait(ms) => MacroRow {
                    kind: MK_WAIT,
                    param: ms.to_string(),
                },
            })
            .collect();
        self.macro_import_open = false;
        self.macro_import_errs.clear();
        self.status = crate::lang::tf(k::MSG_MACRO_IMPORTED, &[("{ok}", &ok.len().to_string())]);
        let line = self.status.clone();
        self.log(&line);
        true
    }

    fn mutex_button_ui(&mut self, ui: &mut egui::Ui, cmd: &str) {
        if ui.button(t(k::UI_BTN_MUTEX)).clicked() && self.mutex_editor_open.as_deref() != Some(cmd)
        {
            self.mutex_editor_open = Some(cmd.to_string());
            self.mutex_edit_for = None; // Force the buffer to reload for this command
        }
        let n = self.mutex_set_of(cmd).len();
        if n > 0 {
            super::pages::count_badge(ui, n);
        }
    }

    pub(super) fn draw_mutex_editor(&mut self, ctx: &egui::Context) {
        let Some(cmd) = self.mutex_editor_open.clone() else {
            return;
        };
        // The command was deleted (not found in cfg) -> close the window automatically (same guard as the macro editor)
        if self.cfg.get("commands").and_then(|v| v.get(&cmd)).is_none() {
            self.mutex_editor_open = None;
            self.mutex_edit_for = None;
            return;
        }
        if self.mutex_edit_for.as_deref() != Some(cmd.as_str()) {
            self.mutex_edit_for = Some(cmd.clone());
            self.mutex_edit = self.mutex_set_of(&cmd);
        }
        let phrase = self
            .cfg
            .get("commands")
            .and_then(|v| v.get(&cmd))
            .and_then(|c| c.get("phrase"))
            .and_then(|v| v.as_str())
            .filter(|s| !s.is_empty())
            .unwrap_or(&cmd)
            .to_string();
        // Candidate row snapshot (taken once before rendering, so the closure no longer borrows self.cfg)
        let mut rows: Vec<MutexRow> = Vec::new();
        // Entries of the top dropdown (**including itself**: it is a "current command" selector, not the candidate table)
        let mut all_cmds: Vec<(String, String)> = Vec::new();
        if let Some(cmds) = self.cfg.get("commands").and_then(|v| v.as_object()) {
            for (id, c) in cmds {
                let display = c
                    .get("phrase")
                    .and_then(|v| v.as_str())
                    .filter(|s| !s.is_empty())
                    .unwrap_or(id)
                    .to_string();
                all_cmds.push((id.clone(), display.clone()));
                if *id == cmd {
                    continue; // Exclude itself from the candidates (a mutex set never contains itself)
                }
                let mode = c
                    .get("mode")
                    .and_then(|v| v.as_str())
                    .unwrap_or("tap")
                    .to_string();
                let key = c
                    .get("key")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                rows.push(MutexRow {
                    id: id.clone(),
                    display,
                    mode,
                    key,
                });
            }
        }
        all_cmds.sort_by(|a, b| {
            a.1.to_lowercase()
                .cmp(&b.1.to_lowercase())
                .then_with(|| a.0.cmp(&b.0))
        });
        let checked: std::collections::BTreeSet<String> = self.mutex_edit.iter().cloned().collect();
        let (col, desc) = self.mutex_sort.unwrap_or((MutexCol::Name, false));
        sort_mutex_rows(&mut rows, &checked, self.mutex_pin_checked, col, desc);
        let mut open = true; // Window close button (egui::Window::open)
        let mut do_save = false;
        let mut do_close = false;
        // The command selected in the top dropdown: only recorded this frame, switched after the window closes (switching writes to disk and reloads buffers)
        let mut switch_to: Option<String> = None;
        egui::Window::new(crate::tfmt!(k::UI_TTL_MUTEX, phrase))
            .id(egui::Id::new("mutex_win"))
            .open(&mut open)
            .collapsible(false)
            .resizable(true)
            .default_size(egui::vec2(520.0, 460.0))
            .min_size(egui::vec2(420.0, 280.0))
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.label(t(k::UI_LBL_CMD));
                    let mut sel = cmd.clone();
                    let cur_text = all_cmds
                        .iter()
                        .find(|(id, _)| *id == cmd)
                        .map(|(_, d)| d.clone())
                        .unwrap_or_else(|| cmd.clone());
                    egui::ComboBox::from_id_salt("mutex_cmd_select")
                        .selected_text(cur_text)
                        // ComboBox::width is the minimum of "total width − 2×padding"; compensate to an actual width of ~190
                        .width(190.0 + 2.0 * ui.spacing().button_padding.x)
                        .show_ui(ui, |ui| {
                            for (id, disp) in &all_cmds {
                                ui.selectable_value(&mut sel, id.clone(), disp);
                            }
                        });
                    if sel != cmd {
                        switch_to = Some(sel);
                    }
                    ui.checkbox(&mut self.mutex_pin_checked, t(k::UI_CHK_PIN_CHECKED));
                });
                ui.label(t(k::UI_HINT_MUTEX));
                ui.separator();
                if rows.is_empty() {
                    ui.weak(t(k::UI_HINT_NO_OTHER_CMDS));
                } else {
                    let list_h = (ui.available_height() - MUTEX_FOOTER_H).max(120.0);
                    egui::ScrollArea::vertical()
                        .id_salt("mutex_rows_scroll")
                        .auto_shrink([false, false])
                        .max_height(list_h)
                        .show(ui, |ui| {
                            egui::Grid::new("mutex_grid")
                                .striped(true)
                                .num_columns(4)
                                .min_col_width(60.0)
                                .show(ui, |ui| {
                                    ui.weak(t(k::UI_TBL_EXCLUSIVE));
                                    sort_header(
                                        ui,
                                        &t(k::UI_TBL_NAME),
                                        MutexCol::Name,
                                        &mut self.mutex_sort,
                                    );
                                    sort_header(
                                        ui,
                                        &t(k::UI_TBL_TRIGGER_MODE),
                                        MutexCol::Mode,
                                        &mut self.mutex_sort,
                                    );
                                    sort_header(
                                        ui,
                                        &t(k::UI_TBL_KEY_MAPPING),
                                        MutexCol::Key,
                                        &mut self.mutex_sort,
                                    );
                                    ui.end_row();
                                    for r in &rows {
                                        let mut on = self.mutex_edit.iter().any(|x| x == &r.id);
                                        if ui.checkbox(&mut on, "").changed() {
                                            self.mutex_edit.retain(|x| x != &r.id);
                                            if on {
                                                self.mutex_edit.push(r.id.clone());
                                            }
                                        }
                                        ui.label(&r.display);
                                        if r.mode == "macro" {
                                            ui.weak(t(k::UI_OPT_MACRO));
                                        } else {
                                            ui.weak(mode_label(&r.mode));
                                        }
                                        if r.mode == "macro" || r.key.is_empty() {
                                            ui.weak("—");
                                        } else {
                                            ui.monospace(&r.key);
                                        }
                                        ui.end_row();
                                    }
                                });
                        });
                }
                ui.separator();
                ui.horizontal(|ui| {
                    if ui
                        .add(egui::Button::new(
                            egui::RichText::new(t(k::UI_BTN_SAVE_EXCLUSION))
                                .color(egui::Color32::from_rgb(76, 175, 80)),
                        ))
                        .clicked()
                    {
                        do_save = true;
                    }
                    if ui.button(t(k::UI_BTN_CLOSE)).clicked() {
                        do_close = true;
                    }
                    ui.weak(crate::lang::tf(
                        k::MSG_MUTEX_SELECTED,
                        &[("{n}", &self.mutex_edit.len().to_string())],
                    ));
                });
            });
        if let Some(new_cmd) = switch_to {
            self.switch_mutex_editor_to(&new_cmd);
        } else if do_save {
            self.apply_mutex_edit(&cmd);
        }
        if do_close {
            open = false;
        }
        if !open {
            if self.mutex_dirty(&cmd) {
                self.apply_mutex_edit(&cmd);
            }
            self.mutex_editor_open = None;
            self.mutex_edit_for = None;
        }
    }

    /// Actual execution of the confirmation dialogs (separate functions: draw_modals renders the shell, probes test the chain directly)
    pub(super) fn exec_confirm(&mut self, conf: Confirm) {
        match conf {
            Confirm::RegenAll { .. } => self.start_proto_full(),
            // "Optimise recognition": persists and prompts for a restart; it does not regenerate on the spot (both directions alike).
            // Either direction invalidates the templates (the cache version includes the switch) and a full rebuild happens at load time, at equal cost;
            // and this process's front end is locked by a OnceLock (udkws_v2::frame_mean_norm has no setter), so
            // recomputing now would inevitably use the old one and force another full pass after restart -- one wasted full encode.
            Confirm::BoostRecognition { enable } => {
                self.settings.frame_mean_norm = enable;
                self.persist_settings();
                self.status = t(k::MSG_BOOST_SWITCH_SAVED);
            }
            Confirm::NoiseModel { enable } => {
                self.settings.noise_model = enable;
                self.persist_settings();
                self.status = t(k::MSG_NOISE_MODEL_SAVED);
            }
        }
    }

    /// Destructive-action confirmation and rename dialogs (called every frame from update, single entry point)
    pub(super) fn draw_modals(&mut self, ctx: &egui::Context) {
        if self.about_open {
            let ver = env!("CARGO_PKG_VERSION");
            egui::Window::new(t(k::UI_TTL_ABOUT))
                .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
                .collapsible(false)
                .resizable(false)
                .show(ctx, |ui| {
                    ui.label(egui::RichText::new(crate::tfmt!(k::UI_APP_NAME_VER, ver)).strong());
                    ui.add_space(4.0);
                    ui.label(t(k::UI_HINT_APP_TAGLINE));
                    ui.add_space(4.0);
                    ui.horizontal(|ui| {
                        ui.label(t(k::UI_LBL_PRIVACY_EMAIL));
                        ui.hyperlink_to(t(k::UI_BTN_PRIVACY_POLICY), crate::update::PRIVACY_URL);
                        ui.label("·");
                        ui.hyperlink_to("shmtty@qq.com", "mailto:shmtty@qq.com");
                    });
                    ui.horizontal(|ui| {
                        ui.label(t(k::UI_LBL_SUPPORT_DEV));
                        ui.hyperlink_to("afdian.com/a/shmtty", "https://afdian.com/a/shmtty");
                    });
                    ui.add_space(4.0);
                    ui.weak(t(k::UI_HINT_PRIVACY_LOCAL));
                    {
                        ui.add_space(2.0);
                        if let Some(new_ver) = &self.latest_update {
                            ui.horizontal(|ui| {
                                ui.label(crate::tfmt!(k::MSG_UPDATE_AVAILABLE_SHORT, new_ver));
                                ui.hyperlink_to(
                                    t(k::UI_BTN_DOWNLOAD_PAGE),
                                    crate::update::DOWNLOAD_URL,
                                );
                            });
                        }
                        ui.horizontal(|ui| {
                            if ui
                                .add_enabled(
                                    !self.update_check_inflight,
                                    egui::Button::new(t(k::UI_BTN_CHECK_UPDATE)),
                                )
                                .clicked()
                            {
                                self.update_check_inflight = true;
                                spawn_update_check(&self.update_check_tx);
                            }
                            if self.update_check_inflight {
                                ui.spinner();
                            }
                        });
                    }
                    ui.separator();
                    if ui.button(t(k::UI_BTN_CLOSE)).clicked() {
                        self.about_open = false;
                    }
                });
        }

        // -- Confirmation dialog --
        if let Some(conf) = self.confirm.clone() {
            let (title, body) = match &conf {
                Confirm::RegenAll { cmds, recs } => (
                    t(k::UI_BTN_REBUILD_TEMPLATE),
                    crate::tfmt!(k::MSG_CONFIRM_REBUILD_ALL, cmds, recs),
                ),
                Confirm::BoostRecognition { .. } => (
                    t(k::UI_CHK_BOOST_RECOGNITION),
                    t(k::MSG_CONFIRM_BOOST_REGEN),
                ),
                Confirm::NoiseModel { .. } => {
                    (t(k::UI_CHK_NOISE_MODEL), t(k::MSG_CONFIRM_NOISE_MODEL))
                }
            };
            let mut exec = false;
            egui::Window::new(title)
                .id(egui::Id::new("confirm_modal"))
                .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
                .collapsible(false)
                .resizable(false)
                .show(ctx, |ui| {
                    ui.label(body);
                    ui.separator();
                    ui.horizontal(|ui| {
                        if ui.button(t(k::UI_BTN_CONFIRM)).clicked() {
                            exec = true;
                        }
                        if ui.button(t(k::UI_BTN_CANCEL)).clicked() {
                            self.confirm = None;
                        }
                    });
                });
            if exec {
                self.exec_confirm(conf);
                self.confirm = None;
            }
        }
        if self.group_new_open {
            let mut apply = false;
            egui::Window::new(t(k::UI_BTN_NEW_GROUP))
                .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
                .collapsible(false)
                .resizable(false)
                .show(ctx, |ui| {
                    ui.horizontal(|ui| {
                        ui.label(t(k::UI_LBL_GROUP_NAME));
                        let resp = ui.text_edit_singleline(&mut self.new_group_name);
                        if resp.lost_focus() && ctx.input(|i| i.key_pressed(egui::Key::Enter)) {
                            apply = true;
                        }
                    });
                    ui.separator();
                    ui.horizontal(|ui| {
                        if ui.button(t(k::UI_BTN_OK)).clicked() {
                            apply = true;
                        }
                        if ui.button(t(k::UI_BTN_CANCEL)).clicked() {
                            self.group_new_open = false;
                            self.new_group_name.clear();
                        }
                    });
                });
            if apply {
                let name = self.new_group_name.trim().to_string();
                if !name.is_empty() {
                    self.save_new_group(&name);
                }
                self.group_new_open = false;
                self.new_group_name.clear();
            }
        }
        // -- Rename dialog --
        if let Some(ren) = self.rename.clone() {
            let title = match &ren {
                Rename::Group(_) => t(k::UI_TTL_RENAME_GROUP),
                Rename::CommandPhrase(_) => t(k::UI_BTN_RENAME_CMD),
            };
            let mut apply = false;
            egui::Window::new(title)
                .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
                .collapsible(false)
                .resizable(false)
                .show(ctx, |ui| {
                    ui.horizontal(|ui| {
                        ui.label(t(k::UI_LBL_NEW_NAME));
                        let resp = ui.text_edit_singleline(&mut self.rename_buf);
                        if resp.lost_focus() && ctx.input(|i| i.key_pressed(egui::Key::Enter)) {
                            apply = true;
                        }
                    });
                    ui.separator();
                    ui.horizontal(|ui| {
                        if ui.button(t(k::UI_BTN_OK)).clicked() {
                            apply = true;
                        }
                        if ui.button(t(k::UI_BTN_CANCEL)).clicked() {
                            self.rename = None;
                        }
                    });
                });
            if apply {
                let new = self.rename_buf.trim().to_string();
                match ren {
                    Rename::Group(old) => self.rename_group(&old, &new),
                    Rename::CommandPhrase(cmd) => {
                        if !new.is_empty() {
                            self.set_cmd_field(
                                &cmd,
                                "phrase",
                                serde_json::Value::String(new.clone()),
                            );
                            self.persist_cfg();
                            self.status = crate::tfmt!(k::MSG_CMD_DISPLAY_RENAMED, cmd, new);
                            self.log(&crate::tfmt!(k::MSG_CMD_RENAME_LOG, cmd, new));
                        }
                    }
                }
                self.rename = None;
            }
        }
        // -- Start-up fatal error box (v2-only: refusing to start when construction fails; the real reason is shown in the box) --
        if let Some(err) = self.fatal_error.clone() {
            let mut ack = false;
            egui::Window::new(t(k::UI_TTL_STARTUP_FAILED))
                .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
                .collapsible(false)
                .resizable(false)
                .show(ctx, |ui| {
                    ui.colored_label(WARN_COLOR, egui::RichText::new(&err));
                    ui.separator();
                    ui.horizontal(|ui| {
                        if ui.button(t(k::UI_BTN_GOT_IT)).clicked() {
                            ack = true;
                        }
                    });
                });
            if ack {
                self.fatal_error = None;
            }
        }
    }
}
