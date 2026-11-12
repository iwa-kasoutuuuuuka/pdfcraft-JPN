//! Scan & OCR ▸ OCR a folder: the dialog (folder, subfolders, where results go, what to skip), the
//! background run with its progress, and the summary.
//!
//! The run is `pdfcraft_engine::ocr_folder`; the engine, language and resolution are the ones
//! chosen in Recognize Text. The worker reports through a shared [`FolderRun`], which the dialog
//! reads while it is open and [`PdfCraftApp::poll_ocr_folder`] reads when it is not.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use egui::{Align, Layout};
use pdfcraft_engine::ocr_folder::{FileStatus, FolderOptions, FolderReport};

use crate::theme::{self, Tokens};
use crate::{PdfCraftApp, widgets};

/// Where the results go.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FolderOutput {
    /// Next to each source, with a suffix on the name.
    Beside,
    /// Into a folder, under the same names.
    Folder,
}

/// The dialog's choices and the run.
pub struct OcrFolderState {
    pub folder: Option<PathBuf>,
    pub recursive: bool,
    pub output: FolderOutput,
    pub output_dir: Option<PathBuf>,
    pub suffix: String,
    pub skip_with_text: bool,
    pub overwrite: bool,
    pub run: Option<Arc<Mutex<FolderRun>>>,
    /// The last finished run, shown until the next starts.
    pub report: Option<Result<FolderReport, String>>,
}

impl Default for OcrFolderState {
    fn default() -> Self {
        Self {
            folder: None,
            recursive: true,
            output: FolderOutput::Beside,
            output_dir: None,
            suffix: pdfcraft_engine::ocr_folder::DEFAULT_SUFFIX.into(),
            skip_with_text: true,
            overwrite: false,
            run: None,
            report: None,
        }
    }
}

/// What the worker shares with the interface.
#[derive(Default)]
pub struct FolderRun {
    pub done: usize,
    pub total: usize,
    pub current: String,
    pub cancel: bool,
    pub result: Option<Result<FolderReport, String>>,
}

/// One line of the summary.
pub(crate) fn summary_line(report: &FolderReport) -> String {
    let mut msg = crate::i18n::fmt(
        tl!("{ok} files made searchable, {skipped} skipped, {failed} failed"),
        &[("ok", &report.ok.to_string()), ("skipped", &report.skipped.to_string()), ("failed", &report.failed.to_string())],
    );
    if report.cancelled {
        msg.push_str(tl!("; stopped before the end"));
    }
    msg
}

fn file_name(path: &std::path::Path) -> String {
    path.file_name().map_or_else(|| path.to_string_lossy().into_owned(), |n| n.to_string_lossy().into_owned())
}

/// Dialog body; returns true to close.
pub(crate) fn body(ui: &mut egui::Ui, app: &mut PdfCraftApp, t: &Tokens) -> bool {
    ui.set_width(560.0);
    ui.label(egui::RichText::new(tl!("OCR a Folder")).font(theme::semibold(18.0)));
    ui.add_space(4.0);
    ui.label(
        egui::RichText::new(tl!(
            "Makes every PDF in a folder searchable, one file at a time. The language and engine are the ones in Recognize Text."
        ))
        .small()
        .color(t.text_faint),
    );
    ui.add_space(8.0);
    // Running: progress and Cancel.
    if let Some(run) = app.ocr_folder.run.clone() {
        let (done, total, current, finished) =
            run.lock().map_or((0, 0, String::new(), true), |r| (r.done, r.total, r.current.clone(), r.result.is_some()));
        if !finished {
            let fraction = if total == 0 { 0.0 } else { done as f32 / total as f32 };
            ui.add(egui::ProgressBar::new(fraction).text(crate::i18n::fmt(
                tl!("File {d} of {t}"),
                &[("d", &(done + 1).min(total.max(1)).to_string()), ("t", &total.max(1).to_string())],
            )));
            ui.label(egui::RichText::new(current).small().color(t.text_muted));
            ui.add_space(8.0);
            let mut close = false;
            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                if widgets::pill_button(ui, tl!("Stop"), false).clicked()
                    && let Ok(mut r) = run.lock()
                {
                    r.cancel = true;
                }
                if widgets::pill_button(ui, tl!("Close"), false).on_hover_text(tl!("The run continues in the background")).clicked() {
                    close = true;
                }
            });
            ui.ctx().request_repaint_after(std::time::Duration::from_millis(200));
            return close;
        }
        app.poll_ocr_folder();
    }
    // Finished: the summary.
    if let Some(report) = app.ocr_folder.report.clone() {
        match report {
            Err(e) => {
                ui.label(egui::RichText::new(e).color(t.text));
            }
            Ok(report) => {
                ui.label(egui::RichText::new(summary_line(&report)).font(theme::semibold(13.0)));
                ui.add_space(4.0);
                egui::ScrollArea::vertical().max_height(260.0).show(ui, |ui| {
                    for f in &report.files {
                        let (mark, detail) = match &f.status {
                            FileStatus::Ok { output, words } => (
                                tl!("Done"),
                                crate::i18n::fmt(
                                    tl!("{words} words, saved as {name}"),
                                    &[("words", &words.to_string()), ("name", &file_name(output))],
                                ),
                            ),
                            FileStatus::Skipped(why) => (tl!("Skipped"), why.clone()),
                            FileStatus::Failed(why) => (tl!("Failed"), why.clone()),
                        };
                        ui.horizontal_wrapped(|ui| {
                            ui.label(egui::RichText::new(mark).strong());
                            ui.label(file_name(&f.path));
                            ui.label(egui::RichText::new(detail).small().color(t.text_muted));
                        });
                    }
                });
            }
        }
        ui.add_space(8.0);
        let mut close = false;
        ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
            if widgets::pill_button(ui, tl!("Close"), true).clicked() {
                close = true;
            }
            if widgets::pill_button(ui, tl!("Another folder"), false).clicked() {
                app.ocr_folder.report = None;
            }
        });
        return close;
    }
    // Options.
    let mut pick_folder = false;
    let mut pick_output = false;
    let mut start = false;
    let mut close = false;
    let s = &mut app.ocr_folder;
    ui.horizontal(|ui| {
        ui.label(tl!("Folder"));
        match &s.folder {
            Some(f) => ui.monospace(f.display().to_string()),
            None => ui.label(egui::RichText::new(tl!("None chosen")).color(t.text_muted)),
        };
        if widgets::pill_button(ui, tl!("Choose folder…"), false).clicked() {
            pick_folder = true;
        }
    });
    ui.add(widgets::checkbox(&mut s.recursive, tl!("Include subfolders")));
    ui.add_space(6.0);
    ui.label(egui::RichText::new(tl!("Save the searchable copies")).font(theme::semibold(12.5)));
    widgets::radio_value(ui, &mut s.output, FolderOutput::Beside, tl!("Next to each original, with a suffix on the name"));
    ui.add_enabled_ui(s.output == FolderOutput::Beside, |ui| {
        ui.horizontal(|ui| {
            ui.add_space(20.0);
            ui.label(tl!("Suffix"));
            ui.add(egui::TextEdit::singleline(&mut s.suffix).desired_width(120.0));
        });
    });
    widgets::radio_value(ui, &mut s.output, FolderOutput::Folder, tl!("In another folder, under the same names"));
    ui.add_enabled_ui(s.output == FolderOutput::Folder, |ui| {
        ui.horizontal(|ui| {
            ui.add_space(20.0);
            match &s.output_dir {
                Some(f) => ui.monospace(f.display().to_string()),
                None => ui.label(egui::RichText::new(tl!("None chosen")).color(t.text_muted)),
            };
            if widgets::pill_button(ui, tl!("Choose folder…"), false).clicked() {
                pick_output = true;
            }
        });
    });
    ui.add_space(6.0);
    ui.add(widgets::checkbox(&mut s.skip_with_text, tl!("Skip files that already have text")));
    ui.add(widgets::checkbox(&mut s.overwrite, tl!("Replace results that already exist (the originals are never replaced)")))
        .on_hover_text(tl!("Without this, a result that already exists is left alone and the file is reported as skipped"));
    ui.add_space(4.0);
    ui.label(
        egui::RichText::new(tl!("Password-protected and damaged files are skipped or reported; the rest are still done."))
            .small()
            .color(t.text_faint),
    );
    let engine_problem = pdfcraft_engine::ocr::engine_unavailable_reason(app.ocr_draft.engine);
    if let Some(why) = &engine_problem {
        ui.label(egui::RichText::new(crate::i18n::t(why).to_string()).small().color(t.text_muted));
    }
    ui.add_space(8.0);
    let s = &app.ocr_folder;
    let ready = s.folder.is_some() && (s.output == FolderOutput::Beside || s.output_dir.is_some());
    ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
        if ui.add_enabled_ui(ready, |ui| widgets::pill_button(ui, tl!("Start"), true)).inner.clicked() {
            start = true;
        }
        if widgets::pill_button(ui, tl!("Cancel"), false).clicked() {
            close = true;
        }
    });
    if pick_folder {
        let dialog = rfd::AsyncFileDialog::new().set_title(tl!("Choose the folder of PDF files").to_string());
        app.ask_one(crate::pickers::Ask::Folder(dialog), None, |app, dir| app.ocr_folder.folder = Some(dir));
    }
    if pick_output {
        let dialog = rfd::AsyncFileDialog::new().set_title(tl!("Choose a folder for the searchable files").to_string());
        app.ask_one(crate::pickers::Ask::Folder(dialog), None, |app, dir| app.ocr_folder.output_dir = Some(dir));
    }
    if start {
        app.start_ocr_folder();
        close = false;
    }
    close
}

/// Whether two paths name the same folder (resolved when both exist, as written otherwise).
fn same_folder(a: &std::path::Path, b: &std::path::Path) -> bool {
    match (std::fs::canonicalize(a), std::fs::canonicalize(b)) {
        (Ok(x), Ok(y)) => x == y,
        _ => a == b,
    }
}

impl PdfCraftApp {
    /// Start the run with the dialog's choices, on a worker thread (inline in tests).
    pub fn start_ocr_folder(&mut self) {
        if self.ocr_folder.run.as_ref().is_some_and(|r| r.lock().is_ok_and(|r| r.result.is_none())) {
            self.notify_tr("Text recognition is already running");
            return;
        }
        let s = &self.ocr_folder;
        let Some(folder) = s.folder.clone() else { return };
        let mut options = FolderOptions::new(folder, self.ocr_draft.settings());
        options.recursive = s.recursive;
        options.skip_with_text = s.skip_with_text;
        options.overwrite = s.overwrite;
        // The dialog promises the originals are never replaced: refuse the two set-ups whose results
        // would land on the sources themselves.
        let onto_sources = match s.output {
            FolderOutput::Beside => s.suffix.trim().is_empty(),
            FolderOutput::Folder => s.output_dir.as_deref().is_some_and(|out| same_folder(out, &options.folder)),
        };
        if onto_sources {
            self.notify_tr("The results would replace the original files: add a suffix or choose another output folder");
            return;
        }
        match s.output {
            FolderOutput::Beside => options.suffix = Some(s.suffix.trim().to_string()),
            FolderOutput::Folder => {
                options.output_dir = s.output_dir.clone();
                options.suffix = Some(String::new());
            }
        }
        // The draft's force option reads even files that have text.
        if options.settings.force_ocr {
            options.skip_with_text = false;
        }
        let run = Arc::new(Mutex::new(FolderRun::default()));
        self.ocr_folder.report = None;
        self.ocr_folder.run = Some(run.clone());
        let lang = crate::i18n::current();
        let work = move || {
            crate::i18n::set_current(lang);
            let progress = run.clone();
            let outcome = pdfcraft_engine::guard(|| {
                pdfcraft_engine::ocr_folder::run(
                    &options,
                    |path, bytes| {
                        if let Some(dir) = path.parent() {
                            std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
                        }
                        crate::editing::write_atomically(&path.to_string_lossy(), bytes).map_err(|e| format!("{}: {e}", path.display()))
                    },
                    |p| match progress.lock() {
                        Ok(mut r) => {
                            r.done = p.done;
                            r.total = p.total;
                            r.current = file_name(p.current);
                            !r.cancel
                        }
                        Err(_) => false,
                    },
                )
            })
            .unwrap_or_else(Err);
            if let Ok(mut r) = run.lock() {
                r.result = Some(outcome);
            }
        };
        if self.run_inline {
            work();
        } else {
            let spawned = std::thread::Builder::new().name("pdfcraft-ocr-folder".into()).spawn(work);
            if spawned.is_err() {
                // No worker: clear the run so the dialog does not wait for one forever.
                self.ocr_folder.run = None;
                self.notify_tr("Could not start text recognition");
                return;
            }
        }
        self.poll_ocr_folder();
    }

    /// Take a finished run's result: the summary for the dialog, a notice if it is not open.
    pub(crate) fn poll_ocr_folder(&mut self) {
        let Some(run) = self.ocr_folder.run.clone() else { return };
        let result = run.lock().ok().and_then(|mut r| r.result.take());
        let Some(result) = result else {
            if let Some(ctx) = &self.ctx {
                ctx.request_repaint_after(std::time::Duration::from_millis(250));
            }
            return;
        };
        self.ocr_folder.run = None;
        let notice = match &result {
            Ok(report) => summary_line(report),
            Err(e) => e.clone(),
        };
        self.ocr_folder.report = Some(result);
        if !matches!(self.dialog, Some(crate::Dialog::OcrFolder)) {
            self.notify(notice);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_defaults_write_beside_the_originals_with_the_suffix() {
        let s = OcrFolderState::default();
        assert_eq!((s.output, s.suffix.as_str(), s.overwrite, s.skip_with_text), (FolderOutput::Beside, "_ocr", false, true));
    }
}
