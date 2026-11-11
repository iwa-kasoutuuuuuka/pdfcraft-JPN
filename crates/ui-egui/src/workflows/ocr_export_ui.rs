//! Scan & OCR ▸ Export recognized text: the dialog (format, pages, vertical text, read scanned
//! pages first) and the background export of the document's text as a .txt or a Word file.
//!
//! The export itself is `pdfcraft_engine::text_export`; this is the dialog, the save panel and the
//! worker thread. Progress and the result go through the same notice-bar channel as the other
//! exports (`export_status`).

use std::sync::{Arc, Mutex};

use egui::{Align, Layout};
use pdfcraft_engine::text_export::{PageSeparator, TextExportOptions, TextFormat, VerticalMode};

use crate::file_filter::AddFilterAnyCase as _;
use crate::marks_ui::PageRange;
use crate::theme::{self, Tokens};
use crate::{PdfCraftApp, widgets};

/// The dialog's choices, kept between uses.
#[derive(Clone, Debug, PartialEq)]
pub struct OcrExportDraft {
    pub format: TextFormat,
    pub range: PageRange,
    pub vertical: VerticalMode,
    pub separator: PageSeparator,
    /// Read the pages that have no text layer first.
    pub ocr_if_needed: bool,
}

impl Default for OcrExportDraft {
    fn default() -> Self {
        Self {
            format: TextFormat::Docx,
            range: PageRange::default(),
            vertical: VerticalMode::Auto,
            separator: PageSeparator::Line,
            ocr_if_needed: true,
        }
    }
}

fn vertical_label(v: VerticalMode) -> &'static str {
    match v {
        VerticalMode::Auto => tl!("Automatic (when the page's text is mostly vertical)"),
        VerticalMode::On => tl!("Always vertical"),
        VerticalMode::Off => tl!("Never vertical"),
    }
}

/// The dialog body; returns (export, cancel).
pub(crate) fn body(ui: &mut egui::Ui, app: &mut PdfCraftApp, t: &Tokens) -> (bool, bool) {
    let count = app.active_ids().and_then(|(_, id)| app.session.get(id)).map_or(0, |d| d.info.pages.len());
    let engine_problem = pdfcraft_engine::ocr::engine_unavailable_reason(app.ocr_draft.engine);
    let d = &mut app.ocr_export_draft;
    ui.label(egui::RichText::new(tl!("Export Recognized Text")).font(theme::semibold(18.0)));
    ui.add_space(4.0);
    ui.label(
        egui::RichText::new(tl!("The text of the pages in reading order; vertical Japanese columns are read right to left."))
            .small()
            .color(t.text_faint),
    );
    ui.add_space(8.0);
    ui.horizontal(|ui| {
        ui.label(tl!("Format"));
        widgets::radio_value(ui, &mut d.format, TextFormat::Docx, tl!("Word (.docx)"));
        widgets::radio_value(ui, &mut d.format, TextFormat::Txt, tl!("Text (.txt)"));
    });
    if d.format == TextFormat::Docx {
        ui.horizontal(|ui| {
            ui.label(tl!("Vertical text"));
            craft_ui::ComboBox::from_id_salt("ocr-export-vertical").selected_text(vertical_label(d.vertical)).show_ui(ui, |ui| {
                for v in [VerticalMode::Auto, VerticalMode::On, VerticalMode::Off] {
                    ui.selectable_value(&mut d.vertical, v, vertical_label(v));
                }
            });
        });
    } else {
        ui.horizontal(|ui| {
            ui.label(tl!("Pages are separated by"));
            widgets::radio_value(ui, &mut d.separator, PageSeparator::Line, tl!("a line with the page number"));
            widgets::radio_value(ui, &mut d.separator, PageSeparator::FormFeed, tl!("form feeds"));
        });
    }
    ui.add_space(6.0);
    ui.add(widgets::checkbox(&mut d.ocr_if_needed, tl!("Recognize pages that have no text first (Japanese)")))
        .on_hover_text(tl!("Pages that are pictures are read with the engine and language chosen in Recognize Text; the document is not changed"));
    if d.ocr_if_needed
        && let Some(why) = engine_problem
    {
        ui.label(egui::RichText::new(crate::i18n::t(&why).to_string()).small().color(t.text_muted));
    }
    ui.add_space(6.0);
    ui.label(egui::RichText::new(tl!("Pages")).font(theme::semibold(12.5)));
    d.range.ui(ui, count);
    ui.add_space(12.0);
    let (mut go, mut cancel) = (false, false);
    ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
        let ok = !d.range.pages(count).is_empty();
        if ui.add_enabled_ui(ok, |ui| widgets::pill_button(ui, tl!("Export"), true)).inner.clicked() {
            go = true;
        }
        if widgets::pill_button(ui, tl!("Cancel"), false).clicked() {
            cancel = true;
        }
    });
    (go, cancel)
}

/// The OCR settings of the export: what Recognize Text shows, but Japanese and the automatic
/// engine when the draft was never changed from its English defaults.
fn ocr_settings(app: &PdfCraftApp) -> pdfcraft_engine::ocr::OcrSettings {
    app.ocr_draft.settings()
}

impl PdfCraftApp {
    /// Ask where to save, then export the active document's text in the background.
    pub fn start_ocr_export(&mut self) {
        if !self.commit_form_typing() {
            return;
        }
        let Some((_, id)) = self.active_ids() else { return };
        if !self.may_copy(id) {
            return;
        }
        let Some(doc) = self.session.get(id) else { return };
        let src = doc.export_source();
        let name = doc.name.clone();
        let stem = name.trim_end_matches(".pdf").trim_end_matches(".PDF").to_string();
        let draft = self.ocr_export_draft.clone();
        let pages = draft.range.pages(src.pages);
        if pages.is_empty() {
            return;
        }
        let options = TextExportOptions {
            format: draft.format,
            vertical: draft.vertical,
            separator: draft.separator,
            ocr_if_needed: draft.ocr_if_needed,
            ocr: ocr_settings(self),
            title: stem.clone(),
            language: None,
        };
        let status: crate::export_ui::ExportStatus = Arc::new(Mutex::new(Some((0, pages.len(), None))));
        let inline = self.run_inline;
        let file_name = format!("{stem}.{}", draft.format.extension());
        let lang = crate::i18n::current();
        let start = move |app: &mut Self, path: std::path::PathBuf| {
            let st = status.clone();
            let shown = path.display().to_string();
            let work = move || {
                crate::i18n::set_current(lang);
                let progress_status = st.clone();
                let total = pages.len();
                let result = pdfcraft_engine::guard(|| {
                    pdfcraft_engine::text_export::export_text(src, &name, &pages, &options, |done, of| {
                        if let Ok(mut s) = progress_status.lock() {
                            *s = Some((done.min(of), total.max(of), None));
                        }
                        true
                    })
                })
                .unwrap_or_else(Err);
                let msg = match result {
                    Ok(out) => match crate::editing::write_atomically(&path.to_string_lossy(), &out.bytes) {
                        Ok(()) => {
                            let mut msg =
                                crate::i18n::fmt(tl!("Exported the text of {n} pages to {path}"), &[("n", &out.pages.to_string()), ("path", &shown)]);
                            if !out.ocr_pages.is_empty() {
                                msg.push_str(&crate::i18n::fmt(tl!("; {n} pages were recognized first"), &[("n", &out.ocr_pages.len().to_string())]));
                            }
                            if !out.empty_pages.is_empty() {
                                msg.push_str(&crate::i18n::fmt(tl!("; {n} pages have no text"), &[("n", &out.empty_pages.len().to_string())]));
                            }
                            msg
                        }
                        Err(e) => crate::i18n::fmt(tl!("Export stopped: {e}"), &[("e", &e.to_string())]),
                    },
                    Err(e) => crate::i18n::fmt(tl!("Export stopped: {e}"), &[("e", &e)]),
                };
                if let Ok(mut s) = st.lock() {
                    let (done, total) = s.as_ref().map_or((0, 0), |(d, t, _)| (*d, *t));
                    *s = Some((done.max(total), total, Some(msg)));
                }
            };
            if inline {
                work();
            } else {
                std::thread::Builder::new().name("pdfcraft-ocr-export".into()).spawn(work).ok();
            }
            app.export_status = Some(status);
        };
        let filter_name = if draft.format == TextFormat::Docx { "Word" } else { "Text" };
        let dialog = rfd::AsyncFileDialog::new()
            .add_filter_any_case(filter_name, &[draft.format.extension()])
            .set_title(tl!("Save the text as").to_string())
            .set_file_name(file_name);
        self.ask_one(crate::pickers::Ask::Save(dialog), None, move |app, path| start(app, path));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_draft_starts_with_word_and_reading_scanned_pages() {
        let d = OcrExportDraft::default();
        assert_eq!((d.format, d.vertical, d.ocr_if_needed), (TextFormat::Docx, VerticalMode::Auto, true));
    }
}
