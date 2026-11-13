//! Scan & OCR ▸ Export tables to CSV: the dialog (pages, one file per table or all in one, read
//! scanned pages first) and the background export of the tables found on the pages as CSV.
//!
//! The detection and the CSV are `pdfcraft_engine::table_csv`; this is the dialog, the save panel
//! and the worker thread. Progress and the result go through the same notice-bar channel as the
//! other exports (`export_status`).

use std::sync::{Arc, Mutex};

use egui::{Align, Layout};
use pdfcraft_engine::table_csv::{CsvLayout, TableCsvOptions};

use crate::file_filter::AddFilterAnyCase as _;
use crate::marks_ui::PageRange;
use crate::theme::{self, Tokens};
use crate::{PdfCraftApp, widgets};

/// The dialog's choices, kept between uses.
#[derive(Clone, Debug, PartialEq)]
pub struct TableCsvDraft {
    pub layout: CsvLayout,
    pub range: PageRange,
    /// Use the ruled lines of drawn tables.
    pub use_rules: bool,
    /// Read the pages that have no text layer first.
    pub ocr_if_needed: bool,
}

impl Default for TableCsvDraft {
    fn default() -> Self {
        Self { layout: CsvLayout::PerTable, range: PageRange::default(), use_rules: true, ocr_if_needed: true }
    }
}

/// The dialog body; returns (export, cancel).
pub(crate) fn body(ui: &mut egui::Ui, app: &mut PdfCraftApp, t: &Tokens) -> (bool, bool) {
    let count = app.active_ids().and_then(|(_, id)| app.session.get(id)).map_or(0, |d| d.info.pages.len());
    let engine_problem = pdfcraft_engine::ocr::engine_unavailable_reason(app.ocr_draft.engine);
    let d = &mut app.table_csv_draft;
    ui.label(egui::RichText::new(tl!("Export Tables to CSV")).font(theme::semibold(18.0)));
    ui.add_space(4.0);
    ui.label(
        egui::RichText::new(tl!("Finds tables from where the text sits on the page and saves them as CSV that Excel opens (UTF-8 with BOM)."))
            .small()
            .color(t.text_faint),
    );
    ui.add_space(8.0);
    ui.horizontal(|ui| {
        ui.label(tl!("Files"));
        widgets::radio_value(ui, &mut d.layout, CsvLayout::PerTable, tl!("One CSV per table"));
        widgets::radio_value(ui, &mut d.layout, CsvLayout::OneFile, tl!("All tables in one CSV"));
    });
    ui.add_space(6.0);
    ui.add(widgets::checkbox(&mut d.use_rules, tl!("Use the ruled lines of tables")))
        .on_hover_text(tl!("Where lines are drawn around the cells, they decide the cells; elsewhere the gaps between words do"));
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

/// Where the file of each table goes in [`CsvLayout::PerTable`]: beside `path`, named from its stem.
fn per_table_path(path: &std::path::Path, table: &pdfcraft_engine::table_csv::FoundTable) -> std::path::PathBuf {
    let stem = path.file_stem().map_or_else(|| "tables".to_string(), |s| s.to_string_lossy().into_owned());
    path.with_file_name(pdfcraft_engine::table_csv::table_file_name(&stem, table))
}

impl PdfCraftApp {
    /// Ask where to save, then export the tables of the active document in the background.
    pub fn start_table_csv_export(&mut self) {
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
        let draft = self.table_csv_draft.clone();
        let pages = draft.range.pages(src.pages);
        if pages.is_empty() {
            return;
        }
        let options =
            TableCsvOptions { layout: draft.layout, ocr_if_needed: draft.ocr_if_needed, ocr: self.ocr_draft.settings(), use_rules: draft.use_rules };
        let status: crate::export_ui::ExportStatus = Arc::new(Mutex::new(Some((0, pages.len(), None))));
        let inline = self.run_inline;
        let lang = crate::i18n::current();
        let start = move |app: &mut Self, path: std::path::PathBuf| {
            let st = status.clone();
            let work = move || {
                crate::i18n::set_current(lang);
                let progress_status = st.clone();
                let total = pages.len();
                let result = pdfcraft_engine::guard(|| {
                    pdfcraft_engine::table_csv::export_tables_csv(src, &name, &pages, &options, |done, of| {
                        if let Ok(mut s) = progress_status.lock() {
                            *s = Some((done.min(of), total.max(of), None));
                        }
                        true
                    })
                })
                .unwrap_or_else(Err);
                let msg = match result {
                    Ok(out) if out.tables.is_empty() => tl!("No tables were found on the chosen pages").to_string(),
                    Ok(out) => {
                        let targets: Vec<std::path::PathBuf> = match options.layout {
                            CsvLayout::OneFile => vec![path.clone()],
                            CsvLayout::PerTable => out.tables.iter().map(|t| per_table_path(&path, t)).collect(),
                        };
                        let written = targets
                            .iter()
                            .zip(&out.files)
                            .try_for_each(|(p, bytes)| crate::editing::write_atomically(&p.to_string_lossy(), bytes).map_err(|e| e.to_string()));
                        match written {
                            Ok(()) => {
                                let shown = match options.layout {
                                    CsvLayout::OneFile => path.display().to_string(),
                                    CsvLayout::PerTable => path.parent().map_or_else(String::new, |p| p.display().to_string()),
                                };
                                let mut msg = crate::i18n::fmt(
                                    tl!("Exported {n} tables as {files} CSV files to {path}"),
                                    &[("n", &out.tables.len().to_string()), ("files", &targets.len().to_string()), ("path", &shown)],
                                );
                                if !out.ocr_pages.is_empty() {
                                    msg.push_str(&crate::i18n::fmt(
                                        tl!("; {n} pages were recognized first"),
                                        &[("n", &out.ocr_pages.len().to_string())],
                                    ));
                                }
                                msg
                            }
                            Err(e) => crate::i18n::fmt(tl!("Export stopped: {e}"), &[("e", &e)]),
                        }
                    }
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
                std::thread::Builder::new().name("pdfcraft-table-csv".into()).spawn(work).ok();
            }
            app.export_status = Some(status);
        };
        let dialog = rfd::AsyncFileDialog::new()
            .add_filter_any_case("CSV", &["csv"])
            .set_title(tl!("Save the tables as").to_string())
            .set_file_name(format!("{stem}.csv"));
        self.ask_one(crate::pickers::Ask::Save(dialog), None, start);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pdfcraft_engine::table_csv::FoundTable;

    #[test]
    fn the_draft_starts_with_a_file_per_table_rules_and_reading_scanned_pages() {
        let d = TableCsvDraft::default();
        assert_eq!((d.layout, d.use_rules, d.ocr_if_needed), (CsvLayout::PerTable, true, true));
    }

    #[test]
    fn per_table_files_go_beside_the_chosen_file() {
        let t = FoundTable { page: 3, index: 2, rows: 2, columns: 2, ruled: false };
        let p = per_table_path(std::path::Path::new("dir/売上.csv"), &t);
        assert_eq!(p, std::path::Path::new("dir/売上_p3_t2.csv"));
    }
}
