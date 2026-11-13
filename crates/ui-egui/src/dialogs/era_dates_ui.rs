//! Edit ▸ Convert dates (和暦/西暦の一括変換): the dates the engine finds in the active document
//! (`Document::find_dates`), each with a check box, the calendar to convert to, and Convert, which
//! applies `Document::date_conversion` as one undoable edit and says what it left unchanged.

use pdfcraft_engine::DocId;
use pdfcraft_engine::era_dates::{Calendar, Conversion, DateSource, DocumentDate};

use crate::{PdfCraftApp, widgets};

#[derive(Default)]
pub(crate) struct Draft {
    to: Calendar,
    /// The document, its edit generation and the calendar the hits were found for.
    key: Option<(DocId, u64, Calendar)>,
    hits: Vec<DocumentDate>,
    /// Which hits to convert, by position in `hits`.
    picked: Vec<bool>,
    withheld: bool,
    /// What the last Convert did.
    message: Option<String>,
}

impl Draft {
    /// The calendar dates are converted to.
    pub(crate) fn set_target(&mut self, to: Calendar) {
        if self.to != to {
            self.to = to;
            self.message = None;
        }
    }
}

fn source_label(h: &DocumentDate) -> String {
    let page = h.page.map(|p| crate::i18n::fmt(tl!("Page {n}"), &[("n", &(p + 1).to_string())]));
    let kind = match h.source {
        DateSource::PageText { .. } => tl!("Page text"),
        DateSource::Comment { .. } => tl!("Comment"),
        DateSource::Field { .. } => tl!("Form field"),
    };
    match page {
        Some(p) => format!("{p} · {kind}"),
        None => kind.to_string(),
    }
}

/// Returns `true` when the dialog should close.
pub(crate) fn body(ui: &mut egui::Ui, app: &mut PdfCraftApp) -> bool {
    ui.heading(tl!("Convert dates (Japanese era / Western)"));
    let Some((_, id)) = app.active_ids() else { return true };
    let Some(doc) = app.session.get(id) else { return true };
    let draft = &mut app.date_convert_draft;
    ui.horizontal(|ui| {
        ui.label(tl!("Convert to"));
        let mut to = draft.to;
        for c in [Calendar::Japanese, Calendar::Western] {
            widgets::radio_value(ui, &mut to, c, tl!(c.label()));
        }
        draft.set_target(to);
    });
    let key = (id, doc.edit_generation(), draft.to);
    if draft.key != Some(key) {
        let search = doc.find_dates(draft.to, None);
        draft.picked = search.hits.iter().map(|h| h.conversion.text().is_some()).collect();
        draft.hits = search.hits;
        draft.withheld = search.page_text_withheld;
        draft.key = Some(key);
    }
    if draft.withheld {
        ui.label(tl!("The document's security doesn't allow copying, so its page text was not searched."));
    }
    let convertible = draft.hits.iter().filter(|h| h.conversion.text().is_some()).count();
    ui.label(crate::i18n::fmt(
        tl!("{found} dates found, {convertible} to convert"),
        &[("found", &draft.hits.len().to_string()), ("convertible", &convertible.to_string())],
    ));
    ui.separator();
    if draft.hits.is_empty() {
        ui.label(tl!("No dates found."));
    } else {
        egui::ScrollArea::vertical().max_height(360.0).auto_shrink([false, true]).show(ui, |ui| {
            egui::Grid::new("convert-dates").striped(true).num_columns(4).show(ui, |ui| {
                for (h, picked) in draft.hits.iter().zip(draft.picked.iter_mut()) {
                    let convertible = h.conversion.text().is_some();
                    ui.add_enabled(convertible, widgets::checkbox_without_text(picked));
                    ui.label(source_label(h));
                    ui.label(&h.original);
                    match &h.conversion {
                        Conversion::To(s) => ui.strong(format!("→ {s}")),
                        Conversion::Already => ui.weak(tl!("Already in this calendar")),
                        Conversion::Invalid(p) => ui.colored_label(ui.visuals().warn_fg_color, tl!(p.label())),
                    };
                    ui.end_row();
                }
            });
        });
    }
    if let Some(m) = &draft.message {
        ui.separator();
        ui.label(m);
    }
    ui.separator();
    let any = draft.picked.iter().any(|p| *p);
    let mut convert = false;
    let mut close = false;
    ui.horizontal(|ui| {
        convert = ui.add_enabled(any, egui::Button::new(tl!("Convert"))).clicked();
        close = ui.button(tl!("Close")).clicked();
    });
    if convert {
        convert_picked(app);
    }
    close
}

/// Convert the picked dates as one undoable edit and say what was left unchanged.
fn convert_picked(app: &mut PdfCraftApp) {
    let Some((_, id)) = app.active_ids() else { return };
    let chosen: Vec<DocumentDate> =
        app.date_convert_draft.hits.iter().zip(&app.date_convert_draft.picked).filter(|(_, p)| **p).map(|(h, _)| h.clone()).collect();
    let Some(doc) = app.session.get(id) else { return };
    let plan = doc.date_conversion(&chosen);
    let applied = match plan.edit {
        Some(edit) => app.apply_edit(edit),
        None => true,
    };
    let mut message = if applied {
        crate::i18n::fmt(
            tl!("{replaced} replaced, {skipped} left unchanged."),
            &[("replaced", &plan.replaced.len().to_string()), ("skipped", &plan.skipped.len().to_string())],
        )
    } else {
        tl!("The dates could not be converted; the document was not changed.").to_string()
    };
    if applied {
        let mut reasons: Vec<&str> = plan.skipped.iter().map(|(_, why)| *why).collect();
        reasons.sort_unstable();
        reasons.dedup();
        for why in reasons {
            message.push('\n');
            message.push_str(tl!(why));
        }
    }
    app.date_convert_draft.message = Some(message);
}
