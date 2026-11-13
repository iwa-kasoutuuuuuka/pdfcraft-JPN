//! pdfcraft-JPN: ready-made business stamps (ビジネス用スタンプ: 承認, 至急, 社外秘...) in the
//! stamps palette, next to the seal. Clicking a word chooses that stamp; a click on the page
//! places it.
//!
//! The picture is drawn by `pdfcraft_engine::seal::business`. Like a seal, the library entry keeps
//! the stamp's fields ([`BusinessDraft`]), so a stamp with a date line is drawn again with the day
//! it is placed on.

use std::sync::Arc;

use egui::Ui;
use pdfcraft_engine::seal::DEFAULT_SEAL_COLOR;
use pdfcraft_engine::seal::business::{
    BUSINESS_STAMP_WORDS, BusinessStampSpec, DEFAULT_BUSINESS_STAMP_SIZE, MAX_BUSINESS_STAMP_NAME_CHARS, MAX_BUSINESS_STAMP_WORD_CHARS, StampFrame,
    business_stamp_edit, business_stamp_pdf,
};
use serde::{Deserialize, Serialize};

use crate::canvas::{DocView, PageToolContext};
use crate::page_cursor::Layer;
use crate::stamps_ui::CustomStamp;
use crate::theme::Tokens;
use crate::{PdfCraftApp, QuickTool, widgets};

/// The fields of a business stamp, as the palette edits them.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct BusinessDraft {
    pub word: String,
    /// A double rule rather than one rounded outline.
    pub double: bool,
    /// A date line: the day the stamp is placed on.
    pub with_date: bool,
    /// The date with its era (令和8.11.13) rather than Western (2026.11.13).
    pub era: bool,
    /// A name line; empty for none.
    pub name: String,
    pub color: [f64; 3],
    /// The word's letter size, in points.
    pub size: f64,
}

impl Default for BusinessDraft {
    fn default() -> Self {
        BusinessDraft {
            word: BUSINESS_STAMP_WORDS.first().copied().unwrap_or_default().to_owned(),
            double: false,
            with_date: false,
            era: true,
            name: String::new(),
            color: DEFAULT_SEAL_COLOR,
            size: DEFAULT_BUSINESS_STAMP_SIZE,
        }
    }
}

impl BusinessDraft {
    /// The stamp to draw, dated `today` when it has a date line.
    pub fn spec(&self, today: (i64, u32, u32)) -> BusinessStampSpec {
        BusinessStampSpec {
            word: self.word.clone(),
            frame: if self.double { StampFrame::Double } else { StampFrame::Rounded },
            date: self.with_date.then_some(today),
            era: self.era,
            name: self.name.clone(),
            color: self.color,
            size: self.size,
        }
    }

    /// The name this stamp has in the stamp library.
    fn label(&self) -> String {
        let mut label = self.word.trim().to_owned();
        if self.with_date {
            label = format!("{label} ({})", tl!("Date"));
        }
        if !self.name.trim().is_empty() {
            label = format!("{label} {}", self.name.trim());
        }
        label
    }
}

/// The palette's preview of the stamp being made: redrawn when its fields or the day change.
#[derive(Default)]
pub struct BusinessPreview {
    key: Option<(BusinessDraft, (i64, u32, u32))>,
    texture: Option<egui::TextureHandle>,
    /// The picture's width over its height.
    aspect: f32,
    error: Option<String>,
}

impl BusinessPreview {
    fn refresh(&mut self, ctx: &egui::Context, draft: &BusinessDraft, today: (i64, u32, u32)) {
        if self.key.as_ref().is_some_and(|(d, t)| d == draft && *t == today) {
            return;
        }
        self.key = Some((draft.clone(), today));
        self.texture = None;
        self.error = None;
        match render(draft, today) {
            Ok(img) => {
                self.aspect = img.size[0] as f32 / img.size[1].max(1) as f32;
                self.texture = Some(ctx.load_texture("business-stamp-preview", img, egui::TextureOptions::LINEAR));
            }
            Err(e) => self.error = Some(e),
        }
    }
}

/// The stamp drawn on a transparent page, three pixels to the point.
fn render(draft: &BusinessDraft, today: (i64, u32, u32)) -> Result<egui::ColorImage, String> {
    if draft.word.trim().is_empty() {
        return Err(tl!("Type a word to see the stamp").to_string());
    }
    let bytes = business_stamp_pdf(&draft.spec(today))?;
    let config = pdfcraft_render::RenderConfig { transparent: true, ..Default::default() };
    let out = pdfcraft_render::PageRenderer::new(Arc::new(bytes), config).render(pdfcraft_render::RenderRequest {
        page: 0,
        scale: 3.0,
        ..Default::default()
    });
    if let Some(e) = out.error {
        return Err(e);
    }
    if out.width == 0 || out.height == 0 {
        return Err(tl!("The stamp could not be drawn").to_string());
    }
    Ok(egui::ColorImage::from_rgba_premultiplied([out.width as usize, out.height as usize], &out.rgba))
}

fn color_button(ui: &mut Ui, c: &mut [f64; 3]) {
    let mut srgb = [0, 1, 2].map(|i| (c[i].clamp(0.0, 1.0) * 255.0).round() as u8);
    if ui.color_edit_button_srgb(&mut srgb).changed() {
        *c = srgb.map(|v| f64::from(v) / 255.0);
    }
}

/// The Business stamps section of the stamps palette.
pub(crate) fn palette_section(app: &mut PdfCraftApp, ui: &mut Ui, t: &Tokens) {
    widgets::section_title(ui, tl!("Business stamps"));
    ui.label(
        egui::RichText::new(tl!("Ready-made stamps such as 承認 or 至急. Click a word, then click on the page to place it."))
            .small()
            .color(t.text_faint),
    );
    let today = app.session.today();
    let mut chosen = false;
    ui.horizontal_wrapped(|ui| {
        for word in BUSINESS_STAMP_WORDS {
            let on = app.business_draft.word == *word;
            if ui.selectable_label(on, *word).clicked() {
                app.business_draft.word = (*word).to_owned();
                chosen = true;
            }
        }
    });
    let d = &mut app.business_draft;
    ui.horizontal(|ui| {
        ui.selectable_value(&mut d.double, false, tl!("Rounded corners"));
        ui.selectable_value(&mut d.double, true, tl!("Double rule"));
    });
    widgets::begin_dialog(ui);
    egui::Grid::new("business-stamp-fields").num_columns(2).spacing([8.0, 6.0]).show(ui, |ui| {
        let l = ui.label(tl!("Word:"));
        ui.add(widgets::line(&mut d.word).desired_width(140.0).char_limit(MAX_BUSINESS_STAMP_WORD_CHARS)).labelled_by(l.id);
        ui.end_row();
        let l = ui.label(tl!("Name:"));
        ui.add(widgets::line(&mut d.name).desired_width(140.0).char_limit(MAX_BUSINESS_STAMP_NAME_CHARS)).labelled_by(l.id);
        ui.end_row();
        ui.label(tl!("Color:"));
        color_button(ui, &mut d.color);
        ui.end_row();
        ui.label(tl!("Size:"));
        ui.add(egui::Slider::new(&mut d.size, 10.0..=36.0).suffix(" pt"));
        ui.end_row();
    });
    ui.add(widgets::checkbox(&mut d.with_date, tl!("Add the date of the day it is placed")));
    if d.with_date {
        ui.add(widgets::checkbox(&mut d.era, tl!("Japanese era (令和8.11.11)")));
    }
    let draft = d.clone();
    let ctx = ui.ctx().clone();
    app.business_preview.refresh(&ctx, &draft, today);
    ui.add_space(4.0);
    let height = (draft.size as f32 * 2.2).clamp(32.0, 90.0);
    match (&app.business_preview.texture, &app.business_preview.error) {
        (Some(tex), _) => {
            let (rect, _) = ui.allocate_exact_size(egui::vec2(ui.available_width(), height + 8.0), egui::Sense::hover());
            ui.painter().rect_filled(rect, 6, egui::Color32::WHITE);
            let size = egui::vec2(height * app.business_preview.aspect, height);
            let size = size * (rect.width() - 8.0).min(size.x).max(1.0) / size.x.max(1.0);
            let image = egui::Rect::from_center_size(rect.center(), size);
            ui.painter().image(tex.id(), image, egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)), egui::Color32::WHITE);
        }
        (None, Some(why)) => {
            ui.label(egui::RichText::new(why).small().color(t.text_muted));
        }
        (None, None) => {}
    }
    ui.add_space(4.0);
    let ready = app.business_preview.texture.is_some();
    let use_it = ui.add_enabled_ui(ready, |ui| widgets::pill_button(ui, tl!("Use this stamp"), true)).inner.clicked();
    if chosen || use_it {
        use_stamp(app, today);
    }
}

/// Keep the stamp in the stamp library (replacing one with the same label) and choose it.
fn use_stamp(app: &mut PdfCraftApp, today: (i64, u32, u32)) {
    let draft = app.business_draft.clone();
    let bytes = match business_stamp_pdf(&draft.spec(today)) {
        Ok(b) => b,
        Err(e) => {
            app.notify_error(&e);
            return;
        }
    };
    let (category, name) = (tl!("Business stamps").to_string(), draft.label());
    let stamp = CustomStamp {
        category: category.clone(),
        name: name.clone(),
        file: "stamp.pdf".into(),
        page: 0,
        data: Arc::new(bytes),
        line: None,
        seal: None,
        business: Some(draft),
    };
    let at = match app.custom_stamps.iter().position(|s| s.category == category && s.name == name) {
        Some(i) => {
            app.custom_stamps[i] = stamp;
            i
        }
        None => {
            app.custom_stamps.push(stamp);
            app.custom_stamps.len() - 1
        }
    };
    app.quick_tool = QuickTool::CustomStamp(at);
}

/// A business stamp on one page: a click that nobody holds (`free`) places it centred on the
/// click, dated `today` when it has a date line. `Ok(true)` once one is placed; `Err` says why it
/// can't be drawn.
pub(crate) fn place(
    cx: PageToolContext<'_>,
    stamp: &BusinessDraft,
    today: (i64, u32, u32),
    author: &str,
    free: bool,
    view: &mut DocView,
) -> Result<bool, String> {
    let PageToolContext { resp, xf, ptr, page, info, .. } = cx;
    let (Some(p), Some(page_info)) = (ptr.within(xf.rect), info.pages.get(page)) else { return Ok(false) };
    view.cursor.claim(Layer::ToolArea, egui::CursorIcon::Crosshair);
    if !(free && resp.clicked()) {
        return Ok(false);
    }
    let (vx, vy) = xf.screen_to_view(p);
    let u = page_info.view_to_user(vx, vy);
    view.pending_edit = Some(business_stamp_edit(&stamp.spec(today), page, page_info, [f64::from(u[0]), f64::from(u[1])], author)?);
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_date_line_takes_the_day_it_is_placed() {
        let mut d = BusinessDraft::default();
        assert_eq!(d.word, "承認", "the first ready-made word is chosen at first");
        assert_eq!(d.spec((2026, 11, 13)).date, None, "no date line unless asked");
        d.with_date = true;
        d.double = true;
        let spec = d.spec((2026, 11, 13));
        assert_eq!((spec.date, spec.frame), (Some((2026, 11, 13)), StampFrame::Double));
    }

    /// A business stamp kept in the stamp library comes back from the settings with its fields;
    /// an older library entry has none.
    #[test]
    fn a_library_business_stamp_keeps_its_fields_through_the_settings() {
        let draft = BusinessDraft { word: "至急".into(), with_date: true, name: "山田".into(), ..BusinessDraft::default() };
        let stamp = CustomStamp {
            category: "Business stamps".into(),
            name: draft.label(),
            file: "stamp.pdf".into(),
            page: 0,
            data: Arc::new(vec![1, 2, 3]),
            line: None,
            seal: None,
            business: Some(draft),
        };
        let json = crate::stamps_ui::encode(std::slice::from_ref(&stamp));
        assert_eq!(crate::stamps_ui::decode(&json), vec![stamp]);
        let old = serde_json::json!([{ "category": "c", "name": "n", "file": "f.png", "data": "AQID" }]);
        assert!(crate::stamps_ui::decode(&old)[0].business.is_none());
    }

    #[test]
    fn the_preview_is_drawn_for_a_stamp_and_refused_for_a_bad_one() {
        let ok = BusinessDraft { word: "OK".into(), ..BusinessDraft::default() };
        assert!(render(&ok, (2026, 11, 13)).is_ok());
        assert!(render(&BusinessDraft { word: " ".into(), ..ok.clone() }, (2026, 11, 13)).is_err(), "no word, no stamp");
        assert!(render(&BusinessDraft { word: "TOO LONG".into(), ..ok }, (2026, 11, 13)).is_err());
    }
}
