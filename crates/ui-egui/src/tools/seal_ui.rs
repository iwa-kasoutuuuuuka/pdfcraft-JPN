//! pdfcraft-JPN: electronic seals (電子印鑑 / ハンコ) in the stamps palette: a name or date seal
//! made from a few fields, kept in the custom stamp library, then placed with a click.
//!
//! A seal is a custom stamp whose picture is drawn by `pdfcraft_engine::seal`. The library entry
//! keeps the seal's fields ([`SealDraft`]), not just a picture, so a date seal is drawn again with
//! the day it is placed on.

use std::sync::Arc;

use egui::Ui;
use pdfcraft_engine::seal::{DEFAULT_SEAL_COLOR, DEFAULT_SEAL_SIZE, SealKind, SealShape, SealSpec, parse_seal_date, seal_edit, seal_pdf};
use serde::{Deserialize, Serialize};

use crate::canvas::{DocView, PageToolContext};
use crate::page_cursor::Layer;
use crate::stamps_ui::CustomStamp;
use crate::theme::Tokens;
use crate::{PdfCraftApp, QuickTool, widgets};

/// The fields of a seal, as the palette edits them.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct SealDraft {
    /// A date seal (データ印) rather than a name seal (認印).
    pub date_seal: bool,
    /// A square outline (角印) rather than a round one.
    pub square: bool,
    pub name: String,
    pub org: String,
    /// The date with its era (令和8.11.11) rather than Western (2026.11.11).
    pub era: bool,
    /// The date is the day the seal is placed on.
    pub today: bool,
    /// The date to write when it isn't today: yyyy-mm-dd.
    pub date: String,
    pub color: [f64; 3],
    /// The seal's side, in points.
    pub size: f64,
}

impl Default for SealDraft {
    fn default() -> Self {
        SealDraft {
            date_seal: false,
            square: false,
            name: String::new(),
            org: String::new(),
            era: true,
            today: true,
            date: String::new(),
            color: DEFAULT_SEAL_COLOR,
            size: DEFAULT_SEAL_SIZE,
        }
    }
}

impl SealDraft {
    /// The seal to draw, with `today` for a date seal that takes the day it is placed on. `Err`
    /// says what to change.
    pub fn spec(&self, today: (i64, u32, u32)) -> Result<SealSpec, String> {
        let date = if self.today {
            today
        } else {
            parse_seal_date(&self.date).ok_or_else(|| tl!("Write the date as yyyy-mm-dd, for example 2026-11-11").to_string())?
        };
        Ok(SealSpec {
            kind: if self.date_seal { SealKind::Date } else { SealKind::Name },
            shape: if self.square { SealShape::Square } else { SealShape::Circle },
            name: self.name.clone(),
            org: self.org.clone(),
            date,
            era: self.era,
            color: self.color,
            size: self.size,
        })
    }

    /// The name this seal has in the stamp library.
    fn label(&self) -> String {
        let kind = if self.date_seal { tl!("Date seal") } else { tl!("Name seal") };
        format!("{kind}: {}", self.name.trim())
    }
}

/// The palette's preview of the seal being made: redrawn when its fields or the day change.
#[derive(Default)]
pub struct SealPreview {
    key: Option<(SealDraft, (i64, u32, u32))>,
    texture: Option<egui::TextureHandle>,
    error: Option<String>,
}

impl SealPreview {
    fn refresh(&mut self, ctx: &egui::Context, draft: &SealDraft, today: (i64, u32, u32)) {
        if self.key.as_ref().is_some_and(|(d, t)| d == draft && *t == today) {
            return;
        }
        self.key = Some((draft.clone(), today));
        self.texture = None;
        self.error = None;
        match render(draft, today) {
            Ok(img) => self.texture = Some(ctx.load_texture("seal-preview", img, egui::TextureOptions::LINEAR)),
            Err(e) => self.error = Some(e),
        }
    }
}

/// The seal drawn on a transparent page, three pixels to the point.
fn render(draft: &SealDraft, today: (i64, u32, u32)) -> Result<egui::ColorImage, String> {
    let mut spec = draft.spec(today)?;
    if spec.name.trim().is_empty() {
        return Err(tl!("Type a name to see the seal").to_string());
    }
    spec.size = spec.size.clamp(pdfcraft_engine::seal::SEAL_SIZE_RANGE.0, pdfcraft_engine::seal::SEAL_SIZE_RANGE.1);
    let bytes = seal_pdf(&spec)?;
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
        return Err(tl!("The seal could not be drawn").to_string());
    }
    Ok(egui::ColorImage::from_rgba_premultiplied([out.width as usize, out.height as usize], &out.rgba))
}

fn color_button(ui: &mut Ui, c: &mut [f64; 3]) {
    let mut srgb = [0, 1, 2].map(|i| (c[i].clamp(0.0, 1.0) * 255.0).round() as u8);
    if ui.color_edit_button_srgb(&mut srgb).changed() {
        *c = srgb.map(|v| f64::from(v) / 255.0);
    }
}

/// The Seal section of the stamps palette.
pub(crate) fn palette_section(app: &mut PdfCraftApp, ui: &mut Ui, t: &Tokens) {
    widgets::section_title(ui, tl!("Seal"));
    ui.label(
        egui::RichText::new(tl!("An electronic seal with your name, or a date stamp. Click on the page to place it.")).small().color(t.text_faint),
    );
    let today = app.session.today();
    let d = &mut app.seal_draft;
    ui.horizontal(|ui| {
        ui.selectable_value(&mut d.date_seal, false, tl!("Name seal"));
        ui.selectable_value(&mut d.date_seal, true, tl!("Date seal"));
    });
    ui.horizontal(|ui| {
        ui.selectable_value(&mut d.square, false, tl!("Round"));
        ui.selectable_value(&mut d.square, true, tl!("Square"));
    });
    egui::Grid::new("seal-fields").num_columns(2).spacing([8.0, 6.0]).show(ui, |ui| {
        let l = ui.label(tl!("Name:"));
        ui.add(widgets::line(&mut d.name).desired_width(140.0).char_limit(pdfcraft_engine::seal::MAX_SEAL_ASCII_CHARS)).labelled_by(l.id);
        ui.end_row();
        if d.date_seal {
            let l = ui.label(tl!("Department:"));
            ui.add(widgets::line(&mut d.org).desired_width(140.0).char_limit(pdfcraft_engine::seal::MAX_SEAL_ORG_CHARS)).labelled_by(l.id);
            ui.end_row();
        }
        ui.label(tl!("Color:"));
        color_button(ui, &mut d.color);
        ui.end_row();
        ui.label(tl!("Size:"));
        ui.add(egui::Slider::new(&mut d.size, 24.0..=96.0).suffix(" pt"));
        ui.end_row();
    });
    if d.date_seal {
        ui.add(widgets::checkbox(&mut d.today, tl!("Date of the day it is placed")));
        if !d.today {
            ui.add(widgets::line(&mut d.date).desired_width(140.0).hint_text("2026-11-11"));
        }
        ui.add(widgets::checkbox(&mut d.era, tl!("Japanese era (令和8.11.11)")));
    }
    let draft = d.clone();
    let ctx = ui.ctx().clone();
    app.seal_preview.refresh(&ctx, &draft, today);
    ui.add_space(4.0);
    let side = (draft.size as f32 * 1.6).clamp(48.0, 150.0);
    match (&app.seal_preview.texture, &app.seal_preview.error) {
        (Some(tex), _) => {
            let (rect, _) = ui.allocate_exact_size(egui::vec2(ui.available_width(), side + 8.0), egui::Sense::hover());
            ui.painter().rect_filled(rect, 6, egui::Color32::WHITE);
            let image = egui::Rect::from_center_size(rect.center(), egui::vec2(side, side));
            ui.painter().image(tex.id(), image, egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)), egui::Color32::WHITE);
        }
        (None, Some(why)) => {
            ui.label(egui::RichText::new(why).small().color(t.text_muted));
        }
        (None, None) => {}
    }
    ui.add_space(4.0);
    let ready = app.seal_preview.texture.is_some();
    if ui.add_enabled_ui(ready, |ui| widgets::pill_button(ui, tl!("Use this seal"), true)).inner.clicked() {
        use_seal(app, today);
    }
}

/// Keep the seal in the stamp library (replacing one with the same label) and choose it.
fn use_seal(app: &mut PdfCraftApp, today: (i64, u32, u32)) {
    let draft = app.seal_draft.clone();
    let bytes = match draft.spec(today).and_then(|s| seal_pdf(&s)) {
        Ok(b) => b,
        Err(e) => {
            app.notify_error(&e);
            return;
        }
    };
    let (category, name) = (tl!("Seals").to_string(), draft.label());
    let stamp = CustomStamp {
        category: category.clone(),
        name: name.clone(),
        file: "seal.pdf".into(),
        page: 0,
        data: Arc::new(bytes),
        line: None,
        seal: Some(draft),
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

/// A seal on one page: a click that nobody holds (`free`) places `stamp`'s seal centred on it,
/// dated `today` when it takes the day. `Ok(true)` once one is placed; `Err` says why it can't be
/// drawn.
pub(crate) fn place(
    cx: PageToolContext<'_>,
    seal: &SealDraft,
    today: (i64, u32, u32),
    author: &str,
    free: bool,
    view: &mut DocView,
) -> Result<bool, String> {
    let PageToolContext { resp, xf, ptr, page, info, .. } = cx;
    let (Some(p), Some(_)) = (ptr.within(xf.rect), info.pages.get(page)) else { return Ok(false) };
    view.cursor.claim(Layer::ToolArea, egui::CursorIcon::Crosshair);
    if !(free && resp.clicked()) {
        return Ok(false);
    }
    let Some(page_info) = info.pages.get(page) else { return Ok(false) };
    let (vx, vy) = xf.screen_to_view(p);
    let u = page_info.view_to_user(vx, vy);
    let spec = seal.spec(today)?;
    view.pending_edit = Some(seal_edit(&spec, page, [f64::from(u[0]), f64::from(u[1])], author)?);
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_date_seal_takes_today_or_the_typed_date() {
        let mut d = SealDraft { date_seal: true, name: "山田".into(), ..SealDraft::default() };
        assert_eq!(d.spec((2026, 11, 11)).unwrap().date, (2026, 11, 11), "the day it is placed");
        d.today = false;
        d.date = "2025-01-05".into();
        assert_eq!(d.spec((2026, 11, 11)).unwrap().date, (2025, 1, 5));
        d.date = "someday".into();
        assert!(d.spec((2026, 11, 11)).is_err(), "a date that isn't one is refused with a message");
        let spec = SealDraft { square: true, ..SealDraft::default() }.spec((2026, 11, 11)).unwrap();
        assert_eq!((spec.kind, spec.shape), (SealKind::Name, SealShape::Square));
    }

    /// A seal kept in the stamp library comes back from the settings with its fields, so a date
    /// seal is drawn again on the day it is placed; an older library entry has no seal.
    #[test]
    fn a_library_seal_keeps_its_fields_through_the_settings() {
        let draft = SealDraft { date_seal: true, org: "営業部".into(), name: "鈴木".into(), era: false, ..SealDraft::default() };
        let stamp = CustomStamp {
            category: "Seals".into(),
            name: draft.label(),
            file: "seal.pdf".into(),
            page: 0,
            data: Arc::new(vec![1, 2, 3]),
            line: None,
            seal: Some(draft.clone()),
        };
        let json = crate::stamps_ui::encode(std::slice::from_ref(&stamp));
        assert_eq!(crate::stamps_ui::decode(&json), vec![stamp]);
        let old = serde_json::json!([{ "category": "c", "name": "n", "file": "f.png", "data": "AQID" }]);
        assert!(crate::stamps_ui::decode(&old)[0].seal.is_none());
    }

    #[test]
    fn the_preview_is_drawn_for_a_seal_and_refused_for_a_bad_one() {
        let ok = SealDraft { name: "Smith".into(), ..SealDraft::default() };
        assert!(render(&ok, (2026, 11, 11)).is_ok());
        assert!(render(&SealDraft::default(), (2026, 11, 11)).is_err(), "no name, no seal");
        let long = SealDraft { name: "A name far too long to fit in a seal".into(), ..SealDraft::default() };
        assert!(render(&long, (2026, 11, 11)).is_err());
    }
}
