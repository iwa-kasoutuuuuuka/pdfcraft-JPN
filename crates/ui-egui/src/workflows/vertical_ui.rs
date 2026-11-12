//! Create ▸ Vertical Japanese document (縦書き文書): a dialog that sets plain text on 原稿用紙, 便箋
//! or plain vertical pages. The engine does the work (`Session::create_vertical_document`, the
//! `create_vertical_document` tool); this is the form, a preview of the first page, and opening the
//! result as a new, unsaved document.

use std::sync::Arc;

use pdfcraft_engine::{VerticalOptions, VerticalPageSize, VerticalTemplate};

use crate::PdfCraftApp;
use crate::widgets;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Orientation {
    /// Landscape for 原稿用紙, portrait for the others.
    #[default]
    Auto,
    Portrait,
    Landscape,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum GridColour {
    #[default]
    Green,
    RedBrown,
}

/// The first page as the dialog shows it, and what the layout came to.
pub struct Preview {
    pub texture: egui::TextureHandle,
    pub pages: usize,
    pub hung: usize,
}

/// The dialog's form.
pub struct VerticalDraft {
    pub text: String,
    pub template: VerticalTemplate,
    pub page: VerticalPageSize,
    pub orientation: Orientation,
    pub serif: bool,
    /// 0: automatic.
    pub font_size: f64,
    /// 0: automatic. Squares across (原稿用紙) or lines (便箋, plain).
    pub columns: usize,
    /// 0: automatic. 原稿用紙 only.
    pub rows: usize,
    pub colour: GridColour,
    pub title: String,
    pub author: String,
    pub preview: Option<Preview>,
    /// Why the last preview or Create failed.
    pub error: Option<String>,
}

impl Default for VerticalDraft {
    fn default() -> Self {
        Self {
            text: String::new(),
            template: VerticalTemplate::Genko,
            page: VerticalPageSize::A4,
            orientation: Orientation::Auto,
            serif: true,
            font_size: 0.0,
            columns: 0,
            rows: 0,
            colour: GridColour::Green,
            title: String::new(),
            author: String::new(),
            preview: None,
            error: None,
        }
    }
}

impl VerticalDraft {
    /// The engine's options for this form.
    pub fn options(&self) -> VerticalOptions {
        let mut o = VerticalOptions::new(self.text.clone(), self.template);
        o.page_size = self.page;
        o.landscape = match self.orientation {
            Orientation::Auto => None,
            Orientation::Portrait => Some(false),
            Orientation::Landscape => Some(true),
        };
        o.serif = self.serif;
        o.font_size = (self.font_size >= 3.0).then_some(self.font_size);
        o.columns = (self.columns > 0).then_some(self.columns);
        o.rows = (self.rows > 0 && self.template == VerticalTemplate::Genko).then_some(self.rows);
        o.grid_colour = match self.colour {
            GridColour::Green => pdfcraft_engine::vertical_grid_colour("green").unwrap_or([0.4, 0.62, 0.45]),
            GridColour::RedBrown => pdfcraft_engine::vertical_grid_colour("red").unwrap_or([0.71, 0.4, 0.31]),
        };
        if self.template == VerticalTemplate::Genko {
            o.title = Some(self.title.clone()).filter(|t| !t.trim().is_empty());
            o.author = Some(self.author.clone()).filter(|t| !t.trim().is_empty());
        }
        o
    }
}

/// The dialog's content. Returns (create, cancel).
pub(crate) fn body(ui: &mut egui::Ui, app: &mut PdfCraftApp) -> (bool, bool) {
    ui.heading(tl!("Create vertical Japanese document"));
    let load;
    let mut preview = false;
    let (mut go, mut cancel) = (false, false);
    {
        let Some(d) = app.vertical_doc.as_mut() else { return (false, true) };
        let label = ui.label(tl!("Text (Aozora-style ruby such as 漢字《かんじ》 is supported)"));
        egui::ScrollArea::vertical().max_height(180.0).show(ui, |ui| {
            ui.add(egui::TextEdit::multiline(&mut d.text).desired_rows(8).desired_width(f32::INFINITY).id_salt("vertical-text"))
                .labelled_by(label.id);
        });
        // Browsers read a picked file asynchronously; the text box takes pasted text there.
        load = cfg!(not(target_arch = "wasm32")) && widgets::pill_button(ui, tl!("Load a .txt file…"), false).clicked();
        ui.add_space(6.0);
        ui.horizontal(|ui| {
            ui.label(tl!("Paper"));
            widgets::radio_value(ui, &mut d.template, VerticalTemplate::Genko, tl!("Genkō yōshi (manuscript paper)"));
            widgets::radio_value(ui, &mut d.template, VerticalTemplate::Binsen, tl!("Letter paper (ruled lines)"));
            widgets::radio_value(ui, &mut d.template, VerticalTemplate::Plain, tl!("Plain"));
        });
        ui.horizontal(|ui| {
            ui.label(tl!("Page size"));
            widgets::radio_value(ui, &mut d.page, VerticalPageSize::A4, "A4");
            widgets::radio_value(ui, &mut d.page, VerticalPageSize::A5, "A5");
            widgets::radio_value(ui, &mut d.page, VerticalPageSize::B4, "B4");
            widgets::radio_value(ui, &mut d.page, VerticalPageSize::B5, "B5");
        });
        ui.horizontal(|ui| {
            ui.label(tl!("Orientation"));
            widgets::radio_value(ui, &mut d.orientation, Orientation::Auto, tl!("Automatic"));
            widgets::radio_value(ui, &mut d.orientation, Orientation::Portrait, tl!("Portrait"));
            widgets::radio_value(ui, &mut d.orientation, Orientation::Landscape, tl!("Landscape"));
        });
        ui.horizontal(|ui| {
            ui.label(tl!("Font"));
            widgets::radio_value(ui, &mut d.serif, true, tl!("Mincho"));
            widgets::radio_value(ui, &mut d.serif, false, tl!("Gothic"));
        });
        let auto = |v: f64, _: std::ops::RangeInclusive<usize>| if v < 0.5 { tl!("Auto").to_string() } else { format!("{v:.0}") };
        ui.horizontal(|ui| {
            let label = ui.label(tl!("Font size (pt)"));
            ui.add(egui::DragValue::new(&mut d.font_size).range(0.0..=72.0).custom_formatter(auto)).labelled_by(label.id);
            let label = ui.label(if d.template == VerticalTemplate::Genko { tl!("Squares across") } else { tl!("Lines per page") });
            ui.add(egui::DragValue::new(&mut d.columns).range(0..=60).custom_formatter(auto)).labelled_by(label.id);
            if d.template == VerticalTemplate::Genko {
                let label = ui.label(tl!("Squares per column"));
                ui.add(egui::DragValue::new(&mut d.rows).range(0..=60).custom_formatter(auto)).labelled_by(label.id);
            }
        });
        if d.template == VerticalTemplate::Genko {
            ui.horizontal(|ui| {
                ui.label(tl!("Grid colour"));
                widgets::radio_value(ui, &mut d.colour, GridColour::Green, tl!("Green"));
                widgets::radio_value(ui, &mut d.colour, GridColour::RedBrown, tl!("Red-brown"));
            });
        } else {
            ui.horizontal(|ui| {
                ui.label(tl!("Line colour"));
                widgets::radio_value(ui, &mut d.colour, GridColour::Green, tl!("Green"));
                widgets::radio_value(ui, &mut d.colour, GridColour::RedBrown, tl!("Red-brown"));
            });
        }
        if d.template == VerticalTemplate::Genko {
            ui.horizontal(|ui| {
                let label = ui.label(tl!("Title"));
                ui.add(egui::TextEdit::singleline(&mut d.title).desired_width(200.0).id_salt("vertical-title")).labelled_by(label.id);
                let label = ui.label(tl!("Author"));
                ui.add(egui::TextEdit::singleline(&mut d.author).desired_width(140.0).id_salt("vertical-author")).labelled_by(label.id);
            });
        }
        if let Some(e) = &d.error {
            ui.colored_label(ui.visuals().error_fg_color, e);
        }
        if let Some(p) = &d.preview {
            ui.add_space(4.0);
            let size = p.texture.size_vec2();
            let scale = (300.0 / size.y.max(1.0)).min(1.0);
            ui.add(egui::Image::new((p.texture.id(), size * scale)));
            ui.label(crate::i18n::fmt(
                tl!("{pages} page(s); {hung} hung punctuation mark(s)"),
                &[("pages", &p.pages.to_string()), ("hung", &p.hung.to_string())],
            ));
        }
        ui.add_space(8.0);
        ui.horizontal(|ui| {
            go = widgets::pill_button(ui, tl!("Create"), true).clicked();
            preview = widgets::pill_button(ui, tl!("Preview first page"), false).clicked();
            cancel = widgets::pill_button(ui, tl!("Cancel"), false).clicked();
        });
    }
    if load {
        app.load_vertical_text();
    }
    if preview {
        app.preview_vertical(ui.ctx());
    }
    (go, cancel)
}

impl PdfCraftApp {
    /// Create ▸ Vertical Japanese document…
    pub(crate) fn create_vertical_dialog(&mut self) {
        self.vertical_doc = Some(VerticalDraft::default());
        self.dialog = Some(crate::Dialog::CreateVertical);
    }

    /// Put a text file's contents into the form's text box.
    #[cfg(not(target_arch = "wasm32"))]
    fn load_vertical_text(&mut self) {
        use crate::file_filter::AddFilterAnyCase as _;
        let dialog = rfd::AsyncFileDialog::new().add_filter_any_case(tl!("Text"), &["txt"]).set_title(tl!("Choose a text file"));
        self.ask_one(crate::pickers::Ask::File(dialog), None, |app, path| match std::fs::read(&path) {
            Ok(bytes) => {
                if let Some(d) = app.vertical_doc.as_mut() {
                    d.text = pdfcraft_engine::decode_text(&bytes);
                    d.preview = None;
                    d.error = None;
                }
            }
            Err(e) => app.notify_fmt("Couldn't read {name}: {e}", &[("name", &path.display().to_string()), ("e", &e.to_string())]),
        });
    }

    #[cfg(target_arch = "wasm32")]
    fn load_vertical_text(&mut self) {}

    /// Lay the text out and show the first page in the form.
    fn preview_vertical(&mut self, ctx: &egui::Context) {
        let Some(d) = self.vertical_doc.as_mut() else { return };
        d.preview = None;
        d.error = None;
        match self.session.create_vertical_document(&d.options()) {
            Ok((bytes, made)) => {
                let config = pdfcraft_render::RenderConfig::default().on_screen();
                let mut renderer = pdfcraft_render::PageRenderer::new(Arc::clone(&bytes), config);
                let out = renderer.render(pdfcraft_render::RenderRequest { page: 0, scale: 0.9, preview: true, ..Default::default() });
                if let Some(e) = &out.error {
                    d.error = Some(e.clone());
                    return;
                }
                let size = [out.width as usize, out.height as usize];
                let image = crate::canvas::texture_image(size, out.rgba);
                let texture = ctx.load_texture("vertical-document-preview", image, egui::TextureOptions::LINEAR);
                d.preview = Some(Preview { texture, pages: made.pages, hung: made.hung });
            }
            Err(e) => d.error = Some(e.to_string()),
        }
    }

    /// Create the document from the form and open it. Returns whether the dialog is done.
    pub(crate) fn finish_vertical_document(&mut self) -> bool {
        let Some(d) = self.vertical_doc.as_mut() else { return true };
        d.error = None;
        let name = match d.title.trim() {
            t if !t.is_empty() && d.template == VerticalTemplate::Genko => format!("{t}.pdf"),
            _ => "縦書き文書.pdf".to_string(),
        };
        let created = self.session.create_vertical_document(&d.options()).map(|(bytes, _)| bytes).map_err(|e| e.to_string());
        match self.open_created_bytes(&name, created) {
            Ok(()) => {
                self.vertical_doc = None;
                true
            }
            Err(e) => {
                if let Some(d) = self.vertical_doc.as_mut() {
                    d.error = Some(e);
                }
                false
            }
        }
    }
}
