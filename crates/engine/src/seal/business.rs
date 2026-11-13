//! pdfcraft-JPN: ready-made business stamps (ビジネス用スタンプ): a word such as 承認, 至急 or
//! 社外秘 in an outline, with an optional date line and name line under it.
//!
//! They are drawn and stored as electronic seals are (see the parent module): a one-page vector
//! PDF placed as a custom stamp ([`Edit::AddCustomStamp`]), so the stamp is one comment that
//! moves, resizes, deletes and undoes like any stamp, and stays sharp at any zoom.
//!
//! The design is our own: a single rounded outline or a double rule (a thick outer frame and a
//! thin inner one), the word large and bold in the middle, and when there are extra lines a thin
//! rule between the word and them.

use pdfcraft_edit::added::{Content, add_content, text_width};
use pdfcraft_render::PageInfo;

use super::{DEFAULT_SEAL_COLOR, check_text, is_seal_char, item, missing_font, num, seal_date_text};
use crate::Edit;
use crate::pictures::MarkFile;

/// The ready-made words, in the order the palette offers them.
pub const BUSINESS_STAMP_WORDS: &[&str] = &["承認", "確認済", "至急", "社外秘", "回覧", "複製厳禁", "済", "重要", "控"];
/// The word's letter size when none is given, in points.
pub const DEFAULT_BUSINESS_STAMP_SIZE: f64 = 20.0;
/// The smallest and largest letter size of the word, in points.
pub const BUSINESS_STAMP_SIZE_RANGE: (f64, f64) = (8.0, 48.0);
/// Most characters of the word.
pub const MAX_BUSINESS_STAMP_WORD_CHARS: usize = 6;
/// Most characters of the name line.
pub const MAX_BUSINESS_STAMP_NAME_CHARS: usize = 12;

/// Lines under the word are this share of the word's size.
const LINE_SCALE: f64 = 0.42;
/// Height of the ink of a line of text, as a share of its size (a safe overestimate).
const INK_HEIGHT: f64 = 0.9;
/// Distance of a line's visual centre below the top of its text box, in em (as the seal's).
const CENTRE_BELOW_TOP: f64 = 0.57;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StampFrame {
    /// One outline with rounded corners.
    Rounded,
    /// A thick outer rectangle and a thin one inside it.
    Double,
}

impl StampFrame {
    pub fn parse(s: &str) -> Option<StampFrame> {
        match s.trim().to_lowercase().as_str() {
            "rounded" | "round" | "角丸" => Some(StampFrame::Rounded),
            "double" | "二重" | "二重枠" => Some(StampFrame::Double),
            _ => None,
        }
    }

    pub fn id(self) -> &'static str {
        match self {
            StampFrame::Rounded => "rounded",
            StampFrame::Double => "double",
        }
    }
}

/// What a business stamp says and how it looks.
#[derive(Clone, Debug, PartialEq)]
pub struct BusinessStampSpec {
    /// The word: one of [`BUSINESS_STAMP_WORDS`] or any other short word.
    pub word: String,
    pub frame: StampFrame,
    /// A date line under the word, as (year, month, day) in the Gregorian calendar.
    pub date: Option<(i64, u32, u32)>,
    /// Write the date with its era (令和8.11.13) instead of Western (2026.11.13).
    pub era: bool,
    /// A name line under the word (and the date); empty for none.
    pub name: String,
    /// The colour, RGB from 0 to 1.
    pub color: [f64; 3],
    /// The word's letter size, in points; the stamp is sized around it.
    pub size: f64,
}

impl Default for BusinessStampSpec {
    fn default() -> Self {
        BusinessStampSpec {
            word: String::new(),
            frame: StampFrame::Rounded,
            date: None,
            era: false,
            name: String::new(),
            color: DEFAULT_SEAL_COLOR,
            size: DEFAULT_BUSINESS_STAMP_SIZE,
        }
    }
}

/// One line of text on a business stamp, in the stamp's own space (points, origin bottom-left).
#[derive(Clone, Debug, PartialEq)]
pub struct StampLine {
    pub text: String,
    pub bold: bool,
    pub size: f64,
    /// The text box handed to the Add text machinery (as wide as the stamp, centred).
    pub rect: [f64; 4],
    /// The box the ink may reach.
    pub ink: [f64; 4],
}

/// Where everything on a business stamp goes.
#[derive(Clone, Debug, PartialEq)]
pub struct BusinessStampLayout {
    pub width: f64,
    pub height: f64,
    pub frame: StampFrame,
    /// Centre lines of the outline rectangles, outermost first, `[x0, y0, x1, y1]`.
    pub outlines: Vec<[f64; 4]>,
    /// Stroke width of each outline.
    pub strokes: Vec<f64>,
    /// Corner radius of a rounded outline's centre line (0 for a double rule).
    pub corner: f64,
    /// The rule between the word and the lines under it, `[x0, y, x1, y]`, and its width.
    pub rule: Option<([f64; 4], f64)>,
    /// The word first, then the date and the name when present.
    pub lines: Vec<StampLine>,
}

impl BusinessStampLayout {
    /// The inside of the innermost outline: everything written lies in it.
    pub fn inner(&self) -> [f64; 4] {
        let (Some(o), Some(w)) = (self.outlines.last(), self.strokes.last()) else { return [0.0, 0.0, self.width, self.height] };
        let h = w / 2.0;
        [o[0] + h, o[1] + h, o[2] - h, o[3] - h]
    }
}

fn check_chars(label: &str, text: &str) -> Result<(), String> {
    if let Some(e) = missing_font(text) {
        return Err(e.replace("on a seal", "on a stamp"));
    }
    if let Some(c) = text.chars().find(|&c| !is_seal_char(c)) {
        return Err(format!("{c:?} can't be drawn in the {label} of a stamp: use Japanese characters, letters and digits"));
    }
    Ok(())
}

/// The layout of a business stamp in points. `Err` says what to change.
pub fn business_stamp_layout(spec: &BusinessStampSpec) -> Result<BusinessStampLayout, String> {
    let s = spec.size;
    if !s.is_finite() || !(BUSINESS_STAMP_SIZE_RANGE.0..=BUSINESS_STAMP_SIZE_RANGE.1).contains(&s) {
        return Err(format!("the stamp's letter size must be {} to {} points", BUSINESS_STAMP_SIZE_RANGE.0, BUSINESS_STAMP_SIZE_RANGE.1));
    }
    if !spec.color.iter().all(|v| v.is_finite() && (0.0..=1.0).contains(v)) {
        return Err("the colour must be RGB values from 0 to 1".into());
    }
    let word = spec.word.trim();
    if word.is_empty() {
        return Err("a business stamp needs a word, for example 承認".into());
    }
    check_text("word", word, MAX_BUSINESS_STAMP_WORD_CHARS)?;
    let name = spec.name.trim();
    check_text("name", name, MAX_BUSINESS_STAMP_NAME_CHARS)?;
    let date = spec.date.map(|d| seal_date_text(d, spec.era)).transpose()?;
    check_chars("word", word)?;
    check_chars("name", name)?;

    let ls = LINE_SCALE * s;
    // Words to write: (text, bold, size).
    let mut texts: Vec<(&str, bool, f64)> = vec![(word, true, s)];
    if let Some(d) = date.as_deref() {
        texts.push((d, false, ls));
    }
    if !name.is_empty() {
        texts.push((name, false, ls));
    }
    let widths: Vec<f64> =
        texts.iter().map(|(t, bold, size)| text_width(&item(t, [0.0, 0.0, 100.0 * size, 2.0 * size], *size, false, *bold, [0.0; 3]))).collect();

    // The frame, measured in from the stamp's edge.
    let (strokes, gap) = match spec.frame {
        StampFrame::Rounded => (vec![0.09 * s], 0.0),
        StampFrame::Double => (vec![0.09 * s, 0.035 * s], 0.13 * s),
    };
    let outer = strokes.first().copied().unwrap_or(0.0) / 2.0 + 0.03 * s;
    let inner_edge = outer + gap + strokes.last().copied().unwrap_or(0.0) / 2.0;
    let (hpad, vpad, line_gap, rule_gap) = (0.32 * s, 0.2 * s, 0.08 * s, 0.14 * s);
    let rule_stroke = 0.035 * s;
    let extra = texts.len().saturating_sub(1);
    let mut content_h = INK_HEIGHT * s;
    if extra > 0 {
        content_h += 2.0 * rule_gap + rule_stroke + extra as f64 * INK_HEIGHT * ls + (extra - 1) as f64 * line_gap;
    }
    let height = 2.0 * (inner_edge + vpad) + content_h;
    let widest = widths.iter().copied().fold(0.0, f64::max);
    let width = (2.0 * (inner_edge + hpad) + widest).max(height);

    let mut outlines = vec![[outer, outer, width - outer, height - outer]];
    if spec.frame == StampFrame::Double {
        let i = outer + gap;
        outlines.push([i, i, width - i, height - i]);
    }
    let corner = match spec.frame {
        StampFrame::Rounded => 0.22 * (height - 2.0 * outer),
        StampFrame::Double => 0.0,
    };
    let mut lines = Vec::new();
    let mut rule = None;
    let mut y = height - inner_edge - vpad;
    for (k, ((text, bold, size), w)) in texts.iter().zip(&widths).enumerate() {
        if k == 1 {
            y -= rule_gap + rule_stroke / 2.0;
            rule = Some(([inner_edge, y, width - inner_edge, y], rule_stroke));
            y -= rule_stroke / 2.0 + rule_gap;
        } else if k > 1 {
            y -= line_gap;
        }
        let half = INK_HEIGHT * size / 2.0;
        let yc = y - half;
        let top = yc + CENTRE_BELOW_TOP * size;
        lines.push(StampLine {
            text: (*text).to_owned(),
            bold: *bold,
            size: *size,
            rect: [0.0, top - 1.2 * size, width, top],
            ink: [(width - w) / 2.0, yc - half, (width + w) / 2.0, yc + half],
        });
        y -= 2.0 * half;
    }
    Ok(BusinessStampLayout { width, height, frame: spec.frame, outlines, strokes, corner, rule, lines })
}

/// A rounded rectangle path (closed, not painted) on the centre line `r` with corner radius `c`.
fn rounded_rect(r: [f64; 4], c: f64) -> String {
    let [x0, y0, x1, y1] = r;
    let c = c.min((x1 - x0) / 2.0).min((y1 - y0) / 2.0).max(0.0);
    let k = 0.552_284_75 * c;
    let p = |x: f64, y: f64| format!("{} {}", num(x), num(y));
    [
        format!("{} m", p(x0 + c, y0)),
        format!("{} l", p(x1 - c, y0)),
        format!("{} {} {} c", p(x1 - c + k, y0), p(x1, y0 + c - k), p(x1, y0 + c)),
        format!("{} l", p(x1, y1 - c)),
        format!("{} {} {} c", p(x1, y1 - c + k), p(x1 - c + k, y1), p(x1 - c, y1)),
        format!("{} l", p(x0 + c, y1)),
        format!("{} {} {} c", p(x0 + c - k, y1), p(x0, y1 - c + k), p(x0, y1 - c)),
        format!("{} l", p(x0, y0 + c)),
        format!("{} {} {} c", p(x0, y0 + c - k), p(x0 + c - k, y0), p(x0 + c, y0)),
        "h\n".to_owned(),
    ]
    .join("\n")
}

/// The outline and the rule, stroked in the stamp's colour.
fn vector_ops(l: &BusinessStampLayout, color: [f64; 3]) -> Vec<u8> {
    let [r, g, b] = color.map(num);
    let mut s = format!("q {r} {g} {b} RG 1 j 1 J\n");
    for (o, w) in l.outlines.iter().zip(&l.strokes) {
        s += &format!("{} w\n", num(*w));
        match l.frame {
            StampFrame::Rounded => s += &format!("{}S\n", rounded_rect(*o, l.corner)),
            StampFrame::Double => s += &format!("{} {} {} {} re S\n", num(o[0]), num(o[1]), num(o[2] - o[0]), num(o[3] - o[1])),
        }
    }
    if let Some(([x0, y0, x1, y1], w)) = l.rule {
        s += &format!("{} w {} {} m {} {} l S\n", num(w), num(x0), num(y0), num(x1), num(y1));
    }
    s += "Q\n";
    s.into_bytes()
}

/// The business stamp as a one-page PDF the size of the stamp (nothing behind it).
pub fn business_stamp_pdf(spec: &BusinessStampSpec) -> Result<Vec<u8>, String> {
    let layout = business_stamp_layout(spec)?;
    let mut doc = pdfcraft_create::blank(layout.width, layout.height, 1).map_err(|e| e.to_string())?;
    for t in &layout.lines {
        add_content(&mut doc, 0, &Content::Text(item(&t.text, t.rect, t.size, false, t.bold, spec.color))).map_err(|e| e.to_string())?;
    }
    pdfcraft_edit::stamp(&mut doc, 0, "Stamp", vector_ops(&layout, spec.color)).map_err(|e| e.to_string())?;
    pdfcraft_cos::write_full(&doc, &pdfcraft_cos::SaveOptions::default()).map_err(|e| e.to_string())
}

/// What the stamp reads, as its comment's contents: the word, then the date and the name.
pub fn business_stamp_description(spec: &BusinessStampSpec) -> String {
    let date = spec.date.and_then(|d| seal_date_text(d, spec.era).ok()).unwrap_or_default();
    [spec.word.trim(), date.as_str(), spec.name.trim()].iter().filter(|t| !t.is_empty()).copied().collect::<Vec<_>>().join(" ")
}

/// The edit that stamps `spec` centred on `at` (user space) on `page` (0-based), upright as the
/// page is displayed (`info` gives its rotation), at the stamp's own size.
pub fn business_stamp_edit(spec: &BusinessStampSpec, page: usize, info: &PageInfo, at: [f64; 2], author: &str) -> Result<Edit, String> {
    let layout = business_stamp_layout(spec)?;
    if !at.iter().all(|v| v.is_finite()) {
        return Err("the stamp's position must be a number".into());
    }
    // As displayed the stamp is width x height; a quarter turn swaps its sides in user space.
    let (w, h) = if matches!(info.rotation, 90 | 270) { (layout.height, layout.width) } else { (layout.width, layout.height) };
    let rect = [at[0] - w / 2.0, at[1] - h / 2.0, at[0] + w / 2.0, at[1] + h / 2.0];
    let bytes = business_stamp_pdf(spec)?;
    Ok(Edit::AddCustomStamp {
        page,
        rect,
        name: business_stamp_description(spec),
        file: MarkFile { name: "stamp.pdf".into(), bytes: std::sync::Arc::new(bytes), page: 0 },
        author: author.into(),
        line: None,
    })
}

#[cfg(test)]
mod tests;
