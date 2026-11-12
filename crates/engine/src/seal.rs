//! pdfcraft-JPN: Japanese electronic seals (電子印鑑 / ハンコ).
//!
//! Two kinds are made here:
//!
//! * **認印 / 名前印** ([`SealKind::Name`]): a ring (or a square, 角印) with the name set
//!   vertically inside, sized so every character cell lies inside the outline.
//! * **データ印** ([`SealKind::Date`]): a ring cut by two horizontal rules into three bands: an
//!   organisation or department on top, the date in the middle (`2026.11.11`, or with an era,
//!   `令和8.11.11`) and the name below.
//!
//! **How a seal is stored.** The seal is drawn as a one-page vector PDF (the outline and rules as
//! paths, the text as added text in the Japanese face from craft-fonts, with its `/ActualText`),
//! and placed as a *custom stamp* ([`Edit::AddCustomStamp`]): a Stamp annotation whose appearance
//! is that page as a form XObject. That makes it a comment like any other stamp (select, move,
//! resize, delete, undo, list in the comment summary) and keeps the seal vector at any zoom. The
//! other route, page content (the Add text machinery), would make the circle and the text
//! separate items to move; a stamp keeps them one.
//!
//! The text face is the document's Japanese face (BIZ UDPGothic, a Gothic); without a Japanese
//! face in the build a seal with Japanese text is refused with a message saying so.

use pdfcraft_edit::added::{AddedText, Align, Content, Family, add_content, text_width};
use pdfcraft_edit::era::{era_year_text, japanese_era};

use crate::Edit;
use crate::pictures::MarkFile;

/// The seal colour until one is chosen: vermilion (朱色).
pub const DEFAULT_SEAL_COLOR: [f64; 3] = [0.85, 0.12, 0.10];
/// The seal's side when none is given, in points (about 17 mm, a usual 認印 size).
pub const DEFAULT_SEAL_SIZE: f64 = 48.0;
/// The smallest and largest seal side, in points.
pub const SEAL_SIZE_RANGE: (f64, f64) = (20.0, 200.0);
/// Most characters of a name set vertically.
pub const MAX_SEAL_NAME_CHARS: usize = 6;
/// Most characters of a name that is all ASCII (set on one line).
pub const MAX_SEAL_ASCII_CHARS: usize = 12;
/// Most characters of the organisation line of a date seal.
pub const MAX_SEAL_ORG_CHARS: usize = 12;

/// The design grid: everything is laid out on a 100 x 100 square and scaled to the seal's size.
const GRID: f64 = 100.0;
/// The outline's centre line sits this far in from the grid's centre.
const OUTLINE_RADIUS: f64 = 46.0;
const RING_STROKE: f64 = 3.6;
const RULE_STROKE: f64 = 2.2;
/// Room kept between the text and the outline's inner edge.
const INNER_GAP: f64 = 3.0;
/// The two rules of a date seal sit this far above and below the centre.
const RULE_OFFSET: f64 = 15.0;
/// Height of the ink of a line of text, as a share of its size (a safe overestimate).
const INK_HEIGHT: f64 = 0.9;
/// Distance of a horizontal line's visual centre below the top of its text box, in em (the
/// baseline sits 0.95 em below the top; the centre of an em box is 0.38 em above the baseline).
const CENTRE_BELOW_TOP: f64 = 0.57;
/// Column pitch of vertical text, in em (see `added::vertical`).
const PITCH: f64 = 1.2;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SealKind {
    /// 認印 / 名前印: the name alone.
    Name,
    /// データ印: organisation, date and name in three bands.
    Date,
}

impl SealKind {
    pub fn parse(s: &str) -> Option<SealKind> {
        match s.trim().to_lowercase().as_str() {
            "name" | "mitome" | "認印" | "名前印" => Some(SealKind::Name),
            "date" | "data" | "データ印" | "日付印" => Some(SealKind::Date),
            _ => None,
        }
    }

    pub fn id(self) -> &'static str {
        match self {
            SealKind::Name => "name",
            SealKind::Date => "date",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SealShape {
    Circle,
    /// 角印.
    Square,
}

impl SealShape {
    pub fn parse(s: &str) -> Option<SealShape> {
        match s.trim().to_lowercase().as_str() {
            "circle" | "round" | "丸" | "丸印" => Some(SealShape::Circle),
            "square" | "角" | "角印" => Some(SealShape::Square),
            _ => None,
        }
    }

    pub fn id(self) -> &'static str {
        match self {
            SealShape::Circle => "circle",
            SealShape::Square => "square",
        }
    }
}

/// What a seal says and how it looks.
#[derive(Clone, Debug, PartialEq)]
pub struct SealSpec {
    pub kind: SealKind,
    pub shape: SealShape,
    /// The name (both kinds).
    pub name: String,
    /// The organisation or department (date seal only).
    pub org: String,
    /// The date as (year, month, day) in the Gregorian calendar (date seal only).
    pub date: (i64, u32, u32),
    /// Write the date with its era (令和8.11.11) instead of Western (2026.11.11).
    pub era: bool,
    /// The seal's colour, RGB from 0 to 1.
    pub color: [f64; 3],
    /// The side of the seal, in points.
    pub size: f64,
}

impl Default for SealSpec {
    fn default() -> Self {
        SealSpec {
            kind: SealKind::Name,
            shape: SealShape::Circle,
            name: String::new(),
            org: String::new(),
            date: (1970, 1, 1),
            era: false,
            color: DEFAULT_SEAL_COLOR,
            size: DEFAULT_SEAL_SIZE,
        }
    }
}

/// The date as a seal writes it: `2026.11.11`, or with `era` `令和8.11.11` (`令和元.5.1` in an
/// era's first year). `Err` for a date that doesn't exist, or before 明治 with an era.
pub fn seal_date_text((y, m, d): (i64, u32, u32), era: bool) -> Result<String, String> {
    let leap = (y % 4 == 0 && y % 100 != 0) || y % 400 == 0;
    let days = match m {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if leap => 29,
        2 => 28,
        _ => 0,
    };
    if !(1..=9999).contains(&y) || d == 0 || d > days {
        return Err(format!("{y}-{m}-{d} is not a date"));
    }
    if !era {
        return Ok(format!("{y}.{m:02}.{d:02}"));
    }
    let (e, year) = japanese_era(y, m, d).ok_or_else(|| format!("{y}-{m}-{d} is before the Meiji era, which began in 1868; turn the era off"))?;
    Ok(format!("{}{}.{m}.{d}", e.name, era_year_text(year)))
}

/// Parse a date typed as `2026-11-11`, `2026.11.11`, `2026/11/11` or `2026年11月11日`.
pub fn parse_seal_date(s: &str) -> Option<(i64, u32, u32)> {
    let digits: Vec<&str> = s.split(|c: char| !c.is_ascii_digit()).filter(|p| !p.is_empty()).collect();
    let [y, m, d] = digits.as_slice() else { return None };
    let date = (y.parse().ok()?, m.parse().ok()?, d.parse().ok()?);
    seal_date_text(date, false).ok()?;
    Some(date)
}

/// One piece of text on a seal, in the seal's own space (points, origin bottom-left).
#[derive(Clone, Debug, PartialEq)]
pub struct SealText {
    pub text: String,
    pub vertical: bool,
    pub bold: bool,
    pub size: f64,
    /// The text box handed to the Add text machinery.
    pub rect: [f64; 4],
    /// Boxes the ink may reach: one per character cell (vertical) or one for the line.
    pub ink: Vec<[f64; 4]>,
}

/// Where everything on a seal goes.
#[derive(Clone, Debug, PartialEq)]
pub struct SealLayout {
    pub size: f64,
    pub shape: SealShape,
    pub centre: [f64; 2],
    /// The outline's centre line: a circle's radius, or a square's half side.
    pub radius: f64,
    pub ring_stroke: f64,
    pub rule_stroke: f64,
    /// The horizontal rules of a date seal, `[x0, y, x1, y]`.
    pub rules: Vec<[f64; 4]>,
    pub texts: Vec<SealText>,
}

impl SealLayout {
    /// The radius of the area inside the outline (a circle's), or half its side (a square's).
    pub fn inner(&self) -> f64 {
        self.radius - self.ring_stroke / 2.0
    }
}

fn is_ascii_text(s: &str) -> bool {
    s.is_ascii()
}

/// A message for Japanese text without a Japanese face, else `None`.
fn missing_font(text: &str) -> Option<String> {
    let c = text.chars().find(|&c| pdfcraft_fonts::shaped_text::is_japanese(c) && !pdfcraft_fonts::japanese_has(c))?;
    Some(format!(
        "no Japanese font is embedded in this build, so {c:?} can't be drawn on a seal; use a build with the Japanese fonts (CRAFT_FONTS_DIR, see AGENTS.md) or write the seal in ASCII letters"
    ))
}

fn item(text: &str, rect: [f64; 4], size: f64, vertical: bool, bold: bool, color: [f64; 3]) -> AddedText {
    AddedText {
        rect,
        text: text.into(),
        family: Family::Helvetica,
        bold,
        italic: false,
        size,
        color,
        ink: None,
        align: Align::Center,
        char_spacing: 0.0,
        aligns: Vec::new(),
        vertical,
    }
}

/// The width of `text` at 10 points.
fn width_per_10(text: &str, bold: bool) -> f64 {
    text_width(&item(text, [0.0, 0.0, GRID * 10.0, 20.0], 10.0, false, bold, [0.0; 3]))
}

/// Half the width of the inside of the outline at `dy` from the centre (0 outside it).
fn half_chord(shape: SealShape, inner: f64, dy: f64) -> f64 {
    match shape {
        SealShape::Circle => (inner * inner - dy * dy).max(0.0).sqrt(),
        SealShape::Square if dy <= inner => inner,
        SealShape::Square => 0.0,
    }
}

/// The columns of a vertical name: one up to three characters, else two.
fn columns_of(chars: &[char]) -> Vec<String> {
    let n = chars.len();
    let first = if n <= 3 { n } else { n.div_ceil(2) };
    chars.chunks(first.max(1)).map(|c| c.iter().collect()).collect()
}

/// A horizontal line fitted in the band around `yc`: the largest size (at most `cap`) at which it
/// is no taller than `max_half` either side of `yc` and no wider than the outline's inside there.
fn fit_line(shape: SealShape, text: &str, bold: bool, yc: f64, max_half: f64, cap: f64, inner: f64) -> Result<SealText, String> {
    let per10 = width_per_10(text, bold);
    let centre = GRID / 2.0;
    let mut size = cap;
    while size >= 5.0 {
        let half_h = INK_HEIGHT * size / 2.0;
        let width = per10 * size / 10.0;
        let reach = (yc - centre).abs() + half_h;
        if half_h <= max_half && width <= 2.0 * half_chord(shape, inner, reach) - 3.0 {
            let top = yc + CENTRE_BELOW_TOP * size;
            return Ok(SealText {
                text: text.into(),
                vertical: false,
                bold,
                size,
                rect: [0.0, top - 1.2 * size, GRID, top],
                ink: vec![[centre - width / 2.0, yc - half_h, centre + width / 2.0, yc + half_h]],
            });
        }
        size -= 0.25;
    }
    Err(format!("\"{text}\" is too long to fit on the seal; shorten it"))
}

fn check_text(label: &str, text: &str, max: usize) -> Result<(), String> {
    if text.chars().any(char::is_control) {
        return Err(format!("the {label} can't contain control characters"));
    }
    if text.chars().count() > max {
        return Err(format!("the {label} can have at most {max} characters"));
    }
    Ok(())
}

/// Characters a seal draws: Japanese text, and ASCII letters, digits and common punctuation.
fn is_seal_char(c: char) -> bool {
    c == ' ' || c.is_ascii_graphic() || (pdfcraft_fonts::shaped_text::is_japanese(c) && !c.is_ascii())
}

/// Lay out a seal on the 100 x 100 grid.
fn layout_grid(spec: &SealSpec) -> Result<SealLayout, String> {
    let name = spec.name.trim();
    let org = spec.org.trim();
    if name.is_empty() {
        return Err("a seal needs a name".into());
    }
    let name_max = if is_ascii_text(name) { MAX_SEAL_ASCII_CHARS } else { MAX_SEAL_NAME_CHARS };
    check_text("name", name, name_max)?;
    let org = if spec.kind == SealKind::Date { org } else { "" };
    check_text("organisation", org, MAX_SEAL_ORG_CHARS)?;
    let date_text = match spec.kind {
        SealKind::Date => Some(seal_date_text(spec.date, spec.era)?),
        SealKind::Name => None,
    };
    for text in [name, org, date_text.as_deref().unwrap_or("")] {
        if let Some(e) = missing_font(text) {
            return Err(e);
        }
        if let Some(c) = text.chars().find(|&c| !is_seal_char(c)) {
            return Err(format!("{c:?} can't be drawn on a seal: use Japanese characters, letters and digits"));
        }
    }
    let centre = GRID / 2.0;
    let inner = OUTLINE_RADIUS - RING_STROKE / 2.0;
    let usable = inner - INNER_GAP;
    let mut layout = SealLayout {
        size: GRID,
        shape: spec.shape,
        centre: [centre, centre],
        radius: OUTLINE_RADIUS,
        ring_stroke: RING_STROKE,
        rule_stroke: RULE_STROKE,
        rules: Vec::new(),
        texts: Vec::new(),
    };
    match spec.kind {
        SealKind::Name if is_ascii_text(name) => {
            // A Latin name on one line, as large as the outline allows.
            layout.texts.push(fit_line(spec.shape, name, true, centre, usable, 30.0, inner)?);
        }
        SealKind::Name => {
            let chars: Vec<char> = name.chars().collect();
            let cols = columns_of(&chars);
            let rows = cols.iter().map(|c| c.chars().count()).max().unwrap_or(1) as f64;
            let ncols = cols.len() as f64;
            // The block of character cells at size 1, in em.
            let (w1, h1) = ((ncols - 1.0) * PITCH + 1.0, rows);
            let size = match spec.shape {
                SealShape::Circle => usable / (w1 / 2.0).hypot(h1 / 2.0),
                SealShape::Square => (2.0 * usable / w1).min(2.0 * usable / h1),
            }
            .min(1.2 * usable);
            let (w, h) = (w1 * size, h1 * size);
            let first_centre = centre + w / 2.0 - size / 2.0;
            let right = first_centre + PITCH * size / 2.0;
            let top = centre + h / 2.0;
            let mut ink = Vec::new();
            for (j, col) in cols.iter().enumerate() {
                let xc = first_centre - PITCH * size * j as f64;
                for i in 0..col.chars().count() {
                    ink.push([xc - size / 2.0, top - (i as f64 + 1.0) * size, xc + size / 2.0, top - i as f64 * size]);
                }
            }
            layout.texts.push(SealText {
                text: cols.join("\n"),
                vertical: true,
                bold: false,
                size,
                rect: [right - ncols * PITCH * size, top - rows * size * 1.05, right, top],
                ink,
            });
        }
        SealKind::Date => {
            layout.rules = vec![
                [centre - inner, centre + RULE_OFFSET, centre + inner, centre + RULE_OFFSET],
                [centre - inner, centre - RULE_OFFSET, centre + inner, centre - RULE_OFFSET],
            ];
            let band_half = RULE_OFFSET - RULE_STROKE / 2.0 - 1.5;
            if !org.is_empty() {
                let (lo, hi) = (centre + RULE_OFFSET + RULE_STROKE / 2.0, centre + usable);
                layout.texts.push(fit_line(spec.shape, org, false, (lo + hi) / 2.0, (hi - lo) / 2.0 - 0.5, 12.0, inner)?);
            }
            layout.texts.push(fit_line(spec.shape, date_text.as_deref().unwrap_or_default(), true, centre, band_half, 14.0, inner)?);
            let (lo, hi) = (centre - usable, centre - RULE_OFFSET - RULE_STROKE / 2.0);
            layout.texts.push(fit_line(spec.shape, name, false, (lo + hi) / 2.0, (hi - lo) / 2.0 - 0.5, 16.0, inner)?);
        }
    }
    Ok(layout)
}

/// A seal's layout in points, scaled to `spec.size`. `Err` says what to change.
pub fn seal_layout(spec: &SealSpec) -> Result<SealLayout, String> {
    if !spec.size.is_finite() || !(SEAL_SIZE_RANGE.0..=SEAL_SIZE_RANGE.1).contains(&spec.size) {
        return Err(format!("the seal size must be {} to {} points", SEAL_SIZE_RANGE.0, SEAL_SIZE_RANGE.1));
    }
    if !spec.color.iter().all(|v| v.is_finite() && (0.0..=1.0).contains(v)) {
        return Err("the colour must be RGB values from 0 to 1".into());
    }
    let mut l = layout_grid(spec)?;
    let k = spec.size / GRID;
    l.size *= k;
    l.centre = l.centre.map(|v| v * k);
    l.radius *= k;
    l.ring_stroke *= k;
    l.rule_stroke *= k;
    for r in &mut l.rules {
        *r = r.map(|v| v * k);
    }
    for t in &mut l.texts {
        t.size *= k;
        t.rect = t.rect.map(|v| v * k);
        for b in &mut t.ink {
            *b = b.map(|v| v * k);
        }
    }
    Ok(l)
}

fn num(v: f64) -> String {
    let s = format!("{v:.3}");
    s.trim_end_matches('0').trim_end_matches('.').to_string()
}

/// The path operators of the outline and rules, stroked in the seal colour.
fn vector_ops(l: &SealLayout, color: [f64; 3]) -> Vec<u8> {
    let [r, g, b] = color.map(num);
    let [cx, cy] = l.centre;
    let mut s = format!("q {r} {g} {b} RG 1 j 1 J {} w\n", num(l.ring_stroke));
    match l.shape {
        SealShape::Circle => {
            let (rad, k) = (l.radius, 0.552_284_75 * l.radius);
            let pts = [
                (cx + rad, cy),
                (cx + rad, cy + k),
                (cx + k, cy + rad),
                (cx, cy + rad),
                (cx - k, cy + rad),
                (cx - rad, cy + k),
                (cx - rad, cy),
                (cx - rad, cy - k),
                (cx - k, cy - rad),
                (cx, cy - rad),
                (cx + k, cy - rad),
                (cx + rad, cy - k),
                (cx + rad, cy),
            ];
            let p = |i: usize| pts.get(i).map_or(String::new(), |(x, y)| format!("{} {}", num(*x), num(*y)));
            s += &format!("{} m\n", p(0));
            for seg in 0..4 {
                s += &format!("{} {} {} c\n", p(seg * 3 + 1), p(seg * 3 + 2), p(seg * 3 + 3));
            }
            s += "h S\n";
        }
        SealShape::Square => {
            s += &format!("{} {} {} {} re S\n", num(cx - l.radius), num(cy - l.radius), num(2.0 * l.radius), num(2.0 * l.radius));
        }
    }
    s += &format!("{} w\n", num(l.rule_stroke));
    for [x0, y0, x1, y1] in &l.rules {
        s += &format!("{} {} m {} {} l S\n", num(*x0), num(*y0), num(*x1), num(*y1));
    }
    s += "Q\n";
    s.into_bytes()
}

/// The seal as a one-page PDF (the page is `spec.size` points square, nothing behind the seal).
pub fn seal_pdf(spec: &SealSpec) -> Result<Vec<u8>, String> {
    let layout = seal_layout(spec)?;
    let mut doc = pdfcraft_create::blank(layout.size, layout.size, 1).map_err(|e| e.to_string())?;
    for t in &layout.texts {
        let text = item(&t.text, t.rect, t.size, t.vertical, t.bold, spec.color);
        add_content(&mut doc, 0, &Content::Text(text)).map_err(|e| e.to_string())?;
    }
    pdfcraft_edit::stamp(&mut doc, 0, "Seal", vector_ops(&layout, spec.color)).map_err(|e| e.to_string())?;
    pdfcraft_cos::write_full(&doc, &pdfcraft_cos::SaveOptions::default()).map_err(|e| e.to_string())
}

/// The text a seal stands for: its comment's contents (印鑑 and what the seal reads).
pub fn seal_description(spec: &SealSpec) -> String {
    let name = spec.name.trim();
    match spec.kind {
        SealKind::Name => format!("印鑑 {name}"),
        SealKind::Date => {
            let date = seal_date_text(spec.date, spec.era).unwrap_or_default();
            format!("印鑑 {} {date} {name}", spec.org.trim()).split_whitespace().collect::<Vec<_>>().join(" ")
        }
    }
}

#[cfg(test)]
mod tests;

/// The edit that stamps `spec` centred on `at` (a point in user space) on `page` (0-based):
/// a Stamp comment whose appearance is the seal.
pub fn seal_edit(spec: &SealSpec, page: usize, at: [f64; 2], author: &str) -> Result<Edit, String> {
    if !at.iter().all(|v| v.is_finite()) {
        return Err("the seal's position must be a number".into());
    }
    let bytes = seal_pdf(spec)?;
    Ok(Edit::AddCustomStamp {
        page,
        rect: [at[0], at[1], at[0], at[1]],
        name: seal_description(spec),
        file: MarkFile { name: "seal.pdf".into(), bytes: std::sync::Arc::new(bytes), page: 0 },
        author: author.into(),
        line: None,
    })
}
