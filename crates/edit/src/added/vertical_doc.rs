//! pdfcraft-JPN: vertical Japanese documents from plain text (縦書き文書).
//!
//! Three templates, all drawn from vector graphics generated here (no asset files) and the
//! Japanese document face (Mincho or Gothic, craft-fonts):
//!
//! * **原稿用紙** (`Template::Genko`): 20 × 20 squares by default, one character per square, the
//!   columns running right to left in two halves of 10 around the centre fold (柱). A paragraph
//!   starts a new column and its first square is left empty (字下げ). 、。 and the other
//!   punctuation take the face's vertical forms, which sit in the upper right of the square.
//!   行頭禁則: a character that may not start a column (、。」） small kana, ー and the like) that
//!   would fall at the top of one is hung (ぶら下げ) below the last square of the previous column,
//!   outside the grid; when two or more such characters follow each other, the last character
//!   before them goes down with them instead (追い出し). An opening bracket never ends a column
//!   (行末禁則). Small kana keep their own square. 縦中横: one or two half-width digits (or `!!`,
//!   `!?`, `?!`) between Japanese characters share one square; any other half-width character
//!   (letters, longer digit runs) takes a square of its own, upright. Ruby (`｜親《るび》`) is
//!   drawn at about a third of the square's size across the right-hand line of its base squares.
//!   An optional title (first column, indented three squares) and author (second column, bottom
//!   aligned) open the first page; the text starts after one blank column.
//! * **便箋** (`Template::Binsen`): vertical ruled lines and normal vertical text flowing in the
//!   lanes between them, laid out by [`super::vertical`].
//! * **Plain** (`Template::Plain`): vertical text with margins and no lines.
//!
//! Every page is `/Direction /R2L` (right binding) through the catalog's `/ViewerPreferences`.
//! Each column is one `/ActualText` span in reading order (right to left, top to bottom), so text
//! extraction and search read the document as written.

use std::collections::BTreeMap;

use pdfcraft_cos::{Dict, Document, ObjRef, Object, PdfString, Stream};
use pdfcraft_fonts::shaped_text::{ShapedFonts, is_japanese};
use pdfcraft_fonts::{ShapedCluster, literal, no_line_end, no_line_start, win_ansi_encodable};

use super::layout::ruby::{self, Segment};
use super::layout::thai_error;
use super::types::{AddedText, Family};
use super::vertical::{Cell, columns_styled, draw_cell, is_combined};
use crate::{EditError, n};

/// The most characters of text a vertical document takes.
pub const MAX_CHARS: usize = 2_000_000;
/// The most pages one document gets.
pub const MAX_PAGES: usize = 5_000;
/// The most squares or columns one page is given.
const MAX_GRID: usize = 60;

/// The paper the text is set on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Template {
    /// 原稿用紙.
    Genko,
    /// 便箋 with vertical rules.
    Binsen,
    /// Vertical text, no lines.
    Plain,
}

impl Template {
    /// `genko`, `binsen` or `plain` (also the Japanese names).
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_lowercase().as_str() {
            "genko" | "genkou" | "genko_yoshi" | "原稿用紙" => Some(Self::Genko),
            "binsen" | "letter" | "便箋" => Some(Self::Binsen),
            "plain" | "none" | "縦書き" | "縦書き文書" => Some(Self::Plain),
            _ => None,
        }
    }
}

/// The page sizes offered (JIS B sizes).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PageSize {
    A4,
    A5,
    B4,
    B5,
}

impl PageSize {
    /// `a4`, `a5`, `b4` or `b5`.
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_lowercase().as_str() {
            "a4" => Some(Self::A4),
            "a5" => Some(Self::A5),
            "b4" => Some(Self::B4),
            "b5" => Some(Self::B5),
            _ => None,
        }
    }

    /// Width and height in points, portrait.
    pub fn portrait(self) -> (f64, f64) {
        match self {
            Self::A4 => (595.276, 841.89),
            Self::A5 => (419.528, 595.276),
            Self::B4 => (728.504, 1031.811),
            Self::B5 => (515.906, 728.504),
        }
    }
}

/// A named grid colour: `green` and `red` (red-brown), or `#RRGGBB`.
pub fn parse_colour(s: &str) -> Option<[f64; 3]> {
    match s.trim().to_lowercase().as_str() {
        "green" | "緑" => Some(GREEN),
        "red" | "brown" | "red-brown" | "赤" => Some(RED_BROWN),
        "gray" | "grey" => Some([0.55; 3]),
        h => {
            let h = h.strip_prefix('#')?;
            if h.len() != 6 || !h.is_ascii() {
                return None;
            }
            let v = |i: usize| u8::from_str_radix(h.get(i..i + 2)?, 16).ok().map(|b| f64::from(b) / 255.0);
            Some([v(0)?, v(2)?, v(4)?])
        }
    }
}

/// The muted green of the default 原稿用紙.
pub const GREEN: [f64; 3] = [0.40, 0.62, 0.45];
/// The red-brown 原稿用紙 also comes in.
pub const RED_BROWN: [f64; 3] = [0.71, 0.40, 0.31];

/// What to make.
#[derive(Clone, Debug)]
pub struct VerticalOptions {
    /// UTF-8 text; `\n` ends a paragraph. May hold Aozora-style ruby (`｜親《るび》`, `漢字《かんじ》`).
    pub text: String,
    pub template: Template,
    pub page_size: PageSize,
    /// `None`: landscape for 原稿用紙, portrait otherwise.
    pub landscape: Option<bool>,
    /// Mincho (serif) when true, Gothic when false.
    pub serif: bool,
    /// Points. 原稿用紙: at most the square (default 80% of it); others default to 14 (便箋) or 12.
    pub font_size: Option<f64>,
    /// 原稿用紙: squares per row of the page (columns of the page, default 20); 便箋 and plain: the
    /// number of lines (default from the font size).
    pub columns: Option<usize>,
    /// 原稿用紙 only: squares per column (default 20).
    pub rows: Option<usize>,
    pub grid_colour: [f64; 3],
    /// Page margin in points for 便箋 and plain.
    pub margin: Option<f64>,
    /// 原稿用紙 only: the title and author of the first page. Both also go into the document's
    /// information.
    pub title: Option<String>,
    pub author: Option<String>,
}

impl VerticalOptions {
    /// 原稿用紙 on landscape A4 in green, set in Mincho.
    pub fn new(text: impl Into<String>, template: Template) -> Self {
        Self {
            text: text.into(),
            template,
            page_size: PageSize::A4,
            landscape: None,
            serif: true,
            font_size: None,
            columns: None,
            rows: None,
            grid_colour: GREEN,
            margin: None,
            title: None,
            author: None,
        }
    }
}

/// What was made, besides the document.
#[derive(Debug)]
pub struct VerticalCreated {
    pub document: Document,
    pub pages: usize,
    /// Columns of text, blank ones included.
    pub columns: usize,
    /// How many characters were hung below a column (ぶら下げ).
    pub hung: usize,
    /// The page size in points, as shown.
    pub page: (f64, f64),
}

/// A character in a square of 原稿用紙, with its ruby (the clusters and how many squares of base
/// they stand over).
#[derive(Clone, Debug)]
struct Sq {
    cell: Option<Cell>,
    ruby: Option<(Vec<ShapedCluster>, usize)>,
}

impl Sq {
    fn blank() -> Self {
        Self { cell: None, ruby: None }
    }

    fn text(&self) -> &str {
        self.cell.as_ref().map_or("", Cell::text)
    }

    /// Whether the character may not start a column.
    fn no_start(&self) -> bool {
        self.text().chars().next().is_some_and(no_line_start)
    }

    fn no_end(&self) -> bool {
        self.text().chars().next().is_some_and(no_line_end)
    }
}

/// One column of 原稿用紙: squares with their row, and a character hung below the grid.
#[derive(Default, Debug)]
struct Col {
    items: Vec<(usize, Sq)>,
    hung: Option<Sq>,
}

/// Make a vertical Japanese document. A clear error when no Japanese face is embedded.
pub fn create_vertical_document(o: &VerticalOptions) -> Result<VerticalCreated, EditError> {
    // Line ends, tabs and other control characters (a form feed from a text dump) are made plain
    // first, in the title and author too: a control character would be drawn as "?".
    let cleaned = VerticalOptions {
        text: plain_text(&o.text, true),
        title: o.title.as_deref().map(|t| plain_text(t, false)),
        author: o.author.as_deref().map(|a| plain_text(a, false)),
        ..o.clone()
    };
    let o = &cleaned;
    if pdfcraft_fonts::document_japanese_font_for_style(o.serif, false).is_none() {
        return Err(EditError::Invalid(
            "this build of PdfCraft has no Japanese font embedded, so a vertical Japanese document can't be made; build it with CRAFT_FONTS_DIR pointing at a craft-fonts checkout"
                .into(),
        ));
    }
    let text = o.text.clone();
    if text.chars().count() > MAX_CHARS {
        return Err(EditError::Invalid(format!("the text is longer than {MAX_CHARS} characters")));
    }
    if let Some((line, c)) = undrawable(&text, o.serif) {
        return Err(EditError::Invalid(format!(
            "line {line}: \"{}\" (U+{:04X}) can't be drawn in a vertical Japanese document: the Japanese font has no glyph for it",
            c.escape_debug(),
            u32::from(c)
        )));
    }
    for (what, field) in [("title", &o.title), ("author", &o.author)] {
        if let Some((_, c)) = field.as_deref().and_then(|f| undrawable(f, o.serif)) {
            return Err(EditError::Invalid(format!(
                "the {what}: \"{}\" (U+{:04X}) can't be drawn in a vertical Japanese document: the Japanese font has no glyph for it",
                c.escape_debug(),
                u32::from(c)
            )));
        }
    }
    let landscape = o.landscape.unwrap_or(o.template == Template::Genko);
    let (pw, ph) = o.page_size.portrait();
    let page = if landscape { (ph, pw) } else { (pw, ph) };
    let mut doc = Document::new_empty();
    let family = if o.serif { Family::Times } else { Family::Helvetica };
    let out = match o.template {
        Template::Genko => genko(&mut doc, o, &text, page, family)?,
        Template::Binsen | Template::Plain => flow(&mut doc, o, &text, page, family)?,
    };
    finish(&mut doc, o, &out, page)?;
    Ok(VerticalCreated { document: doc, pages: out.pages.len(), columns: out.columns, hung: out.hung, page })
}

/// `s` with line ends as `\n` (or spaces, when `lines` is false), tabs as spaces, and every other
/// control character left out.
fn plain_text(s: &str, lines: bool) -> String {
    let s = s.replace("\r\n", "\n").replace('\r', "\n");
    s.chars()
        .filter_map(|c| match c {
            '\n' if lines => Some('\n'),
            '\n' | '\t' => Some(' '),
            c if c.is_control() => None,
            c => Some(c),
        })
        .collect()
}

/// The first character, with its 1-based line, the Japanese face (or the standard fonts) can't draw.
fn undrawable(text: &str, serif: bool) -> Option<(usize, char)> {
    for (i, line) in text.split('\n').enumerate() {
        for c in ruby::without_markup(line).chars() {
            let bad = if c.is_whitespace() || c.is_control() {
                false
            } else if pdfcraft_fonts::shaped_text::is_thai(c) || super::layout::is_arabic(c) || super::layout::is_hebrew(c) {
                true
            } else if is_japanese(c) {
                !pdfcraft_fonts::japanese_has_styled(c, serif)
            } else {
                !win_ansi_encodable(c)
            };
            if bad {
                return Some((i + 1, c));
            }
        }
    }
    None
}

struct Built {
    /// Each page's content stream.
    pages: Vec<Vec<u8>>,
    /// The fonts every page names (Type 3 fonts and the Latin font).
    fonts: Dict,
    columns: usize,
    hung: usize,
}

/// A text item standing for the document's settings, for the vertical drawing code.
fn item(size: f64, family: Family, rect: [f64; 4]) -> AddedText {
    AddedText { vertical: true, rect, size, family, ..AddedText::default() }
}

fn latin_font(doc: &mut Document, family: Family) -> ObjRef {
    let mut font = Dict::new();
    font.set(b"Type".to_vec(), Object::name("Font"));
    font.set(b"Subtype".to_vec(), Object::name("Type1"));
    font.set(b"BaseFont".to_vec(), Object::name(family.base_font(false, false)));
    font.set(b"Encoding".to_vec(), Object::name("WinAnsiEncoding"));
    doc.add(font)
}

/// The `/ActualText` span that opens a column's text.
fn open_span(out: &mut Vec<u8>, text: &str) {
    let logical: String = text.encode_utf16().map(|u| format!("{u:04X}")).collect();
    out.extend(format!("/Span <</ActualText <FEFF{logical}>>> BDC\n").bytes());
}

fn stroke_colour(c: [f64; 3]) -> String {
    format!("{} {} {} RG", n(c[0]), n(c[1]), n(c[2]))
}

// ── 便箋 and plain ──────────────────────────────────────────────────────────────────────────

fn flow(doc: &mut Document, o: &VerticalOptions, text: &str, (w, h): (f64, f64), family: Family) -> Result<Built, EditError> {
    let binsen = o.template == Template::Binsen;
    let margin = o.margin.unwrap_or(if binsen { 42.5 } else { 56.7 });
    if !margin.is_finite() || margin < 0.0 || margin * 2.0 + 20.0 > w.min(h) {
        return Err(EditError::Invalid(format!("the margin {margin} pt doesn't leave room for text on a {w:.0} × {h:.0} pt page")));
    }
    let size = o.font_size.unwrap_or(if binsen { 14.0 } else { 12.0 });
    if !size.is_finite() || !(4.0..=72.0).contains(&size) {
        return Err(EditError::Invalid("font_size must be between 4 and 72 pt".into()));
    }
    let (avail_w, avail_h) = (w - 2.0 * margin, h - 2.0 * margin);
    if avail_h < 2.0 * size {
        return Err(EditError::Invalid("the page is too short for this font size and margin".into()));
    }
    let factor = if binsen { 2.2 } else { 1.8 };
    let (count, pitch) = match o.columns {
        Some(c) => {
            if c == 0 || c > MAX_GRID * 2 {
                return Err(EditError::Invalid(format!("columns must be 1 to {}", MAX_GRID * 2)));
            }
            let pitch = avail_w / c as f64;
            if pitch < 1.2 * size {
                return Err(EditError::Invalid(format!(
                    "{c} lines of {size} pt text don't fit the page width: each line gets {pitch:.1} pt and needs at least {:.1} pt; use fewer lines, a smaller font or smaller margins",
                    1.2 * size
                )));
            }
            (c, pitch)
        }
        None => (((avail_w / (factor * size)).floor() as usize).max(1), factor * size),
    };
    // Lines are centred on the page when the width isn't divided exactly.
    let x_right = (w + count as f64 * pitch) / 2.0;
    let top = h - margin;
    let mut t = item(size, family, [0.0, 0.0, avail_w, avail_h]);
    t.text = text.to_string();
    let mut cols = columns_styled(&t, o.serif)?;
    // Trailing empty columns from a final line break add pages for nothing.
    while cols.len() > 1 && cols.last().is_some_and(Vec::is_empty) {
        cols.pop();
    }
    let per_page = count;
    let page_count = cols.len().div_ceil(per_page).max(1);
    if page_count > MAX_PAGES {
        return Err(EditError::Invalid(format!("the text needs {page_count} pages; at most {MAX_PAGES} are made")));
    }
    let mut fonts = Dict::new();
    let latin = "PCLat";
    let latin_ref = latin_font(doc, family);
    let clusters: Vec<&ShapedCluster> = cols
        .iter()
        .flatten()
        .flat_map(|c| match c {
            Cell::Upright(c) => std::slice::from_ref(c).iter().chain([].iter()),
            Cell::Ruby { base, ruby, .. } => base.iter().chain(ruby.iter()),
            Cell::Sideways(_) | Cell::Combined(_) => [].iter().chain([].iter()),
        })
        .collect();
    let shaped = ShapedFonts::new(clusters.iter().copied(), pdfcraft_fonts::shaped_glyph, &Dict::new(), &mut fonts, "PCJa", &mut |ob| {
        Object::Ref(doc.add(ob))
    })
    .map_err(|(e, c)| thai_error(e, &c.text))?;
    fonts.set(latin.as_bytes().to_vec(), Object::Ref(latin_ref));
    let mut pages = Vec::new();
    for chunk in cols.chunks(per_page) {
        let mut out: Vec<u8> = Vec::new();
        if binsen {
            // 縦罫: a line on each side of every lane.
            out.extend(format!("q {} 0.5 w\n", stroke_colour(o.grid_colour)).bytes());
            let bottom = margin;
            for k in 0..=count {
                let x = x_right - k as f64 * pitch;
                out.extend(format!("{} {} m {} {} l\n", n(x), n(top), n(x), n(bottom)).bytes());
            }
            out.extend_from_slice(b"S Q\n");
        }
        out.extend(format!("BT {}\n", t.fill_op()).bytes());
        for (i, column) in chunk.iter().enumerate() {
            if column.is_empty() {
                continue;
            }
            let centre = x_right - (i as f64 + 0.5) * pitch;
            open_span(&mut out, &column.iter().map(Cell::text).collect::<String>());
            let mut pen = top;
            let mut current: Option<&str> = None;
            let mut ruby_out = Vec::new();
            for cell in column {
                draw_cell(&mut out, &mut ruby_out, &shaped, cell, &t, latin, centre, pen, &mut current);
                pen -= cell_length(cell, &t);
            }
            out.extend_from_slice(b"EMC\n");
            if !ruby_out.is_empty() {
                out.extend_from_slice(b"/Span <</ActualText <FEFF>>> BDC\n");
                out.extend(ruby_out);
                out.extend_from_slice(b"EMC\n");
            }
        }
        out.extend_from_slice(b"ET\n");
        pages.push(out);
    }
    // Cells that hang past the column's end (ぶら下げ).
    let length = avail_h;
    let hung = cols
        .iter()
        .map(|column| {
            let mut used = 0.0;
            let mut hanging = 0usize;
            for (k, cell) in column.iter().enumerate() {
                let l = cell_length(cell, &t);
                if k > 0 && used + l > length + 1e-6 {
                    hanging += 1;
                }
                used += l;
            }
            hanging
        })
        .sum();
    Ok(Built { pages, fonts, columns: cols.len(), hung })
}

/// How far a cell moves the pen down its column (the same measure the layout used).
fn cell_length(cell: &Cell, t: &AddedText) -> f64 {
    match cell {
        Cell::Upright(c) => c.advance * t.size + t.char_spacing,
        Cell::Sideways(s) => t.family.width(s, t.size, t.bold, t.italic) + t.char_spacing * s.chars().count() as f64,
        Cell::Combined(_) => t.size + t.char_spacing,
        Cell::Ruby { base, .. } => base.iter().map(|c| c.advance * t.size + t.char_spacing).sum(),
    }
}

// ── 原稿用紙 ────────────────────────────────────────────────────────────────────────────────

/// The squares of one paragraph.
fn squares(para: &str, serif: bool) -> Result<Vec<Sq>, EditError> {
    let mut out = Vec::new();
    for seg in ruby::ruby_spans(para) {
        match seg {
            Segment::Plain(s) => out.extend(plain_squares(&s, serif)?),
            Segment::Ruby { base, ruby, .. } => {
                let mut base = plain_squares(&base, serif)?;
                let clusters = pdfcraft_fonts::shape_japanese_vertical_styled(&ruby, serif).map_err(|e| thai_error(e, &ruby))?;
                let span = base.len();
                if let Some(first) = base.first_mut().filter(|_| !clusters.is_empty()) {
                    first.ruby = Some((clusters, span));
                }
                out.extend(base);
            }
        }
    }
    Ok(out)
}

fn plain_squares(s: &str, serif: bool) -> Result<Vec<Sq>, EditError> {
    let mut out = Vec::new();
    let mut run = String::new();
    let flush = |run: &mut String, out: &mut Vec<Sq>| -> Result<(), EditError> {
        if !run.is_empty() {
            let shaped = pdfcraft_fonts::shape_japanese_vertical_styled(run, serif).map_err(|e| thai_error(e, run))?;
            out.extend(shaped.into_iter().map(|c| Sq { cell: Some(Cell::Upright(c)), ruby: None }));
            run.clear();
        }
        Ok(())
    };
    let chars: Vec<char> = s.chars().collect();
    let mut i = 0;
    while let Some(&c) = chars.get(i) {
        if c == ' ' || c == '\t' || c == '\u{3000}' {
            flush(&mut run, &mut out)?;
            out.push(Sq::blank());
            i += 1;
        } else if let Some(len) = [2, 1].into_iter().find(|&len| is_combined(&chars, i, len)) {
            flush(&mut run, &mut out)?;
            out.push(Sq { cell: Some(Cell::Combined(chars.get(i..i + len).unwrap_or_default().iter().collect())), ruby: None });
            i += len;
        } else if is_japanese(c) {
            run.push(c);
            i += 1;
        } else {
            flush(&mut run, &mut out)?;
            out.push(Sq { cell: Some(Cell::Combined(c.to_string())), ruby: None });
            i += 1;
        }
    }
    flush(&mut run, &mut out)?;
    Ok(out)
}

/// A ruby's clusters and how many squares of base it stands over.
type RubyOver = Option<(Vec<ShapedCluster>, usize)>;

/// Split a ruby over `span` squares into the part over the first `first` squares and the rest.
fn split_ruby(ruby: Vec<ShapedCluster>, span: usize, first: usize) -> (RubyOver, RubyOver) {
    if first == 0 || span == 0 {
        return (None, Some((ruby, span)));
    }
    if first >= span {
        return (Some((ruby, span)), None);
    }
    let take = (ruby.len() * first + span / 2) / span;
    let mut a = ruby;
    let b = a.split_off(take.min(a.len()));
    ((!a.is_empty()).then_some((a, first)), (!b.is_empty()).then_some((b, span - first)))
}

/// Lay `paras` out in columns of `rows` squares, appending to `cols`. `hung` counts hung characters.
fn flow_squares(paras: Vec<Vec<Sq>>, rows: usize, cols: &mut Vec<Col>, hung: &mut usize) {
    for mut para in paras {
        if para.is_empty() {
            cols.push(Col::default());
            continue;
        }
        let mut col = Col::default();
        // 字下げ: the first square of a paragraph is left empty.
        let mut row = usize::from(rows > 1);
        let mut i = 0;
        while i < para.len() {
            if row >= rows {
                // One or more than one is all that matters: a long run is not counted at every column end.
                let run = para.iter().skip(i).take_while(|s| s.no_start()).take(2).count();
                if run == 1 {
                    // ぶら下げ.
                    if let Some(sq) = para.get(i) {
                        col.hung = Some(sq.clone());
                        *hung += 1;
                    }
                    i += 1;
                    cols.push(std::mem::take(&mut col));
                    row = 0;
                    continue;
                }
                // 追い出し / 行末禁則: when two or more characters that can't start a column follow, or the
                // column would end in an opening bracket, its last square goes down with them.
                let mut carry: Vec<(usize, Sq)> = Vec::new();
                let mut need = run >= 2 || col.items.last().is_some_and(|l| l.1.no_end());
                while need && col.items.len() > 1 && carry.len() < 4 {
                    if let Some(item) = col.items.pop() {
                        carry.insert(0, item);
                    }
                    need = carry.first().is_some_and(|c| c.1.no_start()) || col.items.last().is_some_and(|l| l.1.no_end());
                }
                cols.push(std::mem::take(&mut col));
                for (k, (_, sq)) in carry.into_iter().enumerate() {
                    col.items.push((k, sq));
                }
                row = col.items.len();
                if row >= rows {
                    // Cannot happen (carry is at most four squares); stay safe against a loop.
                    cols.push(std::mem::take(&mut col));
                    row = 0;
                }
                continue;
            }
            // A ruby over squares that run on into the next column is split between the two.
            let split = para.get(i).and_then(|sq| sq.ruby.clone()).filter(|(_, span)| *span > 1 && row + span > rows);
            if let Some((ruby, span)) = split {
                let (a, b) = split_ruby(ruby, span, rows - row);
                if let Some(sq) = para.get_mut(i) {
                    sq.ruby = a;
                }
                if let Some(sq) = para.get_mut(i + rows - row) {
                    sq.ruby = b;
                }
            }
            if let Some(sq) = para.get(i) {
                col.items.push((row, sq.clone()));
            }
            row += 1;
            i += 1;
        }
        cols.push(col);
    }
    keep_ruby_over_its_squares(cols);
}

/// After 追い出し and ぶら下げ have moved squares on, a ruby may stand over squares that are no
/// longer below it in its column. Each ruby keeps the part over the squares that still follow it
/// there; the rest goes to the next square of the text in a grid column (never a hung one).
fn keep_ruby_over_its_squares(cols: &mut [Col]) {
    for c in 0..cols.len() {
        // A hung square stands outside the grid: its ruby goes on to the next column.
        let hung_ruby = cols.get_mut(c).and_then(|col| col.hung.as_mut()).and_then(|sq| sq.ruby.take());
        // The hung square was one of the squares the ruby stood over; it no longer counts.
        if let Some((clusters, span)) = hung_ruby {
            give_ruby(cols, c + 1, 0, (clusters, span.saturating_sub(1).max(1)));
        }
        let len = cols.get(c).map_or(0, |col| col.items.len());
        for j in 0..len {
            let Some((row, Some((ruby, span)))) = cols.get(c).and_then(|col| col.items.get(j)).map(|(r, sq)| (*r, sq.ruby.clone())) else {
                continue;
            };
            let items = cols.get(c).map_or(&[][..], |col| col.items.as_slice());
            let below = (1..span).take_while(|&m| items.get(j + m).is_some_and(|(r, _)| *r == row + m)).count() + 1;
            if below < span {
                // With no later square to take the rest, the whole ruby stays over what is below.
                let later = j + below < len || cols.get(c + 1..).is_some_and(|rest| rest.iter().any(|col| !col.items.is_empty()));
                let (here, rest) = if later { split_ruby(ruby, span, below) } else { (Some((ruby, below)), None) };
                if let Some((_, sq)) = cols.get_mut(c).and_then(|col| col.items.get_mut(j)) {
                    sq.ruby = here;
                }
                if let Some(rest) = rest {
                    if j + below < len {
                        give_ruby(cols, c, j + below, rest);
                    } else {
                        give_ruby(cols, c + 1, 0, rest);
                    }
                }
            }
        }
    }
}

/// Put `ruby` on the square at item `j` of column `c` (or the first square of a later column when
/// that column has none), joining a ruby already there.
fn give_ruby(cols: &mut [Col], c: usize, j: usize, (clusters, span): (Vec<ShapedCluster>, usize)) {
    let Some(col) = cols.get_mut(c..).and_then(|rest| rest.iter_mut().find(|col| !col.items.is_empty())) else { return };
    let at = if j < col.items.len() { j } else { 0 };
    let Some((_, sq)) = col.items.get_mut(at) else { return };
    sq.ruby = Some(match sq.ruby.take() {
        Some((mut own, own_span)) => {
            let mut joined = clusters;
            joined.append(&mut own);
            (joined, span.max(own_span))
        }
        None => (clusters, span),
    });
}

struct Grid {
    rows: usize,
    right_n: usize,
    left_n: usize,
    sq: f64,
    gap: f64,
    x0: f64,
    top: f64,
}

impl Grid {
    /// The right edge of column `c` (0 is the rightmost).
    fn right_edge(&self, c: usize) -> f64 {
        let base = self.x0 + self.left_n as f64 * self.sq + self.gap;
        if c < self.right_n { base + (self.right_n - c) as f64 * self.sq } else { self.x0 + (self.left_n - (c - self.right_n)) as f64 * self.sq }
    }
}

fn genko(doc: &mut Document, o: &VerticalOptions, text: &str, (w, h): (f64, f64), family: Family) -> Result<Built, EditError> {
    let (cols_n, rows) = (o.columns.unwrap_or(20), o.rows.unwrap_or(20));
    if cols_n == 0 || cols_n > MAX_GRID || rows == 0 || rows > MAX_GRID {
        return Err(EditError::Invalid(format!("columns and rows must be 1 to {MAX_GRID}")));
    }
    let gap_squares = if cols_n >= 2 { 1.0 } else { 0.0 };
    let edge = 28.0;
    // One more row of room below the grid for a hung character.
    let sq = ((h - 2.0 * edge) / (rows as f64 + 1.0)).min((w - 2.0 * edge) / (cols_n as f64 + gap_squares));
    if !sq.is_finite() || sq < 6.0 {
        return Err(EditError::Invalid("the page is too small for that many squares; use fewer rows or columns, or a larger page".into()));
    }
    let size = match o.font_size {
        Some(s) if s.is_finite() && s >= 3.0 && s <= sq => s,
        Some(s) => {
            return Err(EditError::Invalid(format!(
                "font_size {s} doesn't fit a square: on this page each square is {sq:.1} pt, so use 3 to {sq:.1}"
            )));
        }
        None => 0.8 * sq,
    };
    let right_n = cols_n.div_ceil(2);
    let left_n = cols_n - right_n;
    let total_w = cols_n as f64 * sq + gap_squares * sq;
    let grid = Grid { rows, right_n, left_n, sq, gap: gap_squares * sq, x0: (w - total_w) / 2.0, top: (h + (rows as f64 + 1.0) * sq) / 2.0 };
    // The text of every paragraph in squares.
    let paras = text.split('\n').map(|p| squares(p, o.serif)).collect::<Result<Vec<_>, _>>()?;
    let mut logical: Vec<Col> = Vec::new();
    let mut hung = 0usize;
    let title = o.title.as_deref().map(str::trim).filter(|t| !t.is_empty());
    let author = o.author.as_deref().map(str::trim).filter(|t| !t.is_empty());
    if let Some(t) = title {
        let sqs = squares(&t.replace('\n', " "), o.serif)?;
        let indent = 3.min(rows.saturating_sub(1));
        if sqs.len() + indent > rows {
            return Err(EditError::Invalid(format!(
                "the title has {} characters; at most {} fit a column of {rows} squares",
                sqs.len(),
                rows.saturating_sub(indent)
            )));
        }
        logical.push(Col { items: sqs.into_iter().enumerate().map(|(k, s)| (indent + k, s)).collect(), hung: None });
    }
    if let Some(a) = author {
        let sqs = squares(&a.replace('\n', " "), o.serif)?;
        // Bottom aligned, one square above the column's end.
        let end = rows.saturating_sub(1);
        if sqs.len() > end {
            return Err(EditError::Invalid(format!("the author has {} characters; at most {end} fit a column of {rows} squares", sqs.len())));
        }
        let start = end - sqs.len();
        logical.push(Col { items: sqs.into_iter().enumerate().map(|(k, s)| (start + k, s)).collect(), hung: None });
    }
    if title.is_some() || author.is_some() {
        logical.push(Col::default());
    }
    flow_squares(paras, rows, &mut logical, &mut hung);
    while logical.len() > 1 && logical.last().is_some_and(|c| c.items.is_empty() && c.hung.is_none()) {
        logical.pop();
    }
    let page_count = logical.len().div_ceil(cols_n).max(1);
    if page_count > MAX_PAGES {
        return Err(EditError::Invalid(format!("the text needs {page_count} pages; at most {MAX_PAGES} are made")));
    }
    let t = item(size, family, [0.0, 0.0, w, h]);
    let mut fonts = Dict::new();
    let latin = "PCLat";
    let latin_ref = latin_font(doc, family);
    let mut clusters: Vec<&ShapedCluster> = Vec::new();
    for sq_ in logical.iter().flat_map(|c| c.items.iter().map(|(_, s)| s).chain(c.hung.iter())) {
        if let Some(Cell::Upright(c)) = &sq_.cell {
            clusters.push(c);
        }
        if let Some((r, _)) = &sq_.ruby {
            clusters.extend(r.iter());
        }
    }
    let shaped = ShapedFonts::new(clusters.iter().copied(), pdfcraft_fonts::shaped_glyph, &Dict::new(), &mut fonts, "PCJa", &mut |ob| {
        Object::Ref(doc.add(ob))
    })
    .map_err(|(e, c)| thai_error(e, &c.text))?;
    fonts.set(latin.as_bytes().to_vec(), Object::Ref(latin_ref));
    let mut pages = Vec::new();
    for chunk in logical.chunks(cols_n) {
        let mut out: Vec<u8> = Vec::new();
        grid_lines(&mut out, &grid, o.grid_colour);
        out.extend(format!("BT {}\n", t.fill_op()).bytes());
        for (c, col) in chunk.iter().enumerate() {
            if col.items.is_empty() && col.hung.is_none() {
                continue;
            }
            let right = grid.right_edge(c);
            let centre = right - sq / 2.0;
            let mut text = String::new();
            let mut by_row: BTreeMap<usize, &Sq> = BTreeMap::new();
            for (row, s) in &col.items {
                by_row.insert(*row, s);
            }
            if let Some(h_) = &col.hung {
                by_row.insert(rows, h_);
            }
            for s in by_row.values() {
                text.push_str(s.text());
            }
            open_span(&mut out, &text);
            let mut current: Option<&str> = None;
            let mut ruby_out = Vec::new();
            // 縦書き: the squares top to bottom; a ruby stands over its base's squares.
            for (&row, s) in &by_row {
                let Some(cell) = &s.cell else { continue };
                // The glyph's em box is centred in the square.
                let pen = grid.top - row as f64 * sq - (sq - size) / 2.0;
                draw_cell(&mut out, &mut ruby_out, &shaped, cell, &t, latin, centre, pen, &mut current);
                if let Some((ruby, span)) = &s.ruby {
                    genko_ruby(&mut ruby_out, &shaped, ruby, *span, right, grid.top - row as f64 * sq, sq);
                }
            }
            out.extend_from_slice(b"EMC\n");
            if !ruby_out.is_empty() {
                out.extend_from_slice(b"/Span <</ActualText <FEFF>>> BDC\n");
                out.extend(ruby_out);
                out.extend_from_slice(b"EMC\n");
            }
        }
        out.extend_from_slice(b"ET\n");
        pages.push(out);
    }
    Ok(Built { pages, fonts, columns: logical.len(), hung })
}

/// A ruby over `span` squares starting at `top`, centred on the squares' right-hand line.
fn genko_ruby(out: &mut Vec<u8>, shaped: &ShapedFonts, ruby: &[ShapedCluster], span: usize, right: f64, top: f64, sq: f64) {
    let length = span as f64 * sq;
    let total: f64 = ruby.iter().map(|c| c.advance).sum();
    if total <= 0.0 {
        return;
    }
    // A third of a square, smaller when the ruby is longer than its base.
    let size = (0.32 * sq).min(length / total);
    let mut pen = top - (length - total * size) / 2.0;
    let [ox, oy] = pdfcraft_fonts::VERTICAL_ORIGIN;
    for c in ruby {
        if let Some((name, code)) = shaped.code(c) {
            out.extend(format!("/{name} {} Tf 1 0 0 1 {} {} Tm ", n(size), n(right - ox * size), n(pen - oy * size)).bytes());
            out.extend(literal(&[code]));
            out.extend_from_slice(b" Tj\n");
        }
        pen -= c.advance * size;
    }
}

/// The grid: two blocks of squares around the centre fold, their outer edges heavier.
fn grid_lines(out: &mut Vec<u8>, g: &Grid, colour: [f64; 3]) {
    let sq = g.sq;
    let bottom = g.top - g.rows as f64 * sq;
    let blocks = [(g.x0, g.left_n), (g.x0 + g.left_n as f64 * sq + g.gap, g.right_n)];
    out.extend(format!("q {} 1 J\n", stroke_colour(colour)).bytes());
    for (x, count) in blocks.into_iter().filter(|(_, c)| *c > 0) {
        let right = x + count as f64 * sq;
        out.extend(b"0.4 w\n");
        for k in 1..count {
            let lx = x + k as f64 * sq;
            out.extend(format!("{} {} m {} {} l\n", n(lx), n(g.top), n(lx), n(bottom)).bytes());
        }
        for r in 1..g.rows {
            let y = g.top - r as f64 * sq;
            out.extend(format!("{} {} m {} {} l\n", n(x), n(y), n(right), n(y)).bytes());
        }
        out.extend_from_slice(b"S\n0.9 w\n");
        out.extend(format!("{} {} {} {} re S\n", n(x), n(bottom), n(right - x), n(g.top - bottom)).bytes());
    }
    out.extend_from_slice(b"Q\n");
}

// ── the document ────────────────────────────────────────────────────────────────────────────

/// The page tree, catalog preferences and information of the finished document.
fn finish(doc: &mut Document, o: &VerticalOptions, built: &Built, (w, h): (f64, f64)) -> Result<(), EditError> {
    let tree = doc
        .root()
        .and_then(|r| doc.get(r).as_dict().and_then(|d| d.reference(b"Pages")))
        .ok_or_else(|| EditError::Invalid("the new document has no page tree".into()))?;
    let mut kids = Vec::new();
    for content in &built.pages {
        let mut resources = Dict::new();
        resources.set(b"Font".to_vec(), Object::Dict(built.fonts.clone()));
        let mut page = Dict::new();
        page.set(b"Type".to_vec(), Object::name("Page"));
        page.set(b"Parent".to_vec(), Object::Ref(tree));
        page.set(b"MediaBox".to_vec(), Object::Array(vec![0.into(), 0.into(), Object::Real(w), Object::Real(h)]));
        page.set(b"Resources".to_vec(), Object::Dict(resources));
        let stream = doc.add(Object::Stream(Stream::flate(Dict::new(), content)));
        page.set(b"Contents".to_vec(), Object::Ref(stream));
        kids.push(Object::Ref(doc.add(page)));
    }
    let count = i64::try_from(kids.len()).map_err(|_| EditError::Invalid("too many pages".into()))?;
    doc.update_dict(tree, |d| {
        d.set(b"Count".to_vec(), Object::Int(count));
        d.set(b"Kids".to_vec(), Object::Array(kids));
    })?;
    // Right binding: the document is read from the right (縦書き).
    let root = doc.root().ok_or_else(|| EditError::Invalid("the new document has no catalog".into()))?;
    doc.update_dict(root, |d| {
        let mut prefs = d.get(b"ViewerPreferences").and_then(|v| v.as_dict().cloned()).unwrap_or_default();
        prefs.set(b"Direction".to_vec(), Object::name("R2L"));
        d.set(b"ViewerPreferences".to_vec(), Object::Dict(prefs));
        d.set(b"Lang".to_vec(), PdfString::text("ja-JP"));
    })?;
    let mut info = Dict::new();
    let title = o.title.as_deref().map(str::trim).filter(|t| !t.is_empty());
    info.set(b"Title".to_vec(), PdfString::text(title.unwrap_or("縦書き文書")));
    if let Some(a) = o.author.as_deref().map(str::trim).filter(|a| !a.is_empty()) {
        info.set(b"Author".to_vec(), PdfString::text(a));
    }
    info.set(b"Producer".to_vec(), PdfString::text("PdfCraft"));
    let info = doc.add(info);
    doc.trailer_mut().set(b"Info".to_vec(), Object::Ref(info));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn have_face() -> bool {
        if pdfcraft_fonts::document_japanese_font_for_style(true, false).is_none() {
            eprintln!("skipping: built without a craft-fonts Jpan face");
            return false;
        }
        true
    }

    /// The columns laid out from `paras`, each as (its rows' text, the hung text).
    fn laid(paras: &[&str], rows: usize) -> Vec<(String, String)> {
        let sq: Vec<Vec<Sq>> = paras.iter().map(|p| squares(p, true).unwrap()).collect();
        let mut cols = Vec::new();
        flow_squares(sq, rows, &mut cols, &mut 0);
        cols.iter()
            .map(|c| {
                // A blank square shows as 　 so the rows can be read off.
                let mut cells = vec!['　'; rows];
                for (r, s) in &c.items {
                    if let (Some(slot), Some(ch)) = (cells.get_mut(*r), s.text().chars().next()) {
                        *slot = ch;
                    }
                }
                (cells.into_iter().collect(), c.hung.as_ref().map(|s| s.text().to_string()).unwrap_or_default())
            })
            .collect()
    }

    /// Every ruby as (column, row, its text, how many squares it stands over).
    fn rubies(paras: &[&str], rows: usize) -> Vec<(usize, usize, String, usize)> {
        let sq: Vec<Vec<Sq>> = paras.iter().map(|p| squares(p, true).unwrap()).collect();
        let mut cols = Vec::new();
        flow_squares(sq, rows, &mut cols, &mut 0);
        let mut out = Vec::new();
        for (c, col) in cols.iter().enumerate() {
            for (r, s) in &col.items {
                if let Some((clusters, span)) = &s.ruby {
                    out.push((c, *r, clusters.iter().map(|k| k.text.as_str()).collect(), *span));
                }
            }
            assert!(col.hung.as_ref().is_none_or(|h| h.ruby.is_none()), "a hung square has no ruby");
        }
        out
    }

    #[test]
    fn a_ruby_follows_its_base_when_part_of_it_goes_down_to_the_next_column() {
        if !have_face() {
            return;
        }
        // 」。 can't start a column, so 京 goes down with them: 東京's ruby is split between the
        // columns instead of standing over the empty square below 東.
        let r = rubies(&["あい東京《とうきょう》」。"], 5);
        assert_eq!(r.len(), 2, "{r:?}");
        assert_eq!((r[0].0, r[0].1, r[0].3), (0, 3, 1), "{r:?}");
        assert_eq!((r[1].0, r[1].1, r[1].3), (1, 0, 1), "{r:?}");
        assert_eq!(format!("{}{}", r[0].2, r[1].2), "とうきょう");
        // A stop inside a base that hangs keeps no ruby; the rest goes on to the next column.
        let r = rubies(&["｜あいうえ、おか《るびるびるび》"], 5);
        assert!(r.iter().all(|x| x.1 + x.3 <= 5), "{r:?}");
        assert_eq!(r.iter().map(|x| x.2.as_str()).collect::<String>(), "るびるびるび", "{r:?}");
    }

    #[test]
    fn a_paragraph_is_indented_one_square_and_starts_a_new_column() {
        if !have_face() {
            return;
        }
        let cols = laid(&["あい", "うえ"], 5);
        assert_eq!(cols, [("　あい　　".to_string(), String::new()), ("　うえ　　".to_string(), String::new())]);
    }

    #[test]
    fn a_stop_that_would_start_a_column_hangs_below_the_previous_one() {
        if !have_face() {
            return;
        }
        // Rows 1-4 hold あいうえ; 、 would start the next column, so it hangs.
        let cols = laid(&["あいうえ、おか"], 5);
        assert_eq!(cols[0], ("　あいうえ".to_string(), "、".to_string()));
        assert_eq!(cols[1].0, "おか　　　");
        // Two in a row: 」。 can't both hang, so え goes down with them.
        let cols = laid(&["あいうえ」。お"], 5);
        assert_eq!(cols[0], ("　あいう　".to_string(), String::new()));
        assert_eq!(cols[1].0, "え」。お　");
        // An opening bracket never ends a column.
        let cols = laid(&["あいう「えお"], 5);
        assert_eq!(cols[0].0, "　あいう　");
        assert_eq!(cols[1].0, "「えお　　");
        // Small kana keep their own square and may not start a column either.
        let cols = laid(&["あいうえっお"], 5);
        assert_eq!(cols[0], ("　あいうえ".to_string(), "っ".to_string()));
    }

    #[test]
    fn digits_share_a_square_and_other_latin_takes_one_each() {
        if !have_face() {
            return;
        }
        let texts = |s: &str| -> Vec<String> { squares(s, true).unwrap().iter().map(|q| q.text().to_string()).collect() };
        assert_eq!(texts("令和7年12月"), ["令", "和", "7", "年", "12", "月"]);
        assert_eq!(texts("第100回"), ["第", "1", "0", "0", "回"]);
        assert_eq!(texts("PDF"), ["P", "D", "F"]);
    }

    #[test]
    fn ruby_stands_over_its_base_squares() {
        if !have_face() {
            return;
        }
        let sq = squares("東京《とうきょう》へ", true).unwrap();
        assert_eq!(sq.iter().map(Sq::text).collect::<String>(), "東京へ");
        assert!(matches!(&sq[0].ruby, Some((r, 2)) if r.len() == 5));
        assert!(sq[1].ruby.is_none());
    }

    fn page_count(doc: &Document) -> usize {
        let tree = doc.root().and_then(|r| doc.get(r).as_dict().and_then(|d| d.reference(b"Pages"))).unwrap();
        doc.get(tree).as_dict().and_then(|d| d.get(b"Count")).and_then(Object::as_int).unwrap() as usize
    }

    #[test]
    fn documents_are_right_bound_and_flow_over_pages() {
        if !have_face() {
            return;
        }
        let long = "あ".repeat(900);
        for template in [Template::Genko, Template::Binsen, Template::Plain] {
            let made = create_vertical_document(&VerticalOptions::new(long.clone(), template)).unwrap();
            if template == Template::Genko {
                // 900 characters and an indent in 20 × 20 = 400 squares a page.
                assert_eq!(made.pages, 3);
            }
            assert_eq!(made.pages, page_count(&made.document));
            let root = made.document.root().and_then(|r| made.document.get(r).as_dict().cloned()).unwrap();
            let prefs = root.get(b"ViewerPreferences").and_then(Object::as_dict).unwrap();
            assert_eq!(prefs.get(b"Direction").and_then(Object::as_name), Some(&b"R2L"[..]));
        }
    }

    #[test]
    fn options_are_checked_with_clear_errors() {
        if !have_face() {
            return;
        }
        let mut o = VerticalOptions::new("あ", Template::Genko);
        o.font_size = Some(500.0);
        let e = create_vertical_document(&o).unwrap_err().to_string();
        assert!(e.contains("font_size"), "{e}");
        let mut o = VerticalOptions::new("あ", Template::Genko);
        o.title = Some("あ".repeat(30));
        assert!(create_vertical_document(&o).unwrap_err().to_string().contains("title"));
        let o = VerticalOptions::new("あ\u{0e01}", Template::Plain);
        assert!(create_vertical_document(&o).unwrap_err().to_string().contains("line 1"));
        // The title is checked like the text.
        let mut o = VerticalOptions::new("あ", Template::Genko);
        o.title = Some("\u{0e01}".into());
        assert!(create_vertical_document(&o).unwrap_err().to_string().contains("the title"));
        let mut o = VerticalOptions::new("あ", Template::Binsen);
        o.columns = Some(500);
        assert!(create_vertical_document(&o).is_err());
    }
}
