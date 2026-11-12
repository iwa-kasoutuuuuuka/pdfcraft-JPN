//! pdfcraft-JPN: vertical writing (縦書き) for added text.
//!
//! The text runs in columns from the top of the box down; the first column sits at the box's
//! right edge and the next ones to its left, as in a Japanese book. Japanese is shaped with the
//! Japanese face for vertical writing (its `vert` forms turn 、。「」ー and the like for columns)
//! and drawn upright from Type 3 glyphs, like Thai and Arabic in horizontal text. Other text
//! (Latin words, numbers) is drawn sideways in the item's standard font, turned 90° clockwise.
//! Each column is one `/ActualText` span, so copy and search read it top to bottom.
//!
//! 縦中横: a run of one or two digits (or `!!`, `!?`, `?!`) between non-ASCII characters is drawn
//! upright, side by side, in one em cell. ルビ: `｜親文字《るび》` is drawn with the ruby at half
//! size to the right of the base, outside the column's text.

use pdfcraft_fonts::ShapedCluster;
use pdfcraft_fonts::shaped_text::{self, ShapedFonts, is_japanese};

use super::layout::ruby::{self, Segment};
use super::layout::{latin_width, norm, thai_error};
use super::types::AddedText;
use super::{Dict, Document, EditError, Object, literal, n, win_ansi};

/// Distance between two columns' centres, in em (the same as the horizontal line pitch).
pub(super) const PITCH: f64 = 1.2;

/// One thing placed in a column.
#[derive(Clone, Debug)]
pub(super) enum Cell {
    /// A Japanese cluster drawn upright; its advance is down the column (em).
    Upright(ShapedCluster),
    /// Text in the standard font, turned sideways.
    Sideways(String),
    /// 縦中横: one or two characters of the standard font, upright and side by side in one em.
    Combined(String),
    /// A base drawn upright with its ruby beside it.
    Ruby { text: String, base: Vec<ShapedCluster>, ruby: Vec<ShapedCluster> },
}

/// Whether `run` (the characters of a paragraph from `at`, `len` of them) is a 縦中横 group:
/// one or two digits, or `!!`, `!?`, `?!`, standing between non-ASCII characters (or the paragraph's ends).
pub(super) fn is_combined(chars: &[char], at: usize, len: usize) -> bool {
    let Some(run) = chars.get(at..at + len) else { return false };
    let alone = at == 0 || chars.get(at - 1).is_none_or(|c| !c.is_ascii());
    let alone = alone && chars.get(at + len).is_none_or(|c| !c.is_ascii());
    let kind = run.iter().all(char::is_ascii_digit) || matches!(run, ['!', '!'] | ['!', '?'] | ['?', '!']);
    alone && kind
}

impl Cell {
    pub(super) fn text(&self) -> &str {
        match self {
            Cell::Upright(c) => &c.text,
            Cell::Sideways(s) | Cell::Combined(s) => s,
            Cell::Ruby { text, .. } => text,
        }
    }

    fn has_ruby(&self) -> bool {
        matches!(self, Cell::Ruby { .. })
    }

    /// The length the cell takes down the column, in points.
    pub(super) fn length(&self, t: &AddedText) -> f64 {
        match self {
            Cell::Upright(c) => c.advance * t.size + t.char_spacing,
            Cell::Sideways(s) => latin_width(t, s) + t.char_spacing * s.chars().count() as f64,
            Cell::Combined(_) => t.size + t.char_spacing,
            Cell::Ruby { base, .. } => base.iter().map(|c| c.advance * t.size + t.char_spacing).sum(),
        }
    }
}

/// The length of each column: the box's height, or its width when the box is a single row high
/// (a box drawn for horizontal text), so a click-made box still holds a useful column.
pub(super) fn column_length(t: &AddedText) -> f64 {
    let r = norm(t.rect);
    let (w, h) = (r[2] - r[0], r[3] - r[1]);
    if h >= 2.0 * t.size { h } else { w.max(h).max(t.size) }
}

/// `para` split into cells: Japanese runs shaped for vertical writing, one cell per cluster;
/// everything else one sideways cell per word (its spaces kept on it).
#[cfg(test)]
fn cells(para: &str) -> Result<Vec<Cell>, EditError> {
    cells_styled(para, false)
}

/// [`cells`] in the serif (Mincho) face when `serif`.
pub(super) fn cells_styled(para: &str, serif: bool) -> Result<Vec<Cell>, EditError> {
    let mut out = Vec::new();
    for seg in ruby::ruby_spans(para) {
        match seg {
            Segment::Plain(s) => out.extend(plain_cells(&s, serif)?),
            Segment::Ruby { base, ruby, .. } => {
                let shape = |s: &str| pdfcraft_fonts::shape_japanese_vertical_styled(s, serif).map_err(|e| thai_error(e, s));
                out.push(Cell::Ruby { base: shape(&base)?, ruby: shape(&ruby)?, text: base });
            }
        }
    }
    Ok(out)
}

/// A stretch of a paragraph that becomes cells of one kind.
#[derive(Clone, Debug, PartialEq)]
pub(super) enum Run {
    /// Japanese, shaped for vertical writing: one upright cell per cluster.
    Japanese(String),
    /// A word of other text (its spaces kept on it): one sideways cell.
    Latin(String),
    /// 縦中横: one combined cell.
    Combined(String),
}

/// `para` (without ruby markup) split into [`Run`]s, in order; together they hold every character.
pub(super) fn runs(para: &str) -> Vec<Run> {
    let mut out = Vec::new();
    let mut run = String::new();
    let mut latin = String::new();
    let flush_run = |run: &mut String, out: &mut Vec<Run>| {
        if !run.is_empty() {
            out.push(Run::Japanese(std::mem::take(run)));
        }
    };
    let flush_latin = |latin: &mut String, out: &mut Vec<Run>| {
        if !latin.is_empty() {
            out.push(Run::Latin(std::mem::take(latin)));
        }
    };
    let chars: Vec<char> = para.chars().collect();
    let mut i = 0;
    while let Some(&c) = chars.get(i) {
        // 縦中横: two characters first, then one.
        if let Some(len) = [2, 1].into_iter().find(|&len| is_combined(&chars, i, len)) {
            flush_run(&mut run, &mut out);
            flush_latin(&mut latin, &mut out);
            out.push(Run::Combined(chars.get(i..i + len).unwrap_or_default().iter().collect()));
            i += len;
            continue;
        }
        i += 1;
        if is_japanese(c) {
            flush_latin(&mut latin, &mut out);
            run.push(c);
        } else {
            flush_run(&mut run, &mut out);
            latin.push(c);
            // A word ends after its spaces.
            if c == ' ' {
                flush_latin(&mut latin, &mut out);
            }
        }
    }
    flush_run(&mut run, &mut out);
    flush_latin(&mut latin, &mut out);
    out
}

/// [`cells`] for text without ruby markup.
fn plain_cells(para: &str, serif: bool) -> Result<Vec<Cell>, EditError> {
    let mut out = Vec::new();
    for run in runs(para) {
        match run {
            Run::Japanese(s) => {
                let shaped = pdfcraft_fonts::shape_japanese_vertical_styled(&s, serif).map_err(|e| thai_error(e, &s))?;
                out.extend(shaped.into_iter().map(Cell::Upright));
            }
            Run::Latin(s) => out.push(Cell::Sideways(s)),
            Run::Combined(s) => out.push(Cell::Combined(s)),
        }
    }
    Ok(out)
}

/// The text's columns, first (rightmost) first. A cell that would overflow a column starts the
/// next one, except one that may not start a line (、。」 and the like, 行頭禁則), which hangs
/// past the column's end instead (ぶら下げ) — one cell at most, so a long run of them (ーーー…) still
/// wraps instead of running off the page. A Latin word longer than a column is broken over columns.
/// Each paragraph starts a new column.
pub(super) fn columns(t: &AddedText) -> Result<Vec<Vec<Cell>>, EditError> {
    columns_styled(t, false)
}

/// [`columns`] in the serif (Mincho) face when `serif`.
pub(super) fn columns_styled(t: &AddedText, serif: bool) -> Result<Vec<Vec<Cell>>, EditError> {
    let length = column_length(t);
    let mut out = Vec::new();
    for para in t.text.split('\n') {
        let mut column: Vec<Cell> = Vec::new();
        let mut used = 0.0;
        for (cell, l) in cells_styled(para, serif)?.into_iter().flat_map(|cell| fit_column(t, cell, length)) {
            let hangs = used <= length && cell.text().chars().next().is_some_and(pdfcraft_fonts::no_line_start);
            if !column.is_empty() && used + l > length && !hangs {
                out.push(std::mem::take(&mut column));
                used = 0.0;
                // A column after a wrap doesn't begin with the spaces where the last one ended.
                if cell.text().trim().is_empty() {
                    continue;
                }
            }
            used += l;
            column.push(cell);
        }
        out.push(column);
    }
    Ok(out)
}

/// `cell` with its length, or a Latin word longer than a column in pieces that each fit one. The
/// pieces are cut where the word's prefix widths say, as the typing layout cuts them.
fn fit_column(t: &AddedText, cell: Cell, length: f64) -> Vec<(Cell, f64)> {
    let l = cell.length(t);
    let Cell::Sideways(word) = &cell else { return vec![(cell, l)] };
    if l <= length || word.chars().count() < 2 {
        return vec![(cell, l)];
    }
    let ends = prefix_ends(t, word);
    let chars: Vec<char> = word.chars().collect();
    let mut out = Vec::new();
    let (mut from, mut start) = (0usize, 0.0);
    while from < chars.len() {
        // At least one character a piece, however narrow the column.
        let mut to = from + 1;
        while to < chars.len() && ends.get(to).is_some_and(|e| e - start <= length) {
            to += 1;
        }
        let end = ends.get(to - 1).copied().unwrap_or(start);
        out.push((Cell::Sideways(chars.get(from..to).unwrap_or_default().iter().collect()), end - start));
        (from, start) = (to, end);
    }
    out
}

/// Where each character of a sideways word ends down the column: the width of the word up to and
/// including it (never going back).
fn prefix_ends(t: &AddedText, word: &str) -> Vec<f64> {
    let mut prefix = String::new();
    let mut last = 0.0f64;
    word.chars()
        .map(|c| {
            prefix.push(c);
            let end = Cell::Sideways(prefix.clone()).length(t);
            last = if end.is_finite() { end.max(last) } else { last };
            last
        })
        .collect()
}

/// How wide a column is: one pitch, and room for ruby on its right when it has any.
fn column_width(t: &AddedText, column: &[Cell]) -> f64 {
    (PITCH + if column.iter().any(Cell::has_ruby) { ruby::RUBY_LEAD } else { 0.0 }) * t.size
}

/// The box vertical text takes, in display space: its top and right edges are the item's, its
/// height the column length and its width one pitch per column.
pub(super) fn vertical_rect(t: &AddedText) -> [f64; 4] {
    let r = norm(t.rect);
    let width = columns(t).map_or(PITCH * t.size, |c| c.iter().map(|col| column_width(t, col)).sum::<f64>().max(PITCH * t.size));
    [r[2] - width, r[3] - column_length(t), r[2], r[3]]
}

/// Draw `t` vertically into `out` (inside the item's `q … Q`), its Type 3 fonts into `fonts`.
pub(super) fn draw_vertical(
    doc: &mut Document,
    t: &AddedText,
    taken: &Dict,
    fonts: &mut Dict,
    out: &mut Vec<u8>,
    latin: &str,
) -> Result<(), EditError> {
    let cols = columns(t)?;
    let r = norm(t.rect);
    let clusters: Vec<&ShapedCluster> = cols
        .iter()
        .flatten()
        .flat_map(|c| match c {
            Cell::Upright(c) => std::slice::from_ref(c).iter().chain([].iter()),
            Cell::Ruby { base, ruby, .. } => base.iter().chain(ruby.iter()),
            Cell::Sideways(_) | Cell::Combined(_) => [].iter().chain([].iter()),
        })
        .collect();
    let shaped = ShapedFonts::new(clusters.iter().copied(), pdfcraft_fonts::shaped_glyph, taken, fonts, "PCJa", &mut |o| Object::Ref(doc.add(o)))
        .map_err(|(e, c)| thai_error(e, &c.text))?;
    out.extend(format!("BT {}\n", t.fill_op()).bytes());
    let size = t.size;
    let mut right = r[2];
    for column in &cols {
        let width = column_width(t, column);
        let text: String = column.iter().map(Cell::text).collect();
        let logical: String = text.encode_utf16().map(|u| format!("{u:04X}")).collect();
        out.extend(format!("/Span <</ActualText <FEFF{logical}>>> BDC\n").bytes());
        // The ruby side of a column is its right.
        let pad = if column.iter().any(Cell::has_ruby) { ruby::RUBY_LEAD * size } else { 0.0 };
        let centre = right - pad - 0.5 * PITCH * size;
        right -= width;
        let mut pen = r[3];
        let mut current: Option<&str> = None;
        let mut ruby_out: Vec<u8> = Vec::new();
        for cell in column {
            draw_cell(out, &mut ruby_out, &shaped, cell, t, latin, centre, pen, &mut current);
            pen -= cell.length(t);
        }
        out.extend_from_slice(b"EMC\n");
        if !ruby_out.is_empty() {
            out.extend_from_slice(b"/Span <</ActualText <FEFF>>> BDC\n");
            out.extend(ruby_out);
            out.extend_from_slice(b"EMC\n");
        }
    }
    out.extend_from_slice(b"ET\n");
    Ok(())
}

/// Draw one cell of a column whose centre line is `centre`, from `pen` (the cell's top) down, into
/// `out`; a ruby goes into `ruby_out`, to be drawn after the column. `current` is the font the text
/// state has selected.
#[allow(clippy::too_many_arguments)]
pub(super) fn draw_cell<'a>(
    out: &mut Vec<u8>,
    ruby_out: &mut Vec<u8>,
    shaped: &'a ShapedFonts,
    cell: &Cell,
    t: &AddedText,
    latin: &'a str,
    centre: f64,
    pen: f64,
    current: &mut Option<&'a str>,
) {
    let size = t.size;
    // Standard-font text turned sideways sits with its x-height on the column's centre line.
    let sideways_shift = 0.26 * size;
    match cell {
        Cell::Upright(c) => {
            // The glyphs carry the face's vertical offsets from VERTICAL_ORIGIN: drawn
            // from there, each lands centred under the previous one.
            if let Some((name, code)) = shaped.code(c) {
                if *current != Some(name) {
                    out.extend(format!("/{name} {} Tf ", n(size)).bytes());
                }
                *current = Some(name);
                let [ox, oy] = pdfcraft_fonts::VERTICAL_ORIGIN;
                out.extend(format!("1 0 0 1 {} {} Tm ", n(centre - ox * size), n(pen - oy * size)).bytes());
                out.extend(literal(&[code]));
                out.extend_from_slice(b" Tj\n");
            }
        }
        Cell::Combined(s) => {
            if *current != Some(latin) {
                out.extend(format!("/{latin} {} Tf ", n(size)).bytes());
            }
            *current = Some(latin);
            let natural = t.family.width(s, size, t.bold, t.italic);
            // Wider than an em: squeezed to fit. Centred on the column, on the em's middle.
            let squeeze = if natural > size { size / natural } else { 1.0 };
            if squeeze < 1.0 {
                out.extend(format!("{} Tz ", n(100.0 * squeeze)).bytes());
            }
            out.extend(format!("1 0 0 1 {} {} Tm ", n(centre - natural * squeeze / 2.0), n(pen - 0.86 * size)).bytes());
            out.extend(literal(&win_ansi(s)));
            out.extend_from_slice(b" Tj\n");
            if squeeze < 1.0 {
                out.extend_from_slice(b"100 Tz ");
            }
        }
        Cell::Ruby { base, ruby, .. } => {
            let [ox, oy] = pdfcraft_fonts::VERTICAL_ORIGIN;
            let mut at = pen;
            for c in base {
                if let Some((name, code)) = shaped.code(c) {
                    out.extend(format!("/{name} {} Tf 1 0 0 1 {} {} Tm ", n(size), n(centre - ox * size), n(at - oy * size)).bytes());
                    out.extend(literal(&[code]));
                    out.extend_from_slice(b" Tj\n");
                }
                at -= c.advance * size + t.char_spacing;
            }
            // The ruby at half size, centred along the base, on its right. It is drawn after the
            // column, in a span of its own that stands for no text.
            let half = size / 2.0;
            let ruby_length: f64 = ruby.iter().map(|c| c.advance * half).sum();
            let mut rp = pen - (cell.length(t) - ruby_length) / 2.0;
            let rc = centre + 0.5 * size + 0.33 * size;
            for c in ruby {
                if let Some((name, code)) = shaped.code(c) {
                    ruby_out.extend(format!("/{name} {} Tf 1 0 0 1 {} {} Tm ", n(half), n(rc - ox * half), n(rp - oy * half)).bytes());
                    ruby_out.extend(literal(&[code]));
                    ruby_out.extend_from_slice(b" Tj\n");
                }
                rp -= c.advance * half;
            }
            *current = None;
        }
        Cell::Sideways(s) => {
            if *current != Some(latin) {
                out.extend(format!("/{latin} {} Tf ", n(size)).bytes());
            }
            *current = Some(latin);
            if t.char_spacing != 0.0 {
                out.extend(format!("{} Tc ", n(t.char_spacing)).bytes());
            }
            // Turned 90° clockwise: the baseline runs down the column.
            out.extend(format!("0 -1 1 0 {} {} Tm ", n(centre - sideways_shift), n(pen)).bytes());
            out.extend(literal(&win_ansi(s)));
            out.extend_from_slice(b" Tj\n");
            if t.char_spacing != 0.0 {
                out.extend_from_slice(b"0 Tc ");
            }
        }
    }
}

/// The first character vertical text can't draw: Thai, Arabic and Hebrew (vertical writing is for
/// Japanese and Western text), Japanese the face lacks, or Western text the standard fonts can't
/// encode.
pub(super) fn first_undrawable_vertical(t: &AddedText) -> Option<char> {
    ruby::without_markup(&t.text).chars().find(|&c| {
        if c == '\n' {
            false
        } else if shaped_text::is_thai(c) || super::layout::is_arabic(c) || super::layout::is_hebrew(c) {
            true
        } else if is_japanese(c) {
            !pdfcraft_fonts::japanese_has(c)
        } else {
            !pdfcraft_fonts::win_ansi_encodable(c)
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::added::{Align, Family};

    fn item(text: &str, rect: [f64; 4]) -> AddedText {
        AddedText {
            vertical: true,
            rect,
            text: text.into(),
            family: Family::Helvetica,
            bold: false,
            italic: false,
            size: 10.0,
            color: [0.0; 3],
            ink: None,
            align: Align::Left,
            char_spacing: 0.0,
            aligns: Vec::new(),
        }
    }

    #[test]
    fn columns_run_right_to_left_and_wrap_at_the_box_height() {
        if pdfcraft_fonts::document_japanese_font_for_style(false, false).is_none() {
            eprintln!("skipping: built without a craft-fonts Jpan face");
            return;
        }
        // 50 pt columns hold five 10 pt characters.
        let t = item("あいうえおかきくけこさ", [100.0, 100.0, 200.0, 150.0]);
        let cols = columns(&t).unwrap();
        let text: Vec<String> = cols.iter().map(|c| c.iter().map(Cell::text).collect()).collect();
        assert_eq!(text, ["あいうえお", "かきくけこ", "さ"]);
        let r = vertical_rect(&t);
        assert_eq!(r[2], 200.0, "the first column sits at the right edge");
        assert!((r[0] - (200.0 - 3.0 * 12.0)).abs() < 1e-9, "{r:?}");
        // 。 hangs at the end of a full column rather than starting the next one.
        let t = item("あいうえお。か", [100.0, 100.0, 200.0, 150.0]);
        let text: Vec<String> = columns(&t).unwrap().iter().map(|c| c.iter().map(Cell::text).collect()).collect();
        assert_eq!(text, ["あいうえお。", "か"]);
        // Latin words are sideways cells of their own.
        let t = item("PDF を開く", [100.0, 100.0, 200.0, 300.0]);
        let cols = columns(&t).unwrap();
        assert!(matches!(&cols[0][0], Cell::Sideways(s) if s == "PDF "));
        assert_eq!(first_undrawable_vertical(&item("日本 ok", [0.0; 4])), None);
        assert_eq!(first_undrawable_vertical(&item("ภาษา", [0.0; 4])), Some('ภ'));
    }

    #[test]
    fn short_digit_runs_between_japanese_are_combined_upright() {
        if pdfcraft_fonts::document_japanese_font_for_style(false, false).is_none() {
            return;
        }
        let kinds = |text: &str| -> Vec<String> {
            cells(text)
                .unwrap()
                .iter()
                .filter(|c| !matches!(c, Cell::Upright(_)))
                .map(|c| match c {
                    Cell::Combined(s) => format!("C:{s}"),
                    Cell::Sideways(s) => format!("S:{s}"),
                    _ => "?".into(),
                })
                .collect()
        };
        assert_eq!(kinds("令和7年12月"), ["C:7", "C:12"]);
        assert_eq!(kinds("第100回"), ["S:100"], "three digits stay sideways");
        assert_eq!(kinds("すごい!!本当?!"), ["C:!!", "C:?!"]);
        assert_eq!(kinds("A 7 年"), ["S:A ", "S:7 "], "next to ASCII: sideways");
        assert_eq!(kinds("7月"), ["C:7"]);
    }

    #[test]
    fn ruby_groups_are_one_cell_with_room_on_the_right() {
        if pdfcraft_fonts::document_japanese_font_for_style(false, false).is_none() {
            return;
        }
        let t = item("東京《とうきょう》へ行く", [100.0, 100.0, 200.0, 300.0]);
        let cols = columns(&t).unwrap();
        assert_eq!(cols.len(), 1);
        assert!(matches!(&cols[0][0], Cell::Ruby { text, .. } if text == "東京"));
        let text: String = cols[0].iter().map(Cell::text).collect();
        assert_eq!(text, "東京へ行く", "the ruby isn't part of the column's text");
        let plain = item("東京へ行く", [100.0, 100.0, 200.0, 300.0]);
        assert!((vertical_rect(&t)[0] - (vertical_rect(&plain)[0] - ruby::RUBY_LEAD * 10.0)).abs() < 1e-9, "a column with ruby is wider");
        assert_eq!(first_undrawable_vertical(&t), None);
    }

    #[test]
    fn vertical_forms_replace_the_horizontal_punctuation() {
        if pdfcraft_fonts::document_japanese_font_for_style(false, false).is_none() {
            return;
        }
        let ids = |s: &str| pdfcraft_fonts::shape_japanese(s).unwrap()[0].glyphs[0].0;
        let vids = |s: &str| pdfcraft_fonts::shape_japanese_vertical(s).unwrap()[0].glyphs[0].0;
        assert_ne!(ids("。"), vids("。"), "the vertical full stop sits top right");
        assert_ne!(ids("「"), vids("「"));
        let v = pdfcraft_fonts::shape_japanese_vertical("あい").unwrap();
        // BIZ UDPGothic is proportional (あ 0.98, い 0.94 em), so the advance is near, not at, an em.
        assert!(v.iter().all(|c| c.advance > 0.5 && c.advance <= 1.1), "each advances about an em down: {v:?}");
    }
}
