//! pdfcraft-JPN: vertical text as it is typed (縦書きのまま入力).
//!
//! The in-place editor shows a vertical item in columns while it is typed. [`VerticalEditLayout`]
//! is that layout, worked out the way [`super::vertical::columns`] places the committed text (the
//! same cells, column length and wrapping, 禁則 hanging included), with every character of the
//! typed text at a known place, so the editor can draw it, put the caret between two characters
//! of a column, hit-test a click and move the caret by column.
//!
//! Ruby markup (`｜親文字《るび》`) is laid out as the characters typed while editing, so the
//! caret can enter the reading; it turns into ruby when the text is committed. Without the
//! Japanese face (a build without craft-fonts) Japanese characters take one em each.
//!
//! Positions are in points in the box's own frame: `x` runs left from the box's right edge (the
//! first column's right side) and `y` down from its top.

use std::ops::Range;

use super::types::AddedText;
use super::vertical::{self, Cell, PITCH, Run};

/// How a [`VerticalGlyph`] is drawn.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VerticalGlyphKind {
    /// Japanese, upright in the column.
    Upright,
    /// Other text, turned 90° clockwise so it reads down the column.
    Sideways,
    /// 縦中横: one or two characters upright and side by side in one em.
    Combined,
}

/// One thing drawn in a column.
#[derive(Clone, Debug, PartialEq)]
pub struct VerticalGlyph {
    /// The characters of the text it draws (character indices, `\n` counted).
    pub chars: Range<usize>,
    pub text: String,
    pub kind: VerticalGlyphKind,
    /// The column it sits in, the first (rightmost) being 0.
    pub column: usize,
    /// Its top, down from the top of the column, in points.
    pub top: f64,
    /// The length it takes down the column, in points.
    pub length: f64,
}

/// Where one character sits: its column, and its top and length down that column.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Place {
    column: usize,
    top: f64,
    length: f64,
}

/// A caret move in vertical text.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CaretMove {
    /// Up the column: the previous character.
    Up,
    /// Down the column: the next character.
    Down,
    /// The next column, to the left (the Left arrow).
    NextColumn,
    /// The previous column, to the right (the Right arrow).
    PreviousColumn,
    /// The top of the caret's column (Home).
    ColumnStart,
    /// The end of the caret's column (End).
    ColumnEnd,
    /// The start of the text.
    TextStart,
    /// The end of the text.
    TextEnd,
}

/// The layout of vertical text being typed. See the module documentation.
#[derive(Clone, Debug, PartialEq)]
pub struct VerticalEditLayout {
    /// What each column draws, column by column, top to bottom.
    pub glyphs: Vec<VerticalGlyph>,
    /// One per character of the text.
    places: Vec<Place>,
    /// Where the caret after the last character goes: (column, y).
    end: (usize, f64),
    /// The length each column's text takes, in points.
    used: Vec<f64>,
    /// The width of one column, in points.
    pub pitch: f64,
    /// The length of a column, in points: where the text wraps.
    pub column_length: f64,
}

/// A finite, positive number, or `fallback`.
fn sane(v: f64, fallback: f64) -> f64 {
    if v.is_finite() && v > 0.0 { v } else { fallback }
}

/// A cell of the typed text: how many characters it holds, its text, its kind, its length down
/// the column and where each of its characters starts within it.
struct EditCell {
    text: String,
    kind: VerticalGlyphKind,
    length: f64,
    /// Each character's (start, length) within the cell.
    offsets: Vec<(f64, f64)>,
}

/// `cell`, or a sideways word longer than a column in pieces that each fit one, cut where
/// `columns` cuts it (the word's prefix widths).
fn fit_column(cell: EditCell, length: f64) -> Vec<EditCell> {
    if cell.kind != VerticalGlyphKind::Sideways || cell.length <= length || cell.offsets.len() < 2 {
        return vec![cell];
    }
    let chars: Vec<char> = cell.text.chars().collect();
    let end_of = |k: usize| cell.offsets.get(k).map_or(0.0, |o| o.0 + o.1);
    let mut out = Vec::new();
    let (mut from, mut start) = (0usize, 0.0);
    while from < chars.len() {
        let mut to = from + 1;
        while to < chars.len() && end_of(to) - start <= length {
            to += 1;
        }
        let end = end_of(to - 1);
        let offsets = cell.offsets.get(from..to).unwrap_or_default().iter().map(|&(s, l)| (s - start, l)).collect();
        out.push(EditCell { text: chars.get(from..to).unwrap_or_default().iter().collect(), kind: cell.kind, length: end - start, offsets });
        (from, start) = (to, end);
    }
    out
}

/// `para` as [`EditCell`]s, with the lengths [`super::vertical::columns`] gives the same cells.
fn edit_cells(t: &AddedText, para: &str, size: f64) -> Vec<EditCell> {
    let spacing = if t.char_spacing.is_finite() { t.char_spacing } else { 0.0 };
    // One em per character: what a cluster takes when the face can't shape it.
    let ems = |s: &str| -> Vec<EditCell> {
        s.chars()
            .map(|c| {
                let length = (size + spacing).max(0.0);
                EditCell { text: c.to_string(), kind: VerticalGlyphKind::Upright, length, offsets: vec![(0.0, length)] }
            })
            .collect()
    };
    let mut out = Vec::new();
    for run in vertical::runs(para) {
        match run {
            Run::Japanese(s) => {
                let want = s.chars().count();
                match pdfcraft_fonts::shape_japanese_vertical(&s) {
                    // Each cluster's characters, in order, make up the run: use the face's advances.
                    Ok(clusters) if clusters.iter().map(|c| c.text.chars().count()).sum::<usize>() == want => {
                        for c in clusters {
                            let length = sane(c.advance * size + spacing, 0.0).max(0.0);
                            let n = c.text.chars().count().max(1);
                            let each = length / n as f64;
                            let offsets = (0..n).map(|k| (each * k as f64, each)).collect();
                            out.push(EditCell { text: c.text, kind: VerticalGlyphKind::Upright, length, offsets });
                        }
                    }
                    _ => out.extend(ems(&s)),
                }
            }
            Run::Latin(s) => {
                let length = sane(Cell::Sideways(s.clone()).length(t), 0.0);
                let mut offsets = Vec::new();
                let mut prefix = String::new();
                let mut start = 0.0;
                for c in s.chars() {
                    prefix.push(c);
                    let end = sane(Cell::Sideways(prefix.clone()).length(t), start).max(start);
                    offsets.push((start, end - start));
                    start = end;
                }
                out.push(EditCell { text: s, kind: VerticalGlyphKind::Sideways, length, offsets });
            }
            Run::Combined(s) => {
                let length = (size + spacing).max(0.0);
                let n = s.chars().count().max(1);
                let each = length / n as f64;
                let offsets = (0..n).map(|k| (each * k as f64, each)).collect();
                out.push(EditCell { text: s, kind: VerticalGlyphKind::Combined, length, offsets });
            }
        }
    }
    out
}

impl VerticalEditLayout {
    /// `t.text` laid out in `t`'s columns. Never fails: text the face can't shape takes one em a
    /// character.
    pub fn new(t: &AddedText) -> Self {
        let size = sane(t.size, 1.0);
        let column_length = sane(vertical::column_length(t), size);
        let mut glyphs = Vec::new();
        let mut places = Vec::new();
        let mut used = Vec::new();
        let mut column = 0usize;
        let mut index = 0usize;
        let mut end = (0, 0.0);
        let paragraphs: Vec<&str> = t.text.split('\n').collect();
        let count = paragraphs.len();
        for (p, para) in paragraphs.into_iter().enumerate() {
            let mut at = 0.0;
            let mut filled = false;
            for cell in edit_cells(t, para, size).into_iter().flat_map(|cell| fit_column(cell, column_length)) {
                let n = cell.offsets.len();
                // One hung cell at most, as `columns` hangs them.
                let hangs = at <= column_length && cell.text.chars().next().is_some_and(pdfcraft_fonts::no_line_start);
                // As `columns` wraps: a cell that overflows a column that has something starts the next.
                if filled && at + cell.length > column_length && !hangs {
                    used.push(at);
                    column = column.saturating_add(1);
                    at = 0.0;
                    filled = false;
                    // The spaces where a column wrapped are not drawn; the caret can still stand there.
                    if cell.text.trim().is_empty() {
                        places.extend((0..n).map(|_| Place { column, top: 0.0, length: 0.0 }));
                        index = index.saturating_add(n);
                        continue;
                    }
                }
                places.extend(cell.offsets.iter().map(|&(start, length)| Place { column, top: at + start, length }));
                glyphs.push(VerticalGlyph {
                    chars: index..index.saturating_add(n),
                    text: cell.text,
                    kind: cell.kind,
                    column,
                    top: at,
                    length: cell.length,
                });
                index = index.saturating_add(n);
                at += cell.length;
                filled = true;
            }
            used.push(at);
            if p + 1 < count {
                // The line break ends its column; the next paragraph starts a new one.
                places.push(Place { column, top: at, length: 0.0 });
                index = index.saturating_add(1);
                column = column.saturating_add(1);
            } else {
                end = (column, at);
            }
        }
        VerticalEditLayout { glyphs, places, end, used, pitch: PITCH * size, column_length }
    }

    /// How many characters the text has (the caret goes from 0 to this).
    pub fn chars(&self) -> usize {
        self.places.len()
    }

    /// How many columns the text takes (at least one).
    pub fn columns(&self) -> usize {
        self.used.len().max(1)
    }

    /// The width the columns take, in points.
    pub fn width(&self) -> f64 {
        self.pitch * self.columns() as f64
    }

    /// The left and right edges of column `c`, as distances left of the box's right edge.
    pub fn column_span(&self, c: usize) -> (f64, f64) {
        (self.pitch * (c as f64 + 1.0), self.pitch * c as f64)
    }

    /// Where the caret before character `index` stands: its column and its distance down it.
    /// Past the end it stands after the last character.
    pub fn caret(&self, index: usize) -> (usize, f64) {
        self.places.get(index).map_or(self.end, |p| (p.column, p.top))
    }

    /// The characters in column `c`, in order.
    fn in_column(&self, c: usize) -> impl Iterator<Item = (usize, &Place)> {
        self.places.iter().enumerate().filter(move |(_, p)| p.column == c)
    }

    /// The caret index nearest the point `x` left of the box's right edge and `y` down from its
    /// top: in the column under `x` (the nearest one outside), before the character whose first
    /// half `y` is in, or at the column's end. The caret always shows in that column: below a
    /// column that wrapped it goes before the column's last character.
    pub fn index_at(&self, x: f64, y: f64) -> usize {
        let last = self.columns().saturating_sub(1);
        let c = if x.is_nan() || x <= 0.0 {
            0
        } else {
            let k = x / self.pitch;
            if k.is_finite() && k < last as f64 + 1.0 { (k.floor() as usize).min(last) } else { last }
        };
        let mut final_char = None;
        for (i, p) in self.in_column(c) {
            if y < p.top + p.length / 2.0 {
                return i;
            }
            final_char = Some(i);
        }
        match final_char {
            // After the column's last character where that still shows in this column (the end
            // of the text); before it where the column wrapped or ends in a line break.
            Some(i) => {
                let after = i.saturating_add(1);
                if self.caret(after).0 == c { after } else { i }
            }
            // An empty column: the caret where it starts.
            None => self.places.iter().position(|p| p.column > c).unwrap_or(self.chars()),
        }
    }

    /// Where `index` goes after `motion`.
    pub fn moved(&self, index: usize, motion: CaretMove) -> usize {
        let index = index.min(self.chars());
        let (c, y) = self.caret(index);
        let centre = |c: usize| self.pitch * (c as f64 + 0.5);
        match motion {
            CaretMove::Up => index.saturating_sub(1),
            CaretMove::Down => index.saturating_add(1).min(self.chars()),
            CaretMove::NextColumn if c >= self.columns().saturating_sub(1) => self.chars(),
            CaretMove::NextColumn => self.index_at(centre(c + 1), y),
            CaretMove::PreviousColumn if c == 0 => 0,
            CaretMove::PreviousColumn => self.index_at(centre(c - 1), y),
            CaretMove::ColumnStart => self.in_column(c).map(|(i, _)| i).next().unwrap_or(index),
            CaretMove::ColumnEnd => self.index_at(centre(c), f64::INFINITY),
            CaretMove::TextStart => 0,
            CaretMove::TextEnd => self.chars(),
        }
    }

    /// The stretches of the columns that the characters `range` take, to highlight a selection:
    /// (column, top, bottom), one per column, in order.
    pub fn selection(&self, range: Range<usize>) -> Vec<(usize, f64, f64)> {
        let mut out: Vec<(usize, f64, f64)> = Vec::new();
        for p in self.places.get(range.start.min(self.chars())..range.end.min(self.chars())).unwrap_or_default() {
            match out.last_mut() {
                Some((c, top, bottom)) if *c == p.column => {
                    *top = top.min(p.top);
                    *bottom = bottom.max(p.top + p.length);
                }
                _ => out.push((p.column, p.top, p.top + p.length)),
            }
        }
        out.retain(|(_, top, bottom)| bottom > top);
        out
    }
}

/// The word around character `index` of `text`, to select on a double click: the run of
/// characters of one kind (kanji, hiragana, katakana, letters and digits) that it is in, or the
/// character itself.
pub fn word_at(text: &str, index: usize) -> Range<usize> {
    #[derive(PartialEq)]
    enum Kind {
        Han,
        Hiragana,
        Katakana,
        Word,
        Other,
    }
    let kind = |c: char| match u32::from(c) {
        0x3400..=0x4DBF | 0x4E00..=0x9FFF | 0xF900..=0xFAFF | 0x20000..=0x3FFFF | 0x3005 => Kind::Han,
        0x3041..=0x309F => Kind::Hiragana,
        0x30A0..=0x30FF | 0x31F0..=0x31FF | 0xFF66..=0xFF9F => Kind::Katakana,
        _ if c.is_alphanumeric() => Kind::Word,
        _ => Kind::Other,
    };
    let chars: Vec<char> = text.chars().collect();
    let Some(last) = chars.len().checked_sub(1) else { return index..index };
    let index = index.min(last);
    let Some(&at) = chars.get(index) else { return index..index };
    let k = kind(at);
    if k == Kind::Other {
        return index..index + 1;
    }
    let mut start = index;
    while start > 0 && chars.get(start - 1).is_some_and(|&c| kind(c) == k) {
        start -= 1;
    }
    let mut end = index;
    while chars.get(end).is_some_and(|&c| kind(c) == k) {
        end += 1;
    }
    start..end.max(start)
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

    fn has_face() -> bool {
        pdfcraft_fonts::document_japanese_font_for_style(false, false).is_some()
    }

    /// Columns' texts as the layout places them.
    fn column_texts(l: &VerticalEditLayout) -> Vec<String> {
        let mut out = vec![String::new(); l.columns()];
        for g in &l.glyphs {
            if let Some(s) = out.get_mut(g.column) {
                s.push_str(&g.text);
            }
        }
        out
    }

    #[test]
    fn the_typed_text_wraps_where_the_committed_text_does() {
        // 50 pt columns hold five 10 pt characters (with the face, or one em each without it).
        for text in ["あいうえおかきくけこさ", "あいうえお。か", "吾輩は猫である。\n名前はまだ無い。2024年", "PDF を開く"]
        {
            let t = item(text, [100.0, 100.0, 200.0, 150.0]);
            let l = VerticalEditLayout::new(&t);
            assert_eq!(l.chars(), text.chars().count());
            if has_face() {
                let committed: Vec<String> = vertical::columns(&t).unwrap().iter().map(|c| c.iter().map(Cell::text).collect()).collect();
                assert_eq!(column_texts(&l), committed, "{text}");
            }
        }
        if has_face() {
            let l = VerticalEditLayout::new(&item("あいうえおかきくけこさ", [100.0, 100.0, 200.0, 150.0]));
            assert_eq!(column_texts(&l), ["あいうえお", "かきくけこ", "さ"]);
            assert!((l.width() - 36.0).abs() < 1e-9);
        }
    }

    /// A long run of characters that may not start a column, and a Latin word longer than one, stay
    /// within the box: one hung cell at most, the word broken over columns, typed as committed.
    #[test]
    fn long_runs_and_long_words_stay_inside_the_column() {
        for text in [
            "あ".to_string() + &"ー".repeat(30),
            "見て https://example.com/abcdefghijklmnopqrstuvwxyz".to_string(),
            "あいうえ」」」」」」」」".to_string(),
        ] {
            let t = item(&text, [100.0, 100.0, 200.0, 150.0]);
            let l = VerticalEditLayout::new(&t);
            assert_eq!(l.chars(), text.chars().count());
            for g in &l.glyphs {
                // 50 pt columns, and one 10 pt cell may hang below.
                assert!(g.top + g.length <= 50.0 + 10.0 + 1e-6, "{text}: {g:?}");
            }
            if has_face() {
                let committed: Vec<String> = vertical::columns(&t).unwrap().iter().map(|c| c.iter().map(Cell::text).collect()).collect();
                assert_eq!(column_texts(&l), committed, "{text}");
                for column in vertical::columns(&t).unwrap() {
                    let used: f64 = column.iter().map(|c| c.length(&t)).sum();
                    assert!(used <= 50.0 + 10.0 + 1e-6, "{text}: {used}");
                }
            }
        }
    }

    #[test]
    fn every_character_has_a_caret_place_in_order() {
        let t = item("一二三四五六七\n\nab cd 7月 ｜東京《とうきょう》", [0.0, 0.0, 100.0, 40.0]);
        let l = VerticalEditLayout::new(&t);
        let n = t.text.chars().count();
        assert_eq!(l.chars(), n, "ruby markup is laid out as typed");
        // Down a column the caret only goes down; a later column is further left.
        let mut last = l.caret(0);
        for i in 1..=n {
            let now = l.caret(i);
            assert!(now.0 > last.0 || now.0 == last.0 && now.1 >= last.1, "caret {i}: {now:?} after {last:?}");
            last = now;
        }
        // The empty paragraph has a column of its own.
        let breaks: Vec<usize> = t.text.chars().enumerate().filter(|(_, c)| *c == '\n').map(|(i, _)| i).collect();
        assert_eq!(l.caret(breaks[1]).0, l.caret(breaks[0]).0 + 1);
        assert_eq!(l.caret(breaks[1]).1, 0.0);
        // Glyphs cover every character but the line breaks, once.
        let covered: usize = l.glyphs.iter().map(|g| g.chars.len()).sum();
        assert!(covered <= n - breaks.len() && covered + 2 >= n - breaks.len(), "{covered} of {n}");
    }

    #[test]
    fn no_text_and_any_index_are_safe() {
        let empty = VerticalEditLayout::new(&item("", [0.0, 0.0, 100.0, 40.0]));
        assert_eq!(empty.chars(), 0);
        assert_eq!(empty.columns(), 1);
        assert_eq!(empty.caret(0), (0, 0.0));
        assert_eq!(empty.caret(usize::MAX), (0, 0.0));
        for m in [
            CaretMove::Up,
            CaretMove::Down,
            CaretMove::NextColumn,
            CaretMove::PreviousColumn,
            CaretMove::ColumnStart,
            CaretMove::ColumnEnd,
            CaretMove::TextStart,
            CaretMove::TextEnd,
        ] {
            assert_eq!(empty.moved(0, m), 0);
            assert_eq!(empty.moved(usize::MAX, m), 0);
        }
        assert_eq!(empty.index_at(f64::NAN, f64::NAN), 0);
        assert_eq!(empty.index_at(-5.0, 1e300), 0);
        assert!(empty.selection(0..usize::MAX).is_empty());
        // Hostile sizes don't make a layout of NaN or loop.
        let mut t = item("あいう abc", [0.0, 0.0, f64::NAN, f64::INFINITY]);
        t.size = f64::NAN;
        t.char_spacing = f64::INFINITY;
        let l = VerticalEditLayout::new(&t);
        assert_eq!(l.chars(), 7);
        assert!(l.pitch.is_finite() && l.pitch > 0.0);
        // Long text stays quick and in order.
        let long = "吾輩は猫である。名前はまだ無い。".repeat(400);
        let l = VerticalEditLayout::new(&item(&long, [0.0, 0.0, 100.0, 200.0]));
        assert_eq!(l.chars(), long.chars().count());
        assert!(l.columns() > 10);
        assert_eq!(l.moved(l.chars(), CaretMove::Down), l.chars());
    }

    #[test]
    fn arrows_move_along_and_across_columns() {
        // Five characters a column without the face; with it the advances are close to an em, so
        // take a box tall enough that three characters fit whatever the face.
        let t = item("あいう\nかきく\nさしす", [0.0, 0.0, 100.0, 60.0]);
        let l = VerticalEditLayout::new(&t);
        assert_eq!(l.columns(), 3);
        // Up/Down: the previous and next character.
        assert_eq!(l.moved(1, CaretMove::Down), 2);
        assert_eq!(l.moved(1, CaretMove::Up), 0);
        assert_eq!(l.moved(0, CaretMove::Up), 0);
        assert_eq!(l.moved(11, CaretMove::Down), 11);
        // Left is the next column (columns run right to left), at the same height.
        assert_eq!(l.moved(1, CaretMove::NextColumn), 5, "い → き");
        assert_eq!(l.moved(5, CaretMove::NextColumn), 9, "き → し");
        assert_eq!(l.moved(9, CaretMove::NextColumn), 11, "past the last column: the end");
        assert_eq!(l.moved(5, CaretMove::PreviousColumn), 1, "き → い");
        assert_eq!(l.moved(1, CaretMove::PreviousColumn), 0, "before the first column: the start");
        // Home and End: the column's ends, the line break's place being the end of its column.
        assert_eq!(l.moved(5, CaretMove::ColumnStart), 4);
        assert_eq!(l.moved(5, CaretMove::ColumnEnd), 7);
        assert_eq!(l.moved(9, CaretMove::ColumnEnd), 11);
        assert_eq!(l.moved(5, CaretMove::TextStart), 0);
        assert_eq!(l.moved(5, CaretMove::TextEnd), 11);
        // The caret at the end of a column stays in it going across.
        assert_eq!(l.moved(3, CaretMove::NextColumn), 7);
    }

    #[test]
    fn a_click_finds_the_caret_in_its_column() {
        let t = item("あいう\nかきく", [0.0, 0.0, 100.0, 60.0]);
        let l = VerticalEditLayout::new(&t);
        let (left, right) = l.column_span(1);
        assert!(left > right);
        let mid = (left + right) / 2.0;
        // Above the column's first character: before it; on a character's second half: after it.
        assert_eq!(l.index_at(mid, -3.0), 4);
        let (_, top_ki) = l.caret(5);
        assert_eq!(l.index_at(mid, top_ki + 0.1), 5);
        assert_eq!(l.index_at(mid, l.caret(6).1 - 0.1), 6);
        assert_eq!(l.index_at(mid, 1e9), 7, "below the text: the end");
        assert_eq!(l.index_at(1.0, 1e9), 3, "below the first column: its line break");
        assert_eq!(l.index_at(1e9, 0.0), 4, "left of every column: the last column");
        assert_eq!(l.index_at(-1e9, 0.0), 0, "right of the box: the first column");
    }

    #[test]
    fn a_wrapped_column_keeps_its_caret() {
        if !has_face() {
            return;
        }
        // "あいうえお" fills the first 50 pt column; か starts the second.
        let l = VerticalEditLayout::new(&item("あいうえおか", [0.0, 0.0, 100.0, 50.0]));
        assert_eq!(l.columns(), 2);
        assert_eq!(l.caret(5).0, 1, "the caret before か is at the top of the second column");
        // Below the first column's text: before its last character, which still shows there.
        assert_eq!(l.index_at(5.0, 1e9), 4);
        assert_eq!(l.moved(5, CaretMove::PreviousColumn), 0);
    }

    #[test]
    fn a_selection_highlights_each_column_it_touches() {
        let l = VerticalEditLayout::new(&item("あいう\nかきく", [0.0, 0.0, 100.0, 60.0]));
        let s = l.selection(1..6);
        assert_eq!(s.iter().map(|(c, ..)| *c).collect::<Vec<_>>(), [0, 1]);
        assert!((s[0].1 - l.caret(1).1).abs() < 1e-9, "from い");
        assert_eq!(s[1].1, 0.0, "the second column from its top");
        assert!((s[1].2 - l.caret(6).1).abs() < 1e-9, "to the end of き");
        assert!(l.selection(3..3).is_empty());
        #[allow(clippy::reversed_empty_ranges, reason = "a reversed range from a caller must be safe")]
        let reversed = 9..2;
        assert!(l.selection(reversed).is_empty());
    }

    #[test]
    fn sideways_words_and_combined_digits_are_placed_per_character() {
        let l = VerticalEditLayout::new(&item("令和7年 PDF", [0.0, 0.0, 100.0, 300.0]));
        let kinds: Vec<(VerticalGlyphKind, &str)> = l.glyphs.iter().map(|g| (g.kind, g.text.as_str())).collect();
        assert!(kinds.contains(&(VerticalGlyphKind::Combined, "7")), "{kinds:?}");
        let pdf = l.glyphs.iter().find(|g| g.kind == VerticalGlyphKind::Sideways && g.text.contains("PDF")).unwrap();
        // Each letter of the sideways word has its own caret place, in order down the column.
        let tops: Vec<f64> = pdf.chars.clone().map(|i| l.caret(i).1).collect();
        assert!(tops.windows(2).all(|w| w[1] > w[0]), "{tops:?}");
    }

    #[test]
    fn a_double_click_selects_a_run_of_one_kind() {
        let s = "吾輩は猫である。PDF版";
        assert_eq!(word_at(s, 0), 0..2, "吾輩");
        assert_eq!(word_at(s, 2), 2..3, "は");
        assert_eq!(word_at(s, 3), 3..4, "猫");
        assert_eq!(word_at(s, 7), 7..8, "。 alone");
        assert_eq!(word_at(s, 9), 8..11, "PDF");
        assert_eq!(word_at(s, 99), 11..12, "past the end: the last character's run");
        assert_eq!(word_at("", 3), 3..3);
    }
}
