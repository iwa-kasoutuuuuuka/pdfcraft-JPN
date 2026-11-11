//! Paragraphs of a page's text, for exports that want prose rather than lines (plain text and
//! Word from the Scan & OCR tool): consecutive lines of one block, joined, with horizontal and
//! vertical (縦書き) text kept apart.
//!
//! The lines come from [`PageText`] already in reading order (columns of vertical text run right
//! to left). Here they are only grouped: a paragraph ends where the writing direction changes,
//! where the next line is not beside or under the previous one (another column or block), or where
//! the gap between them is wider than a line of text usually leaves.

use super::{PageText, TextGlyph};

/// One paragraph of a page, in reading order.
#[derive(Clone, Debug, PartialEq)]
pub struct TextParagraph {
    /// The text, its lines joined: nothing between two lines of Chinese, Japanese or Korean text,
    /// one space between lines of other scripts.
    pub text: String,
    /// The lines run top to bottom (縦書き) instead of left to right.
    pub vertical: bool,
    /// How many glyphs it holds (the weight of this paragraph when judging a whole page).
    pub glyphs: usize,
    /// Box in page view space: [x0, y0, x1, y1].
    pub rect: [f32; 4],
}

/// A line of the page.
struct Line {
    text: String,
    vertical: bool,
    glyphs: usize,
    rect: [f32; 4],
}

/// Whether `c` is written without spaces between words (Han, kana, Hangul and their punctuation,
/// full-width forms).
fn unspaced(c: char) -> bool {
    let n: u32 = c.into();
    (0x2E80..=0x9FFF).contains(&n)
        || (0xAC00..=0xD7AF).contains(&n)
        || (0xF900..=0xFAFF).contains(&n)
        || (0xFF00..=0xFFEF).contains(&n)
        || (0x20000..=0x3FFFF).contains(&n)
}

/// Whether a glyph runs down the page: its writing direction is more vertical than horizontal.
fn runs_down(g: &TextGlyph) -> bool {
    g.direction[1].abs() > g.direction[0].abs()
}

fn grow(r: &mut [f32; 4], o: &[f32; 4]) {
    r[0] = r[0].min(o[0]);
    r[1] = r[1].min(o[1]);
    r[2] = r[2].max(o[2]);
    r[3] = r[3].max(o[3]);
}

/// Spans [a0, a1] and [b0, b1] share some length.
fn overlap(a0: f32, a1: f32, b0: f32, b1: f32) -> bool {
    a1.min(b1) - a0.max(b0) > 0.0
}

impl PageText {
    /// The page's lines, blank ones left out.
    fn text_lines(&self) -> Vec<Line> {
        let mut lines: Vec<Line> = Vec::new();
        let mut start = 0usize;
        while start < self.glyphs.len() {
            let id = self.line_of.get(start).copied();
            let mut end = start + 1;
            while end < self.glyphs.len() && self.line_of.get(end).copied() == id {
                end += 1;
            }
            let text = self.text_of(start..end);
            if !text.trim().is_empty() {
                let mut rect = [f32::INFINITY, f32::INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY];
                let mut down = 0usize;
                for g in self.glyphs.get(start..end).unwrap_or_default() {
                    if g.rect.iter().all(|v| v.is_finite()) {
                        grow(&mut rect, &g.rect);
                    }
                    down += usize::from(runs_down(g));
                }
                if rect.iter().all(|v| v.is_finite()) {
                    lines.push(Line { text: text.trim().to_string(), vertical: down * 2 > end - start, glyphs: end - start, rect });
                }
            }
            start = end;
        }
        lines
    }

    /// The page's text as paragraphs in reading order. A line belongs to the paragraph before it
    /// when it has the same writing direction, lies beside it (vertical text: below it) and is no
    /// farther from it than a gap of about three quarters of a line.
    pub fn paragraphs(&self) -> Vec<TextParagraph> {
        let mut out: Vec<TextParagraph> = Vec::new();
        // The last line added, for the geometry of the next join.
        let mut last: Option<[f32; 4]> = None;
        for line in self.text_lines() {
            let joins = match (out.last(), last) {
                (Some(p), Some(prev)) if p.vertical == line.vertical => {
                    let r = line.rect;
                    if line.vertical {
                        // Columns: the next one lies to the left, one column's width away at most.
                        let width = (r[2] - r[0]).max(1e-3);
                        let gap = prev[0] - r[2];
                        overlap(prev[1], prev[3], r[1], r[3]) && gap <= 0.75 * width && r[0] <= prev[0] + 0.5 * width
                    } else {
                        let height = (r[3] - r[1]).max(1e-3);
                        let gap = r[1] - prev[3];
                        overlap(prev[0], prev[2], r[0], r[2]) && gap <= 0.75 * height && r[1] >= prev[1] - 0.5 * height
                    }
                }
                _ => false,
            };
            if joins && let Some(p) = out.last_mut() {
                let before = p.text.chars().last();
                let after = line.text.chars().next();
                if let (Some(a), Some(b)) = (before, after)
                    && !unspaced(a)
                    && !unspaced(b)
                {
                    p.text.push(' ');
                }
                p.text.push_str(&line.text);
                p.glyphs += line.glyphs;
                grow(&mut p.rect, &line.rect);
            } else {
                out.push(TextParagraph { text: line.text, vertical: line.vertical, glyphs: line.glyphs, rect: line.rect });
            }
            last = Some(line.rect);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn glyph(text: &str, rect: [f32; 4], direction: [f32; 2]) -> TextGlyph {
        let quad = [[rect[0], rect[1]], [rect[2], rect[1]], [rect[2], rect[3]], [rect[0], rect[3]]];
        TextGlyph { text: text.into(), rect, quad, direction }
    }

    /// Each entry is one line: its glyph texts, its box left/top, and whether it runs down.
    fn page(lines: &[(&str, [f32; 2], bool)]) -> PageText {
        let mut t = PageText::default();
        for (n, (text, at, down)) in lines.iter().enumerate() {
            for (k, c) in text.chars().enumerate() {
                let (x, y) = if *down { (at[0], at[1] + 10.0 * k as f32) } else { (at[0] + 10.0 * k as f32, at[1]) };
                let direction = if *down { [0.0, 1.0] } else { [1.0, 0.0] };
                t.glyphs.push(glyph(&c.to_string(), [x, y, x + 10.0, y + 10.0], direction));
                t.line_of.push(n as u32);
                t.space_before.push(false);
            }
        }
        t
    }

    #[test]
    fn lines_close_together_are_one_paragraph_and_a_wide_gap_starts_another() {
        let p = page(&[("日本語の", [10.0, 10.0], false), ("文章です", [10.0, 22.0], false), ("次の段落", [10.0, 60.0], false)]);
        let paragraphs = p.paragraphs();
        assert_eq!(paragraphs.iter().map(|p| p.text.as_str()).collect::<Vec<_>>(), ["日本語の文章です", "次の段落"]);
        assert!(paragraphs.iter().all(|p| !p.vertical));
    }

    #[test]
    fn latin_lines_join_with_a_space() {
        let p = page(&[("Hello", [10.0, 10.0], false), ("world", [10.0, 21.0], false)]);
        assert_eq!(p.paragraphs()[0].text, "Hello world");
    }

    #[test]
    fn vertical_columns_join_and_a_far_column_does_not() {
        // Columns run right to left: x = 100, 88, then a block far to the left at x = 20.
        let p = page(&[("縦書きの", [100.0, 10.0], true), ("文章です", [88.0, 10.0], true), ("別の塊", [20.0, 10.0], true)]);
        let paragraphs = p.paragraphs();
        assert_eq!(paragraphs.iter().map(|p| p.text.as_str()).collect::<Vec<_>>(), ["縦書きの文章です", "別の塊"]);
        assert!(paragraphs.iter().all(|p| p.vertical));
    }

    #[test]
    fn a_change_of_direction_or_a_jump_up_starts_a_paragraph() {
        let p = page(&[
            ("横書き", [10.0, 10.0], false),
            ("縦書き", [100.0, 24.0], true),
            ("二段目", [10.0, 100.0], false),
            ("右の段", [60.0, 12.0], false),
        ]);
        assert_eq!(p.paragraphs().len(), 4);
    }

    #[test]
    fn an_empty_or_blank_page_has_no_paragraphs() {
        assert!(PageText::default().paragraphs().is_empty());
        assert!(page(&[("   ", [0.0, 0.0], false)]).paragraphs().is_empty());
    }
}
