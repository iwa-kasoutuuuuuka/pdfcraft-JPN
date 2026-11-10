//! pdfcraft-JPN: ルビ (ruby, furigana) markup in added text, Aozora Bunko style.
//!
//! `｜親文字《るび》` marks the base explicitly (`｜` U+FF5C or `|`); in the short form
//! `漢字《かんじ》` the base is the run of kanji just before `《`. The markup stays in the
//! item's text, so it remains editable; drawing shows the base, with the ruby at half size above
//! it (horizontal text) or to its right (vertical text). Layout counts only the base, and the
//! page's text layer reads only the base.

use pdfcraft_fonts::ShapedCluster;

use super::super::EditError;
use super::super::types::AddedText;
use super::thai_error;

/// Extra room, in em of the item's size, that a line (horizontal) or column (vertical) with ruby takes.
pub(in super::super) const RUBY_LEAD: f64 = 0.5;

/// One piece of a paragraph: text drawn as it is, or a base with its ruby.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(in super::super) enum Segment {
    Plain(String),
    Ruby {
        base: String,
        ruby: String,
        /// The markup as typed, `｜` and `《…》` included.
        source: String,
    },
}

fn is_kanji(c: char) -> bool {
    matches!(u32::from(c), 0x4E00..=0x9FFF | 0x3400..=0x4DBF | 0xF900..=0xFAFF | 0x20000..=0x2FA1F) || matches!(c, '々' | '〆' | '〇')
}

/// `s` split into plain text and ruby groups. An unmatched `《`, an empty ruby (`《》`), a ruby
/// group with no base, or one that would span a line break stays plain text. Linear in `s`.
pub(in super::super) fn ruby_spans(s: &str) -> Vec<Segment> {
    let chars: Vec<char> = s.chars().collect();
    // The next `》`, `《` or line break at or after each position: where a `《` group ends, if it does.
    let mut stop: Vec<Option<usize>> = vec![None; chars.len() + 1];
    for (i, &c) in chars.iter().enumerate().rev() {
        stop[i] = if matches!(c, '》' | '《' | '\n') { Some(i) } else { stop[i + 1] };
    }
    let mut out = Vec::new();
    let mut plain: Vec<char> = Vec::new();
    // What the pending plain text ends with, kept as it grows: the last bar, whether a space
    // follows it, and the length of the kanji run at its end.
    let (mut bar, mut space_after_bar, mut kanji) = (None::<usize>, false, 0usize);
    let mut i = 0;
    while let Some(&c) = chars.get(i) {
        if c == '《'
            && let Some(close) = stop.get(i + 1).copied().flatten().filter(|&j| chars.get(j) == Some(&'》'))
        {
            let ruby: String = chars.get(i + 1..close).unwrap_or_default().iter().collect();
            if !ruby.trim().is_empty()
                && let Some((keep, base, bar_char)) = split_base(&plain, bar.filter(|_| !space_after_bar), kanji)
            {
                if !keep.is_empty() {
                    out.push(Segment::Plain(keep.iter().collect()));
                }
                let base: String = base.iter().collect();
                let source = format!("{}{base}《{ruby}》", bar_char.map(String::from).unwrap_or_default());
                out.push(Segment::Ruby { base, ruby, source });
                plain.clear();
                (bar, space_after_bar, kanji) = (None, false, 0);
                i = close + 1;
                continue;
            }
        }
        if c == '｜' || c == '|' {
            (bar, space_after_bar) = (Some(plain.len()), false);
        } else if c.is_whitespace() {
            space_after_bar = true;
        }
        kanji = if is_kanji(c) { kanji + 1 } else { 0 };
        plain.push(c);
        i += 1;
    }
    if !plain.is_empty() {
        out.push(Segment::Plain(plain.iter().collect()));
    }
    out
}

/// The pending plain text split into what stays plain, the ruby's base and the bar that opened it.
/// `bar` is the position of an explicit bar with no space after it, `kanji` the length of the kanji
/// run at the text's end.
fn split_base(plain: &[char], bar: Option<usize>, kanji: usize) -> Option<(&[char], &[char], Option<char>)> {
    if let Some(b) = bar {
        let after = plain.get(b + 1..)?;
        if !after.is_empty() {
            return Some((plain.get(..b)?, after, plain.get(b).copied()));
        }
    }
    if kanji == 0 {
        return None;
    }
    let at = plain.len().checked_sub(kanji)?;
    Some((plain.get(..at)?, plain.get(at..)?, None))
}

/// Whether `s` has ruby markup that is drawn as ruby.
pub(in super::super) fn has_ruby(s: &str) -> bool {
    s.contains('《') && ruby_spans(s).iter().any(|seg| matches!(seg, Segment::Ruby { .. }))
}

/// `s` as drawn in the line: the markup and the ruby left out.
pub(in super::super) fn visible(s: &str) -> String {
    if !s.contains('《') {
        return s.to_string();
    }
    ruby_spans(s)
        .into_iter()
        .map(|seg| match seg {
            Segment::Plain(p) => p,
            Segment::Ruby { base, .. } => base,
        })
        .collect()
}

/// `s` with the markup characters removed but the ruby text kept: what has to be drawable.
pub(in super::super) fn without_markup(s: &str) -> String {
    ruby_spans(s)
        .into_iter()
        .map(|seg| match seg {
            Segment::Plain(p) => p,
            Segment::Ruby { base, ruby, .. } => base + &ruby,
        })
        .collect()
}

/// Extra height a horizontal line takes for its ruby, in points.
pub(in super::super) fn lead(t: &AddedText, line: &str) -> f64 {
    if has_ruby(line) { RUBY_LEAD * t.size } else { 0.0 }
}

/// A ruby of a horizontal line.
pub(in super::super) struct Mark {
    /// Where the base starts, from the line's start, and its width (points).
    pub x: f64,
    pub width: f64,
    /// The ruby, shaped, and its width at half the item's size.
    pub clusters: Vec<ShapedCluster>,
    pub w: f64,
}

/// The rubies of `line`, with where their bases lie.
pub(in super::super) fn line_marks(t: &AddedText, line: &str) -> Result<Vec<Mark>, EditError> {
    let mut marks = Vec::new();
    let mut pos = 0.0;
    for seg in ruby_spans(line) {
        match seg {
            Segment::Plain(s) => pos += super::shaped_width(t, &s, false),
            Segment::Ruby { base, ruby, .. } => {
                let width = super::shaped_width(t, &base, false);
                let clusters = pdfcraft_fonts::shape_japanese(&ruby).map_err(|e| thai_error(e, &ruby))?;
                let w = clusters.iter().map(|c| c.advance * t.size / 2.0).sum();
                marks.push(Mark { x: pos, width, clusters, w });
                pos += width;
            }
        }
    }
    Ok(marks)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plain(s: &str) -> Segment {
        Segment::Plain(s.into())
    }
    fn ruby(base: &str, ruby: &str, source: &str) -> Segment {
        Segment::Ruby { base: base.into(), ruby: ruby.into(), source: source.into() }
    }

    #[test]
    fn short_form_takes_the_kanji_run_before_the_bracket() {
        assert_eq!(ruby_spans("へ東京《とうきょう》へ"), [plain("へ"), ruby("東京", "とうきょう", "東京《とうきょう》"), plain("へ")]);
    }

    #[test]
    fn explicit_bar_takes_everything_after_it() {
        assert_eq!(ruby_spans("a｜親文字《るび》b"), [plain("a"), ruby("親文字", "るび", "｜親文字《るび》"), plain("b")]);
        assert_eq!(ruby_spans("|Word《ワード》"), [ruby("Word", "ワード", "|Word《ワード》")]);
        // A space after the bar: not a ruby base, the short form applies.
        assert_eq!(ruby_spans("A | B 漢字《かんじ》"), [plain("A | B "), ruby("漢字", "かんじ", "漢字《かんじ》")]);
    }

    #[test]
    fn unmatched_empty_and_baseless_markup_is_plain() {
        for s in ["東京《とうきょう", "《あ》", "東京《》", "｜《あ》", "あ《い《う》", "東京《と\nう》", "｜あ", "plain"]
        {
            let spans = ruby_spans(s);
            let joined: String = spans
                .iter()
                .map(|seg| match seg {
                    Segment::Plain(p) => p.clone(),
                    Segment::Ruby { source, .. } => source.clone(),
                })
                .collect();
            assert_eq!(joined, s, "markup round-trips");
        }
        assert!(!has_ruby("東京《とうきょう"));
        assert!(!has_ruby("東京《》"));
        assert!(!has_ruby("《あ》"));
        assert_eq!(ruby_spans("あ《い《う》"), [plain("あ《い《う》")]);
    }

    #[test]
    fn parsing_is_linear_in_the_text() {
        let start = std::time::Instant::now();
        // Every group fails (no base), so the pending plain text only grows.
        for unit in ["x《あ》", "|x 《あ》", "《", "｜a《b》c"] {
            let text = unit.repeat(200_000);
            assert!(!has_ruby(&text) || unit == "｜a《b》c");
            assert!(visible(&text).chars().count() <= text.chars().count());
            let _ = without_markup(&text);
        }
        assert!(start.elapsed().as_secs() < 10, "{:?}", start.elapsed());
    }

    #[test]
    fn visible_text_is_the_base_alone() {
        assert_eq!(visible("令和7年に東京《とうきょう》へ"), "令和7年に東京へ");
        assert_eq!(visible("｜親文字《るび》と漢字《かんじ》"), "親文字と漢字");
        assert_eq!(visible("no markup"), "no markup");
        assert_eq!(without_markup("｜親文字《るび》"), "親文字るび");
        assert!(has_ruby("東京《とうきょう》"));
    }
}
