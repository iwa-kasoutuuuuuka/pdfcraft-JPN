//! Read-only document statistics, processing one page at a time without rasterizing it.

use std::collections::{HashMap, HashSet};

use pdfcraft_cos::{Dict, ObjRef, Object};
use pdfcraft_render::{PageRenderer, text::TextFilter};
use unicode_segmentation::UnicodeSegmentation;

use crate::export::ExportSource;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Options {
    pub exclude_tables: bool,
    pub exclude_code: bool,
    /// Figure-marked text and invisible text (including OCR). This does not run OCR.
    pub exclude_images: bool,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Counts {
    pub words: usize,
    /// Unicode scalar values, including whitespace.
    pub characters: usize,
    pub characters_without_spaces: usize,
    /// pdfcraft-JPN: characters without whitespace, punctuation (、。「」 and the like) or symbols
    /// ([`is_punctuation_or_symbol`]): the count Japanese writing usually means by 文字数.
    pub characters_without_punctuation: usize,
    /// pdfcraft-JPN: the lines of 400字詰め原稿用紙 the text fills ([`manuscript_lines`]). Each
    /// page starts a new line. [`Counts::manuscript_sheets`] turns it into sheets.
    pub manuscript_lines: usize,
}

/// Squares on one line of a 400字詰め原稿用紙.
pub const MANUSCRIPT_SQUARES_PER_LINE: usize = 20;
/// Lines on one sheet of a 400字詰め原稿用紙.
pub const MANUSCRIPT_LINES_PER_SHEET: usize = 20;

impl Counts {
    pub fn of(text: &str) -> Self {
        Self {
            words: text.unicode_words().count(),
            characters: text.chars().count(),
            characters_without_spaces: text.chars().filter(|c| !c.is_whitespace()).count(),
            characters_without_punctuation: text.chars().filter(|&c| !c.is_whitespace() && !is_punctuation_or_symbol(c)).count(),
            manuscript_lines: manuscript_lines(text),
        }
    }

    fn add(&mut self, other: Self) {
        self.words = self.words.saturating_add(other.words);
        self.characters = self.characters.saturating_add(other.characters);
        self.characters_without_spaces = self.characters_without_spaces.saturating_add(other.characters_without_spaces);
        self.characters_without_punctuation = self.characters_without_punctuation.saturating_add(other.characters_without_punctuation);
        self.manuscript_lines = self.manuscript_lines.saturating_add(other.manuscript_lines);
    }

    /// The 400字詰め原稿用紙 sheets the text fills, and the lines used on the last of them:
    /// `(0, 0)` without text, `(1, 20)` for exactly one full sheet, `(2, 1)` for one line more.
    pub fn manuscript_sheets(&self) -> (usize, usize) {
        let lines = self.manuscript_lines;
        let sheets = lines.div_ceil(MANUSCRIPT_LINES_PER_SHEET);
        (sheets, lines.saturating_sub(sheets.saturating_sub(1).saturating_mul(MANUSCRIPT_LINES_PER_SHEET)))
    }
}

/// Punctuation and symbols, which Japanese character counts usually leave out: ASCII and
/// Latin-1 punctuation and signs, general punctuation (… ‥ “ ” ‐ —), arrows, mathematical
/// operators, technical symbols, box drawing, shapes (■ ○ ★) and dingbats, CJK punctuation
/// (、。「」『』【】〜 and the full-width space) and the full-width and half-width forms of
/// punctuation (！？（）：・ ｢｣). Letters that live among them stay characters: 々 〆 〇 ー, and
/// ª º µ and the superscripts and fractions of Latin-1.
pub fn is_punctuation_or_symbol(c: char) -> bool {
    c.is_ascii_punctuation()
        || matches!(c,
            '\u{A1}'..='\u{A9}' | '\u{AB}'..='\u{B1}' | '\u{B4}' | '\u{B6}'..='\u{B8}' | '\u{BB}' | '\u{BF}' | '\u{D7}' | '\u{F7}'
            | '\u{2010}'..='\u{205E}'
            | '\u{2190}'..='\u{2BFF}'
            | '\u{3000}'..='\u{3004}' | '\u{3008}'..='\u{3020}' | '\u{3030}' | '\u{303D}' | '\u{30A0}' | '\u{30FB}'
            | '\u{FE30}'..='\u{FE4F}'
            | '\u{FF01}'..='\u{FF0F}' | '\u{FF1A}'..='\u{FF20}' | '\u{FF3B}'..='\u{FF40}' | '\u{FF5B}'..='\u{FF65}' | '\u{FFE0}'..='\u{FFEE}')
}

/// The lines of 400字詰め原稿用紙 (20 squares a line) that `text` fills when it is written in
/// the way a manuscript sheet is filled:
///
/// * every character takes a square of its own, except half-width whitespace, which the text
///   extraction puts between words and lines and which takes none; a full-width space (U+3000)
///   is written as an empty square;
/// * every paragraph (line break) starts a new line, and an empty paragraph is an empty line. No
///   indentation is added: a paragraph's first square is empty only when the text itself starts
///   it with a full-width space. Empty lines before and after the text take no line;
/// * 行頭禁則: a character that may not start a line ([`pdfcraft_fonts::no_line_start`]: 、。」
///   small kana, ー …) that would fall in the first square of a line hangs at the end of the
///   previous line (ぶら下げ) instead of taking a square. When two or more such characters follow
///   each other, or a line would end with an opening bracket (行末禁則), the last character of
///   the line goes down with them (追い出し), as in the 原稿用紙 documents PdfCraft makes.
pub fn manuscript_lines(text: &str) -> usize {
    let breaks = |c: char| matches!(c, '\n' | '\r' | '\u{B}' | '\u{C}' | '\u{85}' | '\u{2028}' | '\u{2029}');
    let text = text.replace("\r\n", "\n");
    let squares = |para: &str| -> Vec<char> { para.chars().filter(|&c| c == '\u{3000}' || !c.is_whitespace()).collect() };
    let paragraphs: Vec<Vec<char>> = text.split(breaks).map(squares).collect();
    let Some(first) = paragraphs.iter().position(|p| !p.is_empty()) else { return 0 };
    let last = paragraphs.iter().rposition(|p| !p.is_empty()).unwrap_or(first);
    paragraphs.get(first..=last).unwrap_or_default().iter().fold(0usize, |lines, p| lines.saturating_add(paragraph_lines(p)))
}

/// The lines one paragraph's `squares` take on the sheet (an empty paragraph takes one).
fn paragraph_lines(squares: &[char]) -> usize {
    use pdfcraft_fonts::{no_line_end, no_line_start};
    let mut lines = 1usize;
    let mut row = 0usize;
    let mut i = 0usize;
    while i < squares.len() {
        if row >= MANUSCRIPT_SQUARES_PER_LINE {
            let run = squares.iter().skip(i).take_while(|&&s| no_line_start(s)).take(2).count();
            let last_opens = i.checked_sub(1).and_then(|k| squares.get(k)).is_some_and(|&p| no_line_end(p));
            if run == 1 && !last_opens {
                // ぶら下げ: it takes no square, and the character after it starts the next line
                // (when there is one).
                i += 1;
                if i < squares.len() {
                    lines = lines.saturating_add(1);
                    row = 0;
                }
                continue;
            }
            // 追い出し: the line's last character goes down with the ones that can't start a
            // line (or because it is an opening bracket), so the new line already holds it.
            lines = lines.saturating_add(1);
            row = usize::from(run >= 2 || last_opens);
            continue;
        }
        row += 1;
        i += 1;
    }
    lines
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Report {
    pub total: Counts,
    /// Zero-based pages, in the requested order, without duplicates.
    pub pages: Vec<(usize, Counts)>,
    pub pages_without_text: usize,
    /// Some structure could not be classified safely; that text was retained.
    pub unclassified_structure: bool,
}

/// Count the current document snapshot. `progress` can cancel between pages. A failed page is
/// an error, never a successful zero or an unlabelled partial total.
pub fn count(source: ExportSource, pages: &[usize], options: Options, progress: impl FnMut(usize) -> bool) -> Result<Report, String> {
    crate::guard(|| count_inner(source, pages, options, progress))?
}

fn count_inner(source: ExportSource, pages: &[usize], options: Options, mut progress: impl FnMut(usize) -> bool) -> Result<Report, String> {
    if pages.iter().any(|&page| page >= source.pages) {
        return Err("the word-count page range is out of bounds".into());
    }
    if !progress(0) {
        return Err("word count cancelled".into());
    }
    let (mut excluded, unclassified) = if options != Options::default() { exclusions(&source, options)? } else { (HashMap::new(), false) };
    let mut renderer = PageRenderer::new(source.bytes, source.config);
    let mut report = Report { unclassified_structure: unclassified, ..Report::default() };
    let mut seen = HashSet::new();
    for &page in pages {
        if !seen.insert(page) {
            continue;
        }
        if !progress(report.pages.len()) {
            return Err("word count cancelled".into());
        }
        let filter = TextFilter {
            tables: options.exclude_tables,
            code: options.exclude_code,
            images: options.exclude_images,
            painted: false,
            mcids: excluded.remove(&page).unwrap_or_default(),
        };
        let text = renderer.filtered_text(page, filter).map_err(|e| format!("page {}: {e}", page.saturating_add(1)))?.plain_text();
        let counts = Counts::of(&text);
        report.pages_without_text += usize::from(text.trim().is_empty());
        report.total.add(counts);
        report.pages.push((page, counts));
    }
    if !progress(report.pages.len()) {
        return Err("word count cancelled".into());
    }
    Ok(report)
}

fn excluded_role(role: &[u8], options: Options) -> bool {
    (options.exclude_tables && matches!(role, b"Table" | b"TR" | b"TD" | b"TH" | b"THead" | b"TBody" | b"TFoot"))
        || (options.exclude_code && role == b"Code")
        || (options.exclude_images && role == b"Figure")
}

fn standard_role(doc: &pdfcraft_cos::Document, map: &Dict, role: &[u8]) -> Vec<u8> {
    let mut role = role.to_vec();
    for _ in 0..32 {
        let Some(next) = map.get(&role).map(|o| doc.resolve(o)).and_then(|o| o.as_name().map(<[u8]>::to_vec)) else { break };
        if next == role {
            break;
        }
        role = next;
    }
    role
}

/// Whether the page's resources hold a Form XObject, whose MCIDs are a separate namespace.
fn page_has_form_xobject(doc: &pdfcraft_cos::Document, page: &pdfcraft_cos::Dict) -> bool {
    let resources = page.get(b"Resources").map(|o| doc.resolve(o));
    let objects = resources.as_ref().and_then(|o| o.as_dict()).and_then(|d| d.get(b"XObject")).map(|o| doc.resolve(o));
    objects
        .as_ref()
        .and_then(|o| o.as_dict())
        .is_some_and(|d| d.iter().any(|(_, value)| matches!(&*doc.resolve(value), Object::Stream(s) if s.dict.name(b"Subtype") == Some(b"Form"))))
}

type Exclusions = HashMap<usize, HashSet<i32>>;

/// Resolve page-local MCIDs through the structure hierarchy, including inherited roles and
/// RoleMap aliases. Form XObjects have a separate MCID namespace that the text device does not
/// expose: retain their page's MCID-tagged text instead of excluding an unrelated same-numbered
/// span. Explicit Table/Code/Figure marked-content tags still work on those pages.
fn exclusions(source: &ExportSource, options: Options) -> Result<(Exclusions, bool), String> {
    let doc = pdfcraft_cos::Document::open_with_password(source.bytes.clone(), source.config.password.as_deref()).map_err(|e| e.to_string())?;
    let catalog = doc.root().and_then(|id| doc.get(id).as_dict().cloned()).unwrap_or_default();
    let Some(root) = catalog.get(b"StructTreeRoot").map(|o| doc.resolve(o)).and_then(|o| o.as_dict().cloned()) else {
        return Ok((HashMap::new(), options.exclude_tables || options.exclude_code));
    };
    let roles = root.get(b"RoleMap").map(|o| doc.resolve(o)).and_then(|o| o.as_dict().cloned()).unwrap_or_default();
    let pages = pdfcraft_model::pages(&doc);
    let page_index: HashMap<_, _> = pages.iter().enumerate().map(|(index, page)| (page.obj, index)).collect();
    let mut stack = root.get(b"K").map(|k| vec![(k.clone(), None::<ObjRef>, false, 0usize)]).unwrap_or_default();
    let mut seen = HashSet::new();
    let mut out: Exclusions = HashMap::new();
    let mut unclassified = false;
    let mut visited = 0usize;
    while let Some((object, page, excluded, depth)) = stack.pop() {
        visited += 1;
        if visited > 100_000 || depth > 128 {
            unclassified = true;
            break;
        }
        if let Some(id) = object.as_ref()
            && !seen.insert(id)
        {
            unclassified = true;
            continue;
        }
        let object = doc.resolve(&object);
        let mut keep = |id: i64, page: Option<ObjRef>| {
            if excluded {
                if let Some((page, id)) = page.and_then(|p| page_index.get(&p).copied()).zip(i32::try_from(id).ok().filter(|id| *id >= 0)) {
                    out.entry(page).or_default().insert(id);
                } else {
                    unclassified = true;
                }
            }
        };
        match &*object {
            Object::Int(id) => keep(*id, page),
            Object::Array(items) => {
                if stack.len().saturating_add(items.len()) > 100_000 {
                    unclassified = true;
                    break;
                }
                stack.extend(items.iter().rev().map(|child| (child.clone(), page, excluded, depth + 1)));
            }
            Object::Dict(dict) => {
                let page = dict.get(b"Pg").and_then(Object::as_ref).or(page);
                if dict.contains(b"Stm") {
                    unclassified = true;
                } else if let Some(id) = dict.get(b"MCID").and_then(Object::as_int) {
                    keep(id, page);
                } else if let Some(kids) = dict.get(b"K") {
                    let excluded = excluded || dict.name(b"S").is_some_and(|role| excluded_role(&standard_role(&doc, &roles, role), options));
                    stack.push((kids.clone(), page, excluded, depth + 1));
                }
            }
            _ => {}
        }
    }
    for (index, page) in pages.iter().enumerate() {
        if !out.contains_key(&index) {
            continue;
        }
        let has_form = page_has_form_xobject(&doc, &page.dict);
        if has_form {
            out.remove(&index);
            unclassified = true;
        }
    }
    Ok((out, unclassified))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn source(bodies: &[&str], structure: &str) -> ExportSource {
        let kids = (0..bodies.len()).map(|i| format!("{} 0 R", 4 + i * 2)).collect::<Vec<_>>().join(" ");
        let mut objects = vec![
            format!("<< /Type /Catalog /Pages 2 0 R {structure} >>"),
            format!("<< /Type /Pages /Kids [{kids}] /Count {} /MediaBox [0 0 200 300] >>", bodies.len()),
            "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>".into(),
        ];
        for (i, body) in bodies.iter().enumerate() {
            objects.push(format!("<< /Type /Page /Parent 2 0 R /Contents {} 0 R /Resources << /Font << /F1 3 0 R >> >> >>", 5 + i * 2));
            objects.push(format!("<< /Length {} >>\nstream\n{body}\nendstream", body.len()));
        }
        let mut bytes = b"%PDF-1.7\n".to_vec();
        let mut offsets = Vec::new();
        for (i, object) in objects.iter().enumerate() {
            offsets.push(bytes.len());
            bytes.extend_from_slice(format!("{} 0 obj\n{object}\nendobj\n", i + 1).as_bytes());
        }
        let xref = bytes.len();
        bytes.extend_from_slice(format!("xref\n0 {}\n0000000000 65535 f \n", objects.len() + 1).as_bytes());
        for offset in offsets {
            bytes.extend_from_slice(format!("{offset:010} 00000 n \n").as_bytes());
        }
        bytes.extend_from_slice(format!("trailer\n<< /Root 1 0 R /Size {} >>\nstartxref\n{xref}\n%%EOF\n", objects.len() + 1).as_bytes());
        ExportSource { bytes: std::sync::Arc::new(bytes), config: Default::default(), pages: bodies.len(), sizes: vec![(200.0, 300.0); bodies.len()] }
    }

    const MARKED: &str = "BT /F1 12 Tf 10 270 Td (Body words) Tj ET
        /Table BMC BT /F1 12 Tf 10 240 Td (Table words) Tj ET EMC
        /Code BMC BT /F1 12 Tf 10 210 Td (Code words) Tj ET EMC
        /Figure BMC BT /F1 12 Tf 10 180 Td (Figure words) Tj ET EMC
        BT /F1 12 Tf 3 Tr 10 150 Td (OCR words) Tj ET";

    #[test]
    fn word_count_filters_are_independent_and_do_not_change_normal_extraction() {
        let src = source(&[MARKED], "");
        for (options, expected) in [
            (Options::default(), 10),
            (Options { exclude_tables: true, ..Options::default() }, 8),
            (Options { exclude_code: true, ..Options::default() }, 8),
            (Options { exclude_images: true, ..Options::default() }, 6),
            (Options { exclude_tables: true, exclude_code: true, exclude_images: true }, 2),
        ] {
            assert_eq!(count(src.clone(), &[0], options, |_| true).unwrap().total.words, expected);
        }
        let mut renderer = PageRenderer::new(src.bytes, src.config);
        assert_eq!(Counts::of(&renderer.filtered_text(0, TextFilter::default()).unwrap().plain_text()).words, 10);
        assert_eq!(renderer.stats().rasterizations, 0);
    }

    #[test]
    fn word_count_resolves_structure_roles_and_page_local_mcids() {
        let body = "BT /F1 12 Tf 10 260 Td (Body words) Tj ET /Span << /MCID 0 >> BDC BT /F1 12 Tf 10 220 Td (Tagged words) Tj ET EMC";
        let src = source(
            &[body, body],
            "/StructTreeRoot << /RoleMap << /Grid /Table >> /K [<< /S /Grid /Pg 4 0 R /K << /S /Span /K 0 >> >> << /S /Code /Pg 6 0 R /K << /Type /MCR /MCID 0 >> >>] >>",
        );
        let tables = count(src.clone(), &[0, 1], Options { exclude_tables: true, ..Options::default() }, |_| true).unwrap();
        assert_eq!(tables.pages.iter().map(|(_, c)| c.words).collect::<Vec<_>>(), vec![2, 4]);
        assert!(!tables.unclassified_structure);
        let code = count(src, &[1, 0], Options { exclude_code: true, ..Options::default() }, |_| true).unwrap();
        assert_eq!(code.pages.iter().map(|(_, c)| c.words).collect::<Vec<_>>(), vec![2, 4]);
    }

    #[test]
    fn word_count_handles_empty_duplicate_invalid_and_cancelled_ranges() {
        let src = source(&["BT /F1 12 Tf 10 220 Td (Two words) Tj ET", ""], "");
        let report = count(src.clone(), &[1, 0, 1], Options::default(), |_| true).unwrap();
        assert_eq!(report.total.words, 2);
        assert_eq!(report.pages.len(), 2);
        assert_eq!(report.pages_without_text, 1);
        assert_eq!(count(src.clone(), &[], Options::default(), |_| true).unwrap().total, Counts::default());
        assert!(count(src.clone(), &[2], Options::default(), |_| true).unwrap_err().contains("bounds"));
        assert!(count(src.clone(), &[0, 1], Options::default(), |done| done < 1).unwrap_err().contains("cancelled"));
        let invalid = ExportSource { bytes: std::sync::Arc::new(b"not a PDF".to_vec()), ..src };
        assert!(count(invalid, &[0], Options::default(), |_| true).is_err());
    }

    #[test]
    fn word_count_actual_text_cannot_reintroduce_excluded_words() {
        let all = source(&["/Span << /ActualText (OCR words) >> BDC BT /F1 12 Tf 3 Tr 10 220 Td (xx) Tj ET EMC"], "");
        let images = Options { exclude_images: true, ..Options::default() };
        assert_eq!(count(all.clone(), &[0], Options::default(), |_| true).unwrap().total.words, 2);
        assert_eq!(count(all, &[0], images, |_| true).unwrap().total.words, 0);
        let mixed = source(
            &[
                "/Span << /ActualText (Body and code) >> BDC BT /F1 12 Tf 10 220 Td (body) Tj ET /Code BMC BT /F1 12 Tf 50 220 Td (code) Tj ET EMC EMC",
            ],
            "",
        );
        assert_eq!(count(mixed.clone(), &[0], Options::default(), |_| true).unwrap().total.words, 3);
        assert!(count(mixed, &[0], Options { exclude_code: true, ..Options::default() }, |_| true).unwrap_err().contains("replacement text"));
    }

    #[test]
    fn word_count_cyclic_structure_is_bounded_and_reported() {
        // The structure reaches the catalog again instead of a structure child.
        let src = source(&[MARKED], "/StructTreeRoot << /K << /S /Table /Pg 4 0 R /K [0 1 0 R] >> >> /K 1 0 R");
        let report = count(src, &[0], Options { exclude_tables: true, ..Options::default() }, |_| true).unwrap();
        assert!(report.unclassified_structure);
    }

    #[test]
    fn word_count_retains_form_namespaces_and_reports_the_limitation() {
        let body = "/Span << /MCID 0 >> BDC BT /F1 12 Tf 10 220 Td (Page words) Tj ET EMC /Fm Do";
        let mut src = source(&[body], "/StructTreeRoot << /K << /S /Table /Pg 4 0 R /K 0 >> >>");
        let mut doc = pdfcraft_cos::Document::open(src.bytes.clone()).unwrap();
        let page = ObjRef::new(4, 0);
        let mut resources = doc.get(page).as_dict().unwrap().get(b"Resources").unwrap().as_dict().unwrap().clone();
        let mut form = Dict::new();
        form.set(b"Type".to_vec(), Object::Name(b"XObject".to_vec()));
        form.set(b"Subtype".to_vec(), Object::Name(b"Form".to_vec()));
        form.set(b"BBox".to_vec(), Object::Array(vec![0.into(), 0.into(), 200.into(), 300.into()]));
        form.set(b"Resources".to_vec(), resources.clone());
        let form = doc.add(pdfcraft_cos::Stream::from_raw(form, b"/Span << /MCID 0 >> BDC BT /F1 12 Tf 10 180 Td (Form words) Tj ET EMC".to_vec()));
        let mut objects = Dict::new();
        objects.set(b"Fm".to_vec(), form);
        resources.set(b"XObject".to_vec(), objects);
        // Inherit resources from the page tree too.
        doc.update_dict(page, |d| {
            d.remove(b"Resources");
        })
        .unwrap();
        doc.update_dict(ObjRef::new(2, 0), |d| d.set(b"Resources".to_vec(), resources)).unwrap();
        src.bytes = std::sync::Arc::new(pdfcraft_cos::write_full(&doc, &Default::default()).unwrap());
        let report = count(src, &[0], Options { exclude_tables: true, ..Options::default() }, |_| true).unwrap();
        assert_eq!(report.total.words, 4, "the unrelated form's MCID must not be excluded");
        assert!(report.unclassified_structure);
    }

    fn sheets(text: &str) -> (usize, usize) {
        Counts::of(text).manuscript_sheets()
    }

    #[test]
    fn manuscript_sheets_are_400_squares_with_a_count_of_lines_on_the_last() {
        let a = |n: usize| "あ".repeat(n);
        assert_eq!(sheets(""), (0, 0));
        assert_eq!(sheets(" \n\n "), (0, 0));
        assert_eq!(sheets(&a(1)), (1, 1));
        assert_eq!(sheets(&a(20)), (1, 1));
        assert_eq!(sheets(&a(21)), (1, 2));
        assert_eq!(sheets(&a(400)), (1, 20), "exactly 400 squares fill one sheet");
        assert_eq!(sheets(&a(401)), (2, 1), "one more starts a second sheet");
        assert_eq!(sheets(&a(800)), (2, 20));
        // Pages add up their lines.
        let mut total = Counts::of(&a(390));
        total.add(Counts::of(&a(30)));
        assert_eq!(total.manuscript_lines, 20 + 2);
        assert_eq!(total.manuscript_sheets(), (2, 2));
    }

    #[test]
    fn manuscript_paragraphs_start_new_lines_without_invented_indentation() {
        let a = |n: usize| "あ".repeat(n);
        assert_eq!(manuscript_lines(&format!("{}\n{}", a(5), a(5))), 2);
        assert_eq!(manuscript_lines(&format!("{}\r\n{}\r{}", a(5), a(5), a(5))), 3);
        assert_eq!(manuscript_lines(&format!("{}\n\n{}", a(5), a(5))), 3, "an empty paragraph is an empty line");
        assert_eq!(manuscript_lines(&format!("\n\n{}\n\n", a(5))), 1, "breaks around the text take no line");
        // No indentation is invented: 20 characters fill one line.
        assert_eq!(manuscript_lines(&a(20)), 1);
        // A full-width space the text starts with is the empty first square.
        assert_eq!(manuscript_lines(&format!("\u{3000}{}", a(19))), 1);
        assert_eq!(manuscript_lines(&format!("\u{3000}{}", a(20))), 2);
        // Half-width spaces from text extraction take no square.
        assert_eq!(manuscript_lines(&format!("{} {}", a(10), a(10))), 1);
    }

    #[test]
    fn manuscript_punctuation_that_would_start_a_line_hangs() {
        let a = |n: usize| "あ".repeat(n);
        // ぶら下げ: the stop hangs at the end of the full line instead of taking a square.
        assert_eq!(manuscript_lines(&format!("{}。", a(20))), 1);
        assert_eq!(manuscript_lines(&format!("{}、{}", a(20), a(20))), 2);
        assert_eq!(sheets(&format!("{}。", a(400))), (1, 20), "a hung stop does not start a sheet");
        assert_eq!(manuscript_lines(&format!("{}。", a(19))), 1, "a stop inside the line takes a square");
        assert_eq!(manuscript_lines(&format!("{}。", a(20 + 19))), 2);
        // 追い出し: 」。 can't both hang, so い goes down with them: い」。 and 18 more fill 21 squares.
        assert_eq!(manuscript_lines(&format!("{}い」。{}", a(19), "う".repeat(18))), 3);
        // 行末禁則: an opening bracket never ends a line.
        assert_eq!(manuscript_lines(&format!("{}「{}", a(19), "う".repeat(19))), 2);
        assert_eq!(manuscript_lines(&format!("{}「{}", a(19), "う".repeat(20))), 3);
    }

    #[test]
    fn japanese_counts_leave_out_whitespace_and_punctuation() {
        let c = Counts::of("「今日は、晴れ。」\n　ＡＢＣ！ 123…★ 々ー〇");
        assert_eq!(c.characters_without_spaces, 21);
        // 「 、 。 」 ！ … ★ and the spaces go; 今日は晴れ ＡＢＣ 123 々ー〇 stay.
        assert_eq!(c.characters_without_punctuation, 14);
        for p in ['、', '。', '「', '」', '『', '』', '【', '】', '・', '〜', '！', '？', '（', '）', '…', '―', '※', '→', '○', '★', '｢', '.', ',']
        {
            assert!(is_punctuation_or_symbol(p), "{p}");
        }
        for l in ['あ', 'ア', '漢', '々', '〆', '〇', 'ー', 'a', 'Ａ', '1', '１', 'ｱ'] {
            assert!(!is_punctuation_or_symbol(l), "{l}");
        }
    }

    #[test]
    fn unicode_words_ignore_punctuation_and_keep_apostrophes() {
        assert_eq!(Counts::of("Hello, Welt! don't … 123").words, 4);
        assert_eq!(Counts::of("Привет мир").words, 2);
        assert_eq!(Counts::of("\t \n...").words, 0);
        assert_eq!(Counts::of("ä b").characters, 3);
        assert_eq!(Counts::of("ä b").characters_without_spaces, 2);
    }
}
