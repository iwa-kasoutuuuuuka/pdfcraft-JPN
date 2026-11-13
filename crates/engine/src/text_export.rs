//! Scan & OCR ▸ Export text: the text of a document (a recognised scan, or any PDF with a text
//! layer) as a plain text file or a Word document.
//!
//! The text is the page's own reading order — columns of vertical (縦書き) Japanese run right to
//! left — grouped into paragraphs ([`pdfcraft_render::TextParagraph`]). Pages without a text layer
//! can be read first ([`TextExportOptions::ocr_if_needed`]); that happens on a scratch copy, so the
//! document being exported is never changed. A Word export tags its text as Japanese (language and
//! East Asian font slot) and writes pages whose text is vertical as top-to-bottom sections.

use std::sync::Arc;

use pdfcraft_export::{TextDocxStyle, TextPage};
use pdfcraft_render::{PageRenderer, TextParagraph};

use crate::Session;
use crate::export::{ExportSource, Exporter};
use crate::ocr::{EngineChoice, OcrSettings, Recognizers, has_own_text};

/// The upper bound on the pages one export takes, so a hostile page count cannot make the output
/// unbounded.
pub const MAX_PAGES: usize = 20_000;

/// The file format of the export.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum TextFormat {
    /// Plain text, UTF-8.
    #[default]
    Txt,
    /// Word (.docx).
    Docx,
}

impl TextFormat {
    /// `txt` or `docx` (also `text`, `word`, `.txt`, `.docx`), case-insensitive.
    pub fn parse(s: &str) -> Option<TextFormat> {
        match s.trim().trim_start_matches('.').to_ascii_lowercase().as_str() {
            "txt" | "text" | "plain" => Some(TextFormat::Txt),
            "docx" | "word" => Some(TextFormat::Docx),
            _ => None,
        }
    }

    pub fn extension(self) -> &'static str {
        match self {
            TextFormat::Txt => "txt",
            TextFormat::Docx => "docx",
        }
    }
}

/// Whether the Word export writes vertical sections.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum VerticalMode {
    /// A page is vertical when most of its text is.
    #[default]
    Auto,
    On,
    Off,
}

impl VerticalMode {
    pub fn parse(s: &str) -> Option<VerticalMode> {
        match s.trim().to_ascii_lowercase().as_str() {
            "auto" => Some(VerticalMode::Auto),
            "on" | "true" | "yes" => Some(VerticalMode::On),
            "off" | "false" | "no" => Some(VerticalMode::Off),
            _ => None,
        }
    }
}

/// How the pages of a text file are told apart.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PageSeparator {
    /// A line `===== 3 / 12 =====` before each page (nothing for a one-page export).
    #[default]
    Line,
    /// A form feed between pages, as `pdftotext` writes them.
    FormFeed,
}

impl PageSeparator {
    pub fn parse(s: &str) -> Option<PageSeparator> {
        match s.trim().to_ascii_lowercase().as_str() {
            "line" | "header" => Some(PageSeparator::Line),
            "formfeed" | "form_feed" | "ff" => Some(PageSeparator::FormFeed),
            _ => None,
        }
    }
}

/// What to export and how.
#[derive(Clone, Debug)]
pub struct TextExportOptions {
    pub format: TextFormat,
    pub vertical: VerticalMode,
    pub separator: PageSeparator,
    /// Read the pages that have no text layer first (a scratch copy is read; the document is not
    /// changed). Needs the recognition engines; `ocr` says how.
    pub ocr_if_needed: bool,
    /// Recognition settings for [`Self::ocr_if_needed`]. Japanese and the automatic engine by default.
    pub ocr: OcrSettings,
    /// The Word document's title.
    pub title: String,
    /// The Word document's language (BCP 47). `None`: taken from `ocr.language`.
    pub language: Option<String>,
}

impl Default for TextExportOptions {
    fn default() -> Self {
        let ocr = OcrSettings {
            language: "ja".into(),
            engine: EngineChoice::Auto,
            strategy: EngineChoice::Auto.default_strategy(),
            ..OcrSettings::default()
        };
        TextExportOptions {
            format: TextFormat::Txt,
            vertical: VerticalMode::Auto,
            separator: PageSeparator::Line,
            ocr_if_needed: false,
            ocr,
            title: String::new(),
            language: None,
        }
    }
}

/// The result of an export.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TextExport {
    /// The file's bytes.
    pub bytes: Vec<u8>,
    /// How many pages it covers.
    pub pages: usize,
    /// The pages (1-based) that were read by OCR first.
    pub ocr_pages: Vec<usize>,
    /// The pages (1-based) written as vertical text.
    pub vertical_pages: Vec<usize>,
    /// The pages (1-based) with no text at all, after OCR when it ran.
    pub empty_pages: Vec<usize>,
    /// How many paragraphs were written.
    pub paragraphs: usize,
    pub warnings: Vec<String>,
}

/// The Word language tag of an OCR language code (`ja`, `ja-vert`, `zh`, `en`, …).
pub fn language_tag(code: &str) -> String {
    match code.split('-').next().unwrap_or_default() {
        "ja" => "ja-JP".into(),
        "zh" => "zh-CN".into(),
        "ko" => "ko-KR".into(),
        "en" => "en-US".into(),
        "" => "en-US".into(),
        other => other.to_string(),
    }
}

/// Whether a page's paragraphs are mostly vertical.
fn mostly_vertical(paragraphs: &[TextParagraph]) -> bool {
    let total: usize = paragraphs.iter().map(|p| p.glyphs).sum();
    let down: usize = paragraphs.iter().filter(|p| p.vertical).map(|p| p.glyphs).sum();
    total > 0 && down.saturating_mul(2) > total
}

/// The bytes of the scratch copy with its missing pages read (`None`: every page had text), and which pages (0-based) were read.
pub(crate) type ReadPages = (Option<Arc<Vec<u8>>>, Vec<usize>);

/// Read the pages `pages` (0-based) of `src` that have no text of their own, on a scratch copy,
/// and return the copy's bytes with the pages read.
pub(crate) fn read_missing(
    src: &ExportSource,
    name: &str,
    pages: &[usize],
    settings: &OcrSettings,
    engines: &dyn Fn(&OcrSettings) -> Result<Recognizers, String>,
    progress: &mut dyn FnMut(usize, usize) -> bool,
) -> Result<ReadPages, String> {
    let mut renderer = PageRenderer::new(src.bytes.clone(), src.config.clone());
    let need: Vec<usize> = pages.iter().copied().filter(|&p| !has_own_text(&mut renderer, p)).collect();
    if need.is_empty() {
        return Ok((None, need));
    }
    let recognizers = engines(settings)?;
    let mut session = Session::new();
    let password = src.config.password.as_deref();
    let id = session.open(name, None, src.bytes.clone(), password).map_err(|e| e.to_string())?;
    if let Some(why) = session.get(id).and_then(|d| d.read_only_reason.clone()) {
        return Err(why);
    }
    let mut settings = settings.clone();
    settings.skip_text_pages = true;
    settings.output_mode = crate::ocr::OutputMode::Searchable;
    let job = session.ocr_job(id, &need, settings).ok_or("the document could not be read")?;
    let found = job.run(&recognizers, progress);
    session.apply_ocr(id, &found).map_err(|e| e.to_string())?;
    let bytes = session.save_bytes(id).map_err(|e| e.to_string())?;
    Ok((Some(bytes), need))
}

/// Export the text of pages `pages` (0-based, in the order given) of `src`.
///
/// `progress(done, total)` is called while OCR runs; returning `false` there stops the reading and
/// exports what was read. An engine that cannot run is an error when OCR is asked for and a page
/// needs it.
pub fn export_text(
    src: ExportSource,
    name: &str,
    pages: &[usize],
    options: &TextExportOptions,
    progress: impl FnMut(usize, usize) -> bool,
) -> Result<TextExport, String> {
    export_text_with(src, name, pages, options, &crate::ocr::recognizers, progress)
}

/// [`export_text`] with the recognition engines supplied by `engines` (tests use a fake one).
pub(crate) fn export_text_with(
    src: ExportSource,
    name: &str,
    pages: &[usize],
    options: &TextExportOptions,
    engines: &dyn Fn(&OcrSettings) -> Result<Recognizers, String>,
    mut progress: impl FnMut(usize, usize) -> bool,
) -> Result<TextExport, String> {
    if pages.is_empty() {
        return Err("no pages to export".into());
    }
    if pages.len() > MAX_PAGES {
        return Err(format!("at most {MAX_PAGES} pages can be exported at once"));
    }
    if let Some(bad) = pages.iter().find(|&&p| p >= src.pages) {
        return Err(format!("page {} does not exist", bad.saturating_add(1)));
    }
    let mut out = TextExport { pages: pages.len(), ..Default::default() };
    let mut source = src;
    if options.ocr_if_needed {
        let (bytes, read) = read_missing(&source, name, pages, &options.ocr, engines, &mut progress)?;
        if let Some(bytes) = bytes {
            source.bytes = bytes;
        }
        out.ocr_pages = read.iter().map(|p| p + 1).collect();
    }
    let sizes = source.sizes.clone();
    let total_pages = source.pages;
    let mut exporter = Exporter::from_source(source);
    let mut docx_pages: Vec<TextPage> = Vec::new();
    let mut text = String::new();
    for (k, &page) in pages.iter().enumerate() {
        let paragraphs = exporter.paragraphs(page)?;
        let vertical = match options.vertical {
            VerticalMode::Auto => mostly_vertical(&paragraphs),
            VerticalMode::On => true,
            VerticalMode::Off => false,
        };
        if paragraphs.is_empty() {
            out.empty_pages.push(page + 1);
        }
        if vertical && !paragraphs.is_empty() {
            out.vertical_pages.push(page + 1);
        }
        out.paragraphs = out.paragraphs.saturating_add(paragraphs.len());
        let body: Vec<String> = paragraphs.into_iter().map(|p| p.text).collect();
        match options.format {
            TextFormat::Txt => {
                if k > 0 && options.separator == PageSeparator::FormFeed {
                    text.push('\u{c}');
                }
                if options.separator == PageSeparator::Line && pages.len() > 1 {
                    if k > 0 {
                        text.push('\n');
                    }
                    text.push_str(&format!("===== {} / {total_pages} =====\n\n", page + 1));
                }
                text.push_str(&body.join("\n\n"));
                text.push('\n');
            }
            TextFormat::Docx => {
                let (width, height) = sizes.get(page).map_or((595.0, 842.0), |&(w, h)| (f64::from(w), f64::from(h)));
                docx_pages.push(TextPage { width, height, paragraphs: body, vertical: vertical && options.vertical != VerticalMode::Off });
            }
        }
    }
    if !out.empty_pages.is_empty() {
        let list = out.empty_pages.iter().map(ToString::to_string).collect::<Vec<_>>().join(", ");
        let hint = if options.ocr_if_needed { "" } else { " (turn on OCR to read scanned pages)" };
        out.warnings.push(format!("no text on page(s) {list}{hint}"));
    }
    out.bytes = match options.format {
        TextFormat::Txt => text.into_bytes(),
        TextFormat::Docx => {
            let tag = options.language.clone().unwrap_or_else(|| language_tag(&options.ocr.language));
            let style = TextDocxStyle { language: tag, ..TextDocxStyle::default() };
            pdfcraft_export::text_docx(&docx_pages, &options.title, &style)
        }
    };
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Edit, Session};
    use pdfcraft_ocr::PlacedWord;

    fn word(text: &str, origin: [f64; 2], across: [f64; 2], up: [f64; 2]) -> PlacedWord {
        PlacedWord { text: text.into(), origin, across, up, confidence: None, source: "test".into() }
    }

    /// `n` blank 200 x 300 pt pages.
    fn blank(n: usize) -> Vec<u8> {
        let kids: Vec<String> = (0..n).map(|i| format!("{} 0 R", 3 + i)).collect();
        let mut objs = vec![
            "<< /Type /Catalog /Pages 2 0 R >>".to_string(),
            format!("<< /Type /Pages /Kids [{}] /Count {n} /MediaBox [0 0 200 300] >>", kids.join(" ")),
        ];
        objs.extend((0..n).map(|_| "<< /Type /Page /Parent 2 0 R >>".to_string()));
        let mut out = b"%PDF-1.7
"
        .to_vec();
        let mut offsets = Vec::new();
        for (i, o) in objs.iter().enumerate() {
            offsets.push(out.len());
            out.extend_from_slice(
                format!(
                    "{} 0 obj
{o}
endobj
",
                    i + 1
                )
                .as_bytes(),
            );
        }
        let xref = out.len();
        out.extend_from_slice(
            format!(
                "xref
0 {}
0000000000 65535 f 
",
                objs.len() + 1
            )
            .as_bytes(),
        );
        for o in offsets {
            out.extend_from_slice(
                format!(
                    "{o:010} 00000 n 
"
                )
                .as_bytes(),
            );
        }
        out.extend_from_slice(
            format!(
                "trailer
<< /Size {} /Root 1 0 R >>
startxref
{xref}
%%EOF
",
                objs.len() + 1
            )
            .as_bytes(),
        );
        out
    }

    /// A document of `pages` 200 x 300 pages, each with `words` (an invisible, fontless text layer,
    /// which is how recognised Japanese is written).
    pub(crate) fn doc_with(words: &[Vec<PlacedWord>]) -> ExportSource {
        let mut s = Session::new().with_clock(|| 1_700_000_000);
        let id = s.open("t.pdf", None, Arc::new(blank(words.len())), None).unwrap();
        for (page, w) in words.iter().enumerate() {
            s.apply(id, Edit::AddOcrText { page, words: w.clone() }).unwrap();
        }
        let bytes = s.save_bytes(id).unwrap();
        let mut s2 = Session::new();
        let id2 = s2.open("t.pdf", None, bytes, None).unwrap();
        s2.get(id2).unwrap().export_source()
    }

    /// A horizontal line of `text` 12 pt high at `y` (PDF coordinates, from the bottom).
    fn row(text: &str, y: f64) -> PlacedWord {
        word(text, [20.0, y], [12.0 * text.chars().count() as f64, 0.0], [0.0, 12.0])
    }

    /// A vertical column of `text` whose left edge is at `x`, from the top at 250.
    fn column(text: &str, x: f64) -> PlacedWord {
        let h = 12.0 * text.chars().count() as f64;
        word(text, [x, 250.0 - h], [12.0, 0.0], [0.0, h])
    }

    fn text_of(src: ExportSource, options: &TextExportOptions) -> TextExport {
        let pages: Vec<usize> = (0..src.pages).collect();
        export_text(src, "t.pdf", &pages, options, |_, _| true).unwrap()
    }

    #[test]
    fn plain_text_keeps_page_order_and_paragraphs() {
        let src = doc_with(&[vec![row("最初の段落です", 250.0), row("続きの行です", 237.0), row("別の段落", 200.0)], vec![row("二ページ目", 250.0)]]);
        let out = text_of(src, &TextExportOptions::default());
        let text = String::from_utf8(out.bytes.clone()).unwrap();
        let find = |s: &str| text.find(s).unwrap_or_else(|| panic!("{s} missing in {text:?}"));
        assert!(find("===== 1 / 2 =====") < find("最初の段落です続きの行です"));
        assert!(find("最初の段落です続きの行です") < find("別の段落"));
        assert!(find("別の段落") < find("===== 2 / 2 ====="));
        assert!(find("===== 2 / 2 =====") < find("二ページ目"));
        assert!(text.contains("最初の段落です続きの行です\n\n別の段落"), "{text:?}");
        assert_eq!(out.pages, 2);
        assert!(out.vertical_pages.is_empty() && out.empty_pages.is_empty(), "{out:?}");
    }

    #[test]
    fn form_feed_separators_and_a_single_page_have_no_header() {
        let src = doc_with(&[vec![row("一", 250.0)], vec![row("二", 250.0)]]);
        let options = TextExportOptions { separator: PageSeparator::FormFeed, ..Default::default() };
        let out = text_of(src, &options);
        assert_eq!(String::from_utf8(out.bytes).unwrap(), "一\n\u{c}二\n");
        let src = doc_with(&[vec![row("一", 250.0)]]);
        assert_eq!(String::from_utf8(text_of(src, &TextExportOptions::default()).bytes).unwrap(), "一\n");
    }

    #[test]
    fn vertical_columns_read_right_to_left_and_a_vertical_page_is_noticed() {
        let src = doc_with(&[vec![column("右の列です", 150.0), column("左の列です", 132.0)]]);
        let out = text_of(src, &TextExportOptions::default());
        assert_eq!(String::from_utf8(out.bytes.clone()).unwrap(), "右の列です左の列です\n");
        assert_eq!(out.vertical_pages, [1]);
    }

    /// The Word package itself is checked in `pdfcraft-export` and end to end through the tool;
    /// here, that the modes pick the pages they should.
    #[test]
    fn docx_modes_pick_the_vertical_pages() {
        let src = doc_with(&[vec![column("右の列です", 150.0), column("左の列です", 132.0)], vec![row("横書きの頁", 250.0)]]);
        let options = TextExportOptions { format: TextFormat::Docx, title: "題".into(), ..Default::default() };
        let out = text_of(src.clone(), &options);
        assert!(out.bytes.starts_with(b"PK"));
        assert_eq!(out.vertical_pages, [1]);
        let off = text_of(src.clone(), &TextExportOptions { vertical: VerticalMode::Off, ..options.clone() });
        assert!(off.vertical_pages.is_empty());
        let on = text_of(src, &TextExportOptions { vertical: VerticalMode::On, ..options });
        assert_eq!(on.vertical_pages, [1, 2]);
    }

    #[test]
    fn pages_without_text_are_listed_and_ocr_is_refused_without_an_engine() {
        let src = doc_with(&[vec![row("あり", 250.0)], vec![]]);
        let out = text_of(src.clone(), &TextExportOptions::default());
        assert_eq!(out.empty_pages, [2]);
        assert!(out.warnings.iter().any(|w| w.contains("page(s) 2")), "{out:?}");
        // A page with no text layer and an engine that cannot run: an honest error, not an empty page.
        let options = TextExportOptions { ocr_if_needed: true, ..Default::default() };
        let broken = |_: &OcrSettings| -> Result<Recognizers, String> { Err("no engine installed".into()) };
        let err = export_text_with(src, "t.pdf", &[0, 1], &options, &broken, |_, _| true).unwrap_err();
        assert!(err.contains("no engine installed"), "{err}");
        // Every page already has text: no engine is needed, nothing is read.
        let full = doc_with(&[vec![row("あり", 250.0)]]);
        let out = export_text_with(full, "t.pdf", &[0], &options, &broken, |_, _| true).unwrap();
        assert!(out.ocr_pages.is_empty());
    }

    /// An engine that reads "日本語" on every page; no models needed.
    struct Fake;

    impl pdfcraft_ocr::Recognizer for Fake {
        fn id(&self) -> &'static str {
            "fake"
        }
        fn available(&self) -> bool {
            true
        }
        fn languages(&self) -> Vec<(String, String)> {
            vec![("ja".into(), "Japanese".into())]
        }
        fn recognize(
            &self,
            _image: &pdfcraft_ocr::OcrImage,
            _options: &pdfcraft_ocr::RecognizeOptions,
        ) -> Result<Vec<pdfcraft_ocr::Line>, pdfcraft_ocr::OcrError> {
            Ok(vec![pdfcraft_ocr::Line { words: vec![pdfcraft_ocr::Word::new("日本語", [10.0, 10.0, 60.0, 30.0], "fake")] }])
        }
    }

    fn fake(_: &OcrSettings) -> Result<Recognizers, String> {
        Ok(Recognizers { primary: Arc::new(Fake), secondary: None })
    }

    #[test]
    fn scanned_pages_are_read_on_a_copy_when_asked_and_the_document_is_left_alone() {
        let mut s = Session::new();
        let text = s.create_from_text("t", "A picture of words").unwrap();
        let id = s.open("text.pdf", None, text, None).unwrap();
        let png = crate::export::Exporter::new(s.get(id).unwrap()).png(0, 100.0).unwrap();
        let scan = s.create_from_images(&[("scan.png".into(), png)]).unwrap();
        let mut s2 = Session::new();
        let id2 = s2.open("scan.pdf", None, scan.clone(), None).unwrap();
        let src = s2.get(id2).unwrap().export_source();
        let original = src.bytes.clone();
        let options = TextExportOptions { ocr_if_needed: true, ..Default::default() };
        let mut progressed = false;
        let out = export_text_with(src.clone(), "scan.pdf", &[0], &options, &fake, |_, _| {
            progressed = true;
            true
        })
        .unwrap();
        assert!(progressed);
        assert_eq!(out.ocr_pages, [1]);
        assert!(String::from_utf8(out.bytes).unwrap().contains("日本語"));
        assert!(out.empty_pages.is_empty(), "{:?}", out.empty_pages);
        assert_eq!(*src.bytes, *original, "the document's bytes are not changed");
        // Without the option the page stays empty.
        let plain = export_text_with(src, "scan.pdf", &[0], &TextExportOptions::default(), &fake, |_, _| true).unwrap();
        assert_eq!(plain.empty_pages, [1]);
    }

    #[test]
    fn bad_page_lists_are_refused() {
        let src = doc_with(&[vec![row("あ", 250.0)]]);
        assert!(export_text(src.clone(), "t.pdf", &[], &TextExportOptions::default(), |_, _| true).is_err());
        assert!(export_text(src, "t.pdf", &[5], &TextExportOptions::default(), |_, _| true).unwrap_err().contains("page 6"));
    }

    #[test]
    fn formats_vertical_modes_and_languages_parse() {
        assert_eq!(TextFormat::parse(".DOCX"), Some(TextFormat::Docx));
        assert_eq!(TextFormat::parse("txt"), Some(TextFormat::Txt));
        assert_eq!(TextFormat::parse("pdf"), None);
        assert_eq!(VerticalMode::parse("Auto"), Some(VerticalMode::Auto));
        assert_eq!(VerticalMode::parse("maybe"), None);
        assert_eq!(language_tag("ja-vert"), "ja-JP");
        assert_eq!(language_tag("en"), "en-US");
    }
}
