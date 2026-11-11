//! The Windows built-in OCR engine (`Windows.Media.Ocr`): no install, reads whatever OCR
//! language packs Windows has (Japanese included, horizontal and vertical).
//!
//! Windows OCR picks ONE writing direction per image, so the page is first cut into
//! whitespace-separated blocks ([`crate::blocks`]) and each block is read on its own. Its
//! words are then regrouped for the invisible text layer: the per-character words of a
//! vertical line become one word (the layer writes a tall box as a column), and consecutive
//! CJK words of a horizontal line become one word per run. On other platforms the engine
//! exists but reports itself unavailable.

use crate::blocks::{BlockRect, Node, read_blocks, segment};
use crate::{LANGUAGES, Line, OcrError, OcrImage, RecognizeOptions, Recognizer, Word, is_cjk};

/// The engine's id: what [`Word::source`], settings and tools name it by.
pub const ID: &str = "windows";

/// One recognised word from the OS: text and box, in the coordinates of the bitmap it read.
type RawWord = (String, [f32; 4]);

/// The primary language subtag Windows OCR uses for a PdfCraft language code.
pub fn primary_tag(code: &str) -> &'static str {
    match code {
        "ja" | "ja-vert" => "ja",
        "zh" => "zh",
        "ko" => "ko",
        "de" => "de",
        "fr" => "fr",
        "es" => "es",
        "it" => "it",
        "pt" => "pt",
        "ru" => "ru",
        "ar" => "ar",
        "nl" => "nl",
        _ => "en",
    }
}

/// The Windows OCR engine.
#[derive(Clone, Copy, Debug, Default)]
pub struct WindowsOcr;

impl WindowsOcr {
    pub fn new() -> WindowsOcr {
        WindowsOcr
    }

    /// The engine when Windows OCR can run here (Windows, with at least one OCR language).
    pub fn find() -> Option<WindowsOcr> {
        WindowsOcr.available().then_some(WindowsOcr)
    }

    fn read_block(&self, engine: &sys::Engine, image: &OcrImage, r: &BlockRect) -> Result<(Vec<Line>, bool), OcrError> {
        // A margin of blank paper around the block (Windows OCR drops text that touches the
        // border), and at least 64 px on a side.
        let pad = (crate::blocks::default_gap(image.width(), image.height()) / 2).max(8);
        let x0 = r.x0.saturating_sub(pad);
        let y0 = r.y0.saturating_sub(pad);
        let x1 = (r.x1 + pad).min(image.width());
        let y1 = (r.y1 + pad).min(image.height());
        let (cw, ch) = ((x1 - x0).max(64), (y1 - y0).max(64));
        let mut bgra = vec![255u8; (cw as usize) * (ch as usize) * 4];
        let stride = image.width() as usize * 4;
        for y in y0..y1 {
            for x in x0..x1 {
                let s = y as usize * stride + x as usize * 4;
                let d = ((y - y0) as usize * cw as usize + (x - x0) as usize) * 4;
                if let (Some(src), Some(dst)) = (image.rgba().get(s..s + 4), bgra.get_mut(d..d + 4)) {
                    let a = src[3] as u32;
                    // Over white paper, so transparent pixels are blank, not black.
                    let over = |c: u8| ((c as u32 * a + 255 * (255 - a)) / 255) as u8;
                    dst.copy_from_slice(&[over(src[2]), over(src[1]), over(src[0]), 255]);
                }
            }
        }
        let raw = engine.read(cw, ch, &bgra)?;
        let lines: Vec<Line> = raw.iter().map(|l| regroup(l, x0 as f32, y0 as f32)).filter(|l| !l.words.is_empty()).collect();
        let vertical = lines.iter().filter(|l| is_vertical_line(l)).count() * 2 > lines.len();
        Ok((lines, vertical))
    }
}

impl Recognizer for WindowsOcr {
    fn id(&self) -> &'static str {
        ID
    }

    fn available(&self) -> bool {
        !sys::available_tags().is_empty()
    }

    fn languages(&self) -> Vec<(String, String)> {
        let tags = sys::available_tags();
        LANGUAGES.iter().filter(|(code, _)| tags.iter().any(|t| t == primary_tag(code))).map(|(c, n)| ((*c).to_string(), (*n).to_string())).collect()
    }

    fn recognize(&self, image: &OcrImage, options: &RecognizeOptions) -> Result<Vec<Line>, OcrError> {
        let engine = sys::Engine::new(primary_tag(&options.language))?;
        let Some(tree) = segment(image.rgba(), image.width(), image.height()) else {
            return Ok(Vec::new());
        };
        // One block: read the page as it is (the cut only trimmed its margins). So does a page
        // that falls into very many blocks (noise, a crafted image): each block is one OS call.
        let whole = Node::Leaf(BlockRect { x0: 0, y0: 0, x1: image.width(), y1: image.height() });
        let tree = match tree {
            Node::Leaf(_) => whole,
            other if other.leaves().len() > MAX_BLOCKS => whole,
            other => other,
        };
        let (lines, _) = read_blocks(&tree, &mut |r| self.read_block(&engine, image, r))?;
        Ok(lines)
    }
}

/// The most blocks a page is read in; past it the page is read whole.
const MAX_BLOCKS: usize = 64;

/// Whether `line` is one vertical column word, as the text layer will write it.
fn is_vertical_line(line: &Line) -> bool {
    match line.words.as_slice() {
        [w] => {
            let (bw, bh) = (w.rect[2] - w.rect[0], w.rect[3] - w.rect[1]);
            w.text.chars().count() >= 2 && bh > 1.5 * bw
        }
        _ => false,
    }
}

fn union(rects: impl Iterator<Item = [f32; 4]>) -> Option<[f32; 4]> {
    rects.reduce(|a, b| [a[0].min(b[0]), a[1].min(b[1]), a[2].max(b[2]), a[3].max(b[3])])
}

/// Whether the words of a line form a column: the box is taller than wide and the words'
/// centres step down the page.
fn is_vertical_run(words: &[RawWord]) -> bool {
    let Some(u) = union(words.iter().map(|w| w.1)) else { return false };
    let stacked = words.windows(2).all(|p| (p[0].1[1] + p[0].1[3]) <= (p[1].1[1] + p[1].1[3]));
    words.len() >= 2 && (u[3] - u[1]) > (u[2] - u[0]) && stacked
}

/// Turn one OS line into a [`Line`] (boxes shifted by the crop origin `ox`, `oy`): a vertical
/// line becomes one word; a horizontal one gets one word per run of CJK words, Latin words
/// staying apart.
fn regroup(words: &[RawWord], ox: f32, oy: f32) -> Line {
    let shift = |r: [f32; 4]| [r[0] + ox, r[1] + oy, r[2] + ox, r[3] + oy];
    let words: Vec<RawWord> = words.iter().filter(|w| !w.0.trim().is_empty()).cloned().collect();
    let mut out: Vec<Word> = Vec::new();
    if is_vertical_run(&words) {
        let text: String = words.iter().flat_map(|w| w.0.chars()).filter(|c| !c.is_whitespace()).collect();
        if let Some(u) = union(words.iter().map(|w| w.1)) {
            out.push(Word::new(text, shift(u), ID));
        }
        return Line { words: out };
    }
    let mut run: Option<(String, [f32; 4])> = None;
    for (text, rect) in &words {
        let cjk = text.chars().all(is_cjk);
        match (&mut run, cjk) {
            (Some((t, r)), true) => {
                t.push_str(text);
                *r = [r[0].min(rect[0]), r[1].min(rect[1]), r[2].max(rect[2]), r[3].max(rect[3])];
            }
            (_, true) => run = Some((text.clone(), *rect)),
            (_, false) => {
                if let Some((t, r)) = run.take() {
                    out.push(Word::new(t, shift(r), ID));
                }
                out.push(Word::new(text.clone(), shift(*rect), ID));
            }
        }
    }
    if let Some((t, r)) = run {
        out.push(Word::new(t, shift(r), ID));
    }
    Line { words: out }
}

#[cfg(windows)]
mod sys {
    use super::RawWord;
    use crate::OcrError;
    use windows::Graphics::Imaging::{BitmapPixelFormat, SoftwareBitmap};
    use windows::Media::Ocr::OcrEngine;
    use windows::Storage::Streams::DataWriter;

    /// The longest one block's recognition may take (the OS normally needs well under a second).
    const RECOGNIZE_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(120);

    fn fail(e: impl std::fmt::Display) -> OcrError {
        OcrError::Recognize(format!("Windows OCR: {e}"))
    }

    /// The primary subtags of the OCR languages Windows has (`ja`, `en`, …).
    pub fn available_tags() -> Vec<String> {
        let Ok(langs) = OcrEngine::AvailableRecognizerLanguages() else { return Vec::new() };
        langs.into_iter().filter_map(|l| l.LanguageTag().ok()).map(|t| t.to_string().split('-').next().unwrap_or_default().to_string()).collect()
    }

    pub struct Engine(OcrEngine);

    impl Engine {
        pub fn new(tag: &str) -> Result<Engine, OcrError> {
            let langs = OcrEngine::AvailableRecognizerLanguages().map_err(fail)?;
            let lang = langs
                .into_iter()
                .find(|l| l.LanguageTag().map(|t| t.to_string().split('-').next().unwrap_or_default() == tag).unwrap_or(false))
                .ok_or_else(|| fail(format!("no OCR language pack for \"{tag}\" is installed (add it in Windows Settings, Time & language)")))?;
            OcrEngine::TryCreateFromLanguage(&lang).map(Engine).map_err(fail)
        }

        /// Read a Bgra8 bitmap; lines of words with boxes in bitmap pixels.
        pub fn read(&self, w: u32, h: u32, bgra: &[u8]) -> Result<Vec<Vec<RawWord>>, OcrError> {
            let max = OcrEngine::MaxImageDimension().map_err(fail)?;
            if w > max || h > max {
                return Err(OcrError::ImageTooLarge);
            }
            let writer = DataWriter::new().map_err(fail)?;
            writer.WriteBytes(bgra).map_err(fail)?;
            let buffer = writer.DetachBuffer().map_err(fail)?;
            let bitmap = SoftwareBitmap::CreateCopyFromBuffer(&buffer, BitmapPixelFormat::Bgra8, w as i32, h as i32).map_err(fail)?;
            // pdfcraft-JPN: wait with a deadline, so a recognition the OS never finishes can't
            // hang the OCR worker (AsyncStatus 0 is Started).
            let op = self.0.RecognizeAsync(&bitmap).map_err(fail)?;
            let deadline = std::time::Instant::now() + RECOGNIZE_TIMEOUT;
            while op.Status().map_err(fail)?.0 == 0 {
                if std::time::Instant::now() >= deadline {
                    let _ = op.Cancel();
                    return Err(OcrError::Recognize(format!("Windows OCR did not finish within {} seconds", RECOGNIZE_TIMEOUT.as_secs())));
                }
                std::thread::sleep(std::time::Duration::from_millis(5));
            }
            let result = op.GetResults().map_err(fail)?;
            let mut out = Vec::new();
            for line in result.Lines().map_err(fail)? {
                let mut words = Vec::new();
                for word in line.Words().map_err(fail)? {
                    let r = word.BoundingRect().map_err(fail)?;
                    words.push((word.Text().map_err(fail)?.to_string(), [r.X, r.Y, r.X + r.Width, r.Y + r.Height]));
                }
                out.push(words);
            }
            Ok(out)
        }
    }
}

#[cfg(not(windows))]
mod sys {
    use super::RawWord;
    use crate::OcrError;

    pub fn available_tags() -> Vec<String> {
        Vec::new()
    }

    pub struct Engine;

    impl Engine {
        pub fn new(_tag: &str) -> Result<Engine, OcrError> {
            Err(OcrError::Recognize("Windows OCR is only available on Windows".into()))
        }

        pub fn read(&self, _w: u32, _h: u32, _bgra: &[u8]) -> Result<Vec<Vec<RawWord>>, OcrError> {
            Err(OcrError::Recognize("Windows OCR is only available on Windows".into()))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn w(t: &str, r: [f32; 4]) -> RawWord {
        (t.to_string(), r)
    }

    #[test]
    fn vertical_line_becomes_one_column_word() {
        let words = vec![w("吾", [100.0, 10.0, 130.0, 40.0]), w("輩", [100.0, 42.0, 130.0, 72.0]), w("は", [101.0, 74.0, 129.0, 104.0])];
        let line = regroup(&words, 5.0, 7.0);
        assert_eq!(line.words.len(), 1);
        assert_eq!(line.words[0].text, "吾輩は");
        assert_eq!(line.words[0].rect, [105.0, 17.0, 135.0, 111.0]);
        assert!(is_vertical_line(&line));
    }

    #[test]
    fn horizontal_cjk_runs_merge_and_latin_stays_apart() {
        let words = vec![
            w("日", [0.0, 0.0, 20.0, 20.0]),
            w("本語", [20.0, 0.0, 60.0, 20.0]),
            w("PdfCraft", [70.0, 0.0, 150.0, 20.0]),
            w("版", [160.0, 0.0, 180.0, 20.0]),
        ];
        let line = regroup(&words, 0.0, 0.0);
        let texts: Vec<_> = line.words.iter().map(|x| x.text.as_str()).collect();
        assert_eq!(texts, ["日本語", "PdfCraft", "版"]);
        assert_eq!(line.words[0].rect, [0.0, 0.0, 60.0, 20.0]);
        assert!(!is_vertical_line(&line));
    }

    #[test]
    fn single_character_and_empty_lines_are_not_columns() {
        let line = regroup(&[w("あ", [0.0, 0.0, 10.0, 30.0])], 0.0, 0.0);
        assert_eq!(line.words.len(), 1);
        assert!(!is_vertical_line(&line));
        assert!(regroup(&[w(" ", [0.0; 4])], 0.0, 0.0).words.is_empty());
    }

    #[test]
    fn language_codes_map_to_windows_tags() {
        assert_eq!(primary_tag("ja-vert"), "ja");
        assert_eq!(primary_tag("zh"), "zh");
        assert_eq!(primary_tag("en"), "en");
        assert_eq!(primary_tag("nope"), "en");
    }

    #[cfg(not(windows))]
    #[test]
    fn unavailable_off_windows() {
        assert!(!WindowsOcr.available());
        assert!(WindowsOcr.languages().is_empty());
    }
}
