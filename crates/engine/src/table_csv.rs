//! Scan & OCR ▸ Export tables to CSV: the tables of a document's pages, found from where its text
//! sits (a recognised scan's invisible text included), as CSV files that Excel opens correctly in
//! Japanese: UTF-8 with a byte order mark, CRLF line ends and RFC 4180 quoting.
//!
//! The text is the page's positioned glyphs ([`pdfcraft_render::PageText`]) gathered into words;
//! the page's ruled lines ([`pdfcraft_edit::page_rules`]) say where a drawn grid's cells are.
//! [`detect`] finds the tables. Pages without a text layer can be read first
//! ([`TableCsvOptions::ocr_if_needed`]), on a scratch copy as the text export does, so the
//! document is never changed.

pub mod detect;

use pdfcraft_render::PageText;

use crate::export::{ExportSource, Exporter};
use crate::ocr::{EngineChoice, OcrSettings, Recognizers};
use crate::text_export::{MAX_PAGES, read_missing};

pub use detect::{Table, Word};

/// How the tables are written.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CsvLayout {
    /// One CSV file for each table.
    #[default]
    PerTable,
    /// Every table in one file, a blank line between two tables.
    OneFile,
}

impl CsvLayout {
    /// `per_table` or `one_file` (also `separate`, `single`, `combined`), case-insensitive.
    pub fn parse(s: &str) -> Option<CsvLayout> {
        match s.trim().to_ascii_lowercase().replace('-', "_").as_str() {
            "per_table" | "separate" | "each" => Some(CsvLayout::PerTable),
            "one_file" | "single" | "combined" | "one" => Some(CsvLayout::OneFile),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            CsvLayout::PerTable => "per_table",
            CsvLayout::OneFile => "one_file",
        }
    }
}

/// What to export and how.
#[derive(Clone, Debug)]
pub struct TableCsvOptions {
    pub layout: CsvLayout,
    /// Read the pages that have no text layer first (a scratch copy is read; the document is not
    /// changed). Needs the recognition engines; `ocr` says how.
    pub ocr_if_needed: bool,
    /// Recognition settings for [`Self::ocr_if_needed`]. Japanese and the automatic engine by default.
    pub ocr: OcrSettings,
    /// Use the page's ruled lines to find the cells of drawn grids (on by default).
    pub use_rules: bool,
}

impl Default for TableCsvOptions {
    fn default() -> Self {
        let ocr = OcrSettings {
            language: "ja".into(),
            engine: EngineChoice::Auto,
            strategy: EngineChoice::Auto.default_strategy(),
            ..OcrSettings::default()
        };
        TableCsvOptions { layout: CsvLayout::PerTable, ocr_if_needed: false, ocr, use_rules: true }
    }
}

/// One table that was found.
#[derive(Clone, Debug, PartialEq)]
pub struct FoundTable {
    /// The page (1-based).
    pub page: usize,
    /// Its place among the tables of that page (1-based, top to bottom).
    pub index: usize,
    pub rows: usize,
    pub columns: usize,
    /// It came from ruled lines rather than from the gaps between words.
    pub ruled: bool,
}

/// The result of an export.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TableCsvExport {
    /// The tables, in page order.
    pub tables: Vec<FoundTable>,
    /// The CSV files: one per table ([`CsvLayout::PerTable`], in the order of `tables`), or a
    /// single one ([`CsvLayout::OneFile`]). Empty when no table was found.
    pub files: Vec<Vec<u8>>,
    /// How many pages were looked at.
    pub pages: usize,
    /// The pages (1-based) that were read by OCR first.
    pub ocr_pages: Vec<usize>,
    /// The pages (1-based) with no text at all, after OCR when it ran.
    pub empty_pages: Vec<usize>,
    pub warnings: Vec<String>,
}

/// The name of the file of a table in [`CsvLayout::PerTable`]: `<stem>_p<page>_t<index>.csv`.
pub fn table_file_name(stem: &str, table: &FoundTable) -> String {
    format!("{stem}_p{}_t{}.csv", table.page, table.index)
}

/// One CSV field, quoted as RFC 4180 asks when it holds a comma, a quote or a line break (quotes
/// doubled). A field that a spreadsheet would run as a formula (it starts with `=`, `@`, a tab or
/// a carriage return, or with `+` or `-` and is not a number or a lone sign) starts with `'`, so a hostile PDF
/// cannot put a formula into the user's spreadsheet.
pub fn csv_field(s: &str) -> String {
    let numeric = |t: &str| !t.is_empty() && t.chars().all(|c| c.is_ascii_digit() || matches!(c, '.' | ',' | '%' | ' '));
    let risky = match s.chars().next() {
        Some('=' | '@' | '\t' | '\r') => true,
        Some('+' | '-') => s.len() > 1 && !numeric(s.get(1..).unwrap_or_default()),
        _ => false,
    };
    let s = if risky { format!("'{s}") } else { s.to_string() };
    if s.contains([',', '"', '\n', '\r']) { format!("\"{}\"", s.replace('"', "\"\"")) } else { s }
}

/// `tables` as one CSV file: a UTF-8 byte order mark, CRLF after every row, every row as wide as
/// its table, and a blank line between two tables.
pub fn csv_bytes<'a>(tables: impl IntoIterator<Item = &'a Table>) -> Vec<u8> {
    let mut out = String::from("\u{feff}");
    for (k, t) in tables.into_iter().enumerate() {
        if k > 0 {
            out.push_str("\r\n");
        }
        let width = t.rows.iter().map(Vec::len).max().unwrap_or(0);
        for row in &t.rows {
            let fields: Vec<String> = (0..width).map(|c| row.get(c).map_or_else(String::new, |s| csv_field(s))).collect();
            out.push_str(&fields.join(","));
            out.push_str("\r\n");
        }
    }
    out.into_bytes()
}

/// Whether a glyph runs down the page (vertical text, which tables are not read from).
fn runs_down(direction: [f32; 2]) -> bool {
    direction[1].abs() > direction[0].abs()
}

/// The words of a page's text: runs of horizontal glyphs of one line with no word break and no
/// gap between them. Vertical text is left out.
pub fn words_of(text: &PageText) -> Vec<Word> {
    let mut out: Vec<Word> = Vec::new();
    let mut current: Option<(u32, Word)> = None;
    for (i, g) in text.glyphs.iter().enumerate() {
        let line = text.line_of.get(i).copied().unwrap_or(u32::MAX);
        let rect = g.rect.map(f64::from);
        let blank = g.text.trim().is_empty();
        if blank || runs_down(g.direction) || !rect.iter().all(|v| v.is_finite()) {
            if let Some((_, w)) = current.take() {
                out.push(w);
            }
            continue;
        }
        let breaks = text.space_before.get(i).copied().unwrap_or(false);
        match current.as_mut() {
            Some((l, w))
                if *l == line && !breaks && {
                    let gap = rect[0] - w.rect[2];
                    let h = (rect[3] - rect[1]).max(1e-3);
                    gap <= 0.1 * h && gap >= -0.5 * h
                } =>
            {
                w.text.push_str(&g.text);
                w.rect = [w.rect[0].min(rect[0]), w.rect[1].min(rect[1]), w.rect[2].max(rect[2]), w.rect[3].max(rect[3])];
            }
            _ => {
                if let Some((_, w)) = current.take() {
                    out.push(w);
                }
                current = Some((line, Word { text: g.text.clone(), rect }));
            }
        }
        if out.len() >= detect::MAX_WORDS {
            break;
        }
    }
    if let Some((_, w)) = current {
        out.push(w);
    }
    out
}

/// The ruled lines of each of `pages` (0-based) in page view space, or nothing when the file
/// cannot be read for them.
fn rules_of(src: &ExportSource, pages: &[usize]) -> Vec<Vec<[f64; 4]>> {
    let Ok(doc) = pdfcraft_cos::Document::open_with_password(src.bytes.clone(), src.config.password.as_deref()) else {
        return vec![Vec::new(); pages.len()];
    };
    let all = pdfcraft_model::pages(&doc);
    pages
        .iter()
        .map(|&p| {
            let Some(page) = all.get(p) else { return Vec::new() };
            let (_, height) = page.display_size(&doc);
            let Some(m) = pdfcraft_content::Matrix(page.view_matrix(&doc)).invert() else { return Vec::new() };
            let view = |x: f64, y: f64| {
                let (a, b) = m.apply(x, y);
                (a, height - b)
            };
            pdfcraft_edit::page_rules(&doc, p)
                .unwrap_or_default()
                .into_iter()
                .map(|r| {
                    let ((ax, ay), (bx, by)) = (view(r[0], r[1]), view(r[2], r[3]));
                    [ax.min(bx), ay.min(by), ax.max(bx), ay.max(by)]
                })
                .collect()
        })
        .collect()
}

/// Find the tables of pages `pages` (0-based, in the order given) of `src` and write them as CSV.
///
/// `progress(done, total)` is called while OCR runs; returning `false` there stops the reading.
/// An engine that cannot run is an error when OCR is asked for and a page needs it.
pub fn export_tables_csv(
    src: ExportSource,
    name: &str,
    pages: &[usize],
    options: &TableCsvOptions,
    progress: impl FnMut(usize, usize) -> bool,
) -> Result<TableCsvExport, String> {
    export_tables_csv_with(src, name, pages, options, &crate::ocr::recognizers, progress)
}

/// [`export_tables_csv`] with the recognition engines supplied by `engines` (tests use a fake one).
pub(crate) fn export_tables_csv_with(
    src: ExportSource,
    name: &str,
    pages: &[usize],
    options: &TableCsvOptions,
    engines: &dyn Fn(&OcrSettings) -> Result<Recognizers, String>,
    mut progress: impl FnMut(usize, usize) -> bool,
) -> Result<TableCsvExport, String> {
    if pages.is_empty() {
        return Err("no pages to export".into());
    }
    if pages.len() > MAX_PAGES {
        return Err(format!("at most {MAX_PAGES} pages can be exported at once"));
    }
    if let Some(bad) = pages.iter().find(|&&p| p >= src.pages) {
        return Err(format!("page {} does not exist", bad.saturating_add(1)));
    }
    let mut out = TableCsvExport { pages: pages.len(), ..Default::default() };
    let mut source = src;
    if options.ocr_if_needed {
        let (bytes, read) = read_missing(&source, name, pages, &options.ocr, engines, &mut progress)?;
        if let Some(bytes) = bytes {
            source.bytes = bytes;
        }
        out.ocr_pages = read.iter().map(|p| p + 1).collect();
    }
    let rules = if options.use_rules { rules_of(&source, pages) } else { vec![Vec::new(); pages.len()] };
    let mut exporter = Exporter::from_source(source);
    let mut found: Vec<Table> = Vec::new();
    for (k, &page) in pages.iter().enumerate() {
        let text = exporter.page_text(page)?;
        let words = words_of(&text);
        if words.is_empty() {
            out.empty_pages.push(page + 1);
            continue;
        }
        let tables = detect::tables(&words, rules.get(k).map_or(&[][..], Vec::as_slice));
        for (n, t) in tables.into_iter().enumerate() {
            out.tables.push(FoundTable { page: page + 1, index: n + 1, rows: t.rows.len(), columns: t.columns(), ruled: t.ruled });
            found.push(t);
        }
    }
    if !out.empty_pages.is_empty() {
        let list = out.empty_pages.iter().map(ToString::to_string).collect::<Vec<_>>().join(", ");
        let hint = if options.ocr_if_needed { "" } else { " (turn on OCR to read scanned pages)" };
        out.warnings.push(format!("no text on page(s) {list}{hint}"));
    }
    if found.is_empty() {
        out.warnings.push("no tables were found".into());
    } else {
        out.files = match options.layout {
            CsvLayout::PerTable => found.iter().map(|t| csv_bytes([t])).collect(),
            CsvLayout::OneFile => vec![csv_bytes(&found)],
        };
    }
    Ok(out)
}

#[cfg(test)]
mod tests;
