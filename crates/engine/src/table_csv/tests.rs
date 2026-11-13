use std::sync::Arc;

use super::*;
use crate::{Edit, Session};
use pdfcraft_ocr::PlacedWord;

/// 200 x 300 pt pages, each drawing its content stream (`""`: none).
fn pages_pdf(contents: &[&str]) -> Vec<u8> {
    let n = contents.len();
    let kids: Vec<String> = (0..n).map(|i| format!("{} 0 R", 3 + 2 * i)).collect();
    let mut objs = vec![
        "<< /Type /Catalog /Pages 2 0 R >>".to_string(),
        format!("<< /Type /Pages /Kids [{}] /Count {n} /MediaBox [0 0 200 300] >>", kids.join(" ")),
    ];
    for (i, c) in contents.iter().enumerate() {
        objs.push(format!("<< /Type /Page /Parent 2 0 R /Contents {} 0 R >>", 4 + 2 * i));
        objs.push(format!("<< /Length {} >>\nstream\n{c}\nendstream", c.len() + 1));
    }
    let mut out = b"%PDF-1.7\n".to_vec();
    let mut offsets = Vec::new();
    for (i, o) in objs.iter().enumerate() {
        offsets.push(out.len());
        out.extend_from_slice(format!("{} 0 obj\n{o}\nendobj\n", i + 1).as_bytes());
    }
    let xref = out.len();
    out.extend_from_slice(format!("xref\n0 {}\n0000000000 65535 f \n", objs.len() + 1).as_bytes());
    for o in offsets {
        out.extend_from_slice(format!("{o:010} 00000 n \n").as_bytes());
    }
    out.extend_from_slice(format!("trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n", objs.len() + 1).as_bytes());
    out
}

/// A recognised word of `text`, its top-left at (`x`, `top`) from the top of a 300 pt page, 10 pt
/// a character and 10 pt high.
fn word(text: &str, x: f64, top: f64) -> PlacedWord {
    let w = 10.0 * text.chars().count() as f64;
    PlacedWord { text: text.into(), origin: [x, 300.0 - top - 10.0], across: [w, 0.0], up: [0.0, 10.0], confidence: None, source: "test".into() }
}

/// A document whose pages draw `contents` and carry `words` as an invisible recognised text
/// layer, saved and opened again.
fn doc(contents: &[&str], words: &[Vec<PlacedWord>]) -> ExportSource {
    let mut s = Session::new().with_clock(|| 1_700_000_000);
    let id = s.open("t.pdf", None, Arc::new(pages_pdf(contents)), None).unwrap();
    for (page, w) in words.iter().enumerate() {
        if !w.is_empty() {
            s.apply(id, Edit::AddOcrText { page, words: w.clone() }).unwrap();
        }
    }
    let bytes = s.save_bytes(id).unwrap();
    let mut s2 = Session::new();
    let id2 = s2.open("t.pdf", None, bytes, None).unwrap();
    s2.get(id2).unwrap().export_source()
}

fn japanese_table() -> Vec<PlacedWord> {
    vec![
        word("品名", 20.0, 20.0),
        word("数量", 90.0, 20.0),
        word("備考", 140.0, 20.0),
        word("りんご", 20.0, 34.0),
        word("3", 90.0, 34.0),
        word("赤い", 140.0, 34.0),
        word("みかん", 20.0, 48.0),
        word("12", 90.0, 48.0),
        word("甘い", 140.0, 48.0),
    ]
}

fn text(bytes: &[u8]) -> &str {
    std::str::from_utf8(bytes).unwrap()
}

#[test]
fn fields_are_quoted_as_rfc_4180_asks_and_formulas_are_defused() {
    assert_eq!(csv_field("東京"), "東京");
    assert_eq!(csv_field("1,200"), "\"1,200\"");
    assert_eq!(csv_field("say \"hi\""), "\"say \"\"hi\"\"\"");
    assert_eq!(csv_field("two\nlines"), "\"two\nlines\"");
    assert_eq!(csv_field("=SUM(A1)"), "'=SUM(A1)");
    assert_eq!(csv_field("@cmd"), "'@cmd");
    assert_eq!(csv_field("+cmd|x"), "'+cmd|x");
    assert_eq!(csv_field("-12.5"), "-12.5");
    assert_eq!(csv_field("-"), "-");
    assert_eq!(csv_field(""), "");
}

#[test]
fn csv_bytes_have_a_bom_crlf_padded_rows_and_a_blank_line_between_tables() {
    let a = Table { rows: vec![vec!["品名".into(), "価格".into()], vec!["りんご".into()]], rect: [0.0; 4], ruled: false };
    let b = Table { rows: vec![vec!["a,b".into(), "x".into()]], rect: [0.0; 4], ruled: false };
    let bytes = csv_bytes([&a, &b]);
    assert!(bytes.starts_with(&[0xEF, 0xBB, 0xBF]));
    assert_eq!(text(&bytes), "\u{feff}品名,価格\r\nりんご,\r\n\r\n\"a,b\",x\r\n");
    assert_eq!(text(&csv_bytes([&b])), "\u{feff}\"a,b\",x\r\n");
}

#[test]
fn a_recognised_japanese_table_is_written_as_csv() {
    let src = doc(&[""], &[japanese_table()]);
    let out = export_tables_csv(src, "t.pdf", &[0], &TableCsvOptions::default(), |_, _| true).unwrap();
    assert_eq!(out.tables, [FoundTable { page: 1, index: 1, rows: 3, columns: 3, ruled: false }]);
    assert_eq!(out.files.len(), 1);
    assert_eq!(text(&out.files[0]), "\u{feff}品名,数量,備考\r\nりんご,3,赤い\r\nみかん,12,甘い\r\n");
}

#[test]
fn ruled_lines_decide_the_cells_and_layouts_split_or_join_the_files() {
    // Page 1: a 2 x 2 grid (x 10..190 split at 100, y from the top 10..70 split at 30) whose
    // second row wraps; page 2: the table without rules.
    let grid = "1 w 10 290 m 190 290 l S 10 270 m 190 270 l S 10 230 m 190 230 l S 10 290 m 10 230 l S 100 290 m 100 230 l S 190 290 m 190 230 l S";
    let ruled = vec![
        word("項目", 15.0, 15.0),
        word("内容", 105.0, 15.0),
        word("住所", 15.0, 35.0),
        word("東京都千代田", 105.0, 35.0),
        word("一丁目", 105.0, 47.0),
    ];
    let src = doc(&[grid, ""], &[ruled, japanese_table()]);
    let out = export_tables_csv(src.clone(), "t.pdf", &[0, 1], &TableCsvOptions::default(), |_, _| true).unwrap();
    assert_eq!(out.tables.len(), 2, "{out:?}");
    assert!(out.tables[0].ruled && !out.tables[1].ruled);
    assert_eq!(text(&out.files[0]), "\u{feff}項目,内容\r\n住所,東京都千代田一丁目\r\n");
    assert_eq!(out.files.len(), 2);
    assert_eq!(table_file_name("doc", &out.tables[1]), "doc_p2_t1.csv");
    let one = TableCsvOptions { layout: CsvLayout::OneFile, ..Default::default() };
    let joined = export_tables_csv(src.clone(), "t.pdf", &[0, 1], &one, |_, _| true).unwrap();
    assert_eq!(joined.files.len(), 1);
    assert_eq!(text(&joined.files[0]), "\u{feff}項目,内容\r\n住所,東京都千代田一丁目\r\n\r\n品名,数量,備考\r\nりんご,3,赤い\r\nみかん,12,甘い\r\n");
    // Rules off: the wrapped line no longer belongs to its cell.
    let plain = TableCsvOptions { use_rules: false, ..Default::default() };
    let unruled = export_tables_csv(src, "t.pdf", &[0], &plain, |_, _| true).unwrap();
    assert!(!text(&unruled.files[0]).contains("一丁目"), "{}", text(&unruled.files[0]));
}

#[test]
fn pages_without_tables_or_text_are_reported_and_ocr_is_refused_without_an_engine() {
    let src = doc(&["", ""], &[vec![word("ただの文章です", 20.0, 20.0)], vec![]]);
    let out = export_tables_csv(src.clone(), "t.pdf", &[0, 1], &TableCsvOptions::default(), |_, _| true).unwrap();
    assert!(out.tables.is_empty() && out.files.is_empty());
    assert_eq!(out.empty_pages, [2]);
    assert!(out.warnings.iter().any(|w| w.contains("no tables")), "{out:?}");
    let options = TableCsvOptions { ocr_if_needed: true, ..Default::default() };
    let broken = |_: &OcrSettings| -> Result<Recognizers, String> { Err("no engine installed".into()) };
    let err = export_tables_csv_with(src.clone(), "t.pdf", &[0, 1], &options, &broken, |_, _| true).unwrap_err();
    assert!(err.contains("no engine installed"), "{err}");
    assert!(export_tables_csv(src.clone(), "t.pdf", &[], &TableCsvOptions::default(), |_, _| true).is_err());
    assert!(export_tables_csv(src, "t.pdf", &[7], &TableCsvOptions::default(), |_, _| true).unwrap_err().contains("page 8"));
}

/// An engine that reads a two-row, two-column table on every page; no models needed.
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
        let w = |t: &str, x: f32, y: f32| pdfcraft_ocr::Word::new(t, [x, y, x + 40.0, y + 20.0], "fake");
        Ok(vec![
            pdfcraft_ocr::Line { words: vec![w("名前", 10.0, 10.0), w("年齢", 150.0, 10.0)] },
            pdfcraft_ocr::Line { words: vec![w("山田", 10.0, 40.0), w("30", 150.0, 40.0)] },
        ])
    }
}

#[test]
fn a_scanned_page_is_read_first_when_asked() {
    let mut s = Session::new();
    let made = s.create_from_text("t", "A picture").unwrap();
    let id = s.open("text.pdf", None, made, None).unwrap();
    let png = crate::export::Exporter::new(s.get(id).unwrap()).png(0, 100.0).unwrap();
    let scan = s.create_from_images(&[("scan.png".into(), png)]).unwrap();
    let mut s2 = Session::new();
    let id2 = s2.open("scan.pdf", None, scan, None).unwrap();
    let src = s2.get(id2).unwrap().export_source();
    let fake = |_: &OcrSettings| -> Result<Recognizers, String> { Ok(Recognizers { primary: Arc::new(Fake), secondary: None }) };
    let options = TableCsvOptions { ocr_if_needed: true, ..Default::default() };
    let out = export_tables_csv_with(src.clone(), "scan.pdf", &[0], &options, &fake, |_, _| true).unwrap();
    assert_eq!(out.ocr_pages, [1]);
    assert_eq!(out.tables.len(), 1, "{out:?}");
    assert_eq!(text(&out.files[0]), "\u{feff}名前,年齢\r\n山田,30\r\n");
    let plain = export_tables_csv_with(src, "scan.pdf", &[0], &TableCsvOptions::default(), &fake, |_, _| true).unwrap();
    assert!(plain.tables.is_empty() && plain.empty_pages == [1], "{plain:?}");
}

#[test]
fn layouts_parse() {
    assert_eq!(CsvLayout::parse("one-file"), Some(CsvLayout::OneFile));
    assert_eq!(CsvLayout::parse("PER_TABLE"), Some(CsvLayout::PerTable));
    assert_eq!(CsvLayout::parse("xlsx"), None);
}
