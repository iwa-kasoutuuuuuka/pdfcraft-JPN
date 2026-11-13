//! pdfcraft-JPN: the 原稿用紙 count of `text_word_count` and the Japanese page numbers of
//! `doc_header_footer`.

use pdfcraft_automation::{Automation, Content, ToolError};
use pdfcraft_engine::Session;
use serde_json::{Value, json};

fn has_japanese_face() -> bool {
    pdfcraft_fonts::document_japanese_font_for_style(false, false).is_some()
}

fn ok(a: &mut Automation, tool: &str, args: Value) -> Value {
    match a.call(tool, &args) {
        Ok(mut c) => match c.remove(0) {
            Content::Json(v) => v,
            other => panic!("{tool}: expected JSON, got {other:?}"),
        },
        Err(e) => panic!("{tool} {args}: {e}"),
    }
}

fn page_texts(a: &mut Automation, doc: u64) -> Vec<String> {
    let pages = ok(a, "text_extract", json!({ "doc": doc }))["pages"].as_array().cloned().unwrap();
    pages.iter().map(|p| p["text"].as_str().unwrap().to_string()).collect()
}

/// A workdir holding `a.pdf`, made from `text` (form feeds start new pages).
fn open(test: &str, text: &str) -> (Automation, u64) {
    let dir = std::env::temp_dir().join(format!("pdfcraft-jacount-{test}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let bytes = Session::new().create_from_text("Count", text).unwrap();
    std::fs::write(dir.join("a.pdf"), bytes.as_ref()).unwrap();
    let mut a = Automation::new().with_root(&dir).unwrap();
    let doc = ok(&mut a, "doc_open", json!({ "path": "a.pdf" }))["doc"].as_u64().unwrap();
    (a, doc)
}

#[test]
fn word_count_reports_japanese_characters_and_manuscript_sheets() {
    // Latin text works in any build: 2 lines on one page, 1 on the other.
    let (mut a, doc) = open("latin", "Hello, world!\nAgain.\u{c}One more.");
    let report = ok(&mut a, "text_word_count", json!({ "doc": doc }));
    let total = &report["total"];
    assert_eq!(total["characters_without_spaces"], 26, "{report}");
    assert_eq!(total["characters_without_punctuation"], 22, "{report}");
    assert_eq!(total["manuscript"], json!({ "lines": 3, "sheets": 1, "last_sheet_lines": 3 }), "{report}");
    assert_eq!(report["pages"][1]["counts"]["manuscript"]["lines"], 1);
    if !has_japanese_face() {
        eprintln!("skipping the Japanese part: built without a craft-fonts Jpan face");
        return;
    }
    // Japanese text from a 原稿用紙 document PdfCraft makes: 400 characters and a stop.
    let made = ok(&mut a, "create_vertical_document", json!({ "text": format!("{}。", "あ".repeat(400)), "template": "genko" }));
    let doc = made["doc"].as_u64().unwrap();
    let report = ok(&mut a, "text_word_count", json!({ "doc": doc }));
    assert_eq!(report["total"]["characters_without_spaces"], 401, "{report}");
    assert_eq!(report["total"]["characters_without_punctuation"], 400, "{report}");
    // Each column the document reads back as is a line of its own; the 20-square columns (and
    // the stop hung below the last of them) each fill one manuscript line.
    let lines = report["total"]["manuscript"]["lines"].as_u64().unwrap();
    let page_lines: u64 = report["pages"].as_array().unwrap().iter().map(|p| p["counts"]["manuscript"]["lines"].as_u64().unwrap()).sum();
    assert_eq!(lines, page_lines);
    let expected: usize = page_texts(&mut a, doc).iter().map(|t| pdfcraft_engine::word_count::Counts::of(t).manuscript_lines).sum();
    assert_eq!(lines as usize, expected);
    assert!(lines >= 20, "{report}");
    let total = pdfcraft_engine::word_count::Counts { manuscript_lines: expected, ..Default::default() };
    let (sheets, last) = total.manuscript_sheets();
    assert_eq!(report["total"]["manuscript"]["sheets"].as_u64().unwrap() as usize, sheets);
    assert_eq!(report["total"]["manuscript"]["last_sheet_lines"].as_u64().unwrap() as usize, last);
}

#[test]
fn header_and_footer_draw_japanese_page_numbers() {
    let (mut a, doc) = open("pages", "One\u{c}Two\u{c}Three");
    // `- 1 -` and `1 / n` are ASCII: any build draws them.
    ok(&mut a, "doc_header_footer", json!({ "doc": doc, "header_center": "<<- 1 ->>", "footer_right": "<<1 / n>>", "start_number": 104 }));
    let texts = page_texts(&mut a, doc);
    assert!(texts[0].contains("- 104 -") && texts[2].contains("106 / 3"), "{texts:?}");
    let kanji =
        json!({ "doc": doc, "footer_center": "<<第一頁>>", "header_left": "<<1ページ>> <<第1頁>> <<一>>", "start_number": 104, "replace": true });
    if !has_japanese_face() {
        // Without a Japanese face the kanji page numbers are refused, saying why.
        let refused = a.call("doc_header_footer", &kanji);
        assert!(matches!(&refused, Err(ToolError::Failed(e)) if e.contains("Japanese")), "{refused:?}");
        return;
    }
    ok(&mut a, "doc_header_footer", kanji);
    let texts = page_texts(&mut a, doc);
    assert!(texts[0].contains("第百四頁") && texts[0].contains("104ページ") && texts[0].contains("第104頁"), "{texts:?}");
    assert!(texts[1].contains("第百五頁") && texts[1].contains("百五"), "{texts:?}");
    assert!(texts[2].contains("第百六頁"), "{texts:?}");
    assert!(!texts[0].contains("- 104 -"), "replace swapped out the first header: {texts:?}");
}
