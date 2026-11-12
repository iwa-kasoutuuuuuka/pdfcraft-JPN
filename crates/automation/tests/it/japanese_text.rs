//! Japanese text matching through the tools: `text_find` with `ignore_variants`, and
//! `redact_find_pii` (personal information as written in Japan) marking, applying and removing it.

use std::path::PathBuf;

use pdfcraft_automation::{Automation, Content};
use serde_json::{Value, json};

fn workdir(test: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("pdfcraft-japanese-text-{test}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn auto(dir: &std::path::Path) -> Automation {
    Automation::new().with_root(dir).unwrap().with_clock(|| 1_700_000_000)
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

/// A blank page with one line of text per entry, 24 points apart.
fn page_with(a: &mut Automation, lines: &[&str]) -> u64 {
    let doc = ok(a, "doc_create", json!({ "from": "blank", "width": 600, "height": 400 }))["doc"].as_u64().unwrap();
    for (i, line) in lines.iter().enumerate() {
        ok(a, "page_add_text", json!({ "doc": doc, "page": 1, "text": line, "at": [30, 30 + 24 * i], "size": 14, "width": 500 }));
    }
    doc
}

fn extracted(a: &mut Automation, doc: u64) -> String {
    ok(a, "text_extract", json!({ "doc": doc }))["pages"][0]["text"].as_str().unwrap().to_string()
}

fn found(a: &mut Automation, doc: u64, query: &str, variants: bool) -> Vec<String> {
    let r = ok(a, "text_find", json!({ "doc": doc, "query": query, "ignore_variants": variants }));
    r["matches"].as_array().unwrap().iter().map(|m| m["text"].as_str().unwrap().to_string()).collect()
}

/// Kana width and script, full-width digits and old kanji are found by the other spelling only with
/// `ignore_variants`; what is highlighted is the text as written.
#[test]
fn text_find_ignores_japanese_variants_when_asked() {
    if pdfcraft_fonts::document_japanese_font().is_none() {
        eprintln!("skipping: built without craft-fonts");
        return;
    }
    let dir = workdir("find");
    let mut a = auto(&dir);
    let doc = page_with(&mut a, &["ガッコウ に行く", "東京大學 國立", "番号 １２３４", "ｶﾀｶﾅ ｶﾞｷﾞ"]);
    // Added text is one run, so a hit highlights the run it is in; `one` checks that exactly one
    // run is found and that it holds the text as written.
    let mut one = |query: &str, variants: bool, written: &str| {
        let hits = found(&mut a, doc, query, variants);
        assert_eq!(hits.len(), 1, "{query}: {hits:?}");
        assert!(hits[0].contains(written), "{query}: {hits:?} lacks {written}");
    };
    one("がっこう", true, "ガッコウ");
    one("大学", true, "大學");
    one("國", false, "國");
    one("国", true, "國");
    one("123", true, "１２３");
    one("カタカナ", true, "ｶﾀｶﾅ");
    one("ガギ", true, "ｶﾞｷﾞ");
    for query in ["がっこう", "国", "123", "カタカナ"] {
        assert!(found(&mut a, doc, query, false).is_empty(), "{query} found without ignore_variants");
    }
    // The highlight is a rectangle on the line as written.
    let r = ok(&mut a, "text_find", json!({ "doc": doc, "query": "国立", "ignore_variants": true }));
    assert_eq!(r["count"], 1);
    assert_eq!(r["matches"][0]["rects"].as_array().unwrap().len(), 1, "{r}");
    std::fs::remove_dir_all(dir).unwrap();
}

/// Personal information is found by category, marked for review, and gone from the text once the
/// redaction is applied; a date and a plain number stay.
#[test]
fn redact_find_pii_marks_and_applies() {
    let dir = workdir("pii");
    let mut a = auto(&dir);
    let doc = page_with(
        &mut a,
        &[
            "Phone 090-1234-5678 or 03-5555-0000",
            "Zip 100-0001 mail taro.yamada@example.co.jp",
            "My Number 1234 5678 9018 (bad 1234 5678 9012)",
            "Date 2024-11-11 order 20241111",
        ],
    );
    let r = ok(&mut a, "redact_find_pii", json!({ "doc": doc }));
    let texts = |r: &Value, category: &str| -> Vec<String> {
        r["hits"].as_array().unwrap().iter().filter(|h| h["category"] == category).map(|h| h["text"].as_str().unwrap().to_string()).collect()
    };
    assert_eq!(texts(&r, "phone"), ["090-1234-5678", "03-5555-0000"], "{r}");
    assert_eq!(texts(&r, "postal-code"), ["100-0001"], "{r}");
    assert_eq!(texts(&r, "email"), ["taro.yamada@example.co.jp"], "{r}");
    assert_eq!(texts(&r, "my-number"), ["1234 5678 9018"], "{r}");
    assert_eq!(r["count"], 5, "{r}");
    assert_eq!(r["by_category"]["phone"], 2);
    assert!(r["hits"][0]["rects"].as_array().is_some_and(|x| !x.is_empty()));
    // Finding alone marks nothing.
    assert!(a.call("redact_apply", &json!({ "doc": doc })).is_err());

    // A chosen category only.
    let only = ok(&mut a, "redact_find_pii", json!({ "doc": doc, "categories": ["email"] }));
    assert_eq!(only["count"], 1);
    assert!(a.call("redact_find_pii", &json!({ "doc": doc, "categories": ["nope"] })).is_err());

    let marked = ok(&mut a, "redact_find_pii", json!({ "doc": doc, "mark": true }));
    assert_eq!(marked["marked"], 5, "{marked}");
    assert_eq!(marked["marks_pending"], 5);
    ok(&mut a, "redact_apply", json!({ "doc": doc }));
    let text = extracted(&mut a, doc);
    for gone in ["090-1234-5678", "03-5555-0000", "100-0001", "example.co.jp", "9018", "5555"] {
        assert!(!text.contains(gone), "{gone} survived: {text}");
    }
    for kept in ["Phone", "2024-11-11", "20241111", "9012"] {
        assert!(text.contains(kept), "{kept} was removed: {text}");
    }
    std::fs::remove_dir_all(dir).unwrap();
}

/// Full-width digits and a postal mark are read too, where the build can draw Japanese.
#[test]
fn redact_find_pii_reads_full_width_text() {
    if pdfcraft_fonts::document_japanese_font().is_none() {
        eprintln!("skipping: built without craft-fonts");
        return;
    }
    let dir = workdir("pii-wide");
    let mut a = auto(&dir);
    let doc = page_with(&mut a, &["電話：０９０－１２３４－５６７８", "〒１００－０００１", "東京都千代田区千代田１－１", "令和６年１１月１１日"]);
    let r = ok(&mut a, "redact_find_pii", json!({ "doc": doc, "categories": ["phone", "postal-code", "address"] }));
    let hits: Vec<(String, String)> = r["hits"]
        .as_array()
        .unwrap()
        .iter()
        .map(|h| (h["category"].as_str().unwrap().to_string(), h["text"].as_str().unwrap().to_string()))
        .collect();
    // Added text is one run, so a hit is the run it is in.
    assert_eq!(hits.len(), 3, "{hits:?}");
    assert!(hits.iter().any(|h| h.0 == "phone" && h.1.contains("０９０－１２３４－５６７８")), "{hits:?}");
    assert!(hits.iter().any(|h| h.0 == "postal-code" && h.1.contains("〒１００－０００１")), "{hits:?}");
    assert!(hits.iter().any(|h| h.0 == "address" && h.1.contains("東京都千代田区千代田１－１")), "{hits:?}");
    ok(&mut a, "redact_find_pii", json!({ "doc": doc, "categories": ["phone", "address"], "mark": true }));
    ok(&mut a, "redact_apply", json!({ "doc": doc }));
    let text = extracted(&mut a, doc);
    assert!(!text.contains("１２３４") && !text.contains("千代田区"), "{text}");
    assert!(text.contains("１００") && text.contains("令和"), "{text}");
    std::fs::remove_dir_all(dir).unwrap();
}

/// A column of vertical added text is one run: every hit in it is counted, and the column is
/// marked once.
#[test]
fn redact_find_pii_counts_every_hit_in_a_vertical_column() {
    if pdfcraft_fonts::document_japanese_font().is_none() {
        eprintln!("skipping: built without craft-fonts");
        return;
    }
    let dir = workdir("pii-vertical");
    let mut a = auto(&dir);
    let doc = ok(&mut a, "doc_create", json!({ "from": "blank", "width": 600, "height": 600 }))["doc"].as_u64().unwrap();
    let text = "電話03-1234-5678　携帯090-1234-5678\n〒100-0001　メール taro@example.jp";
    ok(&mut a, "page_add_text", json!({ "doc": doc, "page": 1, "text": text, "rect": [400, 20, 560, 580], "size": 14, "vertical": true }));
    let r = ok(&mut a, "redact_find_pii", json!({ "doc": doc, "categories": ["phone", "postal-code", "email"] }));
    assert_eq!(r["by_category"]["phone"], 2, "{r}");
    assert_eq!(r["by_category"]["postal-code"], 1, "{r}");
    assert_eq!(r["by_category"]["email"], 1, "{r}");
    let marked = ok(&mut a, "redact_find_pii", json!({ "doc": doc, "categories": ["phone", "postal-code", "email"], "mark": true }));
    let columns = marked["marked"].as_u64().unwrap();
    assert!(columns < 4, "one mark per column, not per hit: {marked}");
    std::fs::remove_dir_all(dir).unwrap();
}
