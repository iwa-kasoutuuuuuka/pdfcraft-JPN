//! 和暦/西暦の一括変換 through the tools: `dates_find_ja` lists the dates in page text, comments
//! and text fields; `dates_convert_ja` replaces them in one undoable edit.

use std::path::PathBuf;

use pdfcraft_automation::{Automation, Content, ToolError};
use serde_json::{Value, json};

fn workdir(test: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("pdfcraft-era-dates-{test}-{}", std::process::id()));
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

/// A page with one line of text per entry, a note whose text holds a date, and a text field
/// holding an era date.
fn document(a: &mut Automation, lines: &[&str]) -> u64 {
    let doc = ok(a, "doc_create", json!({ "from": "blank", "width": 600, "height": 400 }))["doc"].as_u64().unwrap();
    for (i, line) in lines.iter().enumerate() {
        ok(a, "page_add_text", json!({ "doc": doc, "page": 1, "text": line, "at": [30, 30 + 24 * i], "size": 14, "width": 500 }));
    }
    ok(a, "comment_add", json!({ "doc": doc, "page": 1, "type": "note", "at": [20, 300], "contents": "会議は2024年7月1日です", "author": "Ada" }));
    ok(a, "form_add_field", json!({ "doc": doc, "page": 1, "type": "text", "rect": [300, 300, 500, 330], "name": "申請日" }));
    ok(a, "form_fill", json!({ "doc": doc, "values": { "申請日": "R6.5.1" } }));
    doc
}

fn lines(a: &mut Automation, doc: u64) -> Vec<String> {
    let r = ok(a, "text_lines", json!({ "doc": doc, "page": 1 }));
    r["lines"].as_array().unwrap().iter().map(|l| l["text"].as_str().unwrap().to_string()).collect()
}

fn field(a: &mut Automation, doc: u64) -> String {
    let r = ok(a, "form_fields", json!({ "doc": doc }));
    let f = r["fields"].as_array().unwrap().iter().find(|f| f["name"] == "申請日").cloned().unwrap();
    f["value"].as_str().map(str::to_owned).unwrap_or_else(|| f["value"].to_string())
}

fn note(a: &mut Automation, doc: u64) -> String {
    let r = ok(a, "comment_list", json!({ "doc": doc }));
    r["comments"].as_array().unwrap().iter().find(|c| c["type"] == "Text").unwrap()["contents"].as_str().unwrap().to_string()
}

/// (text, status, converted or problem) of each hit.
fn summary(r: &Value) -> Vec<(String, String, String)> {
    r["hits"]
        .as_array()
        .unwrap()
        .iter()
        .map(|h| {
            let detail = h.get("converted").or_else(|| h.get("problem")).and_then(Value::as_str).unwrap_or("");
            (h["text"].as_str().unwrap().to_string(), h["status"].as_str().unwrap().to_string(), detail.to_string())
        })
        .collect()
}

fn s(text: &str, status: &str, detail: &str) -> (String, String, String) {
    (text.into(), status.into(), detail.into())
}

#[test]
fn dates_find_ja_lists_dates_in_page_text_comments_and_fields() {
    if pdfcraft_fonts::document_japanese_font().is_none() {
        eprintln!("skipping: built without craft-fonts");
        return;
    }
    let dir = workdir("find");
    let mut a = auto(&dir);
    let doc = document(&mut a, &["申請日 令和6年5月1日", "期限 2024/6/30", "電話 03-1234-5678", "誤り 平成31年5月1日"]);
    let r = ok(&mut a, "dates_find_ja", json!({ "doc": doc, "to": "western" }));
    assert_eq!(
        summary(&r),
        vec![
            s("令和6年5月1日", "convert", "2024年5月1日"),
            s("2024/6/30", "already", ""),
            s("平成31年5月1日", "invalid", "outside-era"),
            s("2024年7月1日", "already", ""),
            s("R6.5.1", "convert", "2024年5月1日"),
        ],
        "{r}"
    );
    assert_eq!((r["count"].as_u64(), r["convertible"].as_u64(), r["invalid"].as_u64()), (Some(5), Some(2), Some(1)));
    let kinds: Vec<&str> = r["hits"].as_array().unwrap().iter().map(|h| h["source"]["kind"].as_str().unwrap()).collect();
    assert_eq!(kinds, ["page-text", "page-text", "page-text", "comment", "field"]);
    assert_eq!(r["hits"][0]["page"], 1);
    assert_eq!(r["hits"][4]["source"]["field"], "申請日");
    assert!(r["hits"][2]["note"].as_str().unwrap().contains("era"));
    // To the era calendar the other hits convert instead.
    let r = ok(&mut a, "dates_find_ja", json!({ "doc": doc }));
    assert_eq!(
        summary(&r),
        vec![
            s("令和6年5月1日", "already", ""),
            s("2024/6/30", "convert", "令和6年6月30日"),
            s("平成31年5月1日", "invalid", "outside-era"),
            s("2024年7月1日", "convert", "令和6年7月1日"),
            s("R6.5.1", "already", ""),
        ],
        "{r}"
    );
    assert!(matches!(a.call("dates_find_ja", &json!({ "doc": doc, "to": "lunar" })), Err(ToolError::InvalidArgs(_))));
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn dates_convert_ja_replaces_page_text_comments_and_fields_in_one_undo_step() {
    if pdfcraft_fonts::document_japanese_font().is_none() {
        eprintln!("skipping: built without craft-fonts");
        return;
    }
    let dir = workdir("convert");
    let mut a = auto(&dir);
    let doc = document(&mut a, &["申請日 令和6年5月1日", "期限 2024/6/30", "電話 03-1234-5678", "誤り 平成31年5月1日"]);
    let before = lines(&mut a, doc);

    let r = ok(&mut a, "dates_convert_ja", json!({ "doc": doc, "to": "western" }));
    assert_eq!(r["replaced"].as_array().unwrap().len(), 2, "{r}");
    assert_eq!(r["skipped"].as_array().unwrap().len(), 0, "{r}");
    let after = lines(&mut a, doc);
    assert_eq!(after, vec!["申請日 2024年5月1日", "期限 2024/6/30", "電話 03-1234-5678", "誤り 平成31年5月1日"]);
    assert_eq!(field(&mut a, doc), "2024年5月1日");
    assert_eq!(note(&mut a, doc), "会議は2024年7月1日です");
    // Nothing left to convert to the Western calendar.
    let again = ok(&mut a, "dates_find_ja", json!({ "doc": doc, "to": "western" }));
    assert_eq!(again["convertible"], 0, "{again}");

    // One undo step brings every replacement back.
    assert_eq!(ok(&mut a, "edit_undo", json!({ "doc": doc }))["undone"], "Convert dates");
    assert_eq!(lines(&mut a, doc), before);
    assert_eq!(field(&mut a, doc), "R6.5.1");

    // To the era calendar, only the chosen hits; an invalid one is skipped with its reason.
    let found = ok(&mut a, "dates_find_ja", json!({ "doc": doc }));
    assert_eq!(found["hits"][1]["text"], "2024/6/30");
    let r = ok(&mut a, "dates_convert_ja", json!({ "doc": doc, "hits": [2, 3, 4] }));
    assert_eq!(r["replaced"].as_array().unwrap().len(), 2, "{r}");
    assert_eq!(r["skipped"].as_array().unwrap().len(), 1, "{r}");
    assert_eq!(r["skipped"][0]["text"], "平成31年5月1日");
    assert_eq!(lines(&mut a, doc), vec!["申請日 令和6年5月1日", "期限 令和6年6月30日", "電話 03-1234-5678", "誤り 平成31年5月1日"]);
    assert_eq!(note(&mut a, doc), "会議は令和6年7月1日です");
    assert_eq!(field(&mut a, doc), "R6.5.1");

    // Nothing to replace changes nothing.
    let r = ok(&mut a, "dates_convert_ja", json!({ "doc": doc }));
    assert_eq!(r["changed"], false, "{r}");
    assert!(matches!(a.call("dates_convert_ja", &json!({ "doc": doc, "hits": [99] })), Err(ToolError::InvalidArgs(_))));
    std::fs::remove_dir_all(dir).unwrap();
}
