//! `export_tables_csv` end to end: a synthetic PDF with a recognised Japanese table and a ruled
//! one, written as CSV for Excel (UTF-8 with a BOM, CRLF, RFC 4180 quoting).

use std::sync::Arc;

use pdfcraft_automation::{Automation, Content};
use pdfcraft_engine::ocr::PlacedWord;
use pdfcraft_engine::{Edit, Session};
use serde_json::{Value, json};

fn ok(a: &mut Automation, tool: &str, args: Value) -> Value {
    match a.call(tool, &args) {
        Ok(mut c) => match c.remove(0) {
            Content::Json(v) => v,
            other => panic!("{tool}: expected JSON, got {other:?}"),
        },
        Err(e) => panic!("{tool} {args}: {e}"),
    }
}

fn workdir(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("pdfcraft-table-csv-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// 200 x 300 pt pages, each drawing its content stream.
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

/// A recognised word, its top-left at (`x`, `top`) from the top of the page, 10 pt a character.
fn word(text: &str, x: f64, top: f64) -> PlacedWord {
    let w = 10.0 * text.chars().count() as f64;
    PlacedWord { text: text.into(), origin: [x, 290.0 - top], across: [w, 0.0], up: [0.0, 10.0], confidence: None, source: "test".into() }
}

/// Page 1: a table without rules (Japanese, a cell with a comma, one with quotes, one of two
/// Latin words, and a row with a missing cell). Page 2: a ruled 2 x 2 grid whose second row wraps.
fn tables_pdf() -> Vec<u8> {
    let grid = "1 w 10 290 m 190 290 l S 10 270 m 190 270 l S 10 230 m 190 230 l S 10 290 m 10 230 l S 100 290 m 100 230 l S 190 290 m 190 230 l S";
    let mut s = Session::new().with_clock(|| 1_700_000_000);
    let id = s.open("tables.pdf", None, Arc::new(pages_pdf(&["", grid])), None).unwrap();
    let page1 = vec![
        word("品名", 10.0, 20.0),
        word("金額", 100.0, 20.0),
        word("備考", 160.0, 20.0),
        word("りんご", 10.0, 34.0),
        word("1,200", 100.0, 34.0),
        word("\"甘\"", 160.0, 34.0),
        word("Green", 10.0, 48.0),
        word("tea", 63.0, 48.0),
        word("なし", 160.0, 48.0),
    ];
    let page2 = vec![
        word("項目", 15.0, 15.0),
        word("内容", 105.0, 15.0),
        word("住所", 15.0, 35.0),
        word("東京都千代田", 105.0, 35.0),
        word("一丁目", 105.0, 47.0),
    ];
    s.apply(id, Edit::AddOcrText { page: 0, words: page1 }).unwrap();
    s.apply(id, Edit::AddOcrText { page: 1, words: page2 }).unwrap();
    s.save_bytes(id).unwrap().to_vec()
}

const PAGE1: &str = "品名,金額,備考\r\nりんご,\"1,200\",\"\"\"甘\"\"\"\r\nGreen tea,,なし\r\n";
const PAGE2: &str = "項目,内容\r\n住所,東京都千代田一丁目\r\n";

#[test]
fn export_tables_csv_writes_excel_ready_csv_per_table_or_in_one_file() {
    let dir = workdir("export");
    std::fs::write(dir.join("tables.pdf"), tables_pdf()).unwrap();
    let mut a = Automation::new().with_root(&dir).unwrap();
    let doc = ok(&mut a, "doc_open", json!({ "path": "tables.pdf" }))["doc"].as_u64().unwrap();

    // One file per table, named from the path.
    let r = ok(&mut a, "export_tables_csv", json!({ "doc": doc, "path": "out.csv" }));
    assert_eq!(r["layout"], "per_table", "{r}");
    assert_eq!(
        r["tables"],
        json!([{ "page": 1, "index": 1, "rows": 3, "columns": 3, "ruled": false }, { "page": 2, "index": 1, "rows": 2, "columns": 2, "ruled": true }]),
        "{r}"
    );
    let first = std::fs::read(dir.join("out_p1_t1.csv")).unwrap();
    assert!(first.starts_with(&[0xEF, 0xBB, 0xBF]), "UTF-8 byte order mark");
    assert_eq!(String::from_utf8(first).unwrap(), format!("\u{feff}{PAGE1}"));
    assert_eq!(std::fs::read_to_string(dir.join("out_p2_t1.csv")).unwrap(), format!("\u{feff}{PAGE2}"));
    assert!(!dir.join("out.csv").exists());

    // Every table in one file, a blank line between them; a page range picks the second only.
    let r = ok(&mut a, "export_tables_csv", json!({ "doc": doc, "path": "all.csv", "layout": "one_file" }));
    assert_eq!(r["files"].as_array().map(Vec::len), Some(1), "{r}");
    assert_eq!(std::fs::read_to_string(dir.join("all.csv")).unwrap(), format!("\u{feff}{PAGE1}\r\n{PAGE2}"));
    let r = ok(&mut a, "export_tables_csv", json!({ "doc": doc, "path": "two.csv", "layout": "one_file", "pages": [2] }));
    assert_eq!(r["tables"].as_array().map(Vec::len), Some(1), "{r}");
    assert_eq!(std::fs::read_to_string(dir.join("two.csv")).unwrap(), format!("\u{feff}{PAGE2}"));

    // Bad arguments and paths outside the root are refused, and nothing is written.
    assert!(a.call("export_tables_csv", &json!({ "doc": doc, "path": "x.csv", "layout": "xlsx" })).is_err());
    assert!(a.call("export_tables_csv", &json!({ "doc": doc, "path": "../escape.csv" })).is_err());
    assert!(!dir.join("x.csv").exists() && !dir.join("x_p1_t1.csv").exists());
    assert!(!dir.parent().unwrap().join("escape_p1_t1.csv").exists());
}

#[test]
fn a_document_without_tables_writes_nothing_and_says_so() {
    let dir = workdir("none");
    let mut s = Session::new().with_clock(|| 1_700_000_000);
    let id = s.open("prose.pdf", None, Arc::new(pages_pdf(&[""])), None).unwrap();
    s.apply(id, Edit::AddOcrText { page: 0, words: vec![word("表のない普通の文章です", 10.0, 20.0)] }).unwrap();
    std::fs::write(dir.join("prose.pdf"), s.save_bytes(id).unwrap().as_slice()).unwrap();
    let mut a = Automation::new().with_root(&dir).unwrap();
    let doc = ok(&mut a, "doc_open", json!({ "path": "prose.pdf" }))["doc"].as_u64().unwrap();
    let r = ok(&mut a, "export_tables_csv", json!({ "doc": doc, "path": "none.csv", "layout": "one_file" }));
    assert_eq!(r["tables"], json!([]), "{r}");
    assert!(r["warnings"].to_string().contains("no tables"), "{r}");
    assert!(!dir.join("none.csv").exists());
}
