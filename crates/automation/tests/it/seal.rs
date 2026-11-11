//! `page_add_seal`: Japanese electronic seals (電子印鑑) placed as Stamp comments.

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

fn page_text(a: &mut Automation, doc: u64) -> String {
    ok(a, "text_extract", json!({ "doc": doc }))["pages"][0]["text"].as_str().unwrap().to_string()
}

fn open(test: &str) -> (Automation, u64, std::path::PathBuf) {
    let dir = std::env::temp_dir().join(format!("pdfcraft-seal-{test}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let bytes = Session::new().create_from_text("Contract", "Contract text").unwrap();
    std::fs::write(dir.join("a.pdf"), bytes.as_ref()).unwrap();
    // The clock is 2023-11-14, taken as local time (令和5年).
    let mut a = Automation::new().with_root(&dir).unwrap().with_clock(|| 1_700_000_000);
    let doc = ok(&mut a, "doc_open", json!({ "path": "a.pdf" }))["doc"].as_u64().unwrap();
    (a, doc, dir)
}

fn stamps(a: &mut Automation, doc: u64) -> Vec<Value> {
    ok(a, "comment_list", json!({ "doc": doc }))["comments"].as_array().unwrap().iter().filter(|c| c["type"] == "Stamp").cloned().collect()
}

/// A name seal lands as one Stamp comment centred where it was placed, its appearance holds the
/// name as text, and it can be undone and deleted like any stamp.
#[test]
fn a_name_seal_is_a_stamp_comment_with_the_name_as_text() {
    if !has_japanese_face() {
        eprintln!("skipping: built without a craft-fonts Jpan face");
        return;
    }
    let (mut a, doc, dir) = open("name");
    ok(&mut a, "page_add_seal", json!({ "doc": doc, "page": 1, "at": [200, 300], "size": 60, "name": "山田" }));
    let list = stamps(&mut a, doc);
    assert_eq!(list.len(), 1, "{list:?}");
    let r: Vec<f64> = list[0]["rect"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
    assert!((r[2] - r[0] - 60.0).abs() < 1e-6 && (r[3] - r[1] - 60.0).abs() < 1e-6, "{r:?}");
    assert!(((r[0] + r[2]) / 2.0 - 200.0).abs() < 1e-6 && ((r[1] + r[3]) / 2.0 - 300.0).abs() < 1e-6, "{r:?}");
    assert!(page_text(&mut a, doc).contains("山田"), "the seal's name is text on the page");
    // It survives a save and reopen as a stamp with an appearance.
    ok(&mut a, "doc_save", json!({ "doc": doc, "path": "sealed.pdf" }));
    let saved = pdfcraft_cos::Document::open(std::sync::Arc::new(std::fs::read(dir.join("sealed.pdf")).unwrap())).unwrap();
    let has_ap = saved.scan_objects().any(|(_, o)| o.as_dict().is_some_and(|d| d.name(b"Subtype") == Some(&b"Stamp"[..]) && d.get(b"AP").is_some()));
    assert!(has_ap, "the saved Stamp has an appearance stream");
    let reopened = ok(&mut a, "doc_open", json!({ "path": "sealed.pdf" }))["doc"].as_u64().unwrap();
    assert!(page_text(&mut a, reopened).contains("山田"));
    // Delete it like any comment.
    ok(&mut a, "comment_delete", json!({ "doc": doc, "page": 1, "index": 1 }));
    assert!(stamps(&mut a, doc).is_empty());
    assert!(!page_text(&mut a, doc).contains("山田"));
    ok(&mut a, "edit_undo", json!({ "doc": doc }));
    assert_eq!(stamps(&mut a, doc).len(), 1, "undo brings it back");
}

/// A date seal carries the organisation, the date in the era, and the name, with today's date by
/// default.
#[test]
fn a_date_seal_has_the_organisation_date_and_name() {
    if !has_japanese_face() {
        eprintln!("skipping: built without a craft-fonts Jpan face");
        return;
    }
    let (mut a, doc, _dir) = open("date");
    let out = ok(
        &mut a,
        "page_add_seal",
        json!({ "doc": doc, "page": 1, "at": [100, 100], "kind": "date", "org": "営業部", "name": "鈴木", "date": "2026-11-11", "era": true }),
    );
    assert_eq!(out["seal"]["date"], "令和8.11.11", "{out}");
    let text = page_text(&mut a, doc);
    for want in ["営業部", "令和8.11.11", "鈴木"] {
        assert!(text.contains(want), "{want} in {text:?}");
    }
    // No date given: today by the session clock (2023-11-14), Western.
    let out = ok(
        &mut a,
        "page_add_seal",
        json!({ "doc": doc, "page": 1, "at": [300, 100], "kind": "date", "name": "佐藤", "shape": "square", "color": "#0000ff" }),
    );
    assert_eq!(out["seal"]["date"], "2023.11.14", "{out}");
    assert!(page_text(&mut a, doc).contains("2023.11.14"));
    assert_eq!(stamps(&mut a, doc).len(), 2);
}

/// An ASCII name needs no Japanese face.
#[test]
fn an_ascii_name_seal_works_in_any_build() {
    let (mut a, doc, _dir) = open("ascii");
    ok(&mut a, "page_add_seal", json!({ "doc": doc, "page": 1, "at": [100, 100], "name": "Smith" }));
    assert_eq!(stamps(&mut a, doc).len(), 1);
    assert!(page_text(&mut a, doc).contains("Smith"));
}

#[test]
fn bad_seal_arguments_are_refused_with_a_reason() {
    let (mut a, doc, _dir) = open("bad");
    let base = json!({ "doc": doc, "page": 1, "at": [100, 100], "name": "Ab" });
    let with = |k: &str, v: Value| {
        let mut b = base.clone();
        b[k] = v;
        b
    };
    for (why, args) in [
        ("size", with("size", json!(500))),
        ("kind", with("kind", json!("round"))),
        ("shape", with("shape", json!("triangle"))),
        ("date", with("date", json!("2026-02-30"))),
        ("colour", with("color", json!("not-a-colour"))),
        ("name", with("name", json!("   "))),
        ("long name", with("name", json!("Abcdefghijklmnopqrstuvwxyz"))),
        ("page", with("page", json!(9))),
    ] {
        let e = a.call("page_add_seal", &args).unwrap_err();
        assert!(matches!(e, ToolError::InvalidArgs(_)), "{why}: {e}");
    }
    assert!(stamps(&mut a, doc).is_empty(), "a refused seal changes nothing");
    // Without a Japanese face the refusal says so; with one, Japanese is drawn.
    let e = a.call("page_add_seal", &with("name", json!("山田")));
    if has_japanese_face() {
        assert!(e.is_ok());
    } else {
        let msg = e.unwrap_err().to_string();
        assert!(msg.contains("no Japanese font is embedded"), "{msg}");
    }
}
