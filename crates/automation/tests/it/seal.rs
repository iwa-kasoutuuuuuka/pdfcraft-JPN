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

fn rect_of(c: &Value) -> [f64; 4] {
    let v: Vec<f64> = c["rect"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
    [v[0], v[1], v[2], v[3]]
}

/// A business stamp lands as one Stamp comment centred where it was placed, reads as its word,
/// date and name, and undoes like any stamp.
#[test]
fn a_business_stamp_is_a_stamp_comment_with_its_words() {
    if !has_japanese_face() {
        eprintln!("skipping: built without a craft-fonts Jpan face");
        return;
    }
    let (mut a, doc, _dir) = open("business");
    let out = ok(
        &mut a,
        "page_add_business_stamp",
        json!({ "doc": doc, "page": 1, "at": [200, 300], "word": "承認", "date": "2026-11-13", "era": true, "name": "山田" }),
    );
    assert_eq!(out["stamp"]["text"], "承認 令和8.11.13 山田", "{out}");
    let list = stamps(&mut a, doc);
    assert_eq!(list.len(), 1, "{list:?}");
    let r = rect_of(&list[0]);
    assert!(((r[0] + r[2]) / 2.0 - 200.0).abs() < 0.01 && ((r[1] + r[3]) / 2.0 - 300.0).abs() < 0.01, "{r:?}");
    assert!(r[2] - r[0] >= r[3] - r[1], "wider than tall: {r:?}");
    let text = page_text(&mut a, doc);
    for want in ["承認", "令和8.11.13", "山田"] {
        assert!(text.contains(want), "{want} in {text:?}");
    }
    // Every ready-made word, with a double rule and today's date.
    for (i, word) in pdfcraft_engine::seal::business::BUSINESS_STAMP_WORDS.iter().enumerate() {
        let at = [100.0 + 40.0 * (i % 5) as f64, 500.0 + 60.0 * (i / 5) as f64];
        ok(
            &mut a,
            "page_add_business_stamp",
            json!({ "doc": doc, "page": 1, "at": at, "word": word, "frame": "double", "date": "today", "size": 12 }),
        );
    }
    assert_eq!(stamps(&mut a, doc).len(), 1 + pdfcraft_engine::seal::business::BUSINESS_STAMP_WORDS.len());
    assert!(page_text(&mut a, doc).contains("2023.11.14"), "today by the session clock");
    ok(&mut a, "edit_undo", json!({ "doc": doc }));
    assert_eq!(stamps(&mut a, doc).len(), pdfcraft_engine::seal::business::BUSINESS_STAMP_WORDS.len(), "undo takes the last one away");
}

#[test]
fn bad_business_stamp_arguments_are_refused_with_a_reason() {
    let (mut a, doc, _dir) = open("business-bad");
    let base = json!({ "doc": doc, "page": 1, "at": [100, 100], "word": "OK" });
    let with = |k: &str, v: Value| {
        let mut b = base.clone();
        b[k] = v;
        b
    };
    for (why, args) in [
        ("size", with("size", json!(100))),
        ("frame", with("frame", json!("wavy"))),
        ("date", with("date", json!("2026-02-30"))),
        ("colour", with("color", json!("not-a-colour"))),
        ("word", with("word", json!("  "))),
        ("page", with("page", json!(9))),
    ] {
        let e = a.call("page_add_business_stamp", &args).unwrap_err();
        assert!(matches!(e, ToolError::InvalidArgs(_)), "{why}: {e}");
    }
    assert!(stamps(&mut a, doc).is_empty(), "a refused stamp changes nothing");
    ok(&mut a, "page_add_business_stamp", base.clone());
    assert_eq!(stamps(&mut a, doc).len(), 1, "an ASCII word works in any build");
}

/// A document of `pages` pages (the first from text, the rest blank pages of the same size).
fn open_pages(test: &str, pages: usize) -> (Automation, u64, std::path::PathBuf) {
    let (mut a, doc, dir) = open(test);
    for at in 2..=pages {
        ok(&mut a, "page_insert_blank", json!({ "doc": doc, "at": at }));
    }
    (a, doc, dir)
}

/// 契印 across three pages: four half seals, page 2 carrying one at each edge, each half touching
/// the shared edge at the same height; one undo removes them all.
#[test]
fn a_tally_seal_puts_half_a_seal_on_each_side_of_every_page_edge() {
    let (mut a, doc, _dir) = open_pages("tally", 3);
    let width = ok(&mut a, "doc_info", json!({ "doc": doc }))["pages"][0]["width"].as_f64().unwrap();
    let out = ok(&mut a, "page_add_tally_seal", json!({ "doc": doc, "name": "Abe", "size": 60, "y": 200 }));
    assert_eq!(out["tally_seal"]["halves"], 4, "{out}");
    let list = stamps(&mut a, doc);
    assert_eq!(list.len(), 4, "{list:?}");
    let on = |p: u64| list.iter().filter(|c| c["page"] == p).map(rect_of).collect::<Vec<_>>();
    let (p1, p2, p3) = (on(1), on(2), on(3));
    assert_eq!((p1.len(), p2.len(), p3.len()), (1, 2, 1));
    let near = |a: f64, b: f64| (a - b).abs() < 1e-3;
    for r in p1.iter().chain(&p2).chain(&p3) {
        assert!(near(r[2] - r[0], 30.0) && near(r[3] - r[1], 60.0), "half the seal wide: {r:?}");
        assert!(near((r[1] + r[3]) / 2.0, 200.0), "the same height on every page: {r:?}");
    }
    assert!(near(p1[0][2], width), "page 1's half touches its right edge: {p1:?}");
    assert!(p2.iter().any(|r| near(r[0], 0.0)) && p2.iter().any(|r| near(r[2], width)), "page 2 has a half at each edge: {p2:?}");
    assert!(near(p3[0][0], 0.0), "page 3's half touches its left edge: {p3:?}");
    ok(&mut a, "edit_undo", json!({ "doc": doc }));
    assert!(stamps(&mut a, doc).is_empty(), "one undo removes every half");

    // Right to left, pages 2 and 3 only: page 2's half is at its left edge.
    ok(&mut a, "page_add_tally_seal", json!({ "doc": doc, "name": "Abe", "first": 2, "last": 3, "direction": "rtl" }));
    let list = stamps(&mut a, doc);
    let r2: Vec<[f64; 4]> = list.iter().filter(|c| c["page"] == 2).map(rect_of).collect();
    let r3: Vec<[f64; 4]> = list.iter().filter(|c| c["page"] == 3).map(rect_of).collect();
    assert!(r2.len() == 1 && near(r2[0][0], 0.0), "{r2:?}");
    assert!(r3.len() == 1 && near(r3[0][2], width), "{r3:?}");
}

#[test]
fn bad_tally_seal_arguments_are_refused_with_a_reason() {
    let (mut a, doc, _dir) = open_pages("tally-bad", 2);
    let base = json!({ "doc": doc, "name": "Abe" });
    let with = |k: &str, v: Value| {
        let mut b = base.clone();
        b[k] = v;
        b
    };
    for (why, args) in [
        ("one page", with("last", json!(1))),
        ("out of range", with("last", json!(5))),
        ("too high", with("y", json!(5))),
        ("direction", with("direction", json!("up"))),
        ("size", with("size", json!(1000))),
        ("name", with("name", json!(" "))),
    ] {
        let e = a.call("page_add_tally_seal", &args).unwrap_err();
        assert!(matches!(e, ToolError::InvalidArgs(_)), "{why}: {e}");
    }
    assert!(stamps(&mut a, doc).is_empty(), "a refused tally seal changes nothing");
}
