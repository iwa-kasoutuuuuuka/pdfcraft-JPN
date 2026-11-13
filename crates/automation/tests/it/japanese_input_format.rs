//! Japanese input formats on text fields (入力の自動そろえ): `form_set_props` gives a field a
//! format such as 全角カナ, and `form_fill` converts what is filled in on commit.

use pdfcraft_automation::{Automation, Content, ToolError};
use pdfcraft_engine::Session;
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

fn open(test: &str) -> (Automation, u64, std::path::PathBuf) {
    let dir = std::env::temp_dir().join(format!("pdfcraft-ja-format-{test}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let bytes = Session::new().create_blank(300.0, 400.0, 1).unwrap();
    std::fs::write(dir.join("form.pdf"), bytes.as_ref()).unwrap();
    let mut a = Automation::new().with_root(&dir).unwrap().with_clock(|| 1_700_000_000);
    let doc = ok(&mut a, "doc_open", json!({ "path": "form.pdf" }))["doc"].as_u64().unwrap();
    for (name, y) in [("kana", 40.0), ("tel", 80.0)] {
        ok(&mut a, "form_add_field", json!({ "doc": doc, "page": 1, "type": "text", "rect": [30, y, 230, y + 24.0], "name": name }));
    }
    (a, doc, dir)
}

fn field(a: &mut Automation, doc: u64, name: &str) -> Value {
    let fields = ok(a, "form_fields", json!({ "doc": doc }))["fields"].as_array().unwrap().clone();
    fields.into_iter().find(|f| f["name"] == name).unwrap_or_else(|| panic!("no field {name}"))
}

#[test]
fn a_katakana_field_converts_half_width_kana_and_keeps_its_format_after_save() {
    let (mut a, doc, dir) = open("katakana");
    ok(&mut a, "form_set_props", json!({ "doc": doc, "field": "kana", "format": { "type": "japanese", "width": "katakana" } }));
    ok(&mut a, "form_set_props", json!({ "doc": doc, "field": "tel", "format": { "type": "japanese", "width": "halfwidth_digits" } }));
    assert_eq!(field(&mut a, doc, "kana")["format"], json!({ "type": "japanese", "width": "katakana" }));

    ok(&mut a, "form_fill", json!({ "doc": doc, "values": { "kana": "ｶﾞｯｺｳ", "tel": "０３－１２３４ー５６７８" } }));
    assert_eq!(field(&mut a, doc, "kana")["value"], "ガッコウ");
    assert_eq!(field(&mut a, doc, "tel")["value"], "03-1234-5678");

    // Saved and opened again, the format is still there and still converts.
    ok(&mut a, "doc_save", json!({ "doc": doc, "path": "saved.pdf" }));
    // In the file: a keystroke script calling PdfCraftJa_Keystroke, and the format named in the field.
    let saved = pdfcraft_cos::Document::open(std::sync::Arc::new(std::fs::read(dir.join("saved.pdf")).unwrap())).unwrap();
    let kana = saved
        .scan_objects()
        .find_map(|(_, o)| o.as_dict().filter(|d| d.name(b"PdfCraftJaFormat") == Some(&b"katakana"[..])).cloned())
        .expect("the field names its format");
    let aa = saved.resolve(kana.get(b"AA").unwrap()).as_dict().cloned().unwrap();
    let k = saved.resolve(aa.get(b"K").unwrap()).as_dict().cloned().unwrap();
    let js = saved.resolve(k.get(b"JS").unwrap()).as_string().unwrap().to_text();
    assert!(js.contains("PdfCraftJa_Keystroke(\"katakana\");"), "{js}");
    assert!(aa.get(b"F").is_none(), "no format script: the value shows as stored");
    let re = ok(&mut a, "doc_open", json!({ "path": "saved.pdf" }))["doc"].as_u64().unwrap();
    assert_eq!(field(&mut a, re, "kana")["value"], "ガッコウ");
    assert_eq!(field(&mut a, re, "kana")["format"], json!({ "type": "japanese", "width": "katakana" }));
    ok(&mut a, "form_fill", json!({ "doc": re, "values": { "kana": "やまだ ﾀﾛｳ" } }));
    assert_eq!(field(&mut a, re, "kana")["value"], "ヤマダ タロウ");

    // Taking the format away takes its name out of the field too: filling keeps what is typed.
    ok(&mut a, "form_set_props", json!({ "doc": re, "field": "kana", "format": "none" }));
    ok(&mut a, "form_fill", json!({ "doc": re, "values": { "kana": "ｶﾞｯｺｳ" } }));
    assert_eq!(field(&mut a, re, "kana")["value"], "ｶﾞｯｺｳ");
    ok(&mut a, "doc_save", json!({ "doc": re, "path": "plain.pdf" }));
    let plain = ok(&mut a, "doc_open", json!({ "path": "plain.pdf" }))["doc"].as_u64().unwrap();
    assert_eq!(field(&mut a, plain, "kana")["format"], Value::Null);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn an_unknown_width_is_refused_with_the_allowed_ones() {
    let (mut a, doc, dir) = open("unknown");
    for format in [json!({ "type": "japanese", "width": "kana" }), json!({ "type": "japanese" })] {
        match a.call("form_set_props", &json!({ "doc": doc, "field": "kana", "format": format })) {
            Err(ToolError::InvalidArgs(m)) => assert!(m.contains("katakana") && m.contains("halfwidth_digits"), "{m}"),
            other => panic!("expected InvalidArgs, got {other:?}"),
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
}
