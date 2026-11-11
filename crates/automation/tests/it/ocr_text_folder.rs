//! `export_text` (recognised or extracted text as .txt / .docx) and `ocr_folder` (searchable
//! copies of a folder of PDFs), end to end on synthetic documents.

use std::sync::Arc;

use pdfcraft_automation::{Automation, Content, ToolError};
use pdfcraft_engine::ocr::PlacedWord;
use pdfcraft_engine::{Edit, Session, export};
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
    let dir = std::env::temp_dir().join(format!("pdfcraft-ocr-text-folder-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// `n` blank 200 x 300 pt pages.
fn blank(n: usize) -> Vec<u8> {
    let kids: Vec<String> = (0..n).map(|i| format!("{} 0 R", 3 + i)).collect();
    let mut objs = vec![
        "<< /Type /Catalog /Pages 2 0 R >>".to_string(),
        format!("<< /Type /Pages /Kids [{}] /Count {n} /MediaBox [0 0 200 300] >>", kids.join(" ")),
    ];
    objs.extend((0..n).map(|_| "<< /Type /Page /Parent 2 0 R >>".to_string()));
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

fn horizontal(text: &str, y: f64) -> PlacedWord {
    PlacedWord {
        text: text.into(),
        origin: [20.0, y],
        across: [12.0 * text.chars().count() as f64, 0.0],
        up: [0.0, 12.0],
        confidence: None,
        source: "test".into(),
    }
}

fn column(text: &str, x: f64) -> PlacedWord {
    let h = 12.0 * text.chars().count() as f64;
    PlacedWord { text: text.into(), origin: [x, 250.0 - h], across: [12.0, 0.0], up: [0.0, h], confidence: None, source: "test".into() }
}

/// Japanese text written the way recognised text is: page 1 horizontal (two paragraphs), page 2
/// vertical (two columns, the right one first), page 3 horizontal.
fn japanese_pdf() -> Vec<u8> {
    let mut s = Session::new().with_clock(|| 1_700_000_000);
    let id = s.open("ja.pdf", None, Arc::new(blank(3)), None).unwrap();
    let pages = [
        vec![horizontal("最初の段落の一行目です", 250.0), horizontal("最初の段落の二行目です", 237.0), horizontal("二つ目の段落", 190.0)],
        vec![column("右の列の文章", 150.0), column("左の列の文章", 132.0)],
        vec![horizontal("三ページ目の文章", 250.0)],
    ];
    for (page, words) in pages.into_iter().enumerate() {
        s.apply(id, Edit::AddOcrText { page, words }).unwrap();
    }
    s.save_bytes(id).unwrap().to_vec()
}

/// One part of a package written by the export crate's zip writer, inflated.
fn part(zip: &[u8], name: &str) -> String {
    let at = |i: usize, n: usize| zip[i..i + n].iter().rev().fold(0usize, |v, b| v << 8 | *b as usize);
    let mut i = 0;
    while zip.get(i..i + 4) == Some(b"PK\x03\x04") {
        let (size, name_len) = (at(i + 18, 4), at(i + 26, 2));
        let data = i + 30 + name_len;
        if &zip[i + 30..data] == name.as_bytes() {
            let mut s = String::new();
            std::io::Read::read_to_string(&mut flate2::read::DeflateDecoder::new(&zip[data..data + size]), &mut s).unwrap();
            return s;
        }
        i = data + size;
    }
    panic!("{name} missing")
}

fn find_in_order(text: &str, parts: &[&str]) {
    let mut from = 0;
    for p in parts {
        let at = text[from..].find(p).unwrap_or_else(|| panic!("{p} missing (or out of order) in {text}"));
        from += at + p.len();
    }
}

#[test]
fn export_text_writes_txt_and_docx_in_reading_order_with_japanese_tags() {
    let dir = workdir("export");
    std::fs::write(dir.join("ja.pdf"), japanese_pdf()).unwrap();
    let mut a = Automation::new().with_root(&dir).unwrap();
    let doc = ok(&mut a, "doc_open", json!({ "path": "ja.pdf" }))["doc"].as_u64().unwrap();

    // Plain text: UTF-8, page separators, paragraphs, vertical columns right to left.
    let r = ok(&mut a, "export_text", json!({ "doc": doc, "path": "out.txt" }));
    assert_eq!((r["format"].as_str(), r["pages"].as_u64(), r["paragraphs"].as_u64()), (Some("txt"), Some(3), Some(4)), "{r}");
    assert_eq!(r["vertical_pages"], json!([2]), "{r}");
    let txt = std::fs::read_to_string(dir.join("out.txt")).unwrap();
    find_in_order(
        &txt,
        &[
            "===== 1 / 3 =====",
            "最初の段落の一行目です最初の段落の二行目です",
            "二つ目の段落",
            "===== 2 / 3 =====",
            "右の列の文章左の列の文章",
            "===== 3 / 3 =====",
            "三ページ目の文章",
        ],
    );
    // A subset of pages, with form feeds.
    ok(&mut a, "export_text", json!({ "doc": doc, "path": "ff.txt", "pages": [1, 3], "separator": "formfeed" }));
    let ff = std::fs::read_to_string(dir.join("ff.txt")).unwrap();
    assert!(ff.contains('\u{c}') && !ff.contains("右の列") && !ff.contains("====="), "{ff:?}");

    // Word: text in order, East Asian font and language, one vertical section (auto).
    let r = ok(&mut a, "export_text", json!({ "doc": doc, "path": "out.docx" }));
    assert_eq!(r["format"], "docx", "{r}");
    let bytes = std::fs::read(dir.join("out.docx")).unwrap();
    let xml = part(&bytes, "word/document.xml");
    find_in_order(&xml, &["最初の段落の一行目です最初の段落の二行目です", "二つ目の段落", "右の列の文章左の列の文章", "三ページ目の文章"]);
    assert_eq!(xml.matches("w:textDirection w:val=\"tbRl\"").count(), 1, "{xml}");
    let styles = part(&bytes, "word/styles.xml");
    assert!(styles.contains("w:eastAsia=\"ja-JP\""), "{styles}");
    assert!(styles.contains("w:eastAsia=\"Yu Mincho\""), "{styles}");
    // vertical: off writes none, on writes a section per page.
    ok(&mut a, "export_text", json!({ "doc": doc, "path": "off.docx", "vertical": "off" }));
    assert!(!part(&std::fs::read(dir.join("off.docx")).unwrap(), "word/document.xml").contains("tbRl"));
    ok(&mut a, "export_text", json!({ "doc": doc, "path": "on.docx", "vertical": "on" }));
    assert_eq!(part(&std::fs::read(dir.join("on.docx")).unwrap(), "word/document.xml").matches("tbRl").count(), 3);
    // The document is not changed by exporting.
    assert_eq!(ok(&mut a, "doc_list", json!({}))["documents"][0]["dirty"], false);

    // Refusals say what is allowed.
    for (args, needle) in [
        (json!({ "doc": doc, "path": "x.xyz" }), "format"),
        (json!({ "doc": doc, "path": "x.txt", "format": "pdf" }), "format"),
        (json!({ "doc": doc, "path": "x.docx", "vertical": "sideways" }), "vertical"),
        (json!({ "doc": doc, "path": "x.txt", "pages": [9] }), "page 9"),
        (json!({ "doc": doc, "path": "../x.txt" }), "outside"),
    ] {
        let err = a.call("export_text", &args).unwrap_err();
        assert!(err.to_string().contains(needle), "{args}: {err}");
    }
    assert!(!dir.join("x.txt").exists());
}

#[test]
fn export_text_reports_empty_pages_and_reads_scanned_pages_when_asked() {
    let dir = workdir("export-ocr");
    std::fs::write(dir.join("blank.pdf"), blank(1)).unwrap();
    let mut a = Automation::new().with_root(&dir).unwrap();
    let doc = ok(&mut a, "doc_open", json!({ "path": "blank.pdf" }))["doc"].as_u64().unwrap();
    let r = ok(&mut a, "export_text", json!({ "doc": doc, "path": "blank.txt" }));
    assert_eq!(r["empty_pages"], json!([1]), "{r}");
    assert!(r["warnings"][0].as_str().unwrap().contains("OCR"), "{r}");
    if ok(&mut a, "ocr_status", json!({}))["available"] != true {
        // No engine: asking for OCR is an honest error, not an empty file.
        let err =
            a.call("export_text", &json!({ "doc": doc, "path": "e.txt", "ocr_if_needed": true, "engine": "ocrs", "language": "en" })).unwrap_err();
        assert!(matches!(err, ToolError::Failed(_)), "{err}");
        assert!(!dir.join("e.txt").exists());
        eprintln!("skipped: OCR models not installed");
        return;
    }
    // A picture of a sentence is read first, and the open document stays unchanged.
    let mut s = Session::new();
    let text = s.create_from_text("t", "Searchable scans with recognised words").unwrap();
    let id = s.open("text.pdf", None, text, None).unwrap();
    let png = export::Exporter::new(s.get(id).unwrap()).png(0, 150.0).unwrap();
    std::fs::write(dir.join("scan.pdf"), s.create_from_images(&[("scan.png".into(), png)]).unwrap().to_vec()).unwrap();
    let scan = ok(&mut a, "doc_open", json!({ "path": "scan.pdf" }))["doc"].as_u64().unwrap();
    let r = ok(&mut a, "export_text", json!({ "doc": scan, "path": "scan.txt", "ocr_if_needed": true, "engine": "ocrs", "language": "en" }));
    assert_eq!(r["ocr_pages"], json!([1]), "{r}");
    assert!(std::fs::read_to_string(dir.join("scan.txt")).unwrap().to_lowercase().contains("searchable"));
    assert_eq!(ok(&mut a, "doc_list", json!({}))["documents"][0]["dirty"], false);
    assert_eq!(ok(&mut a, "doc_list", json!({}))["documents"][1]["dirty"], false);
}

/// A one-page PDF that is only a picture of `sentence`.
fn scan_pdf(sentence: &str) -> Vec<u8> {
    let mut s = Session::new();
    let text = s.create_from_text("t", sentence).unwrap();
    let id = s.open("text.pdf", None, text, None).unwrap();
    let png = export::Exporter::new(s.get(id).unwrap()).png(0, 150.0).unwrap();
    s.create_from_images(&[("scan.png".into(), png)]).unwrap().to_vec()
}

fn entry<'a>(report: &'a Value, name: &str) -> &'a Value {
    report["files"]
        .as_array()
        .unwrap()
        .iter()
        .find(|f| f["path"].as_str().unwrap().replace('\\', "/").ends_with(name))
        .unwrap_or_else(|| panic!("{name} not in {report}"))
}

#[test]
fn ocr_folder_skips_text_files_reports_bad_ones_and_never_touches_the_sources() {
    let dir = workdir("folder");
    std::fs::create_dir_all(dir.join("in/sub")).unwrap();
    let typed = Session::new().create_from_text("t", "Already typed text").unwrap().to_vec();
    std::fs::write(dir.join("in/typed.pdf"), &typed).unwrap();
    std::fs::write(dir.join("in/broken.pdf"), b"%PDF-1.4 truncated").unwrap();
    std::fs::write(dir.join("in/notes.txt"), b"not a pdf").unwrap();
    std::fs::write(dir.join("in/sub/typed2.pdf"), &typed).unwrap();
    let mut a = Automation::new().with_root(&dir).unwrap();

    // No file needs an engine, so this runs everywhere. Not recursive: the subfolder is left alone.
    let r = ok(&mut a, "ocr_folder", json!({ "folder": "in" }));
    assert_eq!((r["ok"].as_u64(), r["skipped"].as_u64(), r["failed"].as_u64()), (Some(0), Some(1), Some(1)), "{r}");
    assert_eq!(entry(&r, "typed.pdf")["status"], "skipped", "{r}");
    assert!(entry(&r, "typed.pdf")["reason"].as_str().unwrap().contains("text layer"), "{r}");
    assert_eq!(entry(&r, "broken.pdf")["status"], "failed", "{r}");
    assert!(!entry(&r, "broken.pdf")["reason"].as_str().unwrap().is_empty());
    let r = ok(&mut a, "ocr_folder", json!({ "folder": "in", "recursive": true, "output_dir": "out" }));
    assert_eq!((r["skipped"].as_u64(), r["failed"].as_u64()), (Some(2), Some(1)), "{r}");
    assert_eq!(std::fs::read(dir.join("in/typed.pdf")).unwrap(), typed);
    assert!(!dir.join("out/typed.pdf").exists(), "nothing is written for a skipped file");

    // Refusals: outside the root, not a folder, too many files, replacing sources without saying so.
    for (args, needle) in [
        (json!({ "folder": "../elsewhere" }), "outside"),
        (json!({ "folder": "in", "output_dir": "../elsewhere" }), "outside"),
        (json!({ "folder": "in/typed.pdf" }), "folder"),
        (json!({ "folder": "in", "max_files": 1, "recursive": true }), "more than 1 PDF"),
        (json!({ "folder": "in", "suffix": "" }), "overwrite"),
        (json!({ "folder": "in", "suffix": "../x" }), "suffix"),
        (json!({ "folder": "in", "language": "xx" }), "language"),
    ] {
        let err = a.call("ocr_folder", &args).unwrap_err();
        assert!(err.to_string().contains(needle) || matches!(err, ToolError::InvalidArgs(_)), "{args}: {err}");
    }
    assert_eq!(std::fs::read(dir.join("in/typed.pdf")).unwrap(), typed);
}

#[test]
fn ocr_folder_makes_scans_searchable_beside_the_sources_or_in_an_output_folder() {
    let dir = workdir("folder-scans");
    let mut a = Automation::new().with_root(&dir).unwrap();
    if ok(&mut a, "ocr_status", json!({}))["available"] != true {
        eprintln!("skipped: OCR models not installed");
        return;
    }
    std::fs::create_dir_all(dir.join("in/sub")).unwrap();
    let scan = scan_pdf("Folder recognition works well");
    std::fs::write(dir.join("in/scan.pdf"), &scan).unwrap();
    std::fs::write(dir.join("in/sub/scan2.pdf"), scan_pdf("Second scanned sentence here")).unwrap();
    std::fs::write(dir.join("in/typed.pdf"), Session::new().create_from_text("t", "Already typed").unwrap().to_vec()).unwrap();
    std::fs::write(dir.join("in/broken.pdf"), b"garbage").unwrap();
    let base = json!({ "folder": "in", "recursive": true, "engine": "ocrs", "language": "en" });
    let with = |extra: Value| {
        let mut args = base.clone();
        args.as_object_mut().unwrap().extend(extra.as_object().unwrap().clone());
        args
    };
    let r = ok(&mut a, "ocr_folder", base.clone());
    assert_eq!((r["ok"].as_u64(), r["skipped"].as_u64(), r["failed"].as_u64()), (Some(2), Some(1), Some(1)), "{r}");
    assert_eq!(std::fs::read(dir.join("in/scan.pdf")).unwrap(), scan, "the source is not touched");
    let out = ok(&mut a, "doc_open", json!({ "path": "in/scan_ocr.pdf" }))["doc"].as_u64().unwrap();
    assert!(ok(&mut a, "text_find", json!({ "doc": out, "query": "recognition" })).to_string().contains("\"page\""));
    assert!(dir.join("in/sub/scan2_ocr.pdf").exists());
    // A rerun leaves the results alone: nothing is read twice or replaced.
    let again = ok(&mut a, "ocr_folder", base.clone());
    assert_eq!(again["ok"], 0, "{again}");
    // Another output folder, same names.
    let r = ok(&mut a, "ocr_folder", with(json!({ "output_dir": "out" })));
    assert_eq!(r["ok"], 2, "{r}");
    assert!(dir.join("out/scan.pdf").exists() && dir.join("out/sub/scan2.pdf").exists());
    // Replacing the source takes both an empty suffix and overwrite.
    let r = ok(&mut a, "ocr_folder", with(json!({ "suffix": "", "overwrite": true })));
    assert_eq!(r["ok"], 2, "{r}");
    assert_ne!(std::fs::read(dir.join("in/scan.pdf")).unwrap(), scan);
}
