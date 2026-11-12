//! Scan & OCR ▸ Export recognized text and ▸ OCR a folder in the real shell (egui_kittest).

use std::sync::Arc;

use egui_kittest::Harness;
use pdfcraft_engine::ocr::PlacedWord;
use pdfcraft_engine::text_export::TextFormat;
use pdfcraft_engine::{Edit, Session};
use pdfcraft_ui_egui::{Dialog, PdfCraftApp};

/// One blank 200 x 300 pt page.
fn blank() -> Vec<u8> {
    let objs = [
        "<< /Type /Catalog /Pages 2 0 R >>".to_string(),
        "<< /Type /Pages /Kids [3 0 R] /Count 1 /MediaBox [0 0 200 300] >>".to_string(),
        "<< /Type /Page /Parent 2 0 R >>".to_string(),
    ];
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

/// A page whose text layer says 日本語の文章 (a recognised scan).
fn japanese() -> Vec<u8> {
    let mut s = Session::new().with_clock(|| 1_700_000_000);
    let id = s.open("ja.pdf", None, Arc::new(blank()), None).unwrap();
    let word = PlacedWord {
        text: "日本語の文章です".into(),
        origin: [20.0, 250.0],
        across: [96.0, 0.0],
        up: [0.0, 12.0],
        confidence: None,
        source: "test".into(),
    };
    s.apply(id, Edit::AddOcrText { page: 0, words: vec![word] }).unwrap();
    s.save_bytes(id).unwrap().to_vec()
}

fn temp(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("pdfcraft-ocr-text-ui-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

#[test]
fn export_recognized_text_saves_a_word_file_to_the_chosen_path() {
    let dir = temp("export");
    let target = dir.join("out.docx");
    let pick = target.to_string_lossy().into_owned();
    let mut h = Harness::builder().with_size(egui::vec2(1400.0, 900.0)).build_eframe(move |_cc| {
        let mut app = PdfCraftApp::new();
        app.set_option("language", "en").unwrap();
        app.open_bytes("ja.pdf", None, japanese()).unwrap();
        app.run_inline = true;
        app.pick_override = Some(vec![pick]);
        app
    });
    h.run_steps(4);
    assert!(h.state_mut().execute("ocr.export_text"));
    h.run_steps(2);
    assert!(matches!(h.state().dialog, Some(Dialog::ExportOcrText)), "{:?}", h.state().dialog);
    assert_eq!(h.state().ocr_export_draft.format, TextFormat::Docx);
    h.state_mut().ocr_export_draft.ocr_if_needed = false;
    h.state_mut().start_ocr_export();
    h.run_steps(4);
    let bytes = std::fs::read(&target).unwrap();
    assert!(bytes.starts_with(b"PK"), "a Word package");
    let toast = h.state().toast.clone().map(|t| t.0).unwrap_or_default();
    assert!(toast.starts_with("Exported the text of 1 pages"), "{toast}");
    // Plain text through the same dialog.
    let txt = dir.join("out.txt");
    h.state_mut().pick_override = Some(vec![txt.to_string_lossy().into_owned()]);
    h.state_mut().ocr_export_draft.format = TextFormat::Txt;
    h.state_mut().start_ocr_export();
    h.run_steps(4);
    assert_eq!(std::fs::read_to_string(&txt).unwrap().trim(), "日本語の文章です");
}

#[test]
fn ocr_a_folder_reports_each_file_and_keeps_going_past_a_bad_one() {
    let dir = temp("folder");
    let typed = Session::new().create_from_text("t", "Already typed").unwrap().to_vec();
    std::fs::write(dir.join("typed.pdf"), &typed).unwrap();
    std::fs::write(dir.join("broken.pdf"), b"%PDF-1.4 truncated").unwrap();
    let mut h = Harness::builder().with_size(egui::vec2(1400.0, 900.0)).build_eframe(|_cc| {
        let mut app = PdfCraftApp::new();
        app.set_option("language", "en").unwrap();
        app.run_inline = true;
        app
    });
    h.run_steps(2);
    assert!(h.state_mut().execute("ocr.folder"));
    h.run_steps(2);
    assert!(matches!(h.state().dialog, Some(Dialog::OcrFolder)), "{:?}", h.state().dialog);
    h.state_mut().ocr_folder.folder = Some(dir.clone());
    h.state_mut().start_ocr_folder();
    h.run_steps(3);
    assert!(h.state().ocr_folder.run.is_none(), "finished");
    let report = h.state().ocr_folder.report.clone().unwrap().unwrap();
    assert_eq!((report.ok, report.skipped, report.failed), (0, 1, 1), "{report:?}");
    assert_eq!(std::fs::read(dir.join("typed.pdf")).unwrap(), typed, "the original is untouched");
    assert!(!dir.join("typed_ocr.pdf").exists());
}
