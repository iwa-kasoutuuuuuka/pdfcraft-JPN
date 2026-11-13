//! Scan & OCR ▸ Export tables to CSV in the real shell (egui_kittest).

use std::sync::Arc;

use egui_kittest::Harness;
use pdfcraft_engine::ocr::PlacedWord;
use pdfcraft_engine::table_csv::CsvLayout;
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

/// A page whose recognised text is a two-by-two table.
fn table_pdf() -> Vec<u8> {
    let word = |text: &str, x: f64, top: f64| PlacedWord {
        text: text.into(),
        origin: [x, 290.0 - top],
        across: [10.0 * text.chars().count() as f64, 0.0],
        up: [0.0, 10.0],
        confidence: None,
        source: "test".into(),
    };
    let mut s = Session::new().with_clock(|| 1_700_000_000);
    let id = s.open("t.pdf", None, Arc::new(blank()), None).unwrap();
    let words = vec![word("氏名", 20.0, 20.0), word("年齢", 100.0, 20.0), word("山田", 20.0, 34.0), word("30", 100.0, 34.0)];
    s.apply(id, Edit::AddOcrText { page: 0, words }).unwrap();
    s.save_bytes(id).unwrap().to_vec()
}

#[test]
fn export_tables_to_csv_writes_the_chosen_file() {
    let dir = std::env::temp_dir().join(format!("pdfcraft-table-csv-ui-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let target = dir.join("表.csv");
    let pick = target.to_string_lossy().into_owned();
    let mut h = Harness::builder().with_size(egui::vec2(1400.0, 900.0)).build_eframe(move |_cc| {
        let mut app = PdfCraftApp::new();
        app.set_option("language", "en").unwrap();
        app.open_bytes("t.pdf", None, table_pdf()).unwrap();
        app.run_inline = true;
        app.pick_override = Some(vec![pick]);
        app
    });
    h.run_steps(4);
    assert!(h.state_mut().execute("ocr.export_tables_csv"));
    h.run_steps(2);
    assert!(matches!(h.state().dialog, Some(Dialog::ExportTablesCsv)), "{:?}", h.state().dialog);
    h.state_mut().table_csv_draft.ocr_if_needed = false;
    h.state_mut().table_csv_draft.layout = CsvLayout::OneFile;
    h.state_mut().start_table_csv_export();
    h.run_steps(4);
    assert_eq!(std::fs::read_to_string(&target).unwrap(), "\u{feff}氏名,年齢\r\n山田,30\r\n");
    let toast = h.state().toast.clone().map(|t| t.0).unwrap_or_default();
    assert!(toast.starts_with("Exported 1 tables as 1 CSV files"), "{toast}");
    // One file per table, beside the chosen name.
    h.state_mut().pick_override = Some(vec![target.to_string_lossy().into_owned()]);
    h.state_mut().table_csv_draft.layout = CsvLayout::PerTable;
    h.state_mut().start_table_csv_export();
    h.run_steps(4);
    assert!(dir.join("表_p1_t1.csv").exists());
}
