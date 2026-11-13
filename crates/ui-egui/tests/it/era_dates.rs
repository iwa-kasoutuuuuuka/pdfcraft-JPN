//! Edit ▸ Convert dates (和暦/西暦の一括変換) in the real shell: the dialog lists the dates found,
//! the calendar is chosen, and Convert replaces them as one undo step.

use egui_kittest::{Harness, kittest::Queryable};
use pdfcraft_engine::{Edit, FieldValue, NewField, Session};
use pdfcraft_ui_egui::{Dialog, PdfCraftApp};

/// A page with two text fields holding an era date and a date that doesn't exist.
fn fixture() -> Vec<u8> {
    let mut s = Session::new();
    let bytes = s.create_blank(300.0, 400.0, 1).unwrap();
    let id = s.open("dates.pdf", None, bytes, None).unwrap();
    for (name, rect, value) in [("when", [40.0, 290.0, 230.0, 314.0], "R6.5.1"), ("wrong", [40.0, 240.0, 230.0, 264.0], "H31.5.1")] {
        s.apply(id, Edit::AddField { name: Some(name.into()), page: 0, rect, kind: NewField::Text { multiline: false } }).unwrap();
        s.apply(id, Edit::SetFieldValue { name: name.into(), value: FieldValue::Text(value.into()) }).unwrap();
    }
    s.save_bytes(id).unwrap().as_ref().clone()
}

fn value(h: &Harness<'static, PdfCraftApp>, name: &str) -> String {
    let s = h.state();
    let doc = s.session.get(s.views[0].id).unwrap();
    doc.form.iter().find(|f| f.name == name).unwrap().value.first().cloned().unwrap_or_default()
}

#[test]
fn convert_dates_dialog_converts_the_checked_dates_to_the_chosen_calendar() {
    if pdfcraft_fonts::document_japanese_font().is_none() {
        eprintln!("skipping: built without craft-fonts");
        return;
    }
    let mut h = Harness::builder().with_size(egui::vec2(1400.0, 900.0)).build_eframe(|_cc| {
        let mut app = PdfCraftApp::new();
        app.set_option("language", "en").unwrap();
        app.open_bytes("dates.pdf", None, fixture()).unwrap();
        app
    });
    crate::support::settle(&mut h);
    assert!(h.state_mut().execute("edit.convert_dates"));
    h.run_steps(3);
    assert_eq!(h.state().dialog, Some(Dialog::ConvertDates));
    // To the era calendar (the default) both are already era dates, or invalid.
    h.get_by_label("2 dates found, 0 to convert");
    h.get_by_label("Invalid date: the era does not include this day");
    h.get_by_label("Western (2024年5月1日)").click();
    h.run_steps(3);
    h.get_by_label("2 dates found, 1 to convert");
    h.get_by_label("→ 2024年5月1日");
    // Another "Convert" button sits behind the dialog; the dialog's is drawn last.
    h.get_all_by_role_and_label(egui::accesskit::Role::Button, "Convert").last().expect("the dialog's Convert button").click();
    h.run_steps(4);
    assert_eq!(value(&h, "when"), "2024年5月1日");
    assert_eq!(value(&h, "wrong"), "H31.5.1", "an invalid date is never converted");
    h.get_by_label("1 replaced, 0 left unchanged.");
    // The list follows the edit: nothing left to convert.
    h.get_by_label("2 dates found, 0 to convert");
    // One undo step brings it back.
    assert!(h.state_mut().execute("edit.undo"));
    h.run_steps(3);
    assert_eq!(value(&h, "when"), "R6.5.1");
    h.get_by_label("Close").click();
    h.run_steps(3);
    assert_eq!(h.state().dialog, None);
}
