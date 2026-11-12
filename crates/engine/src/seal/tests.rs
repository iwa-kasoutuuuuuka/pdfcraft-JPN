use std::sync::Arc;

use super::*;
use crate::{Edit, Session};

fn has_japanese_face() -> bool {
    pdfcraft_fonts::document_japanese_font_for_style(false, false).is_some()
}

fn name_seal(name: &str, shape: SealShape) -> SealSpec {
    SealSpec { name: name.into(), shape, size: 100.0, ..SealSpec::default() }
}

fn date_seal(org: &str, name: &str, era: bool) -> SealSpec {
    SealSpec { kind: SealKind::Date, org: org.into(), name: name.into(), date: (2026, 11, 11), era, size: 100.0, ..SealSpec::default() }
}

/// Every ink box of the layout lies inside the outline's inner edge.
fn assert_inside(l: &SealLayout) {
    let inner = l.inner();
    for t in &l.texts {
        for b in &t.ink {
            for (x, y) in [(b[0], b[1]), (b[2], b[1]), (b[0], b[3]), (b[2], b[3])] {
                let (dx, dy) = (x - l.centre[0], y - l.centre[1]);
                let inside = match l.shape {
                    SealShape::Circle => dx.hypot(dy) <= inner + 1e-9,
                    SealShape::Square => dx.abs() <= inner + 1e-9 && dy.abs() <= inner + 1e-9,
                };
                assert!(inside, "{:?} corner ({x}, {y}) is outside the {:?} outline (inner {inner})", t.text, l.shape);
            }
        }
    }
}

fn top(t: &SealText) -> f64 {
    t.ink.first().map_or(0.0, |b| b[3])
}

fn bottom(t: &SealText) -> f64 {
    t.ink.first().map_or(0.0, |b| b[1])
}

#[test]
fn dates_are_western_or_in_the_era() {
    assert_eq!(seal_date_text((2026, 11, 11), false).unwrap(), "2026.11.11");
    assert_eq!(seal_date_text((2026, 11, 11), true).unwrap(), "令和8.11.11");
    assert_eq!(seal_date_text((2019, 5, 1), true).unwrap(), "令和元.5.1", "the first year is 元");
    assert_eq!(seal_date_text((2019, 4, 30), true).unwrap(), "平成31.4.30");
    assert!(seal_date_text((2026, 2, 30), false).is_err());
    assert!(seal_date_text((2026, 13, 1), false).is_err());
    assert!(seal_date_text((1800, 1, 1), true).is_err(), "before Meiji there is no era");
    assert_eq!(parse_seal_date("2026年11月11日"), Some((2026, 11, 11)));
    assert_eq!(parse_seal_date("2026-1-5"), Some((2026, 1, 5)));
    assert_eq!(parse_seal_date("2026-02-30"), None);
    assert_eq!(parse_seal_date("tomorrow"), None);
    // An era year or a two-digit year is not taken for a year of the common era.
    assert_eq!(parse_seal_date("令和8年11月11日"), None);
    assert_eq!(parse_seal_date("R8.11.11"), None);
    assert_eq!(parse_seal_date("26-11-11"), None);
}

#[test]
fn a_name_is_set_vertically_and_fits_inside_the_circle() {
    if !has_japanese_face() {
        eprintln!("skipping: built without a craft-fonts Jpan face");
        return;
    }
    for (name, columns) in [("田", 1), ("山田", 1), ("鈴木太", 1), ("佐藤花子", 2), ("山田太郎丸", 2), ("渡辺美智子", 2)] {
        for shape in [SealShape::Circle, SealShape::Square] {
            let l = seal_layout(&name_seal(name, shape)).unwrap();
            assert_inside(&l);
            let [t] = l.texts.as_slice() else { panic!("one text") };
            assert!(t.vertical);
            assert_eq!(t.text.split('\n').count(), columns, "{name}");
            assert_eq!(t.text.replace('\n', ""), name);
            assert_eq!(t.ink.len(), name.chars().count());
            // The text is not tiny: it uses a good share of the outline.
            assert!(t.size > 0.2 * 2.0 * l.inner() / name.chars().count() as f64, "{name} {}", t.size);
        }
    }
    // A longer name is set smaller than a shorter one.
    let two = seal_layout(&name_seal("山田", SealShape::Circle)).unwrap();
    let four = seal_layout(&name_seal("佐藤花子", SealShape::Circle)).unwrap();
    assert!(four.texts[0].size < two.texts[0].size);
    // A seal's size scales everything.
    let small = seal_layout(&SealSpec { size: 50.0, ..name_seal("山田", SealShape::Circle) }).unwrap();
    assert!((small.texts[0].size * 2.0 - two.texts[0].size).abs() < 1e-9);
}

#[test]
fn a_latin_name_is_one_line_inside_the_outline() {
    let l = seal_layout(&name_seal("Smith", SealShape::Circle)).unwrap();
    assert_inside(&l);
    assert!(!l.texts[0].vertical);
    let l = seal_layout(&name_seal("Tanaka T", SealShape::Square)).unwrap();
    assert_inside(&l);
}

#[test]
fn a_date_seal_has_three_bands_in_order() {
    if !has_japanese_face() {
        eprintln!("skipping: built without a craft-fonts Jpan face");
        return;
    }
    let l = seal_layout(&date_seal("営業部", "山田", true)).unwrap();
    assert_inside(&l);
    let texts: Vec<&str> = l.texts.iter().map(|t| t.text.as_str()).collect();
    assert_eq!(texts, ["営業部", "令和8.11.11", "山田"], "top, middle, bottom");
    assert_eq!(l.rules.len(), 2, "two rules");
    let [upper, lower] = [l.rules[0][1], l.rules[1][1]];
    assert!(upper > l.centre[1] && lower < l.centre[1]);
    // Each text sits in its own band.
    assert!(bottom(&l.texts[0]) > upper, "the organisation is above the upper rule");
    assert!(top(&l.texts[1]) < upper && bottom(&l.texts[1]) > lower, "the date is between the rules");
    assert!(top(&l.texts[2]) < lower, "the name is below the lower rule");
    let western = seal_layout(&date_seal("営業部", "山田", false)).unwrap();
    assert_eq!(western.texts[1].text, "2026.11.11");
    // No organisation: the date and the name only, the rules stay.
    let bare = seal_layout(&date_seal("", "山田", false)).unwrap();
    assert_eq!(bare.texts.len(), 2);
    assert_eq!(bare.rules.len(), 2);
    // Square date seals fit too.
    let square = seal_layout(&SealSpec { shape: SealShape::Square, ..date_seal("経理部", "鈴木", false) }).unwrap();
    assert_inside(&square);
}

#[test]
fn bad_input_is_refused_with_a_message() {
    let e = |spec: SealSpec| seal_layout(&spec).unwrap_err();
    assert!(e(name_seal("", SealShape::Circle)).contains("needs a name"));
    assert!(e(name_seal("   ", SealShape::Circle)).contains("needs a name"));
    assert!(e(name_seal("Abcdefghijklmnopqrstuvwxyz", SealShape::Circle)).contains("at most"));
    assert!(e(SealSpec { size: 5.0, ..name_seal("Ab", SealShape::Circle) }).contains("size"));
    assert!(e(SealSpec { size: f64::NAN, ..name_seal("Ab", SealShape::Circle) }).contains("size"));
    assert!(e(SealSpec { color: [2.0, 0.0, 0.0], ..name_seal("Ab", SealShape::Circle) }).contains("colour"));
    assert!(e(SealSpec { date: (2026, 2, 31), ..date_seal("", "Ab", false) }).contains("not a date"));
    assert!(e(name_seal("A\u{1}B", SealShape::Circle)).contains("control"));
    assert!(e(name_seal("Ab😀", SealShape::Circle)).contains("can't be drawn"));
    assert!(e(date_seal("An organisation name far too long to fit", "Ab", false)).contains("at most"));
}

#[test]
fn japanese_without_a_japanese_face_is_refused_clearly() {
    if has_japanese_face() {
        return;
    }
    let e = seal_layout(&name_seal("山田", SealShape::Circle)).unwrap_err();
    assert!(e.contains("no Japanese font is embedded"), "{e}");
    assert!(seal_pdf(&name_seal("山田", SealShape::Circle)).is_err());
    assert!(seal_pdf(&name_seal("Yamada", SealShape::Circle)).is_ok(), "ASCII needs no Japanese face");
}

fn decoded_streams(bytes: Vec<u8>) -> String {
    let doc = pdfcraft_cos::Document::open(Arc::new(bytes)).unwrap();
    doc.scan_objects()
        .filter_map(|(_, o)| match o.as_ref() {
            pdfcraft_cos::Object::Stream(s) => s.decoded().ok().map(|b| String::from_utf8_lossy(&b).into_owned()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

#[test]
fn the_seal_pdf_is_vector_in_the_seal_colour() {
    let spec = SealSpec { color: [0.5, 0.25, 0.125], ..name_seal("Ab", SealShape::Circle) };
    let all = decoded_streams(seal_pdf(&spec).unwrap());
    assert!(all.contains("0.5 0.25 0.125 RG"), "the outline is stroked in the colour: {all}");
    assert!(all.contains(" c\n"), "the ring is a curve");
    assert!(all.contains("0.5 0.25 0.125 rg"), "the text is painted in the colour");
    let square = decoded_streams(seal_pdf(&name_seal("Ab", SealShape::Square)).unwrap());
    assert!(square.contains(" re S"), "a square seal is a rectangle");
}

/// A seal becomes a Stamp comment with an appearance, which can be moved and deleted like any
/// stamp, and the saved file reads back as the seal's text.
#[test]
fn a_placed_seal_is_a_stamp_with_an_appearance_and_its_text() {
    if !has_japanese_face() {
        eprintln!("skipping: built without a craft-fonts Jpan face");
        return;
    }
    let mut session = Session::new().with_clock(|| 1_700_000_000);
    let blank = pdfcraft_create::blank(300.0, 400.0, 1).unwrap();
    let bytes = pdfcraft_cos::write_full(&blank, &pdfcraft_cos::SaveOptions::default()).unwrap();
    let id = session.open("blank.pdf", None, Arc::new(bytes), None).unwrap();
    let spec = SealSpec { size: 60.0, ..date_seal("営業部", "山田", true) };
    let edit = seal_edit(&spec, 0, [150.0, 200.0], "Ada").unwrap();
    assert!(matches!(edit, Edit::AddCustomStamp { .. }));
    session.apply(id, edit).unwrap();
    let annotations = &session.get(id).unwrap().info.annotations;
    let stamps: Vec<_> = annotations.iter().filter(|a| a.subtype == "Stamp").collect();
    assert_eq!(stamps.len(), 1, "one Stamp comment");
    // Its box is the seal, centred where it was placed.
    let rect = stamps[0].rect;
    assert!((rect[2] - rect[0] - 60.0).abs() < 1e-3 && (rect[3] - rect[1] - 60.0).abs() < 1e-3, "{rect:?}");
    assert!(((rect[0] + rect[2]) / 2.0 - 150.0).abs() < 1e-3 && ((rect[1] + rect[3]) / 2.0 - 200.0).abs() < 1e-3);
    let saved = session.save_bytes(id).unwrap();
    // The appearance stream exists and draws the seal (a form XObject of the seal page).
    let saved_doc = pdfcraft_cos::Document::open(saved.clone()).unwrap();
    let ap = saved_doc.scan_objects().find_map(|(_, o)| {
        let d = o.as_dict()?;
        (d.name(b"Subtype") == Some(&b"Stamp"[..])).then(|| d.get(b"AP").cloned()).flatten()
    });
    assert!(ap.is_some(), "the stamp has an /AP");
    // The text is in the saved file as the seal's own words.
    let all = decoded_streams(saved.as_ref().clone());
    assert!(all.contains("RG"), "the outline is drawn in the appearance");
    // Undo removes it, and the seal can be deleted as a comment.
    session.undo(id).unwrap();
    assert!(session.get(id).unwrap().info.annotations.iter().all(|a| a.subtype != "Stamp"));
}
