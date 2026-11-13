use std::sync::Arc;

use super::*;
use crate::Session;

fn has_japanese_face() -> bool {
    pdfcraft_fonts::document_japanese_font_for_style(false, false).is_some()
}

fn stamp(word: &str) -> BusinessStampSpec {
    BusinessStampSpec { word: word.into(), ..BusinessStampSpec::default() }
}

/// Every line's ink lies inside the innermost outline, and the lines go down in order.
fn assert_inside_and_ordered(l: &BusinessStampLayout) {
    let [x0, y0, x1, y1] = l.inner();
    for t in &l.lines {
        let [a, b, c, d] = t.ink;
        assert!(a >= x0 - 1e-9 && b >= y0 - 1e-9 && c <= x1 + 1e-9 && d <= y1 + 1e-9, "{:?} ink {:?} outside {:?}", t.text, t.ink, l.inner());
    }
    for pair in l.lines.windows(2) {
        assert!(pair[0].ink[1] > pair[1].ink[3], "{:?} is above {:?}", pair[0].text, pair[1].text);
    }
    if let Some(([_, y, ..], _)) = l.rule {
        assert!(l.lines[0].ink[1] > y && l.lines.get(1).is_some_and(|t| t.ink[3] < y), "the rule is between the word and the lines");
    }
}

#[test]
fn every_preset_fits_inside_its_frame() {
    if !has_japanese_face() {
        eprintln!("skipping: built without a craft-fonts Jpan face");
        return;
    }
    for word in BUSINESS_STAMP_WORDS {
        for frame in [StampFrame::Rounded, StampFrame::Double] {
            let l = business_stamp_layout(&BusinessStampSpec { frame, ..stamp(word) }).unwrap();
            assert_eq!(l.lines.len(), 1);
            assert_eq!(l.lines[0].text, *word);
            assert!(l.lines[0].bold);
            assert!(l.rule.is_none(), "a word alone has no rule");
            assert_eq!(l.outlines.len(), if frame == StampFrame::Double { 2 } else { 1 });
            assert!(l.width >= l.height, "a stamp is never taller than wide: {word}");
            assert_inside_and_ordered(&l);
        }
    }
    // A longer word makes a wider stamp of the same height.
    let short = business_stamp_layout(&stamp("済")).unwrap();
    let long = business_stamp_layout(&stamp("複製厳禁")).unwrap();
    assert!(long.width > short.width && (long.height - short.height).abs() < 1e-9);
}

#[test]
fn a_date_and_a_name_go_under_the_word_with_a_rule() {
    if !has_japanese_face() {
        eprintln!("skipping: built without a craft-fonts Jpan face");
        return;
    }
    let spec = BusinessStampSpec { date: Some((2026, 11, 13)), era: true, name: "山田".into(), ..stamp("承認") };
    let l = business_stamp_layout(&spec).unwrap();
    let texts: Vec<&str> = l.lines.iter().map(|t| t.text.as_str()).collect();
    assert_eq!(texts, ["承認", "令和8.11.13", "山田"]);
    assert!(l.rule.is_some());
    assert_inside_and_ordered(&l);
    assert!(l.lines[1].size < l.lines[0].size, "the date is smaller than the word");
    let western = business_stamp_layout(&BusinessStampSpec { era: false, ..spec.clone() }).unwrap();
    assert_eq!(western.lines[1].text, "2026.11.13");
    assert!(l.height > business_stamp_layout(&stamp("承認")).unwrap().height, "the lines make it taller");
    assert_eq!(business_stamp_description(&spec), "承認 令和8.11.13 山田");
    // Double rule with lines fits too.
    assert_inside_and_ordered(&business_stamp_layout(&BusinessStampSpec { frame: StampFrame::Double, ..spec }).unwrap());
}

#[test]
fn an_ascii_stamp_needs_no_japanese_face() {
    let l = business_stamp_layout(&BusinessStampSpec { date: Some((2026, 11, 13)), name: "Ada".into(), ..stamp("OK") }).unwrap();
    assert_inside_and_ordered(&l);
    assert!(business_stamp_pdf(&stamp("PAID")).is_ok());
}

#[test]
fn bad_input_is_refused_with_a_message() {
    let e = |spec: BusinessStampSpec| business_stamp_layout(&spec).unwrap_err();
    assert!(e(stamp("")).contains("needs a word"));
    assert!(e(stamp("ABCDEFGH")).contains("at most"));
    assert!(e(BusinessStampSpec { size: 2.0, ..stamp("OK") }).contains("size"));
    assert!(e(BusinessStampSpec { size: f64::INFINITY, ..stamp("OK") }).contains("size"));
    assert!(e(BusinessStampSpec { color: [0.0, -1.0, 0.0], ..stamp("OK") }).contains("colour"));
    assert!(e(BusinessStampSpec { date: Some((2026, 2, 30)), ..stamp("OK") }).contains("not a date"));
    assert!(e(stamp("O\u{7}K")).contains("control"));
    assert!(e(stamp("OK😀")).contains("can't be drawn"));
    assert!(e(BusinessStampSpec { name: "A name much too long".into(), ..stamp("OK") }).contains("at most"));
    if !has_japanese_face() {
        assert!(e(stamp("承認")).contains("no Japanese font"));
    }
}

fn decoded_streams(bytes: &[u8]) -> String {
    let doc = pdfcraft_cos::Document::open(Arc::new(bytes.to_vec())).unwrap();
    doc.scan_objects()
        .filter_map(|(_, o)| match o.as_ref() {
            pdfcraft_cos::Object::Stream(s) => s.decoded().ok().map(|b| String::from_utf8_lossy(&b).into_owned()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

#[test]
fn the_stamp_pdf_is_vector_in_its_colour() {
    let spec = BusinessStampSpec { color: [0.5, 0.25, 0.125], ..stamp("OK") };
    let all = decoded_streams(&business_stamp_pdf(&spec).unwrap());
    assert!(all.contains("0.5 0.25 0.125 RG"), "the outline is stroked in the colour: {all}");
    assert!(all.contains(" c\n"), "the rounded corners are curves");
    assert!(all.contains("0.5 0.25 0.125 rg"), "the word is painted in the colour");
    let double = decoded_streams(&business_stamp_pdf(&BusinessStampSpec { frame: StampFrame::Double, ..stamp("OK") }).unwrap());
    assert_eq!(double.matches(" re S").count(), 2, "a double rule is two rectangles");
}

fn page_info(w: f32, h: f32, rotation: u16) -> PageInfo {
    PageInfo { width: w, height: h, label: "1".into(), crop: [0.0, 0.0, w, h], rotation }
}

/// A placed business stamp is one Stamp comment at the stamp's own size, centred where it was
/// placed, reading as its words, and one undo removes it.
#[test]
fn a_placed_business_stamp_is_a_stamp_comment() {
    if !has_japanese_face() {
        eprintln!("skipping: built without a craft-fonts Jpan face");
        return;
    }
    let mut session = Session::new();
    let blank = pdfcraft_create::blank(300.0, 400.0, 1).unwrap();
    let bytes = pdfcraft_cos::write_full(&blank, &pdfcraft_cos::SaveOptions::default()).unwrap();
    let id = session.open("blank.pdf", None, Arc::new(bytes), None).unwrap();
    let spec = BusinessStampSpec { size: 48.0, date: Some((2026, 11, 13)), name: "山田".into(), ..stamp("複製厳禁") };
    let layout = business_stamp_layout(&spec).unwrap();
    assert!(layout.width > 200.0, "bigger than a natural-size stamp may be, so the box is given");
    let edit = business_stamp_edit(&spec, 0, &page_info(300.0, 400.0, 0), [150.0, 200.0], "Ada").unwrap();
    session.apply(id, edit).unwrap();
    let stamps: Vec<_> = session.get(id).unwrap().info.annotations.iter().filter(|a| a.subtype == "Stamp").cloned().collect();
    assert_eq!(stamps.len(), 1);
    let r = stamps[0].rect.map(f64::from);
    assert!((r[2] - r[0] - layout.width).abs() < 1e-3 && (r[3] - r[1] - layout.height).abs() < 1e-3, "{r:?}");
    assert!(((r[0] + r[2]) / 2.0 - 150.0).abs() < 1e-3 && ((r[1] + r[3]) / 2.0 - 200.0).abs() < 1e-3, "{r:?}");
    session.undo(id).unwrap();
    assert!(session.get(id).unwrap().info.annotations.iter().all(|a| a.subtype != "Stamp"));
}
