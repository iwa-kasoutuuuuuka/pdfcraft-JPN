use std::sync::Arc;

use super::*;
use crate::Session;

fn page_info(w: f32, h: f32, rotation: u16) -> PageInfo {
    PageInfo { width: w, height: h, label: "1".into(), crop: [0.0, 0.0, w, h], rotation }
}

fn seal(name: &str) -> SealSpec {
    SealSpec { name: name.into(), size: 60.0, ..SealSpec::default() }
}

/// Across each pair, the two halves meet at the shared edge: page N's half ends at its right edge
/// and page N+1's begins at its left edge, at the same height, each half the seal wide.
#[test]
fn the_halves_meet_at_the_shared_edge_at_the_same_height() {
    let pages = [page_info(300.0, 400.0, 0), page_info(500.0, 400.0, 0), page_info(300.0, 380.0, 0)];
    let pieces = tally_pieces(&pages, 0, 2, TallyDirection::LeftToRight, Some(100.0), 60.0).unwrap();
    assert_eq!(pieces.len(), 4, "two pairs, two halves each");
    for pair in pieces.chunks(2) {
        let [a, b] = pair else { panic!("pairs") };
        assert_eq!(b.page, a.page + 1);
        assert_eq!((a.half, b.half), (SealHalf::Left, SealHalf::Right), "the left half is on the earlier page");
        let wa = f64::from(pages[a.page].width);
        assert_eq!(a.view[2], wa, "page N's half touches its right edge");
        assert_eq!(b.view[0], 0.0, "page N+1's half touches its left edge");
        for p in [a, b] {
            assert_eq!(p.view[2] - p.view[0], 30.0, "each half is half the seal wide");
            assert_eq!(p.view[3] - p.view[1], 60.0);
            assert_eq!((p.view[1] + p.view[3]) / 2.0, 100.0, "centred at the chosen height");
        }
    }
    // The middle page carries two halves: one at each edge.
    assert_eq!(pieces.iter().filter(|p| p.page == 1).count(), 2);

    // Right to left: mirrored.
    let rtl = tally_pieces(&pages, 0, 1, TallyDirection::RightToLeft, None, 60.0).unwrap();
    let [a, b] = rtl.as_slice() else { panic!("one pair") };
    assert_eq!((a.half, a.view[0]), (SealHalf::Right, 0.0), "page N's half is at its left edge");
    assert_eq!((b.half, b.view[2]), (SealHalf::Left, 500.0), "page N+1's half is at its right edge");
    assert_eq!((a.view[1] + a.view[3]) / 2.0, 200.0, "halfway down by default");
}

#[test]
fn a_bad_range_or_height_is_refused_with_a_message() {
    let pages = [page_info(300.0, 400.0, 0), page_info(300.0, 400.0, 0)];
    let e = |first, last, y| tally_pieces(&pages, first, last, TallyDirection::LeftToRight, y, 60.0).unwrap_err();
    assert!(e(0, 0, None).contains("at least two pages"));
    assert!(e(1, 0, None).contains("at least two pages"));
    assert!(e(0, 2, None).contains("doesn't exist"));
    assert!(e(0, 1, Some(10.0)).contains("doesn't fit"));
    assert!(e(0, 1, Some(395.0)).contains("doesn't fit"));
    assert!(e(0, 1, Some(f64::NAN)).contains("number"));
    let narrow = [page_info(20.0, 400.0, 0), page_info(300.0, 400.0, 0)];
    assert!(tally_pieces(&narrow, 0, 1, TallyDirection::LeftToRight, None, 60.0).unwrap_err().contains("narrow"));
}

/// Each half's picture is the whole seal cut by its crop box to exactly its half.
#[test]
fn each_half_pdf_is_cropped_to_its_half() {
    let spec = seal("Abe");
    for (half, crop) in [(SealHalf::Left, [0.0, 0.0, 30.0, 60.0]), (SealHalf::Right, [30.0, 0.0, 60.0, 60.0])] {
        let bytes = seal_half_pdf(&spec, half).unwrap();
        let doc = pdfcraft_cos::Document::open(Arc::new(bytes)).unwrap();
        let boxes = pdfcraft_organize::page_boxes(&doc).unwrap();
        assert_eq!(boxes[0][0], [0.0, 0.0, 60.0, 60.0], "the whole seal is drawn");
        assert_eq!(boxes[0][1], crop, "{half:?} shows only its half");
    }
    assert_eq!(tally_description(&spec), "契印 Abe");
}

fn three_pages(session: &mut Session) -> crate::DocId {
    let blank = pdfcraft_create::blank(300.0, 400.0, 3).unwrap();
    let bytes = pdfcraft_cos::write_full(&blank, &pdfcraft_cos::SaveOptions::default()).unwrap();
    session.open("blank.pdf", None, Arc::new(bytes), None).unwrap()
}

/// Stamped across a range, every page gets its halves as Stamp comments, each the half's size at
/// the shared edge in user space, and one undo removes them all.
#[test]
fn a_tally_seal_across_a_range_is_one_undo_step() {
    let mut session = Session::new();
    let id = three_pages(&mut session);
    let pages = session.get(id).unwrap().info.pages.clone();
    let edit = tally_seal_edit(&seal("Abe"), &pages, 0, 2, TallyDirection::LeftToRight, Some(100.0), "Ada").unwrap();
    session.apply(id, edit).unwrap();
    let mut stamps: Vec<_> = session.get(id).unwrap().info.annotations.iter().filter(|a| a.subtype == "Stamp").cloned().collect();
    assert_eq!(stamps.len(), 4, "halves on pages 1, 2, 2, 3");
    stamps.sort_by(|a, b| (a.page, a.rect[0] as i64).cmp(&(b.page, b.rect[0] as i64)));
    let rects: Vec<[f64; 4]> = stamps.iter().map(|s| s.rect.map(f64::from)).collect();
    let near = |a: [f64; 4], b: [f64; 4]| a.iter().zip(&b).all(|(x, y)| (x - y).abs() < 1e-3);
    // Page 1's right edge, page 2's left and right edges, page 3's left edge; 100 pt from the top
    // of a 400 pt page is y 300 in user space.
    assert!(near(rects[0], [270.0, 270.0, 300.0, 330.0]), "{rects:?}");
    assert!(near(rects[1], [0.0, 270.0, 30.0, 330.0]), "{rects:?}");
    assert!(near(rects[2], [270.0, 270.0, 300.0, 330.0]), "{rects:?}");
    assert!(near(rects[3], [0.0, 270.0, 30.0, 330.0]), "{rects:?}");
    assert!(stamps.iter().all(|s| s.contents.as_deref().is_some_and(|c| c.contains("契印"))));
    session.undo(id).unwrap();
    assert!(session.get(id).unwrap().info.annotations.iter().all(|a| a.subtype != "Stamp"), "one undo removes every half");
}

/// On a page turned a quarter, the half still sits at the displayed edge.
#[test]
fn on_a_rotated_page_the_half_is_at_the_displayed_edge() {
    let pages = [page_info(400.0, 300.0, 90), page_info(400.0, 300.0, 90)];
    let Edit::Batch { edits, .. } = tally_seal_edit(&seal("Abe"), &pages, 0, 1, TallyDirection::LeftToRight, None, "Ada").unwrap() else {
        panic!("a batch")
    };
    let Some(Edit::AddCustomStamp { rect, .. }) = edits.first() else { panic!("a stamp") };
    // Displayed right edge of a page turned 90° clockwise is user y = 0 (crop is 300 x 400 here,
    // as info gives it: the test's crop is the displayed size, so only the mapping matters).
    let back = pages[0].user_to_view(rect[0] as f32, rect[1] as f32);
    let back2 = pages[0].user_to_view(rect[2] as f32, rect[3] as f32);
    let (x0, x1) = (back[0].min(back2[0]), back[0].max(back2[0]));
    assert!((x1 - 400.0).abs() < 1e-3 && (x1 - x0 - 30.0).abs() < 1e-3, "{back:?} {back2:?}");
}
