//! pdfcraft-JPN: 契印 (tally seal across pages): a seal stamped over the edge between two
//! consecutive pages, half of it on each, so a page taken out or swapped shows a broken seal.
//!
//! On paper the seal straddles the gutter of the two pages laid side by side. Here each page gets
//! a custom stamp ([`Edit::AddCustomStamp`]) of half the seal: the seal is drawn as usual (the
//! parent module) and the stamp's picture is cut to its left or right half by the picture page's
//! crop box, which becomes the form's `/BBox` and so clips the drawing. The two halves sit at the
//! same height, each touching the shared edge, so laid side by side they make the whole seal.
//!
//! For a document read left to right (横書き, bound on the left) page N+1 follows page N on its
//! right: page N's right edge carries the seal's left half and page N+1's left edge its right
//! half. For a document read right to left (縦書き, bound on the right) it is mirrored. Stamping a
//! range puts a seal across every consecutive pair in it (全ページに契印).

use std::sync::Arc;

use pdfcraft_organize::{BoxSpec, PageBox, set_page_box};
use pdfcraft_render::PageInfo;

use super::{SealSpec, seal_document, seal_layout};
use crate::Edit;
use crate::pictures::MarkFile;

/// Most pages one tally seal run may cover.
pub const MAX_TALLY_PAGES: usize = 2000;

/// Which way the pages follow each other.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TallyDirection {
    /// The next page is to the right (横書き, 左綴じ).
    LeftToRight,
    /// The next page is to the left (縦書き, 右綴じ).
    RightToLeft,
}

impl TallyDirection {
    pub fn parse(s: &str) -> Option<TallyDirection> {
        match s.trim().to_lowercase().as_str() {
            "ltr" | "left-to-right" | "left" | "左綴じ" | "横書き" => Some(TallyDirection::LeftToRight),
            "rtl" | "right-to-left" | "right" | "右綴じ" | "縦書き" => Some(TallyDirection::RightToLeft),
            _ => None,
        }
    }

    pub fn id(self) -> &'static str {
        match self {
            TallyDirection::LeftToRight => "ltr",
            TallyDirection::RightToLeft => "rtl",
        }
    }
}

/// A half of a seal.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SealHalf {
    Left,
    Right,
}

/// One half seal on one page: the page (0-based), which half, and where it goes in view space
/// (points from the top-left of the displayed page, y down), `[x0, y0, x1, y1]`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TallyPiece {
    pub page: usize,
    pub half: SealHalf,
    pub view: [f64; 4],
}

/// Where the halves of a `size`-point seal go across every consecutive pair of `first..=last`
/// (0-based). `y` is the seal's centre in points from the top of the displayed pages (the same on
/// both pages of a pair); `None` puts it halfway down the shorter page of each pair. `Err` says
/// what to change.
pub fn tally_pieces(
    pages: &[PageInfo],
    first: usize,
    last: usize,
    direction: TallyDirection,
    y: Option<f64>,
    size: f64,
) -> Result<Vec<TallyPiece>, String> {
    if first >= last {
        return Err("a tally seal (契印) needs at least two pages: the last page must come after the first".into());
    }
    if last >= pages.len() {
        return Err(format!("page {} doesn't exist: the document has {} pages", last + 1, pages.len()));
    }
    if last - first + 1 > MAX_TALLY_PAGES {
        return Err(format!("a tally seal can cover at most {MAX_TALLY_PAGES} pages at once"));
    }
    if !size.is_finite() || size <= 0.0 {
        return Err("the seal size must be a positive number".into());
    }
    if let Some(y) = y
        && !y.is_finite()
    {
        return Err("the seal's height on the page must be a number".into());
    }
    let half = size / 2.0;
    let mut out = Vec::with_capacity(2 * (last - first));
    for n in first..last {
        let (Some(a), Some(b)) = (pages.get(n), pages.get(n + 1)) else { continue };
        let (wa, wb) = (f64::from(a.width), f64::from(b.width));
        let shorter = f64::from(a.height.min(b.height));
        let yc = y.unwrap_or(shorter / 2.0);
        if yc - half < 0.0 || yc + half > shorter {
            return Err(format!(
                "the seal doesn't fit at {yc:.0} pt from the top of pages {} and {} (they are {shorter:.0} pt tall and the seal {size:.0} pt): choose a height from {half:.0} to {:.0} pt",
                n + 1,
                n + 2,
                shorter - half
            ));
        }
        if wa < half || wb < half {
            return Err(format!("pages {} and {} are too narrow for a {size:.0} pt seal", n + 1, n + 2));
        }
        let (y0, y1) = (yc - half, yc + half);
        match direction {
            TallyDirection::LeftToRight => {
                out.push(TallyPiece { page: n, half: SealHalf::Left, view: [wa - half, y0, wa, y1] });
                out.push(TallyPiece { page: n + 1, half: SealHalf::Right, view: [0.0, y0, half, y1] });
            }
            TallyDirection::RightToLeft => {
                out.push(TallyPiece { page: n, half: SealHalf::Right, view: [0.0, y0, half, y1] });
                out.push(TallyPiece { page: n + 1, half: SealHalf::Left, view: [wb - half, y0, wb, y1] });
            }
        }
    }
    Ok(out)
}

/// The seal cut to one half: a one-page PDF of the whole seal whose crop box is that half, so
/// placed as a stamp it shows exactly the half.
pub fn seal_half_pdf(spec: &SealSpec, half: SealHalf) -> Result<Vec<u8>, String> {
    let mut doc = seal_document(spec)?;
    let s = spec.size;
    let crop = match half {
        SealHalf::Left => [0.0, 0.0, s / 2.0, s],
        SealHalf::Right => [s / 2.0, 0.0, s, s],
    };
    set_page_box(&mut doc, &[0], PageBox::Crop, BoxSpec::Rect(crop)).map_err(|e| e.to_string())?;
    pdfcraft_cos::write_full(&doc, &pdfcraft_cos::SaveOptions::default()).map_err(|e| e.to_string())
}

/// The comment text of a tally seal half: 契印 and what the seal reads.
pub fn tally_description(spec: &SealSpec) -> String {
    super::seal_description(spec).replacen("印鑑", "契印", 1)
}

/// The edit that stamps `spec` as a tally seal across every consecutive pair of `first..=last`
/// (0-based), one undo step. See [`tally_pieces`] for `direction` and `y`.
pub fn tally_seal_edit(
    spec: &SealSpec,
    pages: &[PageInfo],
    first: usize,
    last: usize,
    direction: TallyDirection,
    y: Option<f64>,
    author: &str,
) -> Result<Edit, String> {
    // The seal's own checks (size, name, colour) come first, so their messages win.
    seal_layout(spec)?;
    let pieces = tally_pieces(pages, first, last, direction, y, spec.size)?;
    let left = Arc::new(seal_half_pdf(spec, SealHalf::Left)?);
    let right = Arc::new(seal_half_pdf(spec, SealHalf::Right)?);
    let name = tally_description(spec);
    let mut edits = Vec::with_capacity(pieces.len());
    for p in pieces {
        let Some(info) = pages.get(p.page) else { continue };
        let [x0, y0, x1, y1] = p.view.map(|v| v as f32);
        let a = info.view_to_user(x0, y0);
        let b = info.view_to_user(x1, y1);
        let rect = [a[0].min(b[0]), a[1].min(b[1]), a[0].max(b[0]), a[1].max(b[1])].map(f64::from);
        let bytes = match p.half {
            SealHalf::Left => left.clone(),
            SealHalf::Right => right.clone(),
        };
        edits.push(Edit::AddCustomStamp {
            page: p.page,
            rect,
            name: name.clone(),
            file: MarkFile { name: "tally-seal.pdf".into(), bytes, page: 0 },
            author: author.into(),
            line: None,
        });
    }
    Ok(Edit::Batch { label: "Add stamp".into(), edits })
}

#[cfg(test)]
mod tests;
