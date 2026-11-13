//! `page_add_seal`: stamp a Japanese electronic seal (電子印鑑 / ハンコ) on a page. The engine
//! draws it ([`pdfcraft_engine::seal`]) and places it as a Stamp comment.
//!
//! `page_add_business_stamp` (ビジネス用スタンプ: 承認, 至急, 社外秘...) and `page_add_tally_seal`
//! (契印: a seal split over the edge between consecutive pages) are made the same way.

use pdfcraft_engine::seal::business::{
    BUSINESS_STAMP_SIZE_RANGE, BusinessStampSpec, DEFAULT_BUSINESS_STAMP_SIZE, StampFrame, business_stamp_description, business_stamp_edit,
};
use pdfcraft_engine::seal::tally::{TallyDirection, tally_seal_edit};
use pdfcraft_engine::seal::{
    DEFAULT_SEAL_COLOR, DEFAULT_SEAL_SIZE, SEAL_SIZE_RANGE, SealKind, SealShape, SealSpec, parse_seal_date, seal_date_text, seal_edit,
};
use serde_json::{Value, json};

use crate::comments::{DEFAULT_AUTHOR, parse_color, rect_to_user};
use crate::{Args, Automation, Result, ToolError};

impl Automation {
    /// `page_add_seal {doc, page, at, size?, kind?, name, org?, date?, era?, shape?, color?, author?}`.
    pub(crate) fn page_add_seal(&mut self, a: &Args) -> Result<Value> {
        let page = self.page(a)?;
        let info = self.doc(a)?.info.pages[page].clone();
        let [x, y] = a.nums::<2>("at")?.ok_or_else(|| {
            ToolError::InvalidArgs("at: [x, y], the seal's centre in points from the top-left of the displayed page, is required".into())
        })?;
        if !x.is_finite() || !y.is_finite() {
            return Err(ToolError::InvalidArgs("at must be two numbers".into()));
        }
        let kind = match a.opt_str("kind")? {
            None => SealKind::Name,
            Some(k) => SealKind::parse(k).ok_or_else(|| ToolError::InvalidArgs(format!("unknown kind {k:?}: name (認印) or date (データ印)")))?,
        };
        let shape = match a.opt_str("shape")? {
            None => SealShape::Circle,
            Some(s) => SealShape::parse(s).ok_or_else(|| ToolError::InvalidArgs(format!("unknown shape {s:?}: circle or square")))?,
        };
        let size = a.opt_num("size")?.unwrap_or(DEFAULT_SEAL_SIZE);
        if !size.is_finite() || !(SEAL_SIZE_RANGE.0..=SEAL_SIZE_RANGE.1).contains(&size) {
            return Err(ToolError::InvalidArgs(format!("size must be {} to {} points", SEAL_SIZE_RANGE.0, SEAL_SIZE_RANGE.1)));
        }
        let color = match a.opt_str("color")? {
            Some(c) => parse_color(c)?,
            None => DEFAULT_SEAL_COLOR,
        };
        let date = match a.opt_str("date")? {
            None => self.session().today(),
            Some(d) if d.trim().eq_ignore_ascii_case("today") => self.session().today(),
            Some(d) => parse_seal_date(d)
                .ok_or_else(|| ToolError::InvalidArgs(format!("date {d:?} is not a date: write \"today\" or yyyy-mm-dd (2026-11-11)")))?,
        };
        let era = a.opt_bool("era")?.unwrap_or(false);
        let spec =
            SealSpec { kind, shape, name: a.str("name")?.to_owned(), org: a.opt_str("org")?.unwrap_or_default().to_owned(), date, era, color, size };
        let author = a.opt_str("author")?.unwrap_or(DEFAULT_AUTHOR).to_owned();
        let [ux, uy, ..] = rect_to_user(&info, [x, y, x, y]);
        let edit = seal_edit(&spec, page, [ux, uy], &author).map_err(ToolError::InvalidArgs)?;
        let date_text = (kind == SealKind::Date).then(|| seal_date_text(date, era).unwrap_or_default());
        let mut out = self.apply(a, edit)?;
        out["seal"] = json!({ "kind": kind.id(), "shape": shape.id(), "name": spec.name.trim(), "date": date_text });
        Ok(out)
    }

    /// The `date` argument of a business stamp: "today" or yyyy-mm-dd.
    fn stamp_date_arg(&self, d: &str) -> Result<(i64, u32, u32)> {
        if d.trim().eq_ignore_ascii_case("today") {
            return Ok(self.session().today());
        }
        parse_seal_date(d).ok_or_else(|| ToolError::InvalidArgs(format!("date {d:?} is not a date: write \"today\" or yyyy-mm-dd (2026-11-13)")))
    }

    /// `page_add_business_stamp {doc, page, at, word, frame?, date?, era?, name?, size?, color?, author?}`.
    pub(crate) fn page_add_business_stamp(&mut self, a: &Args) -> Result<Value> {
        let page = self.page(a)?;
        let info = self.doc(a)?.info.pages[page].clone();
        let [x, y] = a.nums::<2>("at")?.ok_or_else(|| {
            ToolError::InvalidArgs("at: [x, y], the stamp's centre in points from the top-left of the displayed page, is required".into())
        })?;
        if !x.is_finite() || !y.is_finite() {
            return Err(ToolError::InvalidArgs("at must be two numbers".into()));
        }
        let frame = match a.opt_str("frame")? {
            None => StampFrame::Rounded,
            Some(f) => StampFrame::parse(f).ok_or_else(|| ToolError::InvalidArgs(format!("unknown frame {f:?}: rounded or double")))?,
        };
        let size = a.opt_num("size")?.unwrap_or(DEFAULT_BUSINESS_STAMP_SIZE);
        if !size.is_finite() || !(BUSINESS_STAMP_SIZE_RANGE.0..=BUSINESS_STAMP_SIZE_RANGE.1).contains(&size) {
            return Err(ToolError::InvalidArgs(format!("size must be {} to {} points", BUSINESS_STAMP_SIZE_RANGE.0, BUSINESS_STAMP_SIZE_RANGE.1)));
        }
        let color = match a.opt_str("color")? {
            Some(c) => parse_color(c)?,
            None => DEFAULT_SEAL_COLOR,
        };
        let date = a.opt_str("date")?.map(|d| self.stamp_date_arg(d)).transpose()?;
        let spec = BusinessStampSpec {
            word: a.str("word")?.to_owned(),
            frame,
            date,
            era: a.opt_bool("era")?.unwrap_or(false),
            name: a.opt_str("name")?.unwrap_or_default().to_owned(),
            color,
            size,
        };
        let author = a.opt_str("author")?.unwrap_or(DEFAULT_AUTHOR).to_owned();
        let [ux, uy, ..] = rect_to_user(&info, [x, y, x, y]);
        let edit = business_stamp_edit(&spec, page, &info, [ux, uy], &author).map_err(ToolError::InvalidArgs)?;
        let text = business_stamp_description(&spec);
        let mut out = self.apply(a, edit)?;
        out["stamp"] = json!({ "frame": frame.id(), "text": text });
        Ok(out)
    }

    /// `page_add_tally_seal {doc, first?, last?, name, shape?, size?, color?, direction?, y?, author?}`.
    pub(crate) fn page_add_tally_seal(&mut self, a: &Args) -> Result<Value> {
        let pages = self.doc(a)?.info.pages.clone();
        let n = pages.len() as i64;
        let first = a.opt_int("first")?.unwrap_or(1);
        let last = a.opt_int("last")?.unwrap_or(n);
        for (label, p) in [("first", first), ("last", last)] {
            if p < 1 || p > n {
                return Err(ToolError::InvalidArgs(format!("{label} page {p} is out of range: the document has {n} pages")));
            }
        }
        if first >= last {
            return Err(ToolError::InvalidArgs(format!(
                "a tally seal (契印) goes across two or more pages: last ({last}) must come after first ({first})"
            )));
        }
        let shape = match a.opt_str("shape")? {
            None => SealShape::Circle,
            Some(s) => SealShape::parse(s).ok_or_else(|| ToolError::InvalidArgs(format!("unknown shape {s:?}: circle or square")))?,
        };
        let direction = match a.opt_str("direction")? {
            None => TallyDirection::LeftToRight,
            Some(d) => TallyDirection::parse(d).ok_or_else(|| {
                ToolError::InvalidArgs(format!(
                    "unknown direction {d:?}: ltr (next page to the right, 横書き) or rtl (next page to the left, 縦書き)"
                ))
            })?,
        };
        let size = a.opt_num("size")?.unwrap_or(DEFAULT_SEAL_SIZE);
        if !size.is_finite() || !(SEAL_SIZE_RANGE.0..=SEAL_SIZE_RANGE.1).contains(&size) {
            return Err(ToolError::InvalidArgs(format!("size must be {} to {} points", SEAL_SIZE_RANGE.0, SEAL_SIZE_RANGE.1)));
        }
        let color = match a.opt_str("color")? {
            Some(c) => parse_color(c)?,
            None => DEFAULT_SEAL_COLOR,
        };
        let spec = SealSpec { kind: SealKind::Name, shape, name: a.str("name")?.to_owned(), color, size, ..SealSpec::default() };
        let author = a.opt_str("author")?.unwrap_or(DEFAULT_AUTHOR).to_owned();
        let y = a.opt_num("y")?;
        let (from, to) = (first as usize - 1, last as usize - 1);
        let edit = tally_seal_edit(&spec, &pages, from, to, direction, y, &author).map_err(ToolError::InvalidArgs)?;
        let mut out = self.apply(a, edit)?;
        out["tally_seal"] = json!({ "first": first, "last": last, "direction": direction.id(), "halves": 2 * (last - first) });
        Ok(out)
    }
}
