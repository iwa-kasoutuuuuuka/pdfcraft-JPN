//! `page_add_seal`: stamp a Japanese electronic seal (電子印鑑 / ハンコ) on a page. The engine
//! draws it ([`pdfcraft_engine::seal`]) and places it as a Stamp comment.

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
}
