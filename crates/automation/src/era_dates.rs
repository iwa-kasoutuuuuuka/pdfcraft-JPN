//! `dates_find_ja` and `dates_convert_ja`: find the dates in a document and convert them all to
//! the Japanese era calendar (和暦) or the Western one (西暦). The engine finds, converts and plans
//! the replacement ([`pdfcraft_engine::era_dates`]); these tools report and apply it.

use pdfcraft_engine::era_dates::{Calendar, Conversion, DateSearch, DateSource, DocumentDate};
use serde_json::{Value, json};

use crate::{Args, Automation, Result, ToolError};

fn hit_json(i: usize, h: &DocumentDate) -> Value {
    let source = match &h.source {
        DateSource::PageText { line, .. } => json!({ "kind": h.source.kind(), "line": line + 1 }),
        DateSource::Comment { index, .. } => json!({ "kind": h.source.kind(), "index": index + 1 }),
        DateSource::Field { name } => json!({ "kind": h.source.kind(), "field": name }),
    };
    let mut v = json!({ "hit": i + 1, "page": h.page.map(|p| p + 1), "source": source, "text": h.original });
    match &h.conversion {
        Conversion::To(s) => {
            v["status"] = json!("convert");
            v["converted"] = json!(s);
        }
        Conversion::Already => v["status"] = json!("already"),
        Conversion::Invalid(p) => {
            v["status"] = json!("invalid");
            v["problem"] = json!(p.id());
            v["note"] = json!(p.label());
        }
    }
    v
}

impl Automation {
    /// The target calendar and the dates found, for both tools.
    fn find_dates_ja(&mut self, a: &Args) -> Result<(Calendar, DateSearch)> {
        let to = match a.opt_str("to")? {
            None => Calendar::Japanese,
            Some(t) => {
                Calendar::from_id(t).ok_or_else(|| ToolError::InvalidArgs(format!("unknown calendar {t:?}: japanese (和暦) or western (西暦)")))?
            }
        };
        let pages = match a.opt_ints("pages")? {
            Some(_) => Some(self.pages(a, "pages")?),
            None => None,
        };
        let doc = self.doc(a)?;
        Ok((to, doc.find_dates(to, pages.as_deref())))
    }

    /// `dates_find_ja {doc, to?, pages?}`.
    pub(crate) fn dates_find_ja(&mut self, a: &Args) -> Result<Value> {
        let (to, search) = self.find_dates_ja(a)?;
        let count = |status: &str| search.hits.iter().filter(|h| status_of(&h.conversion) == status).count();
        Ok(json!({
            "to": to.id(),
            "count": search.hits.len(),
            "convertible": count("convert"),
            "already": count("already"),
            "invalid": count("invalid"),
            "page_text_withheld": search.page_text_withheld,
            "hits": search.hits.iter().enumerate().map(|(i, h)| hit_json(i, h)).collect::<Vec<_>>(),
        }))
    }

    /// `dates_convert_ja {doc, to?, pages?, hits?}`.
    pub(crate) fn dates_convert_ja(&mut self, a: &Args) -> Result<Value> {
        let (to, search) = self.find_dates_ja(a)?;
        let chosen: Vec<DocumentDate> = match a.opt_ints("hits")? {
            None => search.hits.iter().filter(|h| h.conversion.text().is_some()).cloned().collect(),
            Some(numbers) => {
                let n = search.hits.len();
                let mut out = Vec::new();
                for k in numbers {
                    let h = usize::try_from(k).ok().and_then(|k| k.checked_sub(1)).and_then(|k| search.hits.get(k));
                    out.push(
                        h.cloned()
                            .ok_or_else(|| ToolError::InvalidArgs(format!("hit {k} is out of range: dates_find_ja found {n} (numbered from 1)")))?,
                    );
                }
                out
            }
        };
        let plan = self.doc(a)?.date_conversion(&chosen);
        let replaced: Vec<Value> = plan
            .replaced
            .iter()
            .filter_map(|&i| chosen.get(i))
            .map(|h| json!({ "page": h.page.map(|p| p + 1), "kind": h.source.kind(), "text": h.original, "converted": h.conversion.text() }))
            .collect();
        let skipped: Vec<Value> = plan
            .skipped
            .iter()
            .filter_map(|&(i, why)| chosen.get(i).map(|h| (h, why)))
            .map(|(h, why)| json!({ "page": h.page.map(|p| p + 1), "kind": h.source.kind(), "text": h.original, "reason": why }))
            .collect();
        let mut out = match plan.edit {
            Some(edit) => self.apply(a, edit)?,
            None => json!({ "changed": false }),
        };
        out["to"] = json!(to.id());
        out["replaced"] = json!(replaced);
        out["skipped"] = json!(skipped);
        out["page_text_withheld"] = json!(search.page_text_withheld);
        Ok(out)
    }
}

fn status_of(c: &Conversion) -> &'static str {
    match c {
        Conversion::To(_) => "convert",
        Conversion::Already => "already",
        Conversion::Invalid(_) => "invalid",
    }
}
