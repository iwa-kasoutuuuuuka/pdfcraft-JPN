//! `create_vertical_document`: a vertical Japanese document (縦書き文書) from plain text.

use pdfcraft_engine::{VerticalOptions, VerticalPageSize, VerticalTemplate, vertical_grid_colour};

use super::error::{Result, ToolError, failed};
use super::json::{Args, summary};
use super::paths::{read_input, write_atomic};
use super::{Automation, Value, json};

fn count(a: &Args, key: &str, max: i64) -> Result<Option<usize>> {
    match a.opt_int(key)? {
        None => Ok(None),
        Some(v) if (1..=max).contains(&v) => Ok(usize::try_from(v).ok()),
        Some(_) => Err(ToolError::InvalidArgs(format!("{key} must be 1 to {max}"))),
    }
}

impl Automation {
    pub(crate) fn create_vertical_document(&mut self, a: &Args) -> Result<Value> {
        let (text, source) = match (a.opt_str("text")?, a.opt_str("text_file")?) {
            (Some(t), None) => (t.to_string(), None),
            (None, Some(p)) => {
                let path = self.resolve(p, false)?;
                (pdfcraft_engine::decode_text(&read_input(&path)?), path.file_stem().map(|s| s.to_string_lossy().into_owned()))
            }
            (Some(_), Some(_)) => return Err(ToolError::InvalidArgs("give `text` or `text_file`, not both".into())),
            (None, None) => return Err(ToolError::InvalidArgs("give the text as `text` or a .txt path as `text_file`".into())),
        };
        let template = match a.opt_str("template")? {
            None => VerticalTemplate::Genko,
            Some(t) => {
                VerticalTemplate::parse(t).ok_or_else(|| ToolError::InvalidArgs(format!("template must be genko, binsen or plain, not {t:?}")))?
            }
        };
        let mut o = VerticalOptions::new(text, template);
        if let Some(s) = a.opt_str("page_size")? {
            o.page_size =
                VerticalPageSize::parse(s).ok_or_else(|| ToolError::InvalidArgs(format!("page_size must be a4, a5, b4 or b5, not {s:?}")))?;
        }
        o.landscape = match a.opt_str("orientation")? {
            None => None,
            Some("landscape") => Some(true),
            Some("portrait") => Some(false),
            Some(other) => return Err(ToolError::InvalidArgs(format!("orientation must be portrait or landscape, not {other:?}"))),
        };
        o.serif = match a.opt_str("font")? {
            None | Some("mincho") => true,
            Some("gothic") => false,
            Some(other) => return Err(ToolError::InvalidArgs(format!("font must be mincho or gothic, not {other:?}"))),
        };
        o.font_size = a.opt_num("font_size")?;
        o.columns = count(a, "columns", 120)?;
        o.rows = count(a, "rows", 60)?;
        if let Some(c) = a.opt_str("grid_colour")? {
            o.grid_colour =
                vertical_grid_colour(c).ok_or_else(|| ToolError::InvalidArgs(format!("grid_colour must be green, red or #RRGGBB, not {c:?}")))?;
        }
        o.margin = a.opt_num("margin")?;
        o.title = a.opt_str("title")?.map(str::to_owned);
        o.author = a.opt_str("author")?.map(str::to_owned);
        // Resolved before the work, so a path outside the root is refused up front.
        let output = a.opt_str("output")?.map(|p| self.resolve(p, true)).transpose()?;
        if let Some(path) = &output
            && path.exists()
            && !a.opt_bool("overwrite")?.unwrap_or(false)
        {
            return Err(ToolError::InvalidArgs(format!("{} already exists; pass overwrite: true to replace it", path.display())));
        }
        let (bytes, made) = self.session.create_vertical_document(&o).map_err(|e| match e {
            // Refusals of the options or the text (no Japanese face, a character it lacks, a grid that
            // doesn't fit): the message says what to change.
            e @ pdfcraft_engine::EditError::Edit(_) => ToolError::InvalidArgs(e.to_string()),
            e => failed(e),
        })?;
        if let Some(path) = &output {
            write_atomic(path, &bytes)?;
        }
        let name = a.opt_str("name")?.map(str::to_owned).unwrap_or_else(|| {
            format!(
                "{}.pdf",
                o.title.as_deref().filter(|t| !t.trim().is_empty()).map(str::to_owned).or(source).unwrap_or_else(|| "縦書き文書".into())
            )
        });
        let id = self.session.open_new(name, bytes).map_err(failed)?;
        let mut out = summary(self.session.get(id).ok_or_else(|| failed("the document vanished"))?);
        if let Some(obj) = out.as_object_mut() {
            obj.insert("columns".into(), json!(made.columns));
            obj.insert("hung".into(), json!(made.hung));
            obj.insert("page_size".into(), json!([made.page.0, made.page.1]));
            obj.insert("output".into(), json!(output.map(|p| p.display().to_string())));
        }
        Ok(out)
    }
}
