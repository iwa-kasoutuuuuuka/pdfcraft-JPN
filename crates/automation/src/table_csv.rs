//! Scan & OCR: `export_tables_csv` (the tables of pages, found from where their text sits, as CSV
//! for Excel). A thin adapter over `pdfcraft_engine::table_csv`, which carries the detection, the
//! CSV writing and their tests.

use pdfcraft_engine::table_csv::{self, CsvLayout, TableCsvOptions};

use super::{Args, Automation, Result, ToolError, Value, failed, json, write_atomic};

/// `path` without a trailing `.csv` (any case), the stem the per-table files are named from.
fn stem_of(path: &str) -> &str {
    let n = path.len();
    match path.get(n.saturating_sub(4)..) {
        Some(ext) if n > 4 && ext.eq_ignore_ascii_case(".csv") => path.get(..n - 4).unwrap_or(path),
        _ => path,
    }
}

impl Automation {
    pub(crate) fn export_tables_csv(&self, a: &Args) -> Result<Value> {
        let doc = self.doc_to_copy(a)?;
        let pages = match a.opt_ints("pages")? {
            Some(_) => self.pages(a, "pages")?,
            None => (0..doc.info.pages.len()).collect(),
        };
        let path_arg = a.str("path")?;
        let layout = match a.opt_str("layout")? {
            Some(l) => CsvLayout::parse(l).ok_or_else(|| ToolError::InvalidArgs(format!("layout must be per_table or one_file, not {l:?}")))?,
            None => CsvLayout::PerTable,
        };
        // Checked before anything is read, so a path outside the root is refused up front.
        self.resolve(path_arg, true)?;
        let options = TableCsvOptions {
            layout,
            ocr_if_needed: a.opt_bool("ocr_if_needed")?.unwrap_or(false),
            ocr: self.japanese_ocr_settings(a)?,
            use_rules: a.opt_bool("use_rules")?.unwrap_or(true),
        };
        let export = table_csv::export_tables_csv(doc.export_source(), &doc.name, &pages, &options, |_, _| true).map_err(failed)?;
        let names: Vec<String> = match layout {
            CsvLayout::OneFile => export.files.iter().map(|_| path_arg.to_string()).collect(),
            CsvLayout::PerTable => export.tables.iter().map(|t| table_csv::table_file_name(stem_of(path_arg), t)).collect(),
        };
        // Every target is resolved before the first is written, so a refusal writes nothing.
        let targets = names.iter().map(|n| self.resolve(n, true)).collect::<Result<Vec<_>>>()?;
        for (target, bytes) in targets.iter().zip(&export.files) {
            write_atomic(target, bytes)?;
        }
        let tables: Vec<Value> = export
            .tables
            .iter()
            .map(|t| json!({ "page": t.page, "index": t.index, "rows": t.rows, "columns": t.columns, "ruled": t.ruled }))
            .collect();
        let mut out = json!({
            "layout": layout.name(),
            "files": targets.iter().map(|p| p.to_string_lossy()).collect::<Vec<_>>(),
            "pages": export.pages,
            "tables": tables,
            "ocr_pages": export.ocr_pages,
            "empty_pages": export.empty_pages,
        });
        if !export.warnings.is_empty() {
            out["warnings"] = json!(export.warnings);
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::stem_of;

    #[test]
    fn the_stem_drops_only_a_csv_extension() {
        assert_eq!(stem_of("out/表.CSV"), "out/表");
        assert_eq!(stem_of("out.csv"), "out");
        assert_eq!(stem_of("out.txt"), "out.txt");
        assert_eq!(stem_of(".csv"), ".csv");
        assert_eq!(stem_of("表"), "表");
    }
}
