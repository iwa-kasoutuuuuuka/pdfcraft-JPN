//! Scan & OCR: `export_text` (the recognised or extracted text as .txt / .docx) and `ocr_folder`
//! (searchable copies of every PDF in a folder). Thin adapters over
//! `pdfcraft_engine::text_export` and `pdfcraft_engine::ocr_folder`, which carry the logic and
//! the tests of the rules.

use pdfcraft_engine::ocr::EngineChoice;
use pdfcraft_engine::ocr_folder::{self, FileStatus, FolderOptions};
use pdfcraft_engine::text_export::{self, PageSeparator, TextExportOptions, TextFormat, VerticalMode};

use super::{Args, Automation, Result, ToolError, Value, failed, json, write_atomic};

impl Automation {
    /// The OCR settings of the two tools: the usual ones, but Japanese and the automatic engine
    /// unless the call says otherwise (the other OCR tools default to English and the built-in
    /// engine).
    pub(crate) fn japanese_ocr_settings(&self, a: &Args) -> Result<pdfcraft_engine::ocr::OcrSettings> {
        let mut settings = self.ocr_settings(a)?;
        if a.get("language").is_none() {
            settings.language = "ja".into();
        }
        if a.get("engine").is_none() {
            settings.engine = EngineChoice::Auto;
        }
        if a.get("strategy").is_none() {
            settings.strategy = settings.engine.default_strategy();
        }
        Ok(settings)
    }

    pub(crate) fn export_text(&self, a: &Args) -> Result<Value> {
        let doc = self.doc_to_copy(a)?;
        let pages = match a.opt_ints("pages")? {
            Some(_) => self.pages(a, "pages")?,
            None => (0..doc.info.pages.len()).collect(),
        };
        let path = self.resolve(a.str("path")?, true)?;
        let extension = path.extension().and_then(std::ffi::OsStr::to_str).unwrap_or("");
        let format = match a.opt_str("format")? {
            Some(f) => TextFormat::parse(f).ok_or_else(|| ToolError::InvalidArgs(format!("unsupported format {f:?} (txt or docx)")))?,
            None => TextFormat::parse(extension).ok_or_else(|| {
                ToolError::InvalidArgs(format!("cannot tell the format from the extension {extension:?}: name it with format (txt or docx)"))
            })?,
        };
        let vertical = match a.opt_str("vertical")? {
            Some(v) => VerticalMode::parse(v).ok_or_else(|| ToolError::InvalidArgs(format!("vertical must be auto, on or off, not {v:?}")))?,
            None => VerticalMode::Auto,
        };
        let separator = match a.opt_str("separator")? {
            Some(s) => PageSeparator::parse(s).ok_or_else(|| ToolError::InvalidArgs(format!("separator must be line or formfeed, not {s:?}")))?,
            None => PageSeparator::Line,
        };
        let ocr = self.japanese_ocr_settings(a)?;
        let stem = doc.name.trim_end_matches(".pdf").trim_end_matches(".PDF").to_string();
        let options = TextExportOptions {
            format,
            vertical,
            separator,
            ocr_if_needed: a.opt_bool("ocr_if_needed")?.unwrap_or(false),
            ocr,
            title: a.opt_str("title")?.map_or(stem, str::to_string),
            language: a.opt_str("document_language")?.map(str::to_string),
        };
        let export = text_export::export_text(doc.export_source(), &doc.name, &pages, &options, |_, _| true).map_err(failed)?;
        write_atomic(&path, &export.bytes)?;
        let mut out = json!({
            "path": path.to_string_lossy(),
            "format": format.extension(),
            "pages": export.pages,
            "paragraphs": export.paragraphs,
            "bytes": export.bytes.len(),
            "ocr_pages": export.ocr_pages,
            "vertical_pages": export.vertical_pages,
            "empty_pages": export.empty_pages,
        });
        if !export.warnings.is_empty() {
            out["warnings"] = json!(export.warnings);
        }
        Ok(out)
    }

    pub(crate) fn ocr_folder(&self, a: &Args) -> Result<Value> {
        use super::file_access::Access;
        let settings = self.japanese_ocr_settings(a)?;
        let folder_arg = a.str("folder")?;
        let folder = self.resolve(folder_arg, false)?;
        let mut options = FolderOptions::new(&folder, settings);
        options.recursive = a.opt_bool("recursive")?.unwrap_or(false);
        options.skip_with_text = a.opt_bool("skip_with_text")?.unwrap_or(true);
        options.overwrite = a.opt_bool("overwrite")?.unwrap_or(false);
        options.suffix = a.opt_str("suffix")?.map(str::to_string);
        if let Some(n) = a.opt_int("max_files")? {
            options.max_files = usize::try_from(n).map_err(|_| Args::wrong("max_files", "1 to 1000"))?.clamp(1, ocr_folder::MAX_FILES);
        }
        match a.opt_str("output_dir")? {
            Some(dir) => options.output_dir = Some(self.resolve(dir, true)?),
            // Results are written beside the sources: the folder must be writable too.
            None => {
                self.resolve_for(folder_arg, Access::Write, false)?;
            }
        }
        // A folder that is not one is the caller's mistake, not a per-file problem.
        if !folder.is_dir() {
            return Err(failed(format!("{folder_arg} is not a folder")));
        }
        let report = ocr_folder::run(&options, |path, bytes| write_atomic(path, bytes).map_err(|e| e.to_string()), |_| true).map_err(failed)?;
        let files: Vec<Value> = report
            .files
            .iter()
            .map(|f| match &f.status {
                FileStatus::Ok { output, words } => {
                    json!({ "path": f.path.to_string_lossy(), "status": "ok", "output": output.to_string_lossy(), "words": words })
                }
                FileStatus::Skipped(why) => json!({ "path": f.path.to_string_lossy(), "status": "skipped", "reason": why }),
                FileStatus::Failed(why) => json!({ "path": f.path.to_string_lossy(), "status": "failed", "reason": why }),
            })
            .collect();
        Ok(json!({
            "folder": folder.to_string_lossy(),
            "ok": report.ok,
            "skipped": report.skipped,
            "failed": report.failed,
            "words": report.words,
            "files": files,
        }))
    }
}
