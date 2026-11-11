//! Scan & OCR ▸ OCR a folder: make every PDF in a folder (and, when asked, its subfolders)
//! searchable, one file at a time.
//!
//! The run is bounded and safe by construction:
//!
//! * Discovery finishes before the first file is read, and refuses a folder with more PDFs than
//!   [`MAX_FILES`] or more bytes than [`MAX_TOTAL_BYTES`] with an error that says what to do.
//!   Links are never followed and folders are walked at most [`MAX_DEPTH`] deep.
//! * A source is never overwritten unless [`FolderOptions::overwrite`] is set. Without it, a result
//!   that would land on an existing file leaves that file alone and is reported as skipped.
//! * One file's failure never stops the run. Each file ends as `ok`, `skipped` (with the reason: it
//!   already has a text layer, it is password protected, its result exists, …) or `failed` (with
//!   the reason), and the report adds them up.
//!
//! The engines are built only when a file needs them, so a folder of PDFs that already have text
//! is processed without any OCR engine installed.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::ocr::{FileJob, OcrSettings, Opened, Recognizers};

/// How the engines are built, when the first file needs reading.
type Engines<'a> = &'a dyn Fn(&OcrSettings) -> Result<Recognizers, String>;

/// Most PDFs one run takes.
pub const MAX_FILES: usize = 1000;
/// Largest single PDF a run reads, in bytes.
pub const MAX_FILE_BYTES: u64 = 256 * 1024 * 1024;
/// Most bytes of PDF one run takes in total.
pub const MAX_TOTAL_BYTES: u64 = 4 * 1024 * 1024 * 1024;
/// Deepest folder level a recursive run descends to.
pub const MAX_DEPTH: usize = 32;
/// Most directory entries looked at while discovering.
pub const MAX_ENTRIES: usize = 50_000;

/// The suffix put on a result written next to its source when none is chosen.
pub const DEFAULT_SUFFIX: &str = "_ocr";

/// What to do.
#[derive(Clone, Debug)]
pub struct FolderOptions {
    pub folder: PathBuf,
    /// Also take the PDFs of subfolders.
    pub recursive: bool,
    /// Write results here, keeping the subfolder layout; `None` writes next to each source.
    pub output_dir: Option<PathBuf>,
    /// Added to the file name before `.pdf`. `None`: [`DEFAULT_SUFFIX`] next to the source, nothing
    /// in `output_dir`.
    pub suffix: Option<String>,
    /// Leave a file that already has a text layer alone.
    pub skip_with_text: bool,
    /// Replace a result (or the source, when the names are the same) that already exists.
    pub overwrite: bool,
    pub settings: OcrSettings,
    pub max_files: usize,
    pub max_file_bytes: u64,
    pub max_total_bytes: u64,
}

impl FolderOptions {
    pub fn new(folder: impl Into<PathBuf>, settings: OcrSettings) -> Self {
        FolderOptions {
            folder: folder.into(),
            recursive: false,
            output_dir: None,
            suffix: None,
            skip_with_text: true,
            overwrite: false,
            settings,
            max_files: MAX_FILES,
            max_file_bytes: MAX_FILE_BYTES,
            max_total_bytes: MAX_TOTAL_BYTES,
        }
    }

    /// The suffix in effect.
    pub fn effective_suffix(&self) -> &str {
        match (&self.suffix, &self.output_dir) {
            (Some(s), _) => s,
            (None, None) => DEFAULT_SUFFIX,
            (None, Some(_)) => "",
        }
    }
}

/// How one file ended.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FileStatus {
    /// Searchable result written to `output`.
    Ok { output: PathBuf, words: usize },
    /// Left alone, and why.
    Skipped(String),
    /// Could not be done, and why.
    Failed(String),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FileReport {
    pub path: PathBuf,
    pub status: FileStatus,
}

/// A run's outcome.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct FolderReport {
    pub files: Vec<FileReport>,
    pub ok: usize,
    pub skipped: usize,
    pub failed: usize,
    pub words: usize,
    /// The run was stopped before every file was taken; the rest are not listed.
    pub cancelled: bool,
}

impl FolderReport {
    fn push(&mut self, path: PathBuf, status: FileStatus) {
        match &status {
            FileStatus::Ok { words, .. } => {
                self.ok += 1;
                self.words = self.words.saturating_add(*words);
            }
            FileStatus::Skipped(_) => self.skipped += 1,
            FileStatus::Failed(_) => self.failed += 1,
        }
        self.files.push(FileReport { path, status });
    }
}

/// A PDF found by discovery.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Found {
    pub path: PathBuf,
    /// The path below the chosen folder.
    pub relative: PathBuf,
    pub bytes: u64,
}

/// Windows junctions and symlinks are reparse points; neither is followed.
fn is_link(metadata: &std::fs::Metadata) -> bool {
    #[cfg(windows)]
    {
        use std::os::windows::fs::MetadataExt;
        metadata.file_attributes() & 0x400 != 0
    }
    #[cfg(not(windows))]
    {
        metadata.file_type().is_symlink()
    }
}

fn is_pdf(path: &Path) -> bool {
    path.extension().and_then(std::ffi::OsStr::to_str).is_some_and(|e| e.eq_ignore_ascii_case("pdf"))
}

/// The PDFs of the folder, sorted, with the limits applied. `skip` is a folder whose contents are
/// left out (the output folder when it lies inside the input).
pub fn discover(folder: &Path, recursive: bool, skip: Option<&Path>, options: &FolderOptions) -> Result<Vec<Found>, String> {
    let meta = std::fs::symlink_metadata(folder).map_err(|e| format!("{}: {e}", folder.display()))?;
    if is_link(&meta) || !meta.is_dir() {
        return Err(format!("{} is not a folder", folder.display()));
    }
    let mut found: Vec<Found> = Vec::new();
    let mut total: u64 = 0;
    let mut seen_entries = 0usize;
    let mut stack: Vec<(PathBuf, PathBuf, usize)> = vec![(folder.to_path_buf(), PathBuf::new(), 0)];
    while let Some((dir, relative, depth)) = stack.pop() {
        let mut entries: Vec<(std::ffi::OsString, PathBuf)> = Vec::new();
        for entry in std::fs::read_dir(&dir).map_err(|e| format!("{}: {e}", dir.display()))? {
            let entry = entry.map_err(|e| format!("{}: {e}", dir.display()))?;
            seen_entries += 1;
            if seen_entries > MAX_ENTRIES {
                return Err(format!("the folder holds more than {MAX_ENTRIES} entries; choose a smaller folder"));
            }
            entries.push((entry.file_name(), entry.path()));
        }
        entries.sort();
        for (name, path) in entries {
            let Ok(meta) = std::fs::symlink_metadata(&path) else { continue };
            if is_link(&meta) {
                continue;
            }
            if meta.is_dir() {
                if recursive && depth < MAX_DEPTH && skip != Some(path.as_path()) {
                    stack.push((path, relative.join(&name), depth + 1));
                }
            } else if meta.is_file() && is_pdf(&path) {
                total = total.saturating_add(meta.len());
                if found.len() >= options.max_files {
                    return Err(format!(
                        "the folder holds more than {} PDF files, the most one run takes; choose a smaller folder{}",
                        options.max_files,
                        if recursive { " or turn off subfolders" } else { "" }
                    ));
                }
                found.push(Found { path, relative: relative.join(&name), bytes: meta.len() });
            }
        }
    }
    if total > options.max_total_bytes {
        return Err(format!(
            "the PDF files add up to {} MiB, more than the {} MiB one run takes; choose a smaller folder",
            total / (1024 * 1024),
            options.max_total_bytes / (1024 * 1024)
        ));
    }
    found.sort_by(|a, b| a.relative.cmp(&b.relative));
    Ok(found)
}

/// The suffix goes into a file name: it may not hold a path or a character a file name cannot.
fn check_suffix(suffix: &str) -> Result<(), String> {
    if suffix.chars().any(|c| matches!(c, '/' | '\\' | ':' | '*' | '?' | '"' | '<' | '>' | '|') || c.is_control()) {
        return Err("the suffix may not contain path separators or characters a file name cannot hold".into());
    }
    Ok(())
}

/// Where the result of `found` goes.
pub fn target_of(options: &FolderOptions, found: &Found) -> Result<PathBuf, String> {
    let suffix = options.effective_suffix();
    check_suffix(suffix)?;
    let stem = found.relative.file_stem().and_then(std::ffi::OsStr::to_str).ok_or("the file name is not valid text")?;
    let name = format!("{stem}{suffix}.pdf");
    Ok(match &options.output_dir {
        Some(dir) => dir.join(found.relative.parent().unwrap_or(Path::new(""))).join(name),
        None => found.path.with_file_name(name),
    })
}

/// The first reason the run as set up cannot go ahead, checked before anything is read.
fn check_setup(options: &FolderOptions) -> Result<(), String> {
    check_suffix(options.effective_suffix())?;
    if options.suffix.is_none() && options.output_dir.is_none() {
        return Ok(());
    }
    if options.effective_suffix().is_empty() && options.output_dir.is_none() && !options.overwrite {
        return Err("an empty suffix without an output folder would replace the source files; set overwrite to allow it".into());
    }
    Ok(())
}

/// What a run reports while it works.
#[derive(Clone, Debug)]
pub struct Progress<'a> {
    /// Files finished.
    pub done: usize,
    pub total: usize,
    /// The file being worked on now.
    pub current: &'a Path,
}

fn is_password_error(why: &str) -> bool {
    let w = why.to_lowercase();
    w.contains("password") || w.contains("encrypt")
}

/// OCR every PDF of the folder.
///
/// `write(path, bytes)` publishes a result (the caller chooses how: atomically, in the app or the
/// tool). `progress` is called before each file and returns `false` to stop (the report says
/// `cancelled`). Returns an error only when the run cannot start; per-file problems are in the report.
pub fn run(
    options: &FolderOptions,
    write: impl FnMut(&Path, &[u8]) -> Result<(), String>,
    progress: impl FnMut(Progress<'_>) -> bool,
) -> Result<FolderReport, String> {
    run_with(options, &crate::ocr::recognizers, write, progress)
}

/// [`run`] with the recognition engines supplied by `engines` (tests use a fake one).
pub(crate) fn run_with(
    options: &FolderOptions,
    engines: Engines<'_>,
    mut write: impl FnMut(&Path, &[u8]) -> Result<(), String>,
    mut progress: impl FnMut(Progress<'_>) -> bool,
) -> Result<FolderReport, String> {
    check_setup(options)?;
    let skip = options.output_dir.as_deref();
    let files = discover(&options.folder, options.recursive, skip, options)?;
    let total = files.len();
    let mut report = FolderReport::default();
    let mut settings = options.settings.clone();
    settings.skip_text_files = options.skip_with_text;
    // Built when the first file needs reading, then reused. A failure to build is the same for every
    // file, so it ends the run instead of failing each file in turn.
    let mut recognizers: Option<Recognizers> = None;
    let suffix = options.effective_suffix().to_string();
    for (i, found) in files.iter().enumerate() {
        if !progress(Progress { done: i, total, current: &found.path }) {
            report.cancelled = true;
            break;
        }
        let path = found.path.clone();
        // Guarded: a panic in a decoder or an engine fails this file, not the run.
        let outcome = crate::guard(|| one(options, &settings, &suffix, found, engines, &mut recognizers, &mut write));
        match outcome {
            Ok(Ok(status)) => report.push(path, status),
            Ok(Err(Fatal(why))) => return Err(why),
            Err(panic) => report.push(path, FileStatus::Failed(format!("internal error: {panic}"))),
        }
    }
    progress(Progress { done: report.files.len(), total, current: Path::new("") });
    Ok(report)
}

/// An error that ends the whole run (the engines cannot start).
struct Fatal(String);

fn one(
    options: &FolderOptions,
    settings: &OcrSettings,
    suffix: &str,
    found: &Found,
    engines: Engines<'_>,
    recognizers: &mut Option<Recognizers>,
    write: &mut impl FnMut(&Path, &[u8]) -> Result<(), String>,
) -> Result<FileStatus, Fatal> {
    use FileStatus::{Failed, Skipped};
    // Results of an earlier run sit beside their sources: they are not sources.
    if options.output_dir.is_none() && !suffix.is_empty() && !options.overwrite {
        let stem = found.relative.file_stem().and_then(std::ffi::OsStr::to_str).unwrap_or("");
        if stem.ends_with(suffix) && stem.len() > suffix.len() {
            return Ok(Skipped(format!("its name ends in {suffix}: taken for the result of an earlier run")));
        }
    }
    let target = match target_of(options, found) {
        Ok(t) => t,
        Err(e) => return Ok(Failed(e)),
    };
    if same_file(&target, &found.path) && !options.overwrite {
        return Ok(Failed("the result would replace the source; set overwrite to allow it".into()));
    }
    if target.exists() && !options.overwrite {
        return Ok(Skipped(format!("{} already exists", target.display())));
    }
    if found.bytes > options.max_file_bytes {
        return Ok(Failed(format!(
            "{} MiB is more than the {} MiB one file may be",
            found.bytes / (1024 * 1024),
            options.max_file_bytes / (1024 * 1024)
        )));
    }
    let bytes = match std::fs::read(&found.path) {
        Ok(b) => b,
        Err(e) => return Ok(Failed(e.to_string())),
    };
    let name = found.path.file_name().and_then(std::ffi::OsStr::to_str).unwrap_or("file.pdf");
    let file = match FileJob::open(name, Arc::new(bytes), None, settings.clone()) {
        Ok(Opened::Skipped(_)) => return Ok(Skipped("it already has a text layer".into())),
        Ok(Opened::Read(file)) => file,
        Err(why) if is_password_error(&why) => return Ok(Skipped(format!("encrypted: {why}"))),
        Err(why) => return Ok(Failed(why)),
    };
    let mut file = file;
    let Some(job) = file.job.take() else { return Ok(Failed("the document could not be read".into())) };
    if recognizers.is_none() {
        *recognizers = Some(engines(settings).map_err(Fatal)?);
    }
    let Some(engines) = recognizers.as_ref() else { return Ok(Failed("no recognition engine".into())) };
    let pages = job.run(engines, |_, _| true);
    if !pages.is_empty() && pages.iter().all(|p| p.skipped.is_some()) {
        return Ok(Skipped("every page already has text".into()));
    }
    let result = match file.finish(pages) {
        Ok(r) => r,
        Err(why) => return Ok(Failed(why)),
    };
    let words = result.words();
    if words == 0 {
        return Ok(Skipped("no text was recognised".into()));
    }
    if let Err(why) = write(&target, &result.bytes) {
        return Ok(Failed(why));
    }
    Ok(FileStatus::Ok { output: target, words })
}

/// Whether two paths name the same file (the result and its source).
fn same_file(a: &Path, b: &Path) -> bool {
    match (std::fs::canonicalize(a), std::fs::canonicalize(b)) {
        (Ok(x), Ok(y)) => x == y,
        _ => a == b,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("pdfcraft-ocr-folder-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn options(folder: &Path) -> FolderOptions {
        FolderOptions::new(folder, OcrSettings::default())
    }

    #[test]
    fn discovery_lists_pdfs_sorted_and_only_descends_when_asked() {
        let dir = temp("discover");
        std::fs::create_dir_all(dir.join("sub")).unwrap();
        for f in ["b.pdf", "a.PDF", "note.txt", "sub/c.pdf"] {
            std::fs::write(dir.join(f), b"x").unwrap();
        }
        let flat = discover(&dir, false, None, &options(&dir)).unwrap();
        assert_eq!(flat.iter().map(|f| f.relative.to_string_lossy().into_owned()).collect::<Vec<_>>(), ["a.PDF", "b.pdf"]);
        let deep = discover(&dir, true, None, &options(&dir)).unwrap();
        assert_eq!(deep.len(), 3);
        assert!(deep.iter().any(|f| f.relative == Path::new("sub").join("c.pdf")));
        // The output folder, inside the input, is not an input.
        let skipped = discover(&dir, true, Some(&dir.join("sub")), &options(&dir)).unwrap();
        assert_eq!(skipped.len(), 2);
    }

    #[test]
    fn discovery_limits_are_refused_with_what_to_do() {
        let dir = temp("limits");
        for f in ["1.pdf", "2.pdf", "3.pdf"] {
            std::fs::write(dir.join(f), b"x").unwrap();
        }
        let mut o = options(&dir);
        o.max_files = 2;
        let err = discover(&dir, false, None, &o).unwrap_err();
        assert!(err.contains("more than 2 PDF files") && err.contains("smaller folder"), "{err}");
        o.max_files = 10;
        o.max_total_bytes = 2;
        assert!(discover(&dir, false, None, &o).unwrap_err().contains("add up to"));
        assert!(discover(&dir.join("nope"), false, None, &options(&dir)).is_err());
        assert!(discover(&dir.join("1.pdf"), false, None, &options(&dir)).unwrap_err().contains("not a folder"));
    }

    #[test]
    fn targets_follow_the_suffix_and_the_output_folder() {
        let dir = temp("targets");
        let found = Found { path: dir.join("sub").join("scan.pdf"), relative: Path::new("sub").join("scan.pdf"), bytes: 1 };
        let mut o = options(&dir);
        assert_eq!(target_of(&o, &found).unwrap(), dir.join("sub").join("scan_ocr.pdf"));
        o.suffix = Some("-txt".into());
        assert_eq!(target_of(&o, &found).unwrap(), dir.join("sub").join("scan-txt.pdf"));
        o.output_dir = Some(dir.join("out"));
        o.suffix = None;
        assert_eq!(target_of(&o, &found).unwrap(), dir.join("out").join("sub").join("scan.pdf"));
        o.suffix = Some("../x".into());
        assert!(target_of(&o, &found).is_err());
    }

    #[test]
    fn an_empty_suffix_beside_the_source_needs_overwrite() {
        let dir = temp("empty-suffix");
        let mut o = options(&dir);
        o.suffix = Some(String::new());
        assert!(run(&o, |_, _| Ok(()), |_| true).unwrap_err().contains("overwrite"));
        o.overwrite = true;
        assert!(run(&o, |_, _| Ok(()), |_| true).is_ok());
    }

    #[test]
    fn a_corrupt_file_fails_alone_and_a_cancel_stops_the_run() {
        let dir = temp("corrupt");
        std::fs::write(dir.join("a.pdf"), b"not a pdf").unwrap();
        std::fs::write(dir.join("b.pdf"), b"%PDF-1.4 broken").unwrap();
        let report = run(&options(&dir), |_, _| Ok(()), |_| true).unwrap();
        assert_eq!((report.ok, report.skipped, report.failed), (0, 0, 2), "{report:?}");
        assert!(report.files.iter().all(|f| matches!(&f.status, FileStatus::Failed(why) if !why.is_empty())));
        let mut calls = 0;
        let report = run(
            &options(&dir),
            |_, _| Ok(()),
            |_| {
                calls += 1;
                calls < 2
            },
        )
        .unwrap();
        assert!(report.cancelled);
        assert_eq!(report.files.len(), 1);
    }

    #[test]
    fn earlier_results_and_existing_outputs_are_left_alone() {
        let dir = temp("existing");
        std::fs::write(dir.join("a.pdf"), b"x").unwrap();
        std::fs::write(dir.join("a_ocr.pdf"), b"y").unwrap();
        let report = run(&options(&dir), |_, _| Ok(()), |_| true).unwrap();
        // a_ocr.pdf is taken for a result; a.pdf's result exists.
        assert_eq!(report.skipped, 2, "{report:?}");
        assert!(report.files.iter().any(|f| matches!(&f.status, FileStatus::Skipped(why) if why.contains("already exists"))), "{report:?}");
        assert_eq!(std::fs::read(dir.join("a_ocr.pdf")).unwrap(), b"y");
    }
}

#[cfg(test)]
mod run_tests;
