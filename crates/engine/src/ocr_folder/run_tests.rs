//! Runs of `ocr_folder` with a fake engine, so no recognition models are needed.

use super::*;
use crate::ocr::{OcrError, Recognizer};
use crate::{Session, export};
use pdfcraft_ocr::{Line, OcrImage, RecognizeOptions, Word};

/// An engine that reads "日本語" on every page.
struct Fake;

impl Recognizer for Fake {
    fn id(&self) -> &'static str {
        "fake"
    }
    fn available(&self) -> bool {
        true
    }
    fn languages(&self) -> Vec<(String, String)> {
        vec![("ja".into(), "Japanese".into())]
    }
    fn recognize(&self, _image: &OcrImage, _options: &RecognizeOptions) -> Result<Vec<Line>, OcrError> {
        Ok(vec![Line { words: vec![Word::new("日本語", [10.0, 10.0, 60.0, 30.0], "fake")] }])
    }
}

fn fake(_: &OcrSettings) -> Result<Recognizers, String> {
    Ok(Recognizers { primary: Arc::new(Fake), secondary: None })
}

fn no_engine(_: &OcrSettings) -> Result<Recognizers, String> {
    Err("no engine installed".into())
}

/// A one-page PDF that is only a picture.
fn scan() -> Vec<u8> {
    let mut s = Session::new();
    let text = s.create_from_text("t", "Scanned words on a picture").unwrap();
    let id = s.open("text.pdf", None, text, None).unwrap();
    let png = export::Exporter::new(s.get(id).unwrap()).png(0, 100.0).unwrap();
    s.create_from_images(&[("scan.png".into(), png)]).unwrap().to_vec()
}

fn typed() -> Vec<u8> {
    Session::new().create_from_text("t", "Already typed").unwrap().to_vec()
}

fn text_of(bytes: Vec<u8>) -> String {
    let mut s = Session::new();
    let id = s.open("x.pdf", None, Arc::new(bytes), None).unwrap();
    crate::tests::page_texts(&s, id).concat()
}

fn temp(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("pdfcraft-ocr-folder-run-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn write_file(path: &Path, bytes: &[u8]) -> Result<(), String> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
    }
    std::fs::write(path, bytes).map_err(|e| e.to_string())
}

fn status<'a>(report: &'a FolderReport, name: &str) -> &'a FileStatus {
    &report.files.iter().find(|f| f.path.ends_with(name)).unwrap_or_else(|| panic!("{name} not in {report:?}")).status
}

#[test]
fn a_folder_is_made_searchable_next_to_the_sources_and_one_bad_file_does_not_stop_it() {
    let dir = temp("beside");
    std::fs::create_dir_all(dir.join("sub")).unwrap();
    std::fs::write(dir.join("scan1.pdf"), scan()).unwrap();
    std::fs::write(dir.join("sub").join("scan2.pdf"), scan()).unwrap();
    std::fs::write(dir.join("typed.pdf"), typed()).unwrap();
    std::fs::write(dir.join("broken.pdf"), b"%PDF-1.4 not really").unwrap();
    std::fs::write(dir.join("notes.txt"), b"ignored").unwrap();
    let before = std::fs::read(dir.join("scan1.pdf")).unwrap();
    let mut o = FolderOptions::new(&dir, OcrSettings::default());
    o.recursive = true;
    let mut seen = Vec::new();
    let report = run_with(&o, &fake, write_file, |p| {
        seen.push((p.done, p.total));
        true
    })
    .unwrap();
    assert_eq!((report.ok, report.skipped, report.failed), (2, 1, 1), "{report:?}");
    assert_eq!(report.words, 2);
    assert!(matches!(status(&report, "typed.pdf"), FileStatus::Skipped(why) if why.contains("text layer")), "{report:?}");
    assert!(matches!(status(&report, "broken.pdf"), FileStatus::Failed(why) if !why.is_empty()), "{report:?}");
    // The sources are untouched; the results hold the recognised text.
    assert_eq!(std::fs::read(dir.join("scan1.pdf")).unwrap(), before);
    assert!(text_of(std::fs::read(dir.join("scan1_ocr.pdf")).unwrap()).contains("日本語"));
    assert!(text_of(std::fs::read(dir.join("sub").join("scan2_ocr.pdf")).unwrap()).contains("日本語"));
    assert_eq!(seen.first(), Some(&(0, 4)));
    assert_eq!(seen.last(), Some(&(4, 4)));
    // A second run finds the results and leaves everything as it was.
    let again = run_with(&o, &fake, write_file, |_| true).unwrap();
    assert_eq!(again.ok, 0, "{again:?}");
    assert!(matches!(status(&again, "scan1.pdf"), FileStatus::Skipped(why) if why.contains("already exists")), "{again:?}");
    assert!(matches!(status(&again, "scan1_ocr.pdf"), FileStatus::Skipped(_)), "{again:?}");
}

#[test]
fn results_can_go_to_an_output_folder_and_text_files_can_be_forced() {
    let dir = temp("outdir");
    let input = dir.join("in");
    std::fs::create_dir_all(input.join("sub")).unwrap();
    std::fs::write(input.join("a.pdf"), scan()).unwrap();
    std::fs::write(input.join("sub").join("b.pdf"), typed()).unwrap();
    let mut o = FolderOptions::new(&input, OcrSettings::default());
    o.recursive = true;
    o.output_dir = Some(dir.join("out"));
    let report = run_with(&o, &fake, write_file, |_| true).unwrap();
    assert_eq!((report.ok, report.skipped), (1, 1), "{report:?}");
    assert!(dir.join("out").join("a.pdf").exists());
    assert!(!input.join("a_ocr.pdf").exists());
    // Without skipping files that have text, a typed file has nothing to read: every page has text.
    o.skip_with_text = false;
    o.overwrite = true;
    let report = run_with(&o, &fake, write_file, |_| true).unwrap();
    assert!(matches!(status(&report, "b.pdf"), FileStatus::Skipped(why) if why.contains("every page")), "{report:?}");
}

#[test]
fn the_source_is_replaced_only_when_overwrite_is_set() {
    let dir = temp("overwrite");
    std::fs::write(dir.join("a.pdf"), scan()).unwrap();
    let before = std::fs::read(dir.join("a.pdf")).unwrap();
    let mut o = FolderOptions::new(&dir, OcrSettings::default());
    o.suffix = Some(String::new());
    assert!(run_with(&o, &fake, write_file, |_| true).unwrap_err().contains("overwrite"));
    assert_eq!(std::fs::read(dir.join("a.pdf")).unwrap(), before);
    o.overwrite = true;
    let report = run_with(&o, &fake, write_file, |_| true).unwrap();
    assert_eq!(report.ok, 1, "{report:?}");
    assert!(text_of(std::fs::read(dir.join("a.pdf")).unwrap()).contains("日本語"));
}

#[test]
fn a_result_never_replaces_another_source_even_with_overwrite() {
    let dir = temp("other-source");
    std::fs::write(dir.join("a.pdf"), scan()).unwrap();
    std::fs::write(dir.join("a_ocr.pdf"), scan()).unwrap();
    let before = std::fs::read(dir.join("a_ocr.pdf")).unwrap();
    let mut o = FolderOptions::new(&dir, OcrSettings::default());
    o.overwrite = true;
    let report = run_with(&o, &fake, write_file, |_| true).unwrap();
    assert!(matches!(status(&report, "a.pdf"), FileStatus::Failed(why) if why.contains("another source")), "{report:?}");
    // a_ocr.pdf is untouched and was read as a source of its own.
    assert_eq!(std::fs::read(dir.join("a_ocr.pdf")).unwrap(), before);
    assert!(matches!(status(&report, "a_ocr.pdf"), FileStatus::Ok { .. }), "{report:?}");
    assert!(std::fs::read(dir.join("a_ocr_ocr.pdf")).is_ok());
}

#[test]
fn files_that_have_text_need_no_engine_but_a_scan_without_one_ends_the_run_with_the_reason() {
    let dir = temp("noengine");
    std::fs::write(dir.join("typed.pdf"), typed()).unwrap();
    let o = FolderOptions::new(&dir, OcrSettings::default());
    let report = run_with(&o, &no_engine, write_file, |_| true).unwrap();
    assert_eq!(report.skipped, 1, "{report:?}");
    std::fs::write(dir.join("scan.pdf"), scan()).unwrap();
    let err = run_with(&o, &no_engine, write_file, |_| true).unwrap_err();
    assert!(err.contains("no engine installed"), "{err}");
    assert!(!dir.join("scan_ocr.pdf").exists());
}

#[test]
fn a_failing_write_fails_the_file_and_a_password_is_a_skip() {
    let dir = temp("writefail");
    std::fs::write(dir.join("a.pdf"), scan()).unwrap();
    let o = FolderOptions::new(&dir, OcrSettings::default());
    let report = run_with(&o, &fake, |_, _| Err("disk full".into()), |_| true).unwrap();
    assert!(matches!(status(&report, "a.pdf"), FileStatus::Failed(why) if why == "disk full"), "{report:?}");
    assert!(is_password_error(&pdfcraft_render::OpenError::NeedsPassword.to_string()));
    assert!(!is_password_error("the file is not a readable PDF"));
}

#[test]
fn an_oversize_file_fails_alone() {
    let dir = temp("oversize");
    std::fs::write(dir.join("a.pdf"), scan()).unwrap();
    let mut o = FolderOptions::new(&dir, OcrSettings::default());
    o.max_file_bytes = 10;
    let report = run_with(&o, &fake, write_file, |_| true).unwrap();
    assert!(matches!(status(&report, "a.pdf"), FileStatus::Failed(why) if why.contains("one file may be")), "{report:?}");
}

#[test]
fn a_stop_takes_effect_between_the_pages_of_a_file() {
    let dir = temp("stop-pages");
    let page = {
        let mut s = Session::new();
        let text = s.create_from_text("t", "Scanned words on a picture").unwrap();
        let id = s.open("text.pdf", None, text, None).unwrap();
        export::Exporter::new(s.get(id).unwrap()).png(0, 100.0).unwrap()
    };
    let pages: Vec<(String, Vec<u8>)> = (0..3).map(|k| (format!("p{k}.png"), page.clone())).collect();
    std::fs::write(dir.join("long.pdf"), Session::new().create_from_images(&pages).unwrap().to_vec()).unwrap();
    let o = FolderOptions::new(&dir, OcrSettings::default());
    let mut calls = 0;
    // The first call is before the file, the second before its first page: stop after one page.
    let report = run_with(&o, &fake, write_file, |_| {
        calls += 1;
        calls < 3
    })
    .unwrap();
    assert!(report.cancelled, "{report:?}");
    assert!(matches!(status(&report, "long.pdf"), FileStatus::Skipped(why) if why.contains("stopped")), "{report:?}");
    assert!(!dir.join("long_ocr.pdf").exists(), "nothing is written for a stopped file");
}
