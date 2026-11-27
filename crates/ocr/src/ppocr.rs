//! PP-OCRv5 (server), the high-accuracy Japanese engine: PaddlePaddle's detection and
//! recognition networks (Apache-2.0, `[[fetched]] optional = true` in ATTRIBUTION.toml, fetched
//! only by `cargo xtask models --optional`, never committed or packaged) run on rten, on the CPU.
//!
//! Written from the public description of the pipeline (docs/ocr-ppocr.ja.md):
//! - **Detection (DB).** The page, scaled so its long side is [`PpOcr::det_side`] (sides rounded
//!   to multiples of 32) and normalised with the ImageNet mean and deviation, gives a text
//!   probability map. It is thresholded at [`BIN_THRESHOLD`]; each 4-connected region whose mean
//!   probability reaches [`BOX_THRESHOLD`] is a box, grown on every side by its
//!   area × [`UNCLIP_RATIO`] / perimeter.
//! - **Direction.** When most boxes are tall (or the language is `ja-vert`), the page is
//!   vertical: its tall boxes are columns and are turned a quarter left before reading (the
//!   column's top becomes the line's left end), and the reading order is columns right to left.
//! - **Recognition (CTC).** Each box is scaled to [`REC_HEIGHT`] px high; widths are rounded up to a
//!   multiple of [`REC_BUCKET`] (padded with mid grey, as PaddleOCR pads) and up to [`REC_BATCH`]
//!   lines of one width go through the network together. Greedy CTC decoding (blank = class 0)
//!   maps class `i` to entry `i - 1` of the model's own dictionary (`character_dict` in its
//!   `inference.yml`), with a space as the optional last class.
//! - **Words.** The time step each character is emitted at places it along its line, so every
//!   CJK character is a word with its own box (a text layer then selects one character at a
//!   time, also in vertical columns) and Latin runs between spaces are words.
//!
//! Nothing the files say is trusted: each file must have the SHA-256 ATTRIBUTION.toml records
//! before it is parsed (rten's parser can spin on bytes that are not a model), output shapes
//! are checked against their lengths, and the counts of boxes and dictionary entries are
//! capped. The networks run behind [`Network`], so tests feed synthetic outputs and the
//! benchmark (`examples/ocr_bench`) can time another runtime with the same pre- and
//! post-processing.

use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};

use super::{Line, OcrError, OcrImage, RecognizeOptions, Recognizer, Word, is_cjk};

/// The model files, as ATTRIBUTION.toml names them.
pub const PPOCR_DETECTION_MODEL: &str = "PP-OCRv5_server_det.onnx";
pub const PPOCR_RECOGNITION_MODEL: &str = "PP-OCRv5_server_rec.onnx";
/// The recogniser's `inference.yml`: its character dictionary.
pub const PPOCR_DICTIONARY: &str = "PP-OCRv5_server_rec.yml";
/// Their SHA-256s, as in ATTRIBUTION.toml. Nothing else is loaded.
pub const PPOCR_DETECTION_MODEL_SHA256: &str = "10803475a591f7dc623e24670fb5752ec94d39a1f8cf069aac1b6f0ce19cfc85";
pub const PPOCR_RECOGNITION_MODEL_SHA256: &str = "d9dc333c9c7b042c6dffb8e33d72b6f65c9c1d463d0a3c2f78174fea55e94752";
pub const PPOCR_DICTIONARY_SHA256: &str = "2c719dba044c4e2228aef8ff92f5f575394d75d24c16de096a33b7cfd902f66d";

/// The languages PP-OCRv5 reads, as `(code, English name)` from [`crate::LANGUAGES`]. `ja-vert`
/// reads every page as vertical; `ja` decides per page.
pub const PPOCR_LANGUAGES: &[(&str, &str)] =
    &[("ja", "Japanese"), ("ja-vert", "Japanese (vertical)"), ("zh", "Chinese (Simplified)"), ("en", "English")];

/// The engine's id: what [`Word::source`] carries and what settings and tools name it by.
pub const PPOCR_ID: &str = "ppocr";

/// The detection input's long side when none is chosen. On real catalogue pages 1600 read best
/// (docs/ocr-ppocr.ja.md §5.5): smaller loses small print, larger splits table cells.
pub const DEFAULT_DET_SIDE: usize = 1600;
/// The range [`PpOcr::with_det_side`] clamps to: a 2560 × 2560 input is 75 MB of f32s.
pub const MIN_DET_SIDE: usize = 32;
pub const MAX_DET_SIDE: usize = 2560;

/// Probability-map threshold for text pixels.
pub const BIN_THRESHOLD: f32 = 0.3;
/// Mean probability a region needs to count as a box.
pub const BOX_THRESHOLD: f32 = 0.6;
/// How far a box grows: area × this / perimeter on each side.
pub const UNCLIP_RATIO: f32 = 1.5;
/// Recogniser input height, in pixels.
pub const REC_HEIGHT: usize = 48;
/// Recogniser widths are rounded up to a multiple of this.
pub const REC_BUCKET: usize = 160;
/// Lines read in one recogniser run, at most.
pub const REC_BATCH: usize = 8;
/// The widest recogniser input, in pixels (PaddleOCR's own limit).
pub const REC_MAX_WIDTH: usize = 3200;
/// Boxes read per page, at most: the largest real catalogue page had about 400. Beyond this the
/// map is noise, and each box costs a recogniser run.
pub const MAX_BOXES: usize = 4000;
/// Dictionary entries, at most (the model's has 18,383).
pub const MAX_DICTIONARY: usize = 100_000;
/// The largest model file read, in bytes (the recogniser is 85 MB, the detector 88 MB).
const MAX_FILE_BYTES: u64 = 512 * 1024 * 1024;

/// One network: an NCHW f32 input in, its first output out as `(shape, values)`. The engine
/// checks the shape against the values; implementations need not be trusted to.
pub trait Network: Send + Sync {
    fn run(&self, shape: [usize; 4], data: Vec<f32>) -> Result<(Vec<usize>, Vec<f32>), OcrError>;

    /// The output's shape and, for each row along its last axis, the best class and its value
    /// (the first largest). Greedy CTC needs nothing more, so a GPU can reduce where the
    /// probabilities are and send back two numbers a row instead of 18,385. The default runs
    /// [`Network::run`] and reduces with [`best_of`].
    fn run_best(&self, shape: [usize; 4], data: Vec<f32>) -> Result<(Vec<usize>, Vec<Best>), OcrError> {
        let (shape, data) = self.run(shape, data)?;
        let best = best_of(&shape, &data)?;
        Ok((shape, best))
    }
}

/// A row's best class and its value.
pub type Best = (usize, f32);

/// The first largest value of each row along the last axis of `shape`.
pub fn best_of(shape: &[usize], data: &[f32]) -> Result<Vec<Best>, OcrError> {
    let classes = shape.last().copied().filter(|&c| c > 0).ok_or_else(|| recognize(format!("output shape {shape:?} has no classes")))?;
    let n = shape.iter().try_fold(1usize, |a, &d| a.checked_mul(d));
    if n != Some(data.len()) {
        return Err(recognize(format!("output shape {shape:?} does not hold its {} values", data.len())));
    }
    Ok(data.chunks(classes).map(|row| row.iter().enumerate().fold((0usize, f32::MIN), |a, (i, &p)| if p > a.1 { (i, p) } else { a })).collect())
}

/// Makes the two networks on another device (a GPU), from the model files' bytes (already
/// checked against their SHA-256). This crate cannot know about GPUs (they are an L7 concern),
/// so the program (L8) registers one with the engine, which hands it to
/// [`PpOcr::load_with_gpu`]. `Err` means it cannot run them here, and the reader stays on the
/// CPU.
/// The detection and recognition networks, in that order.
pub type Networks = (Box<dyn Network>, Box<dyn Network>);

pub trait NetworkFactory: Send + Sync {
    /// What runs the networks, for notes and status ("wgpu on ...").
    fn name(&self) -> String;
    /// The detection and recognition networks.
    fn networks(&self, detection: &[u8], recognition: &[u8]) -> Result<Networks, OcrError>;
}

/// Largest difference between the GPU's and the CPU's outputs the self-check accepts. Both
/// networks end in probabilities (0..1); the prototype kernels differed by under 1e-4.
pub const SELF_CHECK_TOLERANCE: f32 = 1e-3;

/// The first reason this session fell back from the GPU to the CPU, and whether it was handed
/// out ([`take_gpu_fallback_notice`]).
static FALLBACK_NOTICE: Mutex<(Option<String>, bool)> = Mutex::new((None, false));

fn report_fallback(why: &str) {
    let mut n = FALLBACK_NOTICE.lock().unwrap_or_else(|e| e.into_inner());
    if n.0.is_none() {
        n.0 = Some(why.to_string());
    }
}

/// Why recognition moved from the GPU to the CPU, once a session: the first call after the
/// first fallback returns it, every other call `None`. (Each page's notes say it every time.)
pub fn take_gpu_fallback_notice() -> Option<String> {
    let mut n = FALLBACK_NOTICE.lock().unwrap_or_else(|e| e.into_inner());
    if n.1 {
        return None;
    }
    let why = n.0.clone()?;
    n.1 = true;
    Some(why)
}

/// The networks on a GPU, while they behave.
struct Gpu {
    det: Box<dyn Network>,
    rec: Box<dyn Network>,
    name: String,
    /// Cleared on the first failure: the rest of the session reads on the CPU.
    healthy: AtomicBool,
    /// Why it was given up, for every later page's notes.
    failed: Mutex<Option<String>>,
}

#[derive(Clone, Copy)]
enum Net {
    Det,
    Rec,
}

/// An output whose shape matches its length and whose values are all finite: anything else
/// from a GPU is a failure, never a reading.
fn sane(out: (Vec<usize>, Vec<f32>)) -> Result<(Vec<usize>, Vec<f32>), OcrError> {
    let n = out.0.iter().try_fold(1usize, |a, &d| a.checked_mul(d));
    if n != Some(out.1.len()) {
        return Err(recognize(format!("output shape {:?} does not hold its {} values", out.0, out.1.len())));
    }
    if out.1.iter().any(|v| !v.is_finite()) {
        return Err(recognize("output has values that are not finite"));
    }
    Ok(out)
}

/// A best-per-row output that matches its shape, names classes that exist and holds only
/// finite values.
fn sane_best(out: (Vec<usize>, Vec<Best>)) -> Result<(Vec<usize>, Vec<Best>), OcrError> {
    let (lead, classes) = match out.0.split_last() {
        Some((c, lead)) if *c > 0 => (lead, *c),
        _ => return Err(recognize(format!("output shape {:?} has no classes", out.0))),
    };
    let rows = lead.iter().try_fold(1usize, |a, &d| a.checked_mul(d));
    if rows != Some(out.1.len()) {
        return Err(recognize(format!("output shape {:?} does not hold its {} rows", out.0, out.1.len())));
    }
    if out.1.iter().any(|&(c, p)| c >= classes || !p.is_finite()) {
        return Err(recognize("output has classes outside the model or values that are not finite"));
    }
    Ok(out)
}

/// A fixed input for the self-check: a smooth pattern in -1..1.
fn probe(shape: [usize; 4]) -> Vec<f32> {
    let n = shape.iter().product::<usize>();
    (0..n).map(|i| ((i % 997) as f32 * 0.37).sin()).collect()
}

/// Run one network on both and compare (shapes equal, every value within
/// [`SELF_CHECK_TOLERANCE`]).
fn self_check(cpu: &dyn Network, gpu: &dyn Network, shape: [usize; 4]) -> Result<(), String> {
    let want = cpu.run(shape, probe(shape)).and_then(sane).map_err(|e| format!("the CPU check failed: {e}"))?;
    let got = gpu.run(shape, probe(shape)).and_then(sane).map_err(|e| e.to_string())?;
    if got.0 != want.0 {
        return Err(format!("its output shape {:?} differs from the CPU's {:?}", got.0, want.0));
    }
    let worst = got.1.iter().zip(&want.1).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    if worst > SELF_CHECK_TOLERANCE {
        return Err(format!("its results differ from the CPU's by up to {worst}"));
    }
    Ok(())
}

/// The recogniser's self-check: the GPU's best class per row must be one the CPU scores within
/// [`SELF_CHECK_TOLERANCE`] of its own best (a near tie may go either way), with that score.
fn self_check_best(cpu: &dyn Network, gpu: &dyn Network, shape: [usize; 4]) -> Result<(), String> {
    let want = cpu.run(shape, probe(shape)).and_then(sane).map_err(|e| format!("the CPU check failed: {e}"))?;
    let got = gpu.run_best(shape, probe(shape)).and_then(sane_best).map_err(|e| e.to_string())?;
    if got.0 != want.0 {
        return Err(format!("its output shape {:?} differs from the CPU's {:?}", got.0, want.0));
    }
    let classes = want.0.last().copied().unwrap_or(1).max(1);
    for (row, &(c, p)) in want.1.chunks(classes).zip(&got.1) {
        let best = row.iter().copied().fold(f32::MIN, f32::max);
        let at = row.get(c).copied().unwrap_or(f32::MIN);
        if best - at > SELF_CHECK_TOLERANCE || (p - best).abs() > SELF_CHECK_TOLERANCE {
            return Err(format!("its best classes differ from the CPU's (class {c} at {p}, the CPU's best {best})"));
        }
    }
    Ok(())
}

/// A network run by rten on the CPU, from ONNX bytes (for comparing other runtimes with).
pub fn cpu_network(model: Vec<u8>) -> Result<Box<dyn Network>, OcrError> {
    Ok(Box::new(Rten(rten::Model::load(model).map_err(|e| OcrError::Load("rten".into(), e.to_string()))?)))
}

/// rten on the CPU.
struct Rten(rten::Model);

impl Network for Rten {
    fn run(&self, shape: [usize; 4], data: Vec<f32>) -> Result<(Vec<usize>, Vec<f32>), OcrError> {
        use rten_tensor::prelude::*;
        let input = rten_tensor::NdTensor::<f32, 4>::try_from_data(shape, data).map_err(recognize)?;
        let out: rten_tensor::Tensor<f32> = self.0.run_one(input.into(), None).map_err(recognize)?.try_into().map_err(recognize)?;
        Ok((out.shape().to_vec(), out.to_vec()))
    }
}

fn recognize(e: impl std::fmt::Display) -> OcrError {
    OcrError::Recognize(e.to_string())
}

fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    Sha256::digest(bytes).iter().map(|b| format!("{b:02x}")).collect()
}

/// `data` when its SHA-256 is `sha256`, else an error naming the file.
fn verified(name: &str, sha256: &str, data: Vec<u8>) -> Result<Vec<u8>, OcrError> {
    if sha256_hex(&data) == sha256 {
        Ok(data)
    } else {
        Err(OcrError::Load(
            name.into(),
            "not the expected PP-OCRv5 file (SHA-256 differs); fetch it again with `cargo xtask models --optional`".into(),
        ))
    }
}

/// Read a file of at most [`MAX_FILE_BYTES`].
fn read_capped(path: &Path) -> Result<Vec<u8>, OcrError> {
    use std::io::Read;
    let what = |e: std::io::Error| OcrError::Load(path.display().to_string(), e.to_string());
    let file = std::fs::File::open(path).map_err(what)?;
    let mut data = Vec::new();
    file.take(MAX_FILE_BYTES + 1).read_to_end(&mut data).map_err(what)?;
    if data.len() as u64 > MAX_FILE_BYTES {
        return Err(OcrError::Load(path.display().to_string(), format!("larger than {MAX_FILE_BYTES} bytes")));
    }
    Ok(data)
}

/// Where the three PP-OCRv5 files are.
#[derive(Clone, Debug, PartialEq)]
pub struct PpOcrModels {
    pub detection: PathBuf,
    pub recognition: PathBuf,
    pub dictionary: PathBuf,
}

impl PpOcrModels {
    /// The files in `dir`, when all three are there.
    pub fn in_dir(dir: &Path) -> Option<PpOcrModels> {
        let m = PpOcrModels {
            detection: dir.join(PPOCR_DETECTION_MODEL),
            recognition: dir.join(PPOCR_RECOGNITION_MODEL),
            dictionary: dir.join(PPOCR_DICTIONARY),
        };
        (m.detection.is_file() && m.recognition.is_file() && m.dictionary.is_file()).then_some(m)
    }

    /// Look for them where [`crate::Models::find`] looks for the ocrs models.
    pub fn find() -> Option<PpOcrModels> {
        crate::Models::search_dirs().iter().find_map(|d| Self::in_dir(d))
    }
}

/// The dictionary in the recogniser's `inference.yml`: the YAML list under `character_dict:`,
/// one `  - entry` per line (plain, single-quoted or double-quoted), at most [`MAX_DICTIONARY`].
pub fn read_dictionary(yml: &str) -> Result<Vec<String>, OcrError> {
    let mut out = Vec::new();
    let mut inside = false;
    for line in yml.lines() {
        if !inside {
            inside = line.trim_start().starts_with("character_dict:");
            continue;
        }
        let Some(item) = line.strip_prefix("  - ") else { break };
        let quoted = |q: char| item.len() >= 2 && item.starts_with(q) && item.ends_with(q);
        let item = if quoted('\'') {
            item.get(1..item.len() - 1).unwrap_or("").replace("''", "'")
        } else if quoted('"') {
            item.get(1..item.len() - 1).unwrap_or("").replace("\\\"", "\"").replace("\\\\", "\\")
        } else {
            item.to_string()
        };
        if out.len() == MAX_DICTIONARY {
            return Err(OcrError::Load(PPOCR_DICTIONARY.into(), format!("more than {MAX_DICTIONARY} dictionary entries")));
        }
        out.push(item);
    }
    if out.is_empty() {
        return Err(OcrError::Load(PPOCR_DICTIONARY.into(), "no character_dict list".into()));
    }
    Ok(out)
}

/// A loaded PP-OCRv5 reader. Loading takes a few seconds (two 85 MB networks, hashed first);
/// keep one and reuse it. It is `Send + Sync`: pages may be read on several threads at once.
pub struct PpOcr {
    /// The networks on the CPU: the reading when there is no GPU, and the fallback when it fails.
    det: Box<dyn Network>,
    rec: Box<dyn Network>,
    gpu: Option<Gpu>,
    /// The GPU was asked for and is not used: why (every page's notes say it).
    gpu_refused: Option<String>,
    dict: Vec<String>,
    det_side: usize,
    id: &'static str,
}

/// A text box in page pixels, `[x0, y0, x1, y1]` (end-exclusive), with its mean probability.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TextBox {
    pub rect: [usize; 4],
    pub score: f32,
}

impl TextBox {
    fn width(&self) -> usize {
        self.rect[2].saturating_sub(self.rect[0])
    }
    fn height(&self) -> usize {
        self.rect[3].saturating_sub(self.rect[1])
    }
    /// At least 1.5 times as tall as it is wide: a column of vertical text.
    fn is_tall(&self) -> bool {
        self.height() as f32 >= self.width() as f32 * 1.5
    }
}

/// One character the recogniser read, at `[start, end)` along its line in crop pixels.
#[derive(Clone, Debug, PartialEq)]
pub struct ReadChar {
    pub text: String,
    pub start: f32,
    pub end: f32,
    pub probability: f32,
}

impl PpOcr {
    /// Load the files in `models`, each checked against its SHA-256 first.
    pub fn load(models: &PpOcrModels) -> Result<PpOcr, OcrError> {
        Self::from_bytes(read_capped(&models.detection)?, read_capped(&models.recognition)?, read_capped(&models.dictionary)?)
    }

    /// Load the files found by [`PpOcrModels::find`].
    pub fn find() -> Result<PpOcr, OcrError> {
        Self::load(&PpOcrModels::find().ok_or(OcrError::NoPpOcrModels)?)
    }

    /// Load the files from their contents. Bytes that are not the files ATTRIBUTION.toml names
    /// are an error, before anything parses them.
    pub fn from_bytes(detection: Vec<u8>, recognition: Vec<u8>, dictionary: Vec<u8>) -> Result<PpOcr, OcrError> {
        Self::from_bytes_with(detection, recognition, dictionary, None)
    }

    /// [`PpOcr::load`], with the networks also made on a GPU by `factory`. The CPU networks
    /// are loaded too, as the fallback. A factory that cannot make them, or networks that fail
    /// the self-check (the same fixed input through the CPU and the GPU must agree within
    /// [`SELF_CHECK_TOLERANCE`]), leave the reader on the CPU, saying why on every page.
    pub fn load_with_gpu(models: &PpOcrModels, factory: &dyn NetworkFactory) -> Result<PpOcr, OcrError> {
        Self::from_bytes_with(read_capped(&models.detection)?, read_capped(&models.recognition)?, read_capped(&models.dictionary)?, Some(factory))
    }

    fn from_bytes_with(
        detection: Vec<u8>,
        recognition: Vec<u8>,
        dictionary: Vec<u8>,
        factory: Option<&dyn NetworkFactory>,
    ) -> Result<PpOcr, OcrError> {
        let dictionary = verified(PPOCR_DICTIONARY, PPOCR_DICTIONARY_SHA256, dictionary)?;
        let yml = String::from_utf8(dictionary).map_err(|e| OcrError::Load(PPOCR_DICTIONARY.into(), e.to_string()))?;
        let model = |name: &str, sha256: &str, data: Vec<u8>| -> Result<Box<dyn Network>, OcrError> {
            let data = verified(name, sha256, data)?;
            Ok(Box::new(Rten(rten::Model::load(data).map_err(|e| OcrError::Load(name.into(), e.to_string()))?)))
        };
        let detection = verified(PPOCR_DETECTION_MODEL, PPOCR_DETECTION_MODEL_SHA256, detection)?;
        let recognition = verified(PPOCR_RECOGNITION_MODEL, PPOCR_RECOGNITION_MODEL_SHA256, recognition)?;
        let gpu = factory.map(|f| (f.name(), f.networks(&detection, &recognition)));
        let det = model(PPOCR_DETECTION_MODEL, PPOCR_DETECTION_MODEL_SHA256, detection)?;
        let rec = model(PPOCR_RECOGNITION_MODEL, PPOCR_RECOGNITION_MODEL_SHA256, recognition)?;
        let cpu = Self::with_networks(det, rec, &yml, PPOCR_ID)?;
        Ok(match gpu {
            None => cpu,
            Some((name, Ok((det, rec)))) => cpu.with_gpu(det, rec, name),
            Some((name, Err(e))) => cpu.with_gpu_refused(format!("the GPU ({name}) is not used: {e}; reading on the CPU")),
        })
    }

    /// A reader over networks run some other way (tests, the benchmark's other runtimes), with
    /// the dictionary from the recogniser's `inference.yml` and `id` as the words' source.
    pub fn with_networks(det: Box<dyn Network>, rec: Box<dyn Network>, dictionary_yml: &str, id: &'static str) -> Result<PpOcr, OcrError> {
        Ok(PpOcr { det, rec, gpu: None, gpu_refused: None, dict: read_dictionary(dictionary_yml)?, det_side: DEFAULT_DET_SIDE, id })
    }

    /// The same reader, running `det` and `rec` on a GPU named `name` first and its own
    /// networks as the fallback, when they pass the self-check; else it stays on the CPU and
    /// says why on every page.
    pub fn with_gpu(mut self, det: Box<dyn Network>, rec: Box<dyn Network>, name: String) -> PpOcr {
        let checked = self_check(self.det.as_ref(), det.as_ref(), [1, 3, 64, 64])
            .and_then(|()| self_check_best(self.rec.as_ref(), rec.as_ref(), [1, 3, REC_HEIGHT, REC_BUCKET]));
        match checked {
            Ok(()) => {
                self.gpu = Some(Gpu { det, rec, name, healthy: AtomicBool::new(true), failed: Mutex::new(None) });
                self.gpu_refused = None;
                self
            }
            Err(why) => self.with_gpu_refused(format!("the GPU ({name}) is not used: the self-check failed, {why}; reading on the CPU")),
        }
    }

    /// The same reader on the CPU, with `why` the GPU that was asked for is not used (noted on
    /// every page, and the session's fallback notice).
    pub fn with_gpu_refused(mut self, why: String) -> PpOcr {
        report_fallback(&why);
        self.gpu = None;
        self.gpu_refused = Some(why);
        self
    }

    /// What runs the networks now: `"rten"` (the CPU), or the GPU's name while it works.
    pub fn backend(&self) -> String {
        match &self.gpu {
            Some(g) if g.healthy.load(Ordering::Relaxed) => g.name.clone(),
            _ => "rten".into(),
        }
    }

    /// Run one network with `f`: on the GPU while it works (an error, or an output that is not
    /// sane, gives it up for the session, with a note), else on the CPU. `f` is told whether
    /// it runs on the GPU (whose outputs it checks).
    fn exec<T>(&self, net: Net, notes: &mut Vec<String>, f: impl Fn(&dyn Network, bool) -> Result<T, OcrError>) -> Result<T, OcrError> {
        let cpu = match net {
            Net::Det => self.det.as_ref(),
            Net::Rec => self.rec.as_ref(),
        };
        if let Some(gpu) = self.gpu.as_ref().filter(|g| g.healthy.load(Ordering::Relaxed)) {
            let on_gpu = match net {
                Net::Det => gpu.det.as_ref(),
                Net::Rec => gpu.rec.as_ref(),
            };
            match f(on_gpu, true) {
                Ok(out) => return Ok(out),
                Err(e) => {
                    let why = format!("the GPU ({}) failed: {e}; this page and the rest of the session are read on the CPU", gpu.name);
                    gpu.healthy.store(false, Ordering::Relaxed);
                    *gpu.failed.lock().unwrap_or_else(|e| e.into_inner()) = Some(why.clone());
                    report_fallback(&why);
                    notes.push(why);
                }
            }
        }
        f(cpu, false)
    }

    /// [`PpOcr::read`], with the page's notes: why the GPU is not used (every page), or that it
    /// failed on this page.
    pub fn read_noted(&self, image: &OcrImage, vertical: Option<bool>) -> Result<(Vec<Line>, Vec<String>), OcrError> {
        let mut notes = Vec::new();
        if let Some(why) = &self.gpu_refused {
            notes.push(why.clone());
        } else if let Some(g) = self.gpu.as_ref().filter(|g| !g.healthy.load(Ordering::Relaxed))
            && let Some(why) = g.failed.lock().unwrap_or_else(|e| e.into_inner()).clone()
        {
            notes.push(why);
        }
        let lines = self.read_into(image, vertical, &mut notes)?;
        Ok((lines, notes))
    }

    /// The same reader with another detection input size (clamped to
    /// [`MIN_DET_SIDE`]..=[`MAX_DET_SIDE`]).
    pub fn with_det_side(mut self, side: usize) -> PpOcr {
        self.det_side = side.clamp(MIN_DET_SIDE, MAX_DET_SIDE);
        self
    }

    /// The detection input's long side, in pixels.
    pub fn det_side(&self) -> usize {
        self.det_side
    }

    /// Read `image`. `vertical`: `Some(true)` reads it as a vertical page, `Some(false)` as a
    /// horizontal one, `None` decides from its boxes.
    pub fn read(&self, image: &OcrImage, vertical: Option<bool>) -> Result<Vec<Line>, OcrError> {
        self.read_into(image, vertical, &mut Vec::new())
    }

    fn read_into(&self, image: &OcrImage, vertical: Option<bool>, notes: &mut Vec<String>) -> Result<Vec<Line>, OcrError> {
        let (w, h) = (image.width() as usize, image.height() as usize);
        let mut grey = Vec::with_capacity(w.saturating_mul(h));
        for y in 0..h {
            for x in 0..w {
                grey.push(image.gray(x as i64, y as i64).unwrap_or(255));
            }
        }
        let page = Grey { px: &grey, w, h };
        let boxes = self.detect(&page, notes)?;
        let vertical_page = vertical.unwrap_or_else(|| boxes.iter().filter(|b| b.is_tall()).count() * 2 > boxes.len());
        let lines: Vec<Pending> = boxes
            .into_iter()
            .map(|b| {
                let turned = vertical_page && b.is_tall();
                let (along, across) = if turned { (b.height(), b.width()) } else { (b.width(), b.height()) };
                let rw = ((along as f32 * REC_HEIGHT as f32 / across.max(1) as f32).round() as usize).clamp(16, REC_MAX_WIDTH);
                Pending { b, turned, rw }
            })
            .collect();
        let read = self.read_lines(&page, &lines, notes)?;
        let mut placed: Vec<(TextBox, Vec<Word>)> = Vec::new();
        for (line, chars) in lines.iter().zip(read) {
            let words = self.words(line, &chars);
            if !words.is_empty() {
                placed.push((line.b, words));
            }
        }
        order(&mut placed, vertical_page);
        Ok(placed.into_iter().map(|(_, words)| Line { words }).collect())
    }

    /// Text boxes in page pixels.
    fn detect(&self, page: &Grey, notes: &mut Vec<String>) -> Result<Vec<TextBox>, OcrError> {
        let (w, h) = (page.w, page.h);
        if w == 0 || h == 0 {
            return Err(OcrError::EmptyImage);
        }
        let scale = self.det_side as f32 / w.max(h) as f32;
        let round32 = |v: f32| ((v / 32.0).round().max(1.0) as usize).saturating_mul(32).min(MAX_DET_SIDE);
        let (dw, dh) = (round32(w as f32 * scale), round32(h as f32 * scale));
        let (sx, sy) = (w as f32 / dw as f32, h as f32 / dh as f32);
        let (mean, std) = ([0.485f32, 0.456, 0.406], [0.229f32, 0.224, 0.225]);
        let plane = dh * dw;
        let mut input = vec![0.0f32; 3 * plane];
        for y in 0..dh {
            for x in 0..dw {
                let v = page.sample((x as f32 + 0.5) * sx - 0.5, (y as f32 + 0.5) * sy - 0.5) / 255.0;
                for c in 0..3 {
                    if let Some(slot) = input.get_mut(c * plane + y * dw + x) {
                        *slot = (v - mean[c]) / std[c];
                    }
                }
            }
        }
        let shape4 = [1, 3, dh, dw];
        let (shape, data) = self.exec(Net::Det, notes, |n, gpu| {
            let out = n.run(shape4, input.clone())?;
            if gpu { sane(out) } else { Ok(out) }
        })?;
        let (ph, pw) = match shape.as_slice() {
            [1, 1, ph, pw] if *ph > 0 && *pw > 0 && ph.checked_mul(*pw) == Some(data.len()) => (*ph, *pw),
            other => return Err(recognize(format!("unexpected detection output shape {other:?} for {} values", data.len()))),
        };
        Ok(boxes_from_map(&data, pw, ph, w, h))
    }

    /// Read `lines` in batches of equal padded width, cropping each batch only when it runs
    /// (at most [`REC_BATCH`] lines are held at once). Returns each line's characters, in the
    /// order given.
    fn read_lines(&self, page: &Grey, lines: &[Pending], notes: &mut Vec<String>) -> Result<Vec<Vec<ReadChar>>, OcrError> {
        let mut order: Vec<usize> = (0..lines.len()).collect();
        order.sort_by_key(|&i| lines.get(i).map_or(0, |l| bucket(l.rw)));
        let mut out = vec![Vec::new(); lines.len()];
        for group in order.chunk_by(|&a, &b| lines.get(a).map(|l| bucket(l.rw)) == lines.get(b).map(|l| bucket(l.rw))) {
            for batch in group.chunks(REC_BATCH) {
                let Some(bw) = batch.first().and_then(|&k| lines.get(k)).map(|l| bucket(l.rw)) else { continue };
                let n = batch.len();
                let plane = REC_HEIGHT * bw;
                let mut input = vec![0.0f32; n * 3 * plane];
                for (slot, &k) in batch.iter().enumerate() {
                    let Some(line) = lines.get(k) else { continue };
                    let crop = page.crop(line);
                    for y in 0..REC_HEIGHT {
                        let Some(src) = crop.get(y * line.rw..(y + 1) * line.rw) else { continue };
                        for c in 0..3 {
                            let at = (slot * 3 + c) * plane + y * bw;
                            if let Some(dst) = input.get_mut(at..at + line.rw) {
                                dst.copy_from_slice(src);
                            }
                        }
                    }
                }
                let shape4 = [n, 3, REC_HEIGHT, bw];
                let (shape, best) = self.exec(Net::Rec, notes, |net, gpu| {
                    let out = net.run_best(shape4, input.clone())?;
                    if gpu { sane_best(out) } else { Ok(out) }
                })?;
                let (steps, classes) = match shape.as_slice() {
                    [b, t, c] if *b == n && *t > 0 && *c > 0 && b.checked_mul(*t) == Some(best.len()) => (*t, *c),
                    other => return Err(recognize(format!("unexpected recognition output shape {other:?} for {} rows", best.len()))),
                };
                for (slot, &k) in batch.iter().enumerate() {
                    let seq =
                        best.get(slot * steps..(slot + 1) * steps).ok_or_else(|| recognize("the recognition output is shorter than its shape"))?;
                    let rw = lines.get(k).map_or(bw, |l| l.rw);
                    if let Some(o) = out.get_mut(k) {
                        *o = decode(&self.dict, seq, classes, bw as f32 / steps as f32, rw as f32)?;
                    }
                }
            }
        }
        Ok(out)
    }

    /// The words of one line, placed on the page: each CJK character alone, Latin runs split at
    /// spaces. Positions along the recogniser input are scaled back to the box.
    fn words(&self, line: &Pending, chars: &[ReadChar]) -> Vec<Word> {
        let b = line.b.rect;
        let along = if line.turned { line.b.height() } else { line.b.width() } as f32;
        let k = along / line.rw.max(1) as f32;
        let rect = |s: f32, e: f32| -> [f32; 4] {
            let (s, e) = ((s * k).clamp(0.0, along), (e * k).clamp(0.0, along));
            if line.turned {
                [b[0] as f32, b[1] as f32 + s, b[2] as f32, b[1] as f32 + e.max(s + 1.0)]
            } else {
                [b[0] as f32 + s, b[1] as f32, b[0] as f32 + e.max(s + 1.0), b[3] as f32]
            }
        };
        let mut words = Vec::new();
        let mut run: Option<(String, f32, f32, f32, usize)> = None;
        let flush = |run: &mut Option<(String, f32, f32, f32, usize)>, words: &mut Vec<Word>| {
            if let Some((text, s, e, p, n)) = run.take() {
                words.push(Word::new(text, rect(s, e), self.id).with_confidence(p / n.max(1) as f32 * 100.0));
            }
        };
        for c in chars {
            if c.text.trim().is_empty() {
                flush(&mut run, &mut words);
                continue;
            }
            if c.text.chars().all(is_cjk) {
                flush(&mut run, &mut words);
                words.push(Word::new(c.text.clone(), rect(c.start, c.end), self.id).with_confidence(c.probability * 100.0));
                continue;
            }
            match run.as_mut() {
                Some((text, _, e, p, n)) => {
                    text.push_str(&c.text);
                    *e = c.end;
                    *p += c.probability;
                    *n += 1;
                }
                None => run = Some((c.text.clone(), c.start, c.end, c.probability, 1)),
            }
        }
        flush(&mut run, &mut words);
        words
    }
}

impl Recognizer for PpOcr {
    fn id(&self) -> &'static str {
        self.id
    }

    fn available(&self) -> bool {
        true
    }

    fn languages(&self) -> Vec<(String, String)> {
        PPOCR_LANGUAGES.iter().map(|(c, n)| ((*c).to_string(), (*n).to_string())).collect()
    }

    fn recognize(&self, image: &OcrImage, options: &RecognizeOptions) -> Result<Vec<Line>, OcrError> {
        self.read(image, (options.language == "ja-vert").then_some(true))
    }

    fn recognize_noted(&self, image: &OcrImage, options: &RecognizeOptions) -> Result<(Vec<Line>, Vec<String>), OcrError> {
        self.read_noted(image, (options.language == "ja-vert").then_some(true))
    }
}

/// A box waiting to be read: turned a quarter left when it is a column, `rw` px wide once scaled.
struct Pending {
    b: TextBox,
    turned: bool,
    rw: usize,
}

/// The padded recogniser width for a line `rw` px wide.
fn bucket(rw: usize) -> usize {
    rw.div_ceil(REC_BUCKET).max(1).saturating_mul(REC_BUCKET).min(REC_MAX_WIDTH.div_ceil(REC_BUCKET) * REC_BUCKET)
}

/// The page in grey, 0 (ink) … 255 (paper).
struct Grey<'a> {
    px: &'a [u8],
    w: usize,
    h: usize,
}

impl Grey<'_> {
    /// Bilinear sample, white outside the page.
    fn sample(&self, x: f32, y: f32) -> f32 {
        let px = |xx: i64, yy: i64| -> f32 {
            if xx < 0 || yy < 0 || xx as usize >= self.w || yy as usize >= self.h {
                255.0
            } else {
                self.px.get(yy as usize * self.w + xx as usize).copied().map_or(255.0, f32::from)
            }
        };
        if !x.is_finite() || !y.is_finite() {
            return 255.0;
        }
        let (x0, y0) = (x.floor(), y.floor());
        let (fx, fy) = (x - x0, y - y0);
        let (xi, yi) = (x0 as i64, y0 as i64);
        px(xi, yi) * (1.0 - fx) * (1.0 - fy) + px(xi + 1, yi) * fx * (1.0 - fy) + px(xi, yi + 1) * (1.0 - fx) * fy + px(xi + 1, yi + 1) * fx * fy
    }

    /// One line as recogniser input: [`REC_HEIGHT`] × `rw`, normalised to -1..1. A turned line's
    /// pixel (u along it, v across it) comes from (right edge - v, top + u) of its box.
    fn crop(&self, line: &Pending) -> Vec<f32> {
        let b = line.b.rect;
        let (along, across) = if line.turned { (line.b.height(), line.b.width()) } else { (line.b.width(), line.b.height()) };
        let (ku, kv) = (along as f32 / line.rw.max(1) as f32, across as f32 / REC_HEIGHT as f32);
        let mut out = vec![0.0f32; REC_HEIGHT * line.rw];
        for y in 0..REC_HEIGHT {
            for x in 0..line.rw {
                let (u, v) = ((x as f32 + 0.5) * ku - 0.5, (y as f32 + 0.5) * kv - 0.5);
                let (px, py) = if line.turned { (b[2] as f32 - 1.0 - v, b[1] as f32 + u) } else { (b[0] as f32 + u, b[1] as f32 + v) };
                if let Some(slot) = out.get_mut(y * line.rw + x) {
                    *slot = (self.sample(px, py) / 255.0 - 0.5) / 0.5;
                }
            }
        }
        out
    }
}

/// DB post-processing: the boxes of a `pw` × `ph` probability map, scaled to a `w` × `h` page.
/// At most [`MAX_BOXES`], the highest-scoring kept, in map order.
pub fn boxes_from_map(map: &[f32], pw: usize, ph: usize, w: usize, h: usize) -> Vec<TextBox> {
    let len = pw.checked_mul(ph).unwrap_or(0).min(map.len());
    let mut seen = vec![false; len];
    let on = |i: usize| map.get(i).is_some_and(|&p| p > BIN_THRESHOLD);
    let mut boxes = Vec::new();
    let mut stack = Vec::new();
    let (fx, fy) = (w as f32 / pw.max(1) as f32, h as f32 / ph.max(1) as f32);
    for start in 0..len {
        if !on(start) || seen.get(start).copied().unwrap_or(true) {
            continue;
        }
        let (mut x0, mut y0, mut x1, mut y1, mut area, mut score) = (pw, ph, 0usize, 0usize, 0usize, 0.0f32);
        stack.push(start);
        if let Some(s) = seen.get_mut(start) {
            *s = true;
        }
        while let Some(i) = stack.pop() {
            let (x, y) = (i % pw, i / pw);
            (x0, y0, x1, y1) = (x0.min(x), y0.min(y), x1.max(x), y1.max(y));
            area += 1;
            score += map.get(i).copied().unwrap_or(0.0);
            let mut visit = |j: usize| {
                if j < len && on(j) && !seen.get(j).copied().unwrap_or(true) {
                    if let Some(s) = seen.get_mut(j) {
                        *s = true;
                    }
                    stack.push(j);
                }
            };
            if x > 0 {
                visit(i - 1);
            }
            if x + 1 < pw {
                visit(i + 1);
            }
            if y > 0 {
                visit(i - pw);
            }
            if y + 1 < ph {
                visit(i + pw);
            }
        }
        let (bw, bh) = ((x1 - x0 + 1) as f32, (y1 - y0 + 1) as f32);
        let mean = score / area.max(1) as f32;
        if bw.min(bh) < 3.0 || mean.is_nan() || mean < BOX_THRESHOLD {
            continue;
        }
        let grow = bw * bh * UNCLIP_RATIO / (2.0 * (bw + bh));
        let rect = [
            ((x0 as f32 - grow) * fx).max(0.0) as usize,
            ((y0 as f32 - grow) * fy).max(0.0) as usize,
            (((x1 + 1) as f32 + grow) * fx).min(w as f32) as usize,
            (((y1 + 1) as f32 + grow) * fy).min(h as f32) as usize,
        ];
        if rect[2] > rect[0] + 2 && rect[3] > rect[1] + 2 {
            boxes.push(TextBox { rect, score: mean });
        }
    }
    if boxes.len() > MAX_BOXES {
        let mut by_score: Vec<usize> = (0..boxes.len()).collect();
        by_score.sort_by(|&a, &b| {
            let s = |i: usize| boxes.get(i).map_or(0.0, |t| t.score);
            s(b).total_cmp(&s(a))
        });
        let mut keep = vec![false; boxes.len()];
        for &i in by_score.iter().take(MAX_BOXES) {
            if let Some(k) = keep.get_mut(i) {
                *k = true;
            }
        }
        let mut i = 0;
        boxes.retain(|_| {
            let k = keep.get(i).copied().unwrap_or(false);
            i += 1;
            k
        });
    }
    boxes
}

/// Greedy CTC over one line's best class per time step (of `classes`). `step_px` is the input width
/// one time step covers; positions are clamped to the line's `rw` real pixels (the rest is
/// padding). A character runs from its first step to the next character's first step (the
/// last one to the end of its own steps).
pub fn decode(dict: &[String], best: &[Best], classes: usize, step_px: f32, rw: f32) -> Result<Vec<ReadChar>, OcrError> {
    // Classes: blank, the dictionary, then (when the model has one more) a space.
    let space = dict.len() + 1;
    if classes != space && classes != space + 1 {
        return Err(recognize(format!("the recogniser has {classes} classes; its dictionary implies {space} or {}", space + 1)));
    }
    let mut out: Vec<ReadChar> = Vec::new();
    let mut last = 0usize;
    let mut last_end = 0.0f32;
    for (t, &(best, p)) in best.iter().enumerate() {
        if best >= classes {
            return Err(recognize(format!("class {best} of {classes}")));
        }
        let at = (t as f32 * step_px).min(rw);
        if best != 0 && best != last {
            let text = if best == space { " ".to_string() } else { dict.get(best - 1).cloned().unwrap_or_default() };
            if let Some(prev) = out.last_mut() {
                prev.end = at.max(prev.start);
            }
            out.push(ReadChar { text, start: at, end: at, probability: if p.is_finite() { p } else { 0.0 } });
        }
        if best != 0 {
            last_end = ((t + 1) as f32 * step_px).min(rw);
        }
        last = best;
    }
    if let Some(prev) = out.last_mut() {
        prev.end = last_end.max(prev.start);
    }
    Ok(out)
}

/// Reading order: columns right to left (top to bottom inside one) on a vertical page; else
/// rows top to bottom, a row being boxes whose middles are within half the first box's height,
/// left to right.
fn order(read: &mut Vec<(TextBox, Vec<Word>)>, vertical_page: bool) {
    if vertical_page {
        read.sort_by(|a, b| b.0.rect[2].cmp(&a.0.rect[2]).then(a.0.rect[1].cmp(&b.0.rect[1])));
        return;
    }
    // Group into rows first (a total order per step; a tolerance comparator is not one).
    read.sort_by_key(|r| r.0.rect[1] + r.0.rect[3]);
    let mut rows: Vec<Vec<(TextBox, Vec<Word>)>> = Vec::new();
    for r in read.drain(..) {
        let mid = (r.0.rect[1] + r.0.rect[3]) / 2;
        match rows.last_mut() {
            Some(row) if row.first().is_some_and(|f| mid.abs_diff((f.0.rect[1] + f.0.rect[3]) / 2) <= f.0.height() / 2) => row.push(r),
            _ => rows.push(vec![r]),
        }
    }
    for mut row in rows {
        row.sort_by_key(|r| r.0.rect[0]);
        read.extend(row);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const YML: &str = "Global:\n  model_name: test\nPostProcess:\n  name: CTCLabelDecode\n  character_dict:\n  - 日\n  - 本\n  - A\n  - B\n  - ''''\n  - '\"'\nother: 1\n";

    /// A network that answers with a fixed output, or one computed from the input shape.
    /// An output (shape, values) made from the input shape.
    type Answer = Box<dyn Fn([usize; 4]) -> (Vec<usize>, Vec<f32>) + Send + Sync>;

    struct Fixed(Answer);

    impl Network for Fixed {
        fn run(&self, shape: [usize; 4], _data: Vec<f32>) -> Result<(Vec<usize>, Vec<f32>), OcrError> {
            Ok((self.0)(shape))
        }
    }

    fn fixed(f: impl Fn([usize; 4]) -> (Vec<usize>, Vec<f32>) + Send + Sync + 'static) -> Box<dyn Network> {
        Box::new(Fixed(Box::new(f)))
    }

    fn white(w: u32, h: u32) -> OcrImage {
        OcrImage::new(w, h, vec![255; (w * h * 4) as usize]).unwrap()
    }

    /// A detector whose map marks the rectangles `rects` (in map pixels, end-exclusive) at 0.9.
    fn det_marking(rects: Vec<[usize; 4]>) -> Box<dyn Network> {
        fixed(move |[_, _, h, w]| {
            let mut map = vec![0.0f32; w * h];
            for r in &rects {
                for y in r[1]..r[3].min(h) {
                    for x in r[0]..r[2].min(w) {
                        map[y * w + x] = 0.9;
                    }
                }
            }
            (vec![1, 1, h, w], map)
        })
    }

    /// A recogniser that reads `classes_per_line` for each line of the batch: one class per
    /// time step (0 = blank), as one-hot rows over `dict + 2` classes.
    fn rec_reading(classes: Vec<usize>, total: usize) -> Box<dyn Network> {
        fixed(move |[n, _, _, _]| {
            let steps = classes.len();
            let mut data = vec![0.0f32; n * steps * total];
            for b in 0..n {
                for (t, &c) in classes.iter().enumerate() {
                    data[(b * steps + t) * total + c] = 0.95;
                }
            }
            (vec![n, steps, total], data)
        })
    }

    #[test]
    fn the_dictionary_is_the_character_dict_list_with_quotes_undone() {
        let d = read_dictionary(YML).unwrap();
        assert_eq!(d, ["日", "本", "A", "B", "'", "\""]);
        assert!(read_dictionary("Global:\n  x: 1\n").is_err(), "no list is an error, not an empty reader");
        let huge = format!("character_dict:\n{}", "  - x\n".repeat(MAX_DICTIONARY + 1));
        assert!(read_dictionary(&huge).is_err(), "the entry count is capped");
    }

    #[test]
    fn db_boxes_are_the_regions_grown_and_scaled_to_the_page() {
        // A 4 × 20 region in a 40 × 20 map, on a page twice the size.
        let (pw, ph) = (40, 20);
        let mut map = vec![0.0f32; pw * ph];
        for y in 8..12 {
            for x in 10..30 {
                map[y * pw + x] = 0.8;
            }
        }
        let boxes = boxes_from_map(&map, pw, ph, 80, 40);
        assert_eq!(boxes.len(), 1);
        // grow = 20·4·1.5 / (2·24) = 2.5 map px on each side.
        assert_eq!(boxes[0].rect, [15, 11, 65, 29]);
        assert!((boxes[0].score - 0.8).abs() < 1e-6);
        // Faint regions and specks are not boxes.
        let faint: Vec<f32> = map.iter().map(|&p| if p > 0.0 { 0.5 } else { 0.0 }).collect();
        assert!(boxes_from_map(&faint, pw, ph, 80, 40).is_empty());
        let mut speck = vec![0.0f32; pw * ph];
        speck[5 * pw + 5] = 1.0;
        assert!(boxes_from_map(&speck, pw, ph, 80, 40).is_empty());
    }

    /// A map whose stated size disagrees with its length, NaNs, and a map of countless regions
    /// are read without panicking, and the box count is capped.
    #[test]
    fn db_post_processing_survives_hostile_maps() {
        assert!(boxes_from_map(&[0.9; 10], 100, 100, 10, 10).is_empty(), "a map shorter than its size reads only what is there");
        assert!(boxes_from_map(&[f32::NAN; 400], 20, 20, 20, 20).is_empty());
        assert!(boxes_from_map(&[], 0, 0, 0, 0).is_empty());
        assert!(boxes_from_map(&[0.9; 16], usize::MAX, 2, 10, 10).is_empty(), "an overflowing size is empty, not a panic");
        // 4 × 4 blobs on a grid with gaps: (1000/5)² = 40,000 regions.
        let side = 1000;
        let mut map = vec![0.0f32; side * side];
        for y in 0..side {
            for x in 0..side {
                if x % 5 < 4 && y % 5 < 4 {
                    map[y * side + x] = 0.9;
                }
            }
        }
        assert_eq!(boxes_from_map(&map, side, side, side, side).len(), MAX_BOXES);
    }

    #[test]
    fn ctc_collapses_repeats_drops_blanks_and_places_characters() {
        let dict: Vec<String> = ["日", "本", "A"].iter().map(|s| s.to_string()).collect();
        // 5 classes: blank, 日, 本, A, space. Steps: 日 日 _ 本 _ A A space.
        let seq = [1usize, 1, 0, 2, 0, 3, 3, 4];
        let classes = 5;
        let mut data = vec![0.0f32; seq.len() * classes];
        for (t, &c) in seq.iter().enumerate() {
            data[t * classes + c] = 0.9;
        }
        let best = best_of(&[seq.len(), classes], &data).unwrap();
        let read = decode(&dict, &best, classes, 8.0, 64.0).unwrap();
        let text: String = read.iter().map(|c| c.text.as_str()).collect();
        assert_eq!(text, "日本A ");
        assert_eq!((read[0].start, read[0].end), (0.0, 24.0), "日 runs to where 本 starts");
        assert_eq!((read[2].start, read[2].end), (40.0, 56.0));
        // A class count the dictionary does not explain is an error.
        assert!(decode(&dict, &best, 7, 8.0, 64.0).is_err());
        // An output shorter than its shape, or a class beyond the model's, is an error, not a panic.
        assert!(best_of(&[seq.len(), classes], &data[..10]).is_err());
        assert!(best_of(&[seq.len(), 0], &[]).is_err());
        assert!(decode(&dict, &[(9, 0.5)], classes, 8.0, 64.0).is_err());
        // NaN probabilities never panic.
        let nan = best_of(&[8, 5], &[f32::NAN; 40]).unwrap();
        assert!(decode(&dict, &nan, 5, 8.0, 64.0).is_ok());
    }

    #[test]
    fn a_horizontal_line_becomes_one_word_per_kanji_and_one_per_latin_run() {
        // One text row across the middle of a 320 × 64 page; the detector's map is the page
        // scaled to 1600 / 320 = 5× → 1600 × 320.
        let det = det_marking(vec![[100, 120, 1500, 200]]);
        // Classes: blank, 日 本 A B ' ", space = 8. Reads 日 本 A B.
        let rec = rec_reading(vec![1, 0, 2, 0, 0, 3, 4, 0, 0, 0], 8);
        let ocr = PpOcr::with_networks(det, rec, YML, PPOCR_ID).unwrap();
        let lines = ocr.read(&white(320, 64), None).unwrap();
        assert_eq!(lines.len(), 1);
        let texts: Vec<&str> = lines[0].words.iter().map(|w| w.text.as_str()).collect();
        assert_eq!(texts, ["日", "本", "AB"]);
        assert_eq!(lines[0].text(), "日本 AB");
        let r: Vec<[f32; 4]> = lines[0].words.iter().map(|w| w.rect).collect();
        assert!(r[0][2] <= r[1][0] && r[1][2] <= r[2][0], "words run left to right without overlapping: {r:?}");
        assert!(lines[0].words.iter().all(|w| w.source == PPOCR_ID && w.confidence.is_some_and(|c| c > 90.0)));
    }

    #[test]
    fn a_vertical_page_reads_columns_right_to_left_with_characters_down_the_column() {
        // Two tall columns on a 200 × 400 page (map 4×: 800 × 1600).
        let det = det_marking(vec![[100, 100, 180, 1500], [500, 100, 580, 1500]]);
        let rec = rec_reading(vec![1, 0, 0, 2, 0, 0], 8);
        let ocr = PpOcr::with_networks(det, rec, YML, PPOCR_ID).unwrap();
        let lines = ocr.read(&white(200, 400), None).unwrap();
        assert_eq!(lines.len(), 2);
        assert!(lines[0].words[0].rect[0] > lines[1].words[0].rect[0], "the right column comes first");
        let col = &lines[0].words;
        assert_eq!(col.iter().map(|w| w.text.as_str()).collect::<Vec<_>>(), ["日", "本"]);
        assert!(col[0].rect[3] <= col[1].rect[1], "characters run down the column: {:?}", col.iter().map(|w| w.rect).collect::<Vec<_>>());
        // ja-vert forces the direction even when the boxes do not decide it.
        let det = det_marking(vec![[100, 100, 180, 1500]]);
        let rec = rec_reading(vec![1, 0, 0, 2, 0, 0], 8);
        let ocr = PpOcr::with_networks(det, rec, YML, PPOCR_ID).unwrap();
        let lines = ocr.recognize(&white(200, 400), &RecognizeOptions { language: "ja-vert".into() }).unwrap();
        assert!(lines[0].words[0].rect[3] <= lines[0].words[1].rect[1]);
    }

    /// Outputs whose shape disagrees with their length, or with the batch, are errors.
    #[test]
    fn network_outputs_are_not_trusted() {
        let good_rec = || rec_reading(vec![1, 0], 8);
        let page = white(64, 64);
        let cases: Vec<Box<dyn Network>> = vec![
            fixed(|_| (vec![1, 1, 10, 10], vec![0.9; 99])),
            fixed(|_| (vec![1, 2, 10, 10], vec![0.9; 200])),
            fixed(|_| (vec![], vec![])),
            fixed(|_| (vec![1, 1, usize::MAX, 2], vec![0.9; 4])),
        ];
        for det in cases {
            let ocr = PpOcr::with_networks(det, good_rec(), YML, PPOCR_ID).unwrap();
            assert!(matches!(ocr.read(&page, None), Err(OcrError::Recognize(_))));
        }
        let det = || det_marking(vec![[100, 600, 1500, 1000]]);
        let recs: Vec<Box<dyn Network>> = vec![
            fixed(|[n, ..]| (vec![n, 4, 8], vec![0.5; n * 4 * 8 - 1])),
            fixed(|[n, ..]| (vec![n + 1, 4, 8], vec![0.5; (n + 1) * 4 * 8])),
            fixed(|[n, ..]| (vec![n, 4, 9], vec![0.5; n * 4 * 9])),
            fixed(|[n, ..]| (vec![n, usize::MAX, 8], vec![0.5; n])),
        ];
        for rec in recs {
            let ocr = PpOcr::with_networks(det(), rec, YML, PPOCR_ID).unwrap();
            assert!(matches!(ocr.read(&page, None), Err(OcrError::Recognize(_))));
        }
    }

    /// A page with nothing on it reads as no lines; the largest page the image caps allow is
    /// read at the capped detection size, not at its own.
    #[test]
    fn blank_and_huge_pages() {
        let det = fixed(|[_, _, h, w]| {
            assert!(h <= MAX_DET_SIDE && w <= MAX_DET_SIDE, "the detection input is capped: {w} × {h}");
            (vec![1, 1, h, w], vec![0.0; h * w])
        });
        let ocr = PpOcr::with_networks(det, rec_reading(vec![0], 8), YML, PPOCR_ID).unwrap().with_det_side(usize::MAX);
        assert_eq!(ocr.det_side(), MAX_DET_SIDE);
        assert!(ocr.read(&white(1, 1), None).unwrap().is_empty());
        assert!(ocr.read(&white(crate::MAX_SIDE, 640), None).unwrap().is_empty());
        assert!(ocr.read(&white(crate::MAX_SIDE, 1), None).unwrap().is_empty(), "a sliver page still has a 32-pixel detection input");
    }

    /// Files that are not the attributed ones are refused before rten parses them (rten can
    /// spin forever on bytes that are not a model).
    #[test]
    fn files_other_than_the_attributed_ones_are_refused() {
        let yml = std::fs::read(Path::new(env!("CARGO_MANIFEST_DIR")).join("../../assets/models").join(PPOCR_DICTIONARY)).unwrap_or_default();
        for (det, rec, dict) in
            [(vec![0xff; 64], vec![0xff; 64], yml.clone()), (vec![], vec![], vec![]), (vec![0; 8], vec![0; 8], b"character_dict:\n  - a\n".to_vec())]
        {
            match PpOcr::from_bytes(det, rec, dict) {
                Err(OcrError::Load(name, why)) => assert!(why.contains("SHA-256") && name.starts_with("PP-OCRv5"), "{name}: {why}"),
                Err(e) => panic!("unexpected error {e}"),
                Ok(_) => panic!("loaded files that are not the models"),
            }
        }
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(PpOcrModels::in_dir(dir.path()), None);
        for f in [PPOCR_DETECTION_MODEL, PPOCR_RECOGNITION_MODEL] {
            std::fs::write(dir.path().join(f), b"x").unwrap();
        }
        assert_eq!(PpOcrModels::in_dir(dir.path()), None, "all three files are needed");
        std::fs::write(dir.path().join(PPOCR_DICTIONARY), b"x").unwrap();
        let models = PpOcrModels::in_dir(dir.path()).unwrap();
        assert!(matches!(PpOcr::load(&models), Err(OcrError::Load(..))));
    }

    /// The real models, when `cargo xtask models --optional` has fetched them: a synthetic line
    /// of Japanese is read. Skips (passes) without them.
    #[test]
    fn the_real_models_read_when_installed() {
        let Some(models) = PpOcrModels::find() else {
            eprintln!("skipped: the PP-OCRv5 models are not installed (cargo xtask models --optional)");
            return;
        };
        let ocr = PpOcr::load(&models).unwrap();
        // A blank page reads as nothing, through the real networks.
        assert!(ocr.read(&white(640, 480), None).unwrap().is_empty());
    }

    /// A GPU network: `inner` answers, and from its `fail_from`-th run on it returns an error.
    struct Flaky {
        inner: Box<dyn Network>,
        runs: std::sync::atomic::AtomicUsize,
        fail_from: usize,
    }

    impl Network for Flaky {
        fn run(&self, shape: [usize; 4], data: Vec<f32>) -> Result<(Vec<usize>, Vec<f32>), OcrError> {
            let n = self.runs.fetch_add(1, Ordering::Relaxed);
            if n >= self.fail_from {
                return Err(OcrError::Recognize("device lost".into()));
            }
            self.inner.run(shape, data)
        }
    }

    fn flaky(inner: Box<dyn Network>, fail_from: usize) -> Box<dyn Network> {
        Box::new(Flaky { inner, runs: std::sync::atomic::AtomicUsize::new(0), fail_from })
    }

    /// The CPU pair the GPU tests read with: one row of 日本 on a 320 x 64 page.
    fn cpu_reader() -> PpOcr {
        PpOcr::with_networks(det_marking(vec![[100, 120, 1500, 200]]), rec_reading(vec![1, 0, 2, 0], 8), YML, PPOCR_ID).unwrap()
    }

    fn text(lines: &[Line]) -> String {
        lines.iter().map(Line::text).collect::<Vec<_>>().join("/")
    }

    /// A GPU that agrees with the CPU passes the self-check and reads, with no notes.
    #[test]
    fn a_gpu_that_agrees_with_the_cpu_reads() {
        let ocr = cpu_reader().with_gpu(det_marking(vec![[100, 120, 1500, 200]]), rec_reading(vec![1, 0, 2, 0], 8), "test GPU".into());
        assert_eq!(ocr.backend(), "test GPU");
        let (lines, notes) = ocr.read_noted(&white(320, 64), None).unwrap();
        assert_eq!(text(&lines), "日本");
        assert!(notes.is_empty(), "{notes:?}");
    }

    /// A GPU whose results differ from the CPU's (a driver bug) is never used: the reader stays
    /// on the CPU and every page says why.
    #[test]
    fn a_gpu_that_disagrees_with_the_cpu_is_refused_and_every_page_says_so() {
        let wrong = fixed(|[n, _, _, _]| (vec![n, 4, 8], vec![0.5; n * 4 * 8]));
        let ocr = cpu_reader().with_gpu(det_marking(vec![[100, 120, 1500, 200]]), wrong, "test GPU".into());
        assert_eq!(ocr.backend(), "rten");
        for _ in 0..2 {
            let (lines, notes) = ocr.read_noted(&white(320, 64), None).unwrap();
            assert_eq!(text(&lines), "日本", "the CPU reads");
            assert!(notes.len() == 1 && notes[0].contains("self-check") && notes[0].contains("test GPU"), "{notes:?}");
        }
        // NaN and a shape that does not hold its values are refused the same way.
        for bad in [fixed(|[_, _, h, w]| (vec![1, 1, h, w], vec![f32::NAN; h * w])), fixed(|[_, _, h, w]| (vec![1, 1, h, w], vec![0.0; 3]))] {
            let ocr = cpu_reader().with_gpu(bad, rec_reading(vec![1, 0, 2, 0], 8), "test GPU".into());
            assert_eq!(ocr.backend(), "rten");
        }
    }

    /// A GPU that fails mid-session (an error, or an output that is not sane) gives way to the
    /// CPU for that page and the rest of the session; the page is still read, and every page
    /// from then on notes it.
    #[test]
    fn a_gpu_that_fails_falls_back_to_the_cpu_for_the_rest_of_the_session() {
        // Two self-check runs, then one good page (det + rec), then failure.
        let ocr =
            cpu_reader().with_gpu(flaky(det_marking(vec![[100, 120, 1500, 200]]), 2), flaky(rec_reading(vec![1, 0, 2, 0], 8), 2), "test GPU".into());
        let (lines, notes) = ocr.read_noted(&white(320, 64), None).unwrap();
        assert_eq!((text(&lines).as_str(), notes.len()), ("日本", 0));
        let (lines, notes) = ocr.read_noted(&white(320, 64), None).unwrap();
        assert_eq!(text(&lines), "日本", "the failing page is read on the CPU");
        assert!(notes.len() == 1 && notes[0].contains("device lost"), "{notes:?}");
        assert_eq!(ocr.backend(), "rten");
        let (_, notes) = ocr.read_noted(&white(320, 64), None).unwrap();
        assert!(notes.len() == 1 && notes[0].contains("device lost"), "later pages note it too: {notes:?}");
        // The session notice is handed out once (another test may have taken it first).
        let _ = take_gpu_fallback_notice();
        assert_eq!(take_gpu_fallback_notice(), None);
        // A GPU that starts answering NaN after the check is given up the same way.
        let nan_after = |inner: Box<dyn Network>| -> Box<dyn Network> {
            let runs = std::sync::atomic::AtomicUsize::new(0);
            fixed(move |shape| {
                let (s, v) = inner.run(shape, vec![0.0; shape.iter().product()]).unwrap();
                if runs.fetch_add(1, Ordering::Relaxed) >= 1 { (s, vec![f32::NAN; v.len()]) } else { (s, v) }
            })
        };
        let ocr = cpu_reader().with_gpu(nan_after(det_marking(vec![[100, 120, 1500, 200]])), rec_reading(vec![1, 0, 2, 0], 8), "test GPU".into());
        let (lines, notes) = ocr.read_noted(&white(320, 64), None).unwrap();
        assert_eq!(text(&lines), "日本");
        assert!(notes.iter().any(|n| n.contains("not finite")), "{notes:?}");
    }

    /// A factory that cannot make the networks leaves the reader on the CPU, saying why.
    #[test]
    fn a_factory_that_cannot_run_the_models_leaves_the_reader_on_the_cpu() {
        struct NoKernels;
        impl NetworkFactory for NoKernels {
            fn name(&self) -> String {
                "no kernels".into()
            }
            fn networks(&self, _: &[u8], _: &[u8]) -> Result<Networks, OcrError> {
                Err(OcrError::Load("wgpu".into(), "the kernels are not built yet".into()))
            }
        }
        let Some(models) = PpOcrModels::find() else {
            eprintln!("skipped: the PP-OCRv5 models are not installed (cargo xtask models --optional)");
            return;
        };
        let ocr = PpOcr::load_with_gpu(&models, &NoKernels).unwrap();
        assert_eq!(ocr.backend(), "rten");
        let (lines, notes) = ocr.read_noted(&white(320, 64), None).unwrap();
        assert!(lines.is_empty());
        assert!(notes.len() == 1 && notes[0].contains("no kernels") && notes[0].contains("not built yet"), "{notes:?}");
    }
}
