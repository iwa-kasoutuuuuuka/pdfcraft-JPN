//! A minimal PP-OCRv5 reader on rten, for the benchmark only: it answers whether rten loads
//! and runs the PP-OCRv5 server models (PaddlePaddle, Apache-2.0; fetched by
//! `cargo xtask models`, never committed) and how well and how fast they read.
//!
//! Written from the public description of the pipeline (DB text detection, then a CTC line
//! recogniser over the model's own character list): detection on the page scaled so its long
//! side is `det_side`, normalised with the ImageNet mean and deviation; the probability map is
//! thresholded at 0.3 and each connected region is a box, grown by its area × 1.5 / perimeter.
//! Each box is cut from the page, turned a quarter left when it is tall (a vertical line),
//! scaled to 48 px high and read in batches of equal padded width; greedy CTC decoding (blank = 0) maps class `i` to entry
//! `i - 1` of the dictionary, with a space as the last class. No angle classifier, no polygon
//! boxes: enough to measure, not the shipped engine.
//!
//! The networks run through [`Net`]: rten on the CPU (always), or ONNX Runtime with its CUDA
//! provider (`--features ort-cuda`, local benchmarking only; see `ort_net`). Pre- and
//! post-processing are the same code for both, so the scores compare the runtimes alone.

use std::path::{Path, PathBuf};
use std::time::Instant;

use pdfcraft_ocr::{Line, OcrError, OcrImage, RecognizeOptions, Recognizer, Word};
use rten::Model;
use rten_tensor::prelude::*;
use rten_tensor::{NdTensor, Tensor};

/// One network: an NCHW f32 input in, the first output out as (shape, values).
pub trait Net: Send + Sync {
    fn infer(&self, shape: [usize; 4], data: Vec<f32>) -> Result<(Vec<usize>, Vec<f32>), OcrError>;
}

/// rten on the CPU.
struct RtenNet(Model);

impl Net for RtenNet {
    fn infer(&self, shape: [usize; 4], data: Vec<f32>) -> Result<(Vec<usize>, Vec<f32>), OcrError> {
        let input = NdTensor::<f32, 4>::try_from_data(shape, data).map_err(err)?;
        let out: Tensor<f32> = self.0.run_one(input.into(), None).map_err(err)?.try_into().map_err(err)?;
        Ok((out.shape().to_vec(), out.to_vec()))
    }
}

/// ONNX Runtime with the CUDA provider, loaded at run time from `ORT_DYLIB_PATH`. Registration
/// failure is an error (never a silent fall back to ONNX Runtime's CPU provider, which would
/// make the GPU column of the benchmark a lie).
#[cfg(feature = "ort-cuda")]
mod ort_net {
    use std::path::Path;
    use std::sync::{Mutex, OnceLock};

    use ort::environment::Environment;
    use ort::ep;
    use ort::session::Session;
    use ort::value::TensorRef;
    use pdfcraft_ocr::OcrError;

    use super::{Net, err};

    static ENV: OnceLock<Result<Environment, String>> = OnceLock::new();

    fn environment() -> Result<&'static Environment, String> {
        ENV.get_or_init(|| {
            let lib = std::env::var_os("ORT_DYLIB_PATH").ok_or("set ORT_DYLIB_PATH to the onnxruntime.dll of an ONNX Runtime GPU build")?;
            ort::init_from(&lib)
                .map_err(|e| format!("loading ONNX Runtime from {}: {e}", Path::new(&lib).display()))?
                .build()
                .map_err(|e| format!("ONNX Runtime environment: {e}"))
        })
        .as_ref()
        .map_err(Clone::clone)
    }

    pub struct OrtNet(Mutex<Session>);

    impl OrtNet {
        pub fn load(path: &Path) -> Result<OrtNet, String> {
            let env = environment()?;
            let what = |e: ort::Error| format!("{} on CUDA: {e}", path.display());
            let builder = Session::builder(env).map_err(what)?;
            let mut builder = builder.with_execution_providers([ep::CUDA::default().build().error_on_failure()]).map_err(|e| what(e.into()))?;
            let session = builder.commit_from_file(path).map_err(what)?;
            Ok(OrtNet(Mutex::new(session)))
        }
    }

    impl Net for OrtNet {
        fn infer(&self, shape: [usize; 4], data: Vec<f32>) -> Result<(Vec<usize>, Vec<f32>), OcrError> {
            let mut session = self.0.lock().unwrap_or_else(|e| e.into_inner());
            let input = TensorRef::from_array_view((shape, data.as_slice())).map_err(err)?;
            let outputs = session.run(ort::inputs![input]).map_err(err)?;
            let first = outputs.values().next().ok_or_else(|| err("the model has no output"))?;
            let (shape, values) = first.try_extract_tensor::<f32>().map_err(err)?;
            let dims = shape
                .iter()
                .map(|&d| usize::try_from(d).map_err(|_| err(format!("the model output has a negative dimension {d}"))))
                .collect::<Result<Vec<_>, _>>()?;
            Ok((dims, values.to_vec()))
        }
    }
}

/// Which runtime runs the networks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Runtime {
    Rten,
    #[cfg(feature = "ort-cuda")]
    OrtCuda,
}

pub struct PpOcr {
    det: Box<dyn Net>,
    rec: Box<dyn Net>,
    dict: Vec<String>,
    det_side: usize,
    id: &'static str,
    pub load_seconds: f64,
}

/// The files the probe needs, as `cargo xtask models` names them.
pub const DET: &str = "PP-OCRv5_server_det.onnx";
pub const REC: &str = "PP-OCRv5_server_rec.onnx";
pub const REC_CONFIG: &str = "PP-OCRv5_server_rec.yml";

fn err(e: impl std::fmt::Display) -> OcrError {
    OcrError::Recognize(e.to_string())
}

/// The dictionary in the recogniser's `inference.yml`: the YAML list under `character_dict:`.
pub fn read_dict(yml: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut inside = false;
    for line in yml.lines() {
        if line.trim_start().starts_with("character_dict:") {
            inside = true;
            continue;
        }
        if !inside {
            continue;
        }
        let Some(item) = line.strip_prefix("  - ") else { break };
        let item = if item.len() >= 2 && item.starts_with('\'') && item.ends_with('\'') {
            item.get(1..item.len() - 1).unwrap_or("").replace("''", "'")
        } else if item.len() >= 2 && item.starts_with('"') && item.ends_with('"') {
            item.get(1..item.len() - 1).unwrap_or("").replace("\\\"", "\"").replace("\\\\", "\\")
        } else {
            item.to_string()
        };
        out.push(item);
    }
    out
}

impl PpOcr {
    pub fn load(dir: &Path, det_side: usize, runtime: Runtime) -> Result<PpOcr, String> {
        let start = Instant::now();
        let (det, rec, id): (Box<dyn Net>, Box<dyn Net>, &'static str) = match runtime {
            Runtime::Rten => (
                Box::new(RtenNet(Model::load_file(dir.join(DET)).map_err(|e| format!("loading {DET} with rten: {e}"))?)),
                Box::new(RtenNet(Model::load_file(dir.join(REC)).map_err(|e| format!("loading {REC} with rten: {e}"))?)),
                "ppocrv5-rten",
            ),
            #[cfg(feature = "ort-cuda")]
            Runtime::OrtCuda => (Box::new(ort_net::OrtNet::load(&dir.join(DET))?), Box::new(ort_net::OrtNet::load(&dir.join(REC))?), "ppocrv5-cuda"),
        };
        let yml = std::fs::read_to_string(dir.join(REC_CONFIG)).map_err(|e| format!("{REC_CONFIG}: {e}"))?;
        let dict = read_dict(&yml);
        if dict.is_empty() {
            return Err(format!("{REC_CONFIG} has no character_dict"));
        }
        Ok(PpOcr { det, rec, dict, det_side, id, load_seconds: start.elapsed().as_secs_f64() })
    }

    pub fn find(dir: Option<&PathBuf>, det_side: usize, runtime: Runtime) -> Result<PpOcr, String> {
        let dir = dir.cloned().or_else(|| std::env::var_os("PDFCRAFT_MODELS").map(PathBuf::from)).unwrap_or_else(|| PathBuf::from("assets/models"));
        if !dir.join(DET).exists() {
            return Err(format!("no {DET} in {} (run `cargo xtask models --optional`)", dir.display()));
        }
        PpOcr::load(&dir, det_side, runtime)
    }

    /// Text boxes `[x0, y0, x1, y1]` in page pixels.
    fn detect(&self, grey: &[u8], w: usize, h: usize) -> Result<Vec<[usize; 4]>, OcrError> {
        if w == 0 || h == 0 {
            return Err(OcrError::EmptyImage);
        }
        let scale = self.det_side as f32 / w.max(h) as f32;
        let round32 = |v: f32| (((v / 32.0).round() as usize).max(1)) * 32;
        let (dw, dh) = (round32(w as f32 * scale), round32(h as f32 * scale));
        let (sx, sy) = (w as f32 / dw as f32, h as f32 / dh as f32);
        let (mean, std) = ([0.485f32, 0.456, 0.406], [0.229f32, 0.224, 0.225]);
        let plane = dh * dw;
        let mut input = vec![0.0f32; 3 * plane];
        for y in 0..dh {
            for x in 0..dw {
                let v = sample(grey, w, h, (x as f32 + 0.5) * sx - 0.5, (y as f32 + 0.5) * sy - 0.5) / 255.0;
                for c in 0..3 {
                    if let Some(slot) = input.get_mut(c * plane + y * dw + x) {
                        *slot = (v - mean[c]) / std[c];
                    }
                }
            }
        }
        let (shape, data) = self.det.infer([1, 3, dh, dw], input)?;
        let (ph, pw) = match shape.as_slice() {
            [1, 1, ph, pw] if *ph > 0 && *pw > 0 && ph.checked_mul(*pw) == Some(data.len()) => (*ph, *pw),
            other => return Err(err(format!("unexpected detection output shape {other:?}"))),
        };
        let mask: Vec<bool> = data.iter().map(|&p| p > 0.3).collect();
        let mut seen = vec![false; mask.len()];
        let mut boxes = Vec::new();
        let mut stack = Vec::new();
        for start in 0..mask.len() {
            if !mask.get(start).copied().unwrap_or(false) || seen.get(start).copied().unwrap_or(true) {
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
                score += data.get(i).copied().unwrap_or(0.0);
                let mut visit = |j: usize| {
                    if mask.get(j).copied().unwrap_or(false) && !seen.get(j).copied().unwrap_or(true) {
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
            if bw.min(bh) < 3.0 || score / (area as f32) < 0.6 {
                continue;
            }
            let grow = bw * bh * 1.5 / (2.0 * (bw + bh));
            let fx = w as f32 / pw as f32;
            let fy = h as f32 / ph as f32;
            let b = [
                ((x0 as f32 - grow) * fx).max(0.0) as usize,
                ((y0 as f32 - grow) * fy).max(0.0) as usize,
                (((x1 + 1) as f32 + grow) * fx).min(w as f32) as usize,
                (((y1 + 1) as f32 + grow) * fy).min(h as f32) as usize,
            ];
            if b[2] > b[0] + 2 && b[3] > b[1] + 2 {
                boxes.push(b);
            }
        }
        Ok(boxes)
    }

    /// Cut one box out as a recogniser line: 48 px high, `rw` wide, normalised to -1..1. On a
    /// vertical page, tall boxes are columns and are turned a quarter left first (the column's
    /// top becomes the line's left end).
    fn crop(grey: &[u8], w: usize, h: usize, b: [usize; 4], vertical_page: bool) -> (usize, Vec<f32>) {
        let (bw, bh) = (b[2] - b[0], b[3] - b[1]);
        let vertical = vertical_page && is_tall(&b);
        // The crop's size once turned: (cw, ch), and where crop pixel (u along the line, v
        // across it) comes from. Turned left: u runs down the column, v from its right edge.
        let (cw, ch) = if vertical { (bh, bw) } else { (bw, bh) };
        let from = |u: f32, v: f32| if vertical { (b[2] as f32 - 1.0 - v, b[1] as f32 + u) } else { (b[0] as f32 + u, b[1] as f32 + v) };
        let rw = ((cw as f32 * 48.0 / ch.max(1) as f32).round() as usize).clamp(16, REC_MAX_WIDTH);
        let mut line = vec![0.0f32; 48 * rw];
        let (kx, ky) = (cw as f32 / rw as f32, ch as f32 / 48.0);
        for y in 0..48 {
            for x in 0..rw {
                let (px, py) = from((x as f32 + 0.5) * kx - 0.5, (y as f32 + 0.5) * ky - 0.5);
                if let Some(slot) = line.get_mut(y * rw + x) {
                    *slot = (sample(grey, w, h, px, py) / 255.0 - 0.5) / 0.5;
                }
            }
        }
        (rw, line)
    }

    /// Read lines in batches: widths are rounded up to a multiple of [`REC_BUCKET`] (the right
    /// is padded with 0, mid grey after normalisation, as PaddleOCR pads) so a runtime that tunes
    /// per input shape (cuDNN) sees few shapes, and up to [`REC_BATCH`] lines of one width go in
    /// one run. Returns (text, mean confidence) per line, in the order given.
    fn read_lines(&self, lines: &[(usize, Vec<f32>)]) -> Result<Vec<(String, f32)>, OcrError> {
        let bucket = |rw: usize| rw.div_ceil(REC_BUCKET).max(1).saturating_mul(REC_BUCKET).min(REC_MAX_WIDTH.div_ceil(REC_BUCKET) * REC_BUCKET);
        let mut order: Vec<usize> = (0..lines.len()).collect();
        order.sort_by_key(|&i| lines.get(i).map_or(0, |l| bucket(l.0)));
        let mut out = vec![(String::new(), 0.0f32); lines.len()];
        let mut i = 0;
        while i < order.len() {
            let bw = order.get(i).and_then(|&k| lines.get(k)).map_or(REC_BUCKET, |l| bucket(l.0));
            let mut group = Vec::new();
            while let Some(&k) = order.get(i) {
                if group.len() == REC_BATCH || lines.get(k).map_or(0, |l| bucket(l.0)) != bw {
                    break;
                }
                group.push(k);
                i += 1;
            }
            let n = group.len();
            let plane = 48 * bw;
            let mut input = vec![0.0f32; n * 3 * plane];
            for (slot, &k) in group.iter().enumerate() {
                let Some((rw, line)) = lines.get(k) else { continue };
                for y in 0..48 {
                    let Some(src) = line.get(y * rw..(y + 1) * rw) else { continue };
                    for c in 0..3 {
                        let at = (slot * 3 + c) * plane + y * bw;
                        if let Some(dst) = input.get_mut(at..at + rw) {
                            dst.copy_from_slice(src);
                        }
                    }
                }
            }
            let (shape, data) = self.rec.infer([n, 3, 48, bw], input)?;
            let (steps, classes) = match shape.as_slice() {
                [b, t, c] if *b == n && *c > 0 && b.checked_mul(*t).and_then(|bt| bt.checked_mul(*c)) == Some(data.len()) => (*t, *c),
                other => return Err(err(format!("unexpected recognition output shape {other:?}"))),
            };
            for (slot, &k) in group.iter().enumerate() {
                let seq = data
                    .get(slot * steps * classes..(slot + 1) * steps * classes)
                    .ok_or_else(|| err("the recognition output is shorter than its shape"))?;
                if let Some(o) = out.get_mut(k) {
                    *o = self.decode(seq, steps, classes)?;
                }
            }
        }
        Ok(out)
    }

    /// Greedy CTC over one line's `steps × classes` probabilities.
    fn decode(&self, data: &[f32], steps: usize, classes: usize) -> Result<(String, f32), OcrError> {
        // Classes: blank, the dictionary, then (when the model has one more) a space.
        let space = self.dict.len() + 1;
        if classes != space && classes != space + 1 {
            return Err(err(format!("the recogniser has {classes} classes; its dictionary implies {} or {}", space, space + 1)));
        }
        let (mut text, mut last, mut conf, mut n) = (String::new(), 0usize, 0.0f32, 0usize);
        for t in 0..steps {
            let row = data.get(t * classes..(t + 1) * classes).unwrap_or(&[]);
            let (best, p) = row.iter().enumerate().fold((0usize, f32::MIN), |a, (i, &p)| if p > a.1 { (i, p) } else { a });
            if best != 0 && best != last {
                let ch = if best == space { " " } else { self.dict.get(best - 1).map_or("", String::as_str) };
                text.push_str(ch);
                conf += p;
                n += 1;
            }
            last = best;
        }
        Ok((text, if n > 0 { conf / n as f32 } else { 0.0 }))
    }
}

/// Recogniser widths are rounded up to a multiple of this (see [`PpOcr::read_lines`]).
const REC_BUCKET: usize = 160;
/// Lines read in one recogniser run, at most.
const REC_BATCH: usize = 8;
/// The widest recogniser input, in pixels (PaddleOCR's own limit).
const REC_MAX_WIDTH: usize = 3200;

/// A box at least 1.5 times as tall as it is wide: a column of vertical text.
fn is_tall(b: &[usize; 4]) -> bool {
    b[3].saturating_sub(b[1]) as f32 >= b[2].saturating_sub(b[0]) as f32 * 1.5
}

/// Bilinear grey sample, white outside the page.
fn sample(grey: &[u8], w: usize, h: usize, x: f32, y: f32) -> f32 {
    let px = |xx: i64, yy: i64| -> f32 {
        if xx < 0 || yy < 0 || xx as usize >= w || yy as usize >= h {
            255.0
        } else {
            grey.get(yy as usize * w + xx as usize).copied().map_or(255.0, f32::from)
        }
    };
    let (x0, y0) = (x.floor(), y.floor());
    let (fx, fy) = (x - x0, y - y0);
    let (xi, yi) = (x0 as i64, y0 as i64);
    px(xi, yi) * (1.0 - fx) * (1.0 - fy) + px(xi + 1, yi) * fx * (1.0 - fy) + px(xi, yi + 1) * (1.0 - fx) * fy + px(xi + 1, yi + 1) * fx * fy
}

impl Recognizer for PpOcr {
    fn id(&self) -> &'static str {
        self.id
    }
    fn available(&self) -> bool {
        true
    }
    fn languages(&self) -> Vec<(String, String)> {
        vec![("ja".into(), "Japanese".into())]
    }
    fn recognize(&self, image: &OcrImage, _options: &RecognizeOptions) -> Result<Vec<Line>, OcrError> {
        let (w, h) = (image.width() as usize, image.height() as usize);
        let grey: Vec<u8> = (0..h).flat_map(|y| (0..w).map(move |x| (x, y))).map(|(x, y)| image.gray(x as i64, y as i64).unwrap_or(255)).collect();
        let boxes = self.detect(&grey, w, h)?;
        let vertical_page = boxes.iter().filter(|b| is_tall(b)).count() * 2 > boxes.len();
        let lines: Vec<(usize, Vec<f32>)> = boxes.iter().map(|&b| PpOcr::crop(&grey, w, h, b, vertical_page)).collect();
        let texts = self.read_lines(&lines)?;
        let mut read = Vec::new();
        for (b, (text, conf)) in boxes.into_iter().zip(texts) {
            if !text.trim().is_empty() {
                read.push((b, text, conf));
            }
        }
        // Reading order: columns right to left on a vertical page, else rows top to bottom
        // (boxes whose middles are within half a box height share a row, left to right).
        if vertical_page {
            read.sort_by(|a, b| b.0[2].cmp(&a.0[2]).then(a.0[1].cmp(&b.0[1])));
        } else {
            // Group into rows first (a total order per step; a tolerance comparator is not one).
            read.sort_by_key(|r| r.0[1] + r.0[3]);
            let mut rows: Vec<Vec<([usize; 4], String, f32)>> = Vec::new();
            for r in read.drain(..) {
                let mid = (r.0[1] + r.0[3]) / 2;
                match rows.last_mut() {
                    Some(row) if row.first().is_some_and(|f| mid.abs_diff((f.0[1] + f.0[3]) / 2) <= (f.0[3] - f.0[1]) / 2) => row.push(r),
                    _ => rows.push(vec![r]),
                }
            }
            for mut row in rows {
                row.sort_by_key(|r| r.0[0]);
                read.extend(row);
            }
        }
        Ok(read
            .into_iter()
            .map(|(b, text, conf)| Line {
                words: vec![Word::new(text, [b[0] as f32, b[1] as f32, b[2] as f32, b[3] as f32], self.id).with_confidence(conf * 100.0)],
            })
            .collect())
    }
}
