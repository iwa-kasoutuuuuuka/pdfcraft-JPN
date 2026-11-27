//! The OCR benchmark (`crates/ocr/examples/ocr_bench/`) with one more engine, `ppocrv5-wgpu`:
//! PP-OCRv5 with its networks on this machine's GPU through wgpu (this crate), the library's
//! pre- and post-processing, and the engine's own self-check and CPU fallback. Pages whose
//! reading fell back to the CPU say so, so the GPU column is never the CPU's.
//!
//! `cargo run --release -p pdfcraft-ocr-gpu --example ocr_bench_gpu -- --engines ppocrv5,ppocrv5-wgpu`
//! (with `CRAFT_FONTS_DIR` and the models of `cargo xtask models --optional`). Nothing here is
//! shipped; it is a measuring tool.

#[allow(unexpected_cfgs, dead_code)]
#[path = "../../ocr/examples/ocr_bench/main.rs"]
mod bench;

use std::sync::Arc;
use std::task::{Context, Poll, Waker};
use std::time::Instant;

use pdfcraft_ocr::ppocr::{PpOcr, PpOcrModels};
use pdfcraft_ocr::{Line, OcrError, OcrImage, RecognizeOptions, Recognizer};
use pdfcraft_ocr_gpu::{GpuContext, WgpuFactory};

fn ready<F: std::future::Future>(fut: F) -> Option<F::Output> {
    let mut fut = std::pin::pin!(fut);
    let mut cx = Context::from_waker(Waker::noop());
    for _ in 0..10_000_000 {
        if let Poll::Ready(v) = fut.as_mut().poll(&mut cx) {
            return Some(v);
        }
        std::thread::yield_now();
    }
    None
}

/// The engine, printing every page's notes (a fallback to the CPU, and why).
struct Noted(PpOcr);

impl Recognizer for Noted {
    fn id(&self) -> &'static str {
        self.0.id()
    }
    fn available(&self) -> bool {
        self.0.available()
    }
    fn languages(&self) -> Vec<(String, String)> {
        self.0.languages()
    }
    fn recognize(&self, image: &OcrImage, options: &RecognizeOptions) -> Result<Vec<Line>, OcrError> {
        let (lines, notes) = self.0.recognize_noted(image, options)?;
        for n in notes {
            println!("ppocrv5-wgpu: note: {n}");
        }
        Ok(lines)
    }
}

fn gpu_engine(name: &str, args: &bench::Args) -> Option<Box<dyn Recognizer>> {
    if name != "ppocrv5-wgpu" {
        return None;
    }
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = ready(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        force_fallback_adapter: false,
        compatible_surface: None,
        apply_limit_buckets: false,
    }))?
    .ok()?;
    let info = adapter.get_info();
    let a = adapter.limits();
    let limits = wgpu::Limits {
        max_storage_buffer_binding_size: a.max_storage_buffer_binding_size.min(1 << 30),
        max_buffer_size: a.max_buffer_size.min(1 << 30),
        ..wgpu::Limits::default()
    };
    let (device, queue) = ready(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("ocr-bench"),
        required_features: wgpu::Features::empty(),
        required_limits: limits,
        experimental_features: wgpu::ExperimentalFeatures::disabled(),
        memory_hints: wgpu::MemoryHints::Performance,
        trace: wgpu::Trace::Off,
    }))?
    .ok()?;
    let context = match GpuContext::shared(device, queue, info) {
        Ok(c) => c,
        Err(e) => {
            println!("ocr-bench: ppocrv5-wgpu: {e}");
            return None;
        }
    };
    let factory = Arc::new(WgpuFactory::new(context));
    let models = match &args.models {
        Some(dir) => PpOcrModels::in_dir(dir),
        None => PpOcrModels::find(),
    }?;
    let start = Instant::now();
    let reader = match PpOcr::load_with_gpu(&models, factory.as_ref()) {
        Ok(r) => r.with_det_side(args.det_side),
        Err(e) => {
            println!("ocr-bench: ppocrv5-wgpu: {e}");
            return None;
        }
    };
    if reader.backend() == "rten" {
        println!("ocr-bench: ppocrv5-wgpu: the GPU is not used, so this engine would time the CPU; skipped");
        if let Some(why) = pdfcraft_ocr::ppocr::take_gpu_fallback_notice() {
            println!("ocr-bench: ppocrv5-wgpu: {why}");
        }
        return None;
    }
    println!(
        "ocr-bench: PP-OCRv5 server det/rec on {} in {:.1} s, self-check passed (detection side {})",
        reader.backend(),
        start.elapsed().as_secs_f64(),
        reader.det_side()
    );
    Some(Box::new(Noted(reader)))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    bench::run_with(&gpu_engine)
}
