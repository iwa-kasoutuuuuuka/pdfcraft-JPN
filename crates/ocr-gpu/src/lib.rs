//! pdfcraft-ocr-gpu — PP-OCRv5 on the GPU (L7).
//!
//! `pdfcraft_ocr::ppocr` (L4) reads pages with its networks behind the `Network` trait and
//! knows no GPU API; the layering keeps wgpu at L7 and above. This crate is the GPU side:
//!
//! - [`onnx`]: our own minimal reader of the ONNX files (protobuf wire format), with every
//!   length, count and size checked: the graph, its attributes and its weights, nothing run.
//! - [`GpuContext`]: the wgpu device and queue to compute on. The app shares the device it
//!   draws with (eframe's), by the owner's decision; software adapters are refused.
//! - [`WgpuFactory`]: the `NetworkFactory` the program (L8) registers with the engine
//!   (`pdfcraft_engine::ocr::register_gpu`). The engine asks it only when a job turns the GPU
//!   on (`OcrSettings::gpu`, off by default).
//!
//! The compute kernels are not here yet (P2 of docs/ocr-ppocr.ja.md): the factory reads both
//! models, names the operators it has no kernel for, and declines, so PP-OCRv5 reads on the
//! CPU and every page says why. Off the web only: the web build has no PP-OCRv5 models.

#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable)]

pub mod onnx;

#[cfg(not(target_arch = "wasm32"))]
mod context;
#[cfg(not(target_arch = "wasm32"))]
pub use context::{GpuContext, WgpuFactory};

/// The ONNX operators the GPU kernels cover. None yet (P2 adds them); the factory declines a
/// model that uses any other.
pub const SUPPORTED_OPS: &[&str] = &[];

/// The operators of `model` with no GPU kernel, sorted and once each.
pub fn unsupported_ops(model: &onnx::Model) -> Vec<String> {
    model.graph.op_counts().into_keys().filter(|op| !SUPPORTED_OPS.contains(&op.as_str())).collect()
}

/// Read both networks and say what the GPU cannot run: `Ok(())` when every operator has a
/// kernel. The error is one sentence for a page's notes.
pub fn plan(detection: &[u8], recognition: &[u8]) -> Result<(), String> {
    let mut missing = std::collections::BTreeSet::new();
    for (name, bytes) in [(pdfcraft_ocr::ppocr::PPOCR_DETECTION_MODEL, detection), (pdfcraft_ocr::ppocr::PPOCR_RECOGNITION_MODEL, recognition)] {
        let model = onnx::parse(bytes).map_err(|e| format!("{name}: {e}"))?;
        missing.extend(unsupported_ops(&model));
    }
    if missing.is_empty() {
        return Ok(());
    }
    Err(format!("the GPU kernels for {} are not built yet", missing.into_iter().collect::<Vec<_>>().join(", ")))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The real models, when installed: both parse, and with no kernels yet the plan names
    /// their operators instead of claiming the GPU can run them.
    #[test]
    fn the_real_models_parse_and_the_plan_declines_them() {
        let Some(models) = pdfcraft_ocr::ppocr::PpOcrModels::find() else {
            eprintln!("skipped: the PP-OCRv5 models are not installed (cargo xtask models --optional)");
            return;
        };
        let det = std::fs::read(&models.detection).unwrap();
        let rec = std::fs::read(&models.recognition).unwrap();
        let (d, r) = (onnx::parse(&det).unwrap(), onnx::parse(&rec).unwrap());
        // docs/ocr-ppocr.ja.md §4: opset 11 and 11 operator types; opset 10 and 23.
        assert_eq!((d.opset(), d.graph.op_counts().len()), (Some(11), 11), "{:?}", d.graph.op_counts());
        assert_eq!((r.opset(), r.graph.op_counts().len()), (Some(10), 23), "{:?}", r.graph.op_counts());
        // Every Conv's weight is a tensor of the graph, reached through Identity or Constant.
        for g in [&d.graph, &r.graph] {
            for n in g.nodes.iter().filter(|n| n.op == "Conv") {
                let w = n.inputs.get(1).and_then(|i| g.tensor(i)).unwrap_or_else(|| panic!("{} has no weight", n.name));
                assert_eq!(w.dims.len(), 4);
                assert_eq!(w.floats.len(), w.dims.iter().product::<usize>());
            }
        }
        let why = plan(&det, &rec).unwrap_err();
        assert!(why.contains("Conv") && why.contains("not built yet"), "{why}");
    }
}
