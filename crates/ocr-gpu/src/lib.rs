//! pdfcraft-ocr-gpu — PP-OCRv5 on the GPU (L7).
//!
//! `pdfcraft_ocr::ppocr` (L4) reads pages with its networks behind the `Network` trait and
//! knows no GPU API; the layering keeps wgpu at L7 and above. This crate is the GPU side:
//!
//! - [`onnx`]: our own minimal reader of the ONNX files (protobuf wire format), with every
//!   length, count and size checked.
//! - `ir` and `plan`: the graph turned into fused operations once per network (weights folded),
//!   then planned per input size (shapes, the shape arithmetic, shared buffers). No GPU needed,
//!   so they are tested everywhere.
//! - `wgsl`: the kernels, sized by uniforms (one pipeline serves every input size).
//! - `exec`: plans run on a wgpu device, in submissions of about 8 ms, inside error scopes.
//! - [`GpuContext`] and [`WgpuFactory`]: the device the app draws with (eframe's, by the owner's
//!   decision; software adapters are refused), and the `NetworkFactory` the program (L8)
//!   registers with the engine (`pdfcraft_engine::ocr::register_gpu`). The engine asks it only
//!   when a job turns the GPU on (`OcrSettings::gpu`, off by default), self-checks it against
//!   the CPU, and reads on the CPU whenever it fails.
//!
//! Off the web only: the web build has no PP-OCRv5 models.

#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable)]

// On wasm only the preparation is built (there is no device to plan or run for), so the
// fields only planning reads go unread there.
#[cfg_attr(target_arch = "wasm32", allow(dead_code))]
mod eval;
#[cfg_attr(target_arch = "wasm32", allow(dead_code))]
mod ir;
pub mod onnx;
#[cfg(not(target_arch = "wasm32"))]
mod plan;
#[cfg(not(target_arch = "wasm32"))]
mod wgsl;

#[cfg(not(target_arch = "wasm32"))]
mod context;
#[cfg(not(target_arch = "wasm32"))]
mod exec;
#[cfg(not(target_arch = "wasm32"))]
pub use context::{GpuContext, WgpuFactory, networks_on};
#[cfg(not(target_arch = "wasm32"))]
pub use exec::Gpu;

#[cfg(all(test, not(target_arch = "wasm32")))]
mod testmodel;
#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests_gpu;

/// The ONNX operators the GPU kernels cover (PP-OCRv5 server's detector and recogniser, and
/// the shape arithmetic they do).
pub const SUPPORTED_OPS: &[&str] = &[
    "Add",
    "AveragePool",
    "BatchNormalization",
    "Cast",
    "Concat",
    "Constant",
    "Conv",
    "ConvTranspose",
    "Div",
    "Dropout",
    "Exp",
    "Flatten",
    "Gather",
    "Identity",
    "MatMul",
    "MaxPool",
    "Mul",
    "Neg",
    "Pow",
    "ReduceMean",
    "Relu",
    "Reshape",
    "Resize",
    "Shape",
    "Sigmoid",
    "Slice",
    "Softmax",
    "Sqrt",
    "Squeeze",
    "Sub",
    "Tanh",
    "Transpose",
    "Unsqueeze",
];

/// The operators of `model` with no GPU kernel, sorted and once each.
pub fn unsupported_ops(model: &onnx::Model) -> Vec<String> {
    model.graph.op_counts().into_keys().filter(|op| !SUPPORTED_OPS.contains(&op.as_str())).collect()
}

/// Read and prepare both networks; the error is one sentence for a page's notes.
fn prepare_both(detection: &[u8], recognition: &[u8]) -> Result<(ir::Prepared, ir::Prepared), String> {
    let mut out = Vec::with_capacity(2);
    for (name, bytes) in [(pdfcraft_ocr::ppocr::PPOCR_DETECTION_MODEL, detection), (pdfcraft_ocr::ppocr::PPOCR_RECOGNITION_MODEL, recognition)] {
        let model = onnx::parse(bytes).map_err(|e| format!("{name}: {e}"))?;
        let missing = unsupported_ops(&model);
        if !missing.is_empty() {
            return Err(format!("{name}: the GPU has no kernels for {}", missing.join(", ")));
        }
        out.push(ir::prepare(&model).map_err(|e| format!("{name}: {e}"))?);
    }
    let rec = out.pop().ok_or("no recogniser")?;
    let det = out.pop().ok_or("no detector")?;
    Ok((det, rec))
}

/// Whether the GPU can run both networks: `Ok(())` when they read and prepare, else why not.
pub fn plan(detection: &[u8], recognition: &[u8]) -> Result<(), String> {
    prepare_both(detection, recognition).map(|_| ())
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;

    /// Window attributes a hostile model could use to break the size arithmetic (a zero
    /// kernel side made `k - 1` wrap, a zero stride divided by zero) or to make one dispatch
    /// take seconds (a huge kernel) are refused while preparing.
    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn hostile_window_attributes_are_refused() {
        use crate::testmodel::{A, M};
        let refused = |op: &str, w: &[usize], attrs: Vec<(&str, A)>| {
            let mut m = M::new(17);
            if !w.is_empty() {
                m.weight("w", w, 0.1);
                m.node(op, &["x", "w"], "y", &attrs);
            } else {
                m.node(op, &["x"], "y", &attrs);
            }
            let model = onnx::parse(&m.bytes()).unwrap();
            ir::prepare(&model).map(|_| ()).unwrap_err()
        };
        refused("ConvTranspose", &[1, 1, 0, 0], vec![]);
        refused("ConvTranspose", &[1, 1, 2, 2], vec![("strides", A::Is(vec![0, 0]))]);
        refused("ConvTranspose", &[1, 1, 2, 2], vec![("strides", A::Is(vec![2, 2])), ("output_padding", A::Is(vec![5, 5]))]);
        refused("Conv", &[1, 1, 65, 65], vec![]);
        refused("Conv", &[1, 1, 3, 3], vec![("dilations", A::Is(vec![1000, 1]))]);
        refused("Conv", &[1, 1, 3, 3], vec![("pads", A::Is(vec![5000, 0, 0, 0]))]);
        refused("MaxPool", &[], vec![("kernel_shape", A::Is(vec![1024, 1024]))]);
        refused("AveragePool", &[], vec![("kernel_shape", A::Is(vec![0, 2]))]);
        // An ordinary transposed convolution prepares.
        let mut m = M::new(17);
        m.weight("w", &[1, 1, 2, 2], 0.1);
        m.node("ConvTranspose", &["x", "w"], "y", &[("strides", A::Is(vec![2, 2]))]);
        assert!(ir::prepare(&onnx::parse(&m.bytes()).unwrap()).is_ok());
    }

    /// Constants that fold into far more than the weights hold (here ten Concats of four
    /// copies of a weight, 40 M values from 1 M) are refused while preparing, not allocated.
    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn constants_folding_past_the_budget_are_refused() {
        use crate::testmodel::M;
        let mut m = M::new(17);
        m.weight("w", &[1 << 20], 1.0);
        for i in 0..10 {
            m.node("Concat", &["w", "w", "w", "w"], &format!("c{i}"), &[("axis", crate::testmodel::A::I(0))]);
        }
        m.node("Relu", &["x"], "y", &[]);
        let model = onnx::parse(&m.bytes()).unwrap();
        let e = ir::prepare_within(&model, 1 << 24).unwrap_err();
        assert!(e.contains("constants"), "{e}");
        // Within the budget it prepares.
        assert!(ir::prepare_within(&model, 1 << 26).is_ok());
    }

    /// The real models, when installed: both parse with the operator counts of
    /// docs/ocr-ppocr.ja.md §2, prepare, and plan for the sizes the reader uses, with the
    /// detector's head fused and the recogniser's last layer read as best classes.
    #[test]
    fn the_real_models_prepare_and_plan() {
        let Some(models) = pdfcraft_ocr::ppocr::PpOcrModels::find() else {
            eprintln!("skipped: the PP-OCRv5 models are not installed (cargo xtask models --optional)");
            return;
        };
        let det = std::fs::read(&models.detection).unwrap();
        let rec = std::fs::read(&models.recognition).unwrap();
        let (d, r) = (onnx::parse(&det).unwrap(), onnx::parse(&rec).unwrap());
        assert_eq!((d.opset(), d.graph.op_counts().len()), (Some(11), 11), "{:?}", d.graph.op_counts());
        assert_eq!((r.opset(), r.graph.op_counts().len()), (Some(10), 23), "{:?}", r.graph.op_counts());
        assert!(unsupported_ops(&d).is_empty() && unsupported_ops(&r).is_empty());
        let (pd, pr) = prepare_both(&det, &rec).unwrap();
        assert!(pd.ops.iter().any(|o| matches!(o, ir::Op::Head { .. })), "the DB head is one kernel");
        assert!(pr.best.is_some(), "the recogniser ends in a softmax over classes");
        let lim = plan::Limits { max_binding: 1 << 30, max_buffer: 1 << 30, max_groups: 65535 };
        for dims in [[1, 3, 64, 64], [1, 3, 960, 672], [1, 3, 1600, 1152]] {
            let p = plan::plan(&pd, &dims, false, lim).unwrap();
            assert_eq!(p.output_dims, vec![1, 1, dims[2], dims[3]]);
            let total: usize = p.slots.iter().sum::<usize>() * 4;
            eprintln!("det {dims:?}: {} dispatches, {} buffers, {} MB", p.steps.len(), p.slots.len(), total >> 20);
        }
        for dims in [[1, 3, 48, 160], [8, 3, 48, 3200]] {
            let p = plan::plan(&pr, &dims, true, lim).unwrap();
            assert_eq!(p.output_dims, vec![dims[0], dims[3] / 8, 18385]);
            let full = plan::plan(&pr, &dims, false, lim).unwrap();
            assert_eq!(full.output_dims, p.output_dims);
            eprintln!("rec {dims:?}: {} dispatches, {} MB", p.steps.len(), (p.slots.iter().sum::<usize>() * 4) >> 20);
        }
    }
}
