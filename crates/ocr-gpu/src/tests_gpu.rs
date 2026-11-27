//! The kernels against rten (the CPU engine) on small models built here, one per kind of fused
//! operation the PP-OCRv5 networks use, and on the real models when they are installed. They
//! run on whatever adapter this machine has, software ones included (WARP on Windows), and are
//! skipped, saying so, where there is none.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::sync::{Arc, OnceLock};
use std::task::{Context, Poll, Waker};

use pdfcraft_ocr::ppocr::{Network, best_of, cpu_network};

use crate::exec::{Gpu, Net};
use crate::testmodel::{A, M, input};
use crate::{ir, onnx};

fn ready<F: std::future::Future>(fut: F) -> Option<F::Output> {
    let mut fut = std::pin::pin!(fut);
    let mut cx = Context::from_waker(Waker::noop());
    for _ in 0..1_000_000 {
        if let Poll::Ready(v) = fut.as_mut().poll(&mut cx) {
            return Some(v);
        }
        std::thread::yield_now();
    }
    None
}

/// A device for the tests: a GPU if there is one, else a software adapter.
pub fn device() -> Option<Arc<Gpu>> {
    static DEVICE: OnceLock<Option<Arc<Gpu>>> = OnceLock::new();
    DEVICE.get_or_init(new_device).clone()
}

/// A device of its own (for a test that destroys it).
fn new_device() -> Option<Arc<Gpu>> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = [false, true].into_iter().find_map(|fallback| {
        ready(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: fallback,
            compatible_surface: None,
            apply_limit_buckets: false,
        }))?
        .ok()
    })?;
    let info = adapter.get_info();
    let mut limits = wgpu::Limits::default();
    let a = adapter.limits();
    limits.max_storage_buffer_binding_size = a.max_storage_buffer_binding_size.min(1 << 30);
    limits.max_buffer_size = a.max_buffer_size.min(1 << 30);
    let (device, queue) = ready(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("pdfcraft-ocr-gpu tests"),
        required_features: wgpu::Features::empty(),
        required_limits: limits,
        experimental_features: wgpu::ExperimentalFeatures::disabled(),
        memory_hints: wgpu::MemoryHints::Performance,
        trace: wgpu::Trace::Off,
    }))?
    .ok()?;
    eprintln!("GPU tests on {} ({:?}, {:?})", info.name, info.backend, info.device_type);
    Some(Arc::new(Gpu::new(device, queue)))
}

macro_rules! gpu_or_skip {
    () => {
        match device() {
            Some(g) => g,
            None => {
                eprintln!("skipped: no wgpu adapter on this machine");
                return;
            }
        }
    };
}

fn net(gpu: &Arc<Gpu>, bytes: &[u8]) -> Net {
    let model = onnx::parse(bytes).unwrap();
    Net::new(gpu.clone(), ir::prepare(&model).unwrap())
}

/// The model through rten and through the GPU agree within `tol` (relative to the largest
/// value), with the same shape.
fn agree(gpu: &Arc<Gpu>, m: &M, dims: [usize; 4], tol: f32) -> (Vec<usize>, Vec<f32>) {
    let bytes = m.bytes();
    let cpu = cpu_network(bytes.clone()).unwrap();
    let want = cpu.run(dims, input(dims)).unwrap();
    let got = net(gpu, &bytes).run(dims, input(dims)).unwrap();
    assert_eq!(got.0, want.0, "shapes");
    let scale = want.1.iter().fold(1.0f32, |a, v| a.max(v.abs()));
    let worst = got.1.iter().zip(&want.1).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    assert!(worst <= tol * scale, "differs by {worst} (scale {scale})");
    want
}

/// Conv (3×3, stride 2) + BatchNormalization + Relu, a 1×1 conv with a bias and a residual
/// Add, a 2×2 SAME_UPPER conv, and a standalone BN after an Add.
#[test]
fn convolutions_with_folded_norms_activations_and_residuals() {
    let gpu = gpu_or_skip!();
    let mut m = M::new(11);
    m.weight("w0", &[8, 3, 3, 3], 0.3)
        .node("Conv", &["x", "w0"], "c0", &[("kernel_shape", A::Is(vec![3, 3])), ("pads", A::Is(vec![1, 1, 1, 1])), ("strides", A::Is(vec![2, 2]))])
        .weight("g", &[8], 1.0)
        .weight("b", &[8], 0.5)
        .weight("mu", &[8], 0.1)
        .init("var", &[8], &[0.5, 1.0, 1.5, 2.0, 0.7, 0.9, 1.1, 1.3])
        .node("BatchNormalization", &["c0", "g", "b", "mu", "var"], "n0", &[("epsilon", A::F(1e-5))])
        .node("Relu", &["n0"], "r0", &[])
        .weight("w1", &[8, 8, 1, 1], 0.4)
        .weight("b1", &[8], 0.2)
        .node("Conv", &["r0", "w1", "b1"], "c1", &[])
        .node("Add", &["c1", "r0"], "a1", &[])
        .weight("w2", &[8, 8, 2, 2], 0.3)
        .node("Conv", &["a1", "w2"], "c2", &[("auto_pad", A::S("SAME_UPPER"))])
        .node("Add", &["c2", "a1"], "a2", &[])
        .node("BatchNormalization", &["a2", "g", "b", "mu", "var"], "n2", &[])
        .node("Relu", &["n2"], "out", &[]);
    agree(&gpu, &m, [2, 3, 21, 34], 1e-4);
}

/// Depthwise convolutions with strides (2, 1) and (1, 2), a 5×5 one, and a grouped one.
#[test]
fn depthwise_and_grouped_convolutions() {
    let gpu = gpu_or_skip!();
    let mut m = M::new(11);
    m.weight("w0", &[6, 3, 1, 1], 0.5)
        .node("Conv", &["x", "w0"], "c0", &[])
        .weight("d1", &[6, 1, 3, 3], 0.4)
        .node("Conv", &["c0", "d1"], "c1", &[("group", A::I(6)), ("pads", A::Is(vec![1, 1, 1, 1])), ("strides", A::Is(vec![2, 1]))])
        .weight("d2", &[6, 1, 3, 3], 0.4)
        .node("Conv", &["c1", "d2"], "c2", &[("group", A::I(6)), ("pads", A::Is(vec![1, 1, 1, 1])), ("strides", A::Is(vec![1, 2]))])
        .weight("d3", &[6, 1, 5, 5], 0.2)
        .weight("bd3", &[6], 0.1)
        .node("Conv", &["c2", "d3", "bd3"], "c3", &[("group", A::I(6)), ("pads", A::Is(vec![2, 2, 2, 2]))])
        .node("Relu", &["c3"], "r3", &[])
        .weight("g4", &[6, 3, 3, 3], 0.3)
        .node("Conv", &["r3", "g4"], "out", &[("group", A::I(2)), ("pads", A::Is(vec![1, 1, 1, 1]))]);
    agree(&gpu, &m, [1, 3, 17, 23], 1e-4);
}

/// The DB head as the detector has it: a concat of a ×2 nearest Resize and another branch
/// read in place, 3×3 conv + BN + Relu, a 1×1 to one channel + bias, Sigmoid, the mean with
/// the ConvTranspose branch (+ bias, Sigmoid). Also MaxPool 2×2 SAME_UPPER.
#[test]
fn the_db_head_fused_with_its_virtual_concat_and_resize() {
    let gpu = gpu_or_skip!();
    let mut m = M::new(11);
    m.weight("w0", &[4, 3, 3, 3], 0.4)
        .node("Conv", &["x", "w0"], "half", &[("pads", A::Is(vec![1, 1, 1, 1])), ("strides", A::Is(vec![2, 2]))])
        .node(
            "MaxPool",
            &["half"],
            "pool",
            &[("kernel_shape", A::Is(vec![2, 2])), ("strides", A::Is(vec![1, 1])), ("auto_pad", A::S("SAME_UPPER")), ("ceil_mode", A::I(1))],
        )
        .weight("wt", &[4, 1, 2, 2], 0.5)
        .init("bt", &[1, 1, 1, 1], &[0.1])
        .node("ConvTranspose", &["pool", "wt"], "t0", &[("kernel_shape", A::Is(vec![2, 2])), ("strides", A::Is(vec![2, 2]))])
        .node("Add", &["t0", "bt"], "t1", &[])
        .node("Sigmoid", &["t1"], "sig0", &[])
        .init("roi", &[0], &[])
        .init("scales", &[4], &[1.0, 1.0, 2.0, 2.0])
        .node(
            "Resize",
            &["pool", "roi", "scales"],
            "up",
            &[("mode", A::S("nearest")), ("coordinate_transformation_mode", A::S("asymmetric")), ("nearest_mode", A::S("floor"))],
        )
        .node("Concat", &["up", "sig0"], "cat", &[("axis", A::I(1))])
        .weight("w1", &[6, 5, 3, 3], 0.3)
        .node("Conv", &["cat", "w1"], "c1", &[("pads", A::Is(vec![1, 1, 1, 1]))])
        .weight("g", &[6], 1.0)
        .weight("b", &[6], 0.3)
        .weight("mu", &[6], 0.1)
        .init("var", &[6], &[0.5, 1.0, 1.5, 2.0, 0.7, 0.9])
        .node("BatchNormalization", &["c1", "g", "b", "mu", "var"], "n1", &[])
        .node("Relu", &["n1"], "r1", &[])
        .weight("w2", &[1, 6, 1, 1], 0.5)
        .node("Conv", &["r1", "w2"], "c2", &[])
        .init("b2", &[1, 1, 1, 1], &[-0.2])
        .node("Add", &["c2", "b2"], "c3", &[])
        .node("Sigmoid", &["c3"], "sig1", &[])
        .node("Add", &["sig0", "sig1"], "sum", &[])
        .init("half_k", &[1], &[0.5])
        .node("Mul", &["sum", "half_k"], "mean", &[])
        .init("zero", &[], &[0.0])
        .node("Add", &["mean", "zero"], "out", &[]);
    let p = ir::prepare(&onnx::parse(&m.bytes()).unwrap()).unwrap();
    assert!(p.ops.iter().any(|o| matches!(o, ir::Op::Head { other: Some(_), .. })), "{:#?}", p.ops);
    assert!(!p.ops.iter().any(|o| matches!(o, ir::Op::Concat { .. } | ir::Op::Resize { .. })), "the concat and resize are read in place");
    agree(&gpu, &m, [1, 3, 32, 48], 1e-4);
}

/// The recogniser's sequence part: AveragePool, the shape arithmetic (Shape → Slice → Concat →
/// Reshape), Transpose, a layer norm written out, attention (MatMul of two computed values,
/// Softmax on axis 3, Slice + Squeeze), swish, and the last MatMul + bias + Softmax, also read
/// as best classes.
fn sequence_model() -> M {
    let mut m = M::new(10);
    let c = 8;
    m.weight("w0", &[c, 3, 3, 3], 0.4)
        .node("Conv", &["x", "w0"], "c0", &[("pads", A::Is(vec![1, 1, 1, 1]))])
        .node("AveragePool", &["c0"], "p0", &[("kernel_shape", A::Is(vec![3, 2])), ("strides", A::Is(vec![3, 2]))])
        .node("Shape", &["p0"], "s0", &[])
        .init_i("st0", &[1], &[0])
        .init_i("en0", &[1], &[2])
        .node("Slice", &["s0", "st0", "en0"], "s1", &[])
        .init_i("minus1", &[1], &[-1])
        .node("Concat", &["s1", "minus1"], "s2", &[("axis", A::I(0))])
        .node("Reshape", &["p0", "s2"], "flat", &[])
        .node("Transpose", &["flat"], "seq", &[("perm", A::Is(vec![0, 2, 1]))])
        // layer norm
        .node("ReduceMean", &["seq"], "mu", &[("axes", A::Is(vec![-1]))])
        .node("Sub", &["seq", "mu"], "d", &[])
        .init("two", &[], &[2.0])
        .node("Pow", &["d", "two"], "d2", &[])
        .node("ReduceMean", &["d2"], "var", &[("axes", A::Is(vec![-1]))])
        .init("eps", &[], &[1e-6])
        .node("Add", &["var", "eps"], "ve", &[])
        .node("Sqrt", &["ve"], "sd", &[])
        .node("Div", &["d", "sd"], "nrm", &[])
        .weight("gam", &[c], 1.0)
        .node("Mul", &["nrm", "gam"], "ln0", &[])
        .weight("bet", &[c], 0.2)
        .node("Add", &["ln0", "bet"], "ln", &[])
        // qkv = ln · W + b → [N, T, 3, 2, 4] → [3, N, 2, T, 4]
        .weight("wqkv", &[c, 3 * c], 0.4)
        .node("MatMul", &["ln", "wqkv"], "qkv0", &[])
        .weight("bqkv", &[3 * c], 0.1)
        .node("Add", &["qkv0", "bqkv"], "qkv1", &[])
        .init_i("shape5", &[5], &[0, -1, 3, 2, 4])
        .node("Reshape", &["qkv1", "shape5"], "qkv2", &[])
        .node("Transpose", &["qkv2"], "qkv", &[("perm", A::Is(vec![2, 0, 3, 1, 4]))]);
    for (i, name) in ["q", "k", "v"].iter().enumerate() {
        let (st, en, ax) = (format!("slice_{name}_start"), format!("slice_{name}_end"), format!("slice_{name}_axis"));
        m.init_i(&st, &[1], &[i as i64])
            .init_i(&en, &[1], &[i as i64 + 1])
            .init_i(&ax, &[1], &[0])
            .node("Slice", &["qkv", &st, &en, &ax], &format!("{name}0"), &[])
            .node("Squeeze", &[&format!("{name}0")], name, &[("axes", A::Is(vec![0]))]);
    }
    m.init("scale", &[1], &[0.5])
        .node("Mul", &["q", "scale"], "qs", &[])
        .node("Transpose", &["k"], "kt", &[("perm", A::Is(vec![0, 1, 3, 2]))])
        .node("MatMul", &["qs", "kt"], "att0", &[])
        .node("Softmax", &["att0"], "att", &[("axis", A::I(3))])
        .node("MatMul", &["att", "v"], "ctx0", &[])
        .node("Transpose", &["ctx0"], "ctx1", &[("perm", A::Is(vec![0, 2, 1, 3]))])
        .init_i("shape3", &[3], &[0, 0, -1])
        .node("Reshape", &["ctx1", "shape3"], "ctx", &[])
        .node("Add", &["ctx", "ln"], "res", &[])
        // swish MLP
        .weight("wm", &[c, 2 * c], 0.4)
        .node("MatMul", &["res", "wm"], "h0", &[])
        .weight("bm", &[2 * c], 0.1)
        .node("Add", &["h0", "bm"], "h1", &[])
        .node("Sigmoid", &["h1"], "hs", &[])
        .node("Mul", &["h1", "hs"], "h", &[])
        // classes
        .weight("wc", &[2 * c, 37], 0.6)
        .node("MatMul", &["h", "wc"], "l0", &[])
        .weight("bc", &[37], 0.2)
        .node("Add", &["l0", "bc"], "l1", &[])
        .node("Softmax", &["l1"], "out", &[("axis", A::I(2))]);
    m
}

#[test]
fn the_sequence_part_and_its_best_classes() {
    let gpu = gpu_or_skip!();
    let m = sequence_model();
    let dims = [3, 3, 6, 40];
    let (shape, want) = agree(&gpu, &m, dims, 1e-4);
    // 6 rows pool to 2, 40 columns to 20: 40 positions.
    assert_eq!(shape, vec![3, 40, 37]);
    let n = net(&gpu, &m.bytes());
    assert!(n.has_best());
    let (bshape, best) = n.run_best(dims, input(dims)).unwrap();
    assert_eq!(bshape, shape);
    let cpu = best_of(&shape, &want).unwrap();
    assert_eq!(best.len(), cpu.len());
    for (g, c) in best.iter().zip(&cpu) {
        assert_eq!(g.0, c.0, "best class");
        assert!((g.1 - c.1).abs() < 1e-4, "{g:?} {c:?}");
    }
}

/// Different input sizes reuse the network (plans are per size); a size the network cannot
/// take is an error, not a panic, and the next run still works.
#[test]
fn sizes_change_and_bad_sizes_are_errors() {
    let gpu = gpu_or_skip!();
    let m = sequence_model();
    let n = net(&gpu, &m.bytes());
    for dims in [[1, 3, 6, 40], [2, 3, 6, 160], [1, 3, 6, 40]] {
        let cpu = cpu_network(m.bytes()).unwrap().run(dims, input(dims)).unwrap();
        let got = n.run(dims, input(dims)).unwrap();
        assert_eq!(got.0, cpu.0);
    }
    assert!(n.run([1, 3, 2, 40], input([1, 3, 2, 40])).is_err(), "2 rows are fewer than the pooling window's 3");
    assert!(n.run([1, 3, 6, 40], vec![0.0; 3]).is_err(), "too few values");
    assert!(n.run([1, 3, 6, 40], input([1, 3, 6, 40])).is_ok());
    // A batch whose plan would pass the memory budget is read in parts, with the same result.
    let dims = [4, 3, 6, 40];
    let whole = n.run(dims, input(dims)).unwrap();
    let split = Net::new(gpu.clone(), ir::prepare(&onnx::parse(&m.bytes()).unwrap()).unwrap()).with_split_budget(1).run(dims, input(dims)).unwrap();
    assert_eq!(whole.0, split.0);
    let worst = whole.1.iter().zip(&split.1).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    assert!(worst < 1e-5, "{worst}");
}

/// The real networks, when installed: the GPU's output against rten's, at the self-check's
/// sizes and one page-sized input; the recogniser's best classes against the CPU's.
#[test]
fn the_real_models_agree_with_the_cpu() {
    let gpu = gpu_or_skip!();
    let Some(models) = pdfcraft_ocr::ppocr::PpOcrModels::find() else {
        eprintln!("skipped: the PP-OCRv5 models are not installed (cargo xtask models --optional)");
        return;
    };
    let det = std::fs::read(&models.detection).unwrap();
    let rec = std::fs::read(&models.recognition).unwrap();
    let (gd, gr) = crate::networks_on(gpu, &det, &rec).unwrap();
    let (cd, cr) = (cpu_network(det).unwrap(), cpu_network(rec).unwrap());
    for dims in [[1, 3, 64, 64], [1, 3, 672, 480]] {
        let want = cd.run(dims, input(dims)).unwrap();
        let t = std::time::Instant::now();
        let got = gd.run(dims, input(dims)).unwrap();
        eprintln!("det {dims:?} on the GPU: {:.1} ms (first run: builds the kernels)", t.elapsed().as_secs_f64() * 1e3);
        assert_eq!(got.0, want.0);
        let worst = got.1.iter().zip(&want.1).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        let flips = got.1.iter().zip(&want.1).filter(|(a, b)| (**a > 0.3) != (**b > 0.3)).count();
        eprintln!("det {dims:?}: max difference {worst}, {flips} pixels on the other side of 0.3");
        assert!(worst < 1e-3, "{worst}");
    }
    for dims in [[1, 3, 48, 160], [4, 3, 48, 640]] {
        let want = cr.run(dims, input(dims)).unwrap();
        let cpu = best_of(&want.0, &want.1).unwrap();
        let got = gr.run_best(dims, input(dims)).unwrap();
        assert_eq!(got.0, want.0);
        let classes = want.0.last().copied().unwrap();
        for ((g, c), row) in got.1.iter().zip(&cpu).zip(want.1.chunks(classes)) {
            let at = row[g.0];
            assert!(c.1 - at < 1e-3 && (g.1 - c.1).abs() < 1e-3, "GPU {g:?}, CPU {c:?}");
        }
        let full = gr.run(dims, input(dims)).unwrap();
        let worst = full.1.iter().zip(&want.1).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        eprintln!("rec {dims:?}: max difference {worst}");
        assert!(worst < 1e-3);
    }
}

/// An operator with no kernel is named when the model is prepared.
#[test]
fn an_operator_without_a_kernel_is_named() {
    let mut m = M::new(11);
    m.node("LpNormalization", &["x"], "out", &[]);
    let model = onnx::parse(&m.bytes()).unwrap();
    let e = ir::prepare(&model).unwrap_err();
    assert!(e.contains("LpNormalization"), "{e}");
    assert_eq!(crate::unsupported_ops(&model), vec!["LpNormalization".to_string()]);
}

/// A device lost under a network (here destroyed; in the field a driver reset or a TDR) makes
/// the run an error the engine falls back from. wgpu panics in `Device::poll` then, which
/// used to escape the run.
#[test]
fn a_lost_device_is_an_error_not_a_panic() {
    let Some(gpu) = new_device() else {
        eprintln!("skipped: no wgpu adapter on this machine");
        return;
    };
    let m = sequence_model();
    let n = net(&gpu, &m.bytes());
    let dims = [1, 3, 6, 40];
    assert!(n.run(dims, input(dims)).is_ok());
    gpu.device.destroy();
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| (n.run(dims, input(dims)).is_err(), n.run_best(dims, input(dims)).is_err())));
    assert_eq!(r.ok(), Some((true, true)), "a run on a lost device must be an error, not a panic");
}
