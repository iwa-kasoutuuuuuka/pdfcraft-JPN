//! The PP-OCRv5 engines the benchmark times: the library's own (`pdfcraft_ocr::ppocr`, rten on
//! the CPU) and, built with `--features ort-cuda` for local benchmarking only, the same pipeline
//! with its two networks run by ONNX Runtime's CUDA provider. Pre- and post-processing are the
//! library's in both, so the scores compare the runtimes alone.

use std::path::PathBuf;
use std::time::Instant;

use pdfcraft_ocr::ppocr::{PpOcr, PpOcrModels};

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
    use pdfcraft_ocr::ppocr::Network;

    fn err(e: impl std::fmt::Display) -> OcrError {
        OcrError::Recognize(e.to_string())
    }

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

    impl Network for OrtNet {
        fn run(&self, shape: [usize; 4], data: Vec<f32>) -> Result<(Vec<usize>, Vec<f32>), OcrError> {
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

/// A loaded engine (detection input side `det_side`, clamped by the library) and how long
/// loading took, in seconds. `dir`: the model folder; else where the app looks.
pub fn find(dir: Option<&PathBuf>, det_side: usize, runtime: Runtime) -> Result<(PpOcr, f64), String> {
    let start = Instant::now();
    let models = match dir {
        Some(d) => PpOcrModels::in_dir(d),
        None => PpOcrModels::find(),
    }
    .ok_or_else(|| {
        let place = dir.map_or_else(|| "the model folders".to_string(), |d| d.display().to_string());
        format!("no PP-OCRv5 models in {place} (run `cargo xtask models --optional`)")
    })?;
    let engine = load(&models, runtime)?;
    Ok((engine.with_det_side(det_side), start.elapsed().as_secs_f64()))
}

fn load(models: &PpOcrModels, runtime: Runtime) -> Result<PpOcr, String> {
    match runtime {
        Runtime::Rten => PpOcr::load(models).map_err(|e| e.to_string()),
        #[cfg(feature = "ort-cuda")]
        Runtime::OrtCuda => {
            let yml = std::fs::read_to_string(&models.dictionary).map_err(|e| format!("{}: {e}", models.dictionary.display()))?;
            let net = |p: &std::path::Path| ort_net::OrtNet::load(p).map(|n| Box::new(n) as Box<dyn pdfcraft_ocr::ppocr::Network>);
            PpOcr::with_networks(net(&models.detection)?, net(&models.recognition)?, &yml, "ppocr-cuda").map_err(|e| e.to_string())
        }
    }
}
