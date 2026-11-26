//! The device PP-OCRv5 computes on, and the factory the program registers with the engine.

use pdfcraft_ocr::OcrError;
use pdfcraft_ocr::ppocr::{NetworkFactory, Networks};

/// A wgpu device and queue to compute on, with what the adapter says it is.
///
/// The app passes the device it already draws with ([`GpuContext::shared`]), by the owner's
/// decision: no second device, and the drawing's own device-loss handling stays eframe's. The
/// kernels (P2) must therefore keep each submission short (well under a frame, about 16 ms),
/// catch their own validation errors in error scopes so they never reach the drawing's
/// uncaptured-error handler, and treat a lost device as a fallback to the CPU, not a crash.
#[derive(Clone, Debug)]
pub struct GpuContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub adapter: wgpu::AdapterInfo,
}

impl GpuContext {
    /// The drawing's device. A software adapter (WARP, llvmpipe: `DeviceType::Cpu`) is refused:
    /// it would be slower than reading on the CPU directly.
    pub fn shared(device: wgpu::Device, queue: wgpu::Queue, adapter: wgpu::AdapterInfo) -> Result<GpuContext, OcrError> {
        if adapter.device_type == wgpu::DeviceType::Cpu {
            return Err(OcrError::Load("wgpu".into(), format!("{} is a software adapter, not a GPU", adapter.name)));
        }
        Ok(GpuContext { device, queue, adapter })
    }

    /// The adapter's name and backend, as notes show it.
    pub fn describe(&self) -> String {
        format!("wgpu on {} ({:?})", self.adapter.name, self.adapter.backend)
    }
}

/// The GPU backend for PP-OCRv5. The program registers it with the engine
/// (`pdfcraft_engine::ocr::register_gpu`), which asks it only when a job turns the GPU on.
pub struct WgpuFactory {
    context: GpuContext,
}

impl WgpuFactory {
    pub fn new(context: GpuContext) -> WgpuFactory {
        WgpuFactory { context }
    }

    /// The device it computes on.
    pub fn context(&self) -> &GpuContext {
        &self.context
    }
}

impl NetworkFactory for WgpuFactory {
    fn name(&self) -> String {
        self.context.describe()
    }

    /// Reads both models with [`crate::onnx`] and, with no kernels yet, declines with the
    /// operators it cannot run: the engine then reads on the CPU and the pages say why.
    fn networks(&self, detection: &[u8], recognition: &[u8]) -> Result<Networks, OcrError> {
        match crate::plan(detection, recognition) {
            Err(why) => Err(OcrError::Load("wgpu".into(), why)),
            Ok(()) => Err(OcrError::Load("wgpu".into(), "the GPU networks are not built yet".into())),
        }
    }
}
