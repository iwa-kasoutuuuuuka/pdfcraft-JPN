//! The device PP-OCRv5 computes on, and the factory the program registers with the engine.

use std::sync::Arc;

use pdfcraft_ocr::OcrError;
use pdfcraft_ocr::ppocr::{NetworkFactory, Networks};

use crate::exec::{Gpu, Net};

/// A wgpu device and queue to compute on, with what the adapter says it is.
///
/// The app passes the device it already draws with ([`GpuContext::shared`]), by the owner's
/// decision: no second device, and the drawing's own device-loss handling stays eframe's. The
/// kernels therefore submit in pieces of about 8 ms ([`crate::exec`]), catch their own errors in
/// error scopes so they never reach the drawing's uncaptured-error handler, and treat a lost
/// device as a fallback to the CPU, not a crash.
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

    /// A device of its own, for a program that draws nothing (`pdfcraft-cli --gpu`): the
    /// high-performance adapter, with buffers of up to 1 GiB where it allows them (the
    /// detector at its largest side needs more than the default 128 MiB). A software adapter,
    /// or none, is an error naming why; the caller then reads on the CPU.
    pub fn headless() -> Result<GpuContext, OcrError> {
        let load = |why: String| OcrError::Load("wgpu".into(), why);
        // `Instance::new` panics when the program enabled no backend (the crate leaves them to
        // the program: the app gets eframe's, the CLI its own).
        if wgpu::Instance::enabled_backend_features().is_empty() {
            return Err(load("this program was built without a graphics backend".into()));
        }
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = crate::exec::block_on_idle(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
            apply_limit_buckets: false,
        }))
        .ok_or_else(|| load("the graphics driver did not answer".into()))?
        .map_err(|e| load(format!("no graphics adapter: {e}")))?;
        let info = adapter.get_info();
        let a = adapter.limits();
        let limits = wgpu::Limits {
            max_storage_buffer_binding_size: a.max_storage_buffer_binding_size.min(1 << 30),
            max_buffer_size: a.max_buffer_size.min(1 << 30),
            ..wgpu::Limits::default()
        };
        let (device, queue) = crate::exec::block_on_idle(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("pdfcraft-ocr"),
            required_features: wgpu::Features::empty(),
            required_limits: limits,
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::Off,
        }))
        .ok_or_else(|| load("the graphics driver did not answer".into()))?
        .map_err(|e| load(format!("{}: {e}", info.name)))?;
        GpuContext::shared(device, queue, info)
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
    gpu: Arc<Gpu>,
}

impl WgpuFactory {
    pub fn new(context: GpuContext) -> WgpuFactory {
        let gpu = Arc::new(Gpu::new(context.device.clone(), context.queue.clone()));
        WgpuFactory { context, gpu }
    }

    /// The device it computes on.
    pub fn context(&self) -> &GpuContext {
        &self.context
    }
}

/// Both networks on `gpu`, from the model files' bytes.
pub fn networks_on(gpu: Arc<Gpu>, detection: &[u8], recognition: &[u8]) -> Result<Networks, OcrError> {
    crate::exec::guarded("preparing the models", || {
        let (det, rec) = crate::prepare_both(detection, recognition).map_err(|why| OcrError::Load("wgpu".into(), why))?;
        Ok((
            Box::new(Net::new(gpu.clone(), det)) as Box<dyn pdfcraft_ocr::ppocr::Network>,
            Box::new(Net::new(gpu, rec)) as Box<dyn pdfcraft_ocr::ppocr::Network>,
        ))
    })
}

impl NetworkFactory for WgpuFactory {
    fn name(&self) -> String {
        self.context.describe()
    }

    /// Reads both models with [`crate::onnx`], prepares them ([`crate::ir`]), and returns them
    /// on this device. Kernels are built when a size is first read; the engine's self-check
    /// runs one of each before any page.
    fn networks(&self, detection: &[u8], recognition: &[u8]) -> Result<Networks, OcrError> {
        networks_on(self.gpu.clone(), detection, recognition)
    }
}
