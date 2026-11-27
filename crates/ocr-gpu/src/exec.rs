//! Running plans on a wgpu device: pipelines made once per shader, weights uploaded once per
//! network, plans built once per input size (and kept while they fit a memory budget), and
//! each run submitted in pieces short enough not to hold up the drawing that shares the device.
//!
//! Every device call sits inside error scopes (validation, out of memory), so a failure comes
//! back as this run's `Err`, never as the drawing's uncaptured-error handler; a lost device, a
//! timeout or a failed map are errors too. The engine then reads on the CPU.
//!
//! wgpu 30 panics in `Device::poll` when the device is lost (a driver reset, a TDR, memory
//! exhausted): it maps only timeouts to `PollError`. So every poll goes through [`poll`], and
//! every entry point ([`Net::run`], [`Net::run_best`], [`crate::networks_on`]) through
//! [`guarded`], which turn such a panic into this run's error.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll, Waker};
use std::time::{Duration, Instant};

use pdfcraft_ocr::OcrError;
use pdfcraft_ocr::ppocr::{Best, Network};
use wgpu::util::DeviceExt;

use crate::ir::{self, Prepared};
use crate::plan::{self, Bind, Plan, Shader};
use crate::wgsl;

/// How long one piece of a run may take before it counts as a hung GPU.
const SUBMIT_TIMEOUT: Duration = Duration::from_secs(20);
/// What one submission aims at: well under a frame of the drawing.
const SLICE: Duration = Duration::from_millis(8);
/// Memory kept in cached plans (beyond the most recent), per network.
const PLAN_BUDGET: u64 = 256 << 20;
/// A batch whose plan needs more than this is read in parts (the recogniser's eight widest
/// lines would take 1.5 GB at once).
const SPLIT_BUDGET: u64 = 512 << 20;
/// The most buffer memory one plan may take (its intermediates; the weights come on top, at
/// most the model's size). The detector at its largest side needs about 700 MB; a size beyond
/// this is refused, and the engine reads that page on the CPU.
const MAX_PLAN_BYTES: u64 = 2 << 30;

fn gpu_err(e: impl std::fmt::Display) -> OcrError {
    OcrError::Recognize(format!("GPU: {e}"))
}

/// `f`, with a panic inside wgpu (a lost device) as an error naming `what`. The caller keeps
/// no state a half-finished `f` could leave wrong: plans and weights are rebuilt on the next
/// run, and the locks recover from poisoning.
pub(crate) fn guarded<T>(what: &str, f: impl FnOnce() -> Result<T, OcrError>) -> Result<T, OcrError> {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)) {
        Ok(r) => r,
        Err(cause) => {
            let why = cause.downcast_ref::<String>().map(String::as_str).or_else(|| cause.downcast_ref::<&str>().copied()).unwrap_or("a panic");
            Err(gpu_err(format!("{what} stopped (the device may have been lost): {why}")))
        }
    }
}

/// `device.poll`, with wgpu's panic on a lost device as an error.
fn poll(device: &wgpu::Device, ty: wgpu::PollType) -> Result<(), OcrError> {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| device.poll(ty))) {
        Ok(Ok(_)) => Ok(()),
        Ok(Err(e)) => Err(gpu_err(e)),
        Err(_) => Err(gpu_err("the device was lost")),
    }
}

/// Resolve a future the native backends complete during `device.poll` (error scopes, maps).
fn block_on<F: std::future::Future>(device: &wgpu::Device, fut: F) -> Result<F::Output, OcrError> {
    let mut fut = std::pin::pin!(fut);
    let mut cx = Context::from_waker(Waker::noop());
    let start = Instant::now();
    loop {
        if let Poll::Ready(v) = fut.as_mut().poll(&mut cx) {
            return Ok(v);
        }
        if start.elapsed() > SUBMIT_TIMEOUT {
            return Err(gpu_err("the device did not answer"));
        }
        poll(device, wgpu::PollType::Poll)?;
        std::thread::yield_now();
    }
}

/// Resolve a future that needs no device to poll (requesting an adapter or a device: the
/// native backends answer at once), giving up after [`SUBMIT_TIMEOUT`].
pub(crate) fn block_on_idle<F: std::future::Future>(fut: F) -> Option<F::Output> {
    let mut fut = std::pin::pin!(fut);
    let mut cx = Context::from_waker(Waker::noop());
    let start = Instant::now();
    while start.elapsed() < SUBMIT_TIMEOUT {
        if let Poll::Ready(v) = fut.as_mut().poll(&mut cx) {
            return Some(v);
        }
        std::thread::yield_now();
    }
    None
}

/// Validation and out-of-memory errors of `f`'s device calls, as this call's error.
fn scoped<T>(device: &wgpu::Device, f: impl FnOnce() -> Result<T, OcrError>) -> Result<T, OcrError> {
    let oom = device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
    let validation = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let out = f();
    let v = block_on(device, validation.pop())?;
    let o = block_on(device, oom.pop())?;
    if let Some(e) = v.or(o) {
        return Err(gpu_err(e));
    }
    out
}

/// The bytes of a plan's intermediate buffers.
fn plan_bytes(plan: &Plan) -> u64 {
    plan.slots.iter().fold(0u64, |a, &n| a.saturating_add((n.max(1) as u64).saturating_mul(4)))
}

fn bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|f| f.to_le_bytes()).collect()
}

/// The device, its pipelines, and how much work fits one submission.
pub struct Gpu {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    limits: plan::Limits,
    wg_bytes: u32,
    wg_threads: u32,
    storage_per_stage: u32,
    pipelines: Mutex<HashMap<Shader, Arc<wgpu::ComputePipeline>>>,
    /// Work (multiply-adds) per submission, adjusted to [`SLICE`].
    pace: Mutex<f64>,
}

impl Gpu {
    pub fn new(device: wgpu::Device, queue: wgpu::Queue) -> Gpu {
        let l = device.limits();
        let limits = plan::Limits {
            max_binding: l.max_storage_buffer_binding_size,
            max_buffer: l.max_buffer_size,
            max_groups: l.max_compute_workgroups_per_dimension,
        };
        Gpu {
            device,
            queue,
            limits,
            wg_bytes: l.max_compute_workgroup_storage_size,
            wg_threads: l.max_compute_invocations_per_workgroup,
            storage_per_stage: l.max_storage_buffers_per_shader_stage,
            pipelines: Mutex::new(HashMap::new()),
            pace: Mutex::new(2.0e9),
        }
    }

    pub fn limits(&self) -> plan::Limits {
        self.limits
    }

    fn pipeline(&self, shader: &Shader) -> Result<Arc<wgpu::ComputePipeline>, OcrError> {
        if let Some(p) = self.pipelines.lock().unwrap_or_else(|e| e.into_inner()).get(shader) {
            return Ok(p.clone());
        }
        if wgsl::workgroup_bytes(shader) > self.wg_bytes || self.wg_threads < 256 {
            return Err(gpu_err(format!(
                "the device allows {} bytes of workgroup memory and {} threads; the kernels need {} and 256",
                self.wg_bytes,
                self.wg_threads,
                wgsl::workgroup_bytes(shader)
            )));
        }
        let src = wgsl::source(shader);
        let p = scoped(&self.device, || {
            let module = self
                .device
                .create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("pdfcraft-ocr"), source: wgpu::ShaderSource::Wgsl(src.into()) });
            Ok(self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("pdfcraft-ocr"),
                layout: None,
                module: &module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            }))
        })
        .map_err(|e| gpu_err(format!("{shader:?}: {e}")))?;
        let p = Arc::new(p);
        self.pipelines.lock().unwrap_or_else(|e| e.into_inner()).insert(shader.clone(), p.clone());
        Ok(p)
    }

    /// Wait for `index`, failing after [`SUBMIT_TIMEOUT`].
    fn wait(&self, index: wgpu::SubmissionIndex) -> Result<(), OcrError> {
        poll(&self.device, wgpu::PollType::Wait { submission_index: Some(index), timeout: Some(SUBMIT_TIMEOUT) })
    }

    fn storage(&self, len: usize, label: &str) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: (len.max(1) as u64).saturating_mul(4),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }
}

/// A plan with its buffers and bind groups.
struct Built {
    dims: Vec<usize>,
    best: bool,
    plan: Plan,
    slots: Vec<wgpu::Buffer>,
    disps: Vec<(Arc<wgpu::ComputePipeline>, wgpu::BindGroup, [u32; 3], f64)>,
    staging: wgpu::Buffer,
    out_len: usize,
    bytes: u64,
}

/// One network on the GPU.
pub struct Net {
    gpu: Arc<Gpu>,
    p: Prepared,
    consts: Mutex<HashMap<ir::Cid, wgpu::Buffer>>,
    /// Most recent first. The lock also serialises runs: plans own their buffers.
    plans: Mutex<Vec<Built>>,
    /// Buffer bytes each (input size, best) needs, worked out without the device.
    costs: Mutex<HashMap<(Vec<usize>, bool), u64>>,
    split_budget: u64,
}

impl Net {
    pub fn new(gpu: Arc<Gpu>, p: Prepared) -> Net {
        Net {
            gpu,
            p,
            consts: Mutex::new(HashMap::new()),
            plans: Mutex::new(Vec::new()),
            costs: Mutex::new(HashMap::new()),
            split_budget: SPLIT_BUDGET,
        }
    }

    /// The same network with another split budget (tests).
    #[cfg(test)]
    pub fn with_split_budget(mut self, bytes: u64) -> Net {
        self.split_budget = bytes;
        self
    }

    fn cost(&self, dims: &[usize], best: bool) -> Result<u64, OcrError> {
        let key = (dims.to_vec(), best);
        if let Some(&c) = self.costs.lock().unwrap_or_else(|e| e.into_inner()).get(&key) {
            return Ok(c);
        }
        let plan = plan::plan(&self.p, dims, best, self.gpu.limits).map_err(OcrError::Recognize)?;
        let c = plan_bytes(&plan);
        self.costs.lock().unwrap_or_else(|e| e.into_inner()).insert(key, c);
        Ok(c)
    }

    /// [`Net::execute`], reading a batch too large for [`SPLIT_BUDGET`] in two halves (each
    /// split again while needed) and joining their outputs along the batch.
    fn run_parts(&self, dims: [usize; 4], data: &[f32], best: bool) -> Result<(Vec<usize>, Vec<f32>), OcrError> {
        let n = dims[0];
        if n < 2 || self.cost(&dims, best)? <= self.split_budget {
            return self.execute(&dims, data, best);
        }
        let per = data.len() / n;
        if per * n != data.len() {
            return Err(gpu_err(format!("the input has {} values for {dims:?}", data.len())));
        }
        let first = n / 2;
        let (a, b) = data.split_at(first * per);
        let (sa, mut va) = self.run_parts([first, dims[1], dims[2], dims[3]], a, best)?;
        let (sb, vb) = self.run_parts([n - first, dims[1], dims[2], dims[3]], b, best)?;
        match (sa.split_first(), sb.split_first()) {
            (Some((na, ra)), Some((nb, rb))) if ra == rb && *na == first && *nb == n - first => {
                va.extend_from_slice(&vb);
                let mut shape = vec![n];
                shape.extend_from_slice(ra);
                Ok((shape, va))
            }
            _ => Err(gpu_err(format!("the parts of a batch gave shapes {sa:?} and {sb:?}"))),
        }
    }

    fn build(&self, dims: &[usize], best: bool) -> Result<Built, OcrError> {
        let plan = plan::plan(&self.p, dims, best, self.gpu.limits).map_err(OcrError::Recognize)?;
        if plan_bytes(&plan) > MAX_PLAN_BYTES {
            return Err(gpu_err(format!("reading {dims:?} needs {} MB of buffers, more than {} MB", plan_bytes(&plan) >> 20, MAX_PLAN_BYTES >> 20)));
        }
        let g = &self.gpu;
        for s in &plan.steps {
            if s.binds.len() as u32 > g.storage_per_stage {
                return Err(gpu_err(format!("{} binds {} buffers; the device allows {}", s.label, s.binds.len(), g.storage_per_stage)));
            }
        }
        let pipelines = plan.steps.iter().map(|s| g.pipeline(&s.shader)).collect::<Result<Vec<_>, _>>()?;
        scoped(&g.device, || {
            {
                let mut consts = self.consts.lock().unwrap_or_else(|e| e.into_inner());
                for &c in &plan.consts {
                    if consts.contains_key(&c) {
                        continue;
                    }
                    let mut v = ir::floats(&self.p, c);
                    if v.is_empty() {
                        v.push(0.0);
                    }
                    if (v.len() as u64).saturating_mul(4) > g.limits.max_binding {
                        return Err(gpu_err("a weight is larger than the device binds"));
                    }
                    let b = g.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("weight"),
                        contents: &bytes(&v),
                        usage: wgpu::BufferUsages::STORAGE,
                    });
                    consts.insert(c, b);
                }
            }
            let slots: Vec<wgpu::Buffer> = plan.slots.iter().map(|&n| g.storage(n, "intermediate")).collect();
            let consts = self.consts.lock().unwrap_or_else(|e| e.into_inner());
            let mut disps = Vec::with_capacity(plan.steps.len());
            for (s, pipe) in plan.steps.iter().zip(pipelines) {
                let mut words = s.params.clone();
                if words.len() > wgsl::PARAMS {
                    return Err(gpu_err(format!("{} has {} parameters", s.label, words.len())));
                }
                words.resize(wgsl::PARAMS, 0);
                let words: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
                let uniform = g.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("params"),
                    contents: &words,
                    usage: wgpu::BufferUsages::UNIFORM,
                });
                let mut entries = Vec::with_capacity(s.binds.len() + 1);
                for (i, b) in s.binds.iter().enumerate() {
                    let buf = match b {
                        Bind::Val(slot) => slots.get(*slot),
                        Bind::Const(c) => consts.get(c),
                    }
                    .ok_or_else(|| gpu_err(format!("{}: a buffer is missing", s.label)))?;
                    entries.push(wgpu::BindGroupEntry { binding: i as u32, resource: buf.as_entire_binding() });
                }
                entries.push(wgpu::BindGroupEntry { binding: s.binds.len() as u32, resource: uniform.as_entire_binding() });
                let layout = pipe.get_bind_group_layout(0);
                let bind = g.device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some(&s.label), layout: &layout, entries: &entries });
                disps.push((pipe, bind, s.groups, s.work));
            }
            let out_len = match plan.best_rows {
                Some(rows) => rows * 2,
                None => crate::eval::count(&plan.output_dims).map_err(OcrError::Recognize)?,
            };
            let staging = g.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("readback"),
                size: (out_len.max(1) as u64).saturating_mul(4),
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let bytes = plan_bytes(&plan);
            Ok(Built { dims: dims.to_vec(), best, plan, slots, disps, staging, out_len, bytes })
        })
    }

    /// Run on `data` (`dims`), giving the output (or, with `best`, (class, probability) rows
    /// as pairs of floats).
    fn execute(&self, dims: &[usize], data: &[f32], best: bool) -> Result<(Vec<usize>, Vec<f32>), OcrError> {
        let mut plans = self.plans.lock().unwrap_or_else(|e| e.into_inner());
        let at = plans.iter().position(|b| b.dims == dims && b.best == best);
        let built = match at {
            Some(i) => plans.remove(i),
            None => self.build(dims, best)?,
        };
        plans.insert(0, built);
        // Keep the newest, and others while they fit the budget.
        let mut used = 0u64;
        let mut keep = 0;
        for (i, b) in plans.iter().enumerate() {
            used = used.saturating_add(b.bytes);
            if i == 0 || used <= PLAN_BUDGET {
                keep = i + 1;
            } else {
                break;
            }
        }
        plans.truncate(keep);
        let b = plans.first().ok_or_else(|| gpu_err("no plan"))?;
        if data.len() != b.plan.input_len {
            return Err(gpu_err(format!("the input has {} values for {dims:?}", data.len())));
        }
        let g = &self.gpu;
        let input = b.plan.slot_of.get(&b.plan.input).and_then(|&s| b.slots.get(s)).ok_or_else(|| gpu_err("no input buffer"))?;
        let output = b.plan.slot_of.get(&b.plan.output).and_then(|&s| b.slots.get(s)).ok_or_else(|| gpu_err("no output buffer"))?;
        scoped(&g.device, || {
            g.queue.write_buffer(input, 0, &bytes(data));
            let mut i = 0;
            while i < b.disps.len() {
                let budget = *g.pace.lock().unwrap_or_else(|e| e.into_inner());
                let mut enc = g.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("pdfcraft-ocr") });
                let mut work = 0.0;
                {
                    let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("pdfcraft-ocr"), timestamp_writes: None });
                    while let Some((pipe, bind, groups, w)) = b.disps.get(i) {
                        if work > 0.0 && work + w > budget {
                            break;
                        }
                        pass.set_pipeline(pipe);
                        pass.set_bind_group(0, bind, &[]);
                        pass.dispatch_workgroups(groups[0], groups[1], groups[2]);
                        work += w;
                        i += 1;
                    }
                }
                let start = Instant::now();
                let index = g.queue.submit([enc.finish()]);
                g.wait(index)?;
                let took = start.elapsed().as_secs_f64();
                if took > 1e-4 && work > 0.0 {
                    let mut pace = g.pace.lock().unwrap_or_else(|e| e.into_inner());
                    let want = work / took * SLICE.as_secs_f64();
                    *pace = (*pace * 0.5 + want * 0.5).clamp(1.0e7, 1.0e12);
                }
            }
            let mut enc = g.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("pdfcraft-ocr readback") });
            enc.copy_buffer_to_buffer(output, 0, &b.staging, 0, (b.out_len.max(1) as u64).saturating_mul(4));
            let index = g.queue.submit([enc.finish()]);
            let slice = b.staging.slice(..);
            let done: Arc<Mutex<Option<Result<(), wgpu::BufferAsyncError>>>> = Arc::new(Mutex::new(None));
            let tell = done.clone();
            slice.map_async(wgpu::MapMode::Read, move |r| {
                *tell.lock().unwrap_or_else(|e| e.into_inner()) = Some(r);
            });
            g.wait(index)?;
            let mapped = done.lock().unwrap_or_else(|e| e.into_inner()).take();
            match mapped {
                Some(Ok(())) => {}
                Some(Err(e)) => return Err(gpu_err(e)),
                None => return Err(gpu_err("the result was not mapped")),
            }
            let out: Vec<f32> = {
                let view = slice.get_mapped_range().map_err(|e| gpu_err(format!("{e:?}")))?;
                view.as_chunks::<4>().0.iter().take(b.out_len).map(|c| f32::from_le_bytes(*c)).collect()
            };
            b.staging.unmap();
            Ok((b.plan.output_dims.clone(), out))
        })
    }

    pub fn has_best(&self) -> bool {
        self.p.best.is_some()
    }
}

impl Network for Net {
    fn run(&self, shape: [usize; 4], data: Vec<f32>) -> Result<(Vec<usize>, Vec<f32>), OcrError> {
        guarded("the run", || self.run_parts(shape, &data, false))
    }

    fn run_best(&self, shape: [usize; 4], data: Vec<f32>) -> Result<(Vec<usize>, Vec<Best>), OcrError> {
        guarded("the run", || self.best_inner(shape, &data))
    }
}

impl Net {
    fn best_inner(&self, shape: [usize; 4], data: &[f32]) -> Result<(Vec<usize>, Vec<Best>), OcrError> {
        if !self.has_best() {
            let (s, d) = self.run_parts(shape, data, false)?;
            let best = pdfcraft_ocr::ppocr::best_of(&s, &d)?;
            return Ok((s, best));
        }
        let (dims, pairs) = self.run_parts(shape, data, true)?;
        let best = pairs
            .as_chunks::<2>()
            .0
            .iter()
            .map(|p| match p {
                [c, prob] if c.is_finite() && *c >= 0.0 => Ok((*c as usize, *prob)),
                _ => Err(gpu_err("a class that is not a number")),
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok((dims, best))
    }
}
