//! A prepared network planned for one input size: every value's shape, the shape arithmetic
//! worked out, and the list of dispatches (which kernel, which buffers, its parameters and grid),
//! with the intermediate buffers shared by values whose lifetimes do not overlap.
//!
//! Like [`crate::ir`], nothing here touches a GPU.

use std::collections::HashMap;

use crate::eval::{self, Cpu};
use crate::ir::{self, Act, Bin, Kind, Op, Pads, Prepared, Un, Vid, ViewKind};

/// The most dimensions the generic kernels index.
pub const MAX_RANK: usize = 6;
/// Threads per workgroup of the one-dimensional kernels.
pub const WG: usize = 256;
/// The tile of the matrix kernels (rows and columns).
pub const TILE: usize = 64;

/// Which shader a dispatch runs; everything not here is a uniform parameter.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Shader {
    /// Implicit-GEMM convolution: kernel size, stride, dilation, padding before (top, left), the
    /// upsampling of each source.
    ConvGemm {
        k: [usize; 2],
        stride: [usize; 2],
        dil: [usize; 2],
        pad: [usize; 2],
        ups: Vec<[usize; 2]>,
        act: Act,
        residual: bool,
        head: Option<bool>,
    },
    /// `A × B` (+ column bias), batched along the grid's z.
    MatMul {
        bias: bool,
        act: Act,
        residual: bool,
    },
    /// The recogniser's last layer: per row and tile of columns, the max, sum of exponentials
    /// and argmax of `A × B + bias`.
    BestPartial {
        bias: bool,
    },
    BestMerge,
    /// Grouped / depthwise convolution, one thread per output.
    DirectConv {
        k: [usize; 2],
        act: Act,
        residual: bool,
    },
    ConvT {
        k: [usize; 2],
        act: Act,
    },
    Pool {
        max: bool,
        include_pad: bool,
    },
    Resize,
    Copy,
    Binary(Bin),
    Unary(Un),
    Affine(Act),
    ReduceMean,
    Softmax,
    /// out[i] = in[offset + Σ coordᵢ · strideᵢ] (transpose, slice).
    Gather,
}

/// A buffer a dispatch binds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Bind {
    /// An intermediate buffer (after [`allocate`], a slot index).
    Val(Vid),
    Const(ir::Cid),
}

#[derive(Clone, Debug)]
pub struct Step {
    pub label: String,
    pub shader: Shader,
    /// Binding 0 is written; the rest are read.
    pub binds: Vec<Bind>,
    pub params: Vec<u32>,
    pub groups: [u32; 3],
    /// Multiply-adds (or element operations), to pace the submissions.
    pub work: f64,
}

/// One input size's plan.
#[derive(Clone, Debug)]
pub struct Plan {
    pub steps: Vec<Step>,
    /// Elements of each buffer slot.
    pub slots: Vec<usize>,
    /// Value (or scratch) → its slot.
    pub slot_of: HashMap<Vid, usize>,
    pub input: Vid,
    pub input_len: usize,
    pub output: Vid,
    pub output_dims: Vec<usize>,
    /// With `best`: rows of (class, probability) instead of the output tensor.
    pub best_rows: Option<usize>,
    pub consts: Vec<ir::Cid>,
}

/// What the device allows (the parts plans are checked against).
#[derive(Clone, Copy, Debug)]
pub struct Limits {
    pub max_binding: u64,
    pub max_buffer: u64,
    pub max_groups: u32,
}

fn err<T>(why: impl Into<String>) -> Result<T, String> {
    Err(why.into())
}

fn u32_of(v: usize, what: &str) -> Result<u32, String> {
    u32::try_from(v).map_err(|_| format!("{what} {v} is too large for the GPU kernels"))
}

fn i32_bits(v: i64, what: &str) -> Result<u32, String> {
    i32::try_from(v).map(|x| x as u32).map_err(|_| format!("{what} {v} is too large for the GPU kernels"))
}

struct Planner<'p> {
    p: &'p Prepared,
    lim: Limits,
    shapes: HashMap<Vid, Vec<usize>>,
    meta: HashMap<Vid, Cpu>,
    /// A view → the value that owns its buffer.
    base: HashMap<Vid, Vid>,
    steps: Vec<Step>,
    scratch: Vid,
    scratch_len: HashMap<Vid, usize>,
}

/// The product of `f`, or an error naming `what` when it overflows.
fn mul(f: &[usize], what: &str) -> Result<usize, String> {
    f.iter().try_fold(1usize, |a, &v| a.checked_mul(v)).ok_or_else(|| format!("the {what} overflows"))
}

/// The product of `f` as a work estimate (never overflows).
fn work(f: &[usize]) -> f64 {
    f.iter().map(|&v| v as f64).product()
}

/// Output size of a window along one axis.
fn window(input: usize, k: usize, stride: usize, dil: usize, before: usize, after: usize, ceil: bool) -> Result<usize, String> {
    if stride == 0 || dil == 0 {
        return err("a zero stride or dilation");
    }
    let eff = k.checked_sub(1).and_then(|k1| k1.checked_mul(dil)).and_then(|v| v.checked_add(1)).ok_or("kernel size")?;
    let span = input.checked_add(before).and_then(|v| v.checked_add(after)).ok_or("size")?;
    if span < eff {
        return err(format!("a window of {eff} does not fit {input} (+{before}+{after})"));
    }
    let room = span - eff;
    let mut out = if ceil { room.div_ceil(stride) + 1 } else { room / stride + 1 };
    // A last window must start inside the input or its leading padding.
    if ceil && (out - 1).saturating_mul(stride) >= input.saturating_add(before) {
        out -= 1;
    }
    Ok(out)
}

/// SAME padding: (before, after) for one axis.
/// Saturating: a size it gets wrong is refused by [`window`] and [`Planner::set`].
fn same(input: usize, k: usize, stride: usize, dil: usize, upper: bool) -> (usize, usize) {
    let stride = stride.max(1);
    let out = input.div_ceil(stride);
    let eff = k.saturating_sub(1).saturating_mul(dil).saturating_add(1);
    let total = out.saturating_sub(1).saturating_mul(stride).saturating_add(eff).saturating_sub(input);
    let small = total / 2;
    if upper { (small, total - small) } else { (total - small, small) }
}

fn pads_for(p: Pads, ih: usize, iw: usize, k: [usize; 2], s: [usize; 2], d: [usize; 2]) -> [usize; 4] {
    match p {
        Pads::Explicit(v) => v,
        Pads::Same { upper } => {
            let (t, b) = same(ih, k[0], s[0], d[0], upper);
            let (l, r) = same(iw, k[1], s[1], d[1], upper);
            [t, l, b, r]
        }
    }
}

/// A one-dimensional grid of `n` threads.
fn grid(n: usize, lim: Limits) -> Result<[u32; 3], String> {
    let groups = n.div_ceil(WG).max(1);
    let max = lim.max_groups.max(1) as usize;
    let x = groups.min(max);
    let y = groups.div_ceil(x);
    if y > max {
        return err(format!("{n} threads are more than the GPU dispatches at once"));
    }
    Ok([u32_of(x, "groups")?, u32_of(y, "groups")?, 1])
}

impl Planner<'_> {
    fn shape(&self, v: Vid) -> Result<&Vec<usize>, String> {
        if let Some(c) = self.p.konst(v) {
            return Ok(&c.dims);
        }
        self.shapes.get(&v).ok_or_else(|| format!("the shape of {} is not known", self.p.name(v)))
    }

    fn known(&self, v: Vid) -> Result<Cpu, String> {
        if let Some(c) = self.p.konst(v) {
            return Ok(c.clone());
        }
        self.meta.get(&v).cloned().ok_or_else(|| format!("{} is not known before the run", self.p.name(v)))
    }

    fn root(&self, v: Vid) -> Vid {
        let mut v = v;
        for _ in 0..self.base.len() + 1 {
            match self.base.get(&v) {
                Some(&b) => v = b,
                None => break,
            }
        }
        v
    }

    fn bind(&self, v: Vid) -> Bind {
        match self.p.kind(v) {
            Kind::Const(c) => Bind::Const(c),
            _ => Bind::Val(self.root(v)),
        }
    }

    fn set(&mut self, v: Vid, dims: Vec<usize>) -> Result<usize, String> {
        let n = eval::count(&dims)?;
        let bytes = (n as u64).saturating_mul(4);
        if bytes > self.lim.max_binding || bytes > self.lim.max_buffer {
            return err(format!(
                "{} needs {} MB, more than the GPU binds at once ({} MB)",
                self.p.name(v),
                bytes >> 20,
                self.lim.max_binding.min(self.lim.max_buffer) >> 20
            ));
        }
        if n > u32::MAX as usize / 2 {
            return err(format!("{} is too large", self.p.name(v)));
        }
        self.shapes.insert(v, dims);
        Ok(n)
    }

    fn scratch(&mut self, len: usize) -> Result<Vid, String> {
        let bytes = (len as u64).saturating_mul(4);
        if bytes > self.lim.max_binding {
            return err("the recogniser's scratch buffer is too large");
        }
        self.scratch += 1;
        self.scratch_len.insert(self.scratch, len);
        Ok(self.scratch)
    }

    fn step(&mut self, label: &str, shader: Shader, binds: Vec<Bind>, params: Vec<u32>, groups: [u32; 3], work: f64) {
        self.steps.push(Step { label: label.to_string(), shader, binds, params, groups, work });
    }

    fn dims4(&self, v: Vid) -> Result<[usize; 4], String> {
        match self.shape(v)?.as_slice() {
            [a, b, c, d] => Ok([*a, *b, *c, *d]),
            other => err(format!("{} has shape {other:?}, not four dimensions", self.p.name(v))),
        }
    }

    fn conv(&mut self, c: &ir::Conv, head: Option<(ir::Cid, f32, Option<Vid>, f32, f32, Vid)>) -> Result<(), String> {
        let mut batch = None;
        let (mut ih, mut iw, mut cin) = (0usize, 0usize, 0usize);
        let mut src_params = Vec::new();
        for (i, (v, up)) in c.srcs.iter().enumerate() {
            let [b, ch, h, w] = self.dims4(*v)?;
            let (vh, vw) = (h.checked_mul(up[0]).ok_or("size")?, w.checked_mul(up[1]).ok_or("size")?);
            if i == 0 {
                (ih, iw) = (vh, vw);
                batch = Some(b);
            } else if (vh, vw) != (ih, iw) || batch != Some(b) {
                return err(format!("{}: its inputs have different sizes", c.name));
            }
            cin = cin.checked_add(ch).ok_or("channels")?;
            src_params.extend([u32_of(cin, "channels")?, u32_of(h, "height")?, u32_of(w, "width")?]);
        }
        let b = batch.ok_or_else(|| format!("{} has no input", c.name))?;
        if Some(cin) != c.cin_g.checked_mul(c.group) {
            return err(format!("{}: {cin} input channels for weights of {} × {}", c.name, c.cin_g, c.group));
        }
        let pads = pads_for(c.pads, ih, iw, c.k, c.stride, c.dil);
        let oh = window(ih, c.k[0], c.stride[0], c.dil[0], pads[0], pads[2], false)?;
        let ow = window(iw, c.k[1], c.stride[1], c.dil[1], pads[1], pads[3], false)?;
        let npix = oh.checked_mul(ow).ok_or("size")?;
        let kk = mul(&[c.cin_g, c.k[0], c.k[1]], "convolution depth")?;
        let work = work(&[b, c.cout, npix, kk]);
        let (y, out_c) = match head {
            Some((.., y)) => (y, 1),
            None => (c.y, c.cout),
        };
        self.set(y, vec![b, out_c, oh, ow])?;
        if let Some(r) = c.residual
            && self.shape(r)? != &vec![b, c.cout, oh, ow]
        {
            return err(format!("{}: the residual's shape differs", c.name));
        }
        if c.group == 1 {
            if head.is_some() && c.cout > TILE {
                return err(format!("{}: the fused head needs at most {TILE} channels", c.name));
            }
            let groups = [u32_of(npix.div_ceil(TILE), "groups")?, u32_of(c.cout.div_ceil(TILE), "groups")?, u32_of(b, "batch")?];
            if groups.iter().any(|&g| g > self.lim.max_groups) {
                return err(format!("{}: too large to dispatch", c.name));
            }
            let mut params = vec![
                u32_of(c.cout, "channels")?,
                u32_of(npix, "pixels")?,
                u32_of(kk, "depth")?,
                u32_of(ih, "height")?,
                u32_of(iw, "width")?,
                u32_of(ow, "width")?,
            ];
            let mut binds = vec![Bind::Val(self.root(y)), Bind::Const(c.w), Bind::Const(c.st)];
            binds.extend(c.srcs.iter().map(|(v, _)| self.bind(*v)));
            params.extend(src_params);
            let head_shader = match head {
                Some((w2, b2, other, scale, shift, _)) => {
                    binds.push(Bind::Const(w2));
                    if let Some(o) = other {
                        if self.shape(o)? != &vec![b, 1, oh, ow] {
                            return err(format!("{}: the other branch's shape differs", c.name));
                        }
                        binds.push(self.bind(o));
                    }
                    params.extend([b2.to_bits(), scale.to_bits(), shift.to_bits()]);
                    Some(other.is_some())
                }
                None => {
                    if let Some(r) = c.residual {
                        binds.push(self.bind(r));
                    }
                    None
                }
            };
            if binds.len() > 8 {
                return err(format!("{}: {} buffers are more than a kernel binds", c.name, binds.len()));
            }
            let shader = Shader::ConvGemm {
                k: c.k,
                stride: c.stride,
                dil: c.dil,
                pad: [pads[0], pads[1]],
                ups: c.srcs.iter().map(|s| s.1).collect(),
                act: c.act,
                residual: c.residual.is_some() && head.is_none(),
                head: head_shader,
            };
            self.step(&c.name, shader, binds, params, groups, work);
            return Ok(());
        }
        if head.is_some() || c.srcs.len() != 1 || c.srcs.iter().any(|s| s.1 != [1, 1]) {
            return err(format!("{}: a grouped convolution reads one plain input", c.name));
        }
        let x = c.srcs.first().map(|s| s.0).ok_or("no input")?;
        let total = mul(&[b, c.cout, npix], "size")?;
        let per_group = c.cout.checked_div(c.group).ok_or("group 0")?;
        let params = [c.cout, oh, ow, cin, ih, iw, c.cin_g, per_group, c.stride[0], c.stride[1], c.dil[0], c.dil[1], pads[0], pads[1], total]
            .iter()
            .map(|&v| u32_of(v, "size"))
            .collect::<Result<Vec<_>, _>>()?;
        let mut binds = vec![Bind::Val(self.root(y)), self.bind(x), Bind::Const(c.w), Bind::Const(c.st)];
        if let Some(r) = c.residual {
            binds.push(self.bind(r));
        }
        let g = grid(total, self.lim)?;
        self.step(&c.name, Shader::DirectConv { k: c.k, act: c.act, residual: c.residual.is_some() }, binds, params, g, work);
        Ok(())
    }

    fn elementwise(&mut self, label: &str, shader: Shader, y: Vid, dims: Vec<usize>, binds: Vec<Bind>, extra: Vec<u32>) -> Result<(), String> {
        let total = self.set(y, dims)?;
        let mut params = vec![u32_of(total, "size")?];
        params.extend(extra);
        let mut all = vec![Bind::Val(self.root(y))];
        all.extend(binds);
        let g = grid(total, self.lim)?;
        self.step(label, shader, all, params, g, total as f64);
        Ok(())
    }

    /// out = in gathered: dims of the output, strides into the input, offset.
    fn gather(&mut self, label: &str, x: Vid, y: Vid, dims: Vec<usize>, strides: Vec<i64>, offset: i64) -> Result<(), String> {
        if dims.len() > MAX_RANK {
            return err(format!("{label}: {} dimensions", dims.len()));
        }
        let mut extra = vec![u32_of(dims.len(), "rank")?];
        let mut d6 = dims.clone();
        d6.resize(MAX_RANK, 1);
        let mut s6 = strides;
        s6.resize(MAX_RANK, 0);
        for d in &d6 {
            extra.push(u32_of(*d, "size")?);
        }
        for s in &s6 {
            extra.push(i32_bits(*s, "stride")?);
        }
        extra.push(i32_bits(offset, "offset")?);
        let xb = self.bind(x);
        self.elementwise(label, Shader::Gather, y, dims, vec![xb], extra)
    }

    fn op(&mut self, op: &Op) -> Result<(), String> {
        let label = self.p.name(op.output()).to_string();
        match op {
            Op::Conv(c) => self.conv(c, None)?,
            Op::Head { conv, w2, b2, other, scale, shift, y } => self.conv(conv, Some((*w2, *b2, *other, *scale, *shift, *y)))?,
            Op::ConvT { name, x, w, st, cin, cout, k, stride, dil, pads, out_pad, act, y } => {
                let [b, c, ih, iw] = self.dims4(*x)?;
                if c != *cin {
                    return err(format!("{name}: {c} channels for {cin}"));
                }
                // (i - 1) × stride + (k - 1) × dilation + 1 + output padding - both pads.
                let side = |i: usize, k: usize, s: usize, d: usize, op: usize, p0: usize, p1: usize| -> Option<usize> {
                    if s == 0 || d == 0 {
                        return None;
                    }
                    let eff = k.checked_sub(1)?.checked_mul(d)?.checked_add(1)?;
                    i.checked_sub(1)?.checked_mul(s)?.checked_add(eff)?.checked_add(op)?.checked_sub(p0.checked_add(p1)?)
                };
                let oh = side(ih, k[0], stride[0], dil[0], out_pad[0], pads[0], pads[2])
                    .filter(|&v| v > 0)
                    .ok_or_else(|| format!("{name}: no output size"))?;
                let ow = side(iw, k[1], stride[1], dil[1], out_pad[1], pads[1], pads[3])
                    .filter(|&v| v > 0)
                    .ok_or_else(|| format!("{name}: no output size"))?;
                let total = self.set(*y, vec![b, *cout, oh, ow])?;
                let params = [total, *cin, *cout, ih, iw, oh, ow, stride[0], stride[1], dil[0], dil[1], pads[0], pads[1]]
                    .iter()
                    .map(|&v| u32_of(v, "size"))
                    .collect::<Result<Vec<_>, _>>()?;
                let binds = vec![Bind::Val(self.root(*y)), self.bind(*x), Bind::Const(*w), Bind::Const(*st)];
                let g = grid(total, self.lim)?;
                self.step(name, Shader::ConvT { k: *k, act: *act }, binds, params, g, work(&[total, *cin, k[0], k[1]]));
            }
            Op::Pool { x, max, k, stride, pads, ceil, include_pad, y } => {
                let [b, c, ih, iw] = self.dims4(*x)?;
                let pd = pads_for(*pads, ih, iw, *k, *stride, [1, 1]);
                let ceil = *ceil && matches!(pads, Pads::Explicit(_));
                let oh = window(ih, k[0], stride[0], 1, pd[0], pd[2], ceil)?;
                let ow = window(iw, k[1], stride[1], 1, pd[1], pd[3], ceil)?;
                let extra = [c, ih, iw, oh, ow, k[0], k[1], stride[0], stride[1], pd[0], pd[1]].iter().map(|&v| u32_of(v, "size")).collect::<Result<
                    Vec<_>,
                    _,
                >>(
                )?;
                let xb = self.bind(*x);
                self.elementwise(&label, Shader::Pool { max: *max, include_pad: *include_pad }, *y, vec![b, c, oh, ow], vec![xb], extra)?;
            }
            Op::Resize { x, up, y } => {
                let [b, c, ih, iw] = self.dims4(*x)?;
                let (oh, ow) = (mul(&[ih, up[0]], "size")?, mul(&[iw, up[1]], "size")?);
                let extra = [ih, iw, oh, ow, up[0], up[1]].iter().map(|&v| u32_of(v, "size")).collect::<Result<Vec<_>, _>>()?;
                let xb = self.bind(*x);
                self.elementwise(&label, Shader::Resize, *y, vec![b, c, oh, ow], vec![xb], extra)?;
            }
            Op::Concat { xs, axis, y } => {
                let parts: Vec<Vec<usize>> = xs.iter().map(|v| self.shape(*v).cloned()).collect::<Result<_, _>>()?;
                let first = parts.first().ok_or("Concat of nothing")?;
                let ax = eval::axis(*axis, first.len())?;
                let mut dims = first.clone();
                let mut along = 0usize;
                for p in &parts {
                    if p.len() != first.len() || p.iter().enumerate().any(|(i, d)| i != ax && first.get(i) != Some(d)) {
                        return err(format!("{label}: Concat of {first:?} and {p:?}"));
                    }
                    along = along.checked_add(p.get(ax).copied().unwrap_or(0)).ok_or("Concat size")?;
                }
                if let Some(d) = dims.get_mut(ax) {
                    *d = along;
                }
                self.set(*y, dims)?;
                let inner = eval::count(first.get(ax + 1..).unwrap_or(&[]))?;
                let mut off = 0usize;
                for (v, p) in xs.iter().zip(&parts) {
                    let ci = p.get(ax).copied().unwrap_or(0);
                    let n = eval::count(p)?;
                    if n == 0 {
                        continue;
                    }
                    let params = vec![
                        u32_of(n, "size")?,
                        u32_of(mul(&[ci, inner], "size")?, "size")?,
                        u32_of(mul(&[along, inner], "size")?, "size")?,
                        u32_of(mul(&[off, inner], "size")?, "size")?,
                    ];
                    let binds = vec![Bind::Val(self.root(*y)), self.bind(*v)];
                    let g = grid(n, self.lim)?;
                    self.step(&label, Shader::Copy, binds, params, g, n as f64);
                    off = off.saturating_add(ci);
                }
            }
            Op::Binary { op, a, b, y } => {
                let (sa, sb) = (self.shape(*a)?.clone(), self.shape(*b)?.clone());
                let out = eval::broadcast(&sa, &sb)?;
                if out.len() > MAX_RANK {
                    return err(format!("{label}: {} dimensions", out.len()));
                }
                let mut extra = vec![u32_of(out.len(), "rank")?];
                let mut d6 = out.clone();
                d6.resize(MAX_RANK, 1);
                let mut a6 = eval::broadcast_strides(&sa, &out);
                a6.resize(MAX_RANK, 0);
                let mut b6 = eval::broadcast_strides(&sb, &out);
                b6.resize(MAX_RANK, 0);
                for v in d6.iter().chain(&a6).chain(&b6) {
                    extra.push(u32_of(*v, "size")?);
                }
                let binds = vec![self.bind(*a), self.bind(*b)];
                self.elementwise(&label, Shader::Binary(*op), *y, out, binds, extra)?;
            }
            Op::Unary { op, x, y } => {
                let dims = self.shape(*x)?.clone();
                let xb = self.bind(*x);
                self.elementwise(&label, Shader::Unary(*op), *y, dims, vec![xb], vec![])?;
            }
            Op::Affine { x, st, act, y } => {
                let dims = self.shape(*x)?.clone();
                let c = dims.get(1).copied().ok_or_else(|| format!("{label}: no channel axis"))?;
                let inner = eval::count(dims.get(2..).unwrap_or(&[]))?;
                if self.p.consts.get(*st).map(Cpu::len) != c.checked_mul(2) {
                    return err(format!("{label}: {c} channels"));
                }
                let xb = self.bind(*x);
                self.elementwise(
                    &label,
                    Shader::Affine(*act),
                    *y,
                    dims,
                    vec![xb, Bind::Const(*st)],
                    vec![u32_of(c, "channels")?, u32_of(inner, "size")?],
                )?;
            }
            Op::MatMul { a, b, bias, act, residual, y } => self.matmul(&label, *a, *b, *bias, *act, *residual, *y)?,
            Op::ReduceMean { x, axes, keep, y } => {
                let dims = self.shape(*x)?.clone();
                let last = dims.len().checked_sub(1).ok_or("ReduceMean of a scalar")?;
                let ax: Vec<usize> = axes.iter().map(|&a| eval::axis(a, dims.len())).collect::<Result<_, _>>()?;
                if ax != [last] {
                    return err(format!("{label}: only the last axis is reduced"));
                }
                let cols = dims.get(last).copied().unwrap_or(1);
                let rows = eval::count(dims.get(..last).unwrap_or(&[]))?;
                let mut out = dims.get(..last).unwrap_or(&[]).to_vec();
                if *keep {
                    out.push(1);
                }
                self.set(*y, out)?;
                let binds = vec![Bind::Val(self.root(*y)), self.bind(*x)];
                let g = grid(rows, self.lim)?;
                self.step(&label, Shader::ReduceMean, binds, vec![u32_of(rows, "rows")?, u32_of(cols, "columns")?], g, work(&[rows, cols]));
            }
            Op::Softmax { x, axis, y } => {
                let dims = self.shape(*x)?.clone();
                let ax = eval::axis(*axis, dims.len())?;
                let rows = eval::count(dims.get(..ax).unwrap_or(&[]))?;
                let cols = eval::count(dims.get(ax..).unwrap_or(&[]))?;
                self.set(*y, dims)?;
                let binds = vec![Bind::Val(self.root(*y)), self.bind(*x)];
                let max = self.lim.max_groups.max(1) as usize;
                let gx = rows.min(max).max(1);
                let gy = rows.div_ceil(gx).max(1);
                if gy > max {
                    return err(format!("{label}: too many rows"));
                }
                self.step(
                    &label,
                    Shader::Softmax,
                    binds,
                    vec![u32_of(rows, "rows")?, u32_of(cols, "columns")?],
                    [u32_of(gx, "g")?, u32_of(gy, "g")?, 1],
                    work(&[rows, cols, 3]),
                );
            }
            Op::Transpose { x, perm, y } => {
                let dims = self.shape(*x)?.clone();
                let perm: Vec<usize> = if perm.is_empty() {
                    (0..dims.len()).rev().collect()
                } else {
                    perm.iter().map(|&p| eval::axis(p, dims.len())).collect::<Result<_, _>>()?
                };
                let mut seen = perm.clone();
                seen.sort_unstable();
                if seen != (0..dims.len()).collect::<Vec<_>>() {
                    return err(format!("{label}: perm {perm:?}"));
                }
                let st = eval::strides(&dims);
                let out: Vec<usize> = perm.iter().map(|&p| dims.get(p).copied().unwrap_or(1)).collect();
                let strides: Vec<i64> = perm.iter().map(|&p| st.get(p).copied().unwrap_or(0) as i64).collect();
                self.gather(&label, *x, *y, out, strides, 0)?;
            }
            Op::View { x, how, y } => {
                let dims = self.shape(*x)?.clone();
                let out = match how {
                    ViewKind::Identity => dims,
                    ViewKind::Reshape(s) => eval::reshape_dims(&dims, &self.known(*s)?.as_ints()?)?,
                    ViewKind::Flatten(a) => {
                        let a = if *a == dims.len() as i64 { dims.len() } else { eval::axis(*a, dims.len())? };
                        vec![eval::count(dims.get(..a).unwrap_or(&[]))?, eval::count(dims.get(a..).unwrap_or(&[]))?]
                    }
                    ViewKind::Squeeze(axes) => eval::squeeze_dims(&dims, axes)?,
                    ViewKind::Unsqueeze(axes) => eval::unsqueeze_dims(&dims, axes)?,
                };
                self.shapes.insert(*y, out);
                let r = self.root(*x);
                self.base.insert(*y, r);
            }
            Op::Slice { x, starts, ends, axes, steps, y } => {
                let dims = self.shape(*x)?.clone();
                let (s, e) = (self.known(*starts)?.as_ints()?, self.known(*ends)?.as_ints()?);
                let ax = axes.map(|v| self.known(v).and_then(|t| t.as_ints())).transpose()?;
                let stp = steps.map(|v| self.known(v).and_then(|t| t.as_ints())).transpose()?;
                let spans = eval::slice_spans(&dims, &s, &e, ax.as_deref(), stp.as_deref())?;
                let st = eval::strides(&dims);
                let out: Vec<usize> = spans.iter().map(|s| s.len).collect();
                let strides: Vec<i64> = spans
                    .iter()
                    .zip(&st)
                    .map(|(s, &t)| i64::try_from(t).ok().and_then(|t| s.step.checked_mul(t)).ok_or_else(|| format!("{label}: Slice step")))
                    .collect::<Result<_, _>>()?;
                let offset = spans.iter().zip(&st).try_fold(0i64, |a, (s, &t)| {
                    s.start
                        .checked_mul(t)
                        .and_then(|v| i64::try_from(v).ok())
                        .and_then(|v| a.checked_add(v))
                        .ok_or_else(|| format!("{label}: Slice start"))
                })?;
                self.gather(&label, *x, *y, out, strides, offset)?;
            }
            Op::Shape { x, y } => {
                let dims: Vec<i64> = self.shape(*x)?.iter().map(|&d| d as i64).collect();
                self.meta.insert(*y, Cpu::ints(vec![dims.len()], dims));
            }
            Op::Meta { node, ins, y } => {
                let held: Vec<Option<Cpu>> = ins.iter().map(|v| v.map(|v| self.known(v)).transpose()).collect::<Result<_, _>>()?;
                let refs: Vec<Option<&Cpu>> = held.iter().map(Option::as_ref).collect();
                let t = eval::eval(node, &refs).map_err(|e| format!("{}: {e}", node.name))?;
                self.meta.insert(*y, t);
            }
        }
        Ok(())
    }

    /// `a` [.., M, K] × `b` [K, N] (a weight: the leading dims fold into M) or [.., K, N] with
    /// the same leading dims.
    #[allow(clippy::too_many_arguments)]
    fn matmul(&mut self, label: &str, a: Vid, b: Vid, bias: Option<ir::Cid>, act: Act, residual: Option<Vid>, y: Vid) -> Result<(), String> {
        let (sa, sb) = (self.shape(a)?.clone(), self.shape(b)?.clone());
        let (m, k, n, batch, out, b_stride) = match (sa.as_slice(), sb.as_slice()) {
            ([lead @ .., m, k], [kb, n]) if k == kb && !lead.is_empty() || (k == kb && lead.is_empty()) => {
                let rows = eval::count(lead)?.checked_mul(*m).ok_or("size")?;
                let mut out = lead.to_vec();
                out.extend([*m, *n]);
                (rows, *k, *n, 1usize, out, 0usize)
            }
            ([la @ .., m, k], [lb @ .., kb, n]) if k == kb && la == lb => {
                let batch = eval::count(la)?;
                let mut out = la.to_vec();
                out.extend([*m, *n]);
                (*m, *k, *n, batch, out, mul(&[*k, *n], "size")?)
            }
            _ => return err(format!("{label}: MatMul of {sa:?} and {sb:?}")),
        };
        if let Some(r) = residual
            && self.shape(r)? != &out
        {
            return err(format!("{label}: the residual's shape differs"));
        }
        self.set(y, out)?;
        let groups = [u32_of(n.div_ceil(TILE), "g")?, u32_of(m.div_ceil(TILE), "g")?, u32_of(batch, "batch")?];
        if groups.iter().any(|&g| g > self.lim.max_groups) {
            return err(format!("{label}: too large to dispatch"));
        }
        let mut binds = vec![Bind::Val(self.root(y)), self.bind(a), self.bind(b)];
        if let Some(c) = bias {
            binds.push(Bind::Const(c));
        }
        if let Some(r) = residual {
            binds.push(self.bind(r));
        }
        let params =
            [m, n, k, mul(&[m, k], "size")?, b_stride, mul(&[m, n], "size")?].iter().map(|&v| u32_of(v, "size")).collect::<Result<Vec<_>, _>>()?;
        self.step(label, Shader::MatMul { bias: bias.is_some(), act, residual: residual.is_some() }, binds, params, groups, work(&[batch, m, n, k]));
        Ok(())
    }

    /// The recogniser's last layer as (class, probability) per row: `rows` of them.
    fn best(&mut self, best: &ir::Best) -> Result<(Vid, Vec<usize>, usize), String> {
        let sa = self.shape(best.a)?.clone();
        let w = self.p.consts.get(best.w).ok_or("weight")?;
        let [k, n] = match w.dims.as_slice() {
            [k, n] => [*k, *n],
            _ => return err("the last layer's weight is not a matrix"),
        };
        let (last, lead) = sa.split_last().ok_or("the last layer's input is a scalar")?;
        if *last != k {
            return err(format!("the last layer reads {sa:?} with weights {k}×{n}"));
        }
        let rows = eval::count(lead)?;
        let tiles = n.div_ceil(TILE);
        let partial = self.scratch(rows.checked_mul(tiles).and_then(|v| v.checked_mul(3)).ok_or("size")?)?;
        let out = self.scratch(rows.checked_mul(2).ok_or("size")?)?;
        let groups = [u32_of(tiles, "g")?, u32_of(rows.div_ceil(TILE), "g")?, 1];
        if groups.iter().any(|&g| g > self.lim.max_groups) {
            return err("the last layer is too large to dispatch");
        }
        let mut binds = vec![Bind::Val(partial), self.bind(best.a), Bind::Const(best.w)];
        if let Some(c) = best.bias {
            binds.push(Bind::Const(c));
        }
        let params = vec![u32_of(rows, "rows")?, u32_of(n, "classes")?, u32_of(k, "depth")?, u32_of(tiles, "tiles")?];
        self.step("best (partial)", Shader::BestPartial { bias: best.bias.is_some() }, binds, params.clone(), groups, work(&[rows, n, k]));
        let g = grid(rows, self.lim)?;
        self.step("best (merge)", Shader::BestMerge, vec![Bind::Val(out), Bind::Val(partial)], params, g, work(&[rows, tiles]));
        let mut dims = lead.to_vec();
        dims.push(n);
        Ok((out, dims, rows))
    }
}

/// Plan `p` for an input of `dims`; with `best`, the recogniser's tail gives (class,
/// probability) per row.
pub fn plan(p: &Prepared, dims: &[usize], best: bool, lim: Limits) -> Result<Plan, String> {
    let mut pl = Planner {
        p,
        lim,
        shapes: HashMap::new(),
        meta: HashMap::new(),
        base: HashMap::new(),
        steps: Vec::new(),
        scratch: p.names.len(),
        scratch_len: HashMap::new(),
    };
    let input_len = pl.set(p.input, dims.to_vec())?;
    let tail = if best { Some(p.best.as_ref().ok_or("this network does not end in a softmax over classes")?) } else { None };
    let ops = match tail {
        Some(b) => p.ops.get(..b.from).ok_or("the tail")?,
        None => &p.ops,
    };
    for op in ops {
        pl.op(op)?;
    }
    let (output, output_dims, best_rows) = match tail {
        Some(b) => {
            let (out, dims, rows) = pl.best(b)?;
            (out, dims, Some(rows))
        }
        None => {
            let o = pl.root(p.output);
            (o, pl.shape(p.output)?.clone(), None)
        }
    };
    let input = pl.root(p.input);
    let steps = std::mem::take(&mut pl.steps);
    let mut sizes: HashMap<Vid, usize> = pl.scratch_len.clone();
    for (v, d) in &pl.shapes {
        if !pl.base.contains_key(v) {
            sizes.insert(*v, eval::count(d)?);
        }
    }
    allocate(steps, &sizes, input, input_len, output, output_dims, best_rows)
}

/// Give every value written by a step a slot, sharing slots between values whose lifetimes
/// (first write to last read) do not overlap. The input and output keep theirs throughout.
fn allocate(
    mut steps: Vec<Step>,
    sizes: &HashMap<Vid, usize>,
    input: Vid,
    input_len: usize,
    output: Vid,
    output_dims: Vec<usize>,
    best_rows: Option<usize>,
) -> Result<Plan, String> {
    let mut first: HashMap<Vid, usize> = HashMap::new();
    let mut last: HashMap<Vid, usize> = HashMap::new();
    first.insert(input, 0);
    for (i, s) in steps.iter().enumerate() {
        for b in &s.binds {
            if let Bind::Val(v) = b {
                first.entry(*v).or_insert(i);
                last.insert(*v, i);
            }
        }
    }
    last.insert(input, last.get(&input).copied().unwrap_or(0));
    last.insert(output, usize::MAX);
    let mut order: Vec<Vid> = first.keys().copied().collect();
    order.sort_by_key(|v| (first.get(v).copied().unwrap_or(0), *v));
    let mut slots: Vec<usize> = Vec::new();
    let mut free_at: Vec<usize> = Vec::new();
    let mut slot_of = HashMap::new();
    for v in order {
        let need = sizes.get(&v).copied().ok_or("a value without a size")?.max(1);
        let start = first.get(&v).copied().unwrap_or(0);
        let end = last.get(&v).copied().unwrap_or(start);
        // The smallest free slot that holds it, else the largest free one grown, else a new one.
        let free: Vec<usize> = (0..slots.len()).filter(|&s| free_at.get(s).is_some_and(|&f| f < start)).collect();
        let fit = free.iter().copied().filter(|&s| slots.get(s).is_some_and(|&n| n >= need)).min_by_key(|&s| slots.get(s).copied().unwrap_or(0));
        let pick = fit.or_else(|| free.iter().copied().max_by_key(|&s| slots.get(s).copied().unwrap_or(0)));
        let s = match pick {
            Some(s) => {
                if let Some(n) = slots.get_mut(s) {
                    *n = (*n).max(need);
                }
                s
            }
            None => {
                slots.push(need);
                free_at.push(0);
                slots.len() - 1
            }
        };
        if let Some(f) = free_at.get_mut(s) {
            *f = end;
        }
        slot_of.insert(v, s);
    }
    for s in &mut steps {
        for b in &mut s.binds {
            if let Bind::Val(v) = b {
                *v = *slot_of.get(v).ok_or("a value without a slot")?;
            }
        }
    }
    let mut consts: Vec<ir::Cid> =
        steps.iter().flat_map(|s| s.binds.iter()).filter_map(|b| if let Bind::Const(c) = b { Some(*c) } else { None }).collect();
    consts.sort_unstable();
    consts.dedup();
    Ok(Plan { steps, slots, slot_of, input, input_len, output, output_dims, best_rows, consts })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn windows_follow_onnx() {
        assert_eq!(window(48, 3, 2, 1, 1, 1, false).unwrap(), 24);
        assert_eq!(window(5, 2, 2, 1, 0, 0, true).unwrap(), 3);
        assert_eq!(window(5, 2, 2, 1, 0, 0, false).unwrap(), 2);
        assert_eq!(window(6, 3, 3, 1, 0, 0, false).unwrap(), 2);
        assert!(window(1, 3, 1, 1, 0, 0, false).is_err());
        assert_eq!(same(7, 2, 1, 1, true), (0, 1));
        assert_eq!(same(7, 3, 2, 1, true), (1, 1));
        assert_eq!(same(8, 2, 2, 1, false), (0, 0));
        // A zero stride once divided by zero; vast sizes once overflowed.
        assert!(window(5, 2, 0, 1, 0, 0, false).is_err());
        assert!(window(5, 2, 1, 0, 0, 0, true).is_err());
        assert!(window(usize::MAX, 2, 1, 1, 1, 1, false).is_err());
        assert!(window(5, usize::MAX, 1, 2, 0, 0, false).is_err());
        let _ = same(5, 2, 0, 1, true);
        let _ = same(usize::MAX, usize::MAX, usize::MAX, usize::MAX, false);
        assert!(mul(&[usize::MAX, 2], "size").is_err());
        assert_eq!(work(&[usize::MAX, usize::MAX]), (usize::MAX as f64) * (usize::MAX as f64));
    }

    #[test]
    fn slots_are_shared_between_values_that_do_not_overlap() {
        let step = |w: Vid, r: Vid| Step {
            label: String::new(),
            shader: Shader::Copy,
            binds: vec![Bind::Val(w), Bind::Val(r)],
            params: vec![],
            groups: [1, 1, 1],
            work: 0.0,
        };
        // 0 (input) → 1 → 2 → 3 (output): 2 reuses the input's slot once 1 is computed, 3
        // reuses 1's; a step never writes the slot it reads.
        let steps = vec![step(1, 0), step(2, 1), step(3, 2)];
        let sizes = HashMap::from([(0, 10), (1, 10), (2, 10), (3, 4)]);
        let p = allocate(steps, &sizes, 0, 10, 3, vec![4], None).unwrap();
        assert_eq!(p.slots.len(), 2, "{:?}", p.slot_of);
        assert_eq!(p.slot_of.get(&2), p.slot_of.get(&0));
        assert_eq!(p.slot_of.get(&3), p.slot_of.get(&1));
        for s in &p.steps {
            assert_ne!(s.binds.first(), s.binds.get(1));
        }
    }

    #[test]
    fn grids_fit_the_limits() {
        let lim = Limits { max_binding: 1 << 30, max_buffer: 1 << 30, max_groups: 65535 };
        assert_eq!(grid(256, lim).unwrap(), [1, 1, 1]);
        assert_eq!(grid(65536 * 256, lim).unwrap(), [65535, 2, 1]);
        assert!(grid(usize::MAX / 2, lim).is_err());
    }
}
