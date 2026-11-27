//! The model, once per network: ONNX nodes turned into the operations the kernels run, with the
//! weights folded and the cheap steps fused into their neighbours.
//!
//! - `Identity` names resolve to what they copy.
//! - A `Conv` (or `ConvTranspose`, or `MatMul`) absorbs what follows it while it is that value's
//!   only reader: `BatchNormalization`, a per-channel `Add` / `Mul` by a weight (into one scale
//!   and shift per channel), then `Relu`, `Sigmoid` or swish (`x * Sigmoid(x)`), then a residual
//!   `Add` of a value already computed.
//! - A `Concat` along channels read only by convolutions, and a `Resize` (nearest, whole-number
//!   factors) read only by such a concat or convolution, are not computed: the convolution reads
//!   their inputs where they are.
//! - The DB head's last step (a 3×3 convolution to 64 channels, a 1×1 to one, sigmoid, then the
//!   mean with the other branch) is one kernel, so the 64-channel full-size map is never stored.
//! - Weights-only chains are worked out now; shape arithmetic (`Shape` and what reads it) waits
//!   for the input's size ([`Op::Meta`]).
//!
//! Nothing here touches a GPU, so it is tested on every machine.

use std::collections::{HashMap, HashSet};

use crate::eval::{self, Cpu, Data};
use crate::onnx;

pub type Vid = usize;
pub type Cid = usize;

/// What a value is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// A weight (or one worked out from weights): `consts[cid]`.
    Const(Cid),
    /// Small integers known once the input's size is (shape arithmetic).
    Meta,
    /// A tensor on the GPU.
    Run,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Act {
    None,
    Relu,
    Sigmoid,
    Swish,
}

/// A convolution's padding.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pads {
    /// top, left, bottom, right
    Explicit([usize; 4]),
    Same {
        upper: bool,
    },
}

#[derive(Clone, Debug)]
pub struct Conv {
    pub name: String,
    /// The input, as channel runs of other values, each upsampled (rows, columns) on the fly.
    pub srcs: Vec<(Vid, [usize; 2])>,
    /// [Cout, Cin/group, kh, kw]
    pub w: Cid,
    /// Per output channel: scale (Cout values) then shift (Cout values).
    pub st: Cid,
    pub cout: usize,
    pub cin_g: usize,
    pub group: usize,
    pub k: [usize; 2],
    pub stride: [usize; 2],
    pub dil: [usize; 2],
    pub pads: Pads,
    pub act: Act,
    pub residual: Option<Vid>,
    pub y: Vid,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Bin {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Un {
    Relu,
    Sigmoid,
    Sqrt,
    Square,
    Exp,
    Tanh,
}

#[derive(Clone, Debug)]
pub enum ViewKind {
    Identity,
    Reshape(Vid),
    Flatten(i64),
    Squeeze(Vec<i64>),
    Unsqueeze(Vec<i64>),
}

#[derive(Clone, Debug)]
pub enum Op {
    Conv(Conv),
    /// `conv` (≤ 64 channels, one tile) then a 1×1 to one channel with `w2` and `b2`, sigmoid,
    /// plus `other`, times `scale`, plus `shift`.
    Head {
        conv: Conv,
        w2: Cid,
        b2: f32,
        other: Option<Vid>,
        scale: f32,
        shift: f32,
        y: Vid,
    },
    ConvT {
        name: String,
        x: Vid,
        w: Cid,
        st: Cid,
        cin: usize,
        cout: usize,
        k: [usize; 2],
        stride: [usize; 2],
        dil: [usize; 2],
        pads: [usize; 4],
        out_pad: [usize; 2],
        act: Act,
        y: Vid,
    },
    Pool {
        x: Vid,
        max: bool,
        k: [usize; 2],
        stride: [usize; 2],
        pads: Pads,
        ceil: bool,
        include_pad: bool,
        y: Vid,
    },
    Resize {
        x: Vid,
        up: [usize; 2],
        y: Vid,
    },
    Concat {
        xs: Vec<Vid>,
        axis: i64,
        y: Vid,
    },
    Binary {
        op: Bin,
        a: Vid,
        b: Vid,
        y: Vid,
    },
    Unary {
        op: Un,
        x: Vid,
        y: Vid,
    },
    /// Per channel (axis 1): `x * s[c] + t[c]`, then `act`.
    Affine {
        x: Vid,
        st: Cid,
        act: Act,
        y: Vid,
    },
    /// `a × b` (+ a bias along the last axis), then `act`, then `+ residual`.
    MatMul {
        a: Vid,
        b: Vid,
        bias: Option<Cid>,
        act: Act,
        residual: Option<Vid>,
        y: Vid,
    },
    ReduceMean {
        x: Vid,
        axes: Vec<i64>,
        keep: bool,
        y: Vid,
    },
    Softmax {
        x: Vid,
        axis: i64,
        y: Vid,
    },
    Transpose {
        x: Vid,
        perm: Vec<i64>,
        y: Vid,
    },
    View {
        x: Vid,
        how: ViewKind,
        y: Vid,
    },
    Slice {
        x: Vid,
        starts: Vid,
        ends: Vid,
        axes: Option<Vid>,
        steps: Option<Vid>,
        y: Vid,
    },
    Shape {
        x: Vid,
        y: Vid,
    },
    /// Shape arithmetic, worked out with [`eval::eval`] once the sizes are known.
    Meta {
        node: onnx::Node,
        ins: Vec<Option<Vid>>,
        y: Vid,
    },
}

impl Op {
    pub fn output(&self) -> Vid {
        match self {
            Op::Conv(c) => c.y,
            Op::Head { y, .. }
            | Op::ConvT { y, .. }
            | Op::Pool { y, .. }
            | Op::Resize { y, .. }
            | Op::Concat { y, .. }
            | Op::Binary { y, .. }
            | Op::Unary { y, .. }
            | Op::Affine { y, .. }
            | Op::MatMul { y, .. }
            | Op::ReduceMean { y, .. }
            | Op::Softmax { y, .. }
            | Op::Transpose { y, .. }
            | Op::View { y, .. }
            | Op::Slice { y, .. }
            | Op::Shape { y, .. }
            | Op::Meta { y, .. } => *y,
        }
    }
}

/// The last `MatMul` (+ bias) → `Softmax` of a recogniser, when that is how it ends: what
/// `crate::exec` runs instead, giving the best class and its probability per row.
#[derive(Clone, Debug)]
pub struct Best {
    /// Index in `ops` of the `MatMul`; it and the `Softmax` after it are replaced.
    pub from: usize,
    pub a: Vid,
    pub w: Cid,
    pub bias: Option<Cid>,
}

/// A network ready to plan.
#[derive(Clone, Debug, Default)]
pub struct Prepared {
    pub names: Vec<String>,
    pub kinds: Vec<Kind>,
    pub consts: Vec<Cpu>,
    pub ops: Vec<Op>,
    pub input: Vid,
    pub output: Vid,
    pub best: Option<Best>,
}

impl Prepared {
    pub fn kind(&self, v: Vid) -> Kind {
        self.kinds.get(v).copied().unwrap_or(Kind::Run)
    }

    pub fn konst(&self, v: Vid) -> Option<&Cpu> {
        match self.kind(v) {
            Kind::Const(c) => self.consts.get(c),
            _ => None,
        }
    }

    pub fn name(&self, v: Vid) -> &str {
        self.names.get(v).map_or("?", String::as_str)
    }
}

struct Builder<'g> {
    g: &'g onnx::Graph,
    p: Prepared,
    ids: HashMap<String, Vid>,
    alias: HashMap<String, String>,
    /// Value name → indices of the nodes that read it (an output counts as one more reader).
    readers: HashMap<String, Vec<usize>>,
    outputs: HashSet<String>,
    done: HashSet<usize>,
    /// Value → index in `ops` of the op that computes it.
    def: HashMap<Vid, usize>,
    /// Concat outputs computed by their convolutions: their sources.
    virt: HashMap<String, Vec<(String, [usize; 2])>>,
    /// Values held in `p.consts`, weights and folded constants together.
    values: usize,
}

fn err<T>(why: impl Into<String>) -> Result<T, String> {
    Err(why.into())
}

fn usize_of(v: i64, what: &str) -> Result<usize, String> {
    usize::try_from(v).map_err(|_| format!("{what} {v}"))
}

fn pair(v: &[i64], default: usize, what: &str) -> Result<[usize; 2], String> {
    match v {
        [] => Ok([default; 2]),
        [a, b] => Ok([usize_of(*a, what)?, usize_of(*b, what)?]),
        other => err(format!("{what} {other:?} is not two values")),
    }
}

/// The largest kernel side, stride, dilation and padding a network may ask for, and the most
/// multiply-adds one output value may take. PP-OCRv5 stays far below (kernels up to 7, a
/// depth of a few thousand); the bounds keep a hostile model's arithmetic in range and one
/// dispatch short enough not to trip the driver's watchdog.
pub const MAX_KERNEL: usize = 64;
pub const MAX_STRIDE: usize = 64;
pub const MAX_PAD: usize = 1024;
pub const MAX_DEPTH: usize = 1 << 17;
/// Pooling windows may be wider (a pool over a whole row), up to this many values.
pub const MAX_POOL_AREA: usize = 1 << 16;

/// Refuse window attributes outside the bounds above.
fn check_window(name: &str, k: [usize; 2], stride: [usize; 2], dil: [usize; 2], pads: &Pads, depth: usize, max_side: usize) -> Result<(), String> {
    if k.iter().any(|&v| v == 0 || v > max_side) {
        return err(format!("{name}: kernel {k:?} (each side 1 to {max_side})"));
    }
    if stride.iter().chain(&dil).any(|&v| v == 0 || v > MAX_STRIDE) {
        return err(format!("{name}: strides {stride:?} and dilations {dil:?} (each 1 to {MAX_STRIDE})"));
    }
    if let Pads::Explicit(p) = pads
        && p.iter().any(|&v| v > MAX_PAD)
    {
        return err(format!("{name}: pads {p:?} (each at most {MAX_PAD})"));
    }
    if depth > MAX_DEPTH {
        return err(format!("{name}: {depth} multiply-adds per output value (at most {MAX_DEPTH})"));
    }
    Ok(())
}

fn quad(v: &[i64]) -> Result<[usize; 4], String> {
    match v {
        [] => Ok([0; 4]),
        [a, b, c, d] => Ok([usize_of(*a, "pad")?, usize_of(*b, "pad")?, usize_of(*c, "pad")?, usize_of(*d, "pad")?]),
        other => err(format!("pads {other:?} are not four values")),
    }
}

fn pads_of(n: &onnx::Node) -> Result<Pads, String> {
    match n.attrs.get("auto_pad") {
        Some(onnx::Attr::Str(s)) if s == "SAME_UPPER" => Ok(Pads::Same { upper: true }),
        Some(onnx::Attr::Str(s)) if s == "SAME_LOWER" => Ok(Pads::Same { upper: false }),
        Some(onnx::Attr::Str(s)) if s == "VALID" => Ok(Pads::Explicit([0; 4])),
        Some(onnx::Attr::Str(s)) if s != "NOTSET" && !s.is_empty() => err(format!("{} has auto_pad {s}", n.name)),
        _ => Ok(Pads::Explicit(quad(n.ints("pads"))?)),
    }
}

/// Per-channel factors of a weight that broadcasts over [N, C, H, W]: one value, or C values
/// shaped [C], [C, 1, 1] or [1, C, 1, 1].
fn per_channel(t: &Cpu, c: usize) -> Option<Vec<f32>> {
    let v = t.as_floats();
    if v.len() == 1 {
        return Some(vec![*v.first()?; c]);
    }
    let ok = match t.dims.as_slice() {
        [n] => *n == c,
        [n, 1, 1] => *n == c,
        [1, n, 1, 1] => *n == c,
        _ => false,
    };
    (ok && v.len() == c).then_some(v)
}

/// Per-column factors along the last axis: [N] or [1, ..., N].
fn per_column(t: &Cpu, n: usize) -> Option<Vec<f32>> {
    let (last, lead) = t.dims.split_last()?;
    (*last == n && lead.iter().all(|&d| d == 1)).then(|| t.as_floats())
}

impl<'g> Builder<'g> {
    fn resolve(&self, name: &str) -> String {
        let mut n = name;
        for _ in 0..=self.g.nodes.len() {
            match self.alias.get(n) {
                Some(next) => n = next,
                None => break,
            }
        }
        n.to_string()
    }

    fn node_in(&self, n: &onnx::Node, i: usize) -> Option<String> {
        n.inputs.get(i).filter(|s| !s.is_empty()).map(|s| self.resolve(s))
    }

    fn node_out(&self, n: &onnx::Node) -> Result<String, String> {
        n.outputs.first().cloned().ok_or_else(|| format!("{} has no output", n.name))
    }

    fn new_value(&mut self, name: &str, kind: Kind) -> Vid {
        let v = self.p.names.len();
        self.p.names.push(name.to_string());
        self.p.kinds.push(kind);
        self.ids.insert(name.to_string(), v);
        v
    }

    fn add_const(&mut self, t: Cpu) -> Cid {
        self.values = self.values.saturating_add(t.len());
        self.p.consts.push(t);
        self.p.consts.len() - 1
    }

    /// The value named `name`: known, or a weight of the graph.
    fn value(&mut self, name: &str) -> Result<Vid, String> {
        if let Some(&v) = self.ids.get(name) {
            return Ok(v);
        }
        match self.g.tensor(name) {
            Some(t) => {
                let c = Cpu::of(t)?;
                let cid = self.add_const(c);
                Ok(self.new_value(name, Kind::Const(cid)))
            }
            None => err(format!("{name} is read before it is computed")),
        }
    }

    fn opt_value(&mut self, name: Option<String>) -> Result<Option<Vid>, String> {
        name.map(|n| self.value(&n)).transpose()
    }

    fn konst_of(&mut self, name: &str) -> Option<Cpu> {
        let v = self.value(name).ok()?;
        self.p.konst(v).cloned()
    }

    /// The single node reading `value`, if exactly one does and it is not a graph output.
    fn only_reader(&self, value: &str) -> Option<(usize, &'g onnx::Node)> {
        if self.outputs.contains(value) {
            return None;
        }
        match self.readers.get(value).map(Vec::as_slice) {
            Some([i]) if !self.done.contains(i) => self.g.nodes.get(*i).map(|n| (*i, n)),
            _ => None,
        }
    }

    fn readers_of(&self, value: &str) -> Vec<(usize, &'g onnx::Node)> {
        self.readers.get(value).map(|r| r.iter().filter_map(|&i| self.g.nodes.get(i).map(|n| (i, n))).collect()).unwrap_or_default()
    }

    fn computed(&self, name: &str) -> Option<Vid> {
        self.ids.get(name).copied().filter(|v| self.p.kind(*v) == Kind::Run && (self.def.contains_key(v) || *v == self.p.input))
    }

    /// Absorb what follows `cur` into a per-channel scale `s` and shift `t` and an activation.
    /// Returns the value now produced and the activation. `columns` uses the last axis instead
    /// of axis 1 (for `MatMul`), and allows only an added bias.
    fn absorb(&mut self, mut cur: String, s: &mut [f32], t: &mut [f32], columns: bool) -> Result<(String, Act), String> {
        let c = s.len();
        while let Some((i, n)) = self.only_reader(&cur) {
            let other = |b: &Builder<'_>| -> Option<String> {
                let a = b.node_in(n, 0)?;
                let o = b.node_in(n, 1)?;
                if a == cur {
                    Some(o)
                } else if o == cur {
                    Some(a)
                } else {
                    None
                }
            };
            match n.op.as_str() {
                "BatchNormalization" if !columns && self.node_in(n, 0).as_deref() == Some(cur.as_str()) => {
                    let get = |b: &mut Builder<'_>, k: usize| b.node_in(n, k).and_then(|nm| b.konst_of(&nm)).and_then(|t| per_channel(&t, c));
                    let (Some(g), Some(beta), Some(mean), Some(var)) = (get(self, 1), get(self, 2), get(self, 3), get(self, 4)) else { break };
                    let eps = n.float("epsilon", 1e-5);
                    for j in 0..c {
                        let (Some(gj), Some(bj), Some(mj), Some(vj)) = (g.get(j), beta.get(j), mean.get(j), var.get(j)) else { break };
                        let f = gj / (vj + eps).sqrt();
                        if let (Some(sj), Some(tj)) = (s.get_mut(j), t.get_mut(j)) {
                            *sj *= f;
                            *tj = (*tj - mj) * f + bj;
                        }
                    }
                }
                op @ ("Add" | "Mul") => {
                    let Some(o) = other(self) else { break };
                    let Some(k) = self.konst_of(&o) else { break };
                    let f = if columns { per_column(&k, c) } else { per_channel(&k, c) };
                    let Some(f) = f else { break };
                    if columns && op == "Mul" {
                        break;
                    }
                    for j in 0..c {
                        let fj = f.get(j).copied().unwrap_or(0.0);
                        if let (Some(sj), Some(tj)) = (s.get_mut(j), t.get_mut(j)) {
                            if op == "Add" {
                                *tj += fj;
                            } else {
                                *sj *= fj;
                                *tj *= fj;
                            }
                        }
                    }
                }
                "Relu" => {
                    self.done.insert(i);
                    return Ok((self.node_out(n)?, Act::Relu));
                }
                "Sigmoid" => {
                    self.done.insert(i);
                    return Ok((self.node_out(n)?, Act::Sigmoid));
                }
                _ => break,
            }
            self.done.insert(i);
            cur = self.node_out(n)?;
        }
        // Swish: read by a Sigmoid and by the Mul of it with that Sigmoid.
        let rs = self.readers_of(&cur);
        if !self.outputs.contains(&cur)
            && let [(i1, n1), (i2, n2)] = rs.as_slice()
        {
            let (sig, mul) = if n1.op == "Sigmoid" { ((*i1, *n1), (*i2, *n2)) } else { ((*i2, *n2), (*i1, *n1)) };
            if sig.1.op == "Sigmoid" && mul.1.op == "Mul" && !self.done.contains(&sig.0) && !self.done.contains(&mul.0) {
                let sig_out = self.node_out(sig.1)?;
                let ins = (self.node_in(mul.1, 0), self.node_in(mul.1, 1));
                let pairs = ins == (Some(cur.clone()), Some(sig_out.clone())) || ins == (Some(sig_out.clone()), Some(cur.clone()));
                if pairs && self.only_reader(&sig_out).map(|r| r.0) == Some(mul.0) {
                    self.done.insert(sig.0);
                    self.done.insert(mul.0);
                    return Ok((self.node_out(mul.1)?, Act::Swish));
                }
            }
        }
        Ok((cur, Act::None))
    }

    /// A residual `Add` of a value computed before the op now being built.
    fn residual(&mut self, cur: &str) -> Result<Option<(String, Vid)>, String> {
        let Some((i, n)) = self.only_reader(cur) else { return Ok(None) };
        if n.op != "Add" {
            return Ok(None);
        }
        let (a, b) = (self.node_in(n, 0), self.node_in(n, 1));
        let other = if a.as_deref() == Some(cur) {
            b
        } else if b.as_deref() == Some(cur) {
            a
        } else {
            None
        };
        let Some(other) = other else { return Ok(None) };
        let Some(v) = self.computed(&other) else { return Ok(None) };
        self.done.insert(i);
        Ok(Some((self.node_out(n)?, v)))
    }

    fn emit(&mut self, op: Op) {
        self.def.insert(op.output(), self.p.ops.len());
        self.p.ops.push(op);
    }

    fn out_value(&mut self, name: &str, kind: Kind) -> Vid {
        self.new_value(name, kind)
    }

    /// Which concats and resizes the convolutions read in place.
    fn plan_virtual(&mut self) {
        let conv_gemm = |n: &onnx::Node| n.op == "Conv" && n.int("group", 1) == 1;
        let resize_up = |b: &mut Builder<'_>, n: &onnx::Node| -> Option<[usize; 2]> {
            if n.op != "Resize" {
                return None;
            }
            let mode_ok = matches!(n.attrs.get("mode"), Some(onnx::Attr::Str(m)) if m == "nearest")
                && matches!(n.attrs.get("coordinate_transformation_mode"), Some(onnx::Attr::Str(m)) if m == "asymmetric")
                && matches!(n.attrs.get("nearest_mode"), None | Some(onnx::Attr::Str(_)))
                && !matches!(n.attrs.get("nearest_mode"), Some(onnx::Attr::Str(m)) if m != "floor");
            if !mode_ok {
                return None;
            }
            let scales = b.node_in(n, 2).and_then(|s| b.konst_of(&s))?.as_floats();
            whole_up(&scales)
        };
        let nodes = self.g.nodes.clone();
        // Value → the node computing it, looked up once instead of a scan per input.
        let mut producers: HashMap<&str, usize> = HashMap::new();
        for (i, m) in nodes.iter().enumerate() {
            if let Some(o) = m.outputs.first() {
                producers.entry(o.as_str()).or_insert(i);
            }
        }
        for n in &nodes {
            let Ok(out) = self.node_out(n) else { continue };
            if n.op != "Concat" || n.int("axis", 0) != 1 || n.inputs.len() > 4 || self.outputs.contains(&out) {
                continue;
            }
            let rs = self.readers_of(&out);
            if rs.is_empty() || !rs.iter().all(|(_, r)| conv_gemm(r) && self.node_in(r, 0).as_deref() == Some(out.as_str())) {
                continue;
            }
            let mut srcs = Vec::new();
            for k in 0..n.inputs.len() {
                let Some(inp) = self.node_in(n, k) else { continue };
                let producer = producers.get(inp.as_str()).copied();
                let mut src = (inp.clone(), [1, 1]);
                if let Some(pi) = producer
                    && let Some(pn) = nodes.get(pi)
                    && self.readers.get(&inp).map(Vec::len) == Some(1)
                    && !self.outputs.contains(&inp)
                    && let Some(up) = resize_up(self, pn)
                    && let Some(x) = self.node_in(pn, 0)
                {
                    self.done.insert(pi);
                    src = (x, up);
                }
                srcs.push(src);
            }
            self.virt.insert(out, srcs);
        }
        // A Resize read only by convolutions (no concat).
        for (pi, n) in nodes.iter().enumerate() {
            let Ok(out) = self.node_out(n) else { continue };
            if self.done.contains(&pi) || self.outputs.contains(&out) {
                continue;
            }
            let rs = self.readers_of(&out);
            if rs.is_empty() || !rs.iter().all(|(_, r)| conv_gemm(r) && self.node_in(r, 0).as_deref() == Some(out.as_str())) {
                continue;
            }
            if let Some(up) = resize_up(self, n)
                && let Some(x) = self.node_in(n, 0)
            {
                self.done.insert(pi);
                self.virt.insert(out, vec![(x, up)]);
            }
        }
        // The concats themselves are not run.
        for (i, n) in nodes.iter().enumerate() {
            if n.op == "Concat" && n.outputs.first().is_some_and(|o| self.virt.contains_key(o)) {
                self.done.insert(i);
            }
        }
    }

    fn conv(&mut self, n: &'g onnx::Node) -> Result<(), String> {
        let x = self.node_in(n, 0).ok_or("Conv has no input")?;
        let wname = self.node_in(n, 1).ok_or("Conv has no weight")?;
        let w = self.konst_of(&wname).ok_or_else(|| format!("{}: the weight is not constant", n.name))?;
        let [cout, cin_g, kh, kw] = match w.dims.as_slice() {
            [a, b, c, d] => [*a, *b, *c, *d],
            other => return err(format!("{}: weight dims {other:?}", n.name)),
        };
        let group = usize_of(n.int("group", 1), "group")?;
        if group == 0 || cout % group != 0 {
            return err(format!("{}: group {group} for {cout} channels", n.name));
        }
        let k = pair(n.ints("kernel_shape"), 0, "kernel_shape")?;
        if k != [0, 0] && k != [kh, kw] {
            return err(format!("{}: kernel_shape {k:?} differs from the weight's", n.name));
        }
        let stride = pair(n.ints("strides"), 1, "strides")?;
        let dil = pair(n.ints("dilations"), 1, "dilations")?;
        let pads = pads_of(n)?;
        let depth = cin_g.saturating_mul(kh).saturating_mul(kw);
        check_window(&n.name, [kh, kw], stride, dil, &pads, depth, MAX_KERNEL)?;
        let mut s = vec![1.0f32; cout];
        let mut t = match self.node_in(n, 2) {
            Some(b) => self.konst_of(&b).map(|b| b.as_floats()).filter(|v| v.len() == cout).ok_or_else(|| format!("{}: bias", n.name))?,
            None => vec![0.0; cout],
        };
        let out = self.node_out(n)?;
        let (cur, act) = self.absorb(out, &mut s, &mut t, false)?;
        // A 1×1 to one channel then Sigmoid may end the DB head (its Add is the head's own).
        let head_like = cout == 1 && [kh, kw] == [1, 1] && act == Act::Sigmoid;
        let (cur, residual) = match if head_like { None } else { self.residual(&cur)? } {
            Some((o, v)) => (o, Some(v)),
            None => (cur, None),
        };
        let srcs_named = self.virt.get(&x).cloned().unwrap_or_else(|| vec![(x.clone(), [1, 1])]);
        let mut srcs = Vec::new();
        for (name, up) in srcs_named {
            let v = self.value(&name)?;
            if self.p.kind(v) != Kind::Run {
                return err(format!("{}: the input {name} is not computed on the GPU", n.name));
            }
            srcs.push((v, up));
        }
        let wc = self.add_const(w);
        let mut stv = s;
        stv.extend_from_slice(&t);
        let st = self.add_const(Cpu::floats(vec![2 * cout], stv));
        let conv = Conv { name: n.name.clone(), srcs, w: wc, st, cout, cin_g, group, k: [kh, kw], stride, dil, pads, act, residual, y: usize::MAX };

        // The DB head: this 1×1 to one channel reads a ≤ 64-channel convolution only.
        if self.head(&conv, &x, &cur)?.is_some() {
            return Ok(());
        }
        let y = self.out_value(&cur, Kind::Run);
        self.emit(Op::Conv(Conv { y, ..conv }));
        Ok(())
    }

    /// Turn the convolution `a` that computes `x` and this 1×1 (`b`) into one [`Op::Head`].
    fn head(&mut self, b: &Conv, x: &str, cur: &str) -> Result<Option<()>, String> {
        if b.cout != 1
            || b.k != [1, 1]
            || b.stride != [1, 1]
            || b.group != 1
            || b.pads != Pads::Explicit([0; 4])
            || b.act != Act::Sigmoid
            || b.residual.is_some()
            || b.srcs.len() != 1
        {
            return Ok(None);
        }
        let Some(&xv) = self.ids.get(x) else { return Ok(None) };
        let Some(&ai) = self.def.get(&xv) else { return Ok(None) };
        let Some(Op::Conv(a)) = self.p.ops.get(ai) else { return Ok(None) };
        if a.cout > 64 || a.group != 1 || a.residual.is_some() || self.readers.get(x).map(Vec::len) != Some(1) || self.outputs.contains(x) {
            return Ok(None);
        }
        let a = a.clone();
        let w2 = self.p.consts.get(b.w).map(Cpu::as_floats).unwrap_or_default();
        let st2 = self.p.consts.get(b.st).map(Cpu::as_floats).unwrap_or_default();
        let (Some(&s2), Some(&t2)) = (st2.first(), st2.get(1)) else { return Ok(None) };
        if w2.len() != a.cout {
            return Ok(None);
        }
        let mut cur = cur.to_string();
        // + another value computed before `a`, then scalar Mul / Add.
        let mut other = None;
        if let Some((i, n)) = self.only_reader(&cur)
            && n.op == "Add"
        {
            let (p, q) = (self.node_in(n, 0), self.node_in(n, 1));
            let o = if p.as_deref() == Some(cur.as_str()) { q } else { p };
            if let Some(o) = o
                && let Some(v) = self.computed(&o)
                && self.def.get(&v).is_none_or(|&d| d < ai)
            {
                other = Some(v);
                self.done.insert(i);
                cur = self.node_out(n)?;
            }
        }
        let (mut scale, mut shift) = (1.0f32, 0.0f32);
        while let Some((i, n)) = self.only_reader(&cur) {
            if !matches!(n.op.as_str(), "Mul" | "Add") {
                break;
            }
            let (p, q) = (self.node_in(n, 0), self.node_in(n, 1));
            let o = if p.as_deref() == Some(cur.as_str()) { q } else { p };
            let Some(k) = o.and_then(|o| self.konst_of(&o)).and_then(|k| k.scalar()) else { break };
            if n.op == "Mul" {
                scale *= k;
                shift *= k;
            } else {
                shift += k;
            }
            self.done.insert(i);
            cur = self.node_out(n)?;
        }
        // Fold the 1×1's scale into its weights.
        let w2: Vec<f32> = w2.iter().map(|w| w * s2).collect();
        let w2c = self.add_const(Cpu::floats(vec![w2.len()], w2));
        let y = self.out_value(&cur, Kind::Run);
        let op = Op::Head { conv: a, w2: w2c, b2: t2, other, scale, shift, y };
        self.def.insert(y, ai);
        if let Some(slot) = self.p.ops.get_mut(ai) {
            *slot = op;
        }
        self.def.remove(&xv);
        Ok(Some(()))
    }

    fn node(&mut self, idx: usize, n: &'g onnx::Node) -> Result<(), String> {
        if self.done.contains(&idx) {
            return Ok(());
        }
        if n.op == "Conv" {
            return self.conv(n);
        }
        let out = self.node_out(n)?;
        let ins: Vec<Option<String>> = (0..n.inputs.len()).map(|i| self.node_in(n, i)).collect();
        let mut vids = Vec::new();
        for i in &ins {
            vids.push(self.opt_value(i.clone())?);
        }
        let kinds: Vec<Option<Kind>> = vids.iter().map(|v| v.map(|v| self.p.kind(v))).collect();
        let all_const = kinds.iter().flatten().all(|k| matches!(k, Kind::Const(_)));
        let any_run = kinds.iter().flatten().any(|k| *k == Kind::Run);
        let first = vids.first().copied().flatten().ok_or_else(|| format!("{} has no input", n.name))?;

        // Weights only: work it out now.
        if all_const && n.op != "Shape" {
            let held: Vec<Option<Cpu>> = vids.iter().map(|v| v.and_then(|v| self.p.konst(v).cloned())).collect();
            let refs: Vec<Option<&Cpu>> = held.iter().map(Option::as_ref).collect();
            let t = eval::eval(n, &refs).map_err(|e| format!("{}: {e}", n.name))?;
            let c = self.add_const(t);
            self.new_value(&out, Kind::Const(c));
            return Ok(());
        }
        // Shape arithmetic: later, once the sizes are known.
        if !any_run && n.op != "Shape" {
            let y = self.out_value(&out, Kind::Meta);
            self.emit(Op::Meta { node: n.clone(), ins: vids, y });
            return Ok(());
        }
        let arg = |i: usize| vids.get(i).copied().flatten();
        match n.op.as_str() {
            "ConvTranspose" => {
                let w = arg(1).and_then(|v| self.p.konst(v).cloned()).ok_or_else(|| format!("{}: the weight is not constant", n.name))?;
                let [cin, cout_g, kh, kw] = match w.dims.as_slice() {
                    [a, b, c, d] => [*a, *b, *c, *d],
                    other => return err(format!("{}: weight dims {other:?}", n.name)),
                };
                if n.int("group", 1) != 1 || n.attrs.contains_key("output_shape") || !matches!(pads_of(n)?, Pads::Explicit(_)) {
                    return err(format!("{}: only ungrouped ConvTranspose with explicit pads is run", n.name));
                }
                let Pads::Explicit(pads) = pads_of(n)? else { return err("ConvTranspose pads") };
                let stride = pair(n.ints("strides"), 1, "strides")?;
                let dil = pair(n.ints("dilations"), 1, "dilations")?;
                let out_pad = pair(n.ints("output_padding"), 0, "output_padding")?;
                let depth = cin.saturating_mul(kh).saturating_mul(kw);
                check_window(&n.name, [kh, kw], stride, dil, &Pads::Explicit(pads), depth, MAX_KERNEL)?;
                // ONNX: each output padding is below the stride or the dilation.
                let ([o0, o1], [s0, s1], [d0, d1]) = (out_pad, stride, dil);
                if o0 >= s0.max(d0) || o1 >= s1.max(d1) {
                    return err(format!("{}: output_padding {out_pad:?} for strides {stride:?}", n.name));
                }
                let cout = cout_g;
                let mut s = vec![1.0f32; cout];
                let mut t = match arg(2).and_then(|v| self.p.konst(v).cloned()) {
                    Some(b) => b.as_floats(),
                    None => vec![0.0; cout],
                };
                if t.len() != cout {
                    return err(format!("{}: bias", n.name));
                }
                let (cur, act) = self.absorb(out, &mut s, &mut t, false)?;
                let wc = self.add_const(w);
                s.extend_from_slice(&t);
                let st = self.add_const(Cpu::floats(vec![2 * cout], s));
                let y = self.out_value(&cur, Kind::Run);
                let op = Op::ConvT { name: n.name.clone(), x: first, w: wc, st, cin, cout, k: [kh, kw], stride, dil, pads, out_pad, act, y };
                self.emit(op);
            }
            "BatchNormalization" => {
                let get = |b: &Builder<'_>, k: usize| arg(k).and_then(|v| b.p.konst(v)).map(Cpu::as_floats);
                let (Some(g), Some(beta), Some(mean), Some(var)) = (get(self, 1), get(self, 2), get(self, 3), get(self, 4)) else {
                    return err(format!("{}: the statistics are not constant", n.name));
                };
                let c = g.len();
                if [beta.len(), mean.len(), var.len()].iter().any(|&l| l != c) {
                    return err(format!("{}: statistics of different lengths", n.name));
                }
                let eps = n.float("epsilon", 1e-5);
                let mut s = Vec::with_capacity(c);
                let mut t = Vec::with_capacity(c);
                for j in 0..c {
                    let f = g.get(j).copied().unwrap_or(1.0) / (var.get(j).copied().unwrap_or(1.0) + eps).sqrt();
                    s.push(f);
                    t.push(beta.get(j).copied().unwrap_or(0.0) - mean.get(j).copied().unwrap_or(0.0) * f);
                }
                let (cur, act) = self.absorb(out, &mut s, &mut t, false)?;
                s.extend_from_slice(&t);
                let st = self.add_const(Cpu::floats(vec![2 * c], s));
                let y = self.out_value(&cur, Kind::Run);
                self.emit(Op::Affine { x: first, st, act, y });
            }
            op @ ("MaxPool" | "AveragePool") => {
                if n.ints("dilations").iter().any(|&d| d != 1) || n.outputs.len() > 1 {
                    return err(format!("{}: dilated pooling or indices", n.name));
                }
                let y = self.out_value(&out, Kind::Run);
                let k = pair(n.ints("kernel_shape"), 1, "kernel_shape")?;
                let stride = pair(n.ints("strides"), 1, "strides")?;
                let area = k[0].saturating_mul(k[1]);
                check_window(&n.name, k, stride, [1, 1], &pads_of(n)?, 0, MAX_POOL_AREA)?;
                if area > MAX_POOL_AREA {
                    return err(format!("{}: a pooling window of {area} values (at most {MAX_POOL_AREA})", n.name));
                }
                self.emit(Op::Pool {
                    x: first,
                    max: op == "MaxPool",
                    k,
                    stride,
                    pads: pads_of(n)?,
                    ceil: n.int("ceil_mode", 0) != 0,
                    include_pad: n.int("count_include_pad", 0) != 0,
                    y,
                });
            }
            "Resize" => {
                let mode = |k: &str| match n.attrs.get(k) {
                    Some(onnx::Attr::Str(s)) => s.clone(),
                    _ => String::new(),
                };
                if mode("mode") != "nearest"
                    || mode("coordinate_transformation_mode") != "asymmetric"
                    || !matches!(mode("nearest_mode").as_str(), "" | "floor")
                {
                    return err(format!("{}: only nearest, asymmetric, floor resizing is run", n.name));
                }
                let scales = arg(2).and_then(|v| self.p.konst(v)).map(Cpu::as_floats).unwrap_or_default();
                let up = whole_up(&scales).ok_or_else(|| format!("{}: scales {scales:?} are not whole-number factors", n.name))?;
                let y = self.out_value(&out, Kind::Run);
                self.emit(Op::Resize { x: first, up, y });
            }
            "Concat" => {
                let xs: Vec<Vid> = vids.iter().flatten().copied().collect();
                let y = self.out_value(&out, Kind::Run);
                self.emit(Op::Concat { xs, axis: n.int("axis", 0), y });
            }
            op @ ("Add" | "Sub" | "Mul" | "Div" | "Pow") => {
                let b = arg(1).ok_or_else(|| format!("{} has one input", n.name))?;
                let y_name = out;
                if op == "Pow"
                    && let Some(e) = self.p.konst(b).and_then(Cpu::scalar)
                    && (e == 2.0 || e == 0.5 || e == 1.0)
                {
                    let y = self.out_value(&y_name, Kind::Run);
                    if e == 1.0 {
                        self.emit(Op::View { x: first, how: ViewKind::Identity, y });
                    } else {
                        self.emit(Op::Unary { op: if e == 2.0 { Un::Square } else { Un::Sqrt }, x: first, y });
                    }
                    return Ok(());
                }
                let bin = match op {
                    "Add" => Bin::Add,
                    "Sub" => Bin::Sub,
                    "Mul" => Bin::Mul,
                    "Div" => Bin::Div,
                    _ => Bin::Pow,
                };
                let y = self.out_value(&y_name, Kind::Run);
                self.emit(Op::Binary { op: bin, a: first, b, y });
            }
            op @ ("Relu" | "Sigmoid" | "Sqrt" | "Exp" | "Tanh") => {
                let un = match op {
                    "Relu" => Un::Relu,
                    "Sigmoid" => Un::Sigmoid,
                    "Sqrt" => Un::Sqrt,
                    "Exp" => Un::Exp,
                    _ => Un::Tanh,
                };
                let y = self.out_value(&out, Kind::Run);
                self.emit(Op::Unary { op: un, x: first, y });
            }
            "MatMul" => {
                let b = arg(1).ok_or("MatMul has one input")?;
                let cols = match self.p.konst(b) {
                    Some(w) => match w.dims.as_slice() {
                        [_, n] => Some(*n),
                        _ => None,
                    },
                    None => None,
                };
                let (cur, bias, act) = match cols {
                    Some(c) => {
                        let mut s = vec![1.0f32; c];
                        let mut t = vec![0.0f32; c];
                        let (cur, act) = self.absorb(out, &mut s, &mut t, true)?;
                        let bias = (t.iter().any(|&v| v != 0.0)).then(|| self.add_const(Cpu::floats(vec![c], t)));
                        (cur, bias, act)
                    }
                    None => (out, None, Act::None),
                };
                let (cur, residual) = match self.residual(&cur)? {
                    Some((o, v)) => (o, Some(v)),
                    None => (cur, None),
                };
                let y = self.out_value(&cur, Kind::Run);
                self.emit(Op::MatMul { a: first, b, bias, act, residual, y });
            }
            "ReduceMean" => {
                let y = self.out_value(&out, Kind::Run);
                self.emit(Op::ReduceMean { x: first, axes: n.ints("axes").to_vec(), keep: n.int("keepdims", 1) != 0, y });
            }
            "Softmax" => {
                let y = self.out_value(&out, Kind::Run);
                self.emit(Op::Softmax { x: first, axis: n.int("axis", 1), y });
            }
            "Transpose" => {
                let y = self.out_value(&out, Kind::Run);
                self.emit(Op::Transpose { x: first, perm: n.ints("perm").to_vec(), y });
            }
            op @ ("Identity" | "Reshape" | "Flatten" | "Squeeze" | "Unsqueeze") => {
                let how = match op {
                    "Reshape" => ViewKind::Reshape(arg(1).ok_or("Reshape has no shape")?),
                    "Flatten" => ViewKind::Flatten(n.int("axis", 1)),
                    "Squeeze" | "Unsqueeze" => {
                        let axes = match arg(1) {
                            Some(v) => self.p.konst(v).ok_or_else(|| format!("{}: axes are not constant", n.name))?.as_ints()?,
                            None => n.ints("axes").to_vec(),
                        };
                        if op == "Squeeze" { ViewKind::Squeeze(axes) } else { ViewKind::Unsqueeze(axes) }
                    }
                    _ => ViewKind::Identity,
                };
                let y = self.out_value(&out, Kind::Run);
                self.emit(Op::View { x: first, how, y });
            }
            "Slice" => {
                if !n.ints("starts").is_empty() {
                    return err(format!("{}: Slice with attributes (opset < 10)", n.name));
                }
                let (starts, ends) = (arg(1).ok_or("Slice has no starts")?, arg(2).ok_or("Slice has no ends")?);
                let y = self.out_value(&out, Kind::Run);
                self.emit(Op::Slice { x: first, starts, ends, axes: arg(3), steps: arg(4), y });
            }
            "Shape" => {
                if let Some(c) = self.p.konst(first) {
                    let dims: Vec<i64> = c.dims.iter().map(|&d| d as i64).collect();
                    let cid = self.add_const(Cpu::ints(vec![dims.len()], dims));
                    self.new_value(&out, Kind::Const(cid));
                } else {
                    let y = self.out_value(&out, Kind::Meta);
                    self.emit(Op::Shape { x: first, y });
                }
            }
            "Dropout" => {
                let y = self.out_value(&out, Kind::Run);
                self.emit(Op::View { x: first, how: ViewKind::Identity, y });
            }
            other => return err(format!("{other} ({}) has no GPU kernel", n.name)),
        }
        Ok(())
    }

    /// The recogniser's tail: `MatMul` by a weight (+ bias), then a last-axis `Softmax` that is
    /// the output.
    fn find_best(&mut self) {
        let n = self.p.ops.len();
        let (Some(Op::Softmax { x, y, .. }), Some(Op::MatMul { a, b, bias, act: Act::None, residual: None, y: my })) =
            (self.p.ops.get(n.wrapping_sub(1)), self.p.ops.get(n.wrapping_sub(2)))
        else {
            return;
        };
        if *y != self.p.output || x != my {
            return;
        }
        let Kind::Const(w) = self.p.kind(*b) else { return };
        if self.p.consts.get(w).map(|t| t.dims.len()) != Some(2) {
            return;
        }
        self.p.best = Some(Best { from: n - 2, a: *a, w, bias: *bias });
    }
}

/// Integer upsampling factors from Resize scales [1, 1, fh, fw].
fn whole_up(scales: &[f32]) -> Option<[usize; 2]> {
    match scales {
        [n, c, h, w] if *n == 1.0 && *c == 1.0 && h.fract() == 0.0 && w.fract() == 0.0 && *h >= 1.0 && *w >= 1.0 && *h <= 64.0 && *w <= 64.0 => {
            Some([*h as usize, *w as usize])
        }
        _ => None,
    }
}

/// Prepare `model` (one runtime input, one output).
pub fn prepare(model: &onnx::Model) -> Result<Prepared, String> {
    prepare_within(model, onnx::MAX_VALUES)
}

/// [`prepare`], refusing a model whose constants come to more than `max_values`.
pub(crate) fn prepare_within(model: &onnx::Model, max_values: usize) -> Result<Prepared, String> {
    let g = &model.graph;
    let mut alias = HashMap::new();
    for n in &g.nodes {
        if n.op == "Identity"
            && let (Some(i), Some(o)) = (n.inputs.first(), n.outputs.first())
        {
            alias.insert(o.clone(), i.clone());
        }
    }
    let mut b = Builder {
        g,
        p: Prepared::default(),
        ids: HashMap::new(),
        alias,
        readers: HashMap::new(),
        outputs: HashSet::new(),
        done: HashSet::new(),
        def: HashMap::new(),
        virt: HashMap::new(),
        values: 0,
    };
    for (i, n) in g.nodes.iter().enumerate() {
        if n.op == "Identity" || n.op == "Constant" {
            b.done.insert(i);
            continue;
        }
        if !n.domain.is_empty() && n.domain != "ai.onnx" {
            return err(format!("{} is in the operator domain {}", n.name, n.domain));
        }
        for inp in &n.inputs {
            if !inp.is_empty() {
                let r = b.resolve(inp);
                b.readers.entry(r).or_default().push(i);
            }
        }
    }
    let runtime: Vec<&String> = g.inputs.iter().filter(|i| g.tensor(i).is_none()).collect();
    let [input] = runtime.as_slice() else { return err(format!("the model has {} runtime inputs, not one", runtime.len())) };
    let [output] = g.outputs.as_slice() else { return err(format!("the model has {} outputs, not one", g.outputs.len())) };
    let output = b.resolve(output);
    b.outputs.insert(output.clone());
    b.p.input = b.new_value(input, Kind::Run);
    b.plan_virtual();
    for (i, n) in g.nodes.iter().enumerate() {
        b.node(i, n)?;
        // Each node adds at most a few constants of at most `eval::MAX_ELEMENTS`, so checking
        // after each keeps a model whose constants fold into gigabytes (a thousand Concats of a
        // large weight) from getting there.
        if b.values > max_values {
            return err(format!("the model's constants come to more than {max_values} values once worked out"));
        }
    }
    let out = *b.ids.get(&output).ok_or("the output is never computed")?;
    if b.p.kind(out) != Kind::Run {
        return err("the output does not depend on the input");
    }
    b.p.output = out;
    b.find_best();
    Ok(b.p)
}

/// The f32 values of a constant (integers converted).
pub fn floats(p: &Prepared, c: Cid) -> Vec<f32> {
    match p.consts.get(c).map(|t| &t.data) {
        Some(Data::F(v)) => v.clone(),
        Some(Data::I(v)) => v.iter().map(|&i| i as f32).collect(),
        None => Vec::new(),
    }
}
