//! A writer of small ONNX models for tests (protobuf by hand, field numbers from onnx.proto).

#[derive(Default, Clone)]
pub struct W(pub Vec<u8>);

impl W {
    pub fn varint(&mut self, mut v: u64) -> &mut W {
        loop {
            let b = (v & 0x7f) as u8;
            v >>= 7;
            if v == 0 {
                self.0.push(b);
                return self;
            }
            self.0.push(b | 0x80);
        }
    }
    pub fn int(&mut self, field: u64, v: i64) -> &mut W {
        self.varint(field << 3).varint(v as u64)
    }
    pub fn bytes(&mut self, field: u64, b: &[u8]) -> &mut W {
        self.varint(field << 3 | 2).varint(b.len() as u64);
        self.0.extend_from_slice(b);
        self
    }
    pub fn str(&mut self, field: u64, s: &str) -> &mut W {
        self.bytes(field, s.as_bytes())
    }
    pub fn msg(&mut self, field: u64, m: &W) -> &mut W {
        self.bytes(field, &m.0)
    }
}

/// An attribute value.
pub enum A {
    I(i64),
    F(f32),
    Is(Vec<i64>),
    S(&'static str),
}

fn tensor_f(name: &str, dims: &[usize], data: &[f32]) -> W {
    let mut t = W::default();
    for &d in dims {
        t.int(1, d as i64);
    }
    t.int(2, 1).str(8, name);
    let raw: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
    t.bytes(9, &raw);
    t
}

fn tensor_i(name: &str, dims: &[usize], data: &[i64]) -> W {
    let mut t = W::default();
    for &d in dims {
        t.int(1, d as i64);
    }
    t.int(2, 7).str(8, name);
    let raw: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
    t.bytes(9, &raw);
    t
}

/// A model under construction: one float input `x`, one output.
pub struct M {
    nodes: Vec<W>,
    inits: Vec<W>,
    output: String,
    opset: i64,
    count: usize,
}

impl M {
    pub fn new(opset: i64) -> M {
        M { nodes: Vec::new(), inits: Vec::new(), output: String::new(), opset, count: 0 }
    }

    /// Add a node; its output is `out` and becomes the model's output.
    pub fn node(&mut self, op: &str, inputs: &[&str], out: &str, attrs: &[(&str, A)]) -> &mut M {
        let mut n = W::default();
        for i in inputs {
            n.str(1, i);
        }
        n.str(2, out).str(3, &format!("{op}.{}", self.count)).str(4, op);
        self.count += 1;
        // rten wants the window attributes spelled out.
        let has = |k: &str| attrs.iter().any(|(n, _)| *n == k);
        let mut extra: Vec<(&str, A)> = Vec::new();
        if matches!(op, "Conv" | "ConvTranspose" | "MaxPool" | "AveragePool") {
            if !has("strides") {
                extra.push(("strides", A::Is(vec![1, 1])));
            }
            if !has("pads") && !has("auto_pad") {
                extra.push(("pads", A::Is(vec![0, 0, 0, 0])));
            }
        }
        if matches!(op, "Conv" | "ConvTranspose") && !has("dilations") {
            extra.push(("dilations", A::Is(vec![1, 1])));
        }
        for (name, a) in attrs.iter().map(|(n, a)| (*n, a)).chain(extra.iter().map(|(n, a)| (*n, a))) {
            let mut at = W::default();
            at.str(1, name);
            match a {
                A::I(v) => {
                    at.int(20, 2).int(3, *v);
                }
                A::F(v) => {
                    at.int(20, 1).varint(2 << 3 | 5);
                    at.0.extend_from_slice(&v.to_le_bytes());
                }
                A::Is(v) => {
                    at.int(20, 7);
                    for x in v {
                        at.int(8, *x);
                    }
                }
                A::S(s) => {
                    at.int(20, 3).str(4, s);
                }
            }
            n.msg(5, &at);
        }
        self.nodes.push(n);
        self.output = out.to_string();
        self
    }

    pub fn init(&mut self, name: &str, dims: &[usize], data: &[f32]) -> &mut M {
        self.inits.push(tensor_f(name, dims, data));
        self
    }

    pub fn init_i(&mut self, name: &str, dims: &[usize], data: &[i64]) -> &mut M {
        self.inits.push(tensor_i(name, dims, data));
        self
    }

    /// A weight of `dims` with values from a fixed pseudo-random sequence, scaled.
    pub fn weight(&mut self, name: &str, dims: &[usize], scale: f32) -> &mut M {
        let n: usize = dims.iter().product();
        let data: Vec<f32> = (0..n).map(|i| (((i * 7919 + name.len() * 104729) % 2003) as f32 / 1001.5 - 1.0) * scale).collect();
        self.init(name, dims, &data)
    }

    pub fn bytes(&self) -> Vec<u8> {
        let value_info = |name: &str| {
            let mut tt = W::default();
            tt.int(1, 1);
            let mut ty = W::default();
            ty.msg(1, &tt);
            let mut v = W::default();
            v.str(1, name).msg(2, &ty);
            v
        };
        let mut g = W::default();
        for n in &self.nodes {
            g.msg(1, n);
        }
        g.str(2, "test");
        for t in &self.inits {
            g.msg(5, t);
        }
        g.msg(11, &value_info("x")).msg(12, &value_info(&self.output));
        let mut opset = W::default();
        opset.str(1, "").int(2, self.opset);
        let mut m = W::default();
        m.int(1, 7).str(2, "pdfcraft-test").msg(7, &g).msg(8, &opset);
        m.0
    }
}

/// A fixed input for `dims`: a smooth pattern in -1..1.
pub fn input(dims: [usize; 4]) -> Vec<f32> {
    let n: usize = dims.iter().product();
    (0..n).map(|i| ((i % 613) as f32 * 0.41).sin()).collect()
}
