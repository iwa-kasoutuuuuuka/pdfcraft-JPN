//! A minimal ONNX reader: the protobuf wire format walked by hand (no new dependency), for the
//! parts the GPU kernels need: the opset, the graph's nodes with their attributes, its weights
//! (initializers and `Constant` nodes) and its input and output names. Nothing is run.
//!
//! The file is untrusted. Every length is checked against what is left of its message before
//! it is used; varints are at most 10 bytes; nested messages are read only to the depth the
//! schema has (subgraphs in attributes are skipped, never followed); and the counts of nodes,
//! attributes, tensors and weight values are capped ([`Limits`]), so a header cannot make it
//! allocate more than the caps allow. Field numbers follow onnx.proto (ModelProto,
//! GraphProto, NodeProto, AttributeProto, TensorProto, ValueInfoProto, OperatorSetIdProto).

use std::collections::{BTreeMap, HashMap};

/// What a file may hold, at most.
#[derive(Clone, Copy, Debug)]
pub struct Limits {
    pub nodes: usize,
    pub tensors: usize,
    pub attributes_per_node: usize,
    pub names_per_node: usize,
    pub dims: usize,
    /// Weight values (f32 and i64 together) over the whole file.
    pub values: usize,
}

impl Default for Limits {
    /// Well above PP-OCRv5's (about 2,000 nodes and 45 M weights each).
    fn default() -> Limits {
        Limits { nodes: 100_000, tensors: 100_000, attributes_per_node: 64, names_per_node: 256, dims: 8, values: 1 << 28 }
    }
}

/// Why a file was not read.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OnnxError(pub String);

impl std::fmt::Display for OnnxError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "not a readable ONNX model: {}", self.0)
    }
}

impl std::error::Error for OnnxError {}

fn bad<T>(why: impl Into<String>) -> Result<T, OnnxError> {
    Err(OnnxError(why.into()))
}

/// One protobuf field's value.
enum Value<'a> {
    Varint(u64),
    Fixed64,
    Bytes(&'a [u8]),
    Fixed32(u32),
}

struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(buf: &'a [u8]) -> Reader<'a> {
        Reader { buf, pos: 0 }
    }

    fn done(&self) -> bool {
        self.pos >= self.buf.len()
    }

    fn varint(&mut self) -> Result<u64, OnnxError> {
        let mut v = 0u64;
        for i in 0..10 {
            let Some(&b) = self.buf.get(self.pos) else { return bad("a number is cut off") };
            self.pos += 1;
            v |= u64::from(b & 0x7f) << (7 * i);
            if b & 0x80 == 0 {
                return Ok(v);
            }
        }
        bad("a number is longer than 10 bytes")
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8], OnnxError> {
        let end = self.pos.checked_add(n).filter(|&e| e <= self.buf.len());
        let Some(end) = end else { return bad("a field runs past the end of its message") };
        let s = self.buf.get(self.pos..end).unwrap_or(&[]);
        self.pos = end;
        Ok(s)
    }

    fn field(&mut self) -> Result<(u64, Value<'a>), OnnxError> {
        let key = self.varint()?;
        let number = key >> 3;
        if number == 0 {
            return bad("field number 0");
        }
        let value = match key & 7 {
            0 => Value::Varint(self.varint()?),
            1 => {
                self.take(8)?;
                Value::Fixed64
            }
            2 => {
                let n = usize::try_from(self.varint()?).map_err(|_| OnnxError("a length does not fit in memory".into()))?;
                Value::Bytes(self.take(n)?)
            }
            5 => {
                let b = self.take(4)?;
                Value::Fixed32(u32::from_le_bytes(<[u8; 4]>::try_from(b).map_err(|_| OnnxError("a fixed32 is cut off".into()))?))
            }
            w => return bad(format!("wire type {w} is not used by ONNX")),
        };
        Ok((number, value))
    }
}

fn text(b: &[u8]) -> String {
    String::from_utf8_lossy(b).into_owned()
}

/// The running count of weight values, against [`Limits::values`].
struct Budget {
    left: usize,
}

impl Budget {
    fn spend(&mut self, n: usize) -> Result<(), OnnxError> {
        match self.left.checked_sub(n) {
            Some(l) => {
                self.left = l;
                Ok(())
            }
            None => bad("more weight values than the limit"),
        }
    }
}

/// Repeated int64 (packed or not), into `out`, each one charged to `budget`.
fn push_ints(out: &mut Vec<i64>, v: &Value<'_>, budget: &mut Budget) -> Result<(), OnnxError> {
    match v {
        Value::Varint(x) => {
            budget.spend(1)?;
            out.push(*x as i64);
        }
        Value::Bytes(b) => {
            let mut r = Reader::new(b);
            while !r.done() {
                budget.spend(1)?;
                out.push(r.varint()? as i64);
            }
        }
        _ => return bad("an integer list has the wrong wire type"),
    }
    Ok(())
}

/// Repeated float (packed or not).
fn push_floats(out: &mut Vec<f32>, v: &Value<'_>, budget: &mut Budget) -> Result<(), OnnxError> {
    match v {
        Value::Fixed32(x) => {
            budget.spend(1)?;
            out.push(f32::from_bits(*x));
        }
        Value::Bytes(b) => {
            if b.len() % 4 != 0 {
                return bad("a packed float list is not a whole number of floats");
            }
            budget.spend(b.len() / 4)?;
            out.extend(b.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)));
        }
        _ => return bad("a float list has the wrong wire type"),
    }
    Ok(())
}

/// TensorProto data types this reader decodes; others keep their dims and no values.
pub const FLOAT: i64 = 1;
pub const INT64: i64 = 7;

/// A weight: its name, dims, ONNX data type and values (`floats` for FLOAT, `ints` for INT64).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Tensor {
    pub name: String,
    pub dims: Vec<usize>,
    pub data_type: i64,
    pub floats: Vec<f32>,
    pub ints: Vec<i64>,
}

impl Tensor {
    /// The element count its dims say.
    pub fn len(&self) -> Option<usize> {
        self.dims.iter().try_fold(1usize, |a, &d| a.checked_mul(d))
    }

    /// Whether it has no elements.
    pub fn is_empty(&self) -> bool {
        self.len() == Some(0)
    }
}

fn parse_tensor(b: &[u8], limits: &Limits, budget: &mut Budget) -> Result<Tensor, OnnxError> {
    let mut r = Reader::new(b);
    let mut t = Tensor::default();
    let mut dims = Vec::new();
    let mut raw: Option<&[u8]> = None;
    while !r.done() {
        let (number, v) = r.field()?;
        match (number, &v) {
            (1, _) => {
                let mut scratch = Budget { left: limits.dims.saturating_sub(dims.len()) };
                push_ints(&mut dims, &v, &mut scratch).map_err(|_| OnnxError(format!("a tensor has more than {} dims", limits.dims)))?;
            }
            (2, Value::Varint(x)) => t.data_type = *x as i64,
            (4, _) => push_floats(&mut t.floats, &v, budget)?,
            (7, _) => push_ints(&mut t.ints, &v, budget)?,
            (8, Value::Bytes(s)) => t.name = text(s),
            (9, Value::Bytes(s)) => raw = Some(s),
            (13, _) | (14, Value::Varint(1)) => return bad("external tensor data is not supported"),
            _ => {}
        }
    }
    t.dims =
        dims.iter().map(|&d| usize::try_from(d).map_err(|_| OnnxError(format!("tensor {} has a negative dim", t.name)))).collect::<Result<_, _>>()?;
    let n = t.len().ok_or_else(|| OnnxError(format!("tensor {}'s size overflows", t.name)))?;
    if let Some(raw) = raw {
        let width = match t.data_type {
            FLOAT => 4,
            INT64 => 8,
            _ => 0,
        };
        if width > 0 {
            if Some(raw.len()) != n.checked_mul(width) {
                return bad(format!("tensor {} holds {} bytes, its dims say {n} values", t.name, raw.len()));
            }
            budget.spend(n)?;
            if t.data_type == FLOAT {
                t.floats = raw.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect();
            } else {
                t.ints = raw.as_chunks::<8>().0.iter().map(|c| i64::from_le_bytes(*c)).collect();
            }
        }
    }
    let held = if t.data_type == INT64 { t.ints.len() } else { t.floats.len() };
    if matches!(t.data_type, FLOAT | INT64) && held != n {
        return bad(format!("tensor {} has {held} values, its dims say {n}", t.name));
    }
    Ok(t)
}

/// A node attribute's value (graphs are skipped: no subgraphs are followed).
#[derive(Clone, Debug, PartialEq)]
pub enum Attr {
    Int(i64),
    Float(f32),
    Ints(Vec<i64>),
    Floats(Vec<f32>),
    Str(String),
    Tensor(Tensor),
    /// A type this reader does not decode (graphs, sparse tensors, type protos).
    Other(i64),
}

fn parse_attr(b: &[u8], limits: &Limits, budget: &mut Budget) -> Result<(String, Attr), OnnxError> {
    let mut r = Reader::new(b);
    let (mut name, mut kind) = (String::new(), 0i64);
    let (mut f, mut i, mut s, mut t) = (0f32, 0i64, String::new(), None);
    let (mut fs, mut is) = (Vec::new(), Vec::new());
    while !r.done() {
        let (number, v) = r.field()?;
        match (number, &v) {
            (1, Value::Bytes(x)) => name = text(x),
            (2, Value::Fixed32(x)) => f = f32::from_bits(*x),
            (3, Value::Varint(x)) => i = *x as i64,
            (4, Value::Bytes(x)) => s = text(x),
            (5, Value::Bytes(x)) => t = Some(parse_tensor(x, limits, budget)?),
            (7, _) => push_floats(&mut fs, &v, budget)?,
            (8, _) => push_ints(&mut is, &v, budget)?,
            (20, Value::Varint(x)) => kind = *x as i64,
            _ => {}
        }
    }
    let a = match kind {
        1 => Attr::Float(f),
        2 => Attr::Int(i),
        3 => Attr::Str(s),
        4 => Attr::Tensor(t.ok_or_else(|| OnnxError(format!("attribute {name} is a tensor without one")))?),
        6 => Attr::Floats(fs),
        7 => Attr::Ints(is),
        other => Attr::Other(other),
    };
    Ok((name, a))
}

/// One operator in the graph.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Node {
    pub name: String,
    pub op: String,
    pub domain: String,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub attrs: BTreeMap<String, Attr>,
}

impl Node {
    /// An integer attribute, or `default`.
    pub fn int(&self, key: &str, default: i64) -> i64 {
        match self.attrs.get(key) {
            Some(Attr::Int(v)) => *v,
            _ => default,
        }
    }

    /// A float attribute, or `default`.
    pub fn float(&self, key: &str, default: f32) -> f32 {
        match self.attrs.get(key) {
            Some(Attr::Float(v)) => *v,
            _ => default,
        }
    }

    /// An integer list attribute (empty when absent).
    pub fn ints(&self, key: &str) -> &[i64] {
        match self.attrs.get(key) {
            Some(Attr::Ints(v)) => v,
            _ => &[],
        }
    }
}

fn parse_node(b: &[u8], limits: &Limits, budget: &mut Budget) -> Result<Node, OnnxError> {
    let mut r = Reader::new(b);
    let mut n = Node::default();
    while !r.done() {
        let (number, v) = r.field()?;
        match (number, &v) {
            (1, Value::Bytes(x)) => n.inputs.push(text(x)),
            (2, Value::Bytes(x)) => n.outputs.push(text(x)),
            (3, Value::Bytes(x)) => n.name = text(x),
            (4, Value::Bytes(x)) => n.op = text(x),
            (7, Value::Bytes(x)) => n.domain = text(x),
            (5, Value::Bytes(x)) => {
                if n.attrs.len() == limits.attributes_per_node {
                    return bad(format!("node {} has more than {} attributes", n.name, limits.attributes_per_node));
                }
                let (k, a) = parse_attr(x, limits, budget)?;
                n.attrs.insert(k, a);
            }
            _ => {}
        }
        if n.inputs.len() + n.outputs.len() > limits.names_per_node {
            return bad(format!("node {} has more than {} inputs and outputs", n.name, limits.names_per_node));
        }
    }
    Ok(n)
}

fn value_info_name(b: &[u8]) -> Result<String, OnnxError> {
    let mut r = Reader::new(b);
    while !r.done() {
        if let (1, Value::Bytes(v)) = r.field()? {
            return Ok(text(v));
        }
    }
    Ok(String::new())
}

/// The graph: nodes in file order, weights, and the names of its inputs and outputs.
#[derive(Clone, Debug, Default)]
pub struct Graph {
    pub nodes: Vec<Node>,
    pub tensors: Vec<Tensor>,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    /// Tensor name → index in `tensors` (initializers and `Constant` outputs).
    index: HashMap<String, usize>,
    /// An `Identity` output → its input, for names that end at a tensor.
    aliases: HashMap<String, String>,
}

impl Graph {
    /// The weight named `name`, through any chain of `Identity` nodes.
    pub fn tensor(&self, name: &str) -> Option<&Tensor> {
        let mut name = name;
        // A chain is at most as long as the node list (a cycle ends the walk).
        for _ in 0..=self.nodes.len() {
            if let Some(&i) = self.index.get(name) {
                return self.tensors.get(i);
            }
            name = self.aliases.get(name)?;
        }
        None
    }

    /// How often each operator occurs (`Identity` and `Constant` included).
    pub fn op_counts(&self) -> BTreeMap<String, usize> {
        let mut out = BTreeMap::new();
        for n in &self.nodes {
            *out.entry(n.op.clone()).or_insert(0) += 1;
        }
        out
    }
}

/// A model: its IR version, its operator sets as (domain, version), and its graph.
#[derive(Clone, Debug, Default)]
pub struct Model {
    pub ir_version: i64,
    pub opsets: Vec<(String, i64)>,
    pub graph: Graph,
}

impl Model {
    /// The version of the default operator set ("" or "ai.onnx").
    pub fn opset(&self) -> Option<i64> {
        self.opsets.iter().find(|(d, _)| d.is_empty() || d == "ai.onnx").map(|(_, v)| *v)
    }
}

/// Read a model with the default [`Limits`].
pub fn parse(bytes: &[u8]) -> Result<Model, OnnxError> {
    parse_with(bytes, &Limits::default())
}

/// Read a model within `limits`.
pub fn parse_with(bytes: &[u8], limits: &Limits) -> Result<Model, OnnxError> {
    let mut budget = Budget { left: limits.values };
    let mut model = Model::default();
    let mut graph = None;
    let mut r = Reader::new(bytes);
    while !r.done() {
        match r.field()? {
            (1, Value::Varint(v)) => model.ir_version = v as i64,
            (7, Value::Bytes(g)) => graph = Some(g),
            (8, Value::Bytes(o)) => {
                if model.opsets.len() == 64 {
                    return bad("more than 64 operator sets");
                }
                let mut r = Reader::new(o);
                let (mut domain, mut version) = (String::new(), 0i64);
                while !r.done() {
                    match r.field()? {
                        (1, Value::Bytes(d)) => domain = text(d),
                        (2, Value::Varint(v)) => version = v as i64,
                        _ => {}
                    }
                }
                model.opsets.push((domain, version));
            }
            _ => {}
        }
    }
    let Some(g) = graph else { return bad("the model has no graph") };
    let out = &mut model.graph;
    let mut r = Reader::new(g);
    while !r.done() {
        match r.field()? {
            (1, Value::Bytes(v)) => {
                if out.nodes.len() == limits.nodes {
                    return bad(format!("more than {} nodes", limits.nodes));
                }
                out.nodes.push(parse_node(v, limits, &mut budget)?);
            }
            (5, Value::Bytes(v)) => {
                if out.tensors.len() == limits.tensors {
                    return bad(format!("more than {} weights", limits.tensors));
                }
                out.tensors.push(parse_tensor(v, limits, &mut budget)?);
            }
            (11, Value::Bytes(v)) => out.inputs.push(value_info_name(v)?),
            (12, Value::Bytes(v)) => out.outputs.push(value_info_name(v)?),
            _ => {}
        }
        if out.inputs.len() + out.outputs.len() > limits.names_per_node {
            return bad("too many graph inputs and outputs");
        }
    }
    // Constant nodes' values join the weights (moved, not copied).
    for n in &mut out.nodes {
        if n.op == "Constant"
            && let Some(o) = n.outputs.first().cloned()
            && let Some(Attr::Tensor(t)) = n.attrs.remove("value")
        {
            if out.tensors.len() == limits.tensors {
                return bad(format!("more than {} weights", limits.tensors));
            }
            out.tensors.push(Tensor { name: o, ..t });
        }
    }
    for (i, t) in out.tensors.iter().enumerate() {
        out.index.entry(t.name.clone()).or_insert(i);
    }
    for n in &out.nodes {
        if n.op == "Identity"
            && let (Some(i), Some(o)) = (n.inputs.first(), n.outputs.first())
        {
            out.aliases.insert(o.clone(), i.clone());
        }
    }
    Ok(model)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A protobuf writer for test models.
    #[derive(Default)]
    struct W(Vec<u8>);

    impl W {
        fn varint(&mut self, mut v: u64) -> &mut W {
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
        fn int(&mut self, field: u64, v: i64) -> &mut W {
            self.varint(field << 3).varint(v as u64)
        }
        fn bytes(&mut self, field: u64, b: &[u8]) -> &mut W {
            self.varint(field << 3 | 2).varint(b.len() as u64);
            self.0.extend_from_slice(b);
            self
        }
        fn str(&mut self, field: u64, s: &str) -> &mut W {
            self.bytes(field, s.as_bytes())
        }
        fn msg(&mut self, field: u64, m: &W) -> &mut W {
            self.bytes(field, &m.0)
        }
    }

    fn tensor(name: &str, dims: &[i64], data: &[f32]) -> W {
        let mut t = W::default();
        for &d in dims {
            t.int(1, d);
        }
        t.int(2, FLOAT).str(8, name);
        let raw: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
        t.bytes(9, &raw);
        t
    }

    /// A small model: Conv(x, Identity(w)) with ints and a float attribute, a Constant node,
    /// opset 11.
    fn model() -> Vec<u8> {
        let mut strides = W::default();
        strides.str(1, "strides").int(20, 7);
        let mut packed = W::default();
        packed.varint(2).varint(2);
        strides.bytes(8, &packed.0);
        let mut alpha = W::default();
        alpha.str(1, "alpha").int(20, 1);
        alpha.varint(2 << 3 | 5);
        alpha.0.extend_from_slice(&0.5f32.to_le_bytes());
        let mut conv = W::default();
        conv.str(1, "x").str(1, "w_id").str(2, "y").str(3, "conv0").str(4, "Conv").msg(5, &strides).msg(5, &alpha);
        let mut ident = W::default();
        ident.str(1, "w").str(2, "w_id").str(4, "Identity");
        let mut value = W::default();
        value.str(1, "value").int(20, 4).msg(5, &tensor("ignored", &[2], &[1.0, 2.0]));
        let mut constant = W::default();
        constant.str(2, "k").str(4, "Constant").msg(5, &value);
        let mut input = W::default();
        input.str(1, "x");
        let mut output = W::default();
        output.str(1, "y");
        let mut graph = W::default();
        graph.msg(1, &ident).msg(1, &conv).msg(1, &constant).msg(5, &tensor("w", &[1, 1, 1, 2], &[0.25, -1.0])).msg(11, &input).msg(12, &output);
        let mut opset = W::default();
        opset.str(1, "").int(2, 11);
        let mut m = W::default();
        m.int(1, 7).msg(7, &graph).msg(8, &opset);
        m.0
    }

    #[test]
    fn a_small_model_reads_with_its_attributes_weights_and_aliases() {
        let m = parse(&model()).unwrap();
        assert_eq!((m.ir_version, m.opset()), (7, Some(11)));
        let g = &m.graph;
        assert_eq!((g.inputs.as_slice(), g.outputs.as_slice()), (&["x".to_string()][..], &["y".to_string()][..]));
        let conv = g.nodes.iter().find(|n| n.op == "Conv").unwrap();
        assert_eq!(conv.ints("strides"), [2, 2]);
        assert_eq!(conv.float("alpha", 0.0), 0.5);
        assert_eq!(conv.int("group", 1), 1, "absent attributes give the default");
        let w = g.tensor(&conv.inputs[1]).unwrap();
        assert_eq!((w.name.as_str(), w.dims.as_slice(), w.floats.as_slice()), ("w", &[1, 1, 1, 2][..], &[0.25, -1.0][..]));
        assert_eq!(g.tensor("k").map(|t| t.floats.clone()), Some(vec![1.0, 2.0]), "a Constant's value is a weight under its output's name");
        assert_eq!(
            g.op_counts().into_iter().collect::<Vec<_>>(),
            [("Constant".to_string(), 1), ("Conv".to_string(), 1), ("Identity".to_string(), 1)]
        );
        assert!(g.tensor("nothing").is_none());
    }

    /// Every cut of a good file, and every single-byte change of it, is read or refused, never
    /// a panic; random bytes too.
    #[test]
    fn truncated_altered_and_random_bytes_never_panic() {
        let good = model();
        for cut in 0..good.len() {
            let _ = parse(&good[..cut]);
        }
        for i in 0..good.len() {
            for b in [0x00u8, 0x7f, 0x80, 0xff] {
                let mut bytes = good.clone();
                bytes[i] = b;
                let _ = parse(&bytes);
            }
        }
        let mut seed = 0x2545_f491_4f6c_dd1du64;
        for len in 0..3000 {
            let bytes: Vec<u8> = (0..len % 300)
                .map(|_| {
                    seed ^= seed << 13;
                    seed ^= seed >> 7;
                    seed ^= seed << 17;
                    (seed >> 24) as u8
                })
                .collect();
            let _ = parse(&bytes);
        }
        assert!(parse(&[]).is_err(), "no graph");
    }

    /// Headers that promise more than the file holds or the limits allow are refused before
    /// anything that size is allocated.
    #[test]
    fn hostile_sizes_and_counts_are_refused() {
        let wrap = |graph: &W| {
            let mut m = W::default();
            m.msg(7, graph);
            m.0
        };
        // A length far past the end.
        let mut m = W::default();
        m.varint(7 << 3 | 2).varint(u64::MAX >> 1);
        assert!(parse(&m.0).is_err());
        // An 11-byte varint.
        assert!(parse(&[0x08, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x01]).is_err());
        // Dims that multiply past usize, or say more values than the raw bytes hold.
        let mut g = W::default();
        g.msg(5, &tensor("big", &[1 << 40, 1 << 40, 1 << 40], &[]));
        assert!(parse(&wrap(&g)).is_err());
        let mut g = W::default();
        g.msg(5, &tensor("short", &[4], &[1.0]));
        assert!(parse(&wrap(&g)).unwrap_err().0.contains("short"));
        let mut g = W::default();
        g.msg(5, &tensor("neg", &[-1], &[]));
        assert!(parse(&wrap(&g)).is_err());
        // Over the caps: nodes, weight values, attributes per node, dims.
        let limits = Limits { nodes: 2, values: 3, attributes_per_node: 1, dims: 2, ..Limits::default() };
        let mut node = W::default();
        node.str(4, "Relu");
        let mut g = W::default();
        g.msg(1, &node).msg(1, &node).msg(1, &node);
        assert!(parse_with(&wrap(&g), &limits).unwrap_err().0.contains("nodes"));
        let mut g = W::default();
        g.msg(5, &tensor("w", &[4], &[1.0, 2.0, 3.0, 4.0]));
        assert!(parse_with(&wrap(&g), &limits).unwrap_err().0.contains("weight values"));
        let mut a = W::default();
        a.str(1, "a").int(20, 2).int(3, 1);
        let mut node = W::default();
        node.str(4, "Relu").msg(5, &a).msg(5, &a);
        let mut g = W::default();
        g.msg(1, &node);
        assert!(parse_with(&wrap(&g), &limits).unwrap_err().0.contains("attributes"));
        let mut g = W::default();
        g.msg(5, &tensor("w", &[1, 1, 1], &[1.0]));
        assert!(parse_with(&wrap(&g), &limits).unwrap_err().0.contains("dims"));
        // External data is refused, not followed.
        let mut t = tensor("ext", &[1], &[1.0]);
        t.int(14, 1);
        let mut g = W::default();
        g.msg(5, &t);
        assert!(parse(&wrap(&g)).unwrap_err().0.contains("external"));
    }

    /// A cycle of Identity nodes ends the alias walk instead of looping.
    #[test]
    fn identity_cycles_end() {
        let mut a = W::default();
        a.str(1, "b").str(2, "a").str(4, "Identity");
        let mut b = W::default();
        b.str(1, "a").str(2, "b").str(4, "Identity");
        let mut g = W::default();
        g.msg(1, &a).msg(1, &b);
        let mut m = W::default();
        m.msg(7, &g);
        assert!(parse(&m.0).unwrap().graph.tensor("a").is_none());
    }
}
