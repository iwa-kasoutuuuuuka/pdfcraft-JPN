//! The small tensors worked out on the CPU: weights folded at load (`Constant` chains), and the
//! shape arithmetic a graph does at run time (`Shape` → `Slice` → `Concat` → `Reshape`), which
//! is known once the input's size is. Only the operators such graphs use, with every size
//! checked; anything else is an error naming the operator.

use crate::onnx;

/// The most elements one worked-out tensor may have.
pub const MAX_ELEMENTS: usize = 1 << 24;

/// Values: f32 (FLOAT) or i64 (INT64).
#[derive(Clone, Debug, PartialEq)]
pub enum Data {
    F(Vec<f32>),
    I(Vec<i64>),
}

/// A tensor held on the CPU.
#[derive(Clone, Debug, PartialEq)]
pub struct Cpu {
    pub dims: Vec<usize>,
    pub data: Data,
}

impl Cpu {
    pub fn floats(dims: Vec<usize>, v: Vec<f32>) -> Cpu {
        Cpu { dims, data: Data::F(v) }
    }

    pub fn ints(dims: Vec<usize>, v: Vec<i64>) -> Cpu {
        Cpu { dims, data: Data::I(v) }
    }

    /// A weight of the model, when its type is one held here.
    pub fn of(t: &onnx::Tensor) -> Result<Cpu, String> {
        let n = count(&t.dims)?;
        let data = match t.data_type {
            onnx::FLOAT if t.floats.len() == n => Data::F(t.floats.clone()),
            onnx::INT64 if t.ints.len() == n => Data::I(t.ints.clone()),
            other => {
                return Err(format!(
                    "weight {} has data type {other} with {} values for dims {:?}",
                    t.name,
                    t.floats.len().max(t.ints.len()),
                    t.dims
                ));
            }
        };
        Ok(Cpu { dims: t.dims.clone(), data })
    }

    pub fn len(&self) -> usize {
        match &self.data {
            Data::F(v) => v.len(),
            Data::I(v) => v.len(),
        }
    }

    /// The values as integers (floats must be whole numbers).
    pub fn as_ints(&self) -> Result<Vec<i64>, String> {
        match &self.data {
            Data::I(v) => Ok(v.clone()),
            Data::F(v) => v
                .iter()
                .map(
                    |&f| if f.is_finite() && f.fract() == 0.0 && f.abs() < 9.0e15 { Ok(f as i64) } else { Err(format!("{f} is not a whole number")) },
                )
                .collect(),
        }
    }

    /// The values as floats.
    pub fn as_floats(&self) -> Vec<f32> {
        match &self.data {
            Data::F(v) => v.clone(),
            Data::I(v) => v.iter().map(|&i| i as f32).collect(),
        }
    }

    /// The single value of a one-element tensor, as a float.
    pub fn scalar(&self) -> Option<f32> {
        if self.len() == 1 { self.as_floats().first().copied() } else { None }
    }
}

/// The element count of `dims`, checked against [`MAX_ELEMENTS`]-free overflow.
pub fn count(dims: &[usize]) -> Result<usize, String> {
    dims.iter().try_fold(1usize, |a, &d| a.checked_mul(d)).ok_or_else(|| format!("the size {dims:?} overflows"))
}

/// The element count of `dims`, refused when it, or the product with every empty dimension
/// counted as 1, passes [`MAX_ELEMENTS`]. The second bound keeps loops over an empty tensor's
/// other dimensions (`[1 << 62, 0]`) short, and is checked before anything is allocated.
pub fn bounded(dims: &[usize]) -> Result<usize, String> {
    let loose = dims.iter().try_fold(1usize, |a, &d| a.checked_mul(d.max(1)));
    match loose {
        Some(n) if n <= MAX_ELEMENTS => count(dims),
        _ => Err(format!("a worked-out tensor of size {dims:?} is larger than {MAX_ELEMENTS} values")),
    }
}

/// Row-major strides of `dims`.
pub fn strides(dims: &[usize]) -> Vec<usize> {
    let mut s = vec![1usize; dims.len()];
    for i in (0..dims.len().saturating_sub(1)).rev() {
        let next = s.get(i + 1).copied().unwrap_or(1).saturating_mul(dims.get(i + 1).copied().unwrap_or(1));
        if let Some(v) = s.get_mut(i) {
            *v = next;
        }
    }
    s
}

/// The numpy broadcast of two shapes.
pub fn broadcast(a: &[usize], b: &[usize]) -> Result<Vec<usize>, String> {
    let rank = a.len().max(b.len());
    let mut out = Vec::with_capacity(rank);
    for i in 0..rank {
        let da = if i + a.len() >= rank { a.get(i + a.len() - rank).copied().unwrap_or(1) } else { 1 };
        let db = if i + b.len() >= rank { b.get(i + b.len() - rank).copied().unwrap_or(1) } else { 1 };
        out.push(match (da, db) {
            (x, y) if x == y => x,
            (1, y) => y,
            (x, 1) => x,
            _ => return Err(format!("shapes {a:?} and {b:?} do not broadcast")),
        });
    }
    Ok(out)
}

/// The strides that read `shape` broadcast to `out` (0 along the broadcast dimensions).
pub fn broadcast_strides(shape: &[usize], out: &[usize]) -> Vec<usize> {
    let own = strides(shape);
    let lead = out.len().saturating_sub(shape.len());
    (0..out.len())
        .map(|i| {
            if i < lead {
                return 0;
            }
            let j = i - lead;
            if shape.get(j).copied().unwrap_or(1) == 1 { 0 } else { own.get(j).copied().unwrap_or(0) }
        })
        .collect()
}

/// An axis attribute (negative counts from the end) as an index below `rank`.
pub fn axis(a: i64, rank: usize) -> Result<usize, String> {
    let r = rank as i64;
    let v = if a < 0 { a + r } else { a };
    if (0..r).contains(&v) { Ok(v as usize) } else { Err(format!("axis {a} is outside rank {rank}")) }
}

/// One sliced dimension: first index, step, length.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Span {
    pub start: usize,
    pub step: i64,
    pub len: usize,
}

/// ONNX Slice (opset 10+): starts, ends, optional axes and steps, clamped as the spec says.
pub fn slice_spans(dims: &[usize], starts: &[i64], ends: &[i64], axes: Option<&[i64]>, steps: Option<&[i64]>) -> Result<Vec<Span>, String> {
    let mut spans: Vec<Span> = dims.iter().map(|&d| Span { start: 0, step: 1, len: d }).collect();
    if starts.len() != ends.len() {
        return Err("Slice has different numbers of starts and ends".into());
    }
    for (i, (&s, &e)) in starts.iter().zip(ends).enumerate() {
        let ax = match axes {
            Some(a) => axis(*a.get(i).ok_or("Slice has fewer axes than starts")?, dims.len())?,
            None => i,
        };
        let step = match steps {
            Some(st) => *st.get(i).ok_or("Slice has fewer steps than starts")?,
            None => 1,
        };
        if step == 0 {
            return Err("Slice step 0".into());
        }
        let d = i128::from(dims.get(ax).copied().ok_or("Slice axis outside the shape")? as u64);
        let fix = |v: i64| -> i128 { if v < 0 { i128::from(v) + d } else { i128::from(v) } };
        let (start, end) = if step > 0 { (fix(s).clamp(0, d), fix(e).clamp(0, d)) } else { (fix(s).clamp(0, d - 1), fix(e).clamp(-1, d - 1)) };
        let st = i128::from(step);
        let len = if step > 0 { (end - start + st - 1).div_euclid(st) } else { (start - end + (-st) - 1).div_euclid(-st) };
        let len = len.max(0);
        if let Some(sp) = spans.get_mut(ax) {
            *sp = Span {
                start: usize::try_from(start.max(0)).map_err(|_| "Slice start")?,
                step,
                len: usize::try_from(len).map_err(|_| "Slice length")?,
            };
        }
    }
    Ok(spans)
}

/// ONNX Reshape's target: 0 keeps the input's dimension, one -1 is worked out.
pub fn reshape_dims(input: &[usize], target: &[i64]) -> Result<Vec<usize>, String> {
    let total = count(input)?;
    let mut out = Vec::with_capacity(target.len());
    let mut infer = None;
    for (i, &t) in target.iter().enumerate() {
        out.push(match t {
            0 => *input.get(i).ok_or("Reshape copies a dimension the input does not have")?,
            -1 if infer.is_none() => {
                infer = Some(i);
                1
            }
            t if t > 0 => usize::try_from(t).map_err(|_| "Reshape dimension")?,
            t => return Err(format!("Reshape dimension {t}")),
        });
    }
    let known = count(&out)?;
    if let Some(i) = infer {
        if known == 0 || total % known != 0 {
            return Err(format!("Reshape of {input:?} to {target:?}"));
        }
        if let Some(v) = out.get_mut(i) {
            *v = total / known;
        }
    }
    if count(&out)? != total {
        return Err(format!("Reshape of {input:?} to {target:?} changes the size"));
    }
    Ok(out)
}

/// Squeeze: the listed axes (or every 1) removed.
pub fn squeeze_dims(input: &[usize], axes: &[i64]) -> Result<Vec<usize>, String> {
    let drop: Vec<usize> = if axes.is_empty() {
        input.iter().enumerate().filter(|(_, d)| **d == 1).map(|(i, _)| i).collect()
    } else {
        axes.iter().map(|&a| axis(a, input.len())).collect::<Result<_, _>>()?
    };
    for &a in &drop {
        if input.get(a) != Some(&1) {
            return Err(format!("Squeeze of axis {a} of {input:?}"));
        }
    }
    Ok(input.iter().enumerate().filter(|(i, _)| !drop.contains(i)).map(|(_, &d)| d).collect())
}

/// Unsqueeze: a 1 inserted at each listed axis of the output.
pub fn unsqueeze_dims(input: &[usize], axes: &[i64]) -> Result<Vec<usize>, String> {
    let rank = input.len() + axes.len();
    let mut at: Vec<usize> = axes.iter().map(|&a| axis(a, rank)).collect::<Result<_, _>>()?;
    at.sort_unstable();
    at.dedup();
    if at.len() != axes.len() {
        return Err("Unsqueeze repeats an axis".into());
    }
    let mut src = input.iter();
    (0..rank).map(|i| if at.contains(&i) { Ok(1) } else { src.next().copied().ok_or_else(|| "Unsqueeze".to_string()) }).collect()
}

/// The `axes` of Squeeze or Unsqueeze: the attribute (opset < 13) or the second input.
pub fn axes_of(node: &onnx::Node, second: Option<&Cpu>) -> Result<Vec<i64>, String> {
    match second {
        Some(t) => t.as_ints(),
        None => Ok(node.ints("axes").to_vec()),
    }
}

fn gather_slice<T: Copy>(v: &[T], dims: &[usize], spans: &[Span]) -> Result<Vec<T>, String> {
    let out_dims: Vec<usize> = spans.iter().map(|s| s.len).collect();
    let n = bounded(&out_dims)?;
    let st = strides(dims);
    let mut out = Vec::with_capacity(n);
    for flat in 0..n {
        let mut rem = flat;
        let mut src: i128 = 0;
        for (d, sp) in spans.iter().enumerate().rev() {
            let len = sp.len.max(1);
            let c = rem % len;
            rem /= len;
            let pos = sp.start as i128 + c as i128 * i128::from(sp.step);
            src += pos * st.get(d).copied().unwrap_or(0) as i128;
        }
        let i = usize::try_from(src).map_err(|_| "Slice index")?;
        out.push(*v.get(i).ok_or("Slice reads outside its input")?);
    }
    Ok(out)
}

/// Gather along an axis of `len` (`outer` blocks before it, `inner` values after it); the
/// picks are already within the axis, and the output size already bounded.
fn gather_axis<T: Copy>(v: &[T], outer: usize, len: usize, inner: usize, picks: &[i64]) -> Result<Vec<T>, String> {
    let mut out = Vec::with_capacity(outer.saturating_mul(picks.len()).saturating_mul(inner).min(MAX_ELEMENTS));
    for o in 0..outer {
        for &i in picks {
            let at =
                o.checked_mul(len).and_then(|x| x.checked_add(usize::try_from(i).ok()?)).and_then(|x| x.checked_mul(inner)).ok_or("Gather index")?;
            let end = at.checked_add(inner).ok_or("Gather index")?;
            out.extend_from_slice(v.get(at..end).ok_or("Gather reads outside its input")?);
        }
    }
    Ok(out)
}

fn binary_values<T: Copy>(
    a: &[T],
    ad: &[usize],
    b: &[T],
    bd: &[usize],
    f: impl Fn(T, T) -> Result<T, String>,
) -> Result<(Vec<usize>, Vec<T>), String> {
    let out = broadcast(ad, bd)?;
    let n = bounded(&out)?;
    let (sa, sb, so) = (broadcast_strides(ad, &out), broadcast_strides(bd, &out), strides(&out));
    let mut v = Vec::with_capacity(n);
    for flat in 0..n {
        let (mut ia, mut ib) = (0usize, 0usize);
        for d in 0..out.len() {
            let c = (flat / so.get(d).copied().unwrap_or(1).max(1)) % out.get(d).copied().unwrap_or(1).max(1);
            ia += c * sa.get(d).copied().unwrap_or(0);
            ib += c * sb.get(d).copied().unwrap_or(0);
        }
        v.push(f(*a.get(ia).ok_or("broadcast read")?, *b.get(ib).ok_or("broadcast read")?)?);
    }
    Ok((out, v))
}

/// Work out one node whose inputs are all known. `inputs` follow the node's (absent optional
/// inputs are `None`).
pub fn eval(node: &onnx::Node, inputs: &[Option<&Cpu>]) -> Result<Cpu, String> {
    let arg = |i: usize| inputs.get(i).copied().flatten();
    for t in inputs.iter().flatten() {
        bounded(&t.dims)?;
    }
    let first = arg(0).ok_or_else(|| format!("{} has no input", node.op))?;
    let out = match node.op.as_str() {
        "Identity" => first.clone(),
        "Reshape" => {
            let target = arg(1).ok_or("Reshape has no shape")?.as_ints()?;
            Cpu { dims: reshape_dims(&first.dims, &target)?, data: first.data.clone() }
        }
        "Flatten" => {
            let ax = eval_axis_incl(node.int("axis", 1), first.dims.len())?;
            let a = count(first.dims.get(..ax).unwrap_or(&[]))?;
            let b = count(first.dims.get(ax..).unwrap_or(&[]))?;
            Cpu { dims: vec![a, b], data: first.data.clone() }
        }
        "Squeeze" => Cpu { dims: squeeze_dims(&first.dims, &axes_of(node, arg(1))?)?, data: first.data.clone() },
        "Unsqueeze" => Cpu { dims: unsqueeze_dims(&first.dims, &axes_of(node, arg(1))?)?, data: first.data.clone() },
        "Concat" => {
            let parts: Vec<&Cpu> = inputs.iter().flatten().copied().collect();
            concat(&parts, node.int("axis", 0))?
        }
        "Slice" => {
            let (starts, ends) = (arg(1).ok_or("Slice has no starts")?.as_ints()?, arg(2).ok_or("Slice has no ends")?.as_ints()?);
            let axes = arg(3).map(Cpu::as_ints).transpose()?;
            let steps = arg(4).map(Cpu::as_ints).transpose()?;
            let spans = slice_spans(&first.dims, &starts, &ends, axes.as_deref(), steps.as_deref())?;
            let dims = spans.iter().map(|s| s.len).collect();
            let data = match &first.data {
                Data::F(v) => Data::F(gather_slice(v, &first.dims, &spans)?),
                Data::I(v) => Data::I(gather_slice(v, &first.dims, &spans)?),
            };
            Cpu { dims, data }
        }
        "Gather" => {
            let idx = arg(1).ok_or("Gather has no indices")?;
            let ax = axis(node.int("axis", 0), first.dims.len())?;
            let len = first.dims.get(ax).copied().unwrap_or(0) as i64;
            let picks: Vec<i64> = idx.as_ints()?.into_iter().map(|i| if i < 0 { i + len } else { i }).collect();
            if picks.iter().any(|&i| !(0..len).contains(&i)) {
                return Err("Gather index outside the axis".into());
            }
            // The index tensor's shape replaces the axis.
            let mut dims: Vec<usize> = first.dims.get(..ax).unwrap_or(&[]).to_vec();
            dims.extend_from_slice(&idx.dims);
            dims.extend_from_slice(first.dims.get(ax + 1..).unwrap_or(&[]));
            // Checked before anything is allocated: a few indices into a large axis are cheap,
            // many into a wide one are not.
            bounded(&dims)?;
            let outer = count(first.dims.get(..ax).unwrap_or(&[]))?;
            let inner = count(first.dims.get(ax + 1..).unwrap_or(&[]))?;
            let data = match &first.data {
                Data::F(v) => Data::F(gather_axis(v, outer, len as usize, inner, &picks)?),
                Data::I(v) => Data::I(gather_axis(v, outer, len as usize, inner, &picks)?),
            };
            Cpu { dims, data }
        }
        "Cast" => match node.int("to", 1) {
            onnx::FLOAT => Cpu::floats(first.dims.clone(), first.as_floats()),
            onnx::INT64 | 6 => Cpu::ints(
                first.dims.clone(),
                first.as_ints().or_else(|_| Ok::<_, String>(first.as_floats().iter().map(|f| f.trunc() as i64).collect()))?,
            ),
            other => return Err(format!("Cast to type {other}")),
        },
        op @ ("Add" | "Sub" | "Mul" | "Div") => {
            let b = arg(1).ok_or_else(|| format!("{op} has one input"))?;
            match (&first.data, &b.data) {
                (Data::I(x), Data::I(y)) => {
                    let (dims, v) = binary_values(x, &first.dims, y, &b.dims, |p, q| int_op(op, p, q))?;
                    Cpu::ints(dims, v)
                }
                _ => {
                    let (x, y) = (first.as_floats(), b.as_floats());
                    let (dims, v) = binary_values(&x, &first.dims, &y, &b.dims, |p, q| {
                        Ok(match op {
                            "Add" => p + q,
                            "Sub" => p - q,
                            "Mul" => p * q,
                            _ => p / q,
                        })
                    })?;
                    Cpu::floats(dims, v)
                }
            }
        }
        "Neg" => match &first.data {
            Data::I(v) => Cpu::ints(first.dims.clone(), v.iter().map(|x| x.saturating_neg()).collect()),
            Data::F(v) => Cpu::floats(first.dims.clone(), v.iter().map(|x| -x).collect()),
        },
        other => return Err(format!("{other} cannot be worked out ahead of time")),
    };
    if bounded(&out.dims)? != out.len() {
        return Err(format!("{} gave a tensor of {} values for {:?}", node.op, out.len(), out.dims));
    }
    Ok(out)
}

fn eval_axis_incl(a: i64, rank: usize) -> Result<usize, String> {
    if a == rank as i64 { Ok(rank) } else { axis(a, rank) }
}

fn int_op(op: &str, p: i64, q: i64) -> Result<i64, String> {
    match op {
        "Add" => p.checked_add(q),
        "Sub" => p.checked_sub(q),
        "Mul" => p.checked_mul(q),
        // Truncating, as ONNX Runtime divides integers.
        _ => p.checked_div(q),
    }
    .ok_or_else(|| format!("integer {op} of {p} and {q} overflows"))
}

/// Concatenate along `axis` (all parts the same type and shape apart from the axis).
pub fn concat(parts: &[&Cpu], ax: i64) -> Result<Cpu, String> {
    let first = parts.first().ok_or("Concat of nothing")?;
    let rank = first.dims.len();
    let ax = axis(ax, rank)?;
    let mut dims = first.dims.clone();
    let mut along = 0usize;
    for p in parts {
        if p.dims.len() != rank || p.dims.iter().enumerate().any(|(i, d)| i != ax && Some(d) != first.dims.get(i)) {
            return Err(format!("Concat of {:?} and {:?}", first.dims, p.dims));
        }
        along = along.checked_add(p.dims.get(ax).copied().unwrap_or(0)).ok_or("Concat size")?;
    }
    if let Some(d) = dims.get_mut(ax) {
        *d = along;
    }
    let total = bounded(&dims)?;
    let outer = count(first.dims.get(..ax).unwrap_or(&[]))?;
    let inner = count(first.dims.get(ax + 1..).unwrap_or(&[]))?;
    let ints = parts.iter().all(|p| matches!(p.data, Data::I(_)));
    if ints {
        let vals: Vec<&[i64]> = parts.iter().filter_map(|p| if let Data::I(v) = &p.data { Some(v.as_slice()) } else { None }).collect();
        Ok(Cpu::ints(dims, join(&vals, parts, ax, outer, inner, total)?))
    } else {
        // Each part converted once, not once per block.
        let owned: Vec<Vec<f32>> = parts.iter().map(|p| p.as_floats()).collect();
        let vals: Vec<&[f32]> = owned.iter().map(Vec::as_slice).collect();
        Ok(Cpu::floats(dims, join(&vals, parts, ax, outer, inner, total)?))
    }
}

fn join<T: Copy>(vals: &[&[T]], parts: &[&Cpu], ax: usize, outer: usize, inner: usize, total: usize) -> Result<Vec<T>, String> {
    let mut out = Vec::with_capacity(total);
    for o in 0..outer {
        for (v, p) in vals.iter().zip(parts) {
            let run = p.dims.get(ax).copied().unwrap_or(0).checked_mul(inner).ok_or("Concat size")?;
            let start = o.checked_mul(run).ok_or("Concat size")?;
            let end = start.checked_add(run).ok_or("Concat size")?;
            out.extend_from_slice(v.get(start..end).ok_or("Concat read")?);
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn node(op: &str) -> onnx::Node {
        onnx::Node { op: op.into(), ..Default::default() }
    }

    #[test]
    fn shape_arithmetic_works_out() {
        let shape = Cpu::ints(vec![4], vec![8, 120, 1, 40]);
        let sliced =
            eval(&node("Slice"), &[Some(&shape), Some(&Cpu::ints(vec![1], vec![2])), Some(&Cpu::ints(vec![1], vec![i64::MAX])), None, None]).unwrap();
        assert_eq!(sliced, Cpu::ints(vec![2], vec![1, 40]));
        let back = eval(
            &node("Slice"),
            &[Some(&shape), Some(&Cpu::ints(vec![1], vec![-1])), Some(&Cpu::ints(vec![1], vec![-5])), None, Some(&Cpu::ints(vec![1], vec![-1]))],
        )
        .unwrap();
        assert_eq!(back, Cpu::ints(vec![4], vec![40, 1, 120, 8]));
        let joined = eval(&node("Concat"), &[Some(&sliced), Some(&Cpu::ints(vec![1], vec![-1]))]).unwrap();
        assert_eq!(joined.as_ints().unwrap(), vec![1, 40, -1]);
        assert_eq!(reshape_dims(&[8, 120, 1, 40], &[0, 0, -1]).unwrap(), vec![8, 120, 40]);
        assert!(reshape_dims(&[8, 3], &[5, -1]).is_err());
        assert_eq!(squeeze_dims(&[1, 3, 1], &[]).unwrap(), vec![3]);
        assert!(squeeze_dims(&[2, 3], &[0]).is_err());
        assert_eq!(unsqueeze_dims(&[3], &[0, -1]).unwrap(), vec![1, 3, 1]);
        assert_eq!(broadcast(&[8, 1, 120], &[120]).unwrap(), vec![8, 1, 120]);
        assert!(broadcast(&[2, 3], &[4]).is_err());
        assert_eq!(broadcast_strides(&[120], &[8, 40, 120]), vec![0, 0, 1]);
        let mul = eval(&node("Mul"), &[Some(&Cpu::ints(vec![2], vec![3, 4])), Some(&Cpu::ints(vec![], vec![5]))]).unwrap();
        assert_eq!(mul, Cpu::ints(vec![2], vec![15, 20]));
        assert!(eval(&node("Mul"), &[Some(&Cpu::ints(vec![1], vec![i64::MAX])), Some(&Cpu::ints(vec![1], vec![2]))]).is_err());
        assert!(eval(&node("Conv"), &[Some(&shape)]).is_err());
        let g = eval(&node("Gather"), &[Some(&shape), Some(&Cpu::ints(vec![], vec![-1]))]).unwrap();
        assert_eq!(g, Cpu::ints(vec![], vec![40]));
    }

    #[test]
    fn hostile_slices_and_sizes_are_refused_or_clamped() {
        assert!(slice_spans(&[4], &[0], &[4], None, Some(&[0])).is_err());
        assert!(slice_spans(&[4], &[0, 0], &[4], None, None).is_err());
        assert!(slice_spans(&[4], &[0], &[4], Some(&[3]), None).is_err());
        let s = slice_spans(&[4], &[i64::MIN], &[i64::MAX], None, None).unwrap();
        assert_eq!(s, vec![Span { start: 0, step: 1, len: 4 }]);
        let e = slice_spans(&[4], &[3], &[1], None, None).unwrap();
        assert_eq!(e.first().map(|s| s.len), Some(0));
        assert!(count(&[usize::MAX, 2]).is_err());
        assert!(eval(&node("Reshape"), &[Some(&Cpu::ints(vec![2], vec![1, 2])), Some(&Cpu::ints(vec![2], vec![-1, -1]))]).is_err());
    }

    /// Gathering 65536 rows of 65536 values once allocated 65536 copies of the whole table
    /// (34 GB) before the size was checked; an empty tensor with a vast other dimension once
    /// looped for ever in Concat. Both are refused before any work.
    #[test]
    fn large_gathers_and_vast_empty_tensors_are_refused_before_allocating() {
        let wide = Cpu::ints(vec![1, 65536], vec![7; 65536]);
        let picks = Cpu::ints(vec![65536], vec![0; 65536]);
        let gather = node("Gather");
        let start = std::time::Instant::now();
        assert!(eval(&gather, &[Some(&wide), Some(&picks)]).is_err());
        assert!(start.elapsed().as_secs() < 5);
        let vast = Cpu::ints(vec![1 << 62, 0], vec![]);
        let start = std::time::Instant::now();
        assert!(eval(&node("Concat"), &[Some(&vast), Some(&vast)]).is_err());
        assert!(concat(&[&vast, &vast], 1).is_err());
        assert!(start.elapsed().as_secs() < 5);
        // Ordinary gathers still work: columns 2 and 0 of a 2×3 table.
        let table = Cpu::ints(vec![2, 3], vec![1, 2, 3, 4, 5, 6]);
        let mut cols = node("Gather");
        cols.attrs.insert("axis".into(), onnx::Attr::Int(1));
        let got = eval(&cols, &[Some(&table), Some(&Cpu::ints(vec![2], vec![2, 0]))]).unwrap();
        assert_eq!(got, Cpu::ints(vec![2, 2], vec![3, 1, 6, 4]));
    }
}
