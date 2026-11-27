//! WGSL sources for [`plan::Shader`]. Sizes come from a uniform array (`p(i)`), so one pipeline
//! serves every input size; only kernel windows, strides, activations and the buffers bound
//! are compiled in. Every kernel stays within `wgpu::Limits::default()`: 256 threads and at
//! most 12 KiB of workgroup memory ([`workgroup_bytes`]).

use std::fmt::Write as _;

use crate::ir::{Act, Bin, Un};
use crate::plan::Shader;

/// The uniform array: 16 × vec4<u32>.
pub const PARAMS: usize = 64;

/// Workgroup memory a shader declares, to check against the device.
pub fn workgroup_bytes(s: &Shader) -> u32 {
    match s {
        Shader::ConvGemm { .. } | Shader::MatMul { .. } => 8192,
        Shader::BestPartial { .. } => 12288,
        Shader::Softmax => 256,
        _ => 0,
    }
}

fn header(reads: &[&str]) -> String {
    let mut s = String::from("@group(0) @binding(0) var<storage, read_write> y: array<f32>;\n");
    for (i, name) in reads.iter().enumerate() {
        let _ = writeln!(s, "@group(0) @binding({}) var<storage, read> {name}: array<f32>;", i + 1);
    }
    let _ = writeln!(s, "@group(0) @binding({}) var<uniform> P: array<vec4<u32>, 16>;", reads.len() + 1);
    s.push_str(
        "fn p(i: u32) -> u32 { return P[i / 4u][i % 4u]; }\n\
         fn pf(i: u32) -> f32 { return bitcast<f32>(p(i)); }\n\
         fn pi(i: u32) -> i32 { return bitcast<i32>(p(i)); }\n",
    );
    s
}

fn act_fns(a: Act) -> String {
    let (one, four) = match a {
        Act::None => ("v", "v"),
        Act::Relu => ("max(v, 0.0)", "max(v, vec4<f32>(0.0))"),
        Act::Sigmoid => ("1.0 / (1.0 + exp(-v))", "vec4<f32>(1.0) / (vec4<f32>(1.0) + exp(-v))"),
        Act::Swish => ("v / (1.0 + exp(-v))", "v / (vec4<f32>(1.0) + exp(-v))"),
    };
    format!("fn act(v: f32) -> f32 {{ return {one}; }}\nfn act4(v: vec4<f32>) -> vec4<f32> {{ return {four}; }}\n")
}

/// The thread index of a one-dimensional kernel (its grid may be two-dimensional).
const ONE_D: &str = "@compute @workgroup_size(256)\n\
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {\n\
    let i = gid.x + gid.y * nw.x * 256u;\n\
    if (i >= p(0u)) { return; }\n";

/// The shared GEMM main loop: 64×64 tile, 16-deep, 4×4 per thread. `la` and `lb` are the
/// loads of A[gm][gk] and B[gk][n] (the B column is fixed per thread: `bcol`).
fn gemm_loop(la: &str, lb: &str) -> String {
    format!(
        "    let tx = lid % 16u;\n\
    let ty = lid / 16u;\n\
    let ak = lid % 16u;\n\
    let am = lid / 16u;\n\
    let bn = lid % 64u;\n\
    let bk = lid / 64u;\n\
    var c0 = vec4<f32>(0.0);\n\
    var c1 = vec4<f32>(0.0);\n\
    var c2 = vec4<f32>(0.0);\n\
    var c3 = vec4<f32>(0.0);\n\
    for (var k0 = 0u; k0 < K; k0 = k0 + 16u) {{\n\
        for (var j = 0u; j < 4u; j = j + 1u) {{\n\
            let gm = m0 + am + 16u * j;\n\
            let gk = k0 + ak;\n\
            var va = 0.0;\n\
            if (gm < M && gk < K) {{ va = {la}; }}\n\
            As[ak * 64u + am + 16u * j] = va;\n\
            let kk = bk + 4u * j;\n\
            let gk2 = k0 + kk;\n\
            var vb = 0.0;\n\
            if (bvalid && gk2 < K) {{ vb = {lb}; }}\n\
            Bs[kk * 64u + bn] = vb;\n\
        }}\n\
        workgroupBarrier();\n\
        for (var kk = 0u; kk < 16u; kk = kk + 1u) {{\n\
            let ab = kk * 64u + ty * 4u;\n\
            let bb = kk * 64u + tx * 4u;\n\
            let a4 = vec4<f32>(As[ab], As[ab + 1u], As[ab + 2u], As[ab + 3u]);\n\
            let b4 = vec4<f32>(Bs[bb], Bs[bb + 1u], Bs[bb + 2u], Bs[bb + 3u]);\n\
            c0 = c0 + a4.x * b4;\n\
            c1 = c1 + a4.y * b4;\n\
            c2 = c2 + a4.z * b4;\n\
            c3 = c3 + a4.w * b4;\n\
        }}\n\
        workgroupBarrier();\n\
    }}\n"
    )
}

const SHARED: &str = "var<workgroup> As: array<f32, 1024>;\nvar<workgroup> Bs: array<f32, 1024>;\n";

#[allow(clippy::too_many_arguments)]
fn conv_gemm(
    k: [usize; 2],
    stride: [usize; 2],
    dil: [usize; 2],
    pad: [usize; 2],
    ups: &[[usize; 2]],
    act: Act,
    residual: bool,
    head: Option<bool>,
) -> String {
    let mut reads = vec!["w".to_string(), "st".to_string()];
    for i in 0..ups.len() {
        reads.push(format!("s{i}"));
    }
    match head {
        Some(other) => {
            reads.push("w2".into());
            if other {
                reads.push("other".into());
            }
        }
        None if residual => reads.push("r".into()),
        None => {}
    }
    let refs: Vec<&str> = reads.iter().map(String::as_str).collect();
    let mut s = header(&refs);
    s.push_str(SHARED);
    s.push_str(&act_fns(act));
    let _ = write!(
        s,
        "const KH: u32 = {}u;\nconst KW: u32 = {}u;\nconst SH: u32 = {}u;\nconst SW: u32 = {}u;\nconst DH: u32 = {}u;\nconst DW: u32 = {}u;\nconst PT: i32 = {};\nconst PL: i32 = {};\n",
        k[0], k[1], stride[0], stride[1], dil[0], dil[1], pad[0], pad[1]
    );
    // The virtual input: channel runs of the sources, each upsampled.
    s.push_str("fn ld(bi: u32, c: u32, yy: u32, xx: u32) -> f32 {\n");
    for (i, up) in ups.iter().enumerate() {
        let base = 6 + 3 * i;
        let start = if i == 0 { "0u".to_string() } else { format!("p({}u)", base - 3) };
        let body = format!(
            "{{ let c0 = {start}; let ci = p({base}u) - c0; return s{i}[((bi * ci + (c - c0)) * p({}u) + yy / {}u) * p({}u) + xx / {}u]; }}",
            base + 1,
            up[0],
            base + 2,
            up[1]
        );
        if i + 1 == ups.len() {
            let _ = writeln!(s, "    {body}");
        } else {
            let _ = writeln!(s, "    if (c < p({base}u)) {body}");
        }
    }
    s.push_str("}\n");
    s.push_str(
        "fn lb(bi: u32, k: u32, oy: u32, ox: u32) -> f32 {\n\
    let ci = k / (KH * KW);\n\
    let r = k % (KH * KW);\n\
    let ky = r / KW;\n\
    let kx = r % KW;\n\
    let iy = i32(oy * SH + ky * DH) - PT;\n\
    let ix = i32(ox * SW + kx * DW) - PL;\n\
    if (iy < 0 || ix < 0 || iy >= i32(p(3u)) || ix >= i32(p(4u))) { return 0.0; }\n\
    return ld(bi, ci, u32(iy), u32(ix));\n\
}\n",
    );
    let after = 6 + 3 * ups.len();
    match head {
        None => {
            let res = if residual { "o = o + r[at];" } else { "" };
            let _ = write!(
                s,
                "fn put(bi: u32, m: u32, n: u32, v: vec4<f32>) {{\n\
    let M = p(0u);\n\
    let N = p(1u);\n\
    if (m >= M) {{ return; }}\n\
    let sc = st[m];\n\
    let sh = st[M + m];\n\
    for (var j = 0u; j < 4u; j = j + 1u) {{\n\
        if (n + j < N) {{\n\
            let at = (bi * M + m) * N + n + j;\n\
            var o = act(v[j] * sc + sh);\n\
            {res}\n\
            y[at] = o;\n\
        }}\n\
    }}\n\
}}\n"
            );
        }
        Some(_) => {
            s.push_str(
                "fn part(m: u32, v: vec4<f32>) -> vec4<f32> {\n\
    let M = p(0u);\n\
    if (m >= M) { return vec4<f32>(0.0); }\n\
    return w2[m] * act4(v * st[m] + vec4<f32>(st[M + m]));\n\
}\n",
            );
        }
    }
    s.push_str(
        "@compute @workgroup_size(256)\n\
fn main(@builtin(local_invocation_index) lid: u32, @builtin(workgroup_id) wg: vec3<u32>) {\n\
    let M = p(0u);\n\
    let N = p(1u);\n\
    let K = p(2u);\n\
    let OW = max(p(5u), 1u);\n\
    let n0 = wg.x * 64u;\n\
    let m0 = wg.y * 64u;\n\
    let bi = wg.z;\n\
    let pix = n0 + lid % 64u;\n\
    let oy = pix / OW;\n\
    let ox = pix % OW;\n\
    let bvalid = pix < N;\n",
    );
    s.push_str(&gemm_loop("w[gm * K + gk]", "lb(bi, gk2, oy, ox)"));
    match head {
        None => s.push_str(
            "    let n = n0 + tx * 4u;\n\
    let m = m0 + ty * 4u;\n\
    put(bi, m, n, c0);\n\
    put(bi, m + 1u, n, c1);\n\
    put(bi, m + 2u, n, c2);\n\
    put(bi, m + 3u, n, c3);\n\
}\n",
        ),
        Some(other) => {
            let add = if other { "o = o + other[bi * N + n];" } else { "" };
            let _ = write!(
                s,
                "    let m = ty * 4u;\n\
    let z = part(m, c0) + part(m + 1u, c1) + part(m + 2u, c2) + part(m + 3u, c3);\n\
    let at = ty * 64u + tx * 4u;\n\
    As[at] = z.x;\n\
    As[at + 1u] = z.y;\n\
    As[at + 2u] = z.z;\n\
    As[at + 3u] = z.w;\n\
    workgroupBarrier();\n\
    if (lid < 64u) {{\n\
        var sum = 0.0;\n\
        for (var t = 0u; t < 16u; t = t + 1u) {{ sum = sum + As[t * 64u + lid]; }}\n\
        let n = n0 + lid;\n\
        if (n < N) {{\n\
            var o = 1.0 / (1.0 + exp(-(sum + pf({after}u))));\n\
            {add}\n\
            y[bi * N + n] = o * pf({}u) + pf({}u);\n\
        }}\n\
    }}\n\
}}\n",
                after + 1,
                after + 2
            );
        }
    }
    s
}

fn matmul(bias: bool, act: Act, residual: bool) -> String {
    let mut reads = vec!["a", "b"];
    if bias {
        reads.push("bias");
    }
    if residual {
        reads.push("r");
    }
    let mut s = header(&reads);
    s.push_str(SHARED);
    s.push_str(&act_fns(act));
    let add_bias = if bias { "o = o + bias[n + j];" } else { "" };
    let add_res = if residual { "o2 = o2 + r[at];" } else { "" };
    let _ = write!(
        s,
        "fn put(bi: u32, m: u32, n: u32, v: vec4<f32>) {{\n\
    let M = p(0u);\n\
    let N = p(1u);\n\
    if (m >= M) {{ return; }}\n\
    for (var j = 0u; j < 4u; j = j + 1u) {{\n\
        if (n + j < N) {{\n\
            let at = bi * p(5u) + m * N + n + j;\n\
            var o = v[j];\n\
            {add_bias}\n\
            var o2 = act(o);\n\
            {add_res}\n\
            y[at] = o2;\n\
        }}\n\
    }}\n\
}}\n"
    );
    s.push_str(
        "@compute @workgroup_size(256)\n\
fn main(@builtin(local_invocation_index) lid: u32, @builtin(workgroup_id) wg: vec3<u32>) {\n\
    let M = p(0u);\n\
    let N = p(1u);\n\
    let K = p(2u);\n\
    let n0 = wg.x * 64u;\n\
    let m0 = wg.y * 64u;\n\
    let bi = wg.z;\n\
    let abase = bi * p(3u);\n\
    let bbase = bi * p(4u);\n\
    let gn = n0 + lid % 64u;\n\
    let bvalid = gn < N;\n",
    );
    s.push_str(&gemm_loop("a[abase + gm * K + gk]", "b[bbase + gk2 * N + gn]"));
    s.push_str(
        "    let n = n0 + tx * 4u;\n\
    let m = m0 + ty * 4u;\n\
    put(bi, m, n, c0);\n\
    put(bi, m + 1u, n, c1);\n\
    put(bi, m + 2u, n, c2);\n\
    put(bi, m + 3u, n, c3);\n\
}\n",
    );
    s
}

/// A float below every logit, so merges never subtract infinities.
const LOW: &str = "-3.0e38";

fn best_partial(bias: bool) -> String {
    let reads: &[&str] = if bias { &["a", "b", "bias"] } else { &["a", "b"] };
    let mut s = header(reads);
    s.push_str(SHARED);
    s.push_str("var<workgroup> Ia: array<u32, 1024>;\n");
    let add_bias = if bias { "l = l + bias[n];" } else { "" };
    let _ = write!(
        s,
        "fn local(row: u32, tx: u32, n0: u32, v: vec4<f32>) {{\n\
    let N = p(1u);\n\
    var mx = {LOW};\n\
    var sm = 0.0;\n\
    var arg = 0u;\n\
    for (var j = 0u; j < 4u; j = j + 1u) {{\n\
        let n = n0 + tx * 4u + j;\n\
        if (n < N) {{\n\
            var l = v[j];\n\
            {add_bias}\n\
            if (l > mx) {{ sm = sm * exp(mx - l) + 1.0; mx = l; arg = n; }} else {{ sm = sm + exp(l - mx); }}\n\
        }}\n\
    }}\n\
    let at = row * 16u + tx;\n\
    As[at] = mx;\n\
    Bs[at] = sm;\n\
    Ia[at] = arg;\n\
}}\n"
    );
    s.push_str(
        "@compute @workgroup_size(256)\n\
fn main(@builtin(local_invocation_index) lid: u32, @builtin(workgroup_id) wg: vec3<u32>) {\n\
    let M = p(0u);\n\
    let N = p(1u);\n\
    let K = p(2u);\n\
    let n0 = wg.x * 64u;\n\
    let m0 = wg.y * 64u;\n\
    let gn = n0 + lid % 64u;\n\
    let bvalid = gn < N;\n",
    );
    s.push_str(&gemm_loop("a[gm * K + gk]", "b[gk2 * N + gn]"));
    let _ = write!(
        s,
        "    local(ty * 4u, tx, n0, c0);\n\
    local(ty * 4u + 1u, tx, n0, c1);\n\
    local(ty * 4u + 2u, tx, n0, c2);\n\
    local(ty * 4u + 3u, tx, n0, c3);\n\
    workgroupBarrier();\n\
    if (lid < 64u) {{\n\
        var mx = {LOW};\n\
        var sm = 0.0;\n\
        var arg = 0u;\n\
        for (var t = 0u; t < 16u; t = t + 1u) {{\n\
            let at = lid * 16u + t;\n\
            let m = As[at];\n\
            if (m > mx) {{ sm = sm * exp(mx - m) + Bs[at]; mx = m; arg = Ia[at]; }} else {{ sm = sm + Bs[at] * exp(m - mx); }}\n\
        }}\n\
        let row = m0 + lid;\n\
        if (row < M) {{\n\
            let o = (row * p(3u) + wg.x) * 3u;\n\
            y[o] = mx;\n\
            y[o + 1u] = sm;\n\
            y[o + 2u] = f32(arg);\n\
        }}\n\
    }}\n\
}}\n"
    );
    s
}

fn best_merge() -> String {
    let mut s = header(&["part"]);
    s.push_str(ONE_D);
    let _ = write!(
        s,
        "    var mx = {LOW};\n\
    var sm = 0.0;\n\
    var arg = 0.0;\n\
    for (var t = 0u; t < p(3u); t = t + 1u) {{\n\
        let o = (i * p(3u) + t) * 3u;\n\
        let m = part[o];\n\
        if (m > mx) {{ sm = sm * exp(mx - m) + part[o + 1u]; mx = m; arg = part[o + 2u]; }} else {{ sm = sm + part[o + 1u] * exp(m - mx); }}\n\
    }}\n\
    y[i * 2u] = arg;\n\
    y[i * 2u + 1u] = 1.0 / sm;\n\
}}\n"
    );
    s
}

fn direct_conv(k: [usize; 2], act: Act, residual: bool) -> String {
    let reads: &[&str] = if residual { &["x", "w", "st", "r"] } else { &["x", "w", "st"] };
    let mut s = header(reads);
    s.push_str(&act_fns(act));
    let _ = writeln!(s, "const KH: u32 = {}u;\nconst KW: u32 = {}u;", k[0], k[1]);
    s.push_str(
        "@compute @workgroup_size(256)\n\
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {\n\
    let i = gid.x + gid.y * nw.x * 256u;\n\
    if (i >= p(14u)) { return; }\n\
    let COUT = p(0u); let OH = p(1u); let OW = p(2u); let CIN = p(3u); let IH = p(4u); let IW = p(5u);\n\
    let CING = p(6u); let COUTG = max(p(7u), 1u);\n\
    let ox = i % OW; let t1 = i / OW; let oy = t1 % OH; let t2 = t1 / OH; let co = t2 % COUT; let bi = t2 / COUT;\n\
    let cbase = (co / COUTG) * CING;\n\
    var acc = 0.0;\n\
    for (var ci = 0u; ci < CING; ci = ci + 1u) {\n\
        for (var ky = 0u; ky < KH; ky = ky + 1u) {\n\
            let iy = i32(oy * p(8u) + ky * p(10u)) - pi(12u);\n\
            if (iy < 0 || iy >= i32(IH)) { continue; }\n\
            for (var kx = 0u; kx < KW; kx = kx + 1u) {\n\
                let ix = i32(ox * p(9u) + kx * p(11u)) - pi(13u);\n\
                if (ix < 0 || ix >= i32(IW)) { continue; }\n\
                acc = acc + x[((bi * CIN + cbase + ci) * IH + u32(iy)) * IW + u32(ix)] * w[((co * CING + ci) * KH + ky) * KW + kx];\n\
            }\n\
        }\n\
    }\n\
    var o = act(acc * st[co] + st[COUT + co]);\n",
    );
    if residual {
        s.push_str("    o = o + r[i];\n");
    }
    s.push_str("    y[i] = o;\n}\n");
    s
}

fn conv_t(k: [usize; 2], act: Act) -> String {
    let mut s = header(&["x", "w", "st"]);
    s.push_str(&act_fns(act));
    let _ = writeln!(s, "const KH: u32 = {}u;\nconst KW: u32 = {}u;", k[0], k[1]);
    s.push_str(ONE_D);
    s.push_str(
        "    let CIN = p(1u); let COUT = p(2u); let IH = p(3u); let IW = p(4u); let OH = p(5u); let OW = p(6u);\n\
    let SH = i32(p(7u)); let SW = i32(p(8u));\n\
    let ox = i % OW; let t1 = i / OW; let oy = t1 % OH; let t2 = t1 / OH; let co = t2 % COUT; let bi = t2 / COUT;\n\
    var acc = 0.0;\n\
    for (var ky = 0u; ky < KH; ky = ky + 1u) {\n\
        let ty = i32(oy) + pi(11u) - i32(ky * p(9u));\n\
        if (ty < 0 || ty % SH != 0) { continue; }\n\
        let iy = u32(ty / SH);\n\
        if (iy >= IH) { continue; }\n\
        for (var kx = 0u; kx < KW; kx = kx + 1u) {\n\
            let tx = i32(ox) + pi(12u) - i32(kx * p(10u));\n\
            if (tx < 0 || tx % SW != 0) { continue; }\n\
            let ix = u32(tx / SW);\n\
            if (ix >= IW) { continue; }\n\
            for (var ci = 0u; ci < CIN; ci = ci + 1u) {\n\
                acc = acc + x[((bi * CIN + ci) * IH + iy) * IW + ix] * w[((ci * COUT + co) * KH + ky) * KW + kx];\n\
            }\n\
        }\n\
    }\n\
    y[i] = act(acc * st[co] + st[COUT + co]);\n\
}\n",
    );
    s
}

fn pool(max: bool, include_pad: bool) -> String {
    let mut s = header(&["x"]);
    s.push_str(ONE_D);
    s.push_str(
        "    let IH = p(2u); let IW = p(3u); let OH = p(4u); let OW = p(5u);\n\
    let ox = i % OW; let t1 = i / OW; let oy = t1 % OH; let plane = t1 / OH;\n\
    var acc = 0.0;\n\
    var n = 0.0;\n",
    );
    if max {
        let _ = writeln!(s, "    acc = {LOW};");
    }
    let take = if max { "acc = max(acc, v);" } else { "acc = acc + v;" };
    let _ = write!(
        s,
        "    for (var ky = 0u; ky < p(6u); ky = ky + 1u) {{\n\
        let iy = i32(oy * p(8u) + ky) - pi(10u);\n\
        if (iy < 0 || iy >= i32(IH)) {{ continue; }}\n\
        for (var kx = 0u; kx < p(7u); kx = kx + 1u) {{\n\
            let ix = i32(ox * p(9u) + kx) - pi(11u);\n\
            if (ix < 0 || ix >= i32(IW)) {{ continue; }}\n\
            let v = x[(plane * IH + u32(iy)) * IW + u32(ix)];\n\
            {take}\n\
            n = n + 1.0;\n\
        }}\n\
    }}\n"
    );
    if max {
        s.push_str("    y[i] = acc;\n}\n");
    } else if include_pad {
        s.push_str("    y[i] = acc / f32(max(p(6u) * p(7u), 1u));\n}\n");
    } else {
        s.push_str("    y[i] = acc / max(n, 1.0);\n}\n");
    }
    s
}

fn simple(reads: &[&str], body: &str) -> String {
    let mut s = header(reads);
    s.push_str(ONE_D);
    s.push_str(body);
    s.push_str("}\n");
    s
}

fn binary(op: Bin) -> String {
    let expr = match op {
        Bin::Add => "va + vb",
        Bin::Sub => "va - vb",
        Bin::Mul => "va * vb",
        Bin::Div => "va / vb",
        Bin::Pow => "powf(va, vb)",
    };
    let mut s = header(&["a", "b"]);
    if op == Bin::Pow {
        // WGSL leaves pow undefined below zero; ONNX defines whole exponents there.
        s.push_str(
            "fn powf(a: f32, b: f32) -> f32 {\n\
    if (a >= 0.0) { return pow(a, b); }\n\
    if (round(b) != b) { return bitcast<f32>(0x7fc00000u); }\n\
    let m = pow(-a, b);\n\
    if (fract(b * 0.5) != 0.0) { return -m; }\n\
    return m;\n\
}\n",
        );
    }
    s.push_str(ONE_D);
    let _ = write!(
        s,
        "    var rem = i;\n\
    var ia = 0u;\n\
    var ib = 0u;\n\
    for (var d = 0u; d < 6u; d = d + 1u) {{\n\
        let e = 5u - d;\n\
        let dim = max(p(2u + e), 1u);\n\
        let c = rem % dim;\n\
        rem = rem / dim;\n\
        ia = ia + c * p(8u + e);\n\
        ib = ib + c * p(14u + e);\n\
    }}\n\
    let va = a[ia];\n\
    let vb = b[ib];\n\
    y[i] = {expr};\n\
}}\n"
    );
    s
}

fn unary(op: Un) -> String {
    let expr = match op {
        Un::Relu => "max(v, 0.0)",
        Un::Sigmoid => "1.0 / (1.0 + exp(-v))",
        Un::Sqrt => "sqrt(v)",
        Un::Square => "v * v",
        Un::Exp => "exp(v)",
        Un::Tanh => "tanh(v)",
    };
    simple(&["x"], &format!("    let v = x[i];\n    y[i] = {expr};\n"))
}

fn affine(act: Act) -> String {
    let mut s = header(&["x", "st"]);
    s.push_str(&act_fns(act));
    s.push_str(ONE_D);
    s.push_str("    let C = p(1u);\n    let c = (i / max(p(2u), 1u)) % C;\n    y[i] = act(x[i] * st[c] + st[C + c]);\n}\n");
    s
}

fn softmax() -> String {
    let mut s = header(&["x"]);
    s.push_str("var<workgroup> red: array<f32, 64>;\n");
    let _ = write!(
        s,
        "@compute @workgroup_size(64)\n\
fn main(@builtin(local_invocation_index) lid: u32, @builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {{\n\
    let row = wg.x + wg.y * nw.x;\n\
    if (row >= p(0u)) {{ return; }}\n\
    let n = p(1u);\n\
    let base = row * n;\n\
    var m = {LOW};\n\
    for (var j = lid; j < n; j = j + 64u) {{ m = max(m, x[base + j]); }}\n\
    red[lid] = m;\n\
    workgroupBarrier();\n\
    for (var s = 32u; s > 0u; s = s / 2u) {{\n\
        if (lid < s) {{ red[lid] = max(red[lid], red[lid + s]); }}\n\
        workgroupBarrier();\n\
    }}\n\
    let mx = red[0];\n\
    workgroupBarrier();\n\
    var sm = 0.0;\n\
    for (var j = lid; j < n; j = j + 64u) {{ sm = sm + exp(x[base + j] - mx); }}\n\
    red[lid] = sm;\n\
    workgroupBarrier();\n\
    for (var s = 32u; s > 0u; s = s / 2u) {{\n\
        if (lid < s) {{ red[lid] = red[lid] + red[lid + s]; }}\n\
        workgroupBarrier();\n\
    }}\n\
    let total = red[0];\n\
    for (var j = lid; j < n; j = j + 64u) {{ y[base + j] = exp(x[base + j] - mx) / total; }}\n\
}}\n"
    );
    s
}

/// The WGSL for `shader`.
pub fn source(shader: &Shader) -> String {
    match shader {
        Shader::ConvGemm { k, stride, dil, pad, ups, act, residual, head } => conv_gemm(*k, *stride, *dil, *pad, ups, *act, *residual, *head),
        Shader::MatMul { bias, act, residual } => matmul(*bias, *act, *residual),
        Shader::BestPartial { bias } => best_partial(*bias),
        Shader::BestMerge => best_merge(),
        Shader::DirectConv { k, act, residual } => direct_conv(*k, *act, *residual),
        Shader::ConvT { k, act } => conv_t(*k, *act),
        Shader::Pool { max, include_pad } => pool(*max, *include_pad),
        Shader::Resize => simple(
            &["x"],
            "    let IH = p(1u); let IW = p(2u); let OH = p(3u); let OW = p(4u);\n\
    let ox = i % OW; let t1 = i / OW; let oy = t1 % OH; let plane = t1 / OH;\n\
    y[i] = x[(plane * IH + oy / p(5u)) * IW + ox / p(6u)];\n",
        ),
        Shader::Copy => simple(&["x"], "    let bi = max(p(1u), 1u);\n    y[(i / bi) * p(2u) + p(3u) + i % bi] = x[i];\n"),
        Shader::Binary(op) => binary(*op),
        Shader::Unary(op) => unary(*op),
        Shader::Affine(act) => affine(*act),
        Shader::ReduceMean => simple(
            &["x"],
            "    let n = p(1u);\n\
    var sm = 0.0;\n\
    for (var j = 0u; j < n; j = j + 1u) { sm = sm + x[i * n + j]; }\n\
    y[i] = sm / f32(max(n, 1u));\n",
        ),
        Shader::Softmax => softmax(),
        Shader::Gather => simple(
            &["x"],
            "    var rem = i;\n\
    var src = pi(14u);\n\
    for (var d = 0u; d < 6u; d = d + 1u) {\n\
        let e = 5u - d;\n\
        let dim = max(p(2u + e), 1u);\n\
        let c = rem % dim;\n\
        rem = rem / dim;\n\
        src = src + i32(c) * pi(8u + e);\n\
    }\n\
    y[i] = x[u32(src)];\n",
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every shader parses and validates with naga (the compiler wgpu uses), on every machine.
    #[test]
    fn every_shader_validates() {
        let acts = [Act::None, Act::Relu, Act::Sigmoid, Act::Swish];
        let mut shaders = vec![
            Shader::BestPartial { bias: true },
            Shader::BestPartial { bias: false },
            Shader::BestMerge,
            Shader::Pool { max: true, include_pad: false },
            Shader::Pool { max: false, include_pad: false },
            Shader::Pool { max: false, include_pad: true },
            Shader::Resize,
            Shader::Copy,
            Shader::ReduceMean,
            Shader::Softmax,
            Shader::Gather,
        ];
        for op in [Bin::Add, Bin::Sub, Bin::Mul, Bin::Div, Bin::Pow] {
            shaders.push(Shader::Binary(op));
        }
        for op in [Un::Relu, Un::Sigmoid, Un::Sqrt, Un::Square, Un::Exp, Un::Tanh] {
            shaders.push(Shader::Unary(op));
        }
        for act in acts {
            shaders.push(Shader::Affine(act));
            shaders.push(Shader::ConvT { k: [2, 2], act });
            for residual in [false, true] {
                shaders.push(Shader::DirectConv { k: [3, 3], act, residual });
                shaders.push(Shader::MatMul { bias: true, act, residual });
                shaders.push(Shader::MatMul { bias: false, act, residual });
                shaders.push(Shader::ConvGemm { k: [3, 3], stride: [2, 1], dil: [1, 1], pad: [1, 1], ups: vec![[1, 1]], act, residual, head: None });
            }
            for head in [Some(true), Some(false)] {
                shaders.push(Shader::ConvGemm {
                    k: [3, 3],
                    stride: [1, 1],
                    dil: [1, 1],
                    pad: [1, 1],
                    ups: vec![[2, 2], [1, 1]],
                    act,
                    residual: false,
                    head,
                });
            }
        }
        shaders.push(Shader::ConvGemm {
            k: [1, 1],
            stride: [1, 1],
            dil: [1, 1],
            pad: [0, 0],
            ups: vec![[1, 1], [2, 2], [4, 4], [8, 8]],
            act: Act::Relu,
            residual: false,
            head: None,
        });
        for sh in &shaders {
            let src = source(sh);
            let module = naga::front::wgsl::parse_str(&src).unwrap_or_else(|e| panic!("{sh:?}: {}\n{src}", e.emit_to_string(&src)));
            naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::empty())
                .validate(&module)
                .unwrap_or_else(|e| panic!("{sh:?}: {e:?}\n{src}"));
            assert!(workgroup_bytes(sh) <= 16384);
        }
    }
}
