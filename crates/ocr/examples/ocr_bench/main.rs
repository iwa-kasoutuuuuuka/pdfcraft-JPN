//! OCR benchmark on synthetic Japanese scans (`cargo xtask ocr-bench` runs this in release).
//!
//! Pages are drawn here from OFL fonts in craft-fonts (`CRAFT_FONTS_DIR`, default `./craft-fonts`):
//! horizontal gothic and mincho, vertical, ruby (furigana) and a ruled table, each at 300 and
//! 200 dpi and with noise, blur and a 2° skew. The text is our own. Every engine reads every
//! page and is scored against the text drawn:
//!
//! * `cer`: edit distance of the whole reading to the whole ground truth (spaces dropped), over
//!   the ground-truth length. Reading order counts, so a table or ruby page can score badly
//!   for an engine that reads the right characters in another order.
//! * `cer_unordered`: each ground-truth line against its best-matching stretch of the reading,
//!   summed over the lines and divided by the ground-truth length (order-free; text the page
//!   does not hold, such as ruby read as text, is not counted).
//! * `line_recall`: the share of ground-truth lines found anywhere in the reading with at most
//!   20 % edits (order-free).
//! * `read_ratio`: characters read over characters drawn, so an engine that reads far more
//!   than the page holds (ruby as text, noise as letters) cannot hide behind the order-free
//!   scores, which ignore insertions and let repeated cells match the same stretch.
//! * `pages_per_s`: pages read per second of recognition time, one page at a time, over the
//!   pages that were read without an error. Every page is read once before the timed pass, so
//!   per-shape set-up (cuDNN's tuning, plans) is not counted: the figure is the steady state.
//!
//! Usage: `ocr_bench [--engines windows,tesseract,ppocrv5] [--lang ja] [--out FILE] [--images DIR]
//! [--fonts DIR] [--models DIR] [--det-side PX] [--quick]`. `ppocrv5` is the PP-OCRv5 server
//! probe on rten (`ppocr.rs`), reading the models `cargo xtask models --optional` fetched;
//! `ppocrv5-cuda` is the same pipeline on ONNX Runtime's CUDA provider, built only with
//! `--features ort-cuda` and run with `ORT_DYLIB_PATH` set (local GPU benchmarking only). Without a font checkout or an engine it prints why and exits 0.
//! Nothing here is shipped; it is a measuring tool.

mod ppocr;

use std::error::Error;
use std::path::{Path, PathBuf};
use std::time::Instant;

use pdfcraft_ocr::{OcrImage, RecognizeOptions, Recognizer};
use skrifa::instance::{LocationRef, Size};
use skrifa::outline::{DrawSettings, OutlinePen};
use skrifa::{FontRef, MetadataProvider};

type Res<T> = Result<T, Box<dyn Error>>;

// ---------------------------------------------------------------- rasteriser

/// A grey page: ink coverage 0 (paper) … 1 (ink).
struct Page {
    w: usize,
    h: usize,
    ink: Vec<f32>,
}

impl Page {
    fn new(w: usize, h: usize) -> Page {
        Page { w, h, ink: vec![0.0; w * h] }
    }

    fn fill_rect(&mut self, x0: f32, y0: f32, x1: f32, y1: f32) {
        let (xa, xb) = (x0.max(0.0).round() as usize, (x1.round().max(0.0) as usize).min(self.w));
        let (ya, yb) = (y0.max(0.0).round() as usize, (y1.round().max(0.0) as usize).min(self.h));
        for y in ya..yb {
            if let Some(row) = self.ink.get_mut(y * self.w..(y + 1) * self.w) {
                for v in row.iter_mut().take(xb).skip(xa) {
                    *v = 1.0;
                }
            }
        }
    }
}

/// Flattens a glyph outline into closed polylines, in pixels, y down.
struct Flatten {
    ox: f32,
    oy: f32,
    polys: Vec<Vec<(f32, f32)>>,
}

impl Flatten {
    fn pt(&self, x: f32, y: f32) -> (f32, f32) {
        (self.ox + x, self.oy - y)
    }
    fn last(&self) -> (f32, f32) {
        self.polys.last().and_then(|p| p.last()).copied().unwrap_or((self.ox, self.oy))
    }
    fn push(&mut self, p: (f32, f32)) {
        if let Some(poly) = self.polys.last_mut() {
            poly.push(p);
        }
    }
}

impl OutlinePen for Flatten {
    fn move_to(&mut self, x: f32, y: f32) {
        let p = self.pt(x, y);
        self.polys.push(vec![p]);
    }
    fn line_to(&mut self, x: f32, y: f32) {
        let p = self.pt(x, y);
        self.push(p);
    }
    fn quad_to(&mut self, cx: f32, cy: f32, x: f32, y: f32) {
        let (p0, c, p) = (self.last(), self.pt(cx, cy), self.pt(x, y));
        for i in 1..=8 {
            let t = i as f32 / 8.0;
            let u = 1.0 - t;
            self.push((u * u * p0.0 + 2.0 * u * t * c.0 + t * t * p.0, u * u * p0.1 + 2.0 * u * t * c.1 + t * t * p.1));
        }
    }
    fn curve_to(&mut self, c1x: f32, c1y: f32, c2x: f32, c2y: f32, x: f32, y: f32) {
        let (p0, c1, c2, p) = (self.last(), self.pt(c1x, c1y), self.pt(c2x, c2y), self.pt(x, y));
        for i in 1..=12 {
            let t = i as f32 / 12.0;
            let u = 1.0 - t;
            let (a, b, c, d) = (u * u * u, 3.0 * u * u * t, 3.0 * u * t * t, t * t * t);
            self.push((a * p0.0 + b * c1.0 + c * c2.0 + d * p.0, a * p0.1 + b * c1.1 + c * c2.1 + d * p.1));
        }
    }
    fn close(&mut self) {
        if let Some(poly) = self.polys.last_mut()
            && let Some(first) = poly.first().copied()
        {
            poly.push(first);
        }
    }
}

fn add(acc: &mut [f32], idx: i64, v: f32) {
    if idx >= 0
        && let Some(c) = acc.get_mut(idx as usize)
    {
        *c += v;
    }
}

/// Signed-area accumulation of one edge into `acc` (width `w`, height `h`), as font-rs does.
fn edge(acc: &mut [f32], w: i64, h: i64, a: (f32, f32), b: (f32, f32)) {
    if a.1 == b.1 {
        return;
    }
    let (dir, p0, p1) = if a.1 < b.1 { (1.0, a, b) } else { (-1.0, b, a) };
    let dxdy = (p1.0 - p0.0) / (p1.1 - p0.1);
    let mut x = p0.0;
    let mut y = p0.1.max(0.0) as i64;
    if p0.1 < 0.0 {
        x -= p0.1 * dxdy;
    }
    let y_end = h.min(p1.1.ceil() as i64);
    while y < y_end {
        let line = y * w;
        let dy = ((y + 1) as f32).min(p1.1) - (y as f32).max(p0.1);
        let xn = x + dxdy * dy;
        let d = dy * dir;
        let (x0, x1) = if x < xn { (x, xn) } else { (xn, x) };
        let x0f = x0.floor();
        let x0i = x0f as i64;
        let x1c = x1.ceil();
        let x1i = x1c as i64;
        if x1i <= x0i + 1 {
            let xmf = 0.5 * (x + xn) - x0f;
            add(acc, line + x0i, d - d * xmf);
            add(acc, line + x0i + 1, d * xmf);
        } else {
            let s = 1.0 / (x1 - x0);
            let x0fr = x0 - x0f;
            let a0 = 0.5 * s * (1.0 - x0fr) * (1.0 - x0fr);
            let x1fr = x1 - x1c + 1.0;
            let am = 0.5 * s * x1fr * x1fr;
            add(acc, line + x0i, d * a0);
            if x1i == x0i + 2 {
                add(acc, line + x0i + 1, d * (1.0 - a0 - am));
            } else {
                let a1 = s * (1.5 - x0fr);
                add(acc, line + x0i + 1, d * (a1 - a0));
                for xi in x0i + 2..x1i - 1 {
                    add(acc, line + xi, d * s);
                }
                let a2 = a1 + (x1i - x0i - 3) as f32 * s;
                add(acc, line + x1i - 1, d * (1.0 - a2 - am));
            }
            add(acc, line + x1i, d * am);
        }
        x = xn;
        y += 1;
    }
}

struct Face<'a> {
    font: FontRef<'a>,
}

impl<'a> Face<'a> {
    fn new(data: &'a [u8]) -> Res<Face<'a>> {
        Ok(Face { font: FontRef::new(data)? })
    }

    /// Advance of `c` at `px`, or an error when the font lacks it (the corpus must be drawable).
    fn advance(&self, c: char, px: f32) -> Res<f32> {
        let gid = self.font.charmap().map(c).ok_or_else(|| format!("the font has no glyph for {c:?}"))?;
        Ok(self.font.glyph_metrics(Size::new(px), LocationRef::default()).advance_width(gid).unwrap_or(px))
    }

    /// Draw `c` with its pen origin at (`x`, `baseline`); returns the advance.
    fn draw(&self, page: &mut Page, c: char, x: f32, baseline: f32, px: f32) -> Res<f32> {
        let adv = self.advance(c, px)?;
        let gid = self.font.charmap().map(c).ok_or("no glyph")?;
        let Some(glyph) = self.font.outline_glyphs().get(gid) else { return Ok(adv) };
        let (bx, by) = (x.floor() - 2.0, (baseline - px * 1.2).floor());
        let (bw, bh) = ((px * 1.6) as i64 + 6, (px * 1.6) as i64 + 6);
        let mut pen = Flatten { ox: x - bx, oy: baseline - by, polys: Vec::new() };
        glyph.draw(DrawSettings::unhinted(Size::new(px), LocationRef::default()), &mut pen)?;
        let mut acc = vec![0.0f32; (bw * bh + 4) as usize];
        for poly in &pen.polys {
            for pair in poly.windows(2) {
                if let [a, b] = pair {
                    edge(&mut acc, bw, bh, *a, *b);
                }
            }
        }
        for gy in 0..bh {
            let mut sum = 0.0f32;
            for gx in 0..bw {
                sum += acc.get((gy * bw + gx) as usize).copied().unwrap_or(0.0);
                let (px_x, px_y) = (bx as i64 + gx, by as i64 + gy);
                if px_x < 0 || px_y < 0 || px_x as usize >= page.w || px_y as usize >= page.h {
                    continue;
                }
                if let Some(v) = page.ink.get_mut(px_y as usize * page.w + px_x as usize) {
                    *v = v.max(sum.abs().min(1.0));
                }
            }
        }
        Ok(adv)
    }

    fn text_width(&self, s: &str, px: f32) -> Res<f32> {
        s.chars().try_fold(0.0, |w, c| Ok(w + self.advance(c, px)?))
    }

    fn draw_text(&self, page: &mut Page, s: &str, x: f32, baseline: f32, px: f32) -> Res<f32> {
        let mut pen = x;
        for c in s.chars() {
            pen += self.draw(page, c, pen, baseline, px)?;
        }
        Ok(pen - x)
    }
}

// ------------------------------------------------------------------- corpus

const HORIZONTAL_GOTHIC: &[&str] = &[
    "製品カタログ 2027年版",
    "ステンレス製マグカップは軽量で丈夫です。",
    "容量は三百五十ミリリットル、食洗機に対応しています。",
    "ご注文は営業日の午後三時までに承ります。",
    "送料は全国一律で六百六十円となります。",
    "返品・交換は商品到着後七日以内にご連絡ください。",
    "保証期間は購入日から一年間です。",
    "詳しい仕様は裏面の一覧表をご覧ください。",
    "お問い合わせ先 電話 03-1234-5678",
    "受付時間 平日 9:00〜17:30",
];

const HORIZONTAL_MINCHO: &[&str] = &[
    "春の訪れとともに、新しい商品が店頭に並びます。",
    "木製の食器は手に馴染み、使うほどに味わいが増します。",
    "天然素材のみを用い、職人が一つずつ仕上げました。",
    "贈り物にも最適な、化粧箱入りのセットもございます。",
    "在庫には限りがございますので、お早めにどうぞ。",
    "季節ごとの限定色を、年に四回お届けしています。",
    "お手入れは柔らかい布で水拭きするだけです。",
    "直射日光や高温多湿の場所での保管は避けてください。",
];

/// Columns of vertical text, read right to left (no punctuation or long-vowel marks: their
/// vertical forms are a font feature this drawing does not shape).
const VERTICAL: &[&str] = &[
    "日本の四季を映した器を作り続けて五十年",
    "土の温もりと釉薬の深い色合いを大切に",
    "一つ一つ手作業で焼き上げております",
    "毎日の食卓を少しだけ豊かにする品々",
    "贈り物としても多くの方に選ばれています",
    "使うほどに味わいが深まる自慢の逸品です",
    "工房では見学や体験教室も受け付けています",
    "どうぞお気軽にお立ち寄りください",
];

type RubyLine = &'static [(&'static str, &'static str)];
const RUBY: &[RubyLine] = &[
    &[("今日", "きょう"), ("は", ""), ("天気", "てんき"), ("が", ""), ("良", "よ"), ("い", "")],
    &[("山田", "やまだ"), ("先生", "せんせい"), ("の", ""), ("講演", "こうえん"), ("会", "")],
    &[("新製品", "しんせいひん"), ("の", ""), ("発売日", "はつばいび"), ("は", ""), ("来月", "らいげつ"), ("です", "")],
    &[("受付", "うけつけ"), ("は", ""), ("一階", "いっかい"), ("の", ""), ("窓口", "まどぐち"), ("です", "")],
    &[("季節", "きせつ"), ("の", ""), ("果物", "くだもの"), ("を", ""), ("使", "つか"), ("った", ""), ("菓子", "かし")],
    &[("明日", "あした"), ("の", ""), ("会議", "かいぎ"), ("は", ""), ("中止", "ちゅうし"), ("です", "")],
];

const TABLE: &[[&str; 4]] = &[
    ["品名", "容量", "価格", "在庫"],
    ["マグカップ", "350ml", "1,800円", "あり"],
    ["湯呑み", "200ml", "1,200円", "あり"],
    ["急須", "500ml", "3,600円", "残りわずか"],
    ["小皿", "直径12cm", "600円", "あり"],
    ["箸置き", "五個組", "900円", "なし"],
    ["茶托", "二枚組", "1,500円", "あり"],
];

// ------------------------------------------------------------------- layouts

const A4: (f32, f32) = (595.0, 842.0);
const MARGIN: f32 = 56.0;

/// A drawn page and the lines it shows, in reading order.
struct Drawn {
    page: Page,
    truth: Vec<String>,
}

fn blank(dpi: f32) -> (Page, f32) {
    let k = dpi / 72.0;
    (Page::new((A4.0 * k) as usize, (A4.1 * k) as usize), k)
}

fn horizontal(face: &Face, lines: &[&str], dpi: f32) -> Res<Drawn> {
    let (mut page, k) = blank(dpi);
    let size = 12.0 * k;
    let mut y = (MARGIN + 12.0) * k;
    for line in lines {
        face.draw_text(&mut page, line, MARGIN * k, y, size)?;
        y += size * 2.0;
    }
    Ok(Drawn { page, truth: lines.iter().map(|s| (*s).to_string()).collect() })
}

fn vertical(face: &Face, dpi: f32) -> Res<Drawn> {
    let (mut page, k) = blank(dpi);
    let size = 15.0 * k;
    let mut x = (A4.0 - MARGIN) * k - size;
    for col in VERTICAL {
        let mut top = MARGIN * k;
        for c in col.chars() {
            let adv = face.advance(c, size)?;
            face.draw(&mut page, c, x + (size - adv) / 2.0, top + size * 0.86, size)?;
            top += size;
        }
        x -= size * 2.2;
    }
    Ok(Drawn { page, truth: VERTICAL.iter().map(|s| (*s).to_string()).collect() })
}

fn ruby(face: &Face, dpi: f32) -> Res<Drawn> {
    let (mut page, k) = blank(dpi);
    let size = 14.0 * k;
    let small = size * 0.5;
    let mut y = (MARGIN + 30.0) * k;
    let mut truth = Vec::new();
    for line in RUBY {
        let mut x = MARGIN * k;
        let mut text = String::new();
        for (base, reading) in *line {
            let w = face.text_width(base, size)?;
            if !reading.is_empty() {
                let rw = face.text_width(reading, small)?;
                face.draw_text(&mut page, reading, x + (w - rw) / 2.0, y - size * 0.95, small)?;
            }
            face.draw_text(&mut page, base, x, y, size)?;
            x += w;
            text.push_str(base);
        }
        truth.push(text);
        y += size * 3.0;
    }
    Ok(Drawn { page, truth })
}

fn table(face: &Face, dpi: f32) -> Res<Drawn> {
    let (mut page, k) = blank(dpi);
    let size = 11.0 * k;
    let (col_w, row_h) = (118.0 * k, 32.0 * k);
    let (x0, y0) = (MARGIN * k, (MARGIN + 20.0) * k);
    let (cols, rows) = (4usize, TABLE.len());
    let line = (1.2 * k).max(1.0);
    for r in 0..=rows {
        let y = y0 + r as f32 * row_h;
        page.fill_rect(x0, y, x0 + cols as f32 * col_w + line, y + line);
    }
    for c in 0..=cols {
        let x = x0 + c as f32 * col_w;
        page.fill_rect(x, y0, x + line, y0 + rows as f32 * row_h + line);
    }
    let mut truth = Vec::new();
    for (r, row) in TABLE.iter().enumerate() {
        for (c, cell) in row.iter().enumerate() {
            let x = x0 + c as f32 * col_w + 8.0 * k;
            face.draw_text(&mut page, cell, x, y0 + r as f32 * row_h + row_h * 0.64, size)?;
            truth.push((*cell).to_string());
        }
    }
    Ok(Drawn { page, truth })
}

// -------------------------------------------------------------- degradations

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    /// Uniform in [-1, 1).
    fn signed(&mut self) -> f32 {
        ((self.next() >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
}

#[derive(Clone, Copy)]
struct Condition {
    name: &'static str,
    dpi: f32,
    skew_deg: f32,
    blur: usize,
    noise: f32,
}

const CONDITIONS: &[Condition] = &[
    Condition { name: "clean-300", dpi: 300.0, skew_deg: 0.0, blur: 0, noise: 0.0 },
    Condition { name: "clean-200", dpi: 200.0, skew_deg: 0.0, blur: 0, noise: 0.0 },
    Condition { name: "noise-300", dpi: 300.0, skew_deg: 0.0, blur: 0, noise: 40.0 },
    Condition { name: "blur-300", dpi: 300.0, skew_deg: 0.0, blur: 2, noise: 0.0 },
    Condition { name: "skew2-300", dpi: 300.0, skew_deg: 2.0, blur: 0, noise: 0.0 },
    Condition { name: "hard-200", dpi: 200.0, skew_deg: -2.0, blur: 1, noise: 30.0 },
];

fn box_blur(v: &mut [f32], w: usize, h: usize, r: usize) {
    let mut tmp = vec![0.0f32; v.len()];
    for _ in 0..2 {
        for y in 0..h {
            for x in 0..w {
                let (a, b) = (x.saturating_sub(r), (x + r).min(w - 1));
                let mut s = 0.0;
                for xx in a..=b {
                    s += v.get(y * w + xx).copied().unwrap_or(0.0);
                }
                if let Some(t) = tmp.get_mut(y * w + x) {
                    *t = s / (b - a + 1) as f32;
                }
            }
        }
        for y in 0..h {
            for x in 0..w {
                let (a, b) = (y.saturating_sub(r), (y + r).min(h - 1));
                let mut s = 0.0;
                for yy in a..=b {
                    s += tmp.get(yy * w + x).copied().unwrap_or(0.0);
                }
                if let Some(t) = v.get_mut(y * w + x) {
                    *t = s / (b - a + 1) as f32;
                }
            }
        }
    }
}

fn rotate(v: &[f32], w: usize, h: usize, deg: f32) -> Vec<f32> {
    let (s, c) = deg.to_radians().sin_cos();
    let (cx, cy) = (w as f32 / 2.0, h as f32 / 2.0);
    let mut out = vec![0.0f32; v.len()];
    for y in 0..h {
        for x in 0..w {
            let (dx, dy) = (x as f32 - cx, y as f32 - cy);
            let (sx, sy) = (c * dx + s * dy + cx, -s * dx + c * dy + cy);
            if sx < 0.0 || sy < 0.0 || sx >= (w - 1) as f32 || sy >= (h - 1) as f32 {
                continue;
            }
            let (x0, y0) = (sx as usize, sy as usize);
            let (fx, fy) = (sx - x0 as f32, sy - y0 as f32);
            let px = |xx: usize, yy: usize| v.get(yy * w + xx).copied().unwrap_or(0.0);
            let val = px(x0, y0) * (1.0 - fx) * (1.0 - fy)
                + px(x0 + 1, y0) * fx * (1.0 - fy)
                + px(x0, y0 + 1) * (1.0 - fx) * fy
                + px(x0 + 1, y0 + 1) * fx * fy;
            if let Some(o) = out.get_mut(y * w + x) {
                *o = val;
            }
        }
    }
    out
}

/// The grey scan: paper 235, ink 25, then skew, blur and noise.
fn scan(drawn: Page, cond: Condition, seed: u64) -> Vec<u8> {
    let (w, h) = (drawn.w, drawn.h);
    let mut ink = drawn.ink;
    if cond.skew_deg != 0.0 {
        ink = rotate(&ink, w, h, cond.skew_deg);
    }
    if cond.blur > 0 {
        box_blur(&mut ink, w, h, cond.blur);
    }
    let mut rng = Rng(seed | 1);
    ink.iter()
        .map(|&i| {
            let g = 235.0 - i * 210.0 + if cond.noise > 0.0 { rng.signed() * cond.noise } else { 0.0 };
            g.clamp(0.0, 255.0) as u8
        })
        .collect()
}

// -------------------------------------------------------------------- scoring

fn levenshtein(a: &[char], b: &[char]) -> usize {
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    for (i, ca) in a.iter().enumerate() {
        let mut cur = vec![i + 1];
        for (j, cb) in b.iter().enumerate() {
            let sub = prev.get(j).copied().unwrap_or(0) + usize::from(ca != cb);
            let del = prev.get(j + 1).copied().unwrap_or(0) + 1;
            let ins = cur.get(j).copied().unwrap_or(0) + 1;
            cur.push(sub.min(del).min(ins));
        }
        prev = cur;
    }
    prev.last().copied().unwrap_or(0)
}

/// The fewest edits between `pattern` and any substring of `text` (Sellers).
fn best_substring_distance(pattern: &[char], text: &[char]) -> usize {
    let mut prev: Vec<usize> = (0..=pattern.len()).collect();
    let mut best = prev.last().copied().unwrap_or(0);
    for ct in text {
        let mut cur = vec![0usize];
        for (i, cp) in pattern.iter().enumerate() {
            let sub = prev.get(i).copied().unwrap_or(0) + usize::from(cp != ct);
            let del = prev.get(i + 1).copied().unwrap_or(0) + 1;
            let ins = cur.get(i).copied().unwrap_or(0) + 1;
            cur.push(sub.min(del).min(ins));
        }
        best = best.min(cur.last().copied().unwrap_or(best));
        prev = cur;
    }
    best
}

fn squeeze(s: &str) -> Vec<char> {
    s.chars().filter(|c| !c.is_whitespace()).collect()
}

struct Score {
    cer: f64,
    cer_unordered: f64,
    line_recall: f64,
    read_ratio: f64,
}

fn score(truth: &[String], reading: &str) -> Score {
    let read = squeeze(reading);
    let gt: Vec<char> = truth.iter().flat_map(|l| squeeze(l)).collect();
    let cer = levenshtein(&gt, &read) as f64 / gt.len().max(1) as f64;
    let (mut found, mut edits) = (0usize, 0usize);
    for l in truth {
        let p = squeeze(l);
        let d = best_substring_distance(&p, &read);
        edits += d;
        found += usize::from(d * 5 <= p.len());
    }
    let cer_unordered = edits as f64 / gt.len().max(1) as f64;
    let read_ratio = read.len() as f64 / gt.len().max(1) as f64;
    Score { cer, cer_unordered, line_recall: found as f64 / truth.len().max(1) as f64, read_ratio }
}

// ------------------------------------------------------------------- driver

struct Args {
    engines: Vec<String>,
    lang: String,
    out: PathBuf,
    images: Option<PathBuf>,
    fonts: PathBuf,
    models: Option<PathBuf>,
    det_side: usize,
    quick: bool,
}

fn parse_args() -> Res<Args> {
    let mut a = Args {
        engines: vec!["windows".into(), "tesseract".into(), "ppocrv5".into()],
        lang: "ja".into(),
        out: PathBuf::from("target/ocr-bench/report.json"),
        images: None,
        fonts: std::env::var_os("CRAFT_FONTS_DIR").map_or_else(|| PathBuf::from("craft-fonts"), PathBuf::from),
        models: None,
        det_side: 1600,
        quick: false,
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        let mut value = |name: &str| it.next().ok_or_else(|| format!("{name} needs a value"));
        match arg.as_str() {
            "--engines" => a.engines = value("--engines")?.split(',').map(|s| s.trim().to_string()).collect(),
            "--lang" => a.lang = value("--lang")?,
            "--out" => a.out = PathBuf::from(value("--out")?),
            "--images" => a.images = Some(PathBuf::from(value("--images")?)),
            "--fonts" => a.fonts = PathBuf::from(value("--fonts")?),
            "--models" => a.models = Some(PathBuf::from(value("--models")?)),
            "--det-side" => a.det_side = value("--det-side")?.parse::<usize>()?.clamp(32, 4000),
            "--quick" => a.quick = true,
            other => return Err(format!("unknown argument {other}").into()),
        }
    }
    Ok(a)
}

fn find_engine(name: &str, args: &Args) -> Option<Box<dyn Recognizer>> {
    match name {
        "ppocrv5" => load_ppocr(args, ppocr::Runtime::Rten, "rten"),
        #[cfg(feature = "ort-cuda")]
        "ppocrv5-cuda" => load_ppocr(args, ppocr::Runtime::OrtCuda, "ONNX Runtime CUDA"),
        #[cfg(not(feature = "ort-cuda"))]
        "ppocrv5-cuda" => {
            println!("ocr-bench: ppocrv5-cuda needs `--features ort-cuda` (local GPU benchmarking only)");
            None
        }
        "windows" => pdfcraft_ocr::windows_ocr::WindowsOcr::find().map(|e| Box::new(e) as Box<dyn Recognizer>),
        "tesseract" => {
            let program = pdfcraft_ocr::tesseract::configured_path().or_else(pdfcraft_ocr::tesseract::path_search)?;
            let e = pdfcraft_ocr::tesseract::TesseractCli::new(program);
            e.available().then(|| Box::new(e) as Box<dyn Recognizer>)
        }
        _ => None,
    }
}

fn load_ppocr(args: &Args, runtime: ppocr::Runtime, what: &str) -> Option<Box<dyn Recognizer>> {
    match ppocr::PpOcr::find(args.models.as_ref(), args.det_side, runtime) {
        Ok(e) => {
            println!("ocr-bench: PP-OCRv5 server det/rec loaded with {what} in {:.1} s", e.load_seconds);
            Some(Box::new(e))
        }
        Err(e) => {
            println!("ocr-bench: PP-OCRv5 on {what}: {e}");
            None
        }
    }
}

fn save_png(path: &Path, w: usize, h: usize, grey: &[u8]) -> Res<()> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let mut enc = png::Encoder::new(std::io::BufWriter::new(std::fs::File::create(path)?), w as u32, h as u32);
    enc.set_color(png::ColorType::Grayscale);
    enc.set_depth(png::BitDepth::Eight);
    enc.write_header()?.write_image_data(grey)?;
    Ok(())
}

fn rgba(grey: &[u8], w: usize, h: usize) -> Res<OcrImage> {
    let mut px = Vec::with_capacity(grey.len() * 4);
    for g in grey {
        px.extend_from_slice(&[*g, *g, *g, 255]);
    }
    Ok(OcrImage::new(w as u32, h as u32, px)?)
}

struct Case {
    layout: &'static str,
    cond: Condition,
    truth: Vec<String>,
    image: OcrImage,
}

fn build_cases(args: &Args) -> Res<Vec<Case>> {
    let read = |rel: &str| std::fs::read(args.fonts.join(rel)).map_err(|e| format!("{}: {e}", args.fonts.join(rel).display()));
    let gothic = read("fonts/biz-ud-pgothic/BIZUDPGothic-Regular.ttf")?;
    let mincho = read("fonts/biz-ud-mincho/BIZUDMincho-Regular.ttf")?;
    let shippori = read("fonts/shippori-mincho/ShipporiMincho-Regular.ttf")?;
    let (gothic, mincho, shippori) = (Face::new(&gothic)?, Face::new(&mincho)?, Face::new(&shippori)?);
    let conditions: Vec<Condition> =
        if args.quick { CONDITIONS.iter().copied().filter(|c| c.name == "clean-300" || c.name == "hard-200").collect() } else { CONDITIONS.to_vec() };
    let mut cases = Vec::new();
    for (ci, cond) in conditions.iter().enumerate() {
        for (li, layout) in ["horizontal-gothic", "horizontal-mincho", "vertical", "ruby", "table"].into_iter().enumerate() {
            let drawn = match layout {
                "horizontal-gothic" => horizontal(&gothic, HORIZONTAL_GOTHIC, cond.dpi)?,
                "horizontal-mincho" => horizontal(&mincho, HORIZONTAL_MINCHO, cond.dpi)?,
                "vertical" => vertical(&shippori, cond.dpi)?,
                "ruby" => ruby(&mincho, cond.dpi)?,
                _ => table(&gothic, cond.dpi)?,
            };
            let (w, h) = (drawn.page.w, drawn.page.h);
            let truth = drawn.truth;
            let grey = scan(drawn.page, *cond, 0x9E37_79B9 + (ci * 8 + li) as u64);
            if let Some(dir) = &args.images {
                save_png(&dir.join(format!("{layout}-{}.png", cond.name)), w, h, &grey)?;
            }
            cases.push(Case { layout, cond: *cond, truth, image: rgba(&grey, w, h)? });
        }
    }
    Ok(cases)
}

fn main() -> Res<()> {
    let args = parse_args()?;
    if !args.fonts.join("fonts/manifest.txt").exists() {
        println!("ocr-bench: skipped — no craft-fonts checkout at {} (set CRAFT_FONTS_DIR)", args.fonts.display());
        return Ok(());
    }
    let engines: Vec<(String, Box<dyn Recognizer>)> = args.engines.iter().filter_map(|n| find_engine(n, &args).map(|e| (n.clone(), e))).collect();
    for n in &args.engines {
        if !engines.iter().any(|(m, _)| m == n) {
            println!("ocr-bench: engine `{n}` is not available here; skipped");
        }
    }
    if engines.is_empty() {
        println!("ocr-bench: skipped — no engine available (Windows OCR needs a Japanese language pack)");
        return Ok(());
    }
    let cases = build_cases(&args)?;
    let options = RecognizeOptions { language: args.lang.clone() };
    let mut report = Vec::new();
    println!("{:<10} {:<18} {:<10} {:>7} {:>9} {:>12} {:>8}", "engine", "layout", "condition", "cer", "cer_unord", "line_recall", "s/page");
    for (name, engine) in &engines {
        // Warm up on every page: the first read of each input shape loads data or tunes kernels
        // (cuDNN per shape), which the timed pass should not count.
        for c in &cases {
            let _ = engine.recognize(&c.image, &options);
        }
        let mut rows = Vec::new();
        for case in &cases {
            let start = Instant::now();
            let result = engine.recognize(&case.image, &options);
            let secs = start.elapsed().as_secs_f64();
            let (reading, error) = match result {
                Ok(lines) => (lines.iter().map(|l| l.text()).collect::<Vec<_>>().join("\n"), None),
                Err(e) => {
                    println!("{name}: {} {}: error: {e}", case.layout, case.cond.name);
                    (String::new(), Some(e.to_string()))
                }
            };
            let s = score(&case.truth, &reading);
            println!(
                "{:<10} {:<18} {:<10} {:>6.1}% {:>8.1}% {:>11.1}% {:>8.2}",
                name,
                case.layout,
                case.cond.name,
                s.cer * 100.0,
                s.cer_unordered * 100.0,
                s.line_recall * 100.0,
                secs
            );
            rows.push(serde_json::json!({
                "layout": case.layout, "condition": case.cond.name, "dpi": case.cond.dpi,
                "cer": s.cer, "cer_unordered": s.cer_unordered, "line_recall": s.line_recall, "read_ratio": s.read_ratio, "seconds": secs, "error": error,
                "reading": reading,
            }));
        }
        let n = rows.len().max(1) as f64;
        let mean = |key: &str| rows.iter().filter_map(|r| r.get(key).and_then(serde_json::Value::as_f64)).sum::<f64>() / n;
        let ok: Vec<&serde_json::Value> = rows.iter().filter(|r| r.get("error").is_some_and(serde_json::Value::is_null)).collect();
        let errors = rows.len() - ok.len();
        let total_s: f64 = ok.iter().filter_map(|r| r.get("seconds").and_then(serde_json::Value::as_f64)).sum();
        let pages_per_s = if ok.is_empty() { 0.0 } else { ok.len() as f64 / total_s.max(1e-9) };
        println!(
            "{name}: mean cer {:.1}%  mean cer_unordered {:.1}%  mean line_recall {:.1}%  {:.2} pages/s over {} pages, {errors} errors",
            mean("cer") * 100.0,
            mean("cer_unordered") * 100.0,
            mean("line_recall") * 100.0,
            pages_per_s,
            ok.len()
        );
        report.push(serde_json::json!({
            "engine": name, "language": args.lang, "pages": rows.len(),
            "mean_cer": mean("cer"), "mean_cer_unordered": mean("cer_unordered"), "mean_line_recall": mean("line_recall"),
            "pages_per_s": pages_per_s, "errors": errors, "cases": rows,
        }));
    }
    if let Some(dir) = args.out.parent() {
        std::fs::create_dir_all(dir)?;
    }
    std::fs::write(&args.out, serde_json::to_string_pretty(&serde_json::json!({ "engines": report }))?)?;
    println!("ocr-bench: wrote {}", args.out.display());
    Ok(())
}
