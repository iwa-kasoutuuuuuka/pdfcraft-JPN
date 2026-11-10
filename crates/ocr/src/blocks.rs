//! Page block segmentation: a recursive X-Y cut on the ink projection profiles.
//!
//! Engines that pick ONE writing direction per image (Windows OCR) garble a page that mixes a
//! vertical block with a horizontal paragraph. Cutting the page into whitespace-separated
//! blocks first lets each block be read on its own. Pure and deterministic; no platform code.

use crate::Line;

/// A block, in image pixels: `[x0, y0, x1, y1)`, end-exclusive.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BlockRect {
    pub x0: u32,
    pub y0: u32,
    pub x1: u32,
    pub y1: u32,
}

impl BlockRect {
    pub fn width(&self) -> u32 {
        self.x1 - self.x0
    }
    pub fn height(&self) -> u32 {
        self.y1 - self.y0
    }
}

/// The cut tree: `Rows` are stacked top to bottom, `Cols` sit side by side left to right.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Node {
    Leaf(BlockRect),
    Rows(Vec<Node>),
    Cols(Vec<Node>),
}

impl Node {
    /// The leaves in the order they were cut (rows top to bottom, columns left to right).
    pub fn leaves(&self) -> Vec<BlockRect> {
        let mut out = Vec::new();
        self.collect(&mut out);
        out
    }

    fn collect(&self, out: &mut Vec<BlockRect>) {
        match self {
            Node::Leaf(r) => out.push(*r),
            Node::Rows(v) | Node::Cols(v) => v.iter().for_each(|n| n.collect(out)),
        }
    }
}

const MAX_DEPTH: usize = 10;
/// Blocks whose ink box is smaller than this on both sides are specks and dropped.
const MIN_BLOCK: u32 = 6;

/// The smallest gap of blank rows/columns that separates two blocks, for an image `w`×`h`.
pub fn default_gap(w: u32, h: u32) -> u32 {
    (w.min(h) / 40).max(12)
}

/// Otsu's threshold on 0–255 luminance: values below it are ink.
fn otsu(hist: &[u64; 256]) -> u8 {
    let total: u64 = hist.iter().sum();
    let sum_all: u64 = hist.iter().enumerate().map(|(i, c)| i as u64 * c).sum();
    let (mut w_b, mut sum_b, mut best, mut best_t) = (0u64, 0u64, 0f64, 128u8);
    for (t, &c) in hist.iter().enumerate() {
        w_b += c;
        if w_b == 0 {
            continue;
        }
        let w_f = total - w_b;
        if w_f == 0 {
            break;
        }
        sum_b += t as u64 * c;
        let m_b = sum_b as f64 / w_b as f64;
        let m_f = (sum_all - sum_b) as f64 / w_f as f64;
        let var = w_b as f64 * w_f as f64 * (m_b - m_f) * (m_b - m_f);
        if var > best {
            best = var;
            best_t = t as u8;
        }
    }
    best_t
}

/// The ink mask (`true` = ink) of an RGBA raster: Otsu on luminance; a page that is mostly
/// dark is taken to be light-on-dark and inverted. A flat image has no ink.
pub fn ink_mask(rgba: &[u8], w: u32, h: u32) -> Vec<bool> {
    let n = (w as usize) * (h as usize);
    let lum: Vec<u8> =
        rgba.as_chunks::<4>().0.iter().take(n).map(|p| ((299 * p[0] as u32 + 587 * p[1] as u32 + 114 * p[2] as u32) / 1000) as u8).collect();
    let mut hist = [0u64; 256];
    lum.iter().for_each(|&v| hist[v as usize] += 1);
    let (lo, hi) = (lum.iter().min().copied().unwrap_or(0), lum.iter().max().copied().unwrap_or(0));
    if hi.saturating_sub(lo) < 24 {
        return vec![false; n];
    }
    let t = otsu(&hist);
    let mut mask: Vec<bool> = lum.iter().map(|&v| v <= t).collect();
    if mask.iter().filter(|&&b| b).count() * 2 > n {
        mask.iter_mut().for_each(|b| *b = !*b);
    }
    mask
}

/// Ranges `[start, end)` of `profile` where content runs, separated by blank stretches of at
/// least `gap` entries. An entry is blank when its count is at most `noise`.
fn split(profile: &[u32], noise: u32, gap: usize) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut start: Option<usize> = None;
    let mut last_ink = 0usize;
    for (i, &c) in profile.iter().enumerate() {
        if c > noise {
            match start {
                None => start = Some(i),
                Some(s) if i - last_ink > gap => {
                    out.push((s, last_ink + 1));
                    start = Some(i);
                }
                Some(_) => {}
            }
            last_ink = i;
        }
    }
    if let Some(s) = start {
        out.push((s, last_ink + 1));
    }
    out
}

struct Cutter<'a> {
    mask: &'a [bool],
    w: usize,
    gap: usize,
}

impl Cutter<'_> {
    /// The ink bounding box of `r`, or `None` when it holds (almost) nothing.
    fn trim(&self, r: BlockRect) -> Option<BlockRect> {
        let (mut x0, mut y0, mut x1, mut y1) = (u32::MAX, u32::MAX, 0u32, 0u32);
        for y in r.y0..r.y1 {
            let row = self.mask.get(y as usize * self.w + r.x0 as usize..y as usize * self.w + r.x1 as usize)?;
            if let (Some(a), Some(b)) = (row.iter().position(|&b| b), row.iter().rposition(|&b| b)) {
                x0 = x0.min(r.x0 + a as u32);
                x1 = x1.max(r.x0 + b as u32 + 1);
                y0 = y0.min(y);
                y1 = y1.max(y + 1);
            }
        }
        (x1 > x0 && y1 > y0 && (x1 - x0 >= MIN_BLOCK || y1 - y0 >= MIN_BLOCK)).then_some(BlockRect { x0, y0, x1, y1 })
    }

    fn cut(&self, r: BlockRect, depth: usize) -> Option<Node> {
        let r = self.trim(r)?;
        if depth >= MAX_DEPTH {
            return Some(Node::Leaf(r));
        }
        let (rw, rh) = (r.width() as usize, r.height() as usize);
        let mut rows = vec![0u32; rh];
        let mut cols = vec![0u32; rw];
        for (y, row_count) in rows.iter_mut().enumerate() {
            let base = (r.y0 as usize + y) * self.w + r.x0 as usize;
            if let Some(row) = self.mask.get(base..base + rw) {
                for (x, &b) in row.iter().enumerate() {
                    if b {
                        *row_count += 1;
                        cols[x] += 1;
                    }
                }
            }
        }
        let row_cuts = split(&rows, (rw / 200) as u32, self.gap);
        if row_cuts.len() > 1 {
            let kids: Vec<Node> = row_cuts
                .iter()
                .filter_map(|&(a, b)| self.cut(BlockRect { x0: r.x0, y0: r.y0 + a as u32, x1: r.x1, y1: r.y0 + b as u32 }, depth + 1))
                .collect();
            return Some(group(kids, Node::Rows));
        }
        let col_cuts = split(&cols, (rh / 200) as u32, self.gap);
        if col_cuts.len() > 1 {
            let kids: Vec<Node> = col_cuts
                .iter()
                .filter_map(|&(a, b)| self.cut(BlockRect { x0: r.x0 + a as u32, y0: r.y0, x1: r.x0 + b as u32, y1: r.y1 }, depth + 1))
                .collect();
            return Some(group(kids, Node::Cols));
        }
        Some(Node::Leaf(r))
    }
}

fn group(mut kids: Vec<Node>, make: fn(Vec<Node>) -> Node) -> Node {
    if kids.len() == 1 { kids.remove(0) } else { make(kids) }
}

/// Cut a page (RGBA, `w`×`h`) into blocks separated by at least `gap` blank pixels. Returns
/// `None` when the page holds no ink.
pub fn segment_with_gap(rgba: &[u8], w: u32, h: u32, gap: u32) -> Option<Node> {
    if w == 0 || h == 0 || rgba.len() < (w as usize) * (h as usize) * 4 {
        return None;
    }
    let mask = ink_mask(rgba, w, h);
    Cutter { mask: &mask, w: w as usize, gap: gap.max(1) as usize }.cut(BlockRect { x0: 0, y0: 0, x1: w, y1: h }, 0)
}

/// [`segment_with_gap`] with the gap scaled to the image ([`default_gap`]).
pub fn segment(rgba: &[u8], w: u32, h: u32) -> Option<Node> {
    segment_with_gap(rgba, w, h, default_gap(w, h))
}

/// What reading one block produced: its lines (page coordinates) and whether it is vertical.
pub type BlockRead = (Vec<Line>, bool);

/// Read every block with `f` and assemble the page's lines in reading order: rows top to
/// bottom, columns left to right — except that side-by-side blocks of which any is vertical
/// are read right to left, as vertical Japanese is.
pub fn read_blocks<E>(node: &Node, f: &mut impl FnMut(&BlockRect) -> Result<BlockRead, E>) -> Result<BlockRead, E> {
    match node {
        Node::Leaf(r) => f(r),
        Node::Rows(kids) => {
            let (mut lines, mut vertical) = (Vec::new(), false);
            for k in kids {
                let (l, v) = read_blocks(k, f)?;
                lines.extend(l);
                vertical |= v;
            }
            Ok((lines, vertical))
        }
        Node::Cols(kids) => {
            let mut parts = Vec::new();
            for k in kids {
                parts.push(read_blocks(k, f)?);
            }
            let vertical = parts.iter().any(|p| p.1);
            if vertical {
                parts.reverse();
            }
            Ok((parts.into_iter().flat_map(|p| p.0).collect(), vertical))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn page(w: u32, h: u32, boxes: &[[u32; 4]]) -> Vec<u8> {
        let mut px = vec![255u8; (w * h * 4) as usize];
        for b in boxes {
            for y in b[1]..b[3] {
                for x in b[0]..b[2] {
                    let o = ((y * w + x) * 4) as usize;
                    px[o..o + 3].fill(0);
                }
            }
        }
        px
    }

    #[test]
    fn blank_page_has_no_blocks() {
        assert_eq!(segment(&page(200, 200, &[]), 200, 200), None);
    }

    #[test]
    fn single_block_is_one_leaf() {
        let px = page(400, 400, &[[50, 50, 300, 70], [50, 80, 300, 100]]);
        let n = segment_with_gap(&px, 400, 400, 40).unwrap();
        assert_eq!(n, Node::Leaf(BlockRect { x0: 50, y0: 50, x1: 300, y1: 100 }));
    }

    #[test]
    fn vertical_block_above_paragraph_is_split_into_rows() {
        // A tall block top-right and a wide paragraph underneath it.
        let px = page(600, 600, &[[400, 20, 420, 200], [440, 20, 460, 200], [30, 300, 560, 320], [30, 330, 560, 350]]);
        let n = segment_with_gap(&px, 600, 600, 40).unwrap();
        let Node::Rows(kids) = &n else { panic!("{n:?}") };
        assert_eq!(kids.len(), 2);
        assert_eq!(kids[0], Node::Leaf(BlockRect { x0: 400, y0: 20, x1: 460, y1: 200 }));
        assert_eq!(kids[1], Node::Leaf(BlockRect { x0: 30, y0: 300, x1: 560, y1: 350 }));
    }

    #[test]
    fn side_by_side_blocks_are_split_into_columns_and_small_gaps_are_kept() {
        let px = page(600, 300, &[[20, 20, 200, 260], [400, 20, 580, 260]]);
        let n = segment_with_gap(&px, 600, 300, 40).unwrap();
        assert_eq!(n.leaves().len(), 2);
        assert!(matches!(n, Node::Cols(_)));
        // The same page with a gap threshold wider than the gutter stays one block.
        let n = segment_with_gap(&px, 600, 300, 300).unwrap();
        assert_eq!(n.leaves().len(), 1);
    }

    #[test]
    fn specks_are_dropped() {
        let px = page(300, 300, &[[10, 10, 13, 13], [100, 100, 200, 180]]);
        let n = segment_with_gap(&px, 300, 300, 30).unwrap();
        assert_eq!(n.leaves(), vec![BlockRect { x0: 100, y0: 100, x1: 200, y1: 180 }]);
    }

    #[test]
    fn dark_background_is_inverted() {
        let mut px = page(200, 200, &[[0, 0, 200, 200]]);
        for y in 80..120 {
            for x in 50..150 {
                let o = (y * 200 + x) * 4;
                px[o..o + 3].fill(255);
            }
        }
        assert_eq!(segment(&px, 200, 200).unwrap().leaves(), vec![BlockRect { x0: 50, y0: 80, x1: 150, y1: 120 }]);
    }

    #[test]
    fn vertical_siblings_read_right_to_left_and_horizontal_left_to_right() {
        let leaf = |x| Node::Leaf(BlockRect { x0: x, y0: 0, x1: x + 10, y1: 10 });
        let line = |t: &str| Line { words: vec![crate::Word::new(t, [0.0; 4], "t")] };
        let tree = Node::Cols(vec![leaf(0), leaf(100)]);
        let read = |vertical: bool| read_blocks::<()>(&tree, &mut |r| Ok((vec![line(&format!("{}", r.x0))], vertical))).unwrap();
        let texts = |l: &BlockRead| l.0.iter().map(Line::text).collect::<Vec<_>>();
        assert_eq!(texts(&read(true)), ["100", "0"]);
        assert_eq!(texts(&read(false)), ["0", "100"]);
    }
}
