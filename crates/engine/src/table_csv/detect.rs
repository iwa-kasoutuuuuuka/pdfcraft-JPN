//! Tables found in a page's positioned text.
//!
//! Everything here is in page view space (points, origin at the top-left, y down). Two passes:
//!
//! 1. **Ruled tables.** Horizontal and vertical rules that cross each other form a grid; its
//!    column edges are the vertical rules' x positions and its row edges the horizontal rules' y
//!    positions. Each word goes into the cell its centre falls in, so a cell may hold several lines.
//! 2. **Tables without rules**, from the words left over. Words are grouped into lines (their
//!    vertical spans overlap), a line is cut into cells where the gap between two words is wider
//!    than about a character (Japanese has no spaces between words, so a cell is split by its
//!    geometry, never by spaces), and consecutive lines with two or more cells make a table. Its
//!    column edges are the x positions that no cell of any line covers while some line has text on
//!    both sides (a whitespace gutter shared by the rows).
//!
//! The words of a cell are joined as recognised words are ([`pdfcraft_ocr::is_cjk`]): nothing
//! between two Chinese or Japanese characters, a space elsewhere when the words stand apart.

use pdfcraft_ocr::is_cjk;

/// One word (or run of glyphs) of the page, with its box `[x0, y0, x1, y1]` in view space.
#[derive(Clone, Debug, PartialEq)]
pub struct Word {
    pub text: String,
    pub rect: [f64; 4],
}

/// A table found on a page.
#[derive(Clone, Debug, PartialEq)]
pub struct Table {
    /// The rows, top to bottom; every row has the same number of cells.
    pub rows: Vec<Vec<String>>,
    /// The table's box in view space.
    pub rect: [f64; 4],
    /// The table came from ruled lines (otherwise from the gaps between the words).
    pub ruled: bool,
}

impl Table {
    pub fn columns(&self) -> usize {
        self.rows.first().map_or(0, Vec::len)
    }
}

/// At most this many words of a page are looked at.
pub const MAX_WORDS: usize = 200_000;
/// A table has at most this many columns, and at most this many rows: a hostile page laid out as
/// a staircase could otherwise make (words / 2)² cells.
pub const MAX_COLUMNS: usize = 256;
pub const MAX_ROWS: usize = 20_000;
/// A rule at most this thick (points) is a line, not a filled area.
const RULE_THICK: f64 = 3.0;
/// How far apart (points) two rules may be and still mark the same edge, and how far a rule may
/// stop short of another and still meet it.
const RULE_TOL: f64 = 2.0;
/// Rule pairs one page may compare when looking for grids; past it, rules are not used.
const RULE_BUDGET: usize = 4_000_000;
/// Two words of a line farther apart than this many line heights are in different cells.
const CELL_GAP: f64 = 0.8;
/// Two words of a cell farther apart than this many line heights have a space between them
/// (unless both sides are Chinese or Japanese).
const SPACE_GAP: f64 = 0.15;
/// Two lines farther apart than this many line heights are not rows of one table.
const ROW_GAP: f64 = 1.5;
/// Cells longer than this on average are prose (two columns of text side by side), not a table.
const MAX_MEAN_CELL_CHARS: usize = 40;

fn height(r: &[f64; 4]) -> f64 {
    (r[3] - r[1]).max(1e-3)
}

fn usable(w: &Word) -> bool {
    let r = &w.rect;
    r.iter().all(|v| v.is_finite()) && r[2] >= r[0] && r[3] >= r[1] && !w.text.trim().is_empty()
}

/// Append `next` to `out`: a space between them when `spaced` and they are not both Chinese or
/// Japanese characters at the join.
fn join(out: &mut String, next: &str, spaced: bool) {
    let next = next.trim();
    if let (Some(a), Some(b)) = (out.chars().next_back(), next.chars().next())
        && spaced
        && !(is_cjk(a) && is_cjk(b))
    {
        out.push(' ');
    }
    out.push_str(next);
}

/// A line of words: indices into the word list, left to right, and its vertical span.
struct Line {
    words: Vec<usize>,
    top: f64,
    bottom: f64,
}

impl Line {
    fn height(&self) -> f64 {
        (self.bottom - self.top).max(1e-3)
    }
}

/// Group `idx` into lines, top to bottom: a word joins the line before it when their vertical
/// spans share more than half of the smaller height.
fn lines(words: &[Word], idx: &[usize]) -> Vec<Line> {
    let rect = |i: usize| words.get(i).map_or([0.0; 4], |w| w.rect);
    let mut order: Vec<usize> = idx.to_vec();
    order.sort_by(|&a, &b| {
        let (ra, rb) = (rect(a), rect(b));
        (ra[1] + ra[3]).total_cmp(&(rb[1] + rb[3])).then(ra[0].total_cmp(&rb[0]))
    });
    let mut out: Vec<Line> = Vec::new();
    for i in order {
        let r = rect(i);
        match out.last_mut() {
            Some(l) if l.bottom.min(r[3]) - l.top.max(r[1]) > 0.5 * height(&r).min(l.height()) => {
                l.words.push(i);
                l.top = l.top.min(r[1]);
                l.bottom = l.bottom.max(r[3]);
            }
            _ => out.push(Line { words: vec![i], top: r[1], bottom: r[3] }),
        }
    }
    for l in &mut out {
        l.words.sort_by(|&a, &b| rect(a)[0].total_cmp(&rect(b)[0]));
    }
    out
}

/// The text of words that sit on one line, left to right.
fn line_text(words: &[Word], line: &Line) -> String {
    let mut text = String::new();
    let mut right: Option<f64> = None;
    for w in line.words.iter().filter_map(|&i| words.get(i)) {
        let spaced = right.is_some_and(|x| w.rect[0] - x > SPACE_GAP * height(&w.rect).min(line.height()));
        join(&mut text, &w.text, spaced);
        right = Some(right.map_or(w.rect[2], |x| x.max(w.rect[2])));
    }
    text
}

/// The text of a cell's words, which may run over several lines: the lines are joined as
/// recognised words are (nothing between Japanese characters, a space elsewhere).
fn cell_text(words: &[Word], idx: &[usize]) -> String {
    let mut text = String::new();
    for line in lines(words, idx) {
        join(&mut text, &line_text(words, &line), true);
    }
    text
}

/// A run of words of one line that stand close together: one cell of a table without rules.
struct Chunk {
    text: String,
    x0: f64,
    x1: f64,
}

/// Cut a line into chunks where the gap between two words is wider than [`CELL_GAP`] heights.
fn chunks(words: &[Word], line: &Line) -> Vec<Chunk> {
    let mut out: Vec<Chunk> = Vec::new();
    for w in line.words.iter().filter_map(|&i| words.get(i)) {
        let h = height(&w.rect).min(line.height());
        match out.last_mut() {
            Some(c) if w.rect[0] - c.x1 <= CELL_GAP * h => {
                join(&mut c.text, &w.text, w.rect[0] - c.x1 > SPACE_GAP * h);
                c.x1 = c.x1.max(w.rect[2]);
            }
            _ => out.push(Chunk { text: w.text.trim().to_string(), x0: w.rect[0], x1: w.rect[2] }),
        }
    }
    out
}

/// Sorted values merged where neighbours are within `tol`; each group is its mean.
fn cluster(mut values: Vec<f64>, tol: f64) -> Vec<f64> {
    values.retain(|v| v.is_finite());
    values.sort_by(f64::total_cmp);
    let mut out: Vec<(f64, f64, usize)> = Vec::new(); // (last value, sum, count)
    for v in values {
        match out.last_mut() {
            Some((last, sum, n)) if v - *last <= tol => {
                *last = v;
                *sum += v;
                *n += 1;
            }
            _ => out.push((v, v, 1)),
        }
    }
    out.into_iter().map(|(_, sum, n)| sum / n as f64).collect()
}

/// The slot of `v` among ascending `edges`: `k` with `edges[k] <= v < edges[k + 1]`.
fn slot(edges: &[f64], v: f64) -> Option<usize> {
    let k = edges.partition_point(|&e| e <= v);
    (k >= 1 && k < edges.len()).then(|| k - 1)
}

/// Disjoint sets, for joining rules that meet.
struct Sets(Vec<usize>);

impl Sets {
    fn find(&mut self, mut i: usize) -> usize {
        while let Some(&p) = self.0.get(i) {
            if p == i {
                break;
            }
            let grand = self.0.get(p).copied().unwrap_or(p);
            if let Some(slot) = self.0.get_mut(i) {
                *slot = grand;
            }
            i = p;
        }
        i
    }
    fn union(&mut self, a: usize, b: usize) {
        let (ra, rb) = (self.find(a), self.find(b));
        if let Some(slot) = self.0.get_mut(ra) {
            *slot = rb;
        }
    }
}

/// The grids drawn by `rules`: for each, its column edges and row edges (ascending).
fn grids(rules: &[[f64; 4]]) -> Vec<(Vec<f64>, Vec<f64>)> {
    let ok = |r: &&[f64; 4]| r.iter().all(|v| v.is_finite());
    // Horizontal rules as (y, x0, x1), vertical ones as (x, y0, y1).
    let hs: Vec<[f64; 3]> = rules
        .iter()
        .filter(ok)
        .filter(|r| (r[3] - r[1]).abs() <= RULE_THICK && (r[2] - r[0]).abs() > RULE_THICK)
        .map(|r| [(r[1] + r[3]) / 2.0, r[0].min(r[2]), r[0].max(r[2])])
        .collect();
    let vs: Vec<[f64; 3]> = rules
        .iter()
        .filter(ok)
        .filter(|r| (r[2] - r[0]).abs() <= RULE_THICK && (r[3] - r[1]).abs() > RULE_THICK)
        .map(|r| [(r[0] + r[2]) / 2.0, r[1].min(r[3]), r[1].max(r[3])])
        .collect();
    if hs.is_empty() || vs.is_empty() || hs.len().saturating_mul(vs.len()) > RULE_BUDGET {
        return Vec::new();
    }
    let mut sets = Sets((0..hs.len() + vs.len()).collect());
    for (i, h) in hs.iter().enumerate() {
        for (j, v) in vs.iter().enumerate() {
            let meet = h[1] - RULE_TOL <= v[0] && v[0] <= h[2] + RULE_TOL && v[1] - RULE_TOL <= h[0] && h[0] <= v[2] + RULE_TOL;
            if meet {
                sets.union(i, hs.len() + j);
            }
        }
    }
    let mut groups: Vec<(usize, Vec<usize>)> = Vec::new();
    for k in 0..hs.len() + vs.len() {
        let root = sets.find(k);
        match groups.iter_mut().find(|(r, _)| *r == root) {
            Some((_, members)) => members.push(k),
            None => groups.push((root, vec![k])),
        }
    }
    let mut out = Vec::new();
    for (_, members) in groups {
        let h: Vec<&[f64; 3]> = members.iter().filter_map(|&k| hs.get(k)).collect();
        let v: Vec<&[f64; 3]> = members.iter().filter_map(|&k| k.checked_sub(hs.len()).and_then(|j| vs.get(j))).collect();
        if h.is_empty() || v.is_empty() {
            continue;
        }
        // The outer edges come from the rules' ends as well, for grids drawn without a frame.
        let mut xs: Vec<f64> = v.iter().map(|r| r[0]).collect();
        xs.push(h.iter().map(|r| r[1]).fold(f64::INFINITY, f64::min));
        xs.push(h.iter().map(|r| r[2]).fold(f64::NEG_INFINITY, f64::max));
        let mut ys: Vec<f64> = h.iter().map(|r| r[0]).collect();
        ys.push(v.iter().map(|r| r[1]).fold(f64::INFINITY, f64::min));
        ys.push(v.iter().map(|r| r[2]).fold(f64::NEG_INFINITY, f64::max));
        let (xs, ys) = (cluster(xs, RULE_TOL), cluster(ys, RULE_TOL));
        if xs.len() >= 3 && ys.len() >= 2 && xs.len() <= MAX_COLUMNS + 1 && ys.len() <= MAX_ROWS + 1 {
            out.push((xs, ys));
        }
    }
    out
}

/// The table of a grid: the words whose centre lies in it, cell by cell. Rows and columns with no
/// text are left out. `None` when fewer than two cells hold text or fewer than two columns do.
fn ruled_table(words: &[Word], free: &[usize], xs: &[f64], ys: &[f64]) -> Option<(Table, Vec<usize>)> {
    let (cols, rows) = (xs.len().checked_sub(1)?, ys.len().checked_sub(1)?);
    let mut cells: Vec<Vec<Vec<usize>>> = vec![vec![Vec::new(); cols]; rows];
    let mut used = Vec::new();
    for &i in free {
        let Some(w) = words.get(i) else { continue };
        let (cx, cy) = ((w.rect[0] + w.rect[2]) / 2.0, (w.rect[1] + w.rect[3]) / 2.0);
        if let (Some(c), Some(r)) = (slot(xs, cx), slot(ys, cy))
            && let Some(cell) = cells.get_mut(r).and_then(|row| row.get_mut(c))
        {
            cell.push(i);
            used.push(i);
        }
    }
    let text: Vec<Vec<String>> = cells.iter().map(|row| row.iter().map(|idx| cell_text(words, idx)).collect()).collect();
    let keep_col: Vec<bool> = (0..cols).map(|c| text.iter().any(|row| row.get(c).is_some_and(|s| !s.is_empty()))).collect();
    let rows_out: Vec<Vec<String>> = text
        .into_iter()
        .filter(|row| row.iter().any(|s| !s.is_empty()))
        .map(|row| row.into_iter().zip(&keep_col).filter(|(_, k)| **k).map(|(s, _)| s).collect())
        .collect();
    let filled = rows_out.iter().flatten().filter(|s| !s.is_empty()).count();
    if filled < 2 || keep_col.iter().filter(|k| **k).count() < 2 {
        return None;
    }
    let rect = [*xs.first()?, *ys.first()?, *xs.last()?, *ys.last()?];
    Some((Table { rows: rows_out, rect, ruled: true }, used))
}

/// The column edges of a run of chunked lines: inside each maximal x range that no chunk covers
/// while at least two lines have text on both sides of it, its middle.
fn gutters(rows: &[&Vec<Chunk>]) -> Vec<f64> {
    let all = || rows.iter().flat_map(|r| r.iter());
    let mut xs: Vec<f64> = all().flat_map(|c| [c.x0, c.x1]).filter(|v| v.is_finite()).collect();
    xs.sort_by(f64::total_cmp);
    xs.dedup();
    let mut out = Vec::new();
    let mut run: Option<(f64, f64)> = None;
    for pair in xs.windows(2) {
        let (a, b) = (pair[0], pair[1]);
        let covered = all().any(|c| c.x0 < b && c.x1 > a);
        let both_sides = rows.iter().filter(|r| r.iter().any(|c| c.x1 <= a) && r.iter().any(|c| c.x0 >= b)).count();
        if !covered && both_sides >= 2 {
            run = Some(run.map_or((a, b), |(s, _)| (s, b)));
        } else if let Some((s, e)) = run.take() {
            out.push((s + e) / 2.0);
        }
    }
    if let Some((s, e)) = run {
        out.push((s + e) / 2.0);
    }
    out
}

/// The table of a run of lines (`run`, indices into `all`), or `None` when it is not one.
fn gap_table(all: &[Line], chunked: &[Vec<Chunk>], run: &[usize]) -> Option<Table> {
    let rows: Vec<&Vec<Chunk>> = run.iter().filter_map(|&k| chunked.get(k)).collect();
    let multi: Vec<&Vec<Chunk>> = rows.iter().copied().filter(|r| r.len() >= 2).collect();
    let seps = gutters(&multi);
    if seps.is_empty() || seps.len() >= MAX_COLUMNS || rows.len() > MAX_ROWS {
        return None;
    }
    let cols = seps.len() + 1;
    let mut out: Vec<Vec<String>> = Vec::with_capacity(rows.len());
    for row in &rows {
        let mut cells = vec![String::new(); cols];
        for c in row.iter() {
            let k = seps.partition_point(|&s| s <= c.x0);
            if let Some(cell) = cells.get_mut(k) {
                join(cell, &c.text, true);
            }
        }
        out.push(cells);
    }
    let filled: Vec<&String> = out.iter().flatten().filter(|s| !s.is_empty()).collect();
    let chars: usize = filled.iter().map(|s| s.chars().count()).sum();
    if filled.is_empty() || chars / filled.len() > MAX_MEAN_CELL_CHARS {
        return None;
    }
    let lines: Vec<&Line> = run.iter().filter_map(|&k| all.get(k)).collect();
    let x0 = rows.iter().flat_map(|r| r.iter()).map(|c| c.x0).fold(f64::INFINITY, f64::min);
    let x1 = rows.iter().flat_map(|r| r.iter()).map(|c| c.x1).fold(f64::NEG_INFINITY, f64::max);
    let rect = [x0, lines.first()?.top, x1, lines.last()?.bottom];
    Some(Table { rows: out, rect, ruled: false })
}

/// The tables of a page: its words and its ruled lines (`[x0, y0, x1, y1]`, view space), top
/// to bottom. `rules` may be empty.
pub fn tables(words: &[Word], rules: &[[f64; 4]]) -> Vec<Table> {
    let words = words.get(..words.len().min(MAX_WORDS)).unwrap_or_default();
    let mut free: Vec<usize> = (0..words.len()).filter(|&i| words.get(i).is_some_and(usable)).collect();
    let mut out: Vec<Table> = Vec::new();
    for (xs, ys) in grids(rules) {
        if let Some((table, used)) = ruled_table(words, &free, &xs, &ys) {
            let mut used = used;
            used.sort_unstable();
            free.retain(|i| used.binary_search(i).is_err());
            out.push(table);
        }
    }
    let all = lines(words, &free);
    let chunked: Vec<Vec<Chunk>> = all.iter().map(|l| chunks(words, l)).collect();
    let multi = |k: usize| chunked.get(k).is_some_and(|c| c.len() >= 2);
    let close = |a: usize, b: usize| match (all.get(a), all.get(b)) {
        (Some(p), Some(n)) => n.top - p.bottom <= ROW_GAP * p.height().max(n.height()),
        _ => false,
    };
    let mut k = 0;
    while k < all.len() {
        if !multi(k) {
            k += 1;
            continue;
        }
        let mut run = vec![k];
        let mut next = k + 1;
        // A line with one cell stays in the table when a line with several follows it closely
        // (a row whose other cells are empty).
        while next < all.len() && close(next - 1, next) && (multi(next) || (multi(next + 1) && close(next, next + 1))) {
            run.push(next);
            next += 1;
        }
        if run.iter().filter(|&&r| multi(r)).count() >= 2
            && let Some(t) = gap_table(&all, &chunked, &run)
        {
            out.push(t);
        }
        k = next;
    }
    out.sort_by(|a, b| a.rect[1].total_cmp(&b.rect[1]).then(a.rect[0].total_cmp(&b.rect[0])));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A word of `text` at `x`, its top at `y`, 10 pt per character and 10 pt high.
    fn w(text: &str, x: f64, y: f64) -> Word {
        Word { text: text.into(), rect: [x, y, x + 10.0 * text.chars().count() as f64, y + 10.0] }
    }

    fn rows(t: &Table) -> Vec<Vec<&str>> {
        t.rows.iter().map(|r| r.iter().map(String::as_str).collect()).collect()
    }

    #[test]
    fn a_japanese_three_by_three_table_without_rules() {
        let words = [
            w("品名", 20.0, 20.0),
            w("数量", 100.0, 20.0),
            w("単価", 160.0, 20.0),
            w("りんご", 20.0, 34.0),
            w("3", 100.0, 34.0),
            w("120円", 160.0, 34.0),
            w("みかん", 20.0, 48.0),
            w("12", 100.0, 48.0),
            w("80円", 160.0, 48.0),
        ];
        let t = tables(&words, &[]);
        assert_eq!(t.len(), 1, "{t:?}");
        assert_eq!(rows(&t[0]), [["品名", "数量", "単価"], ["りんご", "3", "120円"], ["みかん", "12", "80円"]]);
        assert!(!t[0].ruled);
    }

    #[test]
    fn japanese_characters_of_one_cell_are_joined_without_spaces() {
        // One character a word (as some engines return it), close together: one cell each side.
        let mut words: Vec<Word> = "東京都".chars().enumerate().map(|(k, c)| w(&c.to_string(), 20.0 + 10.5 * k as f64, 20.0)).collect();
        words.push(w("一〇〇", 120.0, 20.0));
        words.extend("大阪府".chars().enumerate().map(|(k, c)| w(&c.to_string(), 20.0 + 10.5 * k as f64, 34.0)));
        words.push(w("二〇〇", 120.0, 34.0));
        let t = tables(&words, &[]);
        assert_eq!(rows(&t[0]), [["東京都", "一〇〇"], ["大阪府", "二〇〇"]]);
    }

    #[test]
    fn ragged_rows_get_empty_cells_in_the_right_column() {
        let words = [
            w("氏名", 20.0, 20.0),
            w("電話", 100.0, 20.0),
            w("備考", 180.0, 20.0),
            w("山田", 20.0, 34.0),
            w("備考あり", 180.0, 34.0),
            w("佐藤", 20.0, 48.0),
            w("0312", 100.0, 48.0),
            // One cell only: kept because a row of several follows.
            w("鈴木", 20.0, 62.0),
            w("高橋", 20.0, 76.0),
            w("0456", 100.0, 76.0),
            w("なし", 180.0, 76.0),
        ];
        let t = tables(&words, &[]);
        assert_eq!(t.len(), 1, "{t:?}");
        assert_eq!(
            rows(&t[0]),
            [
                vec!["氏名", "電話", "備考"],
                vec!["山田", "", "備考あり"],
                vec!["佐藤", "0312", ""],
                vec!["鈴木", "", ""],
                vec!["高橋", "0456", "なし"]
            ]
        );
    }

    #[test]
    fn a_multi_word_latin_cell_keeps_its_spaces() {
        // Words 3 pt apart (a space) stay one cell; 30 pt apart start the next.
        let words = [
            w("Total", 20.0, 20.0),
            w("amount", 73.0, 20.0),
            w("1,200", 160.0, 20.0),
            w("Unit", 20.0, 34.0),
            w("price", 63.0, 34.0),
            w("say \"hi\"", 160.0, 34.0),
        ];
        let t = tables(&words, &[]);
        assert_eq!(rows(&t[0]), [["Total amount", "1,200"], ["Unit price", "say \"hi\""]]);
    }

    #[test]
    fn prose_and_a_single_row_are_not_tables() {
        let words = [w("これは普通の文章の一行目です", 20.0, 20.0), w("これは二行目です", 20.0, 34.0), w("見出し", 20.0, 80.0), w("3", 180.0, 80.0)];
        assert!(tables(&words, &[]).is_empty());
        // Two columns of long text side by side read as prose, not as a table.
        let left = "左の段の文章はこのようにとても長く続いていくものであり表のセルではありませんよね本当に";
        let right = "右の段の文章もまた同じように長く続いていくもので表のセルとは言えない長さがありますよね";
        let words = [w(left, 20.0, 20.0), w(right, 600.0, 20.0), w(left, 20.0, 34.0), w(right, 600.0, 34.0)];
        assert!(tables(&words, &[]).is_empty());
    }

    #[test]
    fn a_ruled_grid_decides_the_cells_and_a_cell_may_hold_two_lines() {
        // A 2 x 2 grid from x 10..210, y 10..70, split at x 110 and y 30.
        let rules = [
            [10.0, 10.0, 210.0, 10.0],
            [10.0, 30.0, 210.0, 30.0],
            [10.0, 70.0, 210.0, 70.0],
            [10.0, 10.0, 10.0, 70.0],
            [110.0, 10.0, 110.0, 70.0],
            [210.0, 10.0, 210.0, 70.0],
        ];
        // The second row's right cell wraps onto a second line.
        let words =
            [w("項目", 15.0, 15.0), w("内容", 115.0, 15.0), w("住所", 15.0, 35.0), w("東京都千代田区", 115.0, 35.0), w("一丁目", 115.0, 47.0)];
        let t = tables(&words, &rules);
        assert_eq!(t.len(), 1, "{t:?}");
        assert!(t[0].ruled);
        assert_eq!(rows(&t[0]), [["項目", "内容"], ["住所", "東京都千代田区一丁目"]]);
        // Without the rules, the wrapped line (one cell, nothing after it) is left out.
        let plain = tables(&words, &[]);
        assert_eq!(rows(&plain[0]), [["項目", "内容"], ["住所", "東京都千代田区"]]);
    }

    #[test]
    fn rules_that_do_not_meet_are_not_a_grid_and_hostile_input_is_ignored() {
        // An underline and a lone vertical bar: no grid.
        assert!(grids(&[[10.0, 10.0, 100.0, 10.0], [300.0, 50.0, 300.0, 90.0]]).is_empty());
        let words = [
            Word { text: "x".into(), rect: [f64::NAN, 0.0, 1.0, 1.0] },
            Word { text: "  ".into(), rect: [0.0, 0.0, 1.0, 1.0] },
            Word { text: "y".into(), rect: [5.0, 5.0, 1.0, 1.0] },
        ];
        assert!(tables(&words, &[[f64::INFINITY, 0.0, 1.0, 0.0]]).is_empty());
    }
}
