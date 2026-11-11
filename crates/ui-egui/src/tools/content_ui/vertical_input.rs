//! pdfcraft-JPN: typing vertical text in place (縦書きのまま入力).
//!
//! A vertical draft keeps egui's text box for the typing itself (the keyboard, the clipboard,
//! Undo and the IME composition all stay egui's), but the box draws nothing: its layout is a
//! hidden, tiny one. The text is drawn here instead, in the columns the page will draw
//! ([`VerticalEditLayout`]), with the caret as a bar across its column, the selection along the
//! columns, and the IME's candidate window placed at that caret. The arrow keys, Home and End are
//! taken before the box sees them and move the caret by the vertical layout: Up and Down along the
//! column, Left to the next column (columns run right to left), Right to the previous one. A
//! press or drag in the box places the caret by the same layout.
//!
//! Ruby markup stays as typed while editing (the reading is text the caret can enter); the page
//! draws it as ruby once the text is committed.

use std::f32::consts::FRAC_PI_2;
use std::sync::Arc;

use egui::text::{CCursor, CCursorRange};
use egui::{Color32, FontId, Pos2, Rect, Stroke, pos2, vec2};
use pdfcraft_engine::{AddedText, VerticalCaretMove as Move, VerticalEditLayout, VerticalGlyph, VerticalGlyphKind};
use pdfcraft_render::DocInfo;

use crate::canvas::PageXform;

/// The last vertical layout worked out, and the text it is for: laying out shapes the Japanese, too
/// slow to repeat every frame for long text.
pub(crate) type VerticalMemo = std::cell::RefCell<Option<(AddedText, Arc<VerticalEditLayout>)>>;

/// `t` laid out vertically, from `memo` when `t` hasn't changed.
pub(super) fn layout(memo: &VerticalMemo, t: &AddedText) -> Arc<VerticalEditLayout> {
    if let Some((key, l)) = memo.borrow().as_ref()
        && key == t
    {
        return Arc::clone(l);
    }
    let l = Arc::new(VerticalEditLayout::new(t));
    *memo.borrow_mut() = Some((t.clone(), Arc::clone(&l)));
    l
}

/// Where a vertical draft is: its page, and the top-right corner of its box in display space.
pub(super) struct Placed<'a> {
    pub xf: &'a PageXform,
    pub info: &'a DocInfo,
    pub page: usize,
    pub right: f64,
    pub top: f64,
}

impl Placed<'_> {
    /// A rectangle of the box's frame ([`VerticalEditLayout`]: `x` left of the right edge, `y`
    /// down from the top) on screen.
    fn rect(&self, left: f64, top: f64, right: f64, bottom: f64) -> Rect {
        super::screen_rect(self.xf, self.info, self.page, [self.right - left, self.top - bottom, self.right - right, self.top - top])
    }

    /// The box the columns take, on screen.
    pub(super) fn frame(&self, l: &VerticalEditLayout) -> Rect {
        self.rect(l.width(), 0.0, 0.0, l.column_length)
    }

    /// A screen point in the box's frame.
    fn to_box(&self, p: Pos2) -> (f64, f64) {
        let [x, y] = super::to_display(self.xf, self.info, self.page, p);
        (self.right - x, self.top - y)
    }
}

/// The caret move a key makes in vertical text, if it is one of the keys taken from the box.
fn motion(key: egui::Key, modifiers: egui::Modifiers) -> Option<Move> {
    use egui::Key;
    Some(match key {
        Key::ArrowUp => Move::Up,
        Key::ArrowDown => Move::Down,
        // Columns run right to left: Left goes on to the next one.
        Key::ArrowLeft => Move::NextColumn,
        Key::ArrowRight => Move::PreviousColumn,
        Key::Home if modifiers.command => Move::TextStart,
        Key::End if modifiers.command => Move::TextEnd,
        Key::Home => Move::ColumnStart,
        Key::End => Move::ColumnEnd,
        _ => return None,
    })
}

/// Whether an IME composition is under way in the box (it decides that the arrow keys are the
/// IME's, and how the composed text is marked).
fn composing_id(input: egui::Id) -> egui::Id {
    input.with("vertical-composing")
}

/// Where the press that is selecting began, as a caret index.
fn anchor_id(input: egui::Id) -> egui::Id {
    input.with("vertical-anchor")
}

/// `range` moved by `motion`, the selection extended with Shift, in text of `l`'s layout.
pub(crate) fn apply(l: &VerticalEditLayout, range: CCursorRange, motion: Move, shift: bool) -> CCursorRange {
    let n = l.chars();
    let (p, s) = (range.primary.index.0.min(n), range.secondary.index.0.min(n));
    let to = match motion {
        // A selection collapses to its end in the direction of the move.
        Move::Up if !shift && p != s => p.min(s),
        Move::Down if !shift && p != s => p.max(s),
        _ => l.moved(p, motion),
    };
    if shift { CCursorRange::two(CCursor::new(s), CCursor::new(to)) } else { CCursorRange::one(CCursor::new(to)) }
}

/// Before the box takes this frame's input: follow the IME composition, and take the keys that
/// move the caret, moving it by `l`.
pub(super) fn before_input(ctx: &egui::Context, input: egui::Id, l: &VerticalEditLayout) {
    let focused = ctx.memory(|m| m.has_focus(input));
    let mut composing = focused && ctx.data(|d| d.get_temp::<bool>(composing_id(input))).unwrap_or(false);
    let mut moves = Vec::new();
    ctx.input_mut(|i| {
        i.events.retain(|e| match e {
            egui::Event::Ime(egui::ImeEvent::Preedit { text, .. }) => {
                composing = !text.is_empty();
                true
            }
            egui::Event::Ime(egui::ImeEvent::Commit(_)) => {
                composing = false;
                true
            }
            // While composing, the keys are the IME's.
            egui::Event::Key { key, pressed, modifiers, .. } if focused && !composing => match motion(*key, *modifiers) {
                Some(m) => {
                    if *pressed {
                        moves.push((m, modifiers.shift));
                    }
                    false
                }
                None => true,
            },
            _ => true,
        });
    });
    ctx.data_mut(|d| d.insert_temp(composing_id(input), composing));
    if moves.is_empty() {
        return;
    }
    let mut state = egui::TextEdit::load_state(ctx, input).unwrap_or_default();
    let mut range = state.cursor.char_range().unwrap_or_else(|| CCursorRange::one(CCursor::new(l.chars())));
    for (m, shift) in moves {
        range = apply(l, range, m, shift);
    }
    state.cursor.set_char_range(Some(range));
    egui::TextEdit::store_state(ctx, input, state);
}

/// How a character is turned for vertical writing when the interface font has no vertical form.
enum Form {
    Upright,
    /// Turned 90° clockwise: brackets, the long vowel mark, dashes.
    Turned,
    /// Moved toward the top right by this many em: 、。 and the small kana.
    Raised(f32),
}

fn form(c: char) -> Form {
    match c {
        'ー' | '－' | '—' | '―' | '‐' | '〜' | '～' | '…' | '‥' | '（' | '）' | '「' | '」' | '『' | '』' | '【' | '】' | '〈' | '〉' | '《'
        | '》' | '〔' | '〕' | '［' | '］' | '｛' | '｝' | '＜' | '＞' | '＝' | '｜' | '：' | '；' | '(' | ')' | '[' | ']' => {
            Form::Turned
        }
        '、' | '。' | '，' | '．' => Form::Raised(0.6),
        'ぁ' | 'ぃ' | 'ぅ' | 'ぇ' | 'ぉ' | 'っ' | 'ゃ' | 'ゅ' | 'ょ' | 'ゎ' | 'ァ' | 'ィ' | 'ゥ' | 'ェ' | 'ォ' | 'ッ' | 'ャ' | 'ュ' | 'ョ' | 'ヮ'
        | 'ヵ' | 'ヶ' => Form::Raised(0.12),
        _ => Form::Upright,
    }
}

/// Draw one glyph of the layout.
fn paint_glyph(painter: &egui::Painter, placed: &Placed<'_>, l: &VerticalEditLayout, g: &VerticalGlyph, font: &FontId, color: Color32) {
    let (left, right) = l.column_span(g.column);
    let cell = placed.rect(left, g.top, right, g.top + g.length);
    let em = font.size;
    match g.kind {
        VerticalGlyphKind::Upright => {
            let count = g.text.chars().count().max(1) as f32;
            let each = cell.height() / count;
            for (k, ch) in g.text.chars().enumerate() {
                let centre = pos2(cell.center().x, cell.top() + each * (k as f32 + 0.5));
                let galley = painter.layout_no_wrap(ch.to_string(), font.clone(), color);
                let size = galley.size();
                match form(ch) {
                    Form::Turned => {
                        painter.add(
                            egui::epaint::TextShape::new(pos2(centre.x + size.y / 2.0, centre.y - size.x / 2.0), galley, color).with_angle(FRAC_PI_2),
                        );
                    }
                    Form::Raised(by) => painter.galley(centre - size / 2.0 + vec2(by * em, -by * em), galley, color),
                    Form::Upright => painter.galley(centre - size / 2.0, galley, color),
                }
            }
        }
        VerticalGlyphKind::Sideways => {
            // Turned 90° clockwise about its top left: the text reads down from the cell's top,
            // its middle on the column's centre line.
            let galley = painter.layout_no_wrap(g.text.clone(), font.clone(), color);
            let h = galley.size().y;
            painter.add(egui::epaint::TextShape::new(pos2(cell.center().x + h / 2.0, cell.top()), galley, color).with_angle(FRAC_PI_2));
        }
        VerticalGlyphKind::Combined => {
            let mut galley = painter.layout_no_wrap(g.text.clone(), font.clone(), color);
            let w = galley.size().x;
            if w > em && w > 0.0 {
                galley = painter.layout_no_wrap(g.text.clone(), FontId::new((font.size * em / w).max(1.0), font.family.clone()), color);
            }
            painter.galley(cell.center() - galley.size() / 2.0, galley, color);
        }
    }
}

/// What the vertical editor draws with.
pub(super) struct Look<'a> {
    pub font: &'a FontId,
    pub color: Color32,
    pub caret: Color32,
    pub selection: Color32,
    pub hint: Color32,
}

/// After the box took this frame's input: place the caret where the box was pressed or dragged,
/// draw the text, the selection and the caret, and put the IME's window at the caret. Returns the
/// caret and selection (characters).
pub(super) fn after_input(
    ui: &egui::Ui,
    input: egui::Id,
    resp: &egui::Response,
    placed: &Placed<'_>,
    l: &VerticalEditLayout,
    text: &str,
    look: &Look<'_>,
) -> CCursorRange {
    let ctx = ui.ctx();
    let n = l.chars();
    let mut state = egui::TextEdit::load_state(ctx, input).unwrap_or_default();
    if let Some(p) = resp.interact_pointer_pos()
        && (resp.is_pointer_button_down_on() || resp.clicked() || resp.drag_stopped())
    {
        let (x, y) = placed.to_box(p);
        let at = l.index_at(x, y);
        let (pressed, shift) = ctx.input(|i| (i.pointer.primary_pressed(), i.modifiers.shift));
        if pressed {
            let anchor = if shift { state.cursor.char_range().map_or(at, |r| r.secondary.index.0.min(n)) } else { at };
            ctx.data_mut(|d| d.insert_temp(anchor_id(input), anchor));
        }
        let anchor = ctx.data(|d| d.get_temp::<usize>(anchor_id(input))).unwrap_or(at).min(n);
        let range = if resp.double_clicked() {
            let w = pdfcraft_engine::vertical_word_at(text, at);
            CCursorRange::two(CCursor::new(w.start.min(n)), CCursor::new(w.end.min(n)))
        } else {
            CCursorRange::two(CCursor::new(anchor), CCursor::new(at))
        };
        state.cursor.set_char_range(Some(range));
        egui::TextEdit::store_state(ctx, input, state.clone());
    }
    let range = state.cursor.char_range().unwrap_or_else(|| CCursorRange::one(CCursor::new(n)));
    let (p, s) = (range.primary.index.0.min(n), range.secondary.index.0.min(n));
    let (a, b) = (p.min(s), p.max(s));
    let focused = ctx.memory(|m| m.has_focus(input));
    let composing = focused && ctx.data(|d| d.get_temp::<bool>(composing_id(input))).unwrap_or(false);
    let painter = ui.painter();
    if focused && !composing && a < b {
        for (c, top, bottom) in l.selection(a..b) {
            let (left, right) = l.column_span(c);
            painter.rect_filled(placed.rect(left, top, right, bottom), 0.0, look.selection);
        }
    }
    for g in &l.glyphs {
        paint_glyph(painter, placed, l, g, look.font, look.color);
    }
    if n == 0 {
        let hint =
            VerticalGlyph { chars: 0..0, text: tl!("Type text").to_string(), kind: VerticalGlyphKind::Sideways, column: 0, top: 0.0, length: 0.0 };
        paint_glyph(painter, placed, l, &hint, look.font, look.hint);
    }
    let em = (l.pitch / 1.2).max(0.0);
    if focused {
        // The text being composed is marked with a line on its right, as vertical text marks it.
        if composing && a < b {
            for (c, top, bottom) in l.selection(a..b) {
                let (left, right) = l.column_span(c);
                let r = placed.rect(left, top, right, bottom);
                let x = r.right() - 0.1 * r.width();
                painter.line_segment([pos2(x, r.top()), pos2(x, r.bottom())], Stroke::new(1.5, look.color));
            }
        }
        let (c, y) = l.caret(p);
        let (left, right) = l.column_span(c);
        let bar = placed.rect(left, y, right, y);
        let inset = 0.12 * bar.width();
        painter.line_segment([pos2(bar.left() + inset, bar.top()), pos2(bar.right() - inset, bar.top())], Stroke::new(2.0, look.caret));
        // The IME's candidate window opens beside the caret's cell, not where the hidden box's caret is.
        let cell = placed.rect(left, y, right, y + em);
        let frame = placed.frame(l);
        let to_global = ctx.layer_transform_to_global(ui.layer_id()).unwrap_or_default();
        ctx.output_mut(|o| {
            if let Some(ime) = o.ime.as_mut()
                && ime.owner == input
            {
                ime.cursor_rect = to_global * cell;
                ime.rect = to_global * frame;
            }
        });
    }
    range
}

#[cfg(test)]
mod tests {
    use super::*;

    fn layout_of(text: &str) -> VerticalEditLayout {
        let t = AddedText { text: text.into(), vertical: true, size: 10.0, rect: [0.0, 0.0, 100.0, 60.0], ..AddedText::default() };
        VerticalEditLayout::new(&t)
    }

    #[test]
    fn the_arrows_follow_the_columns() {
        let l = layout_of("あいう\nかきく");
        let at = |i: usize| CCursorRange::one(CCursor::new(i));
        let index = |r: CCursorRange| (r.secondary.index.0, r.primary.index.0);
        assert_eq!(motion(egui::Key::ArrowLeft, egui::Modifiers::NONE), Some(Move::NextColumn));
        assert_eq!(motion(egui::Key::ArrowRight, egui::Modifiers::NONE), Some(Move::PreviousColumn));
        assert_eq!(motion(egui::Key::Home, egui::Modifiers::COMMAND), Some(Move::TextStart));
        assert_eq!(motion(egui::Key::A, egui::Modifiers::NONE), None);
        assert_eq!(index(apply(&l, at(1), Move::Down, false)), (2, 2));
        assert_eq!(index(apply(&l, at(1), Move::NextColumn, false)), (5, 5));
        // Shift extends from where the selection began; a plain arrow collapses it.
        let sel = apply(&l, at(1), Move::NextColumn, true);
        assert_eq!(index(sel), (1, 5));
        assert_eq!(index(apply(&l, sel, Move::Up, false)), (1, 1));
        assert_eq!(index(apply(&l, sel, Move::Down, false)), (5, 5));
        // Any caret, even one past the text, stays in it.
        assert_eq!(index(apply(&l, at(usize::MAX), Move::Down, false)), (7, 7));
        assert_eq!(index(apply(&layout_of(""), at(3), Move::NextColumn, true)), (0, 0));
    }
}
