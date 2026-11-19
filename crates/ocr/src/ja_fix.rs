//! Japanese post-correction of recognised words (日本語の読み取り補正).
//!
//! OCR engines confuse a few characters that look alike. Each rule here changes a character only
//! where the characters around it make the engine's reading impossible or all but impossible
//! Japanese, never because the alternative is merely likelier:
//!
//! - a small kana (っ ゃ ゅ ょ and their katakana) that no Japanese word can have at that place
//!   becomes the ordinary kana (`見当がっ|ぬ` → `見当がつ|ぬ`);
//! - 一 after katakana (and not before a kanji or digit) is the long-vowel mark ー (`コ一ヒ一` → `コーヒー`), and the other kanji
//!   that look like katakana (力 工 夕 二 八 口) between katakana are the katakana (`ス夕ッフ` →
//!   `スタッフ`); ー between two kanji is 一 (`第ー章` → `第一章`);
//! - a letter that looks like a digit (`O`, `o`, `l`, `I`) between two ASCII digits is the digit
//!   (`2O24` → `2024`).
//!
//! Text without any Japanese character is never touched. The reading order is the one the engine
//! returned, so a rule may look at the character after a word's last one (the next word's first,
//! or the next line's), which is how a kana that was wrapped onto the next line is judged.

use crate::{Line, Word};

/// What a correction pass changed.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Fixed {
    /// Characters replaced.
    pub characters: usize,
}

/// Correct the characters of `lines` in place and say how many changed.
pub fn fix_japanese(lines: &mut [Line]) -> Fixed {
    // One flat list of (line, word, char index) in reading order, with its character, so a rule
    // sees across word and line boundaries.
    let mut slots: Vec<(usize, usize, usize, char)> = Vec::new();
    for (l, line) in lines.iter().enumerate() {
        for (w, word) in line.words.iter().enumerate() {
            slots.extend(word.text.chars().enumerate().map(|(i, c)| (l, w, i, c)));
        }
    }
    if !slots.iter().any(|s| is_japanese(s.3)) {
        return Fixed::default();
    }
    let original: Vec<char> = slots.iter().map(|s| s.3).collect();
    let mut out = original.clone();
    // A fixed character can make its neighbour's reading certain (ハ口一: the 一 is ー, which
    // makes 口 sit between katakana), so the rules run again until nothing changes. Each pass
    // only turns a character into a kana or digit, which no rule turns back: three passes end it.
    for _ in 0..3 {
        let chars = out.clone();
        let mut again = false;
        for i in 0..chars.len() {
            let prev = i.checked_sub(1).and_then(|p| chars.get(p)).copied();
            let next = chars.get(i + 1).copied();
            let tail = chars.get(i + 1..).unwrap_or_default();
            if let Some(c) = chars.get(i).copied().filter(|c| !compound_exception(*c, tail)).and_then(|c| replacement(c, prev, next))
                && let Some(o) = out.get_mut(i)
            {
                *o = c;
                again = true;
            }
        }
        if !again {
            break;
        }
    }
    let chars = original;
    let mut changed = 0usize;
    for (slot, (new, old)) in slots.iter().zip(out.iter().zip(&chars)) {
        if new != old {
            changed += 1;
            set_char(lines, slot.0, slot.1, slot.2, *new);
        }
    }
    Fixed { characters: changed }
}

fn set_char(lines: &mut [Line], l: usize, w: usize, i: usize, c: char) {
    let Some(word) = lines.get_mut(l).and_then(|line| line.words.get_mut(w)) else { return };
    set_word_char(word, i, c);
}

fn set_word_char(word: &mut Word, i: usize, c: char) {
    word.text = word.text.chars().enumerate().map(|(n, o)| if n == i { c } else { o }).collect();
}

/// The character `c` should be, given its neighbours, when the reading is wrong.
fn replacement(c: char, prev: Option<char>, next: Option<char>) -> Option<char> {
    match c {
        // っ before a kana it can never double: a vowel, な行, ま行, や行, ら行, わ行, ん.
        'っ' if next.is_some_and(cannot_follow_small_tsu_hira) && !prev.is_some_and(|p| matches!(p, 'あ' | 'い' | 'う' | 'え' | 'お' | 'わ')) => {
            Some('つ')
        }
        'ッ' if next.is_some_and(cannot_follow_small_tsu_kata) => Some('ツ'),
        // ゃゅょ only follow an い-row kana (and a few loanword kana).
        'ゃ' | 'ゅ' | 'ょ' if prev.is_some_and(is_hiragana) && !prev.is_some_and(takes_small_ya_hira) => Some(big_ya(c)),
        'ャ' | 'ュ' | 'ョ' if prev.is_some_and(is_katakana_letter) && !prev.is_some_and(takes_small_ya_kata) => Some(big_ya(c)),
        // 一 and the katakana look-alikes inside a katakana word.
        // 一 ends a katakana word too (コーヒー); before a kanji it is the numeral (コーヒー一杯).
        // After a katakana word 一 is ー only when the word goes on (katakana) or ends (nothing, a
        // mark, a space or a particle such as を); before a counter ending (一つ, 一か月, 一ずつ) it is the
        // numeral (リンゴ一つ).
        '一' if is_kata_word(prev)
            && next.is_none_or(|n| {
                (is_katakana_letter(n) && !matches!(n, 'ヵ' | 'ヶ'))
                    || n == 'ー'
                    || is_closing(n)
                    || (is_hiragana(n) && !matches!(n, 'つ' | 'か' | 'ず'))
            }) =>
        {
            Some('ー')
        }
        '力' if is_kata_word(prev) && is_kata_word(next) => Some('カ'),
        '工' if is_kata_word(prev) && is_kata_word(next) => Some('エ'),
        '夕' if is_kata_word(prev) && is_kata_word(next) => Some('タ'),
        '二' if is_kata_word(prev) && is_kata_word(next) => Some('ニ'),
        '八' if is_kata_word(prev) && is_kata_word(next) => Some('ハ'),
        '口' if is_kata_word(prev) && is_kata_word(next) => Some('ロ'),
        // ー between a kanji numeral and a kanji is 一 (二ー三); 東京ー大阪 is a dash, left alone.
        'ー' if prev.is_some_and(|p| is_kanji_numeral(p) || p == '第') && next.is_some_and(is_kanji) => Some('一'),
        // A digit-like letter between two digits.
        'O' | 'o' if prev.is_some_and(is_digit) && next.is_some_and(is_digit) => Some('0'),
        'l' | 'I' if prev.is_some_and(is_digit) && next.is_some_and(is_digit) => Some('1'),
        _ => None,
    }
}

fn big_ya(c: char) -> char {
    match c {
        'ゃ' => 'や',
        'ゅ' => 'ゆ',
        'ょ' => 'よ',
        'ャ' => 'ヤ',
        'ュ' => 'ユ',
        'ョ' => 'ヨ',
        other => other,
    }
}

/// Kana a hiragana っ may not precede.
fn cannot_follow_small_tsu_hira(n: char) -> bool {
    matches!(
        n,
        'あ' | 'い'
            | 'う'
            | 'え'
            | 'お'
            | 'な'
            | 'に'
            | 'ぬ'
            | 'ね'
            | 'の'
            | 'ま'
            | 'み'
            | 'む'
            | 'め'
            | 'も'
            | 'や'
            | 'ゆ'
            | 'よ'
            | 'ら'
            | 'り'
            | 'る'
            | 'れ'
            | 'ろ'
            | 'わ'
            | 'を'
            | 'ん'
    )
}

/// Kana a katakana ッ may not precede (ー included: ッー is not a word).
fn cannot_follow_small_tsu_kata(n: char) -> bool {
    matches!(
        n,
        'ア' | 'イ'
            | 'ウ'
            | 'エ'
            | 'オ'
            | 'ナ'
            | 'ニ'
            | 'ヌ'
            | 'ネ'
            | 'ノ'
            | 'マ'
            | 'ミ'
            | 'ム'
            | 'メ'
            | 'モ'
            | 'ヤ'
            | 'ユ'
            | 'ヨ'
            | 'ラ'
            | 'リ'
            | 'ル'
            | 'レ'
            | 'ロ'
            | 'ワ'
            | 'ヲ'
            | 'ン'
            | 'ー'
    )
}

/// The kana before which ゃゅょ is spelt small: the い row, plus て で ふ う (てゅ でゅ ふゃ うょ…).
fn takes_small_ya_hira(p: char) -> bool {
    matches!(p, 'き' | 'ぎ' | 'し' | 'じ' | 'ち' | 'ぢ' | 'に' | 'ひ' | 'び' | 'ぴ' | 'み' | 'り' | 'い' | 'て' | 'で' | 'ふ' | 'う' | 'ゔ')
}

fn takes_small_ya_kata(p: char) -> bool {
    matches!(p, 'キ' | 'ギ' | 'シ' | 'ジ' | 'チ' | 'ヂ' | 'ニ' | 'ヒ' | 'ビ' | 'ピ' | 'ミ' | 'リ' | 'イ' | 'テ' | 'デ' | 'フ' | 'ウ' | 'ヴ')
}

fn is_hiragana(c: char) -> bool {
    ('\u{3041}'..='\u{3096}').contains(&c)
}

/// A katakana syllable (not the ー mark or a dot).
fn is_katakana_letter(c: char) -> bool {
    ('\u{30A1}'..='\u{30FA}').contains(&c)
}

/// A character inside a katakana word: a syllable or ー (not the middle dot, which also
/// separates kanji lists such as 一・二・三).
fn is_kata_word(c: Option<char>) -> bool {
    c.is_some_and(|c| is_katakana_letter(c) || c == 'ー')
}

fn is_kanji_numeral(c: char) -> bool {
    matches!(c, '一' | '二' | '三' | '四' | '五' | '六' | '七' | '八' | '九' | '十' | '百' | '千' | '万' | '〇')
}

fn is_closing(c: char) -> bool {
    matches!(c, '、' | '。' | '，' | '．' | ',' | '.' | ' ' | '　' | ')' | '）' | '」' | '』' | '】')
}

/// Words where the kanji is real: 力アップ (プレゼン力アップ), 口コミ (ホテル口コミ).
fn compound_exception(c: char, tail: &[char]) -> bool {
    let starts = |s: &str| s.chars().count() <= tail.len() && s.chars().zip(tail.iter()).all(|(a, b)| a == *b);
    match c {
        '力' => starts("アップ") || starts("ダウン"),
        '口' => starts("コミ") || starts("座") || starts("ゲンカ"),
        _ => false,
    }
}

fn is_kanji(c: char) -> bool {
    ('\u{4E00}'..='\u{9FFF}').contains(&c) || ('\u{3400}'..='\u{4DBF}').contains(&c) || matches!(c, '々' | '〇')
}

fn is_digit(c: char) -> bool {
    c.is_ascii_digit()
}

fn is_japanese(c: char) -> bool {
    is_hiragana(c) || is_katakana_letter(c) || is_kanji(c)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn line(words: &[&str]) -> Line {
        Line { words: words.iter().map(|t| Word::new(*t, [0.0, 0.0, 10.0, 10.0], "test")).collect() }
    }

    fn fix(texts: &[&[&str]]) -> (Vec<String>, usize) {
        let mut lines: Vec<Line> = texts.iter().map(|w| line(w)).collect();
        let n = fix_japanese(&mut lines).characters;
        (lines.iter().map(Line::text).collect(), n)
    }

    #[test]
    fn real_words_that_look_like_a_misreading_are_left_alone() {
        for t in [
            "リンゴ一つ",
            "セット一つ",
            "ページ一つ目",
            "ボーナス一ヶ月",
            "ボーナス一か月",
            "一・二・三",
            "プレゼン力アップ",
            "ホテル口コミ",
            "ネット口コミで",
            "あっいけない",
            "東京ー大阪",
        ] {
            assert_eq!(fix(&[&[t]]), (vec![t.to_string()], 0), "{t}");
        }
        // An interjection ending a line (えっ) is not turned into えつ by the next line.
        assert_eq!(fix(&[&["えっ"], &["おはよう"]]).1, 0);
        // The genuine misreadings are still corrected.
        assert_eq!(fix(&[&["コーヒ一"]]).0, vec!["コーヒー".to_string()]);
        assert_eq!(fix(&[&["コーヒ一を飲む"]]).0, vec!["コーヒーを飲む".to_string()]);
        assert_eq!(fix(&[&["二ー三"]]).0, vec!["二一三".to_string()]);
    }

    #[test]
    fn a_small_tsu_before_a_vowel_or_na_ma_ra_row_kana_is_tsu() {
        assert_eq!(fix(&[&["見当がっぬ"]]), (vec!["見当がつぬ".to_string()], 1));
        assert_eq!(fix(&[&["やっなん"]]).0, vec!["やつなん".to_string()]);
        assert_eq!(fix(&[&["ホッーム"]]).0, vec!["ホツーム".to_string()]);
    }

    #[test]
    fn a_small_tsu_that_doubles_a_consonant_is_left_alone() {
        for t in ["がっかり", "ちょっと", "きっと", "ベッド", "切手を買った。", "あっ", "ええっ！"] {
            assert_eq!(fix(&[&[t]]), (vec![t.to_string()], 0), "{t}");
        }
    }

    #[test]
    fn a_small_tsu_is_judged_by_the_next_line_when_the_word_wraps() {
        // 「見当がっ|ぬ」: the っ ends a line and the next line starts with ぬ.
        assert_eq!(fix(&[&["見当がっ"], &["ぬ。"]]).0, vec!["見当がつ".to_string(), "ぬ。".to_string()]);
        // 「行っ|た」 is real: た doubles.
        assert_eq!(fix(&[&["学校に行っ"], &["た。"]]).1, 0);
        // The last character of the page has nothing after it: left alone.
        assert_eq!(fix(&[&["あっ"]]).1, 0);
    }

    #[test]
    fn a_small_ya_yu_yo_needs_an_i_row_kana() {
        assert_eq!(fix(&[&["かゃく"]]).0, vec!["かやく".to_string()]);
        assert_eq!(fix(&[&["カョウ"]]).0, vec!["カヨウ".to_string()]);
        for t in ["きゃく", "しゅう", "ちょう", "ミュージック", "シャツ", "てゅ"] {
            assert_eq!(fix(&[&[t]]).1, 0, "{t}");
        }
    }

    #[test]
    fn the_long_vowel_mark_and_katakana_lookalikes_inside_katakana_words() {
        assert_eq!(fix(&[&["コ一ヒ一"]]), (vec!["コーヒー".to_string()], 2));
        assert_eq!(fix(&[&["ス夕ッフ"]]).0, vec!["スタッフ".to_string()]);
        assert_eq!(fix(&[&["サ一バ一"]]).0, vec!["サーバー".to_string()]);
        assert_eq!(fix(&[&["ハロ一"]]).0, vec!["ハロー".to_string()]);
    }

    #[test]
    fn kanji_that_are_really_kanji_stay_kanji() {
        for t in ["一二三", "第一章", "人口が減る", "工場の力", "夕方", "八百屋", "コーヒー一杯", "ビール二本"] {
            assert_eq!(fix(&[&[t]]), (vec![t.to_string()], 0), "{t}");
        }
    }

    #[test]
    fn a_long_vowel_mark_between_two_kanji_is_one() {
        assert_eq!(fix(&[&["第ー章"]]).0, vec!["第一章".to_string()]);
        assert_eq!(fix(&[&["サーバー"]]).1, 0);
    }

    #[test]
    fn a_digit_lookalike_between_digits_is_a_digit() {
        assert_eq!(fix(&[&["2O24年"]]).0, vec!["2024年".to_string()]);
        assert_eq!(fix(&[&["1I1号"]]).0, vec!["111号".to_string()]);
        // Letters beside only one digit stay (OK, 3D, Room 2O).
        assert_eq!(fix(&[&["OK、3D、東京"]]).1, 0);
    }

    #[test]
    fn text_without_japanese_is_never_touched() {
        assert_eq!(fix(&[&["Total 2O24 ー"]]), (vec!["Total 2O24 ー".to_string()], 0));
        assert_eq!(fix(&[&[""]]).1, 0);
        assert_eq!(fix_japanese(&mut []).characters, 0);
    }

    #[test]
    fn corrections_keep_boxes_and_other_words() {
        let mut lines = vec![line(&["東京", "コ一ヒ一"])];
        let rect = lines[0].words[1].rect;
        assert_eq!(fix_japanese(&mut lines).characters, 2);
        assert_eq!(lines[0].words[0].text, "東京");
        assert_eq!(lines[0].words[1].text, "コーヒー");
        assert_eq!(lines[0].words[1].rect, rect);
    }
}
