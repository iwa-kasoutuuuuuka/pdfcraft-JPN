//! Japanese notation folding for search: one canonical form for text that is written in several
//! equivalent ways (表記ゆれ), with a map back to the characters it came from.
//!
//! Three kinds of variation are folded:
//! - width: full-width ASCII (ＡＢＣ１２３) and the ideographic space read as ASCII, half-width
//!   katakana (ｶﾀｶﾅ) as full-width katakana, with a following half-width voiced or semi-voiced
//!   mark (ﾞ ﾟ) or a combining one (U+3099, U+309A) joined to its kana (ｶﾞ is ガ);
//! - kana: hiragana read as katakana (ひらがな is ヒラガナ);
//! - kanji: old forms (旧字体) read as the new form (新字体), from a small hand-written table of
//!   common pairs (國 is 国). The table is data in this file, not a complete Unicode mapping.
//!
//! The long-vowel mark, small kana and letter case are left alone here (case is folded by the
//! caller, as it is for every search).
use std::collections::HashMap;
use std::ops::Range;
use std::sync::OnceLock;

/// Characters folded to the canonical form, and for each one the range of the input characters
/// it stands for (one character, or two where a voiced mark was joined to its kana).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Folded {
    pub chars: Vec<char>,
    /// `origin[i]` is the range of input character indexes that produced `chars[i]`.
    pub origin: Vec<Range<usize>>,
}

/// Old kanji forms (旧字体) and the new form (新字体 or the common modern form) each is read as.
/// Pairs of common characters in names and place names; modest on purpose.
const OLD_NEW: &[(char, char)] = &[
    ('國', '国'),
    ('學', '学'),
    ('會', '会'),
    ('體', '体'),
    ('澤', '沢'),
    ('齋', '斎'),
    ('齊', '斉'),
    ('邊', '辺'),
    ('邉', '辺'),
    ('髙', '高'),
    ('\u{FA11}', '崎'),
    ('德', '徳'),
    ('廣', '広'),
    ('當', '当'),
    ('發', '発'),
    ('實', '実'),
    ('對', '対'),
    ('戰', '戦'),
    ('變', '変'),
    ('傳', '伝'),
    ('圓', '円'),
    ('圖', '図'),
    ('擔', '担'),
    ('條', '条'),
    ('澁', '渋'),
    ('絲', '糸'),
    ('經', '経'),
    ('縣', '県'),
    ('驛', '駅'),
    ('驗', '験'),
    ('營', '営'),
    ('榮', '栄'),
    ('藝', '芸'),
    ('劍', '剣'),
    ('嚴', '厳'),
    ('歐', '欧'),
    ('假', '仮'),
    ('價', '価'),
    ('畫', '画'),
    ('權', '権'),
    ('歡', '歓'),
    ('觀', '観'),
    ('關', '関'),
    ('歸', '帰'),
    ('氣', '気'),
    ('龜', '亀'),
    ('區', '区'),
    ('輕', '軽'),
    ('藏', '蔵'),
    ('繼', '継'),
    ('縱', '縦'),
    ('從', '従'),
    ('獸', '獣'),
    ('讓', '譲'),
    ('釀', '醸'),
    ('辭', '辞'),
    ('濕', '湿'),
    ('壽', '寿'),
    ('收', '収'),
    ('處', '処'),
    ('稱', '称'),
    ('證', '証'),
    ('乘', '乗'),
    ('淨', '浄'),
    ('狀', '状'),
    ('讀', '読'),
    ('賣', '売'),
    ('髮', '髪'),
    ('拂', '払'),
    ('佛', '仏'),
    ('辯', '弁'),
    ('瓣', '弁'),
    ('辨', '弁'),
    ('寶', '宝'),
    ('豐', '豊'),
    ('默', '黙'),
    ('譯', '訳'),
    ('與', '与'),
    ('豫', '予'),
    ('搖', '揺'),
    ('樣', '様'),
    ('禮', '礼'),
    ('靈', '霊'),
    ('勵', '励'),
    ('戀', '恋'),
    ('勞', '労'),
    ('灣', '湾'),
    ('兩', '両'),
    ('聽', '聴'),
    ('廳', '庁'),
    ('應', '応'),
    ('櫻', '桜'),
    ('鹽', '塩'),
    ('產', '産'),
    ('齒', '歯'),
    ('數', '数'),
    ('斷', '断'),
    ('晝', '昼'),
    ('鐵', '鉄'),
    ('轉', '転'),
    ('點', '点'),
    ('殘', '残'),
    ('毆', '殴'),
    ('單', '単'),
    ('淺', '浅'),
    ('續', '続'),
    ('屬', '属'),
    ('屆', '届'),
    ('賴', '頼'),
    ('來', '来'),
    ('號', '号'),
    ('檢', '検'),
    ('險', '険'),
    ('顯', '顕'),
    ('惡', '悪'),
    ('圍', '囲'),
    ('爲', '為'),
    ('舊', '旧'),
    ('據', '拠'),
    ('擧', '挙'),
    ('峽', '峡'),
    ('狹', '狭'),
    ('挾', '挟'),
    ('莊', '荘'),
    ('裝', '装'),
    ('壯', '壮'),
    ('僞', '偽'),
    ('儉', '倹'),
    ('劑', '剤'),
    ('效', '効'),
    ('勸', '勧'),
    ('卷', '巻'),
    ('參', '参'),
    ('慘', '惨'),
    ('勳', '勲'),
    ('圈', '圏'),
    ('壓', '圧'),
    ('增', '増'),
    ('墮', '堕'),
    ('奧', '奥'),
    ('孃', '嬢'),
    ('穩', '穏'),
    ('專', '専'),
    ('將', '将'),
    ('尙', '尚'),
    ('巖', '巌'),
    ('帶', '帯'),
    ('懷', '懐'),
    ('戲', '戯'),
    ('戶', '戸'),
    ('敍', '叙'),
    ('敘', '叙'),
    ('樞', '枢'),
    ('歷', '歴'),
    ('曆', '暦'),
    ('歲', '歳'),
    ('溫', '温'),
    ('滿', '満'),
    ('潛', '潜'),
    ('瀧', '滝'),
    ('燒', '焼'),
    ('燈', '灯'),
    ('爭', '争'),
    ('獨', '独'),
    ('獵', '猟'),
    ('疊', '畳'),
    ('瘦', '痩'),
    ('盜', '盗'),
    ('眞', '真'),
    ('碎', '砕'),
    ('禪', '禅'),
    ('稅', '税'),
    ('總', '総'),
    ('繩', '縄'),
    ('繪', '絵'),
    ('羣', '群'),
    ('肅', '粛'),
    ('腦', '脳'),
    ('臟', '臓'),
    ('莖', '茎'),
    ('藥', '薬'),
    ('蟲', '虫'),
    ('蠶', '蚕'),
    ('衞', '衛'),
    ('襃', '褒'),
    ('覺', '覚'),
    ('觸', '触'),
    ('譽', '誉'),
    ('讚', '讃'),
    ('貳', '弐'),
    ('贊', '賛'),
    ('遞', '逓'),
    ('遲', '遅'),
    ('遙', '遥'),
    ('醉', '酔'),
    ('醫', '医'),
    ('釋', '釈'),
    ('鑛', '鉱'),
    ('鑄', '鋳'),
    ('陷', '陥'),
    ('隱', '隠'),
    ('雜', '雑'),
    ('靜', '静'),
    ('顏', '顔'),
    ('飜', '翻'),
    ('騷', '騒'),
    ('鬪', '闘'),
    ('鷄', '鶏'),
    ('麥', '麦'),
    ('黃', '黄'),
    ('黑', '黒'),
    ('齡', '齢'),
    ('龍', '竜'),
];

fn kanji_table() -> &'static HashMap<char, char> {
    static TABLE: OnceLock<HashMap<char, char>> = OnceLock::new();
    TABLE.get_or_init(|| OLD_NEW.iter().copied().collect())
}

/// Half-width katakana U+FF71..U+FF9D (ｱ..ﾝ) as full-width katakana.
const HALF_KATAKANA: [char; 45] = [
    'ア', 'イ', 'ウ', 'エ', 'オ', 'カ', 'キ', 'ク', 'ケ', 'コ', 'サ', 'シ', 'ス', 'セ', 'ソ', 'タ', 'チ', 'ツ', 'テ', 'ト', 'ナ', 'ニ', 'ヌ', 'ネ',
    'ノ', 'ハ', 'ヒ', 'フ', 'ヘ', 'ホ', 'マ', 'ミ', 'ム', 'メ', 'モ', 'ヤ', 'ユ', 'ヨ', 'ラ', 'リ', 'ル', 'レ', 'ロ', 'ワ', 'ン',
];
/// Half-width small katakana U+FF67..U+FF6F (ｧ..ｯ).
const HALF_SMALL: [char; 9] = ['ァ', 'ィ', 'ゥ', 'ェ', 'ォ', 'ャ', 'ュ', 'ョ', 'ッ'];

/// A half-width character as its full-width form, or `None` when it has none to fold to.
fn half_to_full(c: char) -> Option<char> {
    let u = u32::from(c);
    match u {
        0xFF61 => Some('。'),
        0xFF62 => Some('「'),
        0xFF63 => Some('」'),
        0xFF64 => Some('、'),
        0xFF65 => Some('・'),
        0xFF66 => Some('ヲ'),
        0xFF67..=0xFF6F => HALF_SMALL.get(usize::try_from(u - 0xFF67).ok()?).copied(),
        0xFF70 => Some('ー'),
        0xFF71..=0xFF9D => HALF_KATAKANA.get(usize::try_from(u - 0xFF71).ok()?).copied(),
        _ => None,
    }
}

/// A voiced (`true`) or semi-voiced (`false`) mark that follows a kana: half-width (ﾞ ﾟ),
/// spacing (゛ ゜) or combining.
fn mark_kind(c: char) -> Option<bool> {
    match c {
        '\u{FF9E}' | '\u{309B}' | '\u{3099}' => Some(true),
        '\u{FF9F}' | '\u{309C}' | '\u{309A}' => Some(false),
        _ => None,
    }
}

/// A katakana with its voiced (`true`) or semi-voiced (`false`) mark joined, if the kana takes one.
fn with_mark(base: char, voiced: bool) -> Option<char> {
    const PLAIN: &str = "カキクケコサシスセソタチツテトハヒフヘホウ";
    const VOICED: &str = "ガギグゲゴザジズゼゾダヂヅデドバビブベボヴ";
    const SEMI: &str = "ハヒフヘホ";
    const PAIRS: &str = "パピプペポ";
    if voiced {
        let at = PLAIN.chars().position(|c| c == base)?;
        VOICED.chars().nth(at)
    } else {
        let at = SEMI.chars().position(|c| c == base)?;
        PAIRS.chars().nth(at)
    }
}

/// One character on its own folded: width, hiragana to katakana, old kanji to new.
fn fold_one(c: char) -> char {
    let u = u32::from(c);
    match u {
        0xFF01..=0xFF5E => char::from_u32(u - 0xFEE0).unwrap_or(c),
        0x3000 => ' ',
        0xFF61..=0xFF9D => half_to_full(c).unwrap_or(c),
        // Hiragana letters ぁ..ゖ and the iteration marks ゝ ゞ to katakana.
        0x3041..=0x3096 | 0x309D | 0x309E => char::from_u32(u + 0x60).unwrap_or(c),
        // Unicode keeps the old and new forms apart; only the table joins them.
        0x4E00..=0x9FFF | 0xF900..=0xFAFF => kanji_table().get(&c).copied().unwrap_or(c),
        _ => c,
    }
}

/// Fold `input` to the canonical form, remembering for each output character where it came from.
/// Yields at most one output character per input character.
pub fn fold_chars(input: &[char]) -> Folded {
    let mut chars = Vec::with_capacity(input.len());
    let mut origin = Vec::with_capacity(input.len());
    let mut i = 0;
    while let Some(&c) = input.get(i) {
        let base = fold_one(c);
        // A mark right after a kana joins to it (ｶﾞ, か゛, か + U+3099).
        if let Some(voiced) = input.get(i + 1).copied().and_then(mark_kind)
            && let Some(joined) = with_mark(base, voiced)
        {
            chars.push(joined);
            origin.push(i..i + 2);
            i += 2;
            continue;
        }
        chars.push(base);
        origin.push(i..i + 1);
        i += 1;
    }
    Folded { chars, origin }
}

/// `text` folded to the canonical form, as a string.
pub fn fold_text(text: &str) -> String {
    let input: Vec<char> = text.chars().collect();
    fold_chars(&input).chars.into_iter().collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn width_is_folded() {
        assert_eq!(fold_text("ＡＢＣ１２３"), "ABC123");
        assert_eq!(fold_text("ｶﾀｶﾅ"), "カタカナ");
        assert_eq!(fold_text("ｶﾞｷﾞﾊﾟｳﾞ"), "ガギパヴ");
        assert_eq!(fold_text("ｱｲｳｴｵｧｨｯｰ"), "アイウエオァィッー");
        assert_eq!(fold_text("あ\u{3000}い"), "ア イ");
    }

    #[test]
    fn hiragana_reads_as_katakana() {
        assert_eq!(fold_text("ひらがな"), "ヒラガナ");
        assert_eq!(fold_text("がっこう"), fold_text("ガッコウ"));
        assert_eq!(fold_text("か\u{3099}"), "ガ");
        assert_eq!(fold_text("は\u{309A}"), "パ");
        // A mark on a kana that takes none is kept as it is.
        assert_eq!(fold_text("ア\u{FF9E}"), "ア\u{FF9E}");
    }

    #[test]
    fn old_kanji_read_as_new() {
        assert_eq!(fold_text("國學會體澤齋邊髙\u{FA11}"), "国学会体沢斎辺高崎");
        assert_eq!(fold_text("東京大學"), "東京大学");
    }

    #[test]
    fn the_kanji_table_is_consistent() {
        let table = kanji_table();
        assert_eq!(table.len(), OLD_NEW.len(), "an old form is listed twice");
        for (old, new) in OLD_NEW {
            assert_ne!(old, new);
            assert!(!table.contains_key(new), "{new} is both an old and a new form");
        }
        assert!(OLD_NEW.len() >= 100);
    }

    #[test]
    fn origin_maps_back_to_input_characters() {
        let input: Vec<char> = "Aｶﾞ國b".chars().collect();
        let f = fold_chars(&input);
        assert_eq!(f.chars, vec!['A', 'ガ', '国', 'b']);
        assert_eq!(f.origin, vec![0..1, 1..3, 3..4, 4..5]);
    }

    #[test]
    fn empty_and_lone_marks_are_safe() {
        assert!(fold_chars(&[]).chars.is_empty());
        assert_eq!(fold_text("\u{FF9E}"), "\u{FF9E}");
        assert_eq!(fold_text("ｶ"), "カ");
    }
}
