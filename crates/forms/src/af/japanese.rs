//! Japanese character-width input formats (入力の自動そろえ) for text fields.
//!
//! A form author can ask that what is typed into a field is put into one width or script on
//! commit: 全角カナ, ひらがな, 半角数字, 半角英数 or 全角. Acrobat has no such format of its own, so
//! it is stored as a keystroke script that does the conversion itself on commit: a viewer that
//! runs field JavaScript converts too, and one that doesn't keeps the value as typed. PdfCraft
//! recognises the script by its function name, [`JA_KEYSTROKE`], and runs the conversion natively
//! ([`convert`]); the field dictionary also names the format under [`JA_FORMAT_KEY`], so the choice
//! is still known when another tool drops the field's scripts.

use super::script::js_str;

/// The function the stored keystroke script defines and calls with the format's name.
pub const JA_KEYSTROKE: &str = "PdfCraftJa_Keystroke";

/// The field-dictionary key that records the format by name (`/PdfCraftJaFormat /katakana`).
pub const JA_FORMAT_KEY: &[u8] = b"PdfCraftJaFormat";

/// Which width or script typed text is put into.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JaWidth {
    /// 全角カナ: hiragana and half-width katakana as full-width katakana (ｶﾞ is ガ).
    Katakana,
    /// ひらがな: katakana, full or half width, as hiragana where there is one.
    Hiragana,
    /// 半角数字: full-width digits, signs and hyphens (and a long-vowel mark next to a digit, as
    /// typed in a phone number) as ASCII.
    HalfDigits,
    /// 半角英数: full-width ASCII letters, digits and symbols as ASCII.
    HalfAlnum,
    /// 全角: ASCII and half-width katakana as full width.
    Full,
}

impl JaWidth {
    pub const ALL: [JaWidth; 5] = [JaWidth::Katakana, JaWidth::Hiragana, JaWidth::HalfDigits, JaWidth::HalfAlnum, JaWidth::Full];

    /// The name stored in the script and the field dictionary, and taken by the tools.
    pub fn name(self) -> &'static str {
        match self {
            JaWidth::Katakana => "katakana",
            JaWidth::Hiragana => "hiragana",
            JaWidth::HalfDigits => "halfwidth_digits",
            JaWidth::HalfAlnum => "halfwidth_alnum",
            JaWidth::Full => "fullwidth",
        }
    }

    pub fn from_name(name: &str) -> Option<JaWidth> {
        JaWidth::ALL.into_iter().find(|w| w.name() == name)
    }

    /// How the format is named in a list of them (English; the UI translates it).
    pub fn label(self) -> &'static str {
        match self {
            JaWidth::Katakana => "Full-width katakana",
            JaWidth::Hiragana => "Hiragana",
            JaWidth::HalfDigits => "Half-width digits",
            JaWidth::HalfAlnum => "Half-width letters and digits",
            JaWidth::Full => "Full width",
        }
    }
}

/// U+FF61..=U+FF9F (half-width punctuation, katakana and sound marks) as full width.
const HALF_TO_FULL: [char; 63] = [
    '。', '「', '」', '、', '・', 'ヲ', 'ァ', 'ィ', 'ゥ', 'ェ', 'ォ', 'ャ', 'ュ', 'ョ', 'ッ', 'ー', 'ア', 'イ', 'ウ', 'エ', 'オ', 'カ', 'キ', 'ク',
    'ケ', 'コ', 'サ', 'シ', 'ス', 'セ', 'ソ', 'タ', 'チ', 'ツ', 'テ', 'ト', 'ナ', 'ニ', 'ヌ', 'ネ', 'ノ', 'ハ', 'ヒ', 'フ', 'ヘ', 'ホ', 'マ', 'ミ',
    'ム', 'メ', 'モ', 'ヤ', 'ユ', 'ヨ', 'ラ', 'リ', 'ル', 'レ', 'ロ', 'ワ', 'ン', '゛', '゜',
];

/// Katakana whose voiced form is the next code point (カ is ガ less one).
const VOICED_NEXT: [char; 20] =
    ['カ', 'キ', 'ク', 'ケ', 'コ', 'サ', 'シ', 'ス', 'セ', 'ソ', 'タ', 'チ', 'ツ', 'テ', 'ト', 'ハ', 'ヒ', 'フ', 'ヘ', 'ホ'];

/// Katakana with a semi-voiced form two code points on (ハ, パ).
const SEMI_VOICED: [char; 5] = ['ハ', 'ヒ', 'フ', 'ヘ', 'ホ'];

/// Full-width signs 半角数字 also makes ASCII: ＋ ， － ． （ ）.
const NUMBER_SIGNS: [char; 6] = ['＋', '，', '－', '．', '（', '）'];

fn digit(c: Option<char>) -> bool {
    c.is_some_and(|c| c.is_ascii_digit() || ('０'..='９').contains(&c))
}

/// A sound mark after a kana: 1 voiced (ﾞ or combining), 2 semi-voiced (ﾟ or combining).
fn mark(c: Option<char>) -> u8 {
    match c {
        Some('\u{FF9E}' | '\u{3099}') => 1,
        Some('\u{FF9F}' | '\u{309A}') => 2,
        _ => 0,
    }
}

/// Katakana `k` with sound mark `m` joined, when it has that form.
fn voiced(k: char, m: u8) -> Option<char> {
    let u = u32::from(k);
    let v = match m {
        2 if SEMI_VOICED.contains(&k) => u + 2,
        1 if VOICED_NEXT.contains(&k) => u + 1,
        1 if k == 'ウ' => 0x30F4,
        // ワ ヰ ヱ ヲ: ヷ ヸ ヹ ヺ.
        1 if (0x30EF..=0x30F2).contains(&u) => u + 8,
        1 if k == 'ヽ' => 0x30FE,
        _ => return None,
    };
    char::from_u32(v)
}

/// Katakana as hiragana where there is one (ヷ and the long-vowel mark have none).
fn to_hiragana(k: char) -> char {
    let u = u32::from(k);
    if (0x30A1..=0x30F6).contains(&u) || u == 0x30FD || u == 0x30FE { char::from_u32(u - 0x60).unwrap_or(k) } else { k }
}

/// A hyphen or minus sign: always (‐ ‑ ‒ – −), or only next to a digit (ー ｰ — ―).
fn hyphen(c: char, prev: Option<char>, next: Option<char>) -> bool {
    matches!(c, '\u{2010}'..='\u{2013}' | '\u{2212}') || (matches!(c, 'ー' | 'ｰ' | '\u{2014}' | '\u{2015}') && (digit(prev) || digit(next)))
}

/// `text` put into width `w`. Characters with no counterpart in it are kept as they are.
pub fn convert(w: JaWidth, text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    let mut out = String::with_capacity(text.len());
    let mut i = 0;
    while let Some(&c) = chars.get(i) {
        let prev = i.checked_sub(1).and_then(|p| chars.get(p)).copied();
        let next = chars.get(i + 1).copied();
        i += 1;
        let u = u32::from(c);
        let kana_mode = matches!(w, JaWidth::Katakana | JaWidth::Hiragana);
        if matches!(w, JaWidth::Katakana | JaWidth::Hiragana | JaWidth::Full) {
            // The katakana this character stands for, if it is kana this width changes or joins.
            let half = (0xFF61..=0xFF9F).contains(&u);
            let kana = if half {
                usize::try_from(u - 0xFF61).ok().and_then(|n| HALF_TO_FULL.get(n)).copied()
            } else if kana_mode && ((0x3041..=0x3096).contains(&u) || u == 0x309D || u == 0x309E) {
                char::from_u32(u + 0x60)
            } else if kana_mode && ((0x30A1..=0x30FA).contains(&u) || u == 0x30FD || u == 0x30FE) {
                Some(c)
            } else {
                None
            };
            if let Some(mut k) = kana {
                if let Some(v) = voiced(k, mark(next)) {
                    k = v;
                    i += 1;
                }
                out.push(if w == JaWidth::Hiragana { to_hiragana(k) } else { k });
                continue;
            }
        }
        let c = match w {
            JaWidth::Full => match u {
                0x21..=0x7E => char::from_u32(u + 0xFEE0).unwrap_or(c),
                0x20 => '\u{3000}',
                _ => c,
            },
            JaWidth::HalfAlnum => match u {
                0xFF01..=0xFF5E => char::from_u32(u - 0xFEE0).unwrap_or(c),
                0x3000 => ' ',
                _ if hyphen(c, prev, next) => '-',
                _ => c,
            },
            JaWidth::HalfDigits => match u {
                0xFF10..=0xFF19 => char::from_u32(u - 0xFEE0).unwrap_or(c),
                _ if NUMBER_SIGNS.contains(&c) => char::from_u32(u - 0xFEE0).unwrap_or(c),
                0x3000 => ' ',
                _ if hyphen(c, prev, next) => '-',
                _ => c,
            },
            JaWidth::Katakana | JaWidth::Hiragana => c,
        };
        out.push(c);
    }
    out
}

/// The keystroke script for width `w`: on commit it puts `event.value` into the width as
/// [`convert`] does, so viewers that run field JavaScript convert too. It defines and then calls
/// [`JA_KEYSTROKE`] with the width's name, which is how PdfCraft reads it back.
pub fn keystroke_js(w: JaWidth) -> String {
    let chars = |cs: &[char]| js_str(&cs.iter().collect::<String>());
    let signs = chars(&NUMBER_SIGNS);
    format!(
        "var {JA_KEYSTROKE} = function (m) {{\n\
         if (!event.willCommit) return;\n\
         var s = String(event.value), o = \"\", n = s.length, i, c, k, p, x, t, v;\n\
         var H = {half}, V = {voiced}, P = {semi}, N = {signs};\n\
         var kanaMode = m == \"katakana\" || m == \"hiragana\";\n\
         function dg(d) {{ return (d >= 48 && d <= 57) || (d >= 0xFF10 && d <= 0xFF19); }}\n\
         function mk(d) {{ return d == 0xFF9E || d == 0x3099 ? 1 : d == 0xFF9F || d == 0x309A ? 2 : 0; }}\n\
         function vo(a, b) {{\n\
         var q = String.fromCharCode(a);\n\
         if (b == 2) return P.indexOf(q) >= 0 ? a + 2 : 0;\n\
         if (b != 1) return 0;\n\
         if (V.indexOf(q) >= 0) return a + 1;\n\
         if (a == 0x30A6) return 0x30F4;\n\
         if (a >= 0x30EF && a <= 0x30F2) return a + 8;\n\
         if (a == 0x30FD) return 0x30FE;\n\
         return 0;\n\
         }}\n\
         function hy(d) {{ return (d >= 0x2010 && d <= 0x2013) || d == 0x2212 || ((d == 0x30FC || d == 0xFF70 || d == 0x2014 || d == 0x2015) && (dg(p) || dg(x))); }}\n\
         for (i = 0; i < n; i++) {{\n\
         c = s.charCodeAt(i); p = i > 0 ? s.charCodeAt(i - 1) : 0; x = i + 1 < n ? s.charCodeAt(i + 1) : 0; k = 0;\n\
         if (kanaMode || m == \"fullwidth\") {{\n\
         if (c >= 0xFF61 && c <= 0xFF9F) k = H.charCodeAt(c - 0xFF61);\n\
         else if (kanaMode && ((c >= 0x3041 && c <= 0x3096) || c == 0x309D || c == 0x309E)) k = c + 0x60;\n\
         else if (kanaMode && ((c >= 0x30A1 && c <= 0x30FA) || c == 0x30FD || c == 0x30FE)) k = c;\n\
         }}\n\
         if (k) {{\n\
         t = mk(x); v = t ? vo(k, t) : 0;\n\
         if (v) {{ k = v; i++; }}\n\
         if (m == \"hiragana\" && ((k >= 0x30A1 && k <= 0x30F6) || k == 0x30FD || k == 0x30FE)) k -= 0x60;\n\
         o += String.fromCharCode(k); continue;\n\
         }}\n\
         if (m == \"fullwidth\") {{ if (c >= 0x21 && c <= 0x7E) c += 0xFEE0; else if (c == 0x20) c = 0x3000; }}\n\
         else if (m == \"halfwidth_alnum\") {{ if (c >= 0xFF01 && c <= 0xFF5E) c -= 0xFEE0; else if (c == 0x3000) c = 0x20; else if (hy(c)) c = 45; }}\n\
         else if (m == \"halfwidth_digits\") {{ if ((c >= 0xFF10 && c <= 0xFF19) || N.indexOf(String.fromCharCode(c)) >= 0) c -= 0xFEE0; else if (c == 0x3000) c = 0x20; else if (hy(c)) c = 45; }}\n\
         o += String.fromCharCode(c);\n\
         }}\n\
         event.value = o;\n\
         }};\n\
         {JA_KEYSTROKE}({name});",
        half = chars(&HALF_TO_FULL),
        voiced = chars(&VOICED_NEXT),
        semi = chars(&SEMI_VOICED),
        name = js_str(w.name()),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn half_width_kana_with_sound_marks_joins_into_full_width_katakana() {
        assert_eq!(convert(JaWidth::Katakana, "ｶﾞｯｺｳ"), "ガッコウ");
        assert_eq!(convert(JaWidth::Katakana, "ﾊﾟﾋﾟﾌﾟﾍﾟﾎﾟ"), "パピプペポ");
        assert_eq!(convert(JaWidth::Katakana, "ｳﾞｧｲｵﾘﾝ"), "ヴァイオリン");
        assert_eq!(convert(JaWidth::Katakana, "ｺｰﾋｰ｡"), "コーヒー。");
        // A mark that can't join is kept as the full-width mark.
        assert_eq!(convert(JaWidth::Katakana, "ｱﾞﾟ"), "ア゛゜");
        // Hiragana, with a combining voiced mark, too.
        assert_eq!(convert(JaWidth::Katakana, "がっこう か\u{3099}"), "ガッコウ ガ");
    }

    #[test]
    fn hiragana_takes_katakana_of_both_widths() {
        assert_eq!(convert(JaWidth::Hiragana, "ｶﾞｯｺｳ"), "がっこう");
        assert_eq!(convert(JaWidth::Hiragana, "ヤマダ タロウ"), "やまだ たろう");
        assert_eq!(convert(JaWidth::Hiragana, "ヴヵヶヽヾ"), "ゔゕゖゝゞ");
        // ヷ and the long-vowel mark have no hiragana.
        assert_eq!(convert(JaWidth::Hiragana, "ヷーｰ"), "ヷーー");
    }

    #[test]
    fn half_width_digits_from_phone_and_postal_numbers() {
        assert_eq!(convert(JaWidth::HalfDigits, "０３－１２３４－５６７８"), "03-1234-5678");
        assert_eq!(convert(JaWidth::HalfDigits, "〒１２３ー４５６７"), "〒123-4567");
        assert_eq!(convert(JaWidth::HalfDigits, "（０３）１２３４‐５６７８"), "(03)1234-5678");
        // A long-vowel mark between letters stays: only one next to a digit is a hyphen.
        assert_eq!(convert(JaWidth::HalfDigits, "ラーメン１"), "ラーメン1");
        // Letters are not this width's business.
        assert_eq!(convert(JaWidth::HalfDigits, "ＡＢ１２"), "ＡＢ12");
    }

    #[test]
    fn half_width_letters_and_digits() {
        assert_eq!(convert(JaWidth::HalfAlnum, "ＰｄｆＣｒａｆｔ　２０２６！"), "PdfCraft 2026!");
        assert_eq!(convert(JaWidth::HalfAlnum, "ｔａｒｏ＠ｅｘａｍｐｌｅ．ｊｐ"), "taro@example.jp");
        assert_eq!(convert(JaWidth::HalfAlnum, "東京ｶﾅ"), "東京ｶﾅ");
    }

    #[test]
    fn full_width_takes_ascii_and_half_width_kana() {
        assert_eq!(convert(JaWidth::Full, "ABC 123-ｶﾞ"), "ＡＢＣ\u{3000}１２３－ガ");
        assert_eq!(convert(JaWidth::Full, "ひらがな"), "ひらがな");
    }

    #[test]
    fn text_already_in_the_width_is_unchanged() {
        for (w, t) in [
            (JaWidth::Katakana, "ガッコウ ヴァ 東京 ABC"),
            (JaWidth::Hiragana, "がっこう ー 漢字 123"),
            (JaWidth::HalfDigits, "03-1234-5678"),
            (JaWidth::HalfAlnum, "PdfCraft 2026! 漢字"),
            (JaWidth::Full, "ＡＢＣ　ガッコウ　漢字"),
        ] {
            assert_eq!(convert(w, t), t, "{w:?}");
        }
        assert_eq!(convert(JaWidth::Katakana, ""), "");
    }

    #[test]
    fn characters_without_a_counterpart_are_kept() {
        assert_eq!(convert(JaWidth::Katakana, "漢字😀Aé\u{3099}"), "漢字😀Aé\u{3099}");
        assert_eq!(convert(JaWidth::HalfAlnum, "￥¥〜"), "￥¥〜");
        assert_eq!(convert(JaWidth::Full, "é😀"), "é😀");
    }

    #[test]
    fn names_round_trip() {
        for w in JaWidth::ALL {
            assert_eq!(JaWidth::from_name(w.name()), Some(w));
        }
        assert_eq!(JaWidth::from_name("Katakana"), None);
    }

    #[test]
    fn the_script_is_ascii_and_read_back_by_its_function_name() {
        for w in JaWidth::ALL {
            let js = keystroke_js(w);
            assert!(js.is_ascii(), "stored as a PDF literal string: {js}");
            assert_eq!(super::super::parse_format(&js), Some(super::super::Format::Japanese(w)));
        }
    }
}
