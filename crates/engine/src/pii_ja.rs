//! Personal information as written in Japan, found for redaction (個人情報の自動墨消し).
//!
//! The matchers are hand-written and read the page text after Japanese notation folding
//! ([`pdfcraft_render::text::japanese`]), so full-width digits and letters (０９０－１２３４),
//! dash variants and half-width forms are read as the ASCII they stand for; hits are mapped back
//! to the characters as written. Each category is a precision-first guess meant for review: the
//! hits are marked for redaction and the person looks at them before applying.
//!
//! - My Number (マイナンバー): twelve digits, grouped 4-4-4 or not, whose last digit is the
//!   official check digit. A twelve-digit run with a wrong check digit is not a hit.
//! - Phone numbers (電話番号): landline `0X-XXXX-XXXX` in any area code length, mobile and IP
//!   `070`/`080`/`090`/`050`, `0120`/`0570` and `0800`, with hyphens, spaces, brackets, or none.
//! - Postal codes (郵便番号): `〒123-4567`, `〒1234567` and a bare `123-4567`.
//! - Email addresses (メールアドレス).
//! - Street addresses (住所), a heuristic: a prefecture name, a municipality ending in 市, 区, 町,
//!   村 or 郡, and a street number. A prefecture and city alone, with no number, is not a hit.
use pdfcraft_render::PageText;
use pdfcraft_render::text::japanese::fold_chars;
use std::ops::Range;

/// A kind of personal information.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum PiiCategory {
    MyNumber,
    Phone,
    PostalCode,
    Email,
    Address,
}

/// Every category, in the order they are offered and win an overlap.
pub const PII_CATEGORIES: [PiiCategory; 5] =
    [PiiCategory::MyNumber, PiiCategory::Phone, PiiCategory::PostalCode, PiiCategory::Email, PiiCategory::Address];

/// The categories chosen when none are named: all but the street address heuristic.
pub const PII_DEFAULT_CATEGORIES: [PiiCategory; 4] = [PiiCategory::MyNumber, PiiCategory::Phone, PiiCategory::PostalCode, PiiCategory::Email];

impl PiiCategory {
    /// The stable name used by the automation tool.
    pub fn id(self) -> &'static str {
        match self {
            PiiCategory::MyNumber => "my-number",
            PiiCategory::Phone => "phone",
            PiiCategory::PostalCode => "postal-code",
            PiiCategory::Email => "email",
            PiiCategory::Address => "address",
        }
    }

    /// The English label (translated by the interface).
    pub fn label(self) -> &'static str {
        match self {
            PiiCategory::MyNumber => "My Number (individual number)",
            PiiCategory::Phone => "Phone numbers (Japan)",
            PiiCategory::PostalCode => "Postal codes (Japan)",
            PiiCategory::Email => "Email addresses (personal information)",
            PiiCategory::Address => "Street addresses (Japan, approximate)",
        }
    }

    pub fn from_id(id: &str) -> Option<Self> {
        PII_CATEGORIES.into_iter().find(|c| c.id() == id)
    }
}

/// Whether the twelve digits are a My Number: the last is the check digit of the first eleven
/// (the Digital Agency's algorithm: the digits from the right, P1 to P11, are weighted 2 to 7
/// and then 2 to 6; the check digit is 11 minus the sum modulo 11, or 0 when the remainder is
/// 0 or 1).
pub fn is_valid_my_number(digits: &[u8; 12]) -> bool {
    let mut sum = 0u32;
    for n in 1..=11usize {
        let Some(&p) = digits.get(11 - n) else { return false };
        let q = if n <= 6 { n + 1 } else { n - 5 };
        sum += u32::from(p) * u32::try_from(q).unwrap_or(0);
    }
    let remainder = sum % 11;
    let check = if remainder <= 1 { 0 } else { 11 - remainder };
    digits.get(11).is_some_and(|&d| u32::from(d) == check)
}

fn is_digit(c: Option<&char>) -> bool {
    c.is_some_and(char::is_ascii_digit)
}

/// A dash of any kind (hyphen, non-breaking hyphen, en and em dash, minus sign, horizontal bar,
/// and the long-vowel mark that is often typed for one in a phone number) as a hyphen-minus.
fn dash(c: char) -> char {
    if matches!(c, '\u{2010}'..='\u{2015}' | '\u{2212}' | '\u{30FC}' | '\u{FF0D}' | '\u{FE63}') { '-' } else { c }
}

/// Whether the number starting at `start` is the tail of a longer one: a digit, or a hyphen
/// after a digit, comes right before it. With `spaces`, a space counts like a hyphen.
fn glued_before(t: &[char], start: usize, spaces: bool) -> bool {
    let Some(prev) = start.checked_sub(1).and_then(|p| t.get(p)) else { return false };
    if prev.is_ascii_digit() {
        return true;
    }
    (*prev == '-' || (spaces && *prev == ' ')) && start.checked_sub(2).and_then(|p| t.get(p)).is_some_and(char::is_ascii_digit)
}

/// Whether the number ending at `end` (exclusive) runs on: a digit, or a hyphen and a digit,
/// follows. With `spaces`, a space counts like a hyphen.
fn glued_after(t: &[char], end: usize, spaces: bool) -> bool {
    match t.get(end) {
        Some(c) if c.is_ascii_digit() => true,
        Some(c) if *c == '-' || (spaces && *c == ' ') => is_digit(end.checked_add(1).and_then(|p| t.get(p))),
        _ => false,
    }
}

/// The length of the digit run at `at`, at most `max`.
fn run(t: &[char], at: usize, max: usize) -> usize {
    t.iter().skip(at).take(max).take_while(|c| c.is_ascii_digit()).count()
}

/// The end of a My Number starting at `i`: twelve digits in a row, or three groups of four with
/// the same separator, a hyphen or a space.
fn my_number_at(t: &[char], i: usize) -> Option<usize> {
    let digit = |k: usize| t.get(k).and_then(|c| c.to_digit(10)).and_then(|d| u8::try_from(d).ok());
    let (indexes, end): (Vec<usize>, usize) = if run(t, i, 12) == 12 {
        ((i..i + 12).collect(), i + 12)
    } else {
        let sep = *t.get(i + 4)?;
        if !(sep == '-' || sep == ' ') || t.get(i + 9) != Some(&sep) {
            return None;
        }
        if run(t, i, 4) != 4 || run(t, i + 5, 4) != 4 || run(t, i + 10, 4) != 4 {
            return None;
        }
        ((i..i + 4).chain(i + 5..i + 9).chain(i + 10..i + 14).collect(), i + 14)
    };
    let digits: Vec<u8> = indexes.into_iter().filter_map(digit).collect();
    let digits = <[u8; 12]>::try_from(digits.as_slice()).ok()?;
    (is_valid_my_number(&digits) && !glued_after(t, end, true)).then_some(end)
}

fn my_numbers(t: &[char]) -> Vec<Range<usize>> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < t.len() {
        match (is_digit(t.get(i)) && !glued_before(t, i, true)).then(|| my_number_at(t, i)).flatten() {
            Some(end) => {
                out.push(i..end);
                i = end;
            }
            None => i += 1,
        }
    }
    out
}

/// The length of a phone number starting at `i`, or `None`.
fn phone_at(t: &[char], i: usize) -> Option<usize> {
    let open = t.get(i) == Some(&'(');
    let mut p = i + usize::from(open);
    if t.get(p) != Some(&'0') {
        return None;
    }
    let mut groups: Vec<(usize, usize)> = Vec::new(); // (start, len)
    let (mut outer, mut inner) = (open, false);
    loop {
        let len = run(t, p, 12);
        if len == 0 || len >= 12 {
            return None;
        }
        groups.push((p, len));
        p += len;
        if groups.len() == 3 {
            break;
        }
        match t.get(p) {
            Some('-' | ' ') => p += 1,
            Some(')') if outer || inner => {
                (outer, inner) = (false, false);
                p += 1;
                if matches!(t.get(p), Some('-' | ' ')) {
                    p += 1;
                }
            }
            Some('(') if !outer && !inner && groups.len() == 1 => {
                inner = true;
                p += 1;
            }
            _ => break,
        }
    }
    if outer || inner || open && groups.len() == 1 {
        return None;
    }
    let text = |g: &(usize, usize)| -> String { t.iter().skip(g.0).take(g.1).collect() };
    let lens: Vec<usize> = groups.iter().map(|g| g.1).collect();
    let first = text(groups.first()?);
    let mut chars = first.chars();
    if chars.next() != Some('0') || chars.next().is_none_or(|c| c == '0') {
        return None;
    }
    let ok = match lens.as_slice() {
        // Not separated: ten digits for a landline or 0120, eleven for a mobile or 0800.
        [10] => true,
        [11] => ["050", "070", "080", "090", "0800"].iter().any(|prefix| first.starts_with(prefix)),
        [a, b, c] => {
            let mobile = ["050", "070", "080", "090"].contains(&first.as_str());
            (mobile && (*b, *c) == (4, 4))
                || (first == "0800" && (*b, *c) == (3, 4))
                || (matches!(first.as_str(), "0120" | "0570") && (*b, *c) == (3, 3))
                || (!mobile && (2..=5).contains(a) && a + b == 6 && *c == 4)
        }
        _ => false,
    };
    let end = p;
    (ok && !glued_before(t, if open { i + 1 } else { i }, false) && !glued_after(t, end, false)).then_some(end - i)
}

fn phones(t: &[char]) -> Vec<Range<usize>> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < t.len() {
        match phone_at(t, i) {
            Some(n) if n > 0 => {
                out.push(i..i + n);
                i += n;
            }
            _ => i += 1,
        }
    }
    out
}

fn postal_codes(t: &[char]) -> Vec<Range<usize>> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < t.len() {
        let Some(&c) = t.get(i) else { break };
        let mut found = None;
        if c == '〒' {
            let mut p = i + 1;
            if t.get(p) == Some(&' ') {
                p += 1;
            }
            if run(t, p, 3) == 3 {
                let q = p + 3;
                let r = if t.get(q) == Some(&'-') { q + 1 } else { q };
                if run(t, r, 4) == 4 && !is_digit(t.get(r + 4)) {
                    found = Some(r + 4);
                }
            }
        } else if c.is_ascii_digit()
            && !glued_before(t, i, false)
            && run(t, i, 3) == 3
            && t.get(i + 3) == Some(&'-')
            && run(t, i + 4, 4) == 4
            && !glued_after(t, i + 8, false)
        {
            found = Some(i + 8);
        }
        match found {
            Some(end) => {
                out.push(i..end);
                i = end;
            }
            None => i += 1,
        }
    }
    out
}

fn is_local(c: char) -> bool {
    c.is_ascii_alphanumeric()
        || matches!(c, '.' | '_' | '%' | '+' | '-' | '\'' | '!' | '#' | '$' | '&' | '*' | '/' | '=' | '?' | '^' | '`' | '{' | '|' | '}' | '~')
}

fn emails(t: &[char]) -> Vec<Range<usize>> {
    let mut out: Vec<Range<usize>> = Vec::new();
    for (at, &c) in t.iter().enumerate() {
        if c != '@' {
            continue;
        }
        let mut start = at;
        while start > 0 && t.get(start - 1).is_some_and(|c| is_local(*c)) {
            start -= 1;
        }
        // The local part may not begin with a dot.
        while t.get(start) == Some(&'.') && start < at {
            start += 1;
        }
        let mut end = at + 1;
        while t.get(end).is_some_and(|c| c.is_ascii_alphanumeric() || *c == '.' || *c == '-') {
            end += 1;
        }
        // Not to end on a dot or hyphen (a sentence's full stop).
        while end > at + 1 && matches!(t.get(end - 1), Some('.' | '-')) {
            end -= 1;
        }
        let domain: String = t.iter().skip(at + 1).take(end.saturating_sub(at + 1)).collect();
        let labels: Vec<&str> = domain.split('.').collect();
        let tld_ok = labels.last().is_some_and(|l| l.len() >= 2 && l.chars().all(|c| c.is_ascii_alphabetic()));
        let labels_ok = labels.iter().all(|l| !l.is_empty() && !l.starts_with('-') && !l.ends_with('-'));
        if start < at && labels.len() >= 2 && tld_ok && labels_ok && out.last().is_none_or(|r| r.end <= start) {
            out.push(start..end);
        }
    }
    out
}

const PREFECTURES: [&str; 47] = [
    "北海道",
    "青森県",
    "岩手県",
    "宮城県",
    "秋田県",
    "山形県",
    "福島県",
    "茨城県",
    "栃木県",
    "群馬県",
    "埼玉県",
    "千葉県",
    "東京都",
    "神奈川県",
    "新潟県",
    "富山県",
    "石川県",
    "福井県",
    "山梨県",
    "長野県",
    "岐阜県",
    "静岡県",
    "愛知県",
    "三重県",
    "滋賀県",
    "京都府",
    "大阪府",
    "兵庫県",
    "奈良県",
    "和歌山県",
    "鳥取県",
    "島根県",
    "岡山県",
    "広島県",
    "山口県",
    "徳島県",
    "香川県",
    "愛媛県",
    "高知県",
    "福岡県",
    "佐賀県",
    "長崎県",
    "熊本県",
    "大分県",
    "宮崎県",
    "鹿児島県",
    "沖縄県",
];

/// Characters after the prefecture that end the address: spaces, punctuation and brackets.
fn ends_address(c: char) -> bool {
    c.is_whitespace() || matches!(c, '、' | '。' | ',' | ';' | ':' | '(' | ')' | '「' | '」' | '『' | '』' | '[' | ']' | '\n')
}

fn addresses(t: &[char]) -> Vec<Range<usize>> {
    let mut out = Vec::new();
    let mut i = 0;
    'scan: while i < t.len() {
        for prefecture in PREFECTURES {
            let name: Vec<char> = prefecture.chars().collect();
            if t.get(i..i + name.len()) != Some(name.as_slice()) {
                continue;
            }
            let from = i + name.len();
            let span_end =
                t.iter().enumerate().skip(from).take(40).find(|(_, c)| ends_address(**c)).map_or_else(|| t.len().min(from + 40), |(k, _)| k);
            let span = t.get(from..span_end).unwrap_or_default();
            // A municipality within a dozen characters, then a number: 丁目, 番地 or digits.
            let Some(marker) = span.iter().take(13).position(|c| matches!(c, '市' | '区' | '町' | '村' | '郡')).filter(|m| *m >= 1) else {
                continue;
            };
            let after = span.get(marker + 1..).unwrap_or_default();
            let town_end = after.iter().position(char::is_ascii_digit).unwrap_or(after.len());
            let town = after.get(..town_end).unwrap_or_default();
            // A sentence, not an address: a particle between the municipality and the number.
            if town_end > 8 || town.iter().any(|c| matches!(c, 'ハ' | 'ヲ' | 'ニ' | 'デ' | 'ト' | 'モ' | 'ヘ')) {
                continue;
            }
            let Some(last) = after.iter().rposition(|c| c.is_ascii_digit() || matches!(c, '号' | '番' | '地' | '目')) else { continue };
            if after.iter().take(last + 1).all(|c| !c.is_ascii_digit() && !matches!(c, '目' | '番' | '号')) {
                continue;
            }
            let end = from + marker + 1 + last + 1;
            out.push(i..end);
            i = end;
            continue 'scan;
        }
        i += 1;
    }
    out
}

/// Hits of the chosen categories in `chars` (page text, words and lines separated by a space),
/// as character ranges of `chars`, in order, overlapping hits resolved in favour of the one that
/// starts first, then the category that comes first in [`PII_CATEGORIES`].
pub fn find_pii_chars(categories: &[PiiCategory], chars: &[char]) -> Vec<(PiiCategory, Range<usize>)> {
    let folded = fold_chars(chars);
    let t: Vec<char> = folded.chars.iter().map(|&c| dash(c)).collect();
    let mut hits: Vec<(PiiCategory, Range<usize>)> = Vec::new();
    for category in PII_CATEGORIES.into_iter().filter(|c| categories.contains(c)) {
        let found = match category {
            PiiCategory::MyNumber => my_numbers(&t),
            PiiCategory::Phone => phones(&t),
            PiiCategory::PostalCode => postal_codes(&t),
            PiiCategory::Email => emails(&t),
            PiiCategory::Address => addresses(&t),
        };
        for r in found {
            let start = folded.origin.get(r.start).map(|o| o.start);
            let end = r.end.checked_sub(1).and_then(|k| folded.origin.get(k)).map(|o| o.end);
            if let Some((s, e)) = start.zip(end).filter(|(s, e)| s < e) {
                hits.push((category, s..e));
            }
        }
    }
    hits.sort_by(|a, b| a.1.start.cmp(&b.1.start).then(a.0.cmp(&b.0)));
    let mut kept: Vec<(PiiCategory, Range<usize>)> = Vec::new();
    for hit in hits {
        if kept.last().is_none_or(|k| k.1.end <= hit.1.start) {
            kept.push(hit);
        }
    }
    kept
}

/// Hits of the chosen categories on one page, as glyph ranges in page order. A glyph that carries
/// a whole run of text (an `/ActualText` span, such as a column of vertical added text) can hold
/// several hits: each is kept, on that same glyph, so every one is counted.
pub fn find_pii(text: &PageText, categories: &[PiiCategory]) -> Vec<(PiiCategory, Range<usize>)> {
    let mut hits: Vec<(PiiCategory, Range<usize>)> = Vec::new();
    for &category in categories {
        for r in text.find_with(|chars| find_pii_chars(&[category], chars).into_iter().map(|h| h.1).collect()) {
            hits.push((category, r));
        }
    }
    hits.sort_by(|a, b| a.1.start.cmp(&b.1.start).then(a.0.cmp(&b.0)));
    let mut kept: Vec<(PiiCategory, Range<usize>)> = Vec::new();
    for hit in hits {
        let one_glyph = |k: &(PiiCategory, Range<usize>)| k.1 == hit.1 && hit.1.len() == 1;
        if kept.last().is_none_or(|k| k.1.end <= hit.1.start || one_glyph(k)) {
            kept.push(hit);
        }
    }
    kept
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hits(category: PiiCategory, s: &str) -> Vec<String> {
        let chars: Vec<char> = s.chars().collect();
        find_pii_chars(&[category], &chars).into_iter().map(|(_, r)| chars[r].iter().collect()).collect()
    }

    #[test]
    fn my_number_check_digit() {
        assert!(is_valid_my_number(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 8]));
        assert!(!is_valid_my_number(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2]));
        // A remainder of 0 or 1 gives check digit 0.
        assert!(is_valid_my_number(&[0; 12]));
    }

    #[test]
    fn my_number_forms() {
        let m = PiiCategory::MyNumber;
        assert_eq!(hits(m, "番号は123456789018です"), ["123456789018"]);
        assert_eq!(hits(m, "1234 5678 9018"), ["1234 5678 9018"]);
        assert_eq!(hits(m, "1234-5678-9018"), ["1234-5678-9018"]);
        assert_eq!(hits(m, "１２３４－５６７８－９０１８"), ["１２３４－５６７８－９０１８"]);
        assert_eq!(hits(m, "マイナンバー：１２３４５６７８９０１８"), ["１２３４５６７８９０１８"]);
        // The wrong check digit, a thirteenth digit and a sixteen digit card number are not hits.
        assert!(hits(m, "123456789012").is_empty());
        assert!(hits(m, "1234567890180").is_empty());
        assert!(hits(m, "1234 5678 9018 3456").is_empty());
        assert!(hits(m, "12345678901").is_empty());
    }

    #[test]
    fn phone_numbers() {
        let p = PiiCategory::Phone;
        for ok in [
            "03-1234-5678",
            "03(1234)5678",
            "(03)1234-5678",
            "(03) 1234-5678",
            "0312345678",
            "045-123-4567",
            "0422-12-3456",
            "090-1234-5678",
            "09012345678",
            "080 1234 5678",
            "050-1234-5678",
            "0120-123-456",
            "0120123456",
            "0800-123-4567",
            "０９０－１２３４－５６７８",
            "03‐1234‐5678",
        ] {
            assert_eq!(hits(p, &format!("電話：{ok}。")), [ok], "{ok}");
        }
        assert_eq!(hits(p, "TEL 03-1234-5678 / 090-1111-2222"), ["03-1234-5678", "090-1111-2222"]);
    }

    #[test]
    fn dates_and_other_numbers_are_not_phone_numbers() {
        let p = PiiCategory::Phone;
        for no in [
            "2024-11-11",
            "2024-11-11 12:34:56",
            "令和6年11月11日",
            "1234-5678",
            "0001-0002-0003",
            "123-456-7890",
            "03-1234-56789",
            "00-1234-5678",
            "090-1234-567",
            "12345678901234",
            "ISBN 978-4-00-000000-0",
        ] {
            assert!(hits(p, no).is_empty(), "{no}: {:?}", hits(p, no));
        }
        // The tail of a longer number is no phone number.
        assert!(hits(p, "20240312345678").is_empty());
        assert!(hits(p, "5-03-1234-5678").is_empty());
    }

    #[test]
    fn postal_codes() {
        let z = PiiCategory::PostalCode;
        assert_eq!(hits(z, "〒100-0001 東京都"), ["〒100-0001"]);
        assert_eq!(hits(z, "〒1000001"), ["〒1000001"]);
        assert_eq!(hits(z, "〒 100-0001"), ["〒 100-0001"]);
        assert_eq!(hits(z, "郵便番号 060-0001"), ["060-0001"]);
        assert_eq!(hits(z, "〒１００－０００１"), ["〒１００－０００１"]);
        assert!(hits(z, "2024-11-11").is_empty());
        assert!(hits(z, "03-1234-5678").is_empty());
        assert!(hits(z, "〒100-00012").is_empty());
        assert!(hits(z, "1000-0001").is_empty());
        assert!(hits(z, "100-00012").is_empty());
    }

    #[test]
    fn email_addresses() {
        let e = PiiCategory::Email;
        assert_eq!(hits(e, "連絡先：taro.yamada@example.co.jp。"), ["taro.yamada@example.co.jp"]);
        assert_eq!(hits(e, "a_b+c@mail.example.com."), ["a_b+c@mail.example.com"]);
        assert_eq!(hits(e, "ｔａｒｏ＠ｅｘａｍｐｌｅ．ｃｏｍ"), ["ｔａｒｏ＠ｅｘａｍｐｌｅ．ｃｏｍ"]);
        assert!(hits(e, "@example.com").is_empty());
        assert!(hits(e, "taro@localhost").is_empty());
        assert!(hits(e, "taro@example.c").is_empty());
        assert!(hits(e, "taro @ example.com").is_empty());
    }

    #[test]
    fn street_addresses() {
        let a = PiiCategory::Address;
        assert_eq!(hits(a, "住所：東京都千代田区丸の内1-1-1に住んでいます"), ["東京都千代田区丸の内1-1-1"]);
        assert_eq!(hits(a, "神奈川県横浜市中区本町二丁目10番地"), ["神奈川県横浜市中区本町二丁目10番地"]);
        assert_eq!(hits(a, "大阪府大阪市北区梅田３丁目１番１号 大阪ビル"), ["大阪府大阪市北区梅田３丁目１番１号"]);
        // No number, no municipality, or a sentence: not hits.
        assert!(hits(a, "東京都千代田区").is_empty());
        assert!(hits(a, "東京都内で3人").is_empty());
        assert!(hits(a, "東京都は2024年に条例を改正した").is_empty());
        assert!(hits(a, "京都府の面積は4612平方キロ").is_empty());
    }

    #[test]
    fn overlaps_prefer_the_earlier_hit_and_ranges_map_back_to_the_original() {
        let all = PII_CATEGORIES;
        let chars: Vec<char> = "〒１００－０００１ 03-1234-5678".chars().collect();
        let found = find_pii_chars(&all, &chars);
        assert_eq!(found.len(), 2);
        assert_eq!(found[0], (PiiCategory::PostalCode, 0..9));
        assert_eq!(found[1].0, PiiCategory::Phone);
    }
}
