use super::*;

/// The dates in `text`, each as written and converted to `to` (or the problem's id, or "=" when
/// already in that calendar).
fn conv(text: &str, to: Calendar) -> Vec<(String, String)> {
    find_dates(text)
        .iter()
        .map(|f| {
            let shown = match convert(f, to) {
                Conversion::To(s) => s,
                Conversion::Already => "=".into(),
                Conversion::Invalid(p) => p.id().into(),
            };
            (text[f.range.clone()].to_string(), shown)
        })
        .collect()
}

fn one(text: &str, to: Calendar) -> String {
    let all = conv(text, to);
    assert_eq!(all.len(), 1, "{text:?}: {all:?}");
    all[0].1.clone()
}

fn pairs(list: &[(&str, &str)]) -> Vec<(String, String)> {
    list.iter().map(|(a, b)| (a.to_string(), b.to_string())).collect()
}

use Calendar::{Japanese as J, Western as W};

#[test]
fn era_dates_convert_to_western() {
    assert_eq!(one("令和6年5月1日", W), "2024年5月1日");
    assert_eq!(one("令和元年", W), "2019年");
    assert_eq!(one("令和元年5月", W), "2019年5月");
    assert_eq!(one("R6.5.1", W), "2024年5月1日");
    assert_eq!(one("H31.4.30", W), "2019年4月30日");
    assert_eq!(one("平成31年4月30日", W), "2019年4月30日");
    assert_eq!(one("昭和64年1月7日", W), "1989年1月7日");
    assert_eq!(one("平成元年1月8日", W), "1989年1月8日");
    assert_eq!(one("大正15年12月24日", W), "1926年12月24日");
    assert_eq!(one("明治45年7月29日", W), "1912年7月29日");
    assert_eq!(one("令和 6 年 5 月 1 日", W), "2024年5月1日");
    assert_eq!(one("R6年5月1日", W), "2024年5月1日");
    assert_eq!(one("S64/1/7", W), "1989年1月7日");
    assert_eq!(one("令和6-5-1", W), "2024年5月1日");
    assert_eq!(one("令和06年05月01日", W), "2024年5月1日");
}

#[test]
fn western_dates_convert_to_era() {
    assert_eq!(one("2024年5月1日", J), "令和6年5月1日");
    assert_eq!(one("2024/5/1", J), "令和6年5月1日");
    assert_eq!(one("2024-05-01", J), "令和6年5月1日");
    assert_eq!(one("2024.5.1", J), "令和6年5月1日");
    assert_eq!(one("2019年5月1日", J), "令和元年5月1日");
    assert_eq!(one("2019年4月30日", J), "平成31年4月30日");
    assert_eq!(one("1989年1月7日", J), "昭和64年1月7日");
    assert_eq!(one("1989年1月8日", J), "平成元年1月8日");
    assert_eq!(one("2019年5月", J), "令和元年5月");
    assert_eq!(one("2019年4月", J), "平成31年4月");
    assert_eq!(one("2024 年 5 月 1 日", J), "令和6年5月1日");
}

#[test]
fn era_boundaries_are_exact() {
    // The day after an era's last day belongs to the next era.
    assert_eq!(one("平成31年5月1日", W), "outside-era");
    assert_eq!(one("H31.5.1", W), "outside-era");
    assert_eq!(one("令和元年4月30日", W), "outside-era");
    assert_eq!(one("令和元年4月", W), "outside-era");
    assert_eq!(one("昭和64年1月8日", W), "outside-era");
    assert_eq!(one("平成元年1月7日", W), "outside-era");
    assert_eq!(one("平成32年", W), "outside-era");
    assert_eq!(one("令和0年1月1日", W), "outside-era");
    assert_eq!(one("明治元年1月24日", W), "outside-era");
    // A year that ends in its era is still valid as a year.
    assert_eq!(one("平成31年", W), "2019年");
    assert_eq!(one("昭和64年", W), "1989年");
    assert_eq!(one("昭和64年1月", W), "1989年1月");
}

#[test]
fn invalid_days_and_ambiguous_periods_are_flagged_not_converted() {
    assert_eq!(one("2024年2月30日", J), "no-such-day");
    assert_eq!(one("2023/2/29", J), "no-such-day");
    assert_eq!(one("2024/2/29", J), "令和6年2月29日");
    assert_eq!(one("2024年13月1日", J), "no-such-day");
    assert_eq!(one("令和6年2月30日", W), "no-such-day");
    // Before 明治 there is no era form; a Western date there is still a date.
    assert_eq!(one("1850年1月1日", J), "before-meiji");
    assert_eq!(one("1868年1月24日", J), "before-meiji");
    assert_eq!(one("1868年1月25日", J), "明治元年1月25日");
    assert_eq!(one("1850年1月1日", W), "=");
    // A month that two eras share.
    assert_eq!(one("1989年1月", J), "ambiguous");
    assert_eq!(one("2019年4月", W), "=");
}

#[test]
fn a_date_already_in_the_target_calendar_is_left_alone() {
    assert_eq!(one("令和6年5月1日", J), "=");
    assert_eq!(one("R6.5.1", J), "=");
    assert_eq!(one("2024/5/1", W), "=");
    // Invalid stays invalid in either direction.
    assert_eq!(one("平成31年5月1日", J), "outside-era");
}

#[test]
fn full_width_digits_are_read_and_kept() {
    assert_eq!(one("令和６年５月１日", W), "２０２４年５月１日");
    assert_eq!(one("２０２４年５月１日", J), "令和６年５月１日");
    assert_eq!(one("Ｒ６．５．１", W), "２０２４年５月１日");
    assert_eq!(one("２０２４／５／１", J), "令和６年５月１日");
    assert_eq!(one("令和元年５月１日", W), "２０１９年５月１日");
    assert_eq!(one("令和　６年", W), "２０２４年");
}

#[test]
fn phone_numbers_and_digit_runs_are_not_dates() {
    for text in [
        "03-1234-5678",
        "0120-12-3456",
        "090-2024-0501",
        "TEL 045-123-4567",
        "12024/5/1",
        "2024/5/1/3",
        "2024/5/12345",
        "2024/05-01",
        "2024-5-1-2",
        "1-2024-5-1",
        "部品番号 2024512",
        "20240501",
        "2024年",
        "2024年度",
        "VER6.5.1",
        "AR6.5.1",
        "R6.5",
        "R65.1.1.1",
        "令和六年五月一日",
        "123,456,789",
        "3000/1/1",
        "0999/1/1",
        "合計 1,234 円",
    ] {
        assert_eq!(conv(text, J), vec![], "{text:?}");
    }
}

#[test]
fn dates_in_running_text() {
    assert_eq!(
        conv("申請日:令和6年5月1日、提出期限 2024/6/30 まで。電話 03-1234-5678", W),
        pairs(&[("令和6年5月1日", "2024年5月1日"), ("2024/6/30", "=")])
    );
    assert_eq!(
        conv("申請日:令和6年5月1日、提出期限 2024/6/30 まで。電話 03-1234-5678", J),
        pairs(&[("令和6年5月1日", "="), ("2024/6/30", "令和6年6月30日")])
    );
    // 年度: the year converts and 度 stays.
    assert_eq!(conv("令和6年度予算", W), pairs(&[("令和6年", "2024年")]));
    // A month without its 日 ends the date after 月.
    assert_eq!(conv("2024年5月10日頃と2024年5月", J), pairs(&[("2024年5月10日", "令和6年5月10日"), ("2024年5月", "令和6年5月")]));
    assert_eq!(conv("2024年5月1", J), pairs(&[("2024年5月", "令和6年5月")]));
    // Byte ranges point at the date as written.
    let text = "（日付）R６．５．１です";
    let found = find_dates(text);
    assert_eq!(found.len(), 1);
    assert_eq!(&text[found[0].range.clone()], "R６．５．１");
    assert!(found[0].full_width);
}

#[test]
fn hostile_text_does_not_crash() {
    for text in ["令和", "令和元", "R", "R9", "2024/", "2024/1/", "令和99年99月99日", "平成4294967296年", "9999/99/99", "\u{FF10}", ""] {
        let _ = conv(text, J);
        let _ = conv(text, W);
    }
    let long = "令和6年5月1日".repeat(20_000);
    assert_eq!(find_dates(&long).len(), MAX_DATES_PER_TEXT);
}

#[test]
fn replacing_ranges_keeps_the_rest_of_the_text() {
    let s = "A令和6年5月1日とR6.5.1B";
    let found = find_dates(s);
    let edits: Vec<(Range<usize>, &str)> = found.iter().map(|f| (f.range.clone(), "X")).collect();
    assert_eq!(replaced(s, edits).as_deref(), Some("AXとXB"));
    assert_eq!(replaced("abc", vec![(1..9, "x")]), None);
    assert_eq!(replaced("abc", vec![(0..2, "x"), (1..3, "y")]), None);
}
