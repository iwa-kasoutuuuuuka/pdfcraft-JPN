//! Japanese era (和暦) dates: 令和7年11月10日, R7.11.10. Pure functions on a Gregorian
//! `(year, month, day)`; the date formats of Fill & Sign (`gg` `G` `e` `ee` letters in
//! `pdfcraft_engine::dates`) and the header/footer date tokens both use them.
//!
//! An era starts on its first day, not on 1 January: 2019-04-30 is 平成31年, 2019-05-01 is 令和元年.
//! Dates before Meiji 6's calendar switch are not covered; the era starts used here are the
//! Gregorian dates of each era's first day, from 明治 (1868-01-25) on.

/// One era: its name, the Latin initial used in short forms (R7.11.10) and its first day.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Era {
    pub name: &'static str,
    pub initial: char,
    pub start: (i64, u32, u32),
}

/// The eras, oldest first.
pub const ERAS: [Era; 5] = [
    Era { name: "明治", initial: 'M', start: (1868, 1, 25) },
    Era { name: "大正", initial: 'T', start: (1912, 7, 30) },
    Era { name: "昭和", initial: 'S', start: (1926, 12, 25) },
    Era { name: "平成", initial: 'H', start: (1989, 1, 8) },
    Era { name: "令和", initial: 'R', start: (2019, 5, 1) },
];

/// The era of the Gregorian day and its year number there (1 = 元年), or `None` before 明治 or
/// for a month/day out of range.
pub fn japanese_era(y: i64, m: u32, d: u32) -> Option<(&'static Era, i64)> {
    if !(1..=12).contains(&m) || !(1..=31).contains(&d) {
        return None;
    }
    let era = ERAS.iter().rev().find(|e| (y, m, d) >= e.start)?;
    Some((era, y - era.start.0 + 1))
}

/// The year in an era as written: `元` for the first year, else the number.
pub fn era_year_text(year: i64) -> String {
    if year == 1 { "元".to_string() } else { year.to_string() }
}

/// `令和7年11月10日` (`令和元年5月1日` for year 1); `None` before 明治.
pub fn format_japanese(y: i64, m: u32, d: u32) -> Option<String> {
    let (era, year) = japanese_era(y, m, d)?;
    Some(format!("{}{}年{m}月{d}日", era.name, era_year_text(year)))
}

/// `R7.11.10`; `None` before 明治.
pub fn format_japanese_short(y: i64, m: u32, d: u32) -> Option<String> {
    let (era, year) = japanese_era(y, m, d)?;
    Some(format!("{}{year}.{m}.{d}", era.initial))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn era_boundaries() {
        assert_eq!(format_japanese(2019, 4, 30).as_deref(), Some("平成31年4月30日"));
        assert_eq!(format_japanese(2019, 5, 1).as_deref(), Some("令和元年5月1日"));
        assert_eq!(format_japanese(1989, 1, 7).as_deref(), Some("昭和64年1月7日"));
        assert_eq!(format_japanese(1989, 1, 8).as_deref(), Some("平成元年1月8日"));
        assert_eq!(format_japanese(1926, 12, 24).as_deref(), Some("大正15年12月24日"));
        assert_eq!(format_japanese(1926, 12, 25).as_deref(), Some("昭和元年12月25日"));
        assert_eq!(format_japanese(1912, 7, 29).as_deref(), Some("明治45年7月29日"));
        assert_eq!(format_japanese(1912, 7, 30).as_deref(), Some("大正元年7月30日"));
        assert_eq!(format_japanese(1868, 1, 25).as_deref(), Some("明治元年1月25日"));
        assert_eq!(format_japanese(1868, 1, 24), None);
        assert_eq!(format_japanese(2025, 11, 10).as_deref(), Some("令和7年11月10日"));
        assert_eq!(format_japanese(2026, 11, 10).as_deref(), Some("令和8年11月10日"));
    }

    #[test]
    fn short_form_and_hostile_input() {
        assert_eq!(format_japanese_short(2025, 11, 10).as_deref(), Some("R7.11.10"));
        assert_eq!(format_japanese_short(2019, 5, 1).as_deref(), Some("R1.5.1"));
        assert_eq!(format_japanese_short(1989, 1, 7).as_deref(), Some("S64.1.7"));
        assert_eq!(format_japanese_short(1900, 1, 1).as_deref(), Some("M33.1.1"));
        assert_eq!(japanese_era(2025, 13, 1), None);
        assert_eq!(japanese_era(2025, 1, 0), None);
        assert_eq!(japanese_era(i64::MIN, u32::MAX, u32::MAX), None);
        assert!(japanese_era(i64::MAX, 1, 1).is_some(), "no overflow");
    }
}
