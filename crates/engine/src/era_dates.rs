//! Converting dates between the Japanese era calendar (和暦) and the Western one (西暦), all at
//! once (和暦/西暦の一括変換).
//!
//! [`find_dates`] reads plain text and finds the dates written in it:
//!
//! - era dates in kanji: `令和6年5月1日`, `令和元年`, `平成31年4月`, with spaces between the parts
//!   (`令和 6 年 5 月 1 日`) and the Latin initial in place of the name (`R6年5月1日`);
//! - era dates with separators: `R6.5.1`, `H31/4/30`, `令和6-5-1` (a full date only);
//! - Western dates: `2024年5月1日`, `2024年5月`, and `2024/5/1`, `2024-05-01`, `2024.5.1` (a full
//!   date only, the same separator twice). A Western year alone (`2024年`) is not taken: it is too
//!   often a count or part of a fiscal year (`2024年度`).
//!
//! Full-width digits, letters and separators (`令和６年５月１日`, `Ｒ６．５．１`) read as the ASCII
//! they stand for. A date is not taken from inside a longer run of digits, so a phone number
//! (`03-1234-5678`, `0120-12-3456`) or a part number is never a date: a Western year has four
//! digits starting with 1 or 2, and nothing may be glued before or after the date (a digit, or a
//! separator and a digit). An era initial right after a Latin letter or digit (`VER6.5.1`) is not
//! one either. Kanji numerals (`令和六年`) are not read.
//!
//! An era date is checked against the era's exact days: `平成31年4月30日` is valid, `平成31年5月1日`
//! is not (that day is `令和元年5月1日`), nor is `令和元年4月`, `昭和64年1月8日` or `2月30日`. Such a
//! date is found and reported with its [`DateProblem`], never converted. A Western year or month
//! that spans two eras (`2019年`, `1989年1月`) is [`DateProblem::Ambiguous`] in era form.
//!
//! [`convert`] writes a date in the other calendar, always in kanji style (`令和6年5月1日`,
//! `2024年5月1日`), whatever the source's separators were: `R6.5.1` becomes `2024年5月1日`. Year 1
//! of an era is written `元年`. A date already in the target calendar is left as it is
//! ([`Conversion::Already`]), separators included. Full-width digits in the source give full-width
//! digits in the result.
//!
//! [`Document::find_dates`] looks in the document's page text (by line, as Edit text lists it),
//! the text of its comments and the values of its text fields; [`Document::date_conversion`] plans
//! one undoable edit that replaces the chosen hits. Page text goes through the same in-place line
//! rewrite as Edit text ([`Edit::EditTextLine`]), tried first on a copy of the document: a line
//! the rewrite refuses, or that would not read back as the converted text, is reported and left
//! unchanged, never half-rewritten.

use std::ops::Range;

use chrono::NaiveDate;
use pdfcraft_edit::era::{ERAS, Era, era_year_text, japanese_era};
use pdfcraft_forms::{FieldKind, FieldValue};

use crate::{Document, Edit};

/// A calendar a date is written in, or converted to.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum Calendar {
    /// The Japanese era calendar (和暦): 令和6年5月1日.
    #[default]
    Japanese,
    /// The Western (Gregorian) calendar (西暦): 2024年5月1日.
    Western,
}

impl Calendar {
    /// The stable name used by the automation tools.
    pub fn id(self) -> &'static str {
        match self {
            Calendar::Japanese => "japanese",
            Calendar::Western => "western",
        }
    }

    pub fn from_id(id: &str) -> Option<Self> {
        [Calendar::Japanese, Calendar::Western].into_iter().find(|c| c.id() == id)
    }

    /// The English label (translated by the interface).
    pub fn label(self) -> &'static str {
        match self {
            Calendar::Japanese => "Japanese era (令和6年5月1日)",
            Calendar::Western => "Western (2024年5月1日)",
        }
    }
}

/// Why a date that was found cannot be converted.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DateProblem {
    /// No such day or month (`2月30日`, `13月`).
    NoSuchDay,
    /// The era had not started or had ended (`平成31年5月1日`, `令和元年4月`, `昭和64年1月8日`).
    OutsideEra,
    /// Before 明治, which has no era form here.
    BeforeMeiji,
    /// A year or month that spans two eras (`2019年`, `1989年1月`): which era is meant is unknown.
    Ambiguous,
}

impl DateProblem {
    /// The stable name used by the automation tools.
    pub fn id(self) -> &'static str {
        match self {
            DateProblem::NoSuchDay => "no-such-day",
            DateProblem::OutsideEra => "outside-era",
            DateProblem::BeforeMeiji => "before-meiji",
            DateProblem::Ambiguous => "ambiguous",
        }
    }

    /// The English explanation (translated by the interface).
    pub fn label(self) -> &'static str {
        match self {
            DateProblem::NoSuchDay => "Invalid date: there is no such day",
            DateProblem::OutsideEra => "Invalid date: the era does not include this day",
            DateProblem::BeforeMeiji => "Before Meiji: no era form",
            DateProblem::Ambiguous => "Spans two eras: convert it by hand",
        }
    }
}

/// A Gregorian year, with its month and day when the text gives them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DateValue {
    pub year: i64,
    pub month: Option<u32>,
    pub day: Option<u32>,
}

/// A date found in text.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FoundDate {
    /// Where it is in the text searched, in bytes.
    pub range: Range<usize>,
    /// The calendar it is written in.
    pub calendar: Calendar,
    /// The date, or why it is not one.
    pub value: Result<DateValue, DateProblem>,
    /// Its digits are full-width (`６`), so a conversion writes full-width digits too.
    pub full_width: bool,
}

/// What converting a found date gives.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Conversion {
    /// The date in the target calendar.
    To(String),
    /// Already written in the target calendar: left as it is.
    Already,
    /// Not a valid date, or one the target calendar can't write: left as it is.
    Invalid(DateProblem),
}

impl Conversion {
    /// The converted text, when there is one.
    pub fn text(&self) -> Option<&str> {
        match self {
            Conversion::To(s) => Some(s),
            _ => None,
        }
    }
}

/// The most dates read from one text; a hostile text full of dates stops there.
const MAX_DATES_PER_TEXT: usize = 10_000;

/// A character as the matcher reads it: full-width digits, Latin capitals, separators and the
/// ideographic space as their ASCII forms, and dash variants as a hyphen. One character in, one
/// out, so positions stay put.
fn read_as(c: char) -> char {
    match c {
        '\u{FF10}'..='\u{FF19}' | '\u{FF21}'..='\u{FF3A}' => char::from_u32(u32::from(c) - 0xFEE0).unwrap_or(c),
        '\u{FF0F}' => '/',
        '\u{FF0E}' => '.',
        '\u{FF0D}' | '\u{2010}'..='\u{2012}' | '\u{2212}' => '-',
        '\u{3000}' => ' ',
        _ => c,
    }
}

fn is_separator(c: char) -> bool {
    matches!(c, '/' | '.' | '-')
}

/// The number written at `at` with one to `max` digits, and where it ends; `None` when there is
/// no digit or more than `max` of them.
fn number(t: &[char], at: usize, max: usize) -> Option<(u32, usize)> {
    let len = t.iter().skip(at).take_while(|c| c.is_ascii_digit()).count();
    if len == 0 || len > max {
        return None;
    }
    let mut value = 0u32;
    for c in t.get(at..at + len)? {
        value = value.checked_mul(10)?.checked_add(c.to_digit(10)?)?;
    }
    Some((value, at + len))
}

fn spaces(t: &[char], mut at: usize) -> usize {
    while t.get(at) == Some(&' ') {
        at += 1;
    }
    at
}

/// Whether what ends at `end` runs on into more digits: a digit, or a separator and a digit.
fn runs_on(t: &[char], end: usize) -> bool {
    match t.get(end) {
        Some(c) if c.is_ascii_digit() => true,
        Some(&c) if is_separator(c) => t.get(end + 1).is_some_and(char::is_ascii_digit),
        _ => false,
    }
}

/// Whether what starts at `start` is the tail of something longer: a digit or Latin letter, or a
/// separator after a digit.
fn glued_before(t: &[char], start: usize) -> bool {
    let Some(&prev) = start.checked_sub(1).and_then(|p| t.get(p)) else { return false };
    prev.is_ascii_alphanumeric() || (is_separator(prev) && start.checked_sub(2).and_then(|p| t.get(p)).is_some_and(char::is_ascii_digit))
}

/// The month and day after a year in kanji style, starting right after `年`: `5月1日`, `5月`,
/// or nothing. Gives the month, the day and where the date ends.
fn kanji_tail(t: &[char], after_year: usize) -> (Option<u32>, Option<u32>, usize) {
    let m_at = spaces(t, after_year);
    let Some((month, m_end)) = number(t, m_at, 2) else { return (None, None, after_year) };
    let m_mark = spaces(t, m_end);
    if t.get(m_mark) != Some(&'月') {
        return (None, None, after_year);
    }
    let d_at = spaces(t, m_mark + 1);
    let Some((day, d_end)) = number(t, d_at, 2) else { return (Some(month), None, m_mark + 1) };
    let d_mark = spaces(t, d_end);
    if t.get(d_mark) != Some(&'日') {
        return (Some(month), None, m_mark + 1);
    }
    (Some(month), Some(day), d_mark + 1)
}

/// `.5.1` after a year written with separators: the same separator twice. Gives the month, the
/// day and where the date ends, or `None` when it runs on into more digits.
fn separated_tail(t: &[char], after_year: usize) -> Option<(u32, u32, usize)> {
    let sep = *t.get(after_year).filter(|c| is_separator(**c))?;
    let (month, m_end) = number(t, after_year + 1, 2)?;
    if t.get(m_end) != Some(&sep) {
        return None;
    }
    let (day, end) = number(t, m_end + 1, 2)?;
    (!runs_on(t, end)).then_some((month, day, end))
}

/// The era whose name starts at `at`, or whose Latin initial does (then followed by a digit and
/// not glued to a word before it). Gives the era, where its year starts and whether it was
/// written by name.
fn era_at(t: &[char], at: usize) -> Option<(&'static Era, usize, bool)> {
    for era in &ERAS {
        let name: Vec<char> = era.name.chars().collect();
        if t.get(at..at + name.len()) == Some(name.as_slice()) {
            return Some((era, spaces(t, at + name.len()), true));
        }
    }
    let c = *t.get(at)?;
    let era = ERAS.iter().find(|e| e.initial == c)?;
    (t.get(at + 1).is_some_and(char::is_ascii_digit) && !glued_before(t, at)).then_some((era, at + 1, false))
}

/// The day an era starts, and the last day of it (`None` for the current era).
fn era_days(era: &Era) -> Option<(NaiveDate, Option<NaiveDate>)> {
    let date = |(y, m, d): (i64, u32, u32)| NaiveDate::from_ymd_opt(i32::try_from(y).ok()?, m, d);
    let start = date(era.start)?;
    let next = ERAS.iter().find(|e| e.start > era.start);
    let end = match next {
        Some(n) => Some(date(n.start)?.pred_opt()?),
        None => None,
    };
    Some((start, end))
}

/// The first and last day a date stands for: the day itself, or its whole month or year.
fn period(v: DateValue) -> Option<(NaiveDate, NaiveDate)> {
    let y = i32::try_from(v.year).ok()?;
    match (v.month, v.day) {
        (Some(m), Some(d)) => {
            let day = NaiveDate::from_ymd_opt(y, m, d)?;
            Some((day, day))
        }
        (Some(m), None) => {
            let first = NaiveDate::from_ymd_opt(y, m, 1)?;
            let next = if m == 12 { NaiveDate::from_ymd_opt(y.checked_add(1)?, 1, 1)? } else { NaiveDate::from_ymd_opt(y, m + 1, 1)? };
            Some((first, next.pred_opt()?))
        }
        _ => Some((NaiveDate::from_ymd_opt(y, 1, 1)?, NaiveDate::from_ymd_opt(y, 12, 31)?)),
    }
}

/// The Gregorian date of year `year` of `era`, checked against the era's days.
fn era_value(era: &Era, year: u32, month: Option<u32>, day: Option<u32>) -> Result<DateValue, DateProblem> {
    if year == 0 {
        return Err(DateProblem::OutsideEra);
    }
    let value = DateValue { year: era.start.0 + i64::from(year) - 1, month, day };
    let (first, last) = period(value).ok_or(DateProblem::NoSuchDay)?;
    let (start, end) = era_days(era).ok_or(DateProblem::OutsideEra)?;
    if last < start || end.is_some_and(|end| first > end) {
        return Err(DateProblem::OutsideEra);
    }
    Ok(value)
}

fn western_value(year: u32, month: Option<u32>, day: Option<u32>) -> Result<DateValue, DateProblem> {
    let value = DateValue { year: i64::from(year), month, day };
    period(value).map(|_| value).ok_or(DateProblem::NoSuchDay)
}

/// The era date starting at `at`: a range of characters and the value.
fn era_date(t: &[char], at: usize) -> Option<(Range<usize>, Result<DateValue, DateProblem>)> {
    let (era, year_at, by_name) = era_at(t, at)?;
    // 元 is the first year, in kanji style only.
    let (year, year_end, gan) =
        if t.get(year_at) == Some(&'元') { (1, year_at + 1, true) } else { number(t, year_at, 2).map(|(y, e)| (y, e, false))? };
    let mark = spaces(t, year_end);
    if t.get(mark) == Some(&'年') {
        let (month, day, end) = kanji_tail(t, mark + 1);
        return Some((at..end, era_value(era, year, month, day)));
    }
    // R6.5.1 and 令和6.5.1: no spaces, no 元, a full date.
    if gan || mark != year_end || (!by_name && glued_before(t, at)) {
        return None;
    }
    let (month, day, end) = separated_tail(t, year_end)?;
    Some((at..end, era_value(era, year, Some(month), Some(day))))
}

/// The Western date starting at `at`.
fn western_date(t: &[char], at: usize) -> Option<(Range<usize>, Result<DateValue, DateProblem>)> {
    if !matches!(t.get(at), Some('1' | '2')) || glued_before(t, at) {
        return None;
    }
    let (year, year_end) = number(t, at, 4)?;
    if year_end - at != 4 {
        return None;
    }
    let mark = spaces(t, year_end);
    if t.get(mark) == Some(&'年') {
        // A year on its own is not taken; a month is needed.
        let (month, day, end) = kanji_tail(t, mark + 1);
        month?;
        return Some((at..end, western_value(year, month, day)));
    }
    let (month, day, end) = separated_tail(t, year_end)?;
    Some((at..end, western_value(year, Some(month), Some(day))))
}

/// The dates written in `text`, in order.
pub fn find_dates(text: &str) -> Vec<FoundDate> {
    let original: Vec<(usize, char)> = text.char_indices().collect();
    let t: Vec<char> = original.iter().map(|&(_, c)| read_as(c)).collect();
    let byte_at = |i: usize| original.get(i).map_or(text.len(), |&(b, _)| b);
    let mut out = Vec::new();
    let mut i = 0;
    while i < t.len() && out.len() < MAX_DATES_PER_TEXT {
        let found = era_date(&t, i).map(|(r, v)| (r, v, Calendar::Japanese)).or_else(|| western_date(&t, i).map(|(r, v)| (r, v, Calendar::Western)));
        match found {
            Some((range, value, calendar)) if range.end > range.start => {
                let full_width = original.get(range.clone()).is_some_and(|cs| cs.iter().any(|&(_, c)| ('\u{FF10}'..='\u{FF19}').contains(&c)));
                out.push(FoundDate { range: byte_at(range.start)..byte_at(range.end), calendar, value, full_width });
                i = range.end;
            }
            _ => i += 1,
        }
    }
    out
}

fn widen(s: String) -> String {
    s.chars().map(|c| if c.is_ascii_digit() { char::from_u32(u32::from(c) + 0xFEE0).unwrap_or(c) } else { c }).collect()
}

fn month_day(v: DateValue) -> String {
    let mut s = String::new();
    if let Some(m) = v.month {
        s.push_str(&format!("{m}月"));
        if let Some(d) = v.day {
            s.push_str(&format!("{d}日"));
        }
    }
    s
}

/// The date written in `to`'s calendar, in kanji style.
pub fn convert(found: &FoundDate, to: Calendar) -> Conversion {
    let v = match found.value {
        Ok(v) => v,
        Err(problem) => return Conversion::Invalid(problem),
    };
    if found.calendar == to {
        return Conversion::Already;
    }
    let text = match to {
        Calendar::Western => format!("{}年{}", v.year, month_day(v)),
        Calendar::Japanese => {
            let Some((first, last)) = period(v) else { return Conversion::Invalid(DateProblem::NoSuchDay) };
            let era_of = |d: NaiveDate| {
                use chrono::Datelike;
                japanese_era(i64::from(d.year()), d.month(), d.day())
            };
            let Some((era, year)) = era_of(first) else { return Conversion::Invalid(DateProblem::BeforeMeiji) };
            if era_of(last).map(|(e, _)| e.name) != Some(era.name) {
                return Conversion::Invalid(DateProblem::Ambiguous);
            }
            format!("{}{}年{}", era.name, era_year_text(year), month_day(v))
        }
    };
    Conversion::To(if found.full_width { widen(text) } else { text })
}

/// Where a date in a document is written.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum DateSource {
    /// A line of page text (0-based page, and line as [`Document::text_lines`] numbers it).
    PageText { page: usize, line: usize },
    /// A comment's text (0-based page, and its position in the page's `/Annots`).
    Comment { page: usize, index: usize },
    /// A text field's value.
    Field { name: String },
}

impl DateSource {
    /// The stable name of the kind of place, used by the automation tools.
    pub fn kind(&self) -> &'static str {
        match self {
            DateSource::PageText { .. } => "page-text",
            DateSource::Comment { .. } => "comment",
            DateSource::Field { .. } => "field",
        }
    }
}

/// A date found in a document.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DocumentDate {
    pub source: DateSource,
    /// The page it is on (0-based), when it is on one.
    pub page: Option<usize>,
    /// The whole text it was found in (the line, comment or field value).
    pub context: String,
    /// Where the date is in `context`, in bytes.
    pub range: Range<usize>,
    /// The date as written.
    pub original: String,
    /// What it converts to.
    pub conversion: Conversion,
}

/// The dates in a document.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DateSearch {
    pub hits: Vec<DocumentDate>,
    /// The document's security doesn't allow copying, so its page text was not read (comments
    /// and fields were).
    pub page_text_withheld: bool,
}

/// What [`Document::date_conversion`] will do.
#[derive(Debug, Default)]
pub struct DatePlan {
    /// The edit replacing the dates, `None` when there is nothing to replace.
    pub edit: Option<Edit>,
    /// The positions (in the hits given) that the edit replaces.
    pub replaced: Vec<usize>,
    /// The positions left unchanged, with the reason in English (translated by the interface).
    pub skipped: Vec<(usize, &'static str)>,
}

/// Why a chosen hit is left unchanged.
pub const SKIP_NOT_CONVERTIBLE: &str = "Nothing to convert";
pub const SKIP_TEXT_CHANGED: &str = "The text changed since the dates were found; find them again";
pub const SKIP_LINE_REFUSED: &str = "This line of page text can't be rewritten in place";
pub const SKIP_LINE_READS_DIFFERENTLY: &str = "The rewritten line would not read back as the converted text";
pub const SKIP_FIELD_TOO_LONG: &str = "The converted date is longer than the field allows";

/// Every reason a hit can be skipped, for an interface to translate.
pub const SKIP_REASONS: [&str; 5] = [SKIP_NOT_CONVERTIBLE, SKIP_TEXT_CHANGED, SKIP_LINE_REFUSED, SKIP_LINE_READS_DIFFERENTLY, SKIP_FIELD_TOO_LONG];

/// The undo step's name.
pub const CONVERT_DATES_LABEL: &str = "Convert dates";

fn found_in(source: DateSource, page: Option<usize>, context: &str, to: Calendar, out: &mut Vec<DocumentDate>) {
    for f in find_dates(context) {
        let Some(original) = context.get(f.range.clone()) else { continue };
        out.push(DocumentDate {
            source: source.clone(),
            page,
            context: context.to_owned(),
            original: original.to_owned(),
            conversion: convert(&f, to),
            range: f.range,
        });
    }
}

/// `context` with each `(range, replacement)` put in; `None` when a range doesn't fit.
fn replaced(context: &str, mut edits: Vec<(Range<usize>, &str)>) -> Option<String> {
    edits.sort_by_key(|(r, _)| std::cmp::Reverse(r.start));
    let mut s = context.to_owned();
    let mut limit = s.len();
    for (r, with) in edits {
        if r.end > limit || r.start > r.end || s.get(r.clone()).is_none() {
            return None;
        }
        s.replace_range(r.clone(), with);
        limit = r.start;
    }
    Some(s)
}

/// A text field whose value can be converted: a single line or multiline text value that no
/// date or number format script reads (changing a formatted date's calendar would break it).
fn convertible_field(f: &pdfcraft_forms::Field) -> bool {
    f.kind == FieldKind::Text && !f.read_only() && f.value.len() == 1 && matches!(f.actions.format, pdfcraft_forms::af::Format::None)
}

impl Document {
    /// The dates in the document's page text, comments and text fields, each with what it
    /// converts to in `to`'s calendar: page by page (page text, then comments, then the fields on
    /// that page), then fields on no page. `pages` (0-based) limits the pages read; fields on no
    /// page are read only when it is `None`.
    pub fn find_dates(&self, to: Calendar, pages: Option<&[usize]>) -> DateSearch {
        let n = self.info.pages.len();
        let pages: Vec<usize> = pages.map_or_else(|| (0..n).collect(), |p| p.iter().copied().filter(|&p| p < n).collect());
        let mut search = DateSearch { page_text_withheld: !self.allows_copying(), ..Default::default() };
        let field_page = |f: &pdfcraft_forms::Field| f.widgets.iter().find_map(|w| w.page);
        for &page in &pages {
            if !search.page_text_withheld {
                for (line, l) in self.text_lines(page).iter().enumerate() {
                    found_in(DateSource::PageText { page, line }, Some(page), &l.text, to, &mut search.hits);
                }
            }
            for a in self.info.annotations.iter().filter(|a| a.page == page) {
                if matches!(a.subtype.as_str(), "Popup" | "Link" | "Widget") || (a.grouped && a.parent.is_some()) {
                    continue;
                }
                if let Some(text) = a.contents.as_deref() {
                    found_in(DateSource::Comment { page, index: a.index }, Some(page), text, to, &mut search.hits);
                }
            }
            for f in self.form.iter().filter(|f| convertible_field(f) && field_page(f) == Some(page)) {
                if let Some(value) = f.value.first() {
                    found_in(DateSource::Field { name: f.name.clone() }, Some(page), value, to, &mut search.hits);
                }
            }
        }
        if pages.len() == n {
            for f in self.form.iter().filter(|f| convertible_field(f) && field_page(f).is_none()) {
                if let Some(value) = f.value.first() {
                    found_in(DateSource::Field { name: f.name.clone() }, None, value, to, &mut search.hits);
                }
            }
        }
        search
    }

    /// The text a date's source holds now.
    fn date_source_text(&self, source: &DateSource) -> Option<String> {
        match source {
            DateSource::PageText { page, line } => self.text_lines(*page).get(*line).map(|l| l.text.clone()),
            DateSource::Comment { page, index } => {
                self.info.annotations.iter().find(|a| a.page == *page && a.index == *index).map(|a| a.contents.clone().unwrap_or_default())
            }
            DateSource::Field { name } => self.form.iter().find(|f| &f.name == name).and_then(|f| f.value.first().cloned()),
        }
    }

    /// One undoable edit replacing `hits` (from [`Document::find_dates`]) with their conversions,
    /// and the hits it leaves unchanged with the reason. The edit is planned against the document
    /// as it is now; apply it before any other change.
    ///
    /// Page text lines are rewritten as Edit text rewrites a line. Each rewrite is first tried
    /// on a copy of the document, in the order the edit will make them (bottom line first on each
    /// page, so a line moving to the end of the page's drawing can't renumber one still to do),
    /// and a line is left out when the rewrite is refused or the line would not read back as the
    /// converted text.
    pub fn date_conversion(&self, hits: &[DocumentDate]) -> DatePlan {
        let mut plan = DatePlan::default();
        // Group the convertible hits by where they are, in first-seen order.
        let mut groups: Vec<(DateSource, String, Vec<usize>)> = Vec::new();
        for (i, h) in hits.iter().enumerate() {
            if h.conversion.text().is_none() {
                plan.skipped.push((i, SKIP_NOT_CONVERTIBLE));
                continue;
            }
            match groups.iter_mut().find(|g| g.0 == h.source && g.1 == h.context) {
                Some(g) => g.2.push(i),
                None => groups.push((h.source.clone(), h.context.clone(), vec![i])),
            }
        }
        let mut line_edits: Vec<(usize, usize, String, Vec<usize>)> = Vec::new();
        let mut other_edits: Vec<(Edit, Vec<usize>)> = Vec::new();
        for (source, context, members) in groups {
            let skip = |plan: &mut DatePlan, why| plan.skipped.extend(members.iter().map(|&i| (i, why)));
            if self.date_source_text(&source).as_deref() != Some(context.as_str()) {
                skip(&mut plan, SKIP_TEXT_CHANGED);
                continue;
            }
            let mut ranges: Vec<(Range<usize>, &str)> = Vec::new();
            for &i in &members {
                if let Some(h) = hits.get(i)
                    && let Some(to) = h.conversion.text()
                    && context.get(h.range.clone()) == Some(h.original.as_str())
                {
                    ranges.push((h.range.clone(), to));
                }
            }
            let new = (ranges.len() == members.len()).then(|| replaced(&context, ranges)).flatten();
            let Some(new) = new else {
                skip(&mut plan, SKIP_TEXT_CHANGED);
                continue;
            };
            match source {
                DateSource::PageText { page, line } => line_edits.push((page, line, new, members)),
                DateSource::Comment { page, index } => other_edits.push((Edit::SetAnnotationContents { page, index, text: new }, members)),
                DateSource::Field { name } => {
                    let max = self.form.iter().find(|f| f.name == name).and_then(|f| f.max_len);
                    if max.is_some_and(|max| new.chars().count() > max) {
                        skip(&mut plan, SKIP_FIELD_TOO_LONG);
                        continue;
                    }
                    other_edits.push((Edit::SetFieldValue { name, value: FieldValue::Text(new) }, members));
                }
            }
        }
        // Bottom line first on each page; try each rewrite on a copy.
        line_edits.sort_by(|a, b| a.0.cmp(&b.0).then(b.1.cmp(&a.1)));
        let mut edits: Vec<Edit> = Vec::new();
        let mut trial = self.editor.as_ref().map(|e| e.cos.clone());
        for (page, line, text, members) in line_edits {
            let ok = trial.as_ref().and_then(|cos| {
                let mut next = cos.clone();
                let rewritten = crate::guard(|| pdfcraft_edit::replace_line(&mut next, page, line, &text).map(|_| ())).ok().and_then(Result::ok);
                rewritten.map(|()| next)
            });
            let Some(next) = ok else {
                plan.skipped.extend(members.iter().map(|&i| (i, SKIP_LINE_REFUSED)));
                continue;
            };
            let reads = pdfcraft_edit::text_lines(&next, page).ok().and_then(|ls| ls.get(line).map(|l| l.text == text)).unwrap_or(false);
            if !reads {
                plan.skipped.extend(members.iter().map(|&i| (i, SKIP_LINE_READS_DIFFERENTLY)));
                continue;
            }
            trial = Some(next);
            edits.push(Edit::EditTextLine { page, line, text });
            plan.replaced.extend(members);
        }
        for (edit, members) in other_edits {
            edits.push(edit);
            plan.replaced.extend(members);
        }
        plan.replaced.sort_unstable();
        plan.skipped.sort_unstable();
        if !edits.is_empty() {
            plan.edit = Some(Edit::Batch { label: CONVERT_DATES_LABEL.into(), edits });
        }
        plan
    }
}

#[cfg(test)]
mod tests;
