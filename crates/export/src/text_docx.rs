//! A Word document of recognised or extracted text: one paragraph per block, no layout.
//!
//! Made for the Scan & OCR "Export text" tool, where the page picture is not wanted and Word must
//! show the text right: East Asian text is tagged with its language (`w:lang w:eastAsia`) and its
//! font slot (`w:rFonts w:eastAsia`), and a page whose text runs top to bottom becomes a section
//! with a vertical text direction (`w:textDirection w:val="tbRl"`), so Word lays the columns out
//! right to left as the source did.

use super::Zip;

/// One page of text.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TextPage {
    /// The page's size in points (the section's paper size).
    pub width: f64,
    pub height: f64,
    /// One entry per paragraph, in reading order.
    pub paragraphs: Vec<String>,
    /// The section runs its text top to bottom, columns right to left (縦書き).
    pub vertical: bool,
}

/// How the document tags its text.
#[derive(Clone, Debug, PartialEq)]
pub struct TextDocxStyle {
    /// A BCP 47 tag such as `ja-JP`. Chinese, Japanese and Korean tags set the East Asian
    /// language; any other sets the Latin one.
    pub language: String,
    /// The font Word uses for East Asian text (a name Word resolves on the reader's machine; no
    /// font is embedded).
    pub east_asian_font: String,
    /// The font for Latin text.
    pub latin_font: String,
}

impl Default for TextDocxStyle {
    fn default() -> Self {
        Self { language: "ja-JP".into(), east_asian_font: "Yu Mincho".into(), latin_font: "Century".into() }
    }
}

/// Text valid in XML 1.0 and safe in an attribute or element: control characters (other than
/// tab) and non-characters are dropped, the five markup characters escaped.
fn xml(s: &str) -> String {
    let mut o = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '&' => o.push_str("&amp;"),
            '<' => o.push_str("&lt;"),
            '>' => o.push_str("&gt;"),
            '"' => o.push_str("&quot;"),
            '\t' => o.push(c),
            c if c.is_control() || matches!(u32::from(c), 0xFFFE | 0xFFFF) => {}
            c => o.push(c),
        }
    }
    o
}

/// Whether `tag` names Chinese, Japanese or Korean, by its primary language or script.
fn east_asian(tag: &str) -> bool {
    let primary = tag.split(['-', '_']).next().unwrap_or_default().to_ascii_lowercase();
    matches!(primary.as_str(), "ja" | "zh" | "ko")
}

/// `<w:lang …/>` for `tag`: a CJK tag fills the East Asian slot (Latin text stays English), any
/// other the Latin slot.
fn lang_xml(tag: &str) -> String {
    let tag = xml(tag);
    if east_asian(&tag) { format!("<w:lang w:val=\"en-US\" w:eastAsia=\"{tag}\"/>") } else { format!("<w:lang w:val=\"{tag}\"/>") }
}

fn fonts_xml(style: &TextDocxStyle) -> String {
    format!(
        "<w:rFonts w:ascii=\"{l}\" w:hAnsi=\"{l}\" w:eastAsia=\"{e}\" w:cs=\"{l}\" w:hint=\"eastAsia\"/>",
        l = xml(&style.latin_font),
        e = xml(&style.east_asian_font)
    )
}

fn twips(pt: f64) -> i64 {
    if pt.is_finite() { (pt * 20.0).round().clamp(1440.0, 31680.0) as i64 } else { 11906 }
}

fn section(page: &TextPage) -> String {
    let (w, h) = (twips(page.width), twips(page.height));
    let margin = 1134;
    let direction = if page.vertical { "<w:textDirection w:val=\"tbRl\"/>" } else { "" };
    format!(
        "<w:sectPr><w:type w:val=\"nextPage\"/><w:pgSz w:w=\"{w}\" w:h=\"{h}\"/><w:pgMar w:top=\"{margin}\" w:right=\"{margin}\" w:bottom=\"{margin}\" w:left=\"{margin}\" w:header=\"567\" w:footer=\"567\" w:gutter=\"0\"/>{direction}</w:sectPr>"
    )
}

fn paragraph(text: &str, section: Option<&str>) -> String {
    let props = section.map(|s| format!("<w:pPr>{s}</w:pPr>")).unwrap_or_default();
    let runs = text
        .split('\n')
        .enumerate()
        .map(|(i, line)| {
            let br = if i > 0 { "<w:br/>" } else { "" };
            format!("<w:r><w:rPr>{}</w:rPr>{br}<w:t xml:space=\"preserve\">{}</w:t></w:r>", "<w:rFonts w:hint=\"eastAsia\"/>", xml(line))
        })
        .collect::<String>();
    format!("<w:p>{props}{runs}</w:p>")
}

/// The Word document (.docx) of `pages`. Each page is a section of its own, with its size and text
/// direction; an empty page keeps one blank paragraph so the section exists.
pub fn text_docx(pages: &[TextPage], title: &str, style: &TextDocxStyle) -> Vec<u8> {
    let mut body = String::new();
    for (i, page) in pages.iter().enumerate() {
        let last_page = i + 1 == pages.len();
        let count = page.paragraphs.len();
        for (k, text) in page.paragraphs.iter().enumerate() {
            let closes = k + 1 == count;
            // A section break rides on the last paragraph of its page; the last page's section is
            // the body's own.
            let carried = (closes && !last_page).then(|| section(page));
            body.push_str(&paragraph(text, carried.as_deref()));
        }
        if count == 0 {
            let carried = (!last_page).then(|| section(page));
            body.push_str(&paragraph("", carried.as_deref()));
        }
        if last_page {
            body.push_str(&section(page));
        }
    }
    if pages.is_empty() {
        body.push_str("<w:p/>");
        body.push_str(&section(&TextPage { width: 595.0, height: 842.0, ..Default::default() }));
    }
    let doc = format!(
        "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>\n\
<w:document xmlns:w=\"http://schemas.openxmlformats.org/wordprocessingml/2006/main\"><w:body>{body}</w:body></w:document>"
    );
    let fonts = fonts_xml(style);
    let lang = lang_xml(&style.language);
    let styles = format!(
        "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>\n<w:styles xmlns:w=\"http://schemas.openxmlformats.org/wordprocessingml/2006/main\">\
<w:docDefaults><w:rPrDefault><w:rPr>{fonts}<w:sz w:val=\"21\"/><w:szCs w:val=\"21\"/>{lang}</w:rPr></w:rPrDefault></w:docDefaults>\
<w:style w:type=\"paragraph\" w:default=\"1\" w:styleId=\"Normal\"><w:name w:val=\"Normal\"/><w:pPr><w:spacing w:after=\"120\"/></w:pPr><w:rPr>{fonts}{lang}</w:rPr></w:style>\
</w:styles>"
    );
    let types = "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>\n<Types xmlns=\"http://schemas.openxmlformats.org/package/2006/content-types\">\
<Default Extension=\"rels\" ContentType=\"application/vnd.openxmlformats-package.relationships+xml\"/><Default Extension=\"xml\" ContentType=\"application/xml\"/>\
<Override PartName=\"/word/document.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml\"/>\
<Override PartName=\"/word/styles.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml\"/>\
<Override PartName=\"/docProps/core.xml\" ContentType=\"application/vnd.openxmlformats-package.core-properties+xml\"/></Types>";
    let rels = "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>\n<Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\">\
<Relationship Id=\"rId1\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument\" Target=\"word/document.xml\"/>\
<Relationship Id=\"rId2\" Type=\"http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties\" Target=\"docProps/core.xml\"/></Relationships>";
    let doc_rels = "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>\n<Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\">\
<Relationship Id=\"rIdStyles\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles\" Target=\"styles.xml\"/></Relationships>";
    let core = format!(
        "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>\n<cp:coreProperties xmlns:cp=\"http://schemas.openxmlformats.org/package/2006/metadata/core-properties\" xmlns:dc=\"http://purl.org/dc/elements/1.1/\"><dc:title>{}</dc:title><dc:language>{}</dc:language></cp:coreProperties>",
        xml(title),
        xml(&style.language)
    );
    let mut z = Zip::default();
    z.add("[Content_Types].xml", types.as_bytes(), true);
    z.add("_rels/.rels", rels.as_bytes(), true);
    z.add("docProps/core.xml", core.as_bytes(), true);
    z.add("word/document.xml", doc.as_bytes(), true);
    z.add("word/styles.xml", styles.as_bytes(), true);
    z.add("word/_rels/document.xml.rels", doc_rels.as_bytes(), true);
    z.finish()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::part;

    fn page(vertical: bool, paragraphs: &[&str]) -> TextPage {
        TextPage { width: 595.0, height: 842.0, paragraphs: paragraphs.iter().map(|s| s.to_string()).collect(), vertical }
    }

    #[test]
    fn japanese_text_is_tagged_with_its_language_and_font_slot() {
        let bytes = text_docx(&[page(false, &["日本語の文章", "二つ目 & <段落>"])], "題名", &TextDocxStyle::default());
        let doc = part(&bytes, "word/document.xml");
        assert!(doc.find("日本語の文章") < doc.find("二つ目 &amp; &lt;段落&gt;"), "{doc}");
        assert!(!doc.contains("textDirection"), "{doc}");
        let styles = part(&bytes, "word/styles.xml");
        assert!(styles.contains("w:eastAsia=\"ja-JP\""), "{styles}");
        assert!(styles.contains("w:eastAsia=\"Yu Mincho\""), "{styles}");
        assert!(part(&bytes, "docProps/core.xml").contains("<dc:title>題名</dc:title>"));
    }

    #[test]
    fn a_vertical_page_gets_a_top_to_bottom_section_and_each_page_its_own() {
        let bytes = text_docx(&[page(true, &["縦書き"]), page(false, &["横書き"])], "t", &TextDocxStyle::default());
        let doc = part(&bytes, "word/document.xml");
        // The first page's section rides on its last paragraph; the last page's is the body's own.
        let first = doc.find("tbRl").expect("vertical section");
        assert!(first < doc.find("横書き").unwrap_or(0), "{doc}");
        assert_eq!(doc.matches("<w:sectPr>").count(), 2, "{doc}");
        assert_eq!(doc.matches("tbRl").count(), 1, "{doc}");
    }

    #[test]
    fn markup_and_control_characters_in_the_text_are_neutralised() {
        let bytes = text_docx(&[page(false, &["a\u{1}b</w:t>"])], "t", &TextDocxStyle::default());
        let doc = part(&bytes, "word/document.xml");
        assert!(doc.contains("ab&lt;/w:t&gt;"), "{doc}");
    }

    #[test]
    fn a_non_cjk_language_fills_the_latin_slot_and_no_pages_still_make_a_document() {
        let style = TextDocxStyle { language: "de-DE".into(), ..Default::default() };
        let bytes = text_docx(&[], "t", &style);
        assert!(part(&bytes, "word/styles.xml").contains("<w:lang w:val=\"de-DE\"/>"));
        assert!(part(&bytes, "word/document.xml").contains("<w:sectPr>"));
    }
}
