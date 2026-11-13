//! The keystroke script stored for a Japanese input format (全角カナ, 半角数字, …) converts on
//! commit in a viewer that runs field JavaScript exactly as PdfCraft converts natively. It lives
//! here because forms and js are both layer 3 and may not depend on each other.

use pdfcraft_forms::af::{JaWidth, convert_japanese, japanese_keystroke_js};
use pdfcraft_js::{DocInfo, Event, Limits, run};

fn run_keystroke(js: &str, value: &str, will_commit: bool) -> String {
    let doc = DocInfo { file_name: "form.pdf".into(), num_pages: 1, page: 0, info: Vec::new(), xfa: false, highlight: false };
    let mut ev = Event::field("Keystroke", "kana", value);
    ev.will_commit = will_commit;
    let o = run(js, &ev, &doc, &[], &[], Limits::default());
    assert_eq!(o.error, None, "{js}");
    assert!(o.rc);
    o.value
}

#[test]
fn the_stored_script_converts_as_pdfcraft_does() {
    let samples = [
        "ｶﾞｯｺｳ",
        "ﾊﾟﾋﾟﾌﾟﾍﾟﾎﾟ ｳﾞｧ ﾜﾞｦﾞ ｱﾞﾟ",
        "がっこう か\u{3099} ヴヵヶヽヾ ヷーｰ",
        "０３－１２３４－５６７８ 〒１２３ー４５６７ （０３）‐ ラーメン１",
        "ＰｄｆＣｒａｆｔ　２０２６！ ｔａｒｏ＠ｅｘａｍｐｌｅ．ｊｐ",
        "ABC 123-ｶﾞ ひらがな 漢字😀é ￥¥〜 ―５ —",
        "",
    ];
    for w in JaWidth::ALL {
        let js = japanese_keystroke_js(w);
        for s in samples {
            assert_eq!(run_keystroke(&js, s, true), convert_japanese(w, s), "{w:?} {s:?}");
        }
    }
}

/// Only the commit converts: keystrokes as they are typed are left alone (an IME is still
/// composing them).
#[test]
fn keystrokes_before_the_commit_are_left_alone() {
    let js = japanese_keystroke_js(JaWidth::Katakana);
    assert_eq!(run_keystroke(&js, "ｶﾞｯｺｳ", false), "ｶﾞｯｺｳ");
}
