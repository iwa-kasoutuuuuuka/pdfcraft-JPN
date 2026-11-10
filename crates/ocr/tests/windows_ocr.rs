//! Windows OCR against a real Japanese page (vertical block + horizontal paragraph). Needs the
//! Windows Japanese OCR language, so it is `#[ignore]`d; run with
//! `cargo test -p pdfcraft-ocr --test windows_ocr -- --ignored --nocapture`.
#![cfg(windows)]
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use pdfcraft_ocr::windows_ocr::WindowsOcr;
use pdfcraft_ocr::{OcrImage, RecognizeOptions, Recognizer};

#[test]
#[ignore = "needs the Windows Japanese OCR language and target/jpn-demo/big.png"]
fn reads_vertical_and_horizontal_japanese() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/jpn-demo/big.png");
    let mut decoder = png::Decoder::new(std::io::BufReader::new(std::fs::File::open(path).unwrap()));
    decoder.set_transformations(png::Transformations::EXPAND);
    let mut reader = decoder.read_info().unwrap();
    let mut buf = vec![0; reader.output_buffer_size().unwrap()];
    let info = reader.next_frame(&mut buf).unwrap();
    let px = &buf[..info.buffer_size()];
    let rgba: Vec<u8> = match info.color_type {
        png::ColorType::Rgba => px.to_vec(),
        png::ColorType::Rgb => px.as_chunks::<3>().0.iter().flat_map(|p| [p[0], p[1], p[2], 255]).collect(),
        png::ColorType::Grayscale => px.iter().flat_map(|&g| [g, g, g, 255]).collect(),
        png::ColorType::GrayscaleAlpha => px.as_chunks::<2>().0.iter().flat_map(|p| [p[0], p[0], p[0], p[1]]).collect(),
        other => panic!("{other:?}"),
    };
    let image = OcrImage::new(info.width, info.height, rgba).unwrap();
    let lines = WindowsOcr.recognize(&image, &RecognizeOptions { language: "ja".into() }).unwrap();
    let texts: Vec<String> = lines.iter().map(|l| l.text()).collect();
    for l in &lines {
        println!("{:?} {:?}", l.text(), l.words.iter().map(|w| w.rect).collect::<Vec<_>>());
    }
    let all = texts.join("\n");
    assert!(all.contains("吾輩は猫である"), "{all}");
    assert!(all.contains("横書きの日本語テキスト"), "{all}");
}
