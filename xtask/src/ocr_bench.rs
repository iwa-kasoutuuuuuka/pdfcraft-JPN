//! `cargo xtask ocr-bench`: build and run the `ocr_bench` example of pdfcraft-ocr
//! (`crates/ocr/examples/ocr_bench/`) in release.
//!
//! The example draws synthetic Japanese scans from the OFL fonts of craft-fonts and scores every
//! available engine on them. `CRAFT_FONTS_DIR` points at the craft-fonts checkout (default
//! `./craft-fonts`). With no checkout or no engine it prints why and still succeeds, so a machine
//! without Windows OCR or the fonts is not a failure. Arguments after the command name go to the
//! example (`--engines windows,tesseract`, `--out FILE`, `--images DIR`, `--quick`).

use std::process::Command;

use anyhow::{Context, Result, bail};

use crate::gates::{root, target_dir};

pub fn run(args: &[String]) -> Result<()> {
    let root = root();
    let mut cmd = Command::new(std::env::var("CARGO").unwrap_or_else(|_| "cargo".into()));
    cmd.current_dir(&root).args(["run", "--release", "-p", "pdfcraft-ocr", "--example", "ocr_bench", "--"]);
    if !args.iter().any(|a| a == "--out") {
        cmd.arg("--out").arg(target_dir().join("ocr-bench/report.json"));
    }
    cmd.args(args);
    let status = cmd.status().context("running cargo")?;
    if !status.success() {
        bail!("ocr_bench exited with {status}");
    }
    Ok(())
}
