//! `--gpu` (run, mcp, batch): PP-OCRv5 on a GPU of the process's own. Accepted by each command,
//! said on stderr (the GPU it found, or why it reads on the CPU), and never a failure: a
//! machine without a usable GPU reads on the CPU.

use std::process::{Command, Stdio};

const BIN: &str = env!("CARGO_BIN_EXE_pdfcraft-cli");

fn says_where_ocr_reads(stderr: &str) {
    assert!(stderr.contains("OCR uses the GPU") || stderr.contains("OCR reads on the CPU"), "{stderr}");
}

#[test]
fn run_and_mcp_take_gpu_and_say_where_ocr_reads() {
    let out = Command::new(BIN).args(["run", "doc_list", "--gpu"]).output().unwrap();
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "{stderr}");
    says_where_ocr_reads(&stderr);
    let listed: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert!(listed["documents"].is_array(), "{listed}");

    let out = Command::new(BIN).args(["mcp", "--gpu"]).stdin(Stdio::null()).output().unwrap();
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "{stderr}");
    says_where_ocr_reads(&stderr);

    // Without it, nothing is said about the GPU: the CPU is the default.
    let out = Command::new(BIN).args(["run", "doc_list"]).output().unwrap();
    assert!(out.status.success());
    assert!(!String::from_utf8_lossy(&out.stderr).contains("GPU"));
}
