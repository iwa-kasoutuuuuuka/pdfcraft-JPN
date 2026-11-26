# 高精度な日本語 OCR エンジン (PP-OCRv5): 仕様と段階的ロードマップ

> 状態: 設計メモ (P0 完了、P1 の入口 = rten で PP-OCRv5 が動くことを確認済み、P2 は試験まで: ort + CUDA と burn-wgpu の両方が RTX 5080 で動く。burn は MPL-2.0 の依存のためリポジトリに入れない) | 対象: `crates/ocr`, `crates/engine/src/ocr.rs`, `xtask`

## 1. 目的

既存のエンジン (ocrs, Tesseract, Windows OCR) に加えて、日本語 (横書き・縦書き・ルビ・表) を高精度に読む
OCR エンジンを追加する。PaddlePaddle 公式の PP-OCRv5 サーバーモデルを ONNX 形式で使い、
CPU で動く経路を基本とし、GPU で速くなる経路を任意で用意する。

- 追加するのは `Recognizer` トレイト (`crates/ocr/src/lib.rs`) の実装 1 つ。`id` / `available` / `languages` / `recognize(&OcrImage, &RecognizeOptions) -> Result<Vec<Line>, OcrError>` を満たす。
- `crates/engine/src/ocr.rs` では `EngineChoice` に 1 つ変種を足し、`recognizers()` で組み立てる。
  ページ並列 (`MAX_OCR_WORKERS = 4`) と `read_guarded` の `catch_unwind` はそのまま使う。
- モデルは **リポジトリにコミットしない**。`cargo xtask models --optional` が取得し、
  SHA-256 を検証する。`ATTRIBUTION.toml` に `optional = true` で記録するので、現時点ではリリースパッケージに同梱しない。

## 2. モデルとライセンス

| 部品 | 内容 | サイズ |
| --- | --- | --- |
| 検出 (det) | PP-OCRv5 server、ONNX opset 11 | 88 MB |
| 認識 (rec) | PP-OCRv5 server、ONNX opset 10 | 85 MB |
| 辞書 | rec の `inference.yml` の `character_dict` 18,383 件 (+ blank + 空白) | 小 |

- 入手元は PaddlePaddle 公式の Hugging Face ONNX エクスポート。ライセンスは Apache-2.0 (許可リスト内)。
- 元の場所 (リポジトリと版またはコミット) を記録し、ライセンスファイルと SHA-256 を `ATTRIBUTION.toml` に書く (AGENTS.md §1.3, §1.5)。
  出どころやライセンスが証明できなければ追加しない。
- Adobe 由来の素材は含まれないこと (§1.1) を取得時に確認する。

### 演算子の一覧 (rten での対応可否の確認対象)

- det (opset 11): Add, BatchNormalization, Concat, Conv, ConvTranspose, Identity, MaxPool, Mul, Relu, Resize, Sigmoid
- rec (opset 10): Add, AveragePool, BatchNormalization, Concat, Conv, Div, Identity, MatMul, MaxPool, Mul, Pow, ReduceMean, Relu, Reshape, Shape, Sigmoid, Slice, Softmax, Sqrt, Squeeze, Sub, Transpose, Unsqueeze

## 3. バックエンド方針

| 経路 | 実装 | 位置づけ |
| --- | --- | --- |
| CPU | rten (すでに依存に含む) | 基本。常に使える。P1 で作る |
| GPU (A) | burn-wgpu (純 Rust、wgpu 経由で Vulkan / DX12) | ONNX 取り込みの穴が許す範囲で。P2 で評価 |
| GPU (B) | ort + CUDA。**任意の cargo feature のみ**、リリースには入れない | A が使えないときの代替。ort は rc 版も可 |

ユーザー決定事項:

- CUDA は ort 経由の任意 feature に限る。リリースには含めない。
- DirectML は使わない (再配布 DLL のライセンスが許可リストにないため)。
- GPU 経路の検証は手元のベンチのみ (共有 CI では GPU を使えない)。
- manga-ocr / honmono-ocr は使わない。

設計の参考にしてよいプロジェクト (**設計の参考のみ。コードは写さない**。クリーンルーム、AGENTS.md §2):
pure-onnx-ocr (Apache-2.0, tract)、oar-ocr (Apache-2.0, ort rc.15)、rusto-rs (MIT, rten 0.27 バックエンドあり)、LumenOCR (MIT, ort)。

## 4. 計測 (P0 完了)

`cargo xtask ocr-bench` は craft-fonts の OFL フォント (BIZ UDPGothic, BIZ UDMincho, Shippori Mincho) から合成した日本語スキャンを描く。
フォントまたはエンジンがなければスキップする。結果の JSON は `target/ocr-bench/report.json`。

- レイアウト: horizontal-gothic, horizontal-mincho, vertical, ruby, table
- 条件: clean-300, clean-200, noise-300, blur-300, skew2-300, hard-200 (−2°、ぼかし、ノイズ、200 dpi)
- 指標: `cer` (順序あり)、`cer_unordered`、`line_recall` (編集 20% 以内の行)、pages/s

### Windows OCR のベースライン (言語 ja、30 ページ)

- 平均 CER (順序あり) 40.4%。これを押し上げているのは、ルビ (約 100〜137%、ふりがなを本文として読む) と、表 (約 83〜86%、セルを列方向に読む)。
- 横書き 0〜4%、縦書き 2.8〜7.6%。
- 行再現率 (line recall) 95.4%。
- 4.14 pages/s (1 ページ 0.05〜0.6 秒)。

PP-OCRv5 との比較は次の表。GPU 経路の価値は pages/s で測る。

### 結果 (2026-11-26、手元: Windows 11、論理 32 コア、CPU のみ)

`cargo xtask ocr-bench --engines windows,ppocrv5`（30 ページ = 5 レイアウト × 6 条件）。PP-OCRv5 は
`crates/ocr/examples/ocr_bench/ppocr.rs` の最小パイプライン（DB 検出 → 矩形 → 縦書きは左へ 90° 回転 →
CTC 貪欲デコード、角度分類なし）で rten 0.26 が直接 ONNX を読んで推論する。

**結論: rten 0.26 で PP-OCRv5 server の det/rec は変換なしに読み込め、推論できる。** 動かない演算子はない
（det 11 種・rec 23 種すべて対応）。モデルの読み込みは 13.9 秒（両方、初回）。

| 項目 | Windows OCR | PP-OCRv5 server (rten, CPU) |
| --- | --- | --- |
| 平均 CER (順序あり) | 40.4% | 24.1% |
| 平均 CER (順序なし) | 4.1% | 1.7% |
| 横書き CER (順序なし) | 0〜4.1% | 0〜5.8% (hard-200 以外は 0〜2.1%) |
| 縦書き CER | 2.8〜7.6% | 0〜0.7% |
| ルビ CER (順序なし) | 0〜32.2% | 0〜20.3% |
| 表 CER (順序あり) | 82.8〜85.9% (列ごとに読む) | 0〜2.0% (hard-200 は 25.3%) |
| 行再現率 | 95.4% | 97.2% |
| 速度 | 4.06 pages/s | 0.69 pages/s (約 1.2〜1.6 s/page) |

- 順序ありのルビ CER はどちらも 100% 超。ふりがなを本文として読むため（正解は親文字だけ）。ルビ行の扱いは P1 で決める。
- 精度は PP-OCRv5 が上、特に縦書きと表。速度は CPU で Windows OCR の約 1/6。GPU（P2）とバッチ化（P1）で詰める。
- 合成ページなので実スキャンでの差は未確認（リスク表の「合成ベンチの偏り」）。

## 5. 段階

### P0 ベンチ (完了)

- 完了条件: `cargo xtask ocr-bench` が上記の合成ページを描き、エンジンごとの指標と JSON を出す。Windows OCR のベースラインを取得済み。

### P1 rten による CPU エンジン (`crates/ocr`)

- 検出の後処理 (DB): 確率マップの二値化、輪郭、多角形または最小外接矩形、unclip (領域の拡張)。
- 角度と縦書きの扱い: 行の向きを判定し、縦書きの行は回転して認識に渡す。
- 認識の CTC デコード、同じ高さの行をまとめるバッチ処理。
- 入力の上限 `MAX_SIDE` / `MAX_PIXELS`。巨大な画像でメモリを使い切らない。
- panic しない (AGENTS.md §4): `unwrap` / `expect` / 添字アクセスなし、`get()` と checked 演算、モデルの出力形状は信用しない。クレートは `deny` 属性を付ける。
- 統合: `EngineChoice` に変種を追加 (`parse` の名前も)、`recognizers()` に組み込み、自動化ツール (`crates/automation`、エンドツーエンドのテスト付き)、`parity/acrobat-features.toml` の項目 (テストを `evidence` に引用)、日本語を含む UI 文字列 (craft-i18n)、`README.ja.md` の追記。
- 完了条件:
  - rten で det / rec の両モデルが読み込める（ベンチの試作で確認済み。未対応の演算子なし）。
  - 合成テスト (小さな det / rec 出力と DB 後処理、CTC、空入力、巨大入力、壊れたモデル) が通り、修正前に panic したテストが回帰テストとして残る。
  - `cargo xtask ocr-bench` で §4 の表を埋められる。
  - モデルが無い環境でも `available() == false` で他のエンジンが動く。
  - 品質ゲート (fmt、clippy `-D warnings`、`cargo test -p` 対象クレート、`cargo xtask layers`、`cargo xtask assets`、wasm チェック) が通る。

### P2 GPU

- burn-wgpu で ONNX を取り込む案、または ort + CUDA の任意 feature の案を実測して選ぶ。両方を作る必要はない。
- 比較対象は P1 の CPU 経路。環境は手元の RTX 5080。
- 完了条件: `ocr-bench` で CPU と GPU の pages/s と CER を並べた表がある。CER は CPU と同等以内 (数値の許容は P1 の結果を見て決める)。GPU が使えない環境では CPU に自動で戻る。feature 無効のビルドに影響しない。

#### P2 の途中経過 (2026-11-26、手元の RTX 5080)

**burn-wgpu (リポジトリ外の試験用クレートで確認)**: burn 0.22 と burn-onnx 0.22 の `ModelGen`（build.rs でのコード生成）が、
PP-OCRv5 server の det / rec を**無改変のまま**受け付けた。拒否された演算子も警告もなく、opset 10 / 11 のまま変換は不要。
生成コードには固定サイズが埋め込まれず、1 つのモデルで異なる入力サイズが動く。同じ入力での rten との最大差は
det 3.6e-7、rec 9.3e-5（softmax 出力）で、浮動小数点の誤差の範囲。

| 形状 | burn wgpu (GPU、平均 5 回) | 初回 (シェーダ生成込み) | rten (CPU、平均 5 回) |
| --- | --- | --- | --- |
| det 960×672 | 41 ms | 約 0.8〜1.2 s | 約 590 ms |
| det 1280×896 | 75 ms | 約 1.2 s | 約 980 ms |
| rec 48×320 | 17〜20 ms | 約 0.3〜1.5 s | 約 30 ms |
| rec 48×800 | 23 ms | 約 0.2〜0.3 s | 約 65〜80 ms |

- det は約 15 倍速い。rec は 1 行ずつだと 2〜3 倍にとどまる（1 回 17〜23 ms の下限）。行をまとめるバッチ処理で詰める余地がある。
- 新しい入力サイズの初回はシェーダの生成で 0.2〜1.5 秒かかる。検出の入力サイズを数種類に丸めてキャッシュを効かせる。
- 純 Rust で `unsafe` なし。burn 本体は MIT OR Apache-2.0 だが、**依存ツリーに許可リスト外の MPL-2.0 が 2 つ必須で入る**（`colored` 3.1.1 ← burn-tensor、`buildid` 1.1.0 ← cubecl-runtime。2026-11-26 に `cargo tree` で確認）。依存ツリーも大きく（冷えた状態からのリリースビルドに約 3 分）、
  生成された `.bpk` の重み（約 170 MB）が ONNX と別に要る。burn-onnx 0.22.0 と burn 0.22.x の版を揃える必要がある。
- DX12 と Vulkan のどちらが選ばれたかは未確認。実ページの画像での文字列の一致も未確認。

**ort + CUDA (任意 feature `ort-cuda`、リリースに入れない)**: `ocr-bench` に `ppocrv5-cuda` として実装した。
ONNX Runtime 1.26.0 の GPU 版（NuGet の Microsoft 公式パッケージ、MIT）と、CUDA 12.8 の cudart・cuBLAS、cuDNN 9.7.1
（NVIDIA 公式の再配布用 zip。NVIDIA 独自ライセンスのため、この PC で使うだけで配布しない）を `ORT_DYLIB_PATH` と `PATH` で与える。
CUDA の登録に失敗したら CPU に黙って戻さず、エラーにする。

| エンジン (30 ページ、行はまとめて推論) | 平均 CER (順序なし) | 行再現率 | 速度 | 1 ページの中央値 |
| --- | --- | --- | --- | --- |
| PP-OCRv5 + ort CUDA (RTX 5080) | 0.8% | 99.2% | 3.40 pages/s | 0.31 s |
| PP-OCRv5 + rten (CPU) | 0.8% | 99.2% | 0.61 pages/s | 1.65 s |
| Windows OCR (参考) | 4.1% | 95.4% | 4.06 pages/s | — |

- RTX 5080 (Blackwell, sm_120) で CUDA の実行は成功した。読みは CPU と同じで、精度の差はない。
- 最初は 1 行ずつ推論して 1.13 pages/s だった。cuDNN が入力の形状ごとに調整するため、行ごとに幅が違うと遅い。
  認識の入力幅を 160 px 単位に丸めて右を 0 で埋め（PaddleOCR と同じ埋め方）、同じ幅の行を最大 8 行まとめて推論すると 3.40 pages/s になった。
  この変更で CPU 版の精度も上がった（順序なしの CER 1.7% → 0.8%、行再現率 97.2% → 99.2%）。
- 1 ページずつの計測で、アプリのページ並列 (`MAX_OCR_WORKERS`) は使っていない。残りの時間は主に CPU 側の前処理と後処理。

**P2 の結論**: GPU 経路は 2 本とも RTX 5080 で動く。ただし、どちらもリリースには入れられない。
- ort + CUDA: オーナーの決定で任意 feature に限る。NVIDIA のライブラリは再配布できない。手元の計測と比較用。
- burn-wgpu: 必須の依存に MPL-2.0 が入る。許可リスト（AGENTS.md と deny.toml）に無く、`cargo xtask deny`（all-features）が失敗する。
  許可リストは上流の storytold/craft-contrib のルールでしか変えられないため、オーナーの判断で burn のコードはリポジトリに入れない（2026-11-26）。
  上の数値はリポジトリ外の試験用クレートで測ったもの。

リリースに入れる GPU 経路は、許可リストに収まる別の案を後で探す。候補は wgpu を直接使う自前の推論（演算子は det 11 種・rec 23 種と少ない）、
または MPL-2.0 の 2 クレートを使わない burn の構成（上流の対応待ち）。それまでの既定は CPU の rten とする。

### P3 UI 統合と Automatic の振り分け

- 設定画面にエンジンの選択肢を出す。日本語・中国語・韓国語の `Auto` で PP-OCRv5 をどう組み合わせるか (主にする、副にする、ルビや表のときだけ使う) をベンチで決める。
- 完了条件: UI で選べて、`pdfcraft-cli ui` (`inspect`、`click`、`screenshot`) で実際の画面を確認済み。headless のツールでも同じ結果。
  進捗表示と取り消しが 4 並列のページ読みで動く。

### P4 配布の判断

- 170 MB 級のモデル (det 88 MB + rec 85 MB) を同梱するか、初回に取得するか、同梱しないかを決める。
- **ユーザーの明示的な承認が必要。** 承認前は `optional = true` のまま。
- 完了条件: 決定を `docs/` に記録し、`ATTRIBUTION.toml` と配布物が一致する。`release.yml` の変更は別の PR (AGENTS.md の CI 規則)。

## 6. リスク

| リスク | 内容 | 対処 |
| --- | --- | --- |
| rten の速度 | 演算子はすべて対応済み（§4 の結果）。CPU では約 0.69 pages/s と遅い | P1 でバッチ化と検出の入力サイズを詰め、P2 の GPU 経路で補う |
| burn の ONNX 取り込みの穴 | 同じ演算子が取り込めない、または結果がずれる | CPU 経路の出力と数値で比較。穴があれば ort に切り替える |
| モデルサイズ | 合計 173 MB 程度 (88 + 85)。パッケージとメモリに重い | 同梱は P4 で判断。それまでは任意取得 |
| Blackwell (sm_120) | RTX 5080 向けの CUDA を含む ort のビルドがあるか未確認 | P2 で実機確認。動かなければ burn-wgpu を主にする |
| 検証の範囲 | 共有 CI に GPU がない | GPU 経路は手元のベンチのみで検証し、結果を PR の説明に正直に書く (「実行したものだけ報告」) |
| GPU 経路のライセンス | burn は MPL-2.0 の依存（colored、buildid）を必須で持つ。ort + CUDA は NVIDIA の再配布不可のライブラリが要る | リリースの GPU 経路は許可リストに収まる案が見つかるまで保留。CPU（rten）を既定にする |
| 合成データの偏り | ベンチは合成ページなので実スキャンとの差がある | 実ファイルでの確認は別途。数値は「合成ベンチ」と明記して書く |

## 7. 機能ごとの AGENTS.md の義務

- **資産 (§1)**: モデルと辞書は `ATTRIBUTION.toml` に 1 ファイル 1 項目 (パス/URL、作者、出典の版、SPDX、ライセンスファイル、SHA-256、用途)。`cargo xtask assets --write` のあと `cargo xtask assets`。`ATTRIBUTION.md` は手で編集しない。Adobe 由来は不可。
- **フォント (§1.4)**: ベンチに使う日本語フォントは craft-fonts にあり、このリポジトリにはコミットしない。
- **クリーンルーム (§2)**: 参考プロジェクトのコードを写さない。公開仕様と観測した挙動だけで書く。
- **自動化 (§3)**: すべての機能に headless のツールとエンドツーエンドのテスト。MCP や GPU の経路を既定で有効にしない。
- **落ちない (§4)**: Result で失敗を返し、入力由来の数値はすべて敵対的とみなす。再帰と確保に上限。最後の砦の `catch_unwind` は維持する。
- **ゲート (§5)**: 触ったクレートに scoped な clippy / test、`xtask layers`、`xtask assets`、wasm チェック。UI は L7 より下に egui を持ち込まない。
- **parity**: `parity/acrobat-features.toml` の項目は、テストが通ったときだけ `shipped`。
- **CI**: 機能 PR で CI を変更しない。足りなければ craft-common-actions に issue を出す。
- **ライセンス**: 新しい依存 (ort、burn など) は許可リストを確認する。リストにない識別子は止まって相談する。
