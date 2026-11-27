# 高精度な日本語 OCR エンジン (PP-OCRv5): 仕様と段階的ロードマップ

> 状態: P0 完了、P1 実装済み (エンジン `ppocr`、rten で CPU)、P1.5 実装済み (GPU の差し替え口と L7 クレート `pdfcraft-ocr-gpu`)、P2 実装済み (自前の WGSL 推論で det・rec とも GPU で動く)、P3 実装済み (設定画面の GPU の切り替え、`--gpu`、日本語・中国語の Automatic は PP-OCRv5)。ort + CUDA は手元の比較用の任意 feature のまま。burn は MPL-2.0 の依存のためリポジトリに入れない | 対象: `crates/ocr`, `crates/ocr-gpu`, `crates/engine/src/ocr.rs`, `xtask`

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
- 3.4〜4.1 pages/s（計測ごとの幅。機械の混み具合で変わる。1 ページ 0.05〜0.6 秒）。

PP-OCRv5 との比較は次の表。GPU 経路の価値は pages/s で測る。

### 結果 (2026-11-26、手元: Windows 11、論理 32 コア、CPU のみ)

`cargo xtask ocr-bench --engines windows,ppocrv5`（30 ページ = 5 レイアウト × 6 条件）。PP-OCRv5 は
`crates/ocr/examples/ocr_bench/ppocr.rs` の最小パイプライン（DB 検出 → 矩形 → 縦書きは左へ 90° 回転 →
CTC 貪欲デコード、角度分類なし）で rten 0.26 が直接 ONNX を読んで推論する。

**結論: rten 0.26 で PP-OCRv5 server の det/rec は変換なしに読み込め、推論できる。** 動かない演算子はない
（det 11 種・rec 23 種すべて対応）。モデルの読み込みは 13.9 秒（両方、初回）。

初回の計測（認識を 1 行ずつ推論していた版）。行をまとめて推論する版の数値は、§5 の P2 の表を参照（順序なしの CER 0.8%、行再現率 99.2%、約 0.6 pages/s）。

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

## 4.5 本実装の決定事項 (オーナー、2026-11-26)

| 項目 | 決定 |
| --- | --- |
| GPU 推論の置き場所 | 新しい L7 クレート `pdfcraft-ocr-gpu`（`xtask/src/layers.rs` に 1 行追加）。`crates/ocr` には推論の差し替え口（トレイト）だけを置き、アプリと CLI（L8）が登録する |
| GPU デバイス | 画面描画用（eframe の wgpu）のデバイスとキューを共用する |
| アプリでの既定 | 設定で「GPU で高速化」を有効にしたときだけ。AMD・Intel の GPU で確かめたら既定で有効に切り替える。CLI・MCP は既定で CPU、`--gpu` を明示したときだけ GPU |
| ONNX の読み込み | 自前の最小実装（長さの検査とファジング付き）。依存は増やさない |
| リリースでの扱い | モデル同梱の判断（P4）までは、エンジンの選択肢に PP-OCRv5 を出さない（モデルを `cargo xtask models --optional` で入れた環境だけ） |
| Auto の組み合わせ | P3 でベンチを取って決める |
| GPU 失敗時の通知 | CPU に切り替えたことを 1 セッションに 1 回だけ知らせる。各ページの注記には毎回残す |
| 速度の再確認 | 不要になった（試作 2 で det 全体を実測、960×672 で 25 ms） |

描画用のデバイスを共用するので、本実装では次を守る。OCR の計算は小さな送信に分け、1 回の送信が画面の 1 フレーム（約 16 ms）を大きく超えないようにする。
OCR 側の検証エラーは error scope で捕まえ、描画側の未捕捉エラーにしない。デバイスを失ったときは OCR を CPU に切り替え、描画の復旧は eframe に任せる。
GPU のメモリは中間バッファを生存区間で再利用して抑える（試作では再利用なしで 872 MB 使い、split-K を足すと GPU のメモリが足りなくなった）。

## 5. 段階

### P0 ベンチ (完了)

- 完了条件: `cargo xtask ocr-bench` が上記の合成ページを描き、エンジンごとの指標と JSON を出す。Windows OCR のベースラインを取得済み。

### P1 rten による CPU エンジン (`crates/ocr`) — 実装済み

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

#### P1 の実装 (2026-11-26)

- `crates/ocr/src/ppocr.rs`: エンジン `ppocr`（`PpOcr`、`Recognizer` を実装）。ベンチの試作と同じ流れを、ライブラリの部品として書き直した。
  - モデル 3 ファイルは、`ATTRIBUTION.toml` の SHA-256 と一致したときだけ読み込む（rten の解析器は、モデルでないバイト列で止まらなくなることがあるため）。1 ファイル 512 MB まで。
  - 検出の長辺は既定 1600、32〜2560 に制限。箱は 1 ページ 4000 個まで（スコアの高い順に残す）。辞書は 10 万項目まで。ネットワークの出力の形は長さと突き合わせて検査する。
  - 認識は同じ幅にそろえた行を最大 8 行ずつ。切り出しはそのバッチを読むときだけ作るので、同時に持つ行は 8 行まで。
  - CTC で文字が出た時刻から、行の中の位置を求める。漢字・かなは 1 文字ずつ単語にし、英数字は空白で区切る。縦書きの列でも 1 文字ずつ選べる透明テキストになる。
  - `ja-vert` は縦書きのページとして読む。`ja` はページごとに箱の形で決める。
  - 推論は `Network` トレイトの後ろにある。テストは合成の出力を与え、ベンチの `ort-cuda` は同じ前後処理で CUDA を測る。GPU を登録する口（P1.5）はまだ作っていない。
- engine: `EngineChoice::PpOcr`（`ppocr` / `pp-ocr` / `ppocrv5`）。最初のジョブが読み込み（数秒、ロックの外で行う）、セッション中は使い回す。「自動」ではまだ使わない（P3 で決める）。
- 自動化: `ocr_recognize` などの `engine` に `ppocr` を追加。`ocr_status` が `ppocr` の状態と、無いときの理由を返す。
- UI: エンジンの一覧に「PP-OCRv5」を出すのは、モデルが入っている環境だけ（§4.5 の決定どおり、リリースには出ない）。
- ルビ: 本文とは別の行として読む。順序なしの CER は 0〜6.8% で本文は読めている。ルビを除く処理は入れていない（文字の大きさで判定すると、表の小さい文字も落とすおそれがあるため）。
- 合成 30 ページ（`cargo xtask ocr-bench -- --engines ppocrv5`、手元）: 順序なしの CER は 0.7%、行の再現率は 99.8%、0.50 ページ/秒（試作では 0.8%・99.2%・約 0.6 ページ/秒。速度の差は、PC の負荷の違いを含む）。

### P1.5 GPU の差し替え口 (2026-11-27、実装済み)

- `crates/ocr`（L4）: `ppocr::NetworkFactory` トレイトを追加した。検証済みのモデルのバイト列から、GPU 上の det / rec の `Network` を作る。
  - `PpOcr::load_with_gpu` は CPU のネットワークも必ず読み込み、切り替え先として持つ。
  - 自己検査: 決まった入力（det は 64×64、rec は 48×160）を CPU と GPU の両方に通し、形が同じで差が 1e-3 以内のときだけ GPU を使う。外れたら CPU のままにし、理由を毎ページの注記に残す。
  - 実行中に GPU がエラーを返す、出力の形が値の数と合わない、NaN や inf を返すときは、そのページを CPU で読み直し、セッションの残りも CPU にする。注記はそのページと以後の毎ページに残る。
  - `take_gpu_fallback_notice()` は、最初の切り替えの理由を 1 セッションに 1 回だけ返す（UI での通知は P3）。
  - `Recognizer::recognize_noted`（既定の実装は注記なし）を追加し、`recognize_with_strategy` が各エンジンの注記をページの注記に渡すようにした。
- engine（L6）: `register_gpu(Arc<dyn NetworkFactory>)` と `OcrSettings::gpu`（既定は false）を追加した。
  - GPU を求めたジョブで登録が無いときは、CPU で読み、「GPU のバックエンドが無い」と注記する。
  - キャッシュは GPU の有無ごとに持つ。
- `crates/ocr-gpu`（新しい L7 クレート、`xtask/src/layers.rs` に 1 行）:
  - `onnx`: 自前の最小 ONNX リーダー（依存は増やしていない）。長さはすべて残りの範囲と突き合わせる。varint は 10 バイトまで、サブグラフはたどらない。ノード・重み・属性・次元・重みの値の数には上限がある。外部データは読まない。Identity の循環でも止まる。
    - 切り詰め、1 バイトの書き換え、乱数のバイト列で panic しないことを試験している。
    - 実モデルは det が opset 11・演算子 11 種、rec が opset 10・23 種で、§4 と一致した。Conv の重みは Identity や Constant をたどって全部見つかる。
  - `GpuContext::shared`: 描画用のデバイスとキューを受け取る（§4.5 の決定）。ソフトウェアのアダプター（WARP、llvmpipe）は断る。
  - `WgpuFactory`: 両モデルを読み、カーネルの無い演算子を名前で挙げて断る（`SUPPORTED_OPS` はまだ空）。そのため、いまは GPU を有効にしても CPU で読み、理由が注記に出る。
- アプリ（L8）: eframe の wgpu の描画状態があるとき、`WgpuFactory` を登録する。設定の UI はまだ無い（P3）ので、ジョブが GPU を求めることは無い。CLI / MCP の `--gpu` も P3。
- P2 で守ること（描画用デバイスを共用するため）: 送信は小さく分けて 1 フレーム（約 16 ms）を大きく超えない。検証エラーは error scope で捕まえる。デバイスを失ったら `Network::run` がエラーを返し、上の切り替えで CPU に移る。

### P2 GPU

- burn-wgpu で ONNX を取り込む案、または ort + CUDA の任意 feature の案を実測して選ぶ。両方を作る必要はない。
- 比較対象は P1 の CPU 経路。環境は手元の RTX 5080。
- 完了条件: `ocr-bench` で CPU と GPU の pages/s と CER を並べた表がある。CER は CPU と同等以内 (数値の許容は P1 の結果を見て決める)。GPU が使えない環境では CPU に自動で戻る。feature 無効のビルドに影響しない。

#### P2 の実装 (2026-11-27、自前の WGSL)

`crates/ocr-gpu` に、検出と認識の両方を wgpu の計算シェーダーだけで動かす推論を入れた。新しい外部依存はない（wgpu・naga・skrifa・png はツリーにある版。naga と skrifa・png は開発用の依存だけ）。

- **グラフの変換（`ir`、ネットワークごとに 1 回）**: Identity は名前の別名として解決する。重みだけで決まる部分は読み込み時に計算する。
  - Conv・ConvTranspose・MatMul は、その値だけを読む後続を取り込む。対象は BatchNormalization、チャネルごとの Add・Mul（1 組の倍率とずれにまとめる）、Relu・Sigmoid・swish（`x * Sigmoid(x)`）、すでに計算済みの値との残差 Add。
  - 畳み込みだけが読む Concat（チャネル方向）と Resize（最近傍・整数倍）は作らず、畳み込みが元の値を読む。
  - DB の出力部（3×3 で 64 チャネル → 1×1 で 1 チャネル → Sigmoid → もう一方の枝との平均）は 1 本のカーネルにした。64 チャネル・全解像度の中間値（1600 辺で 470 MB）を作らない。
  - 認識の最後（MatMul + bias → Softmax）は、行ごとの最大のクラスとその確率だけを計算する別の経路を持つ。
- **計画（`plan`、入力の大きさごと）**: 形を決め、Shape から始まる形の計算をここで解く。生存区間が重ならない中間値はバッファを共有する。
  - 既定の上限（1 回にバインドできるバッファの大きさ、ディスパッチの数）は計画の段階で確かめ、超えるときは理由を返す。
- **カーネル（`wgsl`）**: 寸法は uniform で渡すので、1 つのパイプラインがどの入力の大きさにも使える（ページの縦横比ごとに作り直さない）。
  - 畳み込みは implicit GEMM（64×64 のタイル、1 スレッド 4×4）。行列積、DB の出力部、最大クラスの部分集計も同じループを使う。
  - depthwise・グループ畳み込み、ConvTranspose、プーリング、要素ごとの演算（ブロードキャスト）、Softmax、転置・Slice は専用の単純なカーネル。
  - どれも `wgpu::Limits::default()` の範囲（256 スレッド、共有メモリ 12 KiB まで）に収め、使う量はデバイスの上限と突き合わせる。
- **実行（`exec`）**: 重みはネットワークごとに 1 回だけ転送する。計画は最近のものを 256 MB の範囲で残す。
  - 描画とデバイスを共用するので、1 回の送信は約 8 ms に収まるよう、実測の速さから区切る。
  - デバイスの呼び出しはすべて error scope の内側で行う。検証エラー、メモリ不足、20 秒の時間切れ、マップの失敗は、どれもその実行の `Err` になる。engine は CPU に切り替え、注記に残す。
  - 計画が 512 MB を超えるバッチ（認識の幅 3200 の 8 行は 1.5 GB）は、半分ずつに分けて読む。
  - wgpu 30 の `Device::poll` は、デバイスロスト（ドライバーのリセット・TDR・VRAM 枯渇）のとき `Err` を返さず panic する（`PollError` になるのは時間切れだけ）。
    そのため poll と、各入口（`run`・`run_best`・`networks_on`）は panic を受け止めてその実行の `Err` に変える。試験はデバイスを破棄して確かめる。
  - 1 つの計画の中間バッファは 2 GiB まで。それを超える大きさは `Err` になり、そのページは CPU で読む。
- **悪意あるモデルへの備え**: 読み込み時に計算する定数は、出力の大きさを確保の前に検査する（1 つ 1,677 万値まで。空のテンソルでも、0 以外の次元の積で上限を見る）。
  重みと畳み込んだ定数の合計は 2²⁸ 値（`onnx::MAX_VALUES`、ONNX の読み込みと同じ上限）まで。
  - 畳み込み・転置畳み込み・プーリングの窓は、辺 64（プーリングは面積 65,536）、ストライドと dilation 64、パディング 1024、出力 1 値あたりの積和 131,072 まで。
    `output_padding` はストライドか dilation より小さいこと（ONNX の規定）。PP-OCRv5 は窓 7 以下、積和は数千で、十分に下回る。
  - 計画の大きさの計算は、すべて checked か saturating（release ビルドには桁あふれの検査がない）。
- **`Network::run_best`（`crates/ocr`）**: 貪欲 CTC に要るのは各時刻の最大クラスとその確率だけなので、認識はこれで読む。既定の実装は `run` の結果を CPU で集計する。GPU は 18,385 クラスの確率を読み戻さず、1 行あたり 2 つの数だけを返す。
  - 自己検査も、認識はこの形で行う。GPU の最大クラスは、CPU がそのクラスにつけた確率が CPU 自身の最大と 1e-3 以内のときに一致とみなす（拮抗した行はどちらでもよい）。
- **アプリ**: 描画用デバイスのバッファの上限を、アダプターが許す範囲で 1 GiB まで上げて要求する（既定の 128 MiB では、正方形に近い 1600 辺のページの検出が収まらない）。描画は使わない。

**数値の確認（RTX 5080、Vulkan、`cargo test -p pdfcraft-ocr-gpu`、モデルあり）**

| 対象 | rten との最大差 |
| --- | --- |
| det 64×64 | 2.2e-8（0.3 での二値化後の差 0 画素） |
| det 672×480 | 1.7e-8（同 0 画素） |
| rec 1 行 × 160 | 2.7e-6 |
| rec 4 行 × 640 | 7.3e-6 |

- 合成の小さなモデル（畳み込みと正規化・残差、depthwise・グループ、DB の出力部、認識の系列部分と最大クラス、大きさの変化とバッチの分割）でも rten と比べている。
  - アダプターがあればソフトウェアのもの（Windows の WARP）でも走る。無ければ、その旨を出して飛ばす。
  - 全カーネルは naga で、GPU なしに検証している（CI でも走る）。

**ベンチ（P2 の完了条件。2026-11-27、合成 30 ページ、検出の辺 1600）**:
`cargo run --release -p pdfcraft-ocr-gpu --example ocr_bench_gpu -- --engines ppocrv5,ppocrv5-wgpu`。
`ocr_bench_gpu` は pdfcraft-ocr のベンチをそのまま取り込み、engine と同じ自己検査と CPU への切り替えを通る `ppocrv5-wgpu` を足したもの。
CPU に切り替わったページがあれば注記を表示する（今回は 0 件）。

| エンジン | 平均 CER (順序なし) | 行再現率 | 速度 | 1 ページの中央値 |
| --- | --- | --- | --- | --- |
| PP-OCRv5 + 自前 WGSL (RTX 5080、Vulkan) | 0.7% | 99.8% | 2.85 pages/s | 0.36 s |
| PP-OCRv5 + rten (CPU) | 0.7% | 99.8% | 0.49 pages/s | 2.05 s |

- 30 ページすべてで、GPU と CPU の読みの文字列が完全に一致した。CER の許容は「CPU と同一」で満たしている。
- 計測の間、別の作業が GPU をほぼ使い切っていた（使用率 96〜100%）。空いた GPU ではもっと速い見込み（未計測）。
- 起動時のカーネル作成と自己検査は 3.0 秒（rten の読み込みは 0.5 秒）。
- 残りの時間の多くは CPU 側の前処理・後処理と、1 行ずつの認識の送信。NVIDIA 以外の GPU（AMD、Intel の内蔵 GPU）での確認はまだ。

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

**ort + CUDA (任意 feature `ort-cuda`、リリースに入れない)**: ベンチの `ppocrv5-cuda` エンジンとして実装した。`cargo xtask ocr-bench` は feature を付けないので、
`cargo run --release -p pdfcraft-ocr --example ocr_bench --features ort-cuda -- --engines ppocrv5-cuda` で動かす（`ORT_DYLIB_PATH` と `PATH` は下のとおり）。
ONNX Runtime 1.26.0 の GPU 版（NuGet の Microsoft 公式パッケージ、MIT）と、CUDA 12.8 の cudart・cuBLAS、cuDNN 9.7.1
（NVIDIA 公式の再配布用 zip。NVIDIA 独自ライセンスのため、この PC で使うだけで配布しない）を `ORT_DYLIB_PATH` と `PATH` で与える。
CUDA の登録に失敗したらエラーにする（登録のあとで ONNX Runtime が一部のノードを CPU に割り当てることまでは検出しない）。

1 ページの中央値は、出力の JSON の `seconds` から計算した。この時点の計測は最初のページだけを予行していたので、各入力形状の初回の調整（cuDNN）を含む。
いまのベンチは全ページを一度読んでから測る。

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
- 別の作業が GPU をほぼ使い切っている（使用率 96〜99%）状態で測り直すと、30 ページ中 24 ページが cuBLAS のエラー（failure 13、GPU 上で起動できない）になった。
  GPU の取り合いで起きたと見ている。ベンチはこれを各ページのエラーとして表示し、速度の計算から外す。本実装では、このエラーで CPU に切り替える。

**P2 の結論**: GPU 経路は 2 本とも RTX 5080 で動く。ただし、どちらもリリースには入れられない。
- ort + CUDA: オーナーの決定で任意 feature に限る。NVIDIA のライブラリは再配布できない。手元の計測と比較用。
- burn-wgpu: 必須の依存に MPL-2.0 が入る。許可リスト（AGENTS.md と deny.toml）に無く、`cargo xtask deny`（all-features）が失敗する。
  許可リストは上流の storytold/craft-contrib のルールでしか変えられないため、オーナーの判断で burn のコードはリポジトリに入れない（2026-11-26）。
  上の数値はリポジトリ外の試験用クレートで測ったもの。

リリースに入れる GPU 経路は、許可リストに収まる別の案を後で探す。候補は wgpu を直接使う自前の推論（演算子は det 11 種・rec 23 種と少ない）、
または MPL-2.0 の 2 クレートを使わない burn の構成（上流の対応待ち）。それまでの既定は CPU の rten とする。

#### 許可リストに収まる GPU 経路の調査 (2026-11-26)

**必要なカーネル**（ONNX のグラフを属性まで数えた結果）:

- det: 通常の Conv 131 個（1×1 が 50、3×3、9×9、7×7、5×5、1×k / k×1、2×2、ストライド 2 あり）、depthwise Conv 27 個（5×5、3×3 ストライド 2）、
  ConvTranspose 2×2 ストライド 2 が 2 個、BatchNormalization 87 個（読み込み時に Conv の重みへ畳み込める）、MaxPool 2×2（SAME_UPPER）、
  Resize（nearest、asymmetric、floor、整数倍）、Relu、Sigmoid、Add、Mul、Concat（チャネル方向）。
- rec: 上と同じ種類の Conv（depthwise のストライド (2,1)・(1,2) を含む）、AvgPool 3×2、Transformer 部分（SVTR）の MatMul（120 次元。最後は 120×18385）、
  LayerNorm 相当（ReduceMean・Sub・Pow・Sqrt・Div）、Softmax、Reshape・Transpose・Squeeze・Unsqueeze・Slice・Shape。
  形状の計算（Shape → Slice → Reshape）は、入力の幅が決まれば定数に畳める。

**既存のクレート**:

| 候補 | 状態 | 判断 |
| --- | --- | --- |
| wonnx 0.5.1 | 2025-05 にアーカイブ。wgpu 0.19。ConvTranspose と Slice が未実装、入力形状は固定のみ | 使わない |
| oxionnx 0.1.7 (Apache-2.0) | アルファ版。GPU は MatMul・Conv・Softmax・Relu・Resize・要素ごとの演算だけで、ほかは CPU に戻る | 参考にとどめる（CPU と GPU の往復が多く、rten に勝てない見込み） |
| candle | wgpu バックエンドは未マージ（PR が開いたまま） | 使えない |
| ratchet / luminal | LLM 向け、または更新停止 | 使えない |
| burn 0.22 | MPL-2.0 の依存が必須 | 許可リスト外 |
| ort | C++ の ONNX Runtime を包む | 任意 feature のみ（既定の方針どおり） |

**自前の WGSL 推論（推奨案）**:

- wgpu 30.0.1 はすでに eframe 経由で依存ツリーにあり、`cargo xtask deny` も通っている。ライセンスの追加はない（DXC を包む `mach-dxcompiler-rs` は有効にしない）。
- **層の規則**: `cargo xtask layers` は wgpu を UI 系のクレートとして扱う。L4 の `crates/ocr` は wgpu に依存できない。
  そこで `crates/ocr` には推論の差し替え口（ベンチの `Net` に当たるトレイト）だけを置き、wgpu の実装は L7 側に置く。
  アプリが egui の描画用にすでに持っている wgpu のデバイスを共用できる。層の規則を変えずに済む形だが、どこに置くかはオーナーの確認が要る。
- 作るもの:
  - ONNX の読み込みと、入力サイズの段階ごとの実行計画（形状の計算を定数に畳み、計画をキャッシュする）。
  - BN の畳み込みと、Conv に Bias・Relu / Sigmoid・Add を続けて行う融合。
  - Conv の implicit GEMM（共有メモリのタイルとレジスタブロッキング。1×1 はそのまま GEMM）、depthwise Conv、ConvTranspose、プーリング、Resize、
    rec 用の MatMul・LayerNorm・Softmax・Transpose。
  - rten の出力と数値で比べるテスト。panic しない作りで、GPU が無い・失敗したときは rten に戻る。
- 速度の見込み（未計測）: よく調整した f32 の WGSL は cuDNN の FP32 の 2〜4 割程度。det の大きいページでも数十 ms で、CPU より大幅に速い。
  `shader-f16` を使えるともう一段速くなる。NVIDIA 以外の GPU（AMD、Intel の内蔵 GPU）での確認が要る。
- 工数の見込み: 正しい出力まで 3〜5 週、f16 と融合の調整に 1〜2 週。最初に 1×1 Conv と depthwise Conv だけを作り、手元で rten と速度を比べる小さな試作（2〜3 日）で見込みを確かめるのがよい。

**試作の結果 (2026-11-26、リポジトリ外の試験用クレート、RTX 5080)**: wgpu 30.0.1 の既定 feature（DX12 と Vulkan。DXC は使わない）の上に、
1×1 Conv（共有メモリのタイルとレジスタブロッキングの GEMM、Bias と ReLU を融合）と depthwise Conv（入力タイルを共有メモリに置く）を WGSL で書いた。
`unsafe` なし、依存は許可リスト内。wgpu は既定で Vulkan を選んだ。時間は GPU のタイムスタンプで測った（50 回の平均）。CPU は rten 0.26 の `rten::ops::conv`。

| 層（実モデルの形状） | GPU (ms) | rten CPU (ms) | 倍率 | 最大誤差 (abs) |
| --- | --- | --- | --- | --- |
| 1×1 192→192 @120×84 | 0.049 | 2.45 | 50× | 1.1e-6 |
| 1×1 384→384 @60×42 | 0.050 | 1.63 | 33× | 1.6e-6 |
| 1×1 1024→2048 @30×21 | 0.157 | 2.62 | 17× | 2.0e-6 |
| 1×1 3328→1024 @30×21 | 0.318 | 3.98 | 12.5× | 4.4e-6 |
| depthwise 5×5 192ch @120×84 | 0.035 | 2.50 | 72× | 3.6e-7 |
| depthwise 3×3 s2 128ch @240×168 | 0.021 | 1.14 | 54× | 2.4e-7 |
| 測った 11 層の合計 | 0.75 | 18.8 | 25× | — |

- 1×1 Conv は 12〜18 TFLOP/s（FP32 の理論性能の 25〜30%）。誤差は rten 自身の誤差と同程度。
- DX12（FXC）は 1×1 で Vulkan とほぼ同じ、depthwise は約 4 割遅い。
- 最初の素直な書き方では 2〜3 TFLOP/s しか出なかった。ループ変数で添字を取るレジスタ配列があふれるためで、累積を名前付きの vec4 に展開すると 12〜18 TFLOP/s になった。
- `Limits::default()` の範囲（共有メモリ 16 KiB、256 スレッド）に収めたので、性能の低い GPU でも同じシェーダが動くはず（未確認）。
- 見込み（未計測）: det 全体で 960×672 が 40〜60 ms、1280×896 が 65〜100 ms。burn（CubeCL）の 41 / 75 ms に近い。
  不確かなのは、まだ作っていない 3×3 の通常 Conv（implicit GEMM）・ConvTranspose と、多数の小さい層のディスパッチの間接費。

**判断**: 自前の WGSL 推論は、依存を増やさずに burn に近い速さが出る見込み。次の試作は、3×3 の通常 Conv と ConvTranspose を作ること、
そして実際の HGNet のブロック 1 つと DB の出力部をつないで GPU 上で通し、間接費を確かめること。

**試作 2 の結果 (2026-11-26、同じ試験用クレート、DX12)**: 検出（det）の**グラフ全体**を自前の WGSL だけで GPU 上を通した。
実モデルの重み、ディスパッチ 183 回、送信 1 回で、途中の値はすべて GPU 上に置いたまま。入力は合成の文書画像（960×672）。

| 対象 | GPU カーネル | アップロードと読み出しを含む | rten (CPU) | rten との差 |
| --- | --- | --- | --- | --- |
| det 全体 960×672 | 23.6 ms | 25.1 ms | 467 ms | 最大 8.7e-5。0.3 で二値化した後の差は 645,120 画素中 0 |
| DB の出力部（ディスパッチ 4 回） | 3.1 ms | 5.3 ms | 160 ms | 2.7e-5 |
| HGNet の段 1 のブロック（8 回） | 0.9 ms | 3.6 ms | 32.6 ms | 1.2e-5 |
| HGNet の段 3 の軽量ブロック、残差つき（15 回） | 1.5 ms | 4.7 ms | 19.3 ms | 2.5e-5 |

- 作ったもの: 手書きの ONNX（protobuf）読み込み部（長さはすべて検査）。グラフの変換（Identity の別名解決、BN と Bias・Mul の重みへの畳み込み、要素ごとの演算の融合、
  Concat を出力先の配置だけで済ませる、Resize と Concat を次の Conv の読み込みに含める）。
  implicit GEMM の Conv（split-K と DB 出力部の融合つき）、ConvTranspose 2×2、depthwise、Resize、MaxPool。`unsafe` なし、`Limits::default()` の範囲内。
- 3×3 などの通常 Conv は、空間が大きい層で 12〜21 TFLOP/s。1/32・1/16 の解像度の 9×9 Conv は並列度が足りず 0.4〜4 TFLOP/s で、det の時間の半分を占める
  （split-K で詰める見込み。修正後の再計測は実行中）。
- DB 出力部の 64 チャネル・全解像度の中間値（165 MB）は、融合したカーネルで作らずに済ませた。既定の上限（1 バッファ 128 MiB）に収めるために必要な融合。
- 残っていること: 中間バッファの再利用（今は合計 872 MB）、パイプラインのキャッシュと形状ごとの設定の固定（自動調整は全体で 530 秒かかる）、
  入力サイズの段階ごとのシェーダ（今は形状を埋め込んでいる）、1280×896 の実測、DX12 以外（Vulkan、AMD・Intel）での確認、rec の Transformer 部分。
- split-K の再計測（2026-11-26）: 通常 Conv の単体の合計は 11.46 ms → 9.79 ms に下がった。det 全体は、自動調整なしの設定（ディスパッチ 240 回）で
  38.2 ms（rten との差 7.5e-5、二値化後の差 0 画素）と、自動調整した split-K なしの 23.6 ms より遅い。自動調整した版は中間バッファが合計 1.1 GB になり、
  GPU のメモリ不足で組み立てが止まった。中間バッファの再利用（本実装の作業）が入るまで、この計測は完了できない。
- 見込み（未計測）: バッファの再利用と split-K の後で、det 960×672 が約 18〜19 ms、1280×896 が約 32〜36 ms。今の実測の上限は 960×672 で 25.1 ms（split-K なし）。

**rec の Transformer 部分の試作 (2026-11-26、別の試験用クレート、Vulkan と DX12)**: rec の末尾（SVTR）を WGSL で書いた。
構造はグラフから読んだ: 前置 LayerNorm の 2 ブロック（幅 120、8 ヘッド × 15 次元、MLP 120→240→120、swish）、最後の LayerNorm、
Conv の連鎖のあとに 120×18385 の MatMul と Softmax。

- カーネル: LayerNorm、GEMM（swish と残差の融合つき）、注意機構の融合（QKᵀ・online softmax・×V を 1 本で。並べ替えは書き込み先の配置だけで済ませる）、
  Softmax、貪欲 CTC 用の出力層の融合（ロジットを書き出さず、行ごとの最大値・指数和・argmax だけを返す）。
- rten との差: ブロックは最大 1.9e-6、最後の LayerNorm は 1.1e-5、確率は 2.3e-5。融合した出力層の argmax は全行で一致（不一致 0）。
- 時間（Vulkan、GPU タイムスタンプ）: 2 ブロックで 0.11 ms（1 行）／0.11 ms（8 行まとめて）。rten は 0.42 ms／1.06 ms。
  8 行まとめると 1 行あたり約 14 µs まで下がる。出力層の読み出しは、融合しないと 8 行・幅 800 で 59 MB（8.5 ms）、融合すると 6.4 KB（0.06 ms）。
- 分かったこと: この末尾部分は CPU でも rec 全体（1 行 24〜26 ms）の約 5% で、時間の大半は前段の Conv（det と同じ種類の演算）にある。
  よって rec の GPU 化は det の Conv の仕組みをそのまま使える。
- 注意: NVIDIA の Vulkan では、`Limits::default()` の 16 KiB を超える共有メモリ（45 KB）を wgpu が拒否しなかった（DX12 は 32 KB で拒否）。
  本実装では共有メモリの使用量を自分で検査する必要がある。

### P3 UI 統合と Automatic の振り分け (2026-11-27、実装済み)

- **設定画面**: 環境設定に「文字認識」の欄を足し、「GPU で PP-OCRv5 を高速化する」を置いた。既定はオフ。
  - 欄は PP-OCRv5 のモデルがある環境でだけ出す。リリースにはモデルがないので、何も変わらない（§4.5 の決定）。
  - 説明文で、使う GPU の名前（例: `wgpu on NVIDIA GeForce RTX 5080 (Vulkan)`）を示す。GPU がないときは「CPU で読む」と示す。
  - 設定は OCR の他の設定と一緒に保存する（`ocr.gpu`）。認識・書き出し・再読み取りのどのジョブにも渡る。
  - 制御チャネルからは `ocr-gpu true|false` で切り替えられる。
- **デバイス**: アプリは描画用（eframe）の wgpu デバイスを共用する（P1.5 で登録済み）。
  - CLI は `--gpu` のときだけ、自分用のデバイス（`GpuContext::headless`）を作って登録する。
  - ソフトウェアのアダプター、またはアダプターがないときは、stderr に理由を出して CPU で読む。
- **通知**: CPU に切り替えたことは、1 セッションに 1 回だけ知らせる。
  - アプリはトーストで知らせる。ジョブ自身のメッセージが出ている間は待ってから出し、上書きしない。
  - CLI・MCP は、その後の最初のツールの結果の `warnings` で知らせる。
  - 各ページの注記には毎回残す（P1.5 から）。
- **CLI・MCP**: `pdfcraft-cli run … --gpu`、`run --script … --gpu`、`mcp --gpu`、`batch … --gpu`。
  - 既定は CPU。`--gpu` がなければ、どのツールも GPU を求めない（`Automation::with_gpu`）。
  - `mcp install` には `--gpu` を渡す手段をまだ足していない。登録した設定の引数に手で足す。
- **Automatic の振り分け**: 日本語（横・縦）と中国語では、モデルがあれば PP-OCRv5 だけで読む（主にする）。それ以外の言語と、モデルがない環境は従来どおり。
  - 根拠: P3 のベンチ（下の表）。順序なしの CER で比べると、30 件中 28 件で PP-OCRv5 が Windows OCR と同等以上だった。
    劣ったのは 2 件で、ぼかした横書きゴシック（2.6% と 0.5%）と、最も悪い条件のルビ（6.8% と 0.0%）。
    差が大きいのは縦書き（0〜1.4% と 2.8〜5.5%）と表（順序あり 0〜1%、hard-200 だけ 25.3%。Windows OCR は 83〜86%）。
  - 副にする案は採らなかった。弱いページごとに Windows か Tesseract の読み取りがもう 1 回増えるが、精度が上がる見込みがない。
  - ルビや表のときだけ使う案も採らなかった。判定の手間が増え、横書きの本文でも PP-OCRv5 のほうが良いか同じだった。
  - 速度: GPU では Windows OCR に近い（2.23〜2.85 と 3.3〜4.1 pages/s）。CPU では約 1/6 になる。日本語ではこの精度の差が速度に見合うと判断した。速さを優先する人は、エンジンに Windows OCR を指定できる。
  - モデルの読み込みに失敗したとき（ファイルの欠け・不一致）、Automatic は従来のエンジンで読む。PP-OCRv5 を名指ししたジョブは、理由を示して失敗する。

**P3 のベンチ (2026-11-27、合成 30 ページ、検出の辺 1600、RTX 5080・Vulkan)**:
`cargo run --release -p pdfcraft-ocr-gpu --example ocr_bench_gpu -- --engines windows,ppocrv5-wgpu`。
計測の前後で、GPU は別の処理で 90〜98% 使われていた。そのため PP-OCRv5 の速度は P2 の計測（2.85 pages/s）より低い。

| エンジン | 平均 CER (順序あり) | 平均 CER (順序なし) | 行再現率 | 速度 |
| --- | --- | --- | --- | --- |
| Windows OCR | 40.4% | 4.1% | 95.4% | 3.34 pages/s |
| PP-OCRv5 + 自前 WGSL | 23.4% | 0.7% | 99.8% | 2.23 pages/s |

- 順序ありの CER がどちらも高いのはルビのため（ふりがなを本文として読む。正解は親文字だけ）。ルビを除くと、PP-OCRv5 の順序ありは表の hard-200 の 25.3% を除いて 0〜2.6%。

- **実際の画面での確認（2026-11-27、`--control` と `pdfcraft-cli ui`）**:
  - 環境設定で「GPU」を検索すると「文字認識」の欄が出る。切り替えは toggled が false から true に変わり、説明文に `wgpu on NVIDIA GeForce RTX 5080 (Dx12)` と出た。
  - 合成のスキャン 5 ページ（横書き・縦書き・表・ルビ・ノイズ）を、エンジン PP-OCRv5、GPU 有効で認識した。2 回目は 5 ページ・697 語を 3.9 秒で読んだ。CLI の `--gpu` と同じ語数で、CLI の GPU と CPU の結果は全ページで一致した。
  - 1 回目は、モデルの読み込み・自己検査・DX12 のシェーダーの組み立てで 30 秒ほどかかった。その間、進捗は「ページ 1 / 5」のままだった（残りの項目）。
- 残り:
  - 1 回目の準備中に、準備中であることを進捗に出す。
  - AMD・Intel の GPU での確認（確かめたら既定で有効にする、§4.5）。
  - 実スキャンでの Automatic の比較（合成ベンチの偏り、§6）。

### P4 配布の判断

- 170 MB 級のモデル (det 88 MB + rec 85 MB) を同梱するか、初回に取得するか、同梱しないかを決める。
- **ユーザーの明示的な承認が必要。** 承認前は `optional = true` のまま。
- 完了条件: 決定を `docs/` に記録し、`ATTRIBUTION.toml` と配布物が一致する。`release.yml` の変更は別の PR (AGENTS.md の CI 規則)。

## 5.5 実際のカタログでの確認 (2026-11-26)

利用者の製品カタログの PDF から、文字が埋め込まれたページを無作為に 40 枚選び（乱数の種は固定、1 ファイル 1 ページ、日本語が半分以上、200〜6000 字）、
PdfCraft で 300 dpi の画像にして読ませた。埋め込まれた文字を正解とした。数値は集計値だけを残し、文書の内容は残していない。
PDF をそのまま画像にしたもので、スキャンではない。機械は他の作業で混んでいたので、時間は目安。

| エンジン | CER (順序なし) 中央値 / 平均 | 行再現率 中央値 | 時間 中央値 |
| --- | --- | --- | --- |
| Windows OCR (ja) | 9.1〜9.6% / 9.7〜10.1% | 80〜87% | 約 0.3 s |
| PP-OCRv5 + rten、検出 1280 | 3.9〜5.0% / 4.9〜5.0% | 93〜95% | 約 4〜4.5 s |
| PP-OCRv5 + rten、検出 1600 | 3.7% / 4.7% | 94.7% | 1280 とほぼ同じ |

（幅は同じ 40 ページを 2 回測った値。1 回目と 2 回目で前処理を少し変えている。）

- PP-OCRv5 は 40 ページ中 38 ページで Windows OCR より良い。Windows OCR は行を落とす誤りが多く、PP-OCRv5 はわずかに余計な文字を足す（読んだ文字数の比 1.02）。
- 苦手なのは、短い行が多い表や仕様の欄、型番や単位など英数字の多いページ（PP-OCRv5 の最悪でも CER 8〜12%）。
- 検出の入力（長辺）を 1280 から 1600 にすると、表・英数字の多い 21 ページで 9 ページ良くなり 2 ページ悪くなる（中央値 4.0% → 3.2%）。1920 と 2560 は 1600 と同程度で、2560 は約 1.5 倍遅い。
  960 は悪くなる。よってベンチの既定を 1600 に変えた。本実装の既定も 1600 とする。
- 200 dpi に下げると、PP-OCRv5 は CER が約 1 ポイント、行再現率が約 2 ポイント悪くなる。

## 6. リスク

| リスク | 内容 | 対処 |
| --- | --- | --- |
| rten の速度 | 演算子はすべて対応済み（§4 の結果）。CPU では合成ページで約 0.6 pages/s、実カタログで 1 ページ約 4 秒と遅い | P1 でバッチ化と検出の入力サイズを詰め、P2 の GPU 経路で補う |
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
