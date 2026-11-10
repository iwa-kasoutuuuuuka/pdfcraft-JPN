# PdfCraft 日本語版 (pdfcraft-JPN)

[PdfCraft](https://github.com/storytold/pdfcraft)(Rust 製のオープンソース PDF リーダー/エディター)を、日本語で使いやすくしたフォークです。英語の README は [README.md](README.md) を参照してください。

- 画面はすべて日本語(未翻訳 0 件)
- 日本語を PDF に**書き込める**(横書き・**縦書き**・**ルビ**・**縦中横**)
- **右綴じ**の見開き表示
- **和暦**の日付
- 日本語 OCR(**Windows 標準 OCR** / Tesseract、縦書き対応)
- **Claude Code・Antigravity** などの AI エージェントから操作(MCP)

---

## 目次

1. [ビルドとインストール](#1-ビルドとインストール)
2. [日本語・縦書きで書き込む](#2-日本語縦書きで書き込む)
3. [ルビ・縦中横](#3-ルビ縦中横)
4. [右綴じ(右から左への見開き)](#4-右綴じ右から左への見開き)
5. [和暦の日付](#5-和暦の日付)
6. [日本語 OCR(スキャン PDF の文字認識)](#6-日本語-ocrスキャン-pdf-の文字認識)
7. [AI エージェント連携(Claude Code / Antigravity)](#7-ai-エージェント連携claude-code--antigravity)
8. [コマンドラインでの使い方](#8-コマンドラインでの使い方)
9. [配布パッケージと自動更新](#9-配布パッケージと自動更新)
10. [困ったとき・既知の制限](#10-困ったとき既知の制限)
11. [このフォークで変えたこと(開発者向け)](#11-このフォークで変えたこと開発者向け)

---

## 1. ビルドとインストール

必要なもの: [Rust](https://rustup.rs/)(stable)と Git。

```sh
git clone https://github.com/iwa-kasoutuuuuuka/pdfcraft-JPN.git
cd pdfcraft-JPN

# 日本語フォント(BIZ UDPゴシック・BIZ UD明朝・しっぽり明朝, OFL)。
# ./craft-fonts に置くと自動で組み込まれます(CI と同じコミット)。
git clone https://github.com/storytold/craft-fonts.git craft-fonts
git -C craft-fonts checkout b076417b5880bd4154bd22f00695b1d28c6bea83

cargo build --release -p pdfcraft -p pdfcraft-cli
```

できあがるもの:

| ファイル | 役割 |
| --- | --- |
| `target/release/pdfcraft`(Windows は `.exe`) | デスクトップアプリ |
| `target/release/pdfcraft-cli` | コマンドライン / MCP サーバー |

- `craft-fonts` が無くてもビルドはできますが、日本語の**書き込み**ができません(表示はできます)。
- 翻訳は `i18n/` に同梱しているので、追加の設定は不要です。OS の言語が判定できない場合も日本語で起動します。

## 2. 日本語・縦書きで書き込む

次の場所で日本語が使えます。字形は PDF に埋め込まれるので、どのパソコンでも同じように表示され、コピー・検索もできます。

- **テキストを追加**(ページに直接書き込む文字)
- テキストボックスのコメント、Fill & Sign の文字
- ヘッダー/フッター、透かし

横書きは自動で折り返し、「、」「。」「」」などが行頭に来ないよう**禁則処理**をします。

### 縦書きにする

1. 編集 ▸ **テキストを追加** で文字を入力します。
2. 右の書式パネルで **縦書き** ボタン(列のアイコン)を押します。

- 1 列目は枠の右端に置かれ、2 列目以降は左へ並びます。
- 枠の高さが 1 列の長さになります。
- 句読点や括弧は縦書き用の字形に変わり、英単語は横に倒して描かれます。

## 3. ルビ・縦中横

### ルビ(ふりがな)

青空文庫と同じ書き方で、ふりがなを付けられます。横書きでは文字の上に、縦書きでは右側に表示されます。

| 入力 | 意味 |
| --- | --- |
| `東京《とうきょう》` | 直前の漢字の並び(東京)にルビ |
| `｜親文字《るび》` | `｜`(または `|`)から `《` までにルビ |

コピー・検索では、ルビを除いた本文(親文字)だけが取り出されます。

### 縦中横

縦書きの中で、**1〜2 桁の数字**と `!!` `!?` `?!` は自動で正立し、横に並べて 1 文字分に収まります(例: 「令和7年12月」の 7 と 12)。3 桁以上の数字は横倒しのままです。

## 4. 右綴じ(右から左への見開き)

- 右綴じの PDF(`/ViewerPreferences /Direction /R2L`)を開くと、見開き表示でページが**右から左**に並びます。
- **←** キーで次のページへ、**→** キーで前のページへ進みます(本と同じ向き)。
- **ページを整理**画面のサムネイルも右から並び、ドラッグでの並べ替えも右綴じの向きになります。
- 手動で切り替えるには **表示 ▸ ページ表示 ▸ 右から左へのページ順**。
- PDF 自体の設定を変えるには **文書のプロパティ ▸ とじ方向 ▸ 右端**。

## 5. 和暦の日付

| 使う場所 | 書き方 | 例 |
| --- | --- | --- |
| 環境設定 ▸ 日付の形式 | `ggEE年m月d日` / `GE.m.d` | 令和7年11月10日 / R7.11.10 |
| ヘッダー/フッター | `<<ggEE年m月d日>>` / `<<GE.m.d>>` | 同上 |

- 元年にも対応しています(令和元年5月1日)。明治以降のすべての元号を扱えます。
- Fill & Sign の日付スタンプにも反映されます。

## 6. 日本語 OCR(スキャン PDF の文字認識)

**スキャンと OCR ▸ このファイル内** を開き、**テキストを認識** を押します。初期設定は「言語: 日本語」「エンジン: 自動」で、Windows ではそのまま Windows OCR が使われます。縦書きだけのページなら、言語に **日本語(縦書き)** を選ぶこともできます(Tesseract 使用時)。

> エンジンを「内蔵」にすると言語の候補が「英語」だけになります(内蔵エンジンは英語専用のため)。日本語を読むときは「自動」か「Windows OCR」を選んでください。

| エンジン | 準備 | 特徴 |
| --- | --- | --- |
| **Windows OCR** | 不要(Windows 10/11 の日本語言語パックに含まれる) | 速い・高精度。縦書きの列も正しい順で読む。縦書きと横書きが混在するページは段落ごとに分けて認識 |
| Tesseract | [Tesseract](https://github.com/UB-Mannheim/tesseract/wiki) と `jpn.traineddata`・`jpn_vert.traineddata`(縦書き用)を tessdata に入れる | Windows 以外でも使える |
| 内蔵 (ocrs) | `cargo xtask models` | 英語のみ |

- 認識結果は**透明なテキスト**としてページに重なり、検索・コピー・選択ができるようになります(見た目は変わりません)。
- 縦書きの行も、上から下へ・右の列から順に読めるように配置されます。
- 認識した日本語の文字間には余計なスペースが入りません。
- Windows OCR は傾きを補正しません。傾いたスキャンは「傾いた文字行をまっすぐにします」にチェックを入れてください。
- 認識できる言語は Windows の言語設定に入っている OCR 用の言語だけです。日本語が無い場合は **設定 ▸ 時刻と言語 ▸ 言語と地域** で日本語を追加します。

## 7. AI エージェント連携(Claude Code / Antigravity)

`pdfcraft-cli` は MCP サーバーとして動き、AI エージェントから PDF の閲覧・検索・OCR・編集・結合・分割などができます。サーバーは**エージェントが起動したときだけ**動き、ネットワークのポートは開きません。

### 登録(1 コマンド)

```sh
# Claude Code に登録(--root のフォルダの中だけを読み書きできる)
pdfcraft-cli mcp install --target claude-code --root C:\Users\me\Documents\pdf

# Antigravity に登録
pdfcraft-cli mcp install --target antigravity --root C:\Users\me\Documents\pdf

# Claude Desktop / Cursor にも登録できる
pdfcraft-cli mcp install --target claude-desktop --target cursor --root C:\Users\me\Documents\pdf

# 書き込む前に内容だけ確認
pdfcraft-cli mcp install --target antigravity --root C:\Users\me\Documents\pdf --dry-run

# 登録を外す
pdfcraft-cli mcp uninstall --target claude-code
```

主なオプション:

| オプション | 意味 |
| --- | --- |
| `--scope project` | 今のフォルダの `.mcp.json`(Claude Code)や `.cursor/mcp.json` に登録 |
| `--compact` | エージェントに見せるツールを絞って、トークンを節約 |
| `--force` | 既存の別設定を上書き。ドライブ直下やホームフォルダを `--root` にする場合にも必要 |

安全のための挙動:

- 既存の設定ファイルは `<ファイル名>.pdfcraft-bak` にバックアップしてから、`pdfcraft` の項目だけを追加・更新します。
- ドライブ直下やホームフォルダを `--root` にすると、`--force` を付けない限り拒否します。
- Claude Code に登録するときは、Claude Code を一度閉じてから実行してください。登録後は `/mcp` で確認できます。

### スキル / ワークフロー

| エージェント | ファイル | 内容 |
| --- | --- | --- |
| Claude Code | `.claude/skills/pdfcraft/SKILL.md` | PDF を扱う手順、日本語 OCR、縦書きでの書き込み、安全上の注意 |
| Antigravity | `.agent/workflows/pdfcraft.md` | 同上 |

このリポジトリを開けば自動で使われます。他のプロジェクトで使う場合はコピーしてください。詳しくは [docs/agents.ja.md](docs/agents.ja.md) を参照してください。

### エージェントへの頼み方(例)

- 「`契約書.pdf` の 3 ページ目に『令和7年11月10日 承認』と縦書きで書き込んで、別名で保存して」
- 「スキャンした `議事録.pdf` を日本語で OCR して、検索できる PDF にして」
- 「`a.pdf` と `b.pdf` を結合して、2〜4 ページを抜き出して」

## 8. コマンドラインでの使い方

```sh
pdfcraft-cli text 文書.pdf                     # テキストを取り出す(縦書きは列の順)
pdfcraft-cli render 文書.pdf --page 1 --out p1.png
pdfcraft-cli tools                             # 使えるツールと引数(JSON Schema)の一覧
pdfcraft-cli run --script steps.json --root .  # 複数の手順をまとめて実行
```

`steps.json` の例(縦書きでページに書き込む):

```json
[
  {"tool": "doc_create", "args": {"from": "blank", "pages": 1, "width": 420, "height": 595}},
  {"tool": "page_add_text", "args": {"doc": 1, "page": 1, "rect": [40, 40, 380, 300], "size": 18,
    "vertical": true, "text": "令和7年12月、東京《とうきょう》へ行く。"}},
  {"tool": "doc_save", "args": {"doc": 1, "path": "out.pdf"}}
]
```

- `page_add_text` / `content_update` の `vertical` で縦書きを指定します。
- `ocr_recognize` の `engine: "windows"` と `language: "ja"` で日本語 OCR を実行します。

## 9. 配布パッケージと自動更新

### Windows の配布パッケージを作る

[WiX v5](https://wixtoolset.org/)(`dotnet tool install --global wix --version 5.0.2`)を入れてから、PowerShell 7 で次を実行します。フォントのライセンス表記を同梱するため、`CRAFT_FONTS_DIR` を指定します。

```powershell
$env:CRAFT_FONTS_DIR = "$PWD\craft-fonts"
pwsh packaging/windows/package.ps1 -Arch x64
```

`pdfcraft-<version>-windows-x64.msi`(インストーラー)と `…-portable.zip`(展開して使う版)ができます。大きさは元の PdfCraft とほぼ同じ、インストーラーで約 95 MB です。大部分は埋め込みフォント(日本語・各国語で約 44 MB、簡体字中国語で約 13 MB)です。

### 自動更新について

- 更新の確認先は**このフォーク(iwa-kasoutuuuuuka/pdfcraft-JPN)の GitHub Releases** です。元の英語版の更新が案内されることはありません。
- リリースが無いあいだは、更新の通知は出ません。
- Windows の自動インストールは、**コード署名された**ビルド同士でのみ行われます。署名していない自前ビルドでは、通知が出ても自動では入れ替わらず(安全のため)、手動でダウンロードして入れ直します。
- ダウンロードしたファイルは、リリースに記載された SHA-256 と照合されます。

## 10. 困ったとき・既知の制限

| 症状 | 原因と対処 |
| --- | --- |
| 日本語を書き込もうとすると「Japanese text needs a Japanese font」と出る | `craft-fonts` を入れずにビルドしています。[1. ビルド](#1-ビルドとインストール)の手順で `./craft-fonts` を置いてビルドし直してください |
| OCR の「テキストを認識」が押せない | エンジンが「内蔵」になっていて、モデルがありません。エンジンを「自動」か「Windows OCR」にしてください |
| OCR の言語に日本語が出ない | 選んでいるエンジンが日本語に対応していません(内蔵は英語のみ)。Tesseract の場合は `jpn.traineddata` を tessdata に入れてください |
| Claude Code に `pdfcraft` が出てこない | Claude Code を再起動して `/mcp` で確認してください。`--dry-run` で書き込む内容を確認できます |
| 中国語(簡体字)やヒンディー語などが書き込めない | 書き込みに使えるのは日本語フォントにある文字(日本語・英数字・ギリシャ文字・キリル文字・記号の多く)とタイ語・アラビア語・ヘブライ語です |

既知の制限:

- 縦書きのテキストを**入力・編集している間**は、入力欄が横書きで表示されます。確定すると縦書きで描かれます。
- ルビが親文字より長い場合は、はみ出して描かれます(自動で詰めません)。
- 縦中横は 1〜2 桁の数字と `!!` `!?` `?!` だけが対象です。
- macOS 用のアイコン(.icns)は Windows で作ったもので、macOS 上では確認していません。

## 11. このフォークで変えたこと(開発者向け)

| 項目 | 内容 |
| --- | --- |
| 翻訳 | `i18n/`(craft-i18n から取り込み)。`crates/ui-egui/build.rs` が `CRAFT_I18N_DIR` 未指定時に使用。未翻訳の洗い出しは `PDFCRAFT_I18N_DUMP=missing.tsv cargo test -p pdfcraft-ui-egui --lib -- --ignored dump_missing_japanese` |
| 日本語フォント | `./craft-fonts` を `crates/fonts/build.rs` が自動で使用(マニフェストはチェックアウト内のファイルに限定) |
| 日本語の描画 | `crates/fonts/src/shaped_text.rs`(`is_japanese`)、`shaping.rs`(`shape_japanese` / `shape_japanese_vertical`)。Type 3 フォント + ToUnicode で埋め込み |
| 縦書き・ルビ・縦中横 | `crates/edit/src/added/vertical.rs`、`crates/edit/src/added/layout/ruby.rs` |
| 縦書きの抽出 | `crates/render/src/text/device.rs`(`merge_column`) |
| 右綴じ | `crates/ui-egui/src/canvas/zoom.rs`・`navigation.rs`・`organize.rs` |
| 和暦 | `crates/edit/src/era.rs`、`crates/engine/src/dates.rs` |
| OCR | `crates/ocr/src/windows_ocr.rs`・`blocks.rs`・`tesseract.rs`、`crates/engine/src/ocr/layer.rs` |
| MCP 登録 | `apps/pdfcraft-cli/src/mcp_install.rs` |

テスト: `cargo test -p pdfcraft-fonts -p pdfcraft-edit -p pdfcraft-render -p pdfcraft-ocr -p pdfcraft-engine -p pdfcraft-automation -p pdfcraft-cli -p pdfcraft-ui-egui`

## ライセンス

元の PdfCraft と同じく **MIT または Apache-2.0** です。

- 同梱の翻訳(`i18n/`)は [craft-i18n](https://github.com/storytold/craft-i18n)(MIT / Apache-2.0)から取り込んでいます。
- 日本語フォントは [craft-fonts](https://github.com/storytold/craft-fonts) 経由の OFL-1.1 です。

元のプロジェクトの作者と貢献者の皆さんに感謝します。
