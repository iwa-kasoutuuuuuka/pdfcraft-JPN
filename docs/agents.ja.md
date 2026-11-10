# AI エージェントから PdfCraft を使う(日本語ガイド)

`pdfcraft-cli` には MCP サーバー(stdio)が入っています。Claude Code・Antigravity・Claude Desktop・Cursor から、PDF の閲覧、検索、OCR、編集、結合、分割ができます。サーバーは、エージェントが起動したときだけ動き、ポートは開きません。

## 共通: ワンコマンド登録

```powershell
pdfcraft-cli mcp install --target claude-code --root C:\Users\you\Documents\pdf
```

- `--target`: `claude-code`(既定)/ `antigravity` / `claude-desktop` / `cursor`(繰り返し指定可)
- `--scope user|project`: 既定は user。project は claude-code(`.mcp.json`)と cursor(`.cursor\mcp.json`)のみ
- `--root DIR`: PDF を読み書きしてよいフォルダ(user では必須。存在するフォルダ)
- `--compact`: ツール一覧を短くする(`tool_search` / `tool_call` 経由で残りを使う)
- `--name`, `--exe`, `--dry-run`(書き込まず内容を表示), `--force`(違う既存エントリを置換)
- 既存ファイルは `<file>.pdfcraft-bak` にバックアップされ、他のサーバー設定はそのまま残ります。JSON が壊れていると何も書かずに中止します(キーの並びは変わることがあります)。
- 解除: `pdfcraft-cli mcp uninstall --target claude-code`

## Claude Code

```powershell
# 1) 自動登録(~/.claude.json)
pdfcraft-cli mcp install --target claude-code --root C:\Users\you\Documents\pdf
# または手動
claude mcp add pdfcraft -- pdfcraft-cli mcp --root C:\Users\you\Documents\pdf
# プロジェクト共有(.mcp.json をカレントに作成)
pdfcraft-cli mcp install --target claude-code --scope project
```

登録後は Claude Code を再起動するか `/mcp` で確認します。

スキル: このリポジトリの `.claude/skills/pdfcraft/SKILL.md` を使えます。他のプロジェクトでも使うには `%USERPROFILE%\.claude\skills\pdfcraft\SKILL.md` にコピーします。

## Antigravity

```powershell
pdfcraft-cli mcp install --target antigravity --root C:\Users\you\Documents\pdf
```

書き込み先は `%USERPROFILE%\.gemini\antigravity\mcp_config.json` です。手動なら次を追記します(JSON ではパスの `\` を `\\` と二重に書きます)。

```json
{
  "mcpServers": {
    "pdfcraft": {
      "command": "C:\\Users\\you\\.cargo\\bin\\pdfcraft-cli.exe",
      "args": ["mcp", "--root", "C:\\Users\\you\\Documents\\pdf"]
    }
  }
}
```

Antigravity の MCP サーバー一覧を更新してください。ワークフロー: `.agent/workflows/pdfcraft.md`(このリポジトリ同梱。他の場所では `.agent\workflows\` にコピー)。

## Claude Desktop

```powershell
pdfcraft-cli mcp install --target claude-desktop --root C:\Users\you\Documents\pdf
```

書き込み先は `%APPDATA%\Claude\claude_desktop_config.json`(macOS は `~/Library/Application Support/Claude/`、Linux は `~/.config/Claude/`)。Claude Desktop を完全に終了して再起動します。

## Cursor

```powershell
pdfcraft-cli mcp install --target cursor --root C:\Users\you\Documents\pdf
pdfcraft-cli mcp install --target cursor --scope project   # <cwd>\.cursor\mcp.json
```

## 日本語 PDF と OCR

スキャン PDF は `ocr_recognize` を `language: "ja"` と `engine: "auto"` で実行します(Windows では Windows OCR を使用)。tesseract を使う場合は `jpn.traineddata`(縦書きは `jpn_vert.traineddata`)が必要です。状態は `ocr_status` で確認できます。

## うまくいかないとき

- `command` が見つからない: `--exe` で絶対パスを指定するか、`pdfcraft-cli` を `PATH` に置いてください。
- 「a different `pdfcraft` entry already exists」: 内容を確認し、置き換えてよければ `--force`。
- 設定ファイルが壊れていると言われた: ファイルを直すか、`.pdfcraft-bak` から戻してください。
