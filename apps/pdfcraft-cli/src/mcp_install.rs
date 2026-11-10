//! `pdfcraft-cli mcp install` / `mcp uninstall`: write (or remove) the MCP server entry in the
//! config file of an agent (Claude Code, Antigravity, Claude Desktop, Cursor).
//!
//! The logic is split so it can be tested without touching a real home directory: [`config_path_with`]
//! takes the environment as values, [`config_path`] is the only function that reads it, and the merge
//! functions work on text. A config file is only ever rewritten when it parses; every other key and
//! server in it is kept (serde_json without `preserve_order` sorts object keys, so key order may change);
//! the previous file is copied to `<file>.pdfcraft-bak` first and the new one is moved into place from a
//! temp file in the same folder.

use std::path::{Path, PathBuf};

use serde_json::{Map, Value, json};

use super::{CliError, stdout_line};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Target {
    ClaudeCode,
    Antigravity,
    ClaudeDesktop,
    Cursor,
}

impl Target {
    fn parse(s: &str) -> Result<Self, String> {
        match s {
            "claude-code" => Ok(Self::ClaudeCode),
            "antigravity" => Ok(Self::Antigravity),
            "claude-desktop" => Ok(Self::ClaudeDesktop),
            "cursor" => Ok(Self::Cursor),
            other => Err(format!("unknown --target `{other}` (claude-code, antigravity, claude-desktop, cursor)")),
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::ClaudeCode => "Claude Code",
            Self::Antigravity => "Antigravity",
            Self::ClaudeDesktop => "Claude Desktop",
            Self::Cursor => "Cursor",
        }
    }

    fn restart_hint(self) -> (&'static str, &'static str) {
        match self {
            Self::ClaudeCode => ("Claude Code を再起動するか `/mcp` で確認してください", "Restart Claude Code or check with `/mcp`"),
            Self::Antigravity => ("Antigravity の MCP サーバー一覧を更新してください", "Refresh the MCP server list in Antigravity"),
            Self::ClaudeDesktop => ("Claude Desktop を完全に終了して再起動してください", "Quit and restart Claude Desktop"),
            Self::Cursor => ("Cursor を再起動するか設定の MCP を確認してください", "Restart Cursor or check Settings > MCP"),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Scope {
    User,
    Project,
}

/// Where a target keeps its config. Pure: the environment is passed in.
pub(crate) fn config_path_with(
    target: Target,
    scope: Scope,
    home: Option<&Path>,
    appdata: Option<&Path>,
    cwd: &Path,
    windows: bool,
    macos: bool,
) -> Result<PathBuf, String> {
    let need_home = || home.map(Path::to_path_buf).ok_or_else(|| "cannot find the home folder (USERPROFILE / HOME is not set)".to_string());
    match (target, scope) {
        (Target::ClaudeCode, Scope::Project) => Ok(cwd.join(".mcp.json")),
        (Target::Cursor, Scope::Project) => Ok(cwd.join(".cursor").join("mcp.json")),
        (_, Scope::Project) => Err(format!("--scope project is only valid for claude-code and cursor, not {}", target.label())),
        (Target::ClaudeCode, Scope::User) => Ok(need_home()?.join(".claude.json")),
        (Target::Antigravity, Scope::User) => Ok(need_home()?.join(".gemini").join("antigravity").join("mcp_config.json")),
        (Target::Cursor, Scope::User) => Ok(need_home()?.join(".cursor").join("mcp.json")),
        (Target::ClaudeDesktop, Scope::User) => {
            let file = "claude_desktop_config.json";
            if windows {
                let base = appdata.map(Path::to_path_buf).ok_or("cannot find %APPDATA%")?;
                Ok(base.join("Claude").join(file))
            } else if macos {
                Ok(need_home()?.join("Library").join("Application Support").join("Claude").join(file))
            } else {
                Ok(need_home()?.join(".config").join("Claude").join(file))
            }
        }
    }
}

/// The one place that reads the environment for config locations.
pub(crate) fn config_path(target: Target, scope: Scope, cwd: &Path) -> Result<PathBuf, String> {
    let env_path = |k: &str| std::env::var_os(k).filter(|v| !v.is_empty()).map(PathBuf::from);
    let home = env_path("USERPROFILE").or_else(|| env_path("HOME"));
    let appdata = env_path("APPDATA");
    config_path_with(target, scope, home.as_deref(), appdata.as_deref(), cwd, cfg!(windows), cfg!(target_os = "macos"))
}

pub(crate) fn build_entry(target: Target, exe: &str, root: &str, compact: bool) -> Value {
    let mut args = vec![json!("mcp"), json!("--root"), json!(root)];
    if compact {
        args.push(json!("--compact"));
    }
    let mut entry = Map::new();
    if target == Target::ClaudeCode {
        entry.insert("type".into(), json!("stdio"));
    }
    entry.insert("command".into(), json!(exe));
    entry.insert("args".into(), Value::Array(args));
    Value::Object(entry)
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) enum Outcome {
    Added,
    Replaced,
    Unchanged,
}

fn parse_config(existing: Option<&str>) -> Result<Map<String, Value>, String> {
    match existing {
        None => Ok(Map::new()),
        Some(text) if text.trim().is_empty() => Ok(Map::new()),
        Some(text) => match serde_json::from_str::<Value>(text) {
            Ok(Value::Object(m)) => Ok(m),
            Ok(_) => Err("the config file is not a JSON object; nothing was changed".to_string()),
            Err(e) => Err(format!("the config file is not valid JSON ({e}); nothing was changed")),
        },
    }
}

/// Merge `entry` as server `name` into the config text. Never drops other keys or servers.
pub(crate) fn merge(existing: Option<&str>, name: &str, entry: &Value, force: bool) -> Result<(Outcome, Value), String> {
    let mut doc = parse_config(existing)?;
    let servers = doc.entry("mcpServers").or_insert_with(|| Value::Object(Map::new()));
    let Value::Object(servers) = servers else {
        return Err("`mcpServers` in the config file is not an object; nothing was changed".to_string());
    };
    let outcome = match servers.get(name) {
        Some(current) if current == entry => Outcome::Unchanged,
        Some(_) if !force => {
            return Err(format!("a different `{name}` entry already exists; pass --force to replace it, or choose another --name"));
        }
        Some(_) => Outcome::Replaced,
        None => Outcome::Added,
    };
    if outcome != Outcome::Unchanged {
        servers.insert(name.to_string(), entry.clone());
    }
    Ok((outcome, Value::Object(doc)))
}

/// Remove server `name`. `None` when there was nothing to remove.
pub(crate) fn remove(existing: Option<&str>, name: &str) -> Result<Option<Value>, String> {
    let mut doc = parse_config(existing)?;
    let removed = match doc.get_mut("mcpServers") {
        Some(Value::Object(servers)) => servers.remove(name).is_some(),
        Some(_) => return Err("`mcpServers` in the config file is not an object; nothing was changed".to_string()),
        None => false,
    };
    Ok(removed.then_some(Value::Object(doc)))
}

/// Largest config file we are willing to read.
const MAX_CONFIG_BYTES: u64 = 64 * 1024 * 1024;

/// Read a config file: `None` when missing, error when larger than 64 MiB; a leading UTF-8 BOM is dropped.
pub(crate) fn read_config(path: &Path) -> Result<Option<String>, String> {
    use std::io::Read;
    let file = match std::fs::File::open(path) {
        Ok(f) => f,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(format!("cannot read {}: {e}", path.display())),
    };
    let mut bytes = Vec::new();
    file.take(MAX_CONFIG_BYTES + 1).read_to_end(&mut bytes).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    if bytes.len() as u64 > MAX_CONFIG_BYTES {
        return Err(format!("{} is larger than {} MiB; refusing to read it", path.display(), MAX_CONFIG_BYTES / 1024 / 1024));
    }
    let text = String::from_utf8(bytes).map_err(|_| format!("{} is not valid UTF-8", path.display()))?;
    Ok(Some(text.strip_prefix('\u{feff}').map(str::to_string).unwrap_or(text)))
}

fn unix_secs() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs())
}

/// Create `<file>.pdfcraft-bak` (or `.pdfcraft-bak.<unix-time>[.n]` when one exists) holding a copy of `path`.
fn make_backup(path: &Path, dir: &Path, file_name: &str) -> Result<(), String> {
    let first = dir.join(format!("{file_name}.pdfcraft-bak"));
    let ts = unix_secs();
    let candidates =
        std::iter::once(first).chain((0..100).map(|n| {
            if n == 0 { dir.join(format!("{file_name}.pdfcraft-bak.{ts}")) } else { dir.join(format!("{file_name}.pdfcraft-bak.{ts}.{n}")) }
        }));
    for bak in candidates {
        let mut out = match std::fs::OpenOptions::new().write(true).create_new(true).open(&bak) {
            Ok(f) => f,
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(e) => return Err(format!("cannot write the backup {}: {e}", bak.display())),
        };
        let res = (|| -> std::io::Result<()> {
            let mut src = std::fs::File::open(path)?;
            std::io::copy(&mut src, &mut out)?;
            out.sync_all()?;
            if let Ok(m) = src.metadata() {
                let _ = std::fs::set_permissions(&bak, m.permissions());
            }
            Ok(())
        })();
        return res.map_err(|e| format!("cannot write the backup {}: {e}", bak.display()));
    }
    Err(format!("cannot find a free backup name next to {}", path.display()))
}

/// Create a fresh temp file next to the target (never follows a symlink or clobbers a file); 0600 on Unix.
fn create_temp(dir: &Path, file_name: &str) -> Result<(std::fs::File, PathBuf), std::io::Error> {
    let mut last = None;
    for n in 0..32u32 {
        let tmp = dir.join(format!(".{file_name}.pdfcraft-tmp{}.{n}", std::process::id()));
        let mut oo = std::fs::OpenOptions::new();
        oo.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            oo.mode(0o600);
        }
        match oo.open(&tmp) {
            Ok(f) => return Ok((f, tmp)),
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => last = Some(e),
            Err(e) => return Err(e),
        }
    }
    Err(last.unwrap_or_else(|| std::io::Error::other("no free temp file name")))
}

/// Copy an existing file to `<file>.pdfcraft-bak`, then write `content` through a temp file + rename.
/// A symlinked `path` is resolved first, so the link itself is never replaced. The temp file takes
/// over the permissions of the file it replaces (0600 for a new file on Unix).
pub(crate) fn write_with_backup(path: &Path, content: &str) -> Result<(), String> {
    use std::io::Write;
    let is_link = std::fs::symlink_metadata(path).is_ok_and(|m| m.file_type().is_symlink());
    let resolved;
    let path = if is_link {
        resolved = std::fs::canonicalize(path).map_err(|e| format!("cannot resolve the symlink {}: {e}", path.display()))?;
        resolved.as_path()
    } else {
        path
    };
    let err = |what: &str, e: std::io::Error| format!("{what} {}: {e}", path.display());
    let dir = path.parent().filter(|p| !p.as_os_str().is_empty()).unwrap_or_else(|| Path::new("."));
    std::fs::create_dir_all(dir).map_err(|e| err("cannot create the folder for", e))?;
    let file_name = path.file_name().map_or_else(|| "config".to_string(), |n| n.to_string_lossy().into_owned());
    let existing_perms = std::fs::metadata(path).ok().map(|m| m.permissions());
    if existing_perms.is_some() {
        make_backup(path, dir, &file_name)?;
    }
    let (mut file, tmp) = create_temp(dir, &file_name).map_err(|e| err("cannot write", e))?;
    let written = (|| -> std::io::Result<()> {
        file.write_all(content.as_bytes())?;
        if let Some(p) = existing_perms {
            file.set_permissions(p)?;
        }
        file.sync_all()
    })();
    drop(file);
    if let Err(e) = written {
        let _ = std::fs::remove_file(&tmp);
        return Err(err("cannot write", e));
    }
    if let Err(e) = std::fs::rename(&tmp, path) {
        let _ = std::fs::remove_file(&tmp);
        return Err(err("cannot replace", e));
    }
    Ok(())
}

/// `\\?\C:\x` -> `C:\x`, `\\?\UNC\server\share` -> `\\server\share`; any other verbatim path is kept.
pub(crate) fn strip_verbatim(path: PathBuf) -> PathBuf {
    let stripped = {
        let s = path.to_string_lossy();
        s.strip_prefix(r"\\?\").and_then(|rest| {
            if let Some(unc) = rest.strip_prefix("UNC\\") {
                return Some(PathBuf::from(format!(r"\\{unc}")));
            }
            let b = rest.as_bytes();
            (b.len() >= 3 && b[0].is_ascii_alphabetic() && b[1] == b':' && b[2] == b'\\').then(|| PathBuf::from(rest))
        })
    };
    stripped.unwrap_or(path)
}

/// Refuse a root that would expose everything (a filesystem/drive root or the home folder) unless forced.
pub(crate) fn check_root(root: &Path, home: Option<&Path>, force: bool) -> Result<(), String> {
    if force {
        return Ok(());
    }
    let is_home = home.is_some_and(|h| {
        let h = strip_verbatim(std::fs::canonicalize(h).unwrap_or_else(|_| h.to_path_buf()));
        let (a, b) = (h.to_string_lossy(), root.to_string_lossy());
        let (a, b) = (a.trim_end_matches(['/', '\\']), b.trim_end_matches(['/', '\\']));
        if cfg!(windows) { a.eq_ignore_ascii_case(b) } else { a == b }
    });
    if root.parent().is_none() || is_home {
        return Err(format!(
            "--root `{}` is a filesystem root or your home folder; the agent could then read and write everything under it. Choose a narrower folder, or pass --force",
            root.display()
        ));
    }
    Ok(())
}

/// In project scope `.mcp.json` / `.cursor` must not be symlinks (a repo could point them at a private file).
pub(crate) fn check_project_links(cwd: &Path) -> Result<(), String> {
    for name in [".mcp.json", ".cursor"] {
        let p = cwd.join(name);
        if std::fs::symlink_metadata(&p).is_ok_and(|m| m.file_type().is_symlink()) {
            return Err(format!("{} is a symlink; refusing to use it for --scope project", p.display()));
        }
    }
    Ok(())
}

fn pretty(v: &Value) -> String {
    let mut s = serde_json::to_string_pretty(v).unwrap_or_default();
    s.push('\n');
    s
}

// ---- command line ------------------------------------------------------------------------------

#[derive(Debug)]
struct Options {
    targets: Vec<Target>,
    scope: Scope,
    root: Option<String>,
    compact: bool,
    name: String,
    exe: Option<String>,
    dry_run: bool,
    force: bool,
}

fn parse_options(command: &str, args: &[String]) -> Result<Options, String> {
    let mut o = Options {
        targets: Vec::new(),
        scope: Scope::User,
        root: None,
        compact: false,
        name: "pdfcraft".into(),
        exe: None,
        dry_run: false,
        force: false,
    };
    let mut it = args.iter();
    while let Some(arg) = it.next() {
        let mut value = || match it.next() {
            Some(v) if !v.is_empty() && !v.starts_with("--") => Ok(v.clone()),
            _ => Err(format!("mcp {command}: {arg} needs a value")),
        };
        match arg.as_str() {
            "--target" => {
                let t = Target::parse(&value()?)?;
                if !o.targets.contains(&t) {
                    o.targets.push(t);
                }
            }
            "--scope" => {
                o.scope = match value()?.as_str() {
                    "user" => Scope::User,
                    "project" => Scope::Project,
                    other => return Err(format!("unknown --scope `{other}` (user, project)")),
                }
            }
            "--root" => {
                let v = value()?;
                if o.root.is_some() {
                    return Err(format!("mcp {command}: --root may only be supplied once"));
                }
                o.root = Some(v);
            }
            "--name" => o.name = value()?,
            "--exe" => o.exe = Some(value()?),
            "--compact" => o.compact = true,
            "--dry-run" => o.dry_run = true,
            "--force" => o.force = true,
            other => return Err(format!("mcp {command}: unknown option {other}")),
        }
    }
    if o.targets.is_empty() {
        o.targets.push(Target::ClaudeCode);
    }
    Ok(o)
}

fn canonical_dir(p: &str, what: &str) -> Result<String, String> {
    let c = std::fs::canonicalize(p).map_err(|e| format!("{what} `{p}` does not exist or cannot be opened: {e}"))?;
    if !c.is_dir() {
        return Err(format!("{what} `{p}` is not a folder"));
    }
    Ok(strip_verbatim(c).to_string_lossy().into_owned())
}

fn user_scope_note(target: Target, scope: Scope) -> Result<(), CliError> {
    if target == Target::ClaudeCode && scope == Scope::User {
        stdout_line(format_args!(
            "  注意: Claude Code が起動中だと設定を上書きすることがあります。先に終了するか、反映されていなければ再実行してください。キーの並びは変わります(バックアップ .pdfcraft-bak あり)"
        ))?;
        stdout_line(format_args!(
            "  Note: close Claude Code first, or re-run if it overwrote the change. Keys are re-sorted (a .pdfcraft-bak backup is kept)"
        ))?;
    }
    Ok(())
}

/// Entry point for `mcp install` (`install == true`) and `mcp uninstall`.
pub(crate) fn run(install: bool, args: &[String]) -> Result<(), CliError> {
    let command = if install { "install" } else { "uninstall" };
    let o = parse_options(command, args)?;
    let cwd = std::env::current_dir().map_err(|e| format!("cannot read the current folder: {e}"))?;
    let mut entry_for = None;
    if install {
        let root = match (&o.root, o.scope) {
            (Some(r), _) => canonical_dir(r, "--root")?,
            (None, Scope::Project) => canonical_dir(&cwd.to_string_lossy(), "the current folder")?,
            (None, Scope::User) => return Err("mcp install: --root DIR is required for --scope user".into()),
        };
        let home = ["USERPROFILE", "HOME"].iter().find_map(|k| std::env::var_os(k).filter(|v| !v.is_empty())).map(PathBuf::from);
        check_root(Path::new(&root), home.as_deref(), o.force)?;
        let exe = match &o.exe {
            Some(e) => e.clone(),
            None => {
                let e = std::env::current_exe().map_err(|e| format!("cannot find this executable (use --exe): {e}"))?;
                let e = std::fs::canonicalize(&e).unwrap_or(e);
                strip_verbatim(e).to_string_lossy().into_owned()
            }
        };
        entry_for = Some((exe, root));
    }
    for &target in &o.targets {
        let path = config_path(target, o.scope, &cwd)?;
        if o.scope == Scope::Project {
            check_project_links(&cwd)?;
        }
        let existing = read_config(&path)?;
        let label = target.label();
        let at = |m: String| format!("{label} ({}): {m}", path.display());
        let (jp, en) = target.restart_hint();
        if let Some((exe, root)) = &entry_for {
            let entry = build_entry(target, exe, root, o.compact);
            let (outcome, doc) = merge(existing.as_deref(), &o.name, &entry, o.force).map_err(at)?;
            if o.dry_run {
                stdout_line(format_args!("[dry-run] {label}: {}", path.display()))?;
                let shown = json!({ "mcpServers": { o.name.clone(): entry } });
                stdout_line(format_args!("{}", pretty(&shown).trim_end()))?;
                continue;
            }
            if outcome == Outcome::Unchanged {
                stdout_line(format_args!("{label}: already configured / 設定済みです ({})", path.display()))?;
                continue;
            }
            write_with_backup(&path, &pretty(&doc)).map_err(at)?;
            stdout_line(format_args!("{label}: 書き込みました / wrote {}", path.display()))?;
            stdout_line(format_args!("  {jp}"))?;
            stdout_line(format_args!("  {en}"))?;
            user_scope_note(target, o.scope)?;
        } else {
            match remove(existing.as_deref(), &o.name).map_err(at)? {
                None => stdout_line(format_args!("{label}: `{}` はありません / not configured ({})", o.name, path.display()))?,
                Some(doc) => {
                    if o.dry_run {
                        stdout_line(format_args!("[dry-run] {label}: would remove `{}` from {}", o.name, path.display()))?;
                        continue;
                    }
                    write_with_backup(&path, &pretty(&doc)).map_err(at)?;
                    stdout_line(format_args!("{label}: `{}` を削除しました / removed from {}", o.name, path.display()))?;
                    stdout_line(format_args!("  {jp}"))?;
                    stdout_line(format_args!("  {en}"))?;
                    user_scope_note(target, o.scope)?;
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;

    fn entry() -> Value {
        build_entry(Target::Antigravity, "C:\\bin\\pdfcraft-cli.exe", "C:\\work", false)
    }

    #[test]
    fn entry_shapes() {
        let e = build_entry(Target::ClaudeCode, "/x", "/r", true);
        assert_eq!(e, json!({"type":"stdio","command":"/x","args":["mcp","--root","/r","--compact"]}));
        assert_eq!(entry(), json!({"command":"C:\\bin\\pdfcraft-cli.exe","args":["mcp","--root","C:\\work"]}));
    }

    #[test]
    fn paths() {
        let home = Path::new("/h");
        let cwd = Path::new("/c");
        let p = |t, s, w, m| config_path_with(t, s, Some(home), Some(Path::new("/appdata")), cwd, w, m);
        assert_eq!(p(Target::ClaudeCode, Scope::User, false, false).unwrap(), home.join(".claude.json"));
        assert_eq!(p(Target::ClaudeCode, Scope::Project, false, false).unwrap(), cwd.join(".mcp.json"));
        assert_eq!(p(Target::Cursor, Scope::Project, false, false).unwrap(), cwd.join(".cursor").join("mcp.json"));
        assert!(p(Target::Antigravity, Scope::Project, false, false).is_err());
        assert!(p(Target::Antigravity, Scope::User, false, false).unwrap().ends_with("mcp_config.json"));
        assert_eq!(
            p(Target::ClaudeDesktop, Scope::User, true, false).unwrap(),
            Path::new("/appdata").join("Claude").join("claude_desktop_config.json")
        );
        assert!(p(Target::ClaudeDesktop, Scope::User, false, true).unwrap().to_string_lossy().contains("Application Support"));
        assert!(p(Target::ClaudeDesktop, Scope::User, false, false).unwrap().to_string_lossy().contains(".config"));
    }

    #[test]
    fn merge_preserves_other_entries() {
        let existing = r#"{"theme":"dark","mcpServers":{"other":{"command":"x"}}}"#;
        let (o, doc) = merge(Some(existing), "pdfcraft", &entry(), false).unwrap();
        assert_eq!(o, Outcome::Added);
        assert_eq!(doc["theme"], "dark");
        assert_eq!(doc["mcpServers"]["other"], json!({"command":"x"}));
        assert_eq!(doc["mcpServers"]["pdfcraft"], entry());
    }

    #[test]
    fn merge_missing_file_and_idempotence() {
        let (o, doc) = merge(None, "pdfcraft", &entry(), false).unwrap();
        assert_eq!(o, Outcome::Added);
        let text = pretty(&doc);
        let (o2, doc2) = merge(Some(&text), "pdfcraft", &entry(), false).unwrap();
        assert_eq!(o2, Outcome::Unchanged);
        assert_eq!(doc, doc2);
    }

    #[test]
    fn differing_entry_needs_force() {
        let text = pretty(&merge(None, "pdfcraft", &entry(), false).unwrap().1);
        let other = build_entry(Target::Antigravity, "/other", "/r", false);
        assert!(merge(Some(&text), "pdfcraft", &other, false).is_err());
        let (o, doc) = merge(Some(&text), "pdfcraft", &other, true).unwrap();
        assert_eq!(o, Outcome::Replaced);
        assert_eq!(doc["mcpServers"]["pdfcraft"], other);
    }

    #[test]
    fn invalid_json_aborts() {
        assert!(merge(Some("{ not json"), "pdfcraft", &entry(), true).is_err());
        assert!(merge(Some("[1]"), "pdfcraft", &entry(), true).is_err());
        assert!(merge(Some(r#"{"mcpServers":3}"#), "pdfcraft", &entry(), true).is_err());
    }

    #[test]
    fn write_creates_dirs_and_backup() {
        let dir = std::env::temp_dir().join(format!("pdfcraft-install-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let path = dir.join("a").join("mcp.json");
        write_with_backup(&path, "{\"v\":1}\n").unwrap();
        assert!(!dir.join("a").join("mcp.json.pdfcraft-bak").exists());
        write_with_backup(&path, "{\"v\":2}\n").unwrap();
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "{\"v\":2}\n");
        assert_eq!(std::fs::read_to_string(dir.join("a").join("mcp.json.pdfcraft-bak")).unwrap(), "{\"v\":1}\n");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn uninstall_removes_only_named_entry() {
        let existing = r#"{"mcpServers":{"other":{"command":"x"},"pdfcraft":{"command":"y"}},"k":1}"#;
        let doc = remove(Some(existing), "pdfcraft").unwrap().unwrap();
        assert_eq!(doc, json!({"mcpServers":{"other":{"command":"x"}},"k":1}));
        assert!(remove(Some(existing), "nope").unwrap().is_none());
        assert!(remove(None, "pdfcraft").unwrap().is_none());
        assert!(remove(Some("oops"), "pdfcraft").is_err());
    }

    #[test]
    fn verbatim_prefix_is_stripped() {
        assert_eq!(strip_verbatim(PathBuf::from(r"\\?\C:\a\b")), PathBuf::from(r"C:\a\b"));
        assert_eq!(strip_verbatim(PathBuf::from("/a/b")), PathBuf::from("/a/b"));
        assert_eq!(strip_verbatim(PathBuf::from(r"\\?\UNC\srv\share\x")), PathBuf::from(r"\\srv\share\x"));
        assert_eq!(strip_verbatim(PathBuf::from(r"\\?\Volume{1234}\x")), PathBuf::from(r"\\?\Volume{1234}\x"));
    }

    fn tmpdir(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("pdfcraft-install-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn backup_is_not_overwritten() {
        let dir = tmpdir("bak");
        let path = dir.join("c.json");
        std::fs::write(&path, "1").unwrap();
        write_with_backup(&path, "2").unwrap();
        write_with_backup(&path, "3").unwrap();
        assert_eq!(std::fs::read_to_string(dir.join("c.json.pdfcraft-bak")).unwrap(), "1");
        let extra: Vec<_> = std::fs::read_dir(&dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_name().to_string_lossy().starts_with("c.json.pdfcraft-bak."))
            .collect();
        assert_eq!(extra.len(), 1);
        assert_eq!(std::fs::read_to_string(extra[0].path()).unwrap(), "2");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[cfg(unix)]
    #[test]
    fn permissions_kept_and_new_file_private() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tmpdir("perm");
        let path = dir.join("c.json");
        std::fs::write(&path, "1").unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600)).unwrap();
        write_with_backup(&path, "2").unwrap();
        assert_eq!(std::fs::metadata(&path).unwrap().permissions().mode() & 0o777, 0o600);
        let fresh = dir.join("new.json");
        write_with_backup(&fresh, "x").unwrap();
        assert_eq!(std::fs::metadata(&fresh).unwrap().permissions().mode() & 0o777, 0o600);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[cfg(unix)]
    #[test]
    fn symlink_target_is_written_through() {
        let dir = tmpdir("link");
        let real = dir.join("real.json");
        let link = dir.join("link.json");
        std::fs::write(&real, "1").unwrap();
        std::os::unix::fs::symlink(&real, &link).unwrap();
        write_with_backup(&link, "2").unwrap();
        assert!(std::fs::symlink_metadata(&link).unwrap().file_type().is_symlink());
        assert_eq!(std::fs::read_to_string(&real).unwrap(), "2");
        std::fs::remove_file(&link).unwrap();
        std::os::unix::fs::symlink(&real, dir.join(".mcp.json")).unwrap();
        assert!(check_project_links(&dir).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn read_config_bom_and_missing() {
        let dir = tmpdir("read");
        let path = dir.join("c.json");
        assert_eq!(read_config(&path).unwrap(), None);
        std::fs::write(&path, b"\xEF\xBB\xBF{\"a\":1}").unwrap();
        let t = read_config(&path).unwrap().unwrap();
        assert_eq!(t, "{\"a\":1}");
        assert!(merge(Some(&t), "p", &entry(), false).is_ok());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn root_checks() {
        let dir = tmpdir("root");
        let root = dir.ancestors().last().unwrap().to_path_buf();
        assert!(check_root(&root, None, false).is_err());
        assert!(check_root(&root, None, true).is_ok());
        assert!(check_root(&dir, Some(&dir), false).is_err());
        assert!(check_root(&dir, Some(&dir), true).is_ok());
        assert!(check_root(&dir, Some(Path::new("/nonexistent-home")), false).is_ok());
        let file = dir.join("f.txt");
        std::fs::write(&file, "x").unwrap();
        assert!(canonical_dir(&file.to_string_lossy(), "--root").is_err());
        assert!(canonical_dir(&dir.to_string_lossy(), "--root").is_ok());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn option_parsing() {
        let s = |v: &[&str]| v.iter().map(|x| x.to_string()).collect::<Vec<_>>();
        let o = parse_options("install", &s(&["--target", "cursor", "--target", "antigravity", "--root", "/r", "--compact"])).unwrap();
        assert_eq!(o.targets, vec![Target::Cursor, Target::Antigravity]);
        assert!(o.compact);
        assert!(parse_options("install", &s(&["--target", "vim"])).is_err());
        assert!(parse_options("install", &s(&["--bogus"])).is_err());
        assert_eq!(parse_options("install", &[]).unwrap().targets, vec![Target::ClaudeCode]);
    }
}
