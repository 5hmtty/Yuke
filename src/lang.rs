use std::collections::HashMap;
use std::path::Path;
use std::sync::RwLock;

const CN_BUILTIN: &str = include_str!("../lang/zh.json");

pub const DEFAULT_LANG: &str = "zh";

pub fn detect_lang() -> String {
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetUserDefaultUILanguage() -> u16;
    }
    // SAFETY: system call without parameters
    let primary = unsafe { GetUserDefaultUILanguage() } & 0x3ff;
    if primary == 0x0004 {
        "zh".to_string()
    } else {
        "en".to_string()
    }
}

struct Tables {
    /// Key/value table of the current language
    cur: HashMap<String, String>,
    /// Current language ID
    cur_id: String,
    /// CN fallback table
    cn: HashMap<String, String>,
}

/// Current language ID (returns DEFAULT_LANG when uninitialized; used by test assertions only).
#[cfg(test)]
pub fn current() -> String {
    TABLES
        .read()
        .unwrap_or_else(|e| e.into_inner())
        .as_ref()
        .map(|t| t.cur_id.clone())
        .unwrap_or_else(|| DEFAULT_LANG.to_string())
}

static TABLES: RwLock<Option<Tables>> = RwLock::new(None);

/// Parse a flat JSON object ({"source": "translation"}); bad JSON degrades to an empty table (the fallback chain survives, no panic).
fn parse(text: &str) -> HashMap<String, String> {
    serde_json::from_str::<HashMap<String, String>>(text).unwrap_or_else(|e| {
        eprintln!("[lang] WARN language file JSON parse failed ({e}) -- falling back to an empty table");
        HashMap::new()
    })
}

/// Scan `dir` for available language IDs (*.json names without extension, sorted; empty when the directory is absent).
pub fn available(dir: &Path) -> Vec<String> {
    let mut out: Vec<String> = std::fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .filter(|e| e.path().extension().and_then(|s| s.to_str()) == Some("json"))
        .filter_map(|e| {
            e.path()
                .file_stem()
                .and_then(|s| s.to_str())
                .map(|s| s.to_lowercase())
        })
        .collect();
    out.sort();
    out.dedup();
    out
}

/// Initialize/switch the current language to `lang_id` (disk `dir/<lang_id>.json` wins; missing = empty table and the CN fallback applies).
/// `dir` is not created when absent -- the first write belongs to the language selector or the packaging flow.
pub fn set_language(lang_id: &str, dir: &Path) {
    let mut guard = TABLES.write().unwrap_or_else(|e| e.into_inner());
    let tables = guard.get_or_insert_with(|| Tables {
        cur: HashMap::new(),
        cur_id: DEFAULT_LANG.to_string(),
        cn: HashMap::new(),
    });
    ensure_cn(dir, tables);
    let path = dir.join(format!("{lang_id}.json"));
    tables.cur = std::fs::read_to_string(&path)
        .map(|s| parse(&s))
        .unwrap_or_default();
    tables.cur_id = lang_id.to_string();
}

/// Current language ID (returns DEFAULT_LANG when uninitialized).
/// Lookup: current language -> CN fallback -> the key itself (**the semantic id text**).
pub fn t(key: &str) -> String {
    let guard = TABLES.read().unwrap_or_else(|e| e.into_inner());
    match guard.as_ref() {
        Some(t) => t
            .cur
            .get(key)
            .cloned()
            .or_else(|| t.cn.get(key).cloned())
            .unwrap_or_else(|| key.to_string()),
        None => key.to_string(),
    }
}

/// Lookup with parameters: `{name}` placeholders in keys and values are replaced in the order of pairs.
pub fn tf(key: &str, pairs: &[(&str, &str)]) -> String {
    let mut s = t(key);
    for (k, v) in pairs {
        s = s.replace(k, v);
    }
    s
}

/// Initialize the CN fallback table (built-in plus disk merge); used by the first set_language call and by tests.
fn ensure_cn(dir: &Path, tables: &mut Tables) {
    if !tables.cn.is_empty() {
        return;
    }
    tables.cn = parse(CN_BUILTIN);
    if let Ok(s) = std::fs::read_to_string(dir.join(format!("{DEFAULT_LANG}.json"))) {
        for (k, v) in parse(&s) {
            tables.cn.insert(k, v);
        }
    }
}

#[macro_export]
macro_rules! tfmt {
    ($key:expr $(, $arg:expr)* $(,)?) => {{
        let mut s = $crate::lang::t($key);
        $(
            // Named placeholders take priority: {cmd} ← the argument name via stringify (covering most of format!'s implicit named capture)
            s = s.replace(concat!("{", stringify!($arg), "}"), &$arg.to_string());
        )*
        $(
            // Sequential {} placeholders as a fallback (when the key uses no named placeholder)
            let _ = &$arg;
        )*
        let mut s2 = s;
        $(
            s2 = s2.replacen("{}", &$arg.to_string(), 1);
        )*
        s2
    }};
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_json_flat_map_and_bad_json_fallback() {
        let m = parse(r#"{"ui_btn_record": "补录", "ui_btn_delete": "Remove"}"#);
        assert_eq!(m.get("ui_btn_record").map(String::as_str), Some("补录"));
        assert_eq!(m.get("ui_btn_delete").map(String::as_str), Some("Remove"));
        assert!(
            parse("不是 json").is_empty(),
            "bad JSON must degrade to an empty table, not panic"
        );
    }

    #[test]
    fn t_fallback_chain_cn_builtin_last() {
        // Uninitialized (before set_language): fall back to the key itself = the semantic id text
        assert_eq!(t("ui_nonexistent_xyz"), "ui_nonexistent_xyz");
        let dir = std::env::temp_dir().join(format!("viq_lang_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("en.json"),
            r#"{"ui_btn_record": "Record", "ui_only_here": "Only Here"}"#,
        )
        .unwrap();
        // EN: use the EN value when present; a missing key falls back to the built-in CN value; only if both tables miss does it fall back to the key
        set_language("en", &dir);
        assert_eq!(current(), "en");
        assert_eq!(t("ui_btn_record"), "Record");
        assert_eq!(t("ui_only_here"), "Only Here");
        assert_eq!(t("ui_tab_home"), "主页", "a miss on the EN table must fall back to the built-in CN table");
        assert_eq!(
            t("ui_nope_nope"),
            "ui_nope_nope",
            "missing from both tables -> the key itself (the id text)"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn tf_replaces_named_placeholders() {
        let dir = std::env::temp_dir().join(format!("viq_lang_tf_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("en.json"),
            r#"{"deleted={name}": "deleted={name}"}"#,
        )
        .unwrap();
        set_language("en", &dir);
        assert_eq!(tf("deleted={name}", &[("{name}", "cmd1")]), "deleted=cmd1");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn en_covers_all_cn_keys() {
        let cn = parse(CN_BUILTIN);
        assert!(!cn.is_empty(), "the built-in CN fallback table must not be empty");
        let en_text = include_str!("../lang/en.json");
        let en = parse(en_text);
        let missing: Vec<&String> = cn.keys().filter(|k| !en.contains_key(*k)).collect();
        assert!(
            missing.is_empty(),
            "EN.json is missing {} keys (the UI would show Chinese leftovers in EN mode): {:?}",
            missing.len(),
            missing
        );
        let extra: Vec<&String> = en.keys().filter(|k| !cn.contains_key(*k)).collect();
        assert!(
            extra.is_empty(),
            "EN.json has {} dead keys absent from CN: {:?}",
            extra.len(),
            extra
        );
    }

    /// 1) Referencing a nonexistent key should fail at compile time (the `k::` constants come from build.rs, so rustc reports it directly);
    /// 2) a key present in the table but never referenced in source is a zombie -> build.rs errors out.

    /// Only the cheapest invariant is pinned here: **the generated file and the CN table share one source** -- every
    /// `pub const NAME: &str = "id"` in `src/i18n_keys.rs` must have a value in the CN table, and every CN key must
    /// appear in the generated file. This catches hand-edited generated files and lang/zh.json edits without a rebuild.
    #[test]
    fn generated_keys_match_cn_table() {
        let zh = parse(CN_BUILTIN);
        assert!(!zh.is_empty(), "the built-in CN fallback table must not be empty");
        let src = std::fs::read_to_string(
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("src")
                .join("i18n_keys.rs"),
        )
        .expect("src/i18n_keys.rs does not exist -- did the build.rs i18n step run?");
        // ⚠ `gen` is a reserved word in edition 2024; do not use it
        let generated = src
            .lines()
            .filter_map(|l| {
                l.trim()
                    .strip_prefix("pub const ")?
                    .split_once(": &str = \"")
                    .and_then(|(_, v)| v.strip_suffix("\";"))
            })
            .collect::<Vec<_>>();
        assert!(
            generated.len() > 100,
            "unexpected i18n_keys.rs constant count ({}); the generation step may not have run",
            generated.len()
        );
        let missing_in_table: Vec<&&str> = generated
            .iter()
            .filter(|id| !zh.contains_key(**id))
            .collect();
        assert!(
            missing_in_table.is_empty(),
            "i18n_keys.rs has {} constants with no value in lang/zh.json: {:?}",
            missing_in_table.len(),
            missing_in_table
        );
        let missing_in_gen: Vec<&String> = zh
            .keys()
            .filter(|k| !generated.contains(&k.as_str()))
            .collect();
        assert!(
            missing_in_gen.is_empty(),
            "lang/zh.json has {} keys with no generated constant (table edited without a rebuild?): {:?}",
            missing_in_gen.len(),
            missing_in_gen
        );
    }

}
