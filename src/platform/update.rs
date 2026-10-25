//! Update check: only "query the version + link to the download page", no in-app download or install.

//! - Invoked from UI mode only; CLI subcommands and offline tools issue no network requests.
//! - Network failures are silent (no dialog, no interruption of what the user is doing).

use std::time::Duration;

const RELEASES_URL: &str = "https://api.github.com/repos/5hmtty/Yuke/releases/latest";
pub const DOWNLOAD_URL: &str = "https://github.com/5hmtty/Yuke/releases";
/// Privacy-policy link shown in the About dialog (PRIVACY.md in the repository).
pub const PRIVACY_URL: &str = "https://github.com/5hmtty/Yuke/blob/main/PRIVACY.md";
/// HTTP timeout (seconds). The GitHub API answers 403 without a User-Agent header, so one is mandatory.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(5);
const USER_AGENT: &str = concat!("Yuke/", env!("CARGO_PKG_VERSION"), " (update-check)");

/// Returns true when `remote` is newer than `local` (an update should be offered).
pub fn remote_is_newer(local: &str, remote: &str) -> bool {
    let l: Vec<&str> = local.trim_start_matches('v').split('.').collect();
    let r: Vec<&str> = remote.trim_start_matches('v').split('.').collect();
    let n = l.len().max(r.len());
    for i in 0..n {
        let lv = parse_seg(l.get(i).copied().unwrap_or("0"));
        let rv = parse_seg(r.get(i).copied().unwrap_or("0"));
        if rv > lv {
            return true;
        }
        if rv < lv {
            return false;
        }
    }
    false
}

/// Parse one version segment as u64: non-numeric segments (e.g. "beta") and empty ones count as 0 (a simplification --
/// releases here always use plain dotted numeric tags; non-numeric segments only exist as a robustness fallback).
fn parse_seg(s: &str) -> u64 {
    s.trim().parse::<u64>().unwrap_or(0)
}

fn windows_system_proxy() -> Option<(String, u16)> {
    use windows::Win32::System::Registry::{
        HKEY_CURRENT_USER, REG_VALUE_TYPE, RRF_RT_REG_DWORD, RRF_RT_REG_SZ, RegGetValueW,
    };
    use windows::core::{PCWSTR, w};
    const KEY: PCWSTR = w!("Software\\Microsoft\\Windows\\CurrentVersion\\Internet Settings");

    fn reg_string(key: PCWSTR, value: PCWSTR) -> Option<String> {
        let mut buf = [0u16; 512];
        let mut len = buf.len() as u32;
        let mut ty = REG_VALUE_TYPE(0);
        let status = unsafe {
            RegGetValueW(
                HKEY_CURRENT_USER,
                key,
                value,
                RRF_RT_REG_SZ,
                Some(&mut ty),
                Some(buf.as_mut_ptr().cast()),
                Some(&mut len),
            )
        };
        if status.is_err() {
            return None;
        }
        let end = buf.iter().position(|&c| c == 0).unwrap_or(buf.len());
        let s = String::from_utf16_lossy(&buf[..end]).trim().to_string();
        (!s.is_empty()).then_some(s)
    }

    fn reg_dword(key: PCWSTR, value: PCWSTR) -> Option<u32> {
        let mut v = 0u32;
        let status = unsafe {
            RegGetValueW(
                HKEY_CURRENT_USER,
                key,
                value,
                RRF_RT_REG_DWORD,
                None,
                Some(&mut v as *mut u32 as *mut _),
                Some(&mut (std::mem::size_of::<u32>() as u32)),
            )
        };
        (status.is_ok()).then_some(v)
    }

    if reg_dword(KEY, w!("ProxyEnable")) != Some(1) {
        return None;
    }
    let server = reg_string(KEY, w!("ProxyServer"))?;
    parse_proxy_server(&server)
}

/// Parse the ProxyServer registry value: whole-protocol form "1.2.3.4:7890", or per-protocol form
/// "http=1.2.3.4:8080;https=5.6.7.8:7890" (prefer https, fall back to http).
fn parse_proxy_server(v: &str) -> Option<(String, u16)> {
    let entry = if v.contains('=') {
        let kv: Vec<(&str, &str)> = v
            .split(';')
            .filter_map(|kv| kv.split_once('='))
            .map(|(p, a)| (p.trim(), a.trim()))
            .collect();
        // Per-protocol form: prefer https, fall back to http
        kv.iter()
            .find(|(p, _)| *p == "https")
            .or_else(|| kv.iter().find(|(p, _)| *p == "http"))
            .map(|(_, a)| *a)?
    } else {
        v.trim()
    };
    let (host, port) = entry.rsplit_once(':')?;
    let port = port.parse::<u16>().ok()?;
    (!host.is_empty()).then_some((host.to_string(), port))
}

/// Check for updates (blocking; the caller runs it on a background thread):
/// Prefer the 302 redirect method (github.com is far more reachable than api.github.com in some regions, and it does not consume
/// the anonymous 60-requests/hour API quota); on failure fall back to the legacy API -> parse the latest tag (strip 'v') ->
/// compare against `current_version`.
pub fn check_for_update(current_version: &str) -> anyhow::Result<Option<String>> {
    let mut last_err: Option<anyhow::Error> = None;
    for agent in candidate_agents() {
        match latest_tag_via_redirect(&agent).or_else(|redir| {
            latest_tag_via_api(&agent)
                .map_err(|api| anyhow::anyhow!("{redir:#}; API fallback: {api:#}"))
        }) {
            Ok(tag) => {
                return if remote_is_newer(current_version, &tag) {
                    Ok(Some(tag))
                } else {
                    Ok(None)
                };
            }
            Err(e) => last_err = Some(e),
        }
    }
    Err(last_err.unwrap_or_else(|| anyhow::anyhow!("no usable network path")))
}

fn candidate_agents() -> Vec<ureq::Agent> {
    let mut agents = Vec::new();
    let mut push = |proxy: Option<ureq::Proxy>| {
        let mut b = ureq::AgentBuilder::new()
            .timeout(REQUEST_TIMEOUT)
            .redirects(0);
        if let Some(p) = proxy {
            b = b.proxy(p);
        }
        agents.push(b.build());
    };
    push(None); // ① direct
    // ② Windows system proxy (WinINet registry)
    match windows_system_proxy() {
        Some((host, port)) => match ureq::Proxy::new(format!("http://{host}:{port}")) {
            Ok(p) => push(Some(p)),
            Err(_) => {}
        },
        None => {}
    }
    // ③ env proxy (HTTPS_PROXY / https_proxy / ALL_PROXY / all_proxy)
    for key in ["HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy"] {
        if let Ok(url) = std::env::var(key) {
            match ureq::Proxy::new(url.trim()) {
                Ok(p) => {
                    push(Some(p));
                    break;
                }
                Err(_) => {}
            }
        }
    }
    agents
}

/// GitHub releases/latest redirect page: GET answers 302 to /tag/vX.Y.Z.
const RELEASES_LATEST_URL: &str = "https://github.com/5hmtty/Yuke/releases/latest";

fn latest_tag_via_redirect(agent: &ureq::Agent) -> anyhow::Result<String> {
    let resp = agent
        .get(RELEASES_LATEST_URL)
        .set("User-Agent", USER_AGENT)
        .call()
        .map_err(|e| anyhow::anyhow!("failed to reach releases/latest: {e}"))?;
    let loc = resp
        .header("Location")
        .ok_or_else(|| anyhow::anyhow!("HTTP {} response has no Location header", resp.status()))?;
    tag_from_location(loc).ok_or_else(|| anyhow::anyhow!("cannot parse a version from Location: {loc}"))
}

/// Take the last path segment of Location (e.g. "/owner/repo/releases/tag/v1.1.0") and strip the leading
/// 'v'; an empty segment, or one not starting with a digit (untagged and other non-version tags), returns None.
fn tag_from_location(loc: &str) -> Option<String> {
    let tag = loc.rsplit('/').next()?.trim_start_matches('v').to_string();
    let is_numeric = tag.starts_with(|c: char| c.is_ascii_digit());
    (!tag.is_empty() && is_numeric).then_some(tag)
}

/// Legacy API method (api.github.com, anonymous quota of 60/hour) -- the fallback when the 302 method fails.
fn latest_tag_via_api(agent: &ureq::Agent) -> anyhow::Result<String> {
    let resp = agent
        .get(RELEASES_URL)
        .set("User-Agent", USER_AGENT)
        .call()
        .map_err(|e| anyhow::anyhow!("failed to query the latest version: {e}"))?;
    // Any non-2xx (e.g. a 403 rate limit) counts as failure and stays silent.
    if resp.status() < 200 || resp.status() >= 300 {
        return Err(anyhow::anyhow!("GitHub API HTTP {}", resp.status()));
    }
    let text = resp
        .into_string()
        .map_err(|e| anyhow::anyhow!("failed to read the latest-version response: {e}"))?;
    let body: serde_json::Value =
        serde_json::from_str(&text).map_err(|e| anyhow::anyhow!("failed to parse the latest-version response: {e}"))?;
    body.get("tag_name")
        .and_then(|v| v.as_str())
        .map(|s| s.trim_start_matches('v').to_string())
        .ok_or_else(|| anyhow::anyhow!("the response has no tag_name"))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// ProxyServer registry value parsing: whole-protocol / per-protocol / missing port.
    #[test]
    fn parse_proxy_server_forms() {
        assert_eq!(
            parse_proxy_server("1.2.3.4:8080"),
            Some(("1.2.3.4".to_string(), 8080))
        );
        assert_eq!(
            parse_proxy_server("http=1.2.3.4:8080;https=5.6.7.8:7890;ftp=x:1"),
            Some(("5.6.7.8".to_string(), 7890)),
            "scheme-separated form: https, falling back to http"
        );
        assert_eq!(
            parse_proxy_server("http=1.2.3.4:8080"),
            Some(("1.2.3.4".to_string(), 8080))
        );
        assert_eq!(parse_proxy_server("<local>"), None, "a missing port is rejected");
        assert_eq!(parse_proxy_server("localhost"), None, "a missing port is rejected");
    }

    /// Location header parsing (the pure part of the 302 method): relative and absolute URLs, strip 'v', reject non-numeric tags.
    #[test]
    fn tag_from_location_paths() {
        assert_eq!(
            tag_from_location("/5hmtty/Yuke/releases/tag/v1.1.0").as_deref(),
            Some("1.1.0")
        );
        assert_eq!(
            tag_from_location("https://github.com/5hmtty/Yuke/releases/tag/v1.1.0").as_deref(),
            Some("1.1.0")
        );
        assert_eq!(
            tag_from_location("/releases/tag/untagged"),
            None,
            "a non-numeric tag is rejected"
        );
        assert_eq!(tag_from_location("/"), None, "an empty segment is rejected");
    }

    #[test]
    fn version_compare_boundaries() {
        assert!(!remote_is_newer("1.0.0", "1.0.0"));
        assert!(!remote_is_newer("1.0", "1.0.0"), "missing segments padded with 0 (1.0 == 1.0.0)");
        assert!(!remote_is_newer("1.0.0", "1.0"), "1.0.0 == 1.0");
        assert!(!remote_is_newer("v1.0.0", "v1.0.0"), "equal after stripping the leading v");
        assert!(!remote_is_newer("1.0.0", "0.9.0"));
        assert!(
            !remote_is_newer("2.0.0", "1.9.9"),
            "a larger major version is older, not newer"
        );
        assert!(remote_is_newer("1.0.0", "1.0.1"));
        assert!(remote_is_newer("1.0.0", "1.1.0"));
        assert!(remote_is_newer("1.0.0", "2.0.0"));
        assert!(remote_is_newer("1.2", "1.2.1"), "after padding, the longer version is greater");
        assert!(remote_is_newer("1.0.0", "v1.1.0"), "compared after stripping the leading v");
        // Digit overflow: anything beyond u64 counts as 0 (robust, never panics)
        assert!(!remote_is_newer(
            "1.99999999999999999999",
            "1.99999999999999999999"
        ));
    }
}
