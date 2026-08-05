use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use tempfile::NamedTempFile;

use crate::downloads::Transfer;

/// Archive handle that keeps a downloaded temporary file alive while it is
/// being extracted, or references a file copied from a local bundle.
pub(crate) enum Archive {
    Local(PathBuf),
    Temp(NamedTempFile),
}

impl Archive {
    pub(crate) fn path(&self) -> &Path {
        match self {
            Archive::Local(path) => path,
            Archive::Temp(temp) => temp.path(),
        }
    }
}

/// Returns the base URL for runtime assets.
///
/// Defaults to the upstream GitHub release URL. Override with the
/// `KOHARU_RUNTIME_BASE_URL` environment variable to point at a mirror or
/// local HTTP server.
pub(crate) fn runtime_base_url(release: &str) -> String {
    std::env::var("KOHARU_RUNTIME_BASE_URL")
        .unwrap_or_else(|_| format!("https://github.com/mayocream/koharu/releases/download/{release}"))
}

/// Builds the download URL for a runtime asset.
pub(crate) fn runtime_asset_url(release: &str, asset: &str) -> String {
    format!("{}/{asset}", runtime_base_url(release))
}

/// Looks for a pre-shipped runtime asset next to the executable or in the
/// directory pointed to by `KOHARU_RUNTIME_DIR`.
pub(crate) fn find_bundled_asset(asset: &str) -> Option<PathBuf> {
    if let Ok(dir) = std::env::var("KOHARU_RUNTIME_DIR") {
        let path = Path::new(&dir).join(asset);
        if path.is_file() {
            return Some(path);
        }
    }

    if let Ok(exe) = std::env::current_exe() {
        if let Some(parent) = exe.parent() {
            let path = parent.join("runtime-assets").join(asset);
            if path.is_file() {
                return Some(path);
            }
        }
    }

    None
}

/// Fetches a runtime asset, preferring a bundled/local copy when available.
pub(crate) async fn fetch_runtime_asset(release: &str, asset: &str, stage: &Path) -> Result<Archive> {
    if let Some(local) = find_bundled_asset(asset) {
        let dest = stage.join(asset);
        tokio::fs::copy(&local, &dest)
            .await
            .with_context(|| format!("failed to copy bundled asset {} from {}", asset, local.display()))?;
        return Ok(Archive::Local(dest));
    }

    let url = runtime_asset_url(release, asset);
    let temp = tempfile::Builder::new().suffix(".tar.gz").tempfile()?;
    Transfer::new()?
        .fetch(&url, temp.path())
        .await
        .with_context(|| format!("failed to download runtime asset {asset} from {url}"))?;
    Ok(Archive::Temp(temp))
}
