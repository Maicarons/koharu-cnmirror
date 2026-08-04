use anyhow::{Context, Result};
use serde::Deserialize;

use crate::downloads::Transfer;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Platform {
    WindowsX64,
    LinuxX64,
}

impl Platform {
    pub(crate) fn host() -> Result<Self> {
        if cfg!(all(target_os = "windows", target_arch = "x86_64")) {
            Ok(Self::WindowsX64)
        } else if cfg!(all(target_os = "linux", target_arch = "x86_64")) {
            Ok(Self::LinuxX64)
        } else {
            anyhow::bail!("PyPI runtime packages do not support this target")
        }
    }

    fn accepts(self, filename: &str) -> bool {
        match self {
            Self::WindowsX64 => filename.contains("win_amd64"),
            Self::LinuxX64 => filename.contains("manylinux") && filename.contains("x86_64"),
        }
    }
}

#[derive(Deserialize)]
struct Metadata {
    urls: Vec<Distribution>,
}

#[derive(Deserialize)]
struct Distribution {
    filename: String,
    url: String,
}

pub(crate) async fn wheel(project: &str, platform: Platform) -> Result<String> {
    // Try a China-accessible PyPI mirror first, then fall back to the official
    // index. New CUDA wheels may lag on mirrors, so the fallback keeps GPU
    // runtime downloads working everywhere.
    let endpoints = [pypi_endpoint(), "https://pypi.org".to_owned()];
    let mut last_error = None;
    for endpoint in endpoints {
        let url = format!("{endpoint}/pypi/{project}/json");
        let metadata: Result<Metadata> = async {
            Transfer::new()?
                .get(&url)
                .send()
                .await
                .with_context(|| format!("failed to query {project}"))?
                .error_for_status()?
                .json()
                .await
                .with_context(|| format!("invalid PyPI metadata for {project}"))
        }
        .await;
        match metadata {
            Ok(metadata) => {
                if let Some(file) = metadata
                    .urls
                    .into_iter()
                    .find(|file| file.filename.ends_with(".whl") && platform.accepts(&file.filename))
                {
                    return Ok(file.url);
                }
                last_error = Some(anyhow::anyhow!("{project} has no compatible wheel on {endpoint}"));
            }
            Err(error) => last_error = Some(error),
        }
    }
    Err(last_error.unwrap_or_else(|| anyhow::anyhow!("failed to resolve wheel for {project}")))
}

/// Returns the PyPI index used to look up runtime wheels.
///
/// The default points at the Tsinghua mirror for direct access in China.
/// Override it with the `PYPI_ENDPOINT` environment variable.
fn pypi_endpoint() -> String {
    std::env::var("PYPI_ENDPOINT")
        .ok()
        .map(|value| value.trim().to_owned())
        .filter(|value| !value.is_empty())
        .map(|value| value.trim_end_matches('/').to_owned())
        .unwrap_or_else(|| "https://pypi.tuna.tsinghua.edu.cn".to_owned())
}
