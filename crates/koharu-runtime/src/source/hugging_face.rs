use std::{
    collections::HashMap,
    path::{Component, Path, PathBuf},
    sync::{Arc, LazyLock},
};

use anyhow::{Context, ensure};
use serde::Deserialize;
use tokio::sync::{Mutex, OnceCell};

use crate::{Store, downloads::Transfer};

static REVISIONS: LazyLock<Mutex<HashMap<String, Arc<OnceCell<String>>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
enum Revision<'a> {
    Pinned(&'a str),
    Latest,
}

/// An immutable file snapshot hosted by Hugging Face.
///
/// A latest file resolves the repository head once per process. Every file in
/// that repository then uses the same commit, so a model cannot be assembled
/// from different revisions if its repository changes during initialization.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct HuggingFaceFile<'a> {
    repository: &'a str,
    revision: Revision<'a>,
    filename: &'a str,
}

impl<'a> HuggingFaceFile<'a> {
    #[must_use]
    pub const fn pinned(repository: &'a str, revision: &'a str, filename: &'a str) -> Self {
        Self {
            repository,
            revision: Revision::Pinned(revision),
            filename,
        }
    }

    #[must_use]
    pub const fn latest(repository: &'a str, filename: &'a str) -> Self {
        Self {
            repository,
            revision: Revision::Latest,
            filename,
        }
    }

    pub async fn resolve(self) -> anyhow::Result<PathBuf> {
        let mut repository = self.repository.split('/');
        let owner = repository.next().unwrap_or_default();
        let name = repository.next().unwrap_or_default();
        ensure!(
            repository.next().is_none()
                && [owner, name].into_iter().all(|part| {
                    !part.is_empty()
                        && part != "."
                        && part != ".."
                        && part.bytes().all(|byte| {
                            byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.')
                        })
                }),
            "invalid Hugging Face repository {}",
            self.repository
        );
        ensure!(
            !self.filename.is_empty()
                && !self.filename.contains('\\')
                && Path::new(self.filename)
                    .components()
                    .all(|component| matches!(component, Component::Normal(_))),
            "invalid Hugging Face filename {}",
            self.filename
        );

        // Validate a pinned revision up front, then try every download source in
        // priority order. ModelScope is preferred for direct access in China;
        // Hugging Face (hf-mirror by default) is the fallback for repositories
        // that ModelScope does not mirror.
        let pinned = match self.revision {
            Revision::Pinned(revision) => {
                ensure!(is_commit(revision), "invalid pinned revision {revision}");
                Some(revision)
            }
            Revision::Latest => None,
        };

        let mut last_error = None;
        for source in [Source::ModelScope, Source::HuggingFace] {
            let revision = match pinned {
                Some(revision) => revision.to_owned(),
                None => source.latest_revision(self.repository).await?,
            };
            let target = snapshot_path(self.repository, &revision, self.filename);
            match Store::file(target, {
                let repository = self.repository;
                let filename = self.filename;
                let revision = revision.clone();
                move |stage| async move {
                    let url = source.file_url(repository, &revision, filename);
                    Transfer::new()?.fetch(&url, &stage).await
                }
            })
            .await
            {
                Ok(path) => return Ok(path),
                Err(error) => {
                    last_error = Some(error);
                    continue;
                }
            }
        }
        Err(last_error.unwrap_or_else(|| {
            anyhow::anyhow!("all download sources failed for {}", self.repository)
        }))
    }
}

async fn hf_latest_revision(repository: &str) -> anyhow::Result<String> {
    let revision = {
        let mut revisions = REVISIONS.lock().await;
        revisions
            .entry(repository.to_owned())
            .or_insert_with(|| Arc::new(OnceCell::new()))
            .clone()
    };

    revision
        .get_or_try_init(|| async {
            #[derive(Deserialize)]
            struct Repository {
                sha: String,
            }

            let url = format!("{}/api/models/{repository}", hf_endpoint());
            let response = Transfer::new()?
                .get(&url)
                .send()
                .await
                .with_context(|| format!("failed to resolve {repository}"))?
                .error_for_status()
                .with_context(|| format!("failed to resolve {repository}"))?
                .json::<Repository>()
                .await
                .with_context(|| format!("invalid metadata for {repository}"))?;
            ensure!(
                is_commit(&response.sha),
                "{repository} returned invalid commit {}",
                response.sha
            );
            Ok(response.sha)
        })
        .await
        .cloned()
}

/// Returns the Hugging Face endpoint used for both metadata and file downloads.
///
/// The default points at `hf-mirror.com`, a China-accessible mirror of
/// `huggingface.co` with an identical URL layout, so downloads work without
/// extra configuration behind the Great Firewall. Override it with the
/// standard `HF_ENDPOINT` environment variable (for example
/// `https://huggingface.co`) to use the official endpoint or any other
/// API-compatible mirror.
fn hf_endpoint() -> String {
    std::env::var("HF_ENDPOINT")
        .ok()
        .map(|value| value.trim().to_owned())
        .filter(|value| !value.is_empty())
        .map(|value| value.trim_end_matches('/').to_owned())
        .unwrap_or_else(|| "https://hf-mirror.com".to_owned())
}

/// A download backend for model files.
///
/// ModelScope is tried first because it is directly reachable in China; the
/// Hugging Face endpoint (hf-mirror by default) is the fallback for any
/// repository that ModelScope does not mirror.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
enum Source {
    ModelScope,
    HuggingFace,
}

impl Source {
    /// Resolves the revision string used in this source's download URL.
    ///
    /// For Hugging Face we resolve the repository head to an immutable commit so
    /// a model is never assembled from mismatched revisions. ModelScope mirrors
    /// a subset of repositories and does not always carry the exact Hugging Face
    /// commit, so we use its default branch (`master`) instead.
    async fn latest_revision(self, repository: &str) -> anyhow::Result<String> {
        match self {
            Source::HuggingFace => hf_latest_revision(repository).await,
            Source::ModelScope => Ok("master".to_owned()),
        }
    }

    /// Builds the direct download URL for a single file from this source.
    fn file_url(self, repository: &str, revision: &str, filename: &str) -> String {
        match self {
            Source::HuggingFace => format!(
                "{}/{}/resolve/{}/{}",
                hf_endpoint(),
                repository,
                revision,
                filename
            ),
            Source::ModelScope => format!(
                "{}/api/v1/models/{}/repo?Revision={}&FilePath={}",
                modelscope_endpoint(),
                repository,
                revision,
                encode_query(filename)
            ),
        }
    }
}

/// Returns the ModelScope endpoint used for file downloads.
///
/// The default is the public `modelscope.cn` hub, which is directly reachable
/// in China. Override it with the standard `MODELSCOPE_ENDPOINT` environment
/// variable.
fn modelscope_endpoint() -> String {
    std::env::var("MODELSCOPE_ENDPOINT")
        .ok()
        .map(|value| value.trim().to_owned())
        .filter(|value| !value.is_empty())
        .map(|value| value.trim_end_matches('/').to_owned())
        .unwrap_or_else(|| "https://modelscope.cn".to_owned())
}

/// Percent-encodes a value for use inside a ModelScope query parameter, keeping
/// only the RFC 3986 unreserved characters literal.
fn encode_query(value: &str) -> String {
    let mut encoded = String::with_capacity(value.len());
    for byte in value.bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'.' | b'_' | b'~' => {
                encoded.push(byte as char)
            }
            _ => {
                encoded.push('%');
                encoded.push(
                    char::from_digit(u32::from(byte >> 4), 16)
                        .unwrap()
                        .to_ascii_uppercase(),
                );
                encoded.push(
                    char::from_digit(u32::from(byte & 0xf), 16)
                        .unwrap()
                        .to_ascii_uppercase(),
                );
            }
        }
    }
    encoded
}

fn snapshot_path(repository: &str, revision: &str, filename: &str) -> PathBuf {
    Store::root()
        .join("hugging-face")
        .join(repository.replace('/', "--"))
        .join("snapshots")
        .join(revision)
        .join(filename)
}

fn is_commit(revision: &str) -> bool {
    revision.len() == 40 && revision.bytes().all(|byte| byte.is_ascii_hexdigit())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_full_commit_hashes() {
        assert!(is_commit("0123456789abcdef0123456789abcdef01234567"));
        assert!(is_commit("0123456789ABCDEF0123456789ABCDEF01234567"));
        assert!(!is_commit("01234567"));
        assert!(!is_commit("z123456789abcdef0123456789abcdef01234567"));
    }

    #[test]
    fn stores_files_by_immutable_snapshot() {
        let path = snapshot_path(
            "owner/model",
            "0123456789abcdef0123456789abcdef01234567",
            "subdir/model.safetensors",
        );
        assert!(path.ends_with(
            "owner--model/snapshots/0123456789abcdef0123456789abcdef01234567/subdir/model.safetensors"
        ));
    }

    #[tokio::test]
    async fn rejects_paths_that_escape_the_snapshot() {
        let revision = "0123456789abcdef0123456789abcdef01234567";
        assert!(
            HuggingFaceFile::pinned("owner/model", revision, "../model.bin")
                .resolve()
                .await
                .is_err()
        );
        assert!(
            HuggingFaceFile::pinned("owner\\model", revision, "model.bin")
                .resolve()
                .await
                .is_err()
        );
        assert!(
            HuggingFaceFile::pinned("owner/model", "main", "model.bin")
                .resolve()
                .await
                .is_err()
        );
    }
}
