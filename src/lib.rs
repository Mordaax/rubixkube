//! Local Rust tools for a real Kubeflow OptimizationJob demonstration.
//!
//! The upstream controllers remain external dependencies. This crate owns
//! training, protocol-compatible metric delivery and evidence validation.
pub mod report;
pub mod training;

/// Errors returned at command and validation boundaries.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("{0}")]
    Invalid(String),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    #[error(transparent)]
    Http(#[from] reqwest::Error),
}
/// Shared fallible return type.
pub type Result<T> = std::result::Result<T, Error>;
/// Construct a validation or execution error without exposing credentials.
pub fn invalid(message: impl Into<String>) -> Error { Error::Invalid(message.into()) }
