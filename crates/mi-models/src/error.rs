//! Error types for machine-learning models.
//!
//! All fallible model operations report failures through [`ModelError`].
//! Invalid configuration is rejected at construction; backend failures
//! keep their typed source reachable through [`std::error::Error::source`].

use mi_tensor::TensorError;
use std::fmt;

/// Errors produced while building, training, or evaluating models.
#[derive(Debug, Clone, PartialEq)]
pub enum ModelError {
    /// A tensor operation failed inside the backend.
    Tensor(TensorError),
    /// The model configuration is invalid (zero-sized layer, mismatched
    /// parameter layout).
    InvalidConfig {
        /// Human-readable description of the violated rule.
        reason: String,
    },
}

impl fmt::Display for ModelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Tensor(source) => write!(f, "tensor error: {source}"),
            Self::InvalidConfig { reason } => write!(f, "invalid model configuration: {reason}"),
        }
    }
}

impl std::error::Error for ModelError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Tensor(source) => Some(source),
            Self::InvalidConfig { .. } => None,
        }
    }
}

impl From<TensorError> for ModelError {
    fn from(source: TensorError) -> Self {
        Self::Tensor(source)
    }
}

/// Result type used throughout this crate.
pub type ModelResult<T> = Result<T, ModelError>;
