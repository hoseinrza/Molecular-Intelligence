//! Error types for the training infrastructure.

use mi_models::ModelError;
use mi_tensor::TensorError;
use std::fmt;

/// Errors raised while driving a training loop.
#[derive(Debug, Clone, PartialEq)]
pub enum TrainingError {
    /// A tensor operation, readback, or update failed at the project/Burn
    /// boundary.
    Tensor(TensorError),
    /// The model rejected an update or was misconfigured.
    Model(ModelError),
    /// The training configuration itself is unusable (invalid momentum,
    /// schedule bounds, empty data where data is required).
    InvalidConfig(String),
}

impl fmt::Display for TrainingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Tensor(source) => write!(f, "training tensor error: {source}"),
            Self::Model(source) => write!(f, "training model error: {source}"),
            Self::InvalidConfig(reason) => write!(f, "invalid training config: {reason}"),
        }
    }
}

impl std::error::Error for TrainingError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Tensor(source) => Some(source),
            Self::Model(source) => Some(source),
            Self::InvalidConfig(_) => None,
        }
    }
}

impl From<TensorError> for TrainingError {
    fn from(source: TensorError) -> Self {
        Self::Tensor(source)
    }
}

impl From<ModelError> for TrainingError {
    fn from(source: ModelError) -> Self {
        Self::Model(source)
    }
}

/// Result type used by this crate.
pub type TrainingResult<T> = Result<T, TrainingError>;
