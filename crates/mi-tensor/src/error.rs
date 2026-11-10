use std::fmt;

/// Errors detected at the project/Burn boundary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TensorError {
    /// A shape has no elements or overflows the platform element count.
    InvalidShape(String),
    /// The supplied data length differs from the shape's element count.
    DataLength { expected: usize, actual: usize },
    /// An operation received incompatible shapes.
    ShapeMismatch { left: Vec<usize>, right: Vec<usize> },
    /// An axis is outside the tensor rank.
    InvalidDimension { dimension: usize, rank: usize },
    /// An operation requiring at least one value received an empty tensor.
    EmptyTensor,
    /// A value is NaN or infinite.
    NonFiniteValue { index: usize },
    /// Burn currently exposes only floating point tensors through this wrapper.
    UnsupportedDType,
}

impl fmt::Display for TensorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidShape(s) => write!(f, "invalid tensor shape: {s}"),
            Self::DataLength { expected, actual } => write!(f, "shape requires {expected} values, got {actual}"),
            Self::ShapeMismatch { left, right } => write!(f, "shape mismatch: {left:?} versus {right:?}"),
            Self::InvalidDimension { dimension, rank } => write!(f, "dimension {dimension} is outside rank {rank}"),
            Self::EmptyTensor => f.write_str("operation is undefined for an empty tensor"),
            Self::NonFiniteValue { index } => write!(f, "non-finite value at flat index {index}"),
            Self::UnsupportedDType => f.write_str("dtype is not supported by this tensor wrapper"),
        }
    }
}
impl std::error::Error for TensorError {}
/// Result type used by this crate.
pub type TensorResult<T> = Result<T, TensorError>;
