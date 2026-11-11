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
    /// An operation requiring at least one value received an empty tensor.
    EmptyTensor,
    /// A value is NaN or infinite.
    NonFiniteValue { index: usize },
    /// The tensor does not hold exactly one element.
    ExpectedScalar { actual: usize },
    /// Burn failed while reading tensor data back to the host.
    Readback(String),
    /// The operation needs a tensor created on an autodiff-enabled device.
    AutodiffUnavailable,
    /// The requested dtype is not supported by this tensor wrapper.
    UnsupportedDType,
}

impl fmt::Display for TensorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidShape(s) => write!(f, "invalid tensor shape: {s}"),
            Self::DataLength { expected, actual } => {
                write!(f, "shape requires {expected} values, got {actual}")
            }
            Self::ShapeMismatch { left, right } => {
                write!(f, "shape mismatch: {left:?} versus {right:?}")
            }
            Self::EmptyTensor => f.write_str("operation is undefined for an empty tensor"),
            Self::NonFiniteValue { index } => write!(f, "non-finite value at flat index {index}"),
            Self::ExpectedScalar { actual } => {
                write!(f, "expected a single-element tensor, got {actual} elements")
            }
            Self::Readback(message) => write!(f, "host readback failed: {message}"),
            Self::AutodiffUnavailable => f.write_str(
                "autodiff is unavailable: create the tensor with `Cpu::training()` to use this operation",
            ),
            Self::UnsupportedDType => f.write_str("dtype is not supported by this tensor wrapper"),
        }
    }
}

impl std::error::Error for TensorError {}

/// Result type used by this crate.
pub type TensorResult<T> = Result<T, TensorError>;
