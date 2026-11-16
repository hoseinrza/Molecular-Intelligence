//! Error types for scientific evaluation.
//!
//! All fallible metric computations report failures through
//! [`MetricsError`]. An undefined quantity is an error, never a silently
//! substituted number.

use std::fmt;

/// Errors produced when computing evaluation metrics.
#[derive(Debug, Clone, PartialEq)]
pub enum MetricsError {
    /// The prediction and reference slices hold different numbers of
    /// values; metrics are defined only on paired observations.
    LengthMismatch {
        /// Number of predictions supplied.
        predictions: usize,
        /// Number of references supplied.
        references: usize,
    },
    /// No observations were supplied.
    Empty,
    /// The coefficient of determination is undefined because the
    /// references have zero variance (every reference equals its mean) —
    /// there is no variance for the model to explain.
    UndefinedR2,
}

impl fmt::Display for MetricsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LengthMismatch {
                predictions,
                references,
            } => write!(
                f,
                "length mismatch: {predictions} predictions versus {references} references"
            ),
            Self::Empty => f.write_str("no observations supplied"),
            Self::UndefinedR2 => f.write_str(
                "R² is undefined: the references have zero variance, \
                 so there is no variance to explain",
            ),
        }
    }
}

impl std::error::Error for MetricsError {}

/// Result type used throughout this crate.
pub type MetricsResult<T> = Result<T, MetricsError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_messages_are_descriptive() {
        let cases: [(MetricsError, &str); 3] = [
            (
                MetricsError::LengthMismatch {
                    predictions: 3,
                    references: 5,
                },
                "length mismatch: 3 predictions versus 5 references",
            ),
            (MetricsError::Empty, "no observations supplied"),
            (
                MetricsError::UndefinedR2,
                "R² is undefined: the references have zero variance, \
                 so there is no variance to explain",
            ),
        ];
        for (error, expected) in cases {
            assert_eq!(error.to_string(), expected);
        }
    }

    #[test]
    fn errors_are_std_errors() {
        fn assert_error<E: std::error::Error>(_: &E) {}
        assert_error(&MetricsError::Empty);
    }
}
