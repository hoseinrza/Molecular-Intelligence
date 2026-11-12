//! Error types for molecular featurization.
//!
//! All fallible operations in this crate report failures through
//! [`FeaturizationError`]. Invalid input never panics and is never
//! silently ignored; backend-level failures keep their typed source
//! reachable through [`std::error::Error::source`].

use mi_graph::GraphError;
use mi_tensor::TensorError;
use std::fmt;

/// Errors produced while converting molecular structures into tensors.
#[derive(Debug, Clone, PartialEq)]
pub enum FeaturizationError {
    /// The molecular graph could not be built or traversed.
    Graph(GraphError),
    /// A tensor could not be constructed or read back.
    Tensor(TensorError),
    /// An atom has no Cartesian position where the operation requires one
    /// (geometry features need every atom placed in space).
    MissingPosition {
        /// The 0-based index of the position-less atom.
        atom_index: usize,
    },
}

impl fmt::Display for FeaturizationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Graph(source) => write!(f, "molecular graph error: {source}"),
            Self::Tensor(source) => write!(f, "tensor error: {source}"),
            Self::MissingPosition { atom_index } => write!(
                f,
                "atom {atom_index} has no position, which geometry features require"
            ),
        }
    }
}

impl std::error::Error for FeaturizationError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Graph(source) => Some(source),
            Self::Tensor(source) => Some(source),
            Self::MissingPosition { .. } => None,
        }
    }
}

impl From<GraphError> for FeaturizationError {
    fn from(source: GraphError) -> Self {
        Self::Graph(source)
    }
}

impl From<TensorError> for FeaturizationError {
    fn from(source: TensorError) -> Self {
        Self::Tensor(source)
    }
}

/// Result type used throughout this crate.
pub type FeaturizationResult<T> = Result<T, FeaturizationError>;

#[cfg(test)]
mod tests {
    use super::*;
    use mi_chemistry::ChemistryError;
    use mi_tensor::TensorShape;

    #[test]
    fn display_messages_are_descriptive() {
        let cases: [(FeaturizationError, &str); 3] = [
            (
                FeaturizationError::Graph(GraphError::InvalidMolecule(
                    ChemistryError::InvalidMolecularStructure {
                        reason: "empty".into(),
                    },
                )),
                "molecular graph error: source molecule is not a valid molecular structure: invalid molecular structure: empty",
            ),
            (
                FeaturizationError::Tensor(TensorError::InvalidShape("bad".into())),
                "tensor error: invalid tensor shape: bad",
            ),
            (
                FeaturizationError::MissingPosition { atom_index: 2 },
                "atom 2 has no position, which geometry features require",
            ),
        ];
        for (error, expected) in cases {
            assert_eq!(error.to_string(), expected);
        }
    }

    #[test]
    fn sources_are_reachable() {
        let tensor_error = FeaturizationError::Tensor(TensorError::DataLength {
            expected: 6,
            actual: 4,
        });
        let source = std::error::Error::source(&tensor_error)
            .and_then(|source| source.downcast_ref::<TensorError>());
        assert!(source.is_some(), "tensor source must stay reachable");

        let missing = FeaturizationError::MissingPosition { atom_index: 0 };
        assert!(std::error::Error::source(&missing).is_none());
    }

    #[test]
    fn tensor_shape_conversions_flow_through() {
        // TensorShape::new rejects zero dims; the From impl carries it.
        let error: FeaturizationError = TensorShape::new(vec![0, 7]).unwrap_err().into();
        assert!(matches!(error, FeaturizationError::Tensor(_)));
    }
}
