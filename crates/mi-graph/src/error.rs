//! Error types for molecular graph construction and validation.
//!
//! All fallible operations in this crate report failures through
//! [`GraphError`]. Invalid input never panics and is never silently ignored.

use mi_chemistry::ChemistryError;
use std::fmt;

/// Errors produced when building or validating a
/// [`MolecularGraph`](crate::MolecularGraph).
#[derive(Debug, Clone, PartialEq)]
pub enum GraphError {
    /// The source molecule failed its structural validation.
    ///
    /// Graph construction validates the input molecule first; the wrapped
    /// [`ChemistryError`] describes the exact violation and is reachable
    /// through [`std::error::Error::source`].
    InvalidMolecule(ChemistryError),
    /// A node index does not reference an existing node.
    InvalidNodeIndex {
        /// The rejected index.
        index: usize,
        /// The number of nodes available; valid indices are `0..node_count`.
        node_count: usize,
    },
    /// An edge index does not reference an existing edge.
    InvalidEdgeIndex {
        /// The rejected index.
        index: usize,
        /// The number of edges available; valid indices are `0..edge_count`.
        edge_count: usize,
    },
    /// A chemical edge connects a node to itself.
    ///
    /// Unreachable through normal construction (bonds reject self-bonds);
    /// reported by graph validation as a defense-in-depth check.
    SelfLoop {
        /// The node index appearing on both sides of the edge.
        node_index: usize,
    },
    /// Two chemical edges connect the same unordered pair of nodes.
    ///
    /// Unreachable through normal construction (molecules reject duplicate
    /// bonds); reported by graph validation as a defense-in-depth check.
    DuplicateEdge {
        /// The first node index of the duplicated connection.
        first: usize,
        /// The second node index of the duplicated connection.
        second: usize,
    },
    /// The graph violates an internal consistency invariant (broken index
    /// mapping, adjacency that does not match the edges, degree mismatch).
    InconsistentGraph {
        /// Human-readable description of the violated invariant.
        reason: String,
    },
}

impl fmt::Display for GraphError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidMolecule(error) => {
                write!(
                    f,
                    "source molecule is not a valid molecular structure: {error}"
                )
            }
            Self::InvalidNodeIndex { index, node_count } => {
                write!(
                    f,
                    "invalid node index {index}: graph has {node_count} node(s), valid indices are 0..{node_count}"
                )
            }
            Self::InvalidEdgeIndex { index, edge_count } => {
                write!(
                    f,
                    "invalid edge index {index}: graph has {edge_count} edge(s), valid indices are 0..{edge_count}"
                )
            }
            Self::SelfLoop { node_index } => {
                write!(
                    f,
                    "self-loop rejected: node {node_index} cannot bond to itself"
                )
            }
            Self::DuplicateEdge { first, second } => {
                write!(
                    f,
                    "duplicate edge rejected: nodes {first} and {second} are already connected"
                )
            }
            Self::InconsistentGraph { reason } => {
                write!(f, "inconsistent graph structure: {reason}")
            }
        }
    }
}

impl std::error::Error for GraphError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::InvalidMolecule(error) => Some(error),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mi_chemistry::ChemistryError;
    use std::error::Error;

    #[test]
    fn error_display_messages_are_descriptive() {
        let cases: [GraphError; 6] = [
            GraphError::InvalidMolecule(ChemistryError::InvalidMolecularStructure {
                reason: "molecule contains no atoms".to_string(),
            }),
            GraphError::InvalidNodeIndex {
                index: 4,
                node_count: 3,
            },
            GraphError::InvalidEdgeIndex {
                index: 1,
                edge_count: 1,
            },
            GraphError::SelfLoop { node_index: 2 },
            GraphError::DuplicateEdge {
                first: 0,
                second: 1,
            },
            GraphError::InconsistentGraph {
                reason: "degree mismatch".to_string(),
            },
        ];
        for error in cases {
            let message = error.to_string();
            assert!(message.len() > 10, "message too generic: {message:?}");
            assert!(!message.to_lowercase().contains("panic"));
        }
    }

    #[test]
    fn wrapped_molecule_error_is_reachable_as_source() {
        let chemistry_error = ChemistryError::InvalidMolecularStructure {
            reason: "broken".to_string(),
        };
        let error = GraphError::InvalidMolecule(chemistry_error.clone());
        assert_eq!(
            error.source().unwrap().to_string(),
            chemistry_error.to_string()
        );
    }

    #[test]
    fn error_implements_std_error() {
        fn assert_std_error<E: std::error::Error>(_: &E) {}
        assert_std_error(&GraphError::SelfLoop { node_index: 0 });
    }
}
