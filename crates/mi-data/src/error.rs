//! Error types for the scientific dataset layer.
//!
//! All fallible operations in this crate report failures through
//! [`DataError`]. Invalid input never panics and is never silently ignored.

use mi_chemistry::ChemistryError;
use std::fmt;

/// Errors produced when building or using a molecular dataset.
#[derive(Debug, Clone, PartialEq)]
pub enum DataError {
    /// A molecule failed its structural validation and cannot enter a
    /// dataset.
    ///
    /// The wrapped [`ChemistryError`] describes the exact violation and is
    /// reachable through [`std::error::Error::source`].
    InvalidMolecule(ChemistryError),
    /// An entry index does not reference an existing entry.
    InvalidEntryIndex {
        /// The rejected index.
        index: usize,
        /// The number of entries available; valid indices are
        /// `0..entry_count`.
        entry_count: usize,
    },
    /// An energy value is not finite (NaN or infinite).
    InvalidEnergy {
        /// The rejected value, verbatim.
        value: f64,
    },
    /// The dataset violates an internal consistency invariant.
    ///
    /// Defensive: entries validate their molecules at construction, so this
    /// is unreachable through normal use; it defines the contract for
    /// dataset validation.
    InconsistentDataset {
        /// Human-readable description of the violated invariant.
        reason: String,
    },
}

impl fmt::Display for DataError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidMolecule(error) => {
                write!(f, "molecule is not a valid molecular structure: {error}")
            }
            Self::InvalidEntryIndex { index, entry_count } => {
                write!(
                    f,
                    "invalid entry index {index}: dataset has {entry_count} entry(ies), \
                     valid indices are 0..{entry_count}"
                )
            }
            Self::InvalidEnergy { value } => {
                write!(f, "invalid energy value {value}: must be finite")
            }
            Self::InconsistentDataset { reason } => {
                write!(f, "inconsistent dataset: {reason}")
            }
        }
    }
}

impl std::error::Error for DataError {
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
        let cases: [DataError; 4] = [
            DataError::InvalidMolecule(ChemistryError::InvalidMolecularStructure {
                reason: "molecule contains no atoms".to_string(),
            }),
            DataError::InvalidEntryIndex {
                index: 5,
                entry_count: 2,
            },
            DataError::InvalidEnergy { value: f64::NAN },
            DataError::InconsistentDataset {
                reason: "broken invariant".to_string(),
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
        let error = DataError::InvalidMolecule(chemistry_error.clone());
        assert_eq!(
            error.source().unwrap().to_string(),
            chemistry_error.to_string()
        );
    }

    #[test]
    fn error_implements_std_error() {
        fn assert_std_error<E: std::error::Error>(_: &E) {}
        assert_std_error(&DataError::InvalidEnergy { value: 0.0 });
    }
}
