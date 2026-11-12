//! Error types for molecular file I/O.
//!
//! All fallible operations in this crate report failures through
//! [`IoError`]. Invalid input never panics and is never silently ignored;
//! wherever possible the error names the offending **file line** (1-based,
//! counting the whole file, not the record) so a failing dataset can be
//! debugged directly in its source file.

use mi_chemistry::ChemistryError;
use std::fmt;

/// Errors produced when parsing, writing, or importing molecular data.
#[derive(Debug, Clone, PartialEq)]
pub enum IoError {
    /// An input file/string contains no molecule where at least one was
    /// required.
    EmptyInput {
        /// What was being parsed, e.g. `"XYZ"` or `"SDF"`.
        context: &'static str,
    },
    /// A text line does not match the grammar of its format.
    InvalidLine {
        /// 1-based file line the failure was detected on.
        line: usize,
        /// Human-readable description of the expectation that failed.
        reason: String,
    },
    /// The XYZ atom-count line is not a non-negative integer.
    InvalidAtomCount {
        /// 1-based file line of the atom-count declaration.
        line: usize,
        /// The rejected text, verbatim.
        text: String,
    },
    /// The XYZ file declares more or fewer atoms than it provides.
    AtomCountMismatch {
        /// The atom count declared on the count line.
        declared: usize,
        /// The number of atom lines actually found.
        found: usize,
    },
    /// An element symbol does not name one of the 118 known elements.
    UnknownElement {
        /// The rejected symbol, verbatim.
        symbol: String,
        /// 1-based file line the symbol appeared on.
        line: usize,
    },
    /// A coordinate value is not a finite floating-point number.
    InvalidCoordinate {
        /// 1-based file line the value appeared on.
        line: usize,
        /// The rejected text, verbatim.
        value: String,
    },
    /// An atom has no position where the operation requires one (e.g.
    /// writing an XYZ file).
    MissingPosition {
        /// 0-based index of the position-less atom.
        atom_index: usize,
    },
    /// The SDF counts line does not follow the MDL V2000 layout.
    InvalidCountsLine {
        /// 1-based file line of the counts line.
        line: usize,
        /// Human-readable description of the expectation that failed.
        reason: String,
    },
    /// A molfile bond block uses a bond order this crate does not support.
    ///
    /// Supported codes: `1` (single), `2` (double), `3` (triple),
    /// `4` (aromatic). Codes `5`–`8` (dative, wildcard, and other MDL
    /// extensions) have no counterpart in the project's bond model.
    UnsupportedBondOrder {
        /// The rejected bond-order code.
        code: u8,
        /// 1-based file line of the bond block entry.
        line: usize,
    },
    /// A bond type cannot be represented in a written format.
    ///
    /// [`BondType::Unspecified`](mi_chemistry::BondType::Unspecified) has
    /// no MDL code; writing it would silently change its meaning, so it is
    /// rejected instead.
    UnwritableBondType {
        /// 0-based index of the offending bond.
        bond_index: usize,
    },
    /// A molecule record references an atom index outside its atom block.
    InvalidAtomReference {
        /// The rejected 0-based (in-file 1-based) atom index.
        index: usize,
        /// The number of atoms in the record's atom block.
        atom_count: usize,
        /// 1-based file line of the bond block entry.
        line: usize,
    },
    /// Building the molecule failed inside `mi-chemistry` (for example a
    /// duplicate bond in the record's bond block).
    Chemistry {
        /// The wrapped chemistry error.
        source: ChemistryError,
        /// 1-based file line where the offending data appeared.
        line: usize,
    },
    /// The dataset layer rejected a parsed molecule during import (for
    /// example an entry without a valid structure).
    DataRejected {
        /// The wrapped dataset-layer error.
        source: mi_data::DataError,
    },
    /// A file could not be read or written.
    File {
        /// The path involved.
        path: String,
        /// Human-readable description of the failure.
        reason: String,
    },
    /// A metadata field is empty or whitespace-only where content is
    /// required.
    InvalidMetadata {
        /// The rejected field name (e.g. `"source"`).
        field: &'static str,
    },
    /// A property value read from a data field is not a finite number.
    InvalidPropertyValue {
        /// The data-field name (e.g. `"homo"`).
        field: String,
        /// The rejected text, verbatim.
        value: String,
    },
}

impl fmt::Display for IoError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyInput { context } => {
                write!(f, "{context} input contains no molecule")
            }
            Self::InvalidLine { line, reason } => {
                write!(f, "line {line}: {reason}")
            }
            Self::InvalidAtomCount { line, text } => {
                write!(f, "line {line}: atom count {text:?} is not an integer")
            }
            Self::AtomCountMismatch {
                declared,
                found: actual,
            } => {
                write!(
                    f,
                    "atom count mismatch: header declares {declared} atoms, found {actual}"
                )
            }
            Self::UnknownElement { symbol, line } => {
                write!(f, "line {line}: unknown element symbol {symbol:?}")
            }
            Self::InvalidCoordinate { line, value } => {
                write!(
                    f,
                    "line {line}: coordinate {value:?} is not a finite number"
                )
            }
            Self::MissingPosition { atom_index } => {
                write!(
                    f,
                    "atom {atom_index} has no position, which this operation requires"
                )
            }
            Self::InvalidCountsLine { line, reason } => {
                write!(f, "line {line}: invalid counts line: {reason}")
            }
            Self::UnsupportedBondOrder { code, line } => {
                write!(
                    f,
                    "line {line}: unsupported bond order code {code} \
                     (supported: 1, 2, 3, 4)"
                )
            }
            Self::UnwritableBondType { bond_index } => {
                write!(
                    f,
                    "bond {bond_index} has type Unspecified, which has no \
                     writable representation in this format"
                )
            }
            Self::InvalidAtomReference {
                index,
                atom_count,
                line,
            } => {
                write!(
                    f,
                    "line {line}: bond references atom {index}, but the record \
                     has only {atom_count} atoms"
                )
            }
            Self::Chemistry { source, line } => {
                write!(f, "line {line}: {source}")
            }
            Self::DataRejected { source } => {
                write!(f, "dataset rejected this entry: {source}")
            }
            Self::File { path, reason } => {
                write!(f, "file {path:?}: {reason}")
            }
            Self::InvalidMetadata { field } => {
                write!(f, "invalid metadata: {field} must not be empty")
            }
            Self::InvalidPropertyValue { field, value } => {
                write!(f, "data field {field:?}: {value:?} is not a finite number")
            }
        }
    }
}

impl std::error::Error for IoError {}

impl From<ChemistryError> for IoError {
    /// Wraps a chemistry error **without** line information, for uses
    /// outside file parsing (e.g. metadata or import validation). Parsing
    /// code should prefer [`IoError::Chemistry`] to keep the line context.
    fn from(source: ChemistryError) -> Self {
        Self::Chemistry { source, line: 0 }
    }
}

impl From<mi_data::DataError> for IoError {
    /// Wraps a dataset-layer rejection (e.g. an entry built from an
    /// invalid molecule) so import and doctest code can use `?` across
    /// the boundary.
    fn from(source: mi_data::DataError) -> Self {
        Self::DataRejected { source }
    }
}

/// Result type used throughout this crate.
pub type IoResult<T> = Result<T, IoError>;

#[cfg(test)]
mod tests {
    use super::*;
    use mi_chemistry::ChemistryError;

    #[test]
    fn display_messages_name_lines_and_values() {
        let cases: [(IoError, &str); 5] = [
            (
                IoError::InvalidLine {
                    line: 3,
                    reason: "expected 4 fields".into(),
                },
                "line 3: expected 4 fields",
            ),
            (
                IoError::UnknownElement {
                    symbol: "Xx".into(),
                    line: 7,
                },
                "line 7: unknown element symbol \"Xx\"",
            ),
            (
                IoError::AtomCountMismatch {
                    declared: 3,
                    found: 2,
                },
                "atom count mismatch: header declares 3 atoms, found 2",
            ),
            (
                IoError::UnsupportedBondOrder { code: 6, line: 12 },
                "line 12: unsupported bond order code 6 (supported: 1, 2, 3, 4)",
            ),
            (
                IoError::InvalidPropertyValue {
                    field: "homo".into(),
                    value: "n/a".into(),
                },
                "data field \"homo\": \"n/a\" is not a finite number",
            ),
        ];
        for (error, expected) in cases {
            assert_eq!(error.to_string(), expected);
        }
    }

    #[test]
    fn chemistry_errors_are_std_errors() {
        fn assert_error<E: std::error::Error>(_: &E) {}
        assert_error(&IoError::Chemistry {
            source: ChemistryError::SelfBond { atom_index: 0 },
            line: 9,
        });
        assert_error(&IoError::EmptyInput { context: "XYZ" });
    }
}
