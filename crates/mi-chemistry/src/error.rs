//! Error types for chemical data structures.
//!
//! All fallible operations in this crate report failures through
//! [`ChemistryError`]. Invalid input never panics and is never silently
//! ignored.

use std::fmt;

/// Errors produced when constructing or validating chemical data.
#[derive(Debug, Clone, PartialEq)]
pub enum ChemistryError {
    /// An atomic number outside the range `1..=118` was requested.
    InvalidAtomicNumber {
        /// The rejected atomic number.
        value: u8,
    },
    /// No element matches the requested symbol.
    ///
    /// Element symbols are case-sensitive (`"Cl"`, not `"CL"` or `"cl"`).
    UnknownElementSymbol {
        /// The rejected symbol, verbatim.
        symbol: String,
    },
    /// An atom index does not reference an existing atom.
    InvalidAtomIndex {
        /// The rejected index.
        index: usize,
        /// The number of atoms available; valid indices are `0..atom_count`.
        atom_count: usize,
    },
    /// A bond index does not reference an existing bond.
    InvalidBondIndex {
        /// The rejected index.
        index: usize,
        /// The number of bonds available; valid indices are `0..bond_count`.
        bond_count: usize,
    },
    /// A bond was requested between an atom and itself.
    SelfBond {
        /// The atom index appearing on both sides of the bond.
        atom_index: usize,
    },
    /// A bond between the same unordered pair of atoms already exists.
    ///
    /// Molecules in this crate are simple graphs: at most one bond may
    /// connect two atoms, regardless of bond type.
    DuplicateBond {
        /// The first atom index of the rejected bond.
        first: usize,
        /// The second atom index of the rejected bond.
        second: usize,
    },
    /// An isotope mass number is physically impossible for the element.
    ///
    /// The mass number (protons + neutrons) can never be smaller than the
    /// atomic number (protons).
    InvalidIsotopeMassNumber {
        /// The rejected mass number.
        value: u16,
        /// The symbol of the element the isotope was specified for.
        element_symbol: &'static str,
        /// The smallest physically possible mass number (the atomic number).
        minimum_valid: u16,
    },
    /// A position component is not finite (NaN or infinite).
    ///
    /// Only finite Cartesian coordinates can enter the data model; this
    /// keeps every stored position comparable and serializable.
    InvalidPosition {
        /// The rejected component: `"x"`, `"y"`, or `"z"`.
        component: &'static str,
        /// The rejected component value, verbatim.
        value: f64,
    },
    /// The molecular structure violates a structural invariant.
    InvalidMolecularStructure {
        /// Human-readable description of the violated invariant.
        reason: String,
    },
}

impl fmt::Display for ChemistryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidAtomicNumber { value } => {
                write!(f, "invalid atomic number {value}: must be in 1..=118")
            }
            Self::UnknownElementSymbol { symbol } => {
                write!(
                    f,
                    "unknown element symbol {symbol:?} (symbols are case-sensitive, e.g. \"Cl\" not \"CL\")"
                )
            }
            Self::InvalidAtomIndex { index, atom_count } => {
                write!(
                    f,
                    "invalid atom index {index}: molecule has {atom_count} atom(s), valid indices are 0..{atom_count}"
                )
            }
            Self::InvalidBondIndex { index, bond_count } => {
                write!(
                    f,
                    "invalid bond index {index}: molecule has {bond_count} bond(s), valid indices are 0..{bond_count}"
                )
            }
            Self::SelfBond { atom_index } => {
                write!(
                    f,
                    "self-bond rejected: atom {atom_index} cannot bond to itself"
                )
            }
            Self::DuplicateBond { first, second } => {
                write!(
                    f,
                    "duplicate bond rejected: a bond between atoms {first} and {second} already exists"
                )
            }
            Self::InvalidIsotopeMassNumber {
                value,
                element_symbol,
                minimum_valid,
            } => {
                write!(
                    f,
                    "invalid isotope mass number {value} for element {element_symbol}: \
                     the mass number must be at least the atomic number ({minimum_valid})"
                )
            }
            Self::InvalidPosition { component, value } => {
                write!(
                    f,
                    "invalid position: {component} coordinate value {value} is not finite"
                )
            }
            Self::InvalidMolecularStructure { reason } => {
                write!(f, "invalid molecular structure: {reason}")
            }
        }
    }
}

impl std::error::Error for ChemistryError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_display_messages_are_descriptive() {
        let cases: [ChemistryError; 8] = [
            ChemistryError::InvalidAtomicNumber { value: 200 },
            ChemistryError::UnknownElementSymbol {
                symbol: "Xx".to_string(),
            },
            ChemistryError::InvalidAtomIndex {
                index: 5,
                atom_count: 3,
            },
            ChemistryError::InvalidBondIndex {
                index: 2,
                bond_count: 1,
            },
            ChemistryError::SelfBond { atom_index: 4 },
            ChemistryError::DuplicateBond {
                first: 0,
                second: 1,
            },
            ChemistryError::InvalidIsotopeMassNumber {
                value: 3,
                element_symbol: "C",
                minimum_valid: 6,
            },
            ChemistryError::InvalidMolecularStructure {
                reason: "molecule contains no atoms".to_string(),
            },
        ];
        for error in cases {
            let message = error.to_string();
            assert!(message.len() > 10, "message too generic: {message:?}");
            assert!(!message.to_lowercase().contains("panic"));
        }
    }

    #[test]
    fn error_implements_std_error() {
        fn assert_std_error<E: std::error::Error>(_: &E) {}
        assert_std_error(&ChemistryError::SelfBond { atom_index: 0 });
    }
}
