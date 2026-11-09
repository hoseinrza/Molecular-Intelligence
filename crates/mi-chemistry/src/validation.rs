//! Structural validation of molecular data.
//!
//! This module implements the structural invariants of a molecule:
//!
//! 1. A molecule must contain at least one atom (empty molecules are
//!    rejected explicitly rather than silently accepted).
//! 2. Every bond endpoint must reference an existing atom index.
//! 3. No bond may connect an atom to itself.
//! 4. At most one bond may connect a given unordered pair of atoms
//!    (duplicate bonds are rejected).
//!
//! These are *structural* rules only. Chemical valence rules (e.g. "carbon
//! has four bonds") are deliberately **not** enforced: ions, radicals,
//! hypervalent species, and other unusual structures are valid as long as
//! they satisfy the invariants above.

use crate::atom::Atom;
use crate::bond::Bond;
use crate::error::ChemistryError;
use std::collections::HashSet;

/// Validates the structural invariants of an atom list and bond list.
///
/// The duplicate-bond check runs in O(number of bonds) using a hash set of
/// canonical endpoint pairs.
pub(crate) fn validate_structure(atoms: &[Atom], bonds: &[Bond]) -> Result<(), ChemistryError> {
    // Invariant 1: empty molecules are handled explicitly.
    if atoms.is_empty() {
        return Err(ChemistryError::InvalidMolecularStructure {
            reason: "molecule contains no atoms".to_string(),
        });
    }

    let atom_count = atoms.len();
    let mut seen: HashSet<(usize, usize)> = HashSet::with_capacity(bonds.len());
    for bond in bonds {
        let (a, b) = bond.endpoints();
        // Invariant 2: endpoints must reference existing atoms.
        //
        // Defensive: `Molecule::add_bond` already enforces this, but
        // validation must not assume any particular construction path.
        if a >= atom_count {
            return Err(ChemistryError::InvalidAtomIndex {
                index: a,
                atom_count,
            });
        }
        if b >= atom_count {
            return Err(ChemistryError::InvalidAtomIndex {
                index: b,
                atom_count,
            });
        }
        // Invariant 3: no self-bonds (defensive, as above).
        if a == b {
            return Err(ChemistryError::SelfBond { atom_index: a });
        }
        // Invariant 4: no duplicate bonds (bonds are undirected; endpoints
        // are canonicalized at `Bond` construction).
        if !seen.insert((a, b)) {
            return Err(ChemistryError::DuplicateBond {
                first: a,
                second: b,
            });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bond::BondType;
    use crate::element::ChemicalElement;

    fn atoms(count: usize) -> Vec<Atom> {
        let carbon = ChemicalElement::from_symbol("C").unwrap();
        (0..count).map(|_| Atom::new(carbon)).collect()
    }

    #[test]
    fn accepts_a_valid_structure() {
        let atoms = atoms(3);
        let bonds = vec![
            Bond::new(0, 1, BondType::Single).unwrap(),
            Bond::new(1, 2, BondType::Double).unwrap(),
        ];
        assert_eq!(validate_structure(&atoms, &bonds), Ok(()));
    }

    #[test]
    fn rejects_an_empty_molecule_explicitly() {
        let result = validate_structure(&[], &[]);
        assert!(
            matches!(
                result,
                Err(ChemistryError::InvalidMolecularStructure { .. })
            ),
            "empty molecules must be rejected with an explicit error"
        );
    }

    #[test]
    fn rejects_bond_referencing_missing_atom() {
        let atoms = atoms(2);
        let bonds = vec![Bond::new(0, 2, BondType::Single).unwrap()];
        assert_eq!(
            validate_structure(&atoms, &bonds),
            Err(ChemistryError::InvalidAtomIndex {
                index: 2,
                atom_count: 2
            })
        );
    }

    #[test]
    fn self_bonds_cannot_reach_validation() {
        // The defensive self-bond branch in `validate_structure` is
        // unreachable through the public API: `Bond::new` rejects self-bonds
        // at construction (tested in `bond.rs`). This test pins that
        // guarantee.
        assert!(matches!(
            Bond::new(1, 1, BondType::Single),
            Err(ChemistryError::SelfBond { atom_index: 1 })
        ));
    }

    #[test]
    fn rejects_duplicate_bonds_in_any_order() {
        let atoms = atoms(2);
        let bonds = vec![
            Bond::new(0, 1, BondType::Single).unwrap(),
            Bond::new(1, 0, BondType::Double).unwrap(),
        ];
        assert_eq!(
            validate_structure(&atoms, &bonds),
            Err(ChemistryError::DuplicateBond {
                first: 0,
                second: 1
            })
        );
    }
}
