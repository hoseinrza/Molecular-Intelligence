//! Bond representation.

use crate::error::ChemistryError;

/// The order or character of a bond between two atoms.
///
/// # Semantics and limitations
///
/// - [`BondType::Single`], [`BondType::Double`], and [`BondType::Triple`]
///   denote the classical covalent bond orders.
/// - [`BondType::Aromatic`] marks a bond that participates in a delocalized
///   aromatic π system (as in benzene or naphthalene). It does **not** encode
///   a Kekulé structure: which bonds of the ring would alternate single/double
///   is deliberately left unspecified, and no resonance or kekulization logic
///   exists in this crate. An aromatic bond carries no fixed length or order.
/// - [`BondType::Unspecified`] marks a bond whose order is unknown or not yet
///   assigned (e.g. while assembling a structure from partial data). It is a
///   valid state, not a default: bonds must choose their type explicitly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BondType {
    /// A single covalent bond.
    Single,
    /// A double covalent bond.
    Double,
    /// A triple covalent bond.
    Triple,
    /// A delocalized aromatic bond; see the type documentation for
    /// semantics and limitations.
    Aromatic,
    /// A bond of unknown or not-yet-assigned order.
    Unspecified,
}

/// A bond between two atoms of a molecule, identified by atom indices.
///
/// Bonds are **undirected**: the pair `(a, b)` is the same bond as
/// `(b, a)`. Endpoints are stored in canonical (smaller, larger) order at
/// construction, so equality, hashing, and duplicate detection are
/// order-independent:
///
/// ```
/// use mi_chemistry::{Bond, BondType};
///
/// let ab = Bond::new(0, 1, BondType::Single)?;
/// let ba = Bond::new(1, 0, BondType::Single)?;
/// assert_eq!(ab, ba);
/// assert_eq!(ab.endpoints(), (0, 1));
/// # Ok::<(), mi_chemistry::ChemistryError>(())
/// ```
///
/// A bond never connects an atom to itself; such bonds are rejected at
/// construction. Whether the referenced atoms exist is a property of the
/// molecule the bond is added to (see
/// [`Molecule::add_bond`](crate::Molecule::add_bond)).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Bond {
    first: usize,
    second: usize,
    bond_type: BondType,
}

impl Bond {
    /// Creates a bond between the atoms at `first` and `second`.
    ///
    /// The endpoints are stored in canonical (smaller, larger) order;
    /// construction is order-independent.
    ///
    /// # Errors
    ///
    /// Returns [`ChemistryError::SelfBond`] if `first == second`: an atom
    /// cannot bond to itself.
    pub fn new(first: usize, second: usize, bond_type: BondType) -> Result<Self, ChemistryError> {
        if first == second {
            return Err(ChemistryError::SelfBond { atom_index: first });
        }
        Ok(Self {
            first: first.min(second),
            second: second.max(first),
            bond_type,
        })
    }

    /// Returns the smaller of the two atom indices (canonical order).
    pub fn first(&self) -> usize {
        self.first
    }

    /// Returns the larger of the two atom indices (canonical order).
    pub fn second(&self) -> usize {
        self.second
    }

    /// Returns both endpoints in canonical (smaller, larger) order.
    pub fn endpoints(&self) -> (usize, usize) {
        (self.first, self.second)
    }

    /// Returns the type of this bond.
    pub fn bond_type(&self) -> BondType {
        self.bond_type
    }

    /// Returns `true` if this bond connects the two given atom indices, in
    /// any order.
    pub fn connects(&self, a: usize, b: usize) -> bool {
        self.first == a.min(b) && self.second == a.max(b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_bond_is_stored() {
        let bond = Bond::new(0, 1, BondType::Single).unwrap();
        assert_eq!(bond.bond_type(), BondType::Single);
        assert_eq!(bond.endpoints(), (0, 1));
    }

    #[test]
    fn double_bond_is_stored() {
        let bond = Bond::new(0, 1, BondType::Double).unwrap();
        assert_eq!(bond.bond_type(), BondType::Double);
    }

    #[test]
    fn triple_bond_is_stored() {
        let bond = Bond::new(0, 1, BondType::Triple).unwrap();
        assert_eq!(bond.bond_type(), BondType::Triple);
    }

    #[test]
    fn aromatic_bond_is_stored() {
        let bond = Bond::new(0, 1, BondType::Aromatic).unwrap();
        assert_eq!(bond.bond_type(), BondType::Aromatic);
    }

    #[test]
    fn unspecified_bond_is_stored() {
        let bond = Bond::new(0, 1, BondType::Unspecified).unwrap();
        assert_eq!(bond.bond_type(), BondType::Unspecified);
    }

    #[test]
    fn self_bond_is_rejected() {
        assert_eq!(
            Bond::new(2, 2, BondType::Single),
            Err(ChemistryError::SelfBond { atom_index: 2 })
        );
    }

    #[test]
    fn endpoints_are_canonicalized() {
        let bond = Bond::new(3, 1, BondType::Double).unwrap();
        assert_eq!(bond.first(), 1);
        assert_eq!(bond.second(), 3);
        assert_eq!(bond.endpoints(), (1, 3));
        // Order-independent equality and hashing.
        assert_eq!(
            Bond::new(1, 3, BondType::Double),
            Bond::new(3, 1, BondType::Double)
        );
        assert_ne!(
            Bond::new(1, 3, BondType::Double),
            Bond::new(1, 3, BondType::Triple)
        );
    }

    #[test]
    fn connects_is_symmetric() {
        let bond = Bond::new(1, 4, BondType::Single).unwrap();
        assert!(bond.connects(1, 4));
        assert!(bond.connects(4, 1));
        assert!(!bond.connects(1, 2));
        assert!(!bond.connects(0, 4));
        // An atom never "connects to itself" through a valid bond.
        assert!(!bond.connects(1, 1));
    }
}
