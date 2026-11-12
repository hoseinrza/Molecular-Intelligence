//! Molecule representation.

use crate::atom::Atom;
use crate::bond::{Bond, BondType};
use crate::error::ChemistryError;
use crate::validation;

/// A molecular structure: a collection of [`Atom`]s connected by [`Bond`]s.
///
/// # Indexing model
///
/// - Atoms and bonds are stored in insertion order; [`add_atom`](Self::add_atom)
///   and [`add_bond`](Self::add_bond) return the index of the inserted item.
/// - **Indices are stable** for the lifetime of the molecule. This phase has
///   no removal operations, so indices never shift; atom or bond removal is a
///   planned extension that will require explicit index remapping.
/// - Bond endpoints reference atom indices and are validated on insertion.
/// - At most one bond may connect a given unordered pair of atoms;
///   duplicates are rejected (`DuplicateBond`). Molecules are simple graphs,
///   not multi-graphs — parallel edges are a possible future extension.
///
/// # Chemical neutrality of the model
///
/// No assumption is made about net charge or valence: ions, radicals,
/// hypervalent species, under-coordinated atoms, and fragments are all
/// structurally valid as long as the invariants checked by
/// [`validate`](Self::validate) hold.
///
/// # Example
///
/// ```
/// use mi_chemistry::{Atom, BondType, ChemicalElement, Molecule};
///
/// let mut water = Molecule::new().with_name("water").with_identifier("pubchem:962");
/// let oxygen = ChemicalElement::from_symbol("O")?;
/// let hydrogen = ChemicalElement::from_symbol("H")?;
///
/// let o = water.add_atom(Atom::new(oxygen));
/// let h1 = water.add_atom(Atom::new(hydrogen));
/// let h2 = water.add_atom(Atom::new(hydrogen));
///
/// water.add_bond_between(o, h1, BondType::Single)?;
/// water.add_bond_between(o, h2, BondType::Single)?;
///
/// assert_eq!(water.atom_count(), 3);
/// assert_eq!(water.bond_count(), 2);
/// assert_eq!(water.atom(0)?.symbol(), "O");
/// water.validate()?;
/// # Ok::<(), mi_chemistry::ChemistryError>(())
/// ```
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Molecule {
    name: Option<String>,
    identifier: Option<String>,
    atoms: Vec<Atom>,
    bonds: Vec<Bond>,
}

impl Molecule {
    /// Creates an empty molecule with no name and no identifier.
    ///
    /// An empty molecule is a legitimate starting point for incremental
    /// construction, but it is *not* structurally valid:
    /// [`validate`](Self::validate) rejects it explicitly until it contains
    /// at least one atom (see [`is_empty`](Self::is_empty)).
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the molecule name and returns the molecule.
    ///
    /// The name is free-form (e.g. `"water"`, `"aspirin"`).
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Sets the molecular identifier and returns the molecule.
    ///
    /// The identifier is free-form bookkeeping (e.g. a database reference
    /// such as `"pubchem:962"` or an internal dataset key). It is never
    /// interpreted chemically.
    pub fn with_identifier(mut self, identifier: impl Into<String>) -> Self {
        self.identifier = Some(identifier.into());
        self
    }

    /// Adds an atom and returns its index.
    ///
    /// Indices are assigned in insertion order starting at 0 and remain
    /// stable for the lifetime of the molecule.
    pub fn add_atom(&mut self, atom: Atom) -> usize {
        self.atoms.push(atom);
        self.atoms.len() - 1
    }

    /// Adds a previously constructed bond and returns its index.
    ///
    /// The bond's endpoints are checked against the current atom list, and
    /// duplicate bonds (same unordered atom pair, regardless of bond type)
    /// are rejected.
    ///
    /// # Errors
    ///
    /// - [`ChemistryError::InvalidAtomIndex`] if either endpoint does not
    ///   reference an existing atom.
    /// - [`ChemistryError::DuplicateBond`] if a bond between the same
    ///   unordered pair of atoms already exists.
    ///
    /// Self-bonds cannot occur here: [`Bond::new`] rejects them at
    /// construction.
    pub fn add_bond(&mut self, bond: Bond) -> Result<usize, ChemistryError> {
        let atom_count = self.atoms.len();
        let (a, b) = bond.endpoints();
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
        if self
            .bonds
            .iter()
            .any(|existing| existing.endpoints() == (a, b))
        {
            return Err(ChemistryError::DuplicateBond {
                first: a,
                second: b,
            });
        }
        self.bonds.push(bond);
        Ok(self.bonds.len() - 1)
    }

    /// Convenience wrapper: creates a bond between `first` and `second` and
    /// adds it, returning the new bond index.
    ///
    /// # Errors
    ///
    /// In addition to the errors of [`add_bond`](Self::add_bond), returns
    /// [`ChemistryError::SelfBond`] if `first == second`.
    pub fn add_bond_between(
        &mut self,
        first: usize,
        second: usize,
        bond_type: BondType,
    ) -> Result<usize, ChemistryError> {
        let bond = Bond::new(first, second, bond_type)?;
        self.add_bond(bond)
    }

    /// Returns the atom at `index`.
    ///
    /// # Errors
    ///
    /// Returns [`ChemistryError::InvalidAtomIndex`] if `index` is out of
    /// range.
    pub fn atom(&self, index: usize) -> Result<&Atom, ChemistryError> {
        self.atoms
            .get(index)
            .ok_or(ChemistryError::InvalidAtomIndex {
                index,
                atom_count: self.atoms.len(),
            })
    }

    /// Returns the bond at `index`.
    ///
    /// # Errors
    ///
    /// Returns [`ChemistryError::InvalidBondIndex`] if `index` is out of
    /// range.
    pub fn bond(&self, index: usize) -> Result<&Bond, ChemistryError> {
        self.bonds
            .get(index)
            .ok_or(ChemistryError::InvalidBondIndex {
                index,
                bond_count: self.bonds.len(),
            })
    }

    /// Returns the number of atoms.
    pub fn atom_count(&self) -> usize {
        self.atoms.len()
    }

    /// Returns the number of bonds.
    pub fn bond_count(&self) -> usize {
        self.bonds.len()
    }

    /// Returns `true` if the molecule contains no atoms.
    pub fn is_empty(&self) -> bool {
        self.atoms.is_empty()
    }

    /// Returns all atoms in index order.
    pub fn atoms(&self) -> &[Atom] {
        &self.atoms
    }

    /// Returns all bonds in index order.
    pub fn bonds(&self) -> &[Bond] {
        &self.bonds
    }

    /// Returns the molecule name, if set.
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    /// Returns the molecular identifier, if set.
    pub fn identifier(&self) -> Option<&str> {
        self.identifier.as_deref()
    }

    /// Validates the structural invariants of the molecule:
    ///
    /// 1. The molecule is not empty.
    /// 2. Every bond endpoint references an existing atom.
    /// 3. No bond connects an atom to itself.
    /// 4. No duplicate bonds exist.
    ///
    /// These are structural rules only: valence rules are deliberately not
    /// enforced (see the crate documentation).
    ///
    /// # Errors
    ///
    /// Returns [`ChemistryError::InvalidMolecularStructure`] for an empty
    /// molecule and the specific [`ChemistryError`] variants for any
    /// violated invariant.
    pub fn validate(&self) -> Result<(), ChemistryError> {
        validation::validate_structure(&self.atoms, &self.bonds)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::element::ChemicalElement;

    fn element(symbol: &str) -> ChemicalElement {
        ChemicalElement::from_symbol(symbol).unwrap()
    }

    /// Water: O–H single bonds. Atom 0 is the oxygen, atoms 1 and 2 are
    /// hydrogens.
    fn water() -> Molecule {
        let mut molecule = Molecule::new()
            .with_name("water")
            .with_identifier("pubchem:962");
        molecule.add_atom(Atom::new(element("O")));
        molecule.add_atom(Atom::new(element("H")));
        molecule.add_atom(Atom::new(element("H")));
        molecule.add_bond_between(0, 1, BondType::Single).unwrap();
        molecule.add_bond_between(0, 2, BondType::Single).unwrap();
        molecule
    }

    /// Methane: four equivalent C–H single bonds. Atom 0 is the carbon.
    fn methane() -> Molecule {
        let mut molecule = Molecule::new()
            .with_name("methane")
            .with_identifier("pubchem:297");
        molecule.add_atom(Atom::new(element("C")));
        molecule.add_atom(Atom::new(element("H")));
        molecule.add_atom(Atom::new(element("H")));
        molecule.add_atom(Atom::new(element("H")));
        molecule.add_atom(Atom::new(element("H")));
        molecule.add_bond_between(0, 1, BondType::Single).unwrap();
        molecule.add_bond_between(0, 2, BondType::Single).unwrap();
        molecule.add_bond_between(0, 3, BondType::Single).unwrap();
        molecule.add_bond_between(0, 4, BondType::Single).unwrap();
        molecule
    }

    /// Carbon dioxide: two C=O double bonds. Atom 0 is the carbon.
    fn carbon_dioxide() -> Molecule {
        let mut molecule = Molecule::new()
            .with_name("carbon dioxide")
            .with_identifier("pubchem:280");
        molecule.add_atom(Atom::new(element("C")));
        molecule.add_atom(Atom::new(element("O")));
        molecule.add_atom(Atom::new(element("O")));
        molecule.add_bond_between(0, 1, BondType::Double).unwrap();
        molecule.add_bond_between(0, 2, BondType::Double).unwrap();
        molecule
    }

    /// Benzene: a six-membered aromatic ring. Atoms 0..=6 are carbons.
    fn benzene() -> Molecule {
        let mut molecule = Molecule::new()
            .with_name("benzene")
            .with_identifier("pubchem:241");
        for _ in 0..6 {
            molecule.add_atom(Atom::new(element("C")));
        }
        for i in 0..6 {
            molecule
                .add_bond_between(i, (i + 1) % 6, BondType::Aromatic)
                .unwrap();
        }
        molecule
    }

    #[test]
    fn empty_molecule_is_created_explicitly() {
        let molecule = Molecule::new();
        assert!(molecule.is_empty());
        assert_eq!(molecule.atom_count(), 0);
        assert_eq!(molecule.bond_count(), 0);
        assert_eq!(molecule.name(), None);
        assert_eq!(molecule.identifier(), None);
        // Empty molecules are explicitly invalid, not silently accepted.
        assert!(matches!(
            molecule.validate(),
            Err(ChemistryError::InvalidMolecularStructure { .. })
        ));
    }

    #[test]
    fn default_equals_new() {
        assert_eq!(Molecule::default(), Molecule::new());
    }

    #[test]
    fn adding_atoms_returns_stable_indices() {
        let mut molecule = Molecule::new();
        assert_eq!(molecule.add_atom(Atom::new(element("O"))), 0);
        assert_eq!(molecule.add_atom(Atom::new(element("H"))), 1);
        assert_eq!(molecule.add_atom(Atom::new(element("H"))), 2);
        assert_eq!(molecule.atom_count(), 3);
        assert_eq!(molecule.atom(0).unwrap().symbol(), "O");
        assert_eq!(molecule.atom(1).unwrap().symbol(), "H");
        assert_eq!(molecule.atom(2).unwrap().symbol(), "H");
    }

    #[test]
    fn indices_remain_stable_as_the_molecule_grows() {
        let mut molecule = water();
        let first = molecule.atom(0).unwrap().symbol();
        let bond_count = molecule.bond_count();
        molecule.add_atom(Atom::new(element("Na")).with_formal_charge(1));
        // Pre-existing atoms and bonds are untouched by the insertion.
        assert_eq!(molecule.atom(0).unwrap().symbol(), first);
        assert_eq!(molecule.bond_count(), bond_count);
        assert_eq!(molecule.atom(3).unwrap().formal_charge(), 1);
        molecule.validate().unwrap();
    }

    #[test]
    fn water_is_valid() {
        let molecule = water();
        assert_eq!(molecule.name(), Some("water"));
        assert_eq!(molecule.identifier(), Some("pubchem:962"));
        molecule.validate().unwrap();
        assert_eq!(molecule.atom_count(), 3);
        assert_eq!(molecule.bond_count(), 2);
        for bond in molecule.bonds() {
            assert_eq!(bond.bond_type(), BondType::Single);
            assert_eq!(bond.first(), 0, "both bonds connect to the oxygen");
        }
        // Molecular mass from element averages: 15.999 + 2 * 1.008.
        let total: f64 = molecule.atoms().iter().map(Atom::atomic_mass).sum();
        assert!((total - 18.015).abs() < 1e-9);
    }

    #[test]
    fn methane_is_valid() {
        let molecule = methane();
        molecule.validate().unwrap();
        assert_eq!(molecule.atom_count(), 5);
        assert_eq!(molecule.bond_count(), 4);
        let total: f64 = molecule.atoms().iter().map(Atom::atomic_mass).sum();
        assert!((total - (12.011 + 4.0 * 1.008)).abs() < 1e-9);
    }

    #[test]
    fn carbon_dioxide_is_valid() {
        let molecule = carbon_dioxide();
        molecule.validate().unwrap();
        assert_eq!(molecule.bond_count(), 2);
        assert_eq!(molecule.bond(0).unwrap().bond_type(), BondType::Double);
        assert_eq!(molecule.bond(1).unwrap().bond_type(), BondType::Double);
    }

    #[test]
    fn benzene_with_aromatic_bonds_is_valid() {
        let molecule = benzene();
        molecule.validate().unwrap();
        assert_eq!(molecule.atom_count(), 6);
        assert_eq!(molecule.bond_count(), 6);
        for bond in molecule.bonds() {
            assert_eq!(bond.bond_type(), BondType::Aromatic);
        }
        // The ring is closed: each atom participates in exactly two bonds.
        for i in 0..6 {
            let count = molecule
                .bonds()
                .iter()
                .filter(|b| b.connects(i, (i + 1) % 6) || b.connects(i, (i + 5) % 6))
                .count();
            assert_eq!(count, 2, "atom {i} must have two aromatic bonds");
        }
    }

    #[test]
    fn bond_with_invalid_atom_reference_is_rejected() {
        let mut molecule = water();
        // Bond::new succeeds (indices are just numbers), the molecule rejects them.
        let bond = Bond::new(0, 5, BondType::Single).unwrap();
        assert_eq!(
            molecule.add_bond(bond),
            Err(ChemistryError::InvalidAtomIndex {
                index: 5,
                atom_count: 3
            })
        );
        assert_eq!(
            molecule.add_bond_between(7, 0, BondType::Single),
            Err(ChemistryError::InvalidAtomIndex {
                index: 7,
                atom_count: 3
            })
        );
    }

    #[test]
    fn duplicate_bond_is_rejected_regardless_of_order_and_type() {
        let mut molecule = water();
        // Same pair, same type.
        assert_eq!(
            molecule.add_bond_between(0, 1, BondType::Single),
            Err(ChemistryError::DuplicateBond {
                first: 0,
                second: 1
            })
        );
        // Same pair, reversed order.
        assert_eq!(
            molecule.add_bond_between(1, 0, BondType::Single),
            Err(ChemistryError::DuplicateBond {
                first: 0,
                second: 1
            })
        );
        // Same pair, different type: still a duplicate (simple graph).
        assert_eq!(
            molecule.add_bond_between(0, 1, BondType::Double),
            Err(ChemistryError::DuplicateBond {
                first: 0,
                second: 1
            })
        );
    }

    #[test]
    fn self_bond_is_rejected_by_the_molecule() {
        let mut molecule = water();
        assert_eq!(
            molecule.add_bond_between(1, 1, BondType::Single),
            Err(ChemistryError::SelfBond { atom_index: 1 })
        );
    }

    #[test]
    fn atom_and_bond_access_reports_out_of_range_indices() {
        let molecule = water();
        assert_eq!(
            molecule.atom(3),
            Err(ChemistryError::InvalidAtomIndex {
                index: 3,
                atom_count: 3
            })
        );
        assert_eq!(
            molecule.bond(2),
            Err(ChemistryError::InvalidBondIndex {
                index: 2,
                bond_count: 2
            })
        );
        assert!(molecule.atom(2).is_ok());
        assert!(molecule.bond(1).is_ok());
    }

    #[test]
    fn iteration_over_atoms_and_bonds() {
        let molecule = carbon_dioxide();
        let symbols: Vec<&str> = molecule.atoms().iter().map(Atom::symbol).collect();
        assert_eq!(symbols, ["C", "O", "O"]);
        let endpoints: Vec<(usize, usize)> = molecule.bonds().iter().map(Bond::endpoints).collect();
        assert_eq!(endpoints, [(0, 1), (0, 2)]);
        let double_count = molecule
            .bonds()
            .iter()
            .filter(|b| b.bond_type() == BondType::Double)
            .count();
        assert_eq!(double_count, 2);
    }

    #[test]
    fn charged_and_unusual_species_are_structurally_valid() {
        // Hydronium (H3O+) and a bare radical hydrogen fragment must both be
        // structurally valid: the model does not assume neutral or
        // valence-satisfied chemistry.
        let mut hydronium = Molecule::new().with_name("hydronium");
        hydronium.add_atom(Atom::new(element("O")).with_formal_charge(1));
        hydronium.add_atom(Atom::new(element("H")));
        hydronium.add_atom(Atom::new(element("H")));
        hydronium.add_atom(Atom::new(element("H")));
        hydronium.add_bond_between(0, 1, BondType::Single).unwrap();
        hydronium.add_bond_between(0, 2, BondType::Single).unwrap();
        hydronium.add_bond_between(0, 3, BondType::Single).unwrap();
        hydronium.validate().unwrap();

        let mut radical = Molecule::new().with_name("hydrogen radical");
        radical.add_atom(Atom::new(element("H")).with_formal_charge(0));
        radical.validate().unwrap();
    }

    #[test]
    fn molecules_compare_by_value() {
        assert_eq!(water(), water());
        let mut modified = water();
        modified.add_atom(Atom::new(element("H")));
        assert_ne!(water(), modified);
        assert_ne!(water(), methane());
    }
}
