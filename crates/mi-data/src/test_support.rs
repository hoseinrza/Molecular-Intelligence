//! Shared test molecules and property fixtures built with the real crate APIs.
//!
//! Property values are **synthetic** (chosen for exact arithmetic and
//! plausible magnitudes in eV); they are test fixtures, not reference data.

use crate::property::ElectronicProperties;
use mi_chemistry::{Atom, BondType, ChemicalElement, Molecule};

/// Looks up an element by symbol (panics only on a test-data typo).
pub(crate) fn element(symbol: &str) -> ChemicalElement {
    ChemicalElement::from_symbol(symbol).unwrap()
}

/// Water: O–H single bonds. Atom 0 is the oxygen, atoms 1 and 2 hydrogens.
pub(crate) fn water() -> Molecule {
    let mut molecule = Molecule::new().with_name("water");
    molecule.add_atom(Atom::new(element("O")));
    molecule.add_atom(Atom::new(element("H")));
    molecule.add_atom(Atom::new(element("H")));
    molecule.add_bond_between(0, 1, BondType::Single).unwrap();
    molecule.add_bond_between(0, 2, BondType::Single).unwrap();
    molecule
}

/// Methane: four equivalent C–H single bonds. Atom 0 is the carbon.
pub(crate) fn methane() -> Molecule {
    let mut molecule = Molecule::new().with_name("methane");
    molecule.add_atom(Atom::new(element("C")));
    for _ in 0..4 {
        molecule.add_atom(Atom::new(element("H")));
    }
    for hydrogen in 1..=4 {
        molecule
            .add_bond_between(0, hydrogen, BondType::Single)
            .unwrap();
    }
    molecule
}

/// Carbon dioxide: two C=O double bonds. Atom 0 is the carbon.
pub(crate) fn carbon_dioxide() -> Molecule {
    let mut molecule = Molecule::new().with_name("carbon dioxide");
    molecule.add_atom(Atom::new(element("C")));
    molecule.add_atom(Atom::new(element("O")));
    molecule.add_atom(Atom::new(element("O")));
    molecule.add_bond_between(0, 1, BondType::Double).unwrap();
    molecule.add_bond_between(0, 2, BondType::Double).unwrap();
    molecule
}

/// Benzene: a six-membered aromatic ring.
pub(crate) fn benzene() -> Molecule {
    let mut molecule = Molecule::new().with_name("benzene");
    for _ in 0..6 {
        molecule.add_atom(Atom::new(element("C")));
    }
    for carbon in 0..6 {
        molecule
            .add_bond_between(carbon, (carbon + 1) % 6, BondType::Aromatic)
            .unwrap();
    }
    molecule
}

/// Synthetic HOMO/LUMO values (eV); the gap is left for derivation.
pub(crate) fn properties(homo_ev: f64, lumo_ev: f64) -> ElectronicProperties {
    ElectronicProperties::new()
        .with_homo(crate::property::Energy::from_ev(homo_ev).unwrap())
        .with_lumo(crate::property::Energy::from_ev(lumo_ev).unwrap())
}
