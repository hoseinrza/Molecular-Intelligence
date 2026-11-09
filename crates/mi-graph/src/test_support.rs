//! Shared test molecules built with the real `mi-chemistry` API.
//!
//! These builders mirror scientifically sensible species and exercise the
//! chemistry core exactly as an external user would, so graph tests always
//! operate on genuinely valid molecules.

use mi_chemistry::{Atom, Bond, BondType, ChemicalElement, Molecule};

/// Looks up an element by symbol (panics only on a test-data typo).
pub(crate) fn element(symbol: &str) -> ChemicalElement {
    ChemicalElement::from_symbol(symbol).unwrap()
}

/// A plain oxygen atom (water's central atom).
pub(crate) fn water_atom_oxygen() -> Atom {
    Atom::new(element("O"))
}

/// An aromatic bond between atoms 0 and 1.
pub(crate) fn aromatic_bond() -> Bond {
    Bond::new(0, 1, BondType::Aromatic).unwrap()
}

/// Water: O–H single bonds. Atom 0 is the oxygen, atoms 1 and 2 hydrogens.
pub(crate) fn water() -> Molecule {
    let mut molecule = Molecule::new()
        .with_name("water")
        .with_identifier("pubchem:962");
    molecule.add_atom(water_atom_oxygen());
    molecule.add_atom(Atom::new(element("H")));
    molecule.add_atom(Atom::new(element("H")));
    molecule.add_bond_between(0, 1, BondType::Single).unwrap();
    molecule.add_bond_between(0, 2, BondType::Single).unwrap();
    molecule
}

/// Methane: four equivalent C–H single bonds. Atom 0 is the carbon.
pub(crate) fn methane() -> Molecule {
    let mut molecule = Molecule::new()
        .with_name("methane")
        .with_identifier("pubchem:297");
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

/// Benzene: a six-membered aromatic ring. Atoms 0..=5 are carbons.
pub(crate) fn benzene() -> Molecule {
    let mut molecule = Molecule::new()
        .with_name("benzene")
        .with_identifier("pubchem:241");
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

/// A single isolated hydrogen atom (radical) with no bonds.
pub(crate) fn hydrogen_radical() -> Molecule {
    let mut molecule = Molecule::new().with_name("hydrogen radical");
    molecule.add_atom(Atom::new(element("H")));
    molecule
}

/// Two disconnected components: water (atoms 0–2) plus molecular hydrogen
/// (atoms 3–4).
pub(crate) fn water_and_hydrogen() -> Molecule {
    let mut molecule = Molecule::new().with_name("water + hydrogen");
    molecule.add_atom(water_atom_oxygen());
    molecule.add_atom(Atom::new(element("H")));
    molecule.add_atom(Atom::new(element("H")));
    molecule.add_bond_between(0, 1, BondType::Single).unwrap();
    molecule.add_bond_between(0, 2, BondType::Single).unwrap();
    molecule.add_atom(Atom::new(element("H")));
    molecule.add_atom(Atom::new(element("H")));
    molecule.add_bond_between(3, 4, BondType::Single).unwrap();
    molecule
}
