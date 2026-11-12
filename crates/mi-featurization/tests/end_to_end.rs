//! End-to-end featurization checks.
//!
//! Molecule → graph → tensors → readback must agree with the versioned
//! feature schemas, the geometry must match hand-computed distances, and
//! targets must pair with an entry from the dataset layer. Inline
//! fixtures only; everything runs on the CPU backend.

use mi_chemistry::{Atom, BondType, ChemicalElement, Molecule, Position};
use mi_data::{ElectronicProperties, Energy, MolecularEntry};
use mi_featurization::{Targets, bond_distances, distance_matrix, featurize_molecule};
use mi_tensor::Cpu;

fn device() -> mi_tensor::Device {
    Cpu::inference()
}

/// Water with positions: O at origin, two H at ±0.7572 y.
fn water() -> Molecule {
    let mut molecule = Molecule::new().with_name("water");
    let element = |symbol: &str| ChemicalElement::from_symbol(symbol).unwrap();
    molecule.add_atom(Atom::new(element("O")).with_position(Position::new(0.0, 0.0, 0.0).unwrap()));
    molecule
        .add_atom(Atom::new(element("H")).with_position(Position::new(0.0, 0.7572, 0.0).unwrap()));
    molecule
        .add_atom(Atom::new(element("H")).with_position(Position::new(0.0, -0.7572, 0.0).unwrap()));
    molecule.add_bond_between(0, 1, BondType::Single).unwrap();
    molecule.add_bond_between(0, 2, BondType::Single).unwrap();
    molecule
}

#[test]
fn molecule_to_tensors_matches_schema_vectors() {
    let tensors = featurize_molecule(&water(), &device()).unwrap();
    let nodes = tensors.nodes().to_vec().unwrap();
    // Node rows equal NodeFeatures::to_vector for each atom (f32-exact for
    // schema v1 columns).
    assert_eq!(nodes[0..6], [8.0, 0.0, 0.0, 0.0, 2.0, 2.0]);
    assert_eq!(nodes[6..12], [1.0, 0.0, 0.0, 0.0, 1.0, 1.0]);
    assert_eq!(nodes[12..18], [1.0, 0.0, 0.0, 0.0, 1.0, 1.0]);

    // Directed edges: forward then reverse per chemical bond.
    assert_eq!(tensors.edge_index(), &[[0, 1], [1, 0], [0, 2], [2, 0]]);
    let edges = tensors.edge_features().unwrap().to_vec().unwrap();
    for row in 0..4 {
        let start = row * 7;
        assert_eq!(edges[start..start + 7], [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0]);
    }
}

#[test]
fn geometry_agrees_with_hand_computed_distances() {
    let matrix = distance_matrix(&water(), &device()).unwrap();
    let values = matrix.to_vec().unwrap();
    assert_eq!(values[1], 0.7572); // O–H1
    assert_eq!(values[5], 1.5144); // H1–H2
    assert_eq!(bond_distances(&water()).unwrap(), vec![0.7572, 0.7572]);
}

#[test]
fn targets_pair_with_dataset_entries() {
    let entry = MolecularEntry::new(water()).unwrap().with_properties(
        ElectronicProperties::new()
            .with_homo(Energy::from_ev(-10.0).unwrap())
            .with_lumo(Energy::from_ev(2.0).unwrap()),
    );
    let targets = Targets::extract(&entry);
    assert_eq!(targets.gap, Some(12.0), "effective gap is derived");
    let (values, mask) = targets.to_tensors(&device()).unwrap();
    assert_eq!(values.to_vec().unwrap(), vec![-10.0, 2.0, 12.0]);
    assert_eq!(mask.to_vec().unwrap(), vec![1.0, 1.0, 1.0]);
}

#[test]
fn featurization_is_reproducible() {
    let molecule = water();
    let first = featurize_molecule(&molecule, &device()).unwrap();
    let second = featurize_molecule(&molecule, &device()).unwrap();
    assert_eq!(
        first.nodes().to_vec().unwrap(),
        second.nodes().to_vec().unwrap()
    );
    assert_eq!(first.edge_index(), second.edge_index());
    assert_eq!(
        first.edge_features().unwrap().to_vec().unwrap(),
        second.edge_features().unwrap().to_vec().unwrap()
    );
}

#[test]
fn methane_featurizes_end_to_end() {
    // CH4: four C–H bonds around a central carbon (toy geometry).
    let mut molecule = Molecule::new().with_name("methane");
    let element = |symbol: &str| ChemicalElement::from_symbol(symbol).unwrap();
    molecule.add_atom(Atom::new(element("C")).with_position(Position::new(0.0, 0.0, 0.0).unwrap()));
    for coordinates in [
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (-1.0, 0.0, 0.0),
    ] {
        molecule
            .add_atom(Atom::new(element("H")).with_position(
                Position::new(coordinates.0, coordinates.1, coordinates.2).unwrap(),
            ));
    }
    for neighbor in 1..=4 {
        molecule
            .add_bond_between(0, neighbor, BondType::Single)
            .unwrap();
    }

    let tensors = featurize_molecule(&molecule, &device()).unwrap();
    assert_eq!(tensors.nodes().shape().dims(), &[5, 6]);
    assert_eq!(tensors.directed_edge_count(), 8);
    assert_eq!(tensors.edge_features().unwrap().shape().dims(), &[8, 7]);

    let matrix = distance_matrix(&molecule, &device()).unwrap();
    let values = matrix.to_vec().unwrap();
    assert_eq!(values[1], 1.0); // C–H along x
    assert_eq!(values[7], 2.0f64.sqrt() as f32); // H1–H3 (x vs z axes)
}

#[test]
fn unplaced_molecule_rejects_geometry_but_featurizes() {
    // A topologically valid molecule without positions: graph tensors work
    // (topological schemas), geometry refuses with a typed error.
    let mut molecule = Molecule::new();
    let element = |symbol: &str| ChemicalElement::from_symbol(symbol).unwrap();
    molecule.add_atom(Atom::new(element("O")));
    molecule.add_atom(Atom::new(element("H")));
    molecule.add_bond_between(0, 1, BondType::Single).unwrap();

    let tensors = featurize_molecule(&molecule, &device()).unwrap();
    assert_eq!(tensors.nodes().shape().dims(), &[2, 6]);

    let error = distance_matrix(&molecule, &device())
        .map(|_| ())
        .unwrap_err();
    assert!(
        error.to_string().contains("no position"),
        "unexpected: {error}"
    );
}
