//! Round-trip verification for the supported formats.
//!
//! `parse(write(molecule))` must reproduce the chemical content within
//! each format's documented decimal precision: XYZ carries six decimals
//! (tolerance 1e-6 Å), molfiles the standard four (tolerance 1e-4 Å).
//! Elements, names, bond orders, formal charges, and data fields must be
//! preserved exactly.

use mi_chemistry::{Atom, BondType, ChemicalElement, Molecule, Position};
use mi_io::IoError;
use mi_io::sdf::{SdfRecord, parse_sdf, write_sdf_molecule, write_sdf_record, write_sdf_records};
use mi_io::xyz::{parse_xyz, write_xyz};
use std::collections::BTreeMap;

/// Ethanol: C–C, C–O, O–H skeleton with one double bond (C=C).
fn ethanol() -> Molecule {
    let mut molecule = Molecule::new().with_name("ethanol");
    let element = |symbol: &str| ChemicalElement::from_symbol(symbol).unwrap();
    molecule.add_atom(
        Atom::new(element("C")).with_position(Position::new(0.0000, 0.0000, 0.0000).unwrap()),
    );
    molecule.add_atom(
        Atom::new(element("C")).with_position(Position::new(1.5100, 0.1000, 0.2000).unwrap()),
    );
    molecule.add_atom(
        Atom::new(element("O")).with_position(Position::new(2.5000, -0.8500, -0.3500).unwrap()),
    );
    molecule.add_atom(
        Atom::new(element("O")).with_position(Position::new(0.1000, 1.3800, 0.6000).unwrap()),
    );
    molecule.add_bond_between(0, 1, BondType::Double).unwrap();
    molecule.add_bond_between(1, 2, BondType::Single).unwrap();
    molecule.add_bond_between(0, 3, BondType::Single).unwrap();
    molecule
}

fn assert_same_atoms(left: &Molecule, right: &Molecule, tolerance: f64) {
    assert_eq!(left.atom_count(), right.atom_count(), "atom counts differ");
    for (index, (a, b)) in left.atoms().iter().zip(right.atoms()).enumerate() {
        assert_eq!(a.symbol(), b.symbol(), "atom {index} element differs");
        assert_eq!(
            a.formal_charge(),
            b.formal_charge(),
            "atom {index} charge differs"
        );
        match (a.position(), b.position()) {
            (Some(p), Some(q)) => {
                for (axis, (u, v)) in [
                    ("x", (p.x(), q.x())),
                    ("y", (p.y(), q.y())),
                    ("z", (p.z(), q.z())),
                ] {
                    assert!((u - v).abs() < tolerance, "atom {index} {axis}: {u} vs {v}");
                }
            }
            (None, None) => {}
            _ => panic!("atom {index} position presence differs"),
        }
    }
}

fn assert_same_bonds(left: &Molecule, right: &Molecule) {
    assert_eq!(left.bond_count(), right.bond_count(), "bond counts differ");
    for (index, (a, b)) in left.bonds().iter().zip(right.bonds()).enumerate() {
        assert_eq!(
            a.endpoints(),
            b.endpoints(),
            "bond {index} endpoints differ"
        );
        assert_eq!(a.bond_type(), b.bond_type(), "bond {index} type differs");
    }
}

#[test]
fn xyz_round_trip_preserves_structure() {
    let original = ethanol();
    let text = write_xyz(&original).unwrap();
    let parsed = parse_xyz(&text).unwrap();
    assert_eq!(parsed.name(), Some("ethanol"));
    assert_same_atoms(&original, &parsed, 1e-6);
    // XYZ carries no bonds; the round trip cannot preserve them.
    assert_eq!(parsed.bond_count(), 0);
}

#[test]
fn xyz_round_trip_of_a_monoatomic_molecule() {
    let mut molecule = Molecule::new().with_name("single argon");
    molecule.add_atom(
        Atom::new(ChemicalElement::from_symbol("Ar").unwrap())
            .with_position(Position::new(-1.25, 2.5, 0.0).unwrap()),
    );
    let text = write_xyz(&molecule).unwrap();
    let parsed = parse_xyz(&text).unwrap();
    assert_eq!(parsed.name(), Some("single argon"));
    assert_same_atoms(&molecule, &parsed, 1e-6);
}

#[test]
fn xyz_round_trip_of_an_unnamed_molecule() {
    let mut molecule = Molecule::new();
    molecule.add_atom(
        Atom::new(ChemicalElement::from_symbol("Ne").unwrap())
            .with_position(Position::new(0.0, 0.0, 0.0).unwrap()),
    );
    let text = write_xyz(&molecule).unwrap();
    let parsed = parse_xyz(&text).unwrap();
    assert_eq!(parsed.name(), None);
    assert_same_atoms(&molecule, &parsed, 1e-6);
}

#[test]
fn sdf_round_trip_preserves_structure_and_fields() {
    let mut fields = BTreeMap::new();
    fields.insert("homo".to_string(), "-0.3133".to_string());
    fields.insert("gap".to_string(), "0.3764".to_string());
    let record = SdfRecord::new(ethanol(), fields);
    let text = write_sdf_record(&record).unwrap();
    let parsed = parse_sdf(&text).unwrap();
    assert_eq!(parsed.len(), 1);
    let molecule = parsed[0].molecule();
    assert_eq!(molecule.name(), Some("ethanol"));
    assert_same_atoms(record.molecule(), molecule, 1e-4);
    assert_same_bonds(record.molecule(), molecule);
    assert_eq!(parsed[0].data_field("homo"), Some("-0.3133"));
    assert_eq!(parsed[0].data_field("gap"), Some("0.3764"));
}

#[test]
fn sdf_multi_record_round_trip_is_stable() {
    let records = vec![
        SdfRecord::new(ethanol(), BTreeMap::new()),
        SdfRecord::new(ethanol().with_name("second copy"), BTreeMap::new()),
    ];
    let text = write_sdf_records(&records).unwrap();
    let parsed = parse_sdf(&text).unwrap();
    assert_eq!(parsed.len(), 2);
    assert_eq!(parsed[0].molecule().name(), Some("ethanol"));
    assert_eq!(parsed[1].molecule().name(), Some("second copy"));
    for (original, record) in records.iter().zip(&parsed) {
        assert_same_atoms(original.molecule(), record.molecule(), 1e-4);
        assert_same_bonds(original.molecule(), record.molecule());
    }
}

#[test]
fn sdf_round_trip_rejects_unwritable_bonds() {
    let mut molecule = Molecule::new();
    let c = ChemicalElement::from_symbol("C").unwrap();
    molecule.add_atom(Atom::new(c).with_position(Position::new(0.0, 0.0, 0.0).unwrap()));
    molecule.add_atom(Atom::new(c).with_position(Position::new(1.0, 0.0, 0.0).unwrap()));
    molecule
        .add_bond_between(0, 1, BondType::Unspecified)
        .unwrap();
    let error = write_sdf_molecule(&molecule).unwrap_err();
    assert!(matches!(error, IoError::UnwritableBondType { .. }));
}

#[test]
fn files_round_trip_through_disk() {
    // Use the test's own target directory; no external fixtures needed.
    let base = std::env::temp_dir().join(format!("mi_io_round_trip_{}", std::process::id()));
    std::fs::create_dir_all(&base).unwrap();

    let molecule = ethanol();
    let xyz_path = base.join("ethanol.xyz");
    mi_io::write_xyz_file(&molecule, &xyz_path).unwrap();
    let from_xyz = mi_io::parse_xyz_file(&xyz_path).unwrap();
    assert_same_atoms(&molecule, &from_xyz, 1e-6);

    let record = SdfRecord::new(molecule, BTreeMap::new());
    let sdf_path = base.join("ethanol.sdf");
    mi_io::write_sdf_file(std::slice::from_ref(&record), &sdf_path).unwrap();
    let from_sdf = mi_io::parse_sdf_file(&sdf_path).unwrap();
    assert_eq!(from_sdf.len(), 1);
    assert_same_atoms(record.molecule(), from_sdf[0].molecule(), 1e-4);

    std::fs::remove_dir_all(&base).unwrap();
}
