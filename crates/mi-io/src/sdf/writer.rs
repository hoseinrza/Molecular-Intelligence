//! SDF / MDL molfile V2000 writer.
//!
//! Emits the canonical V2000 layout: title/program/comment lines, counts
//! line, atom block (coordinates in the standard `%10.4f` columns), bond
//! block, `M  CHG` lines for formal charges, `M  END`, optional data
//! fields, and the `$$$$` record separator.
//!
//! # Round-trip contract
//!
//! `parse_sdf(write_sdf_record(record))` preserves the molecule name, the
//! element sequence, formal charges, bond orders, data fields, and all
//! coordinates within `1e-4` Å (the format's standard four decimal
//! places — see the integration tests).
//!
//! [`BondType::Unspecified`](mi_chemistry::BondType::Unspecified) has no
//! MDL bond-order code; writing it would silently change its meaning, so
//! it is rejected ([`IoError::UnwritableBondType`]).

use super::parser::SdfRecord;
use crate::error::{IoError, IoResult};
use mi_chemistry::{BondType, Molecule};

/// Serializes one record (molecule + data fields) into SDF text,
/// terminated by `$$$$`.
///
/// # Errors
///
/// Returns [`IoError::UnwritableBondType`] when a bond cannot be
/// represented, and [`IoError::MissingPosition`] when an atom lacks a
/// position.
pub fn write_sdf_record(record: &SdfRecord) -> IoResult<String> {
    let mut output = String::new();
    write_molecule_block(&record.molecule, &mut output)?;
    for tag in record.data_fields.keys() {
        let value = record.data_field(tag).unwrap_or("");
        output.push_str("> <");
        output.push_str(tag);
        output.push_str(">\n");
        output.push_str(value);
        output.push('\n');
        output.push('\n');
    }
    output.push_str("$$$$\n");
    Ok(output)
}

/// Serializes several records into one multi-record SDF document.
///
/// # Errors
///
/// Same as [`write_sdf_record`], applied per record.
pub fn write_sdf_records(records: &[SdfRecord]) -> IoResult<String> {
    let mut output = String::new();
    for record in records {
        output.push_str(&write_sdf_record(record)?);
    }
    Ok(output)
}

/// Serializes one bare molecule into SDF text (no data fields).
///
/// # Errors
///
/// Same as [`write_sdf_record`].
pub fn write_sdf_molecule(molecule: &Molecule) -> IoResult<String> {
    let mut output = String::new();
    write_molecule_block(molecule, &mut output)?;
    output.push_str("$$$$\n");
    Ok(output)
}

/// Writes one molecule's molfile block (everything up to and including
/// `M  END`).
fn write_molecule_block(molecule: &Molecule, output: &mut String) -> IoResult<()> {
    // Title / program / comment lines.
    output.push_str(molecule.name().unwrap_or(""));
    output.push('\n');
    output.push_str("  mi-io        \n");
    output.push('\n');

    // Counts line: atom and bond counts, standard zero padding.
    output.push_str(&format!(
        "{:>3}{:>3}  0  0  0  0  0  0  0  0999 V2000\n",
        molecule.atom_count(),
        molecule.bond_count()
    ));

    // Atom block: x/y/z in the standard 10.4 columns, symbol in 32–34.
    for (index, atom) in molecule.atoms().iter().enumerate() {
        let (x, y, z) = atom
            .position()
            .map(mi_chemistry::Position::components)
            .ok_or(IoError::MissingPosition { atom_index: index })?;
        output.push_str(&format!(
            "{x:>10.4}{y:>10.4}{z:>10.4} {:<3} 0  0  0  0  0  0  0  0  0  0  0  0\n",
            atom.symbol()
        ));
    }

    // Bond block: two 1-based atom indices plus the order code.
    for (index, bond) in molecule.bonds().iter().enumerate() {
        let code = match bond.bond_type() {
            BondType::Single => 1,
            BondType::Double => 2,
            BondType::Triple => 3,
            BondType::Aromatic => 4,
            BondType::Unspecified => {
                return Err(IoError::UnwritableBondType { bond_index: index });
            }
        };
        output.push_str(&format!(
            "{:>3}{:>3}{:>3}  0  0  0  0\n",
            bond.first() + 1,
            bond.second() + 1,
            code
        ));
    }

    // Formal charges as property lines (up to 8 pairs per line).
    let charged: Vec<(usize, i8)> = molecule
        .atoms()
        .iter()
        .enumerate()
        .filter(|(_, atom)| atom.formal_charge() != 0)
        .map(|(index, atom)| (index, atom.formal_charge()))
        .collect();
    for chunk in charged.chunks(8) {
        output.push_str(&format!("M  CHG{:>3}", chunk.len()));
        for (index, charge) in chunk {
            output.push_str(&format!("{:>4}{:>4}", index + 1, charge));
        }
        output.push('\n');
    }

    output.push_str("M  END\n");
    Ok(())
}

/// Writes records to an SDF file on disk.
///
/// # Errors
///
/// Returns [`IoError::File`] when the file cannot be written, plus every
/// error of [`write_sdf_records`].
pub fn write_sdf_file(records: &[SdfRecord], path: impl AsRef<std::path::Path>) -> IoResult<()> {
    let path = path.as_ref();
    let text = write_sdf_records(records)?;
    std::fs::write(path, text).map_err(|error| IoError::File {
        path: path.display().to_string(),
        reason: error.to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use mi_chemistry::{Atom, ChemicalElement, Position};
    use std::collections::BTreeMap;

    fn water() -> Molecule {
        let mut molecule = Molecule::new().with_name("water");
        let o = ChemicalElement::from_symbol("O").unwrap();
        let h = ChemicalElement::from_symbol("H").unwrap();
        molecule.add_atom(Atom::new(o).with_position(Position::new(0.0, 0.0, 0.1173).unwrap()));
        molecule.add_atom(Atom::new(h).with_position(Position::new(0.0, 0.7572, -0.4692).unwrap()));
        molecule
            .add_atom(Atom::new(h).with_position(Position::new(0.0, -0.7572, -0.4692).unwrap()));
        molecule.add_bond_between(0, 1, BondType::Single).unwrap();
        molecule.add_bond_between(0, 2, BondType::Single).unwrap();
        molecule
    }

    #[test]
    fn writes_canonical_v2000_layout() {
        let text = write_sdf_molecule(&water()).unwrap();
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines[0], "water");
        assert_eq!(lines[3], "  3  2  0  0  0  0  0  0  0  0999 V2000");
        assert!(
            lines[4].starts_with("    0.0000    0.0000    0.1173 O"),
            "atom line: {}",
            lines[4]
        );
        assert!(lines[7].starts_with("  1  2  1"), "bond line: {}", lines[7]);
        assert_eq!(lines[9], "M  END");
        assert_eq!(lines[10], "$$$$");
    }

    #[test]
    fn writes_and_rereads_formal_charges() {
        let mut hydroxide = Molecule::new().with_name("hydroxide");
        hydroxide.add_atom(
            Atom::new(ChemicalElement::from_symbol("O").unwrap())
                .with_formal_charge(-1)
                .with_position(Position::new(0.0, 0.0, 0.0).unwrap()),
        );
        hydroxide.add_atom(
            Atom::new(ChemicalElement::from_symbol("H").unwrap())
                .with_position(Position::new(1.0, 0.0, 0.0).unwrap()),
        );
        hydroxide.add_bond_between(0, 1, BondType::Single).unwrap();

        let text = write_sdf_molecule(&hydroxide).unwrap();
        assert!(text.contains("M  CHG"), "charge line missing: {text}");
        let parsed = super::super::parser::parse_sdf(&text).unwrap();
        let molecule = parsed[0].molecule();
        assert_eq!(molecule.atom(0).unwrap().formal_charge(), -1);
        assert_eq!(molecule.atom(1).unwrap().formal_charge(), 0);
    }

    #[test]
    fn data_fields_survive_a_write() {
        let mut fields = BTreeMap::new();
        fields.insert("homo".to_string(), "-10.2755".to_string());
        fields.insert("note".to_string(), "kept verbatim".to_string());
        let record = SdfRecord {
            molecule: water(),
            data_fields: fields,
        };
        let text = write_sdf_record(&record).unwrap();
        assert!(text.contains("> <homo>\n-10.2755\n\n"));
        assert!(text.contains("> <note>\nkept verbatim\n\n"));
        let parsed = super::super::parser::parse_sdf(&text).unwrap();
        assert_eq!(parsed[0].data_field("homo"), Some("-10.2755"));
    }

    #[test]
    fn unspecified_bonds_cannot_be_written() {
        let mut molecule = Molecule::new();
        molecule.add_atom(
            Atom::new(ChemicalElement::from_symbol("C").unwrap())
                .with_position(Position::new(0.0, 0.0, 0.0).unwrap()),
        );
        molecule.add_atom(
            Atom::new(ChemicalElement::from_symbol("C").unwrap())
                .with_position(Position::new(1.0, 0.0, 0.0).unwrap()),
        );
        molecule
            .add_bond_between(0, 1, BondType::Unspecified)
            .unwrap();
        let error = write_sdf_molecule(&molecule).unwrap_err();
        assert!(matches!(
            error,
            IoError::UnwritableBondType { bond_index: 0 }
        ));
    }

    #[test]
    fn multiple_records_are_separated() {
        let records = vec![
            SdfRecord {
                molecule: water(),
                data_fields: BTreeMap::new(),
            },
            SdfRecord {
                molecule: water(),
                data_fields: BTreeMap::new(),
            },
        ];
        let text = write_sdf_records(&records).unwrap();
        let separator_count = text.lines().filter(|line| *line == "$$$$").count();
        assert_eq!(separator_count, 2);
        let parsed = super::super::parser::parse_sdf(&text).unwrap();
        assert_eq!(parsed.len(), 2);
    }

    #[test]
    fn bond_endpoints_are_one_based_and_ascending() {
        let mut molecule = Molecule::new();
        let c = ChemicalElement::from_symbol("C").unwrap();
        molecule.add_atom(Atom::new(c).with_position(Position::new(0.0, 0.0, 0.0).unwrap()));
        molecule.add_atom(Atom::new(c).with_position(Position::new(1.0, 0.0, 0.0).unwrap()));
        molecule.add_bond_between(1, 0, BondType::Double).unwrap();
        let text = write_sdf_molecule(&molecule).unwrap();
        // Canonical order (1, 2), one-based, code 2 for a double bond.
        assert!(text.contains("  1  2  2"), "bond line missing: {text}");
    }
}
