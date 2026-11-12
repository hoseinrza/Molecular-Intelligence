//! XYZ writer.
//!
//! Emits the canonical XYZ layout — atom count, comment, then
//! `Symbol x y z` — with six decimal places per coordinate. Round-trips
//! through [`parse_xyz`](super::parser::parse_xyz) are exact for values
//! that fit six decimals and within `1e-6` Å otherwise (see the
//! integration tests); six decimals is far below the positional accuracy
//! of any quantum-chemical source data this pipeline consumes.

use crate::error::{IoError, IoResult};
use mi_chemistry::Molecule;

/// Serializes one molecule into XYZ text.
///
/// The molecule's name (if any) becomes the comment line; an unnamed
/// molecule gets an empty comment. Every atom must carry a position — a
/// purely topological molecule cannot be serialized to XYZ without
/// silently losing information, so this is an error instead.
///
/// # Errors
///
/// Returns [`IoError::MissingPosition`] when any atom lacks a position.
pub fn write_xyz(molecule: &Molecule) -> IoResult<String> {
    let mut output = String::with_capacity(64 + molecule.atom_count() * 48);
    output.push_str(&molecule.atom_count().to_string());
    output.push('\n');
    output.push_str(molecule.name().unwrap_or(""));
    output.push('\n');
    for (index, atom) in molecule.atoms().iter().enumerate() {
        let (x, y, z) = atom
            .position()
            .map(mi_chemistry::Position::components)
            .ok_or(IoError::MissingPosition { atom_index: index })?;
        output.push_str(atom.symbol());
        for value in [x, y, z] {
            write_space_padded(&mut output, value);
        }
        output.push('\n');
    }
    Ok(output)
}

/// Writes one molecule to an XYZ file on disk.
///
/// # Errors
///
/// Returns [`IoError::File`] when the file cannot be written, plus every
/// error of [`write_xyz`].
pub fn write_xyz_file(molecule: &Molecule, path: impl AsRef<std::path::Path>) -> IoResult<()> {
    let path = path.as_ref();
    let text = write_xyz(molecule)?;
    std::fs::write(path, text).map_err(|error| IoError::File {
        path: path.display().to_string(),
        reason: error.to_string(),
    })
}

/// Appends one coordinate in the conventional `%12.6f` column.
fn write_space_padded(output: &mut String, value: f64) {
    output.push(' ');
    let rendered = format!("{value:.6}");
    for _ in rendered.len()..12 {
        output.push(' ');
    }
    output.push_str(&rendered);
}

#[cfg(test)]
mod tests {
    use super::*;
    use mi_chemistry::{Atom, ChemicalElement, Position};

    fn molecule_with_positions(name: Option<&str>) -> Molecule {
        let mut molecule = Molecule::new();
        if let Some(name) = name {
            molecule = molecule.with_name(name);
        }
        let hydrogen = ChemicalElement::from_symbol("H").unwrap();
        molecule
            .add_atom(Atom::new(hydrogen).with_position(Position::new(0.0, 0.0, -0.4692).unwrap()));
        molecule
    }

    #[test]
    fn writes_count_comment_and_atoms() {
        let molecule = molecule_with_positions(Some("test molecule"));
        let text = write_xyz(&molecule).unwrap();
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines[0], "1");
        assert_eq!(lines[1], "test molecule");
        assert!(lines[2].trim_start().starts_with("H"));
        assert!(lines[2].contains("-0.469200"), "got: {}", lines[2]);
    }

    #[test]
    fn unnamed_molecule_writes_empty_comment() {
        let text = write_xyz(&molecule_with_positions(None)).unwrap();
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines[1], "");
    }

    #[test]
    fn atom_without_position_is_an_error() {
        let mut molecule = Molecule::new();
        molecule.add_atom(Atom::new(ChemicalElement::from_symbol("H").unwrap()));
        let error = write_xyz(&molecule).unwrap_err();
        assert!(matches!(error, IoError::MissingPosition { atom_index: 0 }));
    }
}
