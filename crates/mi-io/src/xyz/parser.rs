//! XYZ parser.
//!
//! Implements the de-facto XYZ format: an atom-count line, a comment line,
//! then one `Symbol x y z` line per atom (coordinates in **Ångström**).
//!
//! # Accepted grammar
//!
//! - Leading blank lines are skipped; the first non-blank line must be a
//!   non-negative integer ≥ 1 (the atom count).
//! - The second line is the comment; non-empty comments become the
//!   molecule name (trimmed).
//! - Each atom line must contain **at least four** whitespace-separated
//!   fields: `Symbol x y z`. Extra trailing fields (charges, masses, …
//!   extended-XYZ columns) are ignored — they carry quantities this phase
//!   does not model, and their presence never changes the parsed
//!   structure.
//! - Anything after the declared number of atom lines (other than blank
//!   lines) is rejected: a trailing second molecule in one file is a data
//!   error here, not a silent truncation.
//!
//! Single-atom files (monoatomic species) are valid. One file always
//! yields exactly one molecule; trajectories are a planned extension.

use crate::error::{IoError, IoResult};
use mi_chemistry::{Atom, ChemicalElement, Molecule, Position};

/// Parses one XYZ document into a validated [`Molecule`].
///
/// The molecule carries the comment line as its name (when non-empty) and
/// every atom's Ångström position. Structural validation is implicit: an
/// XYZ file has no bonds, and a molecule with atoms but no bonds is valid.
///
/// # Errors
///
/// Reports [`IoError::InvalidAtomCount`], [`IoError::AtomCountMismatch`],
/// [`IoError::UnknownElement`], [`IoError::InvalidCoordinate`],
/// [`IoError::InvalidLine`], or [`IoError::EmptyInput`] with the 1-based
/// file line number wherever possible.
pub fn parse_xyz(input: &str) -> IoResult<Molecule> {
    // Enumerate every physical line so errors can point at the file.
    let mut lines = input.lines().enumerate().peekable();

    // Skip leading blank lines; the atom count must follow.
    let count_line = loop {
        match lines.next() {
            Some((_, text)) if text.trim().is_empty() => continue,
            Some((number, text)) => break (number, text),
            None => return Err(IoError::EmptyInput { context: "XYZ" }),
        }
    };

    let declared = count_line
        .1
        .trim()
        .parse::<usize>()
        .map_err(|_| IoError::InvalidAtomCount {
            line: count_line.0 + 1,
            text: count_line.1.trim().to_string(),
        })?;
    if declared == 0 {
        return Err(IoError::InvalidLine {
            line: count_line.0 + 1,
            reason: "atom count must be at least 1".into(),
        });
    }

    // The comment line may itself be blank (an empty title).
    let comment = lines
        .next()
        .map(|(_, text)| text)
        .ok_or(IoError::AtomCountMismatch { declared, found: 0 })?;
    let title = comment.trim();

    let mut molecule = Molecule::new();
    if !title.is_empty() {
        molecule = molecule.with_name(title);
    }

    let mut found = 0usize;
    for (number, text) in lines {
        if text.trim().is_empty() {
            continue;
        }
        if found == declared {
            // More atom lines than declared: reject instead of truncating.
            return Err(IoError::AtomCountMismatch {
                declared,
                found: found + 1,
            });
        }
        let atom = parse_atom_line(text, number + 1)?;
        molecule.add_atom(atom);
        found += 1;
    }
    if found != declared {
        return Err(IoError::AtomCountMismatch { declared, found });
    }
    Ok(molecule)
}

/// Parses one `Symbol x y z` atom line.
fn parse_atom_line(text: &str, line: usize) -> IoResult<Atom> {
    let fields: Vec<&str> = text.split_whitespace().collect();
    if fields.len() < 4 {
        return Err(IoError::InvalidLine {
            line,
            reason: format!(
                "an atom line needs `Symbol x y z` ({} field(s) found)",
                fields.len()
            ),
        });
    }
    let element = ChemicalElement::from_symbol(fields[0]).map_err(|_| IoError::UnknownElement {
        symbol: fields[0].to_string(),
        line,
    })?;
    let x = parse_coordinate(fields[1], line)?;
    let y = parse_coordinate(fields[2], line)?;
    let z = parse_coordinate(fields[3], line)?;
    // All components are finite, so Position::new cannot fail here.
    let position = Position::new(x, y, z).map_err(|source| IoError::Chemistry { source, line })?;
    Ok(Atom::new(element).with_position(position))
}

/// Parses one coordinate, rejecting non-finite values explicitly.
fn parse_coordinate(text: &str, line: usize) -> IoResult<f64> {
    let value: f64 = text.parse().map_err(|_| IoError::InvalidCoordinate {
        line,
        value: text.to_string(),
    })?;
    if !value.is_finite() {
        return Err(IoError::InvalidCoordinate {
            line,
            value: text.to_string(),
        });
    }
    Ok(value)
}

/// Reads and parses an XYZ file from disk.
///
/// # Errors
///
/// Returns [`IoError::File`] when the file cannot be read, plus every
/// error of [`parse_xyz`].
pub fn parse_xyz_file(path: impl AsRef<std::path::Path>) -> IoResult<Molecule> {
    let path = path.as_ref();
    let input = std::fs::read_to_string(path).map_err(|error| IoError::File {
        path: path.display().to_string(),
        reason: error.to_string(),
    })?;
    parse_xyz(&input)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_water_with_name_and_positions() {
        let input = "3\nwater\nO 0.000000 0.000000 0.117300\n\
                     H 0.000000 0.757200 -0.469200\nH 0.000000 -0.757200 -0.469200\n";
        let molecule = parse_xyz(input).unwrap();
        assert_eq!(molecule.name(), Some("water"));
        assert_eq!(molecule.atom_count(), 3);
        assert_eq!(molecule.bond_count(), 0);
        assert_eq!(molecule.atom(0).unwrap().symbol(), "O");
        let (x, y, z) = molecule.atom(0).unwrap().position().unwrap().components();
        assert_eq!((x, y, z), (0.0, 0.0, 0.1173));
        let hydrogen = molecule.atom(1).unwrap().position().unwrap();
        assert_eq!(hydrogen.y(), 0.7572);
    }

    #[test]
    fn parses_monoatomic_molecule() {
        let molecule = parse_xyz("1\nhelium atom\nHe 0.0 0.0 0.0\n").unwrap();
        assert_eq!(molecule.name(), Some("helium atom"));
        assert_eq!(molecule.atom_count(), 1);
        assert_eq!(molecule.atom(0).unwrap().symbol(), "He");
    }

    #[test]
    fn blank_comment_yields_unnamed_molecule() {
        let molecule = parse_xyz("1\n\nNe 1.0 2.0 3.0\n").unwrap();
        assert_eq!(molecule.name(), None);
        let (x, y, z) = molecule.atom(0).unwrap().position().unwrap().components();
        assert_eq!((x, y, z), (1.0, 2.0, 3.0));
    }

    #[test]
    fn leading_blank_lines_are_skipped() {
        let molecule = parse_xyz("\n\n2\ntwo atoms\nH 0 0 0\nH 0 0 1\n").unwrap();
        assert_eq!(molecule.atom_count(), 2);
        assert_eq!(molecule.name(), Some("two atoms"));
    }

    #[test]
    fn trailing_extra_columns_are_ignored() {
        let molecule = parse_xyz("1\nextended-ish\nH 0.0 0.0 0.0 0.5 12.01\n").unwrap();
        assert_eq!(molecule.atom(0).unwrap().symbol(), "H");
        assert_eq!(
            molecule.atom(0).unwrap().position().unwrap().components(),
            (0.0, 0.0, 0.0)
        );
    }

    #[test]
    fn rejects_non_integer_atom_count() {
        let error = parse_xyz("three\nwater\n").unwrap_err();
        assert_eq!(
            error,
            IoError::InvalidAtomCount {
                line: 1,
                text: "three".into(),
            }
        );
    }

    #[test]
    fn rejects_zero_atom_count() {
        let error = parse_xyz("0\nnothing\n").unwrap_err();
        assert!(
            error.to_string().contains("at least 1"),
            "unexpected: {error}"
        );
    }

    #[test]
    fn rejects_missing_comment_line() {
        let error = parse_xyz("2\nH 0.0 0.0 0.0\n").unwrap_err();
        assert!(matches!(error, IoError::AtomCountMismatch { .. }));
    }

    #[test]
    fn rejects_fewer_atoms_than_declared() {
        let error = parse_xyz("3\nwater\nO 0.0 0.0 0.0\nH 0.0 0.0 1.0\n").unwrap_err();
        assert_eq!(
            error,
            IoError::AtomCountMismatch {
                declared: 3,
                found: 2,
            }
        );
    }

    #[test]
    fn rejects_more_atoms_than_declared() {
        let error = parse_xyz("1\ntrailing\nH 0.0 0.0 0.0\nH 0.0 0.0 1.0\n").unwrap_err();
        assert_eq!(
            error,
            IoError::AtomCountMismatch {
                declared: 1,
                found: 2,
            }
        );
    }

    #[test]
    fn rejects_unknown_element() {
        let error = parse_xyz("1\nbad\nXx 0.0 0.0 0.0\n").unwrap_err();
        assert_eq!(
            error,
            IoError::UnknownElement {
                symbol: "Xx".into(),
                line: 3,
            }
        );
    }

    #[test]
    fn rejects_non_numeric_coordinate() {
        let error = parse_xyz("1\nbad\nH zero 0.0 0.0\n").unwrap_err();
        assert_eq!(
            error,
            IoError::InvalidCoordinate {
                line: 3,
                value: "zero".into(),
            }
        );
    }

    #[test]
    fn rejects_non_finite_coordinate() {
        for text in ["inf", "-inf", "nan"] {
            let input = format!("1\nbad\nH {text} 0.0 0.0\n");
            let error = parse_xyz(&input).unwrap_err();
            assert!(matches!(error, IoError::InvalidCoordinate { .. }), "{text}");
        }
    }

    #[test]
    fn rejects_atom_line_with_missing_fields() {
        let error = parse_xyz("1\nbad\nH 0.0 0.0\n").unwrap_err();
        assert!(matches!(error, IoError::InvalidLine { .. }));
        assert!(
            error.to_string().contains("Symbol x y z"),
            "unexpected: {error}"
        );
    }

    #[test]
    fn empty_input_is_rejected() {
        for input in ["", "\n\n"] {
            let error = parse_xyz(input).unwrap_err();
            assert_eq!(error, IoError::EmptyInput { context: "XYZ" }, "{input:?}");
        }
    }
}
