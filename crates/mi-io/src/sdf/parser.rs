//! SDF / MDL molfile V2000 parser.
//!
//! Implements the parts of the MDL Structure Data Format this phase
//! needs, which is also the subset real datasets (e.g. QM9) ship with:
//!
//! - **Multi-record files** — records are separated by `$$$$` lines.
//! - **Atom block** — element, Cartesian position in Ångström, parsed
//!   by fixed columns with a whitespace-token fallback for files whose
//!   columns have drifted.
//! - **Bond block** — order codes `1` (single), `2` (double), `3`
//!   (triple), `4` (aromatic); mapped onto
//!   [`BondType`](mi_chemistry::BondType). Codes `5`–`8` (dative and
//!   other MDL extensions) are rejected rather than guessed.
//! - **Charges** — `M  CHG` property lines set per-atom formal charges.
//! - **Data fields** — `> <TAG>` blocks after `M  END` become the
//!   record's data fields, which is how dataset import attaches
//!   electronic properties.
//!
//! # Deliberate limitations
//!
//! V3000 molfiles, stereo descriptors, atom/bond query features, isotopes
//! (`M  ISO`), and alias lines are not supported: encountering them is a
//! reported error, never a silent simplification.
//!
//! # Column strategy
//!
//! Counts, atom, and bond lines are read by their fixed V2000 column
//! widths first. When a line is too short or fails to parse by columns
//! (some producers emit space-separated fields), the parser retries with
//! whitespace tokens before reporting an error.

use crate::error::{IoError, IoResult};
use mi_chemistry::{Atom, BondType, ChemicalElement, Molecule, Position};
use std::collections::BTreeMap;

/// One parsed SDF record: a [`Molecule`] plus its raw data fields.
///
/// Data fields keep their verbatim string values; interpreting a field as
/// a physical quantity (e.g. an energy) happens in the import layer, where
/// the dataset metadata declares units.
#[derive(Debug, Clone, PartialEq)]
pub struct SdfRecord {
    pub(crate) molecule: Molecule,
    pub(crate) data_fields: BTreeMap<String, String>,
}

impl SdfRecord {
    /// Creates a record from a molecule and its data fields.
    ///
    /// Construction is primarily a parser's job; this constructor exists
    /// for code that assembles records to write (exports, test fixtures).
    pub fn new(molecule: Molecule, data_fields: BTreeMap<String, String>) -> Self {
        Self {
            molecule,
            data_fields,
        }
    }

    /// The parsed molecule (name from the record's title line).
    pub fn molecule(&self) -> &Molecule {
        &self.molecule
    }

    /// The raw data fields of the record, keyed by tag name.
    ///
    /// Multi-line values keep their newlines; whitespace is otherwise
    /// verbatim.
    pub fn data_field(&self, tag: &str) -> Option<&str> {
        self.data_fields.get(tag).map(String::as_str)
    }

    /// All data-field tags in file order (alphabetical, as stored).
    pub fn data_field_tags(&self) -> impl Iterator<Item = &str> + '_ {
        self.data_fields.keys().map(String::as_str)
    }
}

/// Parses an SDF document containing one or more `$$$$`-separated records.
///
/// # Errors
///
/// Returns [`IoError::EmptyInput`] when the document holds no record and
/// a per-record error (with 1-based file line numbers) for the first
/// malformed record. Whitespace-only blocks between `$$$$` separators are
/// skipped; any non-empty block that fails to parse is a data error.
pub fn parse_sdf(input: &str) -> IoResult<Vec<SdfRecord>> {
    let mut records = Vec::new();
    let mut current: Vec<(usize, &str)> = Vec::new();
    for (number, text) in input.lines().enumerate() {
        if text.trim() == "$$$$" {
            if !current.iter().all(|(_, line)| line.trim().is_empty()) {
                records.push(parse_record(&current)?);
            }
            current.clear();
        } else {
            current.push((number, text));
        }
    }
    // A trailing record may lack the final `$$$$` separator.
    if !current.iter().all(|(_, line)| line.trim().is_empty()) {
        records.push(parse_record(&current)?);
    }
    if records.is_empty() {
        return Err(IoError::EmptyInput { context: "SDF" });
    }
    Ok(records)
}

/// Parses one record's lines into a molecule with data fields.
///
/// `lines` carries `(0-based file line number, text)` pairs so every
/// reported error points at the real file position.
fn parse_record(lines: &[(usize, &str)]) -> IoResult<SdfRecord> {
    if lines.len() < 4 {
        return Err(IoError::InvalidLine {
            line: lines.first().map_or(0, |(number, _)| number + 1),
            reason: "a record needs title, program, comment, and counts lines".into(),
        });
    }

    let title = lines[0].1.trim();
    let mut molecule = Molecule::new();
    if !title.is_empty() {
        molecule = molecule.with_name(title);
    }

    let counts_line_number = lines[3].0 + 1;
    let (atom_count, bond_count) = parse_counts_line(lines[3].1, counts_line_number)?;

    // Atoms and bonds are collected before `Molecule::add_atom/add_bond`
    // so `M  CHG` lines (which follow the atom block) can be applied at
    // construction time — atoms are immutable once added.
    let mut atoms = Vec::with_capacity(atom_count);
    let mut bonds = Vec::with_capacity(bond_count);
    for atom_line in &lines[4..4 + atom_count] {
        atoms.push(parse_atom_line(atom_line.1, atom_line.0 + 1)?);
    }
    for bond_line in &lines[4 + atom_count..4 + atom_count + bond_count] {
        bonds.push(parse_bond_line(bond_line.1, bond_line.0 + 1)?);
    }

    // The remainder of the record: property block (`M ...` lines), the
    // `M  END` terminator, then data fields.
    // Each charge entry carries its `M  CHG` line so out-of-range atom
    // references can be reported with a file position instead of being
    // silently dropped.
    let mut formal_charges: BTreeMap<usize, (i8, usize)> = BTreeMap::new();
    let mut cursor = 4 + atom_count + bond_count;
    loop {
        let Some(&(number, text)) = lines.get(cursor) else {
            return Err(IoError::InvalidLine {
                line: lines.last().unwrap().0 + 1,
                reason: "missing `M  END` terminator".into(),
            });
        };
        let trimmed = text.trim_end();
        if trimmed.starts_with("M  END") {
            cursor += 1;
            break;
        }
        if trimmed.starts_with("M  CHG") {
            apply_charge_line(trimmed, number + 1, &mut formal_charges)?;
        }
        cursor += 1;
    }
    for (atom_index, (_, charge_line)) in &formal_charges {
        if *atom_index >= atom_count {
            return Err(IoError::InvalidAtomReference {
                index: *atom_index,
                atom_count,
                line: *charge_line,
            });
        }
    }

    for (index, (element, position)) in atoms.iter().enumerate() {
        let mut atom = Atom::new(*element).with_position(*position);
        if let Some(&(charge, _)) = formal_charges.get(&index) {
            atom = atom.with_formal_charge(charge);
        }
        molecule.add_atom(atom);
    }
    for ((first, second, bond_type), line) in bonds.into_iter().zip(
        lines[4 + atom_count..4 + atom_count + bond_count]
            .iter()
            .map(|&(number, _)| number + 1),
    ) {
        molecule
            .add_bond_between(first, second, bond_type)
            .map_err(|source| IoError::Chemistry { source, line })?;
    }

    let data_fields = parse_data_fields(&lines[cursor..])?;
    Ok(SdfRecord {
        molecule,
        data_fields,
    })
}

/// Parses the V2000 counts line: atom count in columns 1–3, bond count in
/// columns 4–6.
fn parse_counts_line(text: &str, line: usize) -> IoResult<(usize, usize)> {
    if text.contains("V3000") {
        return Err(IoError::InvalidCountsLine {
            line,
            reason: "V3000 molfiles are not supported".into(),
        });
    }
    let counts = parse_columns(text, &[(0, 3), (3, 6)]);
    let atom_count = counts[0].ok_or_else(|| IoError::InvalidCountsLine {
        line,
        reason: "atom count (columns 1-3) is missing or not an integer".into(),
    })?;
    let bond_count = counts[1].ok_or_else(|| IoError::InvalidCountsLine {
        line,
        reason: "bond count (columns 4-6) is missing or not an integer".into(),
    })?;
    if atom_count == 0 {
        return Err(IoError::InvalidCountsLine {
            line,
            reason: "a record must contain at least one atom".into(),
        });
    }
    Ok((atom_count, bond_count))
}

/// Reads fixed `start..end` column groups, falling back to whitespace
/// tokens when the line does not reach the columns.
///
/// Returns `None` for a group that is missing or not an unsigned integer.
/// Columns are read as bytes (all molfile fields are ASCII); a group that
/// falls inside a multi-byte character simply fails to parse and the
/// whitespace fallback takes over — no panics on unusual input.
fn parse_columns(text: &str, groups: &[(usize, usize)]) -> Vec<Option<usize>> {
    let bytes = text.as_bytes();
    let widest = groups.iter().map(|&(_, end)| end).max().unwrap_or(0);
    if bytes.len() < widest {
        return text
            .split_whitespace()
            .map(|token| token.parse::<usize>().ok())
            .collect();
    }
    groups
        .iter()
        .map(|&(start, end)| {
            std::str::from_utf8(&bytes[start..end.min(bytes.len())])
                .ok()
                .and_then(|column| column.trim().parse::<usize>().ok())
        })
        .collect()
}

/// Parses one atom-block line into element + position.
fn parse_atom_line(text: &str, line: usize) -> IoResult<(ChemicalElement, Position)> {
    let (symbol, x, y, z) = parse_atom_fields(text).ok_or(IoError::InvalidLine {
        line,
        reason: "atom block line needs columns for x, y, z, and the element symbol".into(),
    })?;
    let element = ChemicalElement::from_symbol(symbol).map_err(|_| IoError::UnknownElement {
        symbol: symbol.to_string(),
        line,
    })?;
    for value in [x, y, z] {
        if !value.is_finite() {
            return Err(IoError::InvalidCoordinate {
                line,
                value: value.to_string(),
            });
        }
    }
    let position = Position::new(x, y, z).map_err(|source| IoError::Chemistry { source, line })?;
    Ok((element, position))
}

/// Extracts `(symbol, x, y, z)` by fixed columns (10.4f coordinates,
/// symbol in columns 32–34), falling back to whitespace tokens.
///
/// Column reads go through bytes, so multi-byte characters can never
/// panic the slice; they only fail the ASCII-field parse and trigger the
/// fallback.
fn parse_atom_fields(text: &str) -> Option<(&str, f64, f64, f64)> {
    let bytes = text.as_bytes();
    if bytes.len() >= 34 {
        let by_columns = (|| {
            let symbol = std::str::from_utf8(&bytes[31..34]).ok()?.trim();
            let x = text.get(0..10)?.trim().parse::<f64>().ok()?;
            let y = text.get(10..20)?.trim().parse::<f64>().ok()?;
            let z = text.get(20..30)?.trim().parse::<f64>().ok()?;
            if symbol.is_empty() {
                return None;
            }
            Some((symbol, x, y, z))
        })();
        if by_columns.is_some() {
            return by_columns;
        }
    }
    // Whitespace fallback for space-separated (column-drifted) files.
    let tokens: Vec<&str> = text.split_whitespace().collect();
    if tokens.len() < 4 {
        return None;
    }
    Some((
        tokens[3],
        tokens[0].parse().ok()?,
        tokens[1].parse().ok()?,
        tokens[2].parse().ok()?,
    ))
}

/// Parses one bond-block line into `(first, second, type)` with 0-based
/// atom indices.
fn parse_bond_line(text: &str, line: usize) -> IoResult<(usize, usize, BondType)> {
    let (first, second, code) = if text.len() >= 9 {
        (|| {
            Some((
                text.get(0..3)?.trim().parse::<usize>().ok()?,
                text.get(3..6)?.trim().parse::<usize>().ok()?,
                text.get(6..9)?.trim().parse::<u8>().ok()?,
            ))
        })()
    } else {
        (|| {
            let tokens: Vec<&str> = text.split_whitespace().collect();
            if tokens.len() < 3 {
                return None;
            }
            Some((
                tokens[0].parse().ok()?,
                tokens[1].parse().ok()?,
                tokens[2].parse().ok()?,
            ))
        })()
    }
    .ok_or(IoError::InvalidLine {
        line,
        reason: "bond block line needs columns for two atom indices and the bond order".into(),
    })?;
    let bond_type = match code {
        1 => BondType::Single,
        2 => BondType::Double,
        3 => BondType::Triple,
        4 => BondType::Aromatic,
        other => return Err(IoError::UnsupportedBondOrder { code: other, line }),
    };
    // SDF atom indices are 1-based; store 0-based, bounds-checked later.
    Ok((first.saturating_sub(1), second.saturating_sub(1), bond_type))
}

/// Applies one `M  CHG  n  a1 c1 a2 c2 ...` property line.
fn apply_charge_line(
    text: &str,
    line: usize,
    formal_charges: &mut BTreeMap<usize, (i8, usize)>,
) -> IoResult<()> {
    let tokens: Vec<&str> = text.split_whitespace().collect();
    // `M  CHG  <count>  (<atom, charge>)*` — token 2 carries the count.
    if tokens.len() < 3 {
        return Err(IoError::InvalidLine {
            line,
            reason: "`M  CHG` needs a pair count and at least one atom/charge pair".into(),
        });
    }
    let pairs: usize = tokens[2].parse().map_err(|_| IoError::InvalidLine {
        line,
        reason: "`M  CHG` pair count is not an integer".into(),
    })?;
    if pairs == 0 {
        return Err(IoError::InvalidLine {
            line,
            reason: "`M  CHG` must declare at least one pair".into(),
        });
    }
    let expected = 3 + 2 * pairs;
    if tokens.len() < expected {
        return Err(IoError::InvalidLine {
            line,
            reason: format!(
                "`M  CHG` declares {pairs} pair(s) but carries only {} value(s)",
                tokens.len().saturating_sub(3)
            ),
        });
    }
    for pair in 0..pairs {
        let atom_index: usize = tokens[3 + 2 * pair]
            .parse()
            .map_err(|_| IoError::InvalidLine {
                line,
                reason: "`M  CHG` atom index is not an integer".into(),
            })?;
        let charge: i8 = tokens[4 + 2 * pair]
            .parse()
            .map_err(|_| IoError::InvalidLine {
                line,
                reason: "`M  CHG` charge is not an integer".into(),
            })?;
        formal_charges.insert(atom_index.saturating_sub(1), (charge, line));
    }
    Ok(())
}

/// Parses `> <TAG>` data-field blocks after `M  END`.
///
/// A field's value is every line until the next blank line, the next `>`
/// header, or the end of the record; multi-line values keep their
/// newlines. Property lines (`M ...`) after the terminator are malformed
/// — whatever they carry would be silently lost — so they are rejected.
///
/// # Errors
///
/// Returns [`IoError::InvalidLine`] for `M`-block lines after `M  END`.
fn parse_data_fields(lines: &[(usize, &str)]) -> IoResult<BTreeMap<String, String>> {
    let mut fields = BTreeMap::new();
    let mut current_tag: Option<String> = None;
    let mut current_value = String::new();
    for &(number, text) in lines {
        let trimmed_start = text.trim_start();
        if trimmed_start.starts_with("M  ") {
            return Err(IoError::InvalidLine {
                line: number + 1,
                reason: "property-block line after `M  END`; the molecule block                          must be terminated last"
                    .into(),
            });
        }
        if let Some(tag) = extract_tag(trimmed_start) {
            if let Some(tag) = current_tag.take() {
                fields.insert(tag, current_value.trim_end().to_string());
                current_value = String::new();
            }
            current_tag = Some(tag);
        } else if current_tag.is_some() {
            if text.trim().is_empty() {
                fields.insert(
                    current_tag.take().unwrap(),
                    current_value.trim_end().to_string(),
                );
                current_value = String::new();
            } else if !trimmed_start.starts_with('M') {
                if !current_value.is_empty() {
                    current_value.push('\n');
                }
                current_value.push_str(text.trim_end());
            }
        }
    }
    if let Some(tag) = current_tag.take() {
        fields.insert(tag, current_value.trim_end().to_string());
    }
    Ok(fields)
}

/// Extracts the tag between `<` and `>` from a `> <tag>` header line.
fn extract_tag(line: &str) -> Option<String> {
    if !line.trim_start().starts_with('>') {
        return None;
    }
    let start = line.find('<')? + 1;
    let end = line[start..].find('>')? + start;
    let tag = line[start..end].trim();
    if tag.is_empty() {
        return None;
    }
    Some(tag.to_string())
}

/// Reads and parses an SDF file from disk.
///
/// # Errors
///
/// Returns [`IoError::File`] when the file cannot be read, plus every
/// error of [`parse_sdf`].
pub fn parse_sdf_file(path: impl AsRef<std::path::Path>) -> IoResult<Vec<SdfRecord>> {
    let path = path.as_ref();
    let input = std::fs::read_to_string(path).map_err(|error| IoError::File {
        path: path.display().to_string(),
        reason: error.to_string(),
    })?;
    parse_sdf(&input)
}

#[cfg(test)]
mod tests {
    use super::*;
    use mi_chemistry::BondType;

    /// Water: O–H single bonds, standard V2000 column layout.
    const WATER_SDF: &str = "\
water
  mi-io     0
comment line
  3  2  0  0  0  0  0  0  0  0999 V2000
    0.0000    0.0000    0.1173 O   0  0  0  0  0  0  0  0  0  0  0  0
    0.0000    0.7572   -0.4692 H   0  0  0  0  0  0  0  0  0  0  0  0
    0.0000   -0.7572   -0.4692 H   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0  0  0  0
  1  3  1  0  0  0  0
M  END
$$$$
";

    #[test]
    fn parses_water_with_positions_and_bonds() {
        let records = parse_sdf(WATER_SDF).unwrap();
        assert_eq!(records.len(), 1);
        let molecule = records[0].molecule();
        assert_eq!(molecule.name(), Some("water"));
        assert_eq!(molecule.atom_count(), 3);
        assert_eq!(molecule.bond_count(), 2);
        assert_eq!(molecule.atom(0).unwrap().symbol(), "O");
        let (x, y, z) = molecule.atom(0).unwrap().position().unwrap().components();
        assert_eq!((x, y, z), (0.0, 0.0, 0.1173));
        assert_eq!(molecule.bond(0).unwrap().bond_type(), BondType::Single);
        assert_eq!(molecule.bond(0).unwrap().endpoints(), (0, 1));
    }

    #[test]
    fn parses_multiple_records() {
        let input = format!("{WATER_SDF}{WATER_SDF}");
        let records = parse_sdf(&input).unwrap();
        assert_eq!(records.len(), 2);
        for record in &records {
            assert_eq!(record.molecule().atom_count(), 3);
        }
    }

    #[test]
    fn parses_monoatomic_record_without_bonds() {
        let input = "\
single He
  mi-io     0

  1  0  0  0  0  0  0  0  0  0999 V2000
    0.0000    0.0000    0.0000 He  0  0  0  0  0  0  0  0  0  0  0  0
M  END
$$$$
";
        let records = parse_sdf(input).unwrap();
        let molecule = records[0].molecule();
        assert_eq!(molecule.atom_count(), 1);
        assert_eq!(molecule.bond_count(), 0);
        assert_eq!(molecule.atom(0).unwrap().symbol(), "He");
    }

    #[test]
    fn parses_all_supported_bond_orders() {
        let input = "\
orders
  mi-io     0

  5  4  0  0  0  0  0  0  0  0999 V2000
    0.0    0.0    0.0 C   0  0  0  0  0  0  0  0  0  0  0  0
    1.0    0.0    0.0 C   0  0  0  0  0  0  0  0  0  0  0  0
    2.0    0.0    0.0 C   0  0  0  0  0  0  0  0  0  0  0  0
    3.0    0.0    0.0 C   0  0  0  0  0  0  0  0  0  0  0  0
    4.0    0.0    0.0 N   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0  0  0  0
  2  3  2  0  0  0  0
  3  4  3  0  0  0  0
  4  5  4  0  0  0  0
M  END
$$$$
";
        let records = parse_sdf(input).unwrap();
        let molecule = records[0].molecule();
        let expected = [
            BondType::Single,
            BondType::Double,
            BondType::Triple,
            BondType::Aromatic,
        ];
        for (index, bond_type) in expected.iter().enumerate() {
            assert_eq!(molecule.bond(index).unwrap().bond_type(), *bond_type);
        }
    }

    #[test]
    fn parses_formal_charges_from_property_block() {
        // Hydroxide: O(-1)–H.
        let input = "\
hydroxide
  mi-io     0

  2  1  0  0  0  0  0  0  0  0999 V2000
    0.0    0.0    0.0 O   0  0  0  0  0  0  0  0  0  0  0  0
    1.0    0.0    0.0 H   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0  0  0  0
M  CHG  1   1  -1
M  END
$$$$
";
        let records = parse_sdf(input).unwrap();
        let molecule = records[0].molecule();
        assert_eq!(molecule.atom(0).unwrap().formal_charge(), -1);
        assert_eq!(molecule.atom(1).unwrap().formal_charge(), 0);
    }

    #[test]
    fn parses_data_fields_after_mol_end() {
        let input = "\
water
  mi-io     0

  3  2  0  0  0  0  0  0  0  0999 V2000
    0.0000    0.0000    0.1173 O   0  0  0  0  0  0  0  0  0  0  0  0
    0.0000    0.7572   -0.4692 H   0  0  0  0  0  0  0  0  0  0  0  0
    0.0000   -0.7572   -0.4692 H   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0  0  0  0
  1  3  1  0  0  0  0
M  END
> <homo>
-10.2755

> <gap>
11.6200

$$$$
";
        let records = parse_sdf(input).unwrap();
        assert_eq!(records[0].data_field("homo"), Some("-10.2755"));
        assert_eq!(records[0].data_field("gap"), Some("11.6200"));
        assert_eq!(records[0].data_field("missing"), None);
        let tags: Vec<_> = records[0].data_field_tags().collect();
        assert_eq!(tags, vec!["gap", "homo"]);
    }

    #[test]
    fn falls_back_to_whitespace_tokens_for_drifted_columns() {
        let input = "\
drifted
  mi-io     0

  2  1  0  0  0  0  0  0  0  0999 V2000
0.0 0.0 0.0 O 0 0 0
1.0 0.0 0.0 H 0 0 0
  1  2  1  0  0  0  0
M  END
$$$$
";
        let records = parse_sdf(input).unwrap();
        let molecule = records[0].molecule();
        assert_eq!(molecule.atom(0).unwrap().symbol(), "O");
        assert_eq!(molecule.atom(1).unwrap().position().unwrap().x(), 1.0);
    }

    #[test]
    fn rejects_empty_input() {
        for input in ["", "\n$$$$\n\n$$$$\n"] {
            let error = parse_sdf(input).unwrap_err();
            assert_eq!(error, IoError::EmptyInput { context: "SDF" });
        }
    }

    #[test]
    fn rejects_unsupported_bond_order() {
        let input = "\
weird bond
  mi-io     0

  2  1  0  0  0  0  0  0  0  0999 V2000
    0.0    0.0    0.0 C   0  0  0  0  0  0  0  0  0  0  0  0
    1.0    0.0    0.0 C   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  6  0  0  0  0
M  END
$$$$
";
        let error = parse_sdf(input).unwrap_err();
        assert_eq!(error, IoError::UnsupportedBondOrder { code: 6, line: 7 });
    }

    #[test]
    fn rejects_unknown_element_in_atom_block() {
        let input = "\
bad element
  mi-io     0

  1  0  0  0  0  0  0  0  0  0999 V2000
    0.0    0.0    0.0 Xx  0  0  0  0  0  0  0  0  0  0  0  0
M  END
$$$$
";
        let error = parse_sdf(input).unwrap_err();
        assert_eq!(
            error,
            IoError::UnknownElement {
                symbol: "Xx".into(),
                line: 5,
            }
        );
    }

    #[test]
    fn rejects_bond_referencing_missing_atom() {
        let input = "\
bad reference
  mi-io     0

  2  1  0  0  0  0  0  0  0  0999 V2000
    0.0    0.0    0.0 C   0  0  0  0  0  0  0  0  0  0  0  0
    1.0    0.0    0.0 C   0  0  0  0  0  0  0  0  0  0  0  0
  1  3  1  0  0  0  0
M  END
$$$$
";
        let error = parse_sdf(input).unwrap_err();
        assert!(matches!(error, IoError::Chemistry { .. }));
        assert!(error.to_string().contains("atom"), "unexpected: {error}");
    }

    #[test]
    fn rejects_duplicate_bond() {
        let input = "\
duplicated
  mi-io     0

  2  2  0  0  0  0  0  0  0  0999 V2000
    0.0    0.0    0.0 C   0  0  0  0  0  0  0  0  0  0  0  0
    1.0    0.0    0.0 C   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0  0  0  0
  2  1  2  0  0  0  0
M  END
$$$$
";
        let error = parse_sdf(input).unwrap_err();
        assert!(matches!(error, IoError::Chemistry { .. }));
        assert!(
            error.to_string().contains("duplicate"),
            "unexpected: {error}"
        );
    }

    #[test]
    fn rejects_v3000() {
        let input = "\
future format
  mi-io     0

  0  0  0  0  0  0  0  0  0  0999 V3000
M  V30 BEGIN CTAB
M  END
$$$$
";
        let error = parse_sdf(input).unwrap_err();
        assert!(matches!(error, IoError::InvalidCountsLine { .. }));
        assert!(error.to_string().contains("V3000"), "unexpected: {error}");
    }

    #[test]
    fn rejects_record_missing_mol_end() {
        let input = "\
no terminator
  mi-io     0

  1  0  0  0  0  0  0  0  0  0999 V2000
    0.0    0.0    0.0 H   0  0  0  0  0  0  0  0  0  0  0  0
$$$$
";
        let error = parse_sdf(input).unwrap_err();
        assert!(error.to_string().contains("M  END"), "unexpected: {error}");
    }

    #[test]
    fn rejects_truncated_record() {
        let error = parse_sdf("only a title\n$$$$\n").unwrap_err();
        assert!(matches!(error, IoError::InvalidLine { .. }));
    }
}
