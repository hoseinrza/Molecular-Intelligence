//! Invalid-input behavior across formats.
//!
//! Malformed data must surface as typed errors naming the offending file
//! line — never a panic, never a silent simplification. These tests use
//! inline documents only: no external fixtures, no GPU, no network.

use mi_io::{IoError, parse_sdf, parse_xyz};

#[test]
fn xyz_reports_line_numbers_for_every_failure_class() {
    // Unknown element on line 3.
    let error = parse_xyz("1\nbad element\nXx 0.0 0.0 0.0\n").unwrap_err();
    assert_eq!(
        error,
        IoError::UnknownElement {
            symbol: "Xx".into(),
            line: 3,
        }
    );
    // Non-numeric coordinate on line 3.
    let error = parse_xyz("1\nbad coord\nH oops 0.0 0.0\n").unwrap_err();
    assert!(error.to_string().contains("line 3"), "unexpected: {error}");
    // Count mismatch (declared 2, found 1).
    let error = parse_xyz("2\nshort\nH 0.0 0.0 0.0\n").unwrap_err();
    assert!(matches!(error, IoError::AtomCountMismatch { .. }));
}

#[test]
fn xyz_rejects_nan_and_infinity_in_every_coordinate_slot() {
    for coordinates in ["nan 0.0 0.0", "0.0 inf 0.0", "0.0 0.0 -inf"] {
        let input = format!("1\nbad\nH {coordinates}\n");
        let error = parse_xyz(&input).unwrap_err();
        assert!(
            matches!(error, IoError::InvalidCoordinate { .. }),
            "{coordinates}"
        );
    }
}

#[test]
fn xyz_rejects_case_incorrect_element_symbols() {
    // Symbols are case-sensitive in mi-chemistry ("CL" is not chlorine).
    let error = parse_xyz("1\nbad case\nCL 0.0 0.0 0.0\n").unwrap_err();
    assert!(matches!(error, IoError::UnknownElement { .. }));
}

#[test]
fn sdf_reports_line_numbers_for_every_failure_class() {
    // Unknown element in the atom block.
    let input = "\
bad element
  mi-io     0

  1  0  0  0  0  0  0  0  0  0999 V2000
    0.0    0.0    0.0 Zz  0  0  0  0  0  0  0  0  0  0  0  0
M  END
$$$$
";
    let error = parse_sdf(input).unwrap_err();
    assert!(
        matches!(error, IoError::UnknownElement { line: 5, .. }),
        "unexpected: {error}"
    );

    // Malformed counts line.
    let input = "\
bad counts
  mi-io     0

  ??  ??
M  END
$$$$
";
    let error = parse_sdf(input).unwrap_err();
    assert!(matches!(error, IoError::InvalidCountsLine { line: 4, .. }));

    // Non-finite coordinate in the atom block.
    let input = "\
bad position
  mi-io     0

  1  0  0  0  0  0  0  0  0  0999 V2000
    nan    0.0    0.0 H   0  0  0  0  0  0  0  0  0  0  0  0
M  END
$$$$
";
    let error = parse_sdf(input).unwrap_err();
    assert!(matches!(error, IoError::InvalidCoordinate { line: 5, .. }));
}

#[test]
fn sdf_rejects_unsupported_bond_orders_not_silent_guesses() {
    for code in ["5", "8", "9"] {
        let input = format!(
            "\
bad order
  mi-io     0

  2  1  0  0  0  0  0  0  0  0999 V2000
    0.0    0.0    0.0 C   0  0  0  0  0  0  0  0  0  0  0  0
    1.0    0.0    0.0 C   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  {code}  0  0  0  0
M  END
$$$$
"
        );
        let error = parse_sdf(&input).unwrap_err();
        assert!(
            matches!(error, IoError::UnsupportedBondOrder { .. }),
            "code {code}: {error}"
        );
    }
}

#[test]
fn sdf_rejects_bonds_and_charges_referencing_missing_atoms() {
    // Bond to atom 3 in a two-atom record.
    let input = "\
bad bond reference
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

    // Charge on atom 5 in a one-atom record: silently dropping the
    // charge would change the chemistry, so the parser rejects it with
    // the offending line number.
    let input = "\
bad charge reference
  mi-io     0

  1  0  0  0  0  0  0  0  0  0999 V2000
    0.0    0.0    0.0 H   0  0  0  0  0  0  0  0  0  0  0  0
M  CHG  1   5   1
M  END
$$$$
";
    let error = parse_sdf(input).unwrap_err();
    assert!(
        matches!(
            error,
            IoError::InvalidAtomReference {
                index: 4,
                atom_count: 1,
                ..
            }
        ),
        "unexpected: {error}"
    );
}

#[test]
fn sdf_rejects_truncated_and_empty_documents() {
    let error = parse_sdf("").unwrap_err();
    assert_eq!(error, IoError::EmptyInput { context: "SDF" });
    let error = parse_sdf("   \n$$$$\n").unwrap_err();
    assert_eq!(error, IoError::EmptyInput { context: "SDF" });
    let error = parse_sdf("title only\n$$$$\n").unwrap_err();
    assert!(matches!(error, IoError::InvalidLine { .. }));
}

#[test]
fn xyz_rejects_empty_documents() {
    for input in ["", "\n\n\n"] {
        let error = parse_xyz(input).unwrap_err();
        assert_eq!(error, IoError::EmptyInput { context: "XYZ" });
    }
}

#[test]
fn incomplete_property_data_imports_without_properties() {
    // A record with homo but no lumo: absence is legitimate; mi-data
    // derives the gap when both are present and reports none otherwise.
    use mi_io::{DatasetMetadata, EnergyUnit, ImportOptions, import_sdf};
    let input = "\
partial data
  mi-io     0

  1  0  0  0  0  0  0  0  0  0999 V2000
    0.0    0.0    0.0 He  0  0  0  0  0  0  0  0  0  0  0  0
M  END
> <homo>
-0.5

$$$$
";
    let records = parse_sdf(input).unwrap();
    let metadata = DatasetMetadata::new("partial", EnergyUnit::Hartree).unwrap();
    let imported = import_sdf(&records, metadata, &ImportOptions::new()).unwrap();
    assert_eq!(imported.dataset().len(), 1);
    let entry = imported.dataset().entry(0).unwrap();
    assert_eq!(entry.properties().homo().unwrap().hartree(), -0.5);
    assert_eq!(entry.properties().lumo(), None);
    assert_eq!(entry.properties().gap(), None);
}
