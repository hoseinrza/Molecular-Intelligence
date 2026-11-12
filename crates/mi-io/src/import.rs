//! Dataset import: file records to validated, typed datasets.
//!
//! This module is the bridge between raw parsed files and the scientific
//! dataset layer of `mi-data`. Its contract is the project's data policy
//! made executable:
//!
//! - **Nothing is dropped silently.** Every record either becomes a
//!   validated [`MolecularEntry`] or is reported in
//!   [`ImportedDataset::skipped`] with the reason it could not be
//!   imported (invalid structure, unparseable property, ...). Strict
//!   mode turns the first problem into an error instead.
//! - **Properties are optional and unit-explicit.** Electronic
//!   properties are attached only when the corresponding data fields are
//!   present, and every value is converted according to the
//!   [`DatasetMetadata`] energy unit — a Hartree source never leaks
//!   Hartree numbers into an electronvolt dataset.
//! - **Metadata travels with the data.** The result pairs the
//!   [`MolecularDataset`] with its provenance.

use crate::error::{IoError, IoResult};
use crate::metadata::{DatasetMetadata, EnergyUnit};
use crate::sdf::SdfRecord;
use mi_chemistry::Molecule;
use mi_data::{ElectronicProperties, Energy, MolecularDataset, MolecularEntry};
use std::collections::BTreeMap;

/// Mapping from SDF data-field tags to electronic properties.
///
/// The defaults follow the field names used by QM9-style datasets
/// (`homo`, `lumo`, `gap`); override them per source with
/// [`PropertyFields::with_homo_field`] and friends.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PropertyFields {
    homo: String,
    lumo: String,
    gap: String,
}

impl PropertyFields {
    /// Creates the default mapping: `homo`, `lumo`, `gap`.
    pub fn new() -> Self {
        Self {
            homo: "homo".into(),
            lumo: "lumo".into(),
            gap: "gap".into(),
        }
    }

    /// Renames the HOMO data field.
    pub fn with_homo_field(mut self, field: impl Into<String>) -> Self {
        self.homo = field.into();
        self
    }

    /// Renames the LUMO data field.
    pub fn with_lumo_field(mut self, field: impl Into<String>) -> Self {
        self.lumo = field.into();
        self
    }

    /// Renames the HOMO–LUMO gap data field.
    pub fn with_gap_field(mut self, field: impl Into<String>) -> Self {
        self.gap = field.into();
        self
    }

    /// The HOMO data-field tag.
    pub fn homo_field(&self) -> &str {
        &self.homo
    }

    /// The LUMO data-field tag.
    pub fn lumo_field(&self) -> &str {
        &self.lumo
    }

    /// The HOMO–LUMO gap data-field tag.
    pub fn gap_field(&self) -> &str {
        &self.gap
    }
}

impl Default for PropertyFields {
    fn default() -> Self {
        Self::new()
    }
}

/// Options for dataset import.
///
/// `strict` controls the failure policy: when disabled (default), records
/// that cannot be imported are reported in [`ImportedDataset::skipped`]
/// and the rest of the dataset is built; when enabled, the first problem
/// aborts the import with an error. Either way, invalid data is never
/// silently accepted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImportOptions {
    property_fields: PropertyFields,
    strict: bool,
}

impl ImportOptions {
    /// Creates options with the default [`PropertyFields`] and
    /// non-strict failure reporting.
    pub fn new() -> Self {
        Self {
            property_fields: PropertyFields::new(),
            strict: false,
        }
    }

    /// Replaces the data-field mapping used to attach properties.
    pub fn with_property_fields(mut self, fields: PropertyFields) -> Self {
        self.property_fields = fields;
        self
    }

    /// Enables strict mode: the first invalid record aborts the import.
    pub fn with_strict(mut self, strict: bool) -> Self {
        self.strict = strict;
        self
    }

    /// The data-field mapping used to attach properties.
    pub fn property_fields(&self) -> &PropertyFields {
        &self.property_fields
    }

    /// Whether strict mode is enabled.
    pub fn is_strict(&self) -> bool {
        self.strict
    }
}

impl Default for ImportOptions {
    fn default() -> Self {
        Self::new()
    }
}

/// One record that could not become a dataset entry.
///
/// `source_index` counts the records of the input file in file order
/// (0-based), independent of which entries made it into the dataset.
#[derive(Debug, Clone, PartialEq)]
pub struct SkippedEntry {
    /// 0-based record index in the input file.
    pub source_index: usize,
    /// The record's title, when present.
    pub title: Option<String>,
    /// Why the record was not imported.
    pub reason: IoError,
}

/// The result of a successful dataset import.
#[derive(Debug, Clone, PartialEq)]
pub struct ImportedDataset {
    dataset: MolecularDataset,
    metadata: DatasetMetadata,
    skipped: Vec<SkippedEntry>,
}

impl ImportedDataset {
    /// The imported dataset: entries that passed every check.
    pub fn dataset(&self) -> &MolecularDataset {
        &self.dataset
    }

    /// The provenance metadata the import was constructed with.
    pub fn metadata(&self) -> &DatasetMetadata {
        &self.metadata
    }

    /// The records that were not imported, with their reasons.
    ///
    /// Empty when every record imported cleanly. In strict mode this is
    /// always empty (problems surface as errors instead).
    pub fn skipped(&self) -> &[SkippedEntry] {
        &self.skipped
    }

    /// Total records seen in the input (accepted + skipped).
    pub fn record_count(&self) -> usize {
        self.dataset.len() + self.skipped.len()
    }
}

/// Imports one or more SDF records as a [`MolecularDataset`].
///
/// Each record becomes a [`MolecularEntry`]; data fields named by
/// [`ImportOptions::property_fields`] are parsed as finite numbers in the
/// metadata's [`EnergyUnit`] and attached as electronic properties (in
/// electronvolts — `mi-data`'s storage unit). Records that fail any
/// check are skipped (reported) or abort the import (strict mode).
///
/// # Errors
///
/// Returns [`IoError::EmptyInput`] when the input holds no record,
/// [`IoError::InvalidMetadata`] when the metadata is malformed, and — in
/// strict mode — the first record-level error encountered.
pub fn import_sdf(
    records: &[SdfRecord],
    metadata: DatasetMetadata,
    options: &ImportOptions,
) -> IoResult<ImportedDataset> {
    if records.is_empty() {
        return Err(IoError::EmptyInput { context: "SDF" });
    }

    let mut dataset = MolecularDataset::new();
    let mut skipped = Vec::new();
    for (source_index, record) in records.iter().enumerate() {
        let title = record.molecule().name().map(str::to_string);
        match build_entry(record, metadata.energy_unit(), options) {
            Ok((entry, properties)) => {
                dataset.push(entry.with_properties(properties));
            }
            Err(reason) => {
                if options.is_strict() {
                    return Err(reason);
                }
                skipped.push(SkippedEntry {
                    source_index,
                    title,
                    reason,
                });
            }
        }
    }

    Ok(ImportedDataset {
        dataset: dataset.with_name(metadata.source().to_string()),
        metadata,
        skipped,
    })
}

/// Imports a single XYZ molecule as a one-entry dataset.
///
/// XYZ carries no electronic properties, so the entry is structural only;
/// the metadata still travels with the dataset so the structural source
/// is provenance-complete.
///
/// # Errors
///
/// Returns [`IoError::EmptyInput`] when the input holds no molecule.
pub fn import_xyz(molecule: Molecule, metadata: DatasetMetadata) -> IoResult<ImportedDataset> {
    let entry = MolecularEntry::new(molecule).map_err(|source| IoError::DataRejected { source })?;
    let mut dataset = MolecularDataset::new();
    dataset.push(entry);
    Ok(ImportedDataset {
        dataset: dataset.with_name(metadata.source().to_string()),
        metadata,
        skipped: Vec::new(),
    })
}

/// Builds one entry from a record, extracting properties when present.
fn build_entry(
    record: &SdfRecord,
    unit: EnergyUnit,
    options: &ImportOptions,
) -> IoResult<(MolecularEntry, ElectronicProperties)> {
    let entry = MolecularEntry::new(record.molecule().clone())
        .map_err(|source| IoError::DataRejected { source })?;

    let fields = &options.property_fields;
    let homo = record
        .data_field(fields.homo_field())
        .map(|value| parse_energy(fields.homo_field(), value, unit));
    let lumo = record
        .data_field(fields.lumo_field())
        .map(|value| parse_energy(fields.lumo_field(), value, unit));
    let gap = record
        .data_field(fields.gap_field())
        .map(|value| parse_energy(fields.gap_field(), value, unit));

    let mut properties = ElectronicProperties::new();
    if let Some(homo) = homo.transpose()? {
        properties = properties.with_homo(homo);
    }
    if let Some(lumo) = lumo.transpose()? {
        properties = properties.with_lumo(lumo);
    }
    if let Some(gap) = gap.transpose()? {
        properties = properties.with_gap(gap);
    }
    Ok((entry, properties))
}

/// Parses one property value in the metadata's energy unit into the
/// storage unit (electronvolts).
fn parse_energy(field: &str, text: &str, unit: EnergyUnit) -> IoResult<Energy> {
    let value: f64 = text
        .trim()
        .parse()
        .map_err(|_| IoError::InvalidPropertyValue {
            field: field.to_string(),
            value: text.to_string(),
        })?;
    if !value.is_finite() {
        return Err(IoError::InvalidPropertyValue {
            field: field.to_string(),
            value: text.to_string(),
        });
    }
    match unit {
        EnergyUnit::Ev => Energy::from_ev(value),
        EnergyUnit::Hartree => Energy::from_hartree(value),
    }
    .map_err(|source| IoError::DataRejected { source })
}

/// Exposes parsed data-field tags for diagnostics.
///
/// Importing does not require this, but reports and debugging do: the
/// tags a record actually carries are the first thing to inspect when
/// properties come out [`None`].
pub fn record_data_fields(record: &SdfRecord) -> BTreeMap<String, String> {
    record
        .data_field_tags()
        .map(|tag| {
            (
                tag.to_string(),
                record.data_field(tag).unwrap_or("").to_string(),
            )
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::EnergyUnit;
    use crate::sdf::parse_sdf;

    /// Two water records; the second carries homo/lumo/gap fields in
    /// Hartree. Values are toy numbers, not real quantum chemistry.
    const RECORDS: &str = "\
plain water
  mi-io     0

  3  2  0  0  0  0  0  0  0  0999 V2000
    0.0000    0.0000    0.1173 O   0  0  0  0  0  0  0  0  0  0  0  0
    0.0000    0.7572   -0.4692 H   0  0  0  0  0  0  0  0  0  0  0  0
    0.0000   -0.7572   -0.4692 H   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0  0  0  0
  1  3  1  0  0  0  0
M  END
$$$$
water with properties
  mi-io     0

  3  2  0  0  0  0  0  0  0  0999 V2000
    0.0000    0.0000    0.1173 O   0  0  0  0  0  0  0  0  0  0  0  0
    0.0000    0.7572   -0.4692 H   0  0  0  0  0  0  0  0  0  0  0  0
    0.0000   -0.7572   -0.4692 H   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0  0  0  0
  1  3  1  0  0  0  0
M  END
> <homo>
-0.3133

> <lumo>
0.0631

> <gap>
0.3764

$$$$
";

    fn metadata(unit: EnergyUnit) -> DatasetMetadata {
        DatasetMetadata::new("toy-water", unit).unwrap()
    }

    #[test]
    fn imports_every_record_and_attaches_properties() {
        let records = parse_sdf(RECORDS).unwrap();
        let imported = import_sdf(
            &records,
            metadata(EnergyUnit::Hartree),
            &ImportOptions::new(),
        )
        .unwrap();
        assert_eq!(imported.record_count(), 2);
        assert!(imported.skipped().is_empty());
        let dataset = imported.dataset();
        assert_eq!(dataset.len(), 2);
        assert_eq!(dataset.name(), Some("toy-water"));

        let first = dataset.entry(0).unwrap();
        assert_eq!(
            first.properties().homo(),
            None,
            "structural entry stays structural"
        );

        let second = dataset.entry(1).unwrap();
        let homo = second.properties().homo().unwrap();
        // Hartree → eV conversion through mi-data's CODATA factor.
        let expected = -0.3133 * mi_data::Energy::HARTREE_TO_EV;
        assert!((homo.ev() - expected).abs() < 1e-9, "got {}", homo.ev());
        // Gap stored, not derived (it was present in the data fields).
        let gap = second.properties().gap().unwrap();
        let expected_gap = 0.3764 * mi_data::Energy::HARTREE_TO_EV;
        assert!((gap.ev() - expected_gap).abs() < 1e-9);
    }

    #[test]
    fn ev_sources_skip_conversion() {
        let records = parse_sdf(RECORDS).unwrap();
        let imported =
            import_sdf(&records, metadata(EnergyUnit::Ev), &ImportOptions::new()).unwrap();
        let second = imported.dataset().entry(1).unwrap();
        let homo = second.properties().homo().unwrap();
        assert!((homo.ev() - (-0.3133)).abs() < 1e-12, "got {}", homo.ev());
    }

    #[test]
    fn invalid_property_value_skips_the_record() {
        let input = "\
broken properties
  mi-io     0

  1  0  0  0  0  0  0  0  0  0999 V2000
    0.0    0.0    0.0 He  0  0  0  0  0  0  0  0  0  0  0  0
M  END
> <homo>
not-a-number

$$$$
";
        let records = parse_sdf(input).unwrap();
        let imported =
            import_sdf(&records, metadata(EnergyUnit::Ev), &ImportOptions::new()).unwrap();
        assert_eq!(imported.record_count(), 1);
        assert_eq!(imported.dataset().len(), 0);
        let skipped = &imported.skipped()[0];
        assert_eq!(skipped.source_index, 0);
        assert_eq!(skipped.title.as_deref(), Some("broken properties"));
        assert!(matches!(
            skipped.reason,
            IoError::InvalidPropertyValue { .. }
        ));
    }

    #[test]
    fn custom_field_names_are_used() {
        let input = "\
q9 water
  mi-io     0

  3  2  0  0  0  0  0  0  0  0999 V2000
    0.0000    0.0000    0.1173 O   0  0  0  0  0  0  0  0  0  0  0  0
    0.0000    0.7572   -0.4692 H   0  0  0  0  0  0  0  0  0  0  0  0
    0.0000   -0.7572   -0.4692 H   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0  0  0  0
  1  3  1  0  0  0  0
M  END
> <homo_energy>
-10.0

> <lumo_energy>
1.0

$$$$
";
        let options = ImportOptions::new().with_property_fields(
            PropertyFields::new()
                .with_homo_field("homo_energy")
                .with_lumo_field("lumo_energy"),
        );
        let records = parse_sdf(input).unwrap();
        let imported = import_sdf(&records, metadata(EnergyUnit::Ev), &options).unwrap();
        let entry = imported.dataset().entry(0).unwrap();
        assert_eq!(entry.properties().homo().unwrap().ev(), -10.0);
        assert_eq!(entry.properties().lumo().unwrap().ev(), 1.0);
        // Gap absent from the file → derived as lumo − homo by mi-data.
        assert_eq!(entry.properties().homo_lumo_gap().unwrap().ev(), 11.0);
    }

    #[test]
    fn strict_mode_aborts_on_first_problem() {
        let broken = parse_sdf(
            "\
broken
  mi-io     0

  1  0  0  0  0  0  0  0  0  0999 V2000
    0.0    0.0    0.0 He  0  0  0  0  0  0  0  0  0  0  0  0
M  END
> <homo>
nope

$$$$
",
        )
        .unwrap();
        let error = import_sdf(
            &broken,
            metadata(EnergyUnit::Ev),
            &ImportOptions::new().with_strict(true),
        )
        .unwrap_err();
        assert!(matches!(error, IoError::InvalidPropertyValue { .. }));
        // Non-strict mode reports instead of failing.
        let imported =
            import_sdf(&broken, metadata(EnergyUnit::Ev), &ImportOptions::new()).unwrap();
        assert_eq!(imported.skipped().len(), 1);
    }

    #[test]
    fn empty_record_lists_are_rejected() {
        let error = import_sdf(&[], metadata(EnergyUnit::Ev), &ImportOptions::new()).unwrap_err();
        assert_eq!(error, IoError::EmptyInput { context: "SDF" });
    }

    #[test]
    fn xyz_molecules_import_as_structural_entries() {
        let molecule =
            crate::xyz::parse_xyz("2\noxygen fragment\nO 0.0 0.0 0.0\nO 1.2 0.0 0.0\n").unwrap();
        let imported = import_xyz(molecule, metadata(EnergyUnit::Ev)).unwrap();
        assert_eq!(imported.dataset().len(), 1);
        assert!(imported.skipped().is_empty());
        let entry = imported.dataset().entry(0).unwrap();
        assert_eq!(entry.properties().homo(), None);
        assert_eq!(entry.molecule().atom_count(), 2);
    }
}
