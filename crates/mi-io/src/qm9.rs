//! QM9 dataset loading.
//!
//! QM9 ([Ramakrishnan et al., *Scientific Data* 1, 140022 (2014)];
//! [figshare collection](https://doi.org/10.6084/m9.figshare.c.1290126),
//! CC BY 4.0) is the reference dataset of the project's first case study:
//! 133,885 small organic molecules (CHNOF, ≤9 heavy atoms) with properties
//! from **B3LYP/6-31G(2df,p)** DFT calculations. The distribution ships a
//! single `qm9.sdf` carrying every molecule with its properties as data
//! fields — exactly the shape [`crate::sdf`] parses.
//!
//! # What this loader does
//!
//! - Streams records from disk (the file is far too large to materialize)
//!   through [`SdfRecordReader`](crate::sdf::SdfRecordReader) into a
//!   [`MolecularDataset`], attaching the project's three focus properties
//!   (`homo`, `lumo`, `gap`) as [`ElectronicProperties`](mi_data).
//! - Applies the **energy-unit policy explicitly**: the raw fields are in
//!   **Hartree**, so the default conversion is Hartree → eV via
//!   `mi_data::Energy::HARTREE_TO_EV`; a source already in electronvolts
//!   sets [`Qm9Options::with_energy_unit`]. Storage is always eV.
//! - Excludes the molecules listed in the distribution's
//!   `uncharacterized.txt` (calculations with convergence or formatting
//!   problems) — by 1-based record index, which matches the xyz-file
//!   numbering.
//! - Reports everything that did not become an entry: skipped records with
//!   reasons, exclusion counts, and title/position mismatches. Nothing is
//!   dropped silently.
//!
//! # What this loader does *not* claim
//!
//! Predicting QM9's HOMO/LUMO means predicting **DFT** orbital energies —
//! not GW values (see Fediai et al. 2023 for a GW reference set) and not
//! experimental observables. The dataset-level distinction travels in the
//! provenance metadata and the project's scientific documentation.
//!
//! Only the three focus properties enter `mi-data`; the remaining QM9
//! fields (rotational constants, dipole, energies, …) stay available in
//! the source file and are future typed extensions, not silent losses.

use crate::error::{IoError, IoResult};
use crate::import::{ImportAccumulator, PropertyFields, SkippedEntry};
use crate::metadata::{DatasetMetadata, EnergyUnit};
use crate::sdf::{SdfRecord, SdfRecordReader};
use mi_data::MolecularDataset;
use std::collections::BTreeSet;

/// Options for loading a QM9 SDF file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qm9Options {
    property_fields: PropertyFields,
    energy_unit: EnergyUnit,
    strict: bool,
    version: Option<String>,
}

impl Qm9Options {
    /// Creates the defaults: `homo`/`lumo`/`gap` field tags, **Hartree**
    /// source energies (the raw `qm9.sdf` convention), non-strict mode,
    /// no version tag.
    pub fn new() -> Self {
        Self {
            property_fields: PropertyFields::new(),
            energy_unit: EnergyUnit::Hartree,
            strict: false,
            version: None,
        }
    }

    /// Replaces the data-field tag mapping.
    pub fn with_property_fields(mut self, fields: PropertyFields) -> Self {
        self.property_fields = fields;
        self
    }

    /// Overrides the source energy unit (use `EnergyUnit::Ev` for a
    /// pre-converted file).
    pub fn with_energy_unit(mut self, unit: EnergyUnit) -> Self {
        self.energy_unit = unit;
        self
    }

    /// Enables strict mode: the first invalid record aborts the load.
    pub fn with_strict(mut self, strict: bool) -> Self {
        self.strict = strict;
        self
    }

    /// Tags the provenance metadata with a dataset version.
    pub fn with_version(mut self, version: impl Into<String>) -> Self {
        self.version = Some(version.into());
        self
    }

    /// The configured data-field mapping.
    pub fn property_fields(&self) -> &PropertyFields {
        &self.property_fields
    }

    /// The configured source energy unit.
    pub fn energy_unit(&self) -> EnergyUnit {
        self.energy_unit
    }

    /// Whether strict mode is enabled.
    pub fn is_strict(&self) -> bool {
        self.strict
    }
}

impl Default for Qm9Options {
    fn default() -> Self {
        Self::new()
    }
}

/// The result of loading a QM9 SDF file.
#[derive(Debug, Clone)]
pub struct Qm9Report {
    dataset: MolecularDataset,
    metadata: DatasetMetadata,
    skipped: Vec<SkippedEntry>,
    record_count: usize,
    excluded_uncharacterized: usize,
    title_mismatches: usize,
}

impl Qm9Report {
    /// The loaded dataset (only molecules that passed every check).
    pub fn dataset(&self) -> &MolecularDataset {
        &self.dataset
    }

    /// The provenance metadata the load was built with.
    pub fn metadata(&self) -> &DatasetMetadata {
        &self.metadata
    }

    /// The records that were read but could not be imported, with reasons.
    pub fn skipped(&self) -> &[SkippedEntry] {
        &self.skipped
    }

    /// Total records read from the file (accepted + skipped + excluded).
    pub fn record_count(&self) -> usize {
        self.record_count
    }

    /// Records excluded because their index appears in
    /// `uncharacterized.txt`.
    pub fn excluded_uncharacterized(&self) -> usize {
        self.excluded_uncharacterized
    }

    /// How often a `gdb9_<n>` title disagreed with the record's file
    /// position. Zero for a well-formed QM9 file; non-zero means the
    /// exclusion mapping may not line up with the distribution's own
    /// numbering — a visible warning, never a silent assumption.
    pub fn title_mismatches(&self) -> usize {
        self.title_mismatches
    }
}

/// The canonical provenance metadata for the QM9 distribution.
///
/// # Errors
///
/// Returns [`IoError::InvalidMetadata`] if `version` is empty content.
pub fn qm9_metadata(version: Option<String>) -> IoResult<DatasetMetadata> {
    let metadata = DatasetMetadata::new("QM9", EnergyUnit::Hartree)?
        .with_method("DFT")?
        .with_functional("B3LYP")?
        .with_basis_set("6-31G(2df,p)")?
        .with_source_id("doi:10.6084/m9.figshare.c.1290126")?;
    match version {
        Some(version) => metadata.with_version(version),
        None => Ok(metadata),
    }
}

/// Loads a QM9 SDF file into a validated [`MolecularDataset`].
///
/// Records listed in `uncharacterized` (1-based indices, as produced by
/// [`uncharacterized_indices`]) are excluded before import and counted in
/// the report; all other records follow the shared import policy: invalid
/// records are skipped with reasons, or abort the load in strict mode.
///
/// # Errors
///
/// Returns [`IoError::File`] when the file cannot be read, parse errors
/// from malformed records, [`IoError::EmptyInput`] when the file holds no
/// record, and — in strict mode — the first record-level error.
pub fn load_qm9_sdf(
    path: impl AsRef<std::path::Path>,
    options: &Qm9Options,
    uncharacterized: &BTreeSet<usize>,
) -> IoResult<Qm9Report> {
    let reader = SdfRecordReader::from_path(path)?;
    let metadata = qm9_metadata(options.version.clone())?;
    let mut accumulator = ImportAccumulator::new();
    let mut excluded_uncharacterized = 0usize;
    let mut title_mismatches = 0usize;

    for item in reader {
        let record: SdfRecord = item?;
        // The 1-based record position mirrors the distribution's xyz-file
        // numbering; a `gdb9_<n>` title that disagrees is counted as a
        // visible warning instead of silently shifting the exclusion map.
        let position = accumulator.record_count() + excluded_uncharacterized + 1;
        let title_number = title_index(record.molecule().name());
        if let Some(number) = title_number
            && number != position
        {
            title_mismatches += 1;
        }
        let exclusion_key = title_number.unwrap_or(position);
        if uncharacterized.contains(&exclusion_key) {
            excluded_uncharacterized += 1;
            continue;
        }
        accumulator.push(
            position,
            &record,
            options.energy_unit(),
            options.property_fields(),
            options.is_strict(),
        )?;
    }

    if accumulator.record_count() == 0 {
        return Err(IoError::EmptyInput { context: "QM9" });
    }

    let record_count = accumulator.record_count() + excluded_uncharacterized;
    let (dataset, metadata, skipped) = accumulator.finish(metadata);
    Ok(Qm9Report {
        dataset,
        metadata,
        skipped,
        record_count,
        excluded_uncharacterized,
        title_mismatches,
    })
}

/// Parses the distribution's `uncharacterized.txt` into a set of 1-based
/// molecule indices.
///
/// Each line contributes its first whitespace-separated token; lines that
/// do not start with a digit (headers, notes) are skipped; lines that
/// start with digits but fail to parse are errors — corrupted indices are
/// a data condition, not something to guess around.
///
/// # Errors
///
/// Returns [`IoError::File`] when the file cannot be read,
/// [`IoError::InvalidLine`] for malformed numeric lines, and
/// [`IoError::EmptyInput`] when no index could be extracted.
pub fn uncharacterized_indices(path: impl AsRef<std::path::Path>) -> IoResult<BTreeSet<usize>> {
    let path = path.as_ref();
    let content = std::fs::read_to_string(path).map_err(|error| IoError::File {
        path: path.display().to_string(),
        reason: error.to_string(),
    })?;
    let mut indices = BTreeSet::new();
    for (offset, line) in content.lines().enumerate() {
        let first = match line.split_whitespace().next() {
            Some(token) => token,
            None => continue,
        };
        if !first.starts_with(|c: char| c.is_ascii_digit()) {
            continue; // header or note line
        }
        let index: usize = first.parse().map_err(|_| IoError::InvalidLine {
            line: offset + 1,
            reason: format!("uncharacterized index {first:?} is not an integer"),
        })?;
        if index == 0 {
            return Err(IoError::InvalidLine {
                line: offset + 1,
                reason: "molecule indices are 1-based; 0 is invalid".into(),
            });
        }
        indices.insert(index);
        continue;
    }
    if indices.is_empty() {
        return Err(IoError::EmptyInput {
            context: "uncharacterized index list",
        });
    }
    Ok(indices)
}

/// Extracts the 1-based molecule index from a `gdb9_<n>` title.
fn title_index(name: Option<&str>) -> Option<usize> {
    let name = name?;
    let number = name.strip_prefix("gdb9_")?;
    number.parse::<usize>().ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use mi_data::Energy;

    /// Two records with Hartree-valued properties, matching the raw
    /// `qm9.sdf` conventions; the second molecule is listed as
    /// uncharacterized in the fixture index file.
    const RECORDS: &str = "\
gdb9_1
  mi-io     0

  5  4  0  0  0  0  0  0  0  0999 V2000
    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.6300    0.6300    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
   -0.6300    0.6300    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
   -0.6300   -0.6300    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
    0.6300   -0.6300    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0  0  0  0
  1  3  1  0  0  0  0
  1  4  1  0  0  0  0
  1  5  1  0  0  0  0
M  END
> <homo>
-0.4650

> <lumo>
0.0500

> <gap>
0.5150

$$$$
gdb9_2
  mi-io     0

  1  0  0  0  0  0  0  0  0  0999 V2000
    0.0    0.0    0.0 He  0  0  0  0  0  0  0  0  0  0  0  0
M  END
> <homo>
-0.9000

> <lumo>
0.0100

> <gap>
0.9100

$$$$
";

    fn scratch_dir(test: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("mi_io_qm9_{}_{}", std::process::id(), test));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn write_fixtures(test: &str) -> (std::path::PathBuf, std::path::PathBuf, std::path::PathBuf) {
        let dir = scratch_dir(test);
        let sdf = dir.join("qm9_fixture.sdf");
        std::fs::write(&sdf, RECORDS).unwrap();
        let uncharacterized = dir.join("uncharacterized.txt");
        std::fs::write(&uncharacterized, "2\n\n# note line\n7\n").unwrap();
        let missing = dir.join("missing.sdf");
        (sdf, uncharacterized, missing)
    }

    #[test]
    fn loads_properties_with_hartree_to_ev_conversion() {
        let (sdf, uncharacterized, _) = write_fixtures("loads_properties");
        let indices = uncharacterized_indices(&uncharacterized).unwrap();
        assert_eq!(indices, BTreeSet::from([2, 7]));

        let report = load_qm9_sdf(&sdf, &Qm9Options::new(), &indices).unwrap();
        assert_eq!(report.record_count(), 2);
        assert_eq!(report.excluded_uncharacterized(), 1);
        assert_eq!(report.title_mismatches(), 0);
        assert!(report.skipped().is_empty());

        let dataset = report.dataset();
        assert_eq!(dataset.len(), 1);
        let entry = dataset.entry(0).unwrap();
        let homo = entry.properties().homo().unwrap();
        let expected = -0.4650 * Energy::HARTREE_TO_EV;
        assert!((homo.ev() - expected).abs() < 1e-9, "got {}", homo.ev());
        let gap = entry.properties().gap().unwrap();
        let expected_gap = 0.5150 * Energy::HARTREE_TO_EV;
        assert!((gap.ev() - expected_gap).abs() < 1e-9);
        // Provenance travels with the dataset name.
        assert_eq!(dataset.name(), Some("QM9"));
    }

    #[test]
    fn provenance_metadata_is_canonical() {
        let metadata = qm9_metadata(None).unwrap();
        assert_eq!(metadata.source(), "QM9");
        assert_eq!(metadata.energy_unit(), EnergyUnit::Hartree);
        assert_eq!(metadata.functional(), Some("B3LYP"));
        assert_eq!(metadata.basis_set(), Some("6-31G(2df,p)"));
        assert_eq!(metadata.method(), Some("DFT"));
        assert_eq!(
            metadata.source_id(),
            Some("doi:10.6084/m9.figshare.c.1290126")
        );
        assert_eq!(metadata.version(), None);
        let versioned = qm9_metadata(Some("1.0".into())).unwrap();
        assert_eq!(versioned.version(), Some("1.0"));
    }

    #[test]
    fn title_mismatches_are_counted_not_silent() {
        let (sdf, _, _) = write_fixtures("title_mismatches");
        let report = load_qm9_sdf(&sdf, &Qm9Options::new(), &BTreeSet::new()).unwrap();
        assert_eq!(report.title_mismatches(), 0);

        // A file whose second record claims a different index is flagged.
        let dir = scratch_dir("title_mismatches");
        let shifted = dir.join("shifted.sdf");
        std::fs::write(&shifted, RECORDS.replace("gdb9_2", "gdb9_9")).unwrap();
        let report = load_qm9_sdf(&shifted, &Qm9Options::new(), &BTreeSet::new()).unwrap();
        assert_eq!(report.title_mismatches(), 1);
        // The exclusion falls back to the record position when the title
        // does not parse; here nothing is excluded either way.
        assert_eq!(report.excluded_uncharacterized(), 0);
    }

    #[test]
    fn strict_mode_aborts_on_invalid_property() {
        let dir = scratch_dir("strict_mode");
        let broken = dir.join("broken.sdf");
        let input = RECORDS.replace("-0.4650", "not-a-number");
        std::fs::write(&broken, input).unwrap();
        let error = load_qm9_sdf(
            &broken,
            &Qm9Options::new().with_strict(true),
            &BTreeSet::new(),
        )
        .map(|_| ())
        .unwrap_err();
        assert!(matches!(error, IoError::InvalidPropertyValue { .. }));
    }

    #[test]
    fn ev_sources_skip_conversion() {
        let dir = scratch_dir("ev_sources");
        let ev_file = dir.join("ev.sdf");
        // Same record but values already in eV.
        let input = RECORDS
            .replace("-0.4650", "-12.6532")
            .replace("0.0500", "1.3605")
            .replace("0.5150", "14.0137");
        std::fs::write(&ev_file, input).unwrap();
        let options = Qm9Options::new().with_energy_unit(EnergyUnit::Ev);
        let report = load_qm9_sdf(&ev_file, &options, &BTreeSet::new()).unwrap();
        let homo = report
            .dataset()
            .entry(0)
            .unwrap()
            .properties()
            .homo()
            .unwrap();
        assert!((homo.ev() - (-12.6532)).abs() < 1e-6, "got {}", homo.ev());
    }

    #[test]
    fn empty_and_missing_files_are_reported() {
        let (_, _, missing) = write_fixtures("empty_files");
        let error = load_qm9_sdf(&missing, &Qm9Options::new(), &BTreeSet::new()).unwrap_err();
        assert!(matches!(error, IoError::File { .. }));

        let dir = scratch_dir("empty_files");
        let empty = dir.join("empty.sdf");
        std::fs::write(&empty, "   \n$$$$\n").unwrap();
        let error = load_qm9_sdf(&empty, &Qm9Options::new(), &BTreeSet::new()).unwrap_err();
        assert_eq!(error, IoError::EmptyInput { context: "QM9" });
    }

    #[test]
    fn uncharacterized_list_parsing_is_tolerant_but_honest() {
        let (_, uncharacterized, _) = write_fixtures("uncharacterized_parse");
        let indices = uncharacterized_indices(&uncharacterized).unwrap();
        assert_eq!(indices, BTreeSet::from([2, 7]));

        let dir = scratch_dir("uncharacterized_parse");
        let corrupt = dir.join("corrupt.txt");
        std::fs::write(&corrupt, "12\n3x\n").unwrap();
        let error = uncharacterized_indices(&corrupt).unwrap_err();
        assert!(matches!(error, IoError::InvalidLine { line: 2, .. }));

        let zero = dir.join("zero.txt");
        std::fs::write(&zero, "0\n").unwrap();
        let error = uncharacterized_indices(&zero).unwrap_err();
        assert!(error.to_string().contains("1-based"));
    }
}
