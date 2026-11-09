//! Molecular dataset: an indexed, append-only collection of entries.

use crate::entry::MolecularEntry;
use crate::error::DataError;
use crate::property::Energy;

/// Summary statistics over one property, in electronvolts.
///
/// `std_dev` is the **population** standard deviation (divided by `count`),
/// documented explicitly so downstream normalization code knows the
/// convention. Statistics are only produced when at least one entry carries
/// the property.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PropertySummary {
    count: usize,
    mean: f64,
    min: f64,
    max: f64,
    std_dev: f64,
}

impl PropertySummary {
    /// Number of entries contributing to this summary.
    pub const fn count(&self) -> usize {
        self.count
    }

    /// Arithmetic mean, in eV.
    pub const fn mean(&self) -> f64 {
        self.mean
    }

    /// Minimum value, in eV.
    pub const fn min(&self) -> f64 {
        self.min
    }

    /// Maximum value, in eV.
    pub const fn max(&self) -> f64 {
        self.max
    }

    /// Population standard deviation (divided by `count`), in eV.
    pub const fn std_dev(&self) -> f64 {
        self.std_dev
    }
}

/// Coverage and summary statistics of a dataset, per property.
///
/// A property's summary is [`None`] when no entry carries it; its
/// `count()` is the property's coverage (how many entries have it).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct DatasetStatistics {
    entry_count: usize,
    homo: Option<PropertySummary>,
    lumo: Option<PropertySummary>,
    gap: Option<PropertySummary>,
}

impl DatasetStatistics {
    /// Total number of entries in the dataset.
    pub const fn entry_count(&self) -> usize {
        self.entry_count
    }

    /// HOMO summary, if any entry carries a HOMO.
    pub const fn homo(&self) -> Option<&PropertySummary> {
        self.homo.as_ref()
    }

    /// LUMO summary, if any entry carries a LUMO.
    pub const fn lumo(&self) -> Option<&PropertySummary> {
        self.lumo.as_ref()
    }

    /// Effective HOMO–LUMO gap summary (stored values preferred, gaps
    /// derived from HOMO/LUMO otherwise), if any entry yields one.
    pub const fn gap(&self) -> Option<&PropertySummary> {
        self.gap.as_ref()
    }
}

/// A scientific molecular dataset: named, validated, append-only.
///
/// # Indexing model
///
/// - [`push`](Self::push) appends an entry and returns its index; indices
///   are assigned in insertion order starting at 0 and **remain stable**
///   for the lifetime of the dataset (there is no removal in this phase).
/// - Entries are immutable and validated at construction
///   ([`MolecularEntry::new`]), so a dataset never contains structurally
///   invalid molecules or non-finite energies.
/// - An empty dataset is an explicit, valid state (useful while building);
///   [`statistics`](Self::statistics) reports it as zero coverage and
///   [`validate`](Self::validate) accepts it.
///
/// Serialization, file-format loaders, and train/validation/test splitting
/// are planned future extensions (see the architecture documentation).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct MolecularDataset {
    name: Option<String>,
    entries: Vec<MolecularEntry>,
}

impl MolecularDataset {
    /// Creates an empty dataset with no name.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the dataset name and returns the dataset.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Appends an entry and returns its stable index.
    pub fn push(&mut self, entry: MolecularEntry) -> usize {
        self.entries.push(entry);
        self.entries.len() - 1
    }

    /// Returns the entry at `index`.
    ///
    /// # Errors
    ///
    /// Returns [`DataError::InvalidEntryIndex`] if `index` is out of range.
    pub fn entry(&self, index: usize) -> Result<&MolecularEntry, DataError> {
        self.entries.get(index).ok_or(DataError::InvalidEntryIndex {
            index,
            entry_count: self.entries.len(),
        })
    }

    /// All entries in index order.
    pub fn entries(&self) -> &[MolecularEntry] {
        &self.entries
    }

    /// Iterates over the entries in index order.
    pub fn iter(&self) -> std::slice::Iter<'_, MolecularEntry> {
        self.entries.iter()
    }

    /// Number of entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` if the dataset has no entries.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Dataset name, if set.
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    /// Coverage and summary statistics per property, in electronvolts.
    ///
    /// Gap statistics use the *effective* gap (stored values preferred,
    /// derived `LUMO − HOMO` otherwise); entries without a yieldable value
    /// are excluded from that property's summary.
    pub fn statistics(&self) -> DatasetStatistics {
        let homo: Vec<f64> = self
            .iter()
            .filter_map(|entry| entry.properties().homo().map(Energy::ev))
            .collect();
        let lumo: Vec<f64> = self
            .iter()
            .filter_map(|entry| entry.properties().lumo().map(Energy::ev))
            .collect();
        let gap: Vec<f64> = self
            .iter()
            .filter_map(|entry| entry.properties().homo_lumo_gap().map(Energy::ev))
            .collect();
        DatasetStatistics {
            entry_count: self.entries.len(),
            homo: summarize(&homo),
            lumo: summarize(&lumo),
            gap: summarize(&gap),
        }
    }

    /// Re-validates every entry's molecule.
    ///
    /// Entries are validated at construction, so this is defensive; it
    /// defines the contract for any future construction path. An **empty
    /// dataset is valid** (unlike molecules and graphs, a dataset is a
    /// container and legitimately starts empty).
    ///
    /// # Errors
    ///
    /// Returns [`DataError::InvalidMolecule`] for the first structurally
    /// invalid entry.
    pub fn validate(&self) -> Result<(), DataError> {
        for entry in &self.entries {
            entry
                .molecule()
                .validate()
                .map_err(DataError::InvalidMolecule)?;
        }
        Ok(())
    }
}

/// Summarizes a list of finite values; [`None`] when the list is empty.
fn summarize(values: &[f64]) -> Option<PropertySummary> {
    if values.is_empty() {
        return None;
    }
    let count = values.len();
    let mean = values.iter().sum::<f64>() / count as f64;
    let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let variance = values
        .iter()
        .map(|value| (value - mean) * (value - mean))
        .sum::<f64>()
        / count as f64;
    Some(PropertySummary {
        count,
        mean,
        min,
        max,
        std_dev: variance.sqrt(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support;

    #[test]
    fn empty_dataset_is_an_explicit_valid_state() {
        let dataset = MolecularDataset::new();
        assert!(dataset.is_empty());
        assert_eq!(dataset.len(), 0);
        assert_eq!(dataset.name(), None);
        assert!(dataset.validate().is_ok());
        let statistics = dataset.statistics();
        assert_eq!(statistics.entry_count(), 0);
        assert_eq!(statistics.homo(), None);
        assert_eq!(statistics.lumo(), None);
        assert_eq!(statistics.gap(), None);
    }

    #[test]
    fn push_returns_stable_indices() {
        let mut dataset = MolecularDataset::new();
        assert_eq!(
            dataset.push(MolecularEntry::new(test_support::water()).unwrap()),
            0
        );
        assert_eq!(
            dataset.push(MolecularEntry::new(test_support::methane()).unwrap()),
            1
        );
        assert_eq!(
            dataset.push(MolecularEntry::new(test_support::carbon_dioxide()).unwrap()),
            2
        );
        assert_eq!(dataset.len(), 3);
        assert!(!dataset.is_empty());
        // Indices stay meaningful after further growth.
        assert_eq!(dataset.entry(0).unwrap().molecule().atom_count(), 3);
        assert_eq!(dataset.entry(2).unwrap().molecule().atom_count(), 3);
    }

    #[test]
    fn entry_access_reports_out_of_range_indices() {
        let mut dataset = MolecularDataset::new();
        dataset.push(MolecularEntry::new(test_support::water()).unwrap());
        assert_eq!(
            dataset.entry(1),
            Err(DataError::InvalidEntryIndex {
                index: 1,
                entry_count: 1
            })
        );
    }

    #[test]
    fn name_is_carried() {
        let dataset = MolecularDataset::new().with_name("toy dataset");
        assert_eq!(dataset.name(), Some("toy dataset"));
    }

    #[test]
    fn iteration_is_in_index_order() {
        let mut dataset = MolecularDataset::new();
        dataset.push(MolecularEntry::new(test_support::water()).unwrap());
        dataset.push(MolecularEntry::new(test_support::methane()).unwrap());
        let atom_counts: Vec<usize> = dataset.iter().map(|e| e.molecule().atom_count()).collect();
        assert_eq!(atom_counts, [3, 5]);
        assert_eq!(dataset.entries().len(), 2);
    }

    #[test]
    fn statistics_summarize_available_properties_exactly() {
        let mut dataset = MolecularDataset::new();
        // HOMO values 1.0 and 3.0 (eV): mean 2.0, population std-dev 1.0.
        dataset.push(
            MolecularEntry::new(test_support::water())
                .unwrap()
                .with_properties(test_support::properties(1.0, 2.0)),
        );
        dataset.push(
            MolecularEntry::new(test_support::methane())
                .unwrap()
                .with_properties(test_support::properties(3.0, 4.0)),
        );
        // No properties: excluded from every summary.
        dataset.push(MolecularEntry::new(test_support::carbon_dioxide()).unwrap());

        let statistics = dataset.statistics();
        assert_eq!(statistics.entry_count(), 3);
        let homo = statistics.homo().unwrap();
        assert_eq!(
            homo.count(),
            2,
            "coverage excludes entries without the property"
        );
        assert!((homo.mean() - 2.0).abs() < 1e-12);
        assert!((homo.min() - 1.0).abs() < 1e-12);
        assert!((homo.max() - 3.0).abs() < 1e-12);
        assert!((homo.std_dev() - 1.0).abs() < 1e-12);
        let lumo = statistics.lumo().unwrap();
        assert_eq!(lumo.count(), 2);
        assert!((lumo.mean() - 3.0).abs() < 1e-12);
    }

    #[test]
    fn gap_statistics_prefer_stored_then_derived_values() {
        let mut dataset = MolecularDataset::new();
        // Stored gap 0.5 (differs from LUMO - HOMO = 3.0 on purpose).
        dataset.push(
            MolecularEntry::new(test_support::water())
                .unwrap()
                .with_properties(
                    test_support::properties(1.0, 4.0).with_gap(Energy::from_ev(0.5).unwrap()),
                ),
        );
        // No stored gap: derived as LUMO - HOMO = 1.5.
        dataset.push(
            MolecularEntry::new(test_support::methane())
                .unwrap()
                .with_properties(test_support::properties(2.0, 3.5)),
        );
        let statistics = dataset.statistics();
        let gap = statistics.gap().unwrap();
        assert_eq!(gap.count(), 2);
        assert!((gap.min() - 0.5).abs() < 1e-12);
        assert!((gap.max() - 1.5).abs() < 1e-12);
    }

    #[test]
    fn valid_datasets_pass_validation() {
        let mut dataset = MolecularDataset::new();
        dataset.push(MolecularEntry::new(test_support::water()).unwrap());
        dataset.push(MolecularEntry::new(test_support::benzene()).unwrap());
        assert_eq!(dataset.validate(), Ok(()));
    }

    #[test]
    fn datasets_compare_by_value() {
        let mut first = MolecularDataset::new();
        first.push(MolecularEntry::new(test_support::water()).unwrap());
        let mut second = MolecularDataset::new();
        second.push(MolecularEntry::new(test_support::water()).unwrap());
        assert_eq!(first, second);
        second.push(MolecularEntry::new(test_support::methane()).unwrap());
        assert_ne!(first, second);
    }
}
