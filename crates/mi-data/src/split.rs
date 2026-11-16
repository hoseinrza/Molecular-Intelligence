//! Reproducible dataset splitting.
//!
//! Scientific results are only meaningful if they can be regenerated
//! exactly, so a split is a function of the dataset and an explicit seed —
//! never of ambient state. The shuffle is a **Fisher–Yates permutation of
//! entry indices driven by a SplitMix64 generator**, implemented here
//! rather than pulled in as a dependency; the algorithm is pinned by
//! tests, so a given `(dataset, fractions, seed)` triple yields the same
//! split in every build of this crate.
//!
//! # Scope and honest limitations
//!
//! A **random split** puts chemically similar molecules (near-duplicate
//! conformers, homologs) on both sides of the divide, so measured error
//! can overstate generalization. A **scaffold split** (grouping by Bemis–
//! Murcko framework) is the standard harder evaluation and is planned as
//! a follow-up — benchmark reports must state which split they used and
//! ideally report both.

use crate::dataset::MolecularDataset;
use crate::error::DataError;

/// Relative sizes of the three splits; must each be strictly inside
/// `(0, 1)` and sum to 1.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SplitFractions {
    /// Fraction assigned to training.
    pub train: f64,
    /// Fraction assigned to validation.
    pub validation: f64,
    /// Fraction assigned to testing.
    pub test: f64,
}

impl SplitFractions {
    /// The canonical 80/10/10 split used by QM9-style benchmarks.
    pub const DEFAULT: Self = Self {
        train: 0.8,
        validation: 0.1,
        test: 0.1,
    };

    /// Validates a fraction triple.
    ///
    /// # Errors
    ///
    /// Returns [`DataError::InvalidSplit`] when any fraction is outside
    /// `(0, 1)` or the fractions do not sum to 1.
    pub fn new(train: f64, validation: f64, test: f64) -> Result<Self, DataError> {
        for (name, value) in [("train", train), ("validation", validation), ("test", test)] {
            if !(value > 0.0 && value < 1.0) {
                return Err(DataError::InvalidSplit {
                    reason: format!("{name} fraction must be in (0, 1), got {value}"),
                });
            }
        }
        let sum = train + validation + test;
        if (sum - 1.0).abs() > 1e-9 {
            return Err(DataError::InvalidSplit {
                reason: format!("fractions must sum to 1, got {sum}"),
            });
        }
        Ok(Self {
            train,
            validation,
            test,
        })
    }
}

impl Default for SplitFractions {
    fn default() -> Self {
        Self::DEFAULT
    }
}

/// A partition of one dataset into train/validation/test subsets.
///
/// Subsets are independent [`MolecularDataset`]s (cloned entries), named
/// after their role with the source dataset's name as a prefix so
/// provenance travels into every artifact built from them.
#[derive(Debug, Clone, PartialEq)]
pub struct Split {
    train: MolecularDataset,
    validation: MolecularDataset,
    test: MolecularDataset,
}

impl Split {
    /// The training subset.
    pub fn train(&self) -> &MolecularDataset {
        &self.train
    }

    /// The validation subset.
    pub fn validation(&self) -> &MolecularDataset {
        &self.validation
    }

    /// The test subset.
    pub fn test(&self) -> &MolecularDataset {
        &self.test
    }

    /// Total entries across all subsets (equals the source dataset size).
    pub fn total_size(&self) -> usize {
        self.train.len() + self.validation.len() + self.test.len()
    }
}

/// Partitions a dataset into reproducible train/validation/test subsets.
///
/// Entry indices are shuffled with Fisher–Yates under `seed`; the first
/// `train·N` indices form the training subset, the next `validation·N`
/// the validation subset, and the remainder the test subset (so the test
/// subset absorbs rounding and the partition covers the dataset exactly,
/// disjointly).
///
/// # Errors
///
/// Returns [`DataError::InvalidSplit`] for invalid fractions or an empty
/// dataset, and [`DataError::InvalidEntryIndex`] never occurs because the
/// permutation is built from the dataset's own length.
pub fn random_split(
    dataset: &MolecularDataset,
    fractions: SplitFractions,
    seed: u64,
) -> Result<Split, DataError> {
    let total = dataset.len();
    if total == 0 {
        return Err(DataError::InvalidSplit {
            reason: "cannot split an empty dataset".into(),
        });
    }
    let train_size = (fractions.train * total as f64) as usize;
    let validation_size = (fractions.validation * total as f64) as usize;

    let mut order: Vec<usize> = (0..total).collect();
    let mut rng = SplitMix64::new(seed);
    for i in (1..order.len()).rev() {
        let j = rng.below(i + 1);
        order.swap(i, j);
    }

    let build = |indices: &[usize], role: &str| -> MolecularDataset {
        let mut subset = MolecularDataset::new();
        for &index in indices {
            if let Ok(entry) = dataset.entry(index) {
                subset.push(entry.clone());
            }
        }
        let name = match dataset.name() {
            Some(name) => format!("{name}-{role}"),
            None => role.to_string(),
        };
        subset.with_name(name)
    };

    Ok(Split {
        train: build(&order[..train_size], "train"),
        validation: build(
            &order[train_size..train_size + validation_size],
            "validation",
        ),
        test: build(&order[train_size + validation_size..], "test"),
    })
}

/// SplitMix64: a compact, stateless-in-spirit generator whose output
/// sequence is fixed by its specification, used only to permute indices.
struct SplitMix64(u64);

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// A value in `0..bound`. Modulo reduction carries a bias far below
    /// one part in 2^53 for the dataset sizes this crate targets, which
    /// is documented rather than hidden behind rejection sampling.
    fn below(&mut self, bound: usize) -> usize {
        (self.next_u64() % bound as u64) as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MolecularEntry;
    use crate::property::{ElectronicProperties, Energy};
    use mi_chemistry::{Atom, ChemicalElement, Molecule};

    /// A dataset of `count` distinguishable entries (one helium atom each,
    /// indexed by label so membership is checkable).
    fn fixture(count: usize) -> MolecularDataset {
        let mut dataset = MolecularDataset::new().with_name("fixture");
        let helium = ChemicalElement::from_symbol("He").unwrap();
        for index in 0..count {
            let mut molecule = Molecule::new();
            molecule.add_atom(Atom::new(helium).with_label(format!("entry-{index}")));
            let mut properties = ElectronicProperties::new();
            if index % 2 == 0 {
                properties = properties.with_homo(Energy::from_ev(index as f64).unwrap());
            }
            dataset.push(
                MolecularEntry::new(molecule)
                    .unwrap()
                    .with_properties(properties),
            );
        }
        dataset
    }

    fn labels(dataset: &MolecularDataset) -> Vec<String> {
        dataset
            .iter()
            .filter_map(|entry| {
                entry
                    .molecule()
                    .atoms()
                    .first()
                    .and_then(|atom| atom.label().map(str::to_string))
            })
            .collect()
    }

    #[test]
    fn default_fractions_are_eighty_ten_ten() {
        let fractions = SplitFractions::default();
        assert_eq!(
            (fractions.train, fractions.validation, fractions.test),
            (0.8, 0.1, 0.1)
        );
        assert_eq!(
            SplitFractions::new(0.8, 0.1, 0.1).unwrap(),
            SplitFractions::DEFAULT
        );
    }

    #[test]
    fn invalid_fractions_are_rejected() {
        for error in [
            SplitFractions::new(0.0, 0.5, 0.5).unwrap_err(), // train not in (0,1)
            SplitFractions::new(1.0, 0.0, 0.0).unwrap_err(), // validation not in (0,1)
            SplitFractions::new(-0.2, 0.6, 0.6).unwrap_err(),
            SplitFractions::new(0.8, 0.1, 0.05).unwrap_err(), // sum ≠ 1
        ] {
            assert!(matches!(error, DataError::InvalidSplit { .. }), "{error}");
        }
    }

    #[test]
    fn split_sizes_follow_fractions_and_cover_exactly() {
        let dataset = fixture(1000);
        let split = random_split(&dataset, SplitFractions::DEFAULT, 42).unwrap();
        assert_eq!(split.train().len(), 800);
        assert_eq!(split.validation().len(), 100);
        assert_eq!(split.test().len(), 100);
        assert_eq!(split.total_size(), 1000);

        // Subsets are disjoint and jointly cover the source.
        let mut seen = Vec::new();
        for subset in [split.train(), split.validation(), split.test()] {
            seen.extend(labels(subset));
        }
        seen.sort();
        seen.dedup();
        assert_eq!(seen.len(), 1000, "subsets must be disjoint and cover");
    }

    #[test]
    fn same_seed_reproduces_the_same_split() {
        let dataset = fixture(500);
        let first = random_split(&dataset, SplitFractions::DEFAULT, 12345).unwrap();
        let second = random_split(&dataset, SplitFractions::DEFAULT, 12345).unwrap();
        assert_eq!(labels(first.train()), labels(second.train()));
        assert_eq!(labels(first.test()), labels(second.test()));
        assert_eq!(first, second);
    }

    #[test]
    fn different_seeds_shuffle_differently() {
        let dataset = fixture(500);
        let a = random_split(&dataset, SplitFractions::DEFAULT, 1).unwrap();
        let b = random_split(&dataset, SplitFractions::DEFAULT, 2).unwrap();
        assert_ne!(labels(a.train()), labels(b.train()));
    }

    #[test]
    fn subset_names_carry_provenance() {
        let dataset = fixture(30);
        let split = random_split(&dataset, SplitFractions::DEFAULT, 7).unwrap();
        assert_eq!(split.train().name(), Some("fixture-train"));
        assert_eq!(split.validation().name(), Some("fixture-validation"));
        assert_eq!(split.test().name(), Some("fixture-test"));
    }

    #[test]
    fn properties_travel_into_subsets() {
        // Even-indexed entries carry a HOMO; each subset must keep exactly
        // its share of them (statistical sanity, exact share not asserted).
        let dataset = fixture(400);
        let split = random_split(&dataset, SplitFractions::DEFAULT, 9).unwrap();
        let with_homo = |dataset: &MolecularDataset| {
            dataset
                .iter()
                .filter(|entry| entry.properties().homo().is_some())
                .count()
        };
        let total =
            with_homo(split.train()) + with_homo(split.validation()) + with_homo(split.test());
        assert_eq!(total, with_homo(&dataset));
    }

    #[test]
    fn empty_dataset_cannot_be_split() {
        let dataset = MolecularDataset::new();
        let error = random_split(&dataset, SplitFractions::DEFAULT, 1).unwrap_err();
        assert!(error.to_string().contains("empty"));
    }

    #[test]
    fn statistics_are_independent_per_subset() {
        // Entries at even indices carry HOMO = index (eV); the training
        // subset's mean must equal the mean of its own members' values.
        let dataset = fixture(400);
        let split = random_split(&dataset, SplitFractions::DEFAULT, 11).unwrap();
        let train = split.train();
        let statistics = train.statistics();
        let summary = statistics.homo().unwrap();
        let manual_mean: f64 = train
            .iter()
            .filter_map(|entry| entry.properties().homo())
            .map(|energy| energy.ev())
            .sum::<f64>()
            / summary.count() as f64;
        assert!((summary.mean() - manual_mean).abs() < 1e-9);
    }
}
