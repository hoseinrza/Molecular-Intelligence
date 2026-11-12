//! Scientific provenance metadata for imported datasets.
//!
//! Where a dataset came from and how its numbers were produced is part of
//! the science: an energy computed at B3LYP/6-31G(d) in Hartree is a
//! different observable from one measured or computed differently. Every
//! import attaches a [`DatasetMetadata`] record so a dataset can never
//! become anonymous — the unit question ("is this HOMO in eV or Hartree?")
//! is answered by data, not by convention.

use crate::error::{IoError, IoResult};
use std::fmt;

/// The unit electronic energies are reported in by the source data.
///
/// `mi-data` stores all energies as [`electronvolts`](mi_data::Energy);
/// the unit recorded here tells the importer **which conversion, if any**,
/// the source values need on their way in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EnergyUnit {
    /// Electronvolt — already the storage unit; no conversion.
    Ev,
    /// Hartree atomic unit of energy — converted at import using the
    /// CODATA 2018 factor carried by [`mi_data::Energy::HARTREE_TO_EV`].
    Hartree,
}

impl fmt::Display for EnergyUnit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Ev => "eV",
            Self::Hartree => "Eh",
        })
    }
}

/// Scientific provenance of one imported dataset.
///
/// Construction requires the two fields no dataset can do without — the
/// **source** and the **energy unit** — and every remaining field is an
/// explicit, validated builder. Empty or whitespace-only content is
/// rejected so provenance records cannot degrade into empty strings.
///
/// ```
/// use mi_io::{DatasetMetadata, EnergyUnit};
///
/// let metadata = DatasetMetadata::new("QM9", EnergyUnit::Hartree)?
///     .with_method("DFT")?
///     .with_functional("B3LYP")?
///     .with_basis_set("6-31G(2df,p)")?
///     .with_source_id("doi:10.1039/C4SC03082J")?
///     .with_version("1.0")?;
///
/// let rendered = metadata.to_string();
/// assert!(rendered.contains("source: QM9"));
/// assert!(rendered.contains("energy_unit: Eh"));
/// assert!(rendered.contains("functional: B3LYP"));
/// # Ok::<(), mi_io::IoError>(())
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DatasetMetadata {
    source: String,
    method: Option<String>,
    functional: Option<String>,
    basis_set: Option<String>,
    energy_unit: EnergyUnit,
    source_id: Option<String>,
    version: Option<String>,
}

impl DatasetMetadata {
    /// Creates metadata from a required data source name and energy unit.
    ///
    /// # Errors
    ///
    /// Returns [`IoError::InvalidMetadata`] if `source` is empty or
    /// whitespace-only.
    pub fn new(source: impl Into<String>, energy_unit: EnergyUnit) -> IoResult<Self> {
        let source = require_field("source", source)?;
        Ok(Self {
            source,
            method: None,
            functional: None,
            basis_set: None,
            energy_unit,
            source_id: None,
            version: None,
        })
    }

    /// Records the electronic-structure method (e.g. `"DFT"`, `"HF"`,
    /// `"CCSD(T)"`).
    ///
    /// # Errors
    ///
    /// Returns [`IoError::InvalidMetadata`] for empty content.
    pub fn with_method(mut self, method: impl Into<String>) -> IoResult<Self> {
        self.method = Some(require_field("method", method)?);
        Ok(self)
    }

    /// Records the exchange-correlation functional for DFT sources (e.g.
    /// `"B3LYP"`).
    ///
    /// # Errors
    ///
    /// Returns [`IoError::InvalidMetadata`] for empty content.
    pub fn with_functional(mut self, functional: impl Into<String>) -> IoResult<Self> {
        self.functional = Some(require_field("functional", functional)?);
        Ok(self)
    }

    /// Records the atomic orbital basis set (e.g. `"6-31G(2df,p)"`).
    ///
    /// # Errors
    ///
    /// Returns [`IoError::InvalidMetadata`] for empty content.
    pub fn with_basis_set(mut self, basis_set: impl Into<String>) -> IoResult<Self> {
        self.basis_set = Some(require_field("basis_set", basis_set)?);
        Ok(self)
    }

    /// Records a resolvable identifier for the source (DOI, URL, or
    /// database key).
    ///
    /// # Errors
    ///
    /// Returns [`IoError::InvalidMetadata`] for empty content.
    pub fn with_source_id(mut self, source_id: impl Into<String>) -> IoResult<Self> {
        self.source_id = Some(require_field("source_id", source_id)?);
        Ok(self)
    }

    /// Records the dataset version tag (e.g. `"1.0"` or a commit hash).
    ///
    /// # Errors
    ///
    /// Returns [`IoError::InvalidMetadata`] for empty content.
    pub fn with_version(mut self, version: impl Into<String>) -> IoResult<Self> {
        self.version = Some(require_field("version", version)?);
        Ok(self)
    }

    /// The data source name (e.g. `"QM9"`).
    pub fn source(&self) -> &str {
        &self.source
    }

    /// The electronic-structure method, if recorded.
    pub fn method(&self) -> Option<&str> {
        self.method.as_deref()
    }

    /// The DFT functional, if recorded.
    pub fn functional(&self) -> Option<&str> {
        self.functional.as_deref()
    }

    /// The basis set, if recorded.
    pub fn basis_set(&self) -> Option<&str> {
        self.basis_set.as_deref()
    }

    /// The unit the source reports energies in.
    pub fn energy_unit(&self) -> EnergyUnit {
        self.energy_unit
    }

    /// The source identifier (DOI, URL, ...), if recorded.
    pub fn source_id(&self) -> Option<&str> {
        self.source_id.as_deref()
    }

    /// The dataset version tag, if recorded.
    pub fn version(&self) -> Option<&str> {
        self.version.as_deref()
    }
}

impl fmt::Display for DatasetMetadata {
    /// Renders a stable, human-readable provenance record; every recorded
    /// field appears exactly once as `key: value`, and unrecorded fields
    /// are omitted (absence stays explicit).
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "source: {}", self.source)?;
        writeln!(f, "energy_unit: {}", self.energy_unit)?;
        for (label, value) in [
            ("method", self.method.as_deref()),
            ("functional", self.functional.as_deref()),
            ("basis_set", self.basis_set.as_deref()),
            ("source_id", self.source_id.as_deref()),
            ("version", self.version.as_deref()),
        ] {
            if let Some(value) = value {
                writeln!(f, "{label}: {value}")?;
            }
        }
        Ok(())
    }
}

/// Trims a builder field and rejects empty content.
fn require_field(field: &'static str, value: impl Into<String>) -> IoResult<String> {
    let value = value.into();
    if value.trim().is_empty() {
        return Err(IoError::InvalidMetadata { field });
    }
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn required_fields_build_a_record() {
        let metadata = DatasetMetadata::new("QM9", EnergyUnit::Hartree).unwrap();
        assert_eq!(metadata.source(), "QM9");
        assert_eq!(metadata.energy_unit(), EnergyUnit::Hartree);
        assert_eq!(metadata.method(), None);
        assert_eq!(metadata.functional(), None);
        assert_eq!(metadata.basis_set(), None);
        assert_eq!(metadata.source_id(), None);
        assert_eq!(metadata.version(), None);
    }

    #[test]
    fn builder_fields_are_stored() {
        let metadata = DatasetMetadata::new("QM9", EnergyUnit::Ev)
            .unwrap()
            .with_method("DFT")
            .unwrap()
            .with_functional("B3LYP")
            .unwrap()
            .with_basis_set("6-31G(2df,p)")
            .unwrap()
            .with_source_id("doi:10.1039/C4SC03082J")
            .unwrap()
            .with_version("1.0")
            .unwrap();
        assert_eq!(metadata.method(), Some("DFT"));
        assert_eq!(metadata.functional(), Some("B3LYP"));
        assert_eq!(metadata.basis_set(), Some("6-31G(2df,p)"));
        assert_eq!(metadata.source_id(), Some("doi:10.1039/C4SC03082J"));
        assert_eq!(metadata.version(), Some("1.0"));
    }

    #[test]
    fn empty_fields_are_rejected() {
        for error in [
            DatasetMetadata::new("", EnergyUnit::Ev).unwrap_err(),
            DatasetMetadata::new("   ", EnergyUnit::Ev).unwrap_err(),
            DatasetMetadata::new("QM9", EnergyUnit::Ev)
                .unwrap()
                .with_method("  ")
                .unwrap_err(),
        ] {
            assert!(matches!(error, IoError::InvalidMetadata { .. }), "{error}");
        }
    }

    #[test]
    fn display_omits_unrecorded_fields() {
        let metadata = DatasetMetadata::new("toy", EnergyUnit::Ev).unwrap();
        let rendered = metadata.to_string();
        assert!(rendered.contains("source: toy"));
        assert!(rendered.contains("energy_unit: eV"));
        assert!(!rendered.contains("functional"));
    }

    #[test]
    fn energy_unit_displays_in_standard_notation() {
        assert_eq!(EnergyUnit::Ev.to_string(), "eV");
        assert_eq!(EnergyUnit::Hartree.to_string(), "Eh");
    }
}
