//! Regression targets from electronic properties.
//!
//! Training a property-prediction model needs the observables as numbers.
//! `mi-data` keeps properties as explicit [`Option`]s (absence is never a
//! sentinel), while tensors need a fixed-width layout — so this module
//! converts with the **values + mask** convention: one tensor of values in
//! a fixed `[homo, lumo, gap]` layout, and one tensor of presence flags.
//! Missing properties are `0.0` with mask `0.0`; nothing is invented.
//!
//! Units are **electronvolts** — `mi-data`'s storage unit. Sources in
//! Hartree were already converted at the I/O boundary; this module never
//! re-converts or guesses units.

use crate::error::FeaturizationResult;
use mi_data::{Energy, MolecularEntry};
use mi_tensor::{Device, Tensor, TensorShape};

/// The electronic regression targets of one molecule, in electronvolts.
///
/// The fixed layout is `[homo, lumo, gap]` (see [`Targets::WIDTH`]). The
/// gap is the **effective** gap per `mi-data` semantics: the stored value
/// when the source provides one, otherwise derived as `LUMO − HOMO` when
/// both levels are present, otherwise absent.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Targets {
    /// Highest occupied molecular orbital energy (eV), if present.
    pub homo: Option<f32>,
    /// Lowest unoccupied molecular orbital energy (eV), if present.
    pub lumo: Option<f32>,
    /// HOMO–LUMO gap (eV), if present or derivable.
    pub gap: Option<f32>,
}

impl Targets {
    /// Width of the fixed target layout: `[homo, lumo, gap]`.
    pub const WIDTH: usize = 3;

    /// Extracts targets from a dataset entry.
    ///
    /// Every energy is exactly representable as `f32` within the accuracy
    /// of source data; the narrowing mirrors the schema-vector narrowing
    /// in [`crate::graph`].
    pub fn extract(entry: &MolecularEntry) -> Self {
        let properties = entry.properties();
        let as_f32 = |energy: Option<Energy>| energy.map(|energy| energy.ev() as f32);
        Self {
            homo: as_f32(properties.homo()),
            lumo: as_f32(properties.lumo()),
            // The effective gap: stored when present, derived when both
            // levels exist — matching `mi-data`'s own statistics.
            gap: as_f32(properties.homo_lumo_gap()),
        }
    }

    /// Returns `(values, mask)`, both rank-1 tensors of length
    /// [`Targets::WIDTH`].
    ///
    /// `values` holds each present target (and `0.0` for absent slots);
    /// `mask` holds `1.0` for present slots and `0.0` for absent ones.
    /// Model code multiplies element-wise, so absent slots contribute
    /// nothing to a loss without any special-casing.
    ///
    /// # Errors
    ///
    /// Returns [`mi_tensor::TensorError`] if tensor construction fails.
    pub fn to_tensors(&self, device: &Device) -> FeaturizationResult<(Tensor<1>, Tensor<1>)> {
        let mut values = Vec::with_capacity(Self::WIDTH);
        let mut mask = Vec::with_capacity(Self::WIDTH);
        for value in [self.homo, self.lumo, self.gap] {
            match value {
                Some(value) => {
                    values.push(value);
                    mask.push(1.0);
                }
                None => {
                    values.push(0.0);
                    mask.push(0.0);
                }
            }
        }
        let shape = TensorShape::new(vec![Self::WIDTH])?;
        let values = Tensor::from_data(values, shape.clone(), device)?;
        let mask = Tensor::from_data(mask, shape, device)?;
        Ok((values, mask))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mi_chemistry::{Atom, ChemicalElement, Molecule};
    use mi_data::ElectronicProperties;
    use mi_tensor::Cpu;

    fn entry_with(properties: ElectronicProperties) -> MolecularEntry {
        let mut molecule = Molecule::new();
        molecule.add_atom(Atom::new(ChemicalElement::from_symbol("He").unwrap()));
        MolecularEntry::new(molecule)
            .unwrap()
            .with_properties(properties)
    }

    #[test]
    fn full_properties_extract_to_exact_values() {
        let properties = ElectronicProperties::new()
            .with_homo(Energy::from_ev(-10.0).unwrap())
            .with_lumo(Energy::from_ev(1.5).unwrap())
            .with_gap(Energy::from_ev(11.5).unwrap());
        let targets = Targets::extract(&entry_with(properties));
        assert_eq!(targets.homo, Some(-10.0));
        assert_eq!(targets.lumo, Some(1.5));
        assert_eq!(targets.gap, Some(11.5));
    }

    #[test]
    fn gap_is_derived_when_only_levels_are_present() {
        let properties = ElectronicProperties::new()
            .with_homo(Energy::from_ev(-5.0).unwrap())
            .with_lumo(Energy::from_ev(2.0).unwrap());
        let targets = Targets::extract(&entry_with(properties));
        assert_eq!(targets.gap, Some(7.0), "derived as LUMO − HOMO");
    }

    #[test]
    fn absent_properties_stay_absent() {
        let targets = Targets::extract(&entry_with(ElectronicProperties::new()));
        assert_eq!(targets.homo, None);
        assert_eq!(targets.lumo, None);
        assert_eq!(targets.gap, None);
    }

    #[test]
    fn tensors_carry_values_and_mask() {
        // Only levels are present: the gap slot carries the derived gap.
        let properties = ElectronicProperties::new()
            .with_homo(Energy::from_ev(-10.0).unwrap())
            .with_lumo(Energy::from_ev(1.0).unwrap());
        let targets = Targets::extract(&entry_with(properties));
        let (values, mask) = targets.to_tensors(&Cpu::inference()).unwrap();
        assert_eq!(values.shape().dims(), &[3]);
        assert_eq!(mask.shape().dims(), &[3]);
        assert_eq!(values.to_vec().unwrap(), vec![-10.0, 1.0, 11.0]);
        assert_eq!(mask.to_vec().unwrap(), vec![1.0, 1.0, 1.0]);

        // Lumo only: the gap is neither stored nor derivable.
        let properties = ElectronicProperties::new().with_lumo(Energy::from_ev(1.0).unwrap());
        let targets = Targets::extract(&entry_with(properties));
        let (values, mask) = targets.to_tensors(&Cpu::inference()).unwrap();
        assert_eq!(values.to_vec().unwrap(), vec![0.0, 1.0, 0.0]);
        assert_eq!(mask.to_vec().unwrap(), vec![0.0, 1.0, 0.0]);
    }

    #[test]
    fn fully_absent_targets_produce_zero_mask() {
        let targets = Targets::extract(&entry_with(ElectronicProperties::new()));
        let (values, mask) = targets.to_tensors(&Cpu::inference()).unwrap();
        assert_eq!(values.to_vec().unwrap(), vec![0.0, 0.0, 0.0]);
        assert_eq!(mask.to_vec().unwrap(), vec![0.0, 0.0, 0.0]);
    }
}
