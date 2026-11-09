//! Molecular properties with explicit units.
//!
//! Energies are carried in a dedicated [`Energy`] newtype (electronvolts)
//! so that unit confusion — e.g. mixing electronvolts with Hartree atomic
//! units — is a type error, not a silent numerical bug.

use crate::error::DataError;

/// An energy value in electronvolts (eV).
///
/// A validated wrapper around `f64`: only finite values can be constructed,
/// so every [`Energy`] is safe to compare and average. Conversions to and
/// from Hartree atomic units use the CODATA 2018 factor (see
/// [`Energy::HARTREE_TO_EV`]).
///
/// ```
/// use mi_data::Energy;
///
/// let gap = Energy::from_ev(10.5)?;
/// assert_eq!(gap.ev(), 10.5);
///
/// // One Hartree atomic unit of energy.
/// let hartree = Energy::from_hartree(1.0)?;
/// assert_eq!(hartree.ev(), Energy::HARTREE_TO_EV);
/// assert!((hartree.hartree() - 1.0).abs() < 1e-12);
/// # Ok::<(), mi_data::DataError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Energy {
    /// Energy in electronvolts (private; only finite values enter here).
    ev: f64,
}

impl Energy {
    /// Hartree-to-electronvolt conversion factor, CODATA 2018
    /// (NIST/CIAAW recommended value of the hartree in eV).
    pub const HARTREE_TO_EV: f64 = 27.211_386_245_988;

    /// Creates an energy from a value in electronvolts.
    ///
    /// # Errors
    ///
    /// Returns [`DataError::InvalidEnergy`] if `ev` is NaN or infinite.
    pub fn from_ev(ev: f64) -> Result<Self, DataError> {
        if !ev.is_finite() {
            return Err(DataError::InvalidEnergy { value: ev });
        }
        Ok(Self { ev })
    }

    /// Creates an energy from a value in Hartree atomic units.
    ///
    /// # Errors
    ///
    /// Returns [`DataError::InvalidEnergy`] if the result is NaN or
    /// infinite.
    pub fn from_hartree(hartree: f64) -> Result<Self, DataError> {
        Self::from_ev(hartree * Self::HARTREE_TO_EV)
    }

    /// Returns the energy in electronvolts (eV).
    pub const fn ev(self) -> f64 {
        self.ev
    }

    /// Returns the energy in Hartree atomic units (E_h).
    pub fn hartree(self) -> f64 {
        self.ev / Self::HARTREE_TO_EV
    }
}

/// Electronic structure properties of one molecule.
///
/// The project's scientific focus — HOMO, LUMO, and the HOMO–LUMO gap —
/// represented as optional [`Energy`] values: datasets may carry any subset
/// of the properties, and absence is explicit ([`None`]), never a sentinel.
///
/// The gap is treated as follows:
/// - a **stored** gap (as provided by sources such as QM9) is returned by
///   [`ElectronicProperties::homo_lumo_gap`] as-is;
/// - when no gap is stored but both HOMO and LUMO are, the gap is
///   **derived** as `LUMO − HOMO`;
/// - otherwise there is no gap.
///
/// All values are in electronvolts via [`Energy`].
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct ElectronicProperties {
    homo: Option<Energy>,
    lumo: Option<Energy>,
    gap: Option<Energy>,
}

impl ElectronicProperties {
    /// Creates an empty property set (all properties absent).
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the HOMO energy and returns the property set.
    pub fn with_homo(mut self, homo: Energy) -> Self {
        self.homo = Some(homo);
        self
    }

    /// Sets the LUMO energy and returns the property set.
    pub fn with_lumo(mut self, lumo: Energy) -> Self {
        self.lumo = Some(lumo);
        self
    }

    /// Sets the HOMO–LUMO gap and returns the property set.
    pub fn with_gap(mut self, gap: Energy) -> Self {
        self.gap = Some(gap);
        self
    }

    /// HOMO energy, if present.
    pub fn homo(&self) -> Option<Energy> {
        self.homo
    }

    /// LUMO energy, if present.
    pub fn lumo(&self) -> Option<Energy> {
        self.lumo
    }

    /// Stored HOMO–LUMO gap, if present (see
    /// [`homo_lumo_gap`](Self::homo_lumo_gap) for the effective value).
    pub fn gap(&self) -> Option<Energy> {
        self.gap
    }

    /// Effective HOMO–LUMO gap: the stored value when present, otherwise
    /// derived as `LUMO − HOMO` when both are available, otherwise
    /// [`None`].
    pub fn homo_lumo_gap(&self) -> Option<Energy> {
        self.gap.or_else(|| match (self.lumo, self.homo) {
            (Some(lumo), Some(homo)) => Energy::from_ev(lumo.ev() - homo.ev()).ok(),
            _ => None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn energy_roundtrips_through_ev() {
        let energy = Energy::from_ev(-10.5).unwrap();
        assert_eq!(energy.ev(), -10.5);
        assert!((energy.hartree() - -10.5 / Energy::HARTREE_TO_EV).abs() < 1e-12);
    }

    #[test]
    fn one_hartree_converts_exactly_to_codata_factor() {
        let hartree = Energy::from_hartree(1.0).unwrap();
        assert_eq!(hartree.ev(), Energy::HARTREE_TO_EV);
        assert_eq!(hartree.hartree(), 1.0);
    }

    #[test]
    fn non_finite_energies_are_rejected() {
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            // NaN cannot be compared with PartialEq, so variant matching is
            // used instead of value equality.
            assert!(
                matches!(Energy::from_ev(bad), Err(DataError::InvalidEnergy { .. })),
                "from_ev({bad}) must be rejected"
            );
            assert!(
                matches!(
                    Energy::from_hartree(bad),
                    Err(DataError::InvalidEnergy { .. })
                ),
                "from_hartree({bad}) must be rejected"
            );
        }
        // The rejected value is reported verbatim.
        let Err(DataError::InvalidEnergy { value }) = Energy::from_ev(f64::NAN) else {
            panic!("NaN must be rejected");
        };
        assert!(value.is_nan());
    }

    #[test]
    fn energies_order_by_value() {
        assert!(Energy::from_ev(-5.0).unwrap() < Energy::from_ev(0.0).unwrap());
        assert!(Energy::from_ev(1.0).unwrap() <= Energy::from_ev(1.0).unwrap());
    }

    #[test]
    fn default_properties_are_absent() {
        let properties = ElectronicProperties::new();
        assert_eq!(properties.homo(), None);
        assert_eq!(properties.lumo(), None);
        assert_eq!(properties.gap(), None);
        assert_eq!(properties.homo_lumo_gap(), None);
    }

    #[test]
    fn builders_store_each_property() {
        let properties = ElectronicProperties::new()
            .with_homo(Energy::from_ev(-10.0).unwrap())
            .with_lumo(Energy::from_ev(1.0).unwrap())
            .with_gap(Energy::from_ev(11.0).unwrap());
        assert_eq!(properties.homo().map(Energy::ev), Some(-10.0));
        assert_eq!(properties.lumo().map(Energy::ev), Some(1.0));
        assert_eq!(properties.gap().map(Energy::ev), Some(11.0));
    }

    #[test]
    fn gap_is_returned_stored_when_present() {
        let properties = ElectronicProperties::new()
            .with_homo(Energy::from_ev(-10.0).unwrap())
            .with_lumo(Energy::from_ev(1.0).unwrap())
            .with_gap(Energy::from_ev(99.0).unwrap());
        // The stored value wins, even when it disagrees with LUMO - HOMO:
        // sources are trusted for what they provide.
        assert_eq!(properties.homo_lumo_gap().map(Energy::ev), Some(99.0));
    }

    #[test]
    fn gap_is_derived_when_absent_but_homo_and_lumo_exist() {
        let properties = ElectronicProperties::new()
            .with_homo(Energy::from_ev(-10.0).unwrap())
            .with_lumo(Energy::from_ev(1.0).unwrap());
        assert_eq!(properties.homo_lumo_gap().map(Energy::ev), Some(11.0));
    }

    #[test]
    fn gap_is_absent_without_sufficient_inputs() {
        let only_homo = ElectronicProperties::new().with_homo(Energy::from_ev(-10.0).unwrap());
        assert_eq!(only_homo.homo_lumo_gap(), None);
        let only_lumo = ElectronicProperties::new().with_lumo(Energy::from_ev(1.0).unwrap());
        assert_eq!(only_lumo.homo_lumo_gap(), None);
    }
}
