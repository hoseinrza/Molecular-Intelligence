//! Atom representation.

use crate::element::ChemicalElement;
use crate::error::ChemistryError;
use crate::position::Position;

/// A single atom: a chemical element plus per-atom chemical state and an
/// optional Cartesian position.
///
/// An atom bundles a validated [`ChemicalElement`] (identity) with:
///
/// - **Formal charge** — the integer charge assigned by standard valence and
///   formal-charge rules (e.g. `+1` for the nitrogen of an ammonium ion,
///   `-1` for a hydroxide oxygen). It is a bookkeeping quantity, *not* a
///   partial charge from quantum chemistry.
/// - **Isotope mass number** — optional; the integer count of protons plus
///   neutrons (A) of one specific isotope (e.g. `2` for deuterium, `13` for
///   carbon-13). It is distinct from the element's averaged
///   [`atomic mass`](ChemicalElement::atomic_mass) and from the exact
///   isotopic mass in u (which is a future extension).
/// - **Position** — optional; a validated Cartesian position in Ångström
///   (see [`Position`]). Coordinate-based file formats (XYZ, molfiles)
///   always carry one; purely topological uses may omit it.
/// - **Label** — optional free-form bookkeeping text (e.g. `"C_alpha"`);
///   never interpreted chemically.
///
/// Atoms are immutable after construction; use the builder-style methods to
/// configure one:
///
/// ```
/// use mi_chemistry::{Atom, ChemicalElement, Position};
///
/// // A bare carbon atom.
/// let carbon = Atom::new(ChemicalElement::from_symbol("C")?);
///
/// // Carbon-13 carrying a negative formal charge, placed in space.
/// let carbanion_c13 = Atom::new(ChemicalElement::from_symbol("C")?)
///     .with_formal_charge(-1)
///     .with_isotope(13)?
///     .with_position(Position::new(0.0, 0.0, 1.42)?);
/// assert_eq!(carbanion_c13.formal_charge(), -1);
/// assert_eq!(carbanion_c13.isotope_mass_number(), Some(13));
/// assert_eq!(carbanion_c13.position().map(Position::z), Some(1.42));
/// # Ok::<(), mi_chemistry::ChemistryError>(())
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct Atom {
    element: ChemicalElement,
    formal_charge: i8,
    isotope_mass_number: Option<u16>,
    position: Option<Position>,
    label: Option<String>,
}

impl Atom {
    /// Creates an atom of `element` with formal charge `0`, no isotope
    /// specification, no position, and no label.
    ///
    /// The element has already been validated during its own construction,
    /// so this constructor is infallible.
    pub fn new(element: ChemicalElement) -> Self {
        Self {
            element,
            formal_charge: 0,
            isotope_mass_number: None,
            position: None,
            label: None,
        }
    }

    /// Sets the formal charge and returns the atom.
    ///
    /// Any signed charge is accepted: this crate intentionally makes no
    /// assumption about typical oxidation ranges.
    pub fn with_formal_charge(mut self, formal_charge: i8) -> Self {
        self.formal_charge = formal_charge;
        self
    }

    /// Specifies the isotope by mass number and returns the atom.
    ///
    /// The mass number A is the integer count of protons plus neutrons. It
    /// can never be smaller than the atomic number Z (there is no isotope
    /// without the element's protons), so `mass_number < atomic_number` is
    /// rejected.
    ///
    /// # Errors
    ///
    /// Returns [`ChemistryError::InvalidIsotopeMassNumber`] if `mass_number`
    /// is smaller than the element's atomic number (including `0`).
    pub fn with_isotope(mut self, mass_number: u16) -> Result<Self, ChemistryError> {
        let minimum_valid = u16::from(self.element.atomic_number());
        if mass_number < minimum_valid {
            return Err(ChemistryError::InvalidIsotopeMassNumber {
                value: mass_number,
                element_symbol: self.element.symbol(),
                minimum_valid,
            });
        }
        self.isotope_mass_number = Some(mass_number);
        Ok(self)
    }

    /// Sets a free-form label and returns the atom.
    ///
    /// Labels are bookkeeping metadata only; they are never interpreted
    /// chemically.
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    /// Sets the Cartesian position and returns the atom.
    ///
    /// [`Position`] has already been validated during its own construction
    /// (finite components), so this builder is infallible.
    pub fn with_position(mut self, position: Position) -> Self {
        self.position = Some(position);
        self
    }

    /// Returns the chemical element of this atom.
    pub fn element(&self) -> ChemicalElement {
        self.element
    }

    /// Returns the atomic number (proton count) of this atom's element.
    pub fn atomic_number(&self) -> u8 {
        self.element.atomic_number()
    }

    /// Returns the IUPAC symbol of this atom's element (e.g. `"C"`).
    pub fn symbol(&self) -> &'static str {
        self.element.symbol()
    }

    /// Returns the averaged/representative atomic mass of this atom's
    /// element in unified atomic mass units (u).
    ///
    /// Note: this is the element-level value even when an isotope is
    /// specified. The exact mass of a specific isotope is a future
    /// extension; only the integer mass number is carried per atom.
    pub fn atomic_mass(&self) -> f64 {
        self.element.atomic_mass()
    }

    /// Returns the formal charge (0 unless explicitly set).
    pub fn formal_charge(&self) -> i8 {
        self.formal_charge
    }

    /// Returns the isotope mass number, if one was specified.
    pub fn isotope_mass_number(&self) -> Option<u16> {
        self.isotope_mass_number
    }

    /// Returns the Cartesian position in Ångström, if one was set.
    pub fn position(&self) -> Option<Position> {
        self.position
    }

    /// Returns the label, if one was set.
    pub fn label(&self) -> Option<&str> {
        self.label.as_deref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn carbon() -> ChemicalElement {
        ChemicalElement::from_symbol("C").unwrap()
    }

    fn hydrogen() -> ChemicalElement {
        ChemicalElement::from_symbol("H").unwrap()
    }

    #[test]
    fn valid_construction_has_neutral_defaults() {
        let atom = Atom::new(carbon());
        assert_eq!(atom.element(), carbon());
        assert_eq!(atom.atomic_number(), 6);
        assert_eq!(atom.symbol(), "C");
        assert_eq!(atom.atomic_mass(), 12.011);
        assert_eq!(atom.formal_charge(), 0);
        assert_eq!(atom.isotope_mass_number(), None);
        assert_eq!(atom.label(), None);
    }

    #[test]
    fn formal_charge_is_stored_and_returned() {
        // Oxide-style oxygen in a hydroxide ion.
        let hydroxide_oxygen =
            Atom::new(ChemicalElement::from_symbol("O").unwrap()).with_formal_charge(-1);
        assert_eq!(hydroxide_oxygen.formal_charge(), -1);
        // Ammonium nitrogen.
        let ammonium_nitrogen =
            Atom::new(ChemicalElement::from_symbol("N").unwrap()).with_formal_charge(1);
        assert_eq!(ammonium_nitrogen.formal_charge(), 1);
        // Charge zero is explicit and distinct from unset (there is no unset).
        assert_eq!(Atom::new(carbon()).with_formal_charge(0).formal_charge(), 0);
    }

    #[test]
    fn isotope_mass_number_is_represented() {
        // Deuterium: hydrogen with A = 2.
        let deuterium = Atom::new(hydrogen()).with_isotope(2).unwrap();
        assert_eq!(deuterium.isotope_mass_number(), Some(2));
        assert_eq!(deuterium.atomic_number(), 1);
        // Carbon-13.
        let carbon13 = Atom::new(carbon()).with_isotope(13).unwrap();
        assert_eq!(carbon13.isotope_mass_number(), Some(13));
    }

    #[test]
    fn isotope_mass_number_must_not_be_below_atomic_number() {
        // Carbon-3 is physically impossible: A cannot be smaller than Z.
        assert_eq!(
            Atom::new(carbon()).with_isotope(3),
            Err(ChemistryError::InvalidIsotopeMassNumber {
                value: 3,
                element_symbol: "C",
                minimum_valid: 6,
            })
        );
    }

    #[test]
    fn isotope_mass_number_zero_is_invalid_even_for_hydrogen() {
        assert_eq!(
            Atom::new(hydrogen()).with_isotope(0),
            Err(ChemistryError::InvalidIsotopeMassNumber {
                value: 0,
                element_symbol: "H",
                minimum_valid: 1,
            })
        );
    }

    #[test]
    fn position_is_stored_and_optional() {
        let bare = Atom::new(carbon());
        assert_eq!(bare.position(), None);
        let placed = bare
            .clone()
            .with_position(Position::new(0.5, -1.25, 3.0).unwrap());
        assert_eq!(
            placed.position(),
            Some(Position::new(0.5, -1.25, 3.0).unwrap())
        );
        // Atoms differing only in position compare unequal.
        assert_ne!(bare, placed);
    }

    #[test]
    fn label_is_stored_verbatim() {
        let atom = Atom::new(carbon()).with_label("C_alpha");
        assert_eq!(atom.label(), Some("C_alpha"));
        assert_eq!(
            atom.with_label(String::from("owned")).label(),
            Some("owned")
        );
    }

    #[test]
    fn atoms_compare_by_value() {
        assert_eq!(Atom::new(carbon()), Atom::new(carbon()));
        assert_ne!(Atom::new(carbon()), Atom::new(hydrogen()));
        assert_ne!(
            Atom::new(carbon()),
            Atom::new(carbon()).with_formal_charge(1)
        );
        assert_ne!(
            Atom::new(carbon()).with_isotope(12).unwrap(),
            Atom::new(carbon()).with_isotope(13).unwrap()
        );
        assert_ne!(
            Atom::new(carbon()).with_label("a"),
            Atom::new(carbon()).with_label("b")
        );
    }
}
