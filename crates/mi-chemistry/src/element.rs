//! Chemical element reference data and lookup.
//!
//! # Data model
//!
//! [`ChemicalElement`] is a validated handle around an atomic number. All
//! per-element data (symbol, atomic mass) lives in a single static table, so
//! element information is never duplicated across modules. Instances cannot
//! be constructed with an invalid atomic number: the field is private and the
//! constructors are fallible.
//!
//! # Data provenance and license
//!
//! The element table was transcribed from the PubChem PeriodicTable service
//! (U.S. National Library of Medicine,
//! <https://pubchem.ncbi.nlm.nih.gov/rest/pug/periodictable/JSON>),
//! retrieved 2026-11-10. PubChem data is in the public domain. For elements
//! with an IUPAC standard atomic weight, PubChem reports the conventional
//! weight (occasionally rounded, e.g. lead as 207 u); for elements without a
//! standard atomic weight (Tc, Pm, Po, At, Rn, Fr, Ra, and the synthetic
//! elements), the reported value is the atomic mass of a representative
//! isotope (usually the longest-lived one), not a terrestrial average.
//!
//! Atomic masses are expressed in unified atomic mass units (u; daltons).
//! They are averaged or representative values and must not be confused with
//! the integer isotope *mass number* carried by [`crate::atom::Atom`].

use crate::error::ChemistryError;

/// Reference data for a single element.
struct ElementData {
    /// IUPAC element symbol (e.g. `"H"`, `"He"`, `"Cl"`).
    symbol: &'static str,
    /// Atomic mass in unified atomic mass units (u), as reported by PubChem
    /// (see module documentation for provenance and semantics).
    atomic_mass: f64,
}

/// Reference data for all 118 officially recognized elements, indexed by
/// `atomic_number - 1`.
static ELEMENT_DATA: [ElementData; 118] = [
    ElementData {
        symbol: "H",
        atomic_mass: 1.008,
    },
    ElementData {
        symbol: "He",
        atomic_mass: 4.0026,
    },
    ElementData {
        symbol: "Li",
        atomic_mass: 7.0,
    },
    ElementData {
        symbol: "Be",
        atomic_mass: 9.012183,
    },
    ElementData {
        symbol: "B",
        atomic_mass: 10.81,
    },
    ElementData {
        symbol: "C",
        atomic_mass: 12.011,
    },
    ElementData {
        symbol: "N",
        atomic_mass: 14.007,
    },
    ElementData {
        symbol: "O",
        atomic_mass: 15.999,
    },
    ElementData {
        symbol: "F",
        atomic_mass: 18.99840316,
    },
    ElementData {
        symbol: "Ne",
        atomic_mass: 20.18,
    },
    ElementData {
        symbol: "Na",
        atomic_mass: 22.9897693,
    },
    ElementData {
        symbol: "Mg",
        atomic_mass: 24.305,
    },
    ElementData {
        symbol: "Al",
        atomic_mass: 26.981538,
    },
    ElementData {
        symbol: "Si",
        atomic_mass: 28.085,
    },
    ElementData {
        symbol: "P",
        atomic_mass: 30.973762,
    },
    ElementData {
        symbol: "S",
        atomic_mass: 32.07,
    },
    ElementData {
        symbol: "Cl",
        atomic_mass: 35.45,
    },
    ElementData {
        symbol: "Ar",
        atomic_mass: 39.9,
    },
    ElementData {
        symbol: "K",
        atomic_mass: 39.0983,
    },
    ElementData {
        symbol: "Ca",
        atomic_mass: 40.08,
    },
    ElementData {
        symbol: "Sc",
        atomic_mass: 44.95591,
    },
    ElementData {
        symbol: "Ti",
        atomic_mass: 47.867,
    },
    ElementData {
        symbol: "V",
        atomic_mass: 50.9415,
    },
    ElementData {
        symbol: "Cr",
        atomic_mass: 51.996,
    },
    ElementData {
        symbol: "Mn",
        atomic_mass: 54.93804,
    },
    ElementData {
        symbol: "Fe",
        atomic_mass: 55.84,
    },
    ElementData {
        symbol: "Co",
        atomic_mass: 58.93319,
    },
    ElementData {
        symbol: "Ni",
        atomic_mass: 58.693,
    },
    ElementData {
        symbol: "Cu",
        atomic_mass: 63.55,
    },
    ElementData {
        symbol: "Zn",
        atomic_mass: 65.4,
    },
    ElementData {
        symbol: "Ga",
        atomic_mass: 69.723,
    },
    ElementData {
        symbol: "Ge",
        atomic_mass: 72.63,
    },
    ElementData {
        symbol: "As",
        atomic_mass: 74.92159,
    },
    ElementData {
        symbol: "Se",
        atomic_mass: 78.97,
    },
    ElementData {
        symbol: "Br",
        atomic_mass: 79.9,
    },
    ElementData {
        symbol: "Kr",
        atomic_mass: 83.8,
    },
    ElementData {
        symbol: "Rb",
        atomic_mass: 85.468,
    },
    ElementData {
        symbol: "Sr",
        atomic_mass: 87.62,
    },
    ElementData {
        symbol: "Y",
        atomic_mass: 88.90584,
    },
    ElementData {
        symbol: "Zr",
        atomic_mass: 91.22,
    },
    ElementData {
        symbol: "Nb",
        atomic_mass: 92.90637,
    },
    ElementData {
        symbol: "Mo",
        atomic_mass: 95.95,
    },
    ElementData {
        symbol: "Tc",
        atomic_mass: 96.90636,
    },
    ElementData {
        symbol: "Ru",
        atomic_mass: 101.1,
    },
    ElementData {
        symbol: "Rh",
        atomic_mass: 102.9055,
    },
    ElementData {
        symbol: "Pd",
        atomic_mass: 106.42,
    },
    ElementData {
        symbol: "Ag",
        atomic_mass: 107.868,
    },
    ElementData {
        symbol: "Cd",
        atomic_mass: 112.41,
    },
    ElementData {
        symbol: "In",
        atomic_mass: 114.818,
    },
    ElementData {
        symbol: "Sn",
        atomic_mass: 118.71,
    },
    ElementData {
        symbol: "Sb",
        atomic_mass: 121.76,
    },
    ElementData {
        symbol: "Te",
        atomic_mass: 127.6,
    },
    ElementData {
        symbol: "I",
        atomic_mass: 126.9045,
    },
    ElementData {
        symbol: "Xe",
        atomic_mass: 131.29,
    },
    ElementData {
        symbol: "Cs",
        atomic_mass: 132.905452,
    },
    ElementData {
        symbol: "Ba",
        atomic_mass: 137.33,
    },
    ElementData {
        symbol: "La",
        atomic_mass: 138.9055,
    },
    ElementData {
        symbol: "Ce",
        atomic_mass: 140.116,
    },
    ElementData {
        symbol: "Pr",
        atomic_mass: 140.90766,
    },
    ElementData {
        symbol: "Nd",
        atomic_mass: 144.24,
    },
    ElementData {
        symbol: "Pm",
        atomic_mass: 144.91276,
    },
    ElementData {
        symbol: "Sm",
        atomic_mass: 150.4,
    },
    ElementData {
        symbol: "Eu",
        atomic_mass: 151.964,
    },
    ElementData {
        symbol: "Gd",
        atomic_mass: 157.25,
    },
    ElementData {
        symbol: "Tb",
        atomic_mass: 158.92535,
    },
    ElementData {
        symbol: "Dy",
        atomic_mass: 162.5,
    },
    ElementData {
        symbol: "Ho",
        atomic_mass: 164.93033,
    },
    ElementData {
        symbol: "Er",
        atomic_mass: 167.26,
    },
    ElementData {
        symbol: "Tm",
        atomic_mass: 168.93422,
    },
    ElementData {
        symbol: "Yb",
        atomic_mass: 173.05,
    },
    ElementData {
        symbol: "Lu",
        atomic_mass: 174.9667,
    },
    ElementData {
        symbol: "Hf",
        atomic_mass: 178.49,
    },
    ElementData {
        symbol: "Ta",
        atomic_mass: 180.9479,
    },
    ElementData {
        symbol: "W",
        atomic_mass: 183.84,
    },
    ElementData {
        symbol: "Re",
        atomic_mass: 186.207,
    },
    ElementData {
        symbol: "Os",
        atomic_mass: 190.2,
    },
    ElementData {
        symbol: "Ir",
        atomic_mass: 192.22,
    },
    ElementData {
        symbol: "Pt",
        atomic_mass: 195.08,
    },
    ElementData {
        symbol: "Au",
        atomic_mass: 196.96657,
    },
    ElementData {
        symbol: "Hg",
        atomic_mass: 200.59,
    },
    ElementData {
        symbol: "Tl",
        atomic_mass: 204.383,
    },
    ElementData {
        symbol: "Pb",
        atomic_mass: 207.0,
    },
    ElementData {
        symbol: "Bi",
        atomic_mass: 208.9804,
    },
    ElementData {
        symbol: "Po",
        atomic_mass: 208.98243,
    },
    ElementData {
        symbol: "At",
        atomic_mass: 209.98715,
    },
    ElementData {
        symbol: "Rn",
        atomic_mass: 222.01758,
    },
    ElementData {
        symbol: "Fr",
        atomic_mass: 223.01973,
    },
    ElementData {
        symbol: "Ra",
        atomic_mass: 226.02541,
    },
    ElementData {
        symbol: "Ac",
        atomic_mass: 227.02775,
    },
    ElementData {
        symbol: "Th",
        atomic_mass: 232.038,
    },
    ElementData {
        symbol: "Pa",
        atomic_mass: 231.03588,
    },
    ElementData {
        symbol: "U",
        atomic_mass: 238.0289,
    },
    ElementData {
        symbol: "Np",
        atomic_mass: 237.048172,
    },
    ElementData {
        symbol: "Pu",
        atomic_mass: 244.0642,
    },
    ElementData {
        symbol: "Am",
        atomic_mass: 243.06138,
    },
    ElementData {
        symbol: "Cm",
        atomic_mass: 247.07035,
    },
    ElementData {
        symbol: "Bk",
        atomic_mass: 247.07031,
    },
    ElementData {
        symbol: "Cf",
        atomic_mass: 251.07959,
    },
    ElementData {
        symbol: "Es",
        atomic_mass: 252.083,
    },
    ElementData {
        symbol: "Fm",
        atomic_mass: 257.09511,
    },
    ElementData {
        symbol: "Md",
        atomic_mass: 258.09843,
    },
    ElementData {
        symbol: "No",
        atomic_mass: 259.101,
    },
    ElementData {
        symbol: "Lr",
        atomic_mass: 266.12,
    },
    ElementData {
        symbol: "Rf",
        atomic_mass: 267.122,
    },
    ElementData {
        symbol: "Db",
        atomic_mass: 268.126,
    },
    ElementData {
        symbol: "Sg",
        atomic_mass: 269.128,
    },
    ElementData {
        symbol: "Bh",
        atomic_mass: 270.133,
    },
    ElementData {
        symbol: "Hs",
        atomic_mass: 269.1336,
    },
    ElementData {
        symbol: "Mt",
        atomic_mass: 277.154,
    },
    ElementData {
        symbol: "Ds",
        atomic_mass: 282.166,
    },
    ElementData {
        symbol: "Rg",
        atomic_mass: 282.169,
    },
    ElementData {
        symbol: "Cn",
        atomic_mass: 286.179,
    },
    ElementData {
        symbol: "Nh",
        atomic_mass: 286.182,
    },
    ElementData {
        symbol: "Fl",
        atomic_mass: 290.192,
    },
    ElementData {
        symbol: "Mc",
        atomic_mass: 290.196,
    },
    ElementData {
        symbol: "Lv",
        atomic_mass: 293.205,
    },
    ElementData {
        symbol: "Ts",
        atomic_mass: 294.211,
    },
    ElementData {
        symbol: "Og",
        atomic_mass: 295.216,
    },
];

/// A chemical element of the periodic table.
///
/// An element is identified by its atomic number (proton count) and carries
/// its reference symbol and atomic mass from the static data table. Elements
/// are cheap to copy and compare: two elements are equal exactly when their
/// atomic numbers are equal, and ordering follows the atomic number.
///
/// ```
/// use mi_chemistry::ChemicalElement;
///
/// let carbon = ChemicalElement::from_symbol("C")?;
/// assert_eq!(carbon.atomic_number(), 6);
/// assert_eq!(carbon.symbol(), "C");
/// assert_eq!(carbon.atomic_mass(), 12.011);
/// assert_eq!(ChemicalElement::from_number(6)?, carbon);
/// # Ok::<(), mi_chemistry::ChemistryError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ChemicalElement {
    atomic_number: u8,
}

impl ChemicalElement {
    /// Creates an element from its atomic number.
    ///
    /// Valid atomic numbers are `1..=118` (all officially recognized
    /// elements).
    ///
    /// # Errors
    ///
    /// Returns [`ChemistryError::InvalidAtomicNumber`] if `atomic_number` is
    /// `0` or greater than `118`.
    pub fn from_number(atomic_number: u8) -> Result<Self, ChemistryError> {
        if atomic_number == 0 || atomic_number > 118 {
            return Err(ChemistryError::InvalidAtomicNumber {
                value: atomic_number,
            });
        }
        Ok(Self { atomic_number })
    }

    /// Creates an element from its IUPAC symbol.
    ///
    /// Symbols are case-sensitive: `"He"` is helium, but `"HE"`, `"he"`, and
    /// `"He "` are rejected.
    ///
    /// # Errors
    ///
    /// Returns [`ChemistryError::UnknownElementSymbol`] if no element matches
    /// `symbol`.
    pub fn from_symbol(symbol: &str) -> Result<Self, ChemistryError> {
        ELEMENT_DATA
            .iter()
            .position(|data| data.symbol == symbol)
            .map(|index| Self {
                atomic_number: (index + 1) as u8,
            })
            .ok_or_else(|| ChemistryError::UnknownElementSymbol {
                symbol: symbol.to_string(),
            })
    }

    /// Iterates over all 118 officially recognized elements in
    /// atomic-number order, starting with hydrogen.
    pub fn all() -> impl Iterator<Item = Self> {
        (1..=118).map(|z| Self { atomic_number: z })
    }

    /// Returns the atomic number (proton count), in `1..=118`.
    pub const fn atomic_number(self) -> u8 {
        self.atomic_number
    }

    /// Returns the IUPAC element symbol (e.g. `"H"`, `"C"`, `"Og"`).
    pub fn symbol(self) -> &'static str {
        self.data().symbol
    }

    /// Returns the atomic mass in unified atomic mass units (u).
    ///
    /// This is the conventional standard atomic weight where one exists and a
    /// representative isotope mass otherwise (see the module documentation).
    /// It is an averaged/representative value: it is *not* the exact mass of
    /// a specific isotope and must not be confused with the integer isotope
    /// mass number on [`crate::Atom::with_isotope`].
    pub fn atomic_mass(self) -> f64 {
        self.data().atomic_mass
    }

    /// Returns the static reference data for this element.
    ///
    /// The lookup is infallible: `atomic_number` is in `1..=118` by
    /// construction (private field, fallible constructors).
    fn data(self) -> &'static ElementData {
        &ELEMENT_DATA[usize::from(self.atomic_number) - 1]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn lookup_by_atomic_number_returns_hydrogen() {
        let h = ChemicalElement::from_number(1).unwrap();
        assert_eq!(h.atomic_number(), 1);
        assert_eq!(h.symbol(), "H");
        assert_eq!(h.atomic_mass(), 1.008);
    }

    #[test]
    fn lookup_by_atomic_number_returns_carbon() {
        let c = ChemicalElement::from_number(6).unwrap();
        assert_eq!(c.atomic_number(), 6);
        assert_eq!(c.symbol(), "C");
        assert_eq!(c.atomic_mass(), 12.011);
    }

    #[test]
    fn lookup_by_symbol_returns_matching_element() {
        assert_eq!(
            ChemicalElement::from_symbol("C").unwrap(),
            ChemicalElement::from_number(6).unwrap()
        );
        assert_eq!(
            ChemicalElement::from_symbol("Og").unwrap(),
            ChemicalElement::from_number(118).unwrap()
        );
        assert_eq!(
            ChemicalElement::from_symbol("H").unwrap().atomic_mass(),
            1.008
        );
    }

    #[test]
    fn invalid_atomic_number_zero_is_rejected() {
        assert_eq!(
            ChemicalElement::from_number(0),
            Err(ChemistryError::InvalidAtomicNumber { value: 0 })
        );
    }

    #[test]
    fn invalid_atomic_number_above_118_is_rejected() {
        assert_eq!(
            ChemicalElement::from_number(119),
            Err(ChemistryError::InvalidAtomicNumber { value: 119 })
        );
        assert_eq!(
            ChemicalElement::from_number(255),
            Err(ChemistryError::InvalidAtomicNumber { value: 255 })
        );
    }

    #[test]
    fn unknown_symbol_is_rejected() {
        assert_eq!(
            ChemicalElement::from_symbol("Xx"),
            Err(ChemistryError::UnknownElementSymbol {
                symbol: "Xx".to_string()
            })
        );
    }

    #[test]
    fn symbol_lookup_is_case_sensitive() {
        for bad in ["HE", "he", "c", "CL", "H "] {
            assert!(
                matches!(
                    ChemicalElement::from_symbol(bad),
                    Err(ChemistryError::UnknownElementSymbol { .. })
                ),
                "{bad:?} must not match any element"
            );
        }
    }

    #[test]
    fn all_elements_are_available_and_complete() {
        let elements: Vec<_> = ChemicalElement::all().collect();
        assert_eq!(elements.len(), 118);
        assert_eq!(elements[0].symbol(), "H");
        assert_eq!(elements[117].symbol(), "Og");
        for (expected_z, element) in elements.iter().enumerate() {
            assert_eq!(element.atomic_number() as usize, expected_z + 1);
            assert!(!element.symbol().is_empty());
            assert!(
                element.atomic_mass() > 0.0,
                "{} has no mass",
                element.symbol()
            );
        }
    }

    #[test]
    fn symbols_are_unique() {
        let unique: HashSet<&str> = ChemicalElement::all().map(|e| e.symbol()).collect();
        assert_eq!(unique.len(), 118);
    }

    #[test]
    fn symbol_roundtrip_is_identity() {
        for element in ChemicalElement::all() {
            assert_eq!(ChemicalElement::from_symbol(element.symbol()), Ok(element));
        }
    }

    #[test]
    fn element_equality_and_order_follow_atomic_number() {
        assert_eq!(
            ChemicalElement::from_number(8),
            ChemicalElement::from_symbol("O")
        );
        assert_ne!(
            ChemicalElement::from_number(7),
            ChemicalElement::from_number(8)
        );
        assert!(
            ChemicalElement::from_number(1).unwrap() < ChemicalElement::from_number(2).unwrap()
        );
    }

    #[test]
    fn known_reference_masses_match_the_source() {
        // Values cross-checked against the IUPAC conventional weights as
        // reported by PubChem (see module documentation).
        let cases = [
            ("H", 1.008),
            ("He", 4.0026),
            ("C", 12.011),
            ("N", 14.007),
            ("O", 15.999),
            ("F", 18.99840316),
            ("Cl", 35.45),
            ("Fe", 55.84),
            ("Au", 196.96657),
            ("U", 238.0289),
        ];
        for (symbol, mass) in cases {
            assert_eq!(
                ChemicalElement::from_symbol(symbol).unwrap().atomic_mass(),
                mass
            );
        }
    }
}
