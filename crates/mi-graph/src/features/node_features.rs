//! Node feature schema and extraction.

use super::{FeatureColumn, FeatureKind, FeatureSchema};
use mi_chemistry::Atom;

/// Columns of the node feature schema, version 1, in vector order.
const COLUMNS_V1: &[FeatureColumn] = &[
    FeatureColumn::new(
        "atomic_number",
        FeatureKind::Categorical { cardinality: 119 },
    ),
    FeatureColumn::new("formal_charge", FeatureKind::Numerical),
    FeatureColumn::new("isotope_present", FeatureKind::Binary),
    FeatureColumn::new("isotope_mass_number", FeatureKind::Numerical),
    FeatureColumn::new("degree", FeatureKind::Numerical),
    FeatureColumn::new("bond_count", FeatureKind::Numerical),
];

/// Structured features of one graph node (one atom).
///
/// The schema (version 1) has six columns:
///
/// | # | Column | Kind | Encoding |
/// | - | ------ | ---- | -------- |
/// | 0 | `atomic_number` | categorical (cardinality 119) | Integer index `1..=118`; index `0` is reserved for padding/unknown in future embedding tables. |
/// | 1 | `formal_charge` | numerical | Signed formal charge, as-is. |
/// | 2 | `isotope_present` | binary | `1.0` when an isotope mass number is specified, else `0.0`. |
/// | 3 | `isotope_mass_number` | numerical | Mass number when present, else `0.0` (meaningful only when column 2 is `1.0`). |
/// | 4 | `degree` | numerical | Number of adjacent nodes. |
/// | 5 | `bond_count` | numerical | Number of attached chemical bonds. |
///
/// Categorical and numerical features stay distinguishable through
/// [`NodeFeatures::SCHEMA`]; the vector encoding is deterministic and
/// versioned.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeFeatures {
    atomic_number: u8,
    formal_charge: i8,
    isotope_mass_number: Option<u16>,
    degree: usize,
    bond_count: usize,
}

impl NodeFeatures {
    /// Feature schema of this record (version 1).
    pub const SCHEMA: FeatureSchema = FeatureSchema::new(1, COLUMNS_V1);

    /// Extracts node features from an atom and its topological position.
    ///
    /// `degree` is the number of adjacent nodes and `bond_count` the number
    /// of attached chemical bonds. In the current simple-graph model the two
    /// are equal; they are kept as separate features for future multi-graph
    /// support.
    pub fn extract(atom: &Atom, degree: usize, bond_count: usize) -> Self {
        Self {
            atomic_number: atom.atomic_number(),
            formal_charge: atom.formal_charge(),
            isotope_mass_number: atom.isotope_mass_number(),
            degree,
            bond_count,
        }
    }

    /// Atomic number (`1..=118`); the categorical element index.
    pub const fn atomic_number(&self) -> u8 {
        self.atomic_number
    }

    /// Integer formal charge.
    pub const fn formal_charge(&self) -> i8 {
        self.formal_charge
    }

    /// Isotope mass number (protons + neutrons), when specified.
    pub const fn isotope_mass_number(&self) -> Option<u16> {
        self.isotope_mass_number
    }

    /// Number of adjacent nodes.
    pub const fn degree(&self) -> usize {
        self.degree
    }

    /// Number of attached chemical bonds.
    pub const fn bond_count(&self) -> usize {
        self.bond_count
    }

    /// Converts the features to a numerical vector following
    /// [`Self::SCHEMA`] (version 1):
    ///
    /// `[atomic_number, formal_charge, isotope_present, isotope_mass_number,
    /// degree, bond_count]`
    ///
    /// The optional isotope is encoded explicitly (presence flag + value
    /// with `0.0` placeholder when absent); no sentinel values are used.
    pub fn to_vector(&self) -> Vec<f64> {
        let (isotope_present, isotope_value) = match self.isotope_mass_number {
            Some(mass_number) => (1.0, f64::from(mass_number)),
            None => (0.0, 0.0),
        };
        vec![
            f64::from(self.atomic_number),
            f64::from(self.formal_charge),
            isotope_present,
            isotope_value,
            self.degree as f64,
            self.bond_count as f64,
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support;
    use mi_chemistry::ChemicalElement;

    #[test]
    fn extract_captures_atom_properties() {
        // A carbon-13 anion: categorical identity, charge, and isotope.
        let carbon13 = Atom::new(ChemicalElement::from_symbol("C").unwrap())
            .with_formal_charge(-1)
            .with_isotope(13)
            .unwrap();
        let features = NodeFeatures::extract(&carbon13, 3, 3);
        assert_eq!(features.atomic_number(), 6);
        assert_eq!(features.formal_charge(), -1);
        assert_eq!(features.isotope_mass_number(), Some(13));
        assert_eq!(features.degree(), 3);
        assert_eq!(features.bond_count(), 3);
    }

    #[test]
    fn to_vector_matches_schema_layout() {
        let features = NodeFeatures::extract(&test_support::water_atom_oxygen(), 2, 2);
        let vector = features.to_vector();
        assert_eq!(vector.len(), NodeFeatures::SCHEMA.width());
        assert_eq!(vector.len(), NodeFeatures::SCHEMA.columns().len());
        let expected = [8.0, 0.0, 0.0, 0.0, 2.0, 2.0];
        assert_eq!(vector, expected);
    }

    #[test]
    fn optional_isotope_is_encoded_explicitly() {
        let deuterium = Atom::new(ChemicalElement::from_symbol("H").unwrap())
            .with_isotope(2)
            .unwrap();
        let with_isotope = NodeFeatures::extract(&deuterium, 1, 1).to_vector();
        assert_eq!(&with_isotope[2..4], &[1.0, 2.0]);

        let plain = Atom::new(ChemicalElement::from_symbol("H").unwrap());
        let without_isotope = NodeFeatures::extract(&plain, 1, 1).to_vector();
        assert_eq!(&without_isotope[2..4], &[0.0, 0.0]);
    }

    #[test]
    fn schema_is_versioned_and_describes_columns() {
        let schema = NodeFeatures::SCHEMA;
        assert_eq!(schema.version(), 1);
        assert_eq!(schema.width(), 6);
        let names: Vec<&str> = schema.columns().iter().map(FeatureColumn::name).collect();
        assert_eq!(
            names,
            [
                "atomic_number",
                "formal_charge",
                "isotope_present",
                "isotope_mass_number",
                "degree",
                "bond_count"
            ]
        );
        // The element identity column is categorical, not numerical.
        assert_eq!(
            schema.columns()[0].kind(),
            FeatureKind::Categorical { cardinality: 119 }
        );
        assert_eq!(schema.columns()[2].kind(), FeatureKind::Binary);
    }
}
