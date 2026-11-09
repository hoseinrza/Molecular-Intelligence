//! Edge feature schema and extraction.

use super::{FeatureColumn, FeatureKind, FeatureSchema};
use crate::edge::defined_bond_order;
use mi_chemistry::{Bond, BondType};

/// Bond types in one-hot column order (declaration order of [`BondType`]).
const BOND_TYPES: [BondType; 5] = [
    BondType::Single,
    BondType::Double,
    BondType::Triple,
    BondType::Aromatic,
    BondType::Unspecified,
];

/// Columns of the edge feature schema, version 1, in vector order.
const COLUMNS_V1: &[FeatureColumn] = &[
    FeatureColumn::new("bond_type_is_single", FeatureKind::Binary),
    FeatureColumn::new("bond_type_is_double", FeatureKind::Binary),
    FeatureColumn::new("bond_type_is_triple", FeatureKind::Binary),
    FeatureColumn::new("bond_type_is_aromatic", FeatureKind::Binary),
    FeatureColumn::new("bond_type_is_unspecified", FeatureKind::Binary),
    FeatureColumn::new("has_bond_order", FeatureKind::Binary),
    FeatureColumn::new("bond_order", FeatureKind::Numerical),
];

/// Structured features of one graph edge (one chemical bond).
///
/// The schema (version 1) has seven columns:
///
/// | # | Column | Kind | Encoding |
/// | - | ------ | ---- | -------- |
/// | 0–4 | `bond_type_is_{single,double,triple,aromatic,unspecified}` | binary | One-hot encoding of the bond type in [`BondType`] declaration order. |
/// | 5 | `has_bond_order` | binary | `1.0` when a chemically defined bond order exists, else `0.0`. |
/// | 6 | `bond_order` | numerical | `1.0`, `2.0`, or `3.0` when defined, else `0.0` (meaningful only when column 5 is `1.0`). |
///
/// Aromatic and unspecified bonds are **never** assigned an invented order:
/// they are distinguishable through the one-hot columns and carry
/// `bond_order = None` in the structured record.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EdgeFeatures {
    bond_type: BondType,
    bond_order: Option<u8>,
}

impl EdgeFeatures {
    /// Feature schema of this record (version 1).
    pub const SCHEMA: FeatureSchema = FeatureSchema::new(1, COLUMNS_V1);

    /// Extracts edge features from a chemical bond.
    ///
    /// The bond order is populated only when chemically defined: `1` for
    /// single, `2` for double, `3` for triple. Aromatic and unspecified
    /// bonds carry [`None`].
    pub fn extract(bond: &Bond) -> Self {
        Self {
            bond_type: bond.bond_type(),
            bond_order: defined_bond_order(bond.bond_type()),
        }
    }

    /// Bond type of the underlying chemical bond.
    pub const fn bond_type(&self) -> BondType {
        self.bond_type
    }

    /// Chemically defined bond order (`1`, `2`, `3`), or [`None`] for
    /// aromatic and unspecified bonds.
    pub const fn bond_order(&self) -> Option<u8> {
        self.bond_order
    }

    /// Converts the features to a numerical vector following
    /// [`Self::SCHEMA`] (version 1):
    ///
    /// `[is_single, is_double, is_triple, is_aromatic, is_unspecified,
    /// has_bond_order, bond_order]`
    pub fn to_vector(&self) -> Vec<f64> {
        let mut vector = Vec::with_capacity(COLUMNS_V1.len());
        for candidate in BOND_TYPES {
            vector.push(f64::from(u8::from(self.bond_type == candidate)));
        }
        match self.bond_order {
            Some(order) => {
                vector.push(1.0);
                vector.push(f64::from(order));
            }
            None => {
                vector.push(0.0);
                vector.push(0.0);
            }
        }
        vector
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bond_order_is_extracted_only_when_chemically_defined() {
        let cases = [
            (BondType::Single, Some(1)),
            (BondType::Double, Some(2)),
            (BondType::Triple, Some(3)),
            (BondType::Aromatic, None),
            (BondType::Unspecified, None),
        ];
        for (bond_type, expected_order) in cases {
            let bond = Bond::new(0, 1, bond_type).unwrap();
            let features = EdgeFeatures::extract(&bond);
            assert_eq!(features.bond_type(), bond_type);
            assert_eq!(features.bond_order(), expected_order, "{bond_type:?}");
        }
    }

    #[test]
    fn aromatic_bonds_are_not_treated_as_single_or_double() {
        let aromatic = Bond::new(0, 1, BondType::Aromatic).unwrap();
        let vector = EdgeFeatures::extract(&aromatic).to_vector();
        // One-hot hits only the aromatic column.
        assert_eq!(&vector[0..5], &[0.0, 0.0, 0.0, 1.0, 0.0]);
        // No order is invented.
        assert_eq!(&vector[5..7], &[0.0, 0.0]);
    }

    #[test]
    fn to_vector_matches_schema_layout() {
        let double_bond = Bond::new(0, 1, BondType::Double).unwrap();
        let vector = EdgeFeatures::extract(&double_bond).to_vector();
        assert_eq!(vector.len(), EdgeFeatures::SCHEMA.width());
        assert_eq!(vector, [0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn unspecified_bond_carries_no_order() {
        let unspecified = Bond::new(2, 5, BondType::Unspecified).unwrap();
        let features = EdgeFeatures::extract(&unspecified);
        assert_eq!(features.bond_order(), None);
        assert_eq!(features.to_vector()[5..7], [0.0, 0.0]);
    }

    #[test]
    fn schema_is_versioned_and_describes_columns() {
        let schema = EdgeFeatures::SCHEMA;
        assert_eq!(schema.version(), 1);
        assert_eq!(schema.width(), 7);
        let names: Vec<&str> = schema.columns().iter().map(FeatureColumn::name).collect();
        assert_eq!(
            names,
            [
                "bond_type_is_single",
                "bond_type_is_double",
                "bond_type_is_triple",
                "bond_type_is_aromatic",
                "bond_type_is_unspecified",
                "has_bond_order",
                "bond_order"
            ]
        );
        assert!(schema
            .columns()
            .iter()
            .all(|column| column.kind() == FeatureKind::Binary || column.name() == "bond_order"));
    }

    #[test]
    fn extraction_matches_the_source_bond() {
        let triple = Bond::new(0, 1, BondType::Triple).unwrap();
        let features = EdgeFeatures::extract(&triple);
        assert_eq!(features.bond_type(), BondType::Triple);
        assert_eq!(features.bond_order(), Some(3));
    }
}
