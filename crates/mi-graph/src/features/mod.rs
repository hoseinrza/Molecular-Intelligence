//! Explicit, versioned feature schemas for molecular graphs.
//!
//! Features are kept as *structured records* — [`NodeFeatures`] and
//! [`EdgeFeatures`] — never as undocumented raw vectors. Each record carries
//! a [`FeatureSchema`] that describes, column by column, how its
//! [`to_vector`](node_features::NodeFeatures::to_vector) encoding is built:
//! column names, semantic kinds, and a schema version.
//!
//! # Column kinds
//!
//! - [`FeatureKind::Categorical`] — a discrete identifier whose numerical
//!   value is an **index into a future embedding table** with the given
//!   cardinality (e.g. the atomic number). It must never be interpreted as a
//!   magnitude.
//! - [`FeatureKind::Binary`] — a `0.0`/`1.0` flag, including one-hot
//!   indicator columns.
//! - [`FeatureKind::Numerical`] — a real-valued quantity used as-is.
//!
//! Optional values are encoded explicitly as a presence flag followed by the
//! value (with a documented placeholder when absent); no arbitrary
//! floating-point sentinels are used.
//!
//! # Versioning
//!
//! Any change to a schema's columns, ordering, or encoding bumps its version
//! (see each feature type's `SCHEMA`). Consumers of numerical vectors should
//! check [`FeatureSchema::version`] before relying on a layout.

pub mod edge_features;
pub mod node_features;

pub use edge_features::EdgeFeatures;
pub use node_features::NodeFeatures;

/// Classification of a single feature column.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FeatureKind {
    /// A discrete identifier; the vector carries an integer index into a
    /// future embedding table with `cardinality` entries.
    Categorical {
        /// Number of distinct values the embedding table must cover.
        cardinality: usize,
    },
    /// A `0.0`/`1.0` flag (including one-hot indicator columns).
    Binary,
    /// A real-valued quantity.
    Numerical,
}

/// One column of a [`FeatureSchema`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FeatureColumn {
    name: &'static str,
    kind: FeatureKind,
}

impl FeatureColumn {
    pub(crate) const fn new(name: &'static str, kind: FeatureKind) -> Self {
        Self { name, kind }
    }

    /// Column name (stable within a schema version).
    pub const fn name(&self) -> &'static str {
        self.name
    }

    /// Column kind (categorical, binary, or numerical).
    pub const fn kind(&self) -> FeatureKind {
        self.kind
    }
}

/// Versioned description of a feature record's numerical encoding.
///
/// The `columns` slice has exactly the layout produced by the owning
/// feature record's `to_vector` method.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FeatureSchema {
    version: u32,
    columns: &'static [FeatureColumn],
}

impl FeatureSchema {
    pub(crate) const fn new(version: u32, columns: &'static [FeatureColumn]) -> Self {
        Self { version, columns }
    }

    /// Schema version; bump on any layout change.
    pub const fn version(&self) -> u32 {
        self.version
    }

    /// Columns in vector order.
    pub const fn columns(&self) -> &'static [FeatureColumn] {
        self.columns
    }

    /// Number of numerical components produced by `to_vector`.
    pub const fn width(&self) -> usize {
        self.columns.len()
    }
}
