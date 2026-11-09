//! Scientific dataset layer for Molecular Intelligence.
//!
//! Typed, validated, in-memory molecular datasets: each entry pairs a
//! validated [`mi_chemistry::Molecule`] with optional
//! [`ElectronicProperties`] — the project's scientific focus (HOMO, LUMO,
//! and the HOMO–LUMO gap) — with energies carried in an explicit
//! electronvolt [`Energy`] type.
//!
//! # Guarantees
//!
//! - A dataset never holds a structurally invalid molecule or a non-finite
//!   energy; validation happens at entry construction and is re-checkable.
//! - Absent properties are explicit [`Option`]s — no sentinels.
//! - Indices are stable: datasets are append-only in this phase.
//! - Every entry exposes an on-demand molecular graph through `mi-graph`,
//!   bridging datasets to future message-passing models.
//!
//! File-format loading, serialization, and train/validation/test splitting
//! are planned future extensions.
//!
//! # Example
//!
//! ```
//! use mi_data::{ElectronicProperties, Energy, MolecularDataset, MolecularEntry};
//! use mi_chemistry::{Atom, BondType, ChemicalElement, Molecule};
//!
//! let mut water = Molecule::new().with_name("water");
//! let o = water.add_atom(Atom::new(ChemicalElement::from_symbol("O")?));
//! let h1 = water.add_atom(Atom::new(ChemicalElement::from_symbol("H")?));
//! let h2 = water.add_atom(Atom::new(ChemicalElement::from_symbol("H")?));
//! water.add_bond_between(o, h1, BondType::Single)?;
//! water.add_bond_between(o, h2, BondType::Single)?;
//!
//! let properties = ElectronicProperties::new()
//!     .with_homo(Energy::from_ev(-10.0)?)
//!     .with_lumo(Energy::from_ev(1.0)?);
//!
//! let mut dataset = MolecularDataset::new().with_name("toy");
//! dataset.push(MolecularEntry::new(water)?.with_properties(properties));
//!
//! assert_eq!(dataset.len(), 1);
//! let entry = dataset.entry(0)?;
//! assert_eq!(entry.properties().homo_lumo_gap().map(Energy::ev), Some(11.0));
//! assert_eq!(entry.graph()?.node_count(), 3);
//!
//! let statistics = dataset.statistics();
//! assert_eq!(statistics.homo().unwrap().count(), 1);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod dataset;
pub mod entry;
pub mod error;
pub mod property;

#[cfg(test)]
mod test_support;

pub use dataset::{DatasetStatistics, MolecularDataset, PropertySummary};
pub use entry::MolecularEntry;
pub use error::DataError;
pub use property::{ElectronicProperties, Energy};
