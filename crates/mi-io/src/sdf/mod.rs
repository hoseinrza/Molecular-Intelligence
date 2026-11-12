//! SDF / MDL molfile V2000 support: parsing and writing.
//!
//! SDF is the structure-carrier format of the chemical datasets this
//! pipeline targets: elements, positions in Ångström, bonds with orders,
//! formal charges, and arbitrary per-record data fields (where electronic
//! properties travel in datasets like QM9). See [`parser`] for the
//! supported subset and [`writer`] for the emitted layout.
//!
//! # Round-trip contract
//!
//! `parse_sdf(write_sdf_records(records))` preserves names, elements,
//! positions within `1e-4` Å (the format's standard precision), bond
//! orders, formal charges, and data fields. `BondType::Unspecified`
//! cannot be written (no MDL code) and is rejected on write.

mod parser;
mod writer;

pub use parser::{SdfRecord, parse_sdf, parse_sdf_file};
pub use writer::{write_sdf_file, write_sdf_molecule, write_sdf_record, write_sdf_records};
