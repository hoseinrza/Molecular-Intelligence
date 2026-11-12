//! XYZ format support: parsing and writing.
//!
//! XYZ is the lingua franca of coordinate exchange in computational
//! chemistry: element symbols plus Cartesian positions in Ångström, no
//! bonds. This module implements the format faithfully — see
//! [`parser`] for the accepted grammar and [`writer`] for the emitted
//! layout.
//!
//! # Round-trip contract
//!
//! `parse_xyz(write_xyz(molecule))` preserves the element sequence, all
//! coordinates within `1e-6` Å (the writer's decimal precision), and the
//! molecule name. Bonds cannot survive an XYZ round-trip: the format does
//! not carry them.

mod parser;
mod writer;

pub use parser::{parse_xyz, parse_xyz_file};
pub use writer::{write_xyz, write_xyz_file};
