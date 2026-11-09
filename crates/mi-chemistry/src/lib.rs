//! Chemistry core for Molecular Intelligence.
//!
//! Foundational molecular data structures: chemical elements, atoms, bonds,
//! and molecules, with structural validation. Element reference data covers
//! all 118 officially recognized elements (see [`element`] for provenance).
//!
//! # Deliberate scope (Phase 1)
//!
//! This crate implements the *data model only*. It does not parse any
//! molecular format (no SMILES, XYZ, SDF, or MOL files), does not implement
//! graph algorithms, 3D geometry, valence rules, or any quantum chemistry.
//! Parsing and algorithms arrive in later phases on top of these types.
//!
//! # Structure
//!
//! - [`ChemicalElement`] — a validated handle for one of the 118 elements.
//! - [`Atom`] — an element with formal charge, optional isotope mass number,
//!   and optional label.
//! - [`Bond`] / [`BondType`] — an undirected bond between two atom indices.
//! - [`Molecule`] — an indexed, append-only collection of atoms and bonds.
//! - [`ChemistryError`] — all fallible operations report through this type.
//!
//! # Example
//!
//! Build water (H₂O) and validate its structure:
//!
//! ```
//! use mi_chemistry::{Atom, BondType, ChemicalElement, ChemistryError, Molecule};
//!
//! let mut water = Molecule::new().with_name("water").with_identifier("pubchem:962");
//! let oxygen = ChemicalElement::from_symbol("O")?;
//! let hydrogen = ChemicalElement::from_symbol("H")?;
//!
//! let o = water.add_atom(Atom::new(oxygen));
//! let h1 = water.add_atom(Atom::new(hydrogen));
//! let h2 = water.add_atom(Atom::new(hydrogen));
//!
//! water.add_bond_between(o, h1, BondType::Single)?;
//! water.add_bond_between(o, h2, BondType::Single)?;
//!
//! assert_eq!(water.atom_count(), 3);
//! assert_eq!(water.bond_count(), 2);
//! assert_eq!(water.atom(0)?.symbol(), "O");
//!
//! water.validate()?;
//! # Ok::<(), ChemistryError>(())
//! ```

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod atom;
pub mod bond;
pub mod element;
pub mod error;
pub mod molecule;

mod validation;

pub use atom::Atom;
pub use bond::{Bond, BondType};
pub use element::ChemicalElement;
pub use error::ChemistryError;
pub use molecule::Molecule;
