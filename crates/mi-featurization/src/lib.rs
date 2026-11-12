//! Featurization of molecular structures into ML tensors.
//!
//! This crate is the junction between the chemical domain and the
//! numerical substrate: it turns molecular graphs into `mi-tensor`
//! tensors for the machine-learning phases, without either side knowing
//! about the other. It depends on `mi-chemistry`, `mi-graph`, `mi-data`,
//! and `mi-tensor` — and on no ML: featurization is data preparation, not
//! learning.
//!
//! # What it produces
//!
//! - **Graph tensors** ([`featurize_graph`], [`featurize_molecule`]) —
//!   the node feature matrix, the directed edge index, and the edge
//!   feature matrix, assembled from `mi-graph`'s versioned feature
//!   schemas with schema versions carried as provenance
//!   ([`GraphTensors`]).
//! - **Geometry** ([`distance_matrix`], [`bond_distances`]) — Euclidean
//!   distances in Ångström from the Cartesian positions the I/O layer
//!   entered; topological schemas stay untouched by 3D data.
//! - **Targets** ([`Targets`]) — HOMO, LUMO, and gap as a fixed-layout
//!   values/mask tensor pair in electronvolts, with absence encoded by
//!   the mask, never by a sentinel.
//!
//! ```
//! use mi_featurization::{featurize_molecule, Targets};
//! use mi_chemistry::{Atom, BondType, ChemicalElement, Molecule};
//! use mi_data::{ElectronicProperties, Energy, MolecularEntry};
//! use mi_tensor::Cpu;
//!
//! let device = Cpu::inference();
//!
//! // One water molecule with properties, as the dataset layer stores it.
//! let mut water = Molecule::new().with_name("water");
//! let element = |symbol: &str| ChemicalElement::from_symbol(symbol).unwrap();
//! water.add_atom(Atom::new(element("O")));
//! water.add_atom(Atom::new(element("H")));
//! water.add_atom(Atom::new(element("H")));
//! water.add_bond_between(0, 1, BondType::Single)?;
//! water.add_bond_between(0, 2, BondType::Single)?;
//! let entry = MolecularEntry::new(water.clone())?
//!     .with_properties(
//!         ElectronicProperties::new()
//!             .with_homo(Energy::from_ev(-10.0)?)
//!             .with_lumo(Energy::from_ev(2.0)?),
//!     );
//!
//! // Structure → tensors (schema v1: nodes [3, 6], directed edges [4, 7]).
//! let tensors = featurize_molecule(&water, &device)?;
//! assert_eq!(tensors.nodes().shape().dims(), &[3, 6]);
//! assert_eq!(tensors.edge_index(), &[[0, 1], [1, 0], [0, 2], [2, 0]]);
//!
//! // Properties → targets with an explicit presence mask
//! // (gap absent from the source, derived as LUMO − HOMO).
//! let targets = Targets::extract(&entry);
//! let (values, mask) = targets.to_tensors(&device)?;
//! assert_eq!(values.to_vec()?, vec![-10.0, 2.0, 12.0]);
//! assert_eq!(mask.to_vec()?, vec![1.0, 1.0, 1.0]);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! # Guarantees
//!
//! - **No invented features.** Tensors contain exactly what the versioned
//!   schemas define; widths and versions are read from the schemas, never
//!   hardcoded here.
//! - **No silent physics.** Geometry requires every atom to carry a
//!   position; a partially placed molecule is an error, not zeros.
//! - **Explicit absence.** Edge-less graphs carry `None` edge features;
//!   absent properties carry mask `0.0`.
//! - **Inference only.** Featurization never touches autodiff devices;
//!   gradients begin at Phase 6's models.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod error;
pub mod geometry;
pub mod graph;
pub mod targets;

pub use error::{FeaturizationError, FeaturizationResult};
pub use geometry::{bond_distances, distance_matrix};
pub use graph::{GraphTensors, featurize_graph, featurize_molecule};
pub use targets::Targets;
