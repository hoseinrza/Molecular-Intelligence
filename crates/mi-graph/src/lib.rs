//! Molecular graph layer for Molecular Intelligence.
//!
//! Converts validated [`mi_chemistry::Molecule`]s into feature-rich graphs:
//! nodes represent atoms, edges represent chemical bonds, and explicit,
//! versioned feature schemas ([`NodeFeatures`], [`EdgeFeatures`]) describe
//! atomic and bond properties. The representation targets future
//! message-passing neural networks but implements no machine learning.
//!
//! # Semantics
//!
//! - Chemical bonds are **undirected**; edges store canonical
//!   (`source < target`) endpoints and preserve bond types exactly —
//!   aromatic bonds stay aromatic, unspecified bonds carry no invented
//!   order.
//! - [`MolecularGraph::directed_edges`] provides the bidirectional
//!   message-passing **view** without duplicating chemical bond identity.
//! - [`MolecularGraph::adjacency`] is a derived, validated adjacency-list
//!   index for O(degree) neighbor lookup.
//! - Node `i` corresponds to atom `i`, edge `j` to bond `j`; the mapping is
//!   explicit and re-checked by [`MolecularGraph::validate`].
//!
//! The graph is purely topological: no 3D coordinates, no quantum
//! chemistry, no valence rules.
//!
//! # Example
//!
//! ```
//! use mi_chemistry::{Atom, BondType, ChemicalElement, Molecule};
//! use mi_graph::MolecularGraph;
//!
//! let mut water = Molecule::new().with_name("water");
//! let o = water.add_atom(Atom::new(ChemicalElement::from_symbol("O")?));
//! let h1 = water.add_atom(Atom::new(ChemicalElement::from_symbol("H")?));
//! let h2 = water.add_atom(Atom::new(ChemicalElement::from_symbol("H")?));
//! water.add_bond_between(o, h1, BondType::Single)?;
//! water.add_bond_between(o, h2, BondType::Single)?;
//!
//! let graph = MolecularGraph::from_molecule(&water)?;
//! assert_eq!(graph.node_count(), 3);
//! assert_eq!(graph.edge_count(), 2);
//! assert_eq!(graph.degree(0)?, 2, "the oxygen has two neighbors");
//! assert_eq!(graph.directed_edge_count(), 4, "two bonds, both directions");
//! assert!(graph.validate().is_ok());
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod adjacency;
pub mod edge;
pub mod error;
pub mod features;
pub mod graph;
pub mod node;

mod conversion;
#[cfg(test)]
mod test_support;

pub use adjacency::{Adjacency, Neighbor};
pub use edge::{DirectedEdge, Edge};
pub use error::GraphError;
pub use features::edge_features::EdgeFeatures;
pub use features::node_features::NodeFeatures;
pub use features::{FeatureColumn, FeatureKind, FeatureSchema};
pub use graph::MolecularGraph;
pub use node::Node;
