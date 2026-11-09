//! Edge representation — the graph view of a chemical bond.

use crate::features::EdgeFeatures;
use mi_chemistry::{Bond, BondType};

/// Returns the chemically defined bond order for a bond type.
///
/// `Some(1)` for single, `Some(2)` for double, `Some(3)` for triple bonds.
/// Aromatic and unspecified bonds yield [`None`]: aromatic bonds are never
/// reduced to an interpolated order (such as 1.5) or to alternating
/// single/double assignments, and no order is invented for unspecified
/// bonds.
pub(crate) fn defined_bond_order(bond_type: BondType) -> Option<u8> {
    match bond_type {
        BondType::Single => Some(1),
        BondType::Double => Some(2),
        BondType::Triple => Some(3),
        BondType::Aromatic | BondType::Unspecified => None,
    }
}

/// A graph edge representing one chemical bond of the source molecule.
///
/// Edges are **undirected**: endpoints are stored canonically
/// (`source < target`), mirroring [`mi_chemistry::Bond`]. Edge `j`
/// corresponds to bond [`Edge::original_bond_index`] of the source molecule;
/// conversion preserves bond ordering, so the original bond index equals the
/// edge's position in [`MolecularGraph::edges`](crate::MolecularGraph::edges);
/// the correspondence is re-checked by graph validation rather than trusted.
///
/// The bond type is preserved exactly — in particular, aromatic bonds stay
/// aromatic and are never silently rewritten as single or double bonds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Edge {
    original_bond_index: usize,
    source: usize,
    target: usize,
    features: EdgeFeatures,
}

impl Edge {
    /// Creates the edge for `original_bond_index` from its chemical bond.
    pub(crate) fn new(original_bond_index: usize, bond: &Bond) -> Self {
        let (source, target) = bond.endpoints();
        Self {
            original_bond_index,
            source,
            target,
            features: EdgeFeatures::extract(bond),
        }
    }

    /// Index of the bond in the source molecule that this edge represents.
    pub const fn original_bond_index(&self) -> usize {
        self.original_bond_index
    }

    /// Smaller endpoint of the undirected edge (canonical order).
    pub const fn source(&self) -> usize {
        self.source
    }

    /// Larger endpoint of the undirected edge (canonical order).
    pub const fn target(&self) -> usize {
        self.target
    }

    /// Both endpoints in canonical `(smaller, larger)` order.
    pub const fn endpoints(&self) -> (usize, usize) {
        (self.source, self.target)
    }

    /// Structured edge features (schema: [`EdgeFeatures::SCHEMA`]).
    pub const fn features(&self) -> &EdgeFeatures {
        &self.features
    }

    /// Bond type of the underlying chemical bond.
    pub const fn bond_type(&self) -> BondType {
        self.features.bond_type()
    }

    /// Chemically defined bond order (`1`, `2`, `3`), or [`None`] for
    /// aromatic and unspecified bonds.
    pub const fn bond_order(&self) -> Option<u8> {
        self.features.bond_order()
    }
}

/// One direction of an undirected chemical edge — the message-passing view.
///
/// Chemical bonds are undirected, but message-passing neural networks
/// typically require directed edges from every source node to every target
/// node. A `DirectedEdge` is a lightweight view over one direction: it
/// carries only indices and **does not duplicate the chemical bond
/// identity** — both directions of a bond resolve to the same
/// [`Edge`] through [`DirectedEdge::chemical_edge_index`], which also equals
/// the original bond index in the source molecule.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DirectedEdge {
    source: usize,
    target: usize,
    chemical_edge_index: usize,
}

impl DirectedEdge {
    /// Creates the directed view `(source -> target)` backed by the undirected
    /// edge at `chemical_edge_index`.
    pub(crate) const fn new(source: usize, target: usize, chemical_edge_index: usize) -> Self {
        Self {
            source,
            target,
            chemical_edge_index,
        }
    }

    /// Node the message flows from.
    pub const fn source(&self) -> usize {
        self.source
    }

    /// Node the message flows to.
    pub const fn target(&self) -> usize {
        self.target
    }

    /// Index of the backing undirected chemical edge; equals the original
    /// bond index in the source molecule. Both directions of a bond share
    /// this index.
    pub const fn chemical_edge_index(&self) -> usize {
        self.chemical_edge_index
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support;

    #[test]
    fn edge_preserves_bond_identity_and_canonical_endpoints() {
        let bond = Bond::new(3, 1, BondType::Double).unwrap();
        let edge = Edge::new(5, &bond);
        assert_eq!(edge.original_bond_index(), 5);
        assert_eq!(edge.endpoints(), (1, 3));
        assert_eq!(edge.source(), 1);
        assert_eq!(edge.target(), 3);
        assert_eq!(edge.bond_type(), BondType::Double);
        assert_eq!(edge.bond_order(), Some(2));
    }

    #[test]
    fn aromatic_edge_keeps_its_type_without_an_order() {
        let edge = Edge::new(0, &test_support::aromatic_bond());
        assert_eq!(edge.bond_type(), BondType::Aromatic);
        assert_eq!(edge.bond_order(), None);
    }

    #[test]
    fn defined_bond_order_covers_only_classical_orders() {
        assert_eq!(defined_bond_order(BondType::Single), Some(1));
        assert_eq!(defined_bond_order(BondType::Double), Some(2));
        assert_eq!(defined_bond_order(BondType::Triple), Some(3));
        assert_eq!(defined_bond_order(BondType::Aromatic), None);
        assert_eq!(defined_bond_order(BondType::Unspecified), None);
    }

    #[test]
    fn directed_edge_is_a_view_over_the_chemical_edge() {
        let directed = DirectedEdge::new(4, 2, 7);
        assert_eq!(directed.source(), 4);
        assert_eq!(directed.target(), 2);
        assert_eq!(directed.chemical_edge_index(), 7);
        // The opposite direction shares the chemical edge identity.
        let opposite = DirectedEdge::new(2, 4, 7);
        assert_eq!(
            directed.chemical_edge_index(),
            opposite.chemical_edge_index()
        );
        assert_ne!(directed, opposite);
    }

    #[test]
    fn edges_compare_by_value() {
        let first = Edge::new(0, &Bond::new(0, 1, BondType::Single).unwrap());
        let second = Edge::new(0, &Bond::new(1, 0, BondType::Single).unwrap());
        assert_eq!(first, second, "endpoint order must not affect equality");
    }
}
