//! Node representation — the graph view of an atom.

use crate::features::NodeFeatures;
use mi_chemistry::{Atom, ChemicalElement};

/// A graph node representing one atom of the source molecule.
///
/// Node `i` corresponds to atom [`Node::original_atom_index`] of the source
/// molecule. Conversion preserves atom ordering, so the original atom index
/// equals the node's position in [`MolecularGraph::nodes`](crate::MolecularGraph::nodes);
/// the correspondence is re-checked by graph validation rather than trusted.
///
/// A node bundles the element identity (including categorical features such
/// as the atomic number and numerical features such as the formal charge)
/// with topological features (degree, attached bond count) extracted during
/// conversion. Optional values (isotope mass number) stay optional — no
/// sentinel encodings at the structured level.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Node {
    original_atom_index: usize,
    element: ChemicalElement,
    features: NodeFeatures,
}

impl Node {
    /// Creates the node for `original_atom_index` from its atom and
    /// extracted features.
    pub(crate) fn new(original_atom_index: usize, atom: &Atom, features: NodeFeatures) -> Self {
        Self {
            original_atom_index,
            element: atom.element(),
            features,
        }
    }

    /// Index of the atom in the source molecule that this node represents.
    pub const fn original_atom_index(&self) -> usize {
        self.original_atom_index
    }

    /// Chemical element of the atom (element identity feature).
    pub const fn element(&self) -> ChemicalElement {
        self.element
    }

    /// IUPAC symbol of the atom's element (e.g. `"O"`).
    pub fn symbol(&self) -> &'static str {
        self.element.symbol()
    }

    /// Atomic number of the atom (`1..=118`).
    pub const fn atomic_number(&self) -> u8 {
        self.features.atomic_number()
    }

    /// Integer formal charge of the atom.
    pub const fn formal_charge(&self) -> i8 {
        self.features.formal_charge()
    }

    /// Isotope mass number of the atom, when specified on the source atom.
    pub const fn isotope_mass_number(&self) -> Option<u16> {
        self.features.isotope_mass_number()
    }

    /// Degree: the number of adjacent nodes.
    pub const fn degree(&self) -> usize {
        self.features.degree()
    }

    /// Number of attached chemical bonds (equals [`Node::degree`] in the
    /// current simple-graph model; kept separate for future multi-graphs).
    pub const fn bond_count(&self) -> usize {
        self.features.bond_count()
    }

    /// Structured node features (schema: [`NodeFeatures::SCHEMA`]).
    pub const fn features(&self) -> &NodeFeatures {
        &self.features
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support;
    use mi_chemistry::ChemicalElement;

    #[test]
    fn node_preserves_atom_identity_and_topology() {
        let atom = Atom::new(ChemicalElement::from_symbol("N").unwrap())
            .with_formal_charge(1)
            .with_isotope(14)
            .unwrap();
        let features = NodeFeatures::extract(&atom, 4, 4);
        let node = Node::new(7, &atom, features);

        assert_eq!(node.original_atom_index(), 7);
        assert_eq!(node.element(), ChemicalElement::from_symbol("N").unwrap());
        assert_eq!(node.symbol(), "N");
        assert_eq!(node.atomic_number(), 7);
        assert_eq!(node.formal_charge(), 1);
        assert_eq!(node.isotope_mass_number(), Some(14));
        assert_eq!(node.degree(), 4);
        assert_eq!(node.bond_count(), 4);
    }

    #[test]
    fn features_are_accessible_as_a_structured_record() {
        let atom = test_support::water_atom_oxygen();
        let node = Node::new(0, &atom, NodeFeatures::extract(&atom, 2, 2));
        assert_eq!(node.features(), &NodeFeatures::extract(&atom, 2, 2));
        assert_eq!(node.features().atomic_number(), 8);
    }

    #[test]
    fn nodes_compare_by_value() {
        let a = Atom::new(ChemicalElement::from_symbol("O").unwrap());
        let b = Atom::new(ChemicalElement::from_symbol("O").unwrap());
        assert_eq!(
            Node::new(0, &a, NodeFeatures::extract(&a, 1, 1)),
            Node::new(0, &b, NodeFeatures::extract(&b, 1, 1))
        );
        assert_ne!(
            Node::new(0, &a, NodeFeatures::extract(&a, 1, 1)),
            Node::new(1, &a, NodeFeatures::extract(&a, 1, 1))
        );
    }
}
