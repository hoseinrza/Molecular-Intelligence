//! Molecule-to-graph conversion.
//!
//! Conversion is the only way to obtain a [`MolecularGraph`]. It treats
//! `mi-chemistry` as the source of chemical truth: the input molecule is
//! validated first, atom ordering, bond ordering, and bond types are
//! preserved exactly, and features are extracted from the validated data.

use crate::adjacency::Adjacency;
use crate::edge::Edge;
use crate::error::GraphError;
use crate::features::NodeFeatures;
use crate::graph::MolecularGraph;
use crate::node::Node;
use mi_chemistry::Molecule;

impl MolecularGraph {
    /// Builds a molecular graph from a molecule.
    ///
    /// - Node `i` corresponds to atom `i` of `molecule` (atom ordering
    ///   preserved).
    /// - Edge `j` corresponds to bond `j` (bond ordering, endpoints, and
    ///   bond types preserved exactly; aromatic bonds stay aromatic).
    /// - Node features (element identity, formal charge, isotope, degree,
    ///   bond count) and edge features (bond type, chemically defined bond
    ///   order) are extracted during conversion.
    /// - The molecule's name and identifier are carried over as graph
    ///   metadata.
    /// - Isolated atoms and disconnected components are supported.
    ///
    /// # Errors
    ///
    /// Returns [`GraphError::InvalidMolecule`] if `molecule` fails its
    /// structural validation (in particular, empty molecules are rejected —
    /// a molecular graph needs at least one node). The wrapped
    /// [`mi_chemistry::ChemistryError`] describes the exact violation.
    ///
    /// # Example
    ///
    /// ```
    /// # use mi_chemistry::{Atom, BondType, ChemicalElement, Molecule};
    /// # use mi_graph::MolecularGraph;
    /// # fn build_water() -> Result<Molecule, mi_chemistry::ChemistryError> {
    /// #     let mut water = Molecule::new().with_name("water");
    /// #     let o = water.add_atom(Atom::new(ChemicalElement::from_symbol("O")?));
    /// #     let h1 = water.add_atom(Atom::new(ChemicalElement::from_symbol("H")?));
    /// #     let h2 = water.add_atom(Atom::new(ChemicalElement::from_symbol("H")?));
    /// #     water.add_bond_between(o, h1, BondType::Single)?;
    /// #     water.add_bond_between(o, h2, BondType::Single)?;
    /// #     Ok(water)
    /// # }
    /// let water = build_water()?;
    /// let graph = MolecularGraph::from_molecule(&water)?;
    ///
    /// println!("Nodes: {}", graph.node_count());
    /// println!("Edges: {}", graph.edge_count());
    /// assert_eq!(graph.node_count(), 3);
    /// assert_eq!(graph.edge_count(), 2);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn from_molecule(molecule: &Molecule) -> Result<Self, GraphError> {
        // The chemistry core is the source of truth: reject invalid input
        // before any graph data is produced.
        molecule.validate().map_err(GraphError::InvalidMolecule)?;

        let edges: Vec<Edge> = molecule
            .bonds()
            .iter()
            .enumerate()
            .map(|(bond_index, bond)| Edge::new(bond_index, bond))
            .collect();
        let adjacency = Adjacency::from_edges(molecule.atom_count(), &edges);

        let nodes: Vec<Node> = molecule
            .atoms()
            .iter()
            .enumerate()
            .map(|(atom_index, atom)| {
                // The adjacency covers exactly the molecule's atoms, so the
                // degree is available; a gap would be an internal bug, not
                // user input, and is reported instead of defaulted.
                let degree = adjacency
                    .degree(atom_index)
                    .ok_or(GraphError::InconsistentGraph {
                        reason: format!("adjacency is missing node {atom_index}"),
                    })?;
                // In the simple-graph model, degree and attached bond count
                // are equal; they are recorded as separate features.
                Ok(Node::new(
                    atom_index,
                    atom,
                    NodeFeatures::extract(atom, degree, degree),
                ))
            })
            .collect::<Result<Vec<Node>, GraphError>>()?;

        let graph = Self::new(
            molecule.name().map(str::to_owned),
            molecule.identifier().map(str::to_owned),
            nodes,
            edges,
            adjacency,
        );
        // Construction above is proven correct, but validation re-checks the
        // full invariant set and defines the contract for the future.
        graph.validate()?;
        Ok(graph)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support;
    use mi_chemistry::{Bond, BondType, ChemistryError};

    #[test]
    fn rejects_empty_molecules_with_descriptive_error() {
        let error = MolecularGraph::from_molecule(&Molecule::new()).unwrap_err();
        assert!(matches!(
            error,
            GraphError::InvalidMolecule(ChemistryError::InvalidMolecularStructure { .. })
        ));
        // The chemistry error stays reachable through the error chain.
        assert_eq!(
            std::error::Error::source(&error).unwrap().to_string(),
            "invalid molecular structure: molecule contains no atoms"
        );
    }

    #[test]
    fn invalid_bond_endpoints_cannot_reach_conversion() {
        // The chemistry core rejects out-of-range endpoints at bond
        // insertion, so conversion never observes invalid references.
        let mut molecule = test_support::water();
        let dangling = Bond::new(0, 9, BondType::Single).unwrap();
        assert_eq!(
            molecule.add_bond(dangling),
            Err(ChemistryError::InvalidAtomIndex {
                index: 9,
                atom_count: 3
            })
        );
        // The molecule is unchanged and still converts cleanly.
        assert!(MolecularGraph::from_molecule(&molecule).is_ok());
    }

    #[test]
    fn duplicate_bonds_cannot_reach_conversion() {
        let mut molecule = test_support::water();
        assert_eq!(
            molecule.add_bond_between(0, 1, BondType::Double),
            Err(ChemistryError::DuplicateBond {
                first: 0,
                second: 1
            })
        );
        let graph = MolecularGraph::from_molecule(&molecule).unwrap();
        assert_eq!(graph.edge_count(), 2, "the duplicate was never added");
    }

    #[test]
    fn atom_and_bond_ordering_and_metadata_are_preserved() {
        let molecule = test_support::carbon_dioxide();
        let graph = MolecularGraph::from_molecule(&molecule).unwrap();
        let symbols: Vec<&str> = graph.nodes().iter().map(Node::symbol).collect();
        assert_eq!(symbols, ["C", "O", "O"]);
        let types: Vec<BondType> = graph.edges().iter().map(Edge::bond_type).collect();
        assert_eq!(types, [BondType::Double, BondType::Double]);
        assert_eq!(graph.name(), Some("carbon dioxide"));
        assert_eq!(graph.identifier(), Some("pubchem:280"));
    }
}
