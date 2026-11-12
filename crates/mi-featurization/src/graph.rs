//! Graph featurization: molecular graphs as ML tensors.
//!
//! This module is pure assembly: `mi-graph` already owns the versioned
//! feature schemas (`NodeFeatures`, `EdgeFeatures`) and the validated
//! graph traversal (`MolecularGraph::directed_edges`); this layer collects
//! their numerical vectors into `mi-tensor` tensors, one molecule at a
//! time. No feature is invented here — the tensors are exactly what the
//! schemas define, with schema versions carried alongside as provenance.
//!
//! # Conventions
//!
//! - **Directed edges.** Message-passing consumes each bond in both
//!   directions, so `edge_index`/`edge_features` are the *directed* view
//!   (`2E` rows), ordered exactly as
//!   [`MolecularGraph::directed_edges`](mi_graph::MolecularGraph::directed_edges)
//!   yields them (per chemical edge: forward, then reverse).
//! - **Edge-less graphs.** A monoatomic molecule has zero edges, and
//!   [`TensorShape`](mi_tensor::TensorShape) forbids zero-sized
//!   dimensions — so `edge_features` is [`Option`]: `Some` if and only if
//!   `edge_index` is non-empty. Absence is explicit, never a placeholder
//!   row.
//! - **Element narrowing.** Schema vectors are `f64`; tensor ingress is
//!   `f32`. Every schema v1 column (small integers, flags, small counts)
//!   is exactly representable in `f32`, so the narrowing is lossless for
//!   the current schemas — a guarantee this crate's tests pin down.
//! - **Inference only.** Featurized tensors are created on an inference
//!   device and carry no autodiff graph; gradients belong to Phase 6's
//!   models, not to data preparation.

use crate::error::FeaturizationResult;
use mi_chemistry::Molecule;
use mi_graph::{EdgeFeatures, MolecularGraph, NodeFeatures};
use mi_tensor::{Device, Tensor, TensorShape};

/// The featurized numerical representation of one molecular graph.
///
/// Row order is meaningful: node `i` corresponds to graph node `i`
/// (source atom `i`), directed edge `k` corresponds to `edge_index[k]`
/// with features in `edge_features` row `k`.
#[derive(Clone)]
pub struct GraphTensors {
    nodes: Tensor<2>,
    edge_index: Vec<[usize; 2]>,
    edge_features: Option<Tensor<2>>,
    node_schema_version: u32,
    edge_schema_version: u32,
}

impl std::fmt::Debug for GraphTensors {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GraphTensors")
            .field("nodes_shape", &self.nodes.shape().dims().to_vec())
            .field("directed_edge_count", &self.edge_index.len())
            .field(
                "edge_features_shape",
                &self
                    .edge_features
                    .as_ref()
                    .map(|tensor| tensor.shape().dims().to_vec()),
            )
            .field("node_schema_version", &self.node_schema_version)
            .field("edge_schema_version", &self.edge_schema_version)
            .finish()
    }
}

impl GraphTensors {
    /// The node feature matrix, shape `[N, node_width]`.
    pub fn nodes(&self) -> &Tensor<2> {
        &self.nodes
    }

    /// The directed edge endpoints, length `2E`, in
    /// [`directed_edges`](mi_graph::MolecularGraph::directed_edges) order.
    pub fn edge_index(&self) -> &[[usize; 2]] {
        &self.edge_index
    }

    /// The directed edge feature matrix, shape `[2E, edge_width]`, present
    /// if and only if the graph has at least one edge.
    pub fn edge_features(&self) -> Option<&Tensor<2>> {
        self.edge_features.as_ref()
    }

    /// Version of the node feature schema the tensor was built from.
    pub fn node_schema_version(&self) -> u32 {
        self.node_schema_version
    }

    /// Version of the edge feature schema the tensor was built from.
    pub fn edge_schema_version(&self) -> u32 {
        self.edge_schema_version
    }

    /// The number of nodes the tensors represent.
    pub fn node_count(&self) -> usize {
        self.nodes.shape().dims()[0]
    }

    /// The number of directed edges (`2E` for a graph with `E` bonds);
    /// zero for a monoatomic molecule.
    pub fn directed_edge_count(&self) -> usize {
        self.edge_index.len()
    }
}

/// Featurizes a molecular graph into tensors on `device`.
///
/// # Errors
///
/// Returns [`FeaturizationError::Tensor`] if tensor construction fails
/// (which the schema widths and graph invariants make unreachable except
/// through backend failure) and [`FeaturizationError::Graph`] if edge
/// lookup fails.
pub fn featurize_graph(
    graph: &MolecularGraph,
    device: &Device,
) -> FeaturizationResult<GraphTensors> {
    let node_width = NodeFeatures::SCHEMA.width();
    let mut node_data = Vec::with_capacity(graph.node_count() * node_width);
    for node in graph.nodes() {
        node_data.extend(
            node.features()
                .to_vector()
                .into_iter()
                .map(|value| value as f32),
        );
    }
    let nodes = Tensor::from_data(
        node_data,
        TensorShape::new(vec![graph.node_count(), node_width])?,
        device,
    )?;

    let edge_width = EdgeFeatures::SCHEMA.width();
    let mut edge_index = Vec::with_capacity(graph.directed_edge_count());
    let mut edge_data = Vec::with_capacity(graph.directed_edge_count() * edge_width);
    for directed in graph.directed_edges() {
        edge_index.push([directed.source(), directed.target()]);
        let features = graph.edge(directed.chemical_edge_index())?.features();
        edge_data.extend(features.to_vector().into_iter().map(|value| value as f32));
    }
    let edge_features = if edge_index.is_empty() {
        None
    } else {
        let shape = TensorShape::new(vec![edge_index.len(), edge_width])?;
        Some(Tensor::from_data(edge_data, shape, device)?)
    };

    Ok(GraphTensors {
        nodes,
        edge_index,
        edge_features,
        node_schema_version: NodeFeatures::SCHEMA.version(),
        edge_schema_version: EdgeFeatures::SCHEMA.version(),
    })
}

/// Featurizes a molecule: builds its molecular graph, then converts it to
/// tensors.
///
/// # Errors
///
/// Returns [`FeaturizationError::Graph`] if the molecule fails `mi-graph`'s
/// validation (empty molecules are rejected there), plus every error of
/// [`featurize_graph`].
pub fn featurize_molecule(
    molecule: &Molecule,
    device: &Device,
) -> FeaturizationResult<GraphTensors> {
    let graph = MolecularGraph::from_molecule(molecule)?;
    featurize_graph(&graph, device)
}

#[cfg(test)]
mod tests {
    use super::*;
    use mi_chemistry::{Atom, BondType, ChemicalElement, Position};
    use mi_tensor::Cpu;

    fn device() -> Device {
        Cpu::inference()
    }

    /// Water: O–H single bonds; atom 0 is the oxygen.
    fn water() -> Molecule {
        let mut molecule = Molecule::new().with_name("water");
        let element = |symbol: &str| ChemicalElement::from_symbol(symbol).unwrap();
        molecule.add_atom(Atom::new(element("O")));
        molecule.add_atom(Atom::new(element("H")));
        molecule.add_atom(Atom::new(element("H")));
        molecule.add_bond_between(0, 1, BondType::Single).unwrap();
        molecule.add_bond_between(0, 2, BondType::Single).unwrap();
        molecule
    }

    #[test]
    fn water_yields_expected_node_tensor() {
        let tensors = featurize_molecule(&water(), &device()).unwrap();
        assert_eq!(tensors.nodes().shape().dims(), &[3, 6]);
        assert_eq!(tensors.node_count(), 3);
        // Oxygen row: [atomic_number, formal_charge, isotope_present,
        // isotope_mass_number, degree, bond_count].
        let rows = tensors.nodes().to_vec().unwrap();
        assert_eq!(&rows[0..6], &[8.0, 0.0, 0.0, 0.0, 2.0, 2.0]);
        assert_eq!(&rows[6..12], &[1.0, 0.0, 0.0, 0.0, 1.0, 1.0]);
        assert_eq!(&rows[12..18], &[1.0, 0.0, 0.0, 0.0, 1.0, 1.0]);
    }

    #[test]
    fn water_yields_expected_directed_edges() {
        let tensors = featurize_molecule(&water(), &device()).unwrap();
        // Per chemical edge: forward, then reverse.
        assert_eq!(tensors.edge_index(), &[[0, 1], [1, 0], [0, 2], [2, 0]]);
        assert_eq!(tensors.directed_edge_count(), 4);
        assert_eq!(tensors.edge_features().unwrap().shape().dims(), &[4, 7]);
        // Row k corresponds to bond k / 2: single-bond one-hot with order 1.
        let rows = tensors.edge_features().unwrap().to_vec().unwrap();
        for row in 0..4 {
            let start = row * 7;
            assert_eq!(
                &rows[start..start + 7],
                &[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0]
            );
        }
    }

    #[test]
    fn bond_type_order_survives_featurization() {
        // Ethene skeleton: one double bond.
        let mut molecule = Molecule::new();
        let carbon = ChemicalElement::from_symbol("C").unwrap();
        molecule.add_atom(Atom::new(carbon));
        molecule.add_atom(Atom::new(carbon));
        molecule.add_bond_between(0, 1, BondType::Double).unwrap();
        let tensors = featurize_molecule(&molecule, &device()).unwrap();
        let rows = tensors.edge_features().unwrap().to_vec().unwrap();
        // Double-bond one-hot with bond order 2, forward and reverse.
        for row in 0..2 {
            let start = row * 7;
            assert_eq!(
                &rows[start..start + 7],
                &[0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 2.0]
            );
        }
    }

    #[test]
    fn monoatomic_molecule_has_no_edges() {
        let mut molecule = Molecule::new().with_name("argon");
        molecule.add_atom(Atom::new(ChemicalElement::from_symbol("Ar").unwrap()));
        let tensors = featurize_molecule(&molecule, &device()).unwrap();
        assert_eq!(tensors.nodes().shape().dims(), &[1, 6]);
        assert!(tensors.edge_index().is_empty());
        assert!(tensors.edge_features().is_none());
        assert_eq!(tensors.directed_edge_count(), 0);
    }

    #[test]
    fn featurize_graph_matches_featurize_molecule() {
        let molecule = water();
        let graph = MolecularGraph::from_molecule(&molecule).unwrap();
        let from_graph = featurize_graph(&graph, &device()).unwrap();
        let from_molecule = featurize_molecule(&molecule, &device()).unwrap();
        assert_eq!(
            from_graph.nodes().to_vec().unwrap(),
            from_molecule.nodes().to_vec().unwrap()
        );
        assert_eq!(from_graph.edge_index(), from_molecule.edge_index());
        assert_eq!(
            from_graph.edge_features().map(Tensor::to_vec),
            from_molecule.edge_features().map(Tensor::to_vec)
        );
    }

    #[test]
    fn schema_versions_are_carried_as_provenance() {
        let tensors = featurize_molecule(&water(), &device()).unwrap();
        assert_eq!(
            tensors.node_schema_version(),
            NodeFeatures::SCHEMA.version()
        );
        assert_eq!(
            tensors.edge_schema_version(),
            EdgeFeatures::SCHEMA.version()
        );
        assert_eq!(NodeFeatures::SCHEMA.width(), 6);
        assert_eq!(EdgeFeatures::SCHEMA.width(), 7);
    }

    #[test]
    fn isotope_features_flow_through() {
        // Deuterated hydrogen: isotope columns become 1 and 2.
        let mut molecule = Molecule::new();
        let hydrogen = ChemicalElement::from_symbol("H").unwrap();
        molecule.add_atom(Atom::new(hydrogen).with_isotope(2).unwrap());
        let tensors = featurize_molecule(&molecule, &device()).unwrap();
        let row = tensors.nodes().to_vec().unwrap();
        assert_eq!(row, vec![1.0, 0.0, 1.0, 2.0, 0.0, 0.0]);
    }

    #[test]
    fn featurization_is_deterministic() {
        let molecule = water();
        let first = featurize_molecule(&molecule, &device()).unwrap();
        let second = featurize_molecule(&molecule, &device()).unwrap();
        assert_eq!(
            first.nodes().to_vec().unwrap(),
            second.nodes().to_vec().unwrap()
        );
        assert_eq!(first.edge_index(), second.edge_index());
    }

    #[test]
    fn positions_do_not_affect_graph_features() {
        // Node features are topological; adding positions changes nothing.
        let plain = water();
        let mut placed = Molecule::new().with_name("water");
        let element = |symbol: &str| ChemicalElement::from_symbol(symbol).unwrap();
        placed.add_atom(
            Atom::new(element("O")).with_position(Position::new(0.0, 0.0, 0.1173).unwrap()),
        );
        placed.add_atom(
            Atom::new(element("H")).with_position(Position::new(0.0, 0.7572, -0.4692).unwrap()),
        );
        placed.add_atom(
            Atom::new(element("H")).with_position(Position::new(0.0, -0.7572, -0.4692).unwrap()),
        );
        placed.add_bond_between(0, 1, BondType::Single).unwrap();
        placed.add_bond_between(0, 2, BondType::Single).unwrap();
        assert_eq!(
            featurize_molecule(&plain, &device())
                .unwrap()
                .nodes()
                .to_vec()
                .unwrap(),
            featurize_molecule(&placed, &device())
                .unwrap()
                .nodes()
                .to_vec()
                .unwrap()
        );
    }
}
