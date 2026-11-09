//! The molecular graph data model.

use crate::adjacency::{Adjacency, Neighbor};
use crate::edge::{DirectedEdge, Edge, defined_bond_order};
use crate::error::GraphError;
use crate::node::Node;
use std::collections::HashSet;

/// A graph representation of a molecule: nodes for atoms, edges for bonds.
///
/// # Indexing model
///
/// - Node `i` represents atom `i` and edge `j` represents bond `j` of the
///   source molecule (conversion preserves ordering; the mapping is stored
///   explicitly and re-checked by [`MolecularGraph::validate`]).
/// - Indices are stable for the lifetime of the graph: graphs are immutable
///   after construction.
/// - [`MolecularGraph::edges`] is the single source of chemical truth;
///   [`MolecularGraph::adjacency`] is a derived, validated index for
///   O(degree) neighbor lookups.
///
/// # Directed and undirected semantics
///
/// Chemical bonds are undirected and are stored as such (canonical
/// `source < target` endpoints). Message-passing models typically need
/// directed edges; [`MolecularGraph::directed_edges`] provides a
/// bidirectional **view** — two [`DirectedEdge`] entries per chemical edge,
/// sharing the same [`DirectedEdge::chemical_edge_index`] — without
/// duplicating bond identity or features.
///
/// Graphs are constructed exclusively through
/// [`MolecularGraph::from_molecule`]; isolated atoms and disconnected
/// components are fully supported.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MolecularGraph {
    name: Option<String>,
    identifier: Option<String>,
    nodes: Vec<Node>,
    edges: Vec<Edge>,
    adjacency: Adjacency,
}

impl MolecularGraph {
    /// Assembles a graph from validated parts (conversion-only).
    pub(crate) fn new(
        name: Option<String>,
        identifier: Option<String>,
        nodes: Vec<Node>,
        edges: Vec<Edge>,
        adjacency: Adjacency,
    ) -> Self {
        Self {
            name,
            identifier,
            nodes,
            edges,
            adjacency,
        }
    }

    /// Name carried over from the source molecule, if any.
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    /// Identifier carried over from the source molecule, if any.
    pub fn identifier(&self) -> Option<&str> {
        self.identifier.as_deref()
    }

    /// All nodes in index order; node `i` corresponds to atom `i`.
    pub fn nodes(&self) -> &[Node] {
        &self.nodes
    }

    /// All chemical edges in index order; edge `j` corresponds to bond `j`.
    pub fn edges(&self) -> &[Edge] {
        &self.edges
    }

    /// The derived adjacency index.
    pub fn adjacency(&self) -> &Adjacency {
        &self.adjacency
    }

    /// Number of nodes (one per atom of the source molecule).
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Number of undirected chemical edges (one per bond).
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Number of directed edges in the message-passing view (two per
    /// chemical edge).
    pub fn directed_edge_count(&self) -> usize {
        2 * self.edges.len()
    }

    /// Returns the node at `index`.
    ///
    /// # Errors
    ///
    /// Returns [`GraphError::InvalidNodeIndex`] if `index` is out of range.
    pub fn node(&self, index: usize) -> Result<&Node, GraphError> {
        self.nodes.get(index).ok_or(GraphError::InvalidNodeIndex {
            index,
            node_count: self.nodes.len(),
        })
    }

    /// Returns the chemical edge at `index`.
    ///
    /// # Errors
    ///
    /// Returns [`GraphError::InvalidEdgeIndex`] if `index` is out of range.
    pub fn edge(&self, index: usize) -> Result<&Edge, GraphError> {
        self.edges.get(index).ok_or(GraphError::InvalidEdgeIndex {
            index,
            edge_count: self.edges.len(),
        })
    }

    /// Adjacency list of `node`: adjacent nodes with their connecting edge
    /// indices. Isolated atoms have an empty list.
    ///
    /// # Errors
    ///
    /// Returns [`GraphError::InvalidNodeIndex`] if `node` is out of range.
    pub fn neighbors(&self, node: usize) -> Result<&[Neighbor], GraphError> {
        self.adjacency
            .neighbors(node)
            .ok_or(GraphError::InvalidNodeIndex {
                index: node,
                node_count: self.node_count(),
            })
    }

    /// Degree of `node`: the number of adjacent nodes (== incident bonds).
    ///
    /// # Errors
    ///
    /// Returns [`GraphError::InvalidNodeIndex`] if `node` is out of range.
    pub fn degree(&self, node: usize) -> Result<usize, GraphError> {
        self.adjacency
            .degree(node)
            .ok_or(GraphError::InvalidNodeIndex {
                index: node,
                node_count: self.node_count(),
            })
    }

    /// Whether `node` has no bonds (isolated atom).
    ///
    /// # Errors
    ///
    /// Returns [`GraphError::InvalidNodeIndex`] if `node` is out of range.
    pub fn is_isolated(&self, node: usize) -> Result<bool, GraphError> {
        Ok(self.degree(node)? == 0)
    }

    /// All chemical edges incident to `node`, in adjacency order.
    ///
    /// The edge indices come from the validated adjacency, so resolution is
    /// infallible for a constructed graph.
    ///
    /// # Errors
    ///
    /// Returns [`GraphError::InvalidNodeIndex`] if `node` is out of range.
    pub fn incident_edges(
        &self,
        node: usize,
    ) -> Result<impl Iterator<Item = &Edge> + '_, GraphError> {
        let neighbors = self.neighbors(node)?;
        Ok(neighbors
            .iter()
            .map(move |neighbor| &self.edges[neighbor.edge()]))
    }

    /// Bidirectional message-passing view over the chemical edges: for each
    /// undirected edge `(u, v)`, yields `(u -> v)` and `(v -> u)`, both
    /// referencing the same [`DirectedEdge::chemical_edge_index`].
    ///
    /// Order is deterministic: directions are grouped per edge in edge-index
    /// order, forward before reverse.
    pub fn directed_edges(&self) -> impl Iterator<Item = DirectedEdge> + '_ {
        self.edges.iter().enumerate().flat_map(|(index, edge)| {
            let (source, target) = edge.endpoints();
            [
                DirectedEdge::new(source, target, index),
                DirectedEdge::new(target, source, index),
            ]
        })
    }

    /// Validates all structural invariants of the graph:
    ///
    /// 1. The graph contains at least one node.
    /// 2. Node `i` maps to original atom `i`, and edge `j` maps to original
    ///    bond `j`.
    /// 3. Node element identity is consistent with its features.
    /// 4. Edge endpoints are valid node indices, ordered, and not self-loops.
    /// 5. No duplicate chemical edges (same unordered node pair).
    /// 6. Edge bond types and orders are consistent.
    /// 7. The adjacency exactly matches the chemical edges.
    /// 8. Every node's degree and attached-bond count match the adjacency.
    ///
    /// Graphs returned by [`MolecularGraph::from_molecule`] always satisfy
    /// these invariants; validation re-checks them defensively and defines
    /// the contract for any future construction path.
    ///
    /// # Errors
    ///
    /// Returns the most specific [`GraphError`] describing the violated
    /// invariant.
    pub fn validate(&self) -> Result<(), GraphError> {
        let node_count = self.nodes.len();
        if node_count == 0 {
            return Err(GraphError::InconsistentGraph {
                reason: "graph contains no nodes".to_string(),
            });
        }

        for (index, node) in self.nodes.iter().enumerate() {
            if node.original_atom_index() != index {
                return Err(GraphError::InconsistentGraph {
                    reason: format!(
                        "node {index} maps to original atom {}",
                        node.original_atom_index()
                    ),
                });
            }
            if node.element().atomic_number() != node.atomic_number() {
                return Err(GraphError::InconsistentGraph {
                    reason: format!(
                        "node {index} element and features disagree on the atomic number"
                    ),
                });
            }
        }

        let mut seen_pairs: HashSet<(usize, usize)> = HashSet::with_capacity(self.edges.len());
        for (index, edge) in self.edges.iter().enumerate() {
            if edge.original_bond_index() != index {
                return Err(GraphError::InconsistentGraph {
                    reason: format!(
                        "edge {index} maps to original bond {}",
                        edge.original_bond_index()
                    ),
                });
            }
            let (a, b) = edge.endpoints();
            if a >= node_count {
                return Err(GraphError::InvalidNodeIndex {
                    index: a,
                    node_count,
                });
            }
            if b >= node_count {
                return Err(GraphError::InvalidNodeIndex {
                    index: b,
                    node_count,
                });
            }
            if a == b {
                return Err(GraphError::SelfLoop { node_index: a });
            }
            if !seen_pairs.insert((a, b)) {
                return Err(GraphError::DuplicateEdge {
                    first: a,
                    second: b,
                });
            }
            if edge.bond_order() != defined_bond_order(edge.bond_type()) {
                return Err(GraphError::InconsistentGraph {
                    reason: format!("edge {index} bond order disagrees with its bond type"),
                });
            }
        }

        self.adjacency.validate(node_count, &self.edges)?;

        for (index, node) in self.nodes.iter().enumerate() {
            let adjacency_degree =
                self.adjacency
                    .degree(index)
                    .ok_or(GraphError::InconsistentGraph {
                        reason: format!("adjacency is missing node {index}"),
                    })?;
            if adjacency_degree != node.degree() || adjacency_degree != node.bond_count() {
                return Err(GraphError::InconsistentGraph {
                    reason: format!(
                        "node {index} degree mismatch: adjacency reports {adjacency_degree}, \
                         node reports degree {} and {} attached bond(s)",
                        node.degree(),
                        node.bond_count()
                    ),
                });
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support;
    use mi_chemistry::BondType;

    #[test]
    fn water_graph_has_correct_topology() {
        let graph = MolecularGraph::from_molecule(&test_support::water()).unwrap();
        assert_eq!(graph.node_count(), 3);
        assert_eq!(graph.edge_count(), 2);
        assert_eq!(graph.directed_edge_count(), 4);
        // Oxygen degree = 2, hydrogen degrees = 1.
        assert_eq!(graph.degree(0).unwrap(), 2);
        assert_eq!(graph.degree(1).unwrap(), 1);
        assert_eq!(graph.degree(2).unwrap(), 1);
        assert!(!graph.is_isolated(0).unwrap());
        // Atom ordering is preserved: atom 0 is the oxygen.
        assert_eq!(graph.node(0).unwrap().symbol(), "O");
        assert_eq!(graph.node(1).unwrap().symbol(), "H");
        // Source molecule metadata is carried over.
        assert_eq!(graph.name(), Some("water"));
        assert_eq!(graph.identifier(), Some("pubchem:962"));
        graph.validate().unwrap();
    }

    #[test]
    fn methane_graph_has_correct_topology() {
        let graph = MolecularGraph::from_molecule(&test_support::methane()).unwrap();
        assert_eq!(graph.node_count(), 5);
        assert_eq!(graph.edge_count(), 4);
        // Carbon degree = 4.
        assert_eq!(graph.degree(0).unwrap(), 4);
        for hydrogen in 1..5 {
            assert_eq!(graph.degree(hydrogen).unwrap(), 1);
        }
        assert_eq!(graph.node(0).unwrap().symbol(), "C");
        graph.validate().unwrap();
    }

    #[test]
    fn carbon_dioxide_preserves_double_bonds() {
        let graph = MolecularGraph::from_molecule(&test_support::carbon_dioxide()).unwrap();
        assert_eq!(graph.node_count(), 3);
        assert_eq!(graph.edge_count(), 2);
        for edge in graph.edges() {
            assert_eq!(edge.bond_type(), BondType::Double);
            assert_eq!(edge.bond_order(), Some(2));
        }
        graph.validate().unwrap();
    }

    #[test]
    fn benzene_preserves_aromatic_bonds() {
        let graph = MolecularGraph::from_molecule(&test_support::benzene()).unwrap();
        assert_eq!(graph.node_count(), 6);
        assert_eq!(graph.edge_count(), 6);
        assert_eq!(graph.directed_edge_count(), 12);
        for edge in graph.edges() {
            assert_eq!(edge.bond_type(), BondType::Aromatic);
            assert_eq!(
                edge.bond_order(),
                None,
                "no order is invented for aromatic bonds"
            );
        }
        // The ring is a closed cycle: every node has degree 2.
        for node in 0..6 {
            assert_eq!(graph.degree(node).unwrap(), 2);
        }
        graph.validate().unwrap();
    }

    #[test]
    fn single_isolated_atom_is_supported() {
        let graph = MolecularGraph::from_molecule(&test_support::hydrogen_radical()).unwrap();
        assert_eq!(graph.node_count(), 1);
        assert_eq!(graph.edge_count(), 0);
        assert_eq!(graph.directed_edge_count(), 0);
        assert!(graph.is_isolated(0).unwrap());
        assert!(graph.neighbors(0).unwrap().is_empty());
        assert_eq!(graph.node(0).unwrap().degree(), 0);
        assert_eq!(graph.node(0).unwrap().bond_count(), 0);
        graph.validate().unwrap();
    }

    #[test]
    fn disconnected_components_are_supported() {
        let graph = MolecularGraph::from_molecule(&test_support::water_and_hydrogen()).unwrap();
        assert_eq!(graph.node_count(), 5);
        assert_eq!(graph.edge_count(), 3);
        // Water component: degrees 2, 1, 1. Hydrogen component: 1, 1.
        let degrees: Vec<usize> = (0..5).map(|node| graph.degree(node).unwrap()).collect();
        assert_eq!(degrees, [2, 1, 1, 1, 1]);
        // No edge bridges the components: each stays within atoms 0..=2 or 3..=4.
        for edge in graph.edges() {
            let (a, b) = edge.endpoints();
            assert!((a <= 2 && b <= 2) || (a >= 3 && b >= 3));
        }
        graph.validate().unwrap();
    }

    #[test]
    fn directed_view_pairs_share_chemical_identity() {
        let graph = MolecularGraph::from_molecule(&test_support::water()).unwrap();
        let directed: Vec<_> = graph.directed_edges().collect();
        assert_eq!(directed.len(), 4);
        // Deterministic order: per edge, forward direction then reverse.
        assert_eq!(directed[0], DirectedEdge::new(0, 1, 0));
        assert_eq!(directed[1], DirectedEdge::new(1, 0, 0));
        assert_eq!(directed[2], DirectedEdge::new(0, 2, 1));
        assert_eq!(directed[3], DirectedEdge::new(2, 0, 1));
        // Every direction resolves to the same undirected chemical edge.
        for directed_edge in &directed {
            let edge = graph.edge(directed_edge.chemical_edge_index()).unwrap();
            let (a, b) = edge.endpoints();
            assert_eq!(
                (a, b),
                (
                    directed_edge.source().min(directed_edge.target()),
                    directed_edge.source().max(directed_edge.target())
                )
            );
        }
    }

    #[test]
    fn neighbor_and_incident_edge_access_with_error_cases() {
        let graph = MolecularGraph::from_molecule(&test_support::water()).unwrap();
        let neighbors = graph.neighbors(0).unwrap();
        assert_eq!(
            neighbors.iter().map(Neighbor::node).collect::<Vec<_>>(),
            [1, 2]
        );
        assert_eq!(
            neighbors.iter().map(Neighbor::edge).collect::<Vec<_>>(),
            [0, 1]
        );
        let incident: Vec<usize> = graph
            .incident_edges(0)
            .unwrap()
            .map(Edge::original_bond_index)
            .collect();
        assert_eq!(incident, [0, 1]);

        assert_eq!(
            graph.node(3),
            Err(GraphError::InvalidNodeIndex {
                index: 3,
                node_count: 3
            })
        );
        assert_eq!(
            graph.edge(2),
            Err(GraphError::InvalidEdgeIndex {
                index: 2,
                edge_count: 2
            })
        );
        assert_eq!(
            graph.neighbors(3),
            Err(GraphError::InvalidNodeIndex {
                index: 3,
                node_count: 3
            })
        );
        assert_eq!(
            graph.degree(9),
            Err(GraphError::InvalidNodeIndex {
                index: 9,
                node_count: 3
            })
        );
    }

    #[test]
    fn node_and_edge_features_are_extracted() {
        let graph = MolecularGraph::from_molecule(&test_support::carbon_dioxide()).unwrap();
        let carbon = graph.node(0).unwrap();
        assert_eq!(carbon.atomic_number(), 6);
        assert_eq!(carbon.formal_charge(), 0);
        assert_eq!(carbon.degree(), 2);
        assert_eq!(
            carbon.features().to_vector(),
            [6.0, 0.0, 0.0, 0.0, 2.0, 2.0]
        );
        let oxygen = graph.node(1).unwrap();
        assert_eq!(oxygen.atomic_number(), 8);
        assert_eq!(
            oxygen.features().to_vector(),
            [8.0, 0.0, 0.0, 0.0, 1.0, 1.0]
        );
        let double_bond = graph.edge(0).unwrap();
        assert_eq!(double_bond.bond_type(), BondType::Double);
        assert_eq!(
            double_bond.features().to_vector(),
            [0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 2.0]
        );
    }

    #[test]
    fn original_index_mapping_is_preserved_and_validated() {
        let graph = MolecularGraph::from_molecule(&test_support::benzene()).unwrap();
        for (index, node) in graph.nodes().iter().enumerate() {
            assert_eq!(node.original_atom_index(), index);
        }
        for (index, edge) in graph.edges().iter().enumerate() {
            assert_eq!(edge.original_bond_index(), index);
        }
        assert!(graph.validate().is_ok());
    }
}
