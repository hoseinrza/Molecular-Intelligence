//! Adjacency-list representation over the graph's chemical edges.

use crate::edge::Edge;
use crate::error::GraphError;
use std::collections::HashSet;

/// One adjacency entry: an adjacent node and the connecting edge index.
///
/// Entries reference the chemical edges — they never duplicate bond data.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Neighbor {
    node: usize,
    edge: usize,
}

impl Neighbor {
    pub(crate) const fn new(node: usize, edge: usize) -> Self {
        Self { node, edge }
    }

    /// Index of the adjacent node.
    pub const fn node(&self) -> usize {
        self.node
    }

    /// Index of the chemical edge connecting to [`Neighbor::node`].
    pub const fn edge(&self) -> usize {
        self.edge
    }
}

/// Adjacency-list index over the graph's chemical edges.
///
/// `lists[u]` holds one entry per bond incident to node `u`, so neighbor
/// lookup and incident-edge retrieval are O(degree). The adjacency is a
/// **derived view**: it duplicates only indices, never bond data — the
/// [`Edge`] list remains the single source of chemical truth, and
/// [`Adjacency::validate`] enforces exact consistency between the two.
///
/// Isolated atoms (no bonds) have empty lists and are fully supported, as
/// are disconnected molecular components.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Adjacency {
    lists: Vec<Vec<Neighbor>>,
}

impl Adjacency {
    /// Builds the adjacency lists for `node_count` nodes from edges whose
    /// endpoints are canonical (`source < target`) and distinct.
    pub(crate) fn from_edges(node_count: usize, edges: &[Edge]) -> Self {
        let mut lists: Vec<Vec<Neighbor>> = vec![Vec::new(); node_count];
        for (edge_index, edge) in edges.iter().enumerate() {
            let (a, b) = edge.endpoints();
            lists[a].push(Neighbor::new(b, edge_index));
            lists[b].push(Neighbor::new(a, edge_index));
        }
        Self { lists }
    }

    /// Number of nodes the adjacency covers.
    pub fn node_count(&self) -> usize {
        self.lists.len()
    }

    /// Adjacency list of `node`, or [`None`] when out of range.
    pub fn neighbors(&self, node: usize) -> Option<&[Neighbor]> {
        self.lists.get(node).map(Vec::as_slice)
    }

    /// Degree of `node` (number of incident bonds), or [`None`] when out of
    /// range.
    pub fn degree(&self, node: usize) -> Option<usize> {
        self.lists.get(node).map(Vec::len)
    }

    /// Validates exact consistency between the adjacency lists and the edge
    /// list:
    ///
    /// - the lists cover exactly `node_count` nodes,
    /// - every referenced edge exists and actually connects the two nodes,
    /// - every entry has its reverse entry (undirected symmetry),
    /// - no duplicate entries and no self-loops.
    ///
    /// The check is O(nodes + edges) using sets of `(node, node, edge)`
    /// triples.
    pub(crate) fn validate(&self, node_count: usize, edges: &[Edge]) -> Result<(), GraphError> {
        if self.lists.len() != node_count {
            return Err(GraphError::InconsistentGraph {
                reason: format!(
                    "adjacency covers {} node(s) but the graph has {node_count}",
                    self.lists.len()
                ),
            });
        }

        let mut expected: HashSet<(usize, usize, usize)> = HashSet::with_capacity(2 * edges.len());
        for (edge_index, edge) in edges.iter().enumerate() {
            let (a, b) = edge.endpoints();
            if a >= node_count || b >= node_count {
                let index = if a >= node_count { a } else { b };
                return Err(GraphError::InvalidNodeIndex { index, node_count });
            }
            if a == b {
                return Err(GraphError::SelfLoop { node_index: a });
            }
            expected.insert((a, b, edge_index));
            expected.insert((b, a, edge_index));
        }

        let mut actual: HashSet<(usize, usize, usize)> = HashSet::with_capacity(2 * edges.len());
        for (node, list) in self.lists.iter().enumerate() {
            for neighbor in list {
                if neighbor.node >= node_count {
                    return Err(GraphError::InvalidNodeIndex {
                        index: neighbor.node,
                        node_count,
                    });
                }
                if !actual.insert((node, neighbor.node, neighbor.edge)) {
                    return Err(GraphError::InconsistentGraph {
                        reason: format!(
                            "duplicate adjacency entry ({node}, {}, {})",
                            neighbor.node, neighbor.edge
                        ),
                    });
                }
            }
        }

        if actual != expected {
            return Err(GraphError::InconsistentGraph {
                reason: "adjacency lists do not exactly match the chemical edges".to_string(),
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support;
    use mi_chemistry::{Bond, BondType};

    /// Edges of the path graph `0 - 1 - 2`.
    fn path_edges() -> Vec<Edge> {
        vec![
            Edge::new(0, &Bond::new(0, 1, BondType::Single).unwrap()),
            Edge::new(1, &Bond::new(1, 2, BondType::Double).unwrap()),
        ]
    }

    #[test]
    fn neighbor_lookup_and_degrees_are_correct() {
        let adjacency = Adjacency::from_edges(3, &path_edges());
        assert_eq!(adjacency.node_count(), 3);
        assert_eq!(adjacency.degree(0), Some(1));
        assert_eq!(adjacency.degree(1), Some(2));
        assert_eq!(adjacency.degree(2), Some(1));

        let middle = adjacency.neighbors(1).unwrap();
        assert_eq!(middle, &[Neighbor::new(0, 0), Neighbor::new(2, 1),]);
        // Out-of-range lookups are explicit, never panics.
        assert_eq!(adjacency.degree(3), None);
        assert!(adjacency.neighbors(3).is_none());
    }

    #[test]
    fn isolated_nodes_have_empty_lists() {
        let adjacency = Adjacency::from_edges(2, &[]);
        assert_eq!(adjacency.degree(0), Some(0));
        assert!(adjacency.neighbors(0).unwrap().is_empty());
        assert_eq!(adjacency.degree(1), Some(0));
    }

    #[test]
    fn accepts_consistent_lists() {
        let edges = path_edges();
        let adjacency = Adjacency::from_edges(3, &edges);
        assert_eq!(adjacency.validate(3, &edges), Ok(()));
    }

    #[test]
    fn detects_a_missing_reverse_entry() {
        let edges = path_edges();
        // Simulated corruption: node 1's list lost its entry for edge 0.
        let adjacency = Adjacency {
            lists: vec![
                vec![Neighbor::new(1, 0)],
                vec![Neighbor::new(2, 1)],
                vec![Neighbor::new(1, 1)],
            ],
        };
        assert!(matches!(
            adjacency.validate(3, &edges),
            Err(GraphError::InconsistentGraph { .. })
        ));
    }

    #[test]
    fn detects_a_wrong_edge_reference() {
        let edges = path_edges();
        // Simulated corruption: entry claims edge 1 connects 0 and 1, but
        // edge 1 connects 1 and 2.
        let adjacency = Adjacency {
            lists: vec![
                vec![Neighbor::new(1, 1)],
                vec![Neighbor::new(0, 1), Neighbor::new(2, 1)],
                vec![Neighbor::new(1, 1)],
            ],
        };
        assert!(matches!(
            adjacency.validate(3, &edges),
            Err(GraphError::InconsistentGraph { .. })
        ));
    }

    #[test]
    fn detects_duplicate_entries() {
        let edges = path_edges();
        let adjacency = Adjacency {
            lists: vec![
                vec![Neighbor::new(1, 0), Neighbor::new(1, 0)],
                vec![Neighbor::new(0, 0), Neighbor::new(2, 1)],
                vec![Neighbor::new(1, 1)],
            ],
        };
        assert!(matches!(
            adjacency.validate(3, &edges),
            Err(GraphError::InconsistentGraph { .. })
        ));
    }

    #[test]
    fn detects_endpoints_outside_the_node_set() {
        // Edge endpoints come from bonds, which reject self-bonds, so the
        // SelfLoop branch is unreachable through the public API (pinned in
        // `edge.rs` tests). This test exercises the adjacent defensive
        // check: an endpoint outside the validated node set.
        let edges = path_edges();
        let adjacency = Adjacency::from_edges(2, &[]);
        assert_eq!(
            adjacency.validate(2, &edges),
            Err(GraphError::InvalidNodeIndex {
                index: 2,
                node_count: 2
            })
        );
    }

    #[test]
    fn self_loops_cannot_reach_adjacency_validation() {
        // The defensive self-loop branch in `validate` is unreachable
        // through the public API: `Bond::new` rejects self-bonds at
        // construction (tested in `edge.rs`). This test pins that guarantee
        // at the adjacency boundary.
        assert!(matches!(
            Bond::new(1, 1, BondType::Single),
            Err(mi_chemistry::ChemistryError::SelfBond { atom_index: 1 })
        ));
    }

    #[test]
    fn aromatic_edges_are_indexed_like_any_other() {
        let edges = vec![Edge::new(0, &test_support::aromatic_bond())];
        let adjacency = Adjacency::from_edges(2, &edges);
        assert_eq!(adjacency.degree(0), Some(1));
        assert_eq!(adjacency.degree(1), Some(1));
        assert_eq!(adjacency.validate(2, &edges), Ok(()));
    }
}
