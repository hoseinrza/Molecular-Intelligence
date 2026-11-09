# Molecular Graph Architecture

> Scope: **Phase 2** of Molecular Intelligence (`crates/mi-graph`).
> *Implemented* sections describe code that exists today; *limitations* and
> *future compatibility* sections are plans, not capabilities.

## Definition

A **molecular graph** is a topological representation of a molecule: nodes
represent atoms, edges represent chemical bonds, and attached feature
records describe atomic and bond properties. It is **not** a 3D
representation — no coordinates, distances, or geometry exist in this layer.

`MolecularGraph` is constructed exclusively from a validated
`mi_chemistry::Molecule` via `MolecularGraph::from_molecule(&molecule)`.
`mi-chemistry` remains the single source of chemical truth; `mi-graph` never
reinterprets chemistry, it only re-shapes it.

## Node and Edge Semantics

### Nodes

- Node `i` corresponds to atom `i` of the source molecule (atom ordering is
  preserved). The mapping is stored explicitly
  (`Node::original_atom_index`) and re-checked by `MolecularGraph::validate`.
- A node bundles the `ChemicalElement` (element identity) with structured
  `NodeFeatures`: atomic number, formal charge, isotope mass number
  (optional), degree, and attached bond count.
- Degrees are extracted during conversion from the built adjacency.

### Edges

- Edge `j` corresponds to bond `j` (bond ordering preserved; mapping stored
  as `Edge::original_bond_index` and validated).
- Edges are **undirected** with canonical endpoints (`source < target`),
  mirroring `mi_chemistry::Bond`.
- Bond types are preserved exactly. The bond order is populated only when
  chemically defined: `1` single, `2` double, `3` triple. **Aromatic bonds
  carry `None`** — they are never rewritten as single/double, never given
  the interpolated 1.5, and no Kekulé structure is inferred. Unspecified
  bonds also carry `None` — no order is invented.

## Directed and Undirected Representations

Chemical bonds are undirected, but message-passing models consume directed
edges. The layer separates the two deliberately:

- **Undirected truth:** `MolecularGraph::edges() -> &[Edge]` — one edge per
  bond, index-stable, holding the only copy of bond identity and features.
- **Directed view:** `MolecularGraph::directed_edges()` — yields, for each
  edge `(u, v)`, the two directions `(u -> v)` and `(v -> u)` as lightweight
  `DirectedEdge { source, target, chemical_edge_index }` records. The view
  carries **indices only**: both directions of a bond share the same
  `chemical_edge_index` (which equals the original bond index), so bond
  identity and features are never duplicated. Order is deterministic:
  directions are grouped per edge in edge-index order, forward before
  reverse.
- `directed_edge_count() == 2 * edge_count()` is an invariant.

A future MPNN layer can therefore build `edge_index = [2, 2E]` and a doubled
feature matrix directly from `directed_edges()`, resolving shared features
through the chemical edge index.

## Adjacency List Design

`Adjacency` is a **derived index** over the edge list: `lists[u]` holds one
`Neighbor { node, edge }` entry per bond incident to node `u`.

- Neighbor lookup and incident-edge retrieval are O(degree); degree is a
  list length.
- It duplicates **indices only** — never bond data; `edges()` stays the
  single source of truth.
- Isolated atoms have empty lists; disconnected components need no special
  handling (lists are per-node).
- `Adjacency::validate` enforces *exact set equality* between the adjacency
  triples `(u, v, edge)` and the triples implied by the edge list — catching
  missing reverse entries, wrong edge references, duplicates, out-of-range
  endpoints, and self-loops in O(nodes + edges).
- No external graph library is used; the data model is a plain
  `Vec<Vec<Neighbor>>`.

## Node Feature Schema (version 1, width 6)

| # | Column | Kind | Encoding |
| - | ------ | ---- | -------- |
| 0 | `atomic_number` | categorical (cardinality 119) | Integer index `1..=118`; `0` reserved for padding/unknown in future embedding tables. |
| 1 | `formal_charge` | numerical | Signed formal charge, as-is. |
| 2 | `isotope_present` | binary | `1.0` when an isotope is specified, else `0.0`. |
| 3 | `isotope_mass_number` | numerical | Mass number when present, else `0.0` (meaningful only when column 2 is `1.0`). |
| 4 | `degree` | numerical | Number of adjacent nodes. |
| 5 | `bond_count` | numerical | Number of attached bonds (equals `degree` in the simple-graph model; separated for future multi-graphs). |

## Edge Feature Schema (version 1, width 7)

| # | Column | Kind | Encoding |
| - | ------ | ---- | -------- |
| 0–4 | `bond_type_is_{single,double,triple,aromatic,unspecified}` | binary | One-hot of the bond type in `BondType` declaration order. |
| 5 | `has_bond_order` | binary | `1.0` when a chemically defined order exists. |
| 6 | `bond_order` | numerical | `1.0`/`2.0`/`3.0` when defined, else `0.0` (meaningful only when column 5 is `1.0`). |

Schemas are explicit Rust structures (`FeatureSchema`, `FeatureColumn`,
`FeatureKind`) with a version. Categorical and numerical features stay
distinguishable at the schema level; optional values are encoded as
presence-flag + value pairs — no arbitrary floating-point sentinels.
`to_vector()` returns `Vec<f64>` today; a future tensor backend will consume
the same schema.

## Molecule-to-Graph Conversion

`MolecularGraph::from_molecule(&Molecule) -> Result<MolecularGraph, GraphError>`:

1. Validates the molecule (`molecule.validate()`); failures surface as
   `GraphError::InvalidMolecule` with the `ChemistryError` reachable through
   `std::error::Error::source`. Empty molecules are rejected (a graph needs
   at least one node).
2. Builds edges 1:1 from bonds (canonical endpoints, features extracted).
3. Builds the adjacency from the edges.
4. Builds nodes 1:1 from atoms with degrees read from the adjacency.
5. Carries over the molecule's name and identifier as metadata.
6. Runs the full graph validation as a final, defensive check.

Invalid atom/bond references and duplicate bonds are structurally
unreachable: `mi-chemistry` rejects them at construction. Graph validation
re-checks them anyway (defense in depth) and defines the contract for any
future construction path.

## Graph Validation (implemented)

`MolecularGraph::validate` checks: non-empty; node `i` ↔ atom `i` and edge
`j` ↔ bond `j` mappings; node element/feature consistency; endpoints valid,
ordered, self-loop-free; no duplicate unordered pairs; edge bond order
consistent with bond type; adjacency exact-match; per-node degree and bond
count matching the adjacency. Isolated atoms and disconnected components
validate cleanly. No valence rules are imposed.

## Known Limitations

- Purely topological: no coordinates, no distances, no periodic boundaries.
- Simple graphs only: no parallel edges (mirrors `mi-chemistry`), no bond
  removal or in-place editing; graphs are immutable after construction.
- No stereochemistry, atom mapping, ring perception, or aromaticity
  *perception* (aromatic bonds are taken as given from the molecule).
- No graph algorithms (connectivity/paths/cycles are out of scope).
- Feature schema v1 is intentionally minimal; it is not a descriptor set.
- `to_vector()` allocates `Vec<f64>` per call; no caching, no tensors.

## Future Compatibility with MPNN and 3D Models

- The directed view plus shared chemical-edge indices map directly onto the
  standard `[2, 2E]` message-passing edge index with tied edge features.
- `FeatureKind::Categorical` columns are embedding-table indices by
  construction; cardinalities are already declared (119 for elements, 5 for
  bond types via one-hot columns).
- The versioned schemas let `mi-tensor`/`mi-models` consume features without
  guessing layouts, and allow additive evolution (new columns ⇒ new schema
  version).
- 3D models (SchNet/PaiNN-style) will need per-atom positions and
  interatomic distances: positions are an additive node field in
  `mi-chemistry`/`mi-graph`; distances require geometry, which stays out of
  the topological layer.
