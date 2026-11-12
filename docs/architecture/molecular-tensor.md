# Molecular Tensor Architecture

> Scope: **Phase 5 — Molecular Tensor** of Molecular Intelligence
> (`crates/mi-featurization`). *Implemented* sections describe code that
> exists and is tested today; *limitations* and *future work* sections
> are plans, not capabilities.

## Overview

The featurization layer is the sanctioned junction between the chemical
domain and the numerical substrate: it converts molecular structures into
`mi-tensor` tensors so that Phase 6's models can consume them. It depends
on `mi-chemistry`, `mi-graph`, `mi-data`, and `mi-tensor` — and on no ML
library and no learning algorithm. Featurization is data preparation.

```text
Molecule ──► MolecularGraph (mi-graph) ──► GraphTensors (mi-tensor)
                │  versioned schemas v1        │  nodes [N,6]
                │                              │  edge_index [2E]
                └── Positions (Å) ──► geometry │  edge_features [2E,7] (Option)
MolecularEntry ──► properties ──► Targets (values[3] + mask[3])
```

This fulfills the architecture rule from the overview — chemistry, graph,
and dataset types never depend on tensor types, and `mi-tensor` never
imports chemistry — with the junction formalized as its own crate so
Phase 6's `mi-models` can depend on a stable representation API. The
overview's dependency map is updated accordingly.

## What the layer produces

### Graph tensors (`featurize_graph`, `featurize_molecule`)

Pure assembly of what `mi-graph` already defines — no feature is invented
here:

| Tensor | Shape | Content |
| ------ | ----- | ------- |
| `nodes` | `[N, 6]` | `NodeFeatures::to_vector()` per atom — schema v1: atomic number (categorical, cardinality 119, `0` reserved for padding), formal charge, isotope presence, isotope mass number, degree, bond count |
| `edge_index` | `2E` pairs | directed endpoints in `MolecularGraph::directed_edges()` order (per chemical edge: forward, then reverse) — host-side `usize` pairs, not a tensor (see limitations) |
| `edge_features` | `[2E, 7]` | `EdgeFeatures::to_vector()` aligned with `edge_index` — schema v1: one-hot bond type, `has_bond_order`, bond order |

Schema widths and versions are read from `NodeFeatures::SCHEMA` /
`EdgeFeatures::SCHEMA` at featurization time and carried on
`GraphTensors` as provenance, so an ML artifact can record exactly which
feature layout produced it.

**Edge-less graphs** (monoatomic molecules): `edge_features` is
[`Option`](https://doc.rust-lang.org/std/option/) — `Some` if and only if
`edge_index` is non-empty. `TensorShape` forbids zero-sized dimensions,
and a placeholder row would be an invented edge; explicit absence matches
the project's `Option`-carrying convention.

**Element narrowing:** schema vectors are `f64`, tensor ingress is `f32`.
Every schema v1 column (small integers, flags, one-hots) is exactly
representable in `f32`, so the conversion is lossless today — pinned by
tests that compare tensor readbacks against the schema vectors.

### Geometry (`distance_matrix`, `bond_distances`)

Three-dimensional structure is a separate, physically grounded quantity —
the versioned schemas stay topological:

- `distance_matrix(molecule, device)` — symmetric `[N, N]` Euclidean
  distances in **Ångström** (the unit fixed by `Position`), zero
  diagonal. Requires **every** atom to carry a position
  (`FeaturizationError::MissingPosition` names the first offender): a
  partially placed molecule must not silently mix invented zeros into
  physics.
- `bond_distances(molecule)` — one distance per chemical bond in
  `molecule.bonds()` order, for models that want a geometric column on
  bond features.

Computation is `f64` on the host entering `f32` tensors — far below the
accuracy of any source geometry. O(N²) is negligible at molecular sizes.

### Targets (`Targets`)

Fixed layout `[homo, lumo, gap]` in **electronvolts** (the `mi-data`
storage unit; Hartree sources were converted at the I/O boundary and are
never re-converted here):

- `Targets::extract(entry)` reads the dataset entry's properties; the gap
  is the **effective** gap per `mi-data` semantics (stored when present,
  else derived as `LUMO − HOMO`, else absent).
- `to_tensors(device)` returns `(values[3], mask[3])`: absent slots are
  `0.0` with mask `0.0`. Model code multiplies element-wise, so absent
  slots contribute nothing to a loss without special-casing — the mask
  convention mirrors `isotope_present` in the node schema and keeps
  absence explicit, never a sentinel.

## Conventions summary

| Concern | Convention |
| ------- | ---------- |
| Categorical columns | Left as raw indices (schema v1); one-hot/embedding is a **model-side** choice (Phase 6) |
| Edge view | Directed (`2E` rows), `directed_edges()` order |
| Edge-less graphs | `edge_features: None`, empty `edge_index` |
| Missing positions | Typed error, never zeros |
| Missing properties | Mask `0.0`, never sentinels |
| Devices | Inference only; featurized tensors carry no autodiff graph |
| Precision | `f64` feature computation → `f32` tensor ingress (lossless for schema v1) |

## Verification

- **Unit tests** per module: tensor contents against hand-computed schema
  vectors (water, ethene, deuterium), directed-edge ordering, monoatomic
  `None` edges, schema-version provenance, determinism, distance exactness
  (single-axis and 3D Euclidean), missing-position errors, target
  extraction with stored/derived/absent gaps, mask correctness.
- **Integration tests** (`tests/end_to_end.rs`): molecule → tensors →
  readback equals schema vectors; geometry against hand-computed
  distances; targets paired with `MolecularEntry`; methane end-to-end; a
  topologically valid unplaced molecule featurizes but rejects geometry.
- Whole-workspace gates: `fmt --check`, `check`, `test`,
  `clippy --all-targets -D warnings` (all green).

## Limitations and future work

1. **No batching yet.** Featurization is one molecule → one `GraphTensors`
   batch of tensors. Batched training (padding, block-diagonal adjacency,
   per-molecule masks) arrives with the training pipeline phase, where the
   batch conventions can be designed against a real model.
2. **No integer index tensors.** `mi-tensor` is `f32`-only by design
   (Phase 4 backend decision); `edge_index` is therefore host-side
   `usize` pairs. When Phase 6's message passing needs index tensors,
   `mi-tensor` gains an integer tensor type and `edge_index` gains a
   tensor form.
3. **Categorical handling is deferred.** `atomic_number` stays a raw
   index (`0` reserved for padding/unknown by `mi-graph`); whether a model
   one-hots it or embeds it is a modeling decision, not a data decision.
4. **Richer element features** (electronegativity, valence, radii) are a
   schema bump in `mi-graph` (v2), not a featurization change — this layer
   follows whatever the schemas define.
5. **No caching.** `MolecularEntry::graph()` rebuilds the graph per call;
   featurization cost per molecule is microseconds at QM9 scale, so no
   cache exists until a workload justifies one.
