# Dataset Layer Architecture

> Scope: **Phase 3** of Molecular Intelligence (`crates/mi-data`).
> *Implemented* sections describe code that exists today; *limitations* and
> *future work* sections are plans, not capabilities.

## Overview

The dataset layer provides typed, validated, in-memory molecular datasets.
Each **entry** pairs a validated `mi_chemistry::Molecule` with optional
`ElectronicProperties` — the project's scientific focus: HOMO, LUMO, and
the HOMO–LUMO gap. The layer depends on `mi-chemistry` (chemical truth) and
`mi-graph` (on-demand graph views); nothing depends on it yet, keeping the
dependency direction of the architecture intact
(`interface → ML lifecycle → domain/data → core`).

This phase is **in-memory only**: no file parsing, serialization, or
splitting. Loaders (SDF/XYZ/QM-style) and train/validation/test splits are
planned future extensions of this same crate.

## Data Model

```text
MolecularDataset
├── name: Option<String>
└── entries: Vec<MolecularEntry>      (append-only, index = position)
      └── MolecularEntry
          ├── molecule: Molecule       (validated at construction)
          ├── properties: ElectronicProperties
          │     ├── homo: Option<Energy>   (eV)
          │     ├── lumo: Option<Energy>   (eV)
          │     └── gap:  Option<Energy>   (eV)
          └── graph() -> Result<MolecularGraph>   (on demand, via mi-graph)
```

- **Append-only indexing.** `push` returns the entry's index; indices start
  at 0 and remain stable for the dataset's lifetime (no removal exists).
  This matches the stability models of `mi-chemistry` and `mi-graph`.
- **Validation at the boundary.** `MolecularEntry::new` runs the molecule's
  structural validation; empty molecules are rejected (a record without a
  structure is meaningless). `Energy` accepts only finite values, so a
  dataset can never hold NaN/infinite energies. `MolecularDataset::validate`
  re-checks all entries defensively; an **empty dataset is a valid state**
  (a container legitimately starts empty — unlike molecules and graphs).
- **Explicit absence.** Every property is an `Option`; no sentinel values
  anywhere.

## Energy and Units

`Energy` is a validated newtype over `f64` in **electronvolts (eV)** — the
unit confusion with Hartree atomic units becomes a compile-time impossibility.

- `Energy::from_ev(f64) -> Result<Self, DataError>` — rejects non-finite.
- `Energy::from_hartree(f64) -> Result<Self, DataError>` — multiplies by the
  CODATA 2018 factor `HARTREE_TO_EV = 27.211_386_245_988` eV/E_h (NIST
  recommended value of the hartree in electronvolts). Sources reporting in
  Hartree (e.g. QM9-style datasets) convert at the boundary and store eV.
- `ev()` / `hartree()` access both unit systems explicitly.

## Property Semantics

- `homo`, `lumo`, `gap` are independent optionals — sources may provide any
  subset.
- `homo_lumo_gap()` returns the **effective** gap: the stored value when
  present (sources are trusted for what they provide, even if it disagrees
  with HOMO/LUMO), otherwise derived as `LUMO − HOMO` when both exist,
  otherwise `None`.
- Derived gaps are re-validated for finiteness; overflow falls back to
  `None` rather than propagating a non-finite value.

## Statistics

`MolecularDataset::statistics()` yields per-property coverage and summaries
over **available** values only (entries lacking a property are excluded from
that property's summary, and `summary.count()` *is* the coverage):

| Field | Definition |
| ----- | ---------- |
| `count` | number of contributing entries |
| `mean` | arithmetic mean |
| `min` / `max` | extremes |
| `std_dev` | **population** standard deviation (÷ N) — the convention is documented so downstream normalization code is unambiguous |

Gap statistics use the effective gap (stored preferred, derived otherwise).

## Error Handling

All fallible operations report `DataError` (implements `std::error::Error`
with descriptive `Display` messages): `InvalidMolecule` (wrapping
`ChemistryError`, reachable through `Error::source` — the same pattern as
`GraphError`), `InvalidEntryIndex`, `InvalidEnergy`, and
`InconsistentDataset` (defensive). No panics on invalid input.

## Known Limitations

- In-memory and owned (`Molecule` by value): no external references, no
  memory-mapped or streaming datasets, no zero-copy access.
- No file-format loaders or serialization (SDF/XYZ/JSON are future work in
  this crate).
- No train/validation/test splitting or shuffling (planned; a deterministic,
  documented split strategy will accompany it).
- No property deduplication or identifier-uniqueness enforcement.
- Graph views are recomputed on every `entry.graph()` call — no caching.
- Statistics are eager and allocate a `Vec<f64>` per property per call.

## Future Work

- **Loaders/serialization** in this crate: a documented text/JSON
  representation, then standard formats (SDF, XYZ, QM output parsers) built
  on `mi-chemistry`.
- **Splits** with a deterministic, seedable strategy and stratification by
  property ranges.
- **Graph caching** per entry (immutable molecules make graphs pure
  functions of the entry; a lazily-initialized cache is safe) once datasets
  feed model training.
- **Larger property sets** (dipole moments, zero-point energies, atomization
  energies) as additive, versioned extensions of `ElectronicProperties`.
