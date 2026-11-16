# QM9 Pipeline Architecture

> Scope: **Phase 6 — QM9 Pipeline** of Molecular Intelligence
> (`mi-io` streaming + `mi_io::qm9` + `mi-data::split`). *Implemented*
> sections describe code that exists and is tested today; *limitations*
> and *future work* sections are plans, not capabilities.

## Overview

QM9 is the reference dataset of the project's first case study —
benchmarking machine-learning models for **HOMO/LUMO/gap prediction**.
The pipeline turns the raw distribution into validated, typed, split
datasets with full provenance, entirely in Rust:

```text
qm9.sdf (~1.3 GB, 133,885 records)
   │  SdfRecordReader  (streaming, O(one record) memory)
   ▼
mi_io::qm9::load_qm9_sdf
   │  uncharacterized.txt exclusion ──► typed properties (Hartree → eV)
   ▼
MolecularDataset + DatasetMetadata (QM9 · DFT · B3LYP · 6-31G(2df,p))
   │  mi_data::random_split(seed)
   ▼
Split { train, validation, test }      → Phase 7 models consume these
```

## Distribution format (what the loader assumes)

The QM9 distribution (figshare collection
[10.6084/m9.figshare.c.1290126](https://doi.org/10.6084/m9.figshare.c.1290126),
CC BY 4.0) ships:

- **`qm9.sdf`** — all molecules as V2000 molfiles with 3D coordinates and
  19 property **data fields**. The loader reads this file.
- **`uncharacterized.txt`** — indices of molecules whose calculations had
  convergence/formatting problems (≈3,054 entries; the exact count is
  read from the file at runtime, never hardcoded).
- **`gdb9.tar.gz`** — per-molecule xyz files (not read by this pipeline).

**Units matter and are a classic trap:** in the raw distribution the
`homo`, `lumo`, and `gap` data fields are in **Hartree**. Many ecosystems
convert to eV on load (PyTorch Geometric, JARVIS do exactly this). This
pipeline makes the policy explicit — [`Qm9Options`] defaults to
`EnergyUnit::Hartree` and converts with `mi_data::Energy::HARTREE_TO_EV`
(CODATA 2018), so the stored dataset is always in electronvolts; a
pre-converted eV source passes `EnergyUnit::Ev` instead.

Property field tags default to `homo` / `lumo` / `gap` (matching the SDF
data-field names); a file that renamed its fields is handled by
`Qm9Options::with_property_fields`.

## Exclusion and honesty mechanisms

- **Uncharacterized molecules** are excluded before import, keyed by
  1-based record position (which mirrors the distribution's xyz-file
  numbering). The report carries `excluded_uncharacterized`.
- **Title verification:** each `gdb9_<n>` title is checked against the
  record's file position; disagreements are counted in
  `Qm9Report::title_mismatches` — a well-formed file reports zero, so any
  drift between title numbering and file order becomes visible instead of
  silently shifting the exclusion map.
- **Nothing dropped silently:** records that parse but fail import
  (invalid structure, unparseable property) appear in `Qm9Report::skipped`
  with reasons; strict mode aborts instead.

## Split policy

`mi_data::random_split(dataset, fractions, seed)` performs a
Fisher–Yates permutation of entry indices under a SplitMix64 generator
(implemented in-crate, no dependency; the sequence is pinned by tests so
splits are reproducible across builds). The test subset absorbs rounding,
so the three subsets cover the dataset exactly and disjointly.

**Random vs scaffold — the honest generalization question.** A random
split scatters chemically similar molecules across train and test, so
error estimates can overstate generalization to *new chemistry*. The
standard harder evaluation is a **scaffold split** (grouping by Bemis–
Murcko framework); it is planned as a follow-up, arriving through the
Python boundary where scaffold tooling exists. Benchmark reports must
state which split was used and should eventually report both.

## Scientific caveats (keep these visible)

1. **DFT, not GW, not experiment.** QM9 orbital energies come from
   B3LYP/6-31G(2df,p). Predicting them well does not mean predicting GW
   or experimental ionization potentials; Fediai et al. (2023) provide a
   GW reference set for the same molecules as a future target.
2. **The uncharacterized list exists for a reason** — those structures'
   calculations failed or produced problematic output; excluding them is
   the standard practice of QM9-based studies.
3. **QM9 charges/hydrogens are B3LYP-optimal** — geometries are computed
   structures, not crystallographic or measured data.

## Runtime verification checklist (run against a local copy)

Place `qm9.sdf` and `uncharacterized.txt` under `datasets/` (git-ignored)
and check:

1. `uncharacterized_indices` returns ≈3,054 entries (file-dependent).
2. `load_qm9_sdf` reports `record_count == 133_885`.
3. `title_mismatches == 0` (titles line up with positions).
4. First record is `gdb9_1` (methane, CH₄): its `gap` in eV should match
   the published value (Hartree field × 27.211…).
5. Accepted + skipped + excluded = record count, and `skipped` is empty on
   a clean download.

Any mismatch means a format assumption in this document is wrong for that
distribution revision — fix the loader, not the report.

## Usage

```rust
use mi_io::{load_qm9_sdf, uncharacterized_indices, Qm9Options};
use mi_data::random_split;
use std::collections::BTreeSet;

let excluded = uncharacterized_indices("datasets/uncharacterized.txt")?;
let report = load_qm9_sdf("datasets/qm9.sdf", &Qm9Options::new(), &excluded)?;

println!(
    "records {} · excluded {} · accepted {} · skipped {}",
    report.record_count(),
    report.excluded_uncharacterized(),
    report.dataset().len(),
    report.skipped().len(),
);

let split = random_split(report.dataset(), Default::default(), 42)?;
for (name, subset) in [("train", split.train()), ("validation", split.validation()), ("test", split.test())] {
    println!("{name}: {} entries", subset.len());
}
# Ok::<(), mi_io::IoError>(())
```

## Testing

- **Streaming reader:** lazy iteration, trailing record without `$$$$`,
  blank-line handling between/inside records, error line numbers, empty
  documents.
- **QM9 loader** (fixture records in Hartree): property conversion to eV,
  exclusion by position, title-mismatch counting, skip reporting, strict
  mode, eV sources, empty/missing files, tolerant-but-honest
  `uncharacterized.txt` parsing.
- **Split** (`mi-data`): same-seed reproduction, different-seed
  divergence, exact sizes/disjoint cover, fraction validation, empty
  dataset, per-subset statistics, provenance-carrying subset names.
- **Integration** (`tests/qm9_pipeline.rs`): stream → load → split →
  statistics end-to-end.

## Limitations and future work

1. **Random split only** — scaffold split is a planned follow-up (see
   above).
2. **Only the three focus properties are stored** in `mi-data`; the other
   16 QM9 fields remain in the source file and are future typed
   extensions of the property model.
3. **Single-threaded streaming** — a full QM9 load is I/O- and parse-bound
   sequential work; if it ever becomes slow, parallel record parsing can
   be added behind the same API.
4. **No download automation** — the user places the files under
   `datasets/` themselves (the directory is git-ignored by policy).
