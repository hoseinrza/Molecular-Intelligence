# Dataset I/O Architecture

> Scope: **Phase 4 — Molecular Dataset I/O** of Molecular Intelligence
> (`crates/mi-io`). *Implemented* sections describe code that exists and
> is tested today; *limitations* and *future work* sections are plans,
> not capabilities.

## Overview

The I/O layer is the ingestion boundary between raw computational-chemistry
files and the project's typed, validated data model. It depends on
`mi-chemistry` (molecules to build) and `mi-data` (datasets to populate) —
and on nothing else: no ML, no tensors. The chemistry layer stays
learning-algorithm-free, and I/O stays free of learning algorithms.

```text
XYZ file ─┐
          ├─ parse → Molecule ─┐
SDF file ─┘                     ├─ import → MolecularDataset + DatasetMetadata
                                │
SDF data fields ────────────────┘  (electronic properties, unit-converted)
```

Dependency direction stays strictly downward
(`interface → ML lifecycle → domain/data → core`); `mi-io` sits at the
ingestion edge and nothing depends on it yet.

## Formats

### XYZ (`mi_io::xyz`)

The de-facto exchange format for coordinates: atom-count line, comment
line, then `Symbol x y z` per atom, coordinates in **Ångström**.

- Single-atom files are valid molecules; one file yields exactly one
  molecule (trajectories are a planned extension).
- The comment line becomes the molecule name when non-empty.
- Atom lines must carry at least the four fields; **extra trailing
  columns are ignored** (extended-XYZ payloads this phase does not model),
  so legacy files with charges/masses columns still import their
  structure faithfully.
- Anything beyond the declared atom count is an error, never a truncation.
- Writing emits the canonical layout with six decimals per coordinate; a
  write–parse round trip preserves elements, names, and positions within
  `1e-6` Å (integration-tested). XYZ has no bond concept — bonds cannot
  survive an XYZ round trip, by format design.

### SDF / MDL molfile V2000 (`mi_io::sdf`)

The structure-carrier format of chemical datasets (QM9 ships as SDF):
positions, bonds with orders, formal charges, per-record data fields,
multi-record files separated by `$$$$`.

Supported subset — deliberately matched to what real datasets use:

| Feature | Behavior |
| ------- | -------- |
| Records | `$$$$`-separated; whitespace-only blocks skipped; a non-empty block that fails is an error |
| Atom block | Fixed columns first, whitespace-token fallback for column-drifted files; element + position (Å) |
| Bond block | Codes `1`→Single, `2`→Double, `3`→Triple, `4`→Aromatic; codes `5`–`8` are **rejected**, never guessed |
| Charges | `M  CHG` lines applied at construction; atom references out of range are rejected with line numbers |
| Data fields | `> <TAG>` blocks after `M  END`, verbatim values (multi-line kept); interpreted only by the import layer |
| Name | Record title line, when non-empty |

Rejected explicitly (no silent simplification): V3000 molfiles, stereo
descriptors, isotopes (`M  ISO`), alias lines, property lines appearing
after `M  END`.

Writing emits the canonical V2000 layout (`%10.4f` coordinates, `M  CHG`
for formal charges). A write–parse round trip preserves names, elements,
positions within `1e-4` Å (the format's standard precision), bond orders,
formal charges, and data fields (integration-tested).
`BondType::Unspecified` has no MDL code; writing it is an error rather
than a silent reinterpretation.

## Dataset import (`mi_io::import`)

Import turns parsed records into a
[`MolecularDataset`](../../crates/mi-data) with provenance attached.

```text
SdfRecord ──► MolecularEntry (mi-data validation)
   │
   └─ data fields ──► ElectronicProperties (homo / lumo / gap)
              with values converted per DatasetMetadata.energy_unit
```

- **Nothing is dropped silently.** Every record either becomes a
  validated entry or appears in `ImportedDataset::skipped` with its
  0-based record index, title, and the typed reason. Strict mode
  (`ImportOptions::with_strict(true)`) aborts on the first problem
  instead. Import of a document with zero records is an error.
- **Properties are optional.** A record without property fields imports
  as a structural entry; absence of a property is `None` in `mi-data`,
  never a sentinel. The HOMO–LUMO gap is stored when the source provides
  it and derived (by `mi-data`) when both levels are present.
- **Units are explicit.** `EnergyUnit::Hartree` source values are
  converted to electronvolts with `mi_data::Energy::HARTREE_TO_EV`
  (CODATA 2018); `EnergyUnit::Ev` values pass through. The converted,
  storage-unit values are what land in the dataset.
- **Field names are configurable.** `PropertyFields` defaults to the
  QM9-style tags (`homo`, `lumo`, `gap`) and can be remapped per source.

## Scientific metadata (`mi_io::metadata`)

`DatasetMetadata` records provenance: **source** and **energy unit** are
required at construction; method, functional, basis set, source id, and
version are explicit validated builders (empty content rejected). Its
`Display` renders a stable `key: value` provenance block with unrecorded
fields omitted — absence stays explicit, matching the `mi-data`
convention of `Option`-carried properties.

The metadata travels with the imported dataset (`ImportedDataset::
metadata()`) and names the dataset (`MolecularDataset::with_name`) — a
dataset can never become anonymous, and the question "is this HOMO in eV
or Hartree?" is answered by data, not convention.

## Error policy

All failures report [`IoError`](../../crates/mi-io/src/error.rs) — typed,
with the **1-based file line** of the offending data wherever that is
meaningful (malformed lines, unknown elements, bad coordinates, bond
order codes, atom references, `M  CHG` entries). Chemistry-level
failures (duplicate bonds, invalid references) and dataset-level
rejections (invalid entries) are wrapped with their source. Nothing
panics on malformed input; nothing is silently approximated.

## Position support in `mi-chemistry`

File formats carry coordinates, so the data model needed them: `Atom`
now optionally holds a validated `Position` (Ångström, finite-checked,
unit fixed at the type level). Consequences:

- `Atom`/`Molecule`/`ChemistryError` derive `PartialEq` (float positions
  preclude `Eq`); nothing in the workspace relied on `Eq`/`Hash` for
  these types.
- Topological uses are unaffected: position is `Option`, and purely
  graph-level code never reads it.

## Testing

- **Unit tests** in every module: parser grammar, writer layout, error
  construction, metadata builders, import policies (skip reporting,
  strict mode, unit conversion, custom field names).
- **Integration tests** (`crates/mi-io/tests/`), inline documents only —
  no external fixtures, no network:
  - `round_trip.rs` — XYZ and SDF write–parse round trips for
    monoatomic and polyatomic molecules, unnamed molecules, multi-record
    files, data fields, formal charges, and file-level (disk) round
    trips.
  - `invalid_inputs.rs` — line-numbered errors for every failure class,
    NaN/inf coordinates in every slot, case-sensitive symbols,
    unsupported bond orders, truncated/empty documents, incomplete
    property data importing without properties.

## Limitations and future work

1. **XYZ trajectories** (multi-molecule `.xyz` files) are not parsed yet.
2. **V3000 molfiles** are not supported; V2000 covers the datasets this
   pipeline targets (QM9-style).
3. **Stereochemistry, isotopes, aliases, query features** are parsed as
   errors, not ignored — support is a per-feature decision for later
   phases.
4. **Property fields cover electronic energies** (homo/lumo/gap); more
   observables (dipole moments, frequencies) will extend `PropertyFields`
   with their own units.
5. **No incremental/streaming import yet** — files are parsed in memory,
   which fits the current dataset scale.
