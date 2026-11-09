# Chemical Data Model

> Scope: **Phase 1** of Molecular Intelligence (`crates/mi-chemistry`).
> Everything in *Implemented* sections below exists in code today; everything
> in *Limitations* and *Future extension points* is planned and must not be
> assumed to work.

## Overview

The chemistry core implements the foundational data model — chemical
elements, atoms, bonds, and molecules — with structural validation and
total error handling. It is a *data model only*: no file parsing (SMILES,
XYZ, SDF), no graph algorithms, no 3D geometry, no valence rules, no
quantum chemistry. Later phases build on these types.

## Atom Representation

An `Atom` (`mi_chemistry::Atom`) bundles:

| Field                | Type           | Semantics |
| -------------------- | -------------- | --------- |
| element              | `ChemicalElement` | Validated identity: one of the 118 recognized elements. |
| formal charge        | `i8`           | Integer formal charge from standard valence/formal-charge rules (0 by default). |
| isotope mass number  | `Option<u16>`  | Integer protons + neutrons (A) of one specific isotope, if specified. |
| label                | `Option<String>` | Free-form bookkeeping text, never interpreted chemically. |

Design decisions:

- **Immutability.** Atoms are configured through builder-style methods
  (`with_formal_charge`, `with_isotope`, `with_label`) and are immutable
  after construction. There are no public mutable fields.
- **Validated identity.** The element field is a `ChemicalElement`, which
  can only be constructed through fallible lookups
  (`from_number`, `from_symbol`). An atom therefore cannot carry an invalid
  element.
- **No coordinates.** 3D coordinates are deliberately absent in Phase 1;
  adding them later must not change the identity semantics above.

### Atomic number versus isotope mass number

Two distinct integers describe an atom's nuclear composition:

- **Atomic number (Z, `u8`)** — the proton count; fully determines the
  element (H = 1, C = 6, …). `ChemicalElement` is exactly this handle.
- **Isotope mass number (A, `Option<u16>`)** — protons + neutrons of one
  specific isotope (²H → 2, ¹³C → 13). Optional: most workflows operate on
  natural-abundance elements and leave it unset.

The physically impossible case `A < Z` is rejected at construction
(`ChemistryError::InvalidIsotopeMassNumber`); A = 0 is covered by the same
rule. Note that A is a *count*, not a mass: the exact isotopic mass in u
(e.g. 13.003355 u for ¹³C) is **not** carried per atom yet — see
Limitations.

### Formal charge

Formal charge is the integer charge assigned by conventional Lewis-structure
rules (e.g. +1 for ammonium nitrogen, −1 for a hydroxide oxygen). It is
**not** a partial charge from quantum chemistry and **not** the net charge of
the molecule (the net charge is derivable but is not materialized as a
molecule-level field yet). Any `i8` value is accepted: the model makes no
assumption about chemically "reasonable" charge ranges.

## Bond Representation

A `Bond` (`mi_chemistry::Bond`) connects two atoms *by index* with a
`BondType`:

- Bonds are **undirected**. Endpoints are canonicalized (stored as
  smaller/larger) at construction, so `(a, b)` and `(b, a)` are the same
  bond — equality, hashing, and duplicate detection are order-independent.
- Self-bonds (`a == a`) are rejected at construction
  (`ChemistryError::SelfBond`).
- Endpoint *validity* (indices that actually exist) is a property of the
  molecule and is enforced when the bond is added.

### Bond type semantics

| Variant       | Meaning |
| ------------- | ------- |
| `Single`      | Classical single covalent bond. |
| `Double`      | Classical double covalent bond. |
| `Triple`      | Classical triple covalent bond. |
| `Aromatic`    | Bond participating in a delocalized aromatic π system. **Does not** encode a Kekulé structure; which ring bonds would alternate single/double is left unspecified, and no resonance/kekulization logic exists. No fixed length or order is implied. |
| `Unspecified` | Bond order unknown or not yet assigned (e.g. partial data). A valid, explicit state — bonds choose their type explicitly; there is no default bond type. |

## Molecule Representation

A `Molecule` (`mi_chemistry::Molecule`) is an indexed collection of atoms
and bonds plus optional metadata (name, identifier):

- **Indexing model.** Atoms and bonds are stored in insertion order;
  `add_atom`/`add_bond` return the assigned index. Indices are **stable for
  the lifetime of the molecule** because the structure is append-only —
  there is no removal in Phase 1 (removal would shift indices and is
  planned as an explicit, remapping-aware extension).
- **Simple graph.** At most one bond connects a given unordered atom pair;
  duplicates are rejected (`ChemistryError::DuplicateBond`) regardless of
  bond type or endpoint order. Molecules are not multi-graphs yet.
- **No chemical assumptions.** Net charge, radicals, hypervalence, and
  under-coordination are all representable; the model never rejects a
  structure merely for violating textbook valence rules.
- **Access API.** `atom(i)`/`bond(i)` return `Result` (out-of-range is an
  error, not a panic); `atoms()`/`bonds()` return slices for iteration;
  `atom_count()`/`bond_count()`/`is_empty()` are provided.

## Element Data

`ChemicalElement` is a validated handle around the atomic number
(`u8`, valid range 1..=118). All per-element data lives in **one** static
table (no duplication across modules); lookups:

- `from_number(u8) -> Result<ChemicalElement, _>` — by atomic number.
- `from_symbol(&str) -> Result<ChemicalElement, _>` — by IUPAC symbol,
  **case-sensitive** (`"Cl"`, not `"CL"`).
- `all()` — iteration over all 118 elements in atomic-number order.
- `symbol()` / `atomic_number()` / `atomic_mass()` — accessors.

### Data provenance and license

Element data was transcribed from the **PubChem PeriodicTable service**
(U.S. National Library of Medicine,
<https://pubchem.ncbi.nlm.nih.gov/rest/pug/periodictable/JSON>), retrieved
2026-11-10. PubChem data is in the **public domain**.

- For elements with an IUPAC standard atomic weight, PubChem reports the
  conventional weight (occasionally rounded — e.g. lead is reported as
  207 u).
- For elements **without** a standard atomic weight (Tc, Pm, Po, At, Rn,
  Fr, Ra, and the synthetic elements), the value is the atomic mass of a
  representative isotope (usually the longest-lived), not a terrestrial
  average.
- Units: unified atomic mass units (u, daltons).

These values are averaged/representative. They are **not** isotope-resolved
and must not be confused with the integer isotope mass number.

## Validation Rules (implemented)

`Molecule::validate` (backed by `validation::validate_structure`) checks
exactly these structural invariants:

1. The molecule contains at least one atom — empty molecules are rejected
   **explicitly** (`InvalidMolecularStructure`), not silently accepted.
2. Every bond endpoint references an existing atom index.
3. No bond connects an atom to itself.
4. No duplicate bonds exist (hash-set check over canonical endpoint pairs,
   O(number of bonds)).

The same rules are enforced *at mutation time* (`add_bond` rejects invalid
endpoints and duplicates immediately); `validate` re-checks defensively and
defines the contract for molecules assembled by any future construction
path. **Valence rules are deliberately not implemented** — unusual species
must validate cleanly (this is covered by tests: hydronium, hydrogen
radical).

## Error Handling

All fallible operations report `ChemistryError` (implements
`std::error::Error` with descriptive `Display` messages): invalid atomic
number, unknown symbol, invalid atom/bond index, self-bond, duplicate bond,
invalid isotope mass number, invalid molecular structure. Invalid input
never panics and is never silently ignored.

## Current Limitations

- No 3D coordinates and no geometry of any kind.
- No isotopic masses per isotope: only the integer mass number is carried;
  `atomic_mass()` always returns the element-level averaged value.
- No isotopic abundances / natural distribution model.
- Append-only structure: no atom/bond removal (indices would shift).
- Simple graphs only: no parallel (multiple) bonds between one atom pair,
  and no "bond order change in place" API.
- No stereochemistry (E/Z, R/S), no atom mapping, no residues/substructures.
- No valence rules, resonance, kekulization, or aromaticity *perception*
  (aromatic bonds are taken as given, never inferred).
- No molecular format parsing (SMILES, XYZ, SDF, MOL) and no serialization.
- Formal charges only; no partial charges.
- Net molecular charge is not materialized (derivable from atoms).

## Future Extension Points

- **Coordinates**: per-atom 3D positions (and, much later, per-conformer
  ensembles) as an additive field; identity semantics unchanged.
- **Isotopes**: exact isotopic masses and abundances in the element table;
  isotope-aware molecular mass.
- **Removal/editing**: atom/bond removal with explicit index remapping or
  handle-based (generational) indices.
- **Serialization**: MOL/SDF, XYZ, and later SMILES readers/writers built on
  this model (Phase 2+ targets the graph layer).
- **Aromaticity perception and kekulization** as explicit, optional passes.
- **Molecule-level derived properties**: net charge, molecular formula,
  monoisotopic/average mass.
- **Multi-graph support** if a representation requiring parallel bonds is
  needed.
