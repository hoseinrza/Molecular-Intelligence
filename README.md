# Molecular Intelligence

A modular, extensible, Rust-first framework for computational chemistry and
molecular machine learning. The project targets the full path from molecular
representations to trained models and scientific evaluation, with an initial
research focus on molecular property prediction — HOMO, LUMO, and the
HOMO–LUMO gap.

## Current Status: Molecular Tensor (`mi-featurization`)

The workspace foundation, chemistry core (Phase 1, `mi-chemistry`),
molecular graphs (Phase 2, `mi-graph`), the scientific dataset layer
(Phase 3, `mi-data`), and the tensor backend (`mi-tensor`) are complete.
`mi-featurization` now provides the molecular↔numerical junction:
molecular graphs become `mi-tensor` tensors assembled from `mi-graph`'s
versioned feature schemas (node matrix, directed edge index, edge
features, with schema versions as provenance), Cartesian positions become
Euclidean distance features in Ångström, and dataset entries become
HOMO/LUMO/gap regression targets as an explicit values/mask tensor pair
in electronvolts. Absence is always explicit (Option edges, presence
masks) — no invented features, no sentinels. No models or training loops
yet. See
[`docs/architecture/overview.md`](docs/architecture/overview.md),
[`docs/architecture/dataset-layer.md`](docs/architecture/dataset-layer.md),
[`docs/architecture/dataset-io.md`](docs/architecture/dataset-io.md),
and [`docs/architecture/tensor-backend.md`](docs/architecture/tensor-backend.md).

## Requirements

- Rust stable, **Edition 2024** (Rust 1.95 or newer — Burn 0.22.1's MSRV)
  via [rustup](https://rustup.rs)
- Cargo (bundled with rustup)
- `rustfmt` (`rustup component add rustfmt`)

## Repository Layout

```text
molecular-intelligence/
├── Cargo.toml          # Workspace manifest and shared package metadata
├── crates/             # All Rust crates
├── datasets/           # Local datasets (contents git-ignored)
├── experiments/        # Experiment definitions and results
├── checkpoints/        # Model checkpoints (contents git-ignored)
├── benchmarks/         # Benchmark definitions and results
├── configs/            # Configuration files
├── docs/               # architecture/ · scientific-methodology/ · reproducibility/
├── scripts/            # Utility scripts
└── tests/              # Workspace-level integration tests
```

## Crates

| Crate            | Kind | Intended Responsibility |
| ---------------- | ---- | ----------------------- |
| `mi-core`        | lib  | Shared types, errors, and foundational abstractions. |
| `mi-chemistry`   | lib  | Molecular and atomic representations. **Implemented (Phase 1):** elements, atoms, bonds, molecules, structural validation. |
| `mi-graph`       | lib  | Molecular graph representations. **Implemented (Phase 2):** molecular graphs, feature schemas, adjacency, molecule-to-graph conversion. |
| `mi-data`        | lib  | Dataset loading and data processing. **Implemented (Phase 3):** validated in-memory molecular datasets with HOMO/LUMO/gap properties and statistics. |
| `mi-io`          | lib  | Molecular file format I/O. **Implemented (current phase):** XYZ parser/writer; SDF (MDL V2000) parser/writer with bonds, formal charges, data fields, and multi-record files; dataset import with property attachment and energy-unit conversion; scientific provenance metadata; line-numbered typed errors; round-trip-tested formats. |
| `mi-tensor`      | lib  | Numerical and tensor abstractions. **Implemented:** Burn Flex-backed CPU tensors with ingress validation; graph-preserving arithmetic, matrix, activation, and tensor-reduction operations; differentiable MSE/MAE/Huber losses; `require_grad`/`backward`/`grad` autodiff API; separated host-side evaluation utilities. |
| `mi-featurization` | lib | Molecular structure → ML tensors. **Implemented (current phase):** graph tensors from versioned feature schemas, Euclidean distance geometry from Cartesian positions, HOMO/LUMO/gap targets as values/mask tensors in eV; schema-version provenance; explicit-absence conventions. |
| `mi-models`      | lib  | Machine learning model definitions. |
| `mi-training`    | lib  | Training infrastructure. |
| `mi-evaluation`  | lib  | Scientific evaluation and metrics. |
| `mi-inference`   | lib  | Model inference. |
| `mi-python`      | lib  | Future Python interoperability layer. |
| `mi-cli`         | bin  | Command-line interface. |

These responsibilities are declarations of intent; `mi-chemistry`,
`mi-graph`, `mi-data`, `mi-tensor`, `mi-io`, and `mi-featurization` are
implemented, and the rest remains future work.

## Architecture

Dependencies between crates point strictly downward through the layers
`interface → ML lifecycle → domain/data → core`; details, rationale, and the
planned Rust/Python boundary are documented in
[`docs/architecture/overview.md`](docs/architecture/overview.md).

## Quality Checks

```bash
cargo metadata --no-deps       # Workspace manifest sanity
cargo check --workspace        # Type-check every crate
cargo fmt --all -- --check     # Verify formatting
cargo test --workspace         # Run all tests
cargo clippy --workspace --all-targets -- -D warnings
```

> **Note:** the Burn **0.22.1** pin (`flex` + `autodiff`) and the tensor
> layer are verified with local builds and tests; see
> [`docs/architecture/tensor-backend.md`](docs/architecture/tensor-backend.md)
> for the verification environment and remaining limitations.

## Conventions

- All crates inherit shared metadata (`version`, `edition`, `rust-version`,
  `license`) from the root `[workspace.package]` section.
- Scientific logic lives in domain crates; infrastructure lives in
  `mi-tensor`/`mi-training`/`mi-inference`; `mi-cli` and `mi-python` are
  composition roots and contain no scientific logic.
- Large research artifacts (dataset files, checkpoints) are not committed;
  see `.gitignore`.

## License

MIT — see [LICENSE](LICENSE).
