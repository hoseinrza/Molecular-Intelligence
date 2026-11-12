# Molecular Intelligence

A modular, extensible, Rust-first framework for computational chemistry and
molecular machine learning. The project targets the full path from molecular
representations to trained models and scientific evaluation, with an initial
research focus on molecular property prediction — HOMO, LUMO, and the
HOMO–LUMO gap.

## Current Status: Molecular Dataset I/O (`mi-io`)

The workspace foundation, chemistry core (Phase 1, `mi-chemistry`),
molecular graphs (Phase 2, `mi-graph`), the scientific dataset layer
(Phase 3, `mi-data`), and the tensor backend (`mi-tensor`) are complete.
`mi-io` now provides the ingestion boundary for real molecular data: XYZ
parsing/writing (elements plus Cartesian positions in Ångström), SDF/MDL
molfile V2000 parsing/writing (positions, bonds with orders, formal
charges, per-record data fields, multi-record files), dataset import that
attaches electronic properties with explicit energy-unit conversion
(eV/Hartree) and scientific provenance metadata (source, method,
functional, basis set, source id, version), and a strict no-silent-loss
error policy with line-numbered diagnostics. Round-trip behavior is
integration-tested for both formats. No models or training loops yet. See
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
| `mi-models`      | lib  | Machine learning model definitions. |
| `mi-training`    | lib  | Training infrastructure. |
| `mi-evaluation`  | lib  | Scientific evaluation and metrics. |
| `mi-inference`   | lib  | Model inference. |
| `mi-python`      | lib  | Future Python interoperability layer. |
| `mi-cli`         | bin  | Command-line interface. |

These responsibilities are declarations of intent; `mi-chemistry`,
`mi-graph`, `mi-data`, `mi-tensor`, and `mi-io` are implemented, and the
rest remains future work.

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
