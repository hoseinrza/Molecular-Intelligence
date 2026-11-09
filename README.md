# Molecular Intelligence

A modular, extensible, Rust-first framework for computational chemistry and
molecular machine learning. The project targets the full path from molecular
representations to trained models and scientific evaluation, with an initial
research focus on molecular property prediction — HOMO, LUMO, and the
HOMO–LUMO gap.

## Current Status: Phase 2 — Molecular Graph

The workspace foundation (Phase 0) and the chemistry core data model
(Phase 1, `mi-chemistry`: elements, atoms, bonds, molecules, structural
validation) are complete. `mi-graph` now converts validated molecules into
feature-rich molecular graphs: nodes and edges with versioned feature
schemas, adjacency lists, and a directed message-passing view. No datasets,
tensors, models, or training yet. See
[`docs/architecture/overview.md`](docs/architecture/overview.md),
[`docs/scientific-methodology/chemical-data-model.md`](docs/scientific-methodology/chemical-data-model.md),
and [`docs/architecture/molecular-graph.md`](docs/architecture/molecular-graph.md).

## Requirements

- Rust stable, **Edition 2024** (Rust 1.85 or newer) via [rustup](https://rustup.rs)
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
| `mi-data`        | lib  | Dataset loading and data processing. |
| `mi-tensor`      | lib  | Numerical and tensor abstractions. |
| `mi-models`      | lib  | Machine learning model definitions. |
| `mi-training`    | lib  | Training infrastructure. |
| `mi-evaluation`  | lib  | Scientific evaluation and metrics. |
| `mi-inference`   | lib  | Model inference. |
| `mi-python`      | lib  | Future Python interoperability layer. |
| `mi-cli`         | bin  | Command-line interface. |

These responsibilities are declarations of intent; `mi-chemistry` (Phase 1)
and `mi-graph` (Phase 2) are implemented, and the rest remains future work.

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
```

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
