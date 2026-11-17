# Molecular Intelligence

A modular, extensible, Rust-first framework for computational chemistry and
molecular machine learning. The project targets the full path from molecular
representations to trained models and scientific evaluation, with an initial
research focus on molecular property prediction — HOMO, LUMO, and the
HOMO–LUMO gap.

## Current Status: Phase 9 — Training Infrastructure

The full pipeline exists and has produced measured results: raw
QM9 files (`mi-io`, with explicit Hartree→eV conversion and
`uncharacterized.txt` exclusions) → validated dataset (`mi-data`) →
reproducible 80/10/10 split (seeded Fisher–Yates) → descriptors and
targets (`mi-featurization`) → MLP baseline trained with per-sample SGD
on Burn's Flex CPU autodiff (`mi-models`) → MAE/RMSE/R² in electronvolts
(`mi-evaluation`). `mi-benchmark` executes the whole procedure as the
`qm9-baseline` binary: training-subset-only z-score standardization
(`mi-featurization::scaler`), validation-based checkpoint selection, and
JSON + Markdown reports with full provenance (split seed, schema
layouts, hyperparameters, timings) in `benchmarks/`. `mi-training`
(Phase 9) provides the optimization layer — SGD with momentum and Adam
over the leaf-renewal update contract (`Mlp::update_with`), plus
constant/exponential/cosine learning-rate schedules — driving the
recorded full-dataset runs: plain SGD reaches test MAE 0.256 eV (HOMO) /
0.342 eV (LUMO) / 0.415 eV (gap)
([`benchmarks/qm9-baseline/`](benchmarks/qm9-baseline/result.md)), and
Adam with cosine annealing improves on it
([`benchmarks/qm9-adam-cosine/`](benchmarks/qm9-adam-cosine/result.md)).
Scientific caveat kept in the docs: QM9 properties are B3LYP DFT values,
not GW or experimental observables. See
[`docs/architecture/benchmark.md`](docs/architecture/benchmark.md),
[`docs/architecture/baseline-model.md`](docs/architecture/baseline-model.md),
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
| `mi-data`        | lib  | Dataset loading and data processing. **Implemented (Phase 3 + current phase):** validated in-memory molecular datasets with HOMO/LUMO/gap properties, statistics, and reproducible seed-based random train/validation/test splitting. |
| `mi-io`          | lib  | Molecular file format I/O. **Implemented:** XYZ parser/writer; SDF (MDL V2000) parser/writer with bonds, formal charges, data fields, and multi-record files; streaming record reader for dataset-scale files; dataset import with property attachment and energy-unit conversion; QM9 loader (uncharacterized exclusion, provenance metadata); scientific provenance metadata; line-numbered typed errors; round-trip-tested formats. |
| `mi-tensor`      | lib  | Numerical and tensor abstractions. **Implemented:** Burn Flex-backed CPU tensors with ingress validation; graph-preserving arithmetic, matrix, activation, and tensor-reduction operations; differentiable MSE/MAE/Huber losses; `require_grad`/`backward`/`grad` autodiff API; separated host-side evaluation utilities. |
| `mi-featurization` | lib | Molecular structure → ML tensors. **Implemented (current phase):** graph tensors from versioned feature schemas, Euclidean distance geometry from Cartesian positions, HOMO/LUMO/gap targets as values/mask tensors in eV; schema-version provenance; explicit-absence conventions. |
| `mi-models`      | lib  | Machine learning model definitions. **Implemented (current phase):** MLP baseline (deterministic init, per-sample SGD, masked-MSE training) over molecular descriptors on Burn Flex CPU. |
| `mi-training`    | lib  | Training infrastructure. **Implemented (Phase 9):** canonical masked-MSE loop; SGD with momentum and Adam over the leaf-renewal update contract; constant/exponential/cosine learning-rate schedules. |
| `mi-evaluation`  | lib  | Scientific evaluation and metrics. **Implemented (current phase):** MAE/RMSE/R² for regression with typed degenerate-input errors. |
| `mi-inference`   | lib  | Model inference. |
| `mi-python`      | lib  | Future Python interoperability layer. |
| `mi-cli`         | bin  | Command-line interface. |
| `mi-benchmark`   | bin  | Benchmark suite. **Implemented (Phase 8):** `qm9-baseline` — the Experiment 01 pipeline with shared z-score scalers, validation checkpoint selection, and provenance-complete JSON/Markdown reports. |

These responsibilities are declarations of intent; `mi-chemistry`,
`mi-graph`, `mi-data`, `mi-tensor`, `mi-io`, `mi-featurization`,
`mi-models`, `mi-evaluation`, and `mi-benchmark` are implemented
(through the baseline benchmark), and the rest remains future work.

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
