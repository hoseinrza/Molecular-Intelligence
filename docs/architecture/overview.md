# Architecture Overview

> Status: Phase 0 — repository and workspace foundations. Nothing in this
> document describes implemented behavior yet; it records intent and the
> boundaries that future phases must respect.

## Project Vision

Molecular Intelligence is a research-oriented, Rust-first framework for
computational chemistry and molecular machine learning. Its long-term goal is
to cover the full path from raw molecular representations to trained models
and scientific evaluation, with an initial scientific focus on molecular
property prediction — in particular HOMO, LUMO, and the HOMO–LUMO gap.

The project is designed around three principles:

1. **Rust-first.** Core computation, data handling, and modeling live in
   native Rust crates. Python is a boundary, not the foundation.
2. **Modular but minimal.** Each crate owns one clearly bounded
   responsibility. Dependencies between crates are explicit and acyclic.
3. **Science and infrastructure are separated.** Domain knowledge (chemistry,
   molecular graphs, datasets, metrics) is kept apart from plumbing (tensor
   backends, training loops, I/O) so either side can evolve independently.

## Workspace Structure

The repository is a single Cargo workspace (Rust Edition 2024) with all
library and binary crates under `crates/`:

```text
molecular-intelligence/
├── Cargo.toml          # Workspace manifest and shared package metadata
├── crates/             # All Rust crates (see Crate Responsibilities)
├── datasets/           # Local datasets (contents git-ignored)
├── experiments/        # Experiment definitions and results
├── checkpoints/        # Model checkpoints (contents git-ignored)
├── benchmarks/         # Benchmark definitions and results
├── configs/            # Configuration files for experiments and runs
├── docs/               # Architecture, scientific methodology, reproducibility
├── scripts/            # Utility scripts (data prep, CI helpers, …)
└── tests/              # Workspace-level integration tests and fixtures
```

Shared package metadata (`version`, `edition`, `rust-version`, `license`) is
defined once in the root `Cargo.toml` under `[workspace.package]` and
inherited by every crate. Workspace-level dependency inheritance
(`[workspace.dependencies]`) is intentionally unused until a dependency is
genuinely needed by more than one crate.

## Crate Responsibilities

| Crate            | Kind | Responsibility |
| ---------------- | ---- | -------------- |
| `mi-core`        | lib  | Shared types, error types, and foundational abstractions used across the workspace. |
| `mi-chemistry`   | lib  | Molecular and atomic representations (elements, atoms, molecules). |
| `mi-graph`       | lib  | Molecular graph representations built on top of chemical entities. |
| `mi-data`        | lib  | Dataset loading, dataset management, and data processing. |
| `mi-tensor`      | lib  | Numerical and tensor abstractions decoupling models from any specific backend. |
| `mi-models`      | lib  | Machine learning model definitions operating on molecular representations. |
| `mi-training`    | lib  | Training infrastructure: loops, optimization, checkpointing orchestration. |
| `mi-evaluation`  | lib  | Scientific evaluation and metrics (e.g., property-prediction error metrics). |
| `mi-inference`   | lib  | Model inference over trained artifacts. |
| `mi-python`      | lib  | Future Python interoperability layer exposing the Rust API to Python. |
| `mi-cli`         | bin  | Command-line interface; the primary user-facing entry point. |

## Intended Dependency Direction

Dependencies point downward only. No crate may depend on a crate above it in
the following layering, and no cycles are allowed:

```text
        ┌──────────────────────────────────────────────┐
        │            Interface layer                   │
        │   mi-cli (bin)          mi-python            │
        └──────────────┬───────────────────────────────┘
                       │
        ┌──────────────▼───────────────────────────────┐
        │          ML lifecycle layer                  │
        │  mi-training   mi-evaluation   mi-inference  │
        │                mi-models                     │
        └──────┬───────────────┬───────────────┬───────┘
               │               │               │
        ┌──────▼──────┐ ┌──────▼──────┐ ┌──────▼──────┐
        │  mi-data    │ │  mi-graph   │ │  mi-tensor  │
        │  mi-chemistry│ │             │ │             │
        └──────┬──────┘ └──────┬──────┘ └──────┬──────┘
               │               │               │
        ┌──────▼───────────────▼───────────────▼───────┐
        │                  mi-core                     │
        │        (shared types, errors, traits)        │
        └──────────────────────────────────────────────┘
```

Expected (future) internal dependencies:

- `mi-core` — depends on nothing internal.
- `mi-chemistry`, `mi-tensor` — depend on `mi-core`.
- `mi-graph` — depends on `mi-chemistry` (chemical entities to represent as
  graphs) and `mi-core`.
- `mi-data` — depends on `mi-chemistry`/`mi-graph` for decoded content and
  `mi-core`.
- `mi-models` — depends on `mi-graph` (model inputs) and `mi-tensor`
  (computation substrate), plus `mi-core`.
- `mi-training`, `mi-inference` — depend on `mi-models`, `mi-data`,
  `mi-tensor`, `mi-core`.
- `mi-evaluation` — depends on `mi-core` and `mi-data`; it consumes
  predictions and references, not models directly.
- `mi-cli`, `mi-python` — composition roots; may depend on any library crate
  but contain no scientific logic themselves.

At the end of Phase 0 no inter-crate dependencies exist; this diagram is the
contract that later phases extend.

## Separation of Scientific Logic and Infrastructure

Two axes are kept apart:

- **Scientific logic** — what a molecule *is* (`mi-chemistry`), how it becomes
  a graph (`mi-graph`), what datasets mean (`mi-data`), and what numbers are
  scientifically meaningful (`mi-evaluation`). This code is written to be
  correct, inspectable, and unit-testable without heavy dependencies.
- **Infrastructure** — how computation happens (`mi-tensor`), how models are
  optimized (`mi-training`), and how results are served (`mi-inference`,
  `mi-cli`, `mi-python`). This code is written to be swappable and fast.

Concretely: chemical and dataset types never depend on tensor types, and
tensor backends never import chemistry. Models sit at the junction and are the
only place where molecular representations meet numerical computation. This
keeps scientific correctness reviewable in isolation and allows backend or
training-technology changes without touching domain code.

## Future Rust/Python Interoperability

`mi-python` is reserved as the single boundary between the Rust workspace and
the Python ecosystem:

- It will wrap stable, high-level Rust APIs (not internal crates) so the
  binding surface stays small and durable.
- Python will consume the framework through `mi-python` only; it will never
  reach into `mi-chemistry`, `mi-graph`, etc. directly. This keeps the
  internal layout free to change without breaking downstream users.
- The binary interface (`mi-cli`) is expected to remain the primary
  non-Python entry point, so the workspace stays usable with no Python
  installed.

The binding technology (e.g., PyO3/maturin) is deliberately not chosen in
Phase 0.
