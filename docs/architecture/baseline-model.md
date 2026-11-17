# Baseline Model Architecture

> Scope: **Phase 7 — Baseline Model + Metrics** of Molecular
> Intelligence (`mi-models`, `mi-evaluation`, one `mi-tensor` addition,
> one `mi-featurization` addition). *Implemented* sections describe code
> that exists and is tested today; *limitations* and *future work*
> sections are plans, not capabilities.

## Overview

Phase 7 closes the pipeline the research plan's Experiment 01 needs: a
**Rust-native MLP baseline** that trains on featurized molecules with
Burn's autodiff, plus the **MAE / RMSE / R²** metrics the benchmark
reports. The stack decision is Rust-native first: the full chain — QM9
files → dataset → descriptors → training → metrics — runs in native Rust
with no Python in the loop.

```text
MolecularDataset ──► descriptor_vector (18 slots, layout v1) ─┐
MolecularEntry   ──► Targets (values[3] + mask[3], eV) ───────┤
                                                              ▼
                                    Mlp (Burn Flex CPU, autodiff)
                                    · forward: ReLU hidden, linear output
                                    · masked-MSE loss
                                    · per-sample SGD (leaf renewal)
                                                              ▼
                        predictions (eV) ──► mae / rmse / r² (mi-evaluation)
```

## What each crate contributes

### `mi-tensor` — two thin additions
- `operations::mul_scalar` — the optimizer primitive (`lr·∇p`); graph-
  preserving, shape-preserving.
- `Tensor::reshape` — validated (non-zero dims, element count preserved)
  and autodiff-preserving; the model consumes `[1, width]` rows.

Both are thin wrappers over Burn; no numerical code is reimplemented.

### `mi-featurization` — fixed-size descriptor (layout v1)
`descriptor_vector(molecule, device)` returns a rank-1 tensor of width
[`DESCRIPTOR_WIDTH`] = 18:

| Slots | Content | Notes |
| ----- | ------- | ----- |
| 0–5   | per-column sums of `NodeFeatures::to_vector` | total atomic number, net formal charge, isotope sums, degree sums |
| 6–12  | per-column sums of `EdgeFeatures::to_vector` over chemical bonds | bond-type counts, total bond order |
| 13–14 | atom count, bond count | topology |
| 15    | `has_positions` flag | mask convention |
| 16–17 | mean and maximum bond distance (Å) | `0.0` when positions are absent — never invented |

Aggregating the categorical column is physically meaningful (the sum of
atomic numbers is the total electron count). Topology-only molecules get
descriptors too — the geometry flag flips instead of erroring, so
descriptor-based models can consume any dataset entry.

### `mi-models` — the MLP and the training pattern
- `MlpConfig { input_dim, hidden, output_dim }` — empty `hidden` is a
  linear-regression baseline; zero-width layers are rejected.
- Deterministic **Glorot-uniform** initialization from an in-crate
  SplitMix64 generator (seed parameter; no `rand` dependency; the
  sequence is pinned by tests).
- `forward` — batch size **1** (`[1, D] → [1, out]`): the tensor layer
  implements no broadcasting yet, and batch-1 keeps every operation
  same-shaped. Hidden layers apply ReLU; the output layer is linear.
- `update_with(gradients, updater)` — the per-parameter **update hook**
  with the leaf renewal cycle: the updater returns new values, the model
  applies `p ← detach(p_updated).require_grad()`. Burn consumes the
  recorded graph at `backward` (no `retain_graph`), so each step must
  start from fresh leaves; this cycle is the sanctioned pattern and is
  unit-tested for convergence. (`update_with` replaced Phase 7's
  `sgd_step` when optimization moved to `mi-training`; see
  [`training.md`](training.md).)
- `parameter_values()` / `with_parameters(config, values, device)` — the
  snapshot/rebuild pair: training runs on `Cpu::training()`, evaluation
  rebuilds the same architecture (untracked) on `Cpu::inference()`.
- The canonical training loop moved to **`mi-training::train`** in
  Phase 9: per sample, forward → **masked MSE** `Σ(p−y)²·m / Σm`
  (composed from tensor ops, so it stays differentiable; absent targets
  contribute nothing and all-masked samples are skipped *and counted*)
  → `backward` → optimizer step via `update_with`. Returns per-epoch
  host metrics (mean loss, skip counts). `predict` remains the
  evaluation-side forward here.

### `mi-evaluation` — reported metrics
Pure functions over paired observations (`f32` in, `f64` arithmetic):

- **MAE** — average |p − r|, in the property's unit (eV).
- **RMSE** — quadratic mean; weighs outliers more (RMSE ≥ MAE always).
- **R²** — fraction of reference variance explained; negative means
  worse than predicting the mean. **Undefined** (typed error) for
  constant references — never substituted with a fudged number.

Every degenerate input (length mismatch, empty data, zero-variance
references) is a typed [`MetricsError`]; the crate depends on nothing.

## The training recipe (what the tests pin down)

The integration test `crates/mi-models/tests/baseline_pipeline.rs` is
the executable recipe — and it documents three lessons the real
benchmark must follow:

1. **Standardize descriptor inputs** (z-score per column over the
   training samples). Raw columns span wildly different scales (an
   atomic-number sum reaches ~30 while one-hot flags are 0/1); feeding
   them raw into per-sample SGD killed every hidden ReLU unit in the
   first update and froze the network at a constant prediction. With
   standardized inputs the same network converges (loss 414 → 0.0004 in
   800 epochs on the fixture).
2. **Standardize targets** (z-score per slot over present values) and
   **un-scale predictions** before reporting metrics — the reported
   numbers stay in electronvolts.
3. **Seed everything**: the Glorot draw and the dataset split are seeded;
   identical seeds reproduce identical loss curves (asserted).

The overfit proof: 6 fixture molecules, network `[18, 32, 16, 3]`, 800
epochs at lr 0.01 → masked-MSE loss falls by more than ×100, train MAE
< 0.3 eV, train R² > 0.99, all deterministic under the seed.

## Running the real QM9 benchmark (procedure)

With `datasets/qm9.sdf` and `datasets/uncharacterized.txt` in place
(Phase 6's loader):

1. Load: `load_qm9_sdf(...)` → `Qm9Report` → `MolecularDataset`.
2. Split: `random_split(dataset, SplitFractions::DEFAULT, seed)`.
3. Prepare: descriptor + targets per training entry, **z-scored over the
   training subset only** (the validation/test scalers come from train —
   never fit scalers on evaluation data).
4. Train: `train_sgd` on the training subset; track `mean_loss` per
   epoch; select checkpoints on the **validation** subset (MAE).
5. Report: `mae`/`rmse`/`r²` on the test subset, per target slot, with
   the split seed and schema versions recorded for reproducibility.

The reference scale for this experiment (from the QM9 literature): a
well-tuned descriptor-based model reaches ≈ tens-of-meV MAE on HOMO/
LUMO with random splits; anything reported by this framework should
state its split, preprocessing, and hyperparameters alongside.

## Limitations and future work

1. **Batch size 1.** Per-sample SGD is simple and correct but leaves
   performance on the table; batching needs broadcasting in the tensor
   layer (planned `mi-tensor` addition) and padded graph batching for
   message-passing models.
2. ~~**SGD only**~~ — **addressed in Phase 9**: `mi-training` now
   provides SGD with momentum and Adam plus learning-rate schedules
   ([`training.md`](training.md)). Early stopping is still future work;
   the validation subset exists and checkpoint selection runs
   benchmark-side.
3. ~~**Standardization is a documented recipe, not a library API**~~ —
   **addressed in Phase 8**: `mi-featurization::scaler` provides the
   shared fitted z-score scaler.
4. **No uncertainty output** (Experiment 03 of the research plan) —
   ensembles and MC-dropout come after the first benchmark numbers.
5. **The MLP consumes only the fixed-size descriptor**; the graph
   tensors (`GraphTensors`) are ready for a message-passing model (the
   SchNet-class step), which needs index tensors and scatter/gather in
   `mi-tensor` first.
