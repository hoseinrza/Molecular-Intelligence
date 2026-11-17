# Training infrastructure (Phase 9)

> Scope: **Phase 9 — Training Infrastructure** of Molecular
> Intelligence (`mi-training`). *Implemented* sections describe code
> that exists and is tested today; numbers in *results* come from the
> recorded runs in `benchmarks/`.

## Overview

Phase 9 lifts the training recipe out of `mi-models` into its own
architecture layer, as `overview.md` always planned: `mi-training`
depends on `mi-models` and `mi-tensor` (the sanctioned downward
direction), owns the canonical masked-MSE loop, and adds the optimizer
and schedule machinery the baseline deferred.

```text
RegressionSample (values + mask) ──► train(model, samples, optimizer, config, device)
                                        │  per epoch: lr = schedule.lr(epoch, base)
                                        │  per sample: forward → masked MSE → backward
                                        │             → optimizer.step (host math)
                                        │             → Mlp::update_with (leaf renewal)
                                        ▼
                                  EpochMetrics per epoch
```

## The update contract

`mi-models` owns the parameters; `mi-training` owns the update rule.

- [`Mlp::update_with`] replaces the Phase 7 `sgd_step`: the updater
  closure receives the **stable parameter index** (weight then bias per
  layer, matching `parameter_values`), the current parameter, and its
  gradient, and returns updated values. The model renews every updated
  parameter as a fresh autodiff leaf (`detach` + `require_grad`) — the
  sanctioned pattern under Burn's consumed-graph backward contract.
- Optimizers compute updates on the **host** from the gradient readback.
  The update step is explicitly non-differentiable; forward, loss, and
  backward stay inside Burn's autodiff. No project-owned autograd engine
  exists or is planned.
- Optimizer state is keyed by the same stable index, so a parameter that
  was absent from a gradient container keeps both its values and its
  state slot aligned.

## Implemented optimizers

- **`Sgd`** — plain SGD (`momentum = 0`, the Phase 7/8 recipe) or classic
  momentum: `v ← μ·v + ∇p`, `p ← p − lr·v`. Momentum validated to `[0, 1]`.
- **`Adam`** — Kingma & Ba (2015) with bias-corrected moments
  (β₁ = 0.9, β₂ = 0.999, ε = 1e-8 by default, all configurable).
  Unit-tested against hand-computed two-step trajectories (SGD+momentum)
  and the first-step `lr·sign(g)` property (Adam), plus a convergence
  comparison showing Adam beats plain SGD at equal learning rate.

## Implemented schedules

- **`Constant`** — the base rate every epoch.
- **`Exponential { decay }`** — `base · decay^epoch`, decay ∈ (0, 1].
- **`Cosine { total_epochs }`** — annealing from base to zero, clamped
  at the horizon.

Every schedule rejects a non-positive or non-finite base rate with a
typed `TrainingError::InvalidConfig`.

## The canonical loop

`train(model, samples, optimizer, config, device)` runs `config.epochs`
passes over the samples in order. Per epoch it derives the learning rate
from the schedule; per sample it composes the masked MSE
`Σ(p−y)²·m / Σm` from graph-preserving tensor ops, skips (and counts)
samples whose presence mask is all zero, runs `backward`, and applies
the optimizer. It returns one `EpochMetrics` per epoch (mean loss,
skip counts). Batch size remains **1** — batching is a documented
`mi-tensor` future step, not an optimizer concern.

**API movement, documented:** `train_sgd`, `EpochMetrics`, and
`RegressionSample`/`predict` previously lived in `mi-models::training`.
Phase 9 moves the training loop into `mi-training` (its architectural
home); `RegressionSample` and `predict` stay in `mi-models` (model-side
types). The benchmark and the Phase 7 pipeline test were updated to the
new imports; the Phase 8 recorded run predates the move and used the old
API.

## Results: Adam + cosine on the full QM9 benchmark

Direct comparison on identical data, split (seed 42), architecture
(18 → [64, 32] → 3), and 60 epochs; checkpoint selected on validation
MAE in both runs:

| slot | SGD constant (Phase 8) | Adam + cosine (Phase 9) |
| ---- | ---------------------- | ----------------------- |
| homo MAE | 0.2562 eV | **0.2534 eV** |
| lumo MAE | 0.3420 eV | **0.3213 eV** |
| gap MAE  | 0.4153 eV | **0.3987 eV** |
| pooled MAE | 0.3378 eV | **0.3245 eV** |
| R² (homo / lumo / gap) | 0.676 / 0.877 / 0.818 | **0.678 / 0.887 / 0.831** |

The Adam run (`benchmarks/qm9-adam-cosine`: Adam lr 0.002, β₁ 0.9,
β₂ 0.999, cosine over 60 epochs, checkpoint epoch 58) improves every
slot and every R². **Honest reading:** the improvement is real but
modest, and the run has two visible headroom signals. First, validation
MAE was *still falling* at the horizon (0.3277 eV at epoch 59), so a
longer schedule or a larger learning rate should go further. Second,
Adam's per-step host math makes it ~3.4× slower per epoch than plain
SGD here (34 s/epoch versus 10 s/epoch on 104,664 samples at batch
size 1) — device-side optimizer kernels become worthwhile before
optimizer quality is the bottleneck.

## Limitations and future work

1. Batch size 1 (needs broadcasting + batched ops in `mi-tensor`).
2. No early stopping; checkpoint selection stays benchmark-side.
3. Host-side optimizer math is negligible for MLP-scale parameters but
   will want device-side kernels for message-passing models.
4. No gradient clipping (a natural `Optimizer` extension if needed).
