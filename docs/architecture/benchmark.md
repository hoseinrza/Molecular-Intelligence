# QM9 baseline benchmark (Phase 8)

> Scope: **Phase 8 — Benchmark Suite** of Molecular Intelligence
> (`mi-benchmark` binary, one `mi-featurization` addition). *Implemented*
> sections describe code that exists and ran today; numbers in the
> *results* section come from the recorded run in `benchmarks/`.

## Overview

Phase 8 executes the research plan's Experiment 01 end to end on the real
QM9 dataset — the first measured result of the project:

```text
datasets/qm9.sdf ──► mi-io::load_qm9_sdf (uncharacterized exclusions)
                 ──► random_split 80/10/10 (seeded)
                 ──► descriptor_vector (29 slots, layout v2) + Targets (eV)
                 ──► z-score scalers fitted on TRAIN ONLY
                 ──► Mlp per-sample SGD (masked MSE, leaf renewal)
                 ──► checkpoint selection by validation MAE
                 ──► test MAE / RMSE / R² per slot (eV)
```

## What Phase 8 contributes

### `mi-featurization::scaler` — the shared standardizer

`StandardScaler` (single column) and `ColumnScalers` (fixed-width rows or
pre-assembled columns) lift the z-score recipe out of the Phase 7 test
into the library, as `baseline-model.md` planned. Deviations are floored
at `1e-6` so constant columns stay well-defined. Scalers are host `f32`
metadata — no device, no tensors.

### `mi-benchmark` — the `qm9-baseline` binary

One dependency-light binary (`mi-data`, `mi-io`, `mi-featurization`,
`mi-models`, `mi-evaluation`, `mi-tensor`) that:

1. Parses `uncharacterized.txt` (first token of each table row = the
   1-based gdb9 index to exclude) and refuses to run when the file holds
   no rows — exclusions must never be silently disabled.
2. Streams `qm9.sdf` through `mi-io` and prints the `Qm9Report` numbers
   (records, exclusions, skips, title mismatches).
3. Splits 80/10/10 under the recorded seed (`SplitFractions::DEFAULT`).
4. Fits **input** column scalers over training descriptors and **target**
   scalers over present training values per slot — validation/test data
   never touches the fit.
5. Trains with `train_sgd` (masked MSE, per-sample SGD, leaf renewal),
   evaluating pooled validation MAE (eV) every epoch and snapshotting
   `parameter_values()` whenever it improves.
6. Rebuilds the best checkpoint with `Mlp::with_parameters` on
   `Cpu::inference()` and scores the test subset per slot:
   MAE, RMSE, and R² (R² reported as `null` when undefined).
7. Writes `result.json` (tooling) and `result.md` (humans) to the output
   directory, recording dataset provenance, split seed, hyperparameters,
   timings, and the per-epoch trace.

### `scripts/build_qm9_sdf.py` — the data-boundary adapter

The MoleculeNet bundle (`qm9.zip`) ships structures (`gdb9.sdf`, titles
`gdb_<n>`) and properties separately (`gdb9.sdf.csv`, Hartree). The
project's loader reads properties as SDF data fields with `gdb9_<n>`
titles — the original figshare convention. The script reproduces that
convention at the data boundary: it injects `homo`/`lumo`/`gap` (verbatim
Hartree strings) into each record, renames titles, and hard-fails on any
record/CSV-row mismatch. Unit conversion stays inside `mi-io`
(Hartree → eV via the CODATA factor) — the script never converts units.

## Running it

```bash
cargo build -p mi-benchmark --release
./target/x86_64-pc-windows-gnu/release/qm9-baseline.exe \
    --dataset datasets/qm9.sdf \
    --uncharacterized datasets/uncharacterized.txt \
    --out benchmarks/qm9-baseline \
    --epochs 60 --hidden 64,32 --lr 0.01 --seed 42
```

`--batch-size N` (Phase 10) assembles `[B, width]` batches with `cat`
and trains one optimizer step per batch; `--validate-every N` throttles
validation. The recorded batched comparison run lives in
`benchmarks/qm9-batched-adam/` (Adam lr 0.002, cosine, batch 256,
checkpoint epoch 59: test MAE 0.2518 / 0.3195 / 0.3979 eV, 30 s total
wall clock).

The dataset files are not committed (see `.gitignore`); obtain the
MoleculeNet bundle from
`https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/molnet_publish/qm9.zip`
and the exclusion list from figshare file `3195404`, then run
`scripts/build_qm9_sdf.py`.

## Results

Recorded runs live in `benchmarks/`; each `result.json`/`result.md`
carries the split, preprocessing, and hyperparameters alongside every
number, as `baseline-model.md` requires.

**Hyperparameter sweep (layout v1, 300 epochs, batch 256):** the deeper
network [128, 64, 32] with lr 0.003 won the sweep — pooled test MAE
0.2888 eV versus 0.3231 eV for the 60-epoch [64, 32] reference
(−10.6 % from tuning alone; `benchmarks/sweep-*`).

**Descriptor v2 + tuned run** (`benchmarks/qm9-v2-sweep-best/`): layout
v2 descriptors with the winning config reach test MAE
**0.1991 / 0.2612 / 0.3232 eV** (R² 0.803 / 0.922 / 0.884, checkpoint
epoch 287) in 355 s wall clock — −19 % pooled error versus the Phase 10
recorded run, entirely from richer inputs and tuning. Honest reading:
still above the ~0.15 eV RF+ECFP reference (Pereira 2017) and far above
3-D GNNs; the remaining gap is input information, which the graph-model
roadmap targets.

### Run of 2026-11-17 (qm9-baseline-v1)

- **Data**: 133,885 SDF records read, 3,054 excluded via
  `uncharacterized.txt`, 0 skipped, 0 title mismatches → 130,831 entries.
  Load 1.7 s, featurize 1.7 s.
- **Split**: 80/10/10, seed 42 → 104,664 / 13,083 / 13,084.
- **Model**: 18 → [64, 32] → 3; per-sample SGD lr 0.01; 60 epochs; init
  seed 42; masked MSE on z-scored targets (scalers from train only).
- **Checkpoint**: epoch 35 (validation MAE 0.3392 eV).
- **Test metrics** (13,084 test entries, all three slots present):

  | slot | MAE (eV) | RMSE (eV) | R² |
  | ---- | -------- | --------- | -- |
  | homo | 0.2562   | 0.3378    | 0.6757 |
  | lumo | 0.3420   | 0.4445    | 0.8771 |
  | gap  | 0.4153   | 0.5431    | 0.8179 |
  | pooled | 0.3378 | 0.4497    | — |

- **Wall time**: 590 s total (train 586 s), pure Rust on the Flex CPU
  backend, batch size 1.

**Honest reading.** These are the first numbers of the project's
baseline — a fixed 18-slot descriptor MLP with plain per-sample SGD. The
reference scale in `baseline-model.md` (tens of meV for well-tuned
descriptor models) is not matched yet, as expected: no feature
engineering beyond descriptor v1, no learning-rate schedule, no
momentum, batch size 1. The machinery — ingestion, exclusion, split,
standardization, autodiff training, checkpoint selection, per-slot
reporting — is what this benchmark proves; improving the numbers is the
work of later phases (better featurization, schedules, batching,
message-passing models).
