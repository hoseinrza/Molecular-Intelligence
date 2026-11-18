#!/usr/bin/env python3
"""Measures a NumPy reimplementation of the QM9 baseline training loop.

Runs on the SAME machine as the Rust benchmark, with the SAME tensor
shapes as the recorded Phase 8/9 runs (104,664 training rows, 18 -> 64
-> 32 -> 3 MLP, masked MSE), so the comparison isolates the runtime:

- batch-1 SGD (the exact Phase 8 algorithm): timed on the first `steps`
  samples and extrapolated to one epoch;
- batched Adam (batch 256, the vectorized style a NumPy/PyTorch user
  would actually write): timed over full epochs.

Values are synthetic random normals — timing of dense matmuls does not
depend on the chemical values, only the shapes. Prints a JSON result.
"""

import json
import time

import numpy as np

N, D, H1, H2, O = 104_664, 18, 64, 32, 3
STEPS_TIMED = 3_000
ADAM_EPOCHS = 3
BATCH = 256


def main() -> None:
    rng = np.random.default_rng(42)
    X = rng.standard_normal((N, D)).astype(np.float32)
    Y = rng.standard_normal((N, O)).astype(np.float32)
    M = np.ones((N, O), dtype=np.float32)

    def glorot(fan_in, fan_out):
        bound = np.sqrt(6.0 / (fan_in + fan_out))
        return rng.uniform(-bound, bound, size=(fan_in, fan_out)).astype(np.float32)

    W1, b1 = glorot(D, H1), np.zeros(H1, np.float32)
    W2, b2 = glorot(H1, H2), np.zeros(H2, np.float32)
    W3, b3 = glorot(H2, O), np.zeros(O, np.float32)

    def forward(xb):
        z1 = xb @ W1 + b1
        a1 = np.maximum(z1, 0.0)
        z2 = a1 @ W2 + b2
        a2 = np.maximum(z2, 0.0)
        z3 = a2 @ W3 + b3
        return z1, a1, z2, a2, z3

    # --- 1) batch-1 SGD: the Phase 8 algorithm, timed then extrapolated.
    lr = 0.01
    t0 = time.perf_counter()
    for i in range(STEPS_TIMED):
        xb, yb, mb = X[i : i + 1], Y[i : i + 1], M[i : i + 1]
        z1, a1, z2, a2, z3 = forward(xb)
        mask_sum = mb.sum()
        dz3 = 2.0 * (z3 - yb) * mb / mask_sum
        dW3 = a2.T @ dz3
        db3 = dz3.sum(axis=0)
        da2 = dz3 @ W3.T
        dz2 = da2 * (z2 > 0)
        dW2 = a1.T @ dz2
        db2 = dz2.sum(axis=0)
        da1 = dz2 @ W2.T
        dz1 = da1 * (z1 > 0)
        dW1 = xb.T @ dz1
        db1 = dz1.sum(axis=0)
        W3 -= lr * dW3
        b3 -= lr * db3
        W2 -= lr * dW2
        b2 -= lr * db2
        W1 -= lr * dW1
        b1 -= lr * db1
    wall = time.perf_counter() - t0
    per_sample_us = wall / STEPS_TIMED * 1e6
    sgd_epoch_s = per_sample_us * 1e-6 * N

    # --- 2) batched Adam (vectorized): full epochs, wall-clocked.
    params = {
        "W1": W1, "b1": b1, "W2": W2, "b2": b2, "W3": W3, "b3": b3,
    }
    m = {k: np.zeros_like(v) for k, v in params.items()}
    v = {k: np.zeros_like(v) for k, v in params.items()}
    beta1, beta2, eps, lr_adam = 0.9, 0.999, 1e-8, 0.002
    n_batches = N // BATCH
    adam_epoch_times = []
    t = 0
    for _ in range(ADAM_EPOCHS):
        t_start = time.perf_counter()
        order = rng.permutation(N)
        for batch in range(n_batches):
            idx = order[batch * BATCH : (batch + 1) * BATCH]
            xb, yb, mb = X[idx], Y[idx], M[idx]
            z1, a1, z2, a2, z3 = forward(xb)
            mask_sum = mb.sum()
            dz3 = 2.0 * (z3 - yb) * mb / mask_sum
            da2 = dz3 @ W3.T
            dz2 = da2 * (z2 > 0)
            da1 = dz2 @ W2.T
            dz1 = da1 * (z1 > 0)
            grads = {
                "W3": a2.T @ dz3, "b3": dz3.sum(axis=0),
                "W2": a1.T @ dz2, "b2": dz2.sum(axis=0),
                "W1": xb.T @ dz1, "b1": dz1.sum(axis=0),
            }
            t += 1
            bc1 = 1.0 - beta1**t
            bc2 = 1.0 - beta2**t
            for key, g in grads.items():
                m[key] = beta1 * m[key] + (1.0 - beta1) * g
                v[key] = beta2 * v[key] + (1.0 - beta2) * g * g
                m_hat = m[key] / bc1
                v_hat = v[key] / bc2
                params[key] -= lr_adam * m_hat / (np.sqrt(v_hat) + eps)
        adam_epoch_times.append(time.perf_counter() - t_start)

    result = {
        "machine": "local Windows CPU (same host as the Rust benchmark)",
        "shapes": {"rows": N, "input": D, "hidden": [H1, H2], "output": O},
        "numpy_version": np.__version__,
        "batch1_sgd": {
            "steps_timed": STEPS_TIMED,
            "per_sample_us": round(per_sample_us, 1),
            "epoch_seconds_extrapolated": round(sgd_epoch_s, 1),
            "throughput_samples_per_s": round(N / sgd_epoch_s),
        },
        "batched_adam": {
            "batch": BATCH,
            "epochs_timed": ADAM_EPOCHS,
            "epoch_seconds_measured": [
                round(t, 3) for t in adam_epoch_times
            ],
            "epoch_seconds_avg": round(
                sum(adam_epoch_times) / len(adam_epoch_times), 3
            ),
            "throughput_samples_per_s": round(
                N / (sum(adam_epoch_times) / len(adam_epoch_times))
            ),
        },
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
