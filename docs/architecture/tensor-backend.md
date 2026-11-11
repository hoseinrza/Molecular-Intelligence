# Tensor backend

## Selection

`mi-tensor` pins Burn **0.22.1** and enables only its `flex` and `autodiff`
features. In Burn 0.22 the backend is selected at runtime through the
`burn::tensor::Device`: `Device::flex()` is an eager, pure-Rust CPU backend
with no native dependencies, and `device.autodiff()` wraps it in Burn's
autodiff decorator (idempotent, first-order only). The wrapper exposes this
as `Cpu::inference()` and `Cpu::training()`; `Device::is_autodiff()` separates
the two contexts. No CUDA or ROCm feature is enabled.

Burn 0.22.1 declares `rust-version = "1.95"`, so the workspace MSRV was
raised from 1.85 to **1.95** to match.

## Boundary and API

Burn owns tensor computation. `mi_tensor::Tensor<D>` is an ingress wrapper
for finite row-major `f32` data with compile-time rank `D` and validated
runtime dimensions; `TensorShape` checks nonzero dimensions and
element-count overflow. `Tensor::from_data(data, shape, &device)` rejects
rank mismatches, wrong data lengths, and NaN/infinite input values.
`mi-data` remains independent; conversion is an explicit copy from its
numeric values into this crate.

The public wrapper supports Float32. Float64, Int32, and Int64 are listed as
considered dtypes but are not exposed: backend support has not been verified
for this wrapper. Division follows IEEE-754 behavior, so operation results
may be non-finite (`1/0` is `+inf`, `0/0` is NaN) even though tensor ingress
rejects non-finite input values.

## Autodiff contract

Differentiable operations never round-trip through the host: elementwise
arithmetic (`add`, `sub`, `mul`, `div`), `matmul`, `transpose`, activations
(`relu`, `sigmoid`, `tanh`, `gelu`), tensor-valued reductions (`sum_all`,
`mean_all`), and the loss functions all extend Burn's computation graph.

- Leaves are created on `Cpu::training()` and marked with
  `Tensor::require_grad()`; the wrapper returns
  `TensorError::AutodiffUnavailable` instead of panicking when autodiff is
  not enabled on the tensor's device.
- `Tensor::backward()` returns Burn's `Gradients` container and
  `Tensor::grad(&grads)` retrieves a leaf's gradient as a plain tensor.
  Burn consumes the recorded steps reachable from the output (there is no
  `retain_graph`); each fresh forward pass records a new graph, so
  finite-difference tests simply re-run the forward.
- Host-side evaluation utilities are kept separate and return `f32` scalars:
  `Tensor::to_vec`, `Tensor::item`, scalar reductions (`sum`, `mean`, `min`,
  `max`), and `dot`. They must not appear inside training computations.

## Loss functions

`mse`, `mae`, and `huber_loss` are differentiable and return a rank-one
Burn tensor (`Tensor<1>`) holding the mean-reduced loss value. Huber is
expressed as `0.5·d² − 0.5·max(d − δ, 0)²` with `d = |p − t|`, which is
piecewise-identical to the textbook definition and differentiable through
Burn's `clamp_min`. Readback of the scalar value is an evaluation step
(`Tensor::item`).

## Operations and limitations

Burn-backed matrix multiplication, transpose, dot product, and activation
wrappers are exposed for their supported ranks. Current limitations:

- Float32 only (see dtypes above).
- Backward consumes the graph; `retain_graph` semantics are not available
  (Burn does not support them either).
- Gradients are retrievable per leaf tensor; a gradient container API for
  arbitrary parameter sets belongs to the training layer (Phase 5).
- No neural-network modules, optimizers, or training engine exist yet.

Burn Flex is the only enabled backend. A later backend can be added by
enabling its Burn feature and providing an explicit device selection; no
project-owned kernel or autograd engine should be introduced.

## Verification status

Verified locally with Rust 1.99.1 (`stable-x86_64-pc-windows-gnu` target,
which bundles the MinGW linker; the machine has no MSVC Build Tools, and the
default `x86_64-pc-windows-msvc` target therefore cannot link here):
`cargo check`, `cargo test`, `cargo fmt --check`, and `cargo clippy -D
warnings` all pass on this crate. All Burn APIs used are verified against
the pinned 0.22.1 source in the local registry. The declared MSRV of 1.95
follows Burn's own requirement but was not exercised with an exact-1.95
toolchain.
