# Tensor backend

## Selection

`mi-tensor` pins Burn **0.22.1** and enables only its `flex` and `autodiff`
features. Burn's current documentation describes Flex as an eager, pure-Rust
CPU backend with no native dependencies; autodiff is enabled on the device
before creating training tensors. No CUDA or ROCm feature is enabled. The
workspace declares Rust 1.85, but this environment has no Rust toolchain, so
Burn's MSRV compatibility and this implementation's compilation have not been
verified locally. Resolve this before relying on the pin.

## Boundary and API

Burn owns tensor computation. `mi-tensor::Tensor<D>` is an ingress wrapper for
finite row-major `f32` data with compile-time rank `D` and validated runtime
dimensions. `TensorShape` checks nonzero dimensions and element-count overflow.
Burn `Device::flex()` selects CPU inference; `Device::flex().autodiff()` selects
the training context. `mi-data` remains independent; conversion is an explicit
copy from its numeric values into this crate.

The initial public wrapper supports Float32. Float64, Int32, and Int64 are
listed as considered dtypes but are not exposed: backend support has not been
verified for this wrapper. Division follows IEEE-754 behavior, so its result
may be non-finite even though tensor ingress rejects non-finite values.

## Operations and limitations

Burn-backed matrix multiplication, transpose, dot product, and activation
wrappers are exposed for their supported ranks. The current reductions and
loss helpers return host scalars and therefore are convenience/evaluation
operations; they are not a differentiable training-loss API. The arithmetic
and reduction helpers in this initial pass also materialize host values in
places, so they should not be used as an autodiff path. This is a limitation to
resolve before Phase 5.

Burn Flex is the only enabled backend. A later backend can be added by enabling
its Burn feature and providing an explicit device selection; no project-owned
kernel or autograd engine should be introduced.

## Verification status

Burn 0.22.1's published documentation identifies Flex as a CPU backend and
describes autodiff as device context. Local build, test, lint, benchmark, and
MSRV verification remain outstanding because neither `cargo` nor `rustc` is
available in the execution environment.
