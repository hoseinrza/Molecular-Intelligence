//! Elementwise arithmetic operations.
//!
//! All operations run through Burn directly so the autodiff computation
//! graph is preserved; no host round-trip happens in these paths.
use crate::{Tensor, TensorError, TensorResult};
use burn::tensor::Tensor as BurnTensor;

/// Elementwise addition.
pub fn add<const D: usize>(a: &Tensor<D>, b: &Tensor<D>) -> TensorResult<Tensor<D>> {
    binary(a, b, |x, y| x + y)
}

/// Elementwise subtraction.
pub fn sub<const D: usize>(a: &Tensor<D>, b: &Tensor<D>) -> TensorResult<Tensor<D>> {
    binary(a, b, |x, y| x - y)
}

/// Elementwise multiplication.
pub fn mul<const D: usize>(a: &Tensor<D>, b: &Tensor<D>) -> TensorResult<Tensor<D>> {
    binary(a, b, |x, y| x * y)
}

/// Elementwise division.
///
/// Burn applies IEEE-754 semantics, so results may be non-finite (`1/0` is
/// `+inf`, `0/0` is NaN) even though tensor ingress rejects non-finite
/// input values.
pub fn div<const D: usize>(a: &Tensor<D>, b: &Tensor<D>) -> TensorResult<Tensor<D>> {
    binary(a, b, |x, y| x / y)
}

fn binary<const D: usize>(
    a: &Tensor<D>,
    b: &Tensor<D>,
    f: impl FnOnce(BurnTensor<D>, BurnTensor<D>) -> BurnTensor<D>,
) -> TensorResult<Tensor<D>> {
    if a.shape() != b.shape() {
        return Err(TensorError::ShapeMismatch {
            left: a.shape().dims().to_vec(),
            right: b.shape().dims().to_vec(),
        });
    }
    let inner = f(a.inner.clone(), b.inner.clone());
    Ok(Tensor {
        inner,
        shape: a.shape().clone(),
    })
}
