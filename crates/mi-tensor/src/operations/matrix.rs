//! Matrix operations.
//!
//! Matrix dimensions are validated before invoking Burn; the operations run
//! through Burn so the autodiff computation graph is preserved.
use crate::{Tensor, TensorError, TensorResult};

/// Matrix product of rank-two tensors.
pub fn matmul(a: &Tensor<2>, b: &Tensor<2>) -> TensorResult<Tensor<2>> {
    if a.shape().dims()[1] != b.shape().dims()[0] {
        return Err(TensorError::ShapeMismatch {
            left: a.shape().dims().to_vec(),
            right: b.shape().dims().to_vec(),
        });
    }
    let inner = a.inner.clone().matmul(b.inner.clone());
    let shape = crate::TensorShape::new(vec![a.shape().dims()[0], b.shape().dims()[1]])?;
    Ok(Tensor { inner, shape })
}

/// Transposes a rank-two tensor (swaps rows and columns).
pub fn transpose(a: &Tensor<2>) -> Tensor<2> {
    let inner = a.inner.clone().transpose();
    let shape = crate::TensorShape::new(vec![a.shape().dims()[1], a.shape().dims()[0]])
        .expect("swapping the two dimensions of a validated shape is always valid");
    Tensor { inner, shape }
}

/// Dot product of two equal-length rank-one tensors.
///
/// This is a host-side evaluation utility: the result is an `f32` scalar and
/// no gradient is retrievable from it.
pub fn dot(a: &Tensor<1>, b: &Tensor<1>) -> TensorResult<f32> {
    if a.shape() != b.shape() {
        return Err(TensorError::ShapeMismatch {
            left: a.shape().dims().to_vec(),
            right: b.shape().dims().to_vec(),
        });
    }
    let value = (a.inner.clone() * b.inner.clone())
        .sum()
        .try_into_scalar::<f32>()
        .map_err(|e| TensorError::Readback(e.to_string()))?;
    Ok(value)
}
