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

/// Multiplies every element by a scalar.
///
/// The scalar enters the autodiff graph as a constant; on a tracked tensor
/// the result is differentiable with respect to the tensor. This is the
/// optimizer primitive (`lr · ∇p`) and the loss-weighting primitive.
pub fn mul_scalar<const D: usize>(tensor: &Tensor<D>, scalar: f32) -> TensorResult<Tensor<D>> {
    Ok(Tensor {
        inner: tensor.inner.clone().mul_scalar(scalar),
        shape: tensor.shape().clone(),
    })
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Cpu, TensorShape};

    #[test]
    fn mul_scalar_is_exact_and_shape_preserving() {
        let device = Cpu::inference();
        let tensor = Tensor::<1>::from_data(
            vec![1.0, -2.0, 4.0],
            TensorShape::new(vec![3]).unwrap(),
            &device,
        )
        .unwrap();
        let scaled = mul_scalar(&tensor, 2.5).unwrap();
        assert_eq!(scaled.shape().dims(), &[3]);
        assert_eq!(scaled.to_vec().unwrap(), vec![2.5, -5.0, 10.0]);
    }

    #[test]
    fn mul_scalar_is_differentiable() {
        let device = Cpu::training();
        // f(x) = Σ 3·x  →  ∇f = 3 for every element.
        let x = Tensor::<1>::from_data(vec![1.0, 2.0], TensorShape::new(vec![2]).unwrap(), &device)
            .unwrap()
            .require_grad()
            .unwrap();
        let scaled = mul_scalar(&x, 3.0).unwrap();
        let loss = crate::operations::sum_all(&scaled).unwrap();
        let gradients = loss.backward().unwrap();
        let grad = x.grad(&gradients).unwrap();
        assert_eq!(grad.to_vec().unwrap(), vec![3.0, 3.0]);
    }
}
