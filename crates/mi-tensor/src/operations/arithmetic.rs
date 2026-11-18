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

/// Adds a scalar to every element.
///
/// The scalar enters the autodiff graph as a constant; on a tracked tensor
/// the result is differentiable with respect to the tensor.
pub fn add_scalar<const D: usize>(tensor: &Tensor<D>, scalar: f32) -> TensorResult<Tensor<D>> {
    Ok(Tensor {
        inner: tensor.inner.clone() + scalar,
        shape: tensor.shape().clone(),
    })
}

/// Square root of every element (Float tensors).
pub fn sqrt<const D: usize>(tensor: &Tensor<D>) -> TensorResult<Tensor<D>> {
    Ok(Tensor {
        inner: tensor.inner.clone().sqrt(),
        shape: tensor.shape().clone(),
    })
}

/// Adds a broadcast row to a matrix: `x [B, C] + bias [1, C] → [B, C]`.
///
/// Burn broadcasts the single-row `bias` over every row of `x` and its
/// autodiff backward sums the gradient over the broadcast dimension, so
/// this is the batched bias-add primitive for batched training.
///
/// # Errors
///
/// Returns [`TensorError::InvalidShape`] when `bias` does not hold exactly
/// one row and [`TensorError::ShapeMismatch`] when the column counts
/// disagree.
pub fn add_bias(x: &Tensor<2>, bias: &Tensor<2>) -> TensorResult<Tensor<2>> {
    if bias.shape().dims()[0] != 1 {
        return Err(TensorError::InvalidShape(format!(
            "bias must hold exactly one row, got {}",
            bias.shape().dims()[0]
        )));
    }
    if x.shape().dims()[1] != bias.shape().dims()[1] {
        return Err(TensorError::ShapeMismatch {
            left: x.shape().dims().to_vec(),
            right: bias.shape().dims().to_vec(),
        });
    }
    Ok(Tensor {
        inner: x.inner.clone() + bias.inner.clone(),
        shape: x.shape().clone(),
    })
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

    #[test]
    fn add_scalar_and_sqrt_compute_elementwise() {
        let device = Cpu::inference();
        let tensor =
            Tensor::<1>::from_data(vec![4.0, 9.0], TensorShape::new(vec![2]).unwrap(), &device)
                .unwrap();
        let shifted = add_scalar(&tensor, 1.0).unwrap();
        assert_eq!(shifted.to_vec().unwrap(), vec![5.0, 10.0]);
        let rooted = sqrt(&shifted).unwrap();
        assert_eq!(
            rooted.to_vec().unwrap(),
            vec![5.0_f32.sqrt(), 10.0_f32.sqrt()]
        );
    }

    #[test]
    fn add_bias_broadcasts_one_row_over_a_matrix() {
        let device = Cpu::inference();
        let x = Tensor::<2>::from_data(
            vec![1.0, 2.0, 3.0, 4.0],
            TensorShape::new(vec![2, 2]).unwrap(),
            &device,
        )
        .unwrap();
        let bias = Tensor::<2>::from_data(
            vec![10.0, 20.0],
            TensorShape::new(vec![1, 2]).unwrap(),
            &device,
        )
        .unwrap();
        let summed = add_bias(&x, &bias).unwrap();
        assert_eq!(summed.shape().dims(), &[2, 2]);
        assert_eq!(summed.to_vec().unwrap(), vec![11.0, 22.0, 13.0, 24.0]);

        let wrong_rows = Tensor::<2>::from_data(
            vec![1.0, 2.0, 3.0, 4.0],
            TensorShape::new(vec![2, 2]).unwrap(),
            &device,
        )
        .unwrap();
        assert!(matches!(
            add_bias(&x, &wrong_rows),
            Err(TensorError::InvalidShape(_))
        ));
    }

    #[test]
    fn add_bias_gradient_sums_over_the_broadcast_dimension() {
        let device = Cpu::training();
        let x = Tensor::<2>::from_data(
            vec![1.0, 2.0, 3.0, 4.0],
            TensorShape::new(vec![2, 2]).unwrap(),
            &device,
        )
        .unwrap();
        let bias = Tensor::<2>::from_data(
            vec![0.5, -1.0],
            TensorShape::new(vec![1, 2]).unwrap(),
            &device,
        )
        .unwrap()
        .require_grad()
        .unwrap();
        let summed = add_bias(&x, &bias).unwrap();
        let loss = crate::operations::sum_all(&summed).unwrap();
        let gradients = loss.backward().unwrap();
        // ∂Σ(x+b)/∂b sums the ones-gradient over the two rows.
        let grad = bias.grad(&gradients).unwrap().to_vec().unwrap();
        assert_eq!(grad, vec![2.0, 2.0]);
    }
}
