//! Differentiable regression loss functions with mean reduction.
//!
//! Each loss returns a rank-one Burn tensor holding the scalar loss value so
//! it can be backpropagated through with [`Tensor::backward`](crate::Tensor::backward).
//! Host-side scalar readback from the returned tensor is an evaluation step
//! ([`Tensor::item`](crate::Tensor::item)).
use crate::{Tensor, TensorError, TensorResult};

/// Validates paired prediction/target tensors.
fn paired<const D: usize>(p: &Tensor<D>, t: &Tensor<D>) -> TensorResult<()> {
    if p.shape() != t.shape() {
        return Err(TensorError::ShapeMismatch {
            left: p.shape().dims().to_vec(),
            right: t.shape().dims().to_vec(),
        });
    }
    if p.numel() == 0 {
        return Err(TensorError::EmptyTensor);
    }
    Ok(())
}

/// Mean squared error: mean of squared elementwise differences.
pub fn mse<const D: usize>(p: &Tensor<D>, t: &Tensor<D>) -> TensorResult<Tensor<1>> {
    paired(p, t)?;
    let inner = (p.inner.clone() - t.inner.clone()).powf_scalar(2.0).mean();
    let shape = crate::TensorShape::new(vec![1])?;
    Ok(Tensor { inner, shape })
}

/// Mean absolute error: mean of absolute elementwise differences.
pub fn mae<const D: usize>(p: &Tensor<D>, t: &Tensor<D>) -> TensorResult<Tensor<1>> {
    paired(p, t)?;
    let inner = (p.inner.clone() - t.inner.clone()).abs().mean();
    let shape = crate::TensorShape::new(vec![1])?;
    Ok(Tensor { inner, shape })
}

/// Huber (Smooth L1) loss with positive `delta` and mean reduction.
///
/// Per element, with `d = |p - t|`: quadratic branch `0.5 * d^2` when
/// `d <= delta`, linear branch `delta * (d - 0.5 * delta)` otherwise,
/// expressed as `0.5 * d^2 - 0.5 * max(d - delta, 0)^2` to stay differentiable.
pub fn huber_loss<const D: usize>(
    p: &Tensor<D>,
    t: &Tensor<D>,
    delta: f32,
) -> TensorResult<Tensor<1>> {
    if !delta.is_finite() || delta <= 0.0 {
        return Err(TensorError::InvalidShape(
            "Huber delta must be finite and positive".into(),
        ));
    }
    paired(p, t)?;
    let d = (p.inner.clone() - t.inner.clone()).abs();
    let excess = (d.clone() - delta).clamp_min(0.0);
    let inner = (d.powf_scalar(2.0) * 0.5 - excess.powf_scalar(2.0) * 0.5).mean();
    let shape = crate::TensorShape::new(vec![1])?;
    Ok(Tensor { inner, shape })
}
