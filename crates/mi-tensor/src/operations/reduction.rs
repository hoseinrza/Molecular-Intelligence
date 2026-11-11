//! Reductions, in two separated flavors.
//!
//! - [`sum_all`] and [`mean_all`] are differentiable tensor operations: they
//!   return rank-one Burn tensors and extend the autodiff computation graph.
//! - [`sum`], [`mean`], [`min`], and [`max`] are host-side evaluation
//!   utilities: they read an `f32` scalar back from the device and are
//!   intended for evaluation, inspection, and tests.
use crate::{Tensor, TensorError, TensorResult, TensorShape};

/// Sum of all elements, as a differentiable rank-one tensor.
pub fn sum_all<const D: usize>(x: &Tensor<D>) -> TensorResult<Tensor<1>> {
    let inner = x.inner.clone().sum();
    let shape = TensorShape::new(vec![1])?;
    Ok(Tensor { inner, shape })
}

/// Mean of all elements, as a differentiable rank-one tensor.
pub fn mean_all<const D: usize>(x: &Tensor<D>) -> TensorResult<Tensor<1>> {
    if x.numel() == 0 {
        return Err(TensorError::EmptyTensor);
    }
    let inner = x.inner.clone().mean();
    let shape = TensorShape::new(vec![1])?;
    Ok(Tensor { inner, shape })
}

/// Sum of all elements, as a host scalar (evaluation utility).
pub fn sum<const D: usize>(x: &Tensor<D>) -> TensorResult<f32> {
    x.inner
        .clone()
        .sum()
        .try_into_scalar::<f32>()
        .map_err(|e| TensorError::Readback(e.to_string()))
}

/// Mean of all elements, as a host scalar (evaluation utility).
pub fn mean<const D: usize>(x: &Tensor<D>) -> TensorResult<f32> {
    if x.numel() == 0 {
        return Err(TensorError::EmptyTensor);
    }
    x.inner
        .clone()
        .mean()
        .try_into_scalar::<f32>()
        .map_err(|e| TensorError::Readback(e.to_string()))
}

/// Minimum element, as a host scalar (evaluation utility).
pub fn min<const D: usize>(x: &Tensor<D>) -> TensorResult<f32> {
    if x.numel() == 0 {
        return Err(TensorError::EmptyTensor);
    }
    Ok(x.to_vec()?.into_iter().fold(f32::INFINITY, f32::min))
}

/// Maximum element, as a host scalar (evaluation utility).
pub fn max<const D: usize>(x: &Tensor<D>) -> TensorResult<f32> {
    if x.numel() == 0 {
        return Err(TensorError::EmptyTensor);
    }
    Ok(x.to_vec()?.into_iter().fold(f32::NEG_INFINITY, f32::max))
}
