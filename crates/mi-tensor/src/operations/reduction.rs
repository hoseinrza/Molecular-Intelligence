//! Scalar reductions.
use crate::{Tensor, TensorError, TensorResult};
/// Sum all elements.
pub fn sum<const D:usize>(x:&Tensor<D>)->f32 { x.inner.clone().sum().into_scalar() }
/// Mean of all elements.
pub fn mean<const D:usize>(x:&Tensor<D>)->TensorResult<f32> { if x.numel()==0 { Err(TensorError::EmptyTensor) } else { Ok(x.inner.clone().mean().into_scalar()) } }
/// Minimum element.
pub fn min<const D:usize>(x:&Tensor<D>)->TensorResult<f32> { if x.numel()==0 { return Err(TensorError::EmptyTensor); } Ok(x.to_vec()?.into_iter().fold(f32::INFINITY,f32::min)) }
/// Maximum element.
pub fn max<const D:usize>(x:&Tensor<D>)->TensorResult<f32> { if x.numel()==0 { return Err(TensorError::EmptyTensor); } Ok(x.to_vec()?.into_iter().fold(f32::NEG_INFINITY,f32::max)) }
