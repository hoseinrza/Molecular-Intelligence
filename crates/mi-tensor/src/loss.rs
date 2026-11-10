//! Regression loss functions with mean reduction semantics.
use crate::{Tensor, TensorError, TensorResult};
/// Supported regression loss families.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Loss { Mse, Mae, Huber }
fn paired<const D:usize>(p:&Tensor<D>,t:&Tensor<D>)->TensorResult<(Vec<f32>,Vec<f32>)> {
 if p.shape()!=t.shape() { return Err(TensorError::ShapeMismatch { left:p.shape().dims().to_vec(), right:t.shape().dims().to_vec() }); }
 if p.numel()==0 { return Err(TensorError::EmptyTensor); } Ok((p.to_vec()?,t.to_vec()?))
}
/// Mean squared error (mean of squared elementwise differences).
pub fn mse<const D:usize>(p:&Tensor<D>,t:&Tensor<D>)->TensorResult<f32> { let(a,b)=paired(p,t)?; Ok(a.iter().zip(b).map(|(x,y)|(x-y).powi(2)).sum::<f32>()/a.len() as f32) }
/// Mean absolute error.
pub fn mae<const D:usize>(p:&Tensor<D>,t:&Tensor<D>)->TensorResult<f32> { let(a,b)=paired(p,t)?; Ok(a.iter().zip(b).map(|(x,y)|(x-y).abs()).sum::<f32>()/a.len() as f32) }
/// Smooth L1/Huber loss with positive delta and mean reduction.
pub fn huber_loss<const D:usize>(p:&Tensor<D>,t:&Tensor<D>,delta:f32)->TensorResult<f32> { if !delta.is_finite() || delta<=0. { return Err(TensorError::InvalidShape("Huber delta must be finite and positive".into())); } let(a,b)=paired(p,t)?; Ok(a.iter().zip(b).map(|(x,y)| { let e=(x-y).abs(); if e<=delta { 0.5*e*e } else { delta*(e-0.5*delta) } }).sum::<f32>()/a.len() as f32) }
