//! Elementwise arithmetic operations.
use crate::{Tensor, TensorError, TensorResult};
/// Elementwise addition.
pub fn add<const D: usize>(a: &Tensor<D>, b: &Tensor<D>) -> TensorResult<Tensor<D>> { binary(a,b, |x,y| x+y) }
/// Elementwise subtraction.
pub fn sub<const D: usize>(a: &Tensor<D>, b: &Tensor<D>) -> TensorResult<Tensor<D>> { binary(a,b, |x,y| x-y) }
/// Elementwise multiplication.
pub fn mul<const D: usize>(a: &Tensor<D>, b: &Tensor<D>) -> TensorResult<Tensor<D>> { binary(a,b, |x,y| x*y) }
/// Elementwise division; division by zero follows IEEE-754 semantics.
pub fn div<const D: usize>(a: &Tensor<D>, b: &Tensor<D>) -> TensorResult<Tensor<D>> { binary(a,b, |x,y| x/y) }
fn binary<const D: usize>(a: &Tensor<D>, b: &Tensor<D>, f: impl Fn(f32,f32)->f32) -> TensorResult<Tensor<D>> {
 if a.shape()!=b.shape() { return Err(TensorError::ShapeMismatch { left:a.shape().dims().to_vec(), right:b.shape().dims().to_vec() }); }
 let vals=a.to_vec()?.into_iter().zip(b.to_vec()?).map(|(x,y)| f(x,y)).collect(); Tensor::from_data(vals,a.shape().clone())
}
