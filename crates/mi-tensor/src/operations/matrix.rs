//! Matrix operations. Matrix dimensions are validated before invoking Burn.
use crate::{Tensor, TensorError, TensorResult};
/// Matrix product of rank-two tensors.
pub fn matmul(a: &Tensor<2>, b: &Tensor<2>) -> TensorResult<Tensor<2>> {
 if a.shape().dims()[1] != b.shape().dims()[0] { return Err(TensorError::ShapeMismatch { left:a.shape().dims().to_vec(), right:b.shape().dims().to_vec() }); }
 let out=a.inner.clone().matmul(b.inner.clone());
 let sh=crate::TensorShape::new(vec![a.shape().dims()[0],b.shape().dims()[1]])?;
 Ok(Tensor { inner:out, shape:sh })
}
/// Transposes a rank-two tensor.
pub fn transpose(a: &Tensor<2>) -> Tensor<2> { Tensor { inner:a.inner.clone().transpose(), shape:crate::TensorShape::new(vec![a.shape().dims()[1],a.shape().dims()[0]]).expect("valid transpose dimensions") } }
/// Dot product of two equal-length rank-one tensors.
pub fn dot(a: &Tensor<1>, b: &Tensor<1>) -> TensorResult<f32> {
 if a.shape()!=b.shape() { return Err(TensorError::ShapeMismatch { left:a.shape().dims().to_vec(), right:b.shape().dims().to_vec() }); }
 let v=a.inner.clone().mul(b.inner.clone()).sum().into_scalar(); Ok(v)
}
