//! Burn activation functions.
use crate::Tensor;
/// ReLU activation.
pub fn relu<const D:usize>(x:&Tensor<D>)->Tensor<D> { Tensor { inner:x.inner.clone().relu(), shape:x.shape.clone() } }
/// Sigmoid activation.
pub fn sigmoid<const D:usize>(x:&Tensor<D>)->Tensor<D> { Tensor { inner:x.inner.clone().sigmoid(), shape:x.shape.clone() } }
/// Hyperbolic tangent.
pub fn tanh<const D:usize>(x:&Tensor<D>)->Tensor<D> { Tensor { inner:x.inner.clone().tanh(), shape:x.shape.clone() } }
/// Gaussian Error Linear Unit.
pub fn gelu<const D:usize>(x:&Tensor<D>)->Tensor<D> { Tensor { inner:x.inner.clone().gelu(), shape:x.shape.clone() } }
