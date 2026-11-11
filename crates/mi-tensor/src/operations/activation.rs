//! Activation functions.
//!
//! Thin wrappers over Burn's public `activation` module; each call extends
//! Burn's computation graph so activations are differentiable on tensors
//! created on [`Cpu::training`](crate::Cpu::training).
use crate::Tensor;
use burn::tensor::activation;

/// Rectified Linear Unit.
pub fn relu<const D: usize>(x: &Tensor<D>) -> Tensor<D> {
    Tensor {
        inner: activation::relu(x.inner.clone()),
        shape: x.shape.clone(),
    }
}

/// Logistic sigmoid.
pub fn sigmoid<const D: usize>(x: &Tensor<D>) -> Tensor<D> {
    Tensor {
        inner: activation::sigmoid(x.inner.clone()),
        shape: x.shape.clone(),
    }
}

/// Hyperbolic tangent.
pub fn tanh<const D: usize>(x: &Tensor<D>) -> Tensor<D> {
    Tensor {
        inner: activation::tanh(x.inner.clone()),
        shape: x.shape.clone(),
    }
}

/// Exact Gaussian Error Linear Unit.
pub fn gelu<const D: usize>(x: &Tensor<D>) -> Tensor<D> {
    Tensor {
        inner: activation::gelu(x.inner.clone()),
        shape: x.shape.clone(),
    }
}
