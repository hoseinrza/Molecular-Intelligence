//! CPU-first tensor primitives backed by Burn's Flex backend.
//!
//! The public tensor wrapper is deliberately small; Burn performs all numeric
//! work and owns autodiff (selected per-device via `Cpu::inference()` and
//! `Cpu::training()`). Tensors use a compile-time rank and runtime shape.
//! Differentiable operations (`operations`, loss functions, autodiff helpers)
//! never round-trip through the host; host-side evaluation utilities
//! (`to_vec`, `item`, scalar reductions, `dot`) are clearly separated.
mod autograd;
mod backend;
mod dtype;
mod error;
mod loss;
mod shape;
mod tensor;
mod validation;

pub mod operations;

pub use backend::Cpu;
pub use burn::tensor::{Device, Gradients};
pub use dtype::DType;
pub use error::{TensorError, TensorResult};
pub use loss::{huber_loss, mae, mse};
pub use shape::TensorShape;
pub use tensor::Tensor;
