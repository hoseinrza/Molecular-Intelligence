//! CPU-first tensor primitives backed by Burn's Flex backend.
//!
//! The public tensor wrapper is deliberately small; Burn performs all numeric
//! work and owns autodiff. Tensors use a compile-time rank and runtime shape.

mod backend;
mod dtype;
mod error;
mod loss;
mod shape;
mod tensor;
mod validation;
pub mod operations;
pub mod autograd;

pub use backend::{Cpu, InferenceDevice, TrainingDevice};
pub use dtype::DType;
pub use error::{TensorError, TensorResult};
pub use shape::TensorShape;
pub use tensor::Tensor;
pub use loss::{Loss, huber_loss, mae, mse};
