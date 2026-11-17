//! Machine learning model definitions for Molecular Intelligence.
//!
//! This crate is the modeling layer: it turns the featurized numerical
//! representations of `mi-featurization` into trained predictions,
//! exercising `mi-tensor`'s autodiff. Phase 7 delivers the **MLP
//! baseline** — a fully connected network with deterministic
//! initialization, per-sample SGD, and the masked-MSE recipe that
//! respects the project's values/mask target convention.
//!
//! ```
//! use mi_models::{Mlp, MlpConfig};
//! use mi_tensor::{Cpu, Tensor, TensorShape};
//!
//! // A 2 → 3 → 1 regression model on the CPU training device.
//! let device = Cpu::training();
//! let mut model = Mlp::new(
//!     MlpConfig { input_dim: 2, hidden: vec![3], output_dim: 1 },
//!     &device,
//! )?;
//!
//! // One forward pass for a single input row.
//! let input = Tensor::<2>::from_data(
//!     vec![0.5, -0.25],
//!     TensorShape::new(vec![1, 2])?,
//!     &device,
//! )?;
//! let output = model.forward(&input)?;
//! assert_eq!(output.shape().dims(), &[1, 1]);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! # Scope
//!
//! - **Rust-native**: the model trains on Burn's Flex CPU backend with no
//!   Python in the loop (the stack decision of the research roadmap).
//! - **Batch size 1**: parameters update per sample; batching needs
//!   broadcasting in the tensor layer and is a documented future step.
//! - **Model, not trainer**: optimization (SGD/momentum, Adam) and
//!   learning-rate schedules live in `mi-training`, which drives this
//!   crate's models through [`Mlp::update_with`] — the leaf-renewal update
//!   hook that respects Burn's consumed-graph backward contract.
//!
//! Scientific evaluation lives in `mi-evaluation` (MAE/RMSE/R²); this
//! crate reports only training losses.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod error;
pub mod model;
pub mod sample;

pub use error::{ModelError, ModelResult};
pub use model::{Mlp, MlpConfig};
pub use sample::{RegressionSample, predict};
