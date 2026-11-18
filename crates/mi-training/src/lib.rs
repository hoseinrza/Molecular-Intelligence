//! Training infrastructure for Molecular Intelligence.
//!
//! This crate is the optimization layer: it drives [`mi_models::Mlp`]
//! models through the canonical masked-MSE training loop with pluggable
//! optimizers ([`Sgd`] with momentum, [`Adam`]), learning-rate schedules
//! ([`LrSchedule`]), and configurable batching.
//!
//! # The update contract
//!
//! Models expose [`Mlp::update_with`]: the model owns its parameters and
//! renews each updated parameter as a fresh autodiff leaf after every
//! step (Burn consumes the recorded graph at `backward`, so reusing it
//! would panic). Optimizer state lives **on the device** and every update
//! is composed from graph-preserving `mi-tensor` operations — the update
//! itself is explicitly non-differentiable, while forward, loss, and
//! backward stay inside Burn's autodiff, and no host round-trip happens
//! anywhere in the step. No project-owned autograd engine exists or is
//! needed.
//!
//! # Batching
//!
//! Per-sample rows are stacked into `[B, width]` batches with
//! `mi_tensor::operations::cat`, which keeps every sample's leaf connected
//! to the autodiff graph; `batch_size = 1` reproduces the historical
//! per-sample loop exactly. Skip decisions for all-masked batches use the
//! host-side `mask_sum` carried on each sample — no device work is spent
//! deciding.
//!
//! ```
//! use mi_models::{Mlp, MlpConfig, RegressionSample};
//! use mi_tensor::{Cpu, Tensor, TensorShape};
//! use mi_training::{Sgd, TrainConfig, LrSchedule, train};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let device = Cpu::training();
//! let mut model = Mlp::new(
//!     MlpConfig { input_dim: 1, hidden: vec![], output_dim: 1 },
//!     &device,
//! )?;
//! let row_shape = TensorShape::new(vec![1, 1])?;
//! let sample = RegressionSample {
//!     input: Tensor::<2>::from_data(vec![1.0], row_shape.clone(), &device)?,
//!     values: Tensor::<2>::from_data(vec![2.0], row_shape.clone(), &device)?,
//!     mask: Tensor::<2>::from_data(vec![1.0], row_shape.clone(), &device)?,
//!     mask_sum: 1.0,
//! };
//! let mut optimizer = Sgd::new(0.9)?;
//! let history = train(
//!     &mut model,
//!     &[sample],
//!     &mut optimizer,
//!     &TrainConfig {
//!         epochs: 10,
//!         base_lr: 0.1,
//!         schedule: LrSchedule::Constant,
//!         batch_size: 1,
//!     },
//!     &device,
//! )?;
//! assert_eq!(history.len(), 10);
//! assert!(history.last().unwrap().mean_loss < history.first().unwrap().mean_loss);
//! # Ok(())
//! # }
//! ```

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod error;
pub mod optimizer;
pub mod schedule;
pub mod train;

pub use error::{TrainingError, TrainingResult};
pub use optimizer::{Adam, Optimizer, Sgd};
pub use schedule::LrSchedule;
pub use train::{Batch, EpochMetrics, TrainConfig, build_batches, train};
