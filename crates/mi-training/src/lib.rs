//! Training infrastructure for Molecular Intelligence.
//!
//! This crate is the optimization layer: it drives [`mi_models::Mlp`]
//! models through the canonical masked-MSE training loop with pluggable
//! optimizers ([`Sgd`] with momentum, [`Adam`]) and learning-rate
//! schedules ([`LrSchedule`]).
//!
//! # The update contract
//!
//! Models expose [`Mlp::update_with`]: the model owns its parameters and
//! renews each updated parameter as a fresh autodiff leaf after every
//! step (Burn consumes the recorded graph at `backward`, so reusing it
//! would panic). Optimizers compute updates on the host from the
//! gradient readback — the update itself is explicitly non-
//! differentiable, while forward, loss, and backward stay inside Burn's
//! autodiff. No project-owned autograd engine exists or is needed.
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
//! let sample = RegressionSample {
//!     input: Tensor::<1>::from_data(vec![1.0], TensorShape::new(vec![1])?, &device)?,
//!     values: Tensor::<1>::from_data(vec![2.0], TensorShape::new(vec![1])?, &device)?,
//!     mask: Tensor::<1>::from_data(vec![1.0], TensorShape::new(vec![1])?, &device)?,
//! };
//! let mut optimizer = Sgd::new(0.9)?;
//! let history = train(
//!     &mut model,
//!     &[sample],
//!     &mut optimizer,
//!     &TrainConfig { epochs: 10, base_lr: 0.1, schedule: LrSchedule::Constant },
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
pub use train::{EpochMetrics, TrainConfig, train};
