//! Scientific evaluation and metrics for Molecular Intelligence.
//!
//! Regression metrics for the project's property-prediction benchmarks:
//! mean absolute error, root-mean-square error, and the coefficient of
//! determination — the quantities the research plan's Experiment 01
//! reports. All metrics are computed from paired observations in `f64`
//! and every degenerate input (length mismatch, no data, zero-variance
//! references) is a typed error, never a silently substituted number.
//!
//! ```
//! use mi_evaluation::{mae, rmse, r2};
//!
//! let references = [1.0, 2.0, 3.0];
//! let predictions = [1.5, 2.5, 3.5];
//! assert!((mae(&predictions, &references)? - 0.5).abs() < 1e-12);
//! assert!((rmse(&predictions, &references)? - 0.5).abs() < 1e-12);
//! assert!((r2(&predictions, &references)? - 0.625).abs() < 1e-12);
//! # Ok::<(), mi_evaluation::MetricsError>(())
//! ```
//!
//! This crate depends on nothing: metrics are pure functions over paired
//! observations, so scientific evaluation stays reviewable in isolation
//! from models, tensors, and chemistry.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod error;
pub mod metrics;

pub use error::{MetricsError, MetricsResult};
pub use metrics::{mae, r2, rmse};
