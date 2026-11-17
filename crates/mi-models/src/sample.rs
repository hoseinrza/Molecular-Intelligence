//! Training inputs and model-side prediction.
//!
//! [`RegressionSample`] is the one-observation contract between
//! featurization and training: descriptor input, target values, presence
//! mask — all rank-1 tensors already built on the caller's device.
//! [`predict`] is the evaluation-side forward pass.

use crate::error::ModelResult;
use crate::model::Mlp;
use mi_tensor::Tensor;

/// One training observation: a descriptor input, target values, and the
/// presence mask, all rank-1 and already built on the model's device.
#[derive(Clone)]
pub struct RegressionSample {
    /// The fixed-size molecular descriptor (width = model `input_dim`).
    pub input: Tensor<1>,
    /// Target values in the fixed `[homo, lumo, gap]` layout (eV); absent
    /// slots are `0.0`.
    pub values: Tensor<1>,
    /// Presence mask matching `values`: `1.0` present, `0.0` absent.
    pub mask: Tensor<1>,
}

/// Computes predictions for one sample on the model's own device.
///
/// The input is a rank-1 descriptor vector; the result is the rank-1
/// output vector. This is an evaluation utility — it never runs
/// `backward`.
///
/// # Errors
///
/// Returns [`crate::ModelError::Tensor`] for shape or backend failures.
pub fn predict(model: &Mlp, input: &Tensor<1>) -> ModelResult<Tensor<1>> {
    let width = model.config().input_dim;
    let row = input.clone().reshape([1, width])?;
    let output = model.forward(&row)?;
    let output_width = model.config().output_dim;
    Ok(output.reshape([output_width])?)
}
