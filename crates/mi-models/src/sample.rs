//! Training inputs and model-side prediction.
//!
//! [`RegressionSample`] is the one-observation contract between
//! featurization and training: a descriptor **row**, target values, and
//! presence mask — all rank-2 `[1, width]` tensors already built on the
//! caller's device, so the training loop can `cat` them straight into
//! batches without reshaping. [`predict`] is the evaluation-side forward.

use crate::error::ModelResult;
use crate::model::Mlp;
use mi_tensor::Tensor;

/// One training observation: a descriptor row, target values, and the
/// presence mask, all `[1, width]` tensors already built on the model's
/// device.
#[derive(Clone)]
pub struct RegressionSample {
    /// The fixed-size molecular descriptor row `[1, input_dim]`.
    pub input: Tensor<2>,
    /// Target values in the fixed `[homo, lumo, gap]` layout (eV); absent
    /// slots are `0.0`. Shape `[1, target_width]`.
    pub values: Tensor<2>,
    /// Presence mask matching `values`: `1.0` present, `0.0` absent.
    /// Shape `[1, target_width]`.
    pub mask: Tensor<2>,
    /// Host-side total presence weight (`Σm`) for this sample. Computed
    /// once at construction so the training loop can skip all-masked
    /// samples — and count them — without any device round-trip.
    pub mask_sum: f32,
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
