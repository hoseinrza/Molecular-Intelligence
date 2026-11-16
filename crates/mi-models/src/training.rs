//! The canonical SGD regression training loop.
//!
//! This module defines the training pattern once, so every experiment
//! (and the Phase 8 benchmark) runs the same recipe: per-sample forward,
//! **masked MSE** loss that respects the values/mask target convention,
//! backward, leaf-renewal SGD step. Samples must be built on the model's
//! device — the featurization and dataset layers hand over tensors, and
//! mixing devices is a backend error rather than a silent fallback.
//!
//! # Masked loss
//!
//! Targets follow `mi-featurization`'s `Targets` convention: a values
//! vector and a presence-mask vector of the same width. The loss is
//!
//! ```text
//! L = Σ (pᵢ − yᵢ)² · mᵢ  /  Σ mᵢ
//! ```
//!
//! so absent properties contribute nothing, and a sample with no present
//! targets at all is skipped (counted, never averaged as zero).

use crate::error::ModelResult;
use crate::model::Mlp;
use mi_tensor::operations::{div, mul, sub};
use mi_tensor::{Gradients, Tensor};

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

/// Host-side summary of one training epoch.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EpochMetrics {
    /// Zero-based epoch index.
    pub epoch: usize,
    /// Mean of the per-sample masked-MSE losses over the samples that
    /// contributed (see `skipped_samples`).
    pub mean_loss: f32,
    /// Samples skipped because their presence mask was all zero.
    pub skipped_samples: usize,
}

/// Trains `model` with per-sample SGD for `epochs` epochs over `samples`
/// in order, returning one [`EpochMetrics`] per epoch.
///
/// # Errors
///
/// Returns [`ModelError::Tensor`] for shape, readback, or backend
/// failures — for example an input width that disagrees with the model.
pub fn train_sgd(
    model: &mut Mlp,
    samples: &[RegressionSample],
    epochs: usize,
    lr: f32,
) -> ModelResult<Vec<EpochMetrics>> {
    let input_width = model.config().input_dim;
    let target_width = samples
        .first()
        .map(|sample| sample.values.shape().dims()[0])
        .unwrap_or(0);
    let mut history = Vec::with_capacity(epochs);

    for epoch in 0..epochs {
        let mut total_loss = 0.0f64;
        let mut contributing = 0usize;
        let mut skipped_samples = 0usize;

        for sample in samples {
            let input = sample.input.clone().reshape([1, input_width])?;
            let values = sample.values.clone().reshape([1, target_width])?;
            let mask = sample.mask.clone().reshape([1, target_width])?;

            let prediction = model.forward(&input)?;
            let difference = sub(&prediction, &values)?;
            let squared = mul(&difference, &difference)?;
            let masked = mul(&squared, &mask)?;
            let denominator = mi_tensor::operations::sum_all(&mask)?;

            // A sample with no present targets is skipped, not averaged in.
            if denominator.item()? == 0.0 {
                skipped_samples += 1;
                continue;
            }

            let loss = div(&mi_tensor::operations::sum_all(&masked)?, &denominator)?;
            let gradients: Gradients = loss.backward()?;
            model.sgd_step(&gradients, lr)?;
            total_loss += f64::from(loss.item()?);
            contributing += 1;
        }

        let mean_loss = if contributing > 0 {
            (total_loss / contributing as f64) as f32
        } else {
            0.0
        };
        history.push(EpochMetrics {
            epoch,
            mean_loss,
            skipped_samples,
        });
    }
    Ok(history)
}

/// Computes predictions for one sample on the model's own device.
///
/// The input is a rank-1 descriptor vector; the result is the rank-1
/// output vector. This is an evaluation utility — it never runs
/// `backward`.
///
/// # Errors
///
/// Returns [`ModelError::Tensor`] for shape or backend failures.
pub fn predict(model: &Mlp, input: &Tensor<1>) -> ModelResult<Tensor<1>> {
    let width = model.config().input_dim;
    let row = input.clone().reshape([1, width])?;
    let output = model.forward(&row)?;
    let output_width = model.config().output_dim;
    Ok(output.reshape([output_width])?)
}
