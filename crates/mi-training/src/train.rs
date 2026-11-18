//! The canonical training loop — batched.
//!
//! One masked-MSE regression loop, driven by any [`Optimizer`] under any
//! [`LrSchedule`], over batches assembled from per-sample rows with
//! `mi_tensor::operations::cat` (the cat keeps every leaf connected to the
//! autodiff graph). The recipe respects the project's values/mask target
//! convention:
//!
//! ```text
//! L = Σ (pᵢ − yᵢ)² · mᵢ  /  Σ mᵢ        (summed over the batch)
//! ```
//!
//! so absent properties contribute nothing. All-masked batches are skipped
//! (and their samples counted) via the host-side `mask_sum` carried on
//! each [`RegressionSample`] — no device round-trip is spent deciding.
//! With uniform masks (every QM9 sample) the batch loss equals the mean of
//! the per-sample losses; with mixed masks it is the presence-weighted
//! mean, which is the mathematically consistent generalization.
//!
//! `batch_size = 1` reproduces the historical per-sample loop exactly.

use crate::error::{TrainingError, TrainingResult};
use crate::optimizer::Optimizer;
use crate::schedule::LrSchedule;
use mi_models::{Mlp, RegressionSample};
use mi_tensor::operations::{cat, div, mul, sub};
use mi_tensor::{Device, Tensor};

/// One assembled training batch: stacked sample rows plus host metadata.
#[derive(Debug, Clone)]
pub struct Batch {
    /// Stacked input rows `[B, input_dim]`.
    pub input: Tensor<2>,
    /// Stacked target rows `[B, target_width]` (absent slots `0.0`).
    pub values: Tensor<2>,
    /// Stacked presence-mask rows `[B, target_width]`.
    pub mask: Tensor<2>,
    /// How many samples the batch holds.
    pub samples: usize,
    /// Host-side total presence weight (`Σm` over the batch) for the
    /// skip decision.
    pub mask_sum: f32,
}

/// Assembles `[B, width]` batches from per-sample rows.
///
/// Rows are stacked with `cat` along the batch dimension, which keeps each
/// sample's leaf connected to the autodiff graph. The final short batch
/// holds the remainder.
///
/// # Errors
///
/// Returns [`TrainingError::InvalidConfig`] for empty samples or a zero
/// batch size, and [`TrainingError::Tensor`] for stacking failures.
pub fn build_batches(
    samples: &[RegressionSample],
    batch_size: usize,
) -> TrainingResult<Vec<Batch>> {
    if samples.is_empty() {
        return Err(TrainingError::InvalidConfig(
            "cannot train on zero samples".into(),
        ));
    }
    if batch_size == 0 {
        return Err(TrainingError::InvalidConfig(
            "batch size must be positive".into(),
        ));
    }
    let mut batches = Vec::with_capacity(samples.len().div_ceil(batch_size));
    for chunk in samples.chunks(batch_size) {
        let input_rows: Vec<&Tensor<2>> = chunk.iter().map(|s| &s.input).collect();
        let value_rows: Vec<&Tensor<2>> = chunk.iter().map(|s| &s.values).collect();
        let mask_rows: Vec<&Tensor<2>> = chunk.iter().map(|s| &s.mask).collect();
        batches.push(Batch {
            input: cat(&input_rows, 0).map_err(TrainingError::from)?,
            values: cat(&value_rows, 0).map_err(TrainingError::from)?,
            mask: cat(&mask_rows, 0).map_err(TrainingError::from)?,
            samples: chunk.len(),
            mask_sum: chunk.iter().map(|s| s.mask_sum).sum(),
        });
    }
    Ok(batches)
}

/// Host-side summary of one training epoch.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EpochMetrics {
    /// Zero-based epoch index.
    pub epoch: usize,
    /// Presence-weighted mean of the per-batch masked-MSE losses over the
    /// samples that contributed (see
    /// [`EpochMetrics::skipped_samples`]).
    pub mean_loss: f32,
    /// Samples skipped because their presence mask was all zero.
    pub skipped_samples: usize,
}

/// The training run's knobs: epochs, learning rate, schedule, batch size.
#[derive(Debug, Clone)]
pub struct TrainConfig {
    /// Number of epochs (full passes over the batches in order).
    pub epochs: usize,
    /// The learning rate the schedule starts from.
    pub base_lr: f32,
    /// How the learning rate evolves across epochs.
    pub schedule: LrSchedule,
    /// Samples per assembled batch; `1` reproduces the historical
    /// per-sample loop.
    pub batch_size: usize,
}

/// Trains `model` for `config.epochs` epochs, applying `optimizer` once
/// per batch at the scheduled learning rate.
///
/// # Errors
///
/// Returns [`TrainingError`] for tensor, model, or schedule failures.
pub fn train(
    model: &mut Mlp,
    samples: &[RegressionSample],
    optimizer: &mut dyn Optimizer,
    config: &TrainConfig,
    device: &Device,
) -> TrainingResult<Vec<EpochMetrics>> {
    let batches = build_batches(samples, config.batch_size)?;
    let mut history = Vec::with_capacity(config.epochs);

    for epoch in 0..config.epochs {
        let lr = config.schedule.lr(epoch, config.base_lr)?;
        let mut total_loss = 0.0f64;
        let mut contributing = 0usize;
        let mut skipped_samples = 0usize;

        for batch in &batches {
            // A batch whose presence weight is zero carries no learnable
            // signal; skip it (counting its samples) on the host, before
            // any device work.
            if batch.mask_sum == 0.0 {
                skipped_samples += batch.samples;
                continue;
            }

            let prediction = model.forward(&batch.input).map_err(TrainingError::from)?;
            let difference = sub(&prediction, &batch.values).map_err(TrainingError::from)?;
            let squared = mul(&difference, &difference).map_err(TrainingError::from)?;
            let masked = mul(&squared, &batch.mask).map_err(TrainingError::from)?;
            let denominator =
                mi_tensor::operations::sum_all(&batch.mask).map_err(TrainingError::from)?;
            let loss = div(&mi_tensor::operations::sum_all(&masked)?, &denominator)
                .map_err(TrainingError::from)?;
            let gradients = loss.backward().map_err(TrainingError::from)?;
            optimizer.step(model, &gradients, lr, device)?;
            // The batch loss is already a mean over its presence-weighted
            // elements; weight it by the batch sample count so the epoch
            // mean keeps the per-sample semantics.
            total_loss += f64::from(loss.item()?) * batch.samples as f64;
            contributing += batch.samples;
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optimizer::{Adam, Sgd};
    use mi_models::MlpConfig;
    use mi_tensor::{Cpu, Tensor, TensorShape};

    /// A rank-two row `[1, width]` on the given device.
    fn shaped(data: Vec<f32>, device: &Device) -> Tensor<2> {
        Tensor::<2>::from_data(data, TensorShape::new(vec![1, 1]).unwrap(), device).unwrap()
    }

    /// One linear sample: `x = [1]`, target `y`, mask all present.
    fn linear_sample(device: &Device, y: f32) -> RegressionSample {
        RegressionSample {
            input: shaped(vec![1.0], device),
            values: shaped(vec![y], device),
            mask: shaped(vec![1.0], device),
            mask_sum: 1.0,
        }
    }

    fn linear_model(device: &Device, seed: u64) -> Mlp {
        Mlp::new_with_seed(
            MlpConfig {
                input_dim: 1,
                hidden: vec![],
                output_dim: 1,
            },
            device,
            seed,
        )
        .unwrap()
    }

    fn config(epochs: usize, base_lr: f32, batch_size: usize) -> TrainConfig {
        TrainConfig {
            epochs,
            base_lr,
            schedule: LrSchedule::Constant,
            batch_size,
        }
    }

    #[test]
    fn batch_one_matches_hand_computed_sgd_momentum_steps() {
        // loss = (p − y)², p = w·x + b with x = 1, y = 0
        // → ∇w = ∇b = 2p. Step 1 (lr 0.1, momentum 0.5):
        //   v₁ = ∇p₁, w₁ = w₀ − lr·v₁; step 2: v₂ = 0.5·v₁ + ∇p₂.
        let device = Cpu::training();
        let mut model = linear_model(&device, 3);
        let sample = linear_sample(&device, 0.0);
        let initial = model.parameter_values().unwrap();
        let w0 = initial[0][0];
        let b0 = initial[1][0];

        let mut optimizer = Sgd::new(0.5).unwrap();
        train(
            &mut model,
            std::slice::from_ref(&sample),
            &mut optimizer,
            &config(2, 0.1, 1),
            &device,
        )
        .unwrap();

        let p0 = w0 + b0;
        let grad0 = 2.0 * p0;
        let w1 = w0 - 0.1 * grad0;
        let b1 = b0 - 0.1 * grad0;
        let p1 = w1 + b1;
        let grad1 = 2.0 * p1;
        let v2 = 0.5 * grad0 + grad1;
        let w2 = w1 - 0.1 * v2;
        let b2 = b1 - 0.1 * v2;

        let final_values = model.parameter_values().unwrap();
        assert!(
            (final_values[0][0] - w2).abs() < 1e-5,
            "w: {} vs {w2}",
            final_values[0][0]
        );
        assert!(
            (final_values[1][0] - b2).abs() < 1e-5,
            "b: {} vs {b2}",
            final_values[1][0]
        );
    }

    #[test]
    fn batch_size_two_reduces_the_mean_of_per_sample_losses() {
        // With uniform masks the batch loss over {y=2, y=4} at prediction p
        // equals the mean of the two per-sample losses.
        let device = Cpu::training();
        let samples = vec![linear_sample(&device, 2.0), linear_sample(&device, 4.0)];
        let mut model = linear_model(&device, 8);
        let history = train(
            &mut model,
            &samples,
            &mut Sgd::new(0.0).unwrap(),
            &config(1, 0.1, 2),
            &device,
        )
        .unwrap();

        let initial = Mlp::new_with_seed(
            MlpConfig {
                input_dim: 1,
                hidden: vec![],
                output_dim: 1,
            },
            &device,
            8,
        )
        .unwrap()
        .parameter_values()
        .unwrap();
        let p0 = initial[0][0] + initial[1][0];
        let expected = ((p0 - 2.0).powi(2) + (p0 - 4.0).powi(2)) / 2.0;
        assert!(
            (f64::from(history[0].mean_loss) - f64::from(expected)).abs() < 1e-5,
            "batch loss {} vs expected {expected}",
            history[0].mean_loss
        );
    }

    #[test]
    fn adam_first_step_moves_every_parameter_by_about_lr() {
        // Adam's first step is lr · g / (|g| + ε) ≈ lr · sign(g).
        let device = Cpu::training();
        let mut model = linear_model(&device, 5);
        let sample = linear_sample(&device, 0.0);
        let initial = model.parameter_values().unwrap();
        let w0 = initial[0][0];
        let b0 = initial[1][0];

        let mut optimizer = Adam::new().unwrap();
        train(
            &mut model,
            std::slice::from_ref(&sample),
            &mut optimizer,
            &config(1, 0.01, 1),
            &device,
        )
        .unwrap();

        let final_values = model.parameter_values().unwrap();
        let expected_w = w0 - 0.01 * (2.0 * (w0 + b0)).signum();
        let expected_b = b0 - 0.01 * (2.0 * (w0 + b0)).signum();
        assert!(
            (final_values[0][0] - expected_w).abs() < 1e-4,
            "w: {} vs {expected_w}",
            final_values[0][0]
        );
        assert!(
            (final_values[1][0] - expected_b).abs() < 1e-4,
            "b: {} vs {expected_b}",
            final_values[1][0]
        );
    }

    #[test]
    fn adam_converges_faster_than_plain_sgd_on_a_quadratic() {
        let device = Cpu::training();
        let sample = linear_sample(&device, 0.0);
        let run = |optimizer: &mut dyn Optimizer| -> f32 {
            let mut model = linear_model(&device, 11);
            train(
                &mut model,
                std::slice::from_ref(&sample),
                optimizer,
                &config(40, 0.02, 1),
                &device,
            )
            .unwrap();
            model.parameter_values().unwrap()[0][0]
        };
        // Target is p = 0, so the final weight should be near zero; Adam
        // gets closer than plain SGD at the same learning rate.
        let sgd_weight = run(&mut Sgd::new(0.0).unwrap()).abs();
        let adam_weight = run(&mut Adam::new().unwrap()).abs();
        assert!(
            adam_weight < sgd_weight,
            "adam |w|={adam_weight} should beat plain sgd |w|={sgd_weight}"
        );
    }

    #[test]
    fn fully_masked_samples_are_skipped_without_device_work() {
        let device = Cpu::training();
        let mut masked = linear_sample(&device, 5.0);
        masked.mask = shaped(vec![0.0], &device);
        masked.mask_sum = 0.0;
        let mut model = linear_model(&device, 2);
        let history = train(
            &mut model,
            &[masked],
            &mut Sgd::new(0.0).unwrap(),
            &config(3, 0.1, 1),
            &device,
        )
        .unwrap();
        for metrics in history {
            assert_eq!(metrics.skipped_samples, 1);
            assert_eq!(metrics.mean_loss, 0.0, "skipped epochs carry no loss");
        }
    }

    #[test]
    fn remainder_batch_holds_the_leftover_samples() {
        let device = Cpu::training();
        let samples = vec![
            linear_sample(&device, 1.0),
            linear_sample(&device, 2.0),
            linear_sample(&device, 3.0),
            linear_sample(&device, 4.0),
            linear_sample(&device, 5.0),
        ];
        let batches = build_batches(&samples, 2).unwrap();
        assert_eq!(batches.len(), 3);
        assert_eq!(batches[0].samples, 2);
        assert_eq!(batches[1].samples, 2);
        assert_eq!(batches[2].samples, 1);
        assert_eq!(batches[2].input.shape().dims(), &[1, 1]);
        assert!((batches[2].mask_sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn invalid_batch_size_and_empty_samples_are_typed_errors() {
        let device = Cpu::training();
        let mut model = linear_model(&device, 1);
        assert!(matches!(
            train(
                &mut model,
                &[],
                &mut Sgd::new(0.0).unwrap(),
                &config(1, 0.1, 1),
                &device,
            ),
            Err(TrainingError::InvalidConfig(_))
        ));
        let sample = linear_sample(&device, 1.0);
        assert!(matches!(
            train(
                &mut model,
                &[sample],
                &mut Sgd::new(0.0).unwrap(),
                &config(1, 0.1, 0),
                &device,
            ),
            Err(TrainingError::InvalidConfig(_))
        ));
    }
}
