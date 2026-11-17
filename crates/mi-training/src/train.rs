//! The canonical training loop.
//!
//! One masked-MSE regression loop, driven by any [`Optimizer`] under any
//! [`LrSchedule`]. The recipe respects the project's values/mask target
//! convention:
//!
//! ```text
//! L = Σ (pᵢ − yᵢ)² · mᵢ  /  Σ mᵢ
//! ```
//!
//! so absent properties contribute nothing, and a sample with no present
//! targets at all is skipped (counted, never averaged as zero). Samples
//! must be built on the model's device.

use crate::error::{TrainingError, TrainingResult};
use crate::optimizer::Optimizer;
use crate::schedule::LrSchedule;
use mi_models::{Mlp, RegressionSample};
use mi_tensor::Device;
use mi_tensor::operations::{div, mul, sub};

/// Host-side summary of one training epoch.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EpochMetrics {
    /// Zero-based epoch index.
    pub epoch: usize,
    /// Mean of the per-sample masked-MSE losses over the samples that
    /// contributed (see [`EpochMetrics::skipped_samples`]).
    pub mean_loss: f32,
    /// Samples skipped because their presence mask was all zero.
    pub skipped_samples: usize,
}

/// The training run's knobs: epochs, base learning rate, and schedule.
#[derive(Debug, Clone)]
pub struct TrainConfig {
    /// Number of epochs (full passes over the samples in order).
    pub epochs: usize,
    /// The learning rate the schedule starts from.
    pub base_lr: f32,
    /// How the learning rate evolves across epochs.
    pub schedule: LrSchedule,
}

/// Trains `model` for `config.epochs` epochs over `samples` in order,
/// applying `optimizer` once per sample at the scheduled learning rate.
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
    if samples.is_empty() {
        return Err(TrainingError::InvalidConfig(
            "cannot train on zero samples".into(),
        ));
    }
    let input_width = model.config().input_dim;
    let target_width = samples
        .first()
        .map(|sample| sample.values.shape().dims()[0])
        .unwrap_or(0);
    let mut history = Vec::with_capacity(config.epochs);

    for epoch in 0..config.epochs {
        let lr = config.schedule.lr(epoch, config.base_lr)?;
        let mut total_loss = 0.0f64;
        let mut contributing = 0usize;
        let mut skipped_samples = 0usize;

        for sample in samples {
            let input = sample
                .input
                .clone()
                .reshape([1, input_width])
                .map_err(TrainingError::from)?;
            let values = sample
                .values
                .clone()
                .reshape([1, target_width])
                .map_err(TrainingError::from)?;
            let mask = sample
                .mask
                .clone()
                .reshape([1, target_width])
                .map_err(TrainingError::from)?;

            let prediction = model.forward(&input).map_err(TrainingError::from)?;
            let difference = sub(&prediction, &values).map_err(TrainingError::from)?;
            let squared = mul(&difference, &difference).map_err(TrainingError::from)?;
            let masked = mul(&squared, &mask).map_err(TrainingError::from)?;
            let denominator = mi_tensor::operations::sum_all(&mask).map_err(TrainingError::from)?;

            // A sample with no present targets is skipped, not averaged in.
            if denominator.item()? == 0.0 {
                skipped_samples += 1;
                continue;
            }

            let loss = div(&mi_tensor::operations::sum_all(&masked)?, &denominator)
                .map_err(TrainingError::from)?;
            let gradients = loss.backward().map_err(TrainingError::from)?;
            optimizer.step(model, &gradients, lr, device)?;
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optimizer::{Adam, Sgd};
    use crate::schedule::LrSchedule;
    use mi_models::MlpConfig;
    use mi_tensor::{Cpu, Tensor, TensorShape};

    /// A rank-one tensor of width 1 on the given device.
    fn shaped(data: Vec<f32>, device: &Device) -> Tensor<1> {
        Tensor::<1>::from_data(data, TensorShape::new(vec![1]).unwrap(), device).unwrap()
    }

    /// One linear sample: `x = 1`, target `y`, mask all present.
    fn linear_sample(device: &Device, y: f32) -> RegressionSample {
        RegressionSample {
            input: shaped(vec![1.0], device),
            values: shaped(vec![y], device),
            mask: shaped(vec![1.0], device),
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

    #[test]
    fn sgd_with_momentum_matches_hand_computed_steps() {
        // loss = (p − y)², p = w·x + b with x = 1, y = 0
        // → ∇w = ∇b = 2p. Step 1 (lr 0.1, momentum 0.5):
        //   v₁ = ∇p₁, w₁ = w₀ − lr·v₁
        // Step 2: v₂ = 0.5·v₁ + ∇p₂, w₂ = w₁ − lr·v₂.
        let device = Cpu::training();
        let mut model = linear_model(&device, 3);
        let sample = linear_sample(&device, 0.0);
        let initial = model.parameter_values().unwrap(); // [w, b]
        let w0 = initial[0][0];
        let b0 = initial[1][0];

        let mut optimizer = Sgd::new(0.5).unwrap();
        train(
            &mut model,
            std::slice::from_ref(&sample),
            &mut optimizer,
            &TrainConfig {
                epochs: 2,
                base_lr: 0.1,
                schedule: LrSchedule::Constant,
            },
            &device,
        )
        .unwrap();

        let p0 = w0 + b0;
        let grad0 = 2.0 * p0 / 1.0; // masked-MSE divides by Σm = 1
        let w1 = w0 - 0.1 * grad0;
        let b1 = b0 - 0.1 * grad0;
        let p1 = w1 + b1;
        let grad1 = 2.0 * p1;
        let v2_w = 0.5 * grad0 + grad1;
        let v2_b = 0.5 * grad0 + grad1;
        let w2 = w1 - 0.1 * v2_w;
        let b2 = b1 - 0.1 * v2_b;

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
            &[sample],
            &mut optimizer,
            &TrainConfig {
                epochs: 1,
                base_lr: 0.01,
                schedule: LrSchedule::Constant,
            },
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
                &TrainConfig {
                    epochs: 40,
                    base_lr: 0.02,
                    schedule: LrSchedule::Constant,
                },
                &device,
            )
            .unwrap();
            model.parameter_values().unwrap()[0][0]
        };
        // Target is p = 0, so the final weight should be near zero; Adam
        // gets closer than plain SGD at the same learning rate.
        let sgd_weight = (run(&mut Sgd::new(0.0).unwrap()) - 0.0).abs();
        let adam_weight = (run(&mut Adam::new().unwrap()) - 0.0).abs();
        assert!(
            adam_weight < sgd_weight,
            "adam |w|={adam_weight} should beat plain sgd |w|={sgd_weight}"
        );
    }

    #[test]
    fn fully_masked_samples_are_skipped_and_counted() {
        let device = Cpu::training();
        let mut masked = linear_sample(&device, 5.0);
        masked.mask = shaped(vec![0.0], &device);
        let mut model = linear_model(&device, 2);
        let history = train(
            &mut model,
            &[masked],
            &mut Sgd::new(0.0).unwrap(),
            &TrainConfig {
                epochs: 3,
                base_lr: 0.1,
                schedule: LrSchedule::Constant,
            },
            &device,
        )
        .unwrap();
        for metrics in history {
            assert_eq!(metrics.skipped_samples, 1);
            assert_eq!(metrics.mean_loss, 0.0, "skipped epochs carry no loss");
        }
    }

    #[test]
    fn empty_sample_list_is_a_typed_error() {
        let device = Cpu::training();
        let mut model = linear_model(&device, 1);
        let result = train(
            &mut model,
            &[],
            &mut Sgd::new(0.0).unwrap(),
            &TrainConfig {
                epochs: 1,
                base_lr: 0.1,
                schedule: LrSchedule::Constant,
            },
            &device,
        );
        assert!(matches!(result, Err(TrainingError::InvalidConfig(_))));
    }
}
