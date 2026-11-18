//! The MLP baseline model.
//!
//! A fully connected network over `mi-tensor` tensors: Glorot-uniform
//! deterministic initialization, ReLU hidden activations, linear output.
//! Training runs **one sample at a time** (batch size 1) — a documented
//! Phase 7 constraint that keeps every operation same-shaped, since the
//! tensor layer deliberately implements no broadcasting yet.
//!
//! # The parameter lifecycle
//!
//! Parameters live on one device (the caller picks `Cpu::training()` for
//! gradient work). Each SGD step renews every parameter as a **fresh
//! autodiff leaf**: `p ← detach(p − lr·∇p).require_grad()`. Burn consumes
//! the recorded graph at `backward`, so reusing a consumed tensor would
//! panic; the leaf-renewal cycle is the sanctioned way to keep training
//! steps independent. Fresh leaves are also why no `retain_graph` exists.

use crate::error::{ModelError, ModelResult};
use mi_tensor::operations::{add_bias, matmul, relu};
use mi_tensor::{Device, Gradients, Tensor, TensorShape};

/// Layer dimensions of an MLP: `[input, hidden…, output]`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MlpConfig {
    /// Width of the input vector (e.g. the descriptor width).
    pub input_dim: usize,
    /// Widths of the hidden layers in order; empty means a single linear
    /// layer (a linear regression baseline).
    pub hidden: Vec<usize>,
    /// Width of the output vector (3 for HOMO/LUMO/gap).
    pub output_dim: usize,
}

impl MlpConfig {
    /// Validates that every layer width is at least 1.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError::InvalidConfig`] when any width is zero.
    pub fn validate(&self) -> ModelResult<()> {
        let widths = [self.input_dim]
            .into_iter()
            .chain(self.hidden.iter().copied())
            .chain([self.output_dim]);
        for (position, width) in widths.enumerate() {
            if width == 0 {
                return Err(ModelError::InvalidConfig {
                    reason: format!("layer {position} has width 0"),
                });
            }
        }
        Ok(())
    }

    /// The full layer-width chain, input to output.
    fn widths(&self) -> Vec<usize> {
        let mut widths = Vec::with_capacity(self.hidden.len() + 2);
        widths.push(self.input_dim);
        widths.extend(self.hidden.iter().copied());
        widths.push(self.output_dim);
        widths
    }

    /// The number of parameter tensors: one weight and one bias per layer.
    fn parameter_count(&self) -> usize {
        2 * (self.hidden.len() + 1)
    }
}

/// A trained-or-training MLP baseline.
#[derive(Clone)]
pub struct Mlp {
    config: MlpConfig,
    /// One weight per layer, shape `[in, out]`, in layer order.
    weights: Vec<Tensor<2>>,
    /// One bias per layer, shape `[1, out]`, in layer order.
    biases: Vec<Tensor<2>>,
}

impl std::fmt::Debug for Mlp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Tensor values are not printable; the architecture is what matters.
        f.debug_struct("Mlp")
            .field("widths", &self.config.widths())
            .finish()
    }
}

impl Mlp {
    /// Creates an MLP with deterministic Glorot-uniform initialization
    /// (seed 0) and tracks its parameters for gradients. Use
    /// [`Mlp::new_with_seed`] for a different reproducible draw.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError::InvalidConfig`] for zero-width layers and
    /// [`ModelError::Tensor`] for backend failures.
    pub fn new(config: MlpConfig, device: &Device) -> ModelResult<Self> {
        Self::new_with_seed(config, device, 0)
    }

    /// Creates an MLP with a specific reproducible initialization seed.
    ///
    /// # Errors
    ///
    /// Same as [`Mlp::new`].
    pub fn new_with_seed(config: MlpConfig, device: &Device, seed: u64) -> ModelResult<Self> {
        config.validate()?;
        let widths = config.widths();
        let mut rng = SplitMix64::new(seed);
        let mut values = Vec::with_capacity(config.parameter_count());
        for layer in 0..widths.len() - 1 {
            let (fan_in, fan_out) = (widths[layer], widths[layer + 1]);
            // Glorot uniform: U(−b, b) with b = √(6 / (fan_in + fan_out)).
            let bound = (6.0 / (fan_in + fan_out) as f64).sqrt() as f32;
            let weight: Vec<f32> = (0..fan_in * fan_out)
                .map(|_| rng.uniform_signed(bound))
                .collect();
            values.push(weight);
            values.push(vec![0.0; fan_out]);
        }
        Self::build(config, values, device, true)
    }

    /// Rebuilds an MLP from a host-side parameter snapshot, in layer order
    /// (weight then bias per layer), without gradient tracking — the
    /// inference path and the deterministic-construction path.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError::InvalidConfig`] when `values` does not match
    /// the configuration's parameter layout, or zero-width layers are
    /// present.
    pub fn with_parameters(
        config: MlpConfig,
        values: Vec<Vec<f32>>,
        device: &Device,
    ) -> ModelResult<Self> {
        config.validate()?;
        if values.len() != config.parameter_count() {
            return Err(ModelError::InvalidConfig {
                reason: format!(
                    "expected {} parameter tensors, got {}",
                    config.parameter_count(),
                    values.len()
                ),
            });
        }
        Self::build(config, values, device, false)
    }

    /// The model configuration (architecture).
    pub fn config(&self) -> &MlpConfig {
        &self.config
    }

    /// Forward pass for one sample: `input` has shape `[1, input_dim]`.
    /// Hidden layers apply ReLU; the output layer is linear.
    ///
    /// # Errors
    ///
    /// Returns shape errors from the underlying operations when the input
    /// is not a single row of the configured width.
    pub fn forward(&self, input: &Tensor<2>) -> ModelResult<Tensor<2>> {
        let mut activation = input.clone();
        for (layer, (weight, bias)) in self.weights.iter().zip(&self.biases).enumerate() {
            let projected = matmul(&activation, weight)?;
            let biased = add_bias(&projected, bias)?;
            let is_output_layer = layer == self.weights.len() - 1;
            activation = if is_output_layer {
                biased
            } else {
                relu(&biased)
            };
        }
        Ok(activation)
    }

    /// Applies a per-parameter update using the gradients of the most
    /// recent backward pass.
    ///
    /// `updater` receives the stable parameter index (weight then bias per
    /// layer, matching [`parameter_values`](Mlp::parameter_values)), the
    /// current parameter, and its gradient, and returns the updated
    /// values. The model renews every updated parameter as a **fresh
    /// autodiff leaf** (`detach` + `require_grad`): Burn consumes the
    /// recorded graph at `backward`, so a new step must not depend on it.
    /// Parameters absent from `gradients` (they did not influence the
    /// loss) keep their values and their index slot.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError::Tensor`] for updater, readback, or renewal
    /// failures.
    pub fn update_with<F>(&mut self, gradients: &Gradients, mut updater: F) -> ModelResult<()>
    where
        F: FnMut(usize, &Tensor<2>, &Tensor<2>) -> mi_tensor::TensorResult<Tensor<2>>,
    {
        let mut index = 0usize;
        for (weight, bias) in self.weights.iter_mut().zip(&mut self.biases) {
            for parameter in [weight, bias] {
                if let Some(grad) = parameter.grad(gradients) {
                    *parameter = updater(index, parameter, &grad)?.detach().require_grad()?;
                }
                index += 1;
            }
        }
        Ok(())
    }

    /// Copies every parameter to host memory, in layer order (weight then
    /// bias per layer) — the snapshot [`Mlp::with_parameters`] consumes.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError::Tensor`] when a readback fails.
    pub fn parameter_values(&self) -> ModelResult<Vec<Vec<f32>>> {
        // Layer order, weight then bias — the exact order `build` consumes.
        let mut values = Vec::with_capacity(self.config.parameter_count());
        for (weight, bias) in self.weights.iter().zip(&self.biases) {
            values.push(weight.to_vec()?);
            values.push(bias.to_vec()?);
        }
        Ok(values)
    }

    /// Builds layers from host values; `tracked` marks parameters as
    /// gradient leaves (training) or plain tensors (inference).
    fn build(
        config: MlpConfig,
        values: Vec<Vec<f32>>,
        device: &Device,
        tracked: bool,
    ) -> ModelResult<Self> {
        let widths = config.widths();
        let mut weights = Vec::with_capacity(widths.len() - 1);
        let mut biases = Vec::with_capacity(widths.len() - 1);
        let mut consumed = 0usize;
        for layer in 0..widths.len() - 1 {
            let (fan_in, fan_out) = (widths[layer], widths[layer + 1]);
            let weight_shape = TensorShape::new(vec![fan_in, fan_out])?;
            let bias_shape = TensorShape::new(vec![1, fan_out])?;

            let weight_data = values.get(consumed).ok_or(ModelError::InvalidConfig {
                reason: format!("parameter tensor {consumed} (weight) is missing"),
            })?;
            if weight_data.len() != fan_in * fan_out {
                return Err(ModelError::InvalidConfig {
                    reason: format!(
                        "weight {consumed} holds {} values, expected {}",
                        weight_data.len(),
                        fan_in * fan_out
                    ),
                });
            }
            let mut weight = Tensor::<2>::from_data(weight_data.clone(), weight_shape, device)?;
            consumed += 1;

            let bias_data = values.get(consumed).ok_or(ModelError::InvalidConfig {
                reason: format!("parameter tensor {consumed} (bias) is missing"),
            })?;
            if bias_data.len() != fan_out {
                return Err(ModelError::InvalidConfig {
                    reason: format!(
                        "bias {consumed} holds {} values, expected {fan_out}",
                        bias_data.len()
                    ),
                });
            }
            let bias = Tensor::<2>::from_data(bias_data.clone(), bias_shape, device)?;
            consumed += 1;

            if tracked {
                weight = weight.require_grad()?;
                let bias = bias.require_grad()?;
                biases.push(bias);
            } else {
                biases.push(bias);
            }
            weights.push(weight);
        }
        Ok(Self {
            config,
            weights,
            biases,
        })
    }
}

/// SplitMix64: the deterministic initialization generator, pinned by
/// tests so a given seed always produces the same initial parameters.
struct SplitMix64(u64);

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// A uniform value in `[−bound, bound]` with 24-bit resolution.
    fn uniform_signed(&mut self, bound: f32) -> f32 {
        let unit = (self.next_u64() >> 40) as f32 / (1u32 << 24) as f32;
        (unit * 2.0 - 1.0) * bound
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mi_tensor::Cpu;

    fn device() -> Device {
        Cpu::inference()
    }

    fn training_device() -> Device {
        Cpu::training()
    }

    fn config() -> MlpConfig {
        MlpConfig {
            input_dim: 4,
            hidden: vec![3],
            output_dim: 2,
        }
    }

    #[test]
    fn zero_width_layers_are_rejected() {
        for config in [
            MlpConfig {
                input_dim: 0,
                hidden: vec![3],
                output_dim: 2,
            },
            MlpConfig {
                input_dim: 4,
                hidden: vec![0],
                output_dim: 2,
            },
            MlpConfig {
                input_dim: 4,
                hidden: vec![3],
                output_dim: 0,
            },
        ] {
            let error = Mlp::new(config, &device()).unwrap_err();
            assert!(matches!(error, ModelError::InvalidConfig { .. }));
        }
    }

    #[test]
    fn initialization_is_deterministic_per_seed() {
        let first = Mlp::new_with_seed(config(), &training_device(), 42)
            .unwrap()
            .parameter_values()
            .unwrap();
        let second = Mlp::new_with_seed(config(), &training_device(), 42)
            .unwrap()
            .parameter_values()
            .unwrap();
        assert_eq!(first, second);

        let different = Mlp::new_with_seed(config(), &training_device(), 43)
            .unwrap()
            .parameter_values()
            .unwrap();
        assert_ne!(first, different);
    }

    #[test]
    fn forward_shapes_hold_for_config_variants() {
        for config in [
            config(),
            MlpConfig {
                input_dim: 4,
                hidden: vec![],
                output_dim: 2,
            },
            MlpConfig {
                input_dim: 4,
                hidden: vec![3, 5],
                output_dim: 2,
            },
        ] {
            let model = Mlp::new_with_seed(config.clone(), &training_device(), 1).unwrap();
            // The input lives on the model's own device: samples must be
            // built where the model lives, never mixed.
            let input = Tensor::<2>::from_data(
                vec![0.5; config.input_dim],
                TensorShape::new(vec![1, config.input_dim]).unwrap(),
                &training_device(),
            )
            .unwrap();
            let output = model.forward(&input).unwrap();
            assert_eq!(output.shape().dims(), &[1, config.output_dim]);
        }
    }

    #[test]
    fn parameter_snapshot_round_trips() {
        let model = Mlp::new_with_seed(config(), &training_device(), 5).unwrap();
        let values = model.parameter_values().unwrap();
        assert_eq!(values.len(), 4); // weight+bias per layer × 2 layers

        let rebuilt = Mlp::with_parameters(config(), values.clone(), &device()).unwrap();
        assert_eq!(rebuilt.parameter_values().unwrap(), values);

        let wrong_count = vec![values[0].clone()];
        let error = Mlp::with_parameters(config(), wrong_count, &device()).unwrap_err();
        assert!(matches!(error, ModelError::InvalidConfig { .. }));
    }

    #[test]
    fn update_with_reduces_a_quadratic_loss() {
        // Minimize f(w) = Σ (w·x + b − y)² with a plain SGD closure routed
        // through the generic update hook: loss must fall.
        let device = training_device();
        let mut model = Mlp::new_with_seed(
            MlpConfig {
                input_dim: 2,
                hidden: vec![],
                output_dim: 1,
            },
            &device,
            7,
        )
        .unwrap();
        let input = Tensor::<2>::from_data(
            vec![1.0, 2.0],
            TensorShape::new(vec![1, 2]).unwrap(),
            &device,
        )
        .unwrap();
        let target =
            Tensor::<2>::from_data(vec![3.0], TensorShape::new(vec![1, 1]).unwrap(), &device)
                .unwrap();

        let learning_rate = 0.05f32;
        let mut losses = Vec::new();
        for _ in 0..200 {
            let prediction = model.forward(&input).unwrap();
            let difference = mi_tensor::operations::sub(&prediction, &target).unwrap();
            let squared = mi_tensor::operations::mul(&difference, &difference).unwrap();
            let loss = mi_tensor::operations::mean_all(&squared).unwrap();
            let value = loss.item().unwrap();
            losses.push(value);
            let gradients = loss.backward().unwrap();
            model
                .update_with(&gradients, |_index, parameter, gradient| {
                    let values = parameter.to_vec()?;
                    let deltas = gradient.to_vec()?;
                    let updated: Vec<f32> = values
                        .iter()
                        .zip(&deltas)
                        .map(|(value, delta)| value - learning_rate * delta)
                        .collect();
                    Tensor::from_data(updated, parameter.shape().clone(), &device)
                })
                .unwrap();
        }
        assert!(
            losses[losses.len() - 1] < losses[0] * 0.01,
            "loss did not converge: first {} last {}",
            losses[0],
            losses[losses.len() - 1]
        );
    }
}
