//! Optimizers over the leaf-renewal parameter cycle.
//!
//! The optimizers here compute updates on the **host** from the gradient
//! readback — the parameter update itself is an explicitly non-
//! differentiable step (the model detaches and renews every updated
//! parameter as a fresh autodiff leaf through [`Mlp::update_with`]), so
//! no computation graph crosses this boundary. Differentiable work
//! (forward, loss, backward) stays entirely inside Burn.
//!
//! State (velocity, Adam moments) is keyed by the stable parameter index
//! the model hands to [`Mlp::update_with`] — weight then bias per layer,
//! matching [`Mlp::parameter_values`].

use crate::error::{TrainingError, TrainingResult};
use mi_models::Mlp;
use mi_tensor::{Device, Gradients, Tensor};

/// An update rule applied once per training step.
pub trait Optimizer {
    /// Updates every parameter present in `gradients` at learning rate
    /// `lr`, keeping optimizer state across calls.
    ///
    /// # Errors
    ///
    /// Returns [`TrainingError`] for readback or update failures.
    fn step(
        &mut self,
        model: &mut Mlp,
        gradients: &Gradients,
        lr: f32,
        device: &Device,
    ) -> TrainingResult<()>;

    /// A stable, report-friendly identifier of the optimizer and its
    /// hyperparameters.
    fn name(&self) -> String;
}

/// Stochastic gradient descent with classic momentum.
///
/// `v ← momentum·v + ∇p`, then `p ← p − lr·v`. With `momentum = 0` this is
/// plain per-sample SGD (the Phase 7/8 recipe).
#[derive(Debug, Clone, PartialEq)]
pub struct Sgd {
    momentum: f32,
    velocities: Vec<Vec<f32>>,
}

impl Sgd {
    /// Creates the optimizer.
    ///
    /// # Errors
    ///
    /// Returns [`TrainingError::InvalidConfig`] for momentum outside
    /// `[0, 1]`.
    pub fn new(momentum: f32) -> TrainingResult<Self> {
        if !momentum.is_finite() || !(0.0..=1.0).contains(&momentum) {
            return Err(TrainingError::InvalidConfig(format!(
                "momentum must be within [0, 1], got {momentum}"
            )));
        }
        Ok(Self {
            momentum,
            velocities: Vec::new(),
        })
    }

    /// The configured momentum.
    pub const fn momentum(&self) -> f32 {
        self.momentum
    }
}

impl Optimizer for Sgd {
    fn step(
        &mut self,
        model: &mut Mlp,
        gradients: &Gradients,
        lr: f32,
        device: &Device,
    ) -> TrainingResult<()> {
        let velocities = &mut self.velocities;
        let momentum = self.momentum;
        model
            .update_with(
                gradients,
                |index: usize, parameter: &Tensor<2>, gradient: &Tensor<2>| {
                    if index >= velocities.len() {
                        velocities.resize(index + 1, Vec::new());
                    }
                    let velocity = &mut velocities[index];
                    let values = parameter.to_vec()?;
                    let deltas = gradient.to_vec()?;
                    if velocity.len() != values.len() {
                        *velocity = vec![0.0; values.len()];
                    }
                    let updated: Vec<f32> = if momentum == 0.0 {
                        values
                            .iter()
                            .zip(&deltas)
                            .map(|(value, delta)| value - lr * delta)
                            .collect()
                    } else {
                        for (velocity, delta) in velocity.iter_mut().zip(&deltas) {
                            *velocity = momentum * *velocity + delta;
                        }
                        values
                            .iter()
                            .zip(velocity.iter())
                            .map(|(value, velocity)| value - lr * velocity)
                            .collect()
                    };
                    Tensor::from_data(updated, parameter.shape().clone(), device)
                },
            )
            .map_err(TrainingError::from)
    }

    fn name(&self) -> String {
        format!("sgd(momentum={})", self.momentum)
    }
}

/// Adam (Kingma & Ba, 2015) with bias-corrected first and second moments.
#[derive(Debug, Clone, PartialEq)]
pub struct Adam {
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    first_moment: Vec<Vec<f32>>,
    second_moment: Vec<Vec<f32>>,
    timestep: u64,
}

impl Adam {
    /// Creates Adam with the standard defaults (β₁ = 0.9, β₂ = 0.999,
    /// ε = 1e-8).
    pub fn new() -> TrainingResult<Self> {
        Self::with_hyperparameters(0.9, 0.999, 1e-8)
    }

    /// Creates Adam with explicit hyperparameters.
    ///
    /// # Errors
    ///
    /// Returns [`TrainingError::InvalidConfig`] for betas outside `(0, 1)`
    /// or a non-positive epsilon.
    pub fn with_hyperparameters(beta1: f32, beta2: f32, epsilon: f32) -> TrainingResult<Self> {
        for (name, beta) in [("beta1", beta1), ("beta2", beta2)] {
            if !beta.is_finite() || !(0.0..1.0).contains(&beta) {
                return Err(TrainingError::InvalidConfig(format!(
                    "{name} must be within (0, 1), got {beta}"
                )));
            }
        }
        if !epsilon.is_finite() || epsilon <= 0.0 {
            return Err(TrainingError::InvalidConfig(format!(
                "epsilon must be finite and positive, got {epsilon}"
            )));
        }
        Ok(Self {
            beta1,
            beta2,
            epsilon,
            first_moment: Vec::new(),
            second_moment: Vec::new(),
            timestep: 0,
        })
    }

    /// The first-moment decay rate.
    pub const fn beta1(&self) -> f32 {
        self.beta1
    }

    /// The second-moment decay rate.
    pub const fn beta2(&self) -> f32 {
        self.beta2
    }

    /// The numerical-stability constant.
    pub const fn epsilon(&self) -> f32 {
        self.epsilon
    }
}

impl Default for Adam {
    /// The standard defaults; creation cannot fail for these values.
    fn default() -> Self {
        Self::with_hyperparameters(0.9, 0.999, 1e-8)
            .expect("default Adam hyperparameters are valid")
    }
}

impl Optimizer for Adam {
    fn step(
        &mut self,
        model: &mut Mlp,
        gradients: &Gradients,
        lr: f32,
        device: &Device,
    ) -> TrainingResult<()> {
        self.timestep += 1;
        let timestep = self.timestep;
        let bias1 = 1.0 - self.beta1.powi(timestep as i32);
        let bias2 = 1.0 - self.beta2.powi(timestep as i32);
        let (beta1, beta2, epsilon) = (self.beta1, self.beta2, self.epsilon);
        let first = &mut self.first_moment;
        let second = &mut self.second_moment;
        model
            .update_with(
                gradients,
                |index: usize, parameter: &Tensor<2>, gradient: &Tensor<2>| {
                    if index >= first.len() {
                        first.resize(index + 1, Vec::new());
                        second.resize(index + 1, Vec::new());
                    }
                    let m = &mut first[index];
                    let v = &mut second[index];
                    let values = parameter.to_vec()?;
                    let deltas = gradient.to_vec()?;
                    if m.len() != values.len() {
                        *m = vec![0.0; values.len()];
                        *v = vec![0.0; values.len()];
                    }
                    for i in 0..values.len() {
                        m[i] = beta1 * m[i] + (1.0 - beta1) * deltas[i];
                        v[i] = beta2 * v[i] + (1.0 - beta2) * deltas[i] * deltas[i];
                    }
                    let updated: Vec<f32> = values
                        .iter()
                        .zip(m.iter())
                        .zip(v.iter())
                        .map(|((value, m), v)| {
                            let m_hat = m / bias1;
                            let v_hat = v / bias2;
                            value - lr * m_hat / (v_hat.sqrt() + epsilon)
                        })
                        .collect();
                    Tensor::from_data(updated, parameter.shape().clone(), device)
                },
            )
            .map_err(TrainingError::from)
    }

    fn name(&self) -> String {
        format!(
            "adam(beta1={}, beta2={}, epsilon={})",
            self.beta1, self.beta2, self.epsilon
        )
    }
}
