//! Optimizers over the leaf-renewal parameter cycle.
//!
//! Optimizer state (SGD velocity, Adam moments) lives **on the device as
//! tensors** and every update is composed from graph-preserving `mi-tensor`
//! operations — `mul_scalar`, `add`, `sub`, `mul`, `div`, `sqrt`,
//! `add_scalar`. There are no host round-trips in the update path: the
//! update itself is explicitly non-differentiable (the model detaches and
//! renews each updated parameter as a fresh autodiff leaf through
//! [`Mlp::update_with`]), while forward, loss, and backward stay inside
//! Burn's autodiff. No project-owned autograd engine exists or is planned.
//!
//! State is keyed by the stable parameter index the model hands to
//! [`Mlp::update_with`] — weight then bias per layer, matching
//! [`Mlp::parameter_values`].

use crate::error::{TrainingError, TrainingResult};
use mi_models::Mlp;
use mi_tensor::operations::{add, add_scalar, div, mul, mul_scalar, sqrt, sub};
use mi_tensor::{Device, Gradients, Tensor};

/// An update rule applied once per training step.
pub trait Optimizer {
    /// Updates every parameter present in `gradients` at learning rate
    /// `lr`, keeping optimizer state across calls.
    ///
    /// # Errors
    ///
    /// Returns [`TrainingError`] for tensor or update failures.
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
/// plain per-sample SGD. Velocity tensors live on the device and the whole
/// update runs as tensor operations.
#[derive(Debug, Clone)]
pub struct Sgd {
    momentum: f32,
    velocities: Vec<Tensor<2>>,
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
                        velocities
                            .resize(index + 1, Tensor::zeros(parameter.shape().clone(), device)?);
                    }
                    let updated = if momentum == 0.0 {
                        sub(parameter, &mul_scalar(gradient, lr)?)?
                    } else {
                        let velocity = &mut velocities[index];
                        *velocity = add(&mul_scalar(velocity, momentum)?, gradient)?;
                        sub(parameter, &mul_scalar(velocity, lr)?)?
                    };
                    Ok(updated)
                },
            )
            .map_err(TrainingError::from)
    }

    fn name(&self) -> String {
        format!("sgd(momentum={})", self.momentum)
    }
}

/// Adam (Kingma & Ba, 2015) with bias-corrected first and second moments.
///
/// The moment tensors live on the device; the only host-side values are
/// the per-step bias-correction scalars `1/(1 − β^t)`.
#[derive(Debug, Clone)]
pub struct Adam {
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    first_moment: Vec<Tensor<2>>,
    second_moment: Vec<Tensor<2>>,
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
                        first.resize(index + 1, Tensor::zeros(parameter.shape().clone(), device)?);
                        second.resize(index + 1, Tensor::zeros(parameter.shape().clone(), device)?);
                    }
                    let m = &mut first[index];
                    let v = &mut second[index];
                    *m = add(&mul_scalar(m, beta1)?, &mul_scalar(gradient, 1.0 - beta1)?)?;
                    *v = add(
                        &mul_scalar(v, beta2)?,
                        &mul_scalar(&mul(gradient, gradient)?, 1.0 - beta2)?,
                    )?;
                    let m_hat = mul_scalar(m, 1.0 / bias1)?;
                    let v_hat = mul_scalar(v, 1.0 / bias2)?;
                    let denominator = add_scalar(&sqrt(&v_hat)?, epsilon)?;
                    let step = div(&m_hat, &denominator)?;
                    sub(parameter, &mul_scalar(&step, lr)?)
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
