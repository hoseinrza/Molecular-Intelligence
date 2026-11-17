//! Learning-rate schedules over epochs.
//!
//! Schedules map the epoch index to the learning rate an optimizer
//! consumes for that epoch. They are pure host functions over `f32` —
//! nothing here touches a device.

use crate::error::{TrainingError, TrainingResult};

/// How the learning rate evolves across epochs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LrSchedule {
    /// The base learning rate for every epoch.
    Constant,
    /// Exponential decay: `base · decay^epoch`, `decay ∈ (0, 1]`.
    Exponential {
        /// The per-epoch multiplicative factor.
        decay: f32,
    },
    /// Cosine annealing from the base rate to zero over `total_epochs`:
    /// `base · (1 + cos(π·epoch/total)) / 2`, clamped at the horizon.
    Cosine {
        /// The epoch at which the rate reaches zero.
        total_epochs: usize,
    },
}

impl LrSchedule {
    /// The learning rate for one epoch.
    ///
    /// # Errors
    ///
    /// Returns [`TrainingError::InvalidConfig`] when the schedule's own
    /// parameters are out of range or `base` is not positive and finite.
    pub fn lr(&self, epoch: usize, base: f32) -> TrainingResult<f32> {
        if !base.is_finite() || base <= 0.0 {
            return Err(TrainingError::InvalidConfig(format!(
                "learning rate must be finite and positive, got {base}"
            )));
        }
        match *self {
            Self::Constant => Ok(base),
            Self::Exponential { decay } => {
                if !decay.is_finite() || !(0.0..=1.0).contains(&decay) || decay == 0.0 {
                    return Err(TrainingError::InvalidConfig(format!(
                        "decay must be within (0, 1], got {decay}"
                    )));
                }
                Ok(base * decay.powi(epoch.min(i32::MAX as usize) as i32))
            }
            Self::Cosine { total_epochs } => {
                if total_epochs == 0 {
                    return Err(TrainingError::InvalidConfig(
                        "cosine schedule needs a positive total_epochs".into(),
                    ));
                }
                let progress = (epoch.min(total_epochs) as f32) / total_epochs as f32;
                Ok(base * (1.0 + (std::f32::consts::PI * progress).cos()) / 2.0)
            }
        }
    }

    /// A stable, report-friendly identifier of the schedule.
    pub fn name(&self) -> String {
        match *self {
            Self::Constant => "constant".into(),
            Self::Exponential { decay } => format!("exponential(decay={decay})"),
            Self::Cosine { total_epochs } => format!("cosine(total_epochs={total_epochs})"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constant_returns_base() {
        assert_eq!(LrSchedule::Constant.lr(0, 0.01).unwrap(), 0.01);
        assert_eq!(LrSchedule::Constant.lr(999, 0.01).unwrap(), 0.01);
        assert_eq!(LrSchedule::Constant.name(), "constant");
    }

    #[test]
    fn exponential_halves_per_epoch() {
        let schedule = LrSchedule::Exponential { decay: 0.5 };
        assert!((schedule.lr(0, 0.1).unwrap() - 0.1).abs() < 1e-9);
        assert!((schedule.lr(1, 0.1).unwrap() - 0.05).abs() < 1e-9);
        assert!((schedule.lr(2, 0.1).unwrap() - 0.025).abs() < 1e-9);
    }

    #[test]
    fn cosine_anneals_from_base_to_zero() {
        let schedule = LrSchedule::Cosine { total_epochs: 10 };
        assert!((schedule.lr(0, 0.01).unwrap() - 0.01).abs() < 1e-6);
        let mid = schedule.lr(5, 0.01).unwrap();
        assert!((mid - 0.005).abs() < 1e-6, "midpoint is base/2: {mid}");
        assert!(
            schedule.lr(10, 0.01).unwrap() < 1e-6,
            "horizon reaches zero"
        );
        // Beyond the horizon the rate stays clamped at zero.
        assert!(schedule.lr(12, 0.01).unwrap() < 1e-6);
    }

    #[test]
    fn invalid_schedule_parameters_are_typed_errors() {
        assert!(LrSchedule::Exponential { decay: 1.5 }.lr(0, 0.01).is_err());
        assert!(LrSchedule::Exponential { decay: 0.0 }.lr(0, 0.01).is_err());
        assert!(LrSchedule::Cosine { total_epochs: 0 }.lr(0, 0.01).is_err());
        for schedule in [
            LrSchedule::Constant,
            LrSchedule::Exponential { decay: 0.5 },
            LrSchedule::Cosine { total_epochs: 10 },
        ] {
            assert!(schedule.lr(0, 0.0).is_err());
            assert!(schedule.lr(0, -0.01).is_err());
            assert!(schedule.lr(0, f32::NAN).is_err());
        }
    }
}
