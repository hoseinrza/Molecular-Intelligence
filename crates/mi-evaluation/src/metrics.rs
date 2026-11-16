//! Regression metrics for property prediction.
//!
//! These are the quantities the project's benchmark reports (Experiment
//! 01 of the research plan): mean absolute error, root-mean-square error,
//! and the coefficient of determination. All are computed in `f64` from
//! paired `f32` observations — predictions and references must line up
//! one-to-one, and any violation is a typed error, never a silently
//! truncated comparison.
//!
//! # Definitions
//!
//! For paired observations (pᵢ, rᵢ) with i = 1..n and reference mean r̄:
//!
//! - **MAE** = (1/n) · Σ |pᵢ − rᵢ| — the average magnitude of errors,
//!   in the quantity's own unit (eV for the project's targets).
//! - **RMSE** = √( (1/n) · Σ (pᵢ − rᵢ)² ) — the quadratic mean error,
//!   which weighs large errors more heavily; RMSE ≥ MAE always.
//! - **R²** = 1 − Σ (pᵢ − rᵢ)² / Σ (rᵢ − r̄)² — the fraction of reference
//!   variance the predictions explain; 1.0 is perfect, 0.0 is
//!   mean-prediction quality, and negative values are worse than
//!   predicting the mean. Undefined when the references have zero
//!   variance.

use crate::error::{MetricsError, MetricsResult};

/// Mean absolute error: the average magnitude of paired differences.
///
/// # Errors
///
/// Returns [`MetricsError::LengthMismatch`] when the slices differ in
/// length and [`MetricsError::Empty`] when they hold no observations.
pub fn mae(predictions: &[f32], references: &[f32]) -> MetricsResult<f64> {
    let pairs = check_pairs(predictions, references)?;
    let sum: f64 = pairs
        .iter()
        .map(|&(p, r)| (f64::from(p) - f64::from(r)).abs())
        .sum();
    Ok(sum / pairs.len() as f64)
}

/// Root-mean-square error: the quadratic mean of paired differences.
///
/// # Errors
///
/// Same as [`mae`].
pub fn rmse(predictions: &[f32], references: &[f32]) -> MetricsResult<f64> {
    let pairs = check_pairs(predictions, references)?;
    let sum_squares: f64 = pairs
        .iter()
        .map(|&(p, r)| {
            let difference = f64::from(p) - f64::from(r);
            difference * difference
        })
        .sum();
    Ok((sum_squares / pairs.len() as f64).sqrt())
}

/// Coefficient of determination (R²): the fraction of reference variance
/// the predictions explain.
///
/// # Errors
///
/// Returns [`MetricsError::UndefinedR2`] when the references have zero
/// variance (constant references), in addition to the pairing errors of
/// [`mae`].
pub fn r2(predictions: &[f32], references: &[f32]) -> MetricsResult<f64> {
    let pairs = check_pairs(predictions, references)?;
    let n = pairs.len() as f64;
    let mean = pairs.iter().map(|&(_, r)| f64::from(r)).sum::<f64>() / n;
    let residual_sum: f64 = pairs
        .iter()
        .map(|&(p, r)| {
            let difference = f64::from(p) - f64::from(r);
            difference * difference
        })
        .sum();
    let total_sum: f64 = pairs
        .iter()
        .map(|&(_, r)| {
            let deviation = f64::from(r) - mean;
            deviation * deviation
        })
        .sum();
    if total_sum == 0.0 {
        return Err(MetricsError::UndefinedR2);
    }
    Ok(1.0 - residual_sum / total_sum)
}

/// Validates pairing and returns the zipped observations.
fn check_pairs(predictions: &[f32], references: &[f32]) -> MetricsResult<Vec<(f32, f32)>> {
    if predictions.len() != references.len() {
        return Err(MetricsError::LengthMismatch {
            predictions: predictions.len(),
            references: references.len(),
        });
    }
    if predictions.is_empty() {
        return Err(MetricsError::Empty);
    }
    Ok(predictions
        .iter()
        .copied()
        .zip(references.iter().copied())
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mae_is_exact_for_known_values() {
        // |1−2| = 1, |3−1| = 2, |5−5| = 0 → mean 1.
        let value = mae(&[1.0, 3.0, 5.0], &[2.0, 1.0, 5.0]).unwrap();
        assert!((value - 1.0).abs() < 1e-12);
    }

    #[test]
    fn rmse_is_exact_and_ge_mae() {
        // errors: 1, 2, 0 → squared sum 5 → mean 5/3 → √(5/3).
        let value = rmse(&[1.0, 3.0, 5.0], &[2.0, 1.0, 5.0]).unwrap();
        assert!((value - (5.0f64 / 3.0).sqrt()).abs() < 1e-12);
        let mean_absolute = mae(&[1.0, 3.0, 5.0], &[2.0, 1.0, 5.0]).unwrap();
        assert!(value >= mean_absolute);
    }

    #[test]
    fn rmse_penalizes_outliers_more_than_mae() {
        let references = [0.0; 2];
        let steady = rmse(&[1.0, 1.0], &references).unwrap();
        let outlier = rmse(&[0.0, 2.0], &references).unwrap();
        let steady_mae = mae(&[1.0, 1.0], &references).unwrap();
        let outlier_mae = mae(&[0.0, 2.0], &references).unwrap();
        // Same mean absolute error (1), different quadratic weight.
        assert!((steady_mae - outlier_mae).abs() < 1e-12);
        assert!(outlier > steady);
    }

    #[test]
    fn r2_bounds_and_interpretation() {
        let references = [1.0, 2.0, 3.0];
        // Perfect predictions: R² = 1.
        assert!((r2(&references, &references).unwrap() - 1.0).abs() < 1e-12);
        // Predicting the mean: R² = 0.
        let mean_predictions = [2.0, 2.0, 2.0];
        assert!(r2(&mean_predictions, &references).unwrap().abs() < 1e-12);
        // Worse than the mean: R² < 0.
        let worse = [5.0, 0.0, 5.0];
        assert!(r2(&worse, &references).unwrap() < 0.0);
    }

    #[test]
    fn r2_is_exact_for_a_known_case() {
        // references 1,2,3 (variance total 2); errors 0.5, 0.5, 0.5
        // (residual 0.75) → R² = 1 − 0.375.
        let value = r2(&[1.5, 2.5, 3.5], &[1.0, 2.0, 3.0]).unwrap();
        assert!((value - 0.625).abs() < 1e-12);
    }

    #[test]
    fn pairing_errors_are_typed() {
        let length_error = mae(&[1.0, 2.0], &[1.0]).unwrap_err();
        assert_eq!(
            length_error,
            MetricsError::LengthMismatch {
                predictions: 2,
                references: 1,
            }
        );
        assert_eq!(mae(&[], &[]).unwrap_err(), MetricsError::Empty);
    }

    #[test]
    fn constant_references_make_r2_undefined() {
        let error = r2(&[1.0, 2.0], &[5.0, 5.0]).unwrap_err();
        assert_eq!(error, MetricsError::UndefinedR2);
        // MAE and RMSE remain well-defined on the same data.
        assert!(rmse(&[1.0, 2.0], &[5.0, 5.0]).is_ok());
    }
}
