//! Z-score standardization shared by training and evaluation.
//!
//! The baseline recipe (see `docs/architecture/baseline-model.md`) z-scores
//! descriptor inputs and targets **over the training subset only** and
//! un-scales predictions before reporting metrics in electronvolts. The
//! scaler was previously a test-local struct; this module lifts it into the
//! library so the benchmark suite and any experiment share one fitted,
//! serializable-by-values scaler.
//!
//! Arithmetic is plain `f32` host math — scalers are metadata, not tensors,
//! and never touch a device. The deviation is floored at `1e-6` so constant
//! columns stay well-defined instead of dividing by zero.

use crate::error::{FeaturizationError, FeaturizationResult};

/// A fitted per-value z-score transform: `(x − mean) / deviation`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StandardScaler {
    mean: f32,
    deviation: f32,
}

impl StandardScaler {
    /// Fits the scaler over the given values.
    ///
    /// # Errors
    ///
    /// Returns [`FeaturizationError::InvalidScalerInput`] for empty input.
    pub fn fit(values: &[f32]) -> FeaturizationResult<Self> {
        if values.is_empty() {
            return Err(FeaturizationError::InvalidScalerInput(
                "cannot fit a scaler on zero values".into(),
            ));
        }
        let mean = values.iter().sum::<f32>() / values.len() as f32;
        let variance = values
            .iter()
            .map(|value| (value - mean).powi(2))
            .sum::<f32>()
            / values.len() as f32;
        Ok(Self {
            mean,
            deviation: variance.sqrt().max(1e-6),
        })
    }

    /// The fitted mean.
    pub const fn mean(&self) -> f32 {
        self.mean
    }

    /// The fitted (floored) standard deviation.
    pub const fn deviation(&self) -> f32 {
        self.deviation
    }

    /// Standardizes one value.
    pub fn scale(&self, value: f32) -> f32 {
        (value - self.mean) / self.deviation
    }

    /// Reverses [`scale`](StandardScaler::scale).
    pub fn unscale(&self, value: f32) -> f32 {
        value * self.deviation + self.mean
    }
}

/// One scaler per column of a fixed-width feature matrix.
#[derive(Debug, Clone, PartialEq)]
pub struct ColumnScalers {
    scalers: Vec<StandardScaler>,
}

impl ColumnScalers {
    /// Fits one scaler per column over whole rows (every column used).
    ///
    /// # Errors
    ///
    /// Returns [`FeaturizationError::InvalidScalerInput`] for empty input
    /// or rows of inconsistent width.
    pub fn fit_rows(rows: &[Vec<f32>]) -> FeaturizationResult<Self> {
        let Some(first) = rows.first() else {
            return Err(FeaturizationError::InvalidScalerInput(
                "cannot fit column scalers on zero rows".into(),
            ));
        };
        let width = first.len();
        let mut columns = vec![Vec::with_capacity(rows.len()); width];
        for row in rows {
            if row.len() != width {
                return Err(FeaturizationError::InvalidScalerInput(format!(
                    "expected row width {width}, got {}",
                    row.len()
                )));
            }
            for (column, value) in row.iter().enumerate() {
                columns[column].push(*value);
            }
        }
        Self::fit_columns(&columns)
    }

    /// Fits one scaler per column over pre-assembled columns.
    ///
    /// # Errors
    ///
    /// Returns [`FeaturizationError::InvalidScalerInput`] for empty input
    /// or empty columns (a column with no values cannot be fitted).
    pub fn fit_columns(columns: &[Vec<f32>]) -> FeaturizationResult<Self> {
        if columns.is_empty() {
            return Err(FeaturizationError::InvalidScalerInput(
                "cannot fit zero columns".into(),
            ));
        }
        let scalers = columns
            .iter()
            .map(|column| StandardScaler::fit(column))
            .collect::<FeaturizationResult<Vec<_>>>()?;
        Ok(Self { scalers })
    }

    /// The number of columns.
    pub fn width(&self) -> usize {
        self.scalers.len()
    }

    /// The scaler fitted for one column.
    ///
    /// # Errors
    ///
    /// Returns [`FeaturizationError::InvalidScalerInput`] for an
    /// out-of-range column.
    pub fn scaler(&self, column: usize) -> FeaturizationResult<&StandardScaler> {
        self.scalers.get(column).ok_or_else(|| {
            FeaturizationError::InvalidScalerInput(format!(
                "column {column} is outside width {}",
                self.scalers.len()
            ))
        })
    }

    /// Standardizes one row.
    ///
    /// # Errors
    ///
    /// Returns [`FeaturizationError::InvalidScalerInput`] when the row
    /// width differs from the fitted width.
    pub fn scale_row(&self, row: &[f32]) -> FeaturizationResult<Vec<f32>> {
        if row.len() != self.scalers.len() {
            return Err(FeaturizationError::InvalidScalerInput(format!(
                "expected row width {}, got {}",
                self.scalers.len(),
                row.len()
            )));
        }
        Ok(row
            .iter()
            .zip(&self.scalers)
            .map(|(value, scaler)| scaler.scale(*value))
            .collect())
    }

    /// Reverses the standardization of one value in one column.
    ///
    /// # Errors
    ///
    /// Returns [`FeaturizationError::InvalidScalerInput`] for an
    /// out-of-range column.
    pub fn unscale_value(&self, column: usize, value: f32) -> FeaturizationResult<f32> {
        Ok(self.scaler(column)?.unscale(value))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scaler_matches_the_baseline_recipe_values() {
        // The fixture values from the Phase 7 overfit proof.
        let scaler = StandardScaler::fit(&[1.0, 2.0, 3.0]).unwrap();
        assert!((scaler.mean() - 2.0).abs() < 1e-6);
        // Population variance = 2/3 → deviation √(2/3).
        assert!((scaler.deviation() - (2.0f32 / 3.0).sqrt()).abs() < 1e-6);
        assert!((scaler.scale(3.0) - (3.0 - 2.0) / (2.0f32 / 3.0).sqrt()).abs() < 1e-6);
        assert!((scaler.unscale(scaler.scale(3.0)) - 3.0).abs() < 1e-5);
    }

    #[test]
    fn constant_columns_are_floored_not_divided_by_zero() {
        let scaler = StandardScaler::fit(&[5.0, 5.0, 5.0]).unwrap();
        assert_eq!(scaler.deviation(), 1e-6);
        assert_eq!(scaler.scale(5.0), 0.0);
    }

    #[test]
    fn empty_fit_is_a_typed_error() {
        assert!(StandardScaler::fit(&[]).is_err());
        assert!(ColumnScalers::fit_rows(&[]).is_err());
        assert!(ColumnScalers::fit_columns(&[vec![]]).is_err());
    }

    #[test]
    fn column_scalers_round_trip_rows() {
        let rows = vec![vec![0.0, 10.0], vec![1.0, 20.0], vec![2.0, 30.0]];
        let scalers = ColumnScalers::fit_rows(&rows).unwrap();
        assert_eq!(scalers.width(), 2);
        let scaled = scalers.scale_row(&[2.0, 30.0]).unwrap();
        assert!(scaled.iter().all(|value| value.is_finite()));
        for (column, value) in scaled.iter().enumerate() {
            let restored = scalers.unscale_value(column, *value).unwrap();
            assert!((restored - [2.0, 30.0][column]).abs() < 1e-5);
        }
    }

    #[test]
    fn inconsistent_row_widths_are_rejected() {
        let rows = vec![vec![1.0, 2.0], vec![1.0]];
        assert!(matches!(
            ColumnScalers::fit_rows(&rows),
            Err(FeaturizationError::InvalidScalerInput(_))
        ));
        let scalers = ColumnScalers::fit_rows(&[vec![1.0, 2.0]]).unwrap();
        assert!(matches!(
            scalers.scale_row(&[1.0]),
            Err(FeaturizationError::InvalidScalerInput(_))
        ));
    }

    #[test]
    fn columns_fit_independently_of_rows() {
        // Target scalers fit over *present* values only — the columns are
        // pre-assembled by the caller.
        let columns = vec![vec![-12.6, -13.6], vec![1.5, 2.0]];
        let scalers = ColumnScalers::fit_columns(&columns).unwrap();
        assert_eq!(scalers.width(), 2);
        assert!((scalers.scaler(0).unwrap().mean() - (-13.1)).abs() < 1e-6);
    }
}
