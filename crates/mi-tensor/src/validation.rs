//! Boundary validation helpers.
use crate::{TensorError, TensorResult};

/// Rejects NaN and infinite input values.
pub fn finite(values: &[f32]) -> TensorResult<()> {
    if let Some(index) = values.iter().position(|v| !v.is_finite()) {
        Err(TensorError::NonFiniteValue { index })
    } else {
        Ok(())
    }
}
