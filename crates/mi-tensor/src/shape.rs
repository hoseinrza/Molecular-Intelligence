use crate::{TensorError, TensorResult};

/// Validated runtime dimensions. Zero-sized dimensions are rejected.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TensorShape(Vec<usize>);
impl TensorShape {
    /// Validates dimensions and their element-count product.
    pub fn new(dims: impl Into<Vec<usize>>) -> TensorResult<Self> {
        let dims = dims.into();
        if dims.is_empty() || dims.contains(&0) {
            return Err(TensorError::InvalidShape(
                "rank and dimensions must be nonzero".into(),
            ));
        }
        dims.iter()
            .try_fold(1usize, |n, &d| n.checked_mul(d))
            .ok_or_else(|| TensorError::InvalidShape("element count overflows usize".into()))?;
        Ok(Self(dims))
    }
    /// Dimensions as a slice.
    pub fn dims(&self) -> &[usize] {
        &self.0
    }
    /// Rank of this shape.
    pub fn rank(&self) -> usize {
        self.0.len()
    }
    /// Number of elements.
    pub fn numel(&self) -> usize {
        self.0.iter().product()
    }
}
