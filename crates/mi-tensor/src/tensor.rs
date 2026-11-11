//! The public tensor wrapper.
//!
//! Burn performs all numeric work and owns storage. In Burn 0.22 the backend
//! is chosen at runtime through the [`Device`](burn::tensor::Device), so this
//! wrapper carries only the tensor and its validated shape metadata. Host
//! readbacks (`to_vec`, `item`) are evaluation utilities and are never used
//! inside differentiable operations.
use crate::{TensorError, TensorResult, TensorShape, validation};
use burn::tensor::{Tensor as BurnTensor, TensorData};

/// Float32 tensor with compile-time rank and runtime dimensions.
///
/// Tensors created with [`Cpu::inference`](crate::Cpu::inference) run eager
/// CPU evaluation; tensors created with [`Cpu::training`](crate::Cpu::training)
/// additionally participate in Burn's autodiff graph (see
/// [`require_grad`](Tensor::require_grad) and
/// [`backward`](Tensor::backward)).
pub struct Tensor<const D: usize> {
    pub(crate) inner: BurnTensor<D>,
    pub(crate) shape: TensorShape,
}

impl<const D: usize> Clone for Tensor<D> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            shape: self.shape.clone(),
        }
    }
}

impl<const D: usize> Tensor<D> {
    /// Creates a tensor from row-major f32 data on the given device.
    ///
    /// Ingress is validated: the shape rank must match `D`, the data length
    /// must equal the shape's element count, and NaN/infinite values are
    /// rejected. Operation results may still be non-finite (see `div`).
    pub fn from_data(
        data: Vec<f32>,
        shape: TensorShape,
        device: &burn::tensor::Device,
    ) -> TensorResult<Self> {
        if shape.rank() != D {
            return Err(TensorError::InvalidShape(format!(
                "rank {} does not match tensor rank {D}",
                shape.rank()
            )));
        }
        if data.len() != shape.numel() {
            return Err(TensorError::DataLength {
                expected: shape.numel(),
                actual: data.len(),
            });
        }
        validation::finite(&data)?;
        let dims: [usize; D] = shape
            .dims()
            .try_into()
            .map_err(|_| TensorError::InvalidShape("rank mismatch".into()))?;
        let inner = BurnTensor::<D>::from_data(TensorData::new(data, dims), device);
        Ok(Self { inner, shape })
    }

    /// Creates a zero tensor of the requested shape on the given device.
    pub fn zeros(shape: TensorShape, device: &burn::tensor::Device) -> TensorResult<Self> {
        Self::from_data(vec![0.0; shape.numel()], shape, device)
    }

    /// Returns shape metadata.
    pub fn shape(&self) -> &TensorShape {
        &self.shape
    }

    /// Rank.
    pub const fn rank(&self) -> usize {
        D
    }

    /// Element count.
    pub fn numel(&self) -> usize {
        self.shape.numel()
    }

    /// Data type (Burn Flex float tensors in this API use f32).
    pub const fn dtype(&self) -> crate::DType {
        crate::DType::Float32
    }

    /// Copies tensor values to host memory in row-major order.
    ///
    /// This is an evaluation utility; it does not participate in autodiff.
    pub fn to_vec(&self) -> TensorResult<Vec<f32>> {
        self.inner
            .to_data()
            .try_to_vec::<f32>()
            .map_err(|e| TensorError::Readback(e.to_string()))
    }

    /// Copies the single element of a one-element tensor to the host.
    ///
    /// This is an evaluation utility; it does not participate in autodiff.
    pub fn item(&self) -> TensorResult<f32> {
        if self.numel() != 1 {
            return Err(TensorError::ExpectedScalar {
                actual: self.numel(),
            });
        }
        self.inner
            .clone()
            .try_into_scalar::<f32>()
            .map_err(|e| TensorError::Readback(e.to_string()))
    }
}
