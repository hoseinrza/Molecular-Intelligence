use burn::tensor::{Tensor as BurnTensor, TensorData};
use crate::{Cpu, DType, TensorError, TensorResult, TensorShape};

/// Float32 tensor with compile-time rank and runtime dimensions.
/// Burn owns storage and computation; this wrapper adds shape checks at ingress.
pub struct Tensor<const D: usize> { pub(crate) inner: BurnTensor<D>, pub(crate) shape: TensorShape }
impl<const D: usize> Clone for Tensor<D> { fn clone(&self) -> Self { Self { inner: self.inner.clone(), shape: self.shape.clone() } } }
impl<const D: usize> Tensor<D> {
    /// Creates a tensor from row-major f32 data on the Flex CPU backend.
    pub fn from_data(data: Vec<f32>, shape: TensorShape) -> TensorResult<Self> {
        if shape.rank() != D { return Err(TensorError::InvalidShape(format!("rank {} does not match tensor rank {D}", shape.rank()))); }
        if data.len() != shape.numel() { return Err(TensorError::DataLength { expected: shape.numel(), actual: data.len() }); }
        if let Some(index) = data.iter().position(|x| !x.is_finite()) { return Err(TensorError::NonFiniteValue { index }); }
        let dims: [usize; D] = shape.dims().try_into().map_err(|_| TensorError::InvalidShape("rank mismatch".into()))?;
        let inner = BurnTensor::<D>::from_data(TensorData::new(data, dims), &Cpu::inference());
        Ok(Self { inner, shape })
    }
    /// Creates a zero tensor of the requested shape.
    pub fn zeros(shape: TensorShape) -> TensorResult<Self> {
        if shape.rank() != D { return Err(TensorError::InvalidShape(format!("rank {} does not match tensor rank {D}", shape.rank()))); }
        let data = vec![0.0; shape.numel()]; Self::from_data(data, shape)
    }
    /// Returns shape metadata.
    pub fn shape(&self) -> &TensorShape { &self.shape }
    /// Rank.
    pub const fn rank(&self) -> usize { D }
    /// Element count.
    pub fn numel(&self) -> usize { self.shape.numel() }
    /// Data type (Burn Flex float tensors in this API use f32).
    pub const fn dtype(&self) -> DType { DType::Float32 }
    /// Copies tensor values to host memory in row-major order.
    pub fn to_vec(&self) -> TensorResult<Vec<f32>> { self.inner.clone().into_data().to_vec().map_err(|_| TensorError::InvalidShape("Burn data conversion failed".into())) }
}
