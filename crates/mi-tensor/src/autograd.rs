//! Autodiff helpers. The training device explicitly enables Burn's autodiff context.
use burn::tensor::Tensor as BurnTensor;
/// Creates a rank-one training tensor requiring gradients.
pub fn variable(data: Vec<f32>) -> BurnTensor<1> {
    BurnTensor::<1>::from_data(data, &crate::Cpu::training()).require_grad()
}
