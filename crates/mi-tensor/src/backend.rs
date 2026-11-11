use burn::tensor::Device;

/// Explicit CPU selection backed by Burn Flex (pure Rust, no GPU runtime).
///
/// In Burn 0.22 the backend is selected at runtime through the [`Device`],
/// so inference and training share the same device type; the difference is
/// whether autodiff is enabled on it.
#[derive(Debug, Clone, Copy, Default)]
pub struct Cpu;

impl Cpu {
    /// Select Flex CPU for inference.
    pub fn inference() -> Device {
        Device::flex()
    }

    /// Select Flex CPU with autodiff enabled for gradient work.
    ///
    /// Tensors created on this device can participate in Burn's autodiff
    /// graph when marked with [`Tensor::require_grad`](crate::Tensor::require_grad).
    pub fn training() -> Device {
        Device::flex().autodiff()
    }
}
