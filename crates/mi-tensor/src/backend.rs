use burn::tensor::Device;

/// Explicit CPU selection backed by Burn Flex (pure Rust, no GPU runtime).
#[derive(Debug, Clone, Copy, Default)]
pub struct Cpu;
/// Non-differentiating CPU execution context.
pub type InferenceDevice = Device;
/// CPU execution context with Burn autodiff enabled.
pub type TrainingDevice = Device;
impl Cpu {
    /// Select Flex CPU for inference.
    pub fn inference() -> InferenceDevice { Device::flex() }
    /// Select Flex CPU with autodiff for gradient work.
    pub fn training() -> TrainingDevice { Device::flex().autodiff() }
}
