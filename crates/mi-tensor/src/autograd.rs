//! Autodiff API on the tensor wrapper.
//!
//! Everything here delegates to Burn's autodiff backend; no project-owned
//! autograd engine is introduced. Leaves must be created on
//! [`Cpu::training`](crate::Cpu::training) and marked with
//! [`require_grad`](Tensor::require_grad); backward runs from any tracked
//! output tensor and gradients are retrieved per-leaf.
use crate::Tensor;
use crate::{TensorError, TensorResult};
use burn::tensor::Gradients;

impl<const D: usize> Tensor<D> {
    /// Marks this tensor as requiring its gradient after a backward pass.
    ///
    /// Returns an error when the tensor was not created on an
    /// autodiff-enabled device ([`Cpu::training`](crate::Cpu::training)).
    pub fn require_grad(self) -> TensorResult<Self> {
        if !self.inner.is_autodiff() {
            return Err(TensorError::AutodiffUnavailable);
        }
        Ok(Self {
            inner: self.inner.require_grad(),
            shape: self.shape,
        })
    }

    /// Returns whether this tensor's gradient is retained after backward.
    pub fn is_require_grad(&self) -> bool {
        self.inner.is_require_grad()
    }

    /// Returns whether this tensor participates in a recorded autodiff graph.
    pub fn is_tracked(&self) -> bool {
        self.inner.is_tracked()
    }

    /// Returns the tensor with its recorded graph detached.
    ///
    /// The result starts a new lineage; it does not restore the earlier graph.
    pub fn detach(&self) -> Self {
        Self {
            inner: self.inner.clone().detach(),
            shape: self.shape.clone(),
        }
    }

    /// Backpropagates from this tensor and returns the gradient container.
    ///
    /// Burn consumes the recorded steps reachable from this tensor (there is
    /// no `retain_graph`); combine losses that share intermediates into one
    /// tensor before calling `backward`. Each fresh forward pass records a
    /// new graph.
    ///
    /// Returns an error instead of panicking when this tensor does not
    /// participate in an autodiff graph.
    pub fn backward(&self) -> TensorResult<Gradients> {
        if !self.inner.is_tracked() {
            return Err(TensorError::AutodiffUnavailable);
        }
        Ok(self.inner.backward())
    }

    /// Returns this tensor's gradient from a [`Gradients`] container, if present.
    ///
    /// The returned tensor is a plain evaluation tensor (no autodiff
    /// association) with the same shape as `self`.
    pub fn grad(&self, grads: &Gradients) -> Option<Self> {
        self.inner.grad(grads).map(|inner| Self {
            inner,
            shape: self.shape.clone(),
        })
    }
}
