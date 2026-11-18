//! Matrix and structural operations.
//!
//! Dimensions are validated before invoking Burn; the operations run
//! through Burn so the autodiff computation graph is preserved.
use crate::{Tensor, TensorError, TensorResult};
use burn::tensor::Tensor as BurnTensor;

/// Matrix product of rank-two tensors.
pub fn matmul(a: &Tensor<2>, b: &Tensor<2>) -> TensorResult<Tensor<2>> {
    if a.shape().dims()[1] != b.shape().dims()[0] {
        return Err(TensorError::ShapeMismatch {
            left: a.shape().dims().to_vec(),
            right: b.shape().dims().to_vec(),
        });
    }
    let inner = a.inner.clone().matmul(b.inner.clone());
    let shape = crate::TensorShape::new(vec![a.shape().dims()[0], b.shape().dims()[1]])?;
    Ok(Tensor { inner, shape })
}

/// Transposes a rank-two tensor (swaps rows and columns).
pub fn transpose(a: &Tensor<2>) -> Tensor<2> {
    let inner = a.inner.clone().transpose();
    let shape = crate::TensorShape::new(vec![a.shape().dims()[1], a.shape().dims()[0]])
        .expect("swapping the two dimensions of a validated shape is always valid");
    Tensor { inner, shape }
}

/// Concatenates tensors along a dimension.
///
/// Every tensor must share the rank and every dimension except `dim`; the
/// result's `dim` size is the sum. This is the batch-assembly primitive:
/// stacking single-row samples into a `[B, C]` batch keeps the autodiff
/// graph connected to every leaf.
///
/// # Errors
///
/// Returns [`TensorError::EmptyTensor`] for an empty list,
/// [`TensorError::InvalidShape`] when `dim` is outside the rank or a
/// non-`dim` dimension disagrees, and [`TensorError::ShapeMismatch`] when
/// tensor shapes disagree (with the first tensor as reference).
pub fn cat<const D: usize>(tensors: &[&Tensor<D>], dim: usize) -> TensorResult<Tensor<D>> {
    if tensors.is_empty() {
        return Err(TensorError::EmptyTensor);
    }
    if dim >= D {
        return Err(TensorError::InvalidShape(format!(
            "concatenation dim {dim} is outside rank {D}"
        )));
    }
    let reference = tensors[0].shape();
    for tensor in tensors {
        if tensor.shape().rank() != reference.rank()
            || tensor
                .shape()
                .dims()
                .iter()
                .zip(reference.dims())
                .enumerate()
                .any(|(axis, (a, b))| axis != dim && a != b)
        {
            return Err(TensorError::ShapeMismatch {
                left: reference.dims().to_vec(),
                right: tensor.shape().dims().to_vec(),
            });
        }
    }
    let inner = BurnTensor::cat(tensors.iter().map(|t| t.inner.clone()).collect(), dim);
    let mut dims = reference.dims().to_vec();
    dims[dim] = tensors.iter().map(|t| t.shape().dims()[dim]).sum();
    Ok(Tensor {
        inner,
        shape: crate::TensorShape::new(dims)?,
    })
}

/// Dot product of two equal-length rank-one tensors.
///
/// This is a host-side evaluation utility: the result is an `f32` scalar and
/// no gradient is retrievable from it.
pub fn dot(a: &Tensor<1>, b: &Tensor<1>) -> TensorResult<f32> {
    if a.shape() != b.shape() {
        return Err(TensorError::ShapeMismatch {
            left: a.shape().dims().to_vec(),
            right: b.shape().dims().to_vec(),
        });
    }
    let value = (a.inner.clone() * b.inner.clone())
        .sum()
        .try_into_scalar::<f32>()
        .map_err(|e| TensorError::Readback(e.to_string()))?;
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Cpu, TensorShape};

    fn row(data: &[f32]) -> Tensor<2> {
        Tensor::<2>::from_data(
            data.to_vec(),
            TensorShape::new(vec![1, data.len()]).unwrap(),
            &Cpu::inference(),
        )
        .unwrap()
    }

    #[test]
    fn cat_stacks_rows_and_sums_the_dimension() {
        let a = row(&[1.0, 2.0]);
        let b = row(&[3.0, 4.0]);
        let c = row(&[5.0, 6.0]);
        let stacked = cat(&[&a, &b, &c], 0).unwrap();
        assert_eq!(stacked.shape().dims(), &[3, 2]);
        assert_eq!(
            stacked.to_vec().unwrap(),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        );
    }

    #[test]
    fn cat_keeps_the_graph_connected_to_every_leaf() {
        let device = Cpu::training();
        let a = Tensor::<2>::from_data(
            vec![1.0, 2.0],
            TensorShape::new(vec![1, 2]).unwrap(),
            &device,
        )
        .unwrap()
        .require_grad()
        .unwrap();
        let b = Tensor::<2>::from_data(
            vec![3.0, 4.0],
            TensorShape::new(vec![1, 2]).unwrap(),
            &device,
        )
        .unwrap()
        .require_grad()
        .unwrap();
        let stacked = cat(&[&a, &b], 0).unwrap();
        let loss = crate::operations::sum_all(&stacked).unwrap();
        let gradients = loss.backward().unwrap();
        assert_eq!(
            a.grad(&gradients).unwrap().to_vec().unwrap(),
            vec![1.0, 1.0]
        );
        assert_eq!(
            b.grad(&gradients).unwrap().to_vec().unwrap(),
            vec![1.0, 1.0]
        );
    }

    #[test]
    fn cat_rejects_mismatched_shapes_and_bad_dims() {
        let a = row(&[1.0, 2.0]);
        let b = Tensor::<2>::from_data(
            vec![1.0, 2.0, 3.0],
            TensorShape::new(vec![1, 3]).unwrap(),
            &Cpu::inference(),
        )
        .unwrap();
        assert!(matches!(
            cat(&[&a, &b], 0),
            Err(TensorError::ShapeMismatch { .. })
        ));
        let empty: Vec<&Tensor<2>> = Vec::new();
        assert!(matches!(cat(&empty, 0), Err(TensorError::EmptyTensor)));
        let pair = [row(&[1.0, 2.0]), row(&[3.0, 4.0])];
        let refs: Vec<&Tensor<2>> = pair.iter().collect();
        assert!(matches!(cat(&refs, 2), Err(TensorError::InvalidShape(_))));
    }
}
