//! Numerical correctness tests: ingress validation, IEEE-754 semantics,
//! reduction behavior, rank consistency, and inference/training device
//! separation.
use mi_tensor::operations::{self as ops};
use mi_tensor::{Cpu, Tensor, TensorError, TensorShape};

fn shape(dims: &[usize]) -> TensorShape {
    TensorShape::new(dims.to_vec()).expect("valid shape")
}

fn t1(data: &[f32]) -> Tensor<1> {
    Tensor::from_data(data.to_vec(), shape(&[data.len()]), &Cpu::inference()).expect("valid tensor")
}

fn t2(data: &[f32], rows: usize, cols: usize) -> Tensor<2> {
    Tensor::from_data(data.to_vec(), shape(&[rows, cols]), &Cpu::inference()).expect("valid tensor")
}

#[test]
fn shape_rejects_empty_and_zero_dimensions() {
    assert!(matches!(
        TensorShape::new(Vec::<usize>::new()),
        Err(TensorError::InvalidShape(_))
    ));
    assert!(matches!(
        TensorShape::new(vec![0, 3]),
        Err(TensorError::InvalidShape(_))
    ));
}

#[test]
fn shape_rejects_element_count_overflow() {
    let dims = vec![usize::MAX, 2];
    assert!(matches!(
        TensorShape::new(dims),
        Err(TensorError::InvalidShape(_))
    ));
}

#[test]
fn from_data_rejects_rank_mismatch() {
    let s = shape(&[3]);
    assert!(matches!(
        Tensor::<2>::from_data(vec![1.0, 2.0, 3.0], s, &Cpu::inference()),
        Err(TensorError::InvalidShape(_))
    ));
}

#[test]
fn from_data_rejects_wrong_data_length() {
    let s = shape(&[2, 2]);
    assert!(matches!(
        Tensor::<2>::from_data(vec![1.0, 2.0, 3.0], s, &Cpu::inference()),
        Err(TensorError::DataLength {
            expected: 4,
            actual: 3
        })
    ));
}

#[test]
fn from_data_rejects_nan_and_infinity() {
    let s = shape(&[3]);
    let nan = Tensor::<1>::from_data(vec![1.0, f32::NAN, 2.0], s.clone(), &Cpu::inference());
    assert!(matches!(nan, Err(TensorError::NonFiniteValue { index: 1 })));
    let inf = Tensor::<1>::from_data(vec![1.0, f32::INFINITY, 2.0], s, &Cpu::inference());
    assert!(matches!(inf, Err(TensorError::NonFiniteValue { index: 1 })));
}

#[test]
fn metadata_round_trip() {
    let x = t2(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
    assert_eq!(x.rank(), 2);
    assert_eq!(x.numel(), 6);
    assert_eq!(x.shape().dims(), &[2, 3]);
    assert_eq!(x.dtype(), mi_tensor::DType::Float32);
    assert_eq!(
        x.to_vec().expect("readback"),
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    );
}

#[test]
fn zeros_match_shape_and_values() {
    let x = Tensor::<2>::zeros(shape(&[2, 3]), &Cpu::inference()).expect("zeros");
    assert_eq!(x.shape().dims(), &[2, 3]);
    assert!(x.to_vec().expect("readback").iter().all(|v| *v == 0.0));
}

#[test]
fn item_returns_single_element_and_rejects_larger_tensors() {
    let one = t1(&[4.25]);
    assert_eq!(one.item().expect("single element"), 4.25);
    let many = t1(&[1.0, 2.0]);
    assert!(matches!(
        many.item(),
        Err(TensorError::ExpectedScalar { actual: 2 })
    ));
}

#[test]
fn division_follows_ieee_754() {
    let one = t1(&[1.0, -1.0, 0.0, 0.0]);
    let zero = t1(&[0.0, 0.0, 0.0, 0.0]);
    let q = ops::div(&one, &zero).expect("same shapes");
    // NaN never compares equal to itself, so the comparison is elementwise.
    let expected = [f32::INFINITY, f32::NEG_INFINITY, f32::NAN, f32::NAN];
    for (got, want) in q.to_vec().expect("readback").iter().zip(expected) {
        if want.is_nan() {
            assert!(got.is_nan(), "expected NaN, got {got}");
        } else {
            assert_eq!(*got, want);
        }
    }
}

#[test]
fn non_finite_results_are_not_reingested() {
    // Ingress still rejects non-finite values; only operation results may
    // be non-finite.
    let one = t1(&[1.0]);
    let zero = t1(&[0.0]);
    let q = ops::div(&one, &zero).expect("same shapes");
    let inf = q.to_vec().expect("readback")[0];
    assert!(inf.is_infinite());
    let again = Tensor::<1>::from_data(vec![inf], shape(&[1]), &Cpu::inference());
    assert!(matches!(
        again,
        Err(TensorError::NonFiniteValue { index: 0 })
    ));
}

#[test]
fn arithmetic_preserves_values_and_shapes() {
    let a = t1(&[1.0, 2.0, 3.0]);
    let b = t1(&[4.0, 5.0, 6.0]);
    assert_eq!(
        ops::add(&a, &b).expect("add").to_vec().expect("r"),
        vec![5.0, 7.0, 9.0]
    );
    assert_eq!(
        ops::sub(&b, &a).expect("sub").to_vec().expect("r"),
        vec![3.0, 3.0, 3.0]
    );
    assert_eq!(
        ops::mul(&a, &b).expect("mul").to_vec().expect("r"),
        vec![4.0, 10.0, 18.0]
    );
    assert_eq!(
        ops::div(&b, &a).expect("div").to_vec().expect("r"),
        vec![4.0, 2.5, 2.0]
    );
}

#[test]
fn arithmetic_rejects_shape_mismatch() {
    let a = t1(&[1.0, 2.0]);
    let b = t1(&[1.0, 2.0, 3.0]);
    assert!(matches!(
        ops::add(&a, &b),
        Err(TensorError::ShapeMismatch { .. })
    ));
}

#[test]
fn matmul_computes_product_and_shape() {
    let a = t2(&[1.0, 2.0, 3.0, 4.0], 2, 2);
    let b = t2(&[5.0, 6.0, 7.0, 8.0], 2, 2);
    let c = ops::matmul(&a, &b).expect("compatible shapes");
    assert_eq!(c.shape().dims(), &[2, 2]);
    assert_eq!(c.to_vec().expect("readback"), vec![19.0, 22.0, 43.0, 50.0]);
}

#[test]
fn matmul_rejects_incompatible_inner_dimensions() {
    let a = t2(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
    let b = t2(&[1.0, 2.0, 3.0, 4.0], 2, 2);
    assert!(matches!(
        ops::matmul(&a, &b),
        Err(TensorError::ShapeMismatch { .. })
    ));
}

#[test]
fn transpose_swaps_shape_and_values() {
    let a = t2(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
    let at = ops::transpose(&a);
    assert_eq!(at.shape().dims(), &[3, 2]);
    assert_eq!(
        at.to_vec().expect("readback"),
        vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
    );
}

#[test]
fn dot_requires_equal_lengths() {
    let a = t1(&[1.0, 2.0]);
    let b = t1(&[3.0, 4.0, 5.0]);
    assert!(matches!(
        ops::dot(&a, &b),
        Err(TensorError::ShapeMismatch { .. })
    ));
}

#[test]
fn host_reductions_return_expected_scalars() {
    let x = t1(&[3.0, -1.0, 2.5]);
    assert_eq!(ops::sum(&x).expect("sum"), 4.5);
    assert_eq!(ops::mean(&x).expect("mean"), 1.5);
    assert_eq!(ops::min(&x).expect("min"), -1.0);
    assert_eq!(ops::max(&x).expect("max"), 3.0);
    assert_eq!(ops::dot(&x, &x).expect("dot"), 9.0 + 1.0 + 6.25);
}

#[test]
fn differentiable_reductions_return_rank_one_tensors() {
    let x = t1(&[1.0, 2.0, 3.0]);
    let s = ops::sum_all(&x).expect("sum_all");
    let m = ops::mean_all(&x).expect("mean_all");
    assert_eq!(s.shape().dims(), &[1]);
    assert_eq!(s.item().expect("item"), 6.0);
    assert_eq!(m.item().expect("item"), 2.0);
}

#[test]
fn loss_values_match_hand_computed_references() {
    // MSE = mean((p - t)^2) = mean([1, 4, 9]) = 14/3.
    let p = t1(&[1.0, 2.0, 3.0]);
    let t = t1(&[0.0, 0.0, 0.0]);
    let mse = mi_tensor::mse(&p, &t).expect("paired shapes");
    assert_eq!(mse.shape().dims(), &[1]);
    assert!((mse.item().expect("item") - 14.0 / 3.0).abs() < 1e-6);

    // MAE = mean(|p - t|) = 2.
    let mae = mi_tensor::mae(&p, &t).expect("paired shapes");
    assert!((mae.item().expect("item") - 2.0).abs() < 1e-6);

    // Huber, delta = 2: d = [1, 2, 3] -> quadratic 0.5, boundary 2.0,
    // linear 2*(3 - 1) = 4; mean = 6.5/3.
    let huber = mi_tensor::huber_loss(&p, &t, 2.0).expect("paired shapes");
    assert!((huber.item().expect("item") - 6.5 / 3.0).abs() < 1e-6);
}

#[test]
fn huber_quadratic_and_linear_branches_agree_at_delta_boundary() {
    // At d = delta both branches evaluate to 0.5 * delta^2.
    let p = t1(&[1.0, -1.0]);
    let t = t1(&[0.0, 0.0]);
    let loss = mi_tensor::huber_loss(&p, &t, 1.0).expect("paired shapes");
    assert!((loss.item().expect("item") - 0.5).abs() < 1e-6);
}

#[test]
fn huber_rejects_invalid_delta() {
    let p = t1(&[1.0]);
    let t = t1(&[0.0]);
    assert!(matches!(
        mi_tensor::huber_loss(&p, &t, 0.0),
        Err(TensorError::InvalidShape(_))
    ));
    assert!(matches!(
        mi_tensor::huber_loss(&p, &t, -1.0),
        Err(TensorError::InvalidShape(_))
    ));
    assert!(matches!(
        mi_tensor::huber_loss(&p, &t, f32::NAN),
        Err(TensorError::InvalidShape(_))
    ));
}

#[test]
fn losses_reject_shape_mismatch_and_rank_differences() {
    let p = t1(&[1.0, 2.0]);
    let t = t1(&[1.0, 2.0, 3.0]);
    assert!(matches!(
        mi_tensor::mse(&p, &t),
        Err(TensorError::ShapeMismatch { .. })
    ));
    assert!(matches!(
        mi_tensor::mae(&p, &t),
        Err(TensorError::ShapeMismatch { .. })
    ));
    assert!(matches!(
        mi_tensor::huber_loss(&p, &t, 1.0),
        Err(TensorError::ShapeMismatch { .. })
    ));
}

#[test]
fn activations_compute_expected_values() {
    let x = t1(&[0.0, 1.0, -1.0]);
    assert_eq!(ops::relu(&x).to_vec().expect("r"), vec![0.0, 1.0, 0.0]);
    let s = ops::sigmoid(&x).to_vec().expect("r");
    assert!((s[0] - 0.5).abs() < 1e-6);
    assert!((s[1] - 1.0 / (1.0 + (-1.0f32).exp())).abs() < 1e-6);
    let th = ops::tanh(&x).to_vec().expect("r");
    assert!((th[0]).abs() < 1e-6);
    assert!((th[2] - (-(1.0f32)).tanh()).abs() < 1e-6);
}

#[test]
fn inference_and_training_devices_are_distinct() {
    let inference = Cpu::inference();
    let training = Cpu::training();
    assert!(!inference.is_autodiff(), "inference device has no autodiff");
    assert!(training.is_autodiff(), "training device enables autodiff");
    // The idempotent decorator: autodiff(training) == training.
    let twice = training.clone().autodiff();
    assert!(twice.is_autodiff());
}

#[test]
fn clone_is_independent_of_the_original() {
    let a = t1(&[1.0, 2.0]);
    let b = a.clone();
    let c = ops::add(&b, &t1(&[10.0, 20.0])).expect("add");
    assert_eq!(a.to_vec().expect("r"), vec![1.0, 2.0], "original untouched");
    assert_eq!(c.to_vec().expect("r"), vec![11.0, 22.0]);
}

#[test]
fn rank_is_enforced_by_the_type_system_across_ops() {
    // Rank-one and rank-two tensors cannot be mixed: the rank mismatch is a
    // compile error, and same-rank shape mismatches are runtime errors.
    let a = t1(&[1.0, 2.0, 3.0]);
    let b2 = t2(&[1.0, 2.0, 3.0, 4.0], 2, 2);
    let b3 = t1(&[1.0, 2.0, 3.0, 4.0]);
    assert!(matches!(
        ops::add(&a, &b3),
        Err(TensorError::ShapeMismatch { .. })
    ));
    // The following would not compile:
    // ops::add(&a, &b2);
    let _ = b2;
}
