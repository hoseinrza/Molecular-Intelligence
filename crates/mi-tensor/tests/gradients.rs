//! Gradient and autodiff tests.
//!
//! Leaves are created on the training device (`Cpu::training()`) and marked
//! with `require_grad`. Autodiff gradients are verified both against
//! hand-derived analytic results and against central finite differences.
//!
//! Finite-difference settings are chosen for f32 accuracy: with step `h`,
//! the central-difference truncation error is O(h^2) and the roundoff error
//! is O(eps/h) with eps ~ 1.19e-7, so h = 5e-3 keeps both well under the
//! 1e-3 tolerance used here.
use mi_tensor::operations::{self as ops};
use mi_tensor::{Cpu, Tensor, TensorError, TensorShape};

/// Central-difference step.
const H: f32 = 5e-3;
/// Tolerance for gradient comparisons: absolute + relative slack.
const TOL: f32 = 1e-3;

fn shape1(n: usize) -> TensorShape {
    TensorShape::new(vec![n]).expect("valid rank-one shape")
}

fn shape2(rows: usize, cols: usize) -> TensorShape {
    TensorShape::new(vec![rows, cols]).expect("valid rank-two shape")
}

/// Training-device leaf marked for gradient retention.
fn leaf1(data: &[f32]) -> Tensor<1> {
    Tensor::from_data(data.to_vec(), shape1(data.len()), &Cpu::training())
        .expect("valid leaf data")
        .require_grad()
        .expect("training device supports require_grad")
}

/// Training-device rank-two leaf marked for gradient retention.
fn leaf2(data: &[f32], rows: usize, cols: usize) -> Tensor<2> {
    Tensor::from_data(data.to_vec(), shape2(rows, cols), &Cpu::training())
        .expect("valid leaf data")
        .require_grad()
        .expect("training device supports require_grad")
}

/// Training-device constant (no gradient retention).
fn constant1(data: &[f32]) -> Tensor<1> {
    Tensor::from_data(data.to_vec(), shape1(data.len()), &Cpu::training()).expect("constant")
}

/// Training-device rank-two constant (no gradient retention).
fn constant2(data: &[f32], rows: usize, cols: usize) -> Tensor<2> {
    Tensor::from_data(data.to_vec(), shape2(rows, cols), &Cpu::training()).expect("constant")
}

fn assert_close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len(), "gradient length mismatch");
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        let slack = TOL + TOL * a.abs().max(e.abs());
        assert!(
            (a - e).abs() <= slack,
            "element {i}: autodiff {a} versus expected {e} (slack {slack})"
        );
    }
}

/// Backpropagates `f` from a rank-one leaf and returns its gradient.
fn grad1<F: Fn(&Tensor<1>) -> Tensor<1>>(f: F, x: &[f32]) -> Vec<f32> {
    let leaf = leaf1(x);
    let out = f(&leaf);
    let grads = out.backward().expect("tracked output is differentiable");
    leaf.grad(&grads)
        .expect("leaf retained a gradient")
        .to_vec()
        .expect("gradient readback")
}

/// Central finite-difference gradient of a scalar-valued rank-one function.
fn finite_diff1<F: Fn(&Tensor<1>) -> Tensor<1>>(f: F, x: &[f32]) -> Vec<f32> {
    x.iter()
        .enumerate()
        .map(|(i, xi)| {
            let mut xp = x.to_vec();
            let mut xm = x.to_vec();
            xp[i] = xi + H;
            xm[i] = xi - H;
            let yp = f(&Tensor::from_data(xp, shape1(x.len()), &Cpu::training()).expect("fd +"))
                .item()
                .expect("scalar forward");
            let ym = f(&Tensor::from_data(xm, shape1(x.len()), &Cpu::training()).expect("fd -"))
                .item()
                .expect("scalar forward");
            (yp - ym) / (2.0 * H)
        })
        .collect()
}

#[test]
fn scalar_gradient_of_cubic() {
    // f(x) = x^3, f'(x) = 3x^2 = 12 at x = 2.
    let x = leaf1(&[2.0]);
    let x2 = ops::mul(&x, &x).expect("mul");
    let x3 = ops::mul(&x, &x2).expect("mul");
    let grads = x3.backward().expect("differentiable");
    let g = x
        .grad(&grads)
        .expect("grad present")
        .to_vec()
        .expect("readback");
    assert_close(&g, &[12.0]);
}

#[test]
fn scalar_gradient_matches_finite_difference() {
    let x = &[1.7];
    let f = |t: &Tensor<1>| {
        let squared = ops::mul(t, t).expect("mul");
        ops::mul(t, &squared).expect("mul")
    };
    let ad = grad1(f, x);
    let fd = finite_diff1(f, x);
    assert_close(&ad, &fd);
}

#[test]
fn vector_gradient_of_sum_of_squares() {
    // f(x) = sum(x^2), grad = 2x.
    let x = leaf1(&[1.0, -2.0, 3.0]);
    let x2 = ops::mul(&x, &x).expect("mul");
    let s = ops::sum_all(&x2).expect("sum_all");
    let grads = s.backward().expect("differentiable");
    let g = x
        .grad(&grads)
        .expect("grad present")
        .to_vec()
        .expect("readback");
    assert_close(&g, &[2.0, -4.0, 6.0]);
}

#[test]
fn vector_gradient_matches_finite_difference() {
    let x = &[0.5, -1.25, 2.0, 4.0];
    let f = |t: &Tensor<1>| {
        let shifted = ops::add(t, &ops::sigmoid(t)).expect("add");
        let product = ops::mul(t, &shifted).expect("mul");
        ops::sum_all(&product).expect("sum_all")
    };
    let ad = grad1(f, x);
    let fd = finite_diff1(f, x);
    assert_close(&ad, &fd);
}

#[test]
fn matmul_gradients_match_finite_difference() {
    // dL/dA and dL/dB for L = sum(A @ B), verified against central
    // differences per matrix element.
    let a_data = [0.5, -1.0, 2.0, 1.5, 0.25, -0.75];
    let b_data = [1.0, 2.0, -0.5, 0.25, 1.5, -1.0];
    let rows = 2usize;
    let inner = 3usize;
    let cols = 2usize;

    let a = leaf2(&a_data, rows, inner);
    let b = leaf2(&b_data, inner, cols);
    let product = ops::matmul(&a, &b).expect("compatible shapes");
    let out = ops::sum_all(&product).expect("sum_all");
    let grads = out.backward().expect("differentiable");
    let ga = a
        .grad(&grads)
        .expect("grad for A")
        .to_vec()
        .expect("readback");
    let gb = b
        .grad(&grads)
        .expect("grad for B")
        .to_vec()
        .expect("readback");

    let forward = |a_in: &[f32], b_in: &[f32]| -> f32 {
        let av = constant2(a_in, rows, inner);
        let bv = constant2(b_in, inner, cols);
        let product = ops::matmul(&av, &bv).expect("compatible shapes");
        ops::sum_all(&product)
            .expect("sum_all")
            .item()
            .expect("scalar forward")
    };

    let fd_a: Vec<f32> = a_data
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let mut ap = a_data;
            let mut am = a_data;
            ap[i] = v + H;
            am[i] = v - H;
            (forward(&ap, &b_data) - forward(&am, &b_data)) / (2.0 * H)
        })
        .collect();
    let fd_b: Vec<f32> = b_data
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let mut bp = b_data;
            let mut bm = b_data;
            bp[i] = v + H;
            bm[i] = v - H;
            (forward(&a_data, &bp) - forward(&a_data, &bm)) / (2.0 * H)
        })
        .collect();

    assert_close(&ga, &fd_a);
    assert_close(&gb, &fd_b);
}

#[test]
fn sigmoid_gradient_matches_analytic_form() {
    // sigma'(x) = sigma(x) * (1 - sigma(x)).
    let x = leaf1(&[0.5, -1.0, 2.0]);
    let y = ops::sigmoid(&x);
    let sum = ops::sum_all(&y).expect("sum_all");
    let grads = sum.backward().expect("differentiable");
    let g = x
        .grad(&grads)
        .expect("grad present")
        .to_vec()
        .expect("readback");

    let analytic: Vec<f32> = [0.5f32, -1.0, 2.0]
        .iter()
        .map(|v| {
            let s = 1.0 / (1.0 + (-v).exp());
            s * (1.0 - s)
        })
        .collect();
    assert_close(&g, &analytic);
}

#[test]
fn tanh_gradient_matches_analytic_form() {
    // tanh'(x) = 1 - tanh(x)^2.
    let x = leaf1(&[-0.75, 0.25, 1.5]);
    let y = ops::tanh(&x);
    let sum = ops::sum_all(&y).expect("sum_all");
    let grads = sum.backward().expect("differentiable");
    let g = x
        .grad(&grads)
        .expect("grad present")
        .to_vec()
        .expect("readback");

    let analytic: Vec<f32> = [-0.75f32, 0.25, 1.5]
        .iter()
        .map(|v| {
            let t = v.tanh();
            1.0 - t * t
        })
        .collect();
    assert_close(&g, &analytic);
}

#[test]
fn relu_gradient_matches_finite_difference() {
    // Piecewise-linear derivative, evaluated away from the kink at 0.
    let x = &[-1.5f32, 2.0, -0.25, 3.0];
    let f = |t: &Tensor<1>| ops::sum_all(&ops::relu(t)).expect("sum_all");
    let ad = grad1(f, x);
    let fd = finite_diff1(f, x);
    assert_close(&ad, &[0.0, 1.0, 0.0, 1.0]);
    assert_close(&ad, &fd);
}

#[test]
fn gelu_gradient_matches_finite_difference() {
    let x = &[-1.2f32, -0.3, 0.4, 1.8];
    let f = |t: &Tensor<1>| ops::sum_all(&ops::gelu(t)).expect("sum_all");
    let ad = grad1(f, x);
    let fd = finite_diff1(f, x);
    assert_close(&ad, &fd);
}

#[test]
fn mse_gradient_matches_analytic_form() {
    // dL/dp = 2 * (p - t) / n with t = 0, n = 3.
    let p = leaf1(&[1.0, 2.0, 3.0]);
    let t = constant1(&[0.0, 0.0, 0.0]);
    let loss = mi_tensor::mse(&p, &t).expect("paired shapes");
    let grads = loss.backward().expect("differentiable");
    let g = p
        .grad(&grads)
        .expect("grad present")
        .to_vec()
        .expect("readback");
    assert_close(&g, &[2.0 / 3.0, 4.0 / 3.0, 2.0]);
}

#[test]
fn mse_gradient_matches_finite_difference() {
    let p = &[0.3f32, -1.1, 2.4];
    let t_data = [0.5f32, 0.0, -1.0];
    let f = |pred: &Tensor<1>| {
        let t = constant1(&t_data);
        mi_tensor::mse(pred, &t).expect("paired shapes")
    };
    let ad = grad1(f, p);
    let fd = finite_diff1(f, p);
    assert_close(&ad, &fd);
}

#[test]
fn mae_gradient_matches_analytic_form() {
    // dL/dp = sign(p - t) / n with t = 0, n = 3.
    let p = leaf1(&[1.0, -2.0, 3.0]);
    let t = constant1(&[0.0, 0.0, 0.0]);
    let loss = mi_tensor::mae(&p, &t).expect("paired shapes");
    let grads = loss.backward().expect("differentiable");
    let g = p
        .grad(&grads)
        .expect("grad present")
        .to_vec()
        .expect("readback");
    assert_close(&g, &[1.0 / 3.0, -1.0 / 3.0, 1.0 / 3.0]);
}

#[test]
fn huber_gradient_matches_analytic_form() {
    // Quadratic branch for |p| <= 1, linear branch with slope delta = 1
    // beyond it; n = 4.
    let p = leaf1(&[0.5, 2.0, -1.5, 3.0]);
    let t = constant1(&[0.0, 0.0, 0.0, 0.0]);
    let loss = mi_tensor::huber_loss(&p, &t, 1.0).expect("paired shapes");
    let grads = loss.backward().expect("differentiable");
    let g = p
        .grad(&grads)
        .expect("grad present")
        .to_vec()
        .expect("readback");
    assert_close(&g, &[0.125, 0.25, -0.25, 0.25]);
}

#[test]
fn huber_gradient_matches_finite_difference() {
    let p = &[0.2f32, -0.6, 1.4, -2.5];
    let t_data = [0.1f32, 0.2, -0.1, 0.0];
    let f = |pred: &Tensor<1>| {
        let t = constant1(&t_data);
        mi_tensor::huber_loss(pred, &t, 1.0).expect("paired shapes")
    };
    let ad = grad1(f, p);
    let fd = finite_diff1(f, p);
    assert_close(&ad, &fd);
}

#[test]
fn composite_expression_matches_finite_difference() {
    // A multi-operation expression exercising arithmetic, activation, and
    // reduction together.
    let x = &[0.8f32, -1.6, 2.2];
    let f = |t: &Tensor<1>| {
        let gated = ops::mul(&ops::relu(t), &ops::tanh(t)).expect("mul");
        let residual = ops::mul(t, t).expect("mul");
        let combined = ops::add(&gated, &residual).expect("add");
        ops::mean_all(&combined).expect("mean_all")
    };
    let ad = grad1(f, x);
    let fd = finite_diff1(f, x);
    assert_close(&ad, &fd);
}

#[test]
fn constant_leaf_has_no_gradient() {
    let x = leaf1(&[1.0, 2.0]);
    let c = constant1(&[3.0, 4.0]);
    let y = ops::mul(&x, &c).expect("mul");
    let grads = y.backward().expect("differentiable");
    assert!(x.grad(&grads).is_some());
    assert!(c.grad(&grads).is_none(), "constants retain no gradient");
}

#[test]
fn require_grad_fails_on_inference_device() {
    let t =
        Tensor::<1>::from_data(vec![1.0], shape1(1), &Cpu::inference()).expect("inference tensor");
    assert!(matches!(
        t.require_grad(),
        Err(TensorError::AutodiffUnavailable)
    ));
}

#[test]
fn backward_fails_on_untracked_output() {
    // Both tensors are constants on the inference device: no autodiff
    // context exists, so backward must error instead of panicking.
    let a = Tensor::<1>::from_data(vec![1.0, 2.0], shape1(2), &Cpu::inference()).expect("a");
    let b = Tensor::<1>::from_data(vec![3.0, 4.0], shape1(2), &Cpu::inference()).expect("b");
    let y = ops::mul(&a, &b).expect("mul");
    assert!(!y.is_tracked());
    assert!(matches!(
        y.backward(),
        Err(TensorError::AutodiffUnavailable)
    ));
}

#[test]
fn detach_stops_gradient_flow() {
    let x = leaf1(&[1.0, 2.0]);
    let y = ops::mul(&x, &x).expect("mul");
    let d = y.detach();
    assert!(!d.is_tracked(), "detached tensor leaves the graph");
}
