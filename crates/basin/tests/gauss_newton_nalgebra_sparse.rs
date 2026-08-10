#![cfg(feature = "nalgebra")]

use basin::problems::SparseLeastSquares;
use basin::{Executor, GaussNewton, NllsState, TerminationReason};
use nalgebra::DVector;
use nalgebra_sparse::{CooMatrix, CscMatrix};

/// 6×3 sparse design with `b = A·[1,2,3]` so the closed-form
/// least-squares minimum has zero residual at `x* = [1, 2, 3]`.
/// Mirrors the faer-sparse fixture for cross-backend consistency.
fn fixture() -> (
    SparseLeastSquares<CscMatrix<f64>, DVector<f64>>,
    DVector<f64>,
) {
    let mut coo = CooMatrix::<f64>::new(6, 3);
    // Top: identity I₃
    coo.push(0, 0, 1.0);
    coo.push(1, 1, 1.0);
    coo.push(2, 2, 1.0);
    // Bottom: pairwise-sum rows (x0+x1, x0+x2, x1+x2)
    coo.push(3, 0, 1.0);
    coo.push(3, 1, 1.0);
    coo.push(4, 0, 1.0);
    coo.push(4, 2, 1.0);
    coo.push(5, 1, 1.0);
    coo.push(5, 2, 1.0);
    let a = CscMatrix::from(&coo);
    // b = A · [1, 2, 3]: I₃ part → [1, 2, 3]; pairwise → [3, 4, 5].
    let b = DVector::from_vec(vec![1.0, 2.0, 3.0, 3.0, 4.0, 5.0]);
    let initial = DVector::zeros(3);
    (SparseLeastSquares::new(a, b), initial)
}

#[test]
fn gauss_newton_converges_on_sparse_linear_regression() {
    let (problem, initial) = fixture();
    let result =
        Executor::new(problem, GaussNewton::new(), NllsState::new(initial))
            .max_iter(20)
            .run()
            .unwrap();

    assert_eq!(result.reason, TerminationReason::SolverConverged);
    assert!(result.cost() < 1e-20, "cost = {}", result.cost());
    assert!(
        (result.param()[0] - 1.0).abs() < 1e-9,
        "x[0] = {}",
        result.param()[0]
    );
    assert!(
        (result.param()[1] - 2.0).abs() < 1e-9,
        "x[1] = {}",
        result.param()[1]
    );
    assert!(
        (result.param()[2] - 3.0).abs() < 1e-9,
        "x[2] = {}",
        result.param()[2]
    );
}

#[test]
fn gauss_newton_single_step_matches_closed_form() {
    // For linear residuals the GN model is exact, so one full step
    // from x₀ = 0 lands on the closed-form least-squares solution
    // x* = (AᵀA)⁻¹Aᵀb = [1, 2, 3].
    let (problem, initial) = fixture();
    let result =
        Executor::new(problem, GaussNewton::new(), NllsState::new(initial))
            .max_iter(1)
            .run()
            .unwrap();

    assert_eq!(result.reason, TerminationReason::MaxIter);
    assert_eq!(result.iter(), 1);
    assert!(
        (result.param()[0] - 1.0).abs() < 1e-10,
        "x[0] = {}",
        result.param()[0]
    );
    assert!(
        (result.param()[1] - 2.0).abs() < 1e-10,
        "x[1] = {}",
        result.param()[1]
    );
    assert!(
        (result.param()[2] - 3.0).abs() < 1e-10,
        "x[2] = {}",
        result.param()[2]
    );
}

#[test]
fn gauss_newton_emits_solver_converged_via_first_order_optimality() {
    let (problem, initial) = fixture();
    let result =
        Executor::new(problem, GaussNewton::new(), NllsState::new(initial))
            .max_iter(50)
            .run()
            .unwrap();
    assert_eq!(result.reason, TerminationReason::SolverConverged);
}
