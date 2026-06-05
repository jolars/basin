#![cfg(feature = "nalgebra")]

use basin::problems::SparseLeastSquares;
use basin::{BasicState, Executor, LevenbergMarquardt, TerminationReason};
use nalgebra::DVector;
use nalgebra_sparse::{CooMatrix, CscMatrix};

/// Mirror of the GN sparse fixture: 6×3 design with `b = A·[1,2,3]` so
/// the closed-form least-squares minimum has zero residual at
/// `x* = [1, 2, 3]`.
fn fixture() -> (
    SparseLeastSquares<CscMatrix<f64>, DVector<f64>>,
    DVector<f64>,
) {
    let mut coo = CooMatrix::<f64>::new(6, 3);
    coo.push(0, 0, 1.0);
    coo.push(1, 1, 1.0);
    coo.push(2, 2, 1.0);
    coo.push(3, 0, 1.0);
    coo.push(3, 1, 1.0);
    coo.push(4, 0, 1.0);
    coo.push(4, 2, 1.0);
    coo.push(5, 1, 1.0);
    coo.push(5, 2, 1.0);
    let a = CscMatrix::from(&coo);
    let b = DVector::from_vec(vec![1.0, 2.0, 3.0, 3.0, 4.0, 5.0]);
    let initial = DVector::zeros(3);
    (SparseLeastSquares::new(a, b), initial)
}

#[test]
fn levenberg_marquardt_converges_on_sparse_linear_regression() {
    // Sparse LM on a linear residual: damping shifts the first step
    // off the closed-form minimum, but the gain ratio is high so μ
    // shrinks rapidly and the iterate lands at x* in a handful of
    // accepted steps.
    let (problem, initial) = fixture();
    let result = Executor::new(problem, LevenbergMarquardt::new(), BasicState::new(initial))
        .max_iter(50)
        .run()
        .unwrap();

    assert_eq!(result.reason, TerminationReason::SolverConverged);
    assert!(result.cost() < 1e-15, "cost = {}", result.cost());
    assert!(
        (result.param()[0] - 1.0).abs() < 1e-7,
        "x[0] = {}",
        result.param()[0]
    );
    assert!(
        (result.param()[1] - 2.0).abs() < 1e-7,
        "x[1] = {}",
        result.param()[1]
    );
    assert!(
        (result.param()[2] - 3.0).abs() < 1e-7,
        "x[2] = {}",
        result.param()[2]
    );
}

#[test]
fn levenberg_marquardt_handles_sparse_diagonal_damping() {
    // Exercises the sparse Marquardt-damping path end-to-end via a
    // tighter tol_grad — verifies the CSC diagonal extraction
    // (`MatDiagonal`) and the `μ·D` `add_diagonal_vector_in_place`
    // pattern-hit assertion compose correctly with the Cholesky solve.
    let (problem, initial) = fixture();
    let result = Executor::new(
        problem,
        LevenbergMarquardt::new().with_tol_grad(1e-12),
        BasicState::new(initial),
    )
    .max_iter(100)
    .run()
    .unwrap();

    assert_eq!(result.reason, TerminationReason::SolverConverged);
    assert!(result.cost() < 1e-20, "cost = {}", result.cost());
}

#[test]
fn levenberg_marquardt_emits_solver_converged_via_first_order_optimality() {
    let (problem, initial) = fixture();
    let result = Executor::new(problem, LevenbergMarquardt::new(), BasicState::new(initial))
        .max_iter(50)
        .run()
        .unwrap();
    assert_eq!(result.reason, TerminationReason::SolverConverged);
}
