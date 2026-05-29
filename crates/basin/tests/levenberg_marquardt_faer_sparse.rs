#![cfg(feature = "faer")]

use basin::problems::SparseLeastSquares;
use basin::{BasicState, Executor, LevenbergMarquardt, TerminationReason};
use faer::Col;
use faer::sparse::{SparseColMat, Triplet};

type FaerSparseLeastSquares = SparseLeastSquares<SparseColMat<usize, f64>, Col<f64>>;

/// Mirror of the GN sparse fixture: 6×3 design with `b = A·[1,2,3]` so
/// the closed-form least-squares minimum has zero residual at
/// `x* = [1, 2, 3]`.
fn fixture() -> (FaerSparseLeastSquares, Col<f64>) {
    let triplets = [
        Triplet::new(0_usize, 0_usize, 1.0),
        Triplet::new(1, 1, 1.0),
        Triplet::new(2, 2, 1.0),
        Triplet::new(3, 0, 1.0),
        Triplet::new(3, 1, 1.0),
        Triplet::new(4, 0, 1.0),
        Triplet::new(4, 2, 1.0),
        Triplet::new(5, 1, 1.0),
        Triplet::new(5, 2, 1.0),
    ];
    let a =
        SparseColMat::<usize, f64>::try_new_from_triplets(6, 3, &triplets).expect("triplets valid");
    let b = Col::<f64>::from_fn(6, |i| [1.0, 2.0, 3.0, 3.0, 4.0, 5.0][i]);
    let initial = Col::<f64>::zeros(3);
    (SparseLeastSquares::new(a, b), initial)
}

#[test]
fn levenberg_marquardt_converges_on_sparse_linear_regression() {
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
    // Exercises the faer-sparse Marquardt-damping path end-to-end —
    // verifies the col_ptr/row_idx diagonal extraction (`MatDiagonal`)
    // and the `μ·D` `add_diagonal_vector_in_place` value mutation
    // compose correctly with the sparse Cholesky solve.
    let (problem, initial) = fixture();
    let result = Executor::new(
        problem,
        LevenbergMarquardt::new().tol_grad(1e-12),
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
