#![cfg(feature = "faer_all")]

use crate::backend_aliases::faer::Col;
use crate::backend_aliases::faer::sparse::{SparseColMat, Triplet};
use basin::problems::SparseLeastSquares;
use basin::{Executor, GaussNewton, NllsState, TerminationReason};

type FaerSparseLeastSquares =
    SparseLeastSquares<SparseColMat<usize, f64>, Col<f64>>;

/// 6×3 sparse design with `b = A·[1,2,3]` so the closed-form
/// least-squares minimum has zero residual at `x* = [1, 2, 3]`.
/// Sparsity pattern: an I₃ block stacked on a pairwise-sum block.
fn fixture() -> (FaerSparseLeastSquares, Col<f64>) {
    let triplets = [
        // Top: identity I₃
        Triplet::new(0_usize, 0_usize, 1.0),
        Triplet::new(1, 1, 1.0),
        Triplet::new(2, 2, 1.0),
        // Bottom: pairwise-sum rows (x0+x1, x0+x2, x1+x2)
        Triplet::new(3, 0, 1.0),
        Triplet::new(3, 1, 1.0),
        Triplet::new(4, 0, 1.0),
        Triplet::new(4, 2, 1.0),
        Triplet::new(5, 1, 1.0),
        Triplet::new(5, 2, 1.0),
    ];
    let a = SparseColMat::<usize, f64>::try_new_from_triplets(6, 3, &triplets)
        .expect("triplets valid");
    // b = A · [1, 2, 3]: I₃ part gives [1, 2, 3]; pairwise part gives
    // [1+2, 1+3, 2+3] = [3, 4, 5].
    let b = Col::<f64>::from_fn(6, |i| [1.0, 2.0, 3.0, 3.0, 4.0, 5.0][i]);
    let initial = Col::<f64>::zeros(3);
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
    // After the optimum is reached, ‖Jᵀ r‖_∞ = 0 < tol_grad and the
    // solver reports SolverConverged rather than running out of iters.
    let (problem, initial) = fixture();
    let result =
        Executor::new(problem, GaussNewton::new(), NllsState::new(initial))
            .max_iter(50)
            .run()
            .unwrap();
    assert_eq!(result.reason, TerminationReason::SolverConverged);
}

#[path = "support/backend_aliases.rs"]
mod backend_aliases;
