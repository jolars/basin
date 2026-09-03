#![cfg(feature = "nalgebra_all")]

use crate::backend_aliases::nalgebra::DVector;
use crate::backend_aliases::nalgebra_sparse::{CooMatrix, CscMatrix};
use basin::problems::SparseLeastSquaresBoxed;
use basin::{Executor, NllsState, TerminationReason, Trf};

/// 6×3 design: identity stack + pairwise-sum rows. With `b = A·[1, 2, 3]`,
/// the closed-form unconstrained least-squares minimum lands exactly at
/// `x* = [1, 2, 3]`. Bounds are passed in by the caller so the test
/// chooses interior-vs-binding scenarios.
fn fixture(
    lower: DVector<f64>,
    upper: DVector<f64>,
) -> (
    SparseLeastSquaresBoxed<CscMatrix<f64>, DVector<f64>>,
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
    let initial = DVector::from_vec(vec![0.5, 0.5, 0.5]);
    (SparseLeastSquaresBoxed::new(a, b, lower, upper), initial)
}

#[test]
fn trf_with_slack_bounds_reaches_unconstrained_min() {
    // Bounds wide enough to contain `x* = [1, 2, 3]` strictly.
    let (problem, initial) = fixture(
        DVector::from_vec(vec![-10.0, -10.0, -10.0]),
        DVector::from_vec(vec![10.0, 10.0, 10.0]),
    );
    let result = Executor::new(problem, Trf::new(), NllsState::new(initial))
        .max_iter(50)
        .run()
        .unwrap();

    assert_eq!(result.reason, TerminationReason::SolverConverged);
    assert!(
        (result.param()[0] - 1.0).abs() < 1e-5,
        "x[0] = {}",
        result.param()[0]
    );
    assert!(
        (result.param()[1] - 2.0).abs() < 1e-5,
        "x[1] = {}",
        result.param()[1]
    );
    assert!(
        (result.param()[2] - 3.0).abs() < 1e-5,
        "x[2] = {}",
        result.param()[2]
    );
}

#[test]
fn trf_with_binding_upper_bound_converges_to_face() {
    // Upper bound on x[2] is 1.5, below the unconstrained value 3.
    // x[0] = 1 and x[1] = 2 are still interior. The constrained
    // optimum binds the upper bound on x[2].
    let (problem, initial) = fixture(
        DVector::from_vec(vec![-10.0, -10.0, -10.0]),
        DVector::from_vec(vec![10.0, 10.0, 1.5]),
    );
    let result = Executor::new(problem, Trf::new(), NllsState::new(initial))
        .max_iter(200)
        .run()
        .unwrap();

    assert_eq!(result.reason, TerminationReason::SolverConverged);
    // x[2] should be at (or just inside) the upper bound 1.5.
    assert!(
        result.param()[2] <= 1.5 && result.param()[2] >= 1.5 - 1e-3,
        "x[2] = {} should bind the upper bound 1.5",
        result.param()[2]
    );
}

#[test]
fn trf_emits_solver_converged_via_scaled_first_order_optimality() {
    let (problem, initial) = fixture(
        DVector::from_vec(vec![-10.0, -10.0, -10.0]),
        DVector::from_vec(vec![10.0, 10.0, 10.0]),
    );
    let result = Executor::new(problem, Trf::new(), NllsState::new(initial))
        .max_iter(50)
        .run()
        .unwrap();
    assert_eq!(result.reason, TerminationReason::SolverConverged);
}

#[path = "support/backend_aliases.rs"]
mod backend_aliases;
