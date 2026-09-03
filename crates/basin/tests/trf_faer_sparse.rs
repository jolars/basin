#![cfg(feature = "faer_all")]

use crate::backend_aliases::faer::Col;
use crate::backend_aliases::faer::sparse::{SparseColMat, Triplet};
use basin::problems::SparseLeastSquaresBoxed;
use basin::{Executor, NllsState, TerminationReason, Trf};

type FaerSparseLeastSquaresBoxed =
    SparseLeastSquaresBoxed<SparseColMat<usize, f64>, Col<f64>>;

fn fixture(
    lower: Col<f64>,
    upper: Col<f64>,
) -> (FaerSparseLeastSquaresBoxed, Col<f64>) {
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
    let a = SparseColMat::<usize, f64>::try_new_from_triplets(6, 3, &triplets)
        .expect("triplets valid");
    let b = Col::<f64>::from_fn(6, |i| [1.0, 2.0, 3.0, 3.0, 4.0, 5.0][i]);
    let initial = Col::<f64>::from_fn(3, |i| [0.5, 0.5, 0.5][i]);
    (SparseLeastSquaresBoxed::new(a, b, lower, upper), initial)
}

#[test]
fn trf_with_slack_bounds_reaches_unconstrained_min() {
    let (problem, initial) = fixture(
        Col::<f64>::from_fn(3, |_| -10.0),
        Col::<f64>::from_fn(3, |_| 10.0),
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
    let (problem, initial) = fixture(
        Col::<f64>::from_fn(3, |_| -10.0),
        Col::<f64>::from_fn(3, |i| if i == 2 { 1.5 } else { 10.0 }),
    );
    let result = Executor::new(problem, Trf::new(), NllsState::new(initial))
        .max_iter(200)
        .run()
        .unwrap();

    assert_eq!(result.reason, TerminationReason::SolverConverged);
    assert!(
        result.param()[2] <= 1.5 && result.param()[2] >= 1.5 - 1e-3,
        "x[2] = {} should bind the upper bound 1.5",
        result.param()[2]
    );
}

#[test]
fn trf_emits_solver_converged_via_scaled_first_order_optimality() {
    let (problem, initial) = fixture(
        Col::<f64>::from_fn(3, |_| -10.0),
        Col::<f64>::from_fn(3, |_| 10.0),
    );
    let result = Executor::new(problem, Trf::new(), NllsState::new(initial))
        .max_iter(50)
        .run()
        .unwrap();
    assert_eq!(result.reason, TerminationReason::SolverConverged);
}

#[path = "support/backend_aliases.rs"]
mod backend_aliases;
