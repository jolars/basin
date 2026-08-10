#![cfg(feature = "faer")]

use basin::problems::BoothBoxed;
use basin::{
    BoundedCmaEs, CmaEsState, CmaEsTolerance, Executor, PopulationState,
    StepOutcome, TerminationReason,
};
use faer::{Col, Mat};

/// Same seed → same trajectory on the faer backend's bounded variant.
/// Same-backend reproducibility is the load-bearing contract; the
/// nalgebra and faer trajectories may differ (different
/// eigendecomposition routines).
#[test]
fn same_seed_yields_identical_trajectory() {
    let lower = Col::<f64>::from_fn(2, |_| -5.0);
    let upper = Col::<f64>::from_fn(2, |_| 5.0);
    let m0 = Col::<f64>::from_fn(2, |_| 0.0);

    let result_a = Executor::new(
        BoothBoxed::<Col<f64>>::new(lower.clone(), upper.clone()),
        BoundedCmaEs::<Col<f64>, Mat<f64>>::new(42),
        CmaEsState::<Col<f64>, Mat<f64>>::new(m0.clone(), 0.5),
    )
    .max_iter(30)
    .run()
    .unwrap();

    let result_b = Executor::new(
        BoothBoxed::<Col<f64>>::new(lower, upper),
        BoundedCmaEs::<Col<f64>, Mat<f64>>::new(42),
        CmaEsState::<Col<f64>, Mat<f64>>::new(m0, 0.5),
    )
    .max_iter(30)
    .run()
    .unwrap();

    assert_eq!(result_a.cost(), result_b.cost());
    let (a, b) = (result_a.param(), result_b.param());
    assert_eq!(a.nrows(), b.nrows());
    for i in 0..a.nrows() {
        assert_eq!(a[i], b[i]);
    }
}

/// Slack bounds: the unconstrained Booth minimum (1, 3) is interior to
/// [-5, 5]² on the faer backend.
#[test]
fn slack_bounds_recover_unconstrained_minimum() {
    let lower = Col::<f64>::from_fn(2, |_| -5.0);
    let upper = Col::<f64>::from_fn(2, |_| 5.0);
    let m0 = Col::<f64>::from_fn(2, |_| 0.0);

    let result = Executor::new(
        BoothBoxed::<Col<f64>>::new(lower, upper),
        BoundedCmaEs::<Col<f64>, Mat<f64>>::new(7),
        CmaEsState::<Col<f64>, Mat<f64>>::new(m0, 0.5),
    )
    .max_iter(400)
    .run()
    .unwrap();

    let p = result.param();
    assert!(
        (p[0] - 1.0).abs() < 1e-2 && (p[1] - 3.0).abs() < 1e-2,
        "iterate = ({}, {}), expected ≈ (1, 3)",
        p[0],
        p[1]
    );
}

/// Tight [-1, 1]² bounds: constrained minimum is the box corner (1, 1).
#[test]
fn tight_bounds_converge_to_box_corner() {
    let lower = Col::<f64>::from_fn(2, |_| -1.0);
    let upper = Col::<f64>::from_fn(2, |_| 1.0);
    let m0 = Col::<f64>::from_fn(2, |_| 0.0);

    let result = Executor::new(
        BoothBoxed::<Col<f64>>::new(lower, upper),
        BoundedCmaEs::<Col<f64>, Mat<f64>>::new(11),
        CmaEsState::<Col<f64>, Mat<f64>>::new(m0, 0.3),
    )
    .max_iter(800)
    .run()
    .unwrap();

    let p = result.param();
    assert!(
        (p[0] - 1.0).abs() < 1e-2 && (p[1] - 1.0).abs() < 1e-2,
        "iterate = ({}, {}), expected ≈ (1, 1) box corner",
        p[0],
        p[1]
    );
}

/// Wildly infeasible initial mean: starting at (10, 10) with bounds
/// [-1, 1]² should still converge.
#[test]
fn infeasible_initial_mean_converges_to_box_corner() {
    let lower = Col::<f64>::from_fn(2, |_| -1.0);
    let upper = Col::<f64>::from_fn(2, |_| 1.0);
    let m0 = Col::<f64>::from_fn(2, |_| 10.0);

    let result = Executor::new(
        BoothBoxed::<Col<f64>>::new(lower, upper),
        BoundedCmaEs::<Col<f64>, Mat<f64>>::new(5),
        CmaEsState::<Col<f64>, Mat<f64>>::new(m0, 0.3),
    )
    .max_iter(800)
    .run()
    .unwrap();

    let p = result.param();
    assert!(
        (p[0] - 1.0).abs() < 1e-2 && (p[1] - 1.0).abs() < 1e-2,
        "iterate = ({}, {}), expected ≈ (1, 1) box corner from infeasible start",
        p[0],
        p[1]
    );
}

/// TolX termination still fires under bounded CMA-ES on the faer
/// backend.
#[test]
fn slack_bounds_terminate_solver_converged_on_tol_x() {
    let lower = Col::<f64>::from_fn(2, |_| -10.0);
    let upper = Col::<f64>::from_fn(2, |_| 10.0);
    let m0 = Col::<f64>::from_fn(2, |_| 0.0);

    let result = Executor::new(
        BoothBoxed::<Col<f64>>::new(lower, upper),
        BoundedCmaEs::<Col<f64>, Mat<f64>>::new(11),
        CmaEsState::<Col<f64>, Mat<f64>>::new(m0, 0.3),
    )
    .terminate_on(CmaEsTolerance::new(1e-12 * 0.3))
    .max_iter(2000)
    .run()
    .unwrap();

    assert_eq!(result.reason, TerminationReason::CmaEsTolerance);
}

/// `PopulationState` invariants survive iteration on faer.
#[test]
fn population_invariants_hold_after_iteration() {
    let lower = Col::<f64>::from_fn(2, |_| -1.0);
    let upper = Col::<f64>::from_fn(2, |_| 1.0);
    let m0 = Col::<f64>::from_fn(2, |i| if i == 0 { 0.3 } else { 0.4 });
    let lambda = 12;

    let mut stepper = Executor::new(
        BoothBoxed::<Col<f64>>::new(lower, upper),
        BoundedCmaEs::<Col<f64>, Mat<f64>>::new(1234).with_lambda(lambda),
        CmaEsState::<Col<f64>, Mat<f64>>::new(m0, 0.5),
    )
    .max_iter(10)
    .into_stepper()
    .unwrap();

    for _ in 0..10 {
        let StepOutcome::Continue = stepper.step().unwrap() else {
            break;
        };
        let state = stepper.state();
        assert_eq!(state.candidates().len(), lambda);
        assert_eq!(state.costs().len(), lambda);
        for window in state.costs().windows(2) {
            assert!(window[0] <= window[1]);
        }
    }
}
