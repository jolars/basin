//! Bounded CMA-ES convergence over the default `Vec<f64>` backend.
//!
//! Mirrors the box-constrained subset of `tests/bounded_cma_es_nalgebra.rs`,
//! confirming the generic `Solver` impl runs on the hand-rolled `DenseMatrix`
//! covariance, including the extra `MatDiagonal` (diagonal extraction) the
//! adaptive boundary penalty needs.

use basin::problems::BoothBoxed;
use basin::{
    BoundedCmaEs, CmaEsState, CmaEsTolerance, DenseMatrix, Executor, PopulationState, StepOutcome,
    TerminationReason,
};

/// Same seed → same trajectory on the bounded `Vec<f64>` path.
#[test]
fn same_seed_yields_identical_trajectory() {
    let lower = vec![-5.0, -5.0];
    let upper = vec![5.0, 5.0];
    let m0 = vec![0.0, 0.0];

    let result_a = Executor::new(
        BoothBoxed::<Vec<f64>>::new(lower.clone(), upper.clone()),
        BoundedCmaEs::<Vec<f64>, DenseMatrix>::new(42),
        CmaEsState::<Vec<f64>, DenseMatrix>::new(m0.clone(), 0.5),
    )
    .max_iter(30)
    .run()
    .unwrap();

    let result_b = Executor::new(
        BoothBoxed::<Vec<f64>>::new(lower, upper),
        BoundedCmaEs::<Vec<f64>, DenseMatrix>::new(42),
        CmaEsState::<Vec<f64>, DenseMatrix>::new(m0, 0.5),
    )
    .max_iter(30)
    .run()
    .unwrap();

    assert_eq!(result_a.cost(), result_b.cost());
    assert_eq!(result_a.param(), result_b.param());
}

/// Slack [-5, 5]² bounds: the unconstrained Booth minimum (1, 3) is interior,
/// so the bounded solver must reach it on the default backend.
#[test]
fn slack_bounds_recover_unconstrained_minimum() {
    let lower = vec![-5.0, -5.0];
    let upper = vec![5.0, 5.0];
    let m0 = vec![0.0, 0.0];

    let result = Executor::new(
        BoothBoxed::<Vec<f64>>::new(lower, upper),
        BoundedCmaEs::<Vec<f64>, DenseMatrix>::new(7),
        CmaEsState::<Vec<f64>, DenseMatrix>::new(m0, 0.5),
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

/// Tight [-1, 1]² bounds: the constrained Booth minimum is the box corner
/// (1, 1). The penalty must steer the search toward the active corner.
#[test]
fn tight_bounds_converge_to_box_corner() {
    let lower = vec![-1.0, -1.0];
    let upper = vec![1.0, 1.0];
    let m0 = vec![0.0, 0.0];

    let result = Executor::new(
        BoothBoxed::<Vec<f64>>::new(lower, upper),
        BoundedCmaEs::<Vec<f64>, DenseMatrix>::new(11),
        CmaEsState::<Vec<f64>, DenseMatrix>::new(m0, 0.3),
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

/// Wildly infeasible initial mean at (10, 10) with bounds [-1, 1]² should
/// still converge: init-time mean projection plus the adaptive penalty pull
/// the distribution back into feasibility.
#[test]
fn infeasible_initial_mean_converges_to_box_corner() {
    let lower = vec![-1.0, -1.0];
    let upper = vec![1.0, 1.0];
    let m0 = vec![10.0, 10.0];

    let result = Executor::new(
        BoothBoxed::<Vec<f64>>::new(lower, upper),
        BoundedCmaEs::<Vec<f64>, DenseMatrix>::new(5),
        CmaEsState::<Vec<f64>, DenseMatrix>::new(m0, 0.3),
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

/// `with_stds(ones)` reproduces the isotropic default bit-for-bit on the
/// bounded `Vec<f64>` path: the boundary penalty reads `σ² · diag(C)`, which
/// is unchanged when `C = diag(1²) = I`.
#[test]
fn with_stds_ones_matches_default() {
    let lower = vec![-5.0, -5.0];
    let upper = vec![5.0, 5.0];
    let m0 = vec![0.0, 0.0];
    let ones = vec![1.0, 1.0];

    let default = Executor::new(
        BoothBoxed::<Vec<f64>>::new(lower.clone(), upper.clone()),
        BoundedCmaEs::<Vec<f64>, DenseMatrix>::new(42),
        CmaEsState::<Vec<f64>, DenseMatrix>::new(m0.clone(), 0.5),
    )
    .max_iter(40)
    .run()
    .unwrap();

    let with_ones = Executor::new(
        BoothBoxed::<Vec<f64>>::new(lower, upper),
        BoundedCmaEs::<Vec<f64>, DenseMatrix>::new(42),
        CmaEsState::<Vec<f64>, DenseMatrix>::new(m0, 0.5).with_stds(ones),
    )
    .max_iter(40)
    .run()
    .unwrap();

    assert_eq!(default.cost(), with_ones.cost());
    assert_eq!(default.param(), with_ones.param());
    assert_eq!(default.reason, with_ones.reason);
}

/// Solver-internal TolX termination still fires when bounds are slack.
#[test]
fn slack_bounds_terminate_solver_converged_on_tol_x() {
    let lower = vec![-10.0, -10.0];
    let upper = vec![10.0, 10.0];
    let m0 = vec![0.0, 0.0];

    let result = Executor::new(
        BoothBoxed::<Vec<f64>>::new(lower, upper),
        BoundedCmaEs::<Vec<f64>, DenseMatrix>::new(11),
        CmaEsState::<Vec<f64>, DenseMatrix>::new(m0, 0.3),
    )
    .terminate_on(CmaEsTolerance::new(1e-12 * 0.3))
    .max_iter(2000)
    .run()
    .unwrap();

    assert_eq!(result.reason, TerminationReason::CmaEsTolerance);
}

/// `PopulationState` invariants survive iteration on the bounded `Vec<f64>`
/// path: `candidates`/`costs` stay parallel, length-λ, sorted-ascending (on
/// the penalized costs).
#[test]
fn population_invariants_hold_after_iteration() {
    let lower = vec![-1.0, -1.0];
    let upper = vec![1.0, 1.0];
    let m0 = vec![0.3, 0.4];
    let lambda = 12;

    let mut stepper = Executor::new(
        BoothBoxed::<Vec<f64>>::new(lower, upper),
        BoundedCmaEs::<Vec<f64>, DenseMatrix>::new(1234).with_lambda(lambda),
        CmaEsState::<Vec<f64>, DenseMatrix>::new(m0, 0.5),
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
