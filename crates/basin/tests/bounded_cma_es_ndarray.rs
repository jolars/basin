#![cfg(feature = "ndarray")]

use basin::problems::BoothBoxed;
use basin::{
    BoundedCmaEs, CmaEsState, CmaEsTolerance, Executor, PopulationState, StepOutcome,
    TerminationReason,
};
use ndarray::{Array1, Array2};

/// Same seed → same trajectory on the bounded variant. Reproducibility
/// is the load-bearing contract for the seeded RNG; this guards the
/// constant-RNG bug equally on the bounded path.
#[test]
fn same_seed_yields_identical_trajectory() {
    let lower = Array1::from_vec(vec![-5.0, -5.0]);
    let upper = Array1::from_vec(vec![5.0, 5.0]);
    let m0 = Array1::from_vec(vec![0.0, 0.0]);

    let result_a = Executor::new(
        BoothBoxed::<Array1<f64>>::new(lower.clone(), upper.clone()),
        BoundedCmaEs::<Array1<f64>, Array2<f64>>::new(42),
        CmaEsState::<Array1<f64>, Array2<f64>>::new(m0.clone(), 0.5),
    )
    .max_iter(30)
    .run()
    .unwrap();

    let result_b = Executor::new(
        BoothBoxed::<Array1<f64>>::new(lower, upper),
        BoundedCmaEs::<Array1<f64>, Array2<f64>>::new(42),
        CmaEsState::<Array1<f64>, Array2<f64>>::new(m0, 0.5),
    )
    .max_iter(30)
    .run()
    .unwrap();

    assert_eq!(result_a.cost(), result_b.cost());
    assert_eq!(result_a.param(), result_b.param());
}

/// `with_stds(ones)` reproduces the isotropic default bit-for-bit on the
/// bounded variant: the adaptive boundary penalty reads `σ² · diag(C)`,
/// which is unchanged when `C = diag(1²) = I`, so the full trajectory
/// matches.
#[test]
fn with_stds_ones_matches_default() {
    let lower = Array1::from_vec(vec![-5.0, -5.0]);
    let upper = Array1::from_vec(vec![5.0, 5.0]);
    let m0 = Array1::from_vec(vec![0.0, 0.0]);
    let ones = Array1::from_elem(2, 1.0);

    let default = Executor::new(
        BoothBoxed::<Array1<f64>>::new(lower.clone(), upper.clone()),
        BoundedCmaEs::<Array1<f64>, Array2<f64>>::new(42),
        CmaEsState::<Array1<f64>, Array2<f64>>::new(m0.clone(), 0.5),
    )
    .max_iter(40)
    .run()
    .unwrap();

    let with_ones = Executor::new(
        BoothBoxed::<Array1<f64>>::new(lower, upper),
        BoundedCmaEs::<Array1<f64>, Array2<f64>>::new(42),
        CmaEsState::<Array1<f64>, Array2<f64>>::new(m0, 0.5).with_stds(ones),
    )
    .max_iter(40)
    .run()
    .unwrap();

    assert_eq!(default.cost(), with_ones.cost());
    assert_eq!(default.param(), with_ones.param());
    assert_eq!(default.reason, with_ones.reason);
}

/// Anisotropic stds on the bounded variant still recover the interior
/// Booth minimum (1, 3) within budget — the per-coordinate scale flows
/// through the penalty's `σ² · diag(C)` without breaking convergence.
#[test]
fn with_stds_anisotropic_recovers_minimum() {
    let lower = Array1::from_vec(vec![-5.0, -5.0]);
    let upper = Array1::from_vec(vec![5.0, 5.0]);
    let m0 = Array1::from_vec(vec![0.0, 0.0]);
    let stds = Array1::from_vec(vec![0.5, 2.0]);

    let result = Executor::new(
        BoothBoxed::<Array1<f64>>::new(lower, upper),
        BoundedCmaEs::<Array1<f64>, Array2<f64>>::new(7),
        CmaEsState::<Array1<f64>, Array2<f64>>::new(m0, 0.5).with_stds(stds),
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

/// `with_stds` panics on a length mismatch (bounded variant).
#[test]
#[should_panic(expected = "stds.len() == mean.len()")]
fn with_stds_panics_on_length_mismatch() {
    let m0 = Array1::from_vec(vec![0.0, 0.0]);
    let _ =
        CmaEsState::<Array1<f64>, Array2<f64>>::new(m0, 0.5).with_stds(Array1::from_vec(vec![1.0]));
}

/// Slack bounds: the unconstrained Booth minimum (1, 3) is interior to
/// [-5, 5]², so the bounded solver must reach it. Tests that the
/// adaptive penalty machinery doesn't get in the way when the bounds
/// are inactive.
#[test]
fn slack_bounds_recover_unconstrained_minimum() {
    let lower = Array1::from_vec(vec![-5.0, -5.0]);
    let upper = Array1::from_vec(vec![5.0, 5.0]);
    let m0 = Array1::from_vec(vec![0.0, 0.0]);

    let result = Executor::new(
        BoothBoxed::<Array1<f64>>::new(lower, upper),
        BoundedCmaEs::<Array1<f64>, Array2<f64>>::new(7),
        CmaEsState::<Array1<f64>, Array2<f64>>::new(m0, 0.5),
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

/// Tight [-1, 1]² bounds: constrained Booth minimum is the box corner
/// (1, 1). The penalty must steer the search distribution toward the
/// active corner without numerical instability from the rank-µ update.
#[test]
fn tight_bounds_converge_to_box_corner() {
    let lower = Array1::from_vec(vec![-1.0, -1.0]);
    let upper = Array1::from_vec(vec![1.0, 1.0]);
    let m0 = Array1::from_vec(vec![0.0, 0.0]);

    let result = Executor::new(
        BoothBoxed::<Array1<f64>>::new(lower, upper),
        BoundedCmaEs::<Array1<f64>, Array2<f64>>::new(11),
        CmaEsState::<Array1<f64>, Array2<f64>>::new(m0, 0.3),
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
/// [-1, 1]² should still converge. The init-time mean projection plus
/// the adaptive penalty cooperate to pull the search distribution back
/// into feasibility within the iteration budget.
#[test]
fn infeasible_initial_mean_converges_to_box_corner() {
    let lower = Array1::from_vec(vec![-1.0, -1.0]);
    let upper = Array1::from_vec(vec![1.0, 1.0]);
    let m0 = Array1::from_vec(vec![10.0, 10.0]);

    let result = Executor::new(
        BoothBoxed::<Array1<f64>>::new(lower, upper),
        BoundedCmaEs::<Array1<f64>, Array2<f64>>::new(5),
        CmaEsState::<Array1<f64>, Array2<f64>>::new(m0, 0.3),
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

/// Solver-internal TolX termination still fires when bounds are slack.
/// `σ · max_i d_i` collapses below `1e−12 · initial_sigma` once the
/// search distribution contracts onto the unconstrained minimum.
#[test]
fn slack_bounds_terminate_solver_converged_on_tol_x() {
    let lower = Array1::from_vec(vec![-10.0, -10.0]);
    let upper = Array1::from_vec(vec![10.0, 10.0]);
    let m0 = Array1::from_vec(vec![0.0, 0.0]);

    let result = Executor::new(
        BoothBoxed::<Array1<f64>>::new(lower, upper),
        BoundedCmaEs::<Array1<f64>, Array2<f64>>::new(11),
        CmaEsState::<Array1<f64>, Array2<f64>>::new(m0, 0.3),
    )
    .terminate_on(CmaEsTolerance::new(1e-12 * 0.3))
    .max_iter(2000)
    .run()
    .unwrap();

    assert_eq!(result.reason, TerminationReason::CmaEsTolerance);
}

/// `PopulationState` invariants survive iteration on the bounded path:
/// `candidates` and `costs` stay parallel, length-λ, and
/// sorted-ascending. The bounded variant uses **penalized** costs in
/// `state.costs` (so the sort is on the penalized values) — same
/// invariant, different value semantics from the raw cost.
#[test]
fn population_invariants_hold_after_iteration() {
    let lower = Array1::from_vec(vec![-1.0, -1.0]);
    let upper = Array1::from_vec(vec![1.0, 1.0]);
    let m0 = Array1::from_vec(vec![0.3, 0.4]);
    let lambda = 12;

    let mut stepper = Executor::new(
        BoothBoxed::<Array1<f64>>::new(lower, upper),
        BoundedCmaEs::<Array1<f64>, Array2<f64>>::new(1234).with_lambda(lambda),
        CmaEsState::<Array1<f64>, Array2<f64>>::new(m0, 0.5),
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
