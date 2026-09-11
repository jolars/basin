//! Integration tests for the box-constrained MADS variant
//! (`Mads::new().bounded()`), which enforces bounds by the extreme barrier
//! (infeasible poll points get `f = +∞`).
//!
//! Mirrors `bounded_nelder_mead.rs`: slack-bounds/tight-bounds/infeasible-
//! start coverage on `BoothBoxed`. Backend coverage piggybacks on the
//! unbounded `mads_public.rs` tests (the bounded path differs only in the
//! barrier eval closure, which is backend-agnostic).

use basin::problems::BoothBoxed;
use basin::{Executor, Mads, MadsState, State, TerminationReason};

#[test]
fn poll_tolerance_can_be_set_after_bounding() {
    for (tolerance, expected) in [
        (Some(1.0), TerminationReason::MeshTolerance),
        (Some(0.0), TerminationReason::MaxIter),
        (None, TerminationReason::MaxIter),
    ] {
        let solver = Mads::new()
            .bounded()
            .with_absolute_poll_size_tolerance(1.0)
            .with_absolute_poll_size_tolerance(tolerance);
        let configured = Mads::new()
            .with_absolute_poll_size_tolerance(1.0)
            .with_absolute_cost_change_tolerance(None)
            .bounded()
            .with_absolute_poll_size_tolerance(tolerance);
        let problem =
            || BoothBoxed::<Vec<f64>>::new(vec![-5.0; 2], vec![5.0; 2]);
        let direct = Executor::from_start(problem(), solver, vec![0.0; 2])
            .max_iter(1)
            .run()
            .unwrap();
        let wrapped = Executor::from_start(problem(), configured, vec![0.0; 2])
            .max_iter(1)
            .run()
            .unwrap();
        assert_eq!(direct.reason, expected);
        assert_eq!(wrapped.reason, expected);
    }
}

/// Slack bounds: the unconstrained Booth minimum `(1, 3)` lies inside `[-5, 5]²`,
/// so the barrier never fires and MADS recovers the unconstrained optimum.
#[test]
fn slack_bounds_recover_unconstrained_minimum() {
    let problem = BoothBoxed::<Vec<f64>>::new(vec![-5.0, -5.0], vec![5.0, 5.0]);

    let result = Executor::new(
        problem,
        Mads::new()
            .bounded()
            .with_initial_poll_size(1.0)
            .with_minimum_poll_size(1e-6),
        MadsState::new(vec![0.0, 0.0]),
    )
    .max_iter(50_000)
    .max_cost_evals(20_000)
    .run()
    .unwrap();

    assert!(result.best_cost() < 1e-6, "cost = {}", result.best_cost());
    let x = result.best_param();
    assert!(
        (x[0] - 1.0).abs() < 1e-3,
        "x[0] = {} (expected near 1)",
        x[0]
    );
    assert!(
        (x[1] - 3.0).abs() < 1e-3,
        "x[1] = {} (expected near 3)",
        x[1]
    );
}

/// Tight bounds: the unconstrained minimum `(1, 3)` lies outside `[-1, 1]²`; the
/// constrained optimum is the box corner `(1, 1)`. The extreme barrier rejects
/// every poll point that steps out of the box, so MADS pins the incumbent to the
/// active upper face and converges to the corner.
#[test]
fn tight_bounds_converge_to_box_corner() {
    let problem = BoothBoxed::<Vec<f64>>::new(vec![-1.0, -1.0], vec![1.0, 1.0]);

    let result = Executor::new(
        problem,
        Mads::new()
            .bounded()
            .with_initial_poll_size(1.0)
            .with_minimum_poll_size(1e-8),
        MadsState::new(vec![0.0, 0.0]),
    )
    .max_iter(50_000)
    .max_cost_evals(20_000)
    .run()
    .unwrap();

    let x = result.best_param();
    assert!(
        (x[0] - 1.0).abs() < 1e-4,
        "x[0] = {} (expected pinned at upper bound 1)",
        x[0]
    );
    assert!(
        (x[1] - 1.0).abs() < 1e-4,
        "x[1] = {} (expected pinned at upper bound 1)",
        x[1]
    );
}

/// An infeasible start is clamped into the box at `init`, so with `max_iter = 0`
/// the reported iterate is already feasible (the `(10, 10)` start clamps to the
/// `(1, 1)` corner).
#[test]
fn infeasible_start_clamped_at_init() {
    let problem = BoothBoxed::<Vec<f64>>::new(vec![-1.0, -1.0], vec![1.0, 1.0]);

    let result = Executor::new(
        problem,
        Mads::new().bounded(),
        MadsState::new(vec![10.0, 10.0]),
    )
    .max_iter(0)
    .run()
    .unwrap();

    assert_eq!(result.reason, TerminationReason::MaxIter);
    let x = result.state.param();
    assert!(
        (x[0] - 1.0).abs() < 1e-12 && (x[1] - 1.0).abs() < 1e-12,
        "start (10, 10) should clamp to the corner (1, 1); got {x:?}"
    );
}
