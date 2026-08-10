//! Integration tests for the projection-style box-constrained
//! Nelder-Mead variant (`NelderMead::new().projected()` etc.).
//!
//! Mirrors the structure of `projected_gradient_descent_vec.rs`:
//! slack-bounds/tight-bounds/infeasible-start coverage on
//! `BoothBoxed`, plus an adaptive-params smoke test on `RastriginBoxed`.
//! Backend coverage piggybacks on the PGD tests (every shipped backend
//! exercises the same `ClampInPlace` paths); revisit per-backend
//! coverage here only if a backend-specific bug surfaces.

use basin::problems::{BoothBoxed, RastriginBoxed};
use basin::{
    BasicSimplexState, Executor, NelderMead, SimplexTolerance,
    TerminationReason,
};

/// Slack bounds: the unconstrained Booth minimum `(1, 3)` lies inside
/// `[-5, 5]²`, so the projection step should be a no-op for any vertex
/// the simplex actually visits. Sanity-checks that projection doesn't
/// distort convergence when constraints are inactive.
#[test]
fn slack_bounds_recover_unconstrained_minimum() {
    let problem = BoothBoxed::<Vec<f64>>::new(vec![-5.0, -5.0], vec![5.0, 5.0]);
    let initial = vec![0.0, 0.0];

    let result = Executor::new(
        problem,
        NelderMead::new().projected(),
        BasicSimplexState::new(initial),
    )
    .max_iter(2_000)
    .terminate_on(SimplexTolerance::new(1e-8, 1e-8))
    .run()
    .unwrap();

    assert!(result.cost() < 1e-6, "cost = {}", result.cost());
    assert!(
        (result.param()[0] - 1.0).abs() < 1e-3,
        "x[0] = {} (expected near 1)",
        result.param()[0]
    );
    assert!(
        (result.param()[1] - 3.0).abs() < 1e-3,
        "x[1] = {} (expected near 3)",
        result.param()[1]
    );
}

/// Tight bounds: the unconstrained minimum `(1, 3)` lies outside
/// `[-1, 1]²`. The constrained optimum is the box corner `(1, 1)`. The
/// projected NM should drive the simplex to the corner; the
/// reflection/expansion/contraction trial points repeatedly clamp
/// back to the upper face. Load-bearing edge-active test (mirrors the
/// PGD `tight_bounds_converge_to_box_corner` case).
#[test]
fn tight_bounds_converge_to_box_corner() {
    let problem = BoothBoxed::<Vec<f64>>::new(vec![-1.0, -1.0], vec![1.0, 1.0]);
    let initial = vec![0.0, 0.0];

    let result = Executor::new(
        problem,
        NelderMead::new().projected(),
        BasicSimplexState::new(initial),
    )
    .max_iter(2_000)
    .terminate_on(SimplexTolerance::new(1e-10, 1e-10))
    .run()
    .unwrap();

    assert!(
        (result.param()[0] - 1.0).abs() < 1e-4,
        "x[0] = {} (expected pinned at upper bound 1)",
        result.param()[0]
    );
    assert!(
        (result.param()[1] - 1.0).abs() < 1e-4,
        "x[1] = {} (expected pinned at upper bound 1)",
        result.param()[1]
    );
}

/// Initial simplex with at least one infeasible vertex. `init` must
/// project every vertex onto the feasible box before the first cost
/// evaluation, so with `max_iter = 0` every reported vertex is in
/// `[lower, upper]`. Load-bearing test for the `init`-time projection
/// contract.
#[test]
fn infeasible_initial_simplex_is_projected_at_init() {
    use basin::SimplexState;
    let problem = BoothBoxed::<Vec<f64>>::new(vec![-1.0, -1.0], vec![1.0, 1.0]);
    // Two of three vertices live outside the box; init() must clamp them.
    let simplex = vec![vec![0.0, 0.0], vec![10.0, 10.0], vec![-5.0, 0.5]];

    let result = Executor::new(
        problem,
        NelderMead::new().projected(),
        BasicSimplexState::from_simplex(simplex),
    )
    .max_iter(0)
    .run()
    .unwrap();

    assert_eq!(result.reason, TerminationReason::MaxIter);
    for v in result.state.vertices() {
        assert!(
            v[0] >= -1.0 - 1e-12 && v[0] <= 1.0 + 1e-12,
            "x[0] = {} outside [-1, 1]",
            v[0]
        );
        assert!(
            v[1] >= -1.0 - 1e-12 && v[1] <= 1.0 + 1e-12,
            "x[1] = {} outside [-1, 1]",
            v[1]
        );
    }
    // The infeasible vertex (10, 10) must have been clamped to (1, 1).
    let found_corner = result
        .state
        .vertices()
        .iter()
        .any(|v| (v[0] - 1.0).abs() < 1e-12 && (v[1] - 1.0).abs() < 1e-12);
    assert!(
        found_corner,
        "expected the (10, 10) vertex to project to (1, 1); got {:?}",
        result.state.vertices()
    );
}

/// Adaptive-parameter projected NM on a 3D Rastrigin slice. Rastrigin
/// is multimodal but its global minimum sits at the origin, well
/// inside the standard `[-5.12, 5.12]ⁿ` box; from a near-origin start
/// the local descent path should reach the global basin. Loose cost
/// threshold; the test guards algorithm + bounds plumbing, not
/// global-optimization quality.
#[test]
fn adaptive_projected_on_rastrigin_3d() {
    let problem = RastriginBoxed::<Vec<f64>>::with_standard_bounds(3);
    let initial = vec![0.4, -0.3, 0.2];

    let result = Executor::new(
        problem,
        NelderMead::adaptive().projected(),
        BasicSimplexState::new(initial),
    )
    .max_iter(2_000)
    .terminate_on(SimplexTolerance::new(1e-8, 1e-8))
    .run()
    .unwrap();

    // Global optimum is 0 at origin. From a small basin around origin
    // NM should reach it. Generous threshold.
    assert!(result.cost() < 1e-4, "cost = {}", result.cost());
    for (i, &x) in result.param().iter().enumerate() {
        assert!(x.abs() < 1e-2, "x[{i}] = {x} (expected near 0)");
    }
}
