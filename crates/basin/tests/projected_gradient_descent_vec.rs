use basin::problems::BoothBoxed;
use basin::{
    Backtracking, BasicState, Executor, MaxIter, ProjectedGradientDescent,
    ProjectedGradientTolerance, TerminationReason,
};

/// Slack bounds: the unconstrained Booth minimum (1, 3) lies inside
/// [-5, 5]², so the projected solver must match the unconstrained
/// behavior. Sanity check that the projection step is a no-op when
/// constraints are not active.
#[test]
fn slack_bounds_recover_unconstrained_minimum() {
    let problem = BoothBoxed::<Vec<f64>>::new(vec![-5.0, -5.0], vec![5.0, 5.0]);
    let initial = vec![0.0, 0.0];

    let result = Executor::new(
        problem,
        ProjectedGradientDescent::with_line_search(Backtracking::new()),
        BasicState::new(initial),
    )
    .max_iter(2000)
    .run()
    .unwrap();

    assert!(
        (result.param()[0] - 1.0).abs() < 1e-4,
        "x[0] = {} (expected near 1)",
        result.param()[0]
    );
    assert!(
        (result.param()[1] - 3.0).abs() < 1e-4,
        "x[1] = {} (expected near 3)",
        result.param()[1]
    );
}

/// Tight bounds: the unconstrained minimum (1, 3) lies *outside*
/// [-1, 1]². The constrained optimum is the box corner (1, 1)—both
/// gradient components are negative there, pulling the iterate against
/// the upper face. The unprojected ‖∇f‖_∞ ≈ 20 at (1, 1), so
/// `GradientTolerance` would *not* trigger; the projected metric
/// vanishes exactly. Load-bearing edge-active test.
#[test]
fn tight_bounds_converge_to_box_corner() {
    let problem = BoothBoxed::<Vec<f64>>::new(vec![-1.0, -1.0], vec![1.0, 1.0]);
    let initial = vec![0.0, 0.0];

    let result = Executor::new(
        problem,
        ProjectedGradientDescent::with_line_search(Backtracking::new()),
        BasicState::new(initial),
    )
    .max_iter(2000)
    .run()
    .unwrap();

    assert!(
        (result.param()[0] - 1.0).abs() < 1e-6,
        "x[0] = {} (expected pinned at upper bound 1)",
        result.param()[0]
    );
    assert!(
        (result.param()[1] - 1.0).abs() < 1e-6,
        "x[1] = {} (expected pinned at upper bound 1)",
        result.param()[1]
    );
}

/// Infeasible start (10, 10) outside [-1, 1]². `init` must project the
/// param onto the feasible box before any iteration runs, so even with
/// `max_iter = 0` the result reports the projected (1, 1) iterate.
/// Load-bearing test for the `init`-time projection contract.
#[test]
fn infeasible_initial_param_is_projected_at_init() {
    let problem = BoothBoxed::<Vec<f64>>::new(vec![-1.0, -1.0], vec![1.0, 1.0]);
    let initial = vec![10.0, 10.0];

    let result = Executor::new(
        problem,
        ProjectedGradientDescent::new(0.01),
        BasicState::new(initial),
    )
    .max_iter(0)
    .run()
    .unwrap();

    assert_eq!(result.reason, TerminationReason::MaxIter);
    assert_eq!(result.param(), &vec![1.0, 1.0]);
}

/// `ProjectedGradientTolerance` triggers on the tight-bounds setup
/// because the projected-gradient ∞-norm vanishes at (1, 1). A
/// regular `GradientTolerance` would *not* fire here (∇f is non-zero
/// at the constrained optimum)—that's the whole reason this
/// criterion exists.
#[test]
fn projected_gradient_tolerance_triggers_at_corner_minimum() {
    let lower = vec![-1.0, -1.0];
    let upper = vec![1.0, 1.0];
    let problem = BoothBoxed::<Vec<f64>>::new(lower.clone(), upper.clone());
    let initial = vec![0.0, 0.0];

    let result = Executor::new(
        problem,
        ProjectedGradientDescent::with_line_search(Backtracking::new()),
        BasicState::new(initial),
    )
    .max_iter(2000)
    .terminate_on(ProjectedGradientTolerance::new(lower, upper, 1e-7))
    .terminate_on(MaxIter(2000))
    .run()
    .unwrap();

    assert_eq!(result.reason, TerminationReason::ProjectedGradientTolerance);
    assert!(
        (result.param()[0] - 1.0).abs() < 1e-7,
        "x[0] = {}",
        result.param()[0]
    );
    assert!(
        (result.param()[1] - 1.0).abs() < 1e-7,
        "x[1] = {}",
        result.param()[1]
    );
}
