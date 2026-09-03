#![cfg(feature = "nalgebra_all")]

use crate::backend_aliases::nalgebra::DVector;
use basin::problems::BoothBoxed;
use basin::{
    Backtracking, BasicState, Executor, ProjectedGradientDescent,
    ProjectedGradientTolerance, TerminationReason,
};

/// Slack bounds: the unconstrained Booth minimum (1, 3) is interior to
/// [-5, 5]², so the projected solver must reach it.
#[test]
fn slack_bounds_recover_unconstrained_minimum() {
    let lower = DVector::from_vec(vec![-5.0, -5.0]);
    let upper = DVector::from_vec(vec![5.0, 5.0]);
    let problem = BoothBoxed::<DVector<f64>>::new(lower, upper);
    let initial = DVector::from_vec(vec![0.0, 0.0]);

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

/// Tight [-1, 1]² bounds; constrained minimum is the corner (1, 1).
#[test]
fn tight_bounds_converge_to_box_corner() {
    let lower = DVector::from_vec(vec![-1.0, -1.0]);
    let upper = DVector::from_vec(vec![1.0, 1.0]);
    let problem = BoothBoxed::<DVector<f64>>::new(lower, upper);
    let initial = DVector::from_vec(vec![0.0, 0.0]);

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
        "x[0] = {}",
        result.param()[0]
    );
    assert!(
        (result.param()[1] - 1.0).abs() < 1e-6,
        "x[1] = {}",
        result.param()[1]
    );
}

#[test]
fn infeasible_initial_param_is_projected_at_init() {
    let lower = DVector::from_vec(vec![-1.0, -1.0]);
    let upper = DVector::from_vec(vec![1.0, 1.0]);
    let problem = BoothBoxed::<DVector<f64>>::new(lower, upper);
    let initial = DVector::from_vec(vec![10.0, 10.0]);

    let result = Executor::new(
        problem,
        ProjectedGradientDescent::new(0.01),
        BasicState::new(initial),
    )
    .max_iter(0)
    .run()
    .unwrap();

    assert_eq!(result.reason, TerminationReason::MaxIter);
    assert_eq!(result.param()[0], 1.0);
    assert_eq!(result.param()[1], 1.0);
}

#[test]
fn projected_gradient_tolerance_triggers_at_corner_minimum() {
    let lower = DVector::from_vec(vec![-1.0, -1.0]);
    let upper = DVector::from_vec(vec![1.0, 1.0]);
    let problem = BoothBoxed::<DVector<f64>>::new(lower.clone(), upper.clone());
    let initial = DVector::from_vec(vec![0.0, 0.0]);

    let result = Executor::new(
        problem,
        ProjectedGradientDescent::with_line_search(Backtracking::new()),
        BasicState::new(initial),
    )
    .max_iter(2000)
    .terminate_on(ProjectedGradientTolerance::new(lower, upper, 1e-7))
    .run()
    .unwrap();

    assert_eq!(result.reason, TerminationReason::ProjectedGradientTolerance);
}

#[path = "support/backend_aliases.rs"]
mod backend_aliases;
