#![cfg(feature = "faer")]
//! Integration tests for the log-barrier [`BarrierMethod`] on linearly
//! constrained quadratics (faer backend), mirror of
//! `barrier_method_nalgebra.rs`.

use basin::problems::ConstrainedQuadratic;
use basin::{
    Backtracking, BarrierMethod, BasicState, Executor, GradientDescent,
    GradientState, TerminationReason,
};
use faer::{Col, Mat};

/// `min ‖x − (2,2)‖²` s.t. `x₀ + x₁ ≤ 2`; constrained optimum (1, 1).
fn active_problem() -> ConstrainedQuadratic<Mat<f64>, Col<f64>> {
    ConstrainedQuadratic::new(
        Col::from_fn(2, |_| 2.0),
        Mat::from_fn(1, 2, |_, _| 1.0),
        Col::from_fn(1, |_| 2.0),
    )
}

#[test]
fn active_constraint_converges_to_projection() {
    let problem = active_problem();
    let initial = Col::from_fn(2, |_| 0.0); // strictly feasible

    let result = Executor::new(
        problem,
        BarrierMethod::new(GradientDescent::with_line_search(
            Backtracking::new(),
        )),
        BasicState::new(initial),
    )
    .max_iter(50)
    .run()
    .unwrap();

    assert_eq!(result.reason, TerminationReason::SolverConverged);
    assert!(
        (result.param()[0] - 1.0).abs() < 1e-4
            && (result.param()[1] - 1.0).abs() < 1e-4,
        "expected (1, 1), got [{}, {}]",
        result.param()[0],
        result.param()[1]
    );
}

#[test]
fn infeasible_start_is_reported_as_failure() {
    let problem = active_problem();
    let initial = Col::from_fn(2, |_| 2.0); // sum 4 > 2 ⇒ infeasible

    let result = Executor::new(
        problem,
        BarrierMethod::new(GradientDescent::with_line_search(
            Backtracking::new(),
        )),
        BasicState::new(initial),
    )
    .max_iter(50)
    .run()
    .unwrap();

    assert_eq!(result.reason, TerminationReason::SolverFailed);
}

#[test]
fn two_constraints_both_active() {
    // x₀ + x₁ ≤ 2 and x₀ ≤ 0.5; optimum (0.5, 1.5).
    let problem = ConstrainedQuadratic::new(
        Col::from_fn(2, |_| 2.0),
        // A = [[1, 1], [1, 0]], all ones except the (1, 1) entry.
        Mat::from_fn(2, 2, |i, j| if i == 1 && j == 1 { 0.0 } else { 1.0 }),
        Col::from_fn(2, |i| if i == 0 { 2.0 } else { 0.5 }),
    );
    let initial = Col::from_fn(2, |_| 0.0);

    let result = Executor::new(
        problem,
        BarrierMethod::new(GradientDescent::with_line_search(
            Backtracking::new(),
        )),
        BasicState::new(initial),
    )
    .max_iter(50)
    .run()
    .unwrap();

    assert_eq!(result.reason, TerminationReason::SolverConverged);
    assert!(
        (result.param()[0] - 0.5).abs() < 1e-4
            && (result.param()[1] - 1.5).abs() < 1e-4,
        "expected (0.5, 1.5), got [{}, {}]",
        result.param()[0],
        result.param()[1]
    );
}

#[test]
fn eval_counts_are_recorded() {
    let problem = active_problem();
    let initial = Col::from_fn(2, |_| 0.0);

    let result = Executor::new(
        problem,
        BarrierMethod::new(GradientDescent::with_line_search(
            Backtracking::new(),
        )),
        BasicState::new(initial),
    )
    .max_iter(50)
    .run()
    .unwrap();

    assert!(result.cost_evals() > 0, "no cost evals recorded");
    assert!(
        result.state.gradient_evals() > 0,
        "no gradient evals recorded"
    );
}
