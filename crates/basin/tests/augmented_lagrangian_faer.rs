#![cfg(feature = "faer_all")]
//! Integration tests for the augmented-Lagrangian [`AugmentedLagrangianMethod`]
//! on linearly-equality-constrained quadratics (faer backend).

use crate::backend_aliases::faer::{Col, Mat};
use basin::problems::EqualityConstrainedQuadratic;
use basin::{
    AugmentedLagrangianMethod, Backtracking, BasicState, Executor,
    GradientDescent, GradientState, TerminationReason,
};

/// `min ‖x − (2,2)‖²` s.t. `x₀ + x₁ = 2`. Constrained optimum: the
/// projection of (2,2) onto the line `x₀ + x₁ = 2`, namely (1,1).
fn single_row_problem() -> EqualityConstrainedQuadratic<Mat<f64>, Col<f64>> {
    EqualityConstrainedQuadratic::new(
        Col::from_fn(2, |_| 2.0),
        Mat::from_fn(1, 2, |_, _| 1.0),
        Col::from_fn(1, |_| 2.0),
    )
}

#[test]
fn converges_to_affine_projection() {
    let problem = single_row_problem();
    // Start at the *infeasible* origin (sum 0 ≠ 2): no feasibility wall.
    let initial = Col::from_fn(2, |_| 0.0);

    let result = Executor::new(
        problem,
        AugmentedLagrangianMethod::new(GradientDescent::with_line_search(
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
        "expected (1, 1), got {:?}",
        result.param()
    );
}

#[test]
fn fully_determined_system() {
    // Two equalities `x₀ + x₁ = 2` and `x₀ = 0.5` pin the unique point
    // (0.5, 1.5).
    let problem = EqualityConstrainedQuadratic::new(
        Col::from_fn(2, |_| 2.0),
        Mat::from_fn(2, 2, |i, j| if (i, j) == (1, 1) { 0.0 } else { 1.0 }),
        Col::from_fn(2, |i| if i == 0 { 2.0 } else { 0.5 }),
    );
    let initial = Col::from_fn(2, |_| 0.0);

    let result = Executor::new(
        problem,
        AugmentedLagrangianMethod::new(GradientDescent::with_line_search(
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
        "expected (0.5, 1.5), got {:?}",
        result.param()
    );
}

#[test]
fn eval_counts_are_recorded() {
    let problem = single_row_problem();
    let initial = Col::from_fn(2, |_| 0.0);

    let result = Executor::new(
        problem,
        AugmentedLagrangianMethod::new(GradientDescent::with_line_search(
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

#[path = "support/backend_aliases.rs"]
mod backend_aliases;
