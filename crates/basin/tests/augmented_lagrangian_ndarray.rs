#![cfg(feature = "ndarray")]
//! Integration tests for the augmented-Lagrangian [`AugmentedLagrangianMethod`]
//! on linearly-equality-constrained quadratics (ndarray backend). Exercises the
//! `MatVec`/`MatTransposeVec` impls on `ndarray::Array2<f64>`.

use basin::problems::EqualityConstrainedQuadratic;
use basin::{
    AugmentedLagrangianMethod, Backtracking, BasicState, Executor, GradientDescent, GradientState,
    TerminationReason,
};
use ndarray::{Array1, Array2, array};

/// `min ‖x − (2,2)‖²` s.t. `x₀ + x₁ = 2`; constrained optimum (1,1).
fn single_row_problem() -> EqualityConstrainedQuadratic<Array2<f64>, Array1<f64>> {
    EqualityConstrainedQuadratic::new(array![2.0, 2.0], array![[1.0, 1.0]], array![2.0])
}

#[test]
fn converges_to_affine_projection() {
    let problem = single_row_problem();
    let initial = array![0.0, 0.0]; // infeasible start is fine (no wall)

    let result = Executor::new(
        problem,
        AugmentedLagrangianMethod::new(GradientDescent::with_line_search(Backtracking::new())),
        BasicState::new(initial),
    )
    .max_iter(50)
    .run()
    .unwrap();

    assert_eq!(result.reason, TerminationReason::SolverConverged);
    assert!(
        (result.param()[0] - 1.0).abs() < 1e-4 && (result.param()[1] - 1.0).abs() < 1e-4,
        "expected (1, 1), got {:?}",
        result.param()
    );
}

#[test]
fn fully_determined_system() {
    // Two equalities pin the unique point (0.5, 1.5).
    let problem = EqualityConstrainedQuadratic::new(
        array![2.0, 2.0],
        array![[1.0, 1.0], [1.0, 0.0]],
        array![2.0, 0.5],
    );
    let initial = array![0.0, 0.0];

    let result = Executor::new(
        problem,
        AugmentedLagrangianMethod::new(GradientDescent::with_line_search(Backtracking::new())),
        BasicState::new(initial),
    )
    .max_iter(50)
    .run()
    .unwrap();

    assert_eq!(result.reason, TerminationReason::SolverConverged);
    assert!(
        (result.param()[0] - 0.5).abs() < 1e-4 && (result.param()[1] - 1.5).abs() < 1e-4,
        "expected (0.5, 1.5), got {:?}",
        result.param()
    );
}

#[test]
fn eval_counts_are_recorded() {
    let problem = single_row_problem();
    let initial = array![0.0, 0.0];

    let result = Executor::new(
        problem,
        AugmentedLagrangianMethod::new(GradientDescent::with_line_search(Backtracking::new())),
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
