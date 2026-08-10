//! Integration tests for the augmented-Lagrangian [`AugmentedLagrangianMethod`]
//! on linearly-equality-constrained quadratics over the plain `Vec<f64>`
//! backend, using the hand-rolled [`DenseMatrix`]. Mirrors
//! `augmented_lagrangian_nalgebra.rs`.

use basin::problems::EqualityConstrainedQuadratic;
use basin::{
    AugmentedLagrangianMethod, Backtracking, BasicState, DenseMatrix, Executor,
    GradientDescent, GradientState, Lbfgsb, TerminationReason,
};

/// `min ‖x − (2,2)‖²` s.t. `x₀ + x₁ = 2`. The unconstrained min (2,2) is
/// infeasible (sum 4 ≠ 2); the constrained optimum is the projection of (2,2)
/// onto the line `x₀ + x₁ = 2`, namely (1,1).
fn single_row_problem() -> EqualityConstrainedQuadratic<DenseMatrix, Vec<f64>> {
    EqualityConstrainedQuadratic::new(
        vec![2.0, 2.0],
        DenseMatrix::from_row_slice(1, 2, &[1.0, 1.0]),
        vec![2.0],
    )
}

#[test]
fn converges_to_affine_projection() {
    let problem = single_row_problem();
    // Start at the *infeasible* origin (0 + 0 = 0 ≠ 2): the augmented
    // Lagrangian has no feasibility wall, so this is fine.
    let initial = vec![0.0, 0.0];

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
    // Two equalities `x₀ + x₁ = 2` and `x₀ = 0.5` pin a unique point
    // (0.5, 1.5); the quadratic center (2,2) is irrelevant to the solution
    // since the feasible set is a single point.
    let problem = EqualityConstrainedQuadratic::new(
        vec![2.0, 2.0],
        DenseMatrix::from_row_slice(2, 2, &[1.0, 1.0, 1.0, 0.0]),
        vec![2.0, 0.5],
    );
    let initial = vec![0.0, 0.0];

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
    let initial = vec![0.0, 0.0];

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

    // Inner subproblem solves plus the per-outer-iter true-objective evals
    // must have accumulated onto the outer state.
    assert!(result.cost_evals() > 0, "no cost evals recorded");
    assert!(
        result.state.gradient_evals() > 0,
        "no gradient evals recorded"
    );
}

/// An unbounded `Lbfgs` inner (state `LbfgsState`, not `BasicState`) proves the
/// augmented-Lagrangian method is inner-agnostic on the `Vec<f64>` backend too.
/// `L_ρ` is finite everywhere, so the inner's default line search is fine.
/// Converges to the same projection (1,1).
#[test]
fn lbfgs_inner_converges_to_affine_projection() {
    let problem = single_row_problem();
    let initial = vec![0.0, 0.0];

    let result = Executor::new(
        problem,
        AugmentedLagrangianMethod::new(Lbfgsb::new().unbounded()),
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
