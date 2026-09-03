#![cfg(feature = "nalgebra_all")]
//! Integration tests for the augmented-Lagrangian [`AugmentedLagrangianMethod`]
//! on linearly-equality-constrained quadratics (nalgebra backend).

use crate::backend_aliases::nalgebra::{DMatrix, DVector};
use basin::problems::EqualityConstrainedQuadratic;
use basin::{
    AugmentedLagrangianMethod, Backtracking, BasicState, Bfgs, Executor,
    GradientDescent, GradientState, Lbfgsb, TerminationReason,
};

/// `min ‖x − (2,2)‖²` s.t. `x₀ + x₁ = 2`. The unconstrained min (2,2) is
/// infeasible (sum 4 ≠ 2); the constrained optimum is the projection of
/// (2,2) onto the line `x₀ + x₁ = 2`, namely (1,1).
fn single_row_problem()
-> EqualityConstrainedQuadratic<DMatrix<f64>, DVector<f64>> {
    EqualityConstrainedQuadratic::new(
        DVector::from_vec(vec![2.0, 2.0]),
        DMatrix::from_row_slice(1, 2, &[1.0, 1.0]),
        DVector::from_vec(vec![2.0]),
    )
}

#[test]
fn converges_to_affine_projection() {
    let problem = single_row_problem();
    // Start at the *infeasible* origin (0 + 0 = 0 ≠ 2): the augmented
    // Lagrangian has no feasibility wall, so this is fine.
    let initial = DVector::from_vec(vec![0.0, 0.0]);

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
        DVector::from_vec(vec![2.0, 2.0]),
        DMatrix::from_row_slice(2, 2, &[1.0, 1.0, 1.0, 0.0]),
        DVector::from_vec(vec![2.0, 0.5]),
    );
    let initial = DVector::from_vec(vec![0.0, 0.0]);

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
    let initial = DVector::from_vec(vec![0.0, 0.0]);

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

/// A `Bfgs` inner (state `QuasiNewtonState`, not `BasicState`) proves the
/// augmented-Lagrangian method is no longer locked to `BasicState<V>` /
/// `GradientDescent`. `L_ρ` is finite everywhere, so the default Wolfe line
/// search is fine. Converges to the same projection (1,1).
#[test]
fn bfgs_inner_converges_to_affine_projection() {
    let problem = single_row_problem();
    let initial = DVector::from_vec(vec![0.0, 0.0]); // infeasible start is fine

    let result = Executor::new(
        problem,
        AugmentedLagrangianMethod::new(Bfgs::new()),
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

/// An unbounded `Lbfgs` inner (state `LbfgsState`) exercises a third inner
/// state shape. `L_ρ` is finite, so the More–Thuente line search is fine.
#[test]
fn lbfgs_inner_converges_to_affine_projection() {
    let problem = single_row_problem();
    let initial = DVector::from_vec(vec![0.0, 0.0]);

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

#[path = "support/backend_aliases.rs"]
mod backend_aliases;
