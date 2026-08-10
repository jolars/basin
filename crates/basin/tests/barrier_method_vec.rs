//! Integration tests for the log-barrier [`BarrierMethod`] on linearly
//! constrained quadratics over the plain `Vec<f64>` backend, using the
//! hand-rolled [`DenseMatrix`]. Mirrors `barrier_method_nalgebra.rs`; proves
//! the constraint solver now runs on the default backend with no external
//! linear-algebra crate.

use basin::problems::ConstrainedQuadratic;
use basin::{
    Backtracking, BarrierMethod, BasicState, DenseMatrix, Executor,
    GradientDescent, GradientState, TerminationReason,
};

/// `min ‖x − (2,2)‖²` s.t. `x₀ + x₁ ≤ 2`. The unconstrained min (2,2) is
/// infeasible; the constrained optimum is the projection (1,1).
fn active_problem() -> ConstrainedQuadratic<DenseMatrix, Vec<f64>> {
    ConstrainedQuadratic::new(
        vec![2.0, 2.0],
        DenseMatrix::from_row_slice(1, 2, &[1.0, 1.0]),
        vec![2.0],
    )
}

#[test]
fn active_constraint_converges_to_projection() {
    let problem = active_problem();
    let initial = vec![0.0, 0.0]; // strictly feasible (sum 0 < 2)

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
        "expected (1, 1), got {:?}",
        result.param()
    );
}

#[test]
fn inactive_constraint_recovers_unconstrained_minimum() {
    // Center inside the feasible region: unconstrained min (0.5, 0.5),
    // sum 1.0 < 2, so the constraint is slack at the optimum.
    let problem = ConstrainedQuadratic::new(
        vec![0.5, 0.5],
        DenseMatrix::from_row_slice(1, 2, &[1.0, 1.0]),
        vec![2.0],
    );
    let initial = vec![0.0, 0.0];

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
            && (result.param()[1] - 0.5).abs() < 1e-4,
        "expected (0.5, 0.5), got {:?}",
        result.param()
    );
}

#[test]
fn infeasible_start_is_reported_as_failure() {
    let problem = active_problem();
    // sum 4.0 > 2 ⇒ A x₀ ≰ b ⇒ infeasible.
    let initial = vec![2.0, 2.0];

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
fn eval_counts_are_recorded() {
    let problem = active_problem();
    let initial = vec![0.0, 0.0];

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

    // Inner barrier solves plus the per-outer-iter true-objective evals must
    // have accumulated onto the outer state.
    assert!(result.cost_evals() > 0, "no cost evals recorded");
    assert!(
        result.state.gradient_evals() > 0,
        "no gradient evals recorded"
    );
}

/// Two active constraints exercise the multi-row `Aᵀ·(μ/s)` sum in the
/// barrier gradient, and the `DenseMatrix` transpose-matvec on a `2×2`. `min
/// ‖x − (2,2)‖²` s.t. `x₀ + x₁ ≤ 2` and `x₀ ≤ 0.5` has both constraints active
/// at the optimum `(0.5, 1.5)`.
#[test]
fn two_constraints_both_active() {
    let problem = ConstrainedQuadratic::new(
        vec![2.0, 2.0],
        DenseMatrix::from_row_slice(2, 2, &[1.0, 1.0, 1.0, 0.0]),
        vec![2.0, 0.5],
    );
    let initial = vec![0.0, 0.0]; // 0<2 and 0<0.5: strictly feasible

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
        "expected (0.5, 1.5), got {:?}",
        result.param()
    );
}
