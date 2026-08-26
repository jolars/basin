#![cfg(feature = "ndarray")]
//! Integration tests for the log-barrier [`BarrierMethod`] on linearly
//! constrained quadratics (ndarray backend). Exercises the
//! `MatVec`/`MatTransposeVec` impls on `ndarray::Array2<f64>`.

use basin::problems::ConstrainedQuadratic;
use basin::{
    Backtracking, BarrierMethod, BasicState, Executor, GradientDescent,
    GradientState, TerminationReason,
};
use ndarray::{Array1, Array2, array};

/// `min ‖x − (2,2)‖²` s.t. `x₀ + x₁ ≤ 2`; constrained optimum (1,1).
fn active_problem() -> ConstrainedQuadratic<Array2<f64>, Array1<f64>> {
    ConstrainedQuadratic::new(array![2.0, 2.0], array![[1.0, 1.0]], array![2.0])
}

#[test]
fn active_constraint_converges_to_projection() {
    let problem = active_problem();
    let initial = array![0.0, 0.0]; // strictly feasible (sum 0 < 2)

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
fn infeasible_start_runs_phase_one_then_converges() {
    let problem = active_problem();
    let initial = array![2.0, 2.0]; // sum 4.0 > 2 ⇒ infeasible

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
fn eval_counts_are_recorded() {
    let problem = active_problem();
    let initial = array![0.0, 0.0];

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

/// Two active constraints exercise the multi-row transpose-matvec on
/// `Array2`. `x₀ + x₁ ≤ 2` and `x₀ ≤ 0.5` are both active at `(0.5, 1.5)`.
#[test]
fn two_constraints_both_active() {
    let problem = ConstrainedQuadratic::new(
        array![2.0, 2.0],
        array![[1.0, 1.0], [1.0, 0.0]],
        array![2.0, 0.5],
    );
    let initial = array![0.0, 0.0]; // 0<2 and 0<0.5: strictly feasible

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
