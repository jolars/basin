//! Integration tests for the log-barrier [`BarrierMethod`] on linearly
//! constrained quadratics over the plain `Vec<f64>` backend, using the
//! hand-rolled [`DenseMatrix`]. Mirrors `barrier_method_nalgebra.rs`; proves
//! the constraint solver now runs on the default backend with no external
//! linear-algebra crate.

use basin::problems::ConstrainedQuadratic;
use basin::{
    Backtracking, BarrierMethod, BasicState, CostFunction, DenseMatrix,
    Executor, Gradient, GradientDescent, GradientState,
    LinearInequalityConstraints, TargetCost, TerminationReason,
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
fn infeasible_start_runs_phase_one_then_converges() {
    let problem = active_problem();
    // sum 4.0 > 2, so Phase I must find an interior point before Phase II.
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

    assert_eq!(result.reason, TerminationReason::SolverConverged);
    assert!(
        (result.param()[0] - 1.0).abs() < 1e-4
            && (result.param()[1] - 1.0).abs() < 1e-4,
        "expected (1, 1), got {:?}",
        result.param()
    );
}

#[test]
fn target_cost_does_not_bypass_phase_one() {
    let result = Executor::new(
        active_problem(),
        BarrierMethod::new(GradientDescent::with_line_search(
            Backtracking::new(),
        )),
        BasicState::new(vec![2.0, 2.0]),
    )
    .max_iter(50)
    .terminate_on(TargetCost(0.1))
    .run()
    .unwrap();

    assert_eq!(result.reason, TerminationReason::SolverConverged);
    assert!(result.param()[0] + result.param()[1] < 2.0);
}

/// Small fixture for Phase I failure and input-validation cases.
struct LinearProbe {
    a: DenseMatrix,
    b: Vec<f64>,
}

impl CostFunction for LinearProbe {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = std::convert::Infallible;

    fn cost(&self, x: &Vec<f64>) -> Result<f64, Self::Error> {
        Ok(x.iter().map(|xi| xi * xi).sum())
    }
}

impl Gradient for LinearProbe {
    type Gradient = Vec<f64>;

    fn gradient(&self, x: &Vec<f64>) -> Result<Vec<f64>, Self::Error> {
        Ok(x.iter().map(|xi| 2.0 * xi).collect())
    }
}

impl LinearInequalityConstraints for LinearProbe {
    type Matrix = DenseMatrix;

    fn a(&self) -> &DenseMatrix {
        &self.a
    }

    fn b(&self) -> &Vec<f64> {
        &self.b
    }
}

fn run_probe(problem: LinearProbe, initial: Vec<f64>) -> TerminationReason {
    Executor::new(
        problem,
        BarrierMethod::new(GradientDescent::with_line_search(
            Backtracking::new(),
        )),
        BasicState::new(initial),
    )
    .max_iter(50)
    .run()
    .unwrap()
    .reason
}

#[test]
fn boundary_start_runs_phase_one_then_phase_two() {
    let result = Executor::new(
        active_problem(),
        BarrierMethod::new(GradientDescent::with_line_search(
            Backtracking::new(),
        )),
        BasicState::new(vec![1.0, 1.0]),
    )
    .max_iter(50)
    .run()
    .unwrap();

    assert_eq!(result.reason, TerminationReason::SolverConverged);
    assert!(result.param()[0] + result.param()[1] < 2.0);
}

#[test]
fn inconsistent_constraints_report_failure() {
    // x <= 0 and x >= 1 cannot hold simultaneously.
    let problem = LinearProbe {
        a: DenseMatrix::from_row_slice(2, 1, &[1.0, -1.0]),
        b: vec![0.0, -1.0],
    };
    assert_eq!(
        run_probe(problem, vec![0.5]),
        TerminationReason::SolverFailed
    );
}

#[test]
fn distant_feasible_system_is_not_reported_infeasible() {
    // Phase I needs several inner budgets to travel from this scale to the
    // strict half-space. Exhausting one such budget does not center the
    // auxiliary barrier problem and cannot certify infeasibility.
    let problem = LinearProbe {
        a: DenseMatrix::from_row_slice(1, 1, &[1.0]),
        b: vec![0.0],
    };
    let result = Executor::new(
        problem,
        BarrierMethod::new(GradientDescent::with_line_search(
            Backtracking::new(),
        )),
        BasicState::new(vec![1000.0]),
    )
    .max_iter(50)
    .run()
    .unwrap();

    assert_eq!(result.reason, TerminationReason::SolverConverged);
    assert!(
        result.param()[0] < 0.0,
        "expected a strict point, got {:?}",
        result.param()
    );
}

#[test]
fn empty_strict_interior_reports_failure() {
    // x <= 0 and x >= 0 leave the feasible singleton {0}, with no interior.
    let problem = LinearProbe {
        a: DenseMatrix::from_row_slice(2, 1, &[1.0, -1.0]),
        b: vec![0.0, 0.0],
    };
    let result = Executor::new(
        problem,
        BarrierMethod::new(GradientDescent::with_line_search(
            Backtracking::new(),
        )),
        // At the feasible singleton the Phase I gradient is exactly zero, so
        // each auxiliary subproblem is demonstrably centered.
        BasicState::new(vec![0.0]),
    )
    .max_iter(50)
    .run()
    .unwrap();

    assert_eq!(result.reason, TerminationReason::SolverFailed);
}

#[test]
fn nonfinite_initial_parameter_reports_failure() {
    let problem = LinearProbe {
        a: DenseMatrix::from_row_slice(1, 1, &[1.0]),
        b: vec![0.0],
    };
    assert_eq!(
        run_probe(problem, vec![f64::NAN]),
        TerminationReason::SolverFailed
    );
}

#[test]
fn nonfinite_constraint_data_reports_failure() {
    let problem = LinearProbe {
        a: DenseMatrix::from_row_slice(1, 1, &[1.0]),
        b: vec![f64::INFINITY],
    };
    assert_eq!(
        run_probe(problem, vec![0.0]),
        TerminationReason::SolverFailed
    );
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
