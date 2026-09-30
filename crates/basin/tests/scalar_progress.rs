//! Scalar solvers use the same checked records and lifecycle as vector solvers.

use basin::{
    BoxConstraints, Brent, BrentDerivative, CostFunction, EvalCounts, Executor,
    FirstOrderState, GoldenSection, Gradient, GradientState, PointState,
    Solver, State,
};
use std::convert::Infallible;

struct Quadratic;

impl CostFunction for Quadratic {
    type Param = f64;
    type Output = f64;
    type Error = Infallible;

    fn cost(&self, x: &f64) -> Result<f64, Infallible> {
        Ok((x - 0.25).powi(2))
    }
}

impl BoxConstraints for Quadratic {
    fn lower(&self) -> &f64 {
        &-1.0
    }
    fn upper(&self) -> &f64 {
        &2.0
    }
}

impl Gradient for Quadratic {
    type Gradient = f64;

    fn gradient(&self, x: &f64) -> Result<f64, Infallible> {
        Ok(2.0 * (x - 0.25))
    }
}

fn check_point_solver<So>(solver: So)
where
    So: Solver<Quadratic, PointState<f64>, Error = Infallible>,
{
    let initial = Executor::new(Quadratic, solver, PointState::new(0.5))
        .require_evaluated_state()
        .max_iter(0)
        .run_with_solver()
        .unwrap();
    assert_eq!(initial.state.iter(), 0);
    assert!(initial.state.current().is_some());
    assert_eq!(initial.state.counts(), &initial.counts);
    assert_eq!(initial.state.cost_evals(), initial.counts.cost_evals);

    let resumed =
        Executor::resume_from_checkpoint(Quadratic, initial.into_checkpoint())
            .require_evaluated_state()
            .max_iter(4)
            .run_with_solver()
            .unwrap();
    assert_eq!(resumed.state.iter(), 4);
    assert_eq!(resumed.state.counts(), &resumed.counts);
    let (x, cost) = resumed.state.current().unwrap();
    assert_eq!(cost, Quadratic.cost(x).unwrap());
    let (best, best_cost) = resumed.state.best().unwrap();
    assert_eq!(best_cost, Quadratic.cost(best).unwrap());

    let fresh = Executor::new(Quadratic, resumed.solver, resumed.state)
        .require_evaluated_state()
        .max_iter(0)
        .run_with_solver()
        .unwrap();
    assert_eq!(fresh.state.iter(), 0);
    assert!(fresh.counts.cost_evals > 0);
    assert_eq!(fresh.state.counts(), &fresh.counts);
    assert_eq!(fresh.state.best_iter(), 0);
}

#[test]
fn cost_only_scalar_solvers_share_point_progress() {
    check_point_solver(Brent::new());
    check_point_solver(GoldenSection::new());
}

#[test]
fn derivative_brent_publishes_matching_scalar_gradient() {
    let result = Executor::new(
        Quadratic,
        BrentDerivative::new(),
        FirstOrderState::new(0.5),
    )
    .require_evaluated_state()
    .max_iter(10)
    .run_with_solver()
    .unwrap();
    let (x, cost, gradient) = result.state.current().unwrap();
    assert_eq!(cost, Quadratic.cost(x).unwrap());
    assert_eq!(*gradient, Quadratic.gradient(x).unwrap());
    assert_eq!(result.state.gradient(), Some(gradient));
    assert_eq!(result.state.counts(), &result.counts);
    assert_eq!(result.state.gradient_evals(), result.counts.gradient_evals);
    assert_eq!(
        result.counts,
        EvalCounts {
            cost_evals: result.counts.cost_evals,
            gradient_evals: result.counts.cost_evals,
            ..EvalCounts::default()
        }
    );
}
