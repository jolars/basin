//! Least-squares solvers publish shared points and preserve raw work categories.

use basin::{
    DenseMatrix, EvaluationKind, Executor, GaussNewton, Jacobian,
    LevenbergMarquardt, LevenbergMarquardtQr, PointState, Residual, Solver,
    State,
};
use std::convert::Infallible;

struct Linear;
impl Residual for Linear {
    type Param = Vec<f64>;
    type Output = Vec<f64>;
    type Error = Infallible;
    fn residual(&self, x: &Vec<f64>) -> Result<Vec<f64>, Infallible> {
        Ok(x.iter().map(|x| x - 1.0).collect())
    }
}
impl Jacobian for Linear {
    type Jacobian = DenseMatrix;
    fn jacobian(&self, x: &Vec<f64>) -> Result<DenseMatrix, Infallible> {
        use basin::MatrixIdentity;
        Ok(DenseMatrix::identity(x.len()))
    }
}

fn check<So: Solver<Linear, PointState<Vec<f64>>, Error = Infallible>>(
    solver: So,
) {
    let result = Executor::new(Linear, solver, PointState::new(vec![3.0]))
        .require_evaluated_state()
        .max_evaluations(EvaluationKind::Residual, 2)
        .max_iter(10)
        .run_with_solver()
        .unwrap();
    assert_eq!(result.counts.residual_evals, 2);
    assert_eq!(result.state.cost_evals(), 0);
    assert_eq!(result.state.counts(), &result.counts);
    let (x, cost) = result.state.current().unwrap();
    assert_eq!(cost, 0.5 * (x[0] - 1.0).powi(2));
    let fresh = Executor::new(Linear, result.solver, result.state)
        .require_evaluated_state()
        .max_iter(0)
        .run_with_solver()
        .unwrap();
    assert_eq!(fresh.state.iter(), 0);
    assert_eq!(fresh.state.best_iter(), 0);
    assert_eq!(fresh.counts.residual_evals, 1);
    assert_eq!(fresh.counts.jacobian_evals, 1);
    assert_eq!(fresh.state.counts(), &fresh.counts);
}

#[test]
fn residual_drivers_share_point_progress_and_category_budgets() {
    check(GaussNewton::<Vec<f64>, DenseMatrix>::new());
    check(LevenbergMarquardt::<Vec<f64>, DenseMatrix>::new());
    check(LevenbergMarquardtQr::<Vec<f64>, DenseMatrix>::new());
}
