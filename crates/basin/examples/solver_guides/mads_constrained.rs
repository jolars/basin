use basin::{
    Constrained, ConstraintSet, ConstraintSolveError, CostFunction, Executor,
    Mads, NonlinearInequalityValues, ViolationMeasure,
};
use std::convert::Infallible;

struct DiskQuadratic;

impl CostFunction for DiskQuadratic {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = Infallible;

    fn cost(&self, x: &Self::Param) -> Result<f64, Self::Error> {
        Ok((x[0] - 2.0).powi(2) + x[1].powi(2))
    }
}

impl NonlinearInequalityValues for DiskQuadratic {
    type Param = Vec<f64>;
    type Values = Vec<f64>;
    type Error = Infallible;

    fn nonlinear_inequalities(
        &self,
        x: &Self::Param,
    ) -> Result<Self::Values, ConstraintSolveError<Self::Error>> {
        Ok(vec![x[0].powi(2) + x[1].powi(2) - 1.0])
    }
}

fn main() {
    let constraints = ConstraintSet::new().with_nonlinear_inequalities(1);
    let problem = Constrained::new(DiskQuadratic, constraints);
    let solver = Mads::new()
        .with_initial_poll_size(1.0)
        .with_minimum_poll_size(1e-7)
        .constrained();

    let result = Executor::from_start(problem, solver, vec![2.0, 0.0])
        .max_evaluations(basin::EvaluationKind::Cost, 5_000)
        .run()
        .unwrap();

    let (x, cost, violation) = result.state.best().unwrap();
    assert_eq!(violation.measure(), &ViolationMeasure::SquaredL2);
    assert!(violation.value < 1e-10);
    assert!(x[0].powi(2) + x[1].powi(2) <= 1.0 + 1e-5);
    assert!((cost - 1.0).abs() < 1e-3);
}
