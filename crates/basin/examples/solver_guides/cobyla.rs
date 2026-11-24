use basin::{
    Cobyla, Constrained, ConstraintSet, ConstraintSolveError, CostFunction,
    Executor, NonlinearInequalityValues, Termination, ViolationMeasure,
};
use std::convert::Infallible;

struct Disk;

impl CostFunction for Disk {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = Infallible;

    fn cost(&self, x: &Self::Param) -> Result<f64, Self::Error> {
        Ok(x[0] * x[1])
    }
}

impl NonlinearInequalityValues for Disk {
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
    let problem = Constrained::new(Disk, constraints);
    let solver = Cobyla::new()
        .with_initial_radius(0.5)
        .with_final_radius(1e-7);

    let result = Executor::from_start(problem, solver, vec![1.0, 1.0])
        .max_cost_evals(500)
        .run()
        .unwrap();

    assert!(matches!(
        result.report.termination,
        Termination::Converged(_)
    ));
    let (x, cost, violation) = result.state.best().unwrap();
    assert_eq!(violation.measure(), &ViolationMeasure::LInf);
    assert!(violation.value < 1e-5);
    assert!(x[0].powi(2) + x[1].powi(2) <= 1.0 + 1e-5);
    assert!((cost + 0.5).abs() < 1e-3);
}
