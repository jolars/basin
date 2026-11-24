use basin::{
    Constrained, ConstraintJacobian, ConstraintSet, ConstraintSolveError,
    CostFunction, DenseMatrix, Executor, Gradient, NonlinearInequalityValues,
    Slsqp, State, TerminationCode,
};
use std::convert::Infallible;

struct Disk;

impl CostFunction for Disk {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = Infallible;

    fn cost(&self, x: &Self::Param) -> Result<f64, Self::Error> {
        Ok((x[0] - 2.0).powi(2) + x[1].powi(2))
    }
}

impl Gradient for Disk {
    type Gradient = Vec<f64>;

    fn gradient(&self, x: &Self::Param) -> Result<Self::Gradient, Self::Error> {
        Ok(vec![2.0 * (x[0] - 2.0), 2.0 * x[1]])
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

impl ConstraintJacobian for Disk {
    type Param = Vec<f64>;
    type Jacobian = DenseMatrix;
    type Error = Infallible;

    fn constraint_jacobian(
        &self,
        x: &Self::Param,
    ) -> Result<Self::Jacobian, ConstraintSolveError<Self::Error>> {
        Ok(DenseMatrix::from_row_slice(1, 2, &[2.0 * x[0], 2.0 * x[1]]))
    }
}

fn main() {
    let constraints = ConstraintSet::new().with_nonlinear_inequalities(1);
    let problem = Constrained::new(Disk, constraints);
    let solver = Slsqp::new().with_absolute_accuracy_tolerance(1e-10);
    let result = Executor::from_start(problem, solver, vec![2.0, 0.5])
        .max_iter(100)
        .run_with_solver()
        .unwrap();

    assert_eq!(result.report.code(), TerminationCode::SolverConverged);
    let x = result.state.param();
    assert!((x[0] - 1.0).abs() < 1e-5);
    assert!(x[1].abs() < 1e-5);
    assert!((result.state.cost() - 1.0).abs() < 1e-8);
    assert!(result.solver.constraint_violation().unwrap() < 1e-8);
}
