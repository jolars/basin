//! Shared constrained workloads keep timing and allocation probes comparable.

use std::convert::Infallible;

use basin::{
    CostFunction, DenseMatrix, Executor, Gradient,
    OptimizationResultWithSolver, SelectedFirstOrderState, Slsqp, State,
    TerminationCode,
};

pub type Result =
    OptimizationResultWithSolver<SelectedFirstOrderState<Vec<f64>>, Slsqp>;

struct Hs71 {
    lower: Vec<f64>,
    upper: Vec<f64>,
}

impl basin::ProblemConstraints for Hs71 {
    type Constraints = basin::Constraints<
        basin::BoxBounds<Vec<f64>>,
        basin::NoLinearConstraints,
        basin::NonlinearBlocks<basin::NonlinearBlock, basin::NonlinearBlock>,
    >;
    fn constraint_set(
        &self,
    ) -> <Self::Constraints as basin::ConstraintDescription>::Borrowed<'_> {
        basin::Constraints::new()
            .with_bounds(basin::BoxBounds::new(
                Some(&self.lower),
                Some(&self.upper),
            ))
            .with_nonlinear_equalities(1)
            .with_nonlinear_inequalities(1)
    }
}
impl CostFunction for Hs71 {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = Infallible;

    fn cost(&self, x: &Vec<f64>) -> std::result::Result<f64, Infallible> {
        Ok(x[0] * x[3] * (x[0] + x[1] + x[2]) + x[2])
    }
}

impl Gradient for Hs71 {
    type Gradient = Vec<f64>;

    fn gradient(
        &self,
        x: &Vec<f64>,
    ) -> std::result::Result<Vec<f64>, Infallible> {
        Ok(vec![
            x[3] * (2.0 * x[0] + x[1] + x[2]),
            x[0] * x[3],
            x[0] * x[3] + 1.0,
            x[0] * (x[0] + x[1] + x[2]),
        ])
    }
}

impl basin::NonlinearInequalityValues for Hs71 {
    type Param = Vec<f64>;
    type Values = Vec<f64>;
    type Error = Infallible;
    fn nonlinear_inequalities(
        &self,
        x: &Self::Param,
    ) -> std::result::Result<
        Self::Values,
        basin::ConstraintSolveError<Self::Error>,
    > {
        { Ok(vec![25.0 - x.iter().product::<f64>()]) }
            .map_err(basin::ConstraintSolveError::Evaluation)
    }
}
impl basin::NonlinearEqualityValues for Hs71 {
    type Param = Vec<f64>;
    type Values = Vec<f64>;
    type Error = Infallible;
    fn nonlinear_equalities(
        &self,
        x: &Self::Param,
    ) -> std::result::Result<
        Self::Values,
        basin::ConstraintSolveError<Self::Error>,
    > {
        { Ok(Some(vec![x.iter().map(|v| v * v).sum::<f64>() - 40.0])) }
            .map_err(basin::ConstraintSolveError::Evaluation)?
            .ok_or(basin::ConstraintSolveError::InvalidConstraints(
                basin::ConstraintError::DimensionMismatch {
                    block: basin::ConstraintBlock::NonlinearEqualities,
                    axis: basin::ConstraintAxis::Length,
                    expected: 1,
                    actual: 0,
                },
            ))
    }
}

impl basin::ConstraintJacobian for Hs71 {
    type Param = Vec<f64>;
    type Jacobian = DenseMatrix;
    type Error = Infallible;
    fn constraint_jacobian(
        &self,
        x: &Self::Param,
    ) -> std::result::Result<
        Self::Jacobian,
        basin::ConstraintSolveError<Self::Error>,
    > {
        {
            let mut entries: Vec<_> = x.iter().map(|v| 2.0 * v).collect();
            entries.extend((0..4).map(|j| {
                -(0..4).filter(|&k| k != j).map(|k| x[k]).product::<f64>()
            }));
            Ok(DenseMatrix::from_row_slice(2, 4, &entries))
        }
        .map_err(basin::ConstraintSolveError::Evaluation)
    }
}

pub fn hs71() -> Result {
    Executor::from_start(
        Hs71 {
            lower: vec![1.0; 4],
            upper: vec![5.0; 4],
        },
        Slsqp::new().with_absolute_accuracy_tolerance(1e-10),
        vec![1.0, 5.0, 5.0, 1.0],
    )
    .max_iter(200)
    .run_with_solver()
    .unwrap()
}

pub fn verify_hs71(result: &Result) {
    assert_eq!(result.report.code(), TerminationCode::SolverConverged);
    let x = result.state.param();
    assert!(x.iter().all(|v| v.is_finite() && (1.0..=5.0).contains(v)));
    let cost = x[0] * x[3] * (x[0] + x[1] + x[2]) + x[2];
    assert!((cost - result.state.cost()).abs() < 1e-12);
    assert!((cost - 17.014_017_289_156).abs() < 1e-8);
    assert!((x.iter().map(|v| v * v).sum::<f64>() - 40.0).abs() < 1e-9);
    assert!(25.0 - x.iter().product::<f64>() < 1e-9);
    assert!(result.solver.stationarity().unwrap() < 1e-5);
    assert!(result.solver.complementarity().unwrap() < 1e-9);
}
