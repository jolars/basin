//! Shared constrained workloads keep timing and allocation probes comparable.

use std::convert::Infallible;

use basin::{
    ConstraintJacobian, CostFunction, DenseMatrix, Executor, Gradient,
    NonlinearConstraints, OptimizationResult, Slsqp, SlsqpState, State,
    TerminationReason,
};

pub type Result = OptimizationResult<SlsqpState<Vec<f64>>>;

struct Hs71 {
    lower: Vec<f64>,
    upper: Vec<f64>,
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

impl NonlinearConstraints for Hs71 {
    type Matrix = DenseMatrix;

    fn lower(&self) -> Option<&Vec<f64>> {
        Some(&self.lower)
    }
    fn upper(&self) -> Option<&Vec<f64>> {
        Some(&self.upper)
    }
    fn num_nonlinear_constraints(&self) -> usize {
        1
    }
    fn nonlinear_constraints(
        &self,
        x: &Vec<f64>,
    ) -> std::result::Result<Vec<f64>, Infallible> {
        Ok(vec![25.0 - x.iter().product::<f64>()])
    }
    fn num_nonlinear_equalities(&self) -> usize {
        1
    }
    fn nonlinear_equalities(
        &self,
        x: &Vec<f64>,
    ) -> std::result::Result<Option<Vec<f64>>, Infallible> {
        Ok(Some(vec![x.iter().map(|v| v * v).sum::<f64>() - 40.0]))
    }
}

impl ConstraintJacobian for Hs71 {
    fn constraint_jacobian(
        &self,
        x: &Vec<f64>,
    ) -> std::result::Result<DenseMatrix, Infallible> {
        let mut entries: Vec<_> = x.iter().map(|v| 2.0 * v).collect();
        entries.extend((0..4).map(|j| {
            -(0..4).filter(|&k| k != j).map(|k| x[k]).product::<f64>()
        }));
        Ok(DenseMatrix::from_row_slice(2, 4, &entries))
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
    .run()
    .unwrap()
}

pub fn verify_hs71(result: &Result) {
    assert_eq!(result.reason, TerminationReason::SolverConverged);
    let x = result.state.param();
    assert!(x.iter().all(|v| v.is_finite() && (1.0..=5.0).contains(v)));
    let cost = x[0] * x[3] * (x[0] + x[1] + x[2]) + x[2];
    assert!((cost - result.state.cost()).abs() < 1e-12);
    assert!((cost - 17.014_017_289_156).abs() < 1e-8);
    assert!((x.iter().map(|v| v * v).sum::<f64>() - 40.0).abs() < 1e-9);
    assert!(25.0 - x.iter().product::<f64>() < 1e-9);
    assert!(result.state.stationarity().unwrap() < 1e-5);
    assert!(result.state.complementarity().unwrap() < 1e-9);
}
