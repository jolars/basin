use basin::{
    Backtracking, BarrierMethod, Constrained, ConstraintSet, CostFunction,
    DenseMatrix, Executor, Gradient, GradientDescent, LinearBlock,
    TerminationCode,
};
use std::convert::Infallible;

struct Quadratic;

impl CostFunction for Quadratic {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = Infallible;

    fn cost(&self, x: &Self::Param) -> Result<f64, Self::Error> {
        Ok((x[0] - 2.0).powi(2) + (x[1] - 2.0).powi(2))
    }
}

impl Gradient for Quadratic {
    type Gradient = Vec<f64>;

    fn gradient(&self, x: &Self::Param) -> Result<Self::Gradient, Self::Error> {
        Ok(vec![2.0 * (x[0] - 2.0), 2.0 * (x[1] - 2.0)])
    }
}

fn main() {
    let constraints =
        ConstraintSet::new().with_linear_inequalities(LinearBlock::new(
            DenseMatrix::from_row_slice(1, 2, &[1.0, 1.0]),
            vec![2.0],
        ));
    let problem = Constrained::new(Quadratic, constraints);
    let inner = GradientDescent::with_line_search(Backtracking::new())
        .with_absolute_gradient_tolerance(1e-8);
    let solver = BarrierMethod::with_inner_solver(inner);
    let result = Executor::from_start(problem, solver, vec![0.0, 0.0])
        .max_iter(50)
        .run()
        .unwrap();

    assert_eq!(result.report.code(), TerminationCode::SolverConverged);
    let x = result.param();
    assert!(x[0] + x[1] < 2.0);
    assert!((x[0] - 1.0).abs() < 1e-4);
    assert!((x[1] - 1.0).abs() < 1e-4);
    assert!((result.cost() - 2.0).abs() < 1e-4);
}
