use basin::{
    Constrained, ConstraintSet, CostFunction, DenseMatrix, Executor, Lincoa,
    LinearBlock, Termination,
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

fn main() {
    let constraints =
        ConstraintSet::new().with_linear_inequalities(LinearBlock::new(
            DenseMatrix::from_row_slice(1, 2, &[1.0, 1.0]),
            vec![2.0],
        ));
    let problem = Constrained::new(Quadratic, constraints);
    let solver = Lincoa::new()
        .with_initial_radius(0.5)
        .with_final_radius(1e-7);

    let result = Executor::from_start(problem, solver, vec![0.0, 0.0])
        .max_cost_evals(500)
        .run()
        .unwrap();

    assert!(matches!(
        result.report.termination,
        Termination::Converged(_)
    ));
    let incumbent = result.selected_record().expect("expected an incumbent");
    let x = incumbent.param;
    assert!(x[0] + x[1] <= 2.0 + 1e-8);
    assert!((x[0] - 1.0).abs() < 1e-3);
    assert!((x[1] - 1.0).abs() < 1e-3);
}
