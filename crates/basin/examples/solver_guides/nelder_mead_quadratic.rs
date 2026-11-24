use basin::{CostFunction, Executor, NelderMead, Termination};
use std::convert::Infallible;

struct Quadratic;

impl CostFunction for Quadratic {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = Infallible;

    fn cost(&self, x: &Self::Param) -> Result<f64, Self::Error> {
        Ok((x[0] - 1.0).powi(2) + 4.0 * (x[1] + 2.0).powi(2))
    }
}

fn main() {
    let solver = NelderMead::new()
        .with_absolute_simplex_size_tolerance(1e-9)
        .with_relative_simplex_size_tolerance(None)
        .with_absolute_simplex_cost_tolerance(1e-14)
        .with_relative_simplex_cost_tolerance(None);

    let result = Executor::from_start(Quadratic, solver, vec![-3.0, 5.0])
        .max_iter(2_000)
        .run()
        .unwrap();

    assert!(matches!(
        result.report.termination,
        Termination::Converged(_)
    ));
    assert!((result.param()[0] - 1.0).abs() < 1e-7);
    assert!((result.param()[1] + 2.0).abs() < 1e-7);
}
