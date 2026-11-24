use basin::{CostFunction, Executor, Gradient, GradientDescent, Termination};
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

impl Gradient for Quadratic {
    type Gradient = Vec<f64>;

    fn gradient(&self, x: &Self::Param) -> Result<Self::Gradient, Self::Error> {
        Ok(vec![2.0 * (x[0] - 1.0), 8.0 * (x[1] + 2.0)])
    }
}

fn main() {
    let solver =
        GradientDescent::new(0.1).with_absolute_gradient_tolerance(1e-8);

    let result = Executor::from_start(Quadratic, solver, vec![-3.0, 5.0])
        .max_iter(1_000)
        .run()
        .unwrap();

    assert!(matches!(
        result.report.termination,
        Termination::Converged(_)
    ));
    assert!((result.param()[0] - 1.0).abs() < 1e-7);
    assert!((result.param()[1] + 2.0).abs() < 1e-7);
}
