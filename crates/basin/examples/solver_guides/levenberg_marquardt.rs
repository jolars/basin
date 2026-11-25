use basin::{DenseMatrix, Executor, Jacobian, LevenbergMarquardt, Residual};
use std::convert::Infallible;

struct Fit;

impl Residual for Fit {
    type Param = Vec<f64>;
    type Output = Vec<f64>;
    type Error = Infallible;

    fn residual(&self, x: &Self::Param) -> Result<Self::Output, Self::Error> {
        Ok(vec![x[0] - 1.0, 2.0 * (x[1] - 2.0)])
    }
}

impl Jacobian for Fit {
    type Jacobian = DenseMatrix;

    fn jacobian(&self, _: &Self::Param) -> Result<Self::Jacobian, Self::Error> {
        Ok(DenseMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 2.0]))
    }
}

fn main() {
    let solver =
        LevenbergMarquardt::new().with_absolute_gradient_tolerance(1e-8);
    let result = Executor::from_start(Fit, solver, vec![0.0, 0.0])
        .max_iter(100)
        .run()
        .unwrap();

    assert!((result.param()[0] - 1.0).abs() < 1e-6);
    assert!((result.param()[1] - 2.0).abs() < 1e-6);
    assert!(result.current_record().unwrap().cost.unwrap() < 1e-12);
}
