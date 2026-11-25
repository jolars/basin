use basin::{CostFunction, Executor, Gradient, HessianProduct, TrustRegion};
use std::convert::Infallible;

struct Quadratic;

impl CostFunction for Quadratic {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = Infallible;

    fn cost(&self, x: &Self::Param) -> Result<f64, Self::Error> {
        Ok(0.5 * (x[0].powi(2) + 100.0 * x[1].powi(2)))
    }
}

impl Gradient for Quadratic {
    type Gradient = Vec<f64>;

    fn gradient(&self, x: &Self::Param) -> Result<Self::Gradient, Self::Error> {
        Ok(vec![x[0], 100.0 * x[1]])
    }
}

impl HessianProduct for Quadratic {
    fn hessian_product(
        &self,
        _: &Self::Param,
        v: &Self::Param,
    ) -> Result<Self::Param, Self::Error> {
        Ok(vec![v[0], 100.0 * v[1]])
    }
}

fn main() {
    let solver =
        TrustRegion::matrix_free().with_absolute_gradient_tolerance(1e-10);
    let result = Executor::from_start(Quadratic, solver, vec![5.0, 1.0])
        .max_iter(100)
        .run()
        .unwrap();

    assert!(result.current_record().unwrap().cost.unwrap() < 1e-16);
}
