use basin::{CmaEs, CostFunction, DenseMatrix, Executor};
use std::convert::Infallible;

struct Sphere;

impl CostFunction for Sphere {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = Infallible;

    fn cost(&self, x: &Self::Param) -> Result<f64, Self::Error> {
        Ok(x.iter().map(|v| v * v).sum())
    }
}

fn main() {
    let solver = CmaEs::<Vec<f64>, DenseMatrix>::new(7, 0.5)
        .with_absolute_distribution_size_tolerance(1e-10);

    let result = Executor::from_start(Sphere, solver, vec![1.0; 5])
        .max_iter(300)
        .run()
        .unwrap();

    assert!(result.cost() < 1e-6);
    let incumbent = result.incumbent_record().expect("expected an incumbent");
    assert!(incumbent.cost <= result.cost());
    assert!(incumbent.param.iter().all(|v| v.abs() < 1e-3));
}
