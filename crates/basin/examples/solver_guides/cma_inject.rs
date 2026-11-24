use basin::{
    CmaEs, CmaInject, CostFunction, DenseMatrix, Executor, NelderMead,
    PopulationProgress,
};
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
    let outer = CmaEs::<Vec<f64>, DenseMatrix>::new(7, 0.5);
    let inner = NelderMead::adaptive();
    let solver = CmaInject::with_inner_solver(outer, inner)
        .with_k(1)
        .with_inner_max_iter(50);

    let result = Executor::new(
        Sphere,
        solver,
        PopulationProgress::from_point(vec![2.0, -2.0]),
    )
    .max_iter(100)
    .run()
    .unwrap();

    let incumbent = result.incumbent_record().expect("expected an incumbent");
    assert!(incumbent.cost < 1e-4);
    assert!(incumbent.param.iter().all(|v| v.abs() < 0.01));
}
