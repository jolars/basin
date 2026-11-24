use basin::{
    BoundedCmaEs, BoxBounds, Constrained, ConstraintSet, CostFunction,
    DenseMatrix, Executor,
};
use std::convert::Infallible;

struct Quadratic;

impl CostFunction for Quadratic {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = Infallible;

    fn cost(&self, x: &Self::Param) -> Result<f64, Self::Error> {
        assert!((-1.0..=1.0).contains(&x[0]));
        assert!((-2.0..=2.0).contains(&x[1]));
        Ok((x[0] - 2.0).powi(2) + (x[1] + 1.0).powi(2))
    }
}

fn main() {
    let bounds = BoxBounds::new(Some(vec![-1.0, -2.0]), Some(vec![1.0, 2.0]));
    let problem =
        Constrained::new(Quadratic, ConstraintSet::new().with_bounds(bounds));
    let solver = BoundedCmaEs::<Vec<f64>, DenseMatrix>::new(42, 0.6)
        .with_absolute_distribution_size_tolerance(1e-10);

    let result = Executor::from_start(problem, solver, vec![0.0, 1.0])
        .max_iter(500)
        .run()
        .unwrap();

    let incumbent = result.incumbent_record().expect("expected an incumbent");
    assert!((incumbent.cost - 1.0).abs() < 1e-8);
    assert!((incumbent.param[0] - 1.0).abs() < 1e-6);
    assert!((incumbent.param[1] + 1.0).abs() < 1e-4);
}
