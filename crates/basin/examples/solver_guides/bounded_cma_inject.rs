use basin::{
    BoundedCmaEs, BoundedCmaInject, BoxBounds, Constrained, ConstraintSet,
    CostFunction, DenseMatrix, Executor, NelderMead, PopulationProgress,
    RelaxedInner,
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
    let bounds = BoxBounds::new(Some(vec![-5.0; 2]), Some(vec![5.0; 2]));
    let problem =
        Constrained::new(Sphere, ConstraintSet::new().with_bounds(bounds));
    let outer = BoundedCmaEs::<Vec<f64>, DenseMatrix>::new(7, 0.5);
    let inner = NelderMead::adaptive();
    let solver =
        BoundedCmaInject::with_inner_solver(outer, RelaxedInner::new(inner))
            .with_k(1)
            .with_inner_max_iter(50);

    let result = Executor::new(
        problem,
        solver,
        PopulationProgress::from_point(vec![2.0, -2.0]),
    )
    .max_iter(100)
    .run()
    .unwrap();

    let incumbent = result.selected_record().expect("expected an incumbent");
    assert!(incumbent.cost.unwrap() < 1e-4);
    assert!(incumbent.param.iter().all(|v| v.abs() < 0.01));
}
