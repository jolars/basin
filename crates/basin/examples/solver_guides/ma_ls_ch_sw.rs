use basin::{
    BoxBounds, Constrained, ConstraintSet, CostFunction, Executor, MaLsChSw,
    PopulationProgress, SamplingDomain,
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
    let solver = MaLsChSw::<Vec<f64>>::new(7)
        .with_pop_size(20)
        .with_ls_intensity(100);

    let result = Executor::new(
        SamplingDomain::new(problem),
        solver,
        PopulationProgress::<Vec<f64>>::empty(),
    )
    .max_cost_evals(10_000)
    .run()
    .unwrap();

    let incumbent = result.selected_record().expect("expected an incumbent");
    assert!(incumbent.cost.unwrap() < 1e-4);
    assert!(incumbent.param.iter().all(|v| v.abs() < 0.01));
}
