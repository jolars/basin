use basin::{
    BoxBounds, ConstraintDescription, ConstraintSet, Constraints, CostFunction,
    Executor, Gbnm,
};
use std::convert::Infallible;

struct BoundedSphere {
    lower: Vec<f64>,
    upper: Vec<f64>,
}

impl CostFunction for BoundedSphere {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = Infallible;

    fn cost(&self, x: &Self::Param) -> Result<f64, Self::Error> {
        Ok(x.iter().map(|v| v * v).sum())
    }
}

impl Constraints for BoundedSphere {
    type Set = ConstraintSet<BoxBounds<Vec<f64>>>;

    fn constraints(
        &self,
    ) -> <Self::Set as ConstraintDescription>::Borrowed<'_> {
        ConstraintSet::new()
            .with_bounds(BoxBounds::new(Some(&self.lower), Some(&self.upper)))
    }
}

fn main() {
    let problem = BoundedSphere {
        lower: vec![-5.0; 2],
        upper: vec![5.0; 2],
    };
    let solver = Gbnm::new(42);

    let result = Executor::from_start(problem, solver, vec![4.0, 4.0])
        .max_evaluations(basin::EvaluationKind::Cost, 2_000)
        .run()
        .unwrap();

    let incumbent = result.selected_record().expect("expected an incumbent");
    assert!(incumbent.cost.unwrap() < 1e-4);
    assert!(incumbent.param.iter().all(|v| (-5.0..=5.0).contains(v)));
    assert!(
        incumbent.cost.unwrap()
            <= result.current_record().unwrap().cost.unwrap()
    );
}
