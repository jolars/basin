use basin::{
    BoxBounds, ConstraintDescription, ConstraintSet, Constraints, CostFunction,
    Direct, Executor, PointState,
};
use std::convert::Infallible;

struct BoundedQuartic {
    lower: Vec<f64>,
    upper: Vec<f64>,
}

impl CostFunction for BoundedQuartic {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = Infallible;

    fn cost(&self, x: &Self::Param) -> Result<f64, Self::Error> {
        Ok(x.iter()
            .map(|v| 0.5 * (v.powi(4) - 16.0 * v * v + 5.0 * v))
            .sum())
    }
}

impl Constraints for BoundedQuartic {
    type Set = ConstraintSet<BoxBounds<Vec<f64>>>;

    fn constraints(
        &self,
    ) -> <Self::Set as ConstraintDescription>::Borrowed<'_> {
        ConstraintSet::new()
            .with_bounds(BoxBounds::new(Some(&self.lower), Some(&self.upper)))
    }
}

fn main() {
    let problem = BoundedQuartic {
        lower: vec![-5.0; 2],
        upper: vec![5.0; 2],
    };
    let solver = Direct::new();

    let result = Executor::new(problem, solver, PointState::new(vec![0.0; 2]))
        .target_objective(-78.33)
        .max_evaluations(basin::EvaluationKind::Cost, 2_000)
        .run()
        .unwrap();

    let incumbent = result.selected_record().expect("expected an incumbent");
    assert!(incumbent.cost.unwrap() < -78.33);
    assert!(incumbent.param.iter().all(|v| (-5.0..=5.0).contains(v)));
}
