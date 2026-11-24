use basin::{
    BoxBounds, ConstraintDescription, ConstraintSet, Constraints, CostFunction,
    Executor, PopulationProgress, RandomSearch,
};
use std::convert::Infallible;

struct BoundedQuadratic {
    lower: Vec<f64>,
    upper: Vec<f64>,
}

impl CostFunction for BoundedQuadratic {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = Infallible;

    fn cost(&self, x: &Self::Param) -> Result<f64, Self::Error> {
        Ok((x[0] - 1.0).powi(2) + (x[1] + 2.0).powi(2))
    }
}

impl Constraints for BoundedQuadratic {
    type Set = ConstraintSet<BoxBounds<Vec<f64>>>;

    fn constraints(
        &self,
    ) -> <Self::Set as ConstraintDescription>::Borrowed<'_> {
        ConstraintSet::new()
            .with_bounds(BoxBounds::new(Some(&self.lower), Some(&self.upper)))
    }
}

fn main() {
    let problem = BoundedQuadratic {
        lower: vec![-5.0; 2],
        upper: vec![5.0; 2],
    };
    let solver = RandomSearch::new(32, 42);

    let result =
        Executor::new(problem, solver, PopulationProgress::<Vec<f64>>::empty())
            .max_iter(1_000)
            .run()
            .unwrap();

    assert!(result.cost() < 0.05);
    assert!((-5.0..=5.0).contains(&result.param()[0]));
    assert!((-5.0..=5.0).contains(&result.param()[1]));
}
