use basin::{
    BoxBounds, ConstraintDescription, ConstraintSet, Constraints, CostFunction,
    Executor, GoldenSection, PointState, TerminationCode,
};
use std::convert::Infallible;

struct Quadratic {
    lower: f64,
    upper: f64,
}

impl CostFunction for Quadratic {
    type Param = f64;
    type Output = f64;
    type Error = Infallible;

    fn cost(&self, x: &f64) -> Result<f64, Self::Error> {
        Ok((x - 2.0).powi(2))
    }
}

impl Constraints for Quadratic {
    type Set = ConstraintSet<BoxBounds<f64>>;

    fn constraints(
        &self,
    ) -> <Self::Set as ConstraintDescription>::Borrowed<'_> {
        ConstraintSet::new()
            .with_bounds(BoxBounds::new(Some(&self.lower), Some(&self.upper)))
    }
}

fn main() {
    let problem = Quadratic {
        lower: 3.0,
        upper: 5.0,
    };
    let solver = GoldenSection::new().with_absolute_position_tolerance(1e-10);
    let result = Executor::new(problem, solver, PointState::new(4.0))
        .max_iter(100)
        .run()
        .unwrap();

    assert_eq!(result.report.code(), TerminationCode::SolverConverged);
    let incumbent = result.selected_record().expect("expected an incumbent");
    assert!((3.0..=5.0).contains(incumbent.param));
    assert!((incumbent.param - 3.0).abs() < 1e-6);
    assert!((incumbent.cost.unwrap() - 1.0).abs() < 1e-6);
}
