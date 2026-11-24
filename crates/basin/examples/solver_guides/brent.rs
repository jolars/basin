use basin::{
    BoxBounds, Brent, ConstraintDescription, ConstraintSet, Constraints,
    CostFunction, Executor, MinimumBracketer, PointState, TerminationCode,
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
    let search = MinimumBracketer::new(0.0, 0.5, 1.0)
        .bracket(|x: f64| Ok::<_, Infallible>((x - 2.0).powi(2)))
        .unwrap();
    assert!(search.bracketed());

    let (lower, middle, upper) = search.bracket();
    let solver = Brent::new().with_absolute_position_tolerance(1e-10);
    let result = Executor::new(
        Quadratic { lower, upper },
        solver,
        PointState::new(middle),
    )
    .max_iter(100)
    .run()
    .unwrap();

    assert_eq!(result.report.code(), TerminationCode::SolverConverged);
    let incumbent = result.incumbent_record().expect("expected an incumbent");
    assert!((incumbent.param - 2.0).abs() < 1e-6);
    assert!(incumbent.cost < 1e-12);
    let total_evaluations = search.function_evals() + result.counts.cost_evals;
    assert!(total_evaluations > result.counts.cost_evals);
}
