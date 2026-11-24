use basin::{
    BoxBounds, BrentDerivative, ConstraintDescription, ConstraintSet,
    Constraints, CostFunction, Executor, FirstOrderState, Gradient,
    TerminationCode,
};
use std::convert::Infallible;

struct Quartic;

impl CostFunction for Quartic {
    type Param = f64;
    type Output = f64;
    type Error = Infallible;

    fn cost(&self, x: &f64) -> Result<f64, Infallible> {
        assert!((0.0..=1.0).contains(x));
        Ok((x - 0.5).powi(4))
    }
}

impl Gradient for Quartic {
    type Gradient = f64;

    fn gradient(&self, x: &f64) -> Result<f64, Infallible> {
        Ok(4.0 * (x - 0.5).powi(3))
    }
}

impl Constraints for Quartic {
    type Set = ConstraintSet<BoxBounds<f64>>;

    fn constraints(
        &self,
    ) -> <Self::Set as ConstraintDescription>::Borrowed<'_> {
        ConstraintSet::new().with_bounds(BoxBounds::new(Some(&0.0), Some(&1.0)))
    }
}

#[test]
fn brent_shrinks_the_bracket_when_the_minimum_step_reaches_an_endpoint() {
    let result = Executor::new(
        Quartic,
        BrentDerivative::new(),
        FirstOrderState::new(0.25),
    )
    .max_iter(100)
    .run()
    .unwrap();

    assert_eq!(result.report.code(), TerminationCode::SolverConverged);
    assert!((result.best_param() - 0.5).abs() < 1e-8);
}
