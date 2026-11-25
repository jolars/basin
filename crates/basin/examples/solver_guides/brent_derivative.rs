use basin::{
    BoxBounds, BrentDerivative, ConstraintDescription, ConstraintSet,
    Constraints, CostFunction, Executor, FirstOrderState, Gradient,
    Termination,
};
use std::convert::Infallible;

struct Cubic {
    lower: f64,
    upper: f64,
}

impl CostFunction for Cubic {
    type Param = f64;
    type Output = f64;
    type Error = Infallible;

    fn cost(&self, x: &f64) -> Result<f64, Self::Error> {
        Ok(x.powi(3) - 3.0 * x)
    }
}

impl Gradient for Cubic {
    type Gradient = f64;

    fn gradient(&self, x: &f64) -> Result<f64, Self::Error> {
        Ok(3.0 * x * x - 3.0)
    }
}

impl Constraints for Cubic {
    type Set = ConstraintSet<BoxBounds<f64>>;

    fn constraints(
        &self,
    ) -> <Self::Set as ConstraintDescription>::Borrowed<'_> {
        ConstraintSet::new()
            .with_bounds(BoxBounds::new(Some(&self.lower), Some(&self.upper)))
    }
}

fn main() {
    let problem = Cubic {
        lower: 0.0,
        upper: 2.0,
    };
    let solver = BrentDerivative::new().with_absolute_gradient_tolerance(1e-8);
    let result = Executor::new(problem, solver, FirstOrderState::new(0.5))
        .max_iter(100)
        .run()
        .unwrap();

    assert!(matches!(
        result.report.termination,
        Termination::Converged(_)
    ));
    let incumbent = result.selected_record().expect("expected an incumbent");
    assert!((incumbent.param - 1.0).abs() < 1e-6);
    assert!((incumbent.cost.unwrap() + 2.0).abs() < 1e-10);
    assert_eq!(result.counts.cost_evals, result.counts.gradient_evals);
}
