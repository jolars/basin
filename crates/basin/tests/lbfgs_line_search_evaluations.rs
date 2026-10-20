use std::convert::Infallible;

use basin::{
    CostFunction, Executor, FirstOrderState, Gradient, GradientState, Lbfgsb,
};

struct Quadratic {
    lower: Vec<f64>,
    upper: Vec<f64>,
}

impl Quadratic {
    fn new() -> Self {
        Self {
            lower: vec![-2.0],
            upper: vec![2.0],
        }
    }
}

impl basin::Constraints for Quadratic {
    type Set = basin::ConstraintSet<basin::BoxBounds<Vec<f64>>>;
    fn constraints(
        &self,
    ) -> <Self::Set as basin::ConstraintDescription>::Borrowed<'_> {
        basin::ConstraintSet::new().with_bounds(basin::BoxBounds::new(
            Some(&self.lower),
            Some(&self.upper),
        ))
    }
}
impl CostFunction for Quadratic {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = Infallible;

    fn cost(&self, x: &Vec<f64>) -> Result<f64, Infallible> {
        Ok(0.5 * x[0] * x[0])
    }
}

impl Gradient for Quadratic {
    type Gradient = Vec<f64>;

    fn gradient(&self, x: &Vec<f64>) -> Result<Vec<f64>, Infallible> {
        Ok(vec![x[0]])
    }
}

#[test]
fn lbfgs_reuses_the_line_search_evaluation() {
    let result = Executor::new(
        basin::SamplingDomain::new(Quadratic::new()),
        Lbfgsb::new().unbounded(),
        FirstOrderState::new(vec![1.0]),
    )
    .max_iter(1)
    .run()
    .unwrap();

    assert_eq!(result.param(), &[0.0]);
    assert_eq!(result.cost_evals(), 2);
    assert_eq!(result.state.gradient_evals(), 2);
}

#[test]
fn lbfgsb_reuses_the_line_search_evaluation() {
    let result = Executor::new(
        Quadratic::new(),
        Lbfgsb::new(),
        FirstOrderState::new(vec![1.0]),
    )
    .max_iter(1)
    .run()
    .unwrap();

    assert_eq!(result.param(), &[0.0]);
    assert_eq!(result.cost_evals(), 2);
    assert_eq!(result.state.gradient_evals(), 2);
}
