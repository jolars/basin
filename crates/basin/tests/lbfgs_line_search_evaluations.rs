use std::convert::Infallible;

use basin::{
    BoxConstraints, CostFunction, Executor, Gradient, GradientState,
    LbfgsState, Lbfgsb,
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

impl BoxConstraints for Quadratic {
    fn lower(&self) -> &Vec<f64> {
        &self.lower
    }

    fn upper(&self) -> &Vec<f64> {
        &self.upper
    }
}

#[test]
fn lbfgs_reuses_the_line_search_evaluation() {
    let result = Executor::new(
        Quadratic::new(),
        Lbfgsb::new().unbounded(),
        LbfgsState::new(vec![1.0], 10),
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
        LbfgsState::new(vec![1.0], 10),
    )
    .max_iter(1)
    .run()
    .unwrap();

    assert_eq!(result.param(), &[0.0]);
    assert_eq!(result.cost_evals(), 2);
    assert_eq!(result.state.gradient_evals(), 2);
}
