#![cfg(feature = "faer")]

//! L-Bfgs-B convergence tests over the faer backend. Mirrors
//! `tests/lbfgsb_vec.rs` to confirm the `Col<f64>` impl of
//! [`basin::backend::AsFloatSliceMut`] plumbs through correctly.

use basin::problems::BoothBoxed;
use basin::{
    BoxConstraints, CostFunction, Executor, Gradient, LbfgsState, Lbfgsb,
    MaxIter, ProjectedGradientTolerance,
};
use faer::Col;

struct Rosen {
    l: Col<f64>,
    u: Col<f64>,
}
impl CostFunction for Rosen {
    type Param = Col<f64>;
    type Output = f64;
    type Error = std::convert::Infallible;
    fn cost(&self, x: &Col<f64>) -> Result<f64, std::convert::Infallible> {
        Ok((1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0] * x[0]).powi(2))
    }
}
impl Gradient for Rosen {
    type Gradient = Col<f64>;
    fn gradient(
        &self,
        x: &Col<f64>,
    ) -> Result<Col<f64>, std::convert::Infallible> {
        Ok({
            let dfdx0 =
                -2.0 * (1.0 - x[0]) - 400.0 * x[0] * (x[1] - x[0] * x[0]);
            let dfdx1 = 200.0 * (x[1] - x[0] * x[0]);
            Col::from_fn(2, |i| if i == 0 { dfdx0 } else { dfdx1 })
        })
    }
}
impl BoxConstraints for Rosen {
    fn lower(&self) -> &Col<f64> {
        &self.l
    }
    fn upper(&self) -> &Col<f64> {
        &self.u
    }
}

#[test]
fn unbounded_rosenbrock_2d_converges() {
    let problem = Rosen {
        l: Col::from_fn(2, |_| f64::NEG_INFINITY),
        u: Col::from_fn(2, |_| f64::INFINITY),
    };
    let lower = problem.l.clone();
    let upper = problem.u.clone();
    let state = LbfgsState::new(
        Col::from_fn(2, |i| if i == 0 { -1.2 } else { 1.0 }),
        5,
    );

    let result = Executor::new(problem, Lbfgsb::new(), state)
        .terminate_on(MaxIter(200))
        .terminate_on(ProjectedGradientTolerance::new(lower, upper, 1e-8))
        .run()
        .unwrap();

    assert!(result.cost() < 1e-10, "cost = {}", result.cost());
    assert!(
        (result.param()[0] - 1.0).abs() < 1e-4
            && (result.param()[1] - 1.0).abs() < 1e-4,
        "x = ({}, {})",
        result.param()[0],
        result.param()[1]
    );
}

#[test]
fn booth_at_corner_converges() {
    let problem = BoothBoxed::<Col<f64>>::new(
        Col::from_fn(2, |_| -1.0),
        Col::from_fn(2, |_| 1.0),
    );
    let lower = Col::from_fn(2, |_| -1.0);
    let upper = Col::from_fn(2, |_| 1.0);
    let state = LbfgsState::new(Col::from_fn(2, |_| 0.0), 5);

    let result = Executor::new(problem, Lbfgsb::new(), state)
        .terminate_on(MaxIter(100))
        .terminate_on(ProjectedGradientTolerance::new(lower, upper, 1e-8))
        .run()
        .unwrap();

    assert!(
        (result.param()[0] - 1.0).abs() < 1e-5
            && (result.param()[1] - 1.0).abs() < 1e-5,
        "x = ({}, {})",
        result.param()[0],
        result.param()[1]
    );
    assert!(
        (result.cost() - 20.0).abs() < 1e-8,
        "cost = {}",
        result.cost()
    );
}

#[test]
fn booth_slack_bounds_recover_unconstrained_minimum() {
    let problem = BoothBoxed::<Col<f64>>::new(
        Col::from_fn(2, |_| -5.0),
        Col::from_fn(2, |_| 5.0),
    );
    let lower = Col::from_fn(2, |_| -5.0);
    let upper = Col::from_fn(2, |_| 5.0);
    let state = LbfgsState::new(Col::from_fn(2, |_| 0.0), 5);

    let result = Executor::new(problem, Lbfgsb::new(), state)
        .terminate_on(MaxIter(100))
        .terminate_on(ProjectedGradientTolerance::new(lower, upper, 1e-10))
        .run()
        .unwrap();

    assert!(
        (result.param()[0] - 1.0).abs() < 1e-4
            && (result.param()[1] - 3.0).abs() < 1e-4,
        "x = ({}, {})",
        result.param()[0],
        result.param()[1]
    );
    assert!(result.cost() < 1e-8, "cost = {}", result.cost());
}
