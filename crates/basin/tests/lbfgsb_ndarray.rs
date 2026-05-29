#![cfg(feature = "ndarray")]

//! L-Bfgs-B convergence tests over the ndarray backend. Mirrors
//! `tests/lbfgsb_faer.rs` to confirm the `Array1<f64>` impl of
//! [`basin::backend::AsFloatSliceMut`] plumbs through correctly.

use basin::problems::BoothBoxed;
use basin::{
    BoxConstraints, CostFunction, Executor, Gradient, LbfgsState, Lbfgsb, MaxIter,
    ProjectedGradientTolerance,
};
use ndarray::{Array1, array};

struct Rosen {
    l: Array1<f64>,
    u: Array1<f64>,
}
impl CostFunction for Rosen {
    type Param = Array1<f64>;
    type Output = f64;
    type Error = std::convert::Infallible;
    fn cost(&self, x: &Array1<f64>) -> Result<f64, std::convert::Infallible> {
        Ok((1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0] * x[0]).powi(2))
    }
}
impl Gradient for Rosen {
    type Gradient = Array1<f64>;
    fn gradient(&self, x: &Array1<f64>) -> Result<Array1<f64>, std::convert::Infallible> {
        Ok({
            let dfdx0 = -2.0 * (1.0 - x[0]) - 400.0 * x[0] * (x[1] - x[0] * x[0]);
            let dfdx1 = 200.0 * (x[1] - x[0] * x[0]);
            array![dfdx0, dfdx1]
        })
    }
}
impl BoxConstraints for Rosen {
    fn lower(&self) -> &Array1<f64> {
        &self.l
    }
    fn upper(&self) -> &Array1<f64> {
        &self.u
    }
}

#[test]
fn unbounded_rosenbrock_2d_converges() {
    let problem = Rosen {
        l: Array1::from_elem(2, f64::NEG_INFINITY),
        u: Array1::from_elem(2, f64::INFINITY),
    };
    let lower = problem.l.clone();
    let upper = problem.u.clone();
    let state = LbfgsState::new(array![-1.2, 1.0], 5);

    let result = Executor::new(problem, Lbfgsb::new(), state)
        .terminate_on(MaxIter(200))
        .terminate_on(ProjectedGradientTolerance::new(lower, upper, 1e-8))
        .run()
        .unwrap();

    assert!(result.cost() < 1e-10, "cost = {}", result.cost());
    assert!(
        (result.param()[0] - 1.0).abs() < 1e-4 && (result.param()[1] - 1.0).abs() < 1e-4,
        "x = ({}, {})",
        result.param()[0],
        result.param()[1]
    );
}

#[test]
fn booth_at_corner_converges() {
    let problem =
        BoothBoxed::<Array1<f64>>::new(Array1::from_elem(2, -1.0), Array1::from_elem(2, 1.0));
    let lower = Array1::from_elem(2, -1.0);
    let upper = Array1::from_elem(2, 1.0);
    let state = LbfgsState::new(Array1::from_elem(2, 0.0), 5);

    let result = Executor::new(problem, Lbfgsb::new(), state)
        .terminate_on(MaxIter(100))
        .terminate_on(ProjectedGradientTolerance::new(lower, upper, 1e-8))
        .run()
        .unwrap();

    assert!(
        (result.param()[0] - 1.0).abs() < 1e-5 && (result.param()[1] - 1.0).abs() < 1e-5,
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
    let problem =
        BoothBoxed::<Array1<f64>>::new(Array1::from_elem(2, -5.0), Array1::from_elem(2, 5.0));
    let lower = Array1::from_elem(2, -5.0);
    let upper = Array1::from_elem(2, 5.0);
    let state = LbfgsState::new(Array1::from_elem(2, 0.0), 5);

    let result = Executor::new(problem, Lbfgsb::new(), state)
        .terminate_on(MaxIter(100))
        .terminate_on(ProjectedGradientTolerance::new(lower, upper, 1e-10))
        .run()
        .unwrap();

    assert!(
        (result.param()[0] - 1.0).abs() < 1e-4 && (result.param()[1] - 3.0).abs() < 1e-4,
        "x = ({}, {})",
        result.param()[0],
        result.param()[1]
    );
    assert!(result.cost() < 1e-8, "cost = {}", result.cost());
}
