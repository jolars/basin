//! Unconstrained `Lbfgs<Unbounded>` convergence tests across backends.
//!
//! The bounded counterpart is `tests/lbfgsb_{vec,nalgebra,faer}.rs`;
//! this file mirrors the Rosenbrock 2D smoke test but on the
//! Nocedal–Wright two-loop-recursion path that requires no
//! [`BoxConstraints`](basin::BoxConstraints) impl.

use basin::solver::lbfgs::{Bounded, Unbounded};
use basin::{
    CostFunction, Executor, Gradient, GradientTolerance, Lbfgs, LbfgsState, Lbfgsb, MaxIter,
    MoreThuente,
};

/// 2-D Rosenbrock: `f(x) = (1 − x₀)² + 100 (x₁ − x₀²)²`. Used by all
/// three backend variants below — only the parameter type changes.
mod rosenbrock {
    pub fn cost(x0: f64, x1: f64) -> f64 {
        (1.0 - x0).powi(2) + 100.0 * (x1 - x0 * x0).powi(2)
    }
    pub fn grad(x0: f64, x1: f64) -> (f64, f64) {
        let dfdx0 = -2.0 * (1.0 - x0) - 400.0 * x0 * (x1 - x0 * x0);
        let dfdx1 = 200.0 * (x1 - x0 * x0);
        (dfdx0, dfdx1)
    }
}

#[test]
fn rosenbrock_vec() {
    struct Rosen;
    impl CostFunction for Rosen {
        type Param = Vec<f64>;
        type Output = f64;
        type Error = std::convert::Infallible;
        fn cost(&self, x: &Vec<f64>) -> Result<f64, std::convert::Infallible> {
            Ok(rosenbrock::cost(x[0], x[1]))
        }
    }
    impl Gradient for Rosen {
        type Gradient = Vec<f64>;
        fn gradient(&self, x: &Vec<f64>) -> Result<Vec<f64>, std::convert::Infallible> {
            Ok({
                let (a, b) = rosenbrock::grad(x[0], x[1]);
                vec![a, b]
            })
        }
    }

    let state = LbfgsState::new(vec![-1.2, 1.0], 5);
    let result = Executor::new(Rosen, Lbfgs::<Unbounded>::new(), state)
        .terminate_on(MaxIter(200))
        .terminate_on(GradientTolerance(1e-8))
        .run()
        .unwrap();

    assert!(result.cost() < 1e-10, "cost = {}", result.cost());
    assert!(
        (result.param()[0] - 1.0).abs() < 1e-4 && (result.param()[1] - 1.0).abs() < 1e-4,
        "x = {:?}",
        result.param()
    );
}

#[cfg(feature = "nalgebra")]
#[test]
fn rosenbrock_nalgebra() {
    use nalgebra::DVector;

    struct Rosen;
    impl CostFunction for Rosen {
        type Param = DVector<f64>;
        type Output = f64;
        type Error = std::convert::Infallible;
        fn cost(&self, x: &DVector<f64>) -> Result<f64, std::convert::Infallible> {
            Ok(rosenbrock::cost(x[0], x[1]))
        }
    }
    impl Gradient for Rosen {
        type Gradient = DVector<f64>;
        fn gradient(&self, x: &DVector<f64>) -> Result<DVector<f64>, std::convert::Infallible> {
            Ok({
                let (a, b) = rosenbrock::grad(x[0], x[1]);
                DVector::from_vec(vec![a, b])
            })
        }
    }

    let state = LbfgsState::new(DVector::from_vec(vec![-1.2, 1.0]), 5);
    let result = Executor::new(Rosen, Lbfgs::<Unbounded>::new(), state)
        .terminate_on(MaxIter(200))
        .terminate_on(GradientTolerance(1e-8))
        .run()
        .unwrap();

    assert!(result.cost() < 1e-10, "cost = {}", result.cost());
    assert!(
        (result.param()[0] - 1.0).abs() < 1e-4 && (result.param()[1] - 1.0).abs() < 1e-4,
        "x = {:?}",
        result.param()
    );
}

#[cfg(feature = "faer")]
#[test]
fn rosenbrock_faer() {
    use faer::Col;

    struct Rosen;
    impl CostFunction for Rosen {
        type Param = Col<f64>;
        type Output = f64;
        type Error = std::convert::Infallible;
        fn cost(&self, x: &Col<f64>) -> Result<f64, std::convert::Infallible> {
            Ok(rosenbrock::cost(x[0], x[1]))
        }
    }
    impl Gradient for Rosen {
        type Gradient = Col<f64>;
        fn gradient(&self, x: &Col<f64>) -> Result<Col<f64>, std::convert::Infallible> {
            Ok({
                let (a, b) = rosenbrock::grad(x[0], x[1]);
                Col::from_fn(2, |i| if i == 0 { a } else { b })
            })
        }
    }

    let x0 = Col::from_fn(2, |i| if i == 0 { -1.2 } else { 1.0 });
    let state = LbfgsState::new(x0, 5);
    let result = Executor::new(Rosen, Lbfgs::<Unbounded>::new(), state)
        .terminate_on(MaxIter(200))
        .terminate_on(GradientTolerance(1e-8))
        .run()
        .unwrap();

    assert!(result.cost() < 1e-10, "cost = {}", result.cost());
    assert!(
        (result.param()[0] - 1.0).abs() < 1e-4 && (result.param()[1] - 1.0).abs() < 1e-4,
        "x = {:?}",
        (result.param()[0], result.param()[1])
    );
}

#[cfg(feature = "ndarray")]
#[test]
fn rosenbrock_ndarray() {
    use ndarray::{Array1, array};

    struct Rosen;
    impl CostFunction for Rosen {
        type Param = Array1<f64>;
        type Output = f64;
        type Error = std::convert::Infallible;
        fn cost(&self, x: &Array1<f64>) -> Result<f64, std::convert::Infallible> {
            Ok(rosenbrock::cost(x[0], x[1]))
        }
    }
    impl Gradient for Rosen {
        type Gradient = Array1<f64>;
        fn gradient(&self, x: &Array1<f64>) -> Result<Array1<f64>, std::convert::Infallible> {
            Ok({
                let (a, b) = rosenbrock::grad(x[0], x[1]);
                array![a, b]
            })
        }
    }

    let state = LbfgsState::new(array![-1.2, 1.0], 5);
    let result = Executor::new(Rosen, Lbfgs::<Unbounded>::new(), state)
        .terminate_on(MaxIter(200))
        .terminate_on(GradientTolerance(1e-8))
        .run()
        .unwrap();

    assert!(result.cost() < 1e-10, "cost = {}", result.cost());
    assert!(
        (result.param()[0] - 1.0).abs() < 1e-4 && (result.param()[1] - 1.0).abs() < 1e-4,
        "x = {:?}",
        result.param()
    );
}

/// `Lbfgsb` must remain a transparent alias for `Lbfgs<Bounded>`. Any
/// drift here would break the iteration-parity test's import and every
/// other downstream call site that holds an `Lbfgsb<...>` value.
#[test]
fn lbfgsb_alias_compiles() {
    let _: Lbfgsb = Lbfgs::<Bounded>::new();
    let _: Lbfgs<Bounded, MoreThuente> = Lbfgsb::new();
    // `Lbfgs::default()` resolves to the default mode (Bounded) and
    // default line search (MoreThuente); same identity as above.
    let _: Lbfgsb = Lbfgs::default();
}
