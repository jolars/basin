#![cfg(feature = "ndarray")]

//! Bfgs convergence over the ndarray backend (`Array1<f64>` and `Array2<f64>`).
//!
//! Mirrors `tests/bfgs.rs` (nalgebra), `tests/bfgs_vec.rs` (`Vec<f64>`), and
//! `tests/bfgs_faer.rs` (faer): the same generic `Solver` impl drives
//! ndarray's dense inverse-Hessian via `MatVec` + `GeneralRankOneUpdate` on
//! `Array2<f64>`.

use basin::problems::Rosenbrock;
use basin::{
    Bfgs, Executor, GradientTolerance, NdarrayQuasiNewtonState,
    TerminationReason,
};
use ndarray::array;

#[test]
fn bfgs_converges_on_rosenbrock() {
    let problem = Rosenbrock::<ndarray::Array1<f64>>::default();
    let initial = array![-1.2, 1.0];

    let result = Executor::new(
        problem,
        Bfgs::new(),
        NdarrayQuasiNewtonState::new(initial),
    )
    .max_iter(100)
    .run()
    .unwrap();

    assert!(
        result.cost() < 1e-8,
        "expected near-zero cost, got {}",
        result.cost()
    );
    assert!(
        (result.param()[0] - 1.0).abs() < 1e-4,
        "x[0] = {}",
        result.param()[0]
    );
    assert!(
        (result.param()[1] - 1.0).abs() < 1e-4,
        "x[1] = {}",
        result.param()[1]
    );
}

#[test]
fn bfgs_terminates_on_gradient_tolerance() {
    let problem = Rosenbrock::<ndarray::Array1<f64>>::default();
    let initial = array![-1.2, 1.0];

    let result = Executor::new(
        problem,
        Bfgs::new(),
        NdarrayQuasiNewtonState::new(initial),
    )
    .max_iter(200)
    .terminate_on(GradientTolerance(1e-6))
    .run()
    .unwrap();

    assert_eq!(result.reason, TerminationReason::GradientTolerance);
    assert!(result.cost() < 1e-10, "cost = {}", result.cost());
}
