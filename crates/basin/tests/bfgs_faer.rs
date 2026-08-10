#![cfg(feature = "faer")]

//! Bfgs convergence over the faer backend (`Col<f64>` and `Mat<f64>`).
//!
//! Mirrors `tests/bfgs.rs` (nalgebra) and `tests/bfgs_vec.rs` (`Vec<f64>`):
//! the same generic `Solver` impl drives faer's dense inverse-Hessian via
//! `MatVec` + `GeneralRankOneUpdate` on `Mat<f64>`.

use basin::problems::Rosenbrock;
use basin::{
    Bfgs, Executor, FaerQuasiNewtonState, GradientTolerance, TerminationReason,
};
use faer::Col;

#[test]
fn bfgs_converges_on_rosenbrock() {
    let problem = Rosenbrock::<Col<f64>>::default();
    let initial = Col::from_fn(2, |i| if i == 0 { -1.2 } else { 1.0 });

    let result =
        Executor::new(problem, Bfgs::new(), FaerQuasiNewtonState::new(initial))
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
    let problem = Rosenbrock::<Col<f64>>::default();
    let initial = Col::from_fn(2, |i| if i == 0 { -1.2 } else { 1.0 });

    let result =
        Executor::new(problem, Bfgs::new(), FaerQuasiNewtonState::new(initial))
            .max_iter(200)
            .terminate_on(GradientTolerance(1e-6))
            .run()
            .unwrap();

    assert_eq!(result.reason, TerminationReason::GradientTolerance);
    assert!(result.cost() < 1e-10, "cost = {}", result.cost());
}
