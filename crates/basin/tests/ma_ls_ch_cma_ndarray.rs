//! ndarray-backend smoke tests for [`MaLsChCma`]. Convergence on
//! Sphere and Rastrigin to confirm the per-backend trait wiring works;
//! the deeper algorithmic invariants are covered by the nalgebra
//! mirror test (`tests/ma_ls_ch_cma_nalgebra.rs`).

#![cfg(feature = "ndarray")]

use basin::problems::{RastriginBoxed, SphereBoxed};
use basin::{Executor, MaLsChCma, MaLsChState, MaxCostEvals};
use ndarray::{Array1, Array2};

#[test]
fn converges_on_sphere_d10() {
    let problem = SphereBoxed::new(
        Array1::from_elem(10, -5.0),
        Array1::from_elem(10, 5.0),
    );
    let solver =
        MaLsChCma::<Array1<f64>, Array2<f64>>::new(7).with_pop_size(20);
    let result = Executor::new(problem, solver, MaLsChState::new())
        .max_iter(u64::MAX)
        .terminate_on(MaxCostEvals(20_000))
        .run()
        .unwrap();

    assert!(
        result.cost() < 1e-6,
        "Sphere(D=10) ndarray cost {} should be < 1e-6 within 20k evals",
        result.cost()
    );
}

#[test]
fn converges_on_rastrigin_d10() {
    let problem = RastriginBoxed::<Array1<f64>>::with_standard_bounds(10);
    let solver =
        MaLsChCma::<Array1<f64>, Array2<f64>>::new(42).with_pop_size(30);
    let result = Executor::new(problem, solver, MaLsChState::new())
        .max_iter(u64::MAX)
        .terminate_on(MaxCostEvals(50_000))
        .run()
        .unwrap();

    assert!(
        result.cost() < 1.0,
        "Rastrigin(D=10) ndarray cost {} should be < 1.0 within 50k evals",
        result.cost()
    );
}
