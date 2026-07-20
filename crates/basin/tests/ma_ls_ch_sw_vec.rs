//! `Vec<f64>`-backend smoke tests for [`MaLsChSw`] (no feature gate).
//!
//! Deliberately imports **no matrix type**: unlike the CMA variant, the
//! Solis-Wets chain configuration bounds only on the vector tier, so
//! this file doubles as the compile-time proof that `MaLsChSw` needs no
//! `linalg` tier. The deeper algorithmic invariants are covered by the
//! nalgebra mirror test (`tests/ma_ls_ch_sw_nalgebra.rs`).

use basin::problems::{RastriginBoxed, SphereBoxed};
use basin::{Executor, MaLsChSw, MaLsChSwState, MaxCostEvals};

#[test]
fn converges_on_sphere_d10() {
    let problem = SphereBoxed::new(vec![-5.0; 10], vec![5.0; 10]);
    let solver = MaLsChSw::<Vec<f64>>::new(7).with_pop_size(20);
    let result = Executor::new(problem, solver, MaLsChSwState::new())
        .max_iter(u64::MAX)
        .terminate_on(MaxCostEvals(20_000))
        .run()
        .unwrap();

    assert!(
        result.cost() < 1e-6,
        "Sphere(D=10) Vec<f64> cost {} should be < 1e-6 within 20k evals",
        result.cost()
    );
}

#[test]
fn makes_progress_on_rastrigin_d10() {
    let problem = RastriginBoxed::<Vec<f64>>::with_standard_bounds(10);
    let solver = MaLsChSw::<Vec<f64>>::new(42).with_pop_size(30);
    let result = Executor::new(problem, solver, MaLsChSwState::new())
        .max_iter(u64::MAX)
        .terminate_on(MaxCostEvals(50_000))
        .run()
        .unwrap();

    // Rastrigin D=10 starts around ~100-200 for random points in the
    // box; the memetic run must reach the low single digits. (The SW
    // chain is a weaker basin-refiner than CMA on Rastrigin's
    // ill-conditioned local structure, so the threshold is looser than
    // the CMA variant's 1.0.)
    assert!(
        result.cost() < 10.0,
        "Rastrigin(D=10) Vec<f64> cost {} should be < 10 within 50k evals",
        result.cost()
    );
}
