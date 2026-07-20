#![cfg(feature = "nalgebra")]

//! nalgebra-backend mirror of the `Vec<f64>` Solis-Wets suite
//! (`tests/solis_wets_vec.rs`): reduced to the checks that exercise the
//! per-backend trait wiring.

use basin::problems::Sphere;
use basin::{Executor, RhoTolerance, SolisWets, TerminationReason};
use nalgebra::DVector;

#[test]
fn same_seed_yields_identical_trajectory() {
    let run = || {
        Executor::from_start(
            Sphere::<DVector<f64>>::new(),
            SolisWets::new(42),
            DVector::from_vec(vec![2.0, -1.5]),
        )
        .max_iter(200)
        .run()
        .unwrap()
    };
    let result_a = run();
    let result_b = run();
    assert_eq!(result_a.cost(), result_b.cost());
    assert_eq!(result_a.param(), result_b.param());
}

#[test]
fn converges_on_sphere_5d_via_rho_tolerance() {
    let result = Executor::from_start(
        Sphere::<DVector<f64>>::new(),
        SolisWets::new(7),
        DVector::from_vec(vec![2.0, -1.0, 1.5, 0.5, -2.0]),
    )
    .terminate_on(RhoTolerance::new(1e-8))
    .max_iter(100_000)
    .run()
    .unwrap();

    assert_eq!(result.reason, TerminationReason::RhoTolerance);
    assert!(
        result.cost() < 1e-6,
        "sphere 5-D cost = {} (expected < 1e-6)",
        result.cost()
    );
}
