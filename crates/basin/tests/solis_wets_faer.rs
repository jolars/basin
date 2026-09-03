#![cfg(feature = "faer_all")]

//! faer-backend mirror of the `Vec<f64>` Solis-Wets suite
//! (`tests/solis_wets_vec.rs`): reduced to the checks that exercise the
//! per-backend trait wiring.

use crate::backend_aliases::faer::Col;
use basin::problems::Sphere;
use basin::{Executor, RhoTolerance, SolisWets, TerminationReason};

fn col(values: &[f64]) -> Col<f64> {
    Col::<f64>::from_fn(values.len(), |i| values[i])
}

#[test]
fn same_seed_yields_identical_trajectory() {
    let run = || {
        Executor::from_start(
            Sphere::<Col<f64>>::new(),
            SolisWets::new(42),
            col(&[2.0, -1.5]),
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
        Sphere::<Col<f64>>::new(),
        SolisWets::new(7),
        col(&[2.0, -1.0, 1.5, 0.5, -2.0]),
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

#[path = "support/backend_aliases.rs"]
mod backend_aliases;
