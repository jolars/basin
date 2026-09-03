//! Faer-backend smoke tests for [`CmaInject`] with Nelder-Mead inner.
//! Convergence on Rosenbrock 2-D plus the eval-aggregation lower bound
//! confirm the per-backend trait wiring works through the composition
//! boundary; the deeper algorithmic invariants are covered by the
//! nalgebra mirror test (`tests/cma_inject_nalgebra.rs`).

#![cfg(feature = "faer_all")]

use crate::backend_aliases::faer::{Col, Mat};
use basin::problems::{Rosenbrock, Sphere};
use basin::{CmaEs, CmaEsState, CmaInject, Executor, NelderMead};

/// Rosenbrock 2-D from `(-1, 1)`: injecting Nelder-Mead refinements
/// must not break CMA's convergence on the faer backend.
#[test]
fn converges_on_rosenbrock_2d() {
    let m0 = Col::<f64>::from_fn(2, |i| if i == 0 { -1.0 } else { 1.0 });

    let cma = CmaEs::<Col<f64>, Mat<f64>>::new(17);
    let solver = CmaInject::with_inner_solver(cma, NelderMead::adaptive())
        .with_k(1)
        .with_inner_max_iter(30);

    let result = Executor::new(
        Rosenbrock::<Col<f64>>::new(),
        solver,
        CmaEsState::<Col<f64>, Mat<f64>>::new(m0, 0.3),
    )
    .max_iter(200)
    .run()
    .unwrap();

    let p = result.param();
    assert!(
        (p[0] - 1.0).abs() < 1e-3 && (p[1] - 1.0).abs() < 1e-3,
        "rosenbrock 2-D iterate = ({}, {}), expected ≈ (1, 1) within 1e-3",
        p[0],
        p[1]
    );
}

/// Inner Nelder-Mead `cost_evals` roll into the outer state's counter
/// (CONTRIBUTING.md "Solver composition" rule 1); same lower bound as
/// the nalgebra mirror.
#[test]
fn aggregates_inner_cost_evals_into_outer() {
    let m0 = Col::<f64>::from_fn(5, |_| 1.0);
    let n = 5usize;
    let outer_iters: u64 = 20;
    let inner_iters: u64 = 30;
    let k: usize = 1;

    // Vanilla CMA-ES baseline.
    let vanilla = Executor::new(
        Sphere::<Col<f64>>::new(),
        CmaEs::<Col<f64>, Mat<f64>>::new(7),
        CmaEsState::<Col<f64>, Mat<f64>>::new(m0.clone(), 0.3),
    )
    .max_iter(outer_iters)
    .run()
    .unwrap();

    // Memetic variant on the same seed and outer budget.
    let cma = CmaEs::<Col<f64>, Mat<f64>>::new(7);
    let solver = CmaInject::with_inner_solver(cma, NelderMead::adaptive())
        .with_k(k)
        .with_inner_max_iter(inner_iters);

    let memetic = Executor::new(
        Sphere::<Col<f64>>::new(),
        solver,
        CmaEsState::<Col<f64>, Mat<f64>>::new(m0, 0.3),
    )
    .max_iter(outer_iters)
    .run()
    .unwrap();

    let min_extra =
        (outer_iters.saturating_sub(1)) * (k as u64) * (n as u64 + 2);
    assert!(
        memetic.cost_evals() >= vanilla.cost_evals() + min_extra,
        "memetic cost_evals = {} should exceed vanilla {} by at least \
         {} (outer iters × k × (n+2) for NM init + re-eval)",
        memetic.cost_evals(),
        vanilla.cost_evals(),
        min_extra
    );
}

#[path = "support/backend_aliases.rs"]
mod backend_aliases;
