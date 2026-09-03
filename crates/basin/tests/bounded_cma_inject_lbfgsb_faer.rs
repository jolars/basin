//! Faer-backend smoke tests for [`BoundedCmaInject`] with [`Lbfgsb`]
//! inner. Convergence on `BoothBoxed` plus the work-unit aggregation
//! lower bound confirm the per-backend trait wiring works through the
//! composition boundary; the deeper algorithmic invariants are covered
//! by the nalgebra mirror test
//! (`tests/bounded_cma_inject_lbfgsb_nalgebra.rs`).

#![cfg(feature = "faer_all")]

use crate::backend_aliases::faer::{Col, Mat};
use basin::problems::BoothBoxed;
use basin::{BoundedCmaEs, BoundedCmaInject, CmaEsState, Executor, Lbfgsb};

/// BoundedCmaEs + L-Bfgs-B on Booth with slack bounds `[-5, 5]²`; the
/// global min `(1, 3)` is strictly interior, so the inner polish must
/// reach it to L-Bfgs-B precision.
#[test]
fn converges_on_booth_boxed_slack() {
    let lower = Col::<f64>::from_fn(2, |_| -5.0);
    let upper = Col::<f64>::from_fn(2, |_| 5.0);
    let problem = BoothBoxed::<Col<f64>>::new(lower, upper);

    let m0 = Col::<f64>::from_fn(2, |i| if i == 0 { 0.0 } else { 2.0 });

    let cma = BoundedCmaEs::<Col<f64>, Mat<f64>>::new(19);
    let solver = BoundedCmaInject::with_inner_solver(cma, Lbfgsb::new())
        .with_k(1)
        .with_inner_max_iter(50);

    let result = Executor::new(
        problem,
        solver,
        CmaEsState::<Col<f64>, Mat<f64>>::new(m0, 0.5),
    )
    .max_iter(200)
    .run()
    .unwrap();

    let p = result.param();
    let err = (p[0] - 1.0).abs().max((p[1] - 3.0).abs());
    assert!(
        err <= 1e-6,
        "booth-boxed iterate = ({}, {}), expected ≈ (1, 3) within 1e-6 (err = {})",
        p[0],
        p[1],
        err
    );
}

/// L-Bfgs-B work units (cost + gradient evals) roll into the outer
/// state's `cost_evals` (CONTRIBUTING.md "Solver composition" rule 1);
/// same lower bound as the nalgebra mirror.
#[test]
fn aggregates_lbfgsb_work_into_outer() {
    let lower = Col::<f64>::from_fn(2, |_| -5.0);
    let upper = Col::<f64>::from_fn(2, |_| 5.0);

    let m0 = Col::<f64>::from_fn(2, |i| if i == 0 { 0.0 } else { 2.0 });
    let outer_iters: u64 = 20;
    let inner_iters: u64 = 50;
    let k: usize = 1;

    // Vanilla BoundedCmaEs baseline; no TolX criterion registered so it
    // runs the full budget.
    let vanilla = Executor::new(
        BoothBoxed::<Col<f64>>::new(lower.clone(), upper.clone()),
        BoundedCmaEs::<Col<f64>, Mat<f64>>::new(29),
        CmaEsState::<Col<f64>, Mat<f64>>::new(m0.clone(), 0.5),
    )
    .max_iter(outer_iters)
    .run()
    .unwrap();

    // Memetic variant on the same seed and outer budget.
    let cma = BoundedCmaEs::<Col<f64>, Mat<f64>>::new(29);
    let solver = BoundedCmaInject::with_inner_solver(cma, Lbfgsb::new())
        .with_k(k)
        .with_inner_max_iter(inner_iters);

    let memetic = Executor::new(
        BoothBoxed::<Col<f64>>::new(lower, upper),
        solver,
        CmaEsState::<Col<f64>, Mat<f64>>::new(m0, 0.5),
    )
    .max_iter(outer_iters)
    .run()
    .unwrap();

    let min_extra = (outer_iters.saturating_sub(1)) * (k as u64) * 3;
    assert!(
        memetic.cost_evals() >= vanilla.cost_evals() + min_extra,
        "memetic cost_evals = {} should exceed vanilla {} by at least \
         {} (outer iters × k × (L-Bfgs-B init cost + gradient + re-eval))",
        memetic.cost_evals(),
        vanilla.cost_evals(),
        min_extra
    );
}

#[path = "support/backend_aliases.rs"]
mod backend_aliases;
