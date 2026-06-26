//! Integration tests for [`CmaInject`] with Nelder-Mead inner on the
//! nalgebra backend.
//!
//! Two tests below: convergence on Rosenbrock 2-D (the canonical
//! memetic-CMA-ES showcase; Hansen 2011 §4 reports `n×` speedup from
//! injection on this function), and cost-eval aggregation across the
//! outer/inner composition boundary.
//!
//! For the LM-inner variant see `cma_inject_lm_nalgebra.rs`; for
//! L-BFGS-B inner see `bounded_cma_inject_lbfgsb_nalgebra.rs`; for the
//! `SolverFailed`-bubbling contract test (CONTRIBUTING.md "Solver composition"
//! rule 3) see `cma_inject_solver_failed_bubbles.rs`.

#![cfg(feature = "nalgebra")]

use basin::problems::{Rosenbrock, Sphere};
use basin::{CmaEs, CmaEsState, CmaInject, Executor, NelderMead};
use nalgebra::{DMatrix, DVector};

/// Rosenbrock 2-D from `(-1, 1)`: the canonical non-convex banana
/// valley CMA-ES is famously good at. The point of this test is to
/// verify that injecting Nelder-Mead refinements doesn't *break* CMA's
/// convergence (per Hansen 2011 §3: "All update equations starting
/// from (5) are formulated relative to the original sample
/// distribution. This means we are, in principle, free to change the
/// distribution before each iteration step.").
///
/// We do *not* assert speedup vs. vanilla CMA-ES; Hansen's reported
/// `n×` speedup was with gradient/Newton injection, not derivative-free
/// NM polish, and empirically NM polish doesn't beat vanilla CMA-ES on
/// Rosenbrock in eval-count terms. The memetic value is on
/// multi-modal and ill-conditioned constrained problems
/// (Melo & Iacca 2014); convergence on Rosenbrock is the canary, not
/// the marketing.
#[test]
fn converges_on_rosenbrock_2d() {
    let m0 = DVector::from_vec(vec![-1.0, 1.0]);

    let cma = CmaEs::<DVector<f64>, DMatrix<f64>>::new(17);
    let solver = CmaInject::with_inner_solver(cma, NelderMead::adaptive())
        .with_k(1)
        .with_inner_max_iter(30);

    let result = Executor::new(
        Rosenbrock::<DVector<f64>>::new(),
        solver,
        CmaEsState::<DVector<f64>, DMatrix<f64>>::new(m0, 0.3),
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

/// CmaInject must roll inner Nelder-Mead `cost_evals` into the outer
/// state's eval counter (CONTRIBUTING.md "Solver composition" rule 1).
/// Compare CmaInject vs vanilla CmaEs on the same seed/budget/state:
/// the memetic variant invokes the inner `k` times per outer iter, so
/// its public `cost_evals()` must come back strictly larger by at
/// least the inner-init evaluations (`n + 1` NM vertices) plus the
/// per-iter re-evaluation (`+1`). A loose lower bound suffices;
/// NelderMead may terminate early on some iters via `CostTolerance`
/// internals, so we don't pin an exact count.
#[test]
fn aggregates_inner_cost_evals_into_outer() {
    let m0 = DVector::from_vec(vec![1.0; 5]);
    let n = 5usize;
    let outer_iters: u64 = 20;
    let inner_iters: u64 = 30;
    let k: usize = 1;

    // Vanilla CMA-ES baseline.
    let vanilla = Executor::new(
        Sphere::<DVector<f64>>::new(),
        CmaEs::<DVector<f64>, DMatrix<f64>>::new(7),
        CmaEsState::<DVector<f64>, DMatrix<f64>>::new(m0.clone(), 0.3),
    )
    .max_iter(outer_iters)
    .run()
    .unwrap();

    // Memetic variant on the same seed and outer budget.
    let cma = CmaEs::<DVector<f64>, DMatrix<f64>>::new(7);
    let solver = CmaInject::with_inner_solver(cma, NelderMead::adaptive())
        .with_k(k)
        .with_inner_max_iter(inner_iters);

    let memetic = Executor::new(
        Sphere::<DVector<f64>>::new(),
        solver,
        CmaEsState::<DVector<f64>, DMatrix<f64>>::new(m0, 0.3),
    )
    .max_iter(outer_iters)
    .run()
    .unwrap();

    // Per outer iter (after iter 0; CmaInject skips iter-0 injection),
    // CmaInject does at minimum: NM init (n+1 evals) + 1 re-eval =
    // (n+2) extra evals. Across (outer_iters − 0 init-only) iters with
    // k = 1 that's ≥ outer_iters · (n + 2). Using a slightly weaker
    // lower bound to absorb the iter-0 skip + any early-terminating
    // NM iter:
    let min_extra = (outer_iters.saturating_sub(1)) * (k as u64) * (n as u64 + 2);
    assert!(
        memetic.cost_evals() >= vanilla.cost_evals() + min_extra,
        "memetic cost_evals = {} should exceed vanilla {} by at least \
         {} (outer iters × k × (n+2) for NM init + re-eval)",
        memetic.cost_evals(),
        vanilla.cost_evals(),
        min_extra
    );
}
