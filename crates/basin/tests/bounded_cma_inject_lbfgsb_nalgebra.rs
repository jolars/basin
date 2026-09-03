//! Integration tests for [`BoundedCmaInject`] with [`Lbfgsb`] inner on
//! the nalgebra backend (S13b).
//!
//! Two tests: convergence on `BoothBoxed` with bounds that contain the
//! global optimum (so the L-Bfgs-B inner can polish to high precision),
//! and work-unit aggregation (L-Bfgs-B `cost_evals` + `gradient_evals`
//! roll into the outer's `cost_evals`).

#![cfg(feature = "nalgebra_all")]

use crate::backend_aliases::nalgebra::{DMatrix, DVector};
use basin::problems::BoothBoxed;
use basin::{BoundedCmaEs, BoundedCmaInject, CmaEsState, Executor, Lbfgsb};

/// BoundedCmaEs + L-Bfgs-B on Booth with slack bounds `[-5, 5]²`. The
/// global min `(1, 3)` is strictly interior, so both the outer
/// BoundPenalty (no active bounds at the optimum) and the inner
/// L-Bfgs-B (interior iterates) drive convergence. Assert
/// `‖x* − (1, 3)‖_∞ ≤ 1e-6`: L-Bfgs-B precision in the smooth
/// quadratic basin.
///
/// The starting mean is `(0, 2)`, close enough that Hansen's
/// Mahalanobis clip (Hansen 2011 eq. 4) doesn't heavily attenuate the
/// inner's full-amplitude L-Bfgs-B refinements. With a distant start
/// the test would still converge but require many more outer iters
/// as CMA's mean drifts toward the basin; that's a CMA-ES property,
/// not an injection property, and tested elsewhere.
#[test]
fn converges_on_booth_boxed_slack() {
    let lower = DVector::from_vec(vec![-5.0, -5.0]);
    let upper = DVector::from_vec(vec![5.0, 5.0]);
    let problem = BoothBoxed::<DVector<f64>>::new(lower.clone(), upper.clone());

    let m0 = DVector::from_vec(vec![0.0, 2.0]);

    let cma = BoundedCmaEs::<DVector<f64>, DMatrix<f64>>::new(19);
    let solver = BoundedCmaInject::with_inner_solver(cma, Lbfgsb::new())
        .with_k(1)
        .with_inner_max_iter(50);

    let result = Executor::new(
        problem,
        solver,
        CmaEsState::<DVector<f64>, DMatrix<f64>>::new(m0, 0.5),
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

/// BoundedCmaInject's L-Bfgs-B work-unit closure rolls both
/// `cost_evals` and `gradient_evals` into the outer state's
/// `cost_evals`.
///
/// Neither vanilla nor memetic registers the `CmaEsTolerance` (TolX)
/// criterion, so each runs the full `outer_iters` without early
/// termination; otherwise the memetic variant converges faster on TolX
/// and the eval-count comparison gets meaningless. The lower-bound check: per outer iter
/// (after iter 0; CmaInject skips iter-0 injection), L-Bfgs-B init
/// contributes 1 cost + 1 gradient eval (2 work units), and the
/// outer's re-evaluation after clipping is 1 cost, so the floor is
/// `(outer_iters − 1) · k · 3`.
#[test]
fn aggregates_lbfgsb_work_into_outer() {
    let lower = DVector::from_vec(vec![-5.0, -5.0]);
    let upper = DVector::from_vec(vec![5.0, 5.0]);

    let m0 = DVector::from_vec(vec![0.0, 2.0]);
    let outer_iters: u64 = 20;
    let inner_iters: u64 = 50;
    let k: usize = 1;

    // Vanilla BoundedCmaEs baseline; no TolX criterion registered so it
    // runs the full budget.
    let vanilla = Executor::new(
        BoothBoxed::<DVector<f64>>::new(lower.clone(), upper.clone()),
        BoundedCmaEs::<DVector<f64>, DMatrix<f64>>::new(29),
        CmaEsState::<DVector<f64>, DMatrix<f64>>::new(m0.clone(), 0.5),
    )
    .max_iter(outer_iters)
    .run()
    .unwrap();

    // Memetic variant on the same seed and outer budget; no TolX
    // criterion for the same reason.
    let cma = BoundedCmaEs::<DVector<f64>, DMatrix<f64>>::new(29);
    let solver = BoundedCmaInject::with_inner_solver(cma, Lbfgsb::new())
        .with_k(k)
        .with_inner_max_iter(inner_iters);

    let memetic = Executor::new(
        BoothBoxed::<DVector<f64>>::new(lower, upper),
        solver,
        CmaEsState::<DVector<f64>, DMatrix<f64>>::new(m0, 0.5),
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
