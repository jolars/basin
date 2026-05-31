//! Integration tests for [`CmaInject`] with [`LevenbergMarquardt`]
//! inner on the nalgebra backend (S13a).
//!
//! Two tests: convergence on `RosenbrockResiduals` 2-D (CMA's global
//! stage hands a near-basin start to LM, which polishes to high
//! precision) and work-unit aggregation (LM `cost_evals` +
//! `gradient_evals` roll into the outer's `cost_evals`).

#![cfg(feature = "nalgebra")]

use basin::problems::RosenbrockResiduals;
use basin::{BasicPopulationState, CmaEs, CmaInject, Executor, LevenbergMarquardt};
use nalgebra::{DMatrix, DVector};

/// CMA-ES + LM on 2-D Rosenbrock-as-residuals. CMA from a wide-σ start
/// gets near `(1, 1)`; the inner LM (Nielsen damping, default tolerances)
/// drives the iterate down to high precision via the basin's quadratic
/// model. We assert `‖x* − (1, 1)‖_∞ ≤ 1e-6` — LM precision, not CMA
/// precision.
#[test]
fn converges_on_rosenbrock_residuals_2d() {
    let m0 = DVector::from_vec(vec![-1.2, 1.0]);
    let lambda = CmaEs::<DVector<f64>, DMatrix<f64>>::default_lambda(2);

    let cma = CmaEs::<DVector<f64>, DMatrix<f64>>::new(m0, 0.5, 11);
    let solver = CmaInject::with_inner_solver(cma, LevenbergMarquardt::new())
        .with_k(1)
        .with_inner_max_iter(50);

    let result = Executor::new(
        RosenbrockResiduals::<DVector<f64>>::new(),
        solver,
        BasicPopulationState::<DVector<f64>>::with_size(lambda),
    )
    .max_iter(100)
    .run()
    .unwrap();

    let p = result.param();
    let err = (p[0] - 1.0).abs().max((p[1] - 1.0).abs());
    assert!(
        err <= 1e-6,
        "rosenbrock-residuals 2-D iterate = ({}, {}), expected ≈ (1, 1) within 1e-6 (err = {})",
        p[0],
        p[1],
        err
    );
}

/// CmaInject's work-unit closure for LM rolls both `cost_evals`
/// (residual calls) and `gradient_evals` (Jacobian calls) into the
/// outer state's `cost_evals` (CONTRIBUTING.md "Solver composition" rule 1;
/// CMA-ES has no `gradient_evals` field, so derivative-eval counts
/// collapse honestly).
///
/// Lower-bound assertion: each outer iter (after iter 0; CmaInject
/// skips iter-0 injection) runs LM init (1 residual + 1 jacobian eval),
/// at least one LM next_iter (1 residual + 1 jacobian), plus the
/// outer's re-evaluation after clipping (1 cost). So `≥ outer_iters · k
/// · (4 + 1)` extra work units over vanilla — using a slightly weaker
/// lower bound to absorb the iter-0 skip and any LM early-termination
/// via the gradient-norm test.
#[test]
fn aggregates_lm_work_into_outer() {
    let m0 = DVector::from_vec(vec![-1.2, 1.0]);
    let n = 2usize;
    let lambda = CmaEs::<DVector<f64>, DMatrix<f64>>::default_lambda(n);
    let outer_iters: u64 = 20;
    let inner_iters: u64 = 50;
    let k: usize = 1;

    // Vanilla CMA-ES baseline.
    let vanilla = Executor::new(
        RosenbrockResiduals::<DVector<f64>>::new(),
        CmaEs::<DVector<f64>, DMatrix<f64>>::new(m0.clone(), 0.5, 23),
        BasicPopulationState::<DVector<f64>>::with_size(lambda),
    )
    .max_iter(outer_iters)
    .run()
    .unwrap();

    // Memetic variant on the same seed and outer budget.
    let cma = CmaEs::<DVector<f64>, DMatrix<f64>>::new(m0, 0.5, 23);
    let solver = CmaInject::with_inner_solver(cma, LevenbergMarquardt::new())
        .with_k(k)
        .with_inner_max_iter(inner_iters);

    let memetic = Executor::new(
        RosenbrockResiduals::<DVector<f64>>::new(),
        solver,
        BasicPopulationState::<DVector<f64>>::with_size(lambda),
    )
    .max_iter(outer_iters)
    .run()
    .unwrap();

    // Per outer iter (after iter 0), CmaInject does at minimum:
    // LM init (1 residual + 1 jacobian) + 1 re-eval = 3 work units.
    // Across (outer_iters − 1) iters with k = 1 that's a weak floor.
    let min_extra = (outer_iters.saturating_sub(1)) * (k as u64) * 3;
    assert!(
        memetic.cost_evals() >= vanilla.cost_evals() + min_extra,
        "memetic cost_evals = {} should exceed vanilla {} by at least \
         {} (outer iters × k × (LM init residual + jacobian + re-eval))",
        memetic.cost_evals(),
        vanilla.cost_evals(),
        min_extra
    );
}
