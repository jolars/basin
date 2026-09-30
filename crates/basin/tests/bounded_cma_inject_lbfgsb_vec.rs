//! `Vec<f64>`-backend smoke tests for [`BoundedCmaInject`] with
//! [`Lbfgsb`] inner (no feature gate). Convergence on `BoothBoxed`
//! plus the work-unit aggregation lower bound confirm the per-backend
//! trait wiring (the hand-rolled `DenseMatrix` covariance) works
//! through the composition boundary; the deeper algorithmic invariants
//! are covered by the nalgebra mirror test
//! (`tests/bounded_cma_inject_lbfgsb_nalgebra.rs`).

#![cfg(feature = "problems")]

use basin::problems::BoothBoxed;
use basin::{
    BoundedCmaEs, BoundedCmaInject, DenseMatrix, Executor, Lbfgsb,
    PopulationProgress,
};

/// BoundedCmaEs + L-Bfgs-B on Booth with slack bounds `[-5, 5]²`; the
/// global min `(1, 3)` is strictly interior, so the inner polish must
/// reach it to L-Bfgs-B precision.
#[test]
fn converges_on_booth_boxed_slack() {
    let lower = vec![-5.0, -5.0];
    let upper = vec![5.0, 5.0];
    let problem = BoothBoxed::<Vec<f64>>::new(lower, upper);

    let m0 = vec![0.0, 2.0];

    let cma = BoundedCmaEs::<Vec<f64>, DenseMatrix>::new(19, 0.5);
    let solver = BoundedCmaInject::with_inner_solver(cma, Lbfgsb::new())
        .with_k(1)
        .with_inner_max_iter(50);

    let result = Executor::new(
        problem,
        solver,
        PopulationProgress::<Vec<f64>>::from_point(m0),
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

/// Inner cost and gradient calls retain their own categories in outer progress.
#[test]
fn aggregates_lbfgsb_work_into_outer() {
    let lower = vec![-5.0, -5.0];
    let upper = vec![5.0, 5.0];

    let m0 = vec![0.0, 2.0];
    let outer_iters: u64 = 20;
    let inner_iters: u64 = 50;
    let k: usize = 1;

    // Vanilla BoundedCmaEs baseline; no TolX criterion registered so it
    // runs the full budget.
    let vanilla = Executor::new(
        BoothBoxed::<Vec<f64>>::new(lower.clone(), upper.clone()),
        BoundedCmaEs::<Vec<f64>, DenseMatrix>::new(29, 0.5),
        PopulationProgress::<Vec<f64>>::from_point(m0.clone()),
    )
    .max_iter(outer_iters)
    .run()
    .unwrap();

    // Memetic variant on the same seed and outer budget.
    let cma = BoundedCmaEs::<Vec<f64>, DenseMatrix>::new(29, 0.5);
    let solver = BoundedCmaInject::with_inner_solver(cma, Lbfgsb::new())
        .with_k(k)
        .with_inner_max_iter(inner_iters);

    let memetic = Executor::new(
        BoothBoxed::<Vec<f64>>::new(lower, upper),
        solver,
        PopulationProgress::<Vec<f64>>::from_point(m0),
    )
    .max_iter(outer_iters)
    .run()
    .unwrap();

    let injections = outer_iters * k as u64;
    assert!(memetic.cost_evals() >= vanilla.cost_evals() + 2 * injections);
    assert!(memetic.state.counts().gradient_evals >= injections);
    assert_eq!(memetic.state.counts().residual_evals, 0);
}
