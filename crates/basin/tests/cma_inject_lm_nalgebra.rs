//! Integration tests for [`CmaInject`] with [`LevenbergMarquardt`]
//! inner on the nalgebra backend (S13a).
//!
//! Covers convergence on `RosenbrockResiduals` 2-D (CMA's global
//! stage hands a near-basin start to LM, which polishes to high
//! precision) and work-unit aggregation (LM `cost_evals` +
//! each evaluation category remains distinct), and continuation
//! after inner numerical no-progress stops.

#![cfg(all(feature = "nalgebra_all", feature = "problems"))]

use crate::backend_aliases::nalgebra::{DMatrix, DVector};
use basin::problems::RosenbrockResiduals;
use basin::{
    CmaEs, CmaInject, Executor, LevenbergMarquardt, PopulationProgress,
};

/// CMA-ES + LM on 2-D Rosenbrock-as-residuals. CMA from a wide-σ start
/// gets near `(1, 1)`; the inner LM (Nielsen damping, default tolerances)
/// drives the iterate down to high precision via the basin's quadratic
/// model. We assert `‖x* − (1, 1)‖_∞ ≤ 1e-6`, LM precision, not CMA
/// precision.
#[test]
fn converges_on_rosenbrock_residuals_2d() {
    let m0 = DVector::from_vec(vec![-1.2, 1.0]);

    let cma = CmaEs::<DVector<f64>, DMatrix<f64>>::new(11, 0.5);
    let solver = CmaInject::with_inner_solver(cma, LevenbergMarquardt::new())
        .with_k(1)
        .with_inner_max_iter(50);

    let result = Executor::new(
        RosenbrockResiduals::<DVector<f64>>::new(),
        solver,
        PopulationProgress::<DVector<f64>>::from_point(m0),
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

/// Inner residual and Jacobian calls retain their own categories. Injection
/// contributes one extra outer cost call per refined member.
#[test]
fn aggregates_lm_work_into_outer() {
    let m0 = DVector::from_vec(vec![-1.2, 1.0]);
    let outer_iters: u64 = 20;
    let inner_iters: u64 = 50;
    let k: usize = 1;

    // Vanilla CMA-ES baseline.
    let vanilla = Executor::new(
        RosenbrockResiduals::<DVector<f64>>::new(),
        CmaEs::<DVector<f64>, DMatrix<f64>>::new(23, 0.5),
        PopulationProgress::<DVector<f64>>::from_point(m0.clone()),
    )
    .max_iter(outer_iters)
    .run()
    .unwrap();

    // Memetic variant on the same seed and outer budget.
    let cma = CmaEs::<DVector<f64>, DMatrix<f64>>::new(23, 0.5);
    let solver = CmaInject::with_inner_solver(cma, LevenbergMarquardt::new())
        .with_k(k)
        .with_inner_max_iter(inner_iters);

    let memetic = Executor::new(
        RosenbrockResiduals::<DVector<f64>>::new(),
        solver,
        PopulationProgress::<DVector<f64>>::from_point(m0),
    )
    .max_iter(outer_iters)
    .run()
    .unwrap();

    let injections = outer_iters * k as u64;
    assert_eq!(memetic.cost_evals(), vanilla.cost_evals() + injections);
    assert!(memetic.state.counts().residual_evals >= injections);
    assert!(memetic.state.counts().jacobian_evals >= injections);
    assert_eq!(memetic.state.counts().gradient_evals, 0);
}

#[path = "support/backend_aliases.rs"]
mod backend_aliases;

#[test]
fn consumes_lm_no_progress_and_accounts_for_each_fresh_inner_run() {
    use basin::{CostFunction, Jacobian, Residual, State, TerminationReason};
    use std::{
        convert::Infallible,
        sync::{
            Arc, Mutex,
            atomic::{AtomicU64, Ordering},
        },
    };

    #[derive(Clone, Default)]
    struct Affine {
        costs: Arc<AtomicU64>,
        trials: Arc<Mutex<Vec<DVector<f64>>>>,
        jacobians: Arc<AtomicU64>,
    }
    impl CostFunction for Affine {
        type Param = DVector<f64>;
        type Output = f64;
        type Error = Infallible;
        fn cost(&self, x: &DVector<f64>) -> Result<f64, Infallible> {
            self.costs.fetch_add(1, Ordering::Relaxed);
            Ok(0.5 * ((x[0] - 2.).powi(2) + (x[1] - 3.).powi(2)))
        }
    }
    impl Residual for Affine {
        type Param = DVector<f64>;
        type Output = DVector<f64>;
        type Error = Infallible;
        fn residual(
            &self,
            x: &DVector<f64>,
        ) -> Result<DVector<f64>, Infallible> {
            self.trials.lock().unwrap().push(x.clone());
            Ok(DVector::from_vec(vec![x[0] - 2., x[1] - 3.]))
        }
    }
    impl Jacobian for Affine {
        type Jacobian = DMatrix<f64>;
        fn jacobian(
            &self,
            _: &DVector<f64>,
        ) -> Result<DMatrix<f64>, Infallible> {
            self.jacobians.fetch_add(1, Ordering::Relaxed);
            Ok(DMatrix::identity(2, 2))
        }
    }

    let counts = Affine::default();
    let solver = CmaInject::with_inner_solver(
        CmaEs::<DVector<f64>, DMatrix<f64>>::new(11, 0.1),
        LevenbergMarquardt::new()
            .with_tau(1e20)
            .with_absolute_gradient_tolerance(0.),
    )
    .with_k(1)
    .with_inner_max_iter(50);
    let result = Executor::new(
        counts.clone(),
        solver,
        PopulationProgress::from_point(DVector::from_vec(vec![1., 1.])),
    )
    .max_iter(4)
    .run()
    .unwrap();
    assert_eq!(result.reason, TerminationReason::MaxIter);
    assert_eq!(result.state.iter(), 4);
    assert!(result.param().iter().all(|value| value.is_finite()));
    assert!(result.cost().is_finite());
    // Each of the four inner runs exits after its first unchanged trial,
    // despite having a budget of fifty iterations.
    assert_eq!(result.state.counts().residual_evals, 8);
    assert_eq!(result.state.counts().jacobian_evals, 4);
    let trials = counts.trials.lock().unwrap();
    assert_eq!(trials.len(), 8);
    assert_eq!(counts.jacobians.load(Ordering::Relaxed), 4);
    for pair in trials.chunks_exact(2) {
        assert_eq!(pair[0], pair[1]);
    }
    assert_eq!(result.cost_evals(), counts.costs.load(Ordering::Relaxed));
}
