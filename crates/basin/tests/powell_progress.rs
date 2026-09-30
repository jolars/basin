//! Progress ownership and lifecycle for Powell's quadratic-model solvers.

use basin::{
    Bobyqa, BoxConstraints, CostFunction, CountsMirror, DenseMatrix,
    EvalCounts, Executor, Lincoa, LinearConstraints, Newuoa, PointState, State,
};
use std::convert::Infallible;

struct Bowl {
    lower: Vec<f64>,
    upper: Vec<f64>,
}
impl Bowl {
    fn new(n: usize) -> Self {
        Self {
            lower: vec![-5.0; n],
            upper: vec![5.0; n],
        }
    }
}
impl CostFunction for Bowl {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = Infallible;
    fn cost(&self, x: &Vec<f64>) -> Result<f64, Infallible> {
        Ok(x.iter()
            .enumerate()
            .map(|(i, x)| (i + 1) as f64 * x.powi(2))
            .sum())
    }
}
impl BoxConstraints for Bowl {
    fn lower(&self) -> &Vec<f64> {
        &self.lower
    }
    fn upper(&self) -> &Vec<f64> {
        &self.upper
    }
}
impl LinearConstraints for Bowl {
    type Matrix = DenseMatrix;
    fn lower(&self) -> Option<&Vec<f64>> {
        Some(&self.lower)
    }
    fn upper(&self) -> Option<&Vec<f64>> {
        Some(&self.upper)
    }
}

macro_rules! lifecycle {
    ($module:ident, $solver:ident, $state:ident) => {
        mod $module {
            use super::*;
            #[test]
            fn fresh_run_resets_bookkeeping_and_reevaluates() {
                let prior = Executor::from_start(
                    Bowl::new(2),
                    $solver::new(),
                    vec![2.0; 2],
                )
                .max_iter(3)
                .run_with_solver()
                .unwrap();
                assert_eq!(prior.iter(), 3);
                let fresh =
                    Executor::new(Bowl::new(2), prior.solver, prior.state)
                        .max_iter(0)
                        .run_with_solver()
                        .unwrap();
                assert_eq!(fresh.iter(), 0);
                assert_eq!(fresh.state.best_iter(), 0);
                assert_eq!(fresh.counts.cost_evals, 5);
                assert_eq!(fresh.state.cost_evals(), 5);
                assert_eq!(fresh.solver.rho(), Some(1.0));
                assert_eq!(fresh.state.counts(), &fresh.counts);
            }
            #[test]
            fn cost_reader_reports_only_cost_calls() {
                let mut state = $state::<Vec<f64>>::new(vec![2.0; 2]);
                let counts = EvalCounts {
                    cost_evals: 2,
                    gradient_evals: 3,
                    residual_evals: 5,
                    jacobian_evals: 7,
                    hessian_evals: 11,
                    hessian_product_evals: 13,
                };
                state.replace(vec![2.0; 2], 12.0);
                state.mirror(&counts);
                state.update_best();
                assert_eq!(state.cost_evals(), 2);
                assert_eq!(state.counts(), &counts);
                assert_eq!(state.best_counts(), Some(&counts));
                assert_eq!(state.best_cost_evals(), 2);
                state.increment_iter();
                state.mirror(&EvalCounts::default());
                state.update_best();
                assert_eq!(state.best_counts(), Some(&counts));
                assert_eq!(state.best_iter(), 0);
            }

            #[test]
            fn fresh_solver_rebuilds_model_for_a_different_dimension() {
                let used = Executor::from_start(
                    Bowl::new(2),
                    $solver::new(),
                    vec![2.0; 2],
                )
                .max_iter(4)
                .run_with_solver()
                .unwrap();
                let fresh = Executor::new(
                    Bowl::new(3),
                    used.solver,
                    PointState::new(vec![1.5; 3]),
                )
                .require_evaluated_state()
                .max_iter(10)
                .run_with_solver()
                .unwrap();
                let expected = Executor::from_start(
                    Bowl::new(3),
                    $solver::new(),
                    vec![1.5; 3],
                )
                .max_iter(10)
                .run_with_solver()
                .unwrap();
                assert_eq!(fresh.state, expected.state);
                assert_eq!(fresh.counts, expected.counts);
                assert_eq!(fresh.solver.rho(), expected.solver.rho());
            }

            #[test]
            fn owned_checkpoints_continue_without_reevaluating_or_resetting() {
                let make = || {
                    Executor::from_start(
                        Bowl::new(2),
                        $solver::new().with_final_radius(1e-12),
                        vec![2.0; 2],
                    )
                };
                let expected = make().max_iter(20).run_with_solver().unwrap();
                for split in [0, 1, 3, 7, 12] {
                    let checkpoint = make()
                        .max_iter(split)
                        .run_with_solver()
                        .unwrap()
                        .into_checkpoint();
                    let resumed = Executor::resume_from_checkpoint(
                        Bowl::new(2),
                        checkpoint,
                    )
                    .require_evaluated_state()
                    .max_iter(20)
                    .run_with_solver()
                    .unwrap();
                    assert_eq!(
                        resumed.state, expected.state,
                        "split at {split}"
                    );
                    assert_eq!(resumed.counts, expected.counts);
                    assert_eq!(resumed.solver.rho(), expected.solver.rho());
                }
            }

            #[cfg(feature = "serde")]
            #[test]
            fn serialized_checkpoints_preserve_the_model_and_radius_schedule() {
                type Checkpoint =
                    basin::ExactCheckpoint<$solver, PointState<Vec<f64>>>;
                let make = || {
                    Executor::from_start(
                        Bowl::new(2),
                        $solver::new().with_final_radius(1e-12),
                        vec![2.0; 2],
                    )
                };
                let expected = make().max_iter(20).run_with_solver().unwrap();
                let checkpoint = make()
                    .max_iter(7)
                    .run_with_solver()
                    .unwrap()
                    .into_checkpoint();
                let bytes = postcard::to_allocvec(&checkpoint).unwrap();
                let restored: Checkpoint =
                    postcard::from_bytes(&bytes).unwrap();
                let resumed =
                    Executor::resume_from_checkpoint(Bowl::new(2), restored)
                        .require_evaluated_state()
                        .max_iter(20)
                        .run_with_solver()
                        .unwrap();
                assert_eq!(resumed.state, expected.state);
                assert_eq!(resumed.counts, expected.counts);
                assert_eq!(
                    postcard::to_allocvec(&resumed.solver).unwrap(),
                    postcard::to_allocvec(&expected.solver).unwrap()
                );
                let fresh = Executor::new(
                    Bowl::new(2),
                    resumed.solver,
                    PointState::new(vec![2.0; 2]),
                )
                .max_iter(0)
                .run_with_solver()
                .unwrap();
                let initialized = make().max_iter(0).run_with_solver().unwrap();
                assert_eq!(fresh.state, initialized.state);
                assert_eq!(
                    postcard::to_allocvec(&fresh.solver).unwrap(),
                    postcard::to_allocvec(&initialized.solver).unwrap()
                );
            }
        }
    };
}
lifecycle!(newuoa, Newuoa, PointState);
lifecycle!(bobyqa, Bobyqa, PointState);
lifecycle!(lincoa, Lincoa, PointState);
