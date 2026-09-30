//! MADS progress and solver-owned mesh schedules.

use basin::{
    BoxConstraints, CostFunction, Executor, Mads,
    NonlinearInequalityConstraints, Problem, Solver,
};
use basin::{PointState, SelectedState, State};
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
        Ok(x.iter().map(|x| x * x).sum())
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
impl NonlinearInequalityConstraints for Bowl {
    fn num_constraints(&self) -> usize {
        1
    }
    fn constraints(&self, x: &Vec<f64>) -> Result<Vec<f64>, Infallible> {
        Ok(vec![1.0 - x[0]])
    }
}

macro_rules! fresh {
    ($test:ident, $solver:expr) => {
        #[test]
        fn $test() {
            let prior =
                Executor::from_start(Bowl::new(2), $solver, vec![2.0; 2])
                    .max_iter(4)
                    .run_with_solver()
                    .unwrap();
            assert_eq!(prior.iter(), 4);
            let fresh = Executor::new(Bowl::new(2), prior.solver, prior.state)
                .max_iter(0)
                .run_with_solver()
                .unwrap();
            assert_eq!(fresh.iter(), 0);
            assert_eq!(fresh.counts.cost_evals, 1);
        }
    };
}
fresh!(unbounded_fresh_bookkeeping, Mads::new());
fresh!(bounded_fresh_bookkeeping, Mads::new().bounded());
fresh!(constrained_fresh_bookkeeping, Mads::new().constrained());

#[test]
fn constraints_are_counted_as_residual_calls() {
    let mut problem = Problem::new(Bowl::new(2));
    let mut solver = Mads::new().constrained();
    let _ = solver
        .init(&mut problem, basin::SelectedState::new(vec![0.0; 2]))
        .unwrap();
    assert_eq!(problem.counts().cost_evals, 1);
    assert_eq!(problem.counts().residual_evals, 1);
}

macro_rules! lifecycle {
    ($module:ident, $solver:expr, $mode:ty, $state:ident) => {
        mod $module {
            use super::*;
            #[test]
            fn exact_checkpoints_keep_mesh_and_halton_history() {
                let make = || {
                    Executor::from_start(
                        Bowl::new(2),
                        $solver.with_minimum_poll_size(1e-12),
                        vec![2.0; 2],
                    )
                };
                let expected = make().max_iter(20).run_with_solver().unwrap();
                for split in [0, 1, 5, 13] {
                    let checkpoint = make()
                        .max_iter(split)
                        .run_with_solver()
                        .unwrap()
                        .into_checkpoint();
                    let actual = Executor::resume_from_checkpoint(
                        Bowl::new(2),
                        checkpoint,
                    )
                    .require_evaluated_state()
                    .max_iter(20)
                    .run_with_solver()
                    .unwrap();
                    assert_eq!(actual.state, expected.state);
                    assert_eq!(actual.counts, expected.counts);
                    assert_eq!(
                        actual.solver.poll_size(),
                        expected.solver.poll_size()
                    );
                    assert_eq!(
                        actual.solver.mesh_index(),
                        expected.solver.mesh_index()
                    );
                }
            }

            #[test]
            fn fresh_runs_rebuild_for_a_new_dimension() {
                let used =
                    Executor::from_start(Bowl::new(2), $solver, vec![2.0; 2])
                        .max_iter(4)
                        .run_with_solver()
                        .unwrap();
                let actual = Executor::new(
                    Bowl::new(3),
                    used.solver,
                    $state::new(vec![1.5; 3]),
                )
                .max_iter(10)
                .run_with_solver()
                .unwrap();
                let expected =
                    Executor::from_start(Bowl::new(3), $solver, vec![1.5; 3])
                        .max_iter(10)
                        .run_with_solver()
                        .unwrap();
                assert_eq!(actual.state, expected.state);
                assert_eq!(actual.counts, expected.counts);
                assert_eq!(
                    actual.solver.mesh_index(),
                    expected.solver.mesh_index()
                );
            }

            #[cfg(feature = "serde")]
            #[test]
            fn serialized_models_resume_exactly_and_reset_on_a_fresh_run() {
                type Checkpoint =
                    basin::ExactCheckpoint<Mads<$mode>, $state<Vec<f64>>>;
                let make = || {
                    Executor::from_start(
                        Bowl::new(2),
                        $solver.with_minimum_poll_size(1e-12),
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
                let actual =
                    Executor::resume_from_checkpoint(Bowl::new(2), restored)
                        .max_iter(20)
                        .run_with_solver()
                        .unwrap();
                assert_eq!(actual.state, expected.state);
                assert_eq!(actual.counts, expected.counts);
                assert_eq!(
                    postcard::to_allocvec(&actual.solver).unwrap(),
                    postcard::to_allocvec(&expected.solver).unwrap()
                );
                let fresh = Executor::new(
                    Bowl::new(2),
                    actual.solver,
                    $state::new(vec![2.0; 2]),
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
lifecycle!(
    unbounded,
    Mads::new(),
    basin::solver::nelder_mead::Unbounded,
    PointState
);
lifecycle!(
    bounded,
    Mads::new().bounded(),
    basin::solver::mads::Bounded,
    PointState
);
lifecycle!(
    constrained,
    Mads::new().constrained(),
    basin::solver::mads::Constrained,
    SelectedState
);

#[test]
fn constrained_selection_preserves_feasibility_and_publication_counts() {
    let mut stepper = Executor::from_start(
        Bowl::new(2),
        Mads::new().constrained(),
        vec![0.0; 2],
    )
    .require_evaluated_state()
    .max_iter(20)
    .into_stepper()
    .unwrap();
    let initial_cost = stepper.state().cost();
    while stepper.finished().is_none() {
        let previous = stepper.state().clone();
        stepper.step().unwrap();
        let state = stepper.state();
        let (x, cost, violation) = state.current().unwrap();
        assert_eq!(cost, x.iter().map(|x| x * x).sum::<f64>());
        assert_eq!(violation, (1.0 - x[0]).max(0.0).powi(2));
        assert_eq!(state.current(), state.best());
        assert_eq!(state.counts(), stepper.counts());
        assert_eq!(state.counts().residual_evals, state.cost_evals());
        if state.current() == previous.current() {
            assert_eq!(state.best_iter(), previous.best_iter());
            assert_eq!(state.best_counts(), previous.best_counts());
        } else {
            assert_eq!(state.best_iter(), state.iter());
            assert_eq!(state.best_counts(), Some(state.counts()));
        }
    }
    assert!(stepper.state().cost() > initial_cost);
    assert_eq!(stepper.state().current().unwrap().2, 0.0);
}
