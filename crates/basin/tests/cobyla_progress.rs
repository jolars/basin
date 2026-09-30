//! COBYLA's selected progress, callback accounting, and model lifetime.

use basin::{
    Cobyla, CostFunction, Executor, NonlinearInequalityConstraints, Problem,
    SelectedState, Solver, State,
};
use std::convert::Infallible;

struct LowerBound;
impl CostFunction for LowerBound {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = Infallible;
    fn cost(&self, x: &Vec<f64>) -> Result<f64, Infallible> {
        Ok(x.iter().map(|x| x * x).sum())
    }
}
impl NonlinearInequalityConstraints for LowerBound {
    fn num_constraints(&self) -> usize {
        1
    }
    fn constraints(&self, x: &Vec<f64>) -> Result<Vec<f64>, Infallible> {
        Ok(vec![1.0 - x[0]])
    }
}

#[test]
fn fresh_run_resets_iteration_history() {
    let prior = Executor::from_start(LowerBound, Cobyla::new(), vec![0.0, 0.0])
        .max_iter(4)
        .run_with_solver()
        .unwrap();
    assert_eq!(prior.iter(), 4);
    let fresh = Executor::new(LowerBound, prior.solver, prior.state)
        .max_iter(0)
        .run_with_solver()
        .unwrap();
    assert_eq!(fresh.iter(), 0);
    assert_eq!(fresh.counts.cost_evals, 3);
}

#[test]
fn inequality_callbacks_are_residual_evaluations() {
    let mut problem = Problem::new(LowerBound);
    let mut solver = Cobyla::new();
    let _ = solver
        .init(&mut problem, SelectedState::new(vec![0.0, 0.0]))
        .unwrap();
    assert_eq!(problem.counts().cost_evals, 3);
    assert_eq!(problem.counts().residual_evals, 3);
}

#[test]
fn selection_preserves_feasibility_and_stamps_only_changes() {
    let mut stepper = Executor::from_start(
        LowerBound,
        Cobyla::new().with_initial_radius(0.1),
        vec![0.0, 0.0],
    )
    .require_evaluated_state()
    .max_iter(100)
    .into_stepper()
    .unwrap();
    let initial_cost = stepper.state().cost();
    let mut unchanged = 0;
    while stepper.finished().is_none() {
        let previous = stepper.state().clone();
        stepper.step().unwrap();
        let state = stepper.state();
        let (x, cost, violation) = state.current().unwrap();
        assert_eq!(cost, x.iter().map(|x| x * x).sum::<f64>());
        assert_eq!(violation, (1.0 - x[0]).max(0.0));
        assert_eq!(state.current(), state.best());
        assert_eq!(state.counts(), stepper.counts());
        assert_eq!(state.cost_evals(), state.counts().cost_evals);
        assert_eq!(state.counts().residual_evals, state.counts().cost_evals);
        if state.current() == previous.current() {
            unchanged += 1;
            assert_eq!(state.best_iter(), previous.best_iter());
            assert_eq!(state.best_counts(), previous.best_counts());
        } else {
            assert_eq!(state.best_iter(), state.iter());
            assert_eq!(state.best_counts(), Some(state.counts()));
        }
    }
    assert!(unchanged > 0);
    assert!(stepper.state().cost() > initial_cost);
    assert!(stepper.state().current().unwrap().2 < 1e-8);
}

#[test]
fn fresh_runs_rebuild_the_model_for_new_dimensions() {
    let prior = Executor::from_start(LowerBound, Cobyla::new(), vec![0.0; 2])
        .max_iter(6)
        .run_with_solver()
        .unwrap();
    let restarted = Executor::new(
        LowerBound,
        prior.solver,
        SelectedState::new(vec![0.0; 3]),
    )
    .max_iter(15)
    .run_with_solver()
    .unwrap();
    let expected =
        Executor::from_start(LowerBound, Cobyla::new(), vec![0.0; 3])
            .max_iter(15)
            .run_with_solver()
            .unwrap();
    assert_eq!(restarted.state, expected.state);
    assert_eq!(restarted.counts, expected.counts);
    assert_eq!(restarted.solver.rho(), expected.solver.rho());
}

#[test]
fn owned_checkpoints_preserve_filter_and_radius_history() {
    let make = || {
        Executor::from_start(
            LowerBound,
            Cobyla::new()
                .with_initial_radius(0.1)
                .with_final_radius(1e-12),
            vec![0.0; 2],
        )
    };
    let expected = make().max_iter(30).run_with_solver().unwrap();
    for split in [0, 1, 7, 19, 29] {
        let checkpoint = make()
            .max_iter(split)
            .run_with_solver()
            .unwrap()
            .into_checkpoint();
        let resumed = Executor::resume_from_checkpoint(LowerBound, checkpoint)
            .require_evaluated_state()
            .max_iter(30)
            .run_with_solver()
            .unwrap();
        assert_eq!(resumed.state, expected.state, "split at {split}");
        assert_eq!(resumed.counts, expected.counts);
        assert_eq!(resumed.solver.rho(), expected.solver.rho());
    }
}

#[cfg(feature = "serde")]
#[test]
fn serialized_checkpoints_preserve_every_model_buffer() {
    type Checkpoint =
        basin::ExactCheckpoint<Cobyla<Vec<f64>>, SelectedState<Vec<f64>>>;
    let make = || {
        Executor::from_start(
            LowerBound,
            Cobyla::new()
                .with_initial_radius(0.1)
                .with_final_radius(1e-12),
            vec![0.0; 2],
        )
    };
    let expected = make().max_iter(30).run_with_solver().unwrap();
    let checkpoint = make()
        .max_iter(9)
        .run_with_solver()
        .unwrap()
        .into_checkpoint();
    let bytes = postcard::to_allocvec(&checkpoint).unwrap();
    let restored: Checkpoint = postcard::from_bytes(&bytes).unwrap();
    let resumed = Executor::resume_from_checkpoint(LowerBound, restored)
        .max_iter(30)
        .run_with_solver()
        .unwrap();
    assert_eq!(resumed.state, expected.state);
    assert_eq!(resumed.counts, expected.counts);
    assert_eq!(
        postcard::to_allocvec(&resumed.solver).unwrap(),
        postcard::to_allocvec(&expected.solver).unwrap()
    );
    let fresh = Executor::new(
        LowerBound,
        resumed.solver,
        SelectedState::new(vec![0.0; 2]),
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

#[test]
fn residual_budget_counts_constraint_blocks_separately_from_objectives() {
    let result = Executor::from_start(LowerBound, Cobyla::new(), vec![0.0; 2])
        .max_evaluations(basin::EvaluationKind::Residual, 4)
        .run_with_solver()
        .unwrap();
    assert_eq!(result.counts.residual_evals, 4);
    assert_eq!(result.counts.cost_evals, 4);
    assert_eq!(result.state.counts(), &result.counts);
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum CallbackError {
    Inequality,
    Equality,
}

struct FullForm {
    failure: Option<CallbackError>,
}
impl CostFunction for FullForm {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = CallbackError;
    fn cost(&self, x: &Vec<f64>) -> Result<f64, CallbackError> {
        Ok(x.iter().map(|x| x * x).sum())
    }
}
impl basin::NonlinearConstraints for FullForm {
    type Matrix = basin::DenseMatrix;
    fn num_nonlinear_constraints(&self) -> usize {
        1
    }
    fn num_nonlinear_equalities(&self) -> usize {
        1
    }
    fn nonlinear_constraints(
        &self,
        x: &Vec<f64>,
    ) -> Result<Vec<f64>, CallbackError> {
        if self.failure == Some(CallbackError::Inequality) {
            Err(CallbackError::Inequality)
        } else {
            Ok(vec![1.0 - x[0]])
        }
    }
    fn nonlinear_equalities(
        &self,
        x: &Vec<f64>,
    ) -> Result<Option<Vec<f64>>, CallbackError> {
        if self.failure == Some(CallbackError::Equality) {
            Err(CallbackError::Equality)
        } else {
            Ok(Some(vec![x[1]]))
        }
    }
}

#[test]
fn folded_callbacks_count_each_block_including_failure_attempts() {
    for failure in [
        None,
        Some(CallbackError::Inequality),
        Some(CallbackError::Equality),
    ] {
        let mut problem =
            Problem::new(basin::FoldedConstraints::new(FullForm { failure }));
        let mut solver = Cobyla::new();
        let output =
            solver.init(&mut problem, SelectedState::new(vec![0.0; 2]));
        match failure {
            None => {
                assert!(output.is_ok());
                assert_eq!(problem.counts().cost_evals, 3);
                assert_eq!(problem.counts().residual_evals, 6);
            }
            Some(error) => {
                assert_eq!(output.unwrap_err(), error);
                assert_eq!(problem.counts().cost_evals, 1);
                assert_eq!(
                    problem.counts().residual_evals,
                    if error == CallbackError::Inequality {
                        1
                    } else {
                        2
                    }
                );
            }
        }
    }
}

#[test]
fn published_records_keep_the_drivers_nonfinite_moderation() {
    struct Nonfinite(f64);
    impl CostFunction for Nonfinite {
        type Param = Vec<f64>;
        type Output = f64;
        type Error = Infallible;
        fn cost(&self, _: &Vec<f64>) -> Result<f64, Infallible> {
            Ok(self.0)
        }
    }
    impl NonlinearInequalityConstraints for Nonfinite {
        fn num_constraints(&self) -> usize {
            1
        }
        fn constraints(&self, _: &Vec<f64>) -> Result<Vec<f64>, Infallible> {
            Ok(vec![self.0])
        }
    }
    let limit = 2_f64.powi(100);
    for (input, cost, violation) in [
        (f64::NAN, limit, limit),
        (f64::INFINITY, limit, limit),
        (f64::NEG_INFINITY, -limit, 0.0),
    ] {
        let result =
            Executor::from_start(Nonfinite(input), Cobyla::new(), vec![0.0; 2])
                .require_evaluated_state()
                .max_iter(0)
                .run_with_solver()
                .unwrap();
        let (_, actual_cost, actual_violation) =
            result.state.current().unwrap();
        assert_eq!(actual_cost, cost);
        assert_eq!(actual_violation, violation);
        assert_eq!(result.state.current(), result.state.best());
        assert_eq!(result.state.best_counts(), Some(&result.counts));
    }
}

#[test]
fn folded_checkpoints_preserve_constraint_counts_and_selected_records() {
    let problem = || basin::FoldedConstraints::new(FullForm { failure: None });
    let make = || {
        Executor::from_start(
            problem(),
            Cobyla::new().with_initial_radius(0.1),
            vec![0.0; 2],
        )
    };
    let expected = make().max_iter(25).run_with_solver().unwrap();
    let checkpoint = make()
        .max_iter(8)
        .run_with_solver()
        .unwrap()
        .into_checkpoint();
    let resumed = Executor::resume_from_checkpoint(problem(), checkpoint)
        .max_iter(25)
        .run_with_solver()
        .unwrap();
    assert_eq!(resumed.state, expected.state);
    assert_eq!(resumed.counts, expected.counts);
    assert_eq!(resumed.counts.residual_evals, 2 * resumed.counts.cost_evals);
}
