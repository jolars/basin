//! SGD publishes complete objective records without changing its batch cadence.

use basin::{
    CostFunction, EvaluationKind, Executor, MiniBatchGradient, PointState, Sgd,
    State, TerminationReason,
};
use std::convert::Infallible;

struct Quadratic {
    samples: usize,
}
impl CostFunction for Quadratic {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = Infallible;
    fn cost(&self, x: &Vec<f64>) -> Result<f64, Infallible> {
        Ok((0..self.samples)
            .map(|i| x.iter().map(|x| (x - i as f64).powi(2)).sum::<f64>())
            .sum::<f64>()
            / self.samples as f64)
    }
}
impl MiniBatchGradient for Quadratic {
    type Gradient = Vec<f64>;
    fn n_samples(&self) -> usize {
        self.samples
    }
    fn batch_gradient(
        &self,
        x: &Vec<f64>,
        batch: &[usize],
    ) -> Result<Vec<f64>, Infallible> {
        Ok(x.iter()
            .map(|x| {
                batch.iter().map(|i| 2.0 * (x - *i as f64)).sum::<f64>()
                    / batch.len() as f64
            })
            .collect())
    }
}

fn solver() -> Sgd<Vec<f64>> {
    Sgd::new(0.02, 2, 73).with_momentum(0.6)
}
fn problem() -> Quadratic {
    Quadratic { samples: 7 }
}
fn coherent(state: &PointState<Vec<f64>>, problem: &Quadratic) {
    let (x, cost) = state.current().unwrap();
    assert_eq!(cost, problem.cost(x).unwrap());
    if let Some((x, cost)) = state.best() {
        assert_eq!(cost, problem.cost(x).unwrap());
    }
    assert_eq!(state.cost_evals(), state.counts().cost_evals);
}

#[test]
fn refreshes_preserve_batch_trajectory_and_evaluation_cadence() {
    let start = vec![10.0, -5.0];
    for steps in 0..=10 {
        let periodic = Executor::from_start(problem(), solver(), start.clone())
            .require_evaluated_state()
            .max_iter(steps)
            .run_with_solver()
            .unwrap();
        let every = Executor::from_start(
            problem(),
            solver().with_cost_eval_every(1),
            start.clone(),
        )
        .max_iter(steps)
        .run_with_solver()
        .unwrap();
        assert_eq!(periodic.solver.working_param(), Some(every.state.param()));
        coherent(&periodic.state, &problem());
        assert_eq!(periodic.counts, *periodic.state.counts());
        assert_eq!(periodic.counts.gradient_evals, steps);
        assert_eq!(periodic.counts.cost_evals, 1 + steps / 3);
        assert_eq!(periodic.counts.total_work(), 1 + steps / 3 + steps);
        assert_eq!(periodic.state.iter(), steps);
        let last_refresh = Executor::from_start(
            problem(),
            solver().with_cost_eval_every(1),
            start.clone(),
        )
        .max_iter(steps - steps % 3)
        .run()
        .unwrap();
        assert_eq!(periodic.state.param(), last_refresh.param());
        assert_eq!(periodic.state.cost(), last_refresh.cost());
    }
}

#[test]
fn exact_resume_preserves_unpublished_work_and_refresh_phase() {
    let make = || Executor::from_start(problem(), solver(), vec![10.0, -5.0]);
    let expected = make().max_iter(11).run_with_solver().unwrap();
    for split in [0, 1, 2, 3, 4, 8] {
        let checkpoint = make()
            .max_iter(split)
            .run_with_solver()
            .unwrap()
            .into_checkpoint();
        let resumed = Executor::resume_from_checkpoint(problem(), checkpoint)
            .require_evaluated_state()
            .max_iter(11)
            .run_with_solver()
            .unwrap();
        assert_eq!(resumed.state, expected.state);
        assert_eq!(resumed.counts, expected.counts);
        assert_eq!(
            resumed.solver.working_param(),
            expected.solver.working_param()
        );
    }
}

#[test]
fn fresh_run_resets_momentum_batches_refresh_phase_and_progress() {
    let old = Executor::from_start(problem(), solver(), vec![10.0, -5.0])
        .max_iter(5)
        .run_with_solver()
        .unwrap();
    let mut seed = old.state;
    seed.replace(vec![4.0, 5.0, 6.0], -100.0);
    let initialized = Executor::new(Quadratic { samples: 5 }, old.solver, seed)
        .require_evaluated_state()
        .max_iter(0)
        .run_with_solver()
        .unwrap();
    coherent(&initialized.state, &Quadratic { samples: 5 });
    assert_eq!(initialized.state.iter(), 0);
    assert_eq!(initialized.state.best_iter(), 0);
    assert_eq!(initialized.counts.cost_evals, 1);
    assert_eq!(initialized.counts.total_work(), 1);
    assert_eq!(
        initialized.solver.working_param(),
        Some(initialized.state.param())
    );
    let resumed = Executor::resume_from_checkpoint(
        Quadratic { samples: 5 },
        initialized.into_checkpoint(),
    )
    .max_iter(7)
    .run_with_solver()
    .unwrap();
    let rebuilt = Executor::from_start(
        Quadratic { samples: 5 },
        solver(),
        vec![4.0, 5.0, 6.0],
    )
    .max_iter(7)
    .run_with_solver()
    .unwrap();
    assert_eq!(resumed.state, rebuilt.state);
    assert_eq!(resumed.counts, rebuilt.counts);
    assert_eq!(
        resumed.solver.working_param(),
        rebuilt.solver.working_param()
    );
}

#[test]
fn convergence_observes_refreshes_and_budgets_count_every_batch() {
    let result = Executor::from_start(
        problem(),
        Sgd::new(0.0, 2, 73).with_absolute_cost_change_tolerance(0.0),
        vec![10.0, -5.0],
    )
    .max_iter(10)
    .run()
    .unwrap();
    assert_eq!(result.state.iter(), 3);
    assert_eq!(result.reason, TerminationReason::CostTolerance);
    let result = Executor::from_start(problem(), solver(), vec![10.0, -5.0])
        .require_evaluated_state()
        .max_evaluations(EvaluationKind::Gradient, 4)
        .max_iter(10)
        .run()
        .unwrap();
    assert_eq!(result.state.iter(), 4);
    assert_eq!(result.state.counts().gradient_evals, 4);
    coherent(&result.state, &problem());
}

#[cfg(feature = "serde")]
#[test]
fn serialized_checkpoint_preserves_rng_momentum_and_working_iterate() {
    let make = || Executor::from_start(problem(), solver(), vec![10.0, -5.0]);
    let expected = make().max_iter(11).run_with_solver().unwrap();
    let checkpoint = make()
        .max_iter(4)
        .run_with_solver()
        .unwrap()
        .into_checkpoint();
    let bytes = postcard::to_allocvec(&checkpoint).unwrap();
    let restored: basin::ExactCheckpoint<Sgd<Vec<f64>>, PointState<Vec<f64>>> =
        postcard::from_bytes(&bytes).unwrap();
    let resumed = Executor::resume_from_checkpoint(problem(), restored)
        .require_evaluated_state()
        .max_iter(11)
        .run_with_solver()
        .unwrap();
    assert_eq!(resumed.state, expected.state);
    assert_eq!(resumed.counts, expected.counts);
    assert_eq!(
        postcard::to_allocvec(&resumed.solver).unwrap(),
        postcard::to_allocvec(&expected.solver).unwrap()
    );
}
