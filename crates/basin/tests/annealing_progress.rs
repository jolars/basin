//! Shared proposal progress preserves accepted points and complete bookkeeping.

use basin::core::rng::{ChaCha8Rng, RngExt};
use basin::{
    CostFunction, Executor, Neighbor, SimulatedAnnealing, State,
    TemperatureSchedule, TerminationCode,
};
use std::convert::Infallible;

struct Sphere;
impl CostFunction for Sphere {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = Infallible;
    fn cost(&self, x: &Vec<f64>) -> Result<f64, Infallible> {
        Ok(x.iter().map(|x| x * x).sum())
    }
}

#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
struct StatefulNeighbor {
    calls: u64,
}
impl Neighbor<Vec<f64>> for StatefulNeighbor {
    type Error = Infallible;
    fn propose(
        &mut self,
        x: &Vec<f64>,
        _: f64,
        rng: &mut ChaCha8Rng,
    ) -> Result<Vec<f64>, Infallible> {
        self.calls += 1;
        let scale = 1.0 / self.calls as f64;
        Ok(x.iter()
            .map(|x| x + scale * (rng.random::<f64>() - 0.5))
            .collect())
    }
}
fn solver() -> SimulatedAnnealing<StatefulNeighbor> {
    SimulatedAnnealing::new(
        StatefulNeighbor { calls: 0 },
        1.0,
        TemperatureSchedule::geometric(0.9),
        73,
    )
    .with_reannealing_fixed(7)
    .with_reannealing_accepted(5)
    .with_reannealing_best(9)
}

#[test]
fn fresh_annealing_reevaluates_and_resets_progress() {
    let prior = Executor::from_start(Sphere, solver(), vec![2.0, -1.0])
        .max_iter(12)
        .run_with_solver()
        .unwrap();
    let fresh = Executor::new(Sphere, prior.solver, prior.state)
        .max_iter(0)
        .run_with_solver()
        .unwrap();
    assert_eq!(fresh.state.iter(), 0);
    assert_eq!(fresh.counts.cost_evals, 1);
    assert_eq!(fresh.state.best_iter(), 0);
    assert_eq!(fresh.state.accepted_moves(), 0);
    assert_eq!(fresh.state.rejected_moves(), 0);
}

#[test]
fn rejected_proposals_do_not_trigger_iterate_change_stopping() {
    let solver = SimulatedAnnealing::new(
        |x: &Vec<f64>, _: f64, _: &mut ChaCha8Rng| {
            x.iter().map(|x| x + 1000.0).collect()
        },
        1e-9,
        TemperatureSchedule::reciprocal(),
        73,
    )
    .with_absolute_cost_change_tolerance(0.0)
    .with_absolute_step_tolerance(0.0);
    let result = Executor::from_start(Sphere, solver, vec![0.0])
        .max_iter(6)
        .run()
        .unwrap();
    assert_eq!(result.report.code(), TerminationCode::MaxIter);
    assert_eq!(result.iter(), 6);
    assert_eq!(result.state.rejected_moves(), 6);
}

fn coherent(state: &basin::ProposalState<Vec<f64>>) {
    let (x, cost) = state.current().unwrap();
    assert_eq!(cost, Sphere.cost(x).unwrap());
    let (x, cost) = state.best().unwrap();
    assert_eq!(cost, Sphere.cost(x).unwrap());
    assert_eq!(
        state.accepted_moves() + state.rejected_moves(),
        state.iter()
    );
}

#[test]
fn reused_solver_resets_neighbor_rng_schedule_and_dimension() {
    let prior = Executor::from_start(Sphere, solver(), vec![2.0, -1.0])
        .max_iter(19)
        .run_with_solver()
        .unwrap();
    assert_eq!(prior.solver.neighbor().unwrap().calls, 19);
    let mut state = prior.state;
    state.replace(vec![3.0, -2.0, 1.0], -1000.0);
    let initialized = Executor::new(Sphere, prior.solver, state)
        .require_evaluated_state()
        .max_iter(0)
        .run_with_solver()
        .unwrap();
    assert_eq!(initialized.solver.neighbor().unwrap().calls, 0);
    assert_eq!(initialized.solver.temperature(), 1.0);
    assert_eq!(initialized.solver.reannealings(), 0);
    assert_eq!(
        initialized.state.current(),
        Some((&vec![3.0, -2.0, 1.0], 14.0))
    );
    let restarted =
        Executor::resume_from_checkpoint(Sphere, initialized.into_checkpoint())
            .max_iter(30)
            .run_with_solver()
            .unwrap();
    let rebuilt = Executor::from_start(Sphere, solver(), vec![3.0, -2.0, 1.0])
        .max_iter(30)
        .run_with_solver()
        .unwrap();
    coherent(&restarted.state);
    assert_eq!(restarted.state, rebuilt.state);
    assert_eq!(restarted.counts, rebuilt.counts);
    assert_eq!(restarted.solver.neighbor().unwrap().calls, 30);
    assert_eq!(restarted.solver.temperature(), rebuilt.solver.temperature());
    assert_eq!(
        restarted.solver.reannealings(),
        rebuilt.solver.reannealings()
    );
}

#[test]
fn exact_continuation_preserves_all_proposal_and_cooling_history() {
    let make = || Executor::from_start(Sphere, solver(), vec![2.0, -1.0]);
    let expected = make().max_iter(30).run_with_solver().unwrap();
    for split in [0, 1, 6, 7, 13, 29] {
        let checkpoint = make()
            .max_iter(split)
            .run_with_solver()
            .unwrap()
            .into_checkpoint();
        let resumed = Executor::resume_from_checkpoint(Sphere, checkpoint)
            .require_evaluated_state()
            .max_iter(30)
            .run_with_solver()
            .unwrap();
        coherent(&resumed.state);
        assert_eq!(resumed.state, expected.state);
        assert_eq!(resumed.counts, expected.counts);
        assert_eq!(resumed.solver.neighbor().unwrap().calls, 30);
        assert_eq!(resumed.solver.temperature(), expected.solver.temperature());
        assert_eq!(
            resumed.solver.reannealings(),
            expected.solver.reannealings()
        );
    }
}

#[test]
fn independently_seeded_chains_ignore_live_rng_and_neighbor_history() {
    let result = Executor::from_start(Sphere, solver(), vec![2.0, -1.0])
        .max_iter(19)
        .run_with_solver()
        .unwrap();
    let seeded = result.solver.seed_chain(91);
    let seeded_again = result.solver.seed_chain(91);
    assert_eq!(result.solver.neighbor().unwrap().calls, 19);
    let rebuilt = solver().seed_chain(91);
    let run = |solver| {
        Executor::from_start(Sphere, solver, vec![4.0])
            .max_iter(20)
            .run_with_solver()
            .unwrap()
    };
    let first = run(seeded);
    assert_eq!(first.state, run(seeded_again).state);
    assert_eq!(first.state, run(rebuilt).state);
}

#[test]
fn proposal_metadata_uses_publication_counts_and_iteration() {
    use basin::{CountsMirror, EvalCounts, IncumbentState, ProposalState};
    let mut state = ProposalState::new(vec![1.0]);
    assert!(state.current().is_none());
    assert!(state.best().is_none());
    state.replace(vec![1.0], 1.0);
    state.mirror(&EvalCounts {
        cost_evals: 1,
        ..EvalCounts::default()
    });
    state.update_best();
    state.accept_proposal(vec![0.0], 0.0);
    let counts = EvalCounts {
        cost_evals: 2,
        gradient_evals: 3,
        residual_evals: 4,
        jacobian_evals: 5,
        hessian_evals: 6,
        hessian_product_evals: 7,
    };
    state.increment_iter();
    state.mirror(&counts);
    state.update_best();
    assert_eq!(state.last_accepted_iter(), 1);
    assert_eq!(state.best_counts(), Some(&counts));
    assert_eq!(state.incumbent_record().unwrap().iter, 1);
    state.reject_proposal();
    state.increment_iter();
    state.mirror(&EvalCounts {
        cost_evals: 3,
        ..counts
    });
    state.update_best();
    state.update_best();
    assert_eq!(state.last_accepted_iter(), 1);
    assert_eq!(state.best_counts(), Some(&counts));
    // Mid-step publication stamps the completed boundary, without adding an iteration.
    state.accept_proposal(vec![-1.0], -1.0);
    let counts = EvalCounts {
        cost_evals: 4,
        ..counts
    };
    state.mirror(&counts);
    state.update_best();
    assert_eq!(state.last_accepted_iter(), 2);
    assert_eq!(state.best_counts(), Some(&counts));
    assert_eq!(state.best_iter(), 2);
    assert_eq!(state.accepted_moves(), 2);
    assert_eq!(state.rejected_moves(), 1);
    state.reset();
    assert!(state.current().is_none());
    assert!(state.best().is_none());
    assert_eq!(state.counts(), &EvalCounts::default());
    assert_eq!(state.last_accepted_iter(), 0);
    assert_eq!(state.accepted_moves(), 0);
    assert_eq!(state.rejected_moves(), 0);
}

#[cfg(feature = "serde")]
#[test]
fn serialized_checkpoint_preserves_templates_and_live_components() {
    type Checkpoint = basin::ExactCheckpoint<
        SimulatedAnnealing<StatefulNeighbor>,
        basin::ProposalState<Vec<f64>>,
    >;
    let make = || Executor::from_start(Sphere, solver(), vec![2.0, -1.0]);
    let expected = make().max_iter(30).run_with_solver().unwrap();
    let checkpoint = make()
        .max_iter(13)
        .run_with_solver()
        .unwrap()
        .into_checkpoint();
    let bytes = postcard::to_allocvec(&checkpoint).unwrap();
    let restored: Checkpoint = postcard::from_bytes(&bytes).unwrap();
    let resumed = Executor::resume_from_checkpoint(Sphere, restored)
        .max_iter(30)
        .run_with_solver()
        .unwrap();
    assert_eq!(resumed.state, expected.state);
    assert_eq!(resumed.counts, expected.counts);
    assert_eq!(
        postcard::to_allocvec(&resumed.solver).unwrap(),
        postcard::to_allocvec(&expected.solver).unwrap()
    );
    let restarted = Executor::from_start(Sphere, resumed.solver, vec![1.0])
        .max_iter(20)
        .run_with_solver()
        .unwrap();
    let fresh = Executor::from_start(Sphere, solver(), vec![1.0])
        .max_iter(20)
        .run_with_solver()
        .unwrap();
    assert_eq!(restarted.state, fresh.state);
    assert_eq!(
        postcard::to_allocvec(&restarted.solver).unwrap(),
        postcard::to_allocvec(&fresh.solver).unwrap()
    );
}

#[test]
fn raw_cost_budget_and_checked_records_work_without_a_gradient_capability() {
    let result = Executor::from_start(Sphere, solver(), vec![2.0])
        .require_evaluated_state()
        .max_evaluations(basin::EvaluationKind::Cost, 8)
        .max_iter(100)
        .run()
        .unwrap();
    assert_eq!(result.report.code(), TerminationCode::MaxEvaluations);
    assert_eq!(result.iter(), 7);
    assert_eq!(result.state.counts().cost_evals, 8);
    assert_eq!(result.state.counts().total_work(), 8);
}

#[derive(Clone)]
struct CountingRng(u64);
impl rand::TryRng for CountingRng {
    type Error = Infallible;
    fn try_next_u32(&mut self) -> Result<u32, Infallible> {
        Ok(self.try_next_u64()? as u32)
    }
    fn try_next_u64(&mut self) -> Result<u64, Infallible> {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
        Ok(self.0)
    }
    fn try_fill_bytes(&mut self, dst: &mut [u8]) -> Result<(), Infallible> {
        for chunk in dst.chunks_mut(8) {
            let bytes = self.try_next_u64()?.to_le_bytes();
            chunk.copy_from_slice(&bytes[..chunk.len()]);
        }
        Ok(())
    }
}

#[test]
fn fresh_run_resets_captured_closure_state_and_a_custom_rng() {
    let make = || {
        let mut calls = 0;
        SimulatedAnnealing::new_with_rng(
            move |x: &Vec<f64>, _: f64, rng: &mut CountingRng| {
                calls += 1;
                vec![x[0] - f64::from(calls) * rng.random::<f64>()]
            },
            1.0,
            TemperatureSchedule::reciprocal(),
            CountingRng(17),
        )
    };
    let prior = Executor::from_start(Sphere, make(), vec![100.0])
        .max_iter(8)
        .run_with_solver()
        .unwrap();
    let mut state = prior.state;
    state.replace(vec![50.0], -1000.0);
    let restarted = Executor::new(Sphere, prior.solver, state)
        .max_iter(10)
        .run_with_solver()
        .unwrap();
    let expected = Executor::from_start(Sphere, make(), vec![50.0])
        .max_iter(10)
        .run_with_solver()
        .unwrap();
    assert_eq!(restarted.state, expected.state);
    assert_eq!(restarted.counts, expected.counts);
}

#[test]
fn exact_resume_preserves_cost_change_history() {
    let make = || {
        Executor::from_start(
            Sphere,
            SimulatedAnnealing::new(
                |x: &Vec<f64>, _: f64, _: &mut ChaCha8Rng| {
                    x.iter().map(|x| x * 0.5).collect()
                },
                1.0,
                TemperatureSchedule::reciprocal(),
                73,
            )
            .with_absolute_cost_change_tolerance(0.05),
            vec![2.0],
        )
    };
    let expected = make().max_iter(20).run_with_solver().unwrap();
    assert_eq!(expected.report.code(), TerminationCode::CostTolerance);
    assert_eq!(expected.iter(), 4);
    for split in [0, 1, 3] {
        let checkpoint = make()
            .max_iter(split)
            .run_with_solver()
            .unwrap()
            .into_checkpoint();
        let resumed = Executor::resume_from_checkpoint(Sphere, checkpoint)
            .max_iter(20)
            .run_with_solver()
            .unwrap();
        assert_eq!(resumed.report.code(), expected.report.code());
        assert_eq!(resumed.state, expected.state);
        assert_eq!(resumed.counts, expected.counts);
    }
}
