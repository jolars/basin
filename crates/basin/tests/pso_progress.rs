//! Shared swarm progress and solver-owned particle histories.

use basin::{
    BoxConstraints, CostFunction, CountsMirror, EvalCounts, Executor,
    GlobalBestPso, PopulationProgress, State,
};
use std::convert::Infallible;

struct Sphere {
    lower: Vec<f64>,
    upper: Vec<f64>,
}
impl Sphere {
    fn new(n: usize) -> Self {
        Self {
            lower: vec![-5.0; n],
            upper: vec![5.0; n],
        }
    }
}
impl CostFunction for Sphere {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = Infallible;
    fn cost(&self, x: &Vec<f64>) -> Result<f64, Infallible> {
        assert_eq!(x.len(), self.lower.len());
        assert!(x.iter().all(|x| (-5.0..=5.0).contains(x)));
        Ok(x.iter().map(|x| x * x).sum())
    }
}
impl BoxConstraints for Sphere {
    fn lower(&self) -> &Vec<f64> {
        &self.lower
    }
    fn upper(&self) -> &Vec<f64> {
        &self.upper
    }
}

#[test]
fn fresh_swarm_run_resets_progress_and_reevaluates_every_member() {
    let prior = Executor::new(
        Sphere::new(2),
        GlobalBestPso::new(91).with_swarm_size(4),
        PopulationProgress::empty(),
    )
    .max_iter(7)
    .run_with_solver()
    .unwrap();
    let mut members = prior.state.candidates().to_vec();
    members.sort_by(|a, b| a[0].total_cmp(&b[0]));
    let fresh = Executor::new(Sphere::new(2), prior.solver, prior.state)
        .max_iter(0)
        .run_with_solver()
        .unwrap();
    assert_eq!(fresh.iter(), 0);
    assert_eq!(fresh.counts.cost_evals, 4);
    assert_eq!(fresh.state.best_iter(), 0);
    let mut actual = fresh.state.candidates().to_vec();
    actual.sort_by(|a, b| a[0].total_cmp(&b[0]));
    assert_eq!(actual, members);
    assert_eq!(
        fresh.solver.personal_best_positions(),
        fresh.state.candidates()
    );
    assert_eq!(fresh.solver.personal_best_costs(), fresh.state.costs());
}

#[test]
fn swarm_category_readers_do_not_fold_other_work_into_cost_calls() {
    let result = Executor::new(
        Sphere::new(2),
        GlobalBestPso::new(91).with_swarm_size(4),
        PopulationProgress::empty(),
    )
    .max_iter(0)
    .run_with_solver()
    .unwrap();
    let mut state = result.state;
    let counts = EvalCounts {
        cost_evals: 4,
        gradient_evals: 5,
        residual_evals: 6,
        jacobian_evals: 7,
        hessian_evals: 8,
        hessian_product_evals: 9,
    };
    state.reset_best();
    state.mirror(&counts);
    state.update_best();
    assert_eq!(state.cost_evals(), 4);
    assert_eq!(state.best_cost_evals(), 4);
    assert_eq!(state.counts(), &counts);
    assert_eq!(state.best_counts(), Some(&counts));
}

fn coherent(
    state: &PopulationProgress<Vec<f64>>,
    solver: &GlobalBestPso<Vec<f64>>,
) {
    assert_eq!(state.candidates().len(), solver.velocities().len());
    for (i, (point, &cost)) in
        state.candidates().iter().zip(state.costs()).enumerate()
    {
        assert_eq!(cost, point.iter().map(|x| x * x).sum());
        let pbest = &solver.personal_best_positions()[i];
        assert_eq!(
            solver.personal_best_costs()[i],
            pbest.iter().map(|x| x * x).sum()
        );
        assert!(solver.personal_best_costs()[i] <= cost);
    }
    for (point, cost) in [state.current(), state.best(), solver.global_best()]
        .into_iter()
        .flatten()
    {
        assert_eq!(cost, point.iter().map(|x| x * x).sum());
    }
}

#[test]
fn full_population_replacement_keeps_the_historical_swarm_representative() {
    let solver = GlobalBestPso::new(91)
        .with_inertia(1.0)
        .with_cognitive(0.0)
        .with_social(0.0)
        .with_initial_velocities(Some(vec![vec![1.0], vec![1.0]]));
    let state = PopulationProgress::from_population(vec![vec![0.0], vec![1.0]]);
    let result = Executor::new(Sphere::new(1), solver, state)
        .require_evaluated_state()
        .max_iter(2)
        .run_with_solver()
        .unwrap();
    coherent(&result.state, &result.solver);
    assert_eq!(result.state.candidates(), &[vec![2.0], vec![3.0]]);
    assert_eq!(result.state.costs(), &[4.0, 9.0]);
    assert_eq!(result.state.current(), Some((&vec![0.0], 0.0)));
    assert_eq!(result.state.best(), Some((&vec![0.0], 0.0)));
    assert_eq!(result.state.best_iter(), 0);
    assert_eq!(result.state.best_counts().unwrap().cost_evals, 2);
    assert_eq!(result.state.counts().cost_evals, 6);
}

#[test]
fn fresh_runs_reapply_configured_velocities_and_can_return_to_sampling() {
    let make = || {
        GlobalBestPso::new(91)
            .with_swarm_size(2)
            .with_initial_velocities(Some(vec![vec![0.25], vec![-0.5]]))
    };
    let seed =
        || PopulationProgress::from_population(vec![vec![1.0], vec![2.0]]);
    let used = Executor::new(Sphere::new(1), make(), seed())
        .max_iter(8)
        .run_with_solver()
        .unwrap();
    let fresh = Executor::new(Sphere::new(1), used.solver, seed())
        .require_evaluated_state()
        .max_iter(0)
        .run_with_solver()
        .unwrap();
    assert_eq!(fresh.solver.velocities(), &[vec![0.25], vec![-0.5]]);
    let sampled = Executor::new(
        Sphere::new(2),
        fresh.solver.with_initial_velocities(None),
        PopulationProgress::empty(),
    )
    .require_evaluated_state()
    .max_iter(15)
    .run_with_solver()
    .unwrap();
    let expected = Executor::new(
        Sphere::new(2),
        GlobalBestPso::new(91).with_swarm_size(2),
        PopulationProgress::empty(),
    )
    .require_evaluated_state()
    .max_iter(15)
    .run_with_solver()
    .unwrap();
    coherent(&sampled.state, &sampled.solver);
    assert_eq!(sampled.state, expected.state);
    assert_eq!(sampled.solver.velocities(), expected.solver.velocities());
    assert_eq!(
        sampled.solver.personal_best_positions(),
        expected.solver.personal_best_positions()
    );
    assert_eq!(sampled.counts, expected.counts);
}

#[test]
fn checkpoints_keep_rng_draws_and_particle_models_across_every_boundary() {
    let make = || {
        Executor::new(
            Sphere::new(3),
            GlobalBestPso::new(91).with_swarm_size(8),
            PopulationProgress::empty(),
        )
    };
    let expected = make().max_iter(40).run_with_solver().unwrap();
    for split in [0, 1, 7, 19, 39] {
        let checkpoint = make()
            .max_iter(split)
            .run_with_solver()
            .unwrap()
            .into_checkpoint();
        let resumed =
            Executor::resume_from_checkpoint(Sphere::new(3), checkpoint)
                .require_evaluated_state()
                .max_iter(40)
                .run_with_solver()
                .unwrap();
        coherent(&resumed.state, &resumed.solver);
        assert_eq!(resumed.state, expected.state);
        assert_eq!(resumed.solver.velocities(), expected.solver.velocities());
        assert_eq!(
            resumed.solver.personal_best_positions(),
            expected.solver.personal_best_positions()
        );
        assert_eq!(
            resumed.solver.personal_best_costs(),
            expected.solver.personal_best_costs()
        );
        assert_eq!(resumed.counts, expected.counts);
    }
}

#[cfg(feature = "serde")]
#[test]
fn serialized_models_continue_exactly_and_fresh_initialization_clears_scratch()
{
    type Checkpoint = basin::ExactCheckpoint<
        GlobalBestPso<Vec<f64>>,
        PopulationProgress<Vec<f64>>,
    >;
    let make = || {
        Executor::new(
            Sphere::new(2),
            GlobalBestPso::new(91).with_swarm_size(4),
            PopulationProgress::empty(),
        )
    };
    let expected = make().max_iter(25).run_with_solver().unwrap();
    let checkpoint = make()
        .max_iter(11)
        .run_with_solver()
        .unwrap()
        .into_checkpoint();
    let bytes = postcard::to_allocvec(&checkpoint).unwrap();
    let restored: Checkpoint = postcard::from_bytes(&bytes).unwrap();
    let resumed = Executor::resume_from_checkpoint(Sphere::new(2), restored)
        .require_evaluated_state()
        .max_iter(25)
        .run_with_solver()
        .unwrap();
    assert_eq!(resumed.state, expected.state);
    assert_eq!(resumed.counts, expected.counts);
    assert_eq!(
        postcard::to_allocvec(&resumed.solver).unwrap(),
        postcard::to_allocvec(&expected.solver).unwrap()
    );
    let fresh = Executor::new(
        Sphere::new(2),
        resumed.solver,
        PopulationProgress::empty(),
    )
    .require_evaluated_state()
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
#[should_panic(expected = "one velocity per position")]
fn explicit_velocity_count_must_match_the_population() {
    let solver =
        GlobalBestPso::new(91).with_initial_velocities(Some(vec![vec![0.0]]));
    let _ = Executor::new(
        Sphere::new(1),
        solver,
        PopulationProgress::from_population(vec![vec![1.0], vec![2.0]]),
    )
    .max_iter(0)
    .run();
}

#[derive(Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
struct CustomRng(u64);
impl rand::TryRng for CustomRng {
    type Error = Infallible;
    fn try_next_u32(&mut self) -> Result<u32, Infallible> {
        Ok((self.try_next_u64()? >> 32) as u32)
    }
    fn try_next_u64(&mut self) -> Result<u64, Infallible> {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        Ok(self.0)
    }
    fn try_fill_bytes(&mut self, bytes: &mut [u8]) -> Result<(), Infallible> {
        for chunk in bytes.chunks_mut(8) {
            chunk.copy_from_slice(
                &self.try_next_u64()?.to_le_bytes()[..chunk.len()],
            );
        }
        Ok(())
    }
}

#[test]
fn custom_rng_continues_exactly_and_fresh_runs_clone_its_configured_value() {
    let make = || {
        Executor::new(
            Sphere::new(2),
            GlobalBestPso::new_with_rng(CustomRng(37)).with_swarm_size(6),
            PopulationProgress::empty(),
        )
    };
    let expected = make().max_iter(30).run_with_solver().unwrap();
    let checkpoint = make()
        .max_iter(9)
        .run_with_solver()
        .unwrap()
        .into_checkpoint();
    #[cfg(feature = "serde")]
    let checkpoint = {
        let bytes = postcard::to_allocvec(&checkpoint).unwrap();
        postcard::from_bytes::<
            basin::ExactCheckpoint<
                GlobalBestPso<Vec<f64>, f64, CustomRng>,
                PopulationProgress<Vec<f64>>,
            >,
        >(&bytes)
        .unwrap()
    };
    let resumed = Executor::resume_from_checkpoint(Sphere::new(2), checkpoint)
        .require_evaluated_state()
        .max_iter(30)
        .run_with_solver()
        .unwrap();
    assert_eq!(resumed.state, expected.state);
    assert_eq!(resumed.solver.velocities(), expected.solver.velocities());
    assert_eq!(resumed.counts, expected.counts);
    let fresh = Executor::new(
        Sphere::new(2),
        resumed.solver,
        PopulationProgress::empty(),
    )
    .require_evaluated_state()
    .max_iter(30)
    .run_with_solver()
    .unwrap();
    assert_eq!(fresh.state, expected.state);
    assert_eq!(fresh.solver.velocities(), expected.solver.velocities());
    assert_eq!(fresh.counts, expected.counts);
}
