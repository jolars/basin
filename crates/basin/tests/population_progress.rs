//! Shared population records and fresh versus exact stochastic runs.

use basin::{
    BoxConstraints, CostFunction, De, Executor, PopulationProgress,
    RandomSearch, Solver, Ssga, State,
};
use std::convert::Infallible;

struct Sphere {
    lower: Vec<f64>,
    upper: Vec<f64>,
}
impl Sphere {
    fn new() -> Self {
        Self {
            lower: vec![-5.0],
            upper: vec![5.0],
        }
    }
}
impl CostFunction for Sphere {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = Infallible;
    fn cost(&self, x: &Vec<f64>) -> Result<f64, Infallible> {
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

fn fresh_population<So>(solver: So)
where
    So: Solver<Sphere, PopulationProgress<Vec<f64>>, Error = Infallible>,
{
    let prior =
        Executor::new(Sphere::new(), solver, PopulationProgress::empty())
            .max_iter(3)
            .run_with_solver()
            .unwrap();
    let members = prior.state.candidates().to_vec();
    let fresh = Executor::new(Sphere::new(), prior.solver, prior.state)
        .max_iter(0)
        .run_with_solver()
        .unwrap();
    assert_eq!(fresh.iter(), 0);
    assert_eq!(fresh.state.best_iter(), 0);
    assert_eq!(fresh.counts.cost_evals, 4);
    assert_eq!(fresh.state.candidates(), members);
}

#[test]
fn random_search_fresh_run_reevaluates_members_and_resets_bookkeeping() {
    fresh_population(RandomSearch::new(4, 13));
}
#[test]
fn de_fresh_run_reevaluates_members_and_resets_bookkeeping() {
    fresh_population(De::new(13).with_pop_size(4));
}
#[test]
fn ssga_fresh_run_reevaluates_members_and_resets_bookkeeping() {
    fresh_population(Ssga::new(13).with_pop_size(4));
}

fn explicit_population<So>(solver: So)
where
    So: Solver<Sphere, PopulationProgress<Vec<f64>>, Error = Infallible>,
{
    let result = Executor::new(
        Sphere::new(),
        solver,
        PopulationProgress::from_population(vec![
            vec![8.0],
            vec![2.0],
            vec![0.0],
            vec![1.0],
        ]),
    )
    .max_iter(0)
    .run_with_solver()
    .unwrap();
    assert_eq!(
        result.state.candidates(),
        &[vec![0.0], vec![1.0], vec![2.0], vec![5.0]]
    );
    assert_eq!(result.state.costs(), &[0.0, 1.0, 4.0, 25.0]);
    assert_eq!(result.counts.cost_evals, 4);
}

#[test]
fn explicit_initial_populations_are_projected_and_evaluated() {
    explicit_population(RandomSearch::new(4, 13));
    explicit_population(De::new(13).with_pop_size(4));
    explicit_population(Ssga::new(13).with_pop_size(4));
}

#[test]
fn random_search_restarts_its_configured_rng_for_an_empty_population() {
    let make = || {
        Executor::new(
            Sphere::new(),
            RandomSearch::new(4, 13),
            PopulationProgress::empty(),
        )
    };
    let expected = make().max_iter(12).run_with_solver().unwrap();
    let used = make().max_iter(3).run_with_solver().unwrap();
    let restarted =
        Executor::new(Sphere::new(), used.solver, PopulationProgress::empty())
            .max_iter(12)
            .run_with_solver()
            .unwrap();
    assert_eq!(restarted.state.candidates(), expected.state.candidates());
    assert_eq!(restarted.state.costs(), expected.state.costs());
    assert_eq!(restarted.counts, expected.counts);
}

fn coherent(state: &PopulationProgress<Vec<f64>>) {
    for (point, &cost) in state.candidates().iter().zip(state.costs()) {
        assert_eq!(point.iter().map(|x| x * x).sum::<f64>(), cost);
    }
    for (point, cost) in [state.current(), state.best()].into_iter().flatten() {
        assert_eq!(point.iter().map(|x| x * x).sum::<f64>(), cost);
    }
}

#[test]
fn member_order_storage_and_historical_pairs_survive_complete_replacement() {
    use basin::{CountsMirror, EvalCounts, IncumbentState};
    let mut state =
        PopulationProgress::from_population(vec![vec![2.0], vec![1.0]]);
    assert!(state.current().is_none());
    assert!(state.best().is_none());
    assert!(state.evaluated_members().is_none());
    assert!(state.costs().is_empty());
    state
        .replace(vec![vec![2.0], vec![1.0]], vec![4.0, 1.0])
        .unwrap();
    let counts = EvalCounts {
        cost_evals: 2,
        gradient_evals: 3,
        residual_evals: 4,
        jacobian_evals: 5,
        hessian_evals: 6,
        hessian_product_evals: 7,
    };
    state.mirror(&counts);
    state.update_best();
    assert_eq!(state.candidates(), &[vec![2.0], vec![1.0]]);
    assert_eq!(state.current(), Some((&vec![1.0], 1.0)));
    assert_eq!(state.best_counts(), Some(&counts));
    let storage = state.candidates().as_ptr();
    let (mut members, mut costs) = state.take_members();
    assert_eq!(members.as_ptr(), storage);
    assert!(state.current().is_none());
    assert!(state.evaluated_members().is_none());
    members[0][0] = 4.0;
    members[1][0] = 3.0;
    costs.copy_from_slice(&[16.0, 9.0]);
    state.replace(members, costs).unwrap();
    assert_eq!(state.candidates().as_ptr(), storage);
    state.increment_iter();
    state.mirror(&EvalCounts {
        cost_evals: 4,
        ..counts
    });
    state.update_best();
    coherent(&state);
    assert_eq!(state.best(), Some((&vec![1.0], 1.0)));
    assert_eq!(state.current(), Some((&vec![3.0], 9.0)));
    assert_eq!(state.incumbent_record().unwrap().iter, 0);
    assert_eq!(state.best_counts(), Some(&counts));
    state.replace(vec![vec![0.0]], vec![0.0]).unwrap();
    let final_counts = EvalCounts {
        cost_evals: 5,
        ..counts
    };
    state.mirror(&final_counts);
    state.update_best();
    state.increment_iter();
    state.update_best();
    assert_eq!(state.best_iter(), 1);
    assert_eq!(state.best_counts(), Some(&final_counts));
    assert_eq!(state.cost_evals(), 5);
    state.reset();
    assert_eq!(state.candidates(), &[vec![0.0]]);
    assert!(state.current().is_none());
    assert!(state.best().is_none());
    assert_eq!(state.iter(), 0);
    assert_eq!(state.counts(), &EvalCounts::default());
}

#[test]
fn means_compete_with_members_without_changing_their_order() {
    let mut state = PopulationProgress::empty();
    assert!(state.publish_representative(vec![0.0], 0.0).is_err());
    state
        .replace(vec![vec![2.0], vec![1.0]], vec![4.0, 1.0])
        .unwrap();
    state.publish_representative(vec![3.0], 9.0).unwrap();
    state.update_best();
    assert_eq!(state.current(), Some((&vec![3.0], 9.0)));
    assert_eq!(state.best(), Some((&vec![1.0], 1.0)));
    state.publish_representative(vec![0.5], 0.25).unwrap();
    state.increment_iter();
    state.update_best();
    coherent(&state);
    assert_eq!(state.candidates(), &[vec![2.0], vec![1.0]]);
    assert_eq!(state.costs(), &[4.0, 1.0]);
    assert_eq!(state.best(), Some((&vec![0.5], 0.25)));
    assert_eq!(state.best_iter(), 1);
    state
        .replace(vec![vec![4.0], vec![3.0]], vec![16.0, 9.0])
        .unwrap();
    state.update_best();
    assert_eq!(state.current(), Some((&vec![3.0], 9.0)));
    assert_eq!(state.best(), Some((&vec![0.5], 0.25)));
}

#[test]
fn malformed_updates_preserve_the_entire_population_and_mean() {
    let mut state = PopulationProgress::empty();
    state
        .replace(vec![vec![1.0], vec![2.0]], vec![1.0, 4.0])
        .unwrap();
    state.publish_representative(vec![0.0], 0.0).unwrap();
    state.update_best();
    let before = state.clone();
    for (members, costs) in [
        (vec![], vec![]),
        (vec![vec![]], vec![0.0]),
        (vec![vec![1.0], vec![2.0, 3.0]], vec![1.0, 13.0]),
        (vec![vec![1.0], vec![2.0]], vec![1.0]),
    ] {
        assert!(state.replace(members, costs).is_err());
        assert_eq!(state, before);
    }
    assert!(state.publish_representative(vec![0.0, 0.0], 0.0).is_err());
    assert_eq!(state, before);
}

#[test]
fn ties_and_nonfinite_costs_do_not_corrupt_incumbents() {
    let mut state = PopulationProgress::empty();
    state
        .replace(vec![vec![0.0], vec![1.0]], vec![f64::NAN, f64::INFINITY])
        .unwrap();
    state.update_best();
    assert_eq!(state.current(), Some((&vec![1.0], f64::INFINITY)));
    assert!(state.best().is_none());
    state
        .replace(vec![vec![1.0], vec![-1.0]], vec![1.0, 1.0])
        .unwrap();
    state.publish_representative(vec![-1.0], 1.0).unwrap();
    state.update_best();
    assert_eq!(state.best(), Some((&vec![1.0], 1.0)));
    state.replace(vec![vec![-1.0]], vec![1.0]).unwrap();
    state.increment_iter();
    state.update_best();
    assert_eq!(state.best(), Some((&vec![1.0], 1.0)));
    assert_eq!(state.best_iter(), 0);
    state
        .replace(
            vec![vec![1.0], vec![0.0]],
            vec![f64::NAN, f64::NEG_INFINITY],
        )
        .unwrap();
    state.publish_representative(vec![2.0], f64::NAN).unwrap();
    state.update_best();
    assert_eq!(state.best(), Some((&vec![0.0], f64::NEG_INFINITY)));
    assert!(state.costs()[0].is_nan());
}

fn exact_runs<So>(make: impl Fn() -> So)
where
    So: Solver<Sphere, PopulationProgress<Vec<f64>>, Error = Infallible>,
{
    let expected =
        Executor::new(Sphere::new(), make(), PopulationProgress::empty())
            .max_iter(20)
            .run_with_solver()
            .unwrap();
    for split in [0, 1, 7, 19] {
        let checkpoint =
            Executor::new(Sphere::new(), make(), PopulationProgress::empty())
                .max_iter(split)
                .run_with_solver()
                .unwrap()
                .into_checkpoint();
        let resumed =
            Executor::resume_from_checkpoint(Sphere::new(), checkpoint)
                .require_evaluated_state()
                .max_iter(20)
                .run_with_solver()
                .unwrap();
        coherent(&resumed.state);
        assert_eq!(resumed.state, expected.state);
        assert_eq!(resumed.counts, expected.counts);
    }
}

#[test]
fn exact_continuation_keeps_population_rng_and_refinement_schedule() {
    exact_runs(|| RandomSearch::new(4, 27));
    exact_runs(|| De::new(27).with_pop_size(4).with_dither(0.4, 1.0));
    exact_runs(|| Ssga::new(27).with_pop_size(4));
    exact_runs(|| {
        basin::DeInject::with_inner_solver(
            De::new(27).with_pop_size(4),
            basin::NelderMead::new().projected(),
        )
        .with_refine_every(3)
        .with_k(2)
        .with_inner_max_iter(3)
    });
}

#[cfg(feature = "serde")]
fn serialized_runs<So>(make: impl Fn() -> So)
where
    So: Solver<Sphere, PopulationProgress<Vec<f64>>, Error = Infallible>
        + serde::Serialize
        + serde::de::DeserializeOwned,
{
    let expected =
        Executor::new(Sphere::new(), make(), PopulationProgress::empty())
            .max_iter(15)
            .run_with_solver()
            .unwrap();
    let checkpoint =
        Executor::new(Sphere::new(), make(), PopulationProgress::empty())
            .max_iter(6)
            .run_with_solver()
            .unwrap()
            .into_checkpoint();
    let bytes = postcard::to_allocvec(&checkpoint).unwrap();
    let restored: basin::ExactCheckpoint<So, PopulationProgress<Vec<f64>>> =
        postcard::from_bytes(&bytes).unwrap();
    let resumed = Executor::resume_from_checkpoint(Sphere::new(), restored)
        .require_evaluated_state()
        .max_iter(15)
        .run_with_solver()
        .unwrap();
    assert_eq!(resumed.state, expected.state);
    assert_eq!(resumed.counts, expected.counts);
    assert_eq!(
        postcard::to_allocvec(&resumed.solver).unwrap(),
        postcard::to_allocvec(&expected.solver).unwrap()
    );
    let fresh = Executor::new(Sphere::new(), resumed.solver, resumed.state)
        .require_evaluated_state()
        .max_iter(0)
        .run_with_solver()
        .unwrap();
    assert_eq!(fresh.iter(), 0);
    assert_eq!(fresh.state.best_iter(), 0);
    assert_eq!(fresh.counts.cost_evals, 4);
    assert_eq!(fresh.state.candidates(), expected.state.candidates());
}

#[cfg(feature = "serde")]
#[test]
fn serialized_checkpoints_preserve_rng_and_inner_workspace() {
    serialized_runs(|| RandomSearch::new(4, 27));
    serialized_runs(|| De::new(27).with_pop_size(4).with_dither(0.4, 1.0));
    serialized_runs(|| Ssga::new(27).with_pop_size(4));
    serialized_runs(|| {
        basin::DeInject::with_inner_solver(
            De::new(27).with_pop_size(4),
            basin::NelderMead::new().projected(),
        )
        .with_refine_every(3)
        .with_k(2)
        .with_inner_max_iter(3)
    });
}

impl basin::Gradient for Sphere {
    type Gradient = Vec<f64>;
    fn gradient(&self, x: &Vec<f64>) -> Result<Vec<f64>, Infallible> {
        Ok(x.iter().map(|x| 2.0 * x).collect())
    }
}
impl basin::Residual for Sphere {
    type Param = Vec<f64>;
    type Output = Vec<f64>;
    type Error = Infallible;
    fn residual(&self, x: &Vec<f64>) -> Result<Vec<f64>, Infallible> {
        Ok(x.clone())
    }
}
impl basin::Jacobian for Sphere {
    type Jacobian = basin::DenseMatrix;
    fn jacobian(&self, x: &Vec<f64>) -> Result<basin::DenseMatrix, Infallible> {
        Ok(basin::MatrixIdentity::identity(x.len()))
    }
}
impl basin::Hessian for Sphere {
    type Hessian = basin::DenseMatrix;
    fn hessian(&self, x: &Vec<f64>) -> Result<basin::DenseMatrix, Infallible> {
        Ok(basin::MatrixFromDiagonal::from_diagonal(&vec![
            2.0;
            x.len()
        ]))
    }
}
impl basin::HessianProduct for Sphere {
    fn hessian_product(
        &self,
        _: &Vec<f64>,
        v: &Vec<f64>,
    ) -> Result<Vec<f64>, Infallible> {
        Ok(v.iter().map(|x| 2.0 * x).collect())
    }
}

struct ExternalPopulation;
impl Solver<Sphere, PopulationProgress<Vec<f64>>> for ExternalPopulation {
    type Error = Infallible;
    fn init(
        &mut self,
        p: &mut basin::Problem<Sphere>,
        mut state: PopulationProgress<Vec<f64>>,
    ) -> Result<PopulationProgress<Vec<f64>>, Infallible> {
        state.reset();
        let (members, _) = state.take_members();
        let costs = members
            .iter()
            .map(|x| p.cost(x))
            .collect::<Result<Vec<_>, _>>()?;
        state.replace(members, costs).unwrap();
        Ok(state)
    }
    fn next_iter(
        &mut self,
        p: &mut basin::Problem<Sphere>,
        mut state: PopulationProgress<Vec<f64>>,
    ) -> Result<
        (
            PopulationProgress<Vec<f64>>,
            Option<basin::TerminationReason>,
        ),
        Infallible,
    > {
        let (mut members, mut costs) = state.take_members();
        for (member, cost) in members.iter_mut().zip(&mut costs) {
            member[0] *= 0.5;
            *cost = p.cost(member)?;
        }
        p.cost_and_gradient_and_hessian(&members[0])?;
        p.residual_and_jacobian(&members[0])?;
        p.hessian_product(&members[0], &members[0])?;
        state.replace(members, costs).unwrap();
        Ok((state, Some(basin::TerminationReason::UserRequested)))
    }
}

#[test]
fn external_solver_preserves_every_category_at_a_midstep_stop() {
    let result = Executor::new(
        Sphere::new(),
        ExternalPopulation,
        PopulationProgress::from_population(vec![vec![5.0], vec![4.0]]),
    )
    .require_evaluated_state()
    .run_with_solver()
    .unwrap();
    assert_eq!(result.reason, basin::TerminationReason::UserRequested);
    assert_eq!(result.iter(), 0);
    coherent(&result.state);
    assert_eq!(result.state.candidates(), &[vec![2.5], vec![2.0]]);
    assert_eq!(result.state.best(), Some((&vec![2.0], 4.0)));
    assert_eq!(
        result.counts,
        basin::EvalCounts {
            cost_evals: 5,
            gradient_evals: 1,
            residual_evals: 1,
            jacobian_evals: 1,
            hessian_evals: 1,
            hessian_product_evals: 1,
        }
    );
    assert_eq!(result.state.counts(), &result.counts);
    assert_eq!(result.state.best_counts(), Some(&result.counts));
    assert_eq!(result.state.cost_evals(), 5);
    assert_eq!(result.state.best_iter(), 0);
}

#[derive(Default)]
struct PartialRefinement {
    calls: usize,
}
impl basin::InitialState<Vec<f64>> for PartialRefinement {
    type State = basin::PointState<Vec<f64>>;
    fn seed(&self, x: &Vec<f64>) -> Self::State {
        basin::PointState::new(x.clone())
    }
}
impl basin::WarmStart<Vec<f64>> for PartialRefinement {}
impl basin::MemeticInner<Vec<f64>> for PartialRefinement {
    fn seed_scaled(&mut self, x: &Vec<f64>, _: f64) -> Self::State {
        basin::InitialState::seed(self, x)
    }
}
impl Solver<Sphere, basin::PointState<Vec<f64>>> for PartialRefinement {
    type Error = Infallible;
    fn init(
        &mut self,
        p: &mut basin::Problem<Sphere>,
        mut state: basin::PointState<Vec<f64>>,
    ) -> Result<basin::PointState<Vec<f64>>, Infallible> {
        state.reset();
        let point = vec![if self.calls == 0 { 0.5 } else { 0.0 }];
        let (cost, _, _) = p.cost_and_gradient_and_hessian(&point)?;
        p.residual_and_jacobian(&point)?;
        p.hessian_product(&point, &point)?;
        state.replace(point, cost);
        self.calls += 1;
        Ok(state)
    }
    fn next_iter(
        &mut self,
        _: &mut basin::Problem<Sphere>,
        _: basin::PointState<Vec<f64>>,
    ) -> Result<
        (
            basin::PointState<Vec<f64>>,
            Option<basin::TerminationReason>,
        ),
        Infallible,
    > {
        unreachable!("the test inner terminates at initialization")
    }
    fn terminate(
        &self,
        _: &basin::PointState<Vec<f64>>,
    ) -> Option<basin::TerminationReason> {
        Some(if self.calls == 3 {
            basin::TerminationReason::SolverFailed
        } else {
            basin::TerminationReason::SolverConverged
        })
    }
}

#[test]
fn partial_refinement_failure_publishes_prior_improvements_and_all_inner_work()
{
    let solver = basin::DeInject::with_inner_solver(
        De::new(5).with_pop_size(4),
        PartialRefinement::default(),
    )
    .with_k(3);
    let result = Executor::new(
        Sphere::new(),
        solver,
        PopulationProgress::from_population(vec![vec![4.0]; 4]),
    )
    .require_evaluated_state()
    .max_iter(1)
    .run_with_solver()
    .unwrap();
    assert_eq!(result.reason, basin::TerminationReason::SolverFailed);
    assert_eq!(result.iter(), 0);
    coherent(&result.state);
    assert_eq!(result.state.costs(), &[0.0, 0.25, 16.0, 16.0]);
    assert_eq!(result.state.best(), Some((&vec![0.0], 0.0)));
    assert_eq!(
        result.counts,
        basin::EvalCounts {
            cost_evals: 13,
            gradient_evals: 3,
            residual_evals: 3,
            jacobian_evals: 3,
            hessian_evals: 3,
            hessian_product_evals: 3,
        }
    );
    assert_eq!(result.state.counts(), &result.counts);
    assert_eq!(result.state.best_counts(), Some(&result.counts));
    assert_eq!(result.state.cost_evals(), 13);
}
