//! Integration tests for [`BasinHopping`] with a Nelder-Mead inner on the
//! default `Vec<f64>` backend.
//!
//! Mirrors the stochastic-solver test conventions (`cma_es_vec.rs`,
//! `random_search_vec.rs`): seeded-trajectory reproducibility, global
//! convergence on a multimodal landscape, a success-rate-over-seeds check
//! (basin-hopping is stochastic, so one seed proves nothing), and
//! cost-eval aggregation across the inner/outer composition boundary
//! (CONTRIBUTING.md "Solver composition" rule 1). Component-level checks
//! (`RandomDisplacement`, `Metropolis`) live in the solver module's unit
//! tests.

use std::cell::RefCell;
use std::rc::Rc;

use basin::core::rng::Rng;
use basin::problems::{Ackley, Rastrigin};
use basin::{
    BasicState, BasinHopping, Executor, InitialState, MaxIter, NelderMead,
    Problem, SimplexTolerance, Solver, State, StepTaker, TerminationReason,
    WarmStart,
};

/// Same seed in, same trajectory out: the reproducibility contract every
/// stochastic solver in basin honors.
#[test]
fn same_seed_yields_identical_trajectory() {
    let run = |seed: u64| {
        Executor::new(
            Ackley::<Vec<f64>>::new(),
            BasinHopping::new(NelderMead::adaptive(), seed)
                .inner_terminate_on(SimplexTolerance::new(1e-8, 1e-8)),
            BasicState::new(vec![2.0, 2.0]),
        )
        .max_iter(40)
        .run()
        .unwrap()
    };

    let a = run(42);
    let b = run(42);
    assert_eq!(a.param(), b.param());
    assert_eq!(a.cost(), b.cost());
    assert_eq!(a.best_cost(), b.best_cost());
    assert_eq!(a.cost_evals(), b.cost_evals());
}

/// Different seeds explore differently (sanity check that the RNG actually
/// drives the walk).
#[test]
fn different_seeds_yield_different_trajectories() {
    let run = |seed: u64| {
        Executor::new(
            Ackley::<Vec<f64>>::new(),
            BasinHopping::new(NelderMead::adaptive(), seed)
                .with_stepsize(1.5)
                .inner_terminate_on(SimplexTolerance::new(1e-8, 1e-8)),
            BasicState::new(vec![3.0, 3.0]),
        )
        .max_iter(20)
        .run()
        .unwrap()
    };

    let a = run(1);
    let b = run(2);
    assert_ne!(a.param(), b.param());
}

/// Ackley (2-D) has a single broad funnel toward the origin riddled with
/// shallow local minima. A Nelder-Mead-only descent stalls in a ripple;
/// basin-hopping escapes via perturbation + Metropolis and drives the
/// global best to ~0. Assert on `best_cost()` (the lowest minimum ever
/// found), since the *current* iterate may be a transiently accepted
/// uphill hop.
#[test]
fn converges_on_ackley_2d() {
    let result = Executor::new(
        Ackley::<Vec<f64>>::new(),
        BasinHopping::new(NelderMead::adaptive(), 7)
            .with_stepsize(1.0)
            .inner_terminate_on(SimplexTolerance::new(1e-10, 1e-10)),
        BasicState::new(vec![2.5, -2.5]),
    )
    .max_iter(200)
    .run()
    .unwrap();

    assert!(
        result.best_cost() < 1e-6,
        "Ackley(2-D) best cost {} (expected < 1e-6)",
        result.best_cost()
    );
}

/// Rastrigin (2-D): a dense integer lattice of local minima with the
/// global minimum at the origin. Over a spread of seeds basin-hopping
/// should reach the global basin (best cost < 1) in a clear majority of
/// runs. A success-rate assertion is the right shape for a stochastic
/// global solver; a single-seed convergence assertion would be brittle.
#[test]
fn success_rate_over_seeds_on_rastrigin_2d() {
    let trials = 12;
    let successes = (0..trials)
        .filter(|&seed| {
            let result = Executor::new(
                Rastrigin::<Vec<f64>>::new(),
                BasinHopping::new(NelderMead::adaptive(), seed)
                    .with_stepsize(1.5)
                    .inner_terminate_on(SimplexTolerance::new(1e-10, 1e-10)),
                BasicState::new(vec![2.0, 3.0]),
            )
            .max_iter(150)
            .run()
            .unwrap();
            result.best_cost() < 1.0
        })
        .count();

    assert!(
        successes * 2 >= trials as usize,
        "basin-hopping reached Rastrigin's global basin in only {successes}/{trials} runs"
    );
}

/// The inner local minimizations must count toward the outer eval budget
/// (same-problem composition, CONTRIBUTING.md "Solver composition" rule 1).
/// Each hop runs a fresh Nelder-Mead solve (≥ n+1 simplex-init evals) plus
/// the one-off initial relaxation in `init`, so over `H` hops the cumulative
/// `cost_evals` must dwarf the hop count — proof the inner work is folded in
/// rather than dropped.
#[test]
fn aggregates_inner_cost_evals() {
    let hops = 30_u64;
    let n = 2_u64;
    let result = Executor::new(
        Ackley::<Vec<f64>>::new(),
        BasinHopping::new(NelderMead::adaptive(), 3)
            .inner_terminate_on(SimplexTolerance::new(1e-8, 1e-8)),
        BasicState::new(vec![1.0, 1.0]),
    )
    .max_iter(hops)
    .terminate_on(MaxIter(hops))
    .run()
    .unwrap();

    // Lower bound: the initial relaxation plus one (n+1)-vertex simplex init
    // per hop. Real counts are far higher (each NM solve iterates), but this
    // is a robust floor that fails loudly if inner evals are not aggregated.
    let floor = (hops + 1) * (n + 1);
    assert!(
        result.cost_evals() >= floor,
        "cost_evals {} below aggregation floor {} — inner evals not folded in",
        result.cost_evals(),
        floor
    );
}

/// Custom [`StepTaker`] that records how often it is asked to take a step
/// and to adjust its scale. Doubles as proof that the trait is implementable
/// from outside the crate (it only needs `basin::core::rng::Rng`). The
/// no-op displacement keeps the walk stationary so hop counting is exact.
struct RecordingStep {
    calls: Rc<RefCell<(u32, u32)>>,
}

impl StepTaker<Vec<f64>, f64> for RecordingStep {
    fn take_step<R: Rng + ?Sized>(
        &mut self,
        x: &Vec<f64>,
        _rng: &mut R,
    ) -> Vec<f64> {
        self.calls.borrow_mut().0 += 1;
        x.clone()
    }

    fn adjust_scale(&mut self, _factor: f64) {
        self.calls.borrow_mut().1 += 1;
    }
}

/// Adaptive control must fire on a cumulative `nstep % interval == 0`
/// schedule (the SciPy semantics — counters are never reset). Over 35 hops
/// with `interval = 10`, the step taker is asked for 35 displacements and
/// adjusted exactly 3 times (at hops 10, 20, 30). This guards the
/// firing schedule and exercises the public `with_step_taker` extension
/// point.
#[test]
fn adaptive_step_fires_on_cumulative_interval_schedule() {
    let calls = Rc::new(RefCell::new((0_u32, 0_u32)));
    let step = RecordingStep {
        calls: Rc::clone(&calls),
    };

    let hops = 35_u64;
    let _ = Executor::new(
        Ackley::<Vec<f64>>::new(),
        BasinHopping::new(NelderMead::adaptive(), 1)
            .with_step_taker(step)
            .with_adaptive_interval(10)
            .inner_terminate_on(SimplexTolerance::new(1e-8, 1e-8)),
        BasicState::new(vec![0.5, 0.5]),
    )
    .max_iter(hops)
    .terminate_on(MaxIter(hops))
    .run()
    .unwrap();

    let (take_steps, adjusts) = *calls.borrow();
    assert_eq!(take_steps, hops as u32, "one perturbation per hop");
    assert_eq!(adjusts, 3, "adjust at cumulative hops 10, 20, 30");
}

/// Inner solver that delegates setup to a real inner (so the inner state's
/// cost is populated) but always reports a soft `SolverFailed` from
/// `next_iter`. Proves the trait is implementable from outside the crate.
struct FailingInner<I>(I);

impl<I, V> InitialState<V> for FailingInner<I>
where
    I: InitialState<V>,
{
    type State = I::State;
    fn seed(&self, x: &V) -> Self::State {
        self.0.seed(x)
    }
}

impl<I, V> WarmStart<V> for FailingInner<I> where I: WarmStart<V> {}

impl<I, P, S> Solver<P, S> for FailingInner<I>
where
    S: State,
    I: Solver<P, S>,
{
    type Error = I::Error;

    fn init(
        &mut self,
        problem: &mut Problem<P>,
        state: S,
    ) -> Result<S, Self::Error> {
        // Delegate so the inner state's cost gets populated; the outer reads it.
        self.0.init(problem, state)
    }

    fn next_iter(
        &mut self,
        _problem: &mut Problem<P>,
        state: S,
    ) -> Result<(S, Option<TerminationReason>), Self::Error> {
        Ok((state, Some(TerminationReason::SolverFailed)))
    }
}

/// A failed inner solve must *not* terminate the outer walk: SciPy keeps
/// hopping past a failed local minimization (marking the candidate
/// unsuccessful for the acceptance guard), so the run should reach its
/// `MaxIter` cap rather than bubbling `SolverFailed`. With the old
/// terminate-on-failure routing this would have stopped on hop 1.
#[test]
fn failed_inner_solve_does_not_terminate_the_walk() {
    let hops = 5_u64;
    let result = Executor::new(
        Ackley::<Vec<f64>>::new(),
        BasinHopping::new(FailingInner(NelderMead::adaptive()), 3)
            .inner_terminate_on(SimplexTolerance::new(1e-8, 1e-8)),
        BasicState::new(vec![2.0, 2.0]),
    )
    .max_iter(hops)
    .terminate_on(MaxIter(hops))
    .run()
    .unwrap();

    assert_eq!(
        result.reason,
        TerminationReason::MaxIter,
        "walk should run to MaxIter, not stop on the inner's SolverFailed"
    );
    assert_eq!(result.iter(), hops, "all hops should execute");
}
