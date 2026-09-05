//! Complete evolution state for simulated annealing.

use crate::core::math::Scalar;
use crate::core::problem::EvalCounts;
use crate::core::rng::ChaCha8Rng;
use crate::core::state::{
    AcceptanceState, CountsMirror, ExactResumeState, State,
};
use crate::solver::simulated_annealing::{
    Reannealing, ReannealingProgress, TemperatureSchedule,
};

/// State for [`SimulatedAnnealing`](crate::solver::SimulatedAnnealing).
///
/// Unlike states used only for warm starts, this type stores every evolving
/// component of the Markov chain: the stateful neighbor, RNG, cooling phase,
/// reannealing progress, incumbent, best-so-far history, and counters. With
/// the `serde` feature, serializing this state and passing the restored value
/// to [`Executor::resume`](crate::Executor::resume) continues the same chain.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug)]
pub struct SimulatedAnnealingState<P, N, F = f64, R = ChaCha8Rng> {
    pub(crate) param: P,
    pub(crate) cost: Option<F>,
    pub(crate) best_param: Option<P>,
    pub(crate) best_cost: F,
    pub(crate) best_iter: u64,
    pub(crate) best_cost_evals: u64,
    pub(crate) iter: u64,
    pub(crate) eval_counts: EvalCounts,
    pub(crate) neighbor: N,
    pub(crate) rng: R,
    pub(crate) initial_temperature: F,
    pub(crate) schedule: TemperatureSchedule<F>,
    pub(crate) cooling_age: u64,
    pub(crate) reannealing: Option<Reannealing>,
    pub(crate) reannealing_progress: ReannealingProgress,
    pub(crate) accepted_moves: u64,
    pub(crate) rejected_moves: u64,
    pub(crate) reannealings: u64,
    pub(crate) last_accepted_iter: u64,
}

impl<P, N, F, R> SimulatedAnnealingState<P, N, F, R>
where
    F: Scalar,
{
    pub(crate) fn new(
        param: P,
        neighbor: N,
        rng: R,
        initial_temperature: F,
        schedule: TemperatureSchedule<F>,
        reannealing: Option<Reannealing>,
    ) -> Self {
        Self {
            param,
            cost: None,
            best_param: None,
            best_cost: F::infinity(),
            best_iter: 0,
            best_cost_evals: 0,
            iter: 0,
            eval_counts: EvalCounts::default(),
            neighbor,
            rng,
            initial_temperature,
            schedule,
            cooling_age: 0,
            reannealing,
            reannealing_progress: ReannealingProgress::default(),
            accepted_moves: 0,
            rejected_moves: 0,
            reannealings: 0,
            last_accepted_iter: 0,
        }
    }

    /// Temperature that will be used for the next proposal.
    pub fn temperature(&self) -> F {
        self.schedule
            .temperature(self.initial_temperature, self.cooling_age)
    }

    /// Number of accepted proposals.
    pub fn accepted_moves(&self) -> u64 {
        self.accepted_moves
    }

    /// Number of rejected proposals.
    pub fn rejected_moves(&self) -> u64 {
        self.rejected_moves
    }

    /// Number of completed schedule restarts.
    pub fn reannealings(&self) -> u64 {
        self.reannealings
    }

    /// Absolute iteration of the most recently accepted proposal.
    pub fn last_accepted_iter(&self) -> u64 {
        self.last_accepted_iter
    }
}

impl<P, N, F, R> State for SimulatedAnnealingState<P, N, F, R>
where
    P: Clone,
    F: Scalar,
{
    type Param = P;
    type Float = F;

    fn iter(&self) -> u64 {
        self.iter
    }

    fn increment_iter(&mut self) {
        self.iter += 1;
    }

    fn cost_evals(&self) -> u64 {
        self.eval_counts.cost_evals
    }

    fn param(&self) -> &P {
        &self.param
    }

    fn cost(&self) -> F {
        self.cost.expect(
            "SimulatedAnnealingState::cost read before Solver::init evaluated the start point",
        )
    }

    fn best_param(&self) -> &P {
        self.best_param.as_ref().expect(
            "SimulatedAnnealingState::best_param read before Solver::init populated it",
        )
    }

    fn best_cost(&self) -> F {
        self.best_cost
    }

    fn best_iter(&self) -> u64 {
        self.best_iter
    }

    fn best_cost_evals(&self) -> u64 {
        self.best_cost_evals
    }

    fn update_best(&mut self) {
        if let Some(cost) = self.cost {
            if self.best_param.is_none() || cost < self.best_cost {
                self.best_param = Some(self.param.clone());
                self.best_cost = cost;
                self.best_iter = self.iter;
                self.best_cost_evals = self.eval_counts.cost_evals;
            }
        }
    }

    fn reset_best(&mut self) {
        self.best_param = None;
        self.best_cost = F::infinity();
        self.best_iter = 0;
        self.best_cost_evals = 0;
    }
}

impl<P, N, F, R> CountsMirror for SimulatedAnnealingState<P, N, F, R>
where
    SimulatedAnnealingState<P, N, F, R>: State,
{
    fn mirror(&mut self, counts: &EvalCounts) {
        self.eval_counts = *counts;
    }
}

impl<P, N, F, R> ExactResumeState for SimulatedAnnealingState<P, N, F, R>
where
    SimulatedAnnealingState<P, N, F, R>: State,
{
    fn resume_counts(&self) -> EvalCounts {
        self.eval_counts
    }
}

impl<P, N, F, R> AcceptanceState for SimulatedAnnealingState<P, N, F, R>
where
    SimulatedAnnealingState<P, N, F, R>: State,
{
    fn last_accepted_iter(&self) -> u64 {
        self.last_accepted_iter
    }

    fn accepted_moves(&self) -> u64 {
        self.accepted_moves
    }

    fn rejected_moves(&self) -> u64 {
        self.rejected_moves
    }
}
