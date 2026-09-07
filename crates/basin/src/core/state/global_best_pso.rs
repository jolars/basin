//! Complete evolution state for global-best particle swarm optimization.

use crate::core::math::Scalar;
use crate::core::problem::EvalCounts;
use crate::core::rng::ChaCha8Rng;
use crate::core::state::{
    CountsMirror, ExactResumeState, PopulationState, State,
};

/// State for [`GlobalBestPso`](crate::solver::GlobalBestPso).
///
/// Besides the current population, this stores each particle's velocity and
/// personal best, the swarm's global best, and the live random-number
/// generator. Consequently, a serialized initialized state contains the full
/// stochastic evolution snapshot needed by [`Executor::resume`](crate::Executor::resume)
/// when it is paired with the same solver configuration and problem.
///
/// Construct an empty state with [`new`](Self::new), a position warm start
/// with [`from_positions`](Self::from_positions), or a complete position and
/// velocity warm start with
/// [`from_positions_and_velocities`](Self::from_positions_and_velocities).
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug)]
pub struct GlobalBestPsoState<V, F = f64, R = ChaCha8Rng> {
    pub(crate) positions: Vec<V>,
    pub(crate) costs: Vec<F>,
    pub(crate) velocities: Vec<V>,
    pub(crate) personal_best_positions: Vec<V>,
    pub(crate) personal_best_costs: Vec<F>,
    pub(crate) global_best_position: Option<V>,
    pub(crate) global_best_cost: F,
    pub(crate) rng: Option<R>,
    pub(crate) initialized: bool,
    pub(crate) iter: u64,
    pub(crate) eval_counts: EvalCounts,
    pub(crate) best_cost: F,
    pub(crate) best_iter: u64,
    pub(crate) best_cost_evals: u64,
}

impl<V, F: Scalar, R> GlobalBestPsoState<V, F, R> {
    fn empty() -> Self {
        Self {
            positions: Vec::new(),
            costs: Vec::new(),
            velocities: Vec::new(),
            personal_best_positions: Vec::new(),
            personal_best_costs: Vec::new(),
            global_best_position: None,
            global_best_cost: F::infinity(),
            rng: None,
            initialized: false,
            iter: 0,
            eval_counts: EvalCounts::default(),
            best_cost: F::infinity(),
            best_iter: 0,
            best_cost_evals: 0,
        }
    }

    /// Build an empty state that the solver initializes uniformly from the
    /// problem's finite box.
    pub fn new() -> Self {
        Self::empty()
    }

    /// Warm-start the swarm from caller-supplied positions.
    ///
    /// The solver samples initial velocities according to its research
    /// profile and evaluates all positions in [`Solver::init`](crate::Solver::init).
    /// If [`GlobalBestPso::with_swarm_size`](crate::GlobalBestPso::with_swarm_size)
    /// was called, the configured size must equal `positions.len()`.
    ///
    /// # Panics
    ///
    /// Panics if `positions` is empty.
    pub fn from_positions(positions: Vec<V>) -> Self {
        assert!(
            !positions.is_empty(),
            "GlobalBestPsoState requires at least one position"
        );
        let mut state = Self::empty();
        state.positions = positions;
        state
    }

    /// Warm-start the swarm from positions and their matching velocities.
    ///
    /// The solver evaluates the positions and seeds all personal/global bests
    /// during initialization. Out-of-box positions are repaired using the
    /// configured [`PsoBoundaryHandling`](crate::PsoBoundaryHandling) policy.
    ///
    /// # Panics
    ///
    /// Panics if the position list is empty or the two lists differ in length.
    pub fn from_positions_and_velocities(
        positions: Vec<V>,
        velocities: Vec<V>,
    ) -> Self {
        assert!(
            !positions.is_empty(),
            "GlobalBestPsoState requires at least one position"
        );
        assert_eq!(
            positions.len(),
            velocities.len(),
            "GlobalBestPsoState requires one velocity per position"
        );
        let mut state = Self::empty();
        state.positions = positions;
        state.velocities = velocities;
        state
    }

    /// Current particle velocities in the same order as
    /// [`PopulationState::candidates`].
    pub fn velocities(&self) -> &[V] {
        &self.velocities
    }

    /// Personal-best position for each current particle.
    pub fn personal_best_positions(&self) -> &[V] {
        &self.personal_best_positions
    }

    /// Personal-best costs in parallel with
    /// [`personal_best_positions`](Self::personal_best_positions).
    pub fn personal_best_costs(&self) -> &[F] {
        &self.personal_best_costs
    }

    /// Best position found by the whole swarm.
    ///
    /// # Panics
    ///
    /// Panics before [`Solver::init`](crate::Solver::init) initializes the
    /// swarm.
    pub fn global_best_position(&self) -> &V {
        self.global_best_position.as_ref().expect(
            "GlobalBestPsoState::global_best_position read before Solver::init",
        )
    }

    /// Best cost found by the whole swarm.
    pub fn global_best_cost(&self) -> F {
        self.global_best_cost
    }
}

impl<V, F: Scalar, R> Default for GlobalBestPsoState<V, F, R> {
    fn default() -> Self {
        Self::new()
    }
}

impl<V: Clone, F: Scalar, R> State for GlobalBestPsoState<V, F, R> {
    type Param = V;
    type Float = F;

    fn iter(&self) -> u64 {
        self.iter
    }

    fn increment_iter(&mut self) {
        self.iter += 1;
    }

    fn cost_evals(&self) -> u64 {
        self.eval_counts.total_work()
    }

    fn param(&self) -> &V {
        self.global_best_position()
    }

    fn cost(&self) -> F {
        self.global_best_cost
    }

    fn best_param(&self) -> &V {
        self.global_best_position()
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
        if self.global_best_cost < self.best_cost {
            self.best_cost = self.global_best_cost;
            self.best_iter = self.iter;
            self.best_cost_evals = self.eval_counts.total_work();
        }
    }

    fn reset_best(&mut self) {
        self.best_cost = F::infinity();
        self.best_iter = 0;
        self.best_cost_evals = 0;
    }
}

impl<V, F, R> CountsMirror for GlobalBestPsoState<V, F, R>
where
    GlobalBestPsoState<V, F, R>: State,
{
    fn mirror(&mut self, counts: &EvalCounts) {
        self.eval_counts = *counts;
    }
}

impl<V, F, R> ExactResumeState for GlobalBestPsoState<V, F, R>
where
    GlobalBestPsoState<V, F, R>: State,
{
    fn resume_counts(&self) -> EvalCounts {
        self.eval_counts
    }
}

impl<V, F: Scalar, R> PopulationState for GlobalBestPsoState<V, F, R>
where
    V: Clone,
{
    fn candidates(&self) -> &[V] {
        &self.positions
    }

    fn costs(&self) -> &[F] {
        &self.costs
    }
}
