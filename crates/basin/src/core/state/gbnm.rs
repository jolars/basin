//! State for Globalized Bounded Nelder-Mead.

use crate::core::math::Scalar;
use crate::core::problem::EvalCounts;
use crate::core::state::{BasicSimplexState, CountsMirror, State};

/// The restart that produced the active local simplex.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub(crate) enum GbnmRestart {
    Probabilistic,
    SmallTest,
    LargeTest,
}

/// State for [`Gbnm`](crate::solver::Gbnm).
///
/// The active projected Nelder-Mead search is available through
/// [`vertices`](Self::vertices) and [`costs`](Self::costs). GBNM may replace
/// that simplex with a deliberately worse global restart, so this state does
/// not implement [`SimplexState`](crate::SimplexState): applying a local
/// simplex-collapse criterion to the outer global algorithm would stop it at
/// its first local convergence point. Use an evaluation, iteration, time, or
/// target-cost budget instead.
///
/// `param()` and `cost()` describe the best vertex of the active local
/// simplex. `best_param()` and `best_cost()` retain the best point seen over
/// every restart. The state also retains every local-search start and every
/// distinct possible local optimum identified by the Luersen–Le Riche restart
/// tests.
///
/// With the `serde` feature, this state and [`Gbnm`](crate::Gbnm) can be stored
/// together in an [`ExactCheckpoint`](crate::ExactCheckpoint). The solver
/// carries the RNG, so state-only resume is intentionally not supported.
///
/// The scalar `F` defaults to `f64`.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct GbnmState<V, F = f64> {
    pub(crate) initial: V,
    pub(crate) simplex: Option<BasicSimplexState<V, F>>,
    pub(crate) search_starts: Vec<V>,
    pub(crate) local_optima: Vec<(V, F)>,
    pub(crate) restart: GbnmRestart,
    pub(crate) restart_center: Option<V>,
    pub(crate) restart_count: u64,
    pub(crate) iter: u64,
    pub(crate) cost_evals: u64,
    pub(crate) best_param: Option<V>,
    pub(crate) best_cost: F,
    pub(crate) best_iter: u64,
    pub(crate) best_cost_evals: u64,
}

impl<V, F: Scalar> GbnmState<V, F> {
    /// Build an uninitialized GBNM state at the caller's first local-search
    /// start. [`Gbnm::init`](crate::solver::Gbnm) clamps the point into the
    /// problem's box and constructs a regular simplex around it.
    pub fn new(initial: V) -> Self {
        Self {
            initial,
            simplex: None,
            search_starts: Vec::new(),
            local_optima: Vec::new(),
            restart: GbnmRestart::Probabilistic,
            restart_center: None,
            restart_count: 0,
            iter: 0,
            cost_evals: 0,
            best_param: None,
            best_cost: F::infinity(),
            best_iter: 0,
            best_cost_evals: 0,
        }
    }

    /// Vertices of the active local simplex, sorted by ascending cost.
    /// Returns an empty slice before solver initialization.
    pub fn vertices(&self) -> &[V] {
        self.simplex
            .as_ref()
            .map_or(&[], |simplex| simplex.vertices.as_slice())
    }

    /// Costs of the active local simplex, sorted in parallel with
    /// [`vertices`](Self::vertices). Returns an empty slice before solver
    /// initialization.
    pub fn costs(&self) -> &[F] {
        self.simplex
            .as_ref()
            .map_or(&[], |simplex| simplex.costs.as_slice())
    }

    /// Starting centers of all local searches, including the caller-provided
    /// first start after projection into the box.
    pub fn search_starts(&self) -> &[V] {
        &self.search_starts
    }

    /// Distinct possible local optima found by the GBNM convergence tests,
    /// paired with their objective values.
    pub fn local_optima(&self) -> &[(V, F)] {
        &self.local_optima
    }

    /// Number of local-search restarts after the initial simplex.
    pub fn restart_count(&self) -> u64 {
        self.restart_count
    }
}

impl<V: Clone, F: Scalar> State for GbnmState<V, F> {
    type Param = V;
    type Float = F;

    fn iter(&self) -> u64 {
        self.iter
    }

    fn increment_iter(&mut self) {
        self.iter += 1;
    }

    fn cost_evals(&self) -> u64 {
        self.cost_evals
    }

    fn param(&self) -> &V {
        self.simplex
            .as_ref()
            .map(|simplex| &simplex.vertices[0])
            .unwrap_or(&self.initial)
    }

    fn cost(&self) -> F {
        self.simplex
            .as_ref()
            .map(|simplex| simplex.costs[0])
            .expect("GbnmState::cost read before Solver::init populated it")
    }

    fn best_param(&self) -> &V {
        self.best_param.as_ref().expect(
            "GbnmState::best_param read before Solver::init populated it",
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
        let current = self.cost();
        let uninitialized = self.best_param.is_none();
        if uninitialized || (!current.is_nan() && current < self.best_cost) {
            self.best_param = Some(self.param().clone());
            self.best_iter = self.iter;
            self.best_cost_evals = self.cost_evals;
            if !current.is_nan() {
                self.best_cost = current;
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

impl<V, F> CountsMirror for GbnmState<V, F>
where
    GbnmState<V, F>: State,
{
    fn mirror(&mut self, delta: &EvalCounts) {
        self.cost_evals = delta.total_work();
    }
}
