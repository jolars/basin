//! COBYLA solver state.
//!
//! A single-iterate state for Powell's nonlinearly-constrained derivative-free
//! solver [`Cobyla`](crate::solver::Cobyla). It carries the current iterate (the
//! incumbent COBYLA would return) and the current trust-region radius `ρ`, which
//! the natural-convergence criterion
//! [`RhoTolerance`](crate::core::termination::RhoTolerance) binds on.
//!
//! As with [`LincoaState`](crate::core::state::LincoaState), the simplex
//! (`sim` / `simi` / `fval` / `conmat` / `cval`), the penalty parameter `μ`
//! (`cpen`), the ρ/Δ schedule, and the return filter live on the **solver**
//! struct (`Cobyla`'s `CobylaWork`), not here. The constraints describe the
//! *problem* and live problem-side (tenet 4), never on the state.
//!
//! # Current vs best, and why `best ≡ current`
//!
//! COBYLA is a constrained solver: its returned point is chosen from a filter by
//! an L-infinity merit / constraint-violation rule, **not** by objective value
//! alone. The framework's default
//! [`update_best`](State::update_best) ranks by [`cost`](State::cost) only, which
//! cannot see constraint violation and would wrongly prefer a lower-`F`
//! *infeasible* iterate over a higher-`F` feasible one. So this state overrides
//! `update_best` to mirror the *current* iterate: the solver's `next_iter` sets
//! [`param`](State::param) / [`cost`](State::cost) to the work's filter-selected
//! incumbent every step, and `best_*` simply tracks it. The work is the single
//! source of truth for which point is "best".

use crate::core::math::Scalar;
use crate::core::problem::EvalCounts;
use crate::core::state::{CountsMirror, RhoState, State};

/// Solver state for [`Cobyla`](crate::solver::Cobyla).
///
/// Construct with [`new`](Self::new) from the starting point; the solver
/// evaluates the initial simplex and seeds the cost / trust-region radius in
/// [`Solver::init`](crate::core::solver::Solver::init).
///
/// The scalar `F` defaults to `f64` so call sites resolve unchanged.
pub struct CobylaState<V, F = f64> {
    /// Current iterate — the incumbent COBYLA would return (filter-selected by
    /// the work). Initially the user's starting point.
    pub(crate) param: V,
    /// `F(param)`. `None` before [`Solver::init`](crate::core::solver::Solver::init).
    pub(crate) cost: Option<F>,
    /// Current trust-region radius `ρ` — `+∞` before
    /// [`Solver::init`](crate::core::solver::Solver::init) seeds it from
    /// `ρ_beg`. [`RhoTolerance`](crate::core::termination::RhoTolerance) reads it.
    pub(crate) rho: F,

    // --- best evaluated point (mirrors the current iterate; see module docs) ---
    pub(crate) best_param: Option<V>,
    pub(crate) best_cost: F,
    pub(crate) best_iter: u64,
    pub(crate) best_cost_evals: u64,

    // --- counters ---
    pub(crate) iter: u64,
    pub(crate) cost_evals: u64,
}

impl<V, F: Scalar> CobylaState<V, F> {
    /// Build an initial COBYLA state at the starting point `x0`. The solver
    /// evaluates the initial simplex and fills the cost / `ρ` in
    /// [`Solver::init`](crate::core::solver::Solver::init).
    pub fn new(x0: V) -> Self {
        Self {
            param: x0,
            cost: None,
            rho: F::infinity(),
            best_param: None,
            best_cost: F::infinity(),
            best_iter: 0,
            best_cost_evals: 0,
            iter: 0,
            cost_evals: 0,
        }
    }

    /// The current trust-region radius `ρ`. `+∞` before
    /// [`Solver::init`](crate::core::solver::Solver::init); thereafter shrinks
    /// from `ρ_beg` toward `ρ_end`.
    pub fn rho(&self) -> F {
        self.rho
    }
}

impl<V: Clone, F: Scalar> State for CobylaState<V, F> {
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
        &self.param
    }

    /// Cost at the current [`param`](State::param).
    ///
    /// # Panics
    ///
    /// Panics if read before
    /// [`Solver::init`](crate::core::solver::Solver::init) has evaluated the
    /// starting point. By contract the executor calls `init` before any
    /// termination check, so reads from criteria and from
    /// [`OptimizationResult`](crate::core::executor::OptimizationResult) are
    /// safe.
    fn cost(&self) -> F {
        self.cost
            .expect("CobylaState::cost read before Solver::init evaluated the start point")
    }

    fn best_param(&self) -> &V {
        self.best_param
            .as_ref()
            .expect("CobylaState::best_param read before Solver::init populated it")
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

    /// Mirror the current iterate as the best (see module docs): COBYLA's
    /// incumbent is filter-selected by the work, so `best_*` always tracks the
    /// point the solver just reported rather than the cost-minimal one.
    fn update_best(&mut self) {
        if let Some(cost) = self.cost {
            self.best_param = Some(self.param.clone());
            self.best_cost = cost;
            self.best_iter = self.iter;
            self.best_cost_evals = self.cost_evals;
        }
    }

    fn reset_best(&mut self) {
        self.best_param = None;
        self.best_cost = F::infinity();
        self.best_iter = 0;
        self.best_cost_evals = 0;
    }
}

impl<V: Clone, F: Scalar> RhoState for CobylaState<V, F> {
    fn rho(&self) -> F {
        self.rho
    }
}

impl<V, F> CountsMirror for CobylaState<V, F>
where
    CobylaState<V, F>: State,
{
    fn mirror(&mut self, delta: &EvalCounts) {
        // Derivative-free: COBYLA only ever calls the cost / constraint
        // function (one combined evaluation), so every unit of work folds into
        // the single `cost_evals` counter.
        self.cost_evals = delta.total_work();
    }
}
