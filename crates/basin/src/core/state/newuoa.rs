//! NEWUOA solver state.
//!
//! A single-iterate state for Powell's model-based derivative-free solver
//! [`Newuoa`](crate::solver::Newuoa). It carries the current iterate (the best
//! point found so far, since NEWUOA reports the least-`F` point) and the current
//! trust-region radius `ρ`, which the natural-convergence criterion
//! [`RhoTolerance`](crate::core::termination::RhoTolerance) binds on.
//!
//! The quadratic surrogate, the factored inverse-KKT matrix `H`, and the ρ/Δ
//! schedule live on the **solver** struct (`Newuoa`'s `NewuoaWork`), not here:
//! the same split Levenberg-Marquardt uses for its μ/ν/diag working state, and
//! the reason [`NewuoaState`] is generic over the parameter vector `V` only, not
//! the backend matrix. NEWUOA needs no `linalg`-tier ops from `V` (the model
//! algebra is internal `Vec<F>` scratch), so the state is backend-generic.
//!
//! # Current vs best
//!
//! NEWUOA's reported iterate is monotone non-increasing by construction: the
//! solver writes [`param`](State::param)/[`cost`](State::cost) as the
//! least-`F` point seen so far. So [`best_param`](State::best_param)/
//! [`best_cost`](State::best_cost) coincide with the current iterate at every
//! check, like the sorted-simplex/sorted-population shapes.

use crate::core::math::Scalar;
use crate::core::problem::EvalCounts;
use crate::core::state::{CountsMirror, RhoState, State};

/// Solver state for [`Newuoa`](crate::solver::Newuoa).
///
/// Construct with [`new`](Self::new) from the starting point; the solver
/// evaluates it and seeds the cost/trust-region radius in
/// [`Solver::init`](crate::core::solver::Solver::init).
///
/// The scalar `F` defaults to `f64` so call sites resolve unchanged.
pub struct NewuoaState<V, F = f64> {
    /// Current iterate—the best point found so far (NEWUOA reports the
    /// least-`F` point). Initially the user's starting point.
    pub(crate) param: V,
    /// `F(param)`. `None` before [`Solver::init`](crate::core::solver::Solver::init).
    pub(crate) cost: Option<F>,
    /// Current trust-region radius `ρ`—`+∞` before
    /// [`Solver::init`](crate::core::solver::Solver::init) seeds it from
    /// `ρ_beg`. [`RhoTolerance`](crate::core::termination::RhoTolerance) reads it.
    pub(crate) rho: F,

    // --- best evaluated point (coincides with the current iterate) ---
    pub(crate) best_param: Option<V>,
    pub(crate) best_cost: F,
    pub(crate) best_iter: u64,
    pub(crate) best_cost_evals: u64,

    // --- counters ---
    pub(crate) iter: u64,
    pub(crate) cost_evals: u64,
}

impl<V, F: Scalar> NewuoaState<V, F> {
    /// Build an initial NEWUOA state at the starting point `x0`. The solver
    /// evaluates `x0` (and the rest of the initial interpolation set) and fills
    /// the cost/`ρ` in [`Solver::init`](crate::core::solver::Solver::init).
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
    /// from `ρ_beg` toward `ρ_end` on the eq-7.6 schedule.
    pub fn rho(&self) -> F {
        self.rho
    }
}

impl<V: Clone, F: Scalar> State for NewuoaState<V, F> {
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
            .expect("NewuoaState::cost read before Solver::init evaluated the start point")
    }

    fn best_param(&self) -> &V {
        self.best_param
            .as_ref()
            .expect("NewuoaState::best_param read before Solver::init populated it")
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
                self.best_cost_evals = self.cost_evals;
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

impl<V: Clone, F: Scalar> RhoState for NewuoaState<V, F> {
    fn rho(&self) -> F {
        self.rho
    }
}

impl<V, F> CountsMirror for NewuoaState<V, F>
where
    NewuoaState<V, F>: State,
{
    fn mirror(&mut self, delta: &EvalCounts) {
        // Derivative-free: NEWUOA only ever calls the cost function, so every
        // unit of work folds into the single `cost_evals` counter (the same rule
        // as `BasicPopulationState`/`CmaEsState`).
        self.cost_evals = delta.total_work();
    }
}
