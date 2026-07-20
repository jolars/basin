//! Solis-Wets adaptive random-walk state.
//!
//! Carries the accepted iterate `x` **and** the full adaptive machinery
//! of the Solis-Wets step: the bias vector `b` (search momentum), the
//! step size `ρ` (per-coordinate standard deviation of the noise), and
//! the successive success/failure streak counters. Keeping all of it in
//! the state (rather than on the solver) makes a paused
//! `(SolisWets, SolisWetsState)` pair a complete snapshot: an MA-LSCh
//! style outer can store it per individual and resume the local-search
//! chain later, exactly as it stores `(CmaEs, CmaEsState)` pairs.
//!
//! [`SolisWets::init`](crate::solver::SolisWets) is resume-idempotent:
//! the fresh-state sentinel is `cost = None` (set once by `new`), so a
//! resumed state — whose cost is already populated — passes through
//! `init` untouched, mirroring
//! [`CmaEsState`](crate::core::state::CmaEsState)'s empty-population
//! guard.

use crate::core::math::{Scalar, ScaleInPlace, VectorLen};
use crate::core::problem::EvalCounts;
use crate::core::state::{CountsMirror, RhoState, State};

/// Solver state for [`SolisWets`](crate::solver::SolisWets).
///
/// Construct with [`new`](Self::new) (bias `b = 0`, zeroed streak
/// counters). The solver evaluates the start point's cost once in
/// [`Solver::init`](crate::core::solver::Solver::init).
///
/// The Solis-Wets iteration only ever *accepts* improvements, so the
/// current iterate is also the best evaluated point:
/// [`State::param`]/[`State::cost`] and
/// [`State::best_param`]/[`State::best_cost`] coincide at all times.
///
/// The scalar `F` defaults to `f64` so call sites resolve unchanged.
pub struct SolisWetsState<V, F = f64> {
    // --- iterate ---
    /// Current (accepted) point.
    pub(crate) x: V,
    /// `f(x)`. `None` before
    /// [`Solver::init`](crate::core::solver::Solver::init): the
    /// fresh-state sentinel that makes `init` resume-idempotent.
    pub(crate) cost: Option<F>,

    // --- Solis-Wets adaptive machinery ---
    /// Bias vector `b`: momentum toward directions of recorded success.
    pub(crate) bias: V,
    /// Step size `ρ`, the per-coordinate standard deviation of the
    /// sampling noise.
    pub(crate) rho: F,
    /// Successive-success streak `#s` (zeroed on failure and when the
    /// expansion fires).
    pub(crate) num_success: u32,
    /// Successive-failure streak `#f` (zeroed on success and when the
    /// contraction fires).
    pub(crate) num_failure: u32,

    // --- best evaluated point (coincides with x; kept explicit so the
    // --- State contract stays honest under reset_best) ---
    pub(crate) best_param: Option<V>,
    pub(crate) best_cost: F,
    pub(crate) best_iter: u64,
    pub(crate) best_cost_evals: u64,

    // --- counters ---
    pub(crate) iter: u64,
    pub(crate) cost_evals: u64,
}

impl<V, F> SolisWetsState<V, F>
where
    V: Clone + VectorLen + ScaleInPlace<F>,
    F: Scalar,
{
    /// Build an initial Solis-Wets state at `x` with step size `rho`.
    /// The bias starts at `0` and both streak counters at zero; the
    /// solver evaluates `f(x)` in
    /// [`Solver::init`](crate::core::solver::Solver::init).
    ///
    /// # Panics
    ///
    /// Panics if `rho ≤ 0` or `x` is empty.
    pub fn new(x: V, rho: F) -> Self {
        assert!(
            rho > F::zero(),
            "SolisWetsState requires rho > 0, got {:?}",
            rho
        );
        assert!(x.vec_len() >= 1, "SolisWetsState requires a non-empty x");

        let mut bias = x.clone();
        bias.scale_in_place(F::zero());

        Self {
            x,
            cost: None,
            bias,
            rho,
            num_success: 0,
            num_failure: 0,
            best_param: None,
            best_cost: F::infinity(),
            best_iter: 0,
            best_cost_evals: 0,
            iter: 0,
            cost_evals: 0,
        }
    }
}

impl<V, F: Scalar> SolisWetsState<V, F> {
    /// The current step size `ρ` (per-coordinate standard deviation of
    /// the sampling noise). Same value the
    /// [`RhoTolerance`](crate::core::termination::RhoTolerance)
    /// criterion reads through [`RhoState`].
    pub fn rho(&self) -> F {
        self.rho
    }

    /// The current bias vector `b` (search momentum).
    pub fn bias(&self) -> &V {
        &self.bias
    }

    /// Successive successes recorded since the last failure or
    /// step-size expansion.
    pub fn success_count(&self) -> u32 {
        self.num_success
    }

    /// Successive failures recorded since the last success or step-size
    /// contraction.
    pub fn failure_count(&self) -> u32 {
        self.num_failure
    }
}

impl<V: Clone, F: Scalar> State for SolisWetsState<V, F> {
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
        &self.x
    }

    /// Cost at the current iterate, `f(x)`.
    ///
    /// # Panics
    ///
    /// Panics if read before
    /// [`Solver::init`](crate::core::solver::Solver::init) has
    /// evaluated the start point. By contract the executor calls `init`
    /// before any termination check, so reads from criteria and from
    /// [`OptimizationResult`](crate::core::executor::OptimizationResult)
    /// are safe.
    fn cost(&self) -> F {
        self.cost
            .expect("SolisWetsState::cost read before Solver::init evaluated the start point")
    }

    fn best_param(&self) -> &V {
        self.best_param
            .as_ref()
            .expect("SolisWetsState::best_param read before Solver::init populated it")
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
        if let Some(c) = self.cost {
            if self.best_param.is_none() || c < self.best_cost {
                self.best_param = Some(self.x.clone());
                self.best_cost = c;
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

impl<V, F> CountsMirror for SolisWetsState<V, F>
where
    SolisWetsState<V, F>: State,
{
    fn mirror(&mut self, delta: &EvalCounts) {
        // Derivative-free: all work folds into the single `cost_evals`
        // counter, matching `BasicPopulationState` and `CmaEsState`.
        self.cost_evals = delta.total_work();
    }
}

impl<V: Clone, F: Scalar> RhoState for SolisWetsState<V, F> {
    /// For Solis-Wets, `ρ` is the mutation standard deviation rather
    /// than a trust-region radius; the criterion semantics are the same
    /// (`ρ` at its floor ⇒ converged at that resolution).
    fn rho(&self) -> F {
        self.rho
    }
}
