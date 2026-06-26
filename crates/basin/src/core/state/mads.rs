//! MADS solver state.
//!
//! A single-iterate state for the mesh adaptive direct search solver
//! [`Mads`](crate::solver::Mads). It carries the current incumbent (the best
//! feasible point found so far, since MADS only ever moves to an improving mesh
//! point) and the current **poll size** `Δᵖ`, which the natural-convergence
//! criterion [`MeshTolerance`](crate::core::termination::MeshTolerance) binds on.
//!
//! The mesh/poll bookkeeping (the integer mesh index `ℓ`, the Halton-index
//! schedule, the incumbent in flat scratch) lives on the **solver** struct
//! (`Mads`'s `MadsWork`), not here: the same split the Powell-family DFO solvers
//! use. So [`MadsState`] is generic over the parameter vector `V` only: the
//! direction algebra is internal `Vec<f64>`/`Vec<i64>` scratch and needs no
//! `linalg`-tier ops from `V`.
//!
//! # Current vs best
//!
//! MADS's reported iterate is monotone non-increasing by construction (it accepts
//! only improving mesh points), so [`best_param`](State::best_param)/
//! [`best_cost`](State::best_cost) coincide with the current iterate at every
//! check, like the sorted-simplex/Powell-DFO shapes.

use crate::core::math::Scalar;
use crate::core::problem::EvalCounts;
use crate::core::state::{CountsMirror, MeshState, State};

/// Solver state for [`Mads`](crate::solver::Mads).
///
/// Construct with [`new`](Self::new) from the starting point; the solver
/// evaluates it and seeds the cost/poll size in
/// [`Solver::init`](crate::core::solver::Solver::init).
///
/// The scalar `F` defaults to `f64` so call sites resolve unchanged.
pub struct MadsState<V, F = f64> {
    /// Current incumbent—the best feasible point found so far. Initially the
    /// user's starting point.
    pub(crate) param: V,
    /// `f(param)`. `None` before [`Solver::init`](crate::core::solver::Solver::init).
    pub(crate) cost: Option<F>,
    /// Current poll size `Δᵖ`—`+∞` before
    /// [`Solver::init`](crate::core::solver::Solver::init) seeds it. Shrinks
    /// (≈ halving on unsuccessful iterations) toward the configured floor.
    /// [`MeshTolerance`](crate::core::termination::MeshTolerance) reads it.
    pub(crate) poll_size: F,
    /// Current mesh index `ℓ` (OrthoMADS eq. (1)): `ℓ−1` on success, `ℓ+1` on
    /// failure. `0` before init.
    pub(crate) mesh_index: i32,

    // --- best evaluated point (coincides with the current iterate) ---
    pub(crate) best_param: Option<V>,
    pub(crate) best_cost: F,
    pub(crate) best_iter: u64,
    pub(crate) best_cost_evals: u64,

    // --- counters ---
    pub(crate) iter: u64,
    pub(crate) cost_evals: u64,
}

impl<V, F: Scalar> MadsState<V, F> {
    /// Build an initial MADS state at the starting point `x0`. The solver
    /// evaluates `x0` and fills the cost/poll size in
    /// [`Solver::init`](crate::core::solver::Solver::init).
    pub fn new(x0: V) -> Self {
        Self {
            param: x0,
            cost: None,
            poll_size: F::infinity(),
            mesh_index: 0,
            best_param: None,
            best_cost: F::infinity(),
            best_iter: 0,
            best_cost_evals: 0,
            iter: 0,
            cost_evals: 0,
        }
    }

    /// The current poll size `Δᵖ`. `+∞` before
    /// [`Solver::init`](crate::core::solver::Solver::init); thereafter shrinks
    /// toward the configured floor.
    pub fn poll_size(&self) -> F {
        self.poll_size
    }

    /// The current mesh index `ℓ` (OrthoMADS eq. (1)).
    pub fn mesh_index(&self) -> i32 {
        self.mesh_index
    }
}

impl<V: Clone, F: Scalar> State for MadsState<V, F> {
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
            .expect("MadsState::cost read before Solver::init evaluated the start point")
    }

    fn best_param(&self) -> &V {
        self.best_param
            .as_ref()
            .expect("MadsState::best_param read before Solver::init populated it")
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

impl<V: Clone, F: Scalar> MeshState for MadsState<V, F> {
    fn poll_size(&self) -> F {
        self.poll_size
    }

    fn mesh_index(&self) -> i32 {
        self.mesh_index
    }
}

impl<V, F> CountsMirror for MadsState<V, F>
where
    MadsState<V, F>: State,
{
    fn mirror(&mut self, delta: &EvalCounts) {
        // Derivative-free: MADS only ever calls the cost function, so every unit
        // of work folds into the single `cost_evals` counter (the same rule as
        // the other derivative-free states).
        self.cost_evals = delta.total_work();
    }
}

/// Solver state for the **constrained** MADS variant
/// (`Mads<Constrained>`), which handles nonlinear inequality constraints
/// `c(x) ≤ 0` by the progressive barrier (see
/// [`Mads::constrained`](crate::Mads::constrained)).
///
/// It carries the same poll/mesh bookkeeping as [`MadsState`] plus the
/// **aggregate constraint violation** `h(x) = Σⱼ max(cⱼ(x), 0)²` at the reported
/// incumbent ([`constraint_violation`](Self::constraint_violation); `0` ⇔
/// feasible). Construct it with [`new`](Self::new) from the starting point; an
/// **infeasible start is allowed** (the progressive barrier relaxes its way to
/// feasibility).
///
/// # Current vs best
///
/// Unlike the unconstrained [`MadsState`], the reported objective is **not**
/// monotone: a newly found feasible point can have a higher `f` than a prior
/// infeasible incumbent. The solver's progressive-barrier driver owns the
/// incumbent selection, so [`update_best`](State::update_best) mirrors the
/// current iterate unconditionally (the same rule
/// [`CobylaState`](crate::CobylaState) uses), rather than ranking by `cost`.
///
/// The scalar `F` defaults to `f64` so call sites resolve unchanged.
pub struct ConstrainedMadsState<V, F = f64> {
    /// Current reported incumbent—the best feasible point if one has been
    /// found, else the best infeasible point within the barrier threshold.
    pub(crate) param: V,
    /// `f(param)`. `None` before [`Solver::init`](crate::core::solver::Solver::init).
    pub(crate) cost: Option<F>,
    /// Aggregate constraint violation `h(param)` at the reported incumbent.
    /// `+∞` before init; `0` once the incumbent is feasible.
    pub(crate) constraint_violation: F,
    /// Current poll size `Δᵖ`—`+∞` before init.
    pub(crate) poll_size: F,
    /// Current mesh index `ℓ` (OrthoMADS eq. (1)). `0` before init.
    pub(crate) mesh_index: i32,

    // --- best evaluated point (mirrors the current incumbent) ---
    pub(crate) best_param: Option<V>,
    pub(crate) best_cost: F,
    pub(crate) best_iter: u64,
    pub(crate) best_cost_evals: u64,

    // --- counters ---
    pub(crate) iter: u64,
    pub(crate) cost_evals: u64,
}

impl<V, F: Scalar> ConstrainedMadsState<V, F> {
    /// Build an initial constrained-MADS state at the starting point `x0` (which
    /// may be infeasible). The solver evaluates `x0`'s cost and violation and
    /// fills the poll size in [`Solver::init`](crate::core::solver::Solver::init).
    pub fn new(x0: V) -> Self {
        Self {
            param: x0,
            cost: None,
            constraint_violation: F::infinity(),
            poll_size: F::infinity(),
            mesh_index: 0,
            best_param: None,
            best_cost: F::infinity(),
            best_iter: 0,
            best_cost_evals: 0,
            iter: 0,
            cost_evals: 0,
        }
    }

    /// The aggregate constraint violation `h = Σⱼ max(cⱼ, 0)²` at the reported
    /// incumbent. `0` once the incumbent is feasible; `+∞` before
    /// [`Solver::init`](crate::core::solver::Solver::init).
    pub fn constraint_violation(&self) -> F {
        self.constraint_violation
    }

    /// The current poll size `Δᵖ`.
    pub fn poll_size(&self) -> F {
        self.poll_size
    }

    /// The current mesh index `ℓ` (OrthoMADS eq. (1)).
    pub fn mesh_index(&self) -> i32 {
        self.mesh_index
    }
}

impl<V: Clone, F: Scalar> State for ConstrainedMadsState<V, F> {
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
    /// starting point.
    fn cost(&self) -> F {
        self.cost
            .expect("ConstrainedMadsState::cost read before Solver::init evaluated the start point")
    }

    fn best_param(&self) -> &V {
        self.best_param
            .as_ref()
            .expect("ConstrainedMadsState::best_param read before Solver::init populated it")
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

    /// Mirror the current incumbent unconditionally—the progressive-barrier
    /// driver owns incumbent selection, and the reported `cost` is not monotone
    /// across the feasibility transition, so a `cost`-ranked best would be wrong.
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

impl<V: Clone, F: Scalar> MeshState for ConstrainedMadsState<V, F> {
    fn poll_size(&self) -> F {
        self.poll_size
    }

    fn mesh_index(&self) -> i32 {
        self.mesh_index
    }
}

impl<V, F> CountsMirror for ConstrainedMadsState<V, F>
where
    ConstrainedMadsState<V, F>: State,
{
    fn mirror(&mut self, delta: &EvalCounts) {
        // Derivative-free: every unit of work (cost + constraint evals) folds
        // into the single `cost_evals` counter.
        self.cost_evals = delta.total_work();
    }
}
