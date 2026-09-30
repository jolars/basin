//! Solver state shapes.
//!
//! Every [`Solver`](crate::core::solver::Solver) carries its iterate as a
//! [`State`]. The base [`State`] trait is the minimum the executor and
//! generic termination criteria need to read; richer state shapes extend
//! it ([`GradientState`] for first-order solvers, [`SimplexState`] for
//! simplex-based solvers like Nelder-Mead) so termination criteria can
//! bound on the minimum capability they need (tenet 3 in `CONTRIBUTING.md`).
//!
//! `State::Float` is generic across the trait. Every shared progress shape
//! takes an `F: Scalar` parameter that defaults to `f64` and supports `f32`.
//! Solvers (`GradientDescent`, `Bfgs`,
//! both `Lbfgs` modes, the NLLS family, CMA-ES, barrier/AL, etc.) and the
//! shipped termination criteria all carry the same `F = f64` default. See
//! `tests/f32_round_trip.rs` for an end-to-end demonstration that the full
//! pipeline composes at `F = f32`.
//!
//! [`AcceptanceState`] exposes proposal acceptance history to generic stall
//! criteria. [`ExactResumeState`] marks a state that contains the complete
//! evolution snapshot needed by [`Executor::resume`](crate::Executor::resume).
//!
//! External and new solvers can use [`PointState`] or [`FirstOrderState`] for
//! shared progress storage with public, coherent record updates. See
//! [`progress`] for their lifecycle and an external-solver example.
//! Their opt-in [`capabilities`] expose checked current and incumbent records,
//! raw evaluation counts, and objective-selection guarantees. [`SelectedState`]
//! supplies shared progress for solver-selected constrained incumbents;
//! [`SelectedFirstOrderState`] adds matching objective gradients. Every shared
//! state preserves the six raw evaluation categories.
//! [`ProposalState`] adds acceptance bookkeeping while leaving proposal
//! strategies and their evolving machinery in the solver.

pub mod capabilities;
pub mod progress;

pub use capabilities::{
    EvaluatedGradientState, EvaluatedState, IncumbentRef, IncumbentState,
    ObjectiveIncumbentState, RawEvaluationState,
};
pub use progress::{
    FirstOrderState, GradientDimensionMismatch, PointState, ProposalState,
    SelectedState,
};

use crate::core::math::Scalar;
use crate::core::problem::EvalCounts;

/// Minimum information the executor and generic termination criteria
/// need to read from a solver's iterate.
///
/// # Contract
///
/// - **Caller must:** construct via the appropriate concrete state
///   constructor (e.g. [`PointState::new`]) before handing the state to
///   [`Executor`](crate::core::executor::Executor). The executor's `init`
///   call populates derived fields (cost, gradient) before any termination
///   check sees the state.
/// - **Implementor must:** keep [`param`](Self::param) stable between
///   iterations: the returned reference is valid until the next
///   [`Solver::next_iter`](crate::core::solver::Solver::next_iter)
///   returns. [`cost_evals`](Self::cost_evals) reports evaluation work
///   using the state's [`CountsMirror`] mapping, not iterations: a single
///   [`Solver::next_iter`](crate::core::solver::Solver::next_iter) may
///   evaluate the cost many times (line searches, Nelder-Mead shrinks),
///   and users budget against this counter rather than
///   [`iter`](Self::iter).
///
/// # Current vs best
///
/// The trait exposes two parallel views of the iterate stream:
///
/// - **Current** ([`param`](Self::param), [`cost`](Self::cost)): what
///   the solver's reported point and its evaluated cost. Examples include
///   an accepted iterate, a Brent probe, and the CMA-ES distribution mean.
///   The sequence may be nonmonotone. Solvers write these.
/// - **Selected incumbent** ([`best_param`](Self::best_param),
///   [`best_cost`](Self::best_cost), [`best_iter`](Self::best_iter),
///   [`best_cost_evals`](Self::best_cost_evals)): the retained point and
///   the iter/eval counts recorded by the state's selection rule.
///   Most states retain the lowest-cost published point. Constrained
///   states such as [`SelectedState`] instead
///   mirror a solver-selected incumbent whose objective may increase
///   as feasibility improves. The executor calls
///   [`update_best`](Self::update_best) after every
///   successful [`Solver::init`](crate::core::solver::Solver::init)/
///   [`Solver::next_iter`](crate::core::solver::Solver::next_iter).
///   Execution controls like
///   [`no_objective_improvement`](crate::Executor::no_objective_improvement) and
///   [`target_objective`](crate::Executor::target_objective) bind on
///   a checked incumbent and require [`ObjectiveIncumbentState`]; solver-owned
///   cost-change tests bind on `cost()`.
///
/// Best tracking sees published state, not every problem evaluation.
/// For example, [`PopulationProgress`] considers its mean and sampled population.
/// An unreported line-search probe is not automatically considered.
/// Evaluation metadata records the count at publication, which can include
/// evaluations made after the selected point was found.
///
/// Shared states, including [`PopulationProgress`] and [`SimplexProgress`],
/// retain historical points independently of current members or vertices.
pub trait State {
    /// The parameter type the solver iterates over (e.g. `Vec<f64>`,
    /// `nalgebra::DVector<f64>`).
    type Param;
    /// The scalar type of the objective, such as `f64` or `f32`.
    type Float;

    /// Number of fully completed iterations. A
    /// [`Solver::next_iter`](crate::core::solver::Solver::next_iter)
    /// that bails mid-iteration with `Some(reason)` does not increment
    /// this counter; see the
    /// [`executor`](crate::core::executor) module for the exact ordering.
    fn iter(&self) -> u64;
    /// Increment [`iter`](Self::iter) by one. Called by the executor
    /// after a successful [`Solver::next_iter`](crate::core::solver::Solver::next_iter).
    fn increment_iter(&mut self);
    /// Cumulative evaluation work under the state's [`CountsMirror`] mapping.
    /// Every shipped shared state reports only raw cost-function calls.
    /// Custom states define their mapping through [`CountsMirror`].
    /// Diverges from `iter()` whenever a single iteration evaluates the
    /// cost more than once (line searches, Nelder-Mead shrinks, etc.);
    /// this is what users actually budget against.
    ///
    /// Populated by the
    /// [`Executor`](crate::core::executor::Executor) from the wrapper's
    /// `EvalCounts` after every
    /// successful
    /// [`Solver::init`](crate::core::solver::Solver::init)/
    /// [`Solver::next_iter`](crate::core::solver::Solver::next_iter);
    /// the per-state mapping is defined by the state's
    /// [`CountsMirror`] impl. Solvers never write to this counter
    /// directly; the wrapper's counts are authoritative.
    fn cost_evals(&self) -> u64;
    /// Current iterate. Stable between
    /// [`Solver::next_iter`](crate::core::solver::Solver::next_iter)
    /// calls; available after successful initialization, including iter 0.
    /// Some constructors supply it immediately, while an empty population
    /// such as [`PopulationProgress::empty`] must first be initialized.
    fn param(&self) -> &Self::Param;
    /// Cost at the current [`param`](Self::param).
    ///
    /// # Panics
    ///
    /// Shared states panic if `cost()` is read before
    /// [`Solver::init`](crate::core::solver::Solver::init) has populated
    /// the cached cost. By contract the executor calls `init` before any
    /// stopping check, so reads from controls and from
    /// [`OptimizationResult`](crate::core::executor::OptimizationResult)
    /// are safe after successful initialization. Constructors supply no
    /// evaluated records; use [`EvaluatedState::current_record`] or an inherent
    /// checked reader to query availability.
    fn cost(&self) -> Self::Float;

    /// Selected incumbent under this state's best-tracking rule.
    ///
    /// Shared progress states retain this point independently of the current
    /// record, including when all current population members change. Panics on
    /// shared states until an eligible incumbent has been selected.
    fn best_param(&self) -> &Self::Param;
    /// Cost of the selected incumbent. This is the historical minimum for
    /// objective-ordered states, but can increase under constrained
    /// incumbent selection.
    fn best_cost(&self) -> Self::Float;
    /// Iteration recorded by the most recent incumbent update. Objective-
    /// ordered states record strict improvements; states that mirror a
    /// solver-selected incumbent may refresh this on every publication.
    fn best_iter(&self) -> u64;
    /// [`cost_evals`](Self::cost_evals) at the publication boundary that
    /// recorded the incumbent. This need not be the precise evaluation
    /// that found the point.
    fn best_cost_evals(&self) -> u64;
    /// Refresh the incumbent using this state's selection rule. Most
    /// states retain strict objective improvements; population states can
    /// also consider exposed samples, and constrained states can mirror
    /// the solver's selected incumbent.
    ///
    /// Called by the [`Executor`](crate::core::executor::Executor) after
    /// every successful
    /// [`Solver::init`](crate::core::solver::Solver::init) /
    /// [`Solver::next_iter`](crate::core::solver::Solver::next_iter),
    /// once the state's counters have been mirrored. Solvers do not
    /// call this directly.
    fn update_best(&mut self);
    /// Reset the best-so-far slots to their pre-init defaults
    /// (`best_cost = +∞`, all best counters zero).
    ///
    /// Called by fresh [`Executor`](crate::Executor) and
    /// [`run_loop_with_control`](crate::core::executor::run_loop_with_control) paths at run entry so a
    /// reused state tracks per-run best. [`Executor::resume`](crate::Executor::resume)
    /// deliberately preserves the snapshot's best history instead.
    fn reset_best(&mut self);
}

/// States that carry a gradient at the current [`param`](State::param).
///
/// # Contract
///
/// - **Implementor must:** at the end of every successful
///   [`Solver::next_iter`](crate::core::solver::Solver::next_iter)
///   (and at the end of [`Solver::init`](crate::core::solver::Solver::init)
///   for first-order solvers), populate
///   [`gradient`](Self::gradient) so it corresponds to the *current*
///   [`param`](State::param). Termination criteria read it; if it lags
///   behind the param they will fire on stale data.
/// - `None` means "no gradient available at this iterate yet": the
///   only legitimate case is before
///   [`Solver::init`](crate::core::solver::Solver::init) has run, used
///   by gradient-norm convergence checks to skip unavailable data.
pub trait GradientState: State {
    /// Gradient at the current [`param`](State::param), if populated.
    fn gradient(&self) -> Option<&Self::Param>;
    /// Cumulative derivative work under the state's [`CountsMirror`]
    /// mapping. In Basin 1.x this can include Jacobian, Hessian, and
    /// Hessian-product evaluations as well as gradient calls.
    ///
    /// Populated by the
    /// [`Executor`](crate::core::executor::Executor) from the wrapper's
    /// `EvalCounts`; see
    /// [`State::cost_evals`] for the broader rule and
    /// [`CountsMirror`] for the per-state mapping.
    fn gradient_evals(&self) -> u64;
    /// [`gradient_evals`](Self::gradient_evals) at the publication boundary
    /// that recorded the incumbent, the companion to
    /// [`State::best_cost_evals`]. This is not necessarily the count at
    /// the precise evaluation that found the point.
    fn best_gradient_evals(&self) -> u64;
}

/// Bridge from the wrapper's
/// `EvalCounts` to the state's
/// [`State::cost_evals`]/[`GradientState::gradient_evals`] counters.
/// The [`Executor`](crate::core::executor::Executor) calls
/// [`mirror`](Self::mirror) after every successful
/// [`Solver::init`](crate::core::solver::Solver::init)/
/// [`Solver::next_iter`](crate::core::solver::Solver::next_iter),
/// passing the per-run delta of the wrapper's counts on fresh and nested runs.
/// Exact-resume execution instead restores the wrapper's cumulative counts and
/// mirrors those onto the resumed state.
///
/// Public (rather than crate-private) so user-defined state types can
/// be plugged into the [`Executor`](crate::core::executor::Executor):
/// the trait must be impl'able outside basin. Most users won't need
/// this: the shipped state types (`PointState`,
/// `FirstOrderState`, `SimplexProgress`, and `PopulationProgress`) already
/// implement it.
///
/// # Per-state mapping
///
/// - **[`PointState`]/[`FirstOrderState`]/[`ProposalState`]/[`SelectedState`]/[`SelectedFirstOrderState`]/[`SimplexProgress`]/[`PopulationProgress`]** preserve every raw category in
///   their inherent `counts()` and `best_counts()` readers. The cost reader
///   and first-order states' gradient readers report only the named category.
///   Use `total_work()` on the raw counts for an explicit aggregate.
pub trait CountsMirror: State {
    /// Overwrite the state's counters from the per-run wrapper delta.
    /// Called by the executor after every successful
    /// [`Solver::init`](crate::core::solver::Solver::init) /
    /// [`Solver::next_iter`](crate::core::solver::Solver::next_iter).
    fn mirror(&mut self, delta: &EvalCounts);
}

/// Capability marker for states that contain enough information to resume an
/// interrupted run without changing its trajectory.
///
/// Implementations must return the complete wrapper-side evaluation counters
/// represented by the snapshot. [`Executor::resume`](crate::Executor::resume)
/// restores those counters, preserves best-so-far history and absolute
/// iteration numbers, and calls the solver's resume-idempotent `init` method.
/// No shipped shared state implements this marker: built-in solvers require
/// [`ExactCheckpoint`](crate::ExactCheckpoint) for exact continuation. For a
/// custom state-only resume contract, solver-specific evolution data—including
/// RNG and adaptive machinery—must
/// live in the state itself. Exact continuation additionally assumes the same
/// deterministic problem, solver configuration, scalar type, and code;
/// executor-owned termination criteria are not part of the snapshot.
/// This path resets solver-owned convergence history. To retain that history
/// too, use [`ExactCheckpoint`](crate::ExactCheckpoint) and
/// [`Executor::resume_from_checkpoint`](crate::Executor::resume_from_checkpoint).
pub trait ExactResumeState: State {
    /// Complete evaluation counts at the point represented by this snapshot.
    fn resume_counts(&self) -> EvalCounts;
}

/// Minimum state shape for acceptance-stall termination criteria.
pub trait AcceptanceState: State {
    /// Absolute iteration of the most recently accepted proposal.
    fn last_accepted_iter(&self) -> u64;

    /// Cumulative number of accepted proposals.
    fn accepted_moves(&self) -> u64;

    /// Cumulative number of rejected proposals.
    fn rejected_moves(&self) -> u64;
}

/// States built around a simplex of `n + 1` vertices and parallel costs.
///
/// Mirrors [`GradientState`]: the trait exists so convergence checks
/// (e.g. the simplex-collapse test of Lagarias et al. 1998, eq. T1) can
/// bound on a richer view than [`State::param`]/[`State::cost`], which
/// only see the best vertex.
///
/// # Contract
///
/// - **Implementor must:** keep [`vertices`](Self::vertices) and
///   [`costs`](Self::costs) sorted by **ascending cost** at the start and
///   end of every [`Solver::next_iter`](crate::core::solver::Solver::next_iter)
///   call (and at the end of [`Solver::init`](crate::core::solver::Solver::init)).
///   So [`State::param`]/[`State::cost`] always return the current best
///   vertex (`vertices[0]`/`costs[0]`).
/// - **Implementor must:** sort `NaN` costs *last*, so a single bad
///   evaluation can't drag itself to the front and become the
///   "best" vertex.
/// - **Implementor must:** keep the two slices the same length and in
///   parallel order: `costs[i]` is the cost at `vertices[i]`.
pub trait SimplexState: State {
    /// All `n + 1` vertices, sorted by ascending cost.
    fn vertices(&self) -> &[Self::Param];
    /// Costs in parallel with [`vertices`](Self::vertices), sorted ascending.
    fn costs(&self) -> &[Self::Float];
}

/// A population of candidate parameters and matching costs.
///
/// Generic consumers can inspect the whole population while
/// [`State::param`]/[`State::cost`] expose one representative record.
///
/// # Contract
///
/// Keep the two slices the same length and in parallel order at evaluated
/// boundaries: `costs[i]` is the cost at `candidates[i]`. Unevaluated seed
/// storage may have no costs. Member order belongs to the solver; generic
/// consumers must not assume ascending costs or that member zero is best.
/// Solvers may sort complete records, or preserve order to align per-member
/// models and histories. [`PopulationProgress::replace`] retains member order.
///
/// The current record need not be a member. Distribution-based solvers may
/// publish an evaluated mean while retaining their sampled population.
pub trait PopulationState: State {
    /// All candidates in the solver's stored order.
    fn candidates(&self) -> &[Self::Param];
    /// Costs in parallel with [`candidates`](Self::candidates).
    fn costs(&self) -> &[Self::Float];
}

/// State that exposes a trust-region radius or step size `ρ`.
///
/// External states may expose this capability. Built-in shared progress states
/// leave algorithm radii on their solvers, which own the update schedules and
/// numerical convergence checks.
pub trait RhoState: State {
    /// The current trust-region radius or step size `ρ`.
    fn rho(&self) -> Self::Float;
}

/// State that exposes a mesh adaptive direct search poll size.
///
/// External states may expose this capability. Built-in MADS solvers own their
/// mesh and poll diagnostics; shared progress states contain evaluated records.
pub trait MeshState: State {
    /// The current poll size `Δᵖ` (bounds the distance from the incumbent to the
    /// poll trial points).
    fn poll_size(&self) -> Self::Float;
    /// The current mesh index `ℓ` (OrthoMADS eq. (1)).
    fn mesh_index(&self) -> i32;
}

/// FMINSEARCH-style initial simplex from a single starting point.
///
/// Implemented uniformly for cloneable vectors with coordinate access. The
/// usual relative step is 5%; zero coordinates use an absolute `0.00025`.
/// The scalar defaults to `f64` and also supports `f32`.
pub trait IntoInitialSimplex<V, F: Scalar = f64> {
    /// Construct `n + 1` vertices by perturbing each coordinate in turn.
    fn into_initial_simplex(self, relative_step: F) -> Vec<V>;
}

impl<V, F> IntoInitialSimplex<V, F> for V
where
    F: Scalar,
    V: Clone + crate::VectorLen + crate::VectorIndex<F>,
{
    fn into_initial_simplex(self, relative_step: F) -> Vec<V> {
        let n = self.vec_len();
        let mut vertices = Vec::with_capacity(n + 1);
        vertices.push(self.clone());
        for i in 0..n {
            let mut vertex = self.clone();
            let value = self.get_scalar(i);
            vertex.set_scalar(
                i,
                if value != F::zero() {
                    (F::one() + relative_step) * value
                } else {
                    F::from_f64(0.00025).unwrap()
                },
            );
            vertices.push(vertex);
        }
        vertices
    }
}

/// Observable shared simplex progress.
pub mod simplex;
pub use simplex::{SimplexProgress, SimplexShapeError};

/// Observable shared population progress.
pub mod population;
pub use population::{PopulationProgress, PopulationShapeError};

/// Shared constrained progress with a matching objective gradient.
pub mod selected_first_order;
pub use selected_first_order::SelectedFirstOrderState;
