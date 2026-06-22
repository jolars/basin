//! Composition adapter: drive an inner solver from inside an outer
//! solver's [`next_iter`](crate::core::solver::Solver::next_iter).
//!
//! [`InnerExecutor`] mirrors [`Executor`](crate::core::executor::Executor)'s
//! builder ergonomics (`max_iter`, `terminate_on`) but does *not* own the
//! problem — outer solvers store one as a field and call
//! [`InnerExecutor::run`] against the borrowed `&P` they receive in
//! `next_iter`. Internally [`InnerExecutor::run`] is exactly
//! [`run_loop`]; the wrapper just owns the
//! solver, the criteria vec, and the iteration budget so the same set of
//! settings can be reused across outer iters without re-allocating.
//!
//! See `CONTRIBUTING.md` "Solver composition" for the three load-bearing rules
//! (eval aggregation, criteria statelessness across calls, failure
//! routing) every outer solver must follow.

use crate::core::executor::{OptimizationResult, run_loop};
use crate::core::problem::Problem;
use crate::core::solver::Solver;
use crate::core::state::{CountsMirror, State};
use crate::core::termination::TerminationCriterion;

/// Marks a solver as eligible to be the *inner* of a composed solver.
///
/// Composed solvers — [`BarrierMethod`](crate::solver::BarrierMethod) and
/// [`AugmentedLagrangianMethod`](crate::solver::AugmentedLagrangianMethod)
/// re-solving their barrier / augmented-Lagrangian subproblem at each
/// continuation step, or the CMA-injection family via the
/// [`MemeticInner`](crate::solver::MemeticInner) sub-trait — repeatedly seed
/// a fresh inner state at the current outer iterate, drive the inner over it,
/// then read the refined iterate back via [`State::param`].
///
/// The seed/state machinery itself lives on the [`InitialState`] supertrait
/// (which also powers the top-level
/// [`Executor::from_start`](crate::core::executor::Executor::from_start)
/// convenience constructor). `WarmStart` adds no items — it is a marker
/// certifying that a solver's natural [`seed`](InitialState::seed) is
/// *composition-safe*. A solver can build a default initial state
/// ([`InitialState`]) without being a blessed inner: e.g. `TrustRegion` and
/// the Powell / MADS families implement `InitialState` but not `WarmStart`.
pub trait WarmStart<V>: InitialState<V> {}

/// Build a solver's natural initial [`State`] from a starting point.
///
/// `seed(&self, x)` returns a fresh state whose [`State::param`] equals `x`,
/// using the solver's *natural default scale*: there is no outer step-size to
/// track, so a simplex solver picks its own default edge, a Hessian-history
/// solver starts from the identity, a trust-region solver uses its default
/// radius, and so on.
///
/// This is the capability behind
/// [`Executor::from_start`](crate::core::executor::Executor::from_start) —
/// `Executor::from_start(problem, solver, x0)` calls
/// [`seed`](Self::seed) so the caller never names the concrete state type.
/// The composition layer reaches it through the [`WarmStart`] subtrait
/// (which marks a solver as a validated inner); plain `InitialState` callers
/// get the front-door constructor without that promise.
///
/// # Contract
///
/// **Implementor must:** return a state whose [`State::param`] equals `x`
/// (a fresh seed, not a continuation of any previous solve).
pub trait InitialState<V> {
    /// State shape this solver iterates against.
    type State: State<Param = V>;

    /// Build a fresh state seeded at `x` using the solver's natural
    /// default scale.
    fn seed(&self, x: &V) -> Self::State;
}

/// Pre-configured inner solver an outer solver drives once per outer
/// iteration.
///
/// Owns the inner solver, its termination criteria, and its `max_iter`
/// budget. The problem is supplied (borrowed) at [`run`](Self::run) time,
/// so the outer solver can pass the `&P` it receives in
/// [`next_iter`](crate::core::solver::Solver::next_iter) without taking
/// ownership.
///
/// Mirrors [`Executor`](crate::core::executor::Executor)'s builder API:
/// [`max_iter`](Self::max_iter) and [`terminate_on`](Self::terminate_on)
/// are chainable. The differences are (a) the problem isn't owned, and
/// (b) [`run`](Self::run) is reusable — the same `InnerExecutor` is
/// expected to be invoked many times across the outer's lifetime.
///
/// [`run_loop`] stays as the lower-level
/// escape hatch for outer solvers that want to reconstruct criteria per
/// call.
///
/// # Composition contracts
///
/// Three rules outer solvers must follow when consuming the result of
/// [`run`](Self::run); see also `CONTRIBUTING.md` "Solver composition":
///
/// 1. **Eval aggregation.** The [`Problem`] wrapper bumps
///    [`EvalCounts`](crate::core::problem::EvalCounts) on every
///    cost / gradient / residual / Jacobian / Hessian call, and the
///    executor mirrors the per-run delta onto the inner state via
///    [`CountsMirror`]. What the outer must do depends on which problem
///    the inner sees:
///
///    - **Same-problem inner** (the outer passes its own
///      `&mut Problem<P>` to [`run`](Self::run)): the inner's calls bump
///      the *same* wrapper as the outer's, so aggregation happens
///      transparently. No explicit roll-up; the outer state's
///      [`CountsMirror`] impl decides how the counts surface on its
///      [`State::cost_evals`] /
///      [`GradientState::gradient_evals`](crate::core::state::GradientState::gradient_evals).
///    - **Adapter-problem inner** (the outer builds a fresh
///      `Problem::new(adapter)` per outer iter — e.g. the barrier /
///      augmented-Lagrangian methods): after [`run`](Self::run) returns,
///      fold the inner wrapper's counts back into the outer's wrapper via
///      [`EvalCounts::add`](crate::core::problem::EvalCounts::add) on
///      [`Problem::counts_mut`]. Skipping this fold silently corrupts
///      `MaxCostEvals` budgets and the public `result.cost_evals()`.
///
///    See the [`Solver::next_iter`]
///    contract for the canonical wording.
///
/// 2. **Criteria are reset per run.** Criteria registered with
///    [`terminate_on`](Self::terminate_on) live for the whole lifetime of
///    the `InnerExecutor` and are reused on every [`run`](Self::run)
///    call, but each is
///    [`reset`](crate::core::termination::TerminationCriterion::reset) at
///    the start of every run (via [`run_loop`]), so any per-run internal
///    state is cleared first. Stateful criteria are therefore safe to
///    reuse: [`MaxTime`](crate::core::termination::MaxTime) (start instant),
///    [`RelativeGradientTolerance`](crate::core::termination::RelativeGradientTolerance)
///    (anchored `‖∇f_0‖`), and
///    [`NoImprovement`](crate::core::termination::NoImprovement) (stall
///    counter) all behave as freshly constructed each call. A *custom*
///    criterion holding cross-call state must override `reset` to clear it
///    (the default is a no-op).
///
/// 3. **Failure routing.** [`run`](Self::run) returns a full
///    [`OptimizationResult`]; classify the reason. Use
///    [`TerminationReason::is_failure`](crate::core::termination::TerminationReason::is_failure)
///    to decide whether to bubble: `SolverFailed` should bubble via the
///    outer's mid-iter `Option<TerminationReason>` return; everything
///    else (`MaxIter`, `*Tolerance`, `SolverConverged`) is a "clean stop"
///    the outer can consume and continue past.
pub struct InnerExecutor<S, So> {
    solver: So,
    criteria: Vec<Box<dyn TerminationCriterion<S>>>,
    max_iter: u64,
}

impl<S: State + CountsMirror, So> InnerExecutor<S, So> {
    /// Build an inner executor around `solver`. Default `max_iter` is
    /// 1000, mirroring [`Executor::new`](crate::core::executor::Executor::new).
    pub fn new(solver: So) -> Self {
        Self {
            solver,
            criteria: Vec::new(),
            max_iter: 1000,
        }
    }

    /// Set the inner-loop iteration budget. Each call to
    /// [`run`](Self::run) drives the inner solver up to this many
    /// iterations.
    pub fn max_iter(mut self, n: u64) -> Self {
        self.max_iter = n;
        self
    }

    /// Add a termination criterion to the inner loop. Criteria are
    /// checked in insertion order before each inner iteration. See the
    /// type-level "Composition contracts" for the statelessness
    /// requirement that applies because criteria are reused across
    /// [`run`](Self::run) calls.
    pub fn terminate_on<C>(mut self, criterion: C) -> Self
    where
        C: TerminationCriterion<S> + 'static,
    {
        self.criteria.push(Box::new(criterion));
        self
    }

    /// Read-only access to the inner solver. Lets composed outer
    /// solvers dispatch on the inner before [`run`](Self::run) — e.g. to
    /// build an inner state via [`InitialState::seed`] or
    /// `MemeticInner::seed_scaled`. Mutable access goes through
    /// [`run`](Self::run), which already takes `&mut self`.
    pub fn solver(&self) -> &So {
        &self.solver
    }

    /// Drive the inner solver against `problem` from `state`, returning
    /// the final inner state and termination reason. Reusable: call once
    /// per outer iter.
    ///
    /// The inner state's [`State::cost_evals`] reflects only per-run
    /// work (snapshot-relative against the wrapper count at entry), not
    /// cumulative across calls. The wrapper itself accumulates
    /// monotonically — for same-problem composition the outer reads
    /// its own [`Problem::counts`] after `run` to see total work; for
    /// adapter-problem composition the outer builds a fresh inner
    /// `Problem` and folds counts via
    /// [`EvalCounts::add`](crate::core::problem::EvalCounts::add) on
    /// [`Problem::counts_mut`] after `run` returns.
    ///
    /// Internally exactly
    /// [`run_loop`] — `init` is called
    /// on every invocation, so the inner solver sees a fresh setup pass
    /// each time (e.g. seeding cost/gradient at the new starting point).
    pub fn run<P>(
        &mut self,
        problem: &mut Problem<P>,
        state: S,
    ) -> Result<OptimizationResult<S>, So::Error>
    where
        So: Solver<P, S>,
    {
        run_loop(
            problem,
            state,
            &mut self.solver,
            &mut self.criteria,
            self.max_iter,
        )
    }
}
