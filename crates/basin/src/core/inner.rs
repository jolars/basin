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
//! See `AGENTS.md` "Solver composition" for the three load-bearing rules
//! (eval aggregation, criteria statelessness across calls, failure
//! routing) every outer solver must follow.

use crate::core::executor::{run_loop, OptimizationResult};
use crate::core::problem::Problem;
use crate::core::solver::Solver;
use crate::core::state::{CountsMirror, State};
use crate::core::termination::TerminationCriterion;

/// Construct a fresh inner-solver [`State`] seeded at a point.
///
/// Implemented by solvers that can serve as the *inner* of a composed
/// solver — one that repeatedly minimizes a wrapped subproblem starting
/// from the current outer iterate (e.g.
/// [`BarrierMethod`](crate::solver::BarrierMethod) and
/// [`AugmentedLagrangianMethod`](crate::solver::AugmentedLagrangianMethod)
/// re-solving their barrier / augmented-Lagrangian subproblem at each
/// continuation step). The outer solver calls [`seed`](Self::seed) to
/// build a private inner state at the warm-start point, drives the inner
/// over it, then reads the refined iterate back via
/// [`State::param`].
///
/// [`seed`](Self::seed) uses the solver's *natural default scale*: there
/// is no outer step-size to track, so a simplex solver picks its own
/// default edge, a Hessian-history solver starts from the identity, and so
/// on. The CMA-flavored, step-size-scaled variant lives on the
/// [`MemeticInner`](crate::solver::MemeticInner) sub-trait, which extends
/// this one.
///
/// # Contract
///
/// **Implementor must:** return a state whose
/// [`State::param`] equals `x` (a fresh
/// seed, not a continuation of any previous solve), so the outer solver's
/// warm start is honored.
pub trait WarmStart<V> {
    /// State shape this solver iterates against.
    type State: State<Param = V>;

    /// Build a fresh inner state seeded at `x` using the solver's natural
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
/// [`run`](Self::run); see also `AGENTS.md` "Solver composition":
///
/// 1. **Eval aggregation.** The outer must roll the inner's
///    [`State::cost_evals`] (and
///    [`GradientState::gradient_evals`](crate::core::state::GradientState::gradient_evals)
///    when both inner and outer states are
///    [`GradientState`](crate::core::state::GradientState)) into the
///    outer state via the `increment_*_evals` setters. Otherwise
///    `MaxCostEvals` budgets and the public `result.cost_evals()` lie.
///    See the [`Solver::next_iter`]
///    contract for the canonical wording.
///
/// 2. **Criteria statelessness across calls.** Criteria registered with
///    [`terminate_on`](Self::terminate_on) live for the whole lifetime of
///    the `InnerExecutor` and are reused on every [`run`](Self::run)
///    call. They MUST be stateless across runs — fine for
///    [`MaxIter`](crate::core::termination::MaxIter),
///    [`GradientTolerance`](crate::core::termination::GradientTolerance),
///    and [`MaxCostEvals`](crate::core::termination::MaxCostEvals); *not*
///    fine for
///    [`MaxTime`](crate::core::termination::MaxTime), whose internal
///    `start` instant carries across calls and would fire prematurely on
///    later runs. If you need per-run criteria, build a fresh
///    `InnerExecutor` each call (or call
///    [`run_loop`] directly with a
///    fresh `Vec`).
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
    /// solvers dispatch on the inner before / after `run` (e.g. to
    /// construct an inner state via a `MemeticInner::seed` call, or to
    /// read a `MemeticInner::work_units` total off the result).
    /// Mutable access goes through `run`, which already takes
    /// `&mut self`.
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
