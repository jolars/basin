//! The [`Solver`] trait every concrete solver implements. See the trait
//! contract for the lifecycle (`init` once, then repeated `next_iter`,
//! with an optional `terminate` hook) and the
//! [`executor`](crate::core::executor) module for the canonical iteration
//! ordering.

use crate::core::problem::Problem;
use crate::core::state::State;
use crate::core::termination::Termination;

/// Progress published by one solver step. Completion and termination are
/// independent: a finished step can also be the step that stops the run.
/// A continuing partial step publishes progress without an iteration callback.
#[derive(Debug)]
pub struct SolverStep<S: State> {
    /// Coherent progress published by this step.
    pub state: S,
    /// Whether the executor should increment the completed-iteration count.
    pub completed: bool,
    /// An optional stopping decision, independent of iteration completion.
    pub termination: Option<Termination<S::Float>>,
}

impl<S: State> SolverStep<S> {
    /// Publish a completed iteration and continue.
    pub fn completed(state: S) -> Self {
        Self {
            state,
            completed: true,
            termination: None,
        }
    }
    /// Publish a partial step and its stopping explanation.
    pub fn stopped(state: S, termination: Termination<S::Float>) -> Self {
        Self {
            state,
            completed: false,
            termination: Some(termination),
        }
    }
    /// Publish a completed iteration and its stopping explanation.
    pub fn completed_with_termination(
        state: S,
        termination: Termination<S::Float>,
    ) -> Self {
        Self {
            state,
            completed: true,
            termination: Some(termination),
        }
    }
    /// Extract the published state and decision. Callers driving iterations
    /// manually must account for `completed` before consuming this value.
    pub fn into_parts(self) -> (S, bool, Option<Termination<S::Float>>) {
        (self.state, self.completed, self.termination)
    }
}

// Tuple conversion preserves the 1.x convention that a stopping step is partial.
impl<S: State> From<(S, Option<Termination<S::Float>>)> for SolverStep<S> {
    fn from((state, termination): (S, Option<Termination<S::Float>>)) -> Self {
        Self {
            completed: termination.is_none(),
            state,
            termination,
        }
    }
}

/// A concrete optimization algorithm. Implementations carry the
/// solver's configuration and any internal scratch state; the iterate
/// itself lives in `S: State`.
///
/// # Contract
///
/// - **Caller must:** drive the solver through
///   [`Executor`](crate::core::executor::Executor) (or
///   [`run_loop_with_control`](crate::core::executor::run_loop_with_control) for composed
///   solvers). The executor calls [`init`](Self::init) exactly once
///   before any [`next_iter`](Self::next_iter) call, and runs
///   termination checks before each iteration including iter 0. See the
///   [`executor`](crate::core::executor) module docs for the canonical
///   loop ordering.
/// - **Implementor must:** populate every state field termination
///   criteria might read before returning from
///   [`init`](Self::init): at minimum [`State::cost`] for any state
///   whose `cost()` panics on missing data, and
///   [`GradientState::gradient`](crate::core::state::GradientState::gradient)
///   for first-order solvers. After every successful
///   [`next_iter`](Self::next_iter), the same fields must again
///   correspond to the *current* [`State::param`].
/// - **Implementor must:** report mid-iteration failures
///   (line-search bailout, non-descent direction, etc.) via
///   [`SolverStep::termination`] field rather than panicking; and use [`terminate`](Self::terminate)
///   only for clean convergence tests on the current state.
///
/// # Eval counting
///
/// Solvers do **not** maintain eval counters by hand. Every cost/
/// gradient/residual/Jacobian/Hessian call goes through the
/// [`Problem`] wrapper, which bumps
/// [`EvalCounts`](crate::core::problem::EvalCounts) on the wrapper
/// before delegating to the user's problem. The
/// [`Executor`](crate::core::executor::Executor) mirrors those counts
/// onto the state's [`State::cost_evals`]/
/// [`GradientState::gradient_evals`](crate::core::state::GradientState::gradient_evals)
/// after every successful [`init`](Self::init)/
/// [`next_iter`](Self::next_iter); see
/// [`CountsMirror`](crate::core::state::CountsMirror) for the per-state
/// mapping.
///
/// # Error type
///
/// The associated [`Error`](Self::Error) is the **hard-abort** error
/// type the solver propagates out of [`init`](Self::init) and
/// [`next_iter`](Self::next_iter). Concrete impls set
/// `type Error = P::Error;` (or `<P as Residual>::Error` for NLLS
/// solvers) so the user's typed problem error flows untouched out of
/// [`Executor::run`](crate::core::executor::Executor::run). Soft per-point
/// rejection still travels through `Ok(f64::INFINITY)`; see the
/// [`problem`](crate::core::problem) module docs.
pub trait Solver<P, S: State> {
    /// Hard-abort error type, mirroring the underlying problem's
    /// `type Error`. See the [trait docs](Self#error-type).
    type Error;

    /// One-time setup before the iteration loop.
    ///
    /// # Contract
    ///
    /// - **Implementor must:** seed every state field that termination
    ///   criteria or downstream
    ///   [`next_iter`](Self::next_iter) calls will read at iter 0: at
    ///   minimum [`State::cost`], plus
    ///   [`GradientState::gradient`](crate::core::state::GradientState::gradient)
    ///   for first-order solvers and the parallel cost array for
    ///   [`SimplexState`](crate::core::state::SimplexState) solvers.
    ///   Termination criteria run *before* the first
    ///   [`next_iter`](Self::next_iter) call (see the
    ///   [`executor`](crate::core::executor) module docs), so an
    ///   already-optimal initial point must be detectable from the state
    ///   `init` returns.
    /// - **Implementor must:** treat each initialization as a fresh run. Reset
    ///   evolving model, component, RNG, and convergence history while retaining
    ///   configured settings. Reset shared progress and reevaluate its seed,
    ///   including when handed a state from an earlier run. Exact checkpoints
    ///   skip initialization and preserve solver, state, and counts together.
    /// - **Implementor may:** return `Err` to abort the run before the
    ///   first iteration; the error bubbles out of
    ///   [`Executor::run`](crate::core::executor::Executor::run).
    fn init(
        &mut self,
        _problem: &mut Problem<P>,
        state: S,
    ) -> Result<S, Self::Error> {
        Ok(state)
    }

    /// Advance one iteration.
    ///
    /// # Contract
    ///
    /// - **Implementor must:** return a state whose
    ///   [`State::param`], [`State::cost`], and (if `S: GradientState`)
    ///   [`GradientState::gradient`](crate::core::state::GradientState::gradient)
    ///   are mutually consistent at the new iterate. Termination
    ///   criteria evaluated *before* the next iteration assume these
    ///   fields agree.
    /// - **Implementor must:** return a [`SolverStep`] with independent
    ///   completion and termination fields. `completed: true` increments the
    ///   iteration counter, including when `termination` is present. A partial
    ///   stopping step publishes coherent progress without incrementing it.
    ///   Numerical failure and safeguards belong in [`Termination`].
    /// - **Implementor may:** return `Err` to hard-abort the run. This consumes
    ///   the state; the executor exposes counts but promises no recoverable
    ///   checkpoint or final observation.
    /// - **Implementor must (composition, adapter-problem inner only):**
    ///   when running an inner solver against an *adapter problem*
    ///   (e.g. [`LogBarrier`](crate::core::barrier::LogBarrier),
    ///   [`AugmentedLagrangian`](crate::core::augmented_lagrangian::AugmentedLagrangian)),
    ///   i.e. a freshly constructed inner [`Problem`], fold the
    ///   inner wrapper's
    ///   [`EvalCounts`](crate::core::problem::EvalCounts) back into the
    ///   outer's via
    ///   [`EvalCounts::add`](crate::core::problem::EvalCounts::add) on
    ///   [`Problem::counts_mut`]. When the inner shares the outer's
    ///   wrapper (same problem type, passed via `&mut`), counts flow
    ///   through automatically and no roll-up is needed. See
    ///   `CONTRIBUTING.md` "Solver composition" for the failure-routing and
    ///   history-reset contracts that still apply uniformly.
    fn next_iter(
        &mut self,
        problem: &mut Problem<P>,
        state: S,
    ) -> Result<SolverStep<S>, Self::Error>;

    /// Reset per-run convergence history without changing algorithm state.
    ///
    /// Called before initialization for fresh runs and state-only resumes.
    /// Exact solver-aware checkpoint resumes retain this history.
    fn reset_convergence(&mut self) {}

    /// Whether the current boundary can enter cost- and step-change checks.
    ///
    /// Return `true` after initialization to seed their history. Solvers that
    /// can reject every trial in an iteration must return `false` at that
    /// boundary, then `true` after an accepted step. Rejection must not update
    /// the checks' history or turn an unchanged iterate into convergence.
    /// Gradient checks and solver-native convergence tests remain active.
    ///
    /// The default preserves observation of every boundary for existing
    /// implementations. Wrappers should forward this hook to their solver.
    fn should_check_iterate_change(&self) -> bool {
        true
    }

    /// Check convergence at an initialized iteration boundary.
    ///
    /// Stateful checks must be idempotent at the same boundary. The default
    /// preserves existing implementations by calling [`terminate`](Self::terminate).
    /// Model-dependent checks may still run inside `next_iter` when their
    /// diagnostics become available.
    fn check_convergence(
        &mut self,
        _problem: &Problem<P>,
        state: &S,
    ) -> Option<Termination<S::Float>> {
        self.terminate(state)
    }

    /// Optional pre-iteration solver-specific termination test.
    ///
    /// Called after [`RunControl`](crate::RunControl) checks but before each [`next_iter`](Self::next_iter) (including
    /// iter 0, after [`init`](Self::init)). Returning `Some(_)` halts
    /// the executor. Use for clean convergence tests that depend only
    /// on the current state; mid-iter failures should be reported via
    /// [`next_iter`](Self::next_iter)'s return value instead. See the
    /// [`executor`](crate::core::executor) module docs for the full
    /// ordering.
    fn terminate(&self, _state: &S) -> Option<Termination<S::Float>> {
        None
    }
}
