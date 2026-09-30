//! Iteration driver. The high-level entry point is [`Executor`];
//! [`Stepper`] exposes one-iteration-at-a-time control, and [`run_loop_with_control`]
//! is the borrowed-problem variant used by composed solvers.
//!
//! # Canonical iteration ordering
//!
//! [`Executor::run`] (and the equivalent [`Stepper`]/[`run_loop_with_control`]
//! paths) drive the solver through this exact sequence, and every
//! contract elsewhere in the framework cross-links here:
//!
//! 1. Convergence history resets, then [`Solver::init`] is called **once**. The
//!    returned state is what iter-0 sees.
//! 2. Then, repeatedly, before each [`Solver::next_iter`] call
//!    (including the first):
//!    1. An executor-attached [`CancellationToken`] is checked, when
//!       configured. Cancellation stops the run with
//!       [`Termination::Cancelled`]. The borrowed [`run_loop_with_control`] path
//!       has no attached token and skips this step.
//!    2. Execution controls check iteration, legacy cost and gradient budgets,
//!       raw evaluation budgets, elapsed time, target cost, improvement stall,
//!       and acceptance stall, followed by application hooks in insertion
//!       order. See [`RunControl`].
//!    3. [`Solver::check_convergence`] evaluates solver-owned convergence.
//!       Its default delegates to the legacy [`Solver::terminate`] hook.
//!       The first stop ends the run.
//! 3. If nothing fired, [`Solver::next_iter`] returns a [`SolverStep`]. The
//!    executor publishes counts and coherent progress, increments the iteration
//!    counter when `completed` is true, and updates the incumbent.
//! 4. A step with `termination` stops with a report tagged
//!    [`TerminationStage::Step`]. Otherwise execution returns to step 2.
//!
//! Because checks happen *before* iter 0, an already-optimal initial
//! point exits immediately with the corresponding reason rather than
//! taking one redundant step.
//!
//! With [`Executor::require_evaluated_state`], each successful `init` or
//! `next_iter` return is checked for a complete record before bookkeeping or
//! observation, including clean mid-step stops. Restored checkpoints are also
//! validated before observation. Missing records panic as solver contract
//! violations; hard problem errors bypass publication and validation.
//!
//! [`Executor::resume`] restores state-carried evolution data and evaluation
//! counters, while [`Executor::resume_from_checkpoint`] restores the solver,
//! state, and counters from an [`ExactCheckpoint`] and skips `init`.
//! Consume a paused stepper with [`Stepper::into_checkpoint`], or retain the
//! final solver and raw counters with [`Executor::run_with_solver`].

use crate::core::checkpoint::{CheckpointSink, ExactCheckpoint};
use crate::core::convergence::NativeConvergenceTest;
use crate::core::observer::{
    ObservationEvent, Observe, ObserveSolver, ObserverMode, SolverCallback,
    StateObserver,
};
use crate::core::problem::{EvalCounts, Problem};
use crate::core::run_control::RunControl;
use crate::core::solver::Solver;
use crate::core::state::{CountsMirror, ExactResumeState, State};
use crate::core::termination::{
    ApplicationStop, Termination, TerminationReport, TerminationStage,
};
use crate::{Scalar, SolverStep};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

/// Shared, one-shot signal for stopping an [`Executor`] between iterations.
///
/// Clones refer to the same lock-free flag, so a UI or worker thread can keep
/// one handle while the executor owns another. Calling [`cancel`](Self::cancel)
/// is idempotent; a token cannot be reset.
///
/// Cancellation is cooperative: the executor checks after solver
/// initialization and before each new top-level iteration. It does not
/// interrupt an active [`Solver::next_iter`] or problem evaluation. For
/// finer-grained cancellation, return a typed error from the problem method.
///
/// # Example
///
/// ```
/// use basin::CancellationToken;
///
/// let token = CancellationToken::new();
/// let cancel_handle = token.clone();
/// assert!(!token.is_cancelled());
///
/// cancel_handle.cancel();
/// assert!(token.is_cancelled());
/// ```
#[derive(Clone, Debug, Default)]
pub struct CancellationToken {
    cancelled: Arc<AtomicBool>,
}

impl CancellationToken {
    /// Create a token in the active (not cancelled) state.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Request cancellation. Calling this more than once has no further
    /// effect.
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Relaxed);
    }

    /// Whether cancellation has been requested through this token or any of
    /// its clones.
    #[must_use]
    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Relaxed)
    }
}

/// Outcome of an optimization run.
///
/// Owns the final progress, authoritative evaluation counts, and stopping report.
/// Delegates `param()`/`cost()`/`iter()` to the underlying state so
/// callers don't need to import `State` for the common reads.
/// Use [`Executor::run_with_solver`] to retain the final solver as well
/// in an [`OptimizationResultWithSolver`].
pub struct OptimizationResult<S: State> {
    /// Authoritative evaluation counts in this run's scope.
    pub counts: EvalCounts,
    /// Final solver state at termination.
    pub state: S,
    /// Why the executor stopped.
    pub report: TerminationReport<S::Float>,
}

impl<S: State> OptimizationResult<S> {
    /// Final iterate.
    pub fn param(&self) -> &S::Param {
        self.state.param()
    }

    /// Cost at the final iterate.
    pub fn cost(&self) -> S::Float {
        self.state.cost()
    }

    /// Number of fully completed iterations.
    pub fn iter(&self) -> u64 {
        self.state.iter()
    }

    /// Cumulative cost-function evaluations across the run.
    pub fn cost_evals(&self) -> u64 {
        self.state.cost_evals()
    }

    /// Best iterate observed during the run: the lowest-cost point
    /// the executor ever saw. For sorted-simplex/sorted-population
    /// states this coincides with [`param`](Self::param); for non-
    /// monotone single-iterate runs (Brent's probes, future SA) the
    /// two diverge.
    pub fn best_param(&self) -> &S::Param {
        self.state.best_param()
    }

    /// Cost at [`best_param`](Self::best_param).
    pub fn best_cost(&self) -> S::Float {
        self.state.best_cost()
    }

    /// Iteration at which [`best_param`](Self::best_param) was found.
    pub fn best_iter(&self) -> u64 {
        self.state.best_iter()
    }

    /// Cumulative cost evaluations at the moment
    /// [`best_param`](Self::best_param) was found; answers "how many
    /// evals until the solver hit its best?".
    pub fn best_cost_evals(&self) -> u64 {
        self.state.best_cost_evals()
    }

    /// Consume the result and return the final state.
    pub fn into_state(self) -> S {
        self.state
    }
}

/// Outcome of an optimization run that retains the final solver by ownership.
///
/// Returned by [`Executor::run_with_solver`] and
/// [`Stepper::run_to_end_with_solver`]. The solver retains its final model,
/// history, and other evolving machinery. Neither it nor the state needs to
/// implement `Clone` or serialization.
///
/// Convenience readers use the same state semantics as [`OptimizationResult`].
/// Shared states report raw cost calls through [`cost_evals`](Self::cost_evals).
/// Custom states define their own [`CountsMirror`] mapping.
/// [`counts`](Self::counts) always contains the authoritative per-category counters.
pub struct OptimizationResultWithSolver<S: State, So> {
    /// Final solver state at termination.
    pub state: S,
    /// Final solver, including its evolving machinery and convergence history.
    pub solver: So,
    /// Authoritative evaluation counters, including counts restored on resume.
    pub counts: EvalCounts,
    /// Why the executor stopped.
    pub report: TerminationReport<S::Float>,
}

impl<S: State, So> OptimizationResultWithSolver<S, So> {
    /// Native criteria from this stopping event, without consulting retained
    /// solver history. Shared and native criteria can both appear in a report.
    pub fn native_convergence_tests(&self) -> Vec<NativeConvergenceTest> {
        self.report.termination.native_convergence_tests()
    }
}

impl<S: State, So> OptimizationResultWithSolver<S, So> {
    /// Consume the result and return the final state, dropping the solver.
    pub fn into_state(self) -> S {
        self.state
    }

    /// Consume the result and retain ordinary progress and its stop reason.
    ///
    /// Drops the solver while preserving the report and authoritative counts.
    pub fn into_result(self) -> OptimizationResult<S> {
        OptimizationResult {
            state: self.state,
            report: self.report,
            counts: self.counts,
        }
    }

    /// Consume the result into a checkpoint for exact continuation.
    ///
    /// Moves the solver and state without cloning or serialization and
    /// discards the termination reason. Pass the checkpoint to
    /// [`Executor::resume_from_checkpoint`] with the same problem and newly
    /// configured execution policy to continue from this boundary.
    pub fn into_checkpoint(self) -> ExactCheckpoint<So, S> {
        ExactCheckpoint::from_parts(self.solver, self.state, self.counts)
    }
}

impl<S: State, So> OptimizationResultWithSolver<S, So> {
    /// Final iterate.
    pub fn param(&self) -> &S::Param {
        self.state.param()
    }

    /// Cost at the final iterate.
    pub fn cost(&self) -> S::Float {
        self.state.cost()
    }

    /// Number of fully completed iterations.
    pub fn iter(&self) -> u64 {
        self.state.iter()
    }

    /// Cumulative evaluation work under the state's [`CountsMirror`] mapping.
    ///
    /// Use `counts.cost_evals` for the raw number of cost-function calls.
    pub fn cost_evals(&self) -> u64 {
        self.state.cost_evals()
    }

    /// Best iterate under the state's incumbent-selection semantics.
    pub fn best_param(&self) -> &S::Param {
        self.state.best_param()
    }

    /// Cost at [`best_param`](Self::best_param).
    pub fn best_cost(&self) -> S::Float {
        self.state.best_cost()
    }

    /// Iteration at which [`best_param`](Self::best_param) was selected.
    pub fn best_iter(&self) -> u64 {
        self.state.best_iter()
    }

    /// State-mirrored evaluation work when the best iterate was selected.
    pub fn best_cost_evals(&self) -> u64 {
        self.state.best_cost_evals()
    }
}

/// Outcome of a single [`Stepper::step`] call.
///
/// `Stopped` carries the same [`TerminationReport`] the executor would
/// have returned. After `Stopped` is returned once, subsequent calls to
/// `step` keep returning the same `Stopped(reason)` so callers don't
/// have to track whether they're done.
#[derive(Debug, Clone, PartialEq)]
pub enum StepOutcome<F: Scalar = f64> {
    /// Execution may continue after this step's publication. Read the state
    /// iteration count to distinguish completed iterations from partial steps.
    Continue,
    /// Termination fired with the given reason. Subsequent
    /// [`Stepper::step`] calls keep returning this same outcome.
    Stopped(TerminationReport<F>),
}

impl<F: Scalar> StepOutcome<F> {
    /// Borrow the report when this step stopped the run.
    pub fn report(&self) -> Option<&TerminationReport<F>> {
        match self {
            Self::Continue => None,
            Self::Stopped(report) => Some(report),
        }
    }
}

/// Drive a solver one iteration at a time.
///
/// Owns the problem, state, solver, and execution controls, runs
/// `solver.init` exactly once on construction, and exposes
/// [`step`](Self::step)/[`run_to_end`](Self::run_to_end) so callers can
/// interleave their own work between iterations: recording trajectories,
/// animating from a UI, pausing on a button press, evaluating a custom
/// budget, etc.
///
/// [`Executor::run`] is `self.into_stepper().run_to_end()`; the stepper
/// is the building block, the executor is the convenience wrapper.
///
/// # Example
///
/// ```ignore
/// let solver = solver.with_absolute_gradient_tolerance(1e-6);
/// let mut stepper = Executor::new(problem, solver, state)
///     .max_iter(100)
///     .into_stepper()?;
///
/// let reason = loop {
///     match stepper.step()? {
///         StepOutcome::Continue => { /* observe `stepper.state()` */ }
///         StepOutcome::Stopped(reason) => break reason,
///     }
/// };
/// ```
pub struct Stepper<P, S: State, So> {
    problem: Problem<P>,
    // `Option<S>` because `Solver::next_iter` consumes the state by
    // value. A hard error cannot restore that owned value, so the slot
    // remains empty after a failed step. Successful steps restore it.
    state: Option<S>,
    solver: So,
    control: RunControl<S>,
    observers: Vec<(Box<dyn ObserveSolver<S, So>>, ObserverMode)>,
    checkpoints: Vec<(Box<dyn CheckpointSink<So, S>>, ObserverMode)>,
    cancellation_token: Option<CancellationToken>,
    finished: Option<TerminationReport<S::Float>>,
}

impl<P, S, So> Stepper<P, S, So>
where
    S: State + CountsMirror,
    So: Solver<P, S>,
{
    /// Read-only access to the current state, between steps.
    ///
    /// # Panics
    ///
    /// Panics after [`step`](Self::step) returns `Err`, because the
    /// failing solver call consumed the state.
    pub fn state(&self) -> &S {
        self.state
            .as_ref()
            .expect("state slot is Some between steps")
    }

    /// Read the solver's current model and diagnostics between steps.
    /// For registered logging, use [`Executor::observe_solver`] or
    /// [`Executor::observe_solver_with`].
    ///
    /// This borrows the solver without cloning it or performing evaluations.
    /// After a hard error its contents are diagnostic only; exact continuation
    /// requires a successfully published solver-and-state checkpoint.
    pub fn solver(&self) -> &So {
        &self.solver
    }

    /// Wrapper-side evaluation counters. These are authoritative:
    /// solvers can only call into the user's problem through the
    /// wrapper, so every cost/gradient/residual/Jacobian /
    /// Hessian call is reflected here. The state mirror under
    /// [`state`](Self::state) is refreshed after every successful
    /// [`Solver::init`] /
    /// [`Solver::next_iter`];
    /// on the typed-`Err` path the state slot is dropped (see
    /// [`step`](Self::step)) but `counts` is still readable here for
    /// diagnostics.
    pub fn counts(&self) -> &EvalCounts {
        self.problem.counts()
    }

    /// Termination reason if the stepper has stopped, else `None`.
    pub fn finished(&self) -> Option<&TerminationReport<S::Float>> {
        self.finished.as_ref()
    }

    /// Total iterations that have completed so far. Convenience read
    /// equivalent to `self.state().iter()`.
    ///
    /// # Panics
    ///
    /// Panics after [`step`](Self::step) returns `Err`, as
    /// [`state`](Self::state) does.
    pub fn iter(&self) -> u64 {
        self.state().iter()
    }

    /// Run one solver step. Once a `Stopped` outcome has been returned
    /// the stepper is sticky: subsequent calls keep returning the same
    /// `Stopped(reason)` without touching the state or solver.
    ///
    /// Registered observers fire here:
    /// [`observe_iter`](Observe::observe_iter) after completed iterations,
    /// including completed stopping steps, gated by each observer's
    /// [`ObserverMode`]; [`observe_final`](Observe::observe_final) once
    /// when this call first returns [`StepOutcome::Stopped`]. See the
    /// [`observer`](crate::core::observer) module for the lifecycle.
    ///
    /// Returns `Err` when the underlying problem returns `Err` from any
    /// cost/gradient/residual/Jacobian/Hessian call during the
    /// step. A hard error consumes the state and may leave solver machinery
    /// partially updated. It does not set [`finished`](Self::finished).
    /// Callers can inspect [`counts`](Self::counts), then drop the stepper or
    /// call [`into_checkpoint`](Self::into_checkpoint), which returns `None`.
    /// State access and further stepping are not supported after the error.
    /// Observers and checkpoint sinks do not fire on the failed transition.
    pub fn step(&mut self) -> Result<StepOutcome<S::Float>, So::Error> {
        if let Some(report) = &self.finished {
            return Ok(StepOutcome::Stopped(report.clone()));
        }
        let starting_iter = self.state().iter();
        let outcome = if self
            .cancellation_token
            .as_ref()
            .is_some_and(CancellationToken::is_cancelled)
        {
            StepOutcome::Stopped(TerminationReport {
                termination: Termination::Cancelled,
                stage: TerminationStage::Boundary,
                iteration: self.state().iter(),
            })
        } else {
            step_once(
                &mut self.problem,
                &EvalCounts::default(),
                &mut self.state,
                &mut self.solver,
                &mut self.control,
            )?
        };
        match &outcome {
            StepOutcome::Continue => {
                let state = self
                    .state
                    .as_ref()
                    .expect("state slot is Some after Continue");
                let iter = state.iter();
                if iter == starting_iter {
                    return Ok(outcome);
                }
                // `update_best` set `best_iter == iter` iff this iteration
                // strictly improved the incumbent; that is exactly the
                // `NewBest` firing condition.
                let is_new_best = state.best_iter() == iter;
                for (checkpoint, mode) in self.checkpoints.iter_mut() {
                    if mode.fires_on(iter, is_new_best) {
                        checkpoint.save(
                            &self.solver,
                            state,
                            self.problem.counts(),
                        );
                    }
                }
                for (observer, mode) in self.observers.iter_mut() {
                    if mode.fires_on(iter, is_new_best) {
                        observer.observe_iter(state, &self.solver);
                    }
                }
            }
            StepOutcome::Stopped(reason) => {
                self.finished = Some(reason.clone());
                let state =
                    self.state.as_ref().expect("state slot is Some on Stopped");
                for (checkpoint, _mode) in self.checkpoints.iter_mut() {
                    checkpoint.save(&self.solver, state, self.problem.counts());
                }
                if reason.stage == (TerminationStage::Step { completed: true })
                {
                    for (observer, mode) in self.observers.iter_mut() {
                        if mode.fires_on(
                            state.iter(),
                            state.best_iter() == state.iter(),
                        ) {
                            observer.observe_iter(state, &self.solver);
                        }
                    }
                }
                for (observer, _mode) in self.observers.iter_mut() {
                    observer.observe_final(state, &self.solver, reason);
                }
            }
        }
        Ok(outcome)
    }

    /// Drive [`step`](Self::step) to completion and return an
    /// [`OptimizationResult`].
    /// Use [`run_to_end_with_solver`](Self::run_to_end_with_solver) to retain
    /// the final solver as well.
    pub fn run_to_end(self) -> Result<OptimizationResult<S>, So::Error> {
        self.run_to_end_with_solver()
            .map(OptimizationResultWithSolver::into_result)
    }

    /// Drive [`step`](Self::step) to completion, retaining the final solver,
    /// state, raw evaluation counters, and termination reason by ownership.
    ///
    /// Uses the same lifecycle as [`run_to_end`](Self::run_to_end), including
    /// observer and checkpoint callbacks. An already-stopped stepper returns
    /// its recorded reason without repeating final callbacks. No `Clone` or
    /// serialization bounds are required.
    ///
    /// Returns the solver's error on a failed transition, without a partial
    /// result or recoverable checkpoint. Calling this after a previous
    /// [`step`](Self::step) error is unsupported, just as with `run_to_end`.
    pub fn run_to_end_with_solver(
        mut self,
    ) -> Result<OptimizationResultWithSolver<S, So>, So::Error> {
        loop {
            if let StepOutcome::Stopped(reason) = self.step()? {
                return Ok(OptimizationResultWithSolver {
                    state: self
                        .state
                        .take()
                        .expect("state slot is Some on stop"),
                    solver: self.solver,
                    counts: *self.problem.counts(),
                    report: reason,
                });
            }
        }
    }

    /// Consume the stepper into a checkpoint at its current boundary.
    ///
    /// Returns `Some` after successful initialization, between completed
    /// steps, or after a clean stop (including cancellation and a mid-step
    /// stop). Returns `None` after a hard [`step`](Self::step) error consumed
    /// the state; partial solver machinery cannot form an exact checkpoint.
    ///
    /// Moves the solver and state and copies the authoritative counters
    /// without requiring `Clone` or serialization. Extraction performs no
    /// evaluations, convergence checks, or observer/checkpoint callbacks.
    /// The problem, execution policy, and any recorded termination reason
    /// are dropped. Resume through [`Executor::resume_from_checkpoint`],
    /// which describes the requirements for exact continuation.
    ///
    /// # Example
    ///
    /// ```
    /// use basin::{CostFunction, Executor, NelderMead, State};
    /// # fn main() -> Result<(), std::convert::Infallible> {
    /// struct Sphere;
    /// impl CostFunction for Sphere {
    ///     type Param = Vec<f64>;
    ///     type Output = f64;
    ///     type Error = std::convert::Infallible;
    ///     fn cost(&self, x: &Vec<f64>) -> Result<f64, Self::Error> {
    ///         Ok(x.iter().map(|v| v * v).sum())
    ///     }
    /// }
    /// let mut stepper = Executor::from_start(
    ///     Sphere, NelderMead::new(), vec![2.0, 1.0],
    /// ).into_stepper()?;
    /// stepper.step()?;
    /// let checkpoint = stepper.into_checkpoint().unwrap();
    /// assert_eq!(checkpoint.state().iter(), 1);
    /// let result = Executor::resume_from_checkpoint(Sphere, checkpoint)
    ///     .max_iter(10)
    ///     .run()?;
    /// assert_eq!(result.iter(), 10);
    /// # Ok(())
    /// # }
    /// ```
    pub fn into_checkpoint(self) -> Option<ExactCheckpoint<So, S>> {
        Some(ExactCheckpoint::from_parts(
            self.solver,
            self.state?,
            *self.problem.counts(),
        ))
    }

    /// Consume the stepper and return the final state.
    ///
    /// # Panics
    ///
    /// Panics after [`step`](Self::step) returns `Err`, because the
    /// failing solver call consumed the state.
    pub fn into_state(self) -> S {
        self.state.expect("state slot is Some at drop")
    }
}

/// Single-iteration core, shared by [`Stepper::step`] (owned) and
/// [`run_loop_with_control`] (borrowed). Reads the current state via `state_slot`,
/// checks termination, and either returns `Stopped` (slot left
/// untouched) or hands the state to `solver.next_iter`, mirrors the
/// wrapper's counter delta (relative to `baseline`) onto the state,
/// increments the iteration counter, and puts the returned state back.
///
/// The `baseline` captures the wrapper count at the start of the
/// containing run so the state mirror always reflects *per-run* work:
/// for [`Stepper::step`]/[`Executor::run`] it is
/// [`EvalCounts::default`] (fresh wrapper), for nested
/// [`run_loop_with_control`] calls it is the wrapper count at run-loop entry.
///
/// The state slot must be `Some` on entry and is `Some` after an `Ok`
/// return. If [`Solver::next_iter`] returns `Err`, it has consumed the
/// previous state, so the slot remains empty and execution cannot resume
/// from this pair. The wrapper's charged counts remain available.
fn step_once<P, S, So>(
    problem: &mut Problem<P>,
    baseline: &EvalCounts,
    state_slot: &mut Option<S>,
    solver: &mut So,
    control: &mut RunControl<S>,
) -> Result<StepOutcome<S::Float>, So::Error>
where
    S: State + CountsMirror,
    So: Solver<P, S>,
{
    let state = state_slot.as_ref().expect("cannot step after a hard error");
    if let Some(termination) = control
        .check(state, &problem.counts().delta_since(baseline))
        .or_else(|| solver.check_convergence(problem, state))
    {
        return Ok(StepOutcome::Stopped(TerminationReport {
            termination,
            stage: TerminationStage::Boundary,
            iteration: state.iter(),
        }));
    }
    let SolverStep {
        state: mut next,
        completed,
        termination,
    } = solver.next_iter(problem, state_slot.take().unwrap())?;
    control.validate(&next);
    next.mirror(&problem.counts().delta_since(baseline));
    if completed {
        next.increment_iter();
    }
    next.update_best();
    let iteration = next.iter();
    *state_slot = Some(next);
    Ok(match termination {
        Some(termination) => StepOutcome::Stopped(TerminationReport {
            termination,
            stage: TerminationStage::Step { completed },
            iteration,
        }),
        None => StepOutcome::Continue,
    })
}

/// Drive a borrowed solver using execution budgets and application stops.
///
/// Convergence is configured on `solver`. Controls and convergence history
/// reset before initialization; evaluation counts are relative to run entry.
/// The inner state reflects only work performed in this call. Same-problem
/// inner solves share the outer wrapper and aggregate counts automatically.
/// For an adapter problem, use a separate wrapper and fold its counts into
/// the outer wrapper with [`EvalCounts::add`] and [`Problem::counts_mut`].
///
/// Execution controls run before solver convergence. Step completion and
/// termination are independent, as in [`SolverStep`]. This borrowed driver
/// has no executor-attached cancellation token. See [`RunControl`] for check
/// ordering and clock semantics.
pub fn run_loop_with_control<P, S, So>(
    problem: &mut Problem<P>,
    state: S,
    solver: &mut So,
    control: &mut RunControl<S>,
) -> Result<OptimizationResult<S>, So::Error>
where
    S: State + CountsMirror,
    So: Solver<P, S>,
{
    run_segment_with_control(problem, state, solver, control, false)
}

// Local-search chains retain algorithm machinery, but deliberately restart
// controls, convergence history, and per-segment accounting.
pub(crate) fn run_segment_with_control<P, S, So>(
    problem: &mut Problem<P>,
    mut state: S,
    solver: &mut So,
    control: &mut RunControl<S>,
    initialized: bool,
) -> Result<OptimizationResult<S>, So::Error>
where
    S: State + CountsMirror,
    So: Solver<P, S>,
{
    control.reset();
    solver.reset_convergence();
    let baseline = *problem.counts();
    // Reset best-so-far so the state always reflects per-run work,
    // matching the snapshot discipline `state.mirror` uses for eval
    // counters. This makes the same state safe to drive across
    // multiple `run_loop_with_control` calls (e.g. an outer solver re-driving an
    // inner) without best-so-far bleeding from one run into the next.
    state.reset_best();
    let mut state = if initialized {
        state
    } else {
        solver.init(problem, state)?
    };
    control.validate(&state);
    // Mirror init's work onto the state before any termination check.
    state.mirror(&problem.counts().delta_since(&baseline));
    state.update_best();
    let mut slot = Some(state);
    let reason = loop {
        match step_once(problem, &baseline, &mut slot, solver, control)? {
            StepOutcome::Continue => continue,
            StepOutcome::Stopped(reason) => break reason,
        }
    };
    Ok(OptimizationResult {
        state: slot.take().expect("state slot is Some on stop"),
        report: reason,
        counts: problem.counts().delta_since(&baseline),
    })
}

/// User-facing driver. Owns the problem, solver, initial state, and the
/// execution controls; [`run`](Self::run) drives the iteration
/// loop to completion. See the [module docs](self) for the canonical
/// ordering and [`into_stepper`](Self::into_stepper) for one-step-at-a-
/// time control.
///
/// # Examples
///
/// Minimize the 2-D sphere and read the outcome off the
/// [`OptimizationResult`]:
///
/// ```
/// use basin::{
///     FirstOrderState, CostFunction, Executor, Gradient, GradientDescent,
/// };
///
/// struct Sphere;
/// impl CostFunction for Sphere {
///     type Param = Vec<f64>;
///     type Output = f64;
///     type Error = std::convert::Infallible;
///     fn cost(&self, x: &Vec<f64>) -> Result<f64, std::convert::Infallible> {
///         Ok(x.iter().map(|xi| xi * xi).sum())
///     }
/// }
/// impl Gradient for Sphere {
///     type Gradient = Vec<f64>;
///     fn gradient(
///         &self,
///         x: &Vec<f64>,
///     ) -> Result<Vec<f64>, std::convert::Infallible> {
///         Ok(x.iter().map(|xi| 2.0 * xi).collect())
///     }
/// }
///
/// let result = Executor::new(
///     Sphere,
///     (GradientDescent::new(0.1)).with_absolute_gradient_tolerance(1e-9),
///     FirstOrderState::new(vec![3.0, -4.0]),
/// )
/// .max_iter(1_000)
/// .run()
/// .unwrap();
///
/// assert!(result.cost() < 1e-12);
/// ```
pub struct Executor<P, S: State, So> {
    problem: P,
    state: S,
    solver: So,
    control: RunControl<S>,
    observers: Vec<(Box<dyn ObserveSolver<S, So>>, ObserverMode)>,
    checkpoints: Vec<(Box<dyn CheckpointSink<So, S>>, ObserverMode)>,
    cancellation_token: Option<CancellationToken>,
    resume_counts: Option<EvalCounts>,
    skip_init: bool,
}

impl<P, S, So> Executor<P, S, So>
where
    S: State + CountsMirror,
    So: Solver<P, S>,
{
    /// Build an executor from a problem, solver, and initial state. The
    /// default `MaxIter` budget is 1000; override with
    /// [`max_iter`](Self::max_iter).
    pub fn new(problem: P, solver: So, state: S) -> Self {
        Self {
            problem,
            state,
            solver,
            control: RunControl::new(),
            observers: Vec::new(),
            checkpoints: Vec::new(),
            cancellation_token: None,
            resume_counts: None,
            skip_init: false,
        }
    }

    /// Build an executor that continues an exact state snapshot.
    ///
    /// Unlike [`new`](Self::new), this constructor preserves the state's
    /// best-so-far history and restores the problem wrapper's cumulative
    /// evaluation counters. The state must implement [`ExactResumeState`],
    /// which promises that it contains all solver evolution data required for
    /// an unchanged continuation. The solver's `init` implementation must be
    /// resume-idempotent.
    ///
    /// `max_iter` remains an absolute iteration limit: resuming a state at
    /// iteration 40 with `.max_iter(100)` performs at most 60 more iterations.
    /// Exact continuation also requires the same deterministic problem,
    /// solver configuration, scalar type, and code. Execution controls are
    /// configured anew; state-derived checks such as
    /// [`no_acceptance`](Self::no_acceptance) and zero-tolerance
    /// [`no_objective_improvement`](Self::no_objective_improvement) preserve history stored in the
    /// state, while controls with private clocks or anchors begin a new run.
    pub fn resume(problem: P, solver: So, state: S) -> Self
    where
        S: ExactResumeState,
    {
        let resume_counts = state.resume_counts();
        let mut executor = Self::new(problem, solver, state);
        executor.resume_counts = Some(resume_counts);
        executor
    }

    /// Build an executor that continues an exact solver/state checkpoint.
    ///
    /// Unlike [`new`](Self::new), this constructor restores the solver and
    /// problem wrapper's cumulative evaluation counters, preserves the
    /// state's best-so-far history, and does not call [`Solver::init`].
    ///
    /// `max_iter` remains an absolute iteration limit: resuming a state at
    /// iteration 40 with `.max_iter(100)` performs at most 60 more iterations.
    /// Exact continuation also requires the same deterministic problem,
    /// scalar type, backend behavior, and code. The checkpoint does not contain
    /// the problem, execution limits, application hooks, observers,
    /// cancellation token, or checkpoint sinks; configure that execution
    /// policy anew. State-derived checks such as
    /// [`no_acceptance`](Self::no_acceptance) and zero-tolerance
    /// [`no_objective_improvement`](Self::no_objective_improvement) preserve history stored in the
    /// state, while controls with private clocks or anchors begin a new run.
    /// Obtain an owned checkpoint with [`Stepper::into_checkpoint`] or
    /// [`OptimizationResultWithSolver::into_checkpoint`], without cloning
    /// or serializing the solver and state.
    pub fn resume_from_checkpoint(
        problem: P,
        checkpoint: ExactCheckpoint<So, S>,
    ) -> Self {
        let (solver, state, resume_counts) = checkpoint.into_parts();
        let mut executor = Self::new(problem, solver, state);
        executor.resume_counts = Some(resume_counts);
        executor.skip_init = true;
        executor
    }

    /// Build an executor seeding the solver's natural initial state at the
    /// starting point `x0`, instead of constructing the [`State`] by hand.
    ///
    /// `Executor::from_start(problem, solver, x0)` calls
    /// [`InitialState::seed`](crate::core::inner::InitialState::seed), so the
    /// caller never names the concrete state type: the common case reads
    /// `Executor::from_start(problem, TrustRegion::new(), x0).run()`. The
    /// seeded state uses the solver's natural default scale (identity inverse
    /// Hessian, default simplex edge, the solver's default trust radius, …).
    ///
    /// Use [`new`](Self::new) directly to supply a custom initial state (a
    /// pre-built simplex, a warm-started inverse Hessian, an anisotropic
    /// CMA-ES covariance). Solvers whose natural initialization needs more
    /// than a point, namely CMA-ES (step-size σ), the population GA, DE, or
    /// random search (they sample the box), and the bracketing scalar solvers
    /// (Brent, golden-section), deliberately do not implement
    /// [`InitialState`](crate::core::inner::InitialState), so calling
    /// `from_start` with one is a compile error pointing back to
    /// [`new`](Self::new).
    pub fn from_start<V>(problem: P, solver: So, x0: V) -> Self
    where
        So: crate::core::inner::InitialState<V, State = S>,
    {
        let state = solver.seed(&x0);
        Self::new(problem, solver, state)
    }

    /// Set the absolute iteration limit, replacing the previous limit.
    pub fn max_iter(mut self, n: u64) -> Self {
        self.control.max_iter = n;
        self
    }

    crate::core::run_control::control_methods!();

    /// Append an application stop evaluated before solver convergence.
    /// The closure sees an initialized state, including iteration zero.
    pub fn stop_when<C>(mut self, check: C) -> Self
    where
        C: FnMut(&S) -> Option<ApplicationStop> + 'static,
    {
        self.control = std::mem::take(&mut self.control).stop_when(check);
        self
    }

    /// Attach a cooperative cancellation token to this run.
    ///
    /// The executor checks the token after [`Solver::init`] and before every
    /// top-level iteration. A cancellation request returns
    /// `Ok(OptimizationResult)` with [`Termination::Cancelled`]; the
    /// state remains at the last fully completed iteration, including its
    /// best-so-far fields. An in-progress iteration or problem evaluation is
    /// allowed to finish before the token is observed.
    ///
    /// Calling this method again replaces the previously configured token.
    ///
    /// # Example
    ///
    /// ```
    /// use basin::{
    ///     FirstOrderState, CancellationToken, CostFunction, Executor, Gradient,
    ///     GradientDescent, Termination,
    /// };
    ///
    /// struct Sphere;
    /// impl CostFunction for Sphere {
    ///     type Param = Vec<f64>;
    ///     type Output = f64;
    ///     type Error = std::convert::Infallible;
    ///     fn cost(&self, x: &Vec<f64>) -> Result<f64, Self::Error> {
    ///         Ok(x.iter().map(|xi| xi * xi).sum())
    ///     }
    /// }
    /// impl Gradient for Sphere {
    ///     type Gradient = Vec<f64>;
    ///     fn gradient(&self, x: &Vec<f64>) -> Result<Vec<f64>, Self::Error> {
    ///         Ok(x.iter().map(|xi| 2.0 * xi).collect())
    ///     }
    /// }
    ///
    /// let token = CancellationToken::new();
    /// let cancel_handle = token.clone();
    /// cancel_handle.cancel(); // A UI callback or worker may hold this clone.
    ///
    /// let result = Executor::new(
    ///     Sphere,
    ///     GradientDescent::new(0.1),
    ///     FirstOrderState::new(vec![1.0, 1.0]),
    /// )
    /// .with_cancellation_token(token)
    /// .run()
    /// .unwrap();
    ///
    /// assert_eq!(result.report.termination, Termination::Cancelled);
    /// assert_eq!(result.iter(), 0);
    /// ```
    pub fn with_cancellation_token(mut self, token: CancellationToken) -> Self {
        self.cancellation_token = Some(token);
        self
    }

    /// Register an [`Observe`] hook. Observers fire in registration order,
    /// including when mixed with [`observe_solver`](Self::observe_solver) or
    /// [`observe_solver_with`](Self::observe_solver_with);
    /// `mode` gates [`Observe::observe_iter`] only;
    /// [`Observe::observe_init`] and [`Observe::observe_final`] always
    /// fire. See the [`observer`](crate::core::observer) module for the
    /// lifecycle.
    ///
    /// Observers cannot fail the run. Use
    /// [`stop_when`](Self::stop_when) for application stopping, or let
    /// an observer cancel a cloned [`CancellationToken`] for a clean
    /// user-requested stop.
    pub fn observe_with<O>(mut self, observer: O, mode: ObserverMode) -> Self
    where
        O: Observe<S> + 'static,
    {
        self.observers
            .push((Box::new(StateObserver(observer)), mode));
        self
    }

    /// Register an observer that borrows both progress and the solver.
    ///
    /// Hooks fire at the same boundaries as [`observe_with`](Self::observe_with),
    /// in registration order across both kinds of observer. `mode` gates only
    /// completed iterations; initialization and clean termination always fire.
    /// No solver or workspace cloning is required. See [`ObserveSolver`] for
    /// diagnostic semantics and the read-only contract.
    ///
    /// The observer is owned and must have `'static` captures; the solver and
    /// state may borrow data. Use [`observe_solver`](Self::observe_solver) for
    /// a closure with inferred state and solver types.
    pub fn observe_solver_with<O>(
        mut self,
        observer: O,
        mode: ObserverMode,
    ) -> Self
    where
        O: ObserveSolver<S, So> + 'static,
    {
        self.observers.push((Box::new(observer), mode));
        self
    }

    /// Observe progress and solver diagnostics with a closure.
    ///
    /// The callback receives shared state and solver references plus an
    /// [`ObservationEvent`]. It has the same lifecycle and registration order
    /// as [`observe_solver_with`](Self::observe_solver_with). Handle logging
    /// errors inside the callback; cancellation can request a clean stop.
    ///
    /// # Example
    ///
    /// Record annealing's temperature for the next proposal. The solver type,
    /// including its anonymous neighbor closure, is inferred automatically.
    ///
    /// ```
    /// use basin::{CostFunction, Executor, ObserverMode, SimulatedAnnealing,
    ///             State, TemperatureSchedule};
    /// use basin::core::rng::ChaCha8Rng;
    /// use std::convert::Infallible;
    ///
    /// struct Square;
    /// impl CostFunction for Square {
    ///     type Param = f64;
    ///     type Output = f64;
    ///     type Error = Infallible;
    ///     fn cost(&self, x: &f64) -> Result<f64, Infallible> { Ok(x * x) }
    /// }
    /// let solver = SimulatedAnnealing::new(
    ///     |x: &f64, _: f64, _: &mut ChaCha8Rng| x - 0.25,
    ///     8.0, TemperatureSchedule::geometric(0.5), 42,
    /// );
    /// Executor::from_start(Square, solver, 2.0)
    ///     .max_iter(3)
    ///     .observe_solver(
    ///         |state, solver, event| {
    ///             println!("{event:?}: {} {}", state.iter(), solver.temperature());
    ///         },
    ///         ObserverMode::Always,
    ///     )
    ///     .run()?;
    /// # Ok::<(), Infallible>(())
    /// ```
    pub fn observe_solver<C>(self, callback: C, mode: ObserverMode) -> Self
    where
        C: FnMut(&S, &So, ObservationEvent<'_, S::Float>) + 'static,
    {
        self.observe_solver_with(SolverCallback(callback), mode)
    }

    /// Register a solver-aware checkpoint destination.
    ///
    /// `mode` controls saves after completed iterations. Every sink also saves
    /// once when the run stops cleanly, regardless of the mode. Checkpoint
    /// sinks are separate from state-only [`Observe`] implementations because
    /// exact continuation requires both the solver and state.
    ///
    /// Sink failures cannot change the optimization result; implementations
    /// must record or report their own failures. Calling this method again
    /// adds another sink.
    pub fn checkpoint_with<C>(
        mut self,
        checkpoint: C,
        mode: ObserverMode,
    ) -> Self
    where
        C: CheckpointSink<So, S> + 'static,
    {
        self.checkpoints.push((Box::new(checkpoint), mode));
        self
    }

    /// Convert the executor into a [`Stepper`] for one-iteration-at-a-time
    /// control. On a fresh executor or a legacy state-resume executor,
    /// `solver.init` runs here so the returned stepper sits at iter 0 with a
    /// complete state. A solver-aware checkpoint resume skips `init` and uses
    /// its restored iteration boundary directly. All registered observers'
    /// `observe_init` hooks fire in either case. Cancellation is first checked
    /// by the returned stepper's initial [`step`](Stepper::step).
    ///
    /// Returns `Err` when [`Solver::init`] does (e.g. the problem's
    /// initial cost/gradient evaluation `Err`-ed). Observers do *not* fire
    /// on that error path.
    pub fn into_stepper(self) -> Result<Stepper<P, S, So>, So::Error> {
        let Self {
            problem,
            mut state,
            mut solver,
            control,
            mut observers,
            checkpoints,
            cancellation_token,
            resume_counts,
            skip_init,
        } = self;
        let mut problem = Problem::new(problem);
        if let Some(counts) = resume_counts {
            *problem.counts_mut() = counts;
        } else {
            // Fresh top-level wrapper: reset best-so-far so it tracks
            // this run's iterates only, matching the `state.mirror`
            // per-run snapshot discipline.
            state.reset_best();
        }
        let state = if skip_init {
            control.validate(&state);
            state
        } else {
            solver.reset_convergence();
            let mut state = solver.init(&mut problem, state)?;
            control.validate(&state);
            // Mirror init's work onto the state before any termination
            // check. Baseline is zero: this is a fresh top-level wrapper.
            state.mirror(problem.counts());
            state.update_best();
            state
        };
        for (observer, _mode) in observers.iter_mut() {
            observer.observe_init(&state, &solver);
        }
        Ok(Stepper {
            problem,
            state: Some(state),
            solver,
            control,
            observers,
            checkpoints,
            cancellation_token,
            finished: None,
        })
    }

    /// Drive the iteration loop to completion and return the
    /// [`OptimizationResult`].
    /// Use [`run_with_solver`](Self::run_with_solver) to retain the final
    /// solver as well.
    ///
    /// Returns `Err` when the underlying problem returns `Err` from any
    /// cost/gradient/residual/Jacobian/Hessian call (the
    /// `P::Error`-flavored hard-abort path; see the
    /// [`problem`](crate::core::problem) module docs).
    pub fn run(self) -> Result<OptimizationResult<S>, So::Error> {
        self.into_stepper()?.run_to_end()
    }

    /// Drive the iteration loop to completion, retaining the final solver,
    /// state, raw evaluation counters, and termination reason by ownership.
    ///
    /// Follows the same initialization, stopping, and callback lifecycle as
    /// [`run`](Self::run). Neither the solver nor the state needs to implement
    /// `Clone` or serialization. Errors propagate with the same type and
    /// without a partial result.
    ///
    /// # Example
    ///
    /// ```
    /// use basin::{CostFunction, Executor, NelderMead};
    /// # fn main() -> Result<(), std::convert::Infallible> {
    /// struct Sphere;
    /// impl CostFunction for Sphere {
    ///     type Param = Vec<f64>;
    ///     type Output = f64;
    ///     type Error = std::convert::Infallible;
    ///     fn cost(&self, x: &Vec<f64>) -> Result<f64, Self::Error> {
    ///         Ok(x.iter().map(|v| v * v).sum())
    ///     }
    /// }
    /// let result = Executor::from_start(
    ///     Sphere, NelderMead::new(), vec![2.0, 1.0],
    /// ).max_iter(3).run_with_solver()?;
    /// assert_eq!(result.iter(), 3);
    /// let counts = result.counts;
    /// let checkpoint = result.into_checkpoint();
    /// let continued = Executor::resume_from_checkpoint(Sphere, checkpoint)
    ///     .max_iter(10)
    ///     .run_with_solver()?;
    /// assert!(continued.counts.cost_evals > counts.cost_evals);
    /// # Ok(())
    /// # }
    /// ```
    pub fn run_with_solver(
        self,
    ) -> Result<OptimizationResultWithSolver<S, So>, So::Error> {
        self.into_stepper()?.run_to_end_with_solver()
    }
}
