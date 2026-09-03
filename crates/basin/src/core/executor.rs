//! Iteration driver. The high-level entry point is [`Executor`];
//! [`Stepper`] exposes one-iteration-at-a-time control, and [`run_loop`]
//! is the borrowed-problem variant used by composed solvers.
//!
//! # Canonical iteration ordering
//!
//! [`Executor::run`] (and the equivalent [`Stepper`]/[`run_loop`]
//! paths) drive the solver through this exact sequence, and every
//! contract elsewhere in the framework cross-links here:
//!
//! 1. [`Solver::init`] is called **once**, on the initial state. The
//!    returned state is what iter-0 sees.
//! 2. Then, repeatedly, before each [`Solver::next_iter`] call
//!    (including the first):
//!    1. An executor-attached [`CancellationToken`] is checked, when
//!       configured. Cancellation stops the run with
//!       [`TerminationReason::Cancelled`]. The borrowed [`run_loop`] path
//!       has no attached token and skips this step.
//!    2. The built-in [`MaxIter`](crate::core::termination::MaxIter)
//!       limit is checked against [`State::iter`]. If
//!       `state.iter() >= max_iter`, the run stops with
//!       [`TerminationReason::MaxIter`].
//!    3. Each registered [`TerminationCriterion`] is checked **in
//!       insertion order**. The **first to return `Some(reason)` halts
//!       the run**, and later criteria do not run that iteration.
//!    4. The solver's own [`Solver::terminate`] hook is checked.
//!       `Some(_)` halts the run.
//! 3. If nothing fired, [`Solver::next_iter`] is called. It may itself
//!    report a mid-iter termination via its return tuple; in that case
//!    the iteration counter is **not** incremented, so the final
//!    [`State::iter`] reflects the last *fully completed* iteration.
//! 4. Otherwise the iteration counter is incremented and we go back to
//!    step 2.
//!
//! Because checks happen *before* iter 0, an already-optimal initial
//! point exits immediately with the corresponding reason rather than
//! taking one redundant step.

use crate::core::observer::{Observe, ObserverMode};
use crate::core::problem::{EvalCounts, Problem};
use crate::core::solver::Solver;
use crate::core::state::{CountsMirror, State};
use crate::core::termination::{TerminationCriterion, TerminationReason};
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
/// Owns the final solver state plus the reason the executor stopped.
/// Delegates `param()`/`cost()`/`iter()` to the underlying state so
/// callers don't need to import `State` for the common reads.
pub struct OptimizationResult<S> {
    /// Final solver state at termination.
    pub state: S,
    /// Why the executor stopped.
    pub reason: TerminationReason,
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

/// Outcome of a single [`Stepper::step`] call.
///
/// `Stopped` carries the same [`TerminationReason`] the executor would
/// have returned. After `Stopped` is returned once, subsequent calls to
/// `step` keep returning the same `Stopped(reason)` so callers don't
/// have to track whether they're done.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StepOutcome {
    /// The step completed without triggering termination.
    Continue,
    /// Termination fired with the given reason. Subsequent
    /// [`Stepper::step`] calls keep returning this same outcome.
    Stopped(TerminationReason),
}

/// Drive a solver one iteration at a time.
///
/// Owns the problem, state, solver and termination criteria, runs
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
/// let mut stepper = Executor::new(problem, solver, state)
///     .max_iter(100)
///     .terminate_on(GradientTolerance(1e-6))
///     .into_stepper();
///
/// let reason = loop {
///     match stepper.step() {
///         StepOutcome::Continue => { /* observe `stepper.state()` */ }
///         StepOutcome::Stopped(reason) => break reason,
///     }
/// };
/// ```
pub struct Stepper<P, S, So> {
    problem: Problem<P>,
    // `Option<S>` because `Solver::next_iter` consumes the state by
    // value. Take it out, hand it to the solver, put the returned state
    // back. The slot is `Some` whenever a caller can observe it (between
    // `step` calls and at construction/drop), so `state()` and
    // `into_state` can unwrap without checks.
    state: Option<S>,
    solver: So,
    criteria: Vec<Box<dyn TerminationCriterion<S>>>,
    observers: Vec<(Box<dyn Observe<S>>, ObserverMode)>,
    max_iter: u64,
    cancellation_token: Option<CancellationToken>,
    finished: Option<TerminationReason>,
}

impl<P, S, So> Stepper<P, S, So>
where
    S: State + CountsMirror,
    So: Solver<P, S>,
{
    /// Read-only access to the current state, between steps.
    pub fn state(&self) -> &S {
        self.state
            .as_ref()
            .expect("state slot is Some between steps")
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
    pub fn finished(&self) -> Option<&TerminationReason> {
        self.finished.as_ref()
    }

    /// Total iterations that have completed so far. Convenience read
    /// equivalent to `self.state().iter()`.
    pub fn iter(&self) -> u64 {
        self.state().iter()
    }

    /// Advance one iteration. Once a `Stopped` outcome has been returned
    /// the stepper is sticky: subsequent calls keep returning the same
    /// `Stopped(reason)` without touching the state or solver.
    ///
    /// Registered observers fire here:
    /// [`observe_iter`](Observe::observe_iter) on
    /// [`StepOutcome::Continue`], gated by each observer's
    /// [`ObserverMode`]; [`observe_final`](Observe::observe_final) once
    /// when this call first returns [`StepOutcome::Stopped`]. See the
    /// [`observer`](crate::core::observer) module for the lifecycle.
    ///
    /// Returns `Err` when the underlying problem returns `Err` from any
    /// cost/gradient/residual/Jacobian/Hessian call during the
    /// step. The stepper is *not* made sticky on `Err`: the typical
    /// downstream pattern is to surface the error and drop the stepper,
    /// but callers may inspect [`state`](Self::state) and try again.
    /// Observers do *not* fire on the `Err` path (the state has been
    /// consumed by the failing call).
    pub fn step(&mut self) -> Result<StepOutcome, So::Error> {
        if let Some(reason) = self.finished {
            return Ok(StepOutcome::Stopped(reason));
        }
        let outcome = if self
            .cancellation_token
            .as_ref()
            .is_some_and(CancellationToken::is_cancelled)
        {
            StepOutcome::Stopped(TerminationReason::Cancelled)
        } else {
            step_once(
                &mut self.problem,
                &EvalCounts::default(),
                &mut self.state,
                &mut self.solver,
                &mut self.criteria,
                self.max_iter,
            )?
        };
        match outcome {
            StepOutcome::Continue => {
                let state = self
                    .state
                    .as_ref()
                    .expect("state slot is Some after Continue");
                let iter = state.iter();
                // `update_best` set `best_iter == iter` iff this iteration
                // strictly improved the incumbent; that is exactly the
                // `NewBest` firing condition.
                let is_new_best = state.best_iter() == iter;
                for (observer, mode) in self.observers.iter_mut() {
                    if mode.fires_on(iter, is_new_best) {
                        observer.observe_iter(state);
                    }
                }
            }
            StepOutcome::Stopped(reason) => {
                self.finished = Some(reason);
                let state =
                    self.state.as_ref().expect("state slot is Some on Stopped");
                for (observer, _mode) in self.observers.iter_mut() {
                    observer.observe_final(state, &reason);
                }
            }
        }
        Ok(outcome)
    }

    /// Drive [`step`](Self::step) to completion and return an
    /// [`OptimizationResult`].
    pub fn run_to_end(mut self) -> Result<OptimizationResult<S>, So::Error> {
        loop {
            if let StepOutcome::Stopped(reason) = self.step()? {
                return Ok(OptimizationResult {
                    state: self
                        .state
                        .take()
                        .expect("state slot is Some on stop"),
                    reason,
                });
            }
        }
    }

    /// Consume the stepper and return the final state.
    pub fn into_state(self) -> S {
        self.state.expect("state slot is Some at drop")
    }
}

/// Single-iteration core, shared by [`Stepper::step`] (owned) and
/// [`run_loop`] (borrowed). Reads the current state via `state_slot`,
/// checks termination, and either returns `Stopped` (slot left
/// untouched) or hands the state to `solver.next_iter`, mirrors the
/// wrapper's counter delta (relative to `baseline`) onto the state,
/// increments the iteration counter, and puts the returned state back.
///
/// The `baseline` captures the wrapper count at the start of the
/// containing run so the state mirror always reflects *per-run* work:
/// for [`Stepper::step`]/[`Executor::run`] it is
/// [`EvalCounts::default`] (fresh wrapper), for nested
/// [`run_loop`] calls it is the wrapper count at run-loop entry.
///
/// Returns `Err` when [`Solver::next_iter`] does. The state slot is
/// untouched on `Err` (the previous iterate is still readable).
///
/// Invariant: `state_slot` is `Some` on entry and `Some` on return
/// (including on the `Err` path).
fn step_once<P, S, So>(
    problem: &mut Problem<P>,
    baseline: &EvalCounts,
    state_slot: &mut Option<S>,
    solver: &mut So,
    criteria: &mut [Box<dyn TerminationCriterion<S>>],
    max_iter: u64,
) -> Result<StepOutcome, So::Error>
where
    S: State + CountsMirror,
    So: Solver<P, S>,
{
    {
        let state = state_slot
            .as_ref()
            .expect("step_once called with empty state slot");
        if state.iter() >= max_iter {
            return Ok(StepOutcome::Stopped(TerminationReason::MaxIter));
        }
        for criterion in criteria.iter_mut() {
            if let Some(reason) = criterion.check(state) {
                return Ok(StepOutcome::Stopped(reason));
            }
        }
        if let Some(reason) = solver.terminate(state) {
            return Ok(StepOutcome::Stopped(reason));
        }
    }
    let prev = state_slot.take().unwrap();
    let next_iter_result = solver.next_iter(problem, prev);
    let (mut next, mid_iter_reason) = match next_iter_result {
        Ok(t) => t,
        Err(e) => {
            // step_once owes the caller the `state_slot is Some on return`
            // invariant even on the error path; we lost `prev` to
            // `next_iter` (which took it by value), so there's nothing to
            // put back. Mid-iter hard-aborts therefore leave the slot
            // empty and the stepper consumes itself; this is the
            // intentional shape: typed Err is terminal, the typical
            // caller bubbles it out and drops the stepper. The wrapper's
            // own counts are still authoritative on the Err path; see
            // [`Stepper::counts`].
            return Err(e);
        }
    };
    next.mirror(&problem.counts().delta_since(baseline));
    if let Some(reason) = mid_iter_reason {
        // Refresh best-so-far from the mid-iter state too: the solver
        // may have produced its best iterate on the same step that
        // bailed.
        next.update_best();
        *state_slot = Some(next);
        return Ok(StepOutcome::Stopped(reason));
    }
    next.increment_iter();
    next.update_best();
    *state_slot = Some(next);
    Ok(StepOutcome::Continue)
}

/// Drive a solver to completion against a shared [`Problem`] wrapper.
///
/// `Executor` is a thin owning wrapper over this. Composed solvers
/// (e.g. CG inside CMA, NM inside DE) call `run_loop` directly so the
/// inner solver shares the outer's wrapper: inner cost and gradient
/// calls bump the same [`EvalCounts`] as outer calls, so the eval
/// aggregation contract (`CONTRIBUTING.md` "Solver composition" rule 1) is
/// satisfied automatically for same-problem inners. For composed
/// solvers driving an inner against an **adapter problem** (e.g.
/// [`LogBarrier`](crate::core::barrier::LogBarrier)), construct a
/// fresh `Problem::new(adapter)`, pass `&mut` into `run_loop`, then
/// fold the inner wrapper's [`EvalCounts`] back into the outer's via
/// [`EvalCounts::add`] on [`Problem::counts_mut`].
///
/// The inner state's [`State::cost_evals`] (mirrored via
/// [`CountsMirror`]) reflects only *per-run* work: `run_loop` takes
/// a baseline snapshot of [`Problem::counts`] at entry, and the state
/// mirror computes the delta against that. Nested `run_loop` calls
/// against the same wrapper therefore see clean per-call counters.
///
/// Apart from executor-attached cancellation (which is top-level only),
/// semantics match `Executor::run`: each criterion is
/// [`reset`](crate::core::termination::TerminationCriterion::reset) at
/// entry, so a criteria vector reused across calls (as an
/// [`InnerExecutor`](crate::core::inner::InnerExecutor) does) sees fresh
/// per-run state. Then `init` is called once, then on each iteration
/// framework `criteria` are checked in insertion order before
/// the solver's own `terminate` hook, before stepping. `max_iter` is
/// checked against `state.iter()` and exits with `TerminationReason::MaxIter`.
/// `next_iter` may also report a mid-iter termination via its return tuple;
/// in that case the iteration counter is left untouched so the final
/// `state.iter()` still reflects the last fully completed iteration.
pub fn run_loop<P, S, So>(
    problem: &mut Problem<P>,
    mut state: S,
    solver: &mut So,
    criteria: &mut [Box<dyn TerminationCriterion<S>>],
    max_iter: u64,
) -> Result<OptimizationResult<S>, So::Error>
where
    S: State + CountsMirror,
    So: Solver<P, S>,
{
    let baseline = *problem.counts();
    // Reset each criterion's internal per-run state before the run, so a
    // criteria vector reused across `run_loop` calls (e.g. an
    // `InnerExecutor` driven once per outer iter) sees fresh state each
    // call. Stateful criteria (`MaxTime`, `RelativeGradientTolerance`,
    // `NoImprovement`) would otherwise carry state across runs and
    // misbehave; the default `reset` is a no-op for stateless ones.
    for criterion in criteria.iter_mut() {
        criterion.reset();
    }
    // Reset best-so-far so the state always reflects per-run work,
    // matching the snapshot discipline `state.mirror` uses for eval
    // counters. This makes the same state safe to drive across
    // multiple `run_loop` calls (e.g. an outer solver re-driving an
    // inner) without best-so-far bleeding from one run into the next.
    state.reset_best();
    let mut state = solver.init(problem, state)?;
    // Mirror init's work onto the state before any termination check.
    state.mirror(&problem.counts().delta_since(&baseline));
    state.update_best();
    let mut slot = Some(state);
    let reason = loop {
        match step_once(
            problem, &baseline, &mut slot, solver, criteria, max_iter,
        )? {
            StepOutcome::Continue => continue,
            StepOutcome::Stopped(reason) => break reason,
        }
    };
    Ok(OptimizationResult {
        state: slot.take().expect("state slot is Some on stop"),
        reason,
    })
}

/// User-facing driver. Owns the problem, solver, initial state, and the
/// list of termination criteria; [`run`](Self::run) drives the iteration
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
/// use basin::{BasicState, CostFunction, Executor, Gradient, GradientDescent, GradientTolerance};
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
///     fn gradient(&self, x: &Vec<f64>) -> Result<Vec<f64>, std::convert::Infallible> {
///         Ok(x.iter().map(|xi| 2.0 * xi).collect())
///     }
/// }
///
/// let result = Executor::new(Sphere, GradientDescent::new(0.1), BasicState::new(vec![3.0, -4.0]))
///     .max_iter(1_000)
///     .terminate_on(GradientTolerance(1e-9))
///     .run()
///     .unwrap();
///
/// assert!(result.cost() < 1e-12);
/// ```
pub struct Executor<P, S, So> {
    problem: P,
    state: S,
    solver: So,
    max_iter: u64,
    criteria: Vec<Box<dyn TerminationCriterion<S>>>,
    observers: Vec<(Box<dyn Observe<S>>, ObserverMode)>,
    cancellation_token: Option<CancellationToken>,
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
            max_iter: 1000,
            criteria: Vec::new(),
            observers: Vec::new(),
            cancellation_token: None,
        }
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

    /// Convenience setter for the default `MaxIter` criterion. Equivalent
    /// effect to `terminate_on(MaxIter(n))` but mutates a dedicated field
    /// so subsequent calls replace rather than stack.
    pub fn max_iter(mut self, n: u64) -> Self {
        self.max_iter = n;
        self
    }

    /// Add a termination criterion. Criteria are checked in insertion
    /// order before each iteration (and before iter 0); the first to
    /// return `Some(_)` stops the run. See the [module docs](self) for
    /// the full per-iteration ordering.
    pub fn terminate_on<C>(mut self, criterion: C) -> Self
    where
        C: TerminationCriterion<S> + 'static,
    {
        self.criteria.push(Box::new(criterion));
        self
    }

    /// Attach a cooperative cancellation token to this run.
    ///
    /// The executor checks the token after [`Solver::init`] and before every
    /// top-level iteration. A cancellation request returns
    /// `Ok(OptimizationResult)` with [`TerminationReason::Cancelled`]; the
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
    ///     BasicState, CancellationToken, CostFunction, Executor, Gradient,
    ///     GradientDescent, TerminationReason,
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
    ///     BasicState::new(vec![1.0, 1.0]),
    /// )
    /// .with_cancellation_token(token)
    /// .run()
    /// .unwrap();
    ///
    /// assert_eq!(result.reason, TerminationReason::Cancelled);
    /// assert_eq!(result.iter(), 0);
    /// ```
    pub fn with_cancellation_token(mut self, token: CancellationToken) -> Self {
        self.cancellation_token = Some(token);
        self
    }

    /// Register an [`Observe`] hook. Observers fire in registration order;
    /// `mode` gates [`Observe::observe_iter`] only;
    /// [`Observe::observe_init`] and [`Observe::observe_final`] always
    /// fire. See the [`observer`](crate::core::observer) module for the
    /// lifecycle.
    ///
    /// Observers cannot fail the run. Use
    /// [`terminate_on`](Self::terminate_on) for state-based stopping, or let
    /// an observer cancel a cloned [`CancellationToken`] for a clean
    /// user-requested stop.
    pub fn observe_with<O>(mut self, observer: O, mode: ObserverMode) -> Self
    where
        O: Observe<S> + 'static,
    {
        self.observers.push((Box::new(observer), mode));
        self
    }

    /// Convert the executor into a [`Stepper`] for one-iteration-at-a-time
    /// control. `solver.init` runs here so the returned stepper sits at
    /// iter 0 with a complete state; all registered observers' `observe_init`
    /// fire here too. Cancellation is first checked by the returned stepper's
    /// initial [`step`](Stepper::step), after initialization has completed.
    ///
    /// Returns `Err` when [`Solver::init`] does (e.g. the problem's
    /// initial cost/gradient evaluation `Err`-ed). Observers do *not* fire
    /// on that error path.
    pub fn into_stepper(self) -> Result<Stepper<P, S, So>, So::Error> {
        let Self {
            problem,
            mut state,
            mut solver,
            max_iter,
            criteria,
            mut observers,
            cancellation_token,
        } = self;
        let mut problem = Problem::new(problem);
        // Fresh top-level wrapper: reset best-so-far so it tracks
        // this run's iterates only, matching the `state.mirror`
        // per-run snapshot discipline.
        state.reset_best();
        let mut state = solver.init(&mut problem, state)?;
        // Mirror init's work onto the state before any termination
        // check. Baseline is zero: this is a fresh top-level wrapper.
        state.mirror(problem.counts());
        state.update_best();
        for (observer, _mode) in observers.iter_mut() {
            observer.observe_init(&state);
        }
        Ok(Stepper {
            problem,
            state: Some(state),
            solver,
            criteria,
            observers,
            max_iter,
            cancellation_token,
            finished: None,
        })
    }

    /// Drive the iteration loop to completion and return the
    /// [`OptimizationResult`].
    ///
    /// Returns `Err` when the underlying problem returns `Err` from any
    /// cost/gradient/residual/Jacobian/Hessian call (the
    /// `P::Error`-flavored hard-abort path; see the
    /// [`problem`](crate::core::problem) module docs).
    pub fn run(self) -> Result<OptimizationResult<S>, So::Error> {
        self.into_stepper()?.run_to_end()
    }
}
