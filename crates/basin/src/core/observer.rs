//! Read-only side effects fired around the iteration loop.
//!
//! An [`Observe`] implementation watches the run as it happens (logging,
//! progress reporting, recording a trajectory, streaming iterates to a UI)
//! without influencing it. Observers do **not** decide whether to stop; that
//! is the job of [`stop_when`](crate::Executor::stop_when).
//! The two extension points sit side-by-side on
//! [`Executor`](crate::core::executor::Executor):
//!
//! - [`stop_when`](crate::Executor::stop_when):
//!   returns `Option<ApplicationStop>`; framework consumes the result and
//!   stops the run if `Some`.
//! - [`Observe`]: returns `()`; the executor ignores any side effects on the
//!   optimization itself. Failures must be handled inside the observer:
//!   the trait returns no logging error for the executor to propagate.
//!   Panics propagate normally. An observer may request a clean stop by
//!   cancelling a cloned
//!   [`CancellationToken`](crate::core::executor::CancellationToken); the
//!   executor observes it before the next iteration.
//!
//! Like stopping hooks, observers bind on the minimum
//! [`State`](crate::core::state::State) shape they need (tenet 3): a logger
//! that just wants `iter`/`cost` impls `Observe<S: State>`; a gradient-norm
//! observer impls `Observe<S: GradientState>` and is rejected at compile time
//! when attached to a derivative-free run.
//!
//! # Solver diagnostics
//!
//! [`ObserveSolver<S, So>`](ObserveSolver) also borrows the concrete solver.
//! Register it with [`Executor::observe_solver_with`](crate::Executor::observe_solver_with),
//! or use [`Executor::observe_solver`](crate::Executor::observe_solver) for a
//! closure receiving `(&state, &solver, ObservationEvent)`. State observers
//! and solver observers share the lifecycle below, obey the same modes, and
//! fire in registration order, including when mixed.
//!
//! The solver retains ownership of models and workspace. Observation borrows
//! them without cloning or evaluating the problem. Getters retain their
//! individual availability and time semantics: annealing's
//! [`temperature`](crate::SimulatedAnnealing::temperature) describes the next
//! proposal, while SLSQP's [`stationarity`](crate::Slsqp::stationarity) and
//! [`multipliers`](crate::Slsqp::equality_multipliers) are optional and depend
//! on its current model. Callbacks must not evaluate the problem or mutate
//! solver machinery through interior mutability.
//!
//! Observations belong to the registered executor; they do not include every
//! iteration of an outer solver's inner solves. For applications that already
//! drive a [`Stepper`](crate::Stepper), [`Stepper::solver`](crate::Stepper::solver)
//! exposes the same shared borrow between steps. Retain the final solver with
//! [`Executor::run_with_solver`](crate::Executor::run_with_solver) when only
//! final diagnostics are needed.
//!
//! # Lifecycle
//!
//! Three hooks, called in this order during a run:
//!
//! 1. [`observe_init`](Observe::observe_init) fires once after
//!    [`Solver::init`](crate::core::solver::Solver::init) returns and the
//!    state's counter mirror is refreshed, before the first termination
//!    check. A fresh run shows `iter() == 0`. Exact continuation skips solver
//!    initialization and observes the restored iteration and solver instead.
//! 2. [`observe_iter`](Observe::observe_iter) fires after every successfully
//!    completed iteration, after the iteration counter is incremented. On
//!    the first call of a fresh run the state shows `iter() == 1`. Gated by
//!    [`ObserverMode`].
//! 3. [`observe_final`](Observe::observe_final) fires once when the run
//!    stops cleanly with a [`TerminationReport`]. The state shows the iter
//!    count of the last fully-completed iteration.
//!
//! # Edge cases
//!
//! - **Stopping steps.** A step with `completed: false` leaves the iteration
//!   count unchanged and fires only the final hook. A stopping step with
//!   `completed: true` increments the count and fires the gated iteration hook
//!   followed by the final hook. Both receive the same coherent progress.
//! - **Hard error from the problem.** If `next_iter` returns `Err(_)`, the
//!   state is consumed by the failing call and there's nothing to observe;
//!   [`observe_final`](Observe::observe_final) does *not* fire and the
//!   error propagates out of
//!   [`Executor::run`](crate::core::executor::Executor::run) /
//!   [`Stepper::step`](crate::core::executor::Stepper::step). Observers
//!   that need to react to hard aborts should track liveness themselves.
//! - **Reading the reason from inside observers.** The argument to
//!   [`observe_final`](Observe::observe_final) is the
//!   [`TerminationReport`]; state types do not carry it.
//!
//! # Starter observers
//!
//! Two zero-dependency observers ship in core, both binding on the minimum
//! [`State`](crate::core::state::State) shape:
//!
//! - [`Report`]: prints a one-line progress report (iter / cost / best) to
//!   stderr on each fire.
//! - [`History`]: records the `(iter, cost, best_cost)` trajectory for later
//!   plotting or analysis.
//!
//! Heavier integrations (tracing, slog, a TUI) live in satellite crates; a
//! `serde`-gated `CheckpointWriter` (non-wasm) snapshots the state to disk for
//! warm-starting a later run. Solver-aware exact checkpoints live in the
//! sibling [`checkpoint`](crate::core::checkpoint) module.

use crate::core::termination::TerminationReport;

mod adapters;
mod history;
mod report;

pub(crate) use adapters::{SolverCallback, StateObserver};
pub use history::History;
pub use report::Report;

#[cfg(all(feature = "serde", not(target_arch = "wasm32")))]
mod checkpoint;
#[cfg(all(feature = "serde", not(target_arch = "wasm32")))]
pub use checkpoint::{CheckpointWriter, read_checkpoint};

/// Hooks fired around the iteration loop. See the
/// [module docs](self) for lifecycle and edge cases.
///
/// All three methods default to no-ops, so an implementor fills in only what
/// they need. Bind on the minimum state shape required (`S: State`,
/// `S: GradientState`, `S: SimplexState`, …) so a mismatch with the solver is
/// a compile error rather than a runtime no-op.
pub trait Observe<S: crate::State> {
    /// Fired once before the first iteration, after
    /// [`Solver::init`](crate::core::solver::Solver::init) has run and the
    /// state's counter mirror has been refreshed. A fresh run has iteration
    /// zero; exact continuation observes the restored boundary instead.
    ///
    /// Always fires regardless of the observer's [`ObserverMode`]; modes
    /// gate iteration callbacks only.
    fn observe_init(&mut self, _state: &S) {}

    /// Fired after each successfully completed iteration, after
    /// [`State::increment_iter`](crate::core::state::State::increment_iter)
    /// has run, so `state.iter()` returns the count of the iteration that
    /// just finished (1 on the first call of a fresh run, …).
    ///
    /// Gated by [`ObserverMode`]: `Never` skips, `Always` fires every iter,
    /// `Every(n)` fires when `state.iter() % n == 0`.
    fn observe_iter(&mut self, _state: &S) {}

    /// Fired once when the run stops with a clean
    /// [`TerminationReport`]. `state` is the final iterate; `reason` is what
    /// halted the run.
    ///
    /// Does *not* fire when
    /// [`Solver::next_iter`](crate::core::solver::Solver::next_iter)
    /// returns `Err(_)`: in that case the state has been consumed and
    /// there is nothing to observe.
    ///
    /// Always fires regardless of the observer's [`ObserverMode`].
    fn observe_final(
        &mut self,
        _state: &S,
        _reason: &TerminationReport<S::Float>,
    ) {
    }
}

/// Read-only observation of progress and solver-owned diagnostics.
///
/// Register an implementation with
/// [`Executor::observe_solver_with`](crate::Executor::observe_solver_with).
/// For a closure, use [`Executor::observe_solver`](crate::Executor::observe_solver).
/// Bind only on the state and solver capabilities the observer needs; neither
/// argument must implement `Clone` or serialization, and both may borrow data.
/// References passed to a callback cannot be retained beyond that callback.
///
/// All hooks default to no-ops. Their ordering, modes, exact continuation, and
/// error behavior match [`Observe`]; see the [module documentation](self).
/// Observers must use existing diagnostic getters without evaluating the
/// problem or mutating solver machinery through interior mutability. Handle
/// logging errors internally. Panics propagate normally. A cloned
/// [`CancellationToken`](crate::CancellationToken) can request a clean stop.
///
/// # Example
///
/// A reusable logger can bind on the minimum state shape while accepting any
/// annealing neighbor and RNG type:
///
/// ```
/// use basin::{ObserveSolver, SimulatedAnnealing, State};
///
/// struct TemperatureLogger;
/// impl<S, N, R> ObserveSolver<S, SimulatedAnnealing<N, f64, R>>
///     for TemperatureLogger
/// where
///     S: State,
/// {
///     fn observe_iter(&mut self, state: &S, solver: &SimulatedAnnealing<N, f64, R>) {
///         println!("{}: next temperature = {}", state.iter(), solver.temperature());
///     }
/// }
/// ```
pub trait ObserveSolver<S: crate::State, So> {
    /// Fired after successful initialization or exact restoration, before the
    /// first termination check. Counts and incumbent publication are complete.
    /// Always fires regardless of [`ObserverMode`].
    fn observe_init(&mut self, _state: &S, _solver: &So) {}

    /// Fired after a completed iteration, after counts, iteration number, and
    /// incumbent publication are updated. Gated by [`ObserverMode`].
    fn observe_iter(&mut self, _state: &S, _solver: &So) {}

    /// Fired once on a clean stop, regardless of [`ObserverMode`].
    ///
    /// A partial-step stop refreshes counts without incrementing the iteration
    /// counter. The solver may contain diagnostics updated by the final step
    /// or convergence check. Hard errors do not fire this hook.
    fn observe_final(
        &mut self,
        _state: &S,
        _solver: &So,
        _reason: &TerminationReport<S::Float>,
    ) {
    }
}

/// The boundary passed to an [`Executor::observe_solver`](crate::Executor::observe_solver)
/// callback. See the [observer lifecycle](self#lifecycle).
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub enum ObservationEvent<'a, F: crate::Scalar = f64> {
    /// Successful initialization or entry into an exact continuation.
    /// The restored iteration can be nonzero.
    Init,
    /// A completed iteration selected by [`ObserverMode`].
    Iter,
    /// A clean stop, including a partial-step stop. Hard errors do not emit
    /// an event. Repeated stepping after termination does not repeat it.
    Final(&'a TerminationReport<F>),
}

/// Per-registration policy for [`observe_iter`](Observe::observe_iter).
///
/// Every variant gates the iteration callback only: `observe_init` and
/// `observe_final` always fire. A user who wants to fully disable an observer
/// should simply not register it. The same policy applies to [`ObserveSolver`]
/// and [`ObservationEvent::Iter`] callbacks.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ObserverMode {
    /// Skip [`observe_iter`](Observe::observe_iter) entirely. The observer
    /// still sees `observe_init`/`observe_final`.
    Never,
    /// Fire [`observe_iter`](Observe::observe_iter) on every iteration.
    Always,
    /// Fire [`observe_iter`](Observe::observe_iter) on iterations whose
    /// (post-increment) counter is a multiple of `n`. `Every(1)` is
    /// equivalent to [`Always`](Self::Always); `Every(0)` is rejected as
    /// nonsensical and panics on construction via
    /// [`every`](Self::every), but raw construction is permitted and
    /// treated as `Never` (no iter is a multiple of zero under the
    /// `iter % n == 0` rule because `n == 0` would divide-by-zero; the
    /// executor checks for `n != 0` before the modulus).
    Every(u64),
    /// Fire [`observe_iter`](Observe::observe_iter) only on iterations that
    /// strictly improved the best cost so far, i.e. where
    /// [`State::best_iter`](crate::core::state::State::best_iter) equals
    /// [`State::iter`](crate::core::state::State::iter). Ties (a new iterate
    /// merely equal to the incumbent) do *not* fire, matching the strict
    /// `<` test in [`State::update_best`](crate::core::state::State::update_best).
    ///
    /// Pairs naturally with observers that only care about progress, such as
    /// logging or checkpointing the incumbent whenever it moves.
    NewBest,
}

impl ObserverMode {
    /// Construct an [`Every(n)`](Self::Every) mode, panicking when `n == 0`.
    ///
    /// Prefer this over the raw variant when `n` comes from user input.
    pub fn every(n: u64) -> Self {
        assert!(n > 0, "ObserverMode::every(n) requires n > 0");
        ObserverMode::Every(n)
    }

    /// Whether this mode wants [`observe_iter`](Observe::observe_iter) to
    /// fire for an iteration whose (post-increment) counter is `iter`.
    /// `is_new_best` reports whether that iteration strictly improved the
    /// best cost (only [`NewBest`](Self::NewBest) consults it).
    pub(crate) fn fires_on(&self, iter: u64, is_new_best: bool) -> bool {
        match *self {
            ObserverMode::Never => false,
            ObserverMode::Always => true,
            ObserverMode::Every(n) => n != 0 && iter % n == 0,
            ObserverMode::NewBest => is_new_best,
        }
    }
}
