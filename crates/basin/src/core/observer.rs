//! Read-only side effects fired around the iteration loop.
//!
//! An [`Observe`] implementation watches the run as it happens (logging,
//! progress reporting, recording a trajectory, streaming iterates to a UI)
//! without influencing it. Observers do **not** decide whether to stop; that
//! is the job of [`TerminationCriterion`](crate::core::termination::TerminationCriterion).
//! The two extension points sit side-by-side on
//! [`Executor`](crate::core::executor::Executor):
//!
//! - [`TerminationCriterion`](crate::core::termination::TerminationCriterion):
//!   returns `Option<TerminationReason>`; framework consumes the result and
//!   stops the run if `Some`.
//! - [`Observe`]: returns `()`; the executor ignores any side effects on the
//!   optimization itself. Failures must be handled inside the observer:
//!   the trait is infallible by design so a misbehaving logger can't kill
//!   the run.
//!
//! Like termination criteria, observers bind on the minimum
//! [`State`](crate::core::state::State) shape they need (tenet 3): a logger
//! that just wants `iter`/`cost` impls `Observe<S: State>`; a gradient-norm
//! observer impls `Observe<S: GradientState>` and is rejected at compile time
//! when attached to a derivative-free run.
//!
//! # Lifecycle
//!
//! Three hooks, called in this order during a run:
//!
//! 1. [`observe_init`](Observe::observe_init) fires once after
//!    [`Solver::init`](crate::core::solver::Solver::init) returns and the
//!    state's counter mirror is refreshed, before the first termination
//!    check. The state shows `iter() == 0`.
//! 2. [`observe_iter`](Observe::observe_iter) fires after every successfully
//!    completed iteration, after the iteration counter is incremented. On
//!    the first call the state shows `iter() == 1`. Gated by
//!    [`ObserverMode`].
//! 3. [`observe_final`](Observe::observe_final) fires once when the run
//!    stops cleanly with a [`TerminationReason`]. The state shows the iter
//!    count of the last fully-completed iteration.
//!
//! # Edge cases
//!
//! - **Mid-iter termination.** If
//!   [`Solver::next_iter`](crate::core::solver::Solver::next_iter) returns
//!   `(state, Some(reason))`, the iteration counter is *not* incremented,
//!   [`observe_iter`](Observe::observe_iter) does *not* fire for that
//!   partial iteration, and [`observe_final`](Observe::observe_final) fires
//!   with the mid-iter reason.
//! - **Hard error from the problem.** If `next_iter` returns `Err(_)`, the
//!   state is consumed by the failing call and there's nothing to observe;
//!   [`observe_final`](Observe::observe_final) does *not* fire and the
//!   error propagates out of
//!   [`Executor::run`](crate::core::executor::Executor::run) /
//!   [`Stepper::step`](crate::core::executor::Stepper::step). Observers
//!   that need to react to hard aborts should track liveness themselves.
//! - **Reading the reason from inside observers.** The argument to
//!   [`observe_final`](Observe::observe_final) is the
//!   [`TerminationReason`]; state types do not carry it.

use crate::core::termination::TerminationReason;

/// Hooks fired around the iteration loop. See the
/// [module docs](self) for lifecycle and edge cases.
///
/// All three methods default to no-ops, so an implementor fills in only what
/// they need. Bind on the minimum state shape required (`S: State`,
/// `S: GradientState`, `S: SimplexState`, …) so a mismatch with the solver is
/// a compile error rather than a runtime no-op.
pub trait Observe<S> {
    /// Fired once before the first iteration, after
    /// [`Solver::init`](crate::core::solver::Solver::init) has run and the
    /// state's counter mirror has been refreshed. The state's iter counter
    /// is zero.
    ///
    /// Always fires regardless of the observer's [`ObserverMode`]; modes
    /// gate iteration callbacks only.
    fn observe_init(&mut self, _state: &S) {}

    /// Fired after each successfully completed iteration, after
    /// [`State::increment_iter`](crate::core::state::State::increment_iter)
    /// has run, so `state.iter()` returns the count of the iteration that
    /// just finished (1 on the first call, …).
    ///
    /// Gated by [`ObserverMode`]: `Never` skips, `Always` fires every iter,
    /// `Every(n)` fires when `state.iter() % n == 0`.
    fn observe_iter(&mut self, _state: &S) {}

    /// Fired once when the run stops with a clean
    /// [`TerminationReason`]. `state` is the final iterate; `reason` is what
    /// halted the run.
    ///
    /// Does *not* fire when
    /// [`Solver::next_iter`](crate::core::solver::Solver::next_iter)
    /// returns `Err(_)`: in that case the state has been consumed and
    /// there is nothing to observe.
    ///
    /// Always fires regardless of the observer's [`ObserverMode`].
    fn observe_final(&mut self, _state: &S, _reason: &TerminationReason) {}
}

/// Per-registration policy for [`observe_iter`](Observe::observe_iter).
///
/// `Never` and `Every` gate the iteration callback only: `observe_init` and
/// `observe_final` always fire. A user who wants to fully disable an observer
/// should simply not register it.
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
    pub(crate) fn fires_on(&self, iter: u64) -> bool {
        match *self {
            ObserverMode::Never => false,
            ObserverMode::Always => true,
            ObserverMode::Every(n) => n != 0 && iter % n == 0,
        }
    }
}
