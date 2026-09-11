//! Execution budgets and application stopping conditions.
//!
//! Convergence settings belong to the solver. These controls are observed
//! after initialization and between iterations; evaluation limits are not
//! hard caps on work performed inside an iteration.

// Keep the Basin 1.x compatibility bridge and shared check implementations local.
#![allow(deprecated)]

use super::math::Scalar;
use super::problem::EvalCounts;
use super::state::{AcceptanceState, GradientState, State};
use super::termination::{
    NoAcceptance, NoImprovement, TargetCost, TerminationCriterion,
    TerminationReason,
};
use web_time::{Duration, Instant};

type Stop<S> = Box<dyn FnMut(&S) -> Option<TerminationReason>>;

enum Entry<S> {
    Legacy(Box<dyn TerminationCriterion<S>>),
    Hook(Stop<S>),
    Factory {
        make: Box<dyn FnMut() -> Stop<S>>,
        current: Option<Stop<S>>,
    },
}

#[derive(Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub(crate) struct Limits {
    cost: Option<u64>,
    gradient: Option<u64>,
    time: Option<Duration>,
}

/// Budgets and application stops for a borrowed optimization run.
///
/// Used with [`run_loop_with_control`](super::executor::run_loop_with_control).
/// Builders replace their corresponding setting; custom hooks are appended.
/// Checks run in this order: iteration, cost evaluations, gradient evaluations,
/// elapsed time, target cost, improvement stall, acceptance stall, and hooks.
/// The clock starts at the first check after initialization.
pub struct RunControl<S> {
    pub(crate) max_iter: u64,
    pub(crate) limits: Limits,
    start: Option<Instant>,
    target: Option<Box<dyn TerminationCriterion<S>>>,
    improvement: Option<Box<dyn TerminationCriterion<S>>>,
    acceptance: Option<Box<dyn TerminationCriterion<S>>>,
    entries: Vec<Entry<S>>,
}

impl<S> Default for RunControl<S> {
    fn default() -> Self {
        Self::new()
    }
}

impl<S> RunControl<S> {
    /// Create controls with a 1,000-iteration limit and no other stops.
    pub fn new() -> Self {
        Self {
            max_iter: 1000,
            limits: Limits::default(),
            start: None,
            target: None,
            improvement: None,
            acceptance: None,
            entries: Vec::new(),
        }
    }

    /// Set the absolute iteration limit. Replaces the previous limit.
    pub fn max_iter(mut self, limit: u64) -> Self {
        self.max_iter = limit;
        self
    }

    /// Set a boundary-checked cost-evaluation limit, using the state's count.
    pub fn max_cost_evals(mut self, limit: u64) -> Self {
        self.limits.cost = Some(limit);
        self
    }

    /// Set a boundary-checked gradient-evaluation limit using the problem wrapper
    /// count (relative to run entry for borrowed runs).
    pub fn max_gradient_evals(mut self, limit: u64) -> Self
    where
        S: GradientState,
    {
        self.limits.gradient = Some(limit);
        self
    }

    /// Set the time allowed after the first post-initialization check.
    pub fn max_time(mut self, limit: Duration) -> Self {
        self.limits.time = Some(limit);
        self
    }

    /// Stop when the state's best objective value is at most `target`.
    pub fn target_cost<F: Scalar + 'static>(mut self, target: F) -> Self
    where
        S: State<Float = F>,
    {
        assert!(target.is_finite(), "target cost must be finite");
        self.target = Some(Box::new(TargetCost(target)));
        self
    }

    /// Stop after `patience` checks without a best-cost decrease greater
    /// than `min_delta`. Zero delta preserves state-carried stall history.
    pub fn no_improvement<F: Scalar + 'static>(
        mut self,
        patience: u64,
        min_delta: F,
    ) -> Self
    where
        S: State<Float = F>,
    {
        assert!(
            min_delta.is_finite() && min_delta >= F::zero(),
            "minimum improvement must be finite and nonnegative"
        );
        self.improvement =
            Some(Box::new(NoImprovement::new(patience, min_delta)));
        self
    }

    /// Stop after `patience` completed iterations without an accepted move.
    /// Panics when `patience` is zero.
    pub fn no_acceptance(mut self, patience: u64) -> Self
    where
        S: AcceptanceState,
    {
        self.acceptance = Some(Box::new(NoAcceptance::new(patience)));
        self
    }

    /// Append a custom state-based stop, evaluated before solver convergence.
    ///
    /// Return `Some(TerminationReason::UserRequested)` for an application stop,
    /// or another reason describing the condition. A closure's captured state
    /// persists when these controls are reused. Use [`stop_when_factory`](Self::stop_when_factory)
    /// for history that must start fresh on each borrowed run.
    pub fn stop_when<C>(mut self, check: C) -> Self
    where
        C: FnMut(&S) -> Option<TerminationReason> + 'static,
    {
        self.entries.push(Entry::Hook(Box::new(check)));
        self
    }

    /// Append a factory that creates fresh closure history for each run.
    pub fn stop_when_factory<M, C>(mut self, mut make: M) -> Self
    where
        M: FnMut() -> C + 'static,
        C: FnMut(&S) -> Option<TerminationReason> + 'static,
    {
        self.entries.push(Entry::Factory {
            make: Box::new(move || Box::new(make())),
            current: None,
        });
        self
    }

    pub(crate) fn push_legacy(
        &mut self,
        check: Box<dyn TerminationCriterion<S>>,
    ) {
        self.entries.push(Entry::Legacy(check));
    }

    // Borrowed runs reuse controls. Owned executors retain incoming legacy
    // criterion history, matching the original Basin 1.x contract.
    pub(crate) fn reset(&mut self) {
        self.start = None;
        for check in [
            &mut self.target,
            &mut self.improvement,
            &mut self.acceptance,
        ]
        .into_iter()
        .flatten()
        {
            check.reset();
        }
        for entry in &mut self.entries {
            match entry {
                Entry::Legacy(check) => check.reset(),
                Entry::Factory { make, current } => *current = Some(make()),
                Entry::Hook(_) => {}
            }
        }
    }

    #[cfg(feature = "serde")]
    pub(crate) fn has_unserializable_stops(&self) -> bool {
        self.target.is_some()
            || self.improvement.is_some()
            || self.acceptance.is_some()
            || !self.entries.is_empty()
    }

    pub(crate) fn check(
        &mut self,
        state: &S,
        counts: &EvalCounts,
    ) -> Option<TerminationReason>
    where
        S: State,
    {
        if state.iter() >= self.max_iter {
            return Some(TerminationReason::MaxIter);
        }
        if self.limits.cost.is_some_and(|n| state.cost_evals() >= n) {
            return Some(TerminationReason::MaxCostEvals);
        }
        if self
            .limits
            .gradient
            .is_some_and(|n| counts.gradient_evals >= n)
        {
            return Some(TerminationReason::MaxGradientEvals);
        }
        if let Some(limit) = self.limits.time {
            if self.start.get_or_insert_with(Instant::now).elapsed() >= limit {
                return Some(TerminationReason::MaxTime);
            }
        }
        for check in [
            &mut self.target,
            &mut self.improvement,
            &mut self.acceptance,
        ]
        .into_iter()
        .flatten()
        {
            if let Some(reason) = check.check(state) {
                return Some(reason);
            }
        }
        for entry in &mut self.entries {
            let reason = match entry {
                Entry::Legacy(check) => check.check(state),
                Entry::Hook(check) => check(state),
                Entry::Factory { make, current } => {
                    current.get_or_insert_with(make)(state)
                }
            };
            if reason.is_some() {
                return reason;
            }
        }
        None
    }
}

// Keep the owned and borrowed builders identical without duplicating checks.
macro_rules! control_methods {
    () => {
        /// Set a cost-evaluation budget, checked after initialization and between iterations.
        pub fn max_cost_evals(mut self, limit: u64) -> Self {
            self.control =
                std::mem::take(&mut self.control).max_cost_evals(limit);
            self
        }
        /// Set a gradient-evaluation budget, checked between iterations.
        pub fn max_gradient_evals(mut self, limit: u64) -> Self
        where
            S: crate::core::state::GradientState,
        {
            self.control =
                std::mem::take(&mut self.control).max_gradient_evals(limit);
            self
        }
        /// Set a time budget starting at the first post-initialization check.
        pub fn max_time(mut self, limit: web_time::Duration) -> Self {
            self.control = std::mem::take(&mut self.control).max_time(limit);
            self
        }
        /// Stop when the state's best cost reaches the finite target.
        pub fn target_cost<F: crate::core::math::Scalar + 'static>(
            mut self,
            target: F,
        ) -> Self
        where
            S: crate::core::state::State<Float = F>,
        {
            self.control =
                std::mem::take(&mut self.control).target_cost(target);
            self
        }
        /// Stop after `patience` checks without improvement greater than `min_delta`.
        pub fn no_improvement<F: crate::core::math::Scalar + 'static>(
            mut self,
            patience: u64,
            min_delta: F,
        ) -> Self
        where
            S: crate::core::state::State<Float = F>,
        {
            self.control = std::mem::take(&mut self.control)
                .no_improvement(patience, min_delta);
            self
        }
        /// Stop after a positive number of iterations without an accepted move.
        pub fn no_acceptance(mut self, patience: u64) -> Self
        where
            S: crate::core::state::AcceptanceState,
        {
            self.control =
                std::mem::take(&mut self.control).no_acceptance(patience);
            self
        }
        /// Append a factory creating fresh application-stop history for each run.
        pub fn stop_when_factory<M, C>(mut self, make: M) -> Self
        where
            M: FnMut() -> C + 'static,
            C: FnMut(&S) -> Option<crate::TerminationReason> + 'static,
        {
            self.control =
                std::mem::take(&mut self.control).stop_when_factory(make);
            self
        }
    };
}
pub(crate) use control_methods;

// Old outer constructors retain their additional inner-gradient test until
// Basin 2.0. New constructors pass None and use only solver-owned convergence.
pub(crate) fn legacy_inner_control<S, F>(
    max_iter: u64,
    tolerance: Option<F>,
) -> RunControl<S>
where
    F: Scalar + 'static,
    S: GradientState<Float = F>,
    S::Param: crate::NormSquared<F>,
{
    let mut control = RunControl::new().max_iter(max_iter);
    if let Some(tol) = tolerance {
        control
            .push_legacy(Box::new(super::termination::GradientTolerance(tol)));
    }
    control
}
