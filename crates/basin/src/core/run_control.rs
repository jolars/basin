//! Execution budgets and application stopping conditions.
//!
//! Convergence settings belong to the solver. These controls are observed
//! after initialization and between iterations; evaluation limits are not
//! hard caps on work performed inside an iteration.

// Keep the Basin 1.x compatibility bridge and shared check implementations local.
#![allow(deprecated)]

use super::math::Scalar;
use super::problem::{EvalCounts, EvaluationKind};
use super::state::{
    AcceptanceState, EvaluatedState, GradientState, ObjectiveIncumbentState,
    RawEvaluationState, State,
};
use super::termination::{
    NoAcceptance, NoImprovement, TargetCost, TerminationCriterion,
    TerminationReason,
};
use web_time::{Duration, Instant};

#[cfg(test)]
mod tests;

type Stop<S> = Box<dyn FnMut(&S) -> Option<TerminationReason>>;

struct RawBudgets<S> {
    limits: [Option<u64>; 7],
    counts: fn(&S) -> &EvalCounts,
}

struct ObjectiveTarget<F: Scalar>(F);

impl<S: ObjectiveIncumbentState<Float = F>, F: Scalar> TerminationCriterion<S>
    for ObjectiveTarget<F>
{
    fn check(&mut self, state: &S) -> Option<TerminationReason> {
        (state.incumbent_record()?.cost <= self.0)
            .then_some(TerminationReason::TargetCost)
    }
}

struct ObjectiveStall<F: Scalar> {
    patience: u64,
    min_delta: F,
    anchor: Option<(F, u64)>,
}

impl<S: ObjectiveIncumbentState<Float = F>, F: Scalar> TerminationCriterion<S>
    for ObjectiveStall<F>
{
    fn check(&mut self, state: &S) -> Option<TerminationReason> {
        let incumbent = state.incumbent_record()?;
        let improved_at = if self.min_delta == F::zero() {
            incumbent.iter
        } else {
            let (anchor, iter) =
                self.anchor.get_or_insert((incumbent.cost, state.iter()));
            // Subtract costs rather than the tolerance so an overflowing
            // finite decrease or a new negative infinity still improves.
            if incumbent.cost < *anchor
                && *anchor - incumbent.cost > self.min_delta
            {
                *anchor = incumbent.cost;
                *iter = state.iter();
            }
            *iter
        };
        (state.iter().saturating_sub(improved_at) >= self.patience)
            .then_some(TerminationReason::NoImprovement)
    }

    fn reset(&mut self) {
        self.anchor = None;
    }
}

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
/// raw evaluation budgets (in [`EvaluationKind`] order), elapsed time, target
/// cost, improvement stall, acceptance stall, and hooks.
/// The clock starts at the first check after initialization.
/// Publication validation, when enabled, precedes observers and all checks.
pub struct RunControl<S> {
    pub(crate) max_iter: u64,
    pub(crate) limits: Limits,
    raw: Option<RawBudgets<S>>,
    validate: Option<fn(&S)>,
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
            raw: None,
            validate: None,
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

    /// Require a complete evaluated record at every publication boundary.
    ///
    /// Requires [`EvaluatedState`]. Initialization, successful iterations,
    /// clean mid-step stops, and restored checkpoints are validated before
    /// incumbent updates, observers, or stopping checks. Non-finite evaluated
    /// values are allowed. Repeated calls keep validation enabled.
    ///
    /// # Panics
    ///
    /// During execution, panics with `solver published incomplete state` if a
    /// record is unavailable. This is a solver contract violation, not a
    /// numerical termination or a recoverable problem error. Hard problem
    /// errors bypass validation and retain their original error type.
    pub fn require_evaluated_state(mut self) -> Self
    where
        S: EvaluatedState,
    {
        self.validate = Some(|state| {
            assert!(
                state.current_record().is_some(),
                "solver published incomplete state"
            );
        });
        self
    }

    /// Set a raw category or total-work budget, requiring [`RawEvaluationState`].
    ///
    /// Checks run after initialization and between iterations, so even a zero
    /// limit permits initialization. Limits are not hard caps inside a step.
    /// Fresh and nested runs use per-run counts; exact continuation uses
    /// cumulative counts. No gradient capability is required.
    ///
    /// Replaces the limit for `kind`. Different kinds and legacy budgets are
    /// independent; legacy budgets are checked first. Exhaustion reports
    /// [`TerminationReason::MaxEvaluations`] for every kind. Compare the
    /// state's [`raw_counts`](RawEvaluationState::raw_counts) with configured
    /// limits to identify exhausted budgets. Capability controls on an
    /// [`InnerExecutor`](crate::InnerExecutor) cannot be serialized.
    ///
    /// ```compile_fail
    /// use basin::{BasicState, EvaluationKind, RunControl};
    /// let _ = RunControl::<BasicState<Vec<f64>>>::new()
    ///     .max_evaluations(EvaluationKind::Cost, 10);
    /// ```
    pub fn max_evaluations(mut self, kind: EvaluationKind, limit: u64) -> Self
    where
        S: RawEvaluationState,
    {
        let raw = self.raw.get_or_insert(RawBudgets {
            limits: [None; 7],
            counts: S::raw_counts,
        });
        raw.limits[kind as usize] = Some(limit);
        self
    }

    /// Stop when an eligible objective-ordered incumbent reaches `target`.
    ///
    /// Requires [`ObjectiveIncumbentState`]; no incumbent means no target
    /// success. Replaces [`target_cost`](Self::target_cost), and vice versa.
    ///
    /// # Panics
    ///
    /// Panics if `target` is not finite.
    ///
    /// ```compile_fail
    /// use basin::{CobylaState, RunControl};
    /// let _ = RunControl::<CobylaState<Vec<f64>>>::new().target_objective(0.0);
    /// ```
    pub fn target_objective<F: Scalar + 'static>(mut self, target: F) -> Self
    where
        S: ObjectiveIncumbentState<Float = F>,
    {
        assert!(target.is_finite(), "target objective must be finite");
        self.target = Some(Box::new(ObjectiveTarget(target)));
        self
    }

    /// Stop after `patience` completed iterations without objective improvement.
    ///
    /// Requires [`ObjectiveIncumbentState`]. Tracking waits for an incumbent.
    /// Zero `min_delta` uses its publication iteration and preserves stall age
    /// on exact continuation. Positive delta tracks strict decreases exceeding
    /// the delta from a running anchor, starting at the first observation.
    /// Repeated observations do not age the stall. Negative infinity can be
    /// an incumbent and does not itself establish unboundedness.
    ///
    /// Replaces [`no_improvement`](Self::no_improvement), and vice versa.
    /// Borrowed runs reset anchor history. Controls are not part of an exact
    /// checkpoint; reattaching a positive-delta check starts fresh history.
    ///
    /// # Panics
    ///
    /// Panics if `patience` is zero, or `min_delta` is negative or non-finite.
    pub fn no_objective_improvement<F: Scalar + 'static>(
        mut self,
        patience: u64,
        min_delta: F,
    ) -> Self
    where
        S: ObjectiveIncumbentState<Float = F>,
    {
        assert!(patience > 0, "patience must be positive");
        assert!(
            min_delta.is_finite() && min_delta >= F::zero(),
            "minimum improvement must be finite and nonnegative"
        );
        self.improvement = Some(Box::new(ObjectiveStall {
            patience,
            min_delta,
            anchor: None,
        }));
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
            || self.raw.is_some()
            || self.validate.is_some()
            || self.improvement.is_some()
            || self.acceptance.is_some()
            || !self.entries.is_empty()
    }

    pub(crate) fn validate(&self, state: &S) {
        if let Some(validate) = self.validate {
            validate(state);
        }
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
        if let Some(raw) = &self.raw {
            let counts = (raw.counts)(state);
            for kind in EvaluationKind::ALL {
                if raw.limits[kind as usize]
                    .is_some_and(|limit| kind.count(counts) >= limit)
                {
                    return Some(TerminationReason::MaxEvaluations);
                }
            }
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
        /// Validate complete records at publication boundaries.
        ///
        /// See [`RunControl::require_evaluated_state`](crate::RunControl::require_evaluated_state)
        /// for validation ordering and solver-contract panics. This control
        /// cannot be serialized as part of an inner executor.
        pub fn require_evaluated_state(mut self) -> Self
        where
            S: crate::core::state::EvaluatedState,
        {
            self.control =
                std::mem::take(&mut self.control).require_evaluated_state();
            self
        }
        /// Set a raw category or total-work budget at iteration boundaries.
        ///
        /// See [`RunControl::max_evaluations`](crate::RunControl::max_evaluations)
        /// for accounting, precedence, and serialization limits.
        pub fn max_evaluations(
            mut self,
            kind: crate::EvaluationKind,
            limit: u64,
        ) -> Self
        where
            S: crate::core::state::RawEvaluationState,
        {
            self.control =
                std::mem::take(&mut self.control).max_evaluations(kind, limit);
            self
        }
        /// Stop when an eligible objective-ordered incumbent reaches a finite target.
        ///
        /// Replaces `target_cost`, and vice versa. See
        /// [`RunControl::target_objective`](crate::RunControl::target_objective).
        pub fn target_objective<F: crate::core::math::Scalar + 'static>(
            mut self,
            target: F,
        ) -> Self
        where
            S: crate::core::state::ObjectiveIncumbentState<Float = F>,
        {
            self.control =
                std::mem::take(&mut self.control).target_objective(target);
            self
        }
        /// Stop after completed iterations without a sufficient objective decrease.
        ///
        /// Replaces `no_improvement`, and vice versa. See
        /// [`RunControl::no_objective_improvement`](crate::RunControl::no_objective_improvement)
        /// for threshold validation, publication age, and resume behavior.
        pub fn no_objective_improvement<
            F: crate::core::math::Scalar + 'static,
        >(
            mut self,
            patience: u64,
            min_delta: F,
        ) -> Self
        where
            S: crate::core::state::ObjectiveIncumbentState<Float = F>,
        {
            self.control = std::mem::take(&mut self.control)
                .no_objective_improvement(patience, min_delta);
            self
        }
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
