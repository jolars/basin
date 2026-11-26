//! Execution budgets and application stopping conditions.
//!
//! Convergence settings belong to the solver. These controls are observed
//! after initialization and between iterations; evaluation limits are not
//! hard caps on work performed inside an iteration.
//! Evaluation budgets use [`EvalCounts`] callback requests at the
//! [`Problem`](crate::Problem) boundary. Internal adapter probes are excluded,
//! including from [`EvaluationKind::TotalWork`] budgets.

use super::math::Scalar;
use super::problem::{EvalCounts, EvaluationKind};
use super::state::{
    AcceptanceState, EvaluatedState, ObjectiveIncumbentState, State,
};
use super::termination::{ApplicationStop, ExecutionLimit, Stall, Termination};
use web_time::{Duration, Instant};

#[cfg(test)]
mod tests;

type Stop<S> = Box<dyn FnMut(&S) -> Option<Termination<<S as State>::Float>>>;

struct ObjectiveStall<F: Scalar> {
    patience: u64,
    min_delta: F,
    anchor: Option<(F, u64)>,
}

impl<F: Scalar> ObjectiveStall<F> {
    fn check<S: ObjectiveIncumbentState<Float = F>>(
        &mut self,
        state: &S,
    ) -> Option<Termination<<S as State>::Float>> {
        let incumbent = state.selected_record()?;
        let cost = incumbent.cost?;
        let improved_at = if self.min_delta == F::zero() {
            incumbent.iter
        } else {
            let (anchor, iter) =
                self.anchor.get_or_insert((cost, state.iter()));
            // Subtract costs rather than the tolerance so an overflowing
            // finite decrease or a new negative infinity still improves.
            if cost < *anchor && *anchor - cost > self.min_delta {
                *anchor = cost;
                *iter = state.iter();
            }
            *iter
        };
        (state.iter().saturating_sub(improved_at) >= self.patience).then(|| {
            Termination::Stalled(Stall::NoImprovement {
                iterations: state.iter().saturating_sub(improved_at),
                patience: self.patience,
                tolerance: self.min_delta,
                reference_cost: self.anchor.map_or(cost, |(cost, _)| cost),
                best_cost: cost,
            })
        })
    }
}

enum Entry<S: State> {
    Hook(Stop<S>),
    Factory {
        make: Box<dyn FnMut() -> Stop<S>>,
        current: Option<Stop<S>>,
    },
}

impl<S: State> Entry<S> {
    fn reset(&mut self) {
        if let Self::Factory { make, current } = self {
            *current = Some(make());
        }
    }

    fn check(&mut self, state: &S) -> Option<Termination<<S as State>::Float>> {
        match self {
            Self::Hook(check) => check(state),
            Self::Factory { make, current } => {
                current.get_or_insert_with(make)(state)
            }
        }
    }
}

#[derive(Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(deny_unknown_fields))]
pub(crate) struct Limits {
    evaluations: Option<[Option<u64>; EvaluationKind::ALL.len()]>,
    time: Option<Duration>,
}

/// Budgets and application stops for a borrowed optimization run.
///
/// Used with [`run_loop_with_control`](super::executor::run_loop_with_control).
/// Builders replace their corresponding setting; custom hooks are appended.
/// Checks run in this order: iteration, evaluation budgets (in
/// [`EvaluationKind`] order), elapsed time, target cost, improvement stall,
/// acceptance stall, and hooks.
/// The clock starts at the first check after initialization.
/// Publication validation, when enabled, precedes observers and all checks.
pub struct RunControl<S: State> {
    pub(crate) max_iter: u64,
    pub(crate) limits: Limits,
    validate: Option<fn(&S)>,
    start: Option<Instant>,
    target: Option<Stop<S>>,
    improvement: Option<Entry<S>>,
    acceptance: Option<Stop<S>>,
    entries: Vec<Entry<S>>,
}

impl<S: State> Default for RunControl<S> {
    fn default() -> Self {
        Self::new()
    }
}

impl<S: State> RunControl<S> {
    /// Create controls with a 1,000-iteration limit and no other stops.
    pub fn new() -> Self {
        Self {
            max_iter: 1000,
            limits: Limits::default(),
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
                state.evaluated_record().is_some()
                    && state
                        .current_record()
                        .and_then(|record| record.cost)
                        .is_some(),
                "solver published incomplete state"
            );
        });
        self
    }

    /// Set a category or total-work budget using the problem wrapper's counts.
    ///
    /// Checks run after initialization and between iterations, so even a zero
    /// limit permits initialization. Limits are not hard caps inside a step.
    /// Fresh and nested runs use per-run counts; exact continuation uses
    /// cumulative counts. No gradient capability is required.
    ///
    /// Counts follow [`EvalCounts`]: internal adapter evaluations, such as
    /// finite-difference probes, are excluded even from `TotalWork`.
    ///
    /// Replaces the limit for `kind`. Different kinds are independent.
    /// [`ExecutionLimit::Evaluations`] identifies each exhausted kind and its
    /// observed count. With `serde`, these limits are serialized as part of
    /// an [`InnerExecutor`](crate::InnerExecutor).
    ///
    /// ```
    /// use basin::{PointState, EvaluationKind, RunControl};
    /// let _ = RunControl::<PointState<Vec<f64>>>::new()
    ///     .max_evaluations(EvaluationKind::Cost, 10);
    /// ```
    pub fn max_evaluations(mut self, kind: EvaluationKind, limit: u64) -> Self {
        *self.evaluation_budget(kind) = Some(limit);
        self
    }

    pub(crate) fn cap_evaluations(&mut self, kind: EvaluationKind, limit: u64) {
        let budget = self.evaluation_budget(kind);
        *budget = Some(budget.map_or(limit, |existing| existing.min(limit)));
    }

    fn evaluation_budget(&mut self, kind: EvaluationKind) -> &mut Option<u64> {
        let budgets = self
            .limits
            .evaluations
            .get_or_insert([None; EvaluationKind::ALL.len()]);
        &mut budgets[kind as usize]
    }

    /// Stop when an eligible objective-ordered incumbent reaches `target`.
    ///
    /// Requires [`ObjectiveIncumbentState`]; no incumbent means no target
    /// success. Repeated calls replace the previous target.
    ///
    /// # Panics
    ///
    /// Panics if `target` is not finite.
    ///
    /// ```compile_fail
    /// use basin::{SelectedState, RunControl};
    /// let _ = RunControl::<SelectedState<Vec<f64>>>::new().target_objective(0.0);
    /// ```
    pub fn target_objective<F: Scalar + 'static>(mut self, target: F) -> Self
    where
        S: ObjectiveIncumbentState<Float = F>,
    {
        assert!(target.is_finite(), "target objective must be finite");
        self.target = Some(Box::new(move |state| {
            let cost = state.selected_record()?.cost?;
            (cost <= target).then_some(Termination::Target { cost, target })
        }));
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
    /// Repeated calls replace the previous stall check.
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
        self.improvement = Some(Entry::Factory {
            make: Box::new(move || {
                let mut stall = ObjectiveStall {
                    patience,
                    min_delta,
                    anchor: None,
                };
                Box::new(move |state| stall.check(state))
            }),
            current: None,
        });
        self
    }

    /// Set the time allowed after the first post-initialization check.
    pub fn max_time(mut self, limit: Duration) -> Self {
        self.limits.time = Some(limit);
        self
    }

    /// Stop after `patience` completed iterations without an accepted move.
    /// Panics when `patience` is zero.
    pub fn no_acceptance(mut self, patience: u64) -> Self
    where
        S: AcceptanceState,
    {
        assert!(patience > 0, "patience must be positive");
        self.acceptance = Some(Box::new(move |state| {
            (state.iter().saturating_sub(state.last_accepted_iter())
                >= patience)
                .then_some(Termination::Stalled(Stall::NoAcceptedMove {
                    iterations: state
                        .iter()
                        .saturating_sub(state.last_accepted_iter()),
                    patience,
                }))
        }));
        self
    }

    /// Append a custom state-based stop, evaluated before solver convergence.
    ///
    /// Return `Some(ApplicationStop::new("my_stop"))` for an application stop,
    /// or another reason describing the condition. A closure's captured state
    /// persists when these controls are reused. Use [`stop_when_factory`](Self::stop_when_factory)
    /// for history that must start fresh on each borrowed run.
    pub fn stop_when<C>(mut self, mut check: C) -> Self
    where
        C: FnMut(&S) -> Option<ApplicationStop> + 'static,
    {
        self.entries.push(Entry::Hook(Box::new(move |state| {
            check(state).map(Termination::Application)
        })));
        self
    }

    /// Append a factory that creates fresh closure history for each run.
    pub fn stop_when_factory<M, C>(mut self, mut make: M) -> Self
    where
        M: FnMut() -> C + 'static,
        C: FnMut(&S) -> Option<ApplicationStop> + 'static,
    {
        self.entries.push(Entry::Factory {
            make: Box::new(move || {
                let mut check = make();
                Box::new(move |state| {
                    check(state).map(Termination::Application)
                })
            }),
            current: None,
        });
        self
    }

    pub(crate) fn reset(&mut self) {
        self.start = None;
        for entry in self.improvement.iter_mut().chain(&mut self.entries) {
            entry.reset();
        }
    }

    #[cfg(feature = "serde")]
    pub(crate) fn has_unserializable_stops(&self) -> bool {
        self.target.is_some()
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
        baseline: Option<&EvalCounts>,
    ) -> Option<Termination<<S as State>::Float>> {
        let mut limits = Vec::new();
        if state.iter() >= self.max_iter {
            limits.push(ExecutionLimit::Iterations {
                observed: state.iter(),
                limit: self.max_iter,
            });
        }
        if let Some(budgets) = &self.limits.evaluations {
            let delta = baseline.map(|base| counts.delta_since(base));
            let counts = delta.as_ref().unwrap_or(counts);
            for (kind, &budget) in EvaluationKind::ALL.into_iter().zip(budgets)
            {
                let Some(limit) = budget else {
                    continue;
                };
                let observed = kind.count(counts);
                if observed >= limit {
                    limits.push(ExecutionLimit::Evaluations {
                        kind,
                        observed,
                        limit,
                    });
                }
            }
        }
        if let Some(limit) = self.limits.time {
            let elapsed = self.start.get_or_insert_with(Instant::now).elapsed();
            if elapsed >= limit {
                limits.push(ExecutionLimit::Time { elapsed, limit });
            }
        }
        if !limits.is_empty() {
            return Some(Termination::Limit(limits));
        }
        if let Some(reason) =
            self.target.as_mut().and_then(|check| check(state))
        {
            return Some(reason);
        }
        if let Some(reason) = self
            .improvement
            .as_mut()
            .and_then(|check| check.check(state))
        {
            return Some(reason);
        }
        if let Some(reason) =
            self.acceptance.as_mut().and_then(|check| check(state))
        {
            return Some(reason);
        }
        for entry in &mut self.entries {
            if let Some(reason) = entry.check(state) {
                return Some(reason);
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
        ) -> Self {
            self.control =
                std::mem::take(&mut self.control).max_evaluations(kind, limit);
            self
        }
        /// Stop when an eligible objective-ordered incumbent reaches a finite target.
        ///
        /// Repeated calls replace the previous target. See
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
        /// Repeated calls replace the previous stall check. See
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
        /// Set a time budget starting at the first post-initialization check.
        pub fn max_time(mut self, limit: web_time::Duration) -> Self {
            self.control = std::mem::take(&mut self.control).max_time(limit);
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
            C: FnMut(&S) -> Option<crate::ApplicationStop> + 'static,
        {
            self.control =
                std::mem::take(&mut self.control).stop_when_factory(make);
            self
        }
    };
}
pub(crate) use control_methods;
