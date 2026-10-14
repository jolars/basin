//! Native Moré–Thuente predicates and bracket updates.

use crate::{EvalCounts, Scalar};

/// Reason the native search stopped. A selected step is not a Wolfe certificate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SearchExit {
    /// Both native Wolfe comparisons passed.
    Wolfe,
    /// One or more warning comparisons passed.
    Warning,
    /// The evaluation loop exhausted its configured cap.
    Exhausted,
    /// Initial slope or step was invalid.
    InvalidInitial,
    /// The bounded interval was invalid.
    InvalidBounds,
    /// A callback returned an error, including a physical budget denial.
    CallbackError,
}

/// Native bracket state; endpoint values use the original objective.
#[derive(Clone, Copy, Debug)]
pub struct Bracket<F: Scalar = f64> {
    /// Best endpoint step, value, and directional derivative.
    pub best: [F; 3],
    /// Other endpoint step, value, and directional derivative.
    pub other: [F; 3],
    /// Safeguard interval for the next trial.
    pub interval: [F; 2],
    /// Whether the algorithm has bracketed a minimum.
    pub bracketed: bool,
    /// Modified-function stage (1) or original-function stage (2).
    pub stage: u8,
}

/// One attempted fused callback, with comparisons and the subsequent update.
#[derive(Clone, Debug)]
pub struct SearchTrial<F: Scalar = f64> {
    /// Attempted native step along the caller's direction.
    pub step: F,
    /// Completed cost and directional derivative; absent on callback error.
    pub value: Option<[F; 2]>,
    /// Native Armijo right-hand side.
    pub sufficient_decrease_bound: F,
    /// Native strong-curvature upper bound.
    pub curvature_bound: F,
    /// Both Wolfe predicates, absent before a completed callback.
    pub wolfe: Option<[bool; 2]>,
    /// Rounding interval, no resolution, bracket width, upper step, lower step.
    pub warnings: Option<[bool; 5]>,
    /// Bracket immediately before the callback.
    pub before: Bracket<F>,
    /// Bracket and safeguarded step after interpolation, absent on an exit.
    pub update: Option<(Bracket<F>, F)>,
    /// Actual `dcstep` case (1 through 4), absent when no update ran.
    pub interpolation_case: Option<u8>,
    /// Whether interpolation used the stage-one modified function.
    pub modified_function: bool,
    /// Authoritative categories before the callback.
    pub counts_before: EvalCounts,
    /// Authoritative categories after the callback, including denied requests.
    pub counts_after: EvalCounts,
}

/// Opt-in recorder around the original seven-field Moré–Thuente configuration.
///
/// The latest invocation replaces its trial batch, including failed callbacks.
/// Read getters without extra evaluations. A selected step is independent of
/// Wolfe convergence. Diagnostic settings and records are omitted by serde;
/// the wrapped configuration keeps its original serialization shape.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TracedMoreThuente<F: Scalar = f64> {
    /// Original configuration, including its public algorithm controls.
    pub inner: super::MoreThuente<F>,
    #[cfg_attr(feature = "serde", serde(skip, default = "Trace::default"))]
    trace: Trace<F>,
}
impl<F: Scalar> TracedMoreThuente<F> {
    /// Record the native search for this configuration.
    pub fn new(inner: super::MoreThuente<F>) -> Self {
        Self {
            inner,
            trace: Trace {
                enabled: true,
                ..Trace::default()
            },
        }
    }
    /// Enable or disable recording without changing the search configuration.
    pub fn with_trial_diagnostics(mut self, enabled: bool) -> Self {
        self.trace.enabled = enabled;
        self.trace.reset();
        self
    }
    /// Native attempts from the latest invocation, including hard errors.
    pub fn trial_observations(&self) -> &[SearchTrial<F>] {
        &self.trace.trials
    }
    /// Actual native exit, separate from selected-step usability.
    pub fn search_exit(&self) -> Option<SearchExit> {
        self.trace.exit
    }
    /// Selected native step, absent on callback errors and bounded failures.
    pub fn selected_step(&self) -> Option<F> {
        self.trace.selected
    }
}

pub(super) struct Trace<F: Scalar> {
    pub enabled: bool,
    pub trials: Vec<SearchTrial<F>>,
    pub exit: Option<SearchExit>,
    pub selected: Option<F>,
}
impl<F: Scalar> Default for Trace<F> {
    fn default() -> Self {
        Self {
            enabled: false,
            trials: Vec::new(),
            exit: None,
            selected: None,
        }
    }
}
impl<F: Scalar> Trace<F> {
    fn reset(&mut self) {
        self.trials.clear();
        self.exit = None;
        self.selected = None;
    }
    pub(super) fn record_exit(
        &mut self,
        exit: SearchExit,
        selected: Option<F>,
    ) {
        if self.enabled {
            self.exit = Some(exit);
            self.selected = selected;
        }
    }
}

impl<P, V, F> crate::LineSearch<P, V, F> for TracedMoreThuente<F>
where
    F: Scalar,
    P: crate::CostFunction<Param = V, Output = F>
        + crate::Gradient<Gradient = V>,
    V: crate::ScaledAdd<F> + crate::Dot<F> + Clone,
{
    type Error = P::Error;
    fn reset(&mut self) {
        self.trace.reset();
    }
    fn next(
        &mut self,
        p: &mut crate::Problem<P>,
        x: &V,
        cost: F,
        g: &V,
        d: &V,
    ) -> Result<F, P::Error> {
        self.inner
            .search_recorded(p, x, cost, g, d, &mut self.trace)
            .map(|r| match r.outcome {
                crate::LineSearchOutcome::Step(step) => step,
                crate::LineSearchOutcome::Failed => F::zero(),
            })
    }
    fn next_with_evaluation(
        &mut self,
        p: &mut crate::Problem<P>,
        x: &V,
        cost: F,
        g: &V,
        d: &V,
    ) -> Result<crate::LineSearchResult<V, F>, P::Error> {
        self.inner
            .search_recorded(p, x, cost, g, d, &mut self.trace)
    }
    fn next_with_bounds(
        &mut self,
        p: &mut crate::Problem<P>,
        x: &V,
        cost: F,
        g: &V,
        d: &V,
        bounds: crate::LineSearchBounds<F>,
    ) -> Result<crate::LineSearchResult<V, F>, P::Error> {
        let max = self.inner.stpmax.min(bounds.max());
        if !(max.is_finite() && max > F::zero() && self.inner.stpmin <= max) {
            self.trace.reset();
            self.trace.record_exit(SearchExit::InvalidBounds, None);
            return Ok(crate::LineSearchResult::new(
                crate::LineSearchOutcome::Failed,
            ));
        }
        let search = super::MoreThuente {
            alpha_init: bounds.initial().max(self.inner.stpmin).min(max),
            stpmax: max,
            ..self.inner
        };
        search.search_recorded(p, x, cost, g, d, &mut self.trace)
    }
}
