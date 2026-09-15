//! Opt-in, checked progress interfaces for shared and external states.
//!
//! These traits leave the Basin 1.x [`State`] and [`super::GradientState`]
//! readers unchanged. Implement only the capabilities a state can guarantee.
//! Record availability describes evaluation, not numerical validity: an
//! evaluated rejection with infinite cost is still an available record.

use super::State;
use crate::core::math::Scalar;
use crate::core::problem::EvalCounts;

/// Checked access to an evaluated current point and its matching cost.
///
/// Return `None` for an unevaluated seed or incomplete record. Return `Some`
/// only when all advertised current-record data, including derivatives, is
/// populated and belongs to the same point. NaN and infinite values do not
/// themselves make an evaluated record unavailable.
///
/// [`Executor::require_evaluated_state`](crate::Executor::require_evaluated_state)
/// checks this contract at publication boundaries without evaluating the
/// problem. An external implementation remains responsible for ensuring the
/// returned values actually match.
pub trait EvaluatedState: State {
    /// Borrow the complete current record's point and cost, if available.
    fn current_record(&self) -> Option<(&Self::Param, Self::Float)>;
}

/// Checked access to a complete current point, cost, and gradient.
///
/// Availability must agree with [`EvaluatedState::current_record`]. A
/// numerical check requiring a gradient must handle an unavailable record
/// explicitly, rather than silently disabling itself. Attach publication
/// validation to reject incomplete records before checks or observers run.
///
/// A point-only state does not supply this capability:
///
/// ```compile_fail
/// use basin::{EvaluatedGradientState, PointState};
/// fn gradient_check<S: EvaluatedGradientState>(_: &S) {}
/// gradient_check(&PointState::new(vec![1.0]));
/// ```
pub trait EvaluatedGradientState: EvaluatedState {
    /// Borrow the matching point, cost, and gradient, or `None` for a seed.
    fn current_gradient_record(
        &self,
    ) -> Option<(&Self::Param, Self::Float, &Self::Param)>;
}

/// Raw evaluation accounting at the latest publication boundary.
///
/// Preserve every category mirrored from the authoritative [`Problem`](crate::Problem).
/// Fresh and nested runs report per-run deltas; exact continuation reports
/// cumulative work across the resumed run. These counts need not equal the
/// folded Basin 1.x [`State::cost_evals`] reader.
///
/// No gradient or evaluated-record capability is required: a derivative-free
/// outer solver can budget derivative work performed by its inner solvers.
pub trait RawEvaluationState: State {
    /// All six raw categories, without folding or reconstruction.
    fn raw_counts(&self) -> &EvalCounts;
}

/// A borrowed incumbent and the boundary that published its selection.
///
/// Counts include all charged work before publication, rather than only work
/// through the evaluation that found the point. This view owns no parameter
/// storage and does not change a state's serialized representation.
#[derive(Debug)]
pub struct IncumbentRef<'a, V, F: Scalar = f64> {
    /// Selected point, matching `cost`.
    pub param: &'a V,
    /// Objective at the selected point.
    pub cost: F,
    /// Completed-iteration count at the selection's publication boundary.
    pub iter: u64,
    /// Raw counts at the selection's publication boundary.
    pub counts: &'a EvalCounts,
}

/// Checked access to a solver's selected incumbent and publication metadata.
///
/// This capability does not imply objective ordering. A feasibility-first
/// solver may select a higher-cost point. Its selection changes must be
/// explicit events: observing or republishing the same selection must retain
/// its original iteration and counts, including across mid-step publication.
/// Iteration numbers alone do not identify a selection change.
///
/// Return `None` when no incumbent exists. When available, the record must
/// agree with the state's existing best-point, best-cost, and best-iteration
/// readers. Preserve the selected point separately if the current point or
/// population can change independently.
pub trait IncumbentState: State<Float: Scalar> {
    /// Borrow the selected record, if one has been established.
    fn incumbent_record(
        &self,
    ) -> Option<IncumbentRef<'_, Self::Param, Self::Float>>;
}

/// Incumbent selection compatible with objective-only targets and stalls.
///
/// Implementors retain strict objective improvements among eligible published
/// candidates. Equal costs preserve the incumbent and its metadata; NaN and
/// positive infinity never establish an incumbent. Negative infinity may be
/// retained but does not establish unboundedness. A constrained implementation
/// must exclude infeasible candidates and preserve objective ordering among
/// the eligible ones. Feasibility-first selection that can replace an
/// incumbent with a higher-cost point must not implement this trait.
///
/// Best tracking covers published candidates, not every problem evaluation.
/// There is deliberately no blanket implementation for [`State`].
pub trait ObjectiveIncumbentState: IncumbentState {}
