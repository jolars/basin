//! Opt-in, checked progress interfaces for shared and external states.
//!
//! These traits complement the [`State`] and [`super::GradientState`] readers
//! with checked records. Implement only the capabilities a state can guarantee.
//! Record availability describes evaluation, not numerical validity: an
//! evaluated rejection with infinite cost is still an available record.

use super::{State, ViolationReport};
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
    fn evaluated_record(&self) -> Option<(&Self::Param, Self::Float)>;
}

/// Checked access to a complete current point, cost, and gradient.
///
/// Availability must agree with [`EvaluatedState::evaluated_record`]. A
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
/// cumulative work across the resumed run. Shared states use these same raw
/// categories for their cost and gradient readers. Categories follow the
/// callback-request semantics of [`EvalCounts`]; internal adapter probes are
/// excluded.
///
/// No gradient or evaluated-record capability is required: a derivative-free
/// outer solver can budget derivative work performed by its inner solvers.
pub trait RawEvaluationState: State {
    /// All six raw categories, without folding or reconstruction.
    fn raw_counts(&self) -> &EvalCounts;
}

/// Checked constraint-violation reports for current and selected points.
///
/// Each report belongs to the same point as its corresponding progress
/// record. It can be available before the objective is evaluated.
/// Reports preserve evaluated values, including NaN and infinity.
///
/// Reading reports must not evaluate the problem or change state. The report's
/// runtime metadata describes its measure, scaling, and transformation;
/// different solvers may publish different measures through the same shared
/// state. This capability implies neither objective ordering nor a particular
/// feasibility tolerance. Numerical convergence remains solver-owned.
///
/// Unconstrained progress does not advertise this capability:
///
/// ```compile_fail
/// use basin::{ConstraintViolationState, PointState};
/// fn report<S: ConstraintViolationState>(_: &S) {}
/// report(&PointState::new(vec![1.0]));
/// ```
pub trait ConstraintViolationState: State {
    /// Borrow the report matching the current point, or `None`.
    fn current_violation(&self) -> Option<&ViolationReport<Self::Float>>;

    /// Borrow the report matching the selected point, including before its
    /// objective has been evaluated.
    fn selected_violation(&self) -> Option<&ViolationReport<Self::Float>> {
        self.selected_record()?.violation
    }

    /// Borrow the report matching the selected point, or `None`.
    fn incumbent_violation(&self) -> Option<&ViolationReport<Self::Float>>;
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
pub trait ObjectiveIncumbentState: State {}
