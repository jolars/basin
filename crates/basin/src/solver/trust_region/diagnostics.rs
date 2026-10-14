//! Optional native trust-region trial diagnostics.

use crate::{EvalCounts, Scalar};

/// Last phase reached by a trust-region attempt, including interrupted calls.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    /// The strategy is computing its model step.
    Subproblem,
    /// The strategy returned non-finite model reduction.
    InvalidModel,
    /// The model cannot decrease in native arithmetic.
    NoProgress,
    /// A cost call is active and may have failed.
    TrialEvaluation,
    /// An accepted trial's gradient call is active and may have failed.
    GradientRefresh,
    /// The trial was accepted with a coherent evaluation.
    Accepted,
    /// The trial was rejected, retaining the old recommendation.
    Rejected,
}

/// Native scalar measurements at one inner attempt.
///
/// Records persist through solver-aware continuation and reset at initialization.
/// Matrix factorization and strategy-internal residual comparisons are unavailable.
#[derive(Clone, Debug)]
pub struct Trial<F: Scalar = f64> {
    /// Monotone sequence within this solve.
    pub sequence: u64,
    /// Current attempt phase.
    pub phase: Phase,
    /// Radius used to form the step.
    pub radius_before: F,
    /// Radius after the acceptance decision, if available.
    pub radius_after: Option<F>,
    /// Squared Euclidean norm of the model gradient.
    pub gradient_norm_squared: F,
    /// Squared Euclidean norm of the strategy's step.
    pub step_norm_squared: Option<F>,
    /// Native predicted model reduction.
    pub predicted_reduction: Option<F>,
    /// Native trial cost, if the callback completed.
    pub trial_cost: Option<F>,
    /// Actual-to-predicted reduction ratio.
    pub ratio: Option<F>,
    /// Whether the strategy classified its step on the boundary.
    pub hit_boundary: Option<bool>,
    /// Acceptance threshold used by the outer method.
    pub eta: F,
    /// Public logical counts before the strategy call.
    pub counts_before: EvalCounts,
    /// Public logical counts at the most recent completed phase.
    pub counts_after: EvalCounts,
}
