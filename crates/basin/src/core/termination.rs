//! Stopping reasons reported by solvers and execution controls.
//!
//! Configure numerical tolerances on solvers and execution limits or closure
//! hooks on [`Executor`](crate::Executor) and [`RunControl`](crate::RunControl).

use crate::EvaluationKind;
use crate::core::math::Scalar;
use std::time::Duration;

/// Where an executor published a stopping decision.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum TerminationStage {
    /// Before entering the next solver step, including after initialization.
    Boundary,
    /// During a solver step. Completion is independent of termination.
    Step {
        /// Whether this step completed an iteration.
        completed: bool,
    },
}

/// An owned explanation of one stopping event, independent of solver history.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TerminationReport<F: Scalar = f64> {
    /// The stopping decision and its evidence.
    pub termination: Termination<F>,
    /// Where the decision was published.
    pub stage: TerminationStage,
    /// Number of completed iterations when the report was published.
    pub iteration: u64,
}

/// A stopping decision returned with its explanation at the point of detection.
///
/// Numerical failure publishes a coherent state through `Ok`; typed problem
/// callback errors remain `Result::Err`. Convergence describes the tested
/// predicates and does not certify a global optimum or parameter accuracy.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum Termination<F: Scalar = f64> {
    /// One or more numerical convergence predicates passed.
    Converged(Convergence<F>),
    /// All execution budgets exhausted at the observed boundary.
    Limit(Vec<ExecutionLimit>),
    /// The eligible objective incumbent reached the requested target.
    Target {
        /// Eligible incumbent objective at the stopping boundary.
        cost: F,
        /// Requested objective threshold.
        target: F,
    },
    /// A numerical safeguard or configured progress rule stopped the run.
    Stalled(Stall<F>),
    /// The solver could not continue numerically.
    Failed(NumericalFailure<F>),
    /// The executor cancellation token requested a stop.
    Cancelled,
    /// An application hook requested a stop.
    Application(ApplicationStop),
}

/// All sufficient convergence predicates that passed at the stopping stage.
///
/// A compound predicate remains one criterion; its conditions retain their
/// conjunction. Tests unavailable at this stage are not evaluated retroactively.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(
    feature = "serde",
    serde(bound(deserialize = "F: serde::Deserialize<'de>"))
)]
pub struct Convergence<F: Scalar = f64> {
    #[cfg_attr(
        feature = "serde",
        serde(deserialize_with = "deserialize_criteria")
    )]
    criteria: Vec<ConvergenceCriterion<F>>,
}

#[cfg(feature = "serde")]
fn deserialize_criteria<'de, F, D>(
    deserializer: D,
) -> Result<Vec<ConvergenceCriterion<F>>, D::Error>
where
    F: Scalar + serde::Deserialize<'de>,
    D: serde::Deserializer<'de>,
{
    let criteria =
        <Vec<ConvergenceCriterion<F>> as serde::Deserialize>::deserialize(
            deserializer,
        )?;
    if criteria.is_empty() {
        return Err(serde::de::Error::custom(
            "convergence requires at least one satisfied predicate",
        ));
    }
    Ok(criteria)
}

impl<F: Scalar> Convergence<F> {
    /// Construct a value with the supplied initial explanation.
    pub fn new(criterion: ConvergenceCriterion<F>) -> Self {
        Self {
            criteria: vec![criterion],
        }
    }
    /// Read all passing predicates in reporting order.
    pub fn criteria(&self) -> &[ConvergenceCriterion<F>] {
        &self.criteria
    }
    /// Record another predicate passing at the same observation stage.
    pub fn push(&mut self, criterion: ConvergenceCriterion<F>) {
        self.criteria.push(criterion);
    }
    pub(crate) fn append(&mut self, other: Self) {
        self.criteria.extend(other.criteria);
    }
}

/// One precisely identified predicate and the measurements used to evaluate it.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ConvergenceCriterion<F: Scalar = f64> {
    /// The mathematical predicate that passed.
    pub test: ConvergenceTest,
    /// Measurements from the stopping decision.
    pub evidence: ConvergenceEvidence<F>,
}

/// Mathematical identity of a stopping predicate. Norms and reference points
/// are part of the identity, rather than inferred from a tolerance's name.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum ConvergenceTest {
    /// `||g||_2² <= tolerance²`.
    AbsoluteGradientSquared,
    /// `||g||_2² <= tolerance² * ||g_initial||_2²`.
    RelativeGradientSquared,
    /// `||g||_infinity <= tolerance`.
    AbsoluteGradientInfinity,
    /// `||x - projection(x - g)||_infinity <= tolerance`.
    ProjectedGradient,
    /// L-BFGS-B's bound-clipped gradient infinity norm.
    BoundClippedGradient,
    /// `||x_current - x_previous||_2 <= tolerance`.
    AbsoluteStep,
    /// `||x_current - x_previous||_2 <= tolerance * ||x_current||_2`.
    RelativeStep,
    /// `|f_current - f_previous| <= tolerance`.
    AbsoluteCostChange,
    /// `|f_current - f_previous| <= tolerance * |f_previous|`.
    RelativeCostChange,
    /// Configured maximum vertex distance and cost difference from the first
    /// vertex must both pass. A disabled condition is absent from the evidence.
    Simplex,
    /// `max_j |(Jᵀr)_j| / (||J[:,j]|| * ||r||) <= tolerance`.
    GradientOrthogonality,
    /// Robust gradient normalized by model column norms and `sqrt(2*objective)`.
    RobustGradientOrthogonality,
    /// Small absolute actual and predicted reductions relative to the prior
    /// objective, AND gain ratio at most two.
    RelativeModelReduction,
    /// Attempted `||h||_2 <= tolerance * ||x||_2`, using the accepted point
    /// after acceptance and the retained base after rejection.
    RelativeTrialStep,
    /// Updated radius `<= tolerance * sqrt(xᵀ D x)` with Marquardt scaling.
    RelativeTrustRadius,
    /// Coleman–Li bound-scaled gradient infinity norm on free coordinates.
    AbsoluteScaledGradient,
    /// All parameters are fixed by equal bounds.
    NoFreeParameters,
    /// CMA-ES `sigma * max_i d_i < tolerance`.
    DistributionSize,
    /// Algorithm's radius or mutation step reached its configured floor.
    Radius,
    /// MADS poll size reached its configured floor.
    Mesh,
    /// A solver-specific predicate, also available to external solvers.
    /// `key` should include the solver namespace; `definition` states its math.
    Custom {
        /// Namespaced predicate identifier.
        key: String,
        /// Mathematical definition of the predicate.
        definition: String,
    },
}

/// Evidence for a predicate. Values describe the actual decision, including
/// rejected trials, rather than a reconstruction from the final iterate.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum ConvergenceEvidence<F: Scalar = f64> {
    /// `value <= bound` (or the strict comparison documented by the test).
    UpperBound {
        /// Observed value.
        value: F,
        /// Evaluated comparison bound, including any reference scaling.
        bound: F,
        /// Configured tolerance for this predicate.
        tolerance: F,
        /// Reference value used to form the bound, when applicable.
        reference: Option<F>,
    },
    /// Factored values preserve measurements outside the scalar's exponent
    /// range, as used by LM's safeguarded comparisons.
    FactoredUpperBound {
        /// Observed value.
        value: BinaryValue<F>,
        /// Evaluated comparison bound, including any reference scaling.
        bound: BinaryValue<F>,
        /// Configured tolerance for this predicate.
        tolerance: F,
        /// Reference value used to form the bound, when applicable.
        reference: Option<BinaryValue<F>>,
    },
    /// Actual and predicted model reductions with their common reference cost.
    ModelReduction {
        /// Previous objective minus trial objective.
        actual: F,
        /// Predicted decrease from the local model.
        predicted: F,
        /// Objective used as the comparison reference.
        reference_cost: F,
        /// Actual reduction divided by predicted reduction, with the solver's safeguards.
        gain_ratio: F,
        /// Configured tolerance for this predicate.
        tolerance: F,
    },
    /// The configured size and cost-spread conditions, combined with AND.
    Simplex {
        /// Maximum infinity-norm vertex distance from the first vertex, if enabled.
        size: Option<Threshold<F>>,
        /// Maximum absolute cost difference from the first vertex, if enabled.
        cost_spread: Option<Threshold<F>>,
    },
    /// No numerical comparison was required.
    NoFreeParameters,
    /// Named measurements for a custom predicate. An empty collection means
    /// the predicate exposes no scalar measurements.
    Measurements(Vec<Measurement<F>>),
}

/// A nonnegative measurement represented as `significand * 2^exponent`.
/// Keeping the exponent separate avoids overflow and subnormal underflow in
/// the report. Zero has a zero significand; no normalization is required.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct BinaryValue<F: Scalar = f64> {
    /// Nonnegative scalar factor.
    pub significand: F,
    /// Base-two exponent, kept separate from the scalar range.
    pub exponent: i32,
}

#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
/// A scalar comparison value and its configured tolerance.
pub struct Threshold<F: Scalar = f64> {
    /// Observed value.
    pub value: F,
    /// Configured tolerance for this predicate.
    pub tolerance: F,
}

#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
/// A named scalar measurement produced by a predicate or numerical safeguard.
pub struct Measurement<F: Scalar = f64> {
    /// Measurement name defined by the producing predicate.
    pub name: String,
    /// Observed value.
    pub value: F,
}

impl<F: Scalar> ConvergenceCriterion<F> {
    /// Construct a predicate report with its measured value and bound.
    pub fn upper_bound(
        test: ConvergenceTest,
        value: F,
        bound: F,
        tolerance: F,
        reference: Option<F>,
    ) -> Self {
        Self {
            test,
            evidence: ConvergenceEvidence::UpperBound {
                value,
                bound,
                tolerance,
                reference,
            },
        }
    }
    /// Construct a namespaced predicate with its definition and measurements.
    pub fn custom(
        key: impl Into<String>,
        definition: impl Into<String>,
        measurements: Vec<Measurement<F>>,
    ) -> Self {
        Self {
            test: ConvergenceTest::Custom {
                key: key.into(),
                definition: definition.into(),
            },
            evidence: ConvergenceEvidence::Measurements(measurements),
        }
    }
}

/// Exhausted execution budgets, with observed work (which can exceed a limit
/// when a solver step evaluates a batch before the next boundary).
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ExecutionLimit {
    /// Completed-iteration budget.
    Iterations {
        /// Work observed at the stopping boundary.
        observed: u64,
        /// Configured maximum.
        limit: u64,
    },
    /// Cost-evaluation budget using the state's category mapping.
    CostEvaluations {
        /// Work observed at the stopping boundary.
        observed: u64,
        /// Configured maximum.
        limit: u64,
    },
    /// Raw gradient-evaluation budget.
    GradientEvaluations {
        /// Work observed at the stopping boundary.
        observed: u64,
        /// Configured maximum.
        limit: u64,
    },
    /// Raw category or total-work budget.
    Evaluations {
        /// Counted evaluation category or total work.
        kind: EvaluationKind,
        /// Work observed at the stopping boundary.
        observed: u64,
        /// Configured maximum.
        limit: u64,
    },
    /// Elapsed-time budget for this execution segment.
    Time {
        /// Elapsed duration at the stopping boundary.
        elapsed: Duration,
        /// Configured maximum.
        limit: Duration,
    },
}

#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
/// Progress stopped without establishing convergence.
pub enum Stall<F: Scalar = f64> {
    /// No sufficient improvement in the objective incumbent.
    NoImprovement {
        /// Completed iterations since the reference event.
        iterations: u64,
        /// Configured number of iterations without progress.
        patience: u64,
        /// Configured tolerance for this predicate.
        tolerance: F,
        /// Objective used as the comparison reference.
        reference_cost: F,
        /// Current eligible incumbent objective.
        best_cost: F,
    },
    /// No accepted proposal during the configured interval.
    NoAcceptedMove {
        /// Completed iterations since the reference event.
        iterations: u64,
        /// Configured number of iterations without progress.
        patience: u64,
    },
    /// A numerical safeguard stopped progress without proving convergence.
    Numerical {
        /// Explanation from the producing solver.
        message: String,
        /// Available numerical context for this event.
        measurements: Vec<Measurement<F>>,
    },
}

/// Numerical failure, retaining a nested report when an outer solver cannot
/// consume an inner result. The inner report does not certify outer convergence.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum NumericalFailure<F: Scalar = f64> {
    /// A numerical failure detected by a solver.
    Solver {
        /// Explanation from the producing solver.
        message: String,
        /// Available numerical context for this event.
        measurements: Vec<Measurement<F>>,
    },
    /// An inner result the outer solver cannot consume.
    Inner {
        /// The inner stopping event, retaining its stage and completed-iteration count.
        report: Box<TerminationReport<F>>,
    },
}

/// An application hook's own stopping decision, distinct from convergence.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ApplicationStop {
    /// Application-defined identifier for this stop.
    pub code: String,
}
impl ApplicationStop {
    /// Construct a value with the supplied initial explanation.
    pub fn new(code: impl Into<String>) -> Self {
        Self { code: code.into() }
    }
}

/// Explicit policy for consuming a partial inner solution.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PartialResultPolicy {
    /// Allow limits, targets, and stalls to provide partial progress.
    Consume,
    /// Consume only a converged inner result.
    RequireConvergence,
}

impl<F: Scalar> Termination<F> {
    /// Native identities retained for diagnostic consumers migrating from 1.x.
    /// The returned list is derived from this event, never from solver history.
    pub fn native_convergence_tests(
        &self,
    ) -> Vec<crate::NativeConvergenceTest> {
        use crate::NativeConvergenceTest as N;
        use ConvergenceTest as C;
        let Self::Converged(c) = self else {
            return vec![];
        };
        c.criteria()
            .iter()
            .filter_map(|criterion| {
                Some(match criterion.test {
                    C::AbsoluteGradientInfinity => N::AbsoluteGradient,
                    C::GradientOrthogonality => N::GradientOrthogonality,
                    C::RobustGradientOrthogonality => {
                        N::RobustGradientOrthogonality
                    }
                    C::RelativeModelReduction => N::RelativeModelReduction,
                    C::RelativeTrialStep => N::RelativeTrialStep,
                    C::RelativeTrustRadius => N::RelativeTrustRadius,
                    C::AbsoluteScaledGradient => N::AbsoluteScaledGradient,
                    C::NoFreeParameters => N::NoFreeParameters,
                    _ => return None,
                })
            })
            .collect()
    }
    /// Stop because the supplied numerical predicate passed.
    pub fn converged(criterion: ConvergenceCriterion<F>) -> Self {
        Self::Converged(Convergence::new(criterion))
    }
    /// Whether this decision establishes its reported convergence predicates.
    pub fn is_converged(&self) -> bool {
        matches!(self, Self::Converged(_))
    }
    /// Whether this decision reports a numerical failure.
    pub fn is_failure(&self) -> bool {
        matches!(self, Self::Failed(_))
    }
    /// Whether an outer algorithm may consume this result under its explicit
    /// partial-result policy. Cancellation, application stops, and failures
    /// require routing rather than silently continuing.
    pub fn can_continue_as_inner(&self, policy: PartialResultPolicy) -> bool {
        self.is_converged()
            || (policy == PartialResultPolicy::Consume
                && matches!(
                    self,
                    Self::Limit(_) | Self::Target { .. } | Self::Stalled(_)
                ))
    }
    /// Stop with a numerical failure and a published state.
    pub fn numerical_failure(message: impl Into<String>) -> Self {
        Self::Failed(NumericalFailure::Solver {
            message: message.into(),
            measurements: vec![],
        })
    }
    /// Stop on a numerical safeguard without claiming convergence.
    pub fn numerical_stall(message: impl Into<String>) -> Self {
        Self::Stalled(Stall::Numerical {
            message: message.into(),
            measurements: vec![],
        })
    }
    pub(crate) fn merge(self, other: Self) -> Self {
        match (self, other) {
            (Self::Converged(mut a), Self::Converged(b)) => {
                a.append(b);
                Self::Converged(a)
            }
            (Self::Converged(_), b) => b,
            (_, b) if b.is_failure() => b,
            (a, _) => a,
        }
    }
}

impl<F: Scalar> TerminationReport<F> {
    /// Compact compatibility code. Inspect the termination for the full report.
    pub fn code(&self) -> TerminationCode {
        self.termination.code()
    }
    /// Consume a converged inner result, or explicitly accept its partial
    /// progress. Cancellation and application stops propagate as stops.
    pub fn into_outer_termination(
        self,
        policy: PartialResultPolicy,
    ) -> Option<Termination<F>> {
        match self.termination {
            Termination::Converged(_) => None,
            Termination::Cancelled => Some(Termination::Cancelled),
            Termination::Application(stop) => {
                Some(Termination::Application(stop))
            }
            Termination::Limit(_)
            | Termination::Target { .. }
            | Termination::Stalled(_)
                if policy == PartialResultPolicy::Consume =>
            {
                None
            }
            _ => Some(Termination::Failed(NumericalFailure::Inner {
                report: Box::new(self),
            })),
        }
    }
}

impl<F: Scalar> Termination<F> {
    /// Construct a predicate report with its measured value and bound.
    pub fn upper_bound(
        test: ConvergenceTest,
        value: F,
        bound: F,
        tolerance: F,
        reference: Option<F>,
    ) -> Self {
        Self::converged(ConvergenceCriterion::upper_bound(
            test, value, bound, tolerance, reference,
        ))
    }
    /// Construct a namespaced predicate with its definition and measurements.
    pub fn custom(
        key: impl Into<String>,
        definition: impl Into<String>,
        measurements: Vec<Measurement<F>>,
    ) -> Self {
        Self::converged(ConvergenceCriterion::custom(
            key,
            definition,
            measurements,
        ))
    }
    /// A compact compatibility identifier. With simultaneous predicates, this
    /// identifies the first reported predicate; use the full report to inspect
    /// every condition. Its numeric value is explicitly mapped by `as_u8()`.
    pub fn code(&self) -> TerminationCode {
        use ConvergenceTest as C;
        use TerminationCode as R;
        match self {
            Self::Converged(c) => match &c.criteria()[0].test {
                C::AbsoluteGradientSquared => R::GradientTolerance,
                C::RelativeGradientSquared => R::RelativeGradientTolerance,
                C::ProjectedGradient | C::BoundClippedGradient => {
                    R::ProjectedGradientTolerance
                }
                C::AbsoluteStep => R::ParamTolerance,
                C::RelativeStep => R::RelativeParamTolerance,
                C::AbsoluteCostChange => R::CostTolerance,
                C::RelativeCostChange => R::RelativeCostTolerance,
                C::Simplex => R::SimplexTolerance,
                C::DistributionSize => R::CmaEsTolerance,
                C::Radius => R::RhoTolerance,
                C::Mesh => R::MeshTolerance,
                _ => R::SolverConverged,
            },
            Self::Limit(limits) => match limits.first() {
                Some(ExecutionLimit::Iterations { .. }) => R::MaxIter,
                Some(ExecutionLimit::CostEvaluations { .. }) => R::MaxCostEvals,
                Some(ExecutionLimit::GradientEvaluations { .. }) => {
                    R::MaxGradientEvals
                }
                Some(ExecutionLimit::Time { .. }) => R::MaxTime,
                _ => R::MaxEvaluations,
            },
            Self::Target { .. } => R::TargetCost,
            Self::Stalled(Stall::NoImprovement { .. }) => R::NoImprovement,
            Self::Stalled(Stall::NoAcceptedMove { .. }) => R::NoAcceptedMove,
            Self::Stalled(Stall::Numerical { .. }) => R::NumericalNoProgress,
            Self::Failed(_) => R::SolverFailed,
            Self::Cancelled => R::Cancelled,
            Self::Application(_) => R::UserRequested,
        }
    }
}

/// Compatibility identifiers for bindings and compact status displays.
///
/// Use [`Self::as_u8`] for the stable Basin 1.x numeric mapping. The
/// structured [`Termination`] carries the complete stopping explanation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum TerminationCode {
    /// `state.iter() >= max_iter`.
    MaxIter,
    /// Cost-evaluation budget exhausted.
    MaxCostEvals,
    /// Gradient-evaluation budget exhausted.
    MaxGradientEvals,
    /// `‖∇f(x)‖ ≤ tol`.
    GradientTolerance,
    /// `‖∇f(x_k)‖ ≤ tol · ‖∇f(x_0)‖`: gradient norm relative to the
    /// initial gradient (scale-invariant first-order stationarity).
    RelativeGradientTolerance,
    /// `‖x − π_C(x − ∇f(x))‖_∞ ≤ tol`: projected-gradient stationarity
    /// for box-constrained problems. Collapses to the unconstrained
    /// gradient norm when no constraint is active.
    ProjectedGradientTolerance,
    /// `‖x_k − x_{k−1}‖ ≤ tol`.
    ParamTolerance,
    /// `‖x_k − x_{k−1}‖ ≤ tol · ‖x_k‖`: scale-invariant step test
    /// relative to the current iterate.
    RelativeParamTolerance,
    /// `|f_k − f_{k−1}| ≤ tol`.
    CostTolerance,
    /// `|f_k − f_{k−1}| ≤ tol · |f_{k−1}|`: scale-invariant cost
    /// change test relative to the previous objective.
    RelativeCostTolerance,
    /// `f(x_k) ≤ target`: user-supplied target cost reached
    /// (NLopt's `stopval`/SciPy's `f_min`).
    TargetCost,
    /// Best-so-far cost has not improved by more than `tol` in
    /// `patience` consecutive iterations: the early-stopping pattern.
    NoImprovement,
    /// No proposal has been accepted for the configured number of iterations.
    NoAcceptedMove,
    /// Simplex collapsed below the configured tolerance.
    SimplexTolerance,
    /// CMA-ES search distribution collapsed below TolX:
    /// `σ · maxᵢ dᵢ < tol_x` (Hansen 2016 Appendix B.3).
    CmaEsTolerance,
    /// Trust-region radius or step size `ρ` reached the configured
    /// floor: `ρ ≤ rho_end` (Powell-family trust region, Solis-Wets
    /// mutation step).
    RhoTolerance,
    /// MADS poll size reached the configured floor: `Δᵖ ≤ poll_size_min`.
    MeshTolerance,
    /// Wall-clock time limit reached.
    MaxTime,
    /// Cancellation was requested through the executor's cancellation token.
    Cancelled,
    /// An application-specific stopping hook requested termination.
    UserRequested,
    /// Solver determined it has converged (e.g. fixed point reached).
    SolverConverged,
    /// Solver cannot make further progress (e.g. line search failure).
    SolverFailed,
    /// A numerical safeguard stopped the solver without establishing convergence.
    ///
    /// Levenberg-Marquardt reports this after a rejected finite trial whose
    /// computed step leaves every parameter unchanged in floating-point
    /// arithmetic. Trust-region-reflective least squares reports it when a
    /// finite equal-cost rejection is followed by a contracted trial that
    /// changes no parameter. Trust-region minimization reports it when the
    /// predicted reduction is finite and non-positive but the computed
    /// gradient norm is nonzero. The returned point may be inaccurate, heavily
    /// damped, or limited by a small trust radius; this reason does not
    /// establish stationarity or parameter recovery. Outer solvers may consume
    /// the finite result and continue.
    NumericalNoProgress,
    /// A raw evaluation category or total-work budget was exhausted.
    ///
    /// Checked at iteration boundaries through the state's raw-count
    /// capability; distinct from the legacy cost and gradient budget reasons.
    /// Compare [`RawEvaluationState::raw_counts`](crate::RawEvaluationState::raw_counts)
    /// with the configured limits to identify exhausted budgets.
    MaxEvaluations,
}

impl TerminationCode {
    /// Whether this compact identifier denotes a numerical failure.
    /// Prefer [`TerminationReport::into_outer_termination`] with an explicit
    /// [`PartialResultPolicy`] when routing a composed solver's result.
    pub fn is_failure(&self) -> bool {
        matches!(self, Self::SolverFailed)
    }
}

impl TerminationCode {
    /// Stable numeric binding code, preserving the Basin 1.x mapping (0–23).
    ///
    /// | Code | Identifier |
    /// | --- | --- |
    /// | 0 | `MaxIter` |
    /// | 1 | `MaxCostEvals` |
    /// | 2 | `MaxGradientEvals` |
    /// | 3 | `GradientTolerance` |
    /// | 4 | `RelativeGradientTolerance` |
    /// | 5 | `ProjectedGradientTolerance` |
    /// | 6 | `ParamTolerance` |
    /// | 7 | `RelativeParamTolerance` |
    /// | 8 | `CostTolerance` |
    /// | 9 | `RelativeCostTolerance` |
    /// | 10 | `TargetCost` |
    /// | 11 | `NoImprovement` |
    /// | 12 | `NoAcceptedMove` |
    /// | 13 | `SimplexTolerance` |
    /// | 14 | `CmaEsTolerance` |
    /// | 15 | `RhoTolerance` |
    /// | 16 | `MeshTolerance` |
    /// | 17 | `MaxTime` |
    /// | 18 | `Cancelled` |
    /// | 19 | `UserRequested` |
    /// | 20 | `SolverConverged` |
    /// | 21 | `SolverFailed` |
    /// | 22 | `NumericalNoProgress` |
    /// | 23 | `MaxEvaluations` |
    pub const fn as_u8(self) -> u8 {
        match self {
            Self::MaxIter => 0,
            Self::MaxCostEvals => 1,
            Self::MaxGradientEvals => 2,
            Self::GradientTolerance => 3,
            Self::RelativeGradientTolerance => 4,
            Self::ProjectedGradientTolerance => 5,
            Self::ParamTolerance => 6,
            Self::RelativeParamTolerance => 7,
            Self::CostTolerance => 8,
            Self::RelativeCostTolerance => 9,
            Self::TargetCost => 10,
            Self::NoImprovement => 11,
            Self::NoAcceptedMove => 12,
            Self::SimplexTolerance => 13,
            Self::CmaEsTolerance => 14,
            Self::RhoTolerance => 15,
            Self::MeshTolerance => 16,
            Self::MaxTime => 17,
            Self::Cancelled => 18,
            Self::UserRequested => 19,
            Self::SolverConverged => 20,
            Self::SolverFailed => 21,
            Self::NumericalNoProgress => 22,
            Self::MaxEvaluations => 23,
        }
    }
}
