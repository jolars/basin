//! Stopping reasons reported by solvers and execution controls.
//!
//! Configure numerical tolerances on solvers and execution limits or closure
//! hooks on [`Executor`](crate::Executor) and [`RunControl`](crate::RunControl).

/// Why the executor stopped. Returned on
/// [`OptimizationResult::reason`](crate::core::executor::OptimizationResult::reason)
/// and the various step/run hooks.
///
/// Variants are fieldless and preserve numeric casts such as
/// `TerminationReason::MaxIter as u8`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum TerminationReason {
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
    /// (MINPACK `xtol`).
    RelativeParamTolerance,
    /// `|f_k − f_{k−1}| ≤ tol`.
    CostTolerance,
    /// `|f_k − f_{k−1}| ≤ tol · |f_{k−1}|`: scale-invariant cost
    /// reduction test (MINPACK `ftol`).
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

impl TerminationReason {
    /// Whether this reason represents an unrecoverable failure that an
    /// outer solver should bubble (rather than consume and continue).
    ///
    /// Currently only [`SolverFailed`](Self::SolverFailed) qualifies.
    /// [`NumericalNoProgress`](Self::NumericalNoProgress) is a clean inner
    /// stop, although it does not establish convergence or solution accuracy.
    /// [`Cancelled`](Self::Cancelled) is also a clean result rather than a
    /// failure; executor-attached tokens are checked only at top-level
    /// iteration boundaries and do not enter composed inner runs. See
    /// `CONTRIBUTING.md` "Solver composition" for the failure-routing
    /// contract.
    pub fn is_failure(&self) -> bool {
        matches!(self, Self::SolverFailed)
    }
}
