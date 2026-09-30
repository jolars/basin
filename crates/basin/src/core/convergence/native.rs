//! Compatibility access to native solver convergence identities.

/// A native test that participated in a solver's convergence decision.
///
/// These identities supplement [`TerminationCode::SolverConverged`](crate::TerminationCode::SolverConverged)
/// without changing its value or any stopping behavior. They describe the
/// solver's own coordinates and objective, including problem adapters.
/// Passing a test does not establish global optimality or parameter accuracy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum NativeConvergenceTest {
    /// LM's infinity norm of the objective gradient, `‖Jᵀr‖∞ ≤ tolerance`.
    AbsoluteGradient,
    /// LM's maximum absolute cosine between a Jacobian column and the
    /// residual, `max_j |(Jᵀr)_j| / (‖J[:,j]‖ ‖r‖) ≤ tolerance`.
    GradientOrthogonality,
    /// Robust LM's gradient normalized by model column norms and
    /// `sqrt(2 * objective)`, rather than the safeguarded model residual norm.
    RobustGradientOrthogonality,
    /// LM's conjunction of small actual and predicted reductions relative
    /// to the previous objective, with gain ratio at most two.
    RelativeModelReduction,
    /// LM's attempted step satisfies `‖h‖ ≤ tolerance * ‖x‖`, using the
    /// accepted iterate after acceptance and the retained base after rejection.
    /// This differs from movement between published iterates.
    RelativeTrialStep,
    /// LM's updated trust radius satisfies
    /// `radius ≤ tolerance * sqrt(xᵀ D x)` with monotone Marquardt scaling.
    /// This test is inactive under Nielsen damping.
    RelativeTrustRadius,
    /// Full TRF's Coleman–Li scaled gradient satisfies
    /// `max |v ⊙ gradient| ≤ tolerance` over free coordinates.
    AbsoluteScaledGradient,
    /// Full TRF has no free coordinates because every variable is fixed by
    /// equal bounds. No gradient test or Jacobian evaluation is required.
    NoFreeParameters,
}

/// Optional access to the native tests recorded by a solver.
///
/// Implemented by [`LevenbergMarquardt`](crate::LevenbergMarquardt),
/// [`LevenbergMarquardtQr`](crate::LevenbergMarquardtQr),
/// [`TrustRegionReflective`](crate::TrustRegionReflective), and their
/// [`ConfiguredSolver`](super::ConfiguredSolver) wrappers. It adds no
/// requirements to [`Solver`](crate::Solver) or state types.
///
/// Tests are recorded at the stopping decision without extra evaluations.
/// All passing tests at that stage are retained; tests at other stages are
/// not evaluated retroactively. Slice order does not imply precedence.
/// Numerical safeguards, failures, and shared observed convergence checks
/// have their existing [`Termination`](crate::Termination) values
/// and are not native convergence tests.
///
/// Fresh initialization and the next native check or step clear the previous
/// record. Exact checkpoints retain it alongside the solver's history.
/// Consequently, a solver's record can describe an earlier run segment when
/// an executor control stops exact continuation before calling the solver.
/// Prefer [`OptimizationResultWithSolver::native_convergence_tests`](crate::OptimizationResultWithSolver::native_convergence_tests)
/// to inspect a completed run: that accessor derives identities from its owned report.
/// Ordinary results carry the same evidence in `result.report.termination`.
pub trait NativeConvergenceDiagnostics {
    /// Tests recorded at the most recent native convergence decision, or an
    /// empty slice if none is recorded. Reading this never evaluates the problem.
    fn native_convergence_tests(&self) -> &[NativeConvergenceTest];
}
