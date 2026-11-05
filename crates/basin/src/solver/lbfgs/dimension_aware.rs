//! Opt-in dimension-aware reduced solves for L-BFGS-B.

use super::subspace_direction::Workspace;
use super::*;

/// L-BFGS-B with opt-in dimension-aware subspace directions.
///
/// Construct through [`Lbfgs::with_dimension_aware_subspace`], after configuring
/// the underlying solver. Fully free models use inverse two-loop recursion
/// when the parameter dimension is no larger than twice the live history size.
/// A single free coordinate can use a direct reduced BFGS solve once at least
/// three curvature pairs are available.
/// Other cases, or numerically unusable candidates, use the reference solve.
/// Basin's compact-factor checks, Cauchy point, projected-step/backtracking
/// safeguard, line search, and stopping controls remain in effect.
///
/// These are equivalent model solves in exact arithmetic, but rounding can
/// change iterates and callbacks. The default [`Lbfgsb`] keeps its Fortran
/// reference trajectory. Compact factorization is still performed before an
/// alternate direction, preserving its cache and failure/restart contract.
/// Exact checkpoints retain the model and configuration. Serialized checkpoints
/// omit only scratch buffers, which are overwritten before use on continuation.
///
/// ```
/// let solver = basin::Lbfgsb::<Vec<f64>>::new()
///     .with_m_capacity(10)
///     .with_dimension_aware_subspace();
/// assert_eq!(solver.solver().history_capacity(), 10);
/// ```
///
/// # Backends
///
/// Same contiguous `Vec<F>`, nalgebra `DVector<F>`, ndarray `Array1<F>`, and
/// faer `Col<F>` backends and custom slice capabilities as [`Lbfgs`], for both
/// `f32` and `f64`. Custom line searches have the same bounded-search requirement.
///
/// # References
///
/// Uses the BFGS and compact-model identities referenced by [`Lbfgs`]. The
/// dimension-dependent selection follows sisyfuzz's `lbfgsb/model.rs` approach;
/// it retains Basin's L-BFGS-B 3.0 projection rather than sisyfuzz's ray truncation.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DimensionAwareLbfgsb<V, F: Scalar = f64, S = MoreThuente<F>> {
    solver: Lbfgsb<V, F, S>,
    #[cfg_attr(feature = "serde", serde(skip))]
    work: Workspace<F>,
}

impl<V, F: Scalar, S> DimensionAwareLbfgsb<V, F, S> {
    pub(super) fn new(solver: Lbfgsb<V, F, S>) -> Self {
        Self {
            solver,
            work: Workspace::default(),
        }
    }

    /// Inspect the configured solver, history sizes and model observations.
    pub fn solver(&self) -> &Lbfgsb<V, F, S> {
        &self.solver
    }

    /// Recover the reference solver and its current history by ownership.
    /// Continuing with it selects the reference subspace arithmetic.
    pub fn into_inner(self) -> Lbfgsb<V, F, S> {
        self.solver
    }
}

impl<P, V, S, F> Solver<P, FirstOrderState<V, F>>
    for DimensionAwareLbfgsb<V, F, S>
where
    F: Scalar,
    P: CostFunction<Param = V, Output = F>
        + Gradient<Gradient = V>
        + BoxProblem<Param = V>,
    V: AsFloatSliceMut<F> + Clone + Dot<F> + ScaledAdd<F> + VectorLen,
    S: LineSearch<P, V, F, Error = P::Error>,
{
    type Error = crate::ConstraintSolveError<P::Error>;

    fn init(
        &mut self,
        problem: &mut Problem<P>,
        state: FirstOrderState<V, F>,
    ) -> Result<FirstOrderState<V, F>, Self::Error> {
        self.work.reset();
        self.solver.init(problem, state)
    }
    fn next_iter(
        &mut self,
        problem: &mut Problem<P>,
        state: FirstOrderState<V, F>,
    ) -> Result<crate::SolverStep<FirstOrderState<V, F>>, Self::Error> {
        self.solver
            .next_iter_bounded(problem, state, Some(&mut self.work))
    }
    fn reset_convergence(&mut self) {
        <Lbfgsb<V, F, S> as Solver<P, FirstOrderState<V, F>>>::reset_convergence(&mut self.solver);
    }
    fn should_check_iterate_change(&self) -> bool {
        <Lbfgsb<V, F, S> as Solver<P, FirstOrderState<V, F>>>::should_check_iterate_change(&self.solver)
    }
    fn check_convergence(
        &mut self,
        problem: &Problem<P>,
        state: &FirstOrderState<V, F>,
    ) -> Option<Termination<F>> {
        self.solver.check_convergence(problem, state)
    }
    fn terminate(
        &self,
        state: &FirstOrderState<V, F>,
    ) -> Option<Termination<F>> {
        <Lbfgsb<V, F, S> as Solver<P, FirstOrderState<V, F>>>::terminate(
            &self.solver,
            state,
        )
    }
}
