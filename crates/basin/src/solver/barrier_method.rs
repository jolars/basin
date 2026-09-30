//! Log-barrier (sequential unconstrained minimization) method for linear
//! inequality constraints `A x ≤ b`.

use crate::core::barrier::{LogBarrier, strict_feasibility};
use crate::core::constraint::LinearInequalityConstraints;
use crate::core::executor::run_loop_with_control;
use crate::core::inner::{InitialState, WarmStart};
use crate::core::math::{
    MatTransposeVec, MatVec, NegInPlace, NormSquared, Scalar, ScaledAdd,
    VectorIndex, VectorLen,
};
use crate::core::problem::{CostFunction, Gradient, Problem};
use crate::core::solver::Solver;
use crate::core::state::{CountsMirror, PointState, State};
use crate::core::termination::Termination;

/// Two-phase log-barrier method for `min f(x) s.t. A x ≤ b`, layering a
/// barrier on an unconstrained inner solver.
///
/// Phase I automatically finds a strictly feasible point when the supplied
/// start does not satisfy `A x₀ < b`. Phase II then minimizes the log-barrier
/// objective
/// `φ_μ(x) = f(x) − μ · Σ log(bᵢ − aᵢᵀ x)` (the [`LogBarrier`] adapter) with
/// the inner solver `So`, warm-started from the current iterate, then
/// shrinks `μ`. As `μ → 0` the central path converges to the constrained
/// optimum.
///
/// The method is generic over inner solvers that implement [`WarmStart`].
/// The inner state is seeded at the
/// current iterate via [`InitialState::seed`],
/// so each of [`GradientDescent`](crate::solver::GradientDescent)
/// ([`FirstOrderState`](crate::FirstOrderState)), [`Bfgs`](crate::solver::Bfgs)
/// ([`FirstOrderState`](crate::core::state::FirstOrderState)), and unbounded
/// [`Lbfgs`](crate::solver::lbfgs::Lbfgs)
/// ([`FirstOrderState`](crate::core::state::FirstOrderState)) is usable. A
/// least-squares solver
/// ([`LevenbergMarquardt`](crate::solver::LevenbergMarquardt)), because the
/// barrier objective is not a sum of squares and the [`LogBarrier`] adapter
/// exposes only `CostFunction + Gradient`, not `Residual + Jacobian`.
/// A derivative-free solver such as [`NelderMead`](crate::solver::NelderMead)
/// may find a strictly feasible point in Phase I, but
/// cannot provide the gradient convergence certificate used to report an empty
/// strict interior.
///
/// **During Phase II the inner solver must keep iterates feasible.**
/// Feasibility is enforced only by the barrier returning `+∞` outside the
/// feasible set, so the inner solver's step acceptance has to honor that wall:
/// pair the inner with an
/// **Armijo backtracking** line search
/// ([`Backtracking`](crate::line_search::Backtracking)), which shrinks any
/// step whose cost is `+∞`. A fixed step ([`Constant`](crate::line_search::Constant))
/// can overshoot the boundary, and strong-Wolfe searches
/// ([`MoreThuente`](crate::line_search::MoreThuente),
/// [`Wolfe`](crate::line_search::Wolfe)) can stall bracketing on the `+∞`
/// wall; for `GradientDescent`, momentum
/// ([`with_momentum`](crate::solver::GradientDescent::with_momentum)) adds
/// velocity outside the line search's control and can carry the iterate
/// straight through the barrier.
///
/// # Algorithm
///
/// Boyd & Vandenberghe, *Convex Optimization* §11.4.1 followed by §11.3
/// (Alg. 11.1), in the `μ`-shrinking parametrization:
///
/// ```text
/// if A x₀ < b does not hold:
///   solve min s subject to A x - b ≤ s       # Phase I
///   stop Phase I as soon as A x < b
///   if centered and m · μ ≤ phase_one_tol: fail without a strict point
/// μ ← mu0
/// repeat:
///   x ← argmin φ_μ                              # Phase II
///   if m · μ ≤ tol: stop (SolverConverged)   # log-barrier duality gap
///   μ ← μ / reduction
/// ```
///
/// `m · μ` is the exact suboptimality bound for the log barrier (`m` =
/// number of constraints), so the returned iterate is within `tol` of the
/// constrained optimum.
///
/// # Phase I
///
/// The auxiliary scalar is eliminated analytically, so the configured inner
/// solver still works with the original parameter type `V`: for violations
/// `rᵢ = aᵢᵀx - bᵢ`, the reduced objective chooses the unique
/// `s > max rᵢ` satisfying `μ Σᵢ 1/(s-rᵢ) = 1`. Its gradient is
/// `Aᵀ[μ/(s-r)]`. This needs only the same matvec operations as Phase II and
/// therefore preserves every backend. The inner run stops immediately when it
/// produces `A x < b`; Phase II then restarts the `μ` schedule at `mu0`.
///
/// If a centered Phase I subproblem reaches
/// [`with_absolute_phase_one_gap_tolerance`](Self::with_absolute_phase_one_gap_tolerance)
/// without finding a strict point, the constraints are reported as
/// [`SolverFailed`](crate::TerminationCode::SolverFailed). An inner solve that
/// exhausts its iteration budget is not a certificate: Phase I retries the
/// same `μ` from the returned candidate. Numerically, the centered certificate
/// means "no strict interior at the configured scale": exact zero margin
/// cannot be distinguished from an arbitrarily thin interior in finite
/// precision.
///
/// # Termination
///
/// The outer duality-gap test `m · μ ≤ tol` is solver-specific and lives on
/// the solver (tenet 3): it fires via [`terminate`](Solver::terminate) as
/// [`SolverConverged`](crate::TerminationCode::SolverConverged). Pair with the
/// executor's [`max_iter`](crate::Executor::max_iter) as a safety net. A strictly feasible start uses only
/// Phase II; an infeasible or boundary start spends additional outer
/// iterations in Phase I. With the defaults each continuation closes its gap
/// in roughly `log(m · mu0 / tol) / log(reduction)` outer iterations
/// (≈ 9 for the defaults), so an outer `max_iter` of 30–50 is a practical
/// safety budget for both phases.
///
/// Cost- and step-change tests observe only accepted Phase II iterates.
/// The outer [`PointState`] has no gradient capability: the original
/// objective gradient need not vanish at a constrained optimum. Configure
/// gradient convergence on the inner solver.
///
/// # Progress and lifecycle
///
/// The outer state reports the original objective inside the strict domain
/// `A x < b`, and `+∞` elsewhere. Phase I computes this rejection value
/// without calling the original objective or gradient. Its current record is
/// available, but it cannot establish an incumbent until a strictly feasible
/// point with an eligible objective is published. Incumbents retain strict
/// objective improvements among these feasible points.
///
/// Fresh initialization clears progress, reevaluates the seed's domain and
/// objective, and restarts the phase and barrier schedule. Each inner solve
/// starts fresh. Exact continuation retains the phase, schedule, inner solver,
/// progress, and all six raw evaluation categories, and skips initialization.
/// The outer loop does not compute an unused original-objective gradient.
/// Inner adapter evaluations remain charged even when a callback aborts.
///
/// With `serde`, solver-aware checkpoints can serialize the outer method
/// whenever the inner solver supports serialization. Old `BasicState`
/// checkpoint payloads are incompatible with this progress representation.
///
/// # Backends
///
/// Supports `Vec<f32>` and `Vec<f64>` with
/// [`DenseMatrix`](crate::core::math::DenseMatrix), nalgebra `DVector`/`DMatrix`,
/// ndarray `Array1`/`Array2`, and faer `Col`/`Mat`, with either scalar type.
/// The outer method needs [`MatVec`] and [`MatTransposeVec`], with no linear
/// solve. Its vector bounds are [`ScaledAdd`], [`NegInPlace`], [`VectorIndex`],
/// [`VectorLen`], [`NormSquared`], and [`Clone`]. A custom inner solver can
/// impose additional capabilities.
///
/// # Composition
///
/// Use [`with_inner_solver`](Self::with_inner_solver) to supply a solver with
/// its own convergence settings. Each outer iteration starts a fresh inner
/// solve through [`run_loop_with_control`],
/// against the current surrogate problem. Convergence history resets and
/// inner evaluation counts are added to their corresponding outer categories.
///
/// # Examples
///
/// `BarrierMethod` wraps a gradient inner solver (e.g. `BFGS` paired with
/// `Backtracking`) to handle `LinearInequalityConstraints`. See
/// [`ProjectedGradientDescent`](crate::solver::ProjectedGradientDescent)
/// for the simpler box-constrained pattern.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct BarrierMethod<So, F: Scalar = f64> {
    inner_solver: So,
    inner_max_iter: u64,
    mu0: F,
    mu: F,
    reduction: F,
    tol: Option<F>,
    phase_one_tol: F,
    /// `m · μ` of the most recent inner solve; `+∞` until the first solve
    /// so [`terminate`](Solver::terminate) cannot fire at iter 0.
    gap: F,
    phase: BarrierPhase,
    accepted_iterate: bool,
}

#[derive(Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
enum BarrierPhase {
    PhaseOne,
    PhaseTwo,
    Failed,
}

/// Delegates to the configured inner solver but ends a Phase I inner run as
/// soon as the current parameter is strictly feasible for the original
/// constraints. This is important when the auxiliary LP is unbounded below:
/// Phase I only needs one interior point, not the LP optimum.
struct StopAtStrictFeasibility<'a, So> {
    inner: &'a mut So,
}

impl<'a, 'p, P, V, M, S, So, F> Solver<LogBarrier<'p, P, F>, S>
    for StopAtStrictFeasibility<'a, So>
where
    F: Scalar,
    P: CostFunction<Param = V, Output = F>
        + LinearInequalityConstraints<Param = V, Matrix = M>,
    M: MatVec<V>,
    V: ScaledAdd<F> + VectorIndex<F> + VectorLen,
    S: State<Param = V>,
    So: Solver<LogBarrier<'p, P, F>, S>,
{
    type Error = So::Error;

    fn init(
        &mut self,
        problem: &mut Problem<LogBarrier<'p, P, F>>,
        state: S,
    ) -> Result<S, Self::Error> {
        self.inner.init(problem, state)
    }

    fn next_iter(
        &mut self,
        problem: &mut Problem<LogBarrier<'p, P, F>>,
        state: S,
    ) -> Result<crate::SolverStep<S>, Self::Error> {
        let crate::SolverStep {
            state,
            termination: reason,
            completed,
        } = self.inner.next_iter(problem, state)?;
        let inner_failed = reason.as_ref().is_some_and(|reason| {
            !reason.can_continue_as_inner(crate::PartialResultPolicy::Consume)
        });
        if !inner_failed
            && problem.inner().strict_feasibility(state.param()) == Some(true)
        {
            Ok(crate::SolverStep {
                state,
                completed,
                termination: Some(Termination::custom(
                    "barrier.strict_feasibility",
                    "All inequality slacks are strictly positive.",
                    vec![],
                )),
            })
        } else {
            Ok(crate::SolverStep {
                state,
                termination: reason,
                completed,
            })
        }
    }

    fn terminate(&self, state: &S) -> Option<Termination<S::Float>> {
        self.inner.terminate(state)
    }

    fn reset_convergence(&mut self) {
        self.inner.reset_convergence();
    }
    fn check_convergence(
        &mut self,
        problem: &Problem<LogBarrier<'p, P, F>>,
        state: &S,
    ) -> Option<Termination<S::Float>> {
        self.inner.check_convergence(problem, state)
    }
}

impl<So, F: Scalar> BarrierMethod<So, F> {
    /// Build a barrier method around an unconstrained inner solver.
    ///
    /// Defaults: `mu0 = 1.0`, `reduction = 10.0`, `tol = 1e-8`,
    /// `phase_one_tol = 1e-8`, and `inner_max_iter = 50`.
    ///
    /// The `inner_max_iter` default is intentionally modest:
    /// [`with_inner_max_iter`](Self::with_inner_max_iter) is the dominant cost lever
    /// (see its docs) and the outer μ-continuation tolerates loosely-centered
    /// subproblems, so a small budget usually converges to the same point far
    /// more cheaply than a large one.
    /// The supplied inner solver owns its convergence settings.
    pub fn with_inner_solver(inner_solver: So) -> Self {
        Self {
            inner_solver,
            inner_max_iter: 50,
            mu0: F::one(),
            mu: F::one(),
            reduction: F::from_f64(10.0).unwrap(),
            tol: Some(F::from_f64(1e-8).unwrap()),
            phase_one_tol: F::from_f64(1e-8).unwrap(),
            gap: F::infinity(),
            phase: BarrierPhase::PhaseTwo,
            accepted_iterate: false,
        }
    }
}

impl<So, F: Scalar> BarrierMethod<So, F> {
    /// Initial barrier parameter `μ` (default `1.0`).
    ///
    /// # Panics
    ///
    /// Panics unless `mu0 > 0`; a non-positive `μ` is not a barrier.
    pub fn mu0(mut self, mu0: F) -> Self {
        assert!(mu0 > F::zero(), "mu0 must be > 0");
        self.mu0 = mu0;
        self
    }

    /// Per-outer-iteration shrink factor: `μ ← μ / reduction` (default
    /// `10.0`).
    ///
    /// # Panics
    ///
    /// Panics unless `reduction > 1`; otherwise `μ` would not shrink and
    /// the duality gap would never close.
    pub fn with_reduction(mut self, reduction: F) -> Self {
        assert!(reduction > F::one(), "reduction must be > 1");
        self.reduction = reduction;
        self
    }

    /// Stop when the outer duality gap `m · μ` reaches this tolerance
    /// (default `1e-8`). `None` disables the check.
    pub fn with_absolute_duality_gap_tolerance(
        mut self,
        tol: impl Into<Option<F>>,
    ) -> Self {
        self.tol = crate::core::convergence::optional_tolerance(tol);
        self
    }

    /// Set the Phase I accuracy used to classify a constraint system with no
    /// strict interior (default `1e-8`). If a centered Phase I subproblem has
    /// not found `A x < b` once its auxiliary duality gap `m · μ` reaches this
    /// tolerance, the solver reports
    /// [`SolverFailed`](crate::TerminationCode::SolverFailed).
    ///
    /// Finite precision cannot distinguish an empty interior from an
    /// arbitrarily thin one. This tolerance sets the numerical scale for that
    /// decision and must be positive.
    pub fn with_absolute_phase_one_gap_tolerance(
        mut self,
        phase_one_tol: F,
    ) -> Self {
        assert!(phase_one_tol > F::zero(), "phase_one_tol must be > 0");
        self.phase_one_tol = phase_one_tol;
        self
    }

    /// Iteration budget for each inner barrier-subproblem solve (default
    /// `50`).
    ///
    /// **This is the dominant cost lever.** A first-order inner solver
    /// (`GradientDescent`) on the ill-conditioned barrier typically exhausts
    /// this budget rather than its own gradient convergence tolerance,
    /// so total work scales roughly linearly with it. Because the outer
    /// μ-continuation re-solves at each shrinking `μ`, a loosely-centered
    /// (small-budget) subproblem usually still converges to the same point,
    /// often an order of magnitude cheaper. Raise it for hard or higher-
    /// dimensional problems; a Newton-class inner (future work) would center
    /// in far fewer steps and reach its configured tolerance instead.
    ///
    /// # Panics
    ///
    /// Panics unless `inner_max_iter ≥ 1` (a zero budget would never move the
    /// iterate).
    pub fn with_inner_max_iter(mut self, inner_max_iter: u64) -> Self {
        assert!(inner_max_iter >= 1, "inner_max_iter must be ≥ 1");
        self.inner_max_iter = inner_max_iter;
        self
    }
}

impl<So, V, F> InitialState<V> for BarrierMethod<So, F>
where
    F: Scalar,
    V: Clone,
{
    type State = PointState<V, F>;
    fn seed(&self, x: &V) -> Self::State {
        PointState::new(x.clone())
    }
}

impl<P, V, M, So, F> Solver<P, PointState<V, F>> for BarrierMethod<So, F>
where
    F: Scalar,
    P: CostFunction<Param = V, Output = F>
        + Gradient<Gradient = V>
        + LinearInequalityConstraints<Param = V, Matrix = M>,
    M: MatVec<V> + MatTransposeVec<V>,
    V: ScaledAdd<F>
        + NegInPlace
        + VectorIndex<F>
        + VectorLen
        + NormSquared<F>
        + Clone,
    So: WarmStart<V>
        + for<'a> Solver<
            LogBarrier<'a, P, F>,
            So::State,
            Error = <P as CostFunction>::Error,
        >,
    So::State: State<Param = V, Float = F> + CountsMirror,
{
    type Error = <P as CostFunction>::Error;

    fn init(
        &mut self,
        problem: &mut Problem<P>,
        mut state: PointState<V, F>,
    ) -> Result<PointState<V, F>, Self::Error> {
        state.reset();
        self.accepted_iterate = false;
        self.mu = self.mu0;
        self.gap = F::infinity();

        self.phase = match strict_feasibility(problem.inner(), state.param()) {
            Some(true) => BarrierPhase::PhaseTwo,
            Some(false) => BarrierPhase::PhaseOne,
            None => BarrierPhase::Failed,
        };

        if self.phase == BarrierPhase::Failed {
            // Keep cost-based convergence dormant until `next_iter` can
            // report the solver failure through the normal soft-stop path.
            state.replace(state.param().clone(), F::infinity());
            return Ok(state);
        }

        if self.phase == BarrierPhase::PhaseOne {
            // Executor criteria run before `next_iter`, including at iter 0.
            // The infeasible point is not a candidate for the original
            // problem, so exposing its true objective or gradient could let a
            // target or stationarity test bypass Phase I entirely.
            state.replace(state.param().clone(), F::infinity());
            return Ok(state);
        }

        // A feasible start is a valid candidate for the original problem.
        let cost = problem.cost(state.param())?;
        state.replace(state.param().clone(), cost);
        self.accepted_iterate = true;
        Ok(state)
    }

    fn next_iter(
        &mut self,
        problem: &mut Problem<P>,
        mut state: PointState<V, F>,
    ) -> Result<crate::SolverStep<PointState<V, F>>, Self::Error> {
        self.accepted_iterate = false;
        if self.phase == BarrierPhase::Failed {
            return Ok(crate::SolverStep::from((
                state,
                Some(Termination::numerical_failure(
                    "The barrier method could not initialize a finite barrier model.",
                )),
            )));
        }

        if self.phase == BarrierPhase::PhaseOne {
            let mut barrier_wrapper =
                Problem::new(LogBarrier::phase_one(problem.inner(), self.mu));
            let mut control = crate::core::run_control::RunControl::new()
                .max_iter(self.inner_max_iter);
            let inner_state = self.inner_solver.seed(state.param());
            let mut phase_one_solver = StopAtStrictFeasibility {
                inner: &mut self.inner_solver,
            };
            let result = run_loop_with_control(
                &mut barrier_wrapper,
                inner_state,
                &mut phase_one_solver,
                &mut control,
            );

            let inner_counts = *barrier_wrapper.counts();
            problem.counts_mut().add(&inner_counts);
            let result = result?;

            if !result
                .report
                .termination
                .can_continue_as_inner(crate::PartialResultPolicy::Consume)
            {
                self.phase = BarrierPhase::Failed;
                return Ok(crate::SolverStep::from((
                    state,
                    result.report.into_outer_termination(
                        crate::PartialResultPolicy::Consume,
                    ),
                )));
            }

            // Only the inner solver's gradient convergence can certify a
            // centered Phase I subproblem; a budget stop cannot.
            let centered = matches!(
                &result.report.termination,
                Termination::Converged(convergence) if convergence.criteria().iter().any(|criterion|
                    matches!(criterion.test, crate::ConvergenceTest::AbsoluteGradientSquared | crate::ConvergenceTest::RelativeGradientSquared)
                )
            );
            let candidate = result.state.param();
            let feasibility = strict_feasibility(problem.inner(), candidate);
            let Some(is_strictly_feasible) = feasibility else {
                self.phase = BarrierPhase::Failed;
                return Ok(crate::SolverStep::from((
                    state,
                    Some(Termination::numerical_failure(
                        "Phase I produced non-finite constraint slacks.",
                    )),
                )));
            };

            if is_strictly_feasible {
                let cost = problem.cost(candidate)?;
                state.replace(candidate.clone(), cost);
                // Phase II starts a fresh continuation schedule; Phase I's μ
                // controls feasibility accuracy, not objective optimality.
                self.phase = BarrierPhase::PhaseTwo;
                self.mu = self.mu0;
                self.gap = F::infinity();
                self.accepted_iterate = true;
                return Ok(crate::SolverStep::from((state, None)));
            }

            state.replace(candidate.clone(), F::infinity());

            // An unfinished solve supplies neither a Phase I optimum nor the
            // associated m·μ bound. Preserve its progress, but retry this μ
            // instead of turning a shrinking continuation parameter into a
            // false infeasibility certificate.
            if centered {
                let phase_one_gap =
                    F::from_usize(problem.inner().b().vec_len()).unwrap()
                        * self.mu;
                if phase_one_gap <= self.phase_one_tol {
                    self.phase = BarrierPhase::Failed;
                    return Ok(crate::SolverStep::from((
                        state,
                        Some(Termination::numerical_failure(
                            "A centered Phase I subproblem reached its gap threshold without strict feasibility.",
                        )),
                    )));
                }

                self.mu = self.mu / self.reduction;
            }

            return Ok(crate::SolverStep::from((state, None)));
        }

        // Minimize the barrier objective at the current μ on a *separate*
        // inner state seeded (warm-started) at the current iterate. A fresh
        // inner state (rather than threading the outer one) keeps the
        // inner solver's iteration counter from polluting the outer's.
        // Fresh criteria each call satisfies the statelessness contract.
        let mut barrier_wrapper =
            Problem::new(LogBarrier::new(problem.inner(), self.mu));
        let mut control = crate::core::run_control::RunControl::new()
            .max_iter(self.inner_max_iter);
        let inner_state = self.inner_solver.seed(state.param());
        let result = run_loop_with_control(
            &mut barrier_wrapper,
            inner_state,
            &mut self.inner_solver,
            &mut control,
        );

        // Eval aggregation (adapter-problem composition): fold the inner
        // wrapper's per-call counts back into the outer's wrapper. Copy out
        // before borrowing the outer mutably so the LogBarrier's `&P` borrow
        // (still held by `barrier_wrapper`) doesn't collide with the
        // `counts_mut` reborrow.
        let inner_counts = *barrier_wrapper.counts();
        problem.counts_mut().add(&inner_counts);
        let result = result?;

        if !result
            .report
            .termination
            .can_continue_as_inner(crate::PartialResultPolicy::Consume)
        {
            self.phase = BarrierPhase::Failed;
            return Ok(crate::SolverStep::from((
                state,
                result.report.into_outer_termination(
                    crate::PartialResultPolicy::Consume,
                ),
            )));
        }

        // Publish the original objective only after verifying the barrier domain.
        let candidate = result.state.param();
        if strict_feasibility(problem.inner(), candidate) != Some(true) {
            self.phase = BarrierPhase::Failed;
            return Ok(crate::SolverStep::from((
                state,
                Some(Termination::numerical_failure(
                    "The inner barrier solve returned a point outside the strict barrier domain.",
                )),
            )));
        }
        let cost = problem.cost(candidate)?;
        state.replace(candidate.clone(), cost);
        self.accepted_iterate = true;

        // Record the duality gap for this μ, then shrink for the next solve.
        self.gap =
            F::from_usize(problem.inner().b().vec_len()).unwrap() * self.mu;
        self.mu = self.mu / self.reduction;
        Ok(crate::SolverStep::from((state, None)))
    }

    fn should_check_iterate_change(&self) -> bool {
        self.accepted_iterate
    }

    fn terminate(&self, _state: &PointState<V, F>) -> Option<Termination<F>> {
        // Log-barrier duality-gap bound m·μ from the most recent solve.
        if self.phase == BarrierPhase::PhaseTwo
            && self.tol.is_some_and(|tol| self.gap <= tol)
        {
            Some(Termination::custom(
                "barrier.duality_gap",
                "In phase two, m * mu is at most the configured duality-gap tolerance.",
                vec![
                    crate::Measurement {
                        name: "gap".into(),
                        value: self.gap,
                    },
                    crate::Measurement {
                        name: "tolerance".into(),
                        value: self.tol.unwrap(),
                    },
                ],
            ))
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // The builder validation is backend-independent, so a unit inner stand-in
    // (`()`) suffices; these never run the solver, only the builders.

    #[test]
    #[should_panic(expected = "mu0 must be > 0")]
    fn rejects_nonpositive_mu0() {
        let _ = BarrierMethod::with_inner_solver(()).mu0(0.0);
    }

    #[test]
    #[should_panic(expected = "reduction must be > 1")]
    fn rejects_reduction_not_greater_than_one() {
        let _ = BarrierMethod::with_inner_solver(()).with_reduction(1.0);
    }

    #[test]
    #[should_panic(expected = "phase_one_tol must be > 0")]
    fn rejects_nonpositive_phase_one_tol() {
        let _ = BarrierMethod::with_inner_solver(())
            .with_absolute_phase_one_gap_tolerance(0.0);
    }

    #[test]
    #[should_panic(expected = "inner_max_iter must be ≥ 1")]
    fn rejects_zero_inner_max_iter() {
        let _ = BarrierMethod::<_, f64>::with_inner_solver(())
            .with_inner_max_iter(0);
    }
}
