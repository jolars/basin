//! Log-barrier (sequential unconstrained minimization) method for linear
//! inequality constraints `A x ≤ b`.

use crate::core::barrier::{LogBarrier, strict_feasibility};
use crate::core::constraint::LinearInequalityConstraints;
use crate::core::executor::run_loop;
use crate::core::inner::{InitialState, WarmStart};
use crate::core::math::{
    MatTransposeVec, MatVec, NegInPlace, NormSquared, Scalar, ScaledAdd,
    VectorIndex, VectorLen,
};
use crate::core::problem::{CostFunction, Gradient, Problem};
use crate::core::solver::Solver;
use crate::core::state::{BasicState, CountsMirror, GradientState, State};
use crate::core::termination::{
    GradientTolerance, MaxIter, TerminationCriterion, TerminationReason,
};

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
/// The method is generic over the inner solver `So`: any gradient-based
/// solver that implements [`WarmStart`] and
/// iterates over its own [`GradientState`]. The inner state is seeded at the
/// current iterate via [`InitialState::seed`],
/// so each of [`GradientDescent`](crate::solver::GradientDescent)
/// ([`BasicState`]), [`Bfgs`](crate::solver::Bfgs)
/// ([`QuasiNewtonState`](crate::core::state::QuasiNewtonState)), and unbounded
/// [`Lbfgs`](crate::solver::lbfgs::Lbfgs)
/// ([`LbfgsState`](crate::core::state::LbfgsState)) is usable. Two inner kinds
/// are deliberately excluded: a least-squares solver
/// ([`LevenbergMarquardt`](crate::solver::LevenbergMarquardt)), because the
/// barrier objective is not a sum of squares and the [`LogBarrier`] adapter
/// exposes only `CostFunction + Gradient`, not `Residual + Jacobian`, and a
/// derivative-free solver (Nelder-Mead), ruled out by the [`GradientState`]
/// bound.
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
/// [`with_phase_one_tol`](Self::with_phase_one_tol) without finding a strict
/// point, the constraints are reported as
/// [`SolverFailed`](TerminationReason::SolverFailed). An inner solve that
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
/// [`SolverConverged`](TerminationReason::SolverConverged). Pair with the
/// executor's [`MaxIter`] as a safety net. A strictly feasible start uses only
/// Phase II; an infeasible or boundary start spends additional outer
/// iterations in Phase I. With the defaults each continuation closes its gap
/// in roughly `log(m · mu0 / tol) / log(reduction)` outer iterations
/// (≈ 9 for the defaults), so an outer `max_iter` of 30–50 is a practical
/// safety budget for both phases.
///
/// **Do not attach a gradient-norm criterion to the outer executor.** The
/// gap test is the correct optimality measure here. At a constrained
/// optimum the true objective gradient `∇f` does *not* vanish (it points
/// into the active constraint face), so a framework
/// [`GradientTolerance`] /
/// [`RelativeGradientTolerance`](crate::core::termination::RelativeGradientTolerance)
/// on the outer loop would either never fire or fire on the wrong point.
/// (The outer state's gradient is the true `∇f`, seeded only so the state
/// is well-formed; it is not a convergence signal.)
///
/// # Backends
///
/// Requires the constraint matrix to implement
/// [`MatVec`] (`A x`) and [`MatTransposeVec`] (`Aᵀ v`), never a linear
/// solve. All backends supply those two ops, so the method runs on every
/// backend: `Vec<f64>` (via
/// [`DenseMatrix`](crate::core::math::DenseMatrix)), nalgebra
/// (`DMatrix`/`DVector`), faer (`Mat`/`Col`), and `ndarray`
/// (`Array2`/`Array1`).
///
/// # Composition
///
/// Internally drives the inner solver via
/// [`run_loop`] with a **fresh** criteria
/// vector each outer iteration (`MaxIter` + `GradientTolerance` on the
/// current Phase I or Phase II barrier objective). The fresh vector is
/// intrinsic here, since each outer iter minimizes a *different* surrogate
/// (`Problem::new(LogBarrier)` with a shrinking μ), not a reuse-avoidance
/// dodge: criteria
/// [reset](crate::core::termination::TerminationCriterion::reset) per run, so
/// even a stored [`InnerExecutor`](crate::core::inner::InnerExecutor) would
/// reuse stateful criteria safely. The inner runs on its own `So::State` (seeded via
/// [`WarmStart`]) against a fresh `Problem::new(LogBarrier)`; after each
/// solve its
/// [`EvalCounts`](crate::core::problem::EvalCounts) are folded back into
/// the outer wrapper via
/// [`Problem::counts_mut`] (adapter-problem composition, rule 1).
///
/// # Examples
///
/// `BarrierMethod` wraps a gradient inner solver (e.g. `BFGS` paired with
/// `Backtracking`) to handle `LinearInequalityConstraints`. See
/// [`ProjectedGradientDescent`](crate::solver::ProjectedGradientDescent)
/// for the simpler box-constrained pattern.
pub struct BarrierMethod<So, F = f64> {
    inner_solver: So,
    inner_max_iter: u64,
    inner_grad_tol: F,
    mu0: F,
    mu: F,
    reduction: F,
    tol: F,
    phase_one_tol: F,
    /// `m · μ` of the most recent inner solve; `+∞` until the first solve
    /// so [`terminate`](Solver::terminate) cannot fire at iter 0.
    gap: F,
    phase: BarrierPhase,
}

#[derive(Clone, Copy, PartialEq, Eq)]
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
    ) -> Result<(S, Option<TerminationReason>), Self::Error> {
        let (state, reason) = self.inner.next_iter(problem, state)?;
        let inner_failed = reason.is_some_and(|reason| reason.is_failure());
        if !inner_failed
            && problem.inner().strict_feasibility(state.param()) == Some(true)
        {
            Ok((state, Some(TerminationReason::SolverConverged)))
        } else {
            Ok((state, reason))
        }
    }

    fn terminate(&self, state: &S) -> Option<TerminationReason> {
        self.inner.terminate(state)
    }
}

impl<So> BarrierMethod<So> {
    /// Build a barrier method around an unconstrained inner solver.
    ///
    /// Defaults: `mu0 = 1.0`, `reduction = 10.0`, `tol = 1e-8`,
    /// `phase_one_tol = 1e-8`, `inner_max_iter = 50`,
    /// `inner_grad_tol = 1e-8`.
    ///
    /// The `inner_max_iter` default is intentionally modest:
    /// [`with_inner_max_iter`](Self::with_inner_max_iter) is the dominant cost lever
    /// (see its docs) and the outer μ-continuation tolerates loosely-centered
    /// subproblems, so a small budget usually converges to the same point far
    /// more cheaply than a large one.
    pub fn new(inner_solver: So) -> Self {
        Self {
            inner_solver,
            inner_max_iter: 50,
            inner_grad_tol: 1e-8,
            mu0: 1.0,
            mu: 1.0,
            reduction: 10.0,
            tol: 1e-8,
            phase_one_tol: 1e-8,
            gap: f64::INFINITY,
            phase: BarrierPhase::PhaseTwo,
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

    /// Outer duality-gap tolerance: stop once `m · μ ≤ tol` (default
    /// `1e-8`).
    ///
    /// # Panics
    ///
    /// Panics unless `tol > 0`.
    pub fn with_tol(mut self, tol: F) -> Self {
        assert!(tol > F::zero(), "tol must be > 0");
        self.tol = tol;
        self
    }

    /// Phase I accuracy used to classify a constraint system with no strict
    /// interior (default `1e-8`). If a centered Phase I subproblem has not
    /// found `A x < b` once its auxiliary duality gap `m · μ` is at most this
    /// tolerance, the solver reports
    /// [`SolverFailed`](TerminationReason::SolverFailed).
    ///
    /// A finite-precision method cannot distinguish an exactly empty interior
    /// from an arbitrarily thin one. This tolerance therefore means "no strict
    /// interior at this numerical scale."
    ///
    /// # Panics
    ///
    /// Panics unless `phase_one_tol > 0`.
    pub fn with_phase_one_tol(mut self, phase_one_tol: F) -> Self {
        assert!(phase_one_tol > F::zero(), "phase_one_tol must be > 0");
        self.phase_one_tol = phase_one_tol;
        self
    }

    /// Iteration budget for each inner barrier-subproblem solve (default
    /// `50`).
    ///
    /// **This is the dominant cost lever.** A first-order inner solver
    /// (`GradientDescent`) on the ill-conditioned barrier typically exhausts
    /// this budget rather than reaching [`with_inner_grad_tol`](Self::with_inner_grad_tol),
    /// so total work scales roughly linearly with it. Because the outer
    /// μ-continuation re-solves at each shrinking `μ`, a loosely-centered
    /// (small-budget) subproblem usually still converges to the same point,
    /// often an order of magnitude cheaper. Raise it for hard or higher-
    /// dimensional problems; a Newton-class inner (future work) would center
    /// in far fewer steps and reach `inner_grad_tol` instead.
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

    /// Gradient-norm tolerance for each inner barrier-subproblem solve
    /// (default `1e-8`). Inner solves stop at `‖∇φ_μ‖ ≤ inner_grad_tol`.
    ///
    /// Note: with a first-order inner solver this rarely binds; the
    /// ill-conditioned barrier means [`with_inner_max_iter`](Self::with_inner_max_iter)
    /// usually governs instead. It matters for a Newton-class inner.
    ///
    /// # Panics
    ///
    /// Panics unless `inner_grad_tol ≥ 0`.
    pub fn with_inner_grad_tol(mut self, inner_grad_tol: F) -> Self {
        assert!(inner_grad_tol >= F::zero(), "inner_grad_tol must be ≥ 0");
        self.inner_grad_tol = inner_grad_tol;
        self
    }
}

impl<So, V, F> InitialState<V> for BarrierMethod<So, F>
where
    F: Scalar,
    V: Clone,
{
    type State = BasicState<V, F>;
    fn seed(&self, x: &V) -> Self::State {
        BasicState::new(x.clone())
    }
}

impl<P, V, M, So, F> Solver<P, BasicState<V, F>> for BarrierMethod<So, F>
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
    So::State: GradientState<Param = V, Float = F> + CountsMirror,
{
    type Error = <P as CostFunction>::Error;

    fn init(
        &mut self,
        problem: &mut Problem<P>,
        mut state: BasicState<V, F>,
    ) -> Result<BasicState<V, F>, Self::Error> {
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
            state.cost = Some(F::infinity());
            state.gradient = None;
            return Ok(state);
        }

        if self.phase == BarrierPhase::PhaseOne {
            // Executor criteria run before `next_iter`, including at iter 0.
            // The infeasible point is not a candidate for the original
            // problem, so exposing its true objective or gradient could let a
            // target or stationarity test bypass Phase I entirely.
            state.cost = Some(F::infinity());
            state.gradient = None;
            return Ok(state);
        }

        // A feasible start is a valid candidate for the original problem.
        let (cost, grad) = problem.cost_and_gradient(state.param())?;
        state.cost = Some(cost);
        state.gradient = Some(grad);
        Ok(state)
    }

    fn next_iter(
        &mut self,
        problem: &mut Problem<P>,
        mut state: BasicState<V, F>,
    ) -> Result<(BasicState<V, F>, Option<TerminationReason>), Self::Error>
    {
        if self.phase == BarrierPhase::Failed {
            return Ok((state, Some(TerminationReason::SolverFailed)));
        }

        if self.phase == BarrierPhase::PhaseOne {
            let mut barrier_wrapper =
                Problem::new(LogBarrier::phase_one(problem.inner(), self.mu));
            let mut criteria: Vec<Box<dyn TerminationCriterion<So::State>>> = vec![
                Box::new(MaxIter(self.inner_max_iter)),
                Box::new(GradientTolerance(self.inner_grad_tol)),
            ];
            let inner_state = self.inner_solver.seed(state.param());
            let mut phase_one_solver = StopAtStrictFeasibility {
                inner: &mut self.inner_solver,
            };
            let result = run_loop(
                &mut barrier_wrapper,
                inner_state,
                &mut phase_one_solver,
                &mut criteria,
                self.inner_max_iter,
            )?;

            let inner_counts = *barrier_wrapper.counts();
            problem.counts_mut().add(&inner_counts);

            if result.reason.is_failure() {
                self.phase = BarrierPhase::Failed;
                return Ok((state, Some(TerminationReason::SolverFailed)));
            }

            let centered = result.state.gradient().is_some_and(|gradient| {
                gradient.norm_squared()
                    <= self.inner_grad_tol * self.inner_grad_tol
            });
            let candidate = result.state.param();
            let feasibility = strict_feasibility(problem.inner(), candidate);
            let Some(is_strictly_feasible) = feasibility else {
                self.phase = BarrierPhase::Failed;
                return Ok((state, Some(TerminationReason::SolverFailed)));
            };

            state.param = candidate.clone();

            if is_strictly_feasible {
                let (cost, grad) = problem.cost_and_gradient(&state.param)?;
                state.cost = Some(cost);
                state.gradient = Some(grad);
                // Phase II starts a fresh continuation schedule; Phase I's μ
                // controls feasibility accuracy, not objective optimality.
                self.phase = BarrierPhase::PhaseTwo;
                self.mu = self.mu0;
                self.gap = F::infinity();
                return Ok((state, None));
            }

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
                    return Ok((state, Some(TerminationReason::SolverFailed)));
                }

                self.mu = self.mu / self.reduction;
            }

            state.cost = Some(F::infinity());
            state.gradient = None;
            return Ok((state, None));
        }

        // Minimize the barrier objective at the current μ on a *separate*
        // inner state seeded (warm-started) at the current iterate. A fresh
        // inner state (rather than threading the outer one) keeps the
        // inner solver's iteration counter from polluting the outer's.
        // Fresh criteria each call satisfies the statelessness contract.
        let mut barrier_wrapper =
            Problem::new(LogBarrier::new(problem.inner(), self.mu));
        let mut criteria: Vec<Box<dyn TerminationCriterion<So::State>>> = vec![
            Box::new(MaxIter(self.inner_max_iter)),
            Box::new(GradientTolerance(self.inner_grad_tol)),
        ];
        let inner_state = self.inner_solver.seed(state.param());
        let result = run_loop(
            &mut barrier_wrapper,
            inner_state,
            &mut self.inner_solver,
            &mut criteria,
            self.inner_max_iter,
        )?;

        // Eval aggregation (adapter-problem composition): fold the inner
        // wrapper's per-call counts back into the outer's wrapper. Copy out
        // before borrowing the outer mutably so the LogBarrier's `&P` borrow
        // (still held by `barrier_wrapper`) doesn't collide with the
        // `counts_mut` reborrow.
        let inner_counts = *barrier_wrapper.counts();
        problem.counts_mut().add(&inner_counts);

        if result.reason.is_failure() {
            self.phase = BarrierPhase::Failed;
            return Ok((state, Some(TerminationReason::SolverFailed)));
        }

        // Adopt the inner's iterate, then evaluate the *true* f/∇f there
        // (the inner left cost and gradient at the barrier objective).
        let candidate = result.state.param();
        if strict_feasibility(problem.inner(), candidate) != Some(true) {
            self.phase = BarrierPhase::Failed;
            return Ok((state, Some(TerminationReason::SolverFailed)));
        }
        state.param = candidate.clone();
        let (cost, grad) = problem.cost_and_gradient(&state.param)?;
        state.cost = Some(cost);
        state.gradient = Some(grad);

        // Record the duality gap for this μ, then shrink for the next solve.
        self.gap =
            F::from_usize(problem.inner().b().vec_len()).unwrap() * self.mu;
        self.mu = self.mu / self.reduction;
        Ok((state, None))
    }

    fn terminate(
        &self,
        _state: &BasicState<V, F>,
    ) -> Option<TerminationReason> {
        // Log-barrier duality-gap bound m·μ from the most recent solve.
        if self.phase == BarrierPhase::PhaseTwo && self.gap <= self.tol {
            Some(TerminationReason::SolverConverged)
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
        let _ = BarrierMethod::new(()).mu0(0.0);
    }

    #[test]
    #[should_panic(expected = "reduction must be > 1")]
    fn rejects_reduction_not_greater_than_one() {
        let _ = BarrierMethod::new(()).with_reduction(1.0);
    }

    #[test]
    #[should_panic(expected = "tol must be > 0")]
    fn rejects_nonpositive_tol() {
        let _ = BarrierMethod::new(()).with_tol(0.0);
    }

    #[test]
    #[should_panic(expected = "phase_one_tol must be > 0")]
    fn rejects_nonpositive_phase_one_tol() {
        let _ = BarrierMethod::new(()).with_phase_one_tol(0.0);
    }

    #[test]
    #[should_panic(expected = "inner_max_iter must be ≥ 1")]
    fn rejects_zero_inner_max_iter() {
        let _ = BarrierMethod::new(()).with_inner_max_iter(0);
    }

    #[test]
    #[should_panic(expected = "inner_grad_tol must be ≥ 0")]
    fn rejects_negative_inner_grad_tol() {
        let _ = BarrierMethod::new(()).with_inner_grad_tol(-1.0);
    }
}
