//! Augmented-Lagrangian method for linear equality constraints `A x = b`.

use crate::core::augmented_lagrangian::AugmentedLagrangian;
use crate::core::constraint::LinearEqualityConstraints;
use crate::core::executor::run_loop;
use crate::core::inner::WarmStart;
use crate::core::math::{
    Dot, MatTransposeVec, MatVec, NormSquared, Scalar, ScaleInPlace, ScaledAdd,
};
use crate::core::problem::{CostFunction, Gradient, Problem};
use crate::core::solver::Solver;
use crate::core::state::{BasicState, CountsMirror, GradientState, State};
use crate::core::termination::{
    GradientTolerance, MaxIter, TerminationCriterion, TerminationReason,
};

/// Augmented-Lagrangian method for `min f(x) s.t. A x = b` — the
/// equality-constrained analogue of the log-barrier
/// [`BarrierMethod`](crate::solver::BarrierMethod), layering a quadratic
/// penalty plus multiplier estimates on an unconstrained inner solver.
///
/// Each outer iteration minimizes the augmented Lagrangian
/// `L_ρ(x, λ) = f(x) + λᵀ c(x) + (ρ/2)‖c(x)‖²` (with `c(x) = A x − b`, the
/// [`AugmentedLagrangian`] adapter) over `x` with the inner solver `So`,
/// warm-started from the current iterate, then updates the multiplier
/// estimate `λ ← λ + ρ c(x)`. When feasibility fails to improve enough
/// between outer iterations the penalty `ρ` is increased instead. As `λ`
/// approaches the true multipliers `λ*`, the unconstrained minimizer of
/// `L_ρ` approaches the constrained optimum.
///
/// The method is generic over the inner solver `So`: any gradient-based
/// solver that implements [`WarmStart`] and
/// iterates over its own [`GradientState`], seeded at the current iterate via
/// [`WarmStart::seed`]. That covers
/// [`GradientDescent`](crate::solver::GradientDescent) ([`BasicState`]),
/// [`Bfgs`](crate::solver::Bfgs)
/// ([`QuasiNewtonState`](crate::core::state::QuasiNewtonState)), and unbounded
/// [`Lbfgs`](crate::solver::lbfgs::Lbfgs)
/// ([`LbfgsState`](crate::core::state::LbfgsState)). A least-squares inner
/// ([`LevenbergMarquardt`](crate::solver::LevenbergMarquardt)) does not fit —
/// `L_ρ` is not a sum of squares and the [`AugmentedLagrangian`] adapter
/// exposes only `CostFunction + Gradient` — and a derivative-free inner
/// (Nelder-Mead) is excluded by the [`GradientState`] bound.
///
/// # Infeasible starts are fine
///
/// Unlike the [`BarrierMethod`](crate::solver::BarrierMethod), the augmented
/// Lagrangian is finite and smooth *everywhere* — there is no `+∞`
/// feasibility wall — so the starting point need **not** satisfy `A x₀ = b`,
/// and the inner solver may use any line search (Armijo backtracking, Wolfe,
/// Moré–Thuente) or momentum. No phase-1 feasibility solve is required.
///
/// # Algorithm
///
/// Nocedal & Wright, *Numerical Optimization* §17.3 (Alg. 17.4, the
/// LANCELOT-style outer loop), simplified:
///
/// ```text
/// λ ← 0 ∈ ℝᵐ ;  ρ ← rho0
/// repeat:
///   x ← argmin_x L_ρ(x, λ)            # inner solver, warm-started at x
///   c ← A x − b
///   if ‖c‖ ≤ tol: stop (SolverConverged)
///   if ‖c‖ ≤ feasibility_decrease · ‖c_prev‖:
///       λ ← λ + ρ c                    # sufficient feasibility improvement
///   else:
///       ρ ← rho_increase · ρ           # else tighten penalty, keep λ
/// ```
///
/// # Termination
///
/// The outer feasibility test `‖A x − b‖ ≤ tol` is solver-specific and lives
/// on the solver (tenet 3): it fires via [`terminate`](Solver::terminate) as
/// [`SolverConverged`](TerminationReason::SolverConverged). Optimality is the
/// inner solve's job — it drives `‖∇_x L_ρ‖` down — so once the iterate is
/// feasible and the inner solve has converged, the KKT conditions hold. Pair
/// with the executor's [`MaxIter`] as a safety net.
///
/// **Do not attach a gradient-norm criterion to the outer executor.** As with
/// the barrier, at a constrained optimum the *true* objective gradient `∇f`
/// does not vanish — it is balanced by `Aᵀλ*` — so a framework
/// [`GradientTolerance`] on the
/// outer loop would fire on the wrong point or never. (The outer state's
/// gradient is the true `∇f`, seeded only so the state is well-formed; it is
/// not a convergence signal.)
///
/// # Backends
///
/// Requires the constraint matrix to implement [`MatVec`] (`A x`) and
/// [`MatTransposeVec`] (`Aᵀ v`) — never a linear solve. All backends supply
/// those two ops, so the method runs on every backend: `Vec<f64>` (via
/// [`DenseMatrix`](crate::core::math::DenseMatrix)), nalgebra
/// (`DMatrix`/`DVector`), faer (`Mat`/`Col`), and `ndarray`
/// (`Array2`/`Array1`).
///
/// # Composition
///
/// Internally drives the inner solver via
/// [`run_loop`] against a **fresh** inner
/// `Problem` wrapper each outer iteration
/// (the inner is an *adapter-problem* composition — the augmented Lagrangian
/// is a distinct type from the outer problem) with a fresh criteria vector
/// (`MaxIter` + `GradientTolerance` on the augmented Lagrangian). The fresh
/// wrapper is intrinsic here — each outer iter minimizes a *different*
/// surrogate (updated λ, ρ), not a reuse-avoidance dodge: criteria
/// [reset](crate::core::termination::TerminationCriterion::reset) per run, so
/// even a stored [`InnerExecutor`](crate::core::inner::InnerExecutor) would
/// reuse stateful criteria safely. After each inner solve, the inner wrapper's
/// [`EvalCounts`](crate::core::problem::EvalCounts) is folded into the
/// outer's wrapper via
/// [`EvalCounts::add`](crate::core::problem::EvalCounts::add) on
/// [`Problem::counts_mut`](crate::core::problem::Problem::counts_mut).
///
/// # Examples
///
/// `AugmentedLagrangianMethod` wraps a gradient inner solver to handle
/// `LinearEqualityConstraints`, and tolerates infeasible starts. See
/// [`ProjectedGradientDescent`](crate::solver::ProjectedGradientDescent)
/// for the simpler box-constrained pattern.
pub struct AugmentedLagrangianMethod<So, V, F = f64> {
    inner_solver: So,
    inner_max_iter: u64,
    inner_grad_tol: F,
    rho0: F,
    rho: F,
    rho_increase: F,
    feasibility_decrease: F,
    tol: F,
    /// Multiplier estimate `λ ∈ ℝᵐ`; populated in [`init`](Solver::init) with
    /// a zero vector shaped like `b`, then carried across outer iterations.
    lambda: Option<V>,
    /// `‖c(x)‖` of the most recent inner solve; `+∞` until the first solve so
    /// [`terminate`](Solver::terminate) cannot fire at iter 0.
    c_norm: F,
    /// `‖c(x)‖` of the previous outer iteration, for the `ρ`-increase test.
    c_norm_prev: F,
}

impl<So, V> AugmentedLagrangianMethod<So, V> {
    /// Build an augmented-Lagrangian method around an unconstrained inner
    /// solver.
    ///
    /// Defaults: `rho0 = 10.0`, `rho_increase = 10.0`,
    /// `feasibility_decrease = 0.25`, `tol = 1e-8`, `inner_max_iter = 50`,
    /// `inner_grad_tol = 1e-8`.
    pub fn new(inner_solver: So) -> Self {
        Self {
            inner_solver,
            inner_max_iter: 50,
            inner_grad_tol: 1e-8,
            rho0: 10.0,
            rho: 10.0,
            rho_increase: 10.0,
            feasibility_decrease: 0.25,
            tol: 1e-8,
            lambda: None,
            c_norm: f64::INFINITY,
            c_norm_prev: f64::INFINITY,
        }
    }
}

impl<So, V, F: Scalar> AugmentedLagrangianMethod<So, V, F> {
    /// Initial penalty parameter `ρ` (default `10.0`).
    ///
    /// # Panics
    ///
    /// Panics unless `rho0 > 0` — a non-positive penalty is not a penalty.
    pub fn rho0(mut self, rho0: F) -> Self {
        assert!(rho0 > F::zero(), "rho0 must be > 0");
        self.rho0 = rho0;
        self
    }

    /// Penalty growth factor: `ρ ← rho_increase · ρ` when feasibility fails
    /// to improve sufficiently (default `10.0`).
    ///
    /// # Panics
    ///
    /// Panics unless `rho_increase > 1` — otherwise the penalty would not
    /// grow and a stalled iterate could never be pushed onto the feasible
    /// set.
    pub fn with_rho_increase(mut self, rho_increase: F) -> Self {
        assert!(rho_increase > F::one(), "rho_increase must be > 1");
        self.rho_increase = rho_increase;
        self
    }

    /// Required feasibility-decrease ratio `τ ∈ (0, 1)`: the multipliers are
    /// updated only when `‖c(x_k)‖ ≤ τ · ‖c(x_{k−1})‖`; otherwise the penalty
    /// is increased instead (default `0.25`).
    ///
    /// # Panics
    ///
    /// Panics unless `0 < feasibility_decrease < 1`.
    pub fn with_feasibility_decrease(mut self, feasibility_decrease: F) -> Self {
        assert!(
            feasibility_decrease > F::zero() && feasibility_decrease < F::one(),
            "feasibility_decrease must be in (0, 1)"
        );
        self.feasibility_decrease = feasibility_decrease;
        self
    }

    /// Outer feasibility tolerance: stop once `‖A x − b‖ ≤ tol` (default
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

    /// Iteration budget for each inner subproblem solve (default `50`).
    ///
    /// As with the barrier, a first-order inner solver on the (increasingly
    /// ill-conditioned) penalized objective typically exhausts this budget
    /// rather than reaching [`with_inner_grad_tol`](Self::with_inner_grad_tol); the
    /// outer multiplier updates still converge from loosely-minimized
    /// subproblems. Raise it for hard / higher-dimensional problems.
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

    /// Gradient-norm tolerance for each inner subproblem solve (default
    /// `1e-8`). Inner solves stop at `‖∇L_ρ‖ ≤ inner_grad_tol`.
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

// Deprecated setter aliases from the B1 `with_*` rename (0.10.0); remove at 1.0.
impl<So, V, F: Scalar> AugmentedLagrangianMethod<So, V, F> {
    /// Deprecated: renamed to [`with_rho_increase`](Self::with_rho_increase).
    #[deprecated(since = "0.10.0", note = "renamed to `with_rho_increase`")]
    pub fn rho_increase(self, rho_increase: F) -> Self {
        self.with_rho_increase(rho_increase)
    }

    /// Deprecated: renamed to [`with_feasibility_decrease`](Self::with_feasibility_decrease).
    #[deprecated(since = "0.10.0", note = "renamed to `with_feasibility_decrease`")]
    pub fn feasibility_decrease(self, feasibility_decrease: F) -> Self {
        self.with_feasibility_decrease(feasibility_decrease)
    }

    /// Deprecated: renamed to [`with_tol`](Self::with_tol).
    #[deprecated(since = "0.10.0", note = "renamed to `with_tol`")]
    pub fn tol(self, tol: F) -> Self {
        self.with_tol(tol)
    }

    /// Deprecated: renamed to [`with_inner_max_iter`](Self::with_inner_max_iter).
    #[deprecated(since = "0.10.0", note = "renamed to `with_inner_max_iter`")]
    pub fn inner_max_iter(self, inner_max_iter: u64) -> Self {
        self.with_inner_max_iter(inner_max_iter)
    }

    /// Deprecated: renamed to [`with_inner_grad_tol`](Self::with_inner_grad_tol).
    #[deprecated(since = "0.10.0", note = "renamed to `with_inner_grad_tol`")]
    pub fn inner_grad_tol(self, inner_grad_tol: F) -> Self {
        self.with_inner_grad_tol(inner_grad_tol)
    }
}

impl<P, V, M, So, F> Solver<P, BasicState<V, F>> for AugmentedLagrangianMethod<So, V, F>
where
    F: Scalar,
    P: CostFunction<Param = V, Output = F>
        + Gradient<Gradient = V>
        + LinearEqualityConstraints<Param = V, Matrix = M>,
    M: MatVec<V> + MatTransposeVec<V>,
    V: ScaledAdd<F> + Dot<F> + NormSquared<F> + ScaleInPlace<F> + Clone,
    So: WarmStart<V>
        + for<'a> Solver<
            AugmentedLagrangian<'a, P, V, F>,
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
        self.rho = self.rho0;
        self.c_norm = F::infinity();
        self.c_norm_prev = F::infinity();

        // λ ← 0 ∈ ℝᵐ. Clone `b` for the right shape (m entries), then zero it
        // — backend-generic with no "zeros_like" in the math layer. No
        // feasibility precondition: the augmented Lagrangian tolerates an
        // infeasible x₀.
        let mut lambda = problem.inner().b().clone();
        lambda.scale_in_place(F::zero());
        self.lambda = Some(lambda);

        // Seed the *true* objective so framework criteria and the public
        // result read f, not the augmented-Lagrangian value.
        let (cost, grad) = problem.cost_and_gradient(state.param())?;
        state.cost = Some(cost);
        state.gradient = Some(grad);
        Ok(state)
    }

    fn next_iter(
        &mut self,
        problem: &mut Problem<P>,
        mut state: BasicState<V, F>,
    ) -> Result<(BasicState<V, F>, Option<TerminationReason>), Self::Error> {
        // Minimize the augmented Lagrangian at the current (λ, ρ) on a
        // *separate* inner state seeded (warm-started) at the current
        // iterate. Fresh criteria each call satisfies the statelessness
        // contract.
        let lambda = self.lambda.as_ref().expect("init populates lambda");
        let mut al_wrapper =
            Problem::new(AugmentedLagrangian::new(problem.inner(), lambda, self.rho));
        let mut criteria: Vec<Box<dyn TerminationCriterion<So::State>>> = vec![
            Box::new(MaxIter(self.inner_max_iter)),
            Box::new(GradientTolerance(self.inner_grad_tol)),
        ];
        let inner_state = self.inner_solver.seed(state.param());
        let result = run_loop(
            &mut al_wrapper,
            inner_state,
            &mut self.inner_solver,
            &mut criteria,
            self.inner_max_iter,
        )?;

        // Eval aggregation (adapter-problem composition): fold the inner
        // wrapper's per-call counts back into the outer's wrapper. Copy out
        // before borrowing the outer mutably so the AL adapter's `&P` borrow
        // (still held by `al_wrapper`) doesn't collide with the
        // `counts_mut` reborrow.
        let inner_counts = *al_wrapper.counts();
        problem.counts_mut().add(&inner_counts);

        if result.reason.is_failure() {
            return Ok((state, Some(TerminationReason::SolverFailed)));
        }

        // Adopt the inner's iterate, then evaluate the *true* f / ∇f there
        // (the inner left cost/gradient at the augmented-Lagrangian value).
        state.param = result.state.param().clone();
        let (cost, grad) = problem.cost_and_gradient(&state.param)?;
        state.cost = Some(cost);
        state.gradient = Some(grad);

        // Constraint residual c = A x − b at the new iterate.
        let mut c = problem.inner().a().matvec(&state.param);
        c.scaled_add(-F::one(), problem.inner().b());
        self.c_norm_prev = self.c_norm;
        self.c_norm = c.norm_squared().sqrt();

        // Multiplier update vs. penalty increase. The first solve
        // (c_norm_prev = +∞) always takes the update branch.
        if self.c_norm <= self.feasibility_decrease * self.c_norm_prev {
            // Sufficient feasibility improvement: first-order multiplier
            // update λ ← λ + ρ c.
            let lambda = self.lambda.as_mut().expect("init populates lambda");
            lambda.scaled_add(self.rho, &c);
        } else {
            // Feasibility stalled: tighten the penalty, keep λ.
            self.rho = self.rho * self.rho_increase;
        }

        Ok((state, None))
    }

    fn terminate(&self, _state: &BasicState<V, F>) -> Option<TerminationReason> {
        // Feasibility bound ‖A x − b‖ from the most recent solve. Optimality
        // is handled by the inner solve driving ‖∇L_ρ‖ down.
        if self.c_norm <= self.tol {
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
    // (`()`) suffices. The struct's `V` generic is unconstrained by the
    // builders, so pin it to a dummy `Vec<f64>` via turbofish.
    type Builder = AugmentedLagrangianMethod<(), Vec<f64>>;

    #[test]
    #[should_panic(expected = "rho0 must be > 0")]
    fn rejects_nonpositive_rho0() {
        let _ = Builder::new(()).rho0(0.0);
    }

    #[test]
    #[should_panic(expected = "rho_increase must be > 1")]
    fn rejects_rho_increase_not_greater_than_one() {
        let _ = Builder::new(()).with_rho_increase(1.0);
    }

    #[test]
    #[should_panic(expected = "feasibility_decrease must be in (0, 1)")]
    fn rejects_feasibility_decrease_out_of_range() {
        let _ = Builder::new(()).with_feasibility_decrease(1.0);
    }

    #[test]
    #[should_panic(expected = "tol must be > 0")]
    fn rejects_nonpositive_tol() {
        let _ = Builder::new(()).with_tol(0.0);
    }

    #[test]
    #[should_panic(expected = "inner_max_iter must be ≥ 1")]
    fn rejects_zero_inner_max_iter() {
        let _ = Builder::new(()).with_inner_max_iter(0);
    }

    #[test]
    #[should_panic(expected = "inner_grad_tol must be ≥ 0")]
    fn rejects_negative_inner_grad_tol() {
        let _ = Builder::new(()).with_inner_grad_tol(-1.0);
    }
}
