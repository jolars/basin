use crate::core::inner::InitialState;
use crate::core::math::{
    GramMatrix, LinearSolveSpd, MatTransposeVec, NegInPlace, NormInfinity, NormSquared, Scalar,
    ScaledAdd,
};
use crate::core::problem::{Jacobian, Problem, Residual};
use crate::core::solver::Solver;
use crate::core::state::NllsState;
use crate::core::termination::TerminationReason;

/// Pure Gauss-Newton solver for nonlinear least-squares problems
/// `min ½‖r(x)‖²`.
///
/// Each iteration solves the normal equations `(JᵀJ) δ = −Jᵀr` via
/// Cholesky on the Gram matrix `JᵀJ` and takes the full step
/// `x ← x + δ`. No damping, no line search — that's what
/// Levenberg-Marquardt is for. See Madsen, Nielsen, Tingleff (2004),
/// *Methods for Non-Linear Least Squares Problems*, §3.1.
///
/// **Cholesky-on-`JᵀJ` vs QR-on-`J`.** Cholesky on the Gram is the
/// simple path and the only one the
/// [`linalg`](crate::core::math) tier exposes today. It squares the
/// condition number of `J` and fails noisily on rank-deficient `J` —
/// see the [`solve_spd` failure path](#failure-modes) below. QR-on-`J`
/// is more numerically robust but adds a second factorization to the
/// linalg surface; deferred until a solver actually needs it. Pure GN
/// is the right vehicle for Cholesky: when `J` is so ill-conditioned
/// that QR matters in practice, you wanted LM (or TRF) anyway.
///
/// # Failure modes
///
/// - **Rank-deficient `J`** (`JᵀJ` not positive definite) → the
///   Cholesky inside [`LinearSolveSpd`] returns
///   [`NotPositiveDefinite`](crate::core::math::LinearSolveError::NotPositiveDefinite),
///   and the solver returns [`TerminationReason::SolverFailed`]. This
///   is the *correct* behavior for pure GN — Powell's singular
///   function is the canonical example. Reach for Levenberg-Marquardt
///   when this fires.
/// - **Divergence on highly nonlinear / poorly initialized problems.**
///   No safeguard here either; pure GN trusts the linear model. Catch
///   this with a finite [`MaxIter`](crate::core::termination::MaxIter)
///   or [`CostTolerance`](crate::core::termination::CostTolerance) on
///   the executor.
///
/// # Termination
///
/// Beyond the framework criteria
/// ([`MaxIter`](crate::core::termination::MaxIter),
/// [`CostTolerance`](crate::core::termination::CostTolerance),
/// [`ParamTolerance`](crate::core::termination::ParamTolerance), …),
/// the solver emits [`TerminationReason::SolverConverged`] when the
/// first-order optimality measure
/// `‖Jᵀr‖_∞ ≤ tol_grad` (Madsen et al. eq. 3.3a) is satisfied.
/// Default `tol_grad = 1e-8`; set to `0.0` to disable the check.
///
/// Gauss-Newton runs on [`NllsState`], which
/// does **not** impl [`GradientState`](crate::core::state::GradientState):
/// the framework's L2-squared
/// [`GradientTolerance`](crate::core::termination::GradientTolerance) is the
/// wrong metric for NLLS, so attaching it is a **compile error** rather than a
/// criterion that silently never fires. Use
/// [`with_tol_grad`](Self::with_tol_grad) above for the first-order test.
///
/// # Backends
///
/// LA-heavy: the default `Vec<f64>` backend (over the hand-rolled
/// [`DenseMatrix<f64>`](crate::DenseMatrix), via a pure-Rust Cholesky),
/// nalgebra (`DVector<f64>` / `DMatrix<f64>`), and faer (`Col<f64>` /
/// `Mat<f64>`), plus the nalgebra-sparse / faer-sparse matrices.
/// `ndarray::Array1<f64>` produces a compile-time error per tenet 5 — it
/// has no honest [`Jacobian`] impl (no dense matrix type).
///
/// # State convention
///
/// `state.cost` carries the LM convention `½‖r‖²`, derived from the
/// residual the solver computes itself. The bound on `P` is
/// [`Residual`] + [`Jacobian`], not [`CostFunction`](crate::core::problem::CostFunction);
/// problems whose user-facing `cost()` uses an unscaled `Σ rᵢ²` form
/// (e.g. Rosenbrock-as-residuals) will see `state.cost()` differ from
/// `problem.cost(state.param())` by a factor of two. Both go to zero
/// at the optimum, so cost-based termination criteria are unaffected.
///
/// # Examples
///
/// Identical setup to [`LevenbergMarquardt`](crate::LevenbergMarquardt):
/// implement `Residual` + `Jacobian`, then drive a `NllsState` through
/// the `Executor`, swapping `LevenbergMarquardt::new()` for
/// `GaussNewton::new()`.
pub struct GaussNewton<V, M, F = f64> {
    tol_grad: F,

    // Residual and Jacobian caches across iterations. `r_cache` is set
    // to `r(x_new)` after the full GN step and reused at the top of the
    // next iter. `j_cache` is set only by `init` (init's `J(x₀)` is
    // reused for iter 0); after a step `J` is at the old iterate and
    // is dropped.
    r_cache: Option<V>,
    j_cache: Option<M>,
}

impl<V, M> Default for GaussNewton<V, M> {
    fn default() -> Self {
        Self::new()
    }
}

impl<V, M> GaussNewton<V, M> {
    /// Pure Gauss-Newton with the default first-order optimality
    /// tolerance (`tol_grad = 1e-8`).
    pub fn new() -> Self {
        Self {
            tol_grad: 1e-8,
            r_cache: None,
            j_cache: None,
        }
    }
}

impl<V, M, F: Scalar> GaussNewton<V, M, F> {
    /// First-order optimality tolerance: emit
    /// [`TerminationReason::SolverConverged`] when `‖Jᵀr‖_∞ ≤ tol`.
    /// Set to `0.0` to disable the check and rely solely on framework
    /// termination criteria. Default `1e-8`.
    pub fn with_tol_grad(mut self, tol: F) -> Self {
        assert!(tol >= F::zero(), "tol_grad must be ≥ 0");
        self.tol_grad = tol;
        self
    }
}

impl<V, M, F> InitialState<V> for GaussNewton<V, M, F>
where
    F: Scalar,
    V: Clone,
{
    type State = NllsState<V, F>;
    fn seed(&self, x: &V) -> Self::State {
        NllsState::new(x.clone())
    }
}

impl<P, V, M, F> Solver<P, NllsState<V, F>> for GaussNewton<V, M, F>
where
    F: Scalar,
    P: Residual<Param = V, Output = V> + Jacobian<Jacobian = M>,
    V: ScaledAdd<F> + NormSquared<F> + NormInfinity<F> + NegInPlace + Clone,
    M: GramMatrix + MatTransposeVec<V> + LinearSolveSpd<V>,
{
    type Error = <P as Residual>::Error;

    fn init(
        &mut self,
        problem: &mut Problem<P>,
        mut state: NllsState<V, F>,
    ) -> Result<NllsState<V, F>, Self::Error> {
        // Seed cost so iter-0 termination criteria see a populated
        // state. Both `r(x₀)` and `J(x₀)` are stashed so the first
        // `next_iter` doesn't re-evaluate them at the same point.
        let (r, j) = problem.residual_and_jacobian(&state.param)?;
        state.cost = Some(F::from_f64(0.5).unwrap() * r.norm_squared());
        self.r_cache = Some(r);
        self.j_cache = Some(j);
        Ok(state)
    }

    fn next_iter(
        &mut self,
        problem: &mut Problem<P>,
        mut state: NllsState<V, F>,
    ) -> Result<(NllsState<V, F>, Option<TerminationReason>), Self::Error> {
        let r = match self.r_cache.take() {
            Some(r) => r,
            None => problem.residual(&state.param)?,
        };
        let j = match self.j_cache.take() {
            Some(j) => j,
            None => problem.jacobian(&state.param)?,
        };

        // g = Jᵀr is the gradient of ½‖r‖². First-order optimality
        // (Madsen/Nielsen/Tingleff eq. 3.3a) is the canonical NLLS
        // convergence test.
        let g = j.mat_transpose_vec(&r);
        if self.tol_grad > F::zero() && g.norm_infinity() <= self.tol_grad {
            self.r_cache = Some(r);
            self.j_cache = Some(j);
            return Ok((state, Some(TerminationReason::SolverConverged)));
        }

        // Solve (JᵀJ) δ = −Jᵀr. Cholesky failure means JᵀJ is not
        // positive definite (rank-deficient J) — pure GN can't recover,
        // LM in S4 will.
        let gram = j.gram();
        let mut neg_g = g;
        neg_g.neg_in_place();
        let delta = match gram.solve_spd(&neg_g) {
            Ok(d) => d,
            Err(_) => {
                // State unchanged on Cholesky failure; restore caches
                // for any subsequent reuse (e.g. via `InnerExecutor`).
                self.r_cache = Some(r);
                self.j_cache = Some(j);
                return Ok((state, Some(TerminationReason::SolverFailed)));
            }
        };

        // Full GN step. Refresh state.cost from a fresh residual so the
        // post-iteration state is consistent (Solver contract); stash
        // that residual so the next iter reuses it without re-eval.
        // `J(x_new)` is not computed, so `j_cache` stays empty.
        state.param.scaled_add(F::one(), &delta);
        let r_new = problem.residual(&state.param)?;
        state.cost = Some(F::from_f64(0.5).unwrap() * r_new.norm_squared());
        self.r_cache = Some(r_new);
        self.j_cache = None;

        Ok((state, None))
    }
}
