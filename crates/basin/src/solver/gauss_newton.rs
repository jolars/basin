use crate::core::inner::InitialState;
use crate::core::least_squares::evaluation::{Evaluation, NllsStep};
use crate::core::math::{
    GramMatrix, LinearSolveSpd, MatTransposeVec, NegInPlace, NormInfinity,
    NormSquared, Scalar, ScaledAdd,
};
use crate::core::problem::{Jacobian, Problem, Residual};
use crate::core::solver::Solver;
use crate::core::state::NllsState;
use crate::core::termination::TerminationReason;
use crate::{
    LossFunction, RobustLeastSquares, ScaleRowsInPlace, VectorIndex, VectorLen,
};

/// Pure Gauss-Newton solver for nonlinear least-squares problems
/// `min ½‖r(x)‖²`.
///
/// Wrap the raw problem in [`RobustLeastSquares`] to minimize a robust loss.
/// The adapter supplies the safeguarded Gauss–Newton model, and reported
/// costs and convergence use the robust objective and its gradient.
///
/// Each iteration solves the normal equations `(JᵀJ) δ = −Jᵀr` via
/// Cholesky on the Gram matrix `JᵀJ` and takes the full step
/// `x ← x + δ`. No damping, no line search; that's what
/// Levenberg-Marquardt is for. See Madsen, Nielsen, Tingleff (2004),
/// *Methods for Non-Linear Least Squares Problems*, §3.1.
///
/// **Cholesky-on-`JᵀJ` vs QR-on-`J`.** Cholesky on the Gram is the
/// simple path and the only one the
/// [`linalg`](crate::core::math) tier exposes today. It squares the
/// condition number of `J` and fails noisily on rank-deficient `J`;
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
///   is the *correct* behavior for pure GN; Powell's singular
///   function is the canonical example. Reach for Levenberg-Marquardt
///   when this fires.
/// - **Divergence on highly nonlinear or poorly initialized problems.**
///   No safeguard here either; pure GN trusts the linear model. Catch
///   this with a finite [`max_iter`](crate::Executor::max_iter)
///   on the executor and inspect the returned state and stopping reason.
///
/// # Convergence
///
/// The native first-order test is `‖Jᵀr‖_∞ ≤ tolerance`, configured by
/// [`with_absolute_gradient_tolerance`](Self::with_absolute_gradient_tolerance).
/// Its default is `1e-8`; `None` disables it, and zero tests exact stationarity.
/// It reports [`TerminationReason::SolverConverged`] before computing a step.
/// Optional observed cost and step checks are disabled by default and combine
/// with this test using OR. Execution budgets belong on the executor.
///
/// The least-squares gradient is `Jᵀr`. It is computed inside the solver;
/// [`NllsState`] does not expose a [`GradientState`](crate::GradientState).
///
/// # Backends
///
/// LA-heavy: the default `Vec<f64>` backend (over the hand-rolled
/// [`DenseMatrix<f64>`](crate::DenseMatrix), via a pure-Rust Cholesky),
/// nalgebra (`DVector<f64>`/`DMatrix<f64>`), faer (`Col<f64>` /
/// `Mat<f64>`), and ndarray (`Array1<f64>`/`Array2<f64>`, the latter over
/// the same pure-Rust Cholesky), plus the nalgebra-sparse / faer-sparse
/// matrices.
/// All support `f32` and `f64`; use `GaussNewton::default()` for `f32`.
/// Robust objectives additionally require [`ScaleRowsInPlace`] on custom
/// Jacobian types.
///
/// # State convention
///
/// For an ordinary residual problem, `state.cost` carries the LM convention
/// `½‖r‖²`, derived from the
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
    tol_grad: Option<F>,

    // Residual and Jacobian caches across iterations. `r_cache` is set
    // to `r(x_new)` after the full GN step and reused at the top of the
    // next iter. `j_cache` is set only by `init` (init's `J(x₀)` is
    // reused for iter 0); after a step `J` is at the old iterate and
    // is dropped.
    r_cache: Option<V>,
    j_cache: Option<M>,
    model_r_cache: Option<V>,
    failed: bool,
}

impl<V, M, F: Scalar> Default for GaussNewton<V, M, F> {
    fn default() -> Self {
        Self {
            tol_grad: Some(F::from_f64(1e-8).unwrap()),
            r_cache: None,
            j_cache: None,
            model_r_cache: None,
            failed: false,
        }
    }
}

impl<V, M> GaussNewton<V, M> {
    /// Pure Gauss-Newton with the default first-order optimality
    /// tolerance (`tol_grad = 1e-8`).
    pub fn new() -> Self {
        Self::default()
    }
}

impl<V, M, F: Scalar> GaussNewton<V, M, F> {
    /// First-order optimality tolerance: emit
    /// [`TerminationReason::SolverConverged`] when `‖Jᵀr‖_∞ ≤ tol`.
    /// Set to `0.0` to disable the check and rely solely on framework
    /// termination criteria. Default `1e-8`.
    #[deprecated(
        note = "use `with_absolute_gradient_tolerance`; removal scheduled for Basin 2.0"
    )]
    pub fn with_tol_grad(mut self, tol: F) -> Self {
        assert!(tol >= F::zero(), "tol_grad must be ≥ 0");
        self.tol_grad = (tol > F::zero()).then_some(tol);
        self
    }

    /// Configure the infinity norm of J-transpose times residual.
    ///
    /// `None` disables the test; zero requests an exact-zero threshold.
    /// Values must be finite and nonnegative. Enabled tests combine with OR;
    /// each model-based test retains its internal conjunction and observation stage.
    /// Repeated calls replace this setting. Existing solver defaults are retained.
    pub fn with_absolute_gradient_tolerance(
        mut self,
        value: impl Into<Option<F>>,
    ) -> Self {
        self.tol_grad = crate::core::convergence::optional_tolerance(value);
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
        state: NllsState<V, F>,
    ) -> Result<NllsState<V, F>, Self::Error> {
        self.init_evaluated(problem, state)
    }
    fn next_iter(
        &mut self,
        problem: &mut Problem<P>,
        state: NllsState<V, F>,
    ) -> Result<(NllsState<V, F>, Option<TerminationReason>), Self::Error> {
        self.next_evaluated(problem, state)
    }
}

impl<P, L, V, M, F> Solver<RobustLeastSquares<P, L, F>, NllsState<V, F>>
    for GaussNewton<V, M, F>
where
    F: Scalar,
    P: Residual<Param = V, Output = V> + Jacobian<Jacobian = M>,
    V: ScaledAdd<F> + NormSquared<F> + NormInfinity<F> + NegInPlace + Clone,
    M: GramMatrix + MatTransposeVec<V> + LinearSolveSpd<V>,
    L: LossFunction<F>,
    V: VectorLen + VectorIndex<F>,
    M: ScaleRowsInPlace<F>,
{
    type Error = <P as Residual>::Error;
    fn init(
        &mut self,
        problem: &mut Problem<RobustLeastSquares<P, L, F>>,
        state: NllsState<V, F>,
    ) -> Result<NllsState<V, F>, Self::Error> {
        self.init_evaluated(problem, state)
    }
    fn next_iter(
        &mut self,
        problem: &mut Problem<RobustLeastSquares<P, L, F>>,
        state: NllsState<V, F>,
    ) -> Result<(NllsState<V, F>, Option<TerminationReason>), Self::Error> {
        self.next_evaluated(problem, state)
    }
}

impl<V, M, F: Scalar> GaussNewton<V, M, F>
where
    V: ScaledAdd<F> + NormSquared<F> + NormInfinity<F> + NegInPlace + Clone,
    M: GramMatrix + MatTransposeVec<V> + LinearSolveSpd<V>,
{
    fn init_evaluated<E: Evaluation<V, M, F>>(
        &mut self,
        problem: &mut E,
        mut state: NllsState<V, F>,
    ) -> Result<NllsState<V, F>, E::Error> {
        // Seed cost so iter-0 termination criteria see a populated
        // state. Both `r(x₀)` and `J(x₀)` are stashed so the first
        // `next_iter` doesn't re-evaluate them at the same point.
        self.failed = false;
        self.r_cache = None;
        self.j_cache = None;
        self.model_r_cache = None;
        let (r, j) = problem.residual_and_jacobian(&state.param)?;
        state.cost = Some(
            problem.cost(&r, |r| F::from_f64(0.5).unwrap() * r.norm_squared()),
        );
        let Some((model_r, j)) = problem.model(&r, j) else {
            self.failed = true;
            return Ok(state);
        };
        self.r_cache = Some(r);
        self.j_cache = Some(j);
        self.model_r_cache = model_r;
        Ok(state)
    }

    fn next_evaluated<E: Evaluation<V, M, F>>(
        &mut self,
        problem: &mut E,
        mut state: NllsState<V, F>,
    ) -> NllsStep<V, F, E::Error> {
        if self.failed {
            return Ok((state, Some(TerminationReason::SolverFailed)));
        }
        let r = match self.r_cache.take() {
            Some(r) => r,
            None => problem.residual(&state.param)?,
        };
        let (model_r, j) = match self.j_cache.take() {
            Some(j) => (self.model_r_cache.take(), j),
            None => {
                let j = problem.jacobian(&state.param)?;
                let Some(model) = problem.model(&r, j) else {
                    self.failed = true;
                    return Ok((state, Some(TerminationReason::SolverFailed)));
                };
                model
            }
        };

        // The corrected model preserves the objective gradient, so the same
        // first-order optimality test applies to ordinary and robust losses.
        let g = j.mat_transpose_vec(model_r.as_ref().unwrap_or(&r));
        if !problem.valid_vector(&g) {
            self.failed = true;
            return Ok((state, Some(TerminationReason::SolverFailed)));
        }
        if self.tol_grad.is_some_and(|tol| g.norm_infinity() <= tol) {
            self.r_cache = Some(r);
            self.j_cache = Some(j);
            self.model_r_cache = model_r;
            return Ok((state, Some(TerminationReason::SolverConverged)));
        }

        // Solve (JᵀJ) δ = −Jᵀr. Cholesky failure means JᵀJ is not
        // positive definite (rank-deficient J); pure GN can't recover,
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
                self.model_r_cache = model_r;
                return Ok((state, Some(TerminationReason::SolverFailed)));
            }
        };

        // Full GN step. Refresh state.cost from a fresh residual so the
        // post-iteration state is consistent (Solver contract); stash
        // that residual so the next iter reuses it without re-eval.
        // `J(x_new)` is not computed, so `j_cache` stays empty.
        state.param.scaled_add(F::one(), &delta);
        let r_new = problem.residual(&state.param)?;
        state.cost =
            Some(problem.cost(&r_new, |r| {
                F::from_f64(0.5).unwrap() * r.norm_squared()
            }));
        self.r_cache = Some(r_new);
        self.j_cache = None;

        let failed = E::ROBUST && !state.cost.unwrap().is_finite();
        self.failed = failed;
        Ok((state, failed.then_some(TerminationReason::SolverFailed)))
    }
}
