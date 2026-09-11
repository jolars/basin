use crate::core::math::{
    AddDiagonalVectorInPlace, ComponentDivAssign, ComponentMaxAssign,
    ComponentMulAssign, ComponentZip, Dot, FactorizePivotedQr,
    FloorZerosInPlace, GramMatrix, LinearSolveSpd, MatDiagonal,
    MatTransposeVec, NegInPlace, NormInfinity, NormSquared, QrSolveError,
    RegularizedQrSolve, Scalar, ScaleInPlace, ScaledAdd,
};
use crate::core::problem::{Jacobian, Problem, Residual};
use crate::core::solver::Solver;
use crate::core::state::NllsState;
use crate::core::termination::TerminationReason;

mod damping;
mod stopping;
use damping::{scaled_norm, trust_region_step, update_radius};
use stopping::{
    all_finite, finite_unchanged_trial, orthogonality_converged,
    relative_step_converged,
};

/// Damping-parameter selection for both Levenberg-Marquardt factorizations.
///
/// Both strategies use the same monotone Marquardt scaling matrix `D`, gain
/// ratio, and convergence tests. Configure with
/// [`LevenbergMarquardt::with_damping`] or
/// [`LevenbergMarquardtQr::with_damping`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum LmDamping {
    /// Nielsen's smooth gain-ratio update, starting at
    /// [`LevenbergMarquardt::with_tau`]. This is the default.
    #[default]
    Nielsen,
    /// Select damping to fit a scaled trust radius: `sqrt(hᵀDh) ≈ δ`.
    ///
    /// An undamped Gauss-Newton step is used when it fits. Otherwise, a
    /// safeguarded scalar search selects positive damping, reusing the linear
    /// model without extra residual or Jacobian callbacks. Radius initialization
    /// and updates follow MINPACK's [`lmder`](https://netlib.org/minpack/lmder.f).
    /// Configure the initial radius with
    /// [`LevenbergMarquardt::with_initial_step_bound`].
    ///
    /// This is not an exact MINPACK port: parameter selection uses bracketed
    /// solves instead of `lmpar`'s Newton corrections, rank-deficient undamped
    /// systems are regularized rather than truncated, and Basin's acceptance
    /// and stopping tests retain their documented meanings.
    TrustRegion,
}

/// Levenberg-Marquardt solver for nonlinear least-squares problems
/// `min ½‖r(x)‖²`, with Marquardt diagonal scaling and configurable damping.
/// The default is Nielsen's 1999 smooth μ-update; select
/// [`LmDamping::TrustRegion`] with [`Self::with_damping`] for radius-based
/// parameter selection.
///
/// With the default strategy, each iteration solves the damped normal equations
/// `(JᵀJ + μ·D) h = −Jᵀr` via Cholesky, then adapts the damping
/// parameter μ from the gain ratio
/// `ρ = (F(x) − F(x+h)) / (L(0) − L(h))` (Nielsen eq. 2.2). On a
/// successful step (ρ > 0) μ is reduced via the smooth cubic
/// `μ ← μ · max(1/3, 1 − (2ρ−1)³)`; on a failed step (ρ ≤ 0) μ grows
/// geometrically `μ ← μ·ν, ν ← 2ν` with ν initialized to 2; Nielsen
/// shows this avoids the discontinuities of the classical
/// multiply-or-divide threshold rule and lands roughly 25 % fewer
/// iterations on average. See Nielsen, *Damping Parameter in
/// Marquardt's Method* (IMM-REP-1999-05) for the derivation and
/// Madsen, Nielsen, Tingleff (2004), *Methods for Non-Linear Least
/// Squares Problems*, §3.2.
///
/// **Marquardt scaling (`μ·D`, not `μI`).** The damping matrix is the
/// diagonal of the Gram, `D = diag(JᵀJ)` (the per-parameter curvature),
/// rather than the identity. This makes the trust region ellipsoidal
/// in the metric of the columns of `J`, so the algorithm is invariant
/// to diagonal rescaling of the parameters (Marquardt 1963; Moré 1978,
/// *The Levenberg-Marquardt Algorithm: Implementation and Theory*).
/// Isotropic `μI` damping over-damps well-scaled directions and
/// under-damps poorly-scaled ones when the columns of `J` have very
/// different norms (e.g. parameters in a mixed log/linear/angle
/// encoding), which biases the step and can pull the iterate into a
/// worse basin. `D` is maintained as a **monotone running max**
/// `D_k = max(D_{k−1}, diag(J(x_k)ᵀJ(x_k)))` so a column whose
/// curvature momentarily drops keeps the damping floor it earned
/// earlier (Moré 1978; the same safeguard MINPACK applies to its
/// column-norm scaling). Columns that are exactly zero at `x₀` (a
/// parameter with no first-order effect on any residual) would make
/// `μ·D` vanish there and the Gram singular; following MINPACK, their
/// scale is floored to `1` at `init` (see `FloorZerosInPlace`), so a
/// fully-insensitive parameter stays put rather than failing Cholesky.
///
/// With Nielsen damping, `μ₀ = τ` is dimensionless because the
/// per-parameter magnitude now lives in `D` (the initial per-column
/// damping is `τ·diag(J(x₀)ᵀJ(x₀))`). τ is the *relative* trust
/// parameter; use a smaller value (e.g. `1e-6`) when `x₀` is believed
/// close to the optimum, larger (e.g. `1.0`) when far. Default
/// `τ = 10⁻³` matches Nielsen's "moderate trust" recommendation.
///
/// **Trust-region damping.** [`Self::with_damping`] can instead select
/// [`LmDamping::TrustRegion`]. It tries the undamped Gauss-Newton step first,
/// then searches for damping with `sqrt(hᵀDh)` within 10% of the current
/// radius. Rank safeguards or the inner attempt limit may yield a shorter
/// feasible regularized step. [`Self::with_initial_step_bound`] sets the initial
/// radius factor (default `100`); `tau` has no effect on this strategy.
/// Initialization and radius updates follow MINPACK's
/// [`lmder`](https://netlib.org/minpack/lmder.f), while a bracketed scalar
/// search replaces [`lmpar`](https://netlib.org/minpack/lmpar.f)'s Newton
/// corrections. Inner solves reuse the current model and require no additional
/// residual or Jacobian callbacks. Acceptance remains `ρ > 0`, and all
/// convergence settings below keep the same meaning, including the unscaled
/// relative step test. This option therefore does not reproduce MINPACK's
/// complete algorithm or stopping behavior.
///
/// **Linear solve.** Cholesky is the default and retains dense and sparse
/// backend coverage. [`Self::with_pivoted_qr`] selects
/// [`LevenbergMarquardtQr`], which solves the stacked least-squares system
/// `[J; √(μD)]` without forming `JᵀJ`. QR avoids normal-equation roundoff
/// near rank deficiency, but damping and stopping remain independent choices.
/// TRF also uses normal equations; its availability does not imply QR support.
///
/// # Failure modes
///
/// - **Cholesky failure under bumped μ.** Roundoff can defeat positive
///   definiteness when damping is small relative to the Jacobian's
///   conditioning. If no usable step is available, the inner loop increases
///   μ and retries, returning [`TerminationReason::SolverFailed`] if the attempt
///   cap is reached or damping overflows. Positive damping does not guarantee
///   accurate steps from normal equations. Initially zero columns have a unit scaling floor.
/// - **Divergence on highly nonlinear or poorly initialized problems.**
///   The damping itself prevents divergent steps (failed steps are
///   rejected via the gain-ratio test), so divergence manifests as
///   μ growing without bound. Catch this with
///   [`max_iter`](crate::Executor::max_iter) on the executor.
/// - **Unchanged finite trial.** By default, a rejected trial whose computed
///   step leaves every parameter unchanged reports
///   [`TerminationReason::NumericalNoProgress`]. This can occur at an accurate
///   rounded solution or an inaccurate, heavily damped point. An outer solver
///   may consume the result and continue. Configure this independently with
///   [`Self::with_no_progress_check`].
///
/// # Convergence
///
/// Four native tests combine with OR and report
/// [`TerminationReason::SolverConverged`]:
///
/// - [`with_absolute_gradient_tolerance`](Self::with_absolute_gradient_tolerance):
///   `‖Jᵀr‖_∞ ≤ tolerance`, default `1e-8`.
/// - [`with_gradient_orthogonality_tolerance`](Self::with_gradient_orthogonality_tolerance):
///   `max_j |(Jᵀr)_j| / (‖J[:,j]‖ · ‖r‖) ≤ tolerance`, disabled by default.
///   This dimensionless cosine test is invariant to residual scaling.
/// - [`with_relative_model_reduction_tolerance`](Self::with_relative_model_reduction_tolerance):
///   `|actred| ≤ tolerance · f`, `prered ≤ tolerance · f`, and `gain_ratio ≤ 2`,
///   disabled by default. Predicted reduction comes from the LM model, so this
///   differs from an observed cost-change test.
/// - [`with_relative_step_tolerance`](Self::with_relative_step_tolerance):
///   `‖h‖ ≤ tolerance · ‖x‖` for the internal trial step, disabled by default.
///
/// Each accepts a finite nonnegative scalar or `None`. Zero requests an
/// exact-zero threshold. Gradient tests run before computing a step; model
/// reduction and relative step tests run where the trial diagnostics are valid.
/// Observed absolute-step and cost-change checks are also opt-in and combine
/// with native checks using OR. Execution budgets belong on the executor.
/// The least-squares gradient `Jᵀr` is computed internally; [`NllsState`] does
/// not expose a [`GradientState`](crate::GradientState).
/// The numerical no-progress safeguard runs after native convergence tests.
/// Disabling convergence tests does not disable this safeguard; set
/// [`with_no_progress_check(false)`](Self::with_no_progress_check)
/// to retain the previous budget-stop behavior at an unchanged trial.
///
/// # Backends
///
/// LA-heavy: the default `Vec<f64>` backend (over the hand-rolled
/// [`DenseMatrix<f64>`](crate::DenseMatrix), via a pure-Rust Cholesky),
/// nalgebra (`DVector<f64>`/`DMatrix<f64>`), faer (`Col<f64>` /
/// `Mat<f64>`), and ndarray (`Array1<f64>`/`Array2<f64>`, the latter over
/// the same pure-Rust Cholesky) at the dense tier; nalgebra-sparse
/// (`DVector<f64>`/`CscMatrix<f64>`) and faer-sparse (`Col<f64>` /
/// `SparseColMat<usize, f64>`) at the sparse tier.
/// The sparse damping path requires the diagonal of `JᵀJ` to be in the
/// CSC pattern (always true when `J` has no zero columns); see
/// `AddDiagonalVectorInPlace` and `MatDiagonal`.
///
/// # State convention
///
/// `state.cost` carries the LM convention `½‖r‖²`, derived from the
/// residual the solver evaluates itself. The bound on `P` is
/// [`Residual`] + [`Jacobian`], not
/// [`CostFunction`](crate::core::problem::CostFunction); problems
/// whose user-facing `cost()` uses an unscaled `Σ rᵢ²` form will see
/// `state.cost()` differ from `problem.cost(state.param())` by a
/// factor of two. Both go to zero at the optimum, so cost-based
/// termination criteria are unaffected.
///
/// # Examples
///
/// Least-squares fit of an affine residual `r(x) = (x₀ − 1, x₁ − 2)` whose
/// minimum is `(1, 2)`. Levenberg–Marquardt binds on [`Residual`] +
/// [`Jacobian`] (not [`CostFunction`](crate::core::problem::CostFunction))
/// and runs on the matrix-capable backends:
///
/// ```
/// # #[cfg(feature = "nalgebra_v0_35")] {
/// use basin::{NllsState, Executor, Jacobian, LevenbergMarquardt, Residual};
/// use nalgebra::{DMatrix, DVector};
///
/// struct Affine;
/// impl Residual for Affine {
///     type Param = DVector<f64>;
///     type Output = DVector<f64>;
///     type Error = std::convert::Infallible;
///     fn residual(&self, x: &DVector<f64>) -> Result<DVector<f64>, Self::Error> {
///         Ok(DVector::from_vec(vec![x[0] - 1.0, x[1] - 2.0]))
///     }
/// }
/// impl Jacobian for Affine {
///     type Jacobian = DMatrix<f64>;
///     fn jacobian(&self, _x: &DVector<f64>) -> Result<DMatrix<f64>, Self::Error> {
///         Ok(DMatrix::identity(2, 2))
///     }
/// }
///
/// let result = Executor::new(
///     Affine,
///     LevenbergMarquardt::new(),
///     NllsState::new(DVector::from_vec(vec![0.0, 0.0])),
/// )
/// .max_iter(50)
/// .run()
/// .unwrap();
/// assert!((result.param()[0] - 1.0).abs() < 1e-6);
/// assert!((result.param()[1] - 2.0).abs() < 1e-6);
/// # }
/// ```
pub struct LevenbergMarquardt<V, M, F = f64> {
    tol_grad: Option<F>,
    tol_grad_rel: Option<F>,
    tol_cost_rel: Option<F>,
    tol_step_rel: Option<F>,
    numerical_no_progress: bool,
    tau: F,
    damping: LmDamping,
    initial_step_bound: F,
    radius: Option<F>,
    first_trust_step: bool,
    max_inner_attempts: u32,

    mu: Option<F>,
    nu: F,

    // Monotone Marquardt scaling diagonal D = max diag(JᵀJ). Zero
    // columns are floored to one so damping keeps the system nonsingular.
    diag: Option<V>,

    // Rejected steps leave these quantities valid. Accepted steps retain
    // the trial residual but invalidate the linear model and gradient.
    r_cache: Option<V>,
    model_cache: Option<Result<M, QrSolveError>>,
    jtr_cache: Option<V>,
}

impl<V, M> Default for LevenbergMarquardt<V, M> {
    fn default() -> Self {
        Self::new()
    }
}

impl<V, M> LevenbergMarquardt<V, M> {
    /// Levenberg-Marquardt with Nielsen damping: `tol_grad = 1e-8`,
    /// `tol_grad_rel = 0.0` (disabled), `tol_cost_rel = 0.0` (disabled),
    /// `tol_step_rel = 0.0` (disabled), `tau = 1e-3`, `max_inner_attempts = 50`,
    /// and numerical no-progress handling enabled.
    pub fn new() -> Self {
        Self::defaults()
    }
}

impl<V, M, F: Scalar> LevenbergMarquardt<V, M, F> {
    fn defaults() -> Self {
        Self {
            tol_grad: Some(F::from_f64(1e-8).unwrap()),
            tol_grad_rel: None,
            tol_cost_rel: None,
            tol_step_rel: None,
            numerical_no_progress: true,
            tau: F::from_f64(1e-3).unwrap(),
            damping: LmDamping::Nielsen,
            initial_step_bound: F::from_f64(100.0).unwrap(),
            radius: None,
            first_trust_step: true,
            max_inner_attempts: 50,
            mu: None,
            nu: F::from_f64(2.0).unwrap(),
            diag: None,
            r_cache: None,
            model_cache: None,
            jtr_cache: None,
        }
    }

    /// Absolute first-order optimality tolerance: emit
    /// [`TerminationReason::SolverConverged`] when `‖Jᵀr‖_∞ ≤ tol`
    /// (Madsen et al. eq. 3.3a). Set to `0.0` to disable the check and
    /// rely solely on [`with_tol_grad_rel`](Self::with_tol_grad_rel) and/or
    /// framework termination criteria. Default `1e-8`.
    #[allow(deprecated)]
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

    /// Relative (scale-invariant) first-order optimality tolerance,
    /// the MINPACK `gtol` test (Moré 1978): emit
    /// [`TerminationReason::SolverConverged`] when the cosine of the
    /// angle between the residual `r` and every Jacobian column is at
    /// most `tol`, i.e. `max_j |gⱼ| / (‖J·,ⱼ‖ · ‖r‖) ≤ tol` with
    /// `g = Jᵀr`. Being a dimensionless cosine, it is invariant to
    /// scaling of the residuals, so one tolerance ports across problems
    /// with different residual normalizations, unlike the absolute
    /// [`with_tol_grad`](Self::with_tol_grad). Set to `0.0` to disable. Default
    /// `0.0` (disabled); use e.g. `1e-8` for MINPACK `gtol` parity.
    ///
    /// Both gradient tests can be active at once; the solver converges
    /// when *either* fires (matching MINPACK, which checks `ftol`,
    /// `xtol`, and `gtol` independently).
    #[allow(deprecated)]
    #[deprecated(
        note = "use `with_gradient_orthogonality_tolerance`; removal scheduled for Basin 2.0"
    )]
    pub fn with_tol_grad_rel(mut self, tol: F) -> Self {
        assert!(tol >= F::zero(), "tol_grad_rel must be ≥ 0");
        self.tol_grad_rel = (tol > F::zero()).then_some(tol);
        self
    }

    /// Configure the maximum residual/Jacobian-column absolute cosine.
    ///
    /// `None` disables the test; zero requests an exact-zero threshold.
    /// Values must be finite and nonnegative. Enabled tests combine with OR;
    /// each model-based test retains its internal conjunction and observation stage.
    /// Repeated calls replace this setting. Existing solver defaults are retained.
    pub fn with_gradient_orthogonality_tolerance(
        mut self,
        value: impl Into<Option<F>>,
    ) -> Self {
        self.tol_grad_rel = crate::core::convergence::optional_tolerance(value);
        self
    }

    /// Relative cost-reduction tolerance, the MINPACK `ftol` test
    /// (Moré 1978): emit [`TerminationReason::SolverConverged`] when both
    /// the *actual* and the *predicted* reduction in `½‖r‖²` over an
    /// iteration are at most `tol` relative to the current cost, and the
    /// gain ratio is sane:
    ///
    /// ```text
    /// |actred| ≤ tol·F   AND   prered ≤ tol·F   AND   ρ ≤ 2
    /// ```
    ///
    /// with `actred = F(x) − F(x+h)`, `prered = L(0) − L(h)` the model's
    /// predicted reduction, `F = ½‖r‖²`, and `ρ = actred/prered`.
    ///
    /// The `prered` clause is the load-bearing difference from the
    /// framework's [`RelativeCostTolerance`], which sees only the
    /// achieved reduction between consecutive costs and has no access to
    /// the LM model. Predicted reduction is evaluated at the damped step;
    /// excessive damping can make both reductions small even when a weak
    /// direction remains unresolved. This check does not establish parameter
    /// recovery. This model-dependent check belongs on the solver rather
    /// than in the termination layer. Basin uses Nielsen damping and a step
    /// norm test; its complete stopping behavior is not identical to MINPACK.
    ///
    /// Set to `0.0` to disable. Default `0.0` (disabled); use e.g. `1e-8`
    /// for MINPACK `ftol` parity. Converges when *any* enabled test fires
    /// (see [`with_tol_grad`](Self::with_tol_grad)).
    ///
    /// [`RelativeCostTolerance`]: crate::core::termination::RelativeCostTolerance
    #[allow(deprecated)]
    #[deprecated(
        note = "use `with_relative_model_reduction_tolerance`; removal scheduled for Basin 2.0"
    )]
    pub fn with_tol_cost_rel(mut self, tol: F) -> Self {
        assert!(tol >= F::zero(), "tol_cost_rel must be ≥ 0");
        self.tol_cost_rel = (tol > F::zero()).then_some(tol);
        self
    }

    /// Configure both actual and predicted reduction relative to the current cost, with gain ratio at most two.
    ///
    /// `None` disables the test; zero requests an exact-zero threshold.
    /// Values must be finite and nonnegative. Enabled tests combine with OR;
    /// each model-based test retains its internal conjunction and observation stage.
    /// Repeated calls replace this setting. Existing solver defaults are retained.
    pub fn with_relative_model_reduction_tolerance(
        mut self,
        value: impl Into<Option<F>>,
    ) -> Self {
        self.tol_cost_rel = crate::core::convergence::optional_tolerance(value);
        self
    }

    /// Relative step tolerance, the MINPACK `xtol` test (Moré 1978):
    /// emit [`TerminationReason::SolverConverged`] when the accepted (or
    /// attempted) step is negligible relative to the iterate,
    /// `‖h‖ ≤ tol·‖x‖`. Nielsen's smooth μ-update carries no explicit
    /// trust radius `δ`, so the step norm is the natural analog of
    /// MINPACK's `delta ≤ xtol·xnorm`. Set to `0.0` to disable. Default
    /// `0.0` (disabled); use e.g. `1e-8` for MINPACK `xtol` parity.
    /// Converges when *any* enabled test fires (see
    /// [`with_tol_grad`](Self::with_tol_grad)).
    #[allow(deprecated)]
    #[deprecated(
        note = "use `with_relative_step_tolerance`; removal scheduled for Basin 2.0"
    )]
    pub fn with_tol_step_rel(mut self, tol: F) -> Self {
        assert!(tol >= F::zero(), "tol_step_rel must be ≥ 0");
        self.tol_step_rel = (tol > F::zero()).then_some(tol);
        self
    }

    /// Configure the attempted step norm relative to the current iterate norm.
    ///
    /// `None` disables the test; zero requests an exact-zero threshold.
    /// Values must be finite and nonnegative. Enabled tests combine with OR;
    /// each model-based test retains its internal conjunction and observation stage.
    /// Repeated calls replace this setting. Existing solver defaults are retained.
    pub fn with_relative_step_tolerance(
        mut self,
        value: impl Into<Option<F>>,
    ) -> Self {
        self.tol_step_rel = crate::core::convergence::optional_tolerance(value);
        self
    }

    /// Stop after one rejected finite trial whose computed `x + h` equals
    /// the base `x` componentwise. Enabled by default.
    ///
    /// Reports [`TerminationReason::NumericalNoProgress`], which permits an
    /// outer solver to consume the result but makes no convergence, fit, or
    /// parameter-recovery claim. Excessive damping can trigger this safeguard
    /// at an inaccurate point. A rejected trial with different coordinates
    /// does not qualify, even though the stored iterate remains unchanged.
    ///
    /// The usual trial residual callback and damping updates still run. The
    /// parameters, step, residuals, gradient, costs, and reduction diagnostics
    /// must be finite. Native convergence tests take precedence; callback
    /// errors and model-solve failures retain their existing routing.
    /// As a mid-iteration stop, it precedes observed-step and cost-change
    /// checks that would run at the next iteration boundary.
    ///
    /// This safeguard is independent of convergence tolerances: `None` still
    /// disables its particular test, and zero still requests exact zero.
    /// Set this to `false` to recover the previous behavior when all
    /// convergence tests are disabled, including budget stops at zero residual.
    /// Repeated calls replace the setting. Configure before starting a solve.
    pub fn with_no_progress_check(mut self, enabled: bool) -> Self {
        self.numerical_no_progress = enabled;
        self
    }

    /// Select how damping is adapted, independently of the factorization.
    ///
    /// Default: [`LmDamping::Nielsen`]. Both strategies preserve the same
    /// scaling, acceptance (`gain_ratio > 0`), and stopping tests. Configure
    /// this before starting a solve.
    pub fn with_damping(mut self, damping: LmDamping) -> Self {
        self.damping = damping;
        self
    }

    /// Initial scaled trust-radius factor, used by [`LmDamping::TrustRegion`].
    ///
    /// Sets `δ₀ = factor * sqrt(x₀ᵀ D₀ x₀)`, or `factor` when the scaled
    /// starting point is zero. `D₀` is the Marquardt diagonal, including its
    /// unit floor for zero columns. Default: `100`, as in MINPACK. Smaller
    /// values restrict the first step. Has no effect on Nielsen damping.
    ///
    /// # Panics
    ///
    /// Panics unless `factor` is finite and strictly positive.
    pub fn with_initial_step_bound(mut self, factor: F) -> Self {
        assert!(
            factor.is_finite() && factor > F::zero(),
            "initial step bound must be finite and > 0"
        );
        self.initial_step_bound = factor;
        self
    }

    /// Relative initial damping `τ`: `μ₀ = τ`, giving an initial
    /// per-column damping of `τ·diag(J(x₀)ᵀJ(x₀))` under Marquardt
    /// scaling. Use a smaller value (e.g. `1e-6`) when `x₀` is believed
    /// close to the optimum; a larger value (e.g. `1.0`) when far from
    /// it. Default `1e-3` (Nielsen's "moderate trust"). Has no effect on
    /// [`LmDamping::TrustRegion`].
    pub fn with_tau(mut self, tau: F) -> Self {
        assert!(tau > F::zero(), "tau must be > 0");
        self.tau = tau;
        self
    }

    /// Maximum number of damping bumps inside a single outer iteration
    /// before giving up with [`TerminationReason::SolverFailed`]. With Nielsen
    /// damping, each bump multiplies μ by ν (initially 2) and doubles ν;
    /// overflow can end retries before the cap. Default `50`.
    ///
    /// With trust-region damping, this caps all model solves per iteration,
    /// including the undamped attempt and the radius search. At the limit,
    /// the smallest-damping feasible step found is used; if none was found,
    /// the solver reports `SolverFailed` without a trial residual callback.
    pub fn with_max_inner_attempts(mut self, n: u32) -> Self {
        assert!(n > 0, "max_inner_attempts must be > 0");
        self.max_inner_attempts = n;
        self
    }
}

impl<P, V, M, F> Solver<P, NllsState<V, F>> for LevenbergMarquardt<V, M, F>
where
    F: Scalar,
    P: Residual<Param = V, Output = V> + Jacobian<Jacobian = M>,
    V: ScaledAdd<F>
        + NormSquared<F>
        + NormInfinity<F>
        + NegInPlace
        + Dot<F>
        + ScaleInPlace<F>
        + ComponentMulAssign
        + ComponentDivAssign
        + ComponentZip<F>
        + ComponentMaxAssign
        + FloorZerosInPlace<F>
        + Clone,
    M: GramMatrix
        + MatTransposeVec<V>
        + LinearSolveSpd<V>
        + AddDiagonalVectorInPlace<V>
        + MatDiagonal<V>
        + Clone,
{
    type Error = <P as Residual>::Error;
    fn init(
        &mut self,
        problem: &mut Problem<P>,
        state: NllsState<V, F>,
    ) -> Result<NllsState<V, F>, Self::Error> {
        self.init_model::<P, M, NormalEquations>(problem, state)
    }
    fn next_iter(
        &mut self,
        problem: &mut Problem<P>,
        state: NllsState<V, F>,
    ) -> Result<(NllsState<V, F>, Option<TerminationReason>), Self::Error> {
        self.next_iter_model::<P, M, NormalEquations>(problem, state, None)
    }
}

impl<V, C, F: Scalar> LevenbergMarquardt<V, C, F> {
    fn init_model<P, M, Model>(
        &mut self,
        problem: &mut Problem<P>,
        mut state: NllsState<V, F>,
    ) -> Result<NllsState<V, F>, <P as Residual>::Error>
    where
        P: Residual<Param = V, Output = V> + Jacobian<Jacobian = M>,
        M: MatTransposeVec<V>,
        Model: LinearModel<M, V, F, Cache = C>,
        V: ScaledAdd<F>
            + NormSquared<F>
            + NormInfinity<F>
            + NegInPlace
            + Dot<F>
            + ScaleInPlace<F>
            + ComponentMulAssign
            + ComponentDivAssign
            + ComponentZip<F>
            + ComponentMaxAssign
            + FloorZerosInPlace<F>
            + Clone,
    {
        // Seed both the state and the cross-iteration caches from one
        // residual/Jacobian evaluation.
        let (r, j) = problem.residual_and_jacobian(&state.param)?;
        state.cost = Some(F::from_f64(0.5).unwrap() * r.norm_squared());

        let a = Model::prepare(&j, &r);
        self.diag = a.as_ref().ok().map(|a| {
            let mut d = Model::diagonal(a);
            d.floor_zeros_in_place(F::one());
            d
        });

        self.radius = if self.damping == LmDamping::TrustRegion {
            self.diag.as_ref().map(|d| {
                let xnorm = scaled_norm(&state.param, d);
                self.initial_step_bound
                    * if xnorm == F::zero() { F::one() } else { xnorm }
            })
        } else {
            None
        };
        self.first_trust_step = true;
        self.mu = Some(if self.damping == LmDamping::Nielsen {
            self.tau
        } else {
            F::zero()
        });
        self.nu = F::from_f64(2.0).unwrap();
        self.jtr_cache = Some(j.mat_transpose_vec(&r));
        self.model_cache = Some(a);
        self.r_cache = Some(r);
        Ok(state)
    }

    fn next_iter_model<P, M, Model>(
        &mut self,
        problem: &mut Problem<P>,
        mut state: NllsState<V, F>,
        rank_tolerance: Option<F>,
    ) -> LmStep<V, F, <P as Residual>::Error>
    where
        P: Residual<Param = V, Output = V> + Jacobian<Jacobian = M>,
        M: MatTransposeVec<V>,
        Model: LinearModel<M, V, F, Cache = C>,
        V: ScaledAdd<F>
            + NormSquared<F>
            + NormInfinity<F>
            + NegInPlace
            + Dot<F>
            + ScaleInPlace<F>
            + ComponentMulAssign
            + ComponentDivAssign
            + ComponentZip<F>
            + ComponentMaxAssign
            + FloorZerosInPlace<F>
            + Clone,
    {
        let r = match self.r_cache.take() {
            Some(r) => r,
            None => problem.residual(&state.param)?,
        };

        let (a, g) = match (self.model_cache.take(), self.jtr_cache.take()) {
            (Some(a), Some(g)) => (a, g),
            _ => {
                let j = problem.jacobian(&state.param)?;
                (Model::prepare(&j, &r), j.mat_transpose_vec(&r))
            }
        };
        let a = match a {
            Ok(a) => a,
            Err(error) => {
                self.model_cache = Some(Err(error));
                self.r_cache = Some(r);
                self.jtr_cache = Some(g);
                return Ok((state, Some(TerminationReason::SolverFailed)));
            }
        };
        // Squaring a finite gradient can overflow even when the QR step is valid.
        if (Model::CHECK_FINITE || self.damping == LmDamping::TrustRegion)
            && (!r.norm_squared().is_finite() || !g.norm_infinity().is_finite())
        {
            self.model_cache = Some(Ok(a));
            self.r_cache = Some(r);
            self.jtr_cache = Some(g);
            return Ok((state, Some(TerminationReason::SolverFailed)));
        }
        let diag_cur = Model::diagonal(&a);

        // MINPACK's absolute and relative first-order tests:
        //   * absolute   ‖Jᵀr‖_∞ ≤ tol_grad           (Madsen et al. 3.3a)
        //   * relative   max_j |gⱼ|/(‖J·,ⱼ‖·‖r‖) ≤ tol_grad_rel  (MINPACK gtol)
        let abs_converged =
            self.tol_grad.is_some_and(|tol| g.norm_infinity() <= tol);
        let rel_converged = self
            .tol_grad_rel
            .is_some_and(|tol| orthogonality_converged(&g, &diag_cur, &r, tol));
        if abs_converged || rel_converged {
            // Termination does not move the iterate, so the caches remain valid.
            self.r_cache = Some(r);
            self.model_cache = Some(Ok(a));
            self.jtr_cache = Some(g);
            return Ok((state, Some(TerminationReason::SolverConverged)));
        }

        // Moré's monotone scaling keeps the damped Gram positive definite.
        let mut d = self
            .diag
            .take()
            .expect("diag not set: Solver::init must run before next_iter");
        d.component_max_assign(&diag_cur);

        let mut mu = self
            .mu
            .expect("mu not set: Solver::init must run before next_iter");
        let mut nu = self.nu;

        // Increase damping when the model solve reports recoverable rank loss.
        let two = F::from_f64(2.0).unwrap();
        let half = F::from_f64(0.5).unwrap();
        let one_third = F::from_f64(1.0 / 3.0).unwrap();
        let step = if self.damping == LmDamping::TrustRegion {
            let radius = self.radius.expect("trust radius not initialized");
            let mut scaled_gradient = g.clone();
            scaled_gradient.component_div_assign(&d);
            let gradient_norm = g.dot(&scaled_gradient).sqrt();
            trust_region_step(
                radius,
                &mut mu,
                self.max_inner_attempts,
                gradient_norm,
                |mu| Model::solve(&a, &g, &d, mu, rank_tolerance),
                |h| scaled_norm(h, &d),
            )
        } else {
            let mut attempts = 0;
            loop {
                match Model::solve(&a, &g, &d, mu, rank_tolerance) {
                    Ok(step) => break Ok(step),
                    Err(failure) => {
                        attempts += 1;
                        if failure == ModelSolveError::Failed
                            || attempts >= self.max_inner_attempts
                            || !mu.is_finite()
                        {
                            break Err(failure);
                        }
                        mu = mu * nu;
                        nu = nu * two;
                    }
                }
            }
        };
        let h = match step {
            Ok(h) => h,
            Err(_) => {
                self.mu = Some(mu);
                self.nu = nu;
                self.diag = Some(d);
                self.r_cache = Some(r);
                self.model_cache = Some(Ok(a));
                self.jtr_cache = Some(g);
                return Ok((state, Some(TerminationReason::SolverFailed)));
            }
        };

        // Predicted reduction, Nielsen eq. 2.3, with diagonal scaling.
        // Form hᵀDh as h·(D ⊙ h) without materializing μDh − g.
        let mut dh = d.clone();
        dh.component_mul_assign(&h);
        let l_diff = half * (mu * h.dot(&dh) - h.dot(&g));

        let mut x_trial = state.param.clone();
        x_trial.scaled_add(F::one(), &h);
        let r_trial = problem.residual(&x_trial)?;
        state.cost_evals += 1;
        let f_trial = half * r_trial.norm_squared();

        let prev_cost = state
            .cost
            .expect("cost not set: Solver::init must run before next_iter");
        let actual_diff = prev_cost - f_trial;
        let rho = if l_diff > F::zero() {
            actual_diff / l_diff
        } else {
            F::zero()
        };

        // Compare trial coordinates before acceptance can replace the base.
        // Rejection alone leaves the iterate unchanged without proving that
        // the computed step was lost to rounding.
        let numerical_no_progress = self.numerical_no_progress
            && rho <= F::zero()
            && finite_unchanged_trial(&state.param, &x_trial)
            && [prev_cost, f_trial, actual_diff, l_diff, rho]
                .iter()
                .all(|value| value.is_finite())
            && [&h, &r, &r_trial, &g].into_iter().all(all_finite);

        if self.damping == LmDamping::TrustRegion {
            let pnorm = h.dot(&dh).sqrt();
            let radius =
                self.radius.as_mut().expect("trust radius not initialized");
            if self.first_trust_step && pnorm > F::zero() {
                *radius = radius.min(pnorm);
            }
            update_radius(radius, &mut mu, pnorm, rho, actual_diff, h.dot(&g));
        }

        if rho > F::zero() {
            // Nielsen eq. 2.5 with β=2, γ=3, p=3.
            state.param = x_trial;
            state.cost = Some(f_trial);
            if self.damping == LmDamping::Nielsen {
                let factor = F::one() - (two * rho - F::one()).powi(3);
                mu = mu * factor.max(one_third);
                nu = two;
            }
            self.first_trust_step = false;
            self.r_cache = Some(r_trial);
            self.model_cache = None;
            self.jtr_cache = None;
        } else {
            // Preserve iterate-dependent caches and increase damping.
            if self.damping == LmDamping::Nielsen {
                mu = mu * nu;
                nu = nu * two;
            }
            self.r_cache = Some(r);
            self.model_cache = Some(Ok(a));
            self.jtr_cache = Some(g);
        }

        self.mu = Some(mu);
        self.nu = nu;
        self.diag = Some(d);

        // Check MINPACK's ftol and xtol after committing an accepted step.
        //
        //   * tol_cost_rel  |actred| ≤ tol·F  AND  prered ≤ tol·F  AND  ρ ≤ 2.
        //     `|actred|` mirrors MINPACK's `dabs(actred)`.
        //   * tol_step_rel  ‖h‖ ≤ tol_step_rel·‖x‖, the step is negligible
        //     relative to the iterate, including after a rejected trial.
        let cost_rel_converged = self.tol_cost_rel.is_some_and(|tol| {
            actual_diff.abs() <= tol * prev_cost
                && l_diff <= tol * prev_cost
                && rho <= two
        });
        let step_rel_converged = self
            .tol_step_rel
            .is_some_and(|tol| relative_step_converged(&h, &state.param, tol));
        if cost_rel_converged || step_rel_converged {
            return Ok((state, Some(TerminationReason::SolverConverged)));
        }

        if numerical_no_progress {
            return Ok((state, Some(TerminationReason::NumericalNoProgress)));
        }

        Ok((state, None))
    }
}

type LmStep<V, F, E> = Result<(NllsState<V, F>, Option<TerminationReason>), E>;

#[derive(PartialEq)]
enum ModelSolveError {
    Retry,
    Failed,
}

trait LinearModel<M, V, F: Scalar> {
    type Cache;
    const CHECK_FINITE: bool;
    fn prepare(j: &M, r: &V) -> Result<Self::Cache, QrSolveError>;
    fn diagonal(cache: &Self::Cache) -> V;
    // Only rank loss can be repaired by increasing damping on the QR route.
    fn solve(
        cache: &Self::Cache,
        g: &V,
        d: &V,
        mu: F,
        tolerance: Option<F>,
    ) -> Result<V, ModelSolveError>;
}
struct NormalEquations;
impl<M, V, F: Scalar> LinearModel<M, V, F> for NormalEquations
where
    M: GramMatrix
        + MatDiagonal<V>
        + LinearSolveSpd<V>
        + AddDiagonalVectorInPlace<V>
        + Clone,
    V: Clone + NegInPlace + ScaleInPlace<F>,
{
    type Cache = M;
    const CHECK_FINITE: bool = false;
    fn prepare(j: &M, _: &V) -> Result<M, QrSolveError> {
        Ok(j.gram())
    }
    fn diagonal(cache: &M) -> V {
        cache.diagonal()
    }
    fn solve(
        cache: &M,
        g: &V,
        d: &V,
        mu: F,
        _: Option<F>,
    ) -> Result<V, ModelSolveError> {
        let mut a = cache.clone();
        let mut diagonal = d.clone();
        diagonal.scale_in_place(mu);
        a.add_diagonal_vector_in_place(&diagonal);
        let mut rhs = g.clone();
        rhs.neg_in_place();
        a.solve_spd(&rhs).map_err(|_| ModelSolveError::Retry)
    }
}
struct PivotedQr;
impl<M, V, F: Scalar> LinearModel<M, V, F> for PivotedQr
where
    M: FactorizePivotedQr<V, F>,
    V: Clone + NegInPlace,
{
    type Cache = M::Factorization;
    const CHECK_FINITE: bool = true;
    fn prepare(j: &M, r: &V) -> Result<Self::Cache, QrSolveError> {
        let mut rhs = r.clone();
        rhs.neg_in_place();
        j.factorize_pivoted_qr(&rhs)
    }
    fn diagonal(cache: &Self::Cache) -> V {
        cache.column_norms_squared()
    }
    fn solve(
        cache: &Self::Cache,
        _: &V,
        d: &V,
        mu: F,
        tolerance: Option<F>,
    ) -> Result<V, ModelSolveError> {
        cache.solve_regularized(mu, d, tolerance).map_err(|e| {
            if e == QrSolveError::RankDeficient {
                ModelSolveError::Retry
            } else {
                ModelSolveError::Failed
            }
        })
    }
}

/// Levenberg-Marquardt with column-pivoted QR and configurable damping.
///
/// Construct with [`LevenbergMarquardt::with_pivoted_qr`] or [`Self::new`].
/// This uses the same scaling, gain ratio, damping update, and stopping tests
/// as [`LevenbergMarquardt`], solving `[J; sqrt(μD)] h ≈ [-r; 0]` without
/// forming `JᵀJ`. Nielsen damping is the default; [`Self::with_damping`]
/// selects the optional scaled trust-radius strategy. QR improves step accuracy
/// near rank deficiency; neither strategy guarantees MINPACK's nonlinear
/// convergence trajectory or evaluation counts.
///
/// # Rank and failures
///
/// Rank is checked after diagonal regularization and column equilibration.
/// The default threshold is `epsilon(F) * (m+n)`; see
/// [`Self::with_rank_tolerance`]. Nielsen damping increases after rank loss,
/// reusing the factorization, until the attempt limit yields `SolverFailed`.
/// Trust-region damping uses rank loss to bound the parameter search, retaining
/// a feasible regularized step if available and otherwise yielding `SolverFailed`.
/// Non-finite factorization or solve arithmetic yields `SolverFailed`
/// immediately. Problem callback errors propagate unchanged. No truncated
/// solution or normal-equation fallback is used.
/// The default numerical no-progress safeguard and its opt-out are shared
/// with Cholesky; see [`Self::with_no_progress_check`].
///
/// # Backends
///
/// `Vec<F>` with [`DenseMatrix`](crate::DenseMatrix), nalgebra
/// `DVector<F>`/`DMatrix<F>`, ndarray `Array1<F>`/`Array2<F>`, and faer
/// `Col<F>`/`Mat<F>`, for `f32` and `f64`, in pure Rust. Sparse matrices
/// deliberately lack [`FactorizePivotedQr`]: nalgebra-sparse has no QR, and
/// faer's sparse QR does not provide numerical column pivoting.
///
/// # References
///
/// Madsen, Nielsen & Tingleff (2004), *Methods for Non-Linear Least Squares
/// Problems*, §3.2; MINPACK's [`qrfac`](https://netlib.org/minpack/qrfac.f)
/// and [`qrsolv`](https://netlib.org/minpack/qrsolv.f) (Garbow, Hillstrom &
/// Moré, 1980). Optional trust-region damping follows MINPACK's
/// [`lmder`](https://netlib.org/minpack/lmder.f) radius updates, using bracketed
/// parameter selection instead of [`lmpar`](https://netlib.org/minpack/lmpar.f).
/// The rank policy differs from MINPACK's truncated solve.
///
/// # Example
///
/// ```
/// use basin::{DenseMatrix, LevenbergMarquardt, LevenbergMarquardtQr};
/// let solver: LevenbergMarquardtQr<Vec<f64>, DenseMatrix> =
///     LevenbergMarquardt::new()
///         .with_absolute_gradient_tolerance(1e-10)
///         .with_pivoted_qr();
/// ```
pub struct LevenbergMarquardtQr<V, M, F: Scalar = f64>
where
    M: FactorizePivotedQr<V, F>,
{
    inner: LevenbergMarquardt<V, M::Factorization, F>,
    rank_tolerance: Option<F>,
}

impl<V, M, F: Scalar> LevenbergMarquardt<V, M, F> {
    /// Select pivoted QR while preserving configuration and resetting caches.
    ///
    /// This additive route requires [`FactorizePivotedQr`] only on the
    /// returned solver. Existing Cholesky-only matrix implementations retain
    /// their original solver bounds. Configure this before starting a solve.
    pub fn with_pivoted_qr(self) -> LevenbergMarquardtQr<V, M, F>
    where
        M: FactorizePivotedQr<V, F>,
    {
        LevenbergMarquardtQr {
            inner: LevenbergMarquardt {
                tol_grad: self.tol_grad,
                tol_grad_rel: self.tol_grad_rel,
                tol_cost_rel: self.tol_cost_rel,
                tol_step_rel: self.tol_step_rel,
                numerical_no_progress: self.numerical_no_progress,
                tau: self.tau,
                damping: self.damping,
                initial_step_bound: self.initial_step_bound,
                max_inner_attempts: self.max_inner_attempts,
                ..LevenbergMarquardt::defaults()
            },
            rank_tolerance: None,
        }
    }
}

impl<V, M, F: Scalar> Default for LevenbergMarquardtQr<V, M, F>
where
    M: FactorizePivotedQr<V, F>,
{
    fn default() -> Self {
        Self::new()
    }
}
impl<V, M, F: Scalar> LevenbergMarquardtQr<V, M, F>
where
    M: FactorizePivotedQr<V, F>,
{
    /// QR with the same defaults as [`LevenbergMarquardt::new`].
    pub fn new() -> Self {
        Self {
            inner: LevenbergMarquardt::defaults(),
            rank_tolerance: None,
        }
    }
    /// Override the dimensionless augmented-system rank threshold.
    ///
    /// Default: `epsilon(F) * (m+n)`. Finite values in `[0,1)` are valid;
    /// other values panic. Zero detects only exactly zero triangular pivots.
    /// See [`RegularizedQrSolve::solve_regularized`] for the rank contract.
    #[allow(deprecated)]
    #[deprecated(
        note = "use `with_relative_rank_tolerance`; removal scheduled for Basin 2.0"
    )]
    pub fn with_rank_tolerance(self, tol: F) -> Self {
        self.with_relative_rank_tolerance(tol)
    }

    /// Configure the relative rank tolerance.
    /// Retains the algorithm's existing formula, validation, and default.
    pub fn with_relative_rank_tolerance(mut self, tol: F) -> Self {
        assert!(
            tol.is_finite() && tol >= F::zero() && tol < F::one(),
            "rank tolerance must be finite and in [0,1)"
        );
        self.rank_tolerance = Some(tol);
        self
    }
    /// Configure [`LevenbergMarquardt::with_tol_grad`] for the QR route.
    #[allow(deprecated)]
    #[deprecated(
        note = "use `with_absolute_gradient_tolerance`; removal scheduled for Basin 2.0"
    )]
    pub fn with_tol_grad(mut self, value: F) -> Self {
        self.inner = self.inner.with_tol_grad(value);
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
        self.inner = self.inner.with_absolute_gradient_tolerance(value);
        self
    }
    /// Configure [`LevenbergMarquardt::with_tol_grad_rel`] for the QR route.
    #[allow(deprecated)]
    #[deprecated(
        note = "use `with_gradient_orthogonality_tolerance`; removal scheduled for Basin 2.0"
    )]
    pub fn with_tol_grad_rel(mut self, value: F) -> Self {
        self.inner = self.inner.with_tol_grad_rel(value);
        self
    }

    /// Configure the maximum residual/Jacobian-column absolute cosine.
    ///
    /// `None` disables the test; zero requests an exact-zero threshold.
    /// Values must be finite and nonnegative. Enabled tests combine with OR;
    /// each model-based test retains its internal conjunction and observation stage.
    /// Repeated calls replace this setting. Existing solver defaults are retained.
    pub fn with_gradient_orthogonality_tolerance(
        mut self,
        value: impl Into<Option<F>>,
    ) -> Self {
        self.inner = self.inner.with_gradient_orthogonality_tolerance(value);
        self
    }
    /// Configure [`LevenbergMarquardt::with_tol_cost_rel`] for the QR route.
    #[allow(deprecated)]
    #[deprecated(
        note = "use `with_relative_model_reduction_tolerance`; removal scheduled for Basin 2.0"
    )]
    pub fn with_tol_cost_rel(mut self, value: F) -> Self {
        self.inner = self.inner.with_tol_cost_rel(value);
        self
    }

    /// Configure both actual and predicted reduction relative to the current cost, with gain ratio at most two.
    ///
    /// `None` disables the test; zero requests an exact-zero threshold.
    /// Values must be finite and nonnegative. Enabled tests combine with OR;
    /// each model-based test retains its internal conjunction and observation stage.
    /// Repeated calls replace this setting. Existing solver defaults are retained.
    pub fn with_relative_model_reduction_tolerance(
        mut self,
        value: impl Into<Option<F>>,
    ) -> Self {
        self.inner = self.inner.with_relative_model_reduction_tolerance(value);
        self
    }
    /// Configure [`LevenbergMarquardt::with_tol_step_rel`] for the QR route.
    #[allow(deprecated)]
    #[deprecated(
        note = "use `with_relative_step_tolerance`; removal scheduled for Basin 2.0"
    )]
    pub fn with_tol_step_rel(mut self, value: F) -> Self {
        self.inner = self.inner.with_tol_step_rel(value);
        self
    }

    /// Configure the attempted step norm relative to the current iterate norm.
    ///
    /// `None` disables the test; zero requests an exact-zero threshold.
    /// Values must be finite and nonnegative. Enabled tests combine with OR;
    /// each model-based test retains its internal conjunction and observation stage.
    /// Repeated calls replace this setting. Existing solver defaults are retained.
    pub fn with_relative_step_tolerance(
        mut self,
        value: impl Into<Option<F>>,
    ) -> Self {
        self.inner = self.inner.with_relative_step_tolerance(value);
        self
    }
    /// Configure [`LevenbergMarquardt::with_no_progress_check`] for the QR route.
    pub fn with_no_progress_check(mut self, enabled: bool) -> Self {
        self.inner = self.inner.with_no_progress_check(enabled);
        self
    }
    /// Configure [`LevenbergMarquardt::with_damping`] for the QR route.
    pub fn with_damping(mut self, damping: LmDamping) -> Self {
        self.inner = self.inner.with_damping(damping);
        self
    }
    /// Configure [`LevenbergMarquardt::with_initial_step_bound`] for the QR route.
    pub fn with_initial_step_bound(mut self, factor: F) -> Self {
        self.inner = self.inner.with_initial_step_bound(factor);
        self
    }
    /// Configure [`LevenbergMarquardt::with_tau`] for the QR route.
    pub fn with_tau(mut self, value: F) -> Self {
        self.inner = self.inner.with_tau(value);
        self
    }
    /// Configure [`LevenbergMarquardt::with_max_inner_attempts`] for the QR route.
    pub fn with_max_inner_attempts(mut self, value: u32) -> Self {
        self.inner = self.inner.with_max_inner_attempts(value);
        self
    }
}
impl<P, V, M, F> Solver<P, NllsState<V, F>> for LevenbergMarquardtQr<V, M, F>
where
    F: Scalar,
    P: Residual<Param = V, Output = V> + Jacobian<Jacobian = M>,
    V: ScaledAdd<F>
        + NormSquared<F>
        + NormInfinity<F>
        + NegInPlace
        + Dot<F>
        + ScaleInPlace<F>
        + ComponentMulAssign
        + ComponentDivAssign
        + ComponentZip<F>
        + ComponentMaxAssign
        + FloorZerosInPlace<F>
        + Clone,
    M: FactorizePivotedQr<V, F> + MatTransposeVec<V>,
{
    type Error = <P as Residual>::Error;
    fn init(
        &mut self,
        problem: &mut Problem<P>,
        state: NllsState<V, F>,
    ) -> Result<NllsState<V, F>, Self::Error> {
        self.inner.init_model::<P, M, PivotedQr>(problem, state)
    }
    fn next_iter(
        &mut self,
        problem: &mut Problem<P>,
        state: NllsState<V, F>,
    ) -> Result<(NllsState<V, F>, Option<TerminationReason>), Self::Error> {
        self.inner.next_iter_model::<P, M, PivotedQr>(
            problem,
            state,
            self.rank_tolerance,
        )
    }
}

impl<V: Clone, M, F: Scalar> crate::core::inner::InitialState<V>
    for LevenbergMarquardtQr<V, M, F>
where
    M: FactorizePivotedQr<V, F>,
{
    type State = NllsState<V, F>;
    fn seed(&self, x: &V) -> Self::State {
        NllsState::new(x.clone())
    }
}
impl<V: Clone, M, F: Scalar> crate::core::inner::WarmStart<V>
    for LevenbergMarquardtQr<V, M, F>
where
    M: FactorizePivotedQr<V, F>,
{
}
impl<V: Clone, M, F: Scalar> super::cma_inject::MemeticInner<V, F>
    for LevenbergMarquardtQr<V, M, F>
where
    M: FactorizePivotedQr<V, F>,
{
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DenseMatrix, Executor};

    #[derive(Clone)]
    struct CholeskyOnly(DenseMatrix);
    impl GramMatrix for CholeskyOnly {
        fn gram(&self) -> Self {
            Self(self.0.gram())
        }
    }
    impl MatDiagonal<Vec<f64>> for CholeskyOnly {
        fn diagonal(&self) -> Vec<f64> {
            self.0.diagonal()
        }
    }
    impl MatTransposeVec<Vec<f64>> for CholeskyOnly {
        fn mat_transpose_vec(&self, v: &Vec<f64>) -> Vec<f64> {
            self.0.mat_transpose_vec(v)
        }
    }
    impl AddDiagonalVectorInPlace<Vec<f64>> for CholeskyOnly {
        fn add_diagonal_vector_in_place(&mut self, d: &Vec<f64>) {
            self.0.add_diagonal_vector_in_place(d);
        }
    }
    impl LinearSolveSpd<Vec<f64>> for CholeskyOnly {
        fn solve_spd(
            &self,
            b: &Vec<f64>,
        ) -> Result<Vec<f64>, crate::LinearSolveError> {
            self.0.solve_spd(b)
        }
    }
    struct Fit;
    impl Residual for Fit {
        type Param = Vec<f64>;
        type Output = Vec<f64>;
        type Error = std::convert::Infallible;
        fn residual(&self, x: &Vec<f64>) -> Result<Vec<f64>, Self::Error> {
            Ok(vec![x[0] - 1.])
        }
    }
    impl Jacobian for Fit {
        type Jacobian = CholeskyOnly;
        fn jacobian(&self, _: &Vec<f64>) -> Result<CholeskyOnly, Self::Error> {
            Ok(CholeskyOnly(DenseMatrix::from_row_slice(1, 1, &[1.])))
        }
    }
    #[test]
    fn legacy_annotations_and_cholesky_only_capabilities_still_work() {
        let solver: LevenbergMarquardt<Vec<f64>, CholeskyOnly> =
            LevenbergMarquardt::new();
        let result = Executor::from_start(Fit, solver, vec![0.])
            .max_iter(50)
            .run()
            .unwrap();
        assert_eq!(result.reason, TerminationReason::SolverConverged);
        assert!((result.param()[0] - 1.).abs() < 1e-8);
    }
}
