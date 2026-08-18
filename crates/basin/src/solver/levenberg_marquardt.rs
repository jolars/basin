use crate::core::math::{
    AddDiagonalVectorInPlace, ComponentDivAssign, ComponentMaxAssign,
    ComponentMulAssign, Dot, FloorZerosInPlace, GramMatrix, LinearSolveSpd,
    MatDiagonal, MatTransposeVec, NegInPlace, NormInfinity, NormSquared,
    Scalar, ScaleInPlace, ScaledAdd,
};
use crate::core::problem::{Jacobian, Problem, Residual};
use crate::core::solver::Solver;
use crate::core::state::NllsState;
use crate::core::termination::TerminationReason;

/// Levenberg-Marquardt solver for nonlinear least-squares problems
/// `min ½‖r(x)‖²`, with Marquardt diagonal scaling and the Nielsen
/// 1999 smooth μ-update.
///
/// Each iteration solves the damped normal equations
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
/// Initial damping is `μ₀ = τ`, dimensionless, because the
/// per-parameter magnitude now lives in `D` (the initial per-column
/// damping is `τ·diag(J(x₀)ᵀJ(x₀))`). τ is the *relative* trust
/// parameter; use a smaller value (e.g. `1e-6`) when `x₀` is believed
/// close to the optimum, larger (e.g. `1.0`) when far. Default
/// `τ = 10⁻³` matches Nielsen's "moderate trust" recommendation.
///
/// **Cholesky-on-(JᵀJ + μ·D) vs QR-on-stacked-system.** The damping
/// makes the SPD path strictly better-conditioned than pure
/// Gauss-Newton's `JᵀJ`: `μ·D` regularizes the rank deficiency that
/// makes GN fail. We stay on the SPD path because that's the only one
/// the [`linalg`](crate::core::math) tier exposes today, and the
/// regularization is sufficient for unconstrained LM.
/// QR-on-stacked-system (`[J; √(μD)]`) is more robust to ill-conditioned
/// `J` near rank deficiency but adds a second factorization route to
/// the linalg surface; deferred until S6 (TRF), where rank-deficient
/// Jacobians and box constraints make QR materially better.
///
/// # Failure modes
///
/// - **Cholesky failure under bumped μ.** When the initial damping is
///   too small to make `JᵀJ + μ·D` SPD (effectively never, for any
///   sensible `JᵀJ` and finite μ), the inner damping loop bumps μ via
///   `μ := μ·ν, ν := 2ν` and retries. Default
///   [`with_max_inner_attempts`](Self::with_max_inner_attempts) is 50, far more
///   than enough; in practice the first attempt succeeds. If the cap
///   is exhausted (μ overflowing to `inf`), the solver returns
///   [`TerminationReason::SolverFailed`]. Note that bumping μ cannot
///   rescue a coordinate whose `D` entry is zero (`μ·0 = 0`); the
///   `init` zero-column floor exists precisely to keep `D > 0`.
/// - **Divergence on highly nonlinear or poorly initialized problems.**
///   The damping itself prevents divergent steps (failed steps are
///   rejected via the gain-ratio test), so divergence manifests as
///   μ growing without bound. Catch this with
///   [`MaxIter`](crate::core::termination::MaxIter) on the executor.
///
/// # Termination
///
/// Beyond the framework criteria
/// ([`MaxIter`](crate::core::termination::MaxIter),
/// [`CostTolerance`](crate::core::termination::CostTolerance),
/// [`ParamTolerance`](crate::core::termination::ParamTolerance), …),
/// the solver emits [`TerminationReason::SolverConverged`] when any of
/// four MINPACK-style tests is satisfied: the same independent
/// `info`-code structure MINPACK uses, so converging on whichever fires
/// first:
///
/// - **`tol_grad`**: absolute first-order optimality (Madsen et al.
///   eq. 3.3a): `‖Jᵀr‖_∞ ≤ tol_grad`. Default `1e-8`; `0.0` disables.
/// - **`tol_grad_rel`**: relative first-order optimality, MINPACK
///   `gtol` (Moré 1978): the cosine of the angle between the residual
///   `r` and every column of `J`,
///   `max_j |gⱼ| / (‖J·,ⱼ‖ · ‖r‖) ≤ tol_grad_rel`. This measure is
///   dimensionless (invariant to scaling the residuals), so a single
///   tolerance is portable across problems whose residuals carry
///   different normalizations (where the absolute `‖Jᵀr‖_∞` is too
///   tight for some and too loose for others). Default `0.0`
///   (disabled); set e.g. `1e-8` for parity. The per-column norms
///   `‖J·,ⱼ‖ = √diag(JᵀJ)ⱼ` reuse the Marquardt scaling diagonal the
///   solver already forms.
/// - **`tol_cost_rel`**: relative cost reduction, MINPACK `ftol` (Moré 1978):
///   `|actred| ≤ tol·F  ∧  prered ≤ tol·F  ∧  ρ ≤ 2`, with the actual
///   and *predicted* per-iteration reductions in `F = ½‖r‖²`. The
///   `prered` clause is what the framework's
///   [`RelativeCostTolerance`](crate::core::termination::RelativeCostTolerance)
///   cannot express: it gates on the LM model, so the solver iterates
///   through temporary settling points (small actual gain, large
///   predicted gain) instead of stopping short. Default `0.0`
///   (disabled). See [`with_tol_cost_rel`](Self::with_tol_cost_rel).
/// - **`tol_step_rel`**: relative step, MINPACK `xtol` (Moré 1978):
///   `‖h‖ ≤ tol·‖x‖`. Default `0.0` (disabled). See
///   [`with_tol_step_rel`](Self::with_tol_step_rel).
///
/// The two gradient tests run before the step is computed (a step at a
/// stationary point is wasted); `tol_cost_rel`/`tol_step_rel` run after, since they need
/// the attempted step and its predicted and actual reduction.
///
/// LM runs on [`NllsState`], which does
/// **not** impl [`GradientState`](crate::core::state::GradientState): the
/// framework's L2-squared
/// [`GradientTolerance`](crate::core::termination::GradientTolerance) is the
/// wrong metric for NLLS (the canonical first-order test is the ∞-norm of
/// `Jᵀr`), so attaching it (or any other gradient criterion) is a **compile
/// error** rather than a criterion that silently never fires. Use the solver's
/// own [`with_tol_grad`](Self::with_tol_grad) /
/// [`with_tol_grad_rel`](Self::with_tol_grad_rel) for the first-order tests.
/// Same choice as [`GaussNewton`](super::GaussNewton) and
/// [`Trf`](super::Trf).
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
/// # #[cfg(feature = "nalgebra")] {
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
    tol_grad: F,
    tol_grad_rel: F,
    tol_cost_rel: F,
    tol_step_rel: F,
    tau: F,
    max_inner_attempts: u32,

    mu: Option<F>,
    nu: F,

    // Monotone Marquardt scaling diagonal D = max diag(JᵀJ). Zero
    // columns are floored to one so damping keeps the system nonsingular.
    diag: Option<V>,

    // Rejected steps leave these quantities valid. Accepted steps retain
    // the trial residual but invalidate the Gram matrix and gradient.
    r_cache: Option<V>,
    gram_cache: Option<M>,
    jtr_cache: Option<V>,
}

impl<V, M> Default for LevenbergMarquardt<V, M> {
    fn default() -> Self {
        Self::new()
    }
}

impl<V, M> LevenbergMarquardt<V, M> {
    /// Levenberg-Marquardt with Nielsen's defaults: `tol_grad = 1e-8`,
    /// `tol_grad_rel = 0.0` (disabled), `tol_cost_rel = 0.0` (disabled),
    /// `tol_step_rel = 0.0` (disabled), `tau = 1e-3`, `max_inner_attempts = 50`.
    pub fn new() -> Self {
        Self {
            tol_grad: 1e-8,
            tol_grad_rel: 0.0,
            tol_cost_rel: 0.0,
            tol_step_rel: 0.0,
            tau: 1e-3,
            max_inner_attempts: 50,
            mu: None,
            nu: 2.0,
            diag: None,
            r_cache: None,
            gram_cache: None,
            jtr_cache: None,
        }
    }
}

impl<V, M, F: Scalar> LevenbergMarquardt<V, M, F> {
    /// Absolute first-order optimality tolerance: emit
    /// [`TerminationReason::SolverConverged`] when `‖Jᵀr‖_∞ ≤ tol`
    /// (Madsen et al. eq. 3.3a). Set to `0.0` to disable the check and
    /// rely solely on [`with_tol_grad_rel`](Self::with_tol_grad_rel) and/or
    /// framework termination criteria. Default `1e-8`.
    pub fn with_tol_grad(mut self, tol: F) -> Self {
        assert!(tol >= F::zero(), "tol_grad must be ≥ 0");
        self.tol_grad = tol;
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
    pub fn with_tol_grad_rel(mut self, tol: F) -> Self {
        assert!(tol >= F::zero(), "tol_grad_rel must be ≥ 0");
        self.tol_grad_rel = tol;
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
    /// the LM model. At a *temporary settling point* a single step's
    /// actual gain can be small while the model still predicts substantial
    /// progress; gating on `prered` keeps LM iterating through such points
    /// to the true minimum, where a plain achieved-reduction test would
    /// stop short. This is exactly MINPACK's behavior and the reason
    /// `ftol` belongs on the solver rather than in the termination layer.
    ///
    /// Set to `0.0` to disable. Default `0.0` (disabled); use e.g. `1e-8`
    /// for MINPACK `ftol` parity. Converges when *any* enabled test fires
    /// (see [`with_tol_grad`](Self::with_tol_grad)).
    ///
    /// [`RelativeCostTolerance`]: crate::core::termination::RelativeCostTolerance
    pub fn with_tol_cost_rel(mut self, tol: F) -> Self {
        assert!(tol >= F::zero(), "tol_cost_rel must be ≥ 0");
        self.tol_cost_rel = tol;
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
    pub fn with_tol_step_rel(mut self, tol: F) -> Self {
        assert!(tol >= F::zero(), "tol_step_rel must be ≥ 0");
        self.tol_step_rel = tol;
        self
    }

    /// Relative initial damping `τ`: `μ₀ = τ`, giving an initial
    /// per-column damping of `τ·diag(J(x₀)ᵀJ(x₀))` under Marquardt
    /// scaling. Use a smaller value (e.g. `1e-6`) when `x₀` is believed
    /// close to the optimum; a larger value (e.g. `1.0`) when far from
    /// it. Default `1e-3` (Nielsen's "moderate trust").
    pub fn with_tau(mut self, tau: F) -> Self {
        assert!(tau > F::zero(), "tau must be > 0");
        self.tau = tau;
        self
    }

    /// Maximum number of damping bumps inside a single outer iteration
    /// before giving up with [`TerminationReason::SolverFailed`]. Each
    /// bump multiplies μ by ν (initially 2) and doubles ν. With the
    /// default 50, μ grows by a factor of `2^50 ≈ 10¹⁵` before bailing,
    /// effectively unreachable in practice. Default `50`.
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
        mut state: NllsState<V, F>,
    ) -> Result<NllsState<V, F>, Self::Error> {
        // Seed both the state and the cross-iteration caches from one
        // residual/Jacobian evaluation.
        let (r, j) = problem.residual_and_jacobian(&state.param)?;
        state.cost = Some(F::from_f64(0.5).unwrap() * r.norm_squared());

        let a = j.gram();
        let mut d = a.diagonal();
        d.floor_zeros_in_place(F::one());
        self.diag = Some(d);

        self.mu = Some(self.tau);
        self.nu = F::from_f64(2.0).unwrap();
        self.jtr_cache = Some(j.mat_transpose_vec(&r));
        self.gram_cache = Some(a);
        self.r_cache = Some(r);
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

        let (a, g) = match (self.gram_cache.take(), self.jtr_cache.take()) {
            (Some(a), Some(g)) => (a, g),
            _ => {
                let j = problem.jacobian(&state.param)?;
                (j.gram(), j.mat_transpose_vec(&r))
            }
        };
        let diag_cur = a.diagonal();

        // MINPACK's absolute and relative first-order tests:
        //   * absolute   ‖Jᵀr‖_∞ ≤ tol_grad           (Madsen et al. 3.3a)
        //   * relative   max_j |gⱼ|/(‖J·,ⱼ‖·‖r‖) ≤ tol_grad_rel  (MINPACK gtol)
        // The relative measure is the cosine between r and each Jacobian
        // column. Squaring avoids a square root:
        // `max_j gⱼ²/diag(JᵀJ)ⱼ ≤ tol_grad_rel²·‖r‖²`. A zero column has
        // `diag(JᵀJ)ⱼ = 0` and `gⱼ = 0`; flooring the denominator to 1
        // makes that term `0/1 = 0` rather than `0/0 = NaN`, which is
        // MINPACK's "skip zero columns" behavior.
        let abs_converged =
            self.tol_grad > F::zero() && g.norm_infinity() <= self.tol_grad;
        let rel_converged = self.tol_grad_rel > F::zero() && {
            let mut cos_sq = g.clone();
            cos_sq.component_mul_assign(&g);
            let mut denom = diag_cur.clone();
            denom.floor_zeros_in_place(F::one());
            cos_sq.component_div_assign(&denom);
            cos_sq.norm_infinity()
                <= self.tol_grad_rel * self.tol_grad_rel * r.norm_squared()
        };
        if abs_converged || rel_converged {
            // Termination does not move the iterate, so the caches remain valid.
            self.r_cache = Some(r);
            self.gram_cache = Some(a);
            self.jtr_cache = Some(g);
            return Ok((state, Some(TerminationReason::SolverConverged)));
        }

        let mut neg_g = g.clone();
        neg_g.neg_in_place();

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

        // Increase damping if roundoff defeats the Cholesky factorization.
        let two = F::from_f64(2.0).unwrap();
        let half = F::from_f64(0.5).unwrap();
        let one_third = F::from_f64(1.0 / 3.0).unwrap();
        let h;
        let mut attempts: u32 = 0;
        loop {
            let mut a_damped = a.clone();
            let mut damping = d.clone();
            damping.scale_in_place(mu);
            a_damped.add_diagonal_vector_in_place(&damping);
            match a_damped.solve_spd(&neg_g) {
                Ok(step) => {
                    h = step;
                    break;
                }
                Err(_) => {
                    attempts += 1;
                    if attempts >= self.max_inner_attempts || !mu.is_finite() {
                        self.mu = Some(mu);
                        self.nu = nu;
                        self.diag = Some(d);
                        self.r_cache = Some(r);
                        self.gram_cache = Some(a);
                        self.jtr_cache = Some(g);
                        return Ok((
                            state,
                            Some(TerminationReason::SolverFailed),
                        ));
                    }
                    mu = mu * nu;
                    nu = nu * two;
                }
            }
        }

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

        if rho > F::zero() {
            // Nielsen eq. 2.5 with β=2, γ=3, p=3.
            state.param = x_trial;
            state.cost = Some(f_trial);
            let factor = F::one() - (two * rho - F::one()).powi(3);
            mu = mu * factor.max(one_third);
            nu = two;
            self.r_cache = Some(r_trial);
            self.gram_cache = None;
            self.jtr_cache = None;
        } else {
            // Preserve iterate-dependent caches and increase damping.
            mu = mu * nu;
            nu = nu * two;
            self.r_cache = Some(r);
            self.gram_cache = Some(a);
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
        //     relative to the iterate. Squared on both sides to avoid a sqrt.
        let cost_rel_converged = self.tol_cost_rel > F::zero()
            && actual_diff.abs() <= self.tol_cost_rel * prev_cost
            && l_diff <= self.tol_cost_rel * prev_cost
            && rho <= two;
        let step_rel_converged = self.tol_step_rel > F::zero()
            && h.norm_squared()
                <= self.tol_step_rel
                    * self.tol_step_rel
                    * state.param.norm_squared();
        if cost_rel_converged || step_rel_converged {
            return Ok((state, Some(TerminationReason::SolverConverged)));
        }

        Ok((state, None))
    }
}
