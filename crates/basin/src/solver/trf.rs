use crate::core::constraint::BoxConstraints;
use crate::core::inner::InitialState;
use crate::core::math::{
    AddDiagonalVectorInPlace, BoxAffineScaling, Dot, GramMatrix,
    LinearSolveSpd, MatTransposeVec, MaxDiagonal, NegInPlace, NormSquared,
    Scalar, ScaledAdd,
};
use crate::core::problem::{Jacobian, Problem, Residual};
use crate::core::solver::Solver;
use crate::core::state::NllsState;
use crate::core::termination::TerminationReason;

/// Levenberg-Marquardt with box bounds (TRF, trust-region-reflective)
/// for nonlinear least-squares problems `min ½‖r(x)‖²` subject to
/// `lower ≤ x ≤ upper`. The first n-D box-constrained NLLS solver in
/// basin and the natural extension of [`LevenbergMarquardt`](super::LevenbergMarquardt)
/// to bounded problems.
///
/// # Algorithm
///
/// At each iteration the solver computes the Coleman-Li affine scaling
/// from Branch-Coleman-Li 1999: a diagonal trust-region matrix
/// `D = diag(|v|^{-1/2})` and a diagonal curvature correction
/// `C = D · diag(g) · J^v · D` (also diagonal, non-negative), and
/// solves the damped, scaled normal equations
///
/// ```text
/// (JᵀJ + diag(c) + μ · diag(d²)) h = −g
/// ```
///
/// via Cholesky on the SPD Gram. The unconstrained step `h` is then
/// scaled back into the open feasible region by the largest
/// `α ∈ (0, 1]` such that `x + α·h` stays inside `(lower, upper)`,
/// multiplied by a strict-interior factor `θ ≈ 0.99995`. The damping
/// `μ` adapts via the Nielsen smooth cubic gain-ratio update (the same
/// machinery as [`LevenbergMarquardt`](super::LevenbergMarquardt)), and
/// initial `μ₀ = τ · max diag(JᵀJ + diag(c))`.
///
/// The four-case dispatch defining `v(x)` and the elementwise diagonals
/// `d²[i] = 1/|v_i|` and `c[i] = |g_i|/|v_i|` (or 0 for infinite bounds)
/// follow Branch-Coleman-Li 1999 eqs (i)–(iv); see
/// `references/branch-coleman-li-1999/source.marker.md:43-72` and
/// `references/branch-coleman-li-1999/NOTES.md`.
///
/// # Reduction to LM
///
/// When `lower = -∞` and `upper = +∞` element-wise, the BCL scaling
/// reduces to `D = I`, `C = 0`, the step-back is a no-op, and the
/// algorithm becomes exactly Levenberg-Marquardt with Nielsen's μ-update:
/// same iterates, same convergence. `Trf` strictly subsumes
/// [`LevenbergMarquardt`](super::LevenbergMarquardt) at the trait-bound
/// level (the reverse is a compile error: LM bounds on
/// `Residual + Jacobian` only, not [`BoxConstraints`]).
///
/// # What basin's S6 ships, and what it doesn't
///
/// Basin's `Trf` is a deliberate simplification of the full STIR
/// algorithm in BCL §4. It ships:
///
/// - The Coleman-Li affine scaling matrix `D` and curvature correction
///   `C` (BCL eqs 2.1–2.6).
/// - LM-style μ-adaptation via Nielsen smooth cubic, in lieu of the
///   explicit trust-region radius `Δ` of BCL FIG.6.
/// - Strict-interior step-back to keep iterates in the *open* box
///   (`D` is undefined on a finite face).
/// - First-order optimality termination via `‖v ⊙ Jᵀr‖_∞ ≤ tol_grad`.
///
/// What it does **not** ship:
///
/// - **STIR 2D subspace** (BCL FIG.5). The full-space subproblem is
///   solved each iteration. Adequate for small and medium dense and
///   sparse problems; large-scale Krylov inner solves wait for a
///   future session.
/// - **Reflection technique** (BCL §2 / FIG.2). The unconstrained step
///   is straight-line stepped back to the box boundary, never
///   reflected off it. Reflection saves ~2-3× iterations on problems
///   where many components bind (BCL Table 1); defer until a test
///   case demands it.
/// - **Explicit trust-region radius `Δ`** with Moré-Sorensen-style
///   λ-adaptation (BCL FIG.6). The LM-style μ-update is simpler and
///   reuses [`LevenbergMarquardt`](super::LevenbergMarquardt)'s
///   machinery.
/// - **Negative-curvature termination clause** (BCL §6). The
///   `‖D·g‖_∞ ≤ τ` clause alone is used; the curvature test would need
///   an eigendecomposition or Lanczos pass that isn't worth the
///   surface before STIR lands.
///
/// # Failure modes
///
/// - **Cholesky failure under bumped μ.** The damped, scaled Gram
///   `JᵀJ + diag(c) + μ·diag(d²)` is SPD by construction for `μ > 0`,
///   so Cholesky should succeed on the first attempt. The retry loop
///   bumps μ via `μ ← μ·ν, ν ← 2ν` if it doesn't, capped at
///   [`with_max_inner_attempts`](Self::with_max_inner_attempts) (default 50).
///   Cap exhaustion or μ overflowing to `inf` returns
///   [`TerminationReason::SolverFailed`].
/// - **Boundary starting point.** `D` is undefined where `v_i = 0`
///   (i.e. on a finite face). [`init`](Solver::init) projects the
///   starting iterate strictly into `(lower, upper)` via
///   `BoxAffineScaling::project_strictly_inside`, so feasible-but-
///   on-boundary starts are silently corrected.
///
/// # Termination
///
/// Beyond the framework criteria
/// ([`MaxIter`](crate::core::termination::MaxIter),
/// [`CostTolerance`](crate::core::termination::CostTolerance),
/// [`ParamTolerance`](crate::core::termination::ParamTolerance), …),
/// the solver emits [`TerminationReason::SolverConverged`] when
/// `‖v ⊙ Jᵀr‖_∞ ≤ tol_grad` (equivalently `max_i |g_i| · |v_i|`,
/// where `v_i` is BCL's signed distance-to-bound). The metric goes to
/// zero at any KKT point (interior *or* face-active), so it works
/// uniformly across the corner/edge/interior cases. Collapses to
/// LM's `‖Jᵀr‖_∞` when no constraint is active. Default
/// `tol_grad = 1e-8`; set to `0.0` to disable the check.
///
/// TRF runs on [`NllsState`], which does
/// **not** impl [`GradientState`](crate::core::state::GradientState), so the
/// framework gradient criteria are a **compile error** rather than a silent
/// no-op; use [`with_tol_grad`](Self::with_tol_grad) above. This is the same
/// choice as [`LevenbergMarquardt`](super::LevenbergMarquardt): the L2-squared
/// [`GradientTolerance`](crate::core::termination::GradientTolerance) is the
/// wrong metric for NLLS, and
/// [`ProjectedGradientTolerance`](crate::core::termination::ProjectedGradientTolerance)
/// uses the unscaled projected-gradient measure rather than the scaled one
/// TRF's KKT test uses.
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
///
/// The sparse damping path requires the diagonal of `JᵀJ` to be in the
/// CSC pattern (always true when `J` has no zero columns); see
/// `AddDiagonalVectorInPlace`.
///
/// # State convention
///
/// Same as [`LevenbergMarquardt`](super::LevenbergMarquardt):
/// `state.cost` carries the LM convention `½‖r‖²`. The bound on `P`
/// includes [`BoxConstraints`] (which inherits
/// [`CostFunction`](crate::core::problem::CostFunction)) but the solver
/// never calls `cost()`; it computes `½‖r‖²` from the residual it
/// evaluates itself. Problems whose user-facing `cost()` uses an
/// unscaled `Σ rᵢ²` form (e.g.
/// [`BoothBoxedResiduals`](crate::problems::BoothBoxedResiduals)) will
/// see `state.cost()` differ from `problem.cost(state.param())` by a
/// factor of two; both go to zero at the optimum.
///
/// # Examples
///
/// See [`LevenbergMarquardt`](crate::LevenbergMarquardt) for the
/// `Residual` + `Jacobian` least-squares pattern; `Trf` additionally
/// requires the problem to implement `BoxConstraints` and is constructed
/// with `Trf::new()`.
pub struct Trf<V, M, F = f64> {
    tol_grad: F,
    tau: F,
    rstep: F,
    theta: F,
    max_inner_attempts: u32,

    // Runtime state, populated by `init` and mutated by `next_iter`
    // through `&mut self`.
    mu: Option<F>,
    nu: F,

    // Residual and Jacobian caches across iterations, same shape as
    // [`LevenbergMarquardt`](super::LevenbergMarquardt). On accept the
    // trial residual is at the new iterate (so it's stashed) but the
    // Jacobian there is unknown (so it's cleared); on reject both are
    // unchanged at the current iterate (both stashed).
    r_cache: Option<V>,
    j_cache: Option<M>,
}

impl<V, M> Default for Trf<V, M> {
    fn default() -> Self {
        Self::new()
    }
}

impl<V, M> Trf<V, M> {
    /// `Trf` with the canonical defaults: `tol_grad = 1e-8`,
    /// `tau = 1e-3`, `rstep = 1e-10`, `theta = 0.99995`,
    /// `max_inner_attempts = 50`.
    pub fn new() -> Self {
        Self {
            tol_grad: 1e-8,
            tau: 1e-3,
            rstep: 1e-10,
            theta: 0.99995,
            max_inner_attempts: 50,
            mu: None,
            nu: 2.0,
            r_cache: None,
            j_cache: None,
        }
    }
}

impl<V, M, F: Scalar> Trf<V, M, F> {
    /// First-order optimality tolerance: emit
    /// [`TerminationReason::SolverConverged`] when
    /// `‖D · Jᵀr‖_∞ ≤ tol`. Set to `0.0` to disable. Default `1e-8`.
    pub fn with_tol_grad(mut self, tol: F) -> Self {
        assert!(tol >= F::zero(), "tol_grad must be ≥ 0");
        self.tol_grad = tol;
        self
    }

    /// Initial damping scale `τ` in `μ₀ = τ · max diag(JᵀJ + diag(c))`.
    /// Smaller (e.g. `1e-6`) when `x₀` is believed close to the
    /// optimum; larger (e.g. `1.0`) when far from it. Default `1e-3`.
    pub fn with_tau(mut self, tau: F) -> Self {
        assert!(tau > F::zero(), "tau must be > 0");
        self.tau = tau;
        self
    }

    /// Strict-interior projection scale at `init`. Components within
    /// `rstep · max(1, |bound|)` of a finite bound are nudged inward.
    /// Default `1e-10` matches SciPy's `make_strictly_feasible`.
    pub fn with_rstep(mut self, rstep: F) -> Self {
        assert!(rstep > F::zero(), "rstep must be > 0");
        self.rstep = rstep;
        self
    }

    /// Strict-interior step-back factor: when the unconstrained step
    /// would land on or beyond a face, the actual step is scaled by
    /// `theta · τ_max` instead of `τ_max` to keep the iterate strictly
    /// inside. Must be in `(0, 1)`. Default `0.99995`.
    pub fn with_theta(mut self, theta: F) -> Self {
        assert!(
            theta > F::zero() && theta < F::one(),
            "theta must be in (0, 1), got {:?}",
            theta
        );
        self.theta = theta;
        self
    }

    /// Maximum number of damping bumps inside a single outer iteration
    /// before giving up with [`TerminationReason::SolverFailed`]. Each
    /// bump multiplies μ by ν (initially 2) and doubles ν. Default
    /// `50` is effectively unreachable in practice (μ grows by `2^50 ≈
    /// 10¹⁵` before bailing). Default `50`.
    pub fn with_max_inner_attempts(mut self, n: u32) -> Self {
        assert!(n > 0, "max_inner_attempts must be > 0");
        self.max_inner_attempts = n;
        self
    }
}

impl<V, M, F> InitialState<V> for Trf<V, M, F>
where
    F: Scalar,
    V: Clone,
{
    type State = NllsState<V, F>;
    fn seed(&self, x: &V) -> Self::State {
        NllsState::new(x.clone())
    }
}

impl<P, V, M, F> Solver<P, NllsState<V, F>> for Trf<V, M, F>
where
    F: Scalar,
    P: Residual<Param = V, Output = V>
        + Jacobian<Jacobian = M>
        + BoxConstraints<Param = V>,
    V: ScaledAdd<F>
        + NormSquared<F>
        + NegInPlace
        + Dot<F>
        + BoxAffineScaling<F>
        + Clone,
    M: GramMatrix
        + MatTransposeVec<V>
        + LinearSolveSpd<V>
        + AddDiagonalVectorInPlace<V>
        + MaxDiagonal<F>
        + Clone,
{
    type Error = <P as Residual>::Error;

    fn init(
        &mut self,
        problem: &mut Problem<P>,
        mut state: NllsState<V, F>,
    ) -> Result<NllsState<V, F>, Self::Error> {
        // Project the starting iterate strictly into (lower, upper).
        // D is undefined where v_i = 0 (a finite face), so an
        // on-boundary or infeasible start is silently corrected.
        state.param.project_strictly_inside(
            problem.inner().lower(),
            problem.inner().upper(),
            self.rstep,
        );

        let (r, j) = problem.residual_and_jacobian(&state.param)?;
        state.cost = Some(F::from_f64(0.5).unwrap() * r.norm_squared());

        // μ₀ = τ · max diag(JᵀJ + diag(c)). The C-correction is
        // typically small; the τ · max diag scaling matches Nielsen's
        // recommendation for LM, generalized to the BCL M-matrix.
        let g = j.mat_transpose_vec(&r);
        let mut d_sq = state.param.clone();
        let mut c_diag = state.param.clone();
        state.param.compute_cl_scaling(
            &g,
            problem.inner().lower(),
            problem.inner().upper(),
            &mut d_sq,
            &mut c_diag,
        );

        let mut a = j.gram();
        a.add_diagonal_vector_in_place(&c_diag);
        let max_diag = a.max_diagonal().max(F::one());
        self.mu = Some(self.tau * max_diag);
        self.nu = F::from_f64(2.0).unwrap();
        self.r_cache = Some(r);
        self.j_cache = Some(j);
        Ok(state)
    }

    fn next_iter(
        &mut self,
        problem: &mut Problem<P>,
        mut state: NllsState<V, F>,
    ) -> Result<(NllsState<V, F>, Option<TerminationReason>), Self::Error> {
        // Use cached `r`/`J` when available (set by init or by the
        // previous accept-or-reject branch). Only count an eval when the
        // cache misses.
        let r = match self.r_cache.take() {
            Some(r) => r,
            None => problem.residual(&state.param)?,
        };
        let j = match self.j_cache.take() {
            Some(j) => j,
            None => problem.jacobian(&state.param)?,
        };

        let g = j.mat_transpose_vec(&r);

        // Compute the Coleman-Li affine scaling diagonals at the
        // current iterate. d_sq[i] = 1/|v_i|, c_diag[i] = |g_i|/|v_i|
        // (or 0 for infinite bounds).
        let mut d_sq = state.param.clone();
        let mut c_diag = state.param.clone();
        state.param.compute_cl_scaling(
            &g,
            problem.inner().lower(),
            problem.inner().upper(),
            &mut d_sq,
            &mut c_diag,
        );

        // First-order optimality: ‖v ⊙ Jᵀr‖_∞ ≤ tol_grad, equal to
        // `max_i |g_i| / d_sq_i` in our representation (since
        // `d_sq[i] = 1/|v_i|`). Goes to zero at any KKT point, interior
        // *or* face-active. Collapses to LM's `‖Jᵀr‖_∞` when bounds are
        // infinite (then `|v_i| = 1`, `d_sq = 1`, division is identity).
        if self.tol_grad > F::zero()
            && g.cl_kkt_inf_norm(&d_sq) <= self.tol_grad
        {
            // Restore caches; init resets them on each reuse, but
            // mirroring LM's pattern keeps the contract uniform.
            self.r_cache = Some(r);
            self.j_cache = Some(j);
            return Ok((state, Some(TerminationReason::SolverConverged)));
        }

        let mut neg_g = g.clone();
        neg_g.neg_in_place();

        let m = j.gram();

        let mut mu = self
            .mu
            .expect("mu not set: Solver::init must run before next_iter");
        let mut nu = self.nu;

        // Inner damping loop: solve (J^TJ + diag(c) + μ·diag(d_sq)) h = −g.
        // The damped, scaled Gram is SPD by construction for μ > 0; the
        // retry path matters only for pathological cases where roundoff
        // breaks SPD-ness at the chosen μ.
        let two = F::from_f64(2.0).unwrap();
        let half = F::from_f64(0.5).unwrap();
        let one_third = F::from_f64(1.0 / 3.0).unwrap();
        let h;
        let mut attempts: u32 = 0;
        loop {
            let mut a_damped = m.clone();
            // damping_vec = c + μ · d_sq.
            let mut damping_vec = c_diag.clone();
            damping_vec.scaled_add(mu, &d_sq);
            a_damped.add_diagonal_vector_in_place(&damping_vec);
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
                        // State unchanged; restore both caches.
                        self.r_cache = Some(r);
                        self.j_cache = Some(j);
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

        // Step-back to the open feasible region. The unconstrained
        // Newton step h might land on or beyond a face; scale it down
        // by min(1, θ · τ_max) so the iterate stays strictly inside.
        let tau_max = state.param.max_feasible_step(
            &h,
            problem.inner().lower(),
            problem.inner().upper(),
        );
        let alpha = if tau_max >= F::one() {
            F::one()
        } else {
            self.theta * tau_max
        };

        // Trial step.
        let mut x_trial = state.param.clone();
        x_trial.scaled_add(alpha, &h);
        let r_trial = problem.residual(&x_trial)?;
        let f_trial = half * r_trial.norm_squared();

        let prev_cost = state
            .cost
            .expect("cost not set: Solver::init must run before next_iter");

        // BCL gain ratio with the C-correction (eq. ψ_k from
        // `source.marker.md:138`). Numerator is "actual reduction in
        // M-model": Δf − ½ s^T C s. Denominator is the predicted
        // reduction in BCL's *undamped* M-model evaluated at s = α·h,
        // derived from the Lagrangian (M + μD²) h = −g:
        //
        //   −ψ_k(α·h) = −α(1 − ½α) h^T g + ½ α² μ ‖D·h‖²
        //
        // For α = 1 this reduces to ½(μ ‖D·h‖² − h^T g), which mirrors
        // Nielsen's LM formula with D folded in.
        let h_t_g = h.dot(&g);
        let dh_norm_sq = h.weighted_norm_squared(&d_sq);
        let predicted = -alpha * (F::one() - half * alpha) * h_t_g
            + half * alpha * alpha * mu * dh_norm_sq;
        let half_s_t_c_s =
            half * alpha * alpha * h.weighted_norm_squared(&c_diag);
        let actual = prev_cost - f_trial - half_s_t_c_s;

        let rho = if predicted > F::zero() {
            actual / predicted
        } else {
            F::zero()
        };

        if rho > F::zero() {
            // Accept. Update x and cost; adapt μ via Nielsen smooth
            // cubic with β=2, γ=3, p=3 (matches LevenbergMarquardt).
            // Stash the trial residual (now at the new iterate); clear
            // the Jacobian cache since J(x_trial) was not computed.
            state.param = x_trial;
            state.cost = Some(f_trial);
            let factor = F::one() - (two * rho - F::one()).powi(3);
            mu = mu * factor.max(one_third);
            nu = two;
            self.r_cache = Some(r_trial);
            self.j_cache = None;
        } else {
            // Reject. Bump μ geometrically; double ν so consecutive
            // rejections escalate damping faster. Both r and J remain
            // valid at the unchanged iterate.
            mu = mu * nu;
            nu = nu * two;
            self.r_cache = Some(r);
            self.j_cache = Some(j);
        }

        self.mu = Some(mu);
        self.nu = nu;
        Ok((state, None))
    }
}
