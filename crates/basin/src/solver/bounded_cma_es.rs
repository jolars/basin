use std::collections::VecDeque;

use crate::core::constraint::BoxConstraints;
use crate::core::math::{
    ClampInPlace, ComponentMulAssign, MatDiagonal, MatTransposeVec, MatVec, MatrixFromDiagonal,
    MatrixIdentity, NormSquared, RankOneUpdate, SampleStandardNormal, ScaleInPlace, ScaledAdd,
    SymmetricEigen, VectorLen,
};
use crate::core::problem::{CostFunction, Problem};
use crate::core::rng::{ChaCha8Rng, SeedableRng};
use crate::core::solver::Solver;
use crate::core::state::BasicPopulationState;
use crate::core::termination::TerminationReason;

use super::cma_es::{compute_weights, expected_norm_n01, sort_population_ascending};

/// Box-constrained `(µ/µ_W, λ)`-CMA-ES with adaptive quadratic boundary
/// penalty (Hansen `BoundPenalty`, the default in `pycma`).
///
/// This is the constrained sibling of [`CmaEs`](super::cma_es::CmaEs).
/// The CMA-ES core (sampling, recombination, σ adaptation, covariance
/// update, eigendecomposition, TolX termination) is identical;
/// references to "the CMA core" below point at
/// [`CmaEs`'s docs and source](super::cma_es::CmaEs) for the algorithm
/// summary and Hansen 2016 fixture.
///
/// # Bound handling — adaptive quadratic penalty
///
/// Per generation, for each sample `x_k`:
///
/// ```text
/// x_k_rep = clamp(x_k, lower, upper)             # repaired sample
/// f_raw   = problem.cost(&x_k_rep)               # f at repaired point
/// pen     = (1/n) · Σ_i γ_i (x_k[i] − x_k_rep[i])²   # quadratic penalty
/// f_pen   = f_raw + pen
/// ```
///
/// The **un-repaired** `x_k` enters recombination (so the covariance
/// learns "don't go that way"); the **penalized** `f_pen` is what the
/// population is sorted by. `γ ∈ R^n` is initialized to `1` and adapted
/// each generation from the IQR of recent fitness values and the per-
/// coordinate variances `σ² · diag(C)` — see
/// `references/pycma-bound-handling/NOTES.md` for the full rule.
///
/// ## Why this strategy and not the others
///
/// Four bound-handling families circulate in the CMA-ES literature:
///
/// - **Resampling** (reject-and-redraw infeasible samples). Cheap but
///   the rejection rate explodes when the optimum sits on a face of the
///   feasible box, and the implicit sampling distribution is distorted
///   by truncation. Bad default.
/// - **Reflection / clipping**. Cheap, unprincipled. Clipping puts a
///   delta on the distribution that fights covariance adaptation;
///   reflection aliases multimodally near corners.
/// - **Adaptive quadratic penalty** (this solver, Hansen / pycma).
///   Self-tuning, no extra knobs leaked to the user, battle-tested
///   across the BBOB benchmark suite.
/// - **Smooth-boundary transformation** (pycma's `BoundTransform`).
///   Maps `R^n → [l, u]^n` smoothly. Distorts the optimization
///   landscape near active bounds, slowing convergence on coordinates
///   whose optimum is exactly on a face.
///
/// Adaptive penalty is the only one with a serious reference
/// implementation (pycma) and a self-adapting coefficient. **BIPOP**
/// is sometimes lumped with these but is a population-restart scheme,
/// orthogonal to bound handling — it's reserved for restart machinery
/// (S11).
///
/// # Contract
///
/// - **Caller must:** implement
///   [`CostFunction<Param = V, Output = f64>`] **and**
///   [`BoxConstraints`] on the same problem type. The bounds live on
///   the problem (tenet 4 in `AGENTS.md`); handing this solver a
///   problem without `BoxConstraints` is a compile-time error.
/// - **Caller must:** ensure `lower[i] ≤ upper[i]` for every component
///   ([`f64::clamp`] panics otherwise) and `initial_sigma > 0`.
/// - **Caller must:** hand in a
///   [`BasicPopulationState::with_size(λ)`](crate::BasicPopulationState::with_size)
///   matching the solver's λ. The default
///   `λ = 4 + ⌊3 ln n⌋` is exposed via
///   [`default_lambda`](Self::default_lambda).
/// - **Implementor (this solver) must:** maintain the
///   [`PopulationState`](crate::core::state::PopulationState)
///   sorted-by-cost invariant on `state.candidates` / `state.costs`,
///   where `state.costs` carries the **penalized** fitness values
///   (raw fitness is held in solver-internal state for the γ-update
///   IQR). The initial mean is projected onto `[lower, upper]` once at
///   iter 0 so the iter-0 search distribution is centered in
///   feasibility.
///
/// # Termination
///
/// Same TolX criterion as the unconstrained
/// [`CmaEs`](super::cma_es::CmaEs): `σ · max_i d_i < tol_x`. Bounded-
/// CMA-ES adds no new termination criteria of its own; feasibility is
/// enforced sample-wise by construction (every evaluated point is
/// inside the box because we evaluate at `clamp(x_k, lower, upper)`),
/// and the framework's
/// [`MaxIter`](crate::core::termination::MaxIter) /
/// [`MaxCostEvals`](crate::core::termination::MaxCostEvals) work
/// against [`BasicPopulationState`]
/// without modification.
///
/// # Reproducibility
///
/// Same as [`CmaEs`](super::cma_es::CmaEs): a [`ChaCha8Rng`] seeded
/// from `seed: u64` makes the iterate trajectory deterministic on
/// every platform basin builds for, including `wasm32-unknown-unknown`.
///
/// # Backends
///
/// LA-heavy: requires symmetric eigendecomposition, scalar-and-rank-1
/// matrix updates, matrix-vector / transposed matrix-vector products,
/// **plus** [`MatDiagonal<V>`] (extracts `diag(C)` for the σ²·diag(C)
/// per-axis variances the γ-update reads). Wired and tested for the
/// default `Vec<f64>` / [`DenseMatrix`](crate::DenseMatrix) backend
/// (pure-Rust cyclic Jacobi eigensolver — no feature flag, `wasm`-clean),
/// `nalgebra::DVector<f64>` / `nalgebra::DMatrix<f64>` (feature
/// `nalgebra`), `ndarray::Array1<f64>` / `ndarray::Array2<f64>` (feature
/// `ndarray`, also wired to the cyclic Jacobi solver — `wasm`-clean),
/// and `faer::Col<f64>` / `faer::Mat<f64>` (feature `faer`) — same
/// coverage as [`CmaEs`](super::cma_es::CmaEs).
///
/// # Examples
///
/// See [`CmaEs`](crate::CmaEs) (and [`RandomSearch`](crate::RandomSearch)
/// for the population `Executor` pattern); `BoundedCmaEs` additionally
/// requires `BoxConstraints` on the problem.
pub struct BoundedCmaEs<V, M> {
    initial_mean: V,
    initial_sigma: f64,
    lambda_override: Option<usize>,
    seed: u64,
    tol_x_override: Option<f64>,
    /// Per-coordinate initial standard deviations (pycma's `CMA_stds`).
    /// `None` keeps the isotropic `C = I` default; `Some(stds)` seeds the
    /// initial covariance to `diag(stds²)`. Set via [`with_stds`](Self::with_stds).
    stds_override: Option<V>,

    state: Option<Working<V, M>>,
}

/// Solver-internal mutable state, populated in [`Solver::init`] and
/// updated each [`Solver::next_iter`]. Mirrors `cma_es::Working` and
/// adds the BoundPenalty fields (`gamma`, `hist`, `raw_costs`, …).
///
/// `pub(crate)` so sibling solvers in `crate::solver` can read the
/// post-update `m`, `σ`, `B`, `D^{-1}` they need for injection-style
/// composition. `BoundedCmaInject` uses these to clip injected `y_i`
/// in Mahalanobis distance per Hansen 2011 eq. 4, mirroring how
/// `CmaInject` reads from `CmaEs::Working`. Not a stable public surface.
pub(crate) struct Working<V, M> {
    // --- CMA-ES constants (computed once at init) ---
    pub(crate) n: usize,
    lambda: usize,
    mu: usize,
    weights: Vec<f64>,
    mu_eff: f64,
    sum_w: f64,
    c_sigma: f64,
    d_sigma: f64,
    c_c: f64,
    c_1: f64,
    c_mu: f64,
    expected_norm: f64,
    h_sigma_threshold: f64,
    tol_x: f64,

    // --- BoundPenalty constants (computed once at init) ---
    /// `min(1, mu_eff / (10·n))`. Damping factor on the γ multiplicative
    /// update; pycma `boundary_handler.py:716`.
    damp: f64,
    /// `3 · max(1, sqrt(n) / mu_eff)`. The σ-unit slack threshold
    /// before γ_i is raised on coordinate `i`; pycma `boundary_handler.py:730`.
    edist_threshold: f64,
    /// Cap on the `hist` deque length: `20 + ⌊3n/λ⌋`. Pycma
    /// `boundary_handler.py:711`.
    hist_cap: usize,

    // --- CMA-ES mutable iterate ---
    pub(crate) m: V,
    pub(crate) sigma: f64,
    p_sigma: V,
    p_c: V,
    c: M,
    /// Eigenvectors of `c` from the most recent eigendecomposition.
    pub(crate) b: M,
    d: V,
    /// Reciprocals of square-roots of `c`'s eigenvalues, used for
    /// `C^{-1/2} = B D^{-1} Bᵀ`.
    pub(crate) d_inv: V,
    rng: ChaCha8Rng,
    generation: u64,

    // --- BoundPenalty mutable iterate ---
    /// Per-coordinate quadratic-penalty weights `γ ∈ R^n`. Initialized
    /// to all-ones (pycma's scalar-1 default, broadcast); upgraded to
    /// `2 · dfit` per coordinate the first generation a γ update sees
    /// the mean violating any bound. `pub(crate)` so the sibling
    /// `BoundedCmaInject` can apply the same penalty to injected
    /// candidates (consistent ranking with regular samples).
    pub(crate) gamma: V,
    /// True once `gamma` has been calibrated from the fitness history.
    /// Until then `gamma` stays at the conservative initial 1; the
    /// penalty is still applied during this period (just with a tiny
    /// coefficient).
    weights_initialized: bool,
    /// Recent fitness IQR estimates, normalized by the average per-
    /// coordinate variance. Front-loaded (newest first) — pushed via
    /// `push_front`, trimmed via `pop_back`.
    hist: VecDeque<f64>,
    /// Sidecar storage: raw f-values (un-penalized) of the most recent
    /// generation, in *sample* order (not in `state.costs`'s sorted-by-
    /// penalized-cost order — γ-update only reads these as a flat bag
    /// of values for the IQR computation, so order is irrelevant).
    raw_costs: Vec<f64>,
}

impl<V, M> BoundedCmaEs<V, M> {
    /// Build a bounded CMA-ES with the default population size
    /// `λ = 4 + ⌊3 ln n⌋` (Hansen 2016 eq. 48), the default TolX
    /// `tol_x = 1e−12 · initial_sigma`, and a seeded RNG.
    ///
    /// # Panics
    ///
    /// Panics if `initial_sigma ≤ 0`.
    pub fn new(initial_mean: V, initial_sigma: f64, seed: u64) -> Self {
        assert!(
            initial_sigma > 0.0,
            "BoundedCmaEs requires initial_sigma > 0, got {}",
            initial_sigma
        );
        Self {
            initial_mean,
            initial_sigma,
            lambda_override: None,
            seed,
            tol_x_override: None,
            stds_override: None,
            state: None,
        }
    }

    /// Override the default population size. The default
    /// `4 + ⌊3 ln n⌋` is what Hansen's tutorial recommends and is
    /// honest for general black-box use; increasing `λ` improves
    /// global-search robustness at the cost of per-iter convergence
    /// rate (Hansen 2016 Section A *Default Parameters*).
    ///
    /// # Panics
    ///
    /// Panics if `lambda < 4`. Smaller populations are explicitly not
    /// recommended (Hansen 2016 footnote 30).
    pub fn with_lambda(mut self, lambda: usize) -> Self {
        assert!(
            lambda >= 4,
            "BoundedCmaEs requires lambda >= 4, got {} (Hansen 2016 footnote 30: \
             smaller populations have strong adverse effects on performance)",
            lambda
        );
        self.lambda_override = Some(lambda);
        self
    }

    /// Override the default TolX (`1e−12 · initial_sigma`, scaled by
    /// `maxᵢ stdsᵢ` when [`with_stds`](Self::with_stds) is set); see
    /// [`CmaEs::with_tol_x`](super::cma_es::CmaEs::with_tol_x).
    pub fn with_tol_x(mut self, tol_x: f64) -> Self {
        self.tol_x_override = Some(tol_x);
        self
    }

    /// Default population size for dimension `n`: `4 + ⌊3 ln n⌋`
    /// (Hansen 2016 eq. 48). Same formula as
    /// [`CmaEs::default_lambda`](super::cma_es::CmaEs::default_lambda).
    pub fn default_lambda(n: usize) -> usize {
        4 + (3.0 * (n as f64).ln()).floor() as usize
    }

    /// Read-only access to the post-update CMA-ES iterate (`m`, `σ`,
    /// `B`, `D^{-1}`, `n`), used by sibling solvers that compose with
    /// bounded CMA-ES — currently only `BoundedCmaInject`, which needs
    /// `C^{-1/2} = B D^{-1} Bᵀ` to clip injected `y_i` per Hansen 2011
    /// eq. 4. Mirrors [`CmaEs::working`](super::cma_es::CmaEs::working).
    /// `None` before [`Solver::init`] has run.
    pub(crate) fn working(&self) -> Option<&Working<V, M>> {
        self.state.as_ref()
    }
}

impl<V, M> BoundedCmaEs<V, M>
where
    V: VectorLen + std::ops::Index<usize, Output = f64>,
{
    /// Set per-coordinate initial standard deviations (pycma's
    /// `CMA_stds`), seeding an anisotropic initial covariance
    /// `C = diag(stds²)` instead of the isotropic default `C = I`. See
    /// [`CmaEs::with_stds`](super::cma_es::CmaEs::with_stds) for the full
    /// semantics; the adaptive boundary penalty benefits from the same
    /// per-coordinate scale (its γ-update reads `σ² · diag(C)`).
    ///
    /// # Panics
    ///
    /// Panics if `stds.len() != initial_mean.len()` or if any entry is not
    /// strictly positive.
    pub fn with_stds(mut self, stds: V) -> Self {
        let n = self.initial_mean.vec_len();
        assert_eq!(
            stds.vec_len(),
            n,
            "BoundedCmaEs::with_stds requires stds.len() == initial_mean.len(), got {} vs {}",
            stds.vec_len(),
            n
        );
        for i in 0..n {
            assert!(
                stds[i] > 0.0,
                "BoundedCmaEs::with_stds requires every std > 0, got stds[{}] = {}",
                i,
                stds[i]
            );
        }
        self.stds_override = Some(stds);
        self
    }
}

impl<V, M> BoundedCmaEs<V, M>
where
    V: VectorLen + Clone + ComponentMulAssign + std::ops::IndexMut<usize, Output = f64>,
    M: MatrixIdentity + MatrixFromDiagonal<V>,
{
    /// Build [`Working`] from `self`'s user-provided settings. Called
    /// once from [`Solver::init`].
    fn build_working(&self) -> Working<V, M> {
        let n = self.initial_mean.vec_len();
        assert!(
            n >= 1,
            "BoundedCmaEs requires the initial mean to be non-empty"
        );
        let lambda = self
            .lambda_override
            .unwrap_or_else(|| Self::default_lambda(n));
        let mu = lambda / 2;

        // Same provisional-µ_eff trick as CmaEs (Hansen Appendix A:
        // µ_eff is invariant under positive-weight rescaling).
        let alpha_cov = 2.0;
        let raw: Vec<f64> = (1..=lambda)
            .map(|i| ((lambda as f64 + 1.0) / 2.0).ln() - (i as f64).ln())
            .collect();
        let sum_pos: f64 = raw[..mu].iter().sum();
        let mu_eff_provisional = sum_pos.powi(2) / raw[..mu].iter().map(|w| w * w).sum::<f64>();

        let c_1 = alpha_cov / ((n as f64 + 1.3).powi(2) + mu_eff_provisional);
        let c_mu_unbounded = alpha_cov * (mu_eff_provisional - 2.0 + 1.0 / mu_eff_provisional)
            / ((n as f64 + 2.0).powi(2) + alpha_cov * mu_eff_provisional / 2.0);
        let c_mu = (1.0 - c_1).min(c_mu_unbounded);

        let (weights, mu_eff, sum_w) = compute_weights(n, lambda, c_1, c_mu);

        let c_sigma = (mu_eff + 2.0) / (n as f64 + mu_eff + 5.0);
        let d_sigma = {
            let inner = ((mu_eff - 1.0) / (n as f64 + 1.0)).sqrt() - 1.0;
            1.0 + 2.0 * inner.max(0.0) + c_sigma
        };
        let c_c = (4.0 + mu_eff / n as f64) / (n as f64 + 4.0 + 2.0 * mu_eff / n as f64);

        let expected_norm = expected_norm_n01(n);
        let h_sigma_threshold = (1.4 + 2.0 / (n as f64 + 1.0)) * expected_norm;
        // Default TolX scales with the largest initial axis std (see
        // `CmaEs::build_working`); reduces to `1e−12 · initial_sigma`
        // without stds. An explicit override still wins.
        let max_std = self
            .stds_override
            .as_ref()
            .map(|s| (0..n).map(|i| s[i]).fold(0.0_f64, f64::max))
            .unwrap_or(1.0);
        let tol_x = self
            .tol_x_override
            .unwrap_or(1e-12 * self.initial_sigma * max_std);

        // BoundPenalty constants: damp, edist threshold, hist cap.
        let damp = (mu_eff / (10.0 * n as f64)).min(1.0);
        let edist_threshold = 3.0 * (n as f64).sqrt().max(1.0) / mu_eff.max(f64::MIN_POSITIVE);
        // Pycma uses `mueff` (not `max(1, mueff)`) in the denominator;
        // we mirror but defensively floor to avoid div-by-zero on
        // pathological `lambda` (mu_eff is always > 0 for lambda >= 4
        // with the default weights).
        let hist_cap = 20 + (3 * n) / lambda;

        // gamma starts at 1.0 per pycma convention (scalar broadcast in
        // pycma; we materialize as a vector of ones for type uniformity).
        let mut gamma = self.initial_mean.clone();
        for i in 0..n {
            gamma[i] = 1.0;
        }

        // Initial covariance: isotropic `C = I` by default, or anisotropic
        // `C = diag(stds²)` when per-coordinate stds are set (eigendecomp is
        // exactly B = I, D = diag(stds), seeded directly in `init`).
        let c = match self.stds_override.as_ref() {
            Some(stds) => {
                let mut sq = stds.clone();
                sq.component_mul_assign(stds);
                M::from_diagonal(&sq)
            }
            None => M::identity(n),
        };

        Working {
            n,
            lambda,
            mu,
            weights,
            mu_eff,
            sum_w,
            c_sigma,
            d_sigma,
            c_c,
            c_1,
            c_mu,
            expected_norm,
            h_sigma_threshold,
            tol_x,
            damp,
            edist_threshold,
            hist_cap,
            m: self.initial_mean.clone(),
            sigma: self.initial_sigma,
            p_sigma: self.initial_mean.clone(),
            p_c: self.initial_mean.clone(),
            c,
            b: M::identity(n),
            d: self.initial_mean.clone(),
            d_inv: self.initial_mean.clone(),
            rng: ChaCha8Rng::seed_from_u64(self.seed),
            generation: 0,
            gamma,
            weights_initialized: false,
            hist: VecDeque::new(),
            raw_costs: Vec::with_capacity(lambda),
        }
    }
}

/// Apply the adaptive boundary penalty to a single sample. Returns
/// `(raw, penalized)`. The repair clamps `x` into `[lower, upper]` and
/// `f` is evaluated at the *clamped* point; the penalty is the mean of
/// `γ_i · (x[i] − clamp(x)[i])²` over coordinates (matches
/// pycma `boundary_handler.py:655`'s `/ N` divisor).
///
/// `pub(crate)` so [`BoundedCmaInject`](crate::solver::BoundedCmaInject)
/// can rank injected candidates by the same penalized fitness regular
/// samples are sorted on — using raw cost on injection would let an
/// out-of-box LM/L-BFGS-B refinement (e.g. landing at the unconstrained
/// minimum) skip the penalty and pollute `state.costs`.
pub(crate) fn evaluate_with_penalty<P, V>(
    problem: &mut Problem<P>,
    x: &V,
    lower: &V,
    upper: &V,
    gamma: &V,
    n: usize,
) -> Result<(f64, f64), P::Error>
where
    P: CostFunction<Param = V, Output = f64>,
    V: Clone + ClampInPlace + std::ops::Index<usize, Output = f64>,
{
    let mut x_rep = x.clone();
    x_rep.clamp_in_place(lower, upper);
    let raw = problem.cost(&x_rep)?;
    let mut penalty = 0.0;
    for i in 0..n {
        let dx = x[i] - x_rep[i];
        penalty += gamma[i] * dx * dx;
    }
    penalty /= n as f64;
    Ok((raw, raw + penalty))
}

/// γ adaptation. Mirrors pycma `BoundPenalty.update`
/// (`boundary_handler.py:669`–`749`). Reads the previous generation's
/// raw fitness from `w.raw_costs`, the current mean / σ / C from `w`,
/// and the bounds from `problem`; writes back to `w.gamma`,
/// `w.weights_initialized`, and `w.hist`.
fn update_gamma<P, V, M>(w: &mut Working<V, M>, problem: &P)
where
    P: BoxConstraints<Param = V>,
    V: Clone
        + ClampInPlace
        + std::ops::Index<usize, Output = f64>
        + std::ops::IndexMut<usize, Output = f64>,
    M: MatDiagonal<V>,
{
    if w.raw_costs.is_empty() {
        return;
    }

    // varis[i] = σ² · diag(C)[i]. Per-axis variance of N(m, σ²C).
    let diag_c = w.c.diagonal();
    let mut mean_varis = 0.0;
    for i in 0..w.n {
        mean_varis += w.sigma * w.sigma * diag_c[i];
    }
    mean_varis /= w.n as f64;

    // dmean[i] = (m[i] − clamp(m)[i]) / sqrt(varis[i]). Mean violation
    // in σ-units along axis i; zero if the mean is feasible on i.
    let mut m_rep = w.m.clone();
    m_rep.clamp_in_place(problem.lower(), problem.upper());
    let mut dmean: Vec<f64> = Vec::with_capacity(w.n);
    let mut any_violation = false;
    for i in 0..w.n {
        let var_i = w.sigma * w.sigma * diag_c[i];
        let d = (w.m[i] - m_rep[i]) / var_i.sqrt();
        if d != 0.0 {
            any_violation = true;
        }
        dmean.push(d);
    }

    // Fitness IQR (pycma's offset definition: indices 3l/4 and l/4 with
    // l = 1 + λ, no interpolation), normalized by mean per-axis variance.
    let mut sorted = w.raw_costs.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let l = 1 + sorted.len();
    let val = (sorted[3 * l / 4] - sorted[l / 4]) / mean_varis;

    // Push to hist (front), trim to hist_cap.
    if val.is_finite() && val > 0.0 {
        w.hist.push_front(val);
    } else if val == f64::INFINITY && !w.hist.is_empty() {
        let max_hist = w.hist.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        w.hist.push_front(max_hist);
    }
    while w.hist.len() > w.hist_cap {
        w.hist.pop_back();
    }
    if w.hist.is_empty() {
        return;
    }

    // dfit = median(hist).
    let mut hsorted: Vec<f64> = w.hist.iter().cloned().collect();
    hsorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let dfit = hsorted[hsorted.len() / 2];

    // Initialize γ on the first generation that sees an infeasible mean.
    // (We skip pycma's `countiter == 2` re-init path — see
    // references/pycma-bound-handling/NOTES.md "Implementation deltas".)
    if any_violation && !w.weights_initialized {
        let init_val = 2.0 * dfit;
        for i in 0..w.n {
            w.gamma[i] = init_val;
        }
        w.weights_initialized = true;
    }

    // Update γ each generation once initialized:
    // - raise γ_i where |dmean_i| − edist_threshold > 0;
    // - decay γ entries that exceed 5·dfit.
    // pycma's active branch (`if 1 < 3:` at boundary_handler.py:731);
    // the elif/else legacy branches are dead code.
    if w.weights_initialized {
        for (i, dmean_i) in dmean.iter().enumerate() {
            let edist_i = dmean_i.abs() - w.edist_threshold;
            if edist_i > 0.0 {
                let factor = ((edist_i / 3.0).tanh() / 2.0 * w.damp).exp();
                w.gamma[i] *= factor;
            }
        }
        let cap = 5.0 * dfit;
        let decay = (-w.damp / 3.0).exp();
        for i in 0..w.n {
            if w.gamma[i] > cap {
                w.gamma[i] *= decay;
            }
        }
    }
}

impl<P, V, M> Solver<P, BasicPopulationState<V>> for BoundedCmaEs<V, M>
where
    P: CostFunction<Param = V, Output = f64> + BoxConstraints,
    V: VectorLen
        + Clone
        + ScaledAdd<f64>
        + ScaleInPlace
        + ComponentMulAssign
        + ClampInPlace
        + NormSquared
        + SampleStandardNormal
        + std::ops::Index<usize, Output = f64>
        + std::ops::IndexMut<usize, Output = f64>,
    M: MatrixIdentity
        + MatrixFromDiagonal<V>
        + MatVec<V>
        + MatTransposeVec<V>
        + MatDiagonal<V>
        + ScaleInPlace
        + RankOneUpdate<V>
        + SymmetricEigen<V>
        + Clone,
{
    type Error = P::Error;

    fn init(
        &mut self,
        problem: &mut Problem<P>,
        mut state: BasicPopulationState<V>,
    ) -> Result<BasicPopulationState<V>, Self::Error> {
        // Idempotent: a paused BoundedCmaEs re-entered via `run_loop`
        // must not have its evolution state rebuilt. Mirrors the
        // CmaEs::init early-return; see that solver's docs for the
        // chain-resumption use case.
        if self.state.is_some() {
            return Ok(state);
        }
        let mut w = self.build_working();
        // Seed (b, d, d_inv): isotropic (1, …, 1) by default, or
        // (d, d_inv) = (stds, 1/stds) for the diagonal `C = diag(stds²)`
        // (b stays identity). See `CmaEs::init` for why no eigensolve runs
        // here — the diagonal decomposition is exact and order-preserving.
        w.p_sigma.scale_in_place(0.0);
        w.p_c.scale_in_place(0.0);
        if let Some(stds) = self.stds_override.as_ref() {
            for i in 0..w.n {
                w.d[i] = stds[i];
                w.d_inv[i] = 1.0 / stds[i];
            }
        } else {
            for i in 0..w.n {
                w.d[i] = 1.0;
                w.d_inv[i] = 1.0;
            }
        }

        // Project an infeasible initial mean once at iter 0 so the iter-0
        // search distribution is centered in feasibility. Mirrors
        // ProjectedGradientDescent::init's iter-0 projection. pycma's
        // BoundPenalty doesn't project the mean — we do because (a) the
        // user sees `state.param()` which is a sample drawn from N(m, σ²C),
        // and (b) it shaves a few generations off the initial recovery
        // when the user passes an out-of-box starting mean.
        let lo = problem.inner().lower().clone();
        let hi = problem.inner().upper().clone();
        w.m.clamp_in_place(&lo, &hi);

        // First generation: x_k = m + σ B (D ⊙ z_k). Isotropic default keeps
        // the fast path x_k = m + σ z_k (bit-identical when B = I, D = 1);
        // the anisotropic case applies the B·D map.
        let anisotropic = self.stds_override.is_some();
        state.candidates.clear();
        state.costs.clear();
        w.raw_costs.clear();
        for _k in 0..w.lambda {
            let z_k = V::sample_standard_normal(&w.m, &mut w.rng);
            let mut x_k = w.m.clone();
            if anisotropic {
                let mut bd_z = z_k;
                bd_z.component_mul_assign(&w.d);
                let bd_z = w.b.matvec(&bd_z);
                x_k.scaled_add(w.sigma, &bd_z);
            } else {
                x_k.scaled_add(w.sigma, &z_k);
            }
            let (raw, pen) = evaluate_with_penalty(problem, &x_k, &lo, &hi, &w.gamma, w.n)?;
            state.candidates.push(x_k);
            state.costs.push(pen);
            w.raw_costs.push(raw);
        }
        sort_population_ascending(&mut state.candidates, &mut state.costs);

        self.state = Some(w);
        Ok(state)
    }

    fn next_iter(
        &mut self,
        problem: &mut Problem<P>,
        mut state: BasicPopulationState<V>,
    ) -> Result<(BasicPopulationState<V>, Option<TerminationReason>), Self::Error> {
        let w = self
            .state
            .as_mut()
            .expect("BoundedCmaEs::init must run before next_iter");

        w.generation += 1;

        // Recombination uses the un-repaired samples. y_{i:λ} = (x_{i:λ} − m) / σ
        // for the *previous* m, σ. (state.candidates carries the most recent
        // generation's x's, sorted ascending by *penalized* cost — for
        // recombination only the rank order matters.)
        let mut y_sorted: Vec<V> = state
            .candidates
            .iter()
            .map(|x| {
                let mut y = x.clone();
                y.scaled_add(-1.0, &w.m);
                y.scale_in_place(1.0 / w.sigma);
                y
            })
            .collect();

        // ⟨y⟩_w = Σ_{i=1..µ} w_i y_{i:λ}.
        let mut y_w = w.m.clone();
        y_w.scale_in_place(0.0);
        for (i, y_i) in y_sorted.iter().enumerate().take(w.mu) {
            y_w.scaled_add(w.weights[i], y_i);
        }

        // m ← m + σ ⟨y⟩_w.
        w.m.scaled_add(w.sigma, &y_w);

        // C^{−1/2} ⟨y⟩_w = B (D^{−1} ⊙ Bᵀ ⟨y⟩_w).
        let mut bt_y_w = w.b.mat_transpose_vec(&y_w);
        bt_y_w.component_mul_assign(&w.d_inv);
        let c_invsqrt_y_w = w.b.matvec(&bt_y_w);

        // p_σ ← (1 − c_σ) p_σ + √(c_σ(2 − c_σ) µ_eff) C^{−1/2} ⟨y⟩_w.
        w.p_sigma.scale_in_place(1.0 - w.c_sigma);
        let coef_sigma = (w.c_sigma * (2.0 - w.c_sigma) * w.mu_eff).sqrt();
        w.p_sigma.scaled_add(coef_sigma, &c_invsqrt_y_w);

        // σ ← σ exp((c_σ / d_σ) (‖p_σ‖ / E‖N(0,I)‖ − 1)).
        let p_sigma_norm = w.p_sigma.norm_squared().sqrt();
        let log_factor = (w.c_sigma / w.d_sigma) * (p_sigma_norm / w.expected_norm - 1.0);
        w.sigma *= log_factor.exp();

        // h_σ test (Hansen 2016 p. 31, denominator uses 2(g+1)).
        let g_for_h = (w.generation + 1) as i32;
        let exponent = 2 * g_for_h;
        let denom = (1.0 - (1.0 - w.c_sigma).powi(exponent)).sqrt();
        let h_sigma = if p_sigma_norm / denom < w.h_sigma_threshold {
            1.0
        } else {
            0.0
        };

        // p_c update.
        w.p_c.scale_in_place(1.0 - w.c_c);
        let coef_c = h_sigma * (w.c_c * (2.0 - w.c_c) * w.mu_eff).sqrt();
        w.p_c.scaled_add(coef_c, &y_w);

        // C update (eq. 47).
        let delta_h = (1.0 - h_sigma) * w.c_c * (2.0 - w.c_c);
        let c_scale = 1.0 + w.c_1 * delta_h - w.c_1 - w.c_mu * w.sum_w;
        w.c.scale_in_place(c_scale);
        w.c.rank_one_update(w.c_1, &w.p_c);
        for (i, y_i) in y_sorted.iter().enumerate() {
            let w_i = w.weights[i];
            let w_i_o = if w_i >= 0.0 {
                w_i
            } else {
                let mut bt_y = w.b.mat_transpose_vec(y_i);
                bt_y.component_mul_assign(&w.d_inv);
                let cinv_norm_sq = bt_y.norm_squared();
                if cinv_norm_sq > 0.0 {
                    w_i * (w.n as f64) / cinv_norm_sq
                } else {
                    0.0
                }
            };
            if w_i_o != 0.0 {
                w.c.rank_one_update(w.c_mu * w_i_o, y_i);
            }
        }
        drop(std::mem::take(&mut y_sorted));

        // Refresh eigendecomposition of the new C.
        let (b_new, eigs) = match w.c.try_eigh() {
            Ok(pair) => pair,
            Err(_) => return Ok((state, Some(TerminationReason::SolverFailed))),
        };
        w.b = b_new;
        for i in 0..w.n {
            let lam = eigs[i].max(1e-30);
            let s = lam.sqrt();
            w.d[i] = s;
            w.d_inv[i] = 1.0 / s;
        }

        // γ adaptation — runs after the m / σ / C update so it sees
        // the post-recombination state, before the new generation is
        // sampled. Consumes `w.raw_costs` (previous generation's raw
        // fitness, in sample order — γ-update only needs the IQR).
        update_gamma(w, problem.inner());

        // Sample the new generation: x_k = m + σ B (D ⊙ z_k); evaluate
        // at the repaired point, accumulate raw and penalized costs.
        state.candidates.clear();
        state.costs.clear();
        w.raw_costs.clear();
        let lo = problem.inner().lower().clone();
        let hi = problem.inner().upper().clone();
        for _k in 0..w.lambda {
            let z_k = V::sample_standard_normal(&w.m, &mut w.rng);
            let mut bd_z = z_k;
            bd_z.component_mul_assign(&w.d);
            let bd_z = w.b.matvec(&bd_z);
            let mut x_k = w.m.clone();
            x_k.scaled_add(w.sigma, &bd_z);
            let (raw, pen) = evaluate_with_penalty(problem, &x_k, &lo, &hi, &w.gamma, w.n)?;
            state.candidates.push(x_k);
            state.costs.push(pen);
            w.raw_costs.push(raw);
        }
        sort_population_ascending(&mut state.candidates, &mut state.costs);

        Ok((state, None))
    }

    fn terminate(&self, _state: &BasicPopulationState<V>) -> Option<TerminationReason> {
        let w = self.state.as_ref()?;
        let mut max_d = 0.0_f64;
        for i in 0..w.n {
            let v = w.d[i];
            if v > max_d {
                max_d = v;
            }
        }
        if w.sigma * max_d < w.tol_x {
            return Some(TerminationReason::SolverConverged);
        }
        None
    }
}
