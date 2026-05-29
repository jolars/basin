use crate::core::math::{
    ComponentMulAssign, MatTransposeVec, MatVec, MatrixFromDiagonal, MatrixIdentity, NormSquared,
    RankOneUpdate, SampleStandardNormal, ScaleInPlace, ScaledAdd, SymmetricEigen, VectorLen,
};
use crate::core::problem::{CostFunction, Problem};
use crate::core::rng::{ChaCha8Rng, SeedableRng};
use crate::core::solver::Solver;
use crate::core::state::BasicPopulationState;
use crate::core::termination::TerminationReason;

/// `(µ/µ_W, λ)`-CMA Evolution Strategy with negative weights (aCMA-ES)
/// from Hansen 2016 (*The CMA Evolution Strategy: A Tutorial*).
///
/// Stochastic, derivative-free, population-based — the standard
/// black-box optimizer for ill-conditioned, non-separable, non-convex
/// continuous problems. Uses a multivariate normal `N(m, σ²C)` to
/// sample candidates, then adapts `m`, `σ`, and the covariance `C` from
/// the selected best `µ` candidates plus their conjugate evolution
/// path. Hansen 2016 Figure 6 / eqs (38)–(47) is the algorithm-summary
/// fixture; section A is the parameter table.
///
/// # Algorithm
///
/// At [`init`](Solver::init): set `m = initial_mean`,
/// `σ = initial_sigma`, `p_σ = p_c = 0`, `C = I` (or `C = diag(stds²)`
/// when [`with_stds`](Self::with_stds) is set), and sample the first
/// generation `x_k = m + σ B (D ⊙ z_k)` with `z_k ~ N(0, I)` (which is
/// `m + σ z_k` in the isotropic default, where `B D = I`).
///
/// Each [`next_iter`](Solver::next_iter) processes the previous
/// generation's evaluations and samples a fresh generation:
///
/// ```text
/// generation ← generation + 1
///
/// # use sorted x_{i:λ} from previous generation (state.candidates)
/// y_{i:λ} = (x_{i:λ} − m) / σ
/// ⟨y⟩_w = Σ_{i=1..µ} w_i y_{i:λ}                          # eq. 41
/// m ← m + c_m σ ⟨y⟩_w  (with c_m = 1)                     # eq. 42
///
/// # step-size: conjugate path + log-update
/// C^{−1/2} ⟨y⟩_w = B (D^{−1} ⊙ Bᵀ ⟨y⟩_w)
/// p_σ ← (1−c_σ) p_σ + √(c_σ(2−c_σ) µ_eff) · C^{−1/2} ⟨y⟩_w  # eq. 43
/// σ ← σ · exp((c_σ/d_σ) (‖p_σ‖ / E‖N(0,I)‖ − 1))           # eq. 44
///
/// # rank-1 + rank-µ update (with negative-weight rescaling)
/// h_σ = 1 iff ‖p_σ‖ / √(1−(1−c_σ)^(2(g+1))) < (1.4+2/(n+1))·E‖N(0,I)‖
/// p_c ← (1−c_c) p_c + h_σ √(c_c(2−c_c) µ_eff) ⟨y⟩_w        # eq. 45
/// w_i° = w_i if w_i ≥ 0 else w_i · n / ‖C^{−1/2} y_{i:λ}‖²  # eq. 46
/// δ_h = (1−h_σ) c_c (2−c_c)
/// C ← (1 + c_1 δ_h − c_1 − c_µ Σ w_j) C
///     + c_1 p_c p_cᵀ + c_µ Σ_i w_i° y_{i:λ} y_{i:λ}ᵀ        # eq. 47
///
/// # refresh eigendecomposition of new C → (B, d²)
/// d_i ← max(d²_i, 0)^(1/2);  d_i^{−1} ← 1 / d_i
///
/// # sample new generation
/// for k = 1..λ:  z_k ~ N(0, I);  x_k = m + σ B (d ⊙ z_k)
/// ```
///
/// The eigendecomposition is refreshed every iteration. Hansen's
/// suggested optimization (eigendecompose every `max(1, ⌊1/(10n(c_1+c_µ))⌋)`
/// generations, Appendix B.2) is deferred — at small to moderate `n`
/// the cost is dominated by `f` evaluations anyway, and the refresh
/// frequency would change the per-iteration cost calculus.
///
/// # Default parameters
///
/// All defaults follow Hansen 2016 Table 1 (the 2016 negative-weights
/// setting); see [`new`](Self::new) and the per-field doc comments
/// below for the exact formulas. The user supplies only `n` (via the
/// initial mean's length), the initial mean, the initial step-size,
/// and the seed.
///
/// # Reproducibility
///
/// The solver carries a [`ChaCha8Rng`] seeded from the `seed: u64`
/// passed to [`new`](Self::new) — same seed → same iterate trajectory
/// on every platform basin builds for (including
/// `wasm32-unknown-unknown`).
///
/// # Contract
///
/// - **Caller must:** implement [`CostFunction<Param = V, Output = f64>`]
///   on the problem. CMA-ES is derivative-free; no [`Gradient`](crate::Gradient) /
///   [`Jacobian`](crate::Jacobian) required.
/// - **Caller must:** hand in a
///   [`BasicPopulationState::with_size(λ)`](crate::BasicPopulationState::with_size)
///   matching the solver's λ. The default
///   λ = `4 + ⌊3 ln n⌋` is exposed via [`default_lambda`](Self::default_lambda).
/// - **Caller must:** ensure `initial_sigma > 0`.
/// - **Implementor (this solver) must:** maintain the
///   [`PopulationState`](crate::core::state::PopulationState)
///   sorted-by-cost invariant on `state.candidates` / `state.costs`
///   at the start and end of every iteration, and seed `state.cost()`
///   with the best of the first sampled generation.
///
/// # Termination
///
/// Solver-internal: `σ · max d_i < tol_x` → [`TerminationReason::SolverConverged`]
/// (CMA-ES TolX, Hansen 2016 Appendix B.3). Defaults to
/// `1e−12 · initial_sigma` per Hansen's recommendation (scaled by
/// `maxᵢ stdsᵢ` when [`with_stds`](Self::with_stds) is set, to stay
/// relative to the initial spread). Pair with the
/// framework's [`MaxIter`](crate::core::termination::MaxIter) /
/// [`MaxCostEvals`](crate::core::termination::MaxCostEvals) for budget
/// control; both work on
/// [`BasicPopulationState`]
/// without modification. Other CMA-ES termination heuristics
/// (NoEffectAxis, NoEffectCoord, ConditionCov, EqualFunValues,
/// Stagnation, TolXUp, TolFun) are out of scope for S8 vanilla and
/// will land alongside the bounded variant in S9 / restart machinery
/// in S11.
///
/// # Backends
///
/// LA-heavy: requires symmetric eigendecomposition, scalar-and-rank-1
/// matrix updates, and matrix-vector / transposed matrix-vector
/// products. Wired and tested for the default `Vec<f64>` /
/// [`DenseMatrix`](crate::DenseMatrix) backend (pure-Rust cyclic Jacobi
/// eigensolver — no feature flag, `wasm`-clean), `nalgebra::DVector<f64>`
/// / `nalgebra::DMatrix<f64>` (feature `nalgebra`), `ndarray::Array1<f64>`
/// / `ndarray::Array2<f64>` (feature `ndarray`, also wired to the cyclic
/// Jacobi solver — `wasm`-clean), and `faer::Col<f64>` / `faer::Mat<f64>`
/// (feature `faer`). Sparse covariance is not meaningful for CMA-ES — the
/// rank-µ update densifies any starting pattern.
///
/// # Examples
///
/// See [`RandomSearch`](crate::RandomSearch) for the population-based
/// `Executor` pattern — a `BasicPopulationState` sized to λ. Construct the
/// solver with `CmaEs::new` and pass the same λ to the state.
pub struct CmaEs<V, M> {
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
/// updated each [`Solver::next_iter`].
///
/// `pub(crate)` (not public) so sibling solvers in `crate::solver` can
/// read the post-update `m`, `σ`, `B`, `D^{-1}` they need for
/// injection-style composition (`CmaInject` uses these to clip injected
/// `y_i` in Mahalanobis distance per Hansen 2011 eq. 4). Not a stable
/// public surface.
pub(crate) struct Working<V, M> {
    // --- constants (computed once at init) ---
    pub(crate) n: usize,
    lambda: usize,
    mu: usize,
    /// All λ recombination weights (sum of positives = 1; negatives
    /// scaled per Hansen Table 1 rows (50)–(53)).
    weights: Vec<f64>,
    /// `µ_eff = (Σ_{i=1..µ} w_i)² / Σ_{i=1..µ} w_i² = 1 / Σ w_i²`
    /// because the positive weights sum to 1.
    mu_eff: f64,
    /// `Σ_{i=1..λ} w_i`. Negative when negative weights are in use
    /// (default setting); the C-update scalar `1 − c_µ · sum_w`
    /// inflates rather than decays C as a result. With Hansen's
    /// `α_µ_minus = 1 + c_1/c_µ` choice, `c_1 + c_µ · sum_w ≈ 0`,
    /// so the C scalar is approximately 1 (eq. 47).
    sum_w: f64,
    c_sigma: f64,
    d_sigma: f64,
    c_c: f64,
    c_1: f64,
    c_mu: f64,
    expected_norm: f64,
    /// `(1.4 + 2/(n+1)) · E‖N(0,I)‖` — RHS of the h_σ test (eq. 47
    /// callout footnote / Hansen 2016 p. 31).
    h_sigma_threshold: f64,
    tol_x: f64,

    // --- mutable iterate ---
    pub(crate) m: V,
    pub(crate) sigma: f64,
    p_sigma: V,
    p_c: V,
    c: M,
    /// Eigenvectors of `c` from the most recent eigendecomposition.
    pub(crate) b: M,
    /// Square roots of eigenvalues (the diagonal `D` in Hansen's
    /// `B D Bᵀ` factorization).
    d: V,
    /// Reciprocals of `d`, used for `C^{−1/2} = B D^{−1} Bᵀ`.
    pub(crate) d_inv: V,

    rng: ChaCha8Rng,
    /// Generation counter for the h_σ formula (Hansen 2016 p. 31:
    /// uses `(1−c_σ)^{2(g+1)}` in the bound). Incremented at the top
    /// of every [`Solver::next_iter`].
    generation: u64,
}

impl<V, M> CmaEs<V, M> {
    /// Build a CMA-ES with the default population size
    /// `λ = 4 + ⌊3 ln n⌋` (Hansen 2016 eq. 48), the default TolX
    /// `tol_x = 1e−12 · initial_sigma`, and a seeded RNG.
    ///
    /// # Panics
    ///
    /// Panics if `initial_sigma ≤ 0`.
    pub fn new(initial_mean: V, initial_sigma: f64, seed: u64) -> Self {
        assert!(
            initial_sigma > 0.0,
            "CmaEs requires initial_sigma > 0, got {}",
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
    /// Panics if `lambda < 4`. Smaller populations are explicitly
    /// not recommended (Hansen 2016 footnote 30: "Decreasing λ is not
    /// recommended").
    pub fn with_lambda(mut self, lambda: usize) -> Self {
        assert!(
            lambda >= 4,
            "CmaEs requires lambda >= 4, got {} (Hansen 2016 footnote 30: \
             smaller populations have strong adverse effects on performance)",
            lambda
        );
        self.lambda_override = Some(lambda);
        self
    }

    /// Override the default TolX. The check fires when
    /// `σ · max_i d_i < tol_x`, where `d_i` are square roots of `C`'s
    /// eigenvalues — i.e. the largest standard deviation of any axis of
    /// the search distribution drops below the tolerance. Hansen 2016
    /// Appendix B.3 default is `1e−12 · initial_sigma` (scaled by
    /// `maxᵢ stdsᵢ` when [`with_stds`](Self::with_stds) is set). An
    /// explicit override here wins regardless of the builder-call order.
    pub fn with_tol_x(mut self, tol_x: f64) -> Self {
        self.tol_x_override = Some(tol_x);
        self
    }

    /// Default population size for dimension `n`: `4 + ⌊3 ln n⌋`
    /// (Hansen 2016 eq. 48). Exposed so callers building a
    /// [`BasicPopulationState::with_size`] can match the solver's
    /// internal default without re-deriving the formula.
    pub fn default_lambda(n: usize) -> usize {
        4 + (3.0 * (n as f64).ln()).floor() as usize
    }

    /// Read-only access to the post-update CMA-ES iterate (`m`, `σ`,
    /// `B`, `D^{-1}`, `n`), used by sibling solvers that compose with
    /// CMA-ES — currently only `CmaInject`, which needs `C^{-1/2} =
    /// B D^{-1} Bᵀ` to clip injected `y_i` per Hansen 2011 eq. 4.
    /// `None` before [`Solver::init`] has run.
    pub(crate) fn working(&self) -> Option<&Working<V, M>> {
        self.state.as_ref()
    }
}

impl<V, M> CmaEs<V, M>
where
    V: VectorLen + std::ops::Index<usize, Output = f64>,
{
    /// Set per-coordinate initial standard deviations (pycma's
    /// `CMA_stds`), seeding an anisotropic initial covariance
    /// `C = diag(stds²)` instead of the isotropic default `C = I`. The
    /// first generation then samples `m + σ · diag(stds) · N(0, I)` — i.e.
    /// optimizing in coordinates rescaled by `1/stds`. `σ` (from
    /// [`new`](Self::new)) remains the scalar overall step-size; `stds`
    /// only sets the *shape*. Leaving this unset keeps `C = I`.
    ///
    /// Use this on problems whose parameters live on heterogeneous scales,
    /// so the search does not have to spend generations learning the
    /// scaling through covariance adaptation.
    ///
    /// When set, the default TolX also scales with the largest initial
    /// axis: `tol_x = 1e−12 · initial_sigma · maxᵢ stdsᵢ` (so termination
    /// stays relative to the initial spread). An explicit
    /// [`with_tol_x`](Self::with_tol_x) overrides this regardless of order.
    ///
    /// # Panics
    ///
    /// Panics if `stds.len() != initial_mean.len()` or if any entry is not
    /// strictly positive (a non-positive std would make `1/stds` non-finite
    /// in the `C^{−1/2}` factor).
    pub fn with_stds(mut self, stds: V) -> Self {
        let n = self.initial_mean.vec_len();
        assert_eq!(
            stds.vec_len(),
            n,
            "CmaEs::with_stds requires stds.len() == initial_mean.len(), got {} vs {}",
            stds.vec_len(),
            n
        );
        for i in 0..n {
            assert!(
                stds[i] > 0.0,
                "CmaEs::with_stds requires every std > 0, got stds[{}] = {}",
                i,
                stds[i]
            );
        }
        self.stds_override = Some(stds);
        self
    }
}

/// Asymptotic expansion of `E‖N(0, I_n)‖ = √2 Γ((n+1)/2) / Γ(n/2)`.
/// Accurate to ~10 digits for `n ≥ 1`; avoids needing `lgamma` (which
/// is not in stable `std`).
pub(crate) fn expected_norm_n01(n: usize) -> f64 {
    let n = n as f64;
    n.sqrt() * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n * n))
}

/// Compute the recombination weights and derived constants per
/// Hansen 2016 Table 1 rows (49)–(53), plus `µ_eff` and `µ_eff_neg`.
/// Returns `(weights, mu_eff, sum_w)`.
pub(crate) fn compute_weights(
    n: usize,
    lambda: usize,
    c_1: f64,
    c_mu: f64,
) -> (Vec<f64>, f64, f64) {
    let mu = lambda / 2;
    // Raw preliminary weights w_i' = ln((λ+1)/2) − ln i (eq. 49).
    let raw: Vec<f64> = (1..=lambda)
        .map(|i| ((lambda as f64 + 1.0) / 2.0).ln() - (i as f64).ln())
        .collect();

    // Positive sum and negative sum (over raw values).
    let sum_pos: f64 = raw[..mu].iter().sum();
    // µ_eff is defined on the *positive* weights only and is invariant
    // under positive-rescaling, so compute it from raw[..mu] (eq. 8 /
    // Table 1 caption).
    let raw_pos_norm_sq: f64 = raw[..mu].iter().map(|w| w * w).sum();
    let mu_eff = sum_pos.powi(2) / raw_pos_norm_sq;

    // µ_eff_neg from negative-portion raws (Table 1 caption).
    let sum_neg: f64 = raw[mu..].iter().sum();
    let raw_neg_norm_sq: f64 = raw[mu..].iter().map(|w| w * w).sum();
    let mu_eff_neg = if raw_neg_norm_sq > 0.0 {
        sum_neg.powi(2) / raw_neg_norm_sq
    } else {
        0.0
    };

    // Three bounds on the negative-weight scale (eqs. 50–52).
    let alpha_mu_minus = 1.0 + c_1 / c_mu;
    let alpha_mu_eff_minus = 1.0 + 2.0 * mu_eff_neg / (mu_eff + 2.0);
    let alpha_pos_def_minus = (1.0 - c_1 - c_mu) / (n as f64 * c_mu);
    let alpha_neg = alpha_mu_minus
        .min(alpha_mu_eff_minus)
        .min(alpha_pos_def_minus);

    // Final weights (eq. 53):
    // - positive: w_i = w_i' / Σ|w_j'|+ (positives sum to 1).
    // - negative: w_i = (alpha_neg / Σ|w_j'|−) · w_i'.
    let sum_abs_neg: f64 = raw[mu..].iter().map(|w| -w).sum();
    let mut weights = Vec::with_capacity(lambda);
    for (i, &raw_i) in raw.iter().enumerate() {
        let w = if i < mu {
            raw_i / sum_pos
        } else if sum_abs_neg > 0.0 {
            alpha_neg * raw_i / sum_abs_neg
        } else {
            0.0
        };
        weights.push(w);
    }

    let sum_w: f64 = weights.iter().sum();
    (weights, mu_eff, sum_w)
}

impl<V, M> CmaEs<V, M>
where
    V: VectorLen + Clone + ComponentMulAssign + std::ops::Index<usize, Output = f64>,
    M: MatrixIdentity + MatrixFromDiagonal<V>,
{
    /// Build [`Working`] from `self`'s user-provided settings. Called
    /// once from [`Solver::init`].
    fn build_working(&self) -> Working<V, M> {
        let n = self.initial_mean.vec_len();
        assert!(n >= 1, "CmaEs requires the initial mean to be non-empty");
        let lambda = self
            .lambda_override
            .unwrap_or_else(|| Self::default_lambda(n));
        let mu = lambda / 2;
        // Hansen Table 1 rows (55)–(58).
        let alpha_cov = 2.0;
        // The c_1 / c_µ formulas need µ_eff, which depends on positive
        // weights only. Compute µ_eff once from the raw weights to feed
        // c_1 / c_µ, then re-derive the final negative weights against
        // those c_1 / c_µ via `compute_weights` (Hansen explains the
        // apparent circular dependency in Appendix A: µ_eff is invariant
        // under positive-weight rescaling, so a one-shot computation
        // suffices).
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
        // d_σ = 1 + 2 · max(0, √((µ_eff−1)/(n+1)) − 1) + c_σ
        // (Hansen 2016 Table 1 row 55).
        let d_sigma = {
            let inner = ((mu_eff - 1.0) / (n as f64 + 1.0)).sqrt() - 1.0;
            1.0 + 2.0 * inner.max(0.0) + c_sigma
        };
        let c_c = (4.0 + mu_eff / n as f64) / (n as f64 + 4.0 + 2.0 * mu_eff / n as f64);

        let expected_norm = expected_norm_n01(n);
        let h_sigma_threshold = (1.4 + 2.0 / (n as f64 + 1.0)) * expected_norm;
        // Default TolX scales with the largest initial axis std so the
        // convergence test stays relative to the initial spread: with
        // anisotropic stds the largest single-axis std is
        // `initial_sigma · maxᵢ stdsᵢ`, and the terminate check is
        // `σ · maxᵢ dᵢ < tol_x`. Reduces to `1e−12 · initial_sigma` when
        // stds are absent (max_std = 1). An explicit override still wins.
        let max_std = self
            .stds_override
            .as_ref()
            .map(|s| (0..n).map(|i| s[i]).fold(0.0_f64, f64::max))
            .unwrap_or(1.0);
        let tol_x = self
            .tol_x_override
            .unwrap_or(1e-12 * self.initial_sigma * max_std);

        // Initial covariance: isotropic `C = I` by default, or anisotropic
        // `C = diag(stds²)` when per-coordinate stds are set. For a diagonal
        // C the eigendecomposition is exactly `B = I`, `D = diag(stds)`, so
        // `init` seeds (b, d, d_inv) directly without an eigensolve.
        let c = match self.stds_override.as_ref() {
            Some(stds) => {
                let mut sq = stds.clone();
                sq.component_mul_assign(stds);
                M::from_diagonal(&sq)
            }
            None => M::identity(n),
        };

        // Initial mutable state. The vectors p_σ, p_c, d, d_inv are
        // sized like `initial_mean` via clone; their values are
        // overwritten in `init` (zeros for the paths; for the d-vectors,
        // ones when C = I or stds when anisotropic).
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
        }
    }
}

/// Sort `candidates` and `costs` jointly by ascending cost. NaN costs
/// sort last (mirrors `nelder_mead::sort_simplex` /
/// `random_search::sort_population_ascending`).
pub(crate) fn sort_population_ascending<V>(candidates: &mut [V], costs: &mut [f64]) {
    let n = candidates.len();
    debug_assert_eq!(n, costs.len());
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&i, &j| {
        costs[i]
            .partial_cmp(&costs[j])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    apply_permutation(candidates, &idx);
    apply_permutation(costs, &idx);
}

fn apply_permutation<T>(slice: &mut [T], idx: &[usize]) {
    let mut visited = vec![false; slice.len()];
    for start in 0..slice.len() {
        if visited[start] || idx[start] == start {
            visited[start] = true;
            continue;
        }
        let mut current = start;
        loop {
            let next = idx[current];
            visited[current] = true;
            if next == start {
                break;
            }
            slice.swap(current, next);
            current = next;
        }
    }
}

impl<P, V, M> Solver<P, BasicPopulationState<V>> for CmaEs<V, M>
where
    P: CostFunction<Param = V, Output = f64>,
    V: VectorLen
        + Clone
        + ScaledAdd<f64>
        + ScaleInPlace
        + ComponentMulAssign
        + NormSquared
        + SampleStandardNormal
        + std::ops::Index<usize, Output = f64>
        + std::ops::IndexMut<usize, Output = f64>,
    M: MatrixIdentity
        + MatrixFromDiagonal<V>
        + MatVec<V>
        + MatTransposeVec<V>
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
        // Idempotent: if a previous init already seeded the internal
        // state, return the caller-provided state untouched. This lets
        // chain-style outer solvers (e.g. MaLsChCma) call `run_loop`
        // repeatedly on a paused CmaEs without clobbering its evolution
        // state on every entry. For non-resumption use this is a no-op:
        // a freshly constructed CmaEs has `self.state == None` and
        // proceeds through the full setup below.
        if self.state.is_some() {
            return Ok(state);
        }
        let mut w = self.build_working();
        // Zero the path vectors and seed (b, d, d_inv). For the isotropic
        // default (C = I) that is (d, d_inv) = (1, …, 1). With per-coordinate
        // stds the covariance is the diagonal `C = diag(stds²)`, whose
        // eigendecomposition is exactly `B = I`, `D = diag(stds)`, so we seed
        // d = stds, d_inv = 1/stds directly (no eigensolve — a generic
        // `try_eigh` could reorder eigenvalues and scramble the per-coordinate
        // correspondence). b stays the identity from `build_working`.
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

        // First generation: x_k = m + σ B (D ⊙ z_k). The isotropic default
        // keeps the fast path x_k = m + σ z_k (B = I, D = 1 makes the two
        // bit-identical); the anisotropic case applies the B·D map.
        let anisotropic = self.stds_override.is_some();
        state.candidates.clear();
        state.costs.clear();
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
            let cost = problem.cost(&x_k)?;
            state.candidates.push(x_k);
            state.costs.push(cost);
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
            .expect("CmaEs::init must run before next_iter");

        w.generation += 1;

        // Rebuild y_{i:λ} = (x_{i:λ} − m) / σ for the *previous* m, σ.
        // (state.candidates carries the most recent generation's x's,
        // sorted ascending by cost.)
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

        // m ← m + σ ⟨y⟩_w (c_m = 1 by default).
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

        // p_c ← (1 − c_c) p_c + h_σ √(c_c(2 − c_c) µ_eff) ⟨y⟩_w.
        w.p_c.scale_in_place(1.0 - w.c_c);
        let coef_c = h_sigma * (w.c_c * (2.0 - w.c_c) * w.mu_eff).sqrt();
        w.p_c.scaled_add(coef_c, &y_w);

        // C update (eq. 47):
        //   C ← (1 + c_1 δ_h − c_1 − c_µ Σ w_j) C
        //       + c_1 p_c p_cᵀ
        //       + c_µ Σ_i w_i° y_{i:λ} y_{i:λ}ᵀ
        // with w_i° = w_i for w_i ≥ 0, else w_i · n / ‖C^{−1/2} y_{i:λ}‖².
        let delta_h = (1.0 - h_sigma) * w.c_c * (2.0 - w.c_c);
        let c_scale = 1.0 + w.c_1 * delta_h - w.c_1 - w.c_mu * w.sum_w;
        w.c.scale_in_place(c_scale);
        w.c.rank_one_update(w.c_1, &w.p_c);
        // Negative-weight path rescales by n / ‖C^{−1/2} y_i‖²;
        // positive-weight path uses w_i directly (eq. 46).
        for (i, y_i) in y_sorted.iter().enumerate() {
            let w_i = w.weights[i];
            let w_i_o = if w_i >= 0.0 {
                w_i
            } else {
                // ‖C^{−1/2} y_i‖² = ‖D^{−1} ⊙ Bᵀ y_i‖² (orthogonal B).
                let mut bt_y = w.b.mat_transpose_vec(y_i);
                bt_y.component_mul_assign(&w.d_inv);
                let cinv_norm_sq = bt_y.norm_squared();
                if cinv_norm_sq > 0.0 {
                    w_i * (w.n as f64) / cinv_norm_sq
                } else {
                    // Pathological zero-direction; drop this contribution.
                    0.0
                }
            };
            if w_i_o != 0.0 {
                w.c.rank_one_update(w.c_mu * w_i_o, y_i);
            }
        }
        // Drop y_sorted now to free memory before the eigendecomposition.
        drop(std::mem::take(&mut y_sorted));

        // Refresh eigendecomposition of the new C.
        let (b_new, eigs) = match w.c.try_eigh() {
            Ok(pair) => pair,
            Err(_) => return Ok((state, Some(TerminationReason::SolverFailed))),
        };
        w.b = b_new;
        // d_i = √max(λ_i, 0); d_inv_i = 1/d_i. Floating-point can produce
        // tiny negative eigenvalues even when the algorithm preserves
        // positive definiteness; clamp to a small positive floor before
        // taking the square root.
        for i in 0..w.n {
            let lam = eigs[i].max(1e-30);
            let s = lam.sqrt();
            w.d[i] = s;
            w.d_inv[i] = 1.0 / s;
        }

        // Sample new generation: x_k = m + σ B (D ⊙ z_k).
        state.candidates.clear();
        state.costs.clear();
        for _k in 0..w.lambda {
            let z_k = V::sample_standard_normal(&w.m, &mut w.rng);
            let mut bd_z = z_k;
            bd_z.component_mul_assign(&w.d);
            let bd_z = w.b.matvec(&bd_z);
            let mut x_k = w.m.clone();
            x_k.scaled_add(w.sigma, &bd_z);
            let cost = problem.cost(&x_k)?;
            state.candidates.push(x_k);
            state.costs.push(cost);
        }
        sort_population_ascending(&mut state.candidates, &mut state.costs);

        Ok((state, None))
    }

    fn terminate(&self, _state: &BasicPopulationState<V>) -> Option<TerminationReason> {
        let w = self.state.as_ref()?;
        // TolX (Hansen 2016 Appendix B.3): stop when the largest
        // standard deviation of any axis of the search distribution
        // drops below `tol_x`. `σ · max_i d_i` is the largest
        // single-axis standard deviation.
        let max_d = w.d_iter_max();
        if w.sigma * max_d < w.tol_x {
            return Some(TerminationReason::SolverConverged);
        }
        None
    }
}

impl<V, M> Working<V, M>
where
    V: std::ops::Index<usize, Output = f64> + VectorLen,
{
    fn d_iter_max(&self) -> f64 {
        let mut m = 0.0_f64;
        for i in 0..self.n {
            let v = self.d[i];
            if v > m {
                m = v;
            }
        }
        m
    }
}
