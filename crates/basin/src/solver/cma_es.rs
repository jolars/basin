use std::marker::PhantomData;

use crate::core::math::{
    ComponentMulAssign, MatTransposeVec, MatVec, MatrixFromDiagonal, MatrixIdentity, NormSquared,
    RankOneUpdate, SampleStandardNormal, Scalar, ScaleInPlace, ScaledAdd, SymmetricEigen,
    VectorLen,
};
use crate::core::problem::{CostFunction, Problem};
use crate::core::rng::{ChaCha8Rng, SeedableRng};
use crate::core::solver::Solver;
use crate::core::state::CmaEsState;
use crate::core::termination::TerminationReason;

/// `(µ/µ_W, λ)`-CMA Evolution Strategy with negative weights (aCMA-ES)
/// from Hansen 2016 (*The CMA Evolution Strategy: A Tutorial*).
///
/// Stochastic, derivative-free, population-based—the standard
/// black-box optimizer for ill-conditioned, non-separable, non-convex
/// continuous problems. Uses a multivariate normal `N(m, σ²C)` to
/// sample candidates, then adapts `m`, `σ`, and the covariance `C` from
/// the selected best `µ` candidates plus their conjugate evolution
/// path. Hansen 2016 Figure 6 / eqs (38)–(47) is the algorithm-summary
/// fixture; section A is the parameter table.
///
/// # Algorithm
///
/// The initial distribution (`m`, `σ`, and `C = I` or
/// `C = diag(stds²)`) is supplied by the caller via
/// [`CmaEsState::new(mean, sigma)`](crate::CmaEsState::new) (optionally
/// `.with_stds(stds)`). At [`init`](Solver::init) the solver computes
/// its derived constants, samples the first generation
/// `x_k = m + σ B (D ⊙ z_k)` with `z_k ~ N(0, I)`, and evaluates `f(m)`.
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
/// generations, Appendix B.2) is deferred—at small to moderate `n`
/// the cost is dominated by `f` evaluations anyway, and the refresh
/// frequency would change the per-iteration cost calculus.
///
/// # Result—mean vs best sample
///
/// CMA-ES's recommended solution is the distribution mean (pycma's
/// `xfavorite`), so [`State::param`](crate::State::param) /
/// [`State::cost`](crate::State::cost) on [`CmaEsState`] return `m` and
/// `f(m)`—the solver evaluates `f(m)` once per generation (so the
/// per-generation cost budget is `λ + 1`, not `λ`). The best evaluated
/// point ever seen (`xbest`) is available via
/// [`State::best_param`](crate::State::best_param) /
/// [`State::best_cost`](crate::State::best_cost), so an
/// [`OptimizationResult`](crate::core::executor::OptimizationResult)
/// surfaces both.
///
/// # Default parameters
///
/// All defaults follow Hansen 2016 Table 1 (the 2016 negative-weights
/// setting); see the per-field doc comments below for the exact
/// formulas. The user supplies the initial mean/step-size (via
/// [`CmaEsState`]) and the seed (via [`new`](Self::new)); `n` is the
/// mean's length and λ defaults to `4 + ⌊3 ln n⌋`.
///
/// # Reproducibility
///
/// The solver carries a [`ChaCha8Rng`] seeded from the `seed: u64`
/// passed to [`new`](Self::new)—same seed → same iterate trajectory
/// on every platform basin builds for (including
/// `wasm32-unknown-unknown`).
///
/// # Contract
///
/// - **Caller must:** implement [`CostFunction<Param = V, Output = f64>`]
///   on the problem. CMA-ES is derivative-free; no [`Gradient`](crate::Gradient) /
///   [`Jacobian`](crate::Jacobian) required.
/// - **Caller must:** hand in a
///   [`CmaEsState::new(mean, sigma)`](crate::CmaEsState::new) (optionally
///   `.with_stds(stds)`). The solver derives λ = `4 + ⌊3 ln n⌋` (override
///   via [`with_lambda`](Self::with_lambda); the default is exposed as
///   [`default_lambda`](Self::default_lambda)) and fills the first
///   generation in [`init`](Solver::init).
/// - **Implementor (this solver) must:** maintain the
///   [`PopulationState`](crate::core::state::PopulationState)
///   sorted-by-cost invariant on `state.candidates`/`state.costs`
///   at the start and end of every iteration.
///
/// # Termination
///
/// The canonical TolX test (`σ · max d_i < tol_x`, Hansen 2016 Appendix
/// B.3) is the framework criterion
/// [`CmaEsTolerance`](crate::core::termination::CmaEsTolerance), which
/// binds on [`CmaEsState`] and fires
/// [`TerminationReason::CmaEsTolerance`]. Register it on the
/// [`Executor`](crate::core::executor::Executor)—Hansen's recommended
/// value is `1e−12 · initial_sigma` (scale by `maxᵢ stdsᵢ` when an
/// anisotropic initial covariance is used). Pair with the framework's
/// [`MaxIter`](crate::core::termination::MaxIter) /
/// [`MaxCostEvals`](crate::core::termination::MaxCostEvals) for budget
/// control. Other CMA-ES termination heuristics (NoEffectAxis,
/// NoEffectCoord, ConditionCov, EqualFunValues, Stagnation, TolXUp,
/// TolFun) are out of scope for now.
///
/// # Backends
///
/// LA-heavy: requires symmetric eigendecomposition, scalar-and-rank-1
/// matrix updates, and matrix-vector/transposed matrix-vector
/// products. Wired and tested for the default `Vec<f64>` /
/// [`DenseMatrix`](crate::DenseMatrix) backend (pure-Rust cyclic Jacobi
/// eigensolver—no feature flag, `wasm`-clean), `nalgebra::DVector<f64>`
/// / `nalgebra::DMatrix<f64>` (feature `nalgebra`), `ndarray::Array1<f64>`
/// / `ndarray::Array2<f64>` (feature `ndarray`, also wired to the cyclic
/// Jacobi solver—`wasm`-clean), and `faer::Col<f64>`/`faer::Mat<f64>`
/// (feature `faer`). Sparse covariance is not meaningful for CMA-ES—the
/// rank-µ update densifies any starting pattern.
///
/// # Examples
///
/// See [`RandomSearch`](crate::RandomSearch) for the population-based
/// `Executor` pattern. Construct the solver with `CmaEs::new(seed)` and
/// the initial distribution with `CmaEsState::new(mean, sigma)`.
pub struct CmaEs<V, M, F = f64> {
    lambda_override: Option<usize>,
    /// Derived CMA constants, computed once at [`Solver::init`] from the
    /// state's dimension. Cached on the solver (config-only) rather than
    /// in the state; persists across `run_loop` re-entry so a resumed
    /// solver skips recomputation.
    constants: Option<CmaConstants<F>>,
    rng: ChaCha8Rng,
    _marker: PhantomData<(V, M)>,
}

/// Derived CMA-ES constants (Hansen 2016 Table 1), computed once at
/// [`Solver::init`] from `n` and `λ`. Pure functions of the
/// hyperparameters—no mutable iterate (that lives in
/// [`CmaEsState`]).
pub(crate) struct CmaConstants<F = f64> {
    pub(crate) n: usize,
    pub(crate) lambda: usize,
    pub(crate) mu: usize,
    /// All λ recombination weights (sum of positives = 1; negatives
    /// scaled per Hansen Table 1 rows (50)–(53)).
    pub(crate) weights: Vec<F>,
    /// `µ_eff = (Σ_{i=1..µ} w_i)² / Σ_{i=1..µ} w_i² = 1 / Σ w_i²`
    /// because the positive weights sum to 1.
    pub(crate) mu_eff: F,
    /// `Σ_{i=1..λ} w_i`. Negative when negative weights are in use
    /// (default setting); the C-update scalar `1 − c_µ · sum_w`
    /// inflates rather than decays C as a result. With Hansen's
    /// `α_µ_minus = 1 + c_1/c_µ` choice, `c_1 + c_µ · sum_w ≈ 0`,
    /// so the C scalar is approximately 1 (eq. 47).
    pub(crate) sum_w: F,
    pub(crate) c_sigma: F,
    pub(crate) d_sigma: F,
    pub(crate) c_c: F,
    pub(crate) c_1: F,
    pub(crate) c_mu: F,
    pub(crate) expected_norm: F,
    /// `(1.4 + 2/(n+1)) · E‖N(0,I)‖`—RHS of the h_σ test (eq. 47
    /// callout footnote / Hansen 2016 p. 31).
    pub(crate) h_sigma_threshold: F,
}

impl<V, M, F: Scalar> CmaEs<V, M, F> {
    /// Build a CMA-ES with the default population size
    /// `λ = 4 + ⌊3 ln n⌋` (Hansen 2016 eq. 48) and a seeded RNG. The
    /// initial mean, step-size, and (optional) per-coordinate stds are
    /// supplied via [`CmaEsState`]; TolX is the
    /// [`CmaEsTolerance`](crate::core::termination::CmaEsTolerance)
    /// criterion.
    pub fn new(seed: u64) -> Self {
        Self {
            lambda_override: None,
            constants: None,
            rng: ChaCha8Rng::seed_from_u64(seed),
            _marker: PhantomData,
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

    /// Default population size for dimension `n`: `4 + ⌊3 ln n⌋`
    /// (Hansen 2016 eq. 48). Exposed so callers can match the solver's
    /// internal default without re-deriving the formula.
    pub fn default_lambda(n: usize) -> usize {
        4 + (3.0 * (n as f64).ln()).floor() as usize
    }
}

/// Asymptotic expansion of `E‖N(0, I_n)‖ = √2 Γ((n+1)/2) / Γ(n/2)`.
/// Accurate to ~10 digits for `n ≥ 1`; avoids needing `lgamma` (which
/// is not in stable `std`).
pub(crate) fn expected_norm_n01<F: Scalar>(n: usize) -> F {
    let n_f = F::from_usize(n).unwrap();
    let one = F::one();
    let four = F::from_f64(4.0).unwrap();
    let twenty_one = F::from_f64(21.0).unwrap();
    n_f.sqrt() * (one - one / (four * n_f) + one / (twenty_one * n_f * n_f))
}

/// Compute the recombination weights and derived constants per
/// Hansen 2016 Table 1 rows (49)–(53), plus `µ_eff` and `µ_eff_neg`.
/// Returns `(weights, mu_eff, sum_w)`.
pub(crate) fn compute_weights<F: Scalar>(
    n: usize,
    lambda: usize,
    c_1: F,
    c_mu: F,
) -> (Vec<F>, F, F) {
    let mu = lambda / 2;
    let one = F::one();
    let two = F::from_f64(2.0).unwrap();
    let zero = F::zero();
    let lambda_f = F::from_usize(lambda).unwrap();
    // Raw preliminary weights w_i' = ln((λ+1)/2) − ln i (eq. 49).
    let raw: Vec<F> = (1..=lambda)
        .map(|i| ((lambda_f + one) / two).ln() - F::from_usize(i).unwrap().ln())
        .collect();

    // Positive sum and negative sum (over raw values).
    let sum_pos: F = raw[..mu].iter().copied().sum();
    // µ_eff is defined on the *positive* weights only and is invariant
    // under positive-rescaling, so compute it from raw[..mu] (eq. 8 /
    // Table 1 caption).
    let raw_pos_norm_sq: F = raw[..mu].iter().map(|w| *w * *w).sum();
    let mu_eff = sum_pos * sum_pos / raw_pos_norm_sq;

    // µ_eff_neg from negative-portion raws (Table 1 caption).
    let sum_neg: F = raw[mu..].iter().copied().sum();
    let raw_neg_norm_sq: F = raw[mu..].iter().map(|w| *w * *w).sum();
    let mu_eff_neg = if raw_neg_norm_sq > zero {
        sum_neg * sum_neg / raw_neg_norm_sq
    } else {
        zero
    };

    // Three bounds on the negative-weight scale (eqs. 50–52).
    let alpha_mu_minus = one + c_1 / c_mu;
    let alpha_mu_eff_minus = one + two * mu_eff_neg / (mu_eff + two);
    let alpha_pos_def_minus = (one - c_1 - c_mu) / (F::from_usize(n).unwrap() * c_mu);
    let alpha_neg = alpha_mu_minus
        .min(alpha_mu_eff_minus)
        .min(alpha_pos_def_minus);

    // Final weights (eq. 53):
    // - positive: w_i = w_i' / Σ|w_j'|+ (positives sum to 1).
    // - negative: w_i = (alpha_neg / Σ|w_j'|−) · w_i'.
    let sum_abs_neg: F = raw[mu..].iter().map(|w| -*w).sum();
    let mut weights = Vec::with_capacity(lambda);
    for (i, &raw_i) in raw.iter().enumerate() {
        let w = if i < mu {
            raw_i / sum_pos
        } else if sum_abs_neg > zero {
            alpha_neg * raw_i / sum_abs_neg
        } else {
            zero
        };
        weights.push(w);
    }

    let sum_w: F = weights.iter().copied().sum();
    (weights, mu_eff, sum_w)
}

/// Compute the derived CMA-ES constants (Hansen 2016 Table 1) for
/// dimension `n` and population size `lambda`. Shared by [`CmaEs`] and
/// [`BoundedCmaEs`](crate::solver::BoundedCmaEs)'s init.
pub(crate) fn compute_constants<F: Scalar>(n: usize, lambda: usize) -> CmaConstants<F> {
    let mu = lambda / 2;
    let one = F::one();
    let two = F::from_f64(2.0).unwrap();
    let zero = F::zero();
    let n_f = F::from_usize(n).unwrap();
    let lambda_f = F::from_usize(lambda).unwrap();
    // Hansen Table 1 rows (55)–(58).
    let alpha_cov = two;
    // The c_1 / c_µ formulas need µ_eff, which depends on positive
    // weights only. Compute µ_eff once from the raw weights to feed
    // c_1 / c_µ, then re-derive the final negative weights against
    // those c_1 / c_µ via `compute_weights` (Hansen explains the
    // apparent circular dependency in Appendix A: µ_eff is invariant
    // under positive-weight rescaling, so a one-shot computation
    // suffices).
    let raw: Vec<F> = (1..=lambda)
        .map(|i| ((lambda_f + one) / two).ln() - F::from_usize(i).unwrap().ln())
        .collect();
    let sum_pos: F = raw[..mu].iter().copied().sum();
    let mu_eff_provisional = sum_pos * sum_pos / raw[..mu].iter().map(|w| *w * *w).sum::<F>();

    let c_1 = alpha_cov
        / ((n_f + F::from_f64(1.3).unwrap()) * (n_f + F::from_f64(1.3).unwrap())
            + mu_eff_provisional);
    let c_mu_unbounded = alpha_cov * (mu_eff_provisional - two + one / mu_eff_provisional)
        / ((n_f + two) * (n_f + two) + alpha_cov * mu_eff_provisional / two);
    let c_mu = (one - c_1).min(c_mu_unbounded);

    let (weights, mu_eff, sum_w) = compute_weights::<F>(n, lambda, c_1, c_mu);

    let c_sigma = (mu_eff + two) / (n_f + mu_eff + F::from_f64(5.0).unwrap());
    // d_σ = 1 + 2 · max(0, √((µ_eff−1)/(n+1)) − 1) + c_σ
    // (Hansen 2016 Table 1 row 55).
    let d_sigma = {
        let inner = ((mu_eff - one) / (n_f + one)).sqrt() - one;
        one + two * inner.max(zero) + c_sigma
    };
    let c_c = (F::from_f64(4.0).unwrap() + mu_eff / n_f)
        / (n_f + F::from_f64(4.0).unwrap() + two * mu_eff / n_f);

    let expected_norm = expected_norm_n01::<F>(n);
    let h_sigma_threshold = (F::from_f64(1.4).unwrap() + two / (n_f + one)) * expected_norm;

    CmaConstants {
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
    }
}

/// Sample a fresh generation `x_k = m + σ B (D ⊙ z_k)` with
/// `z_k ~ N(0, I)` into `state.candidates` (cleared first). The
/// isotropic default (`B = I`, `D = 1`) reduces to `m + σ z_k`
/// bit-identically, so the single general path is used throughout.
/// Shared by [`CmaEs`] init and `next_iter`.
pub(crate) fn sample_generation<V, M, F>(
    state: &mut CmaEsState<V, M, F>,
    lambda: usize,
    rng: &mut ChaCha8Rng,
) where
    F: Scalar,
    V: VectorLen + Clone + ScaledAdd<F> + ComponentMulAssign + SampleStandardNormal,
    M: MatVec<V>,
{
    state.candidates.clear();
    for _ in 0..lambda {
        let z_k = V::sample_standard_normal(&state.m, rng);
        let mut bd_z = z_k;
        bd_z.component_mul_assign(&state.d);
        let bd_z = state.b.matvec(&bd_z);
        let mut x_k = state.m.clone();
        x_k.scaled_add(state.sigma, &bd_z);
        state.candidates.push(x_k);
    }
}

/// Sort `candidates` and `costs` jointly by ascending cost. NaN costs
/// sort last (mirrors `nelder_mead::sort_simplex` /
/// `random_search::sort_population_ascending`).
pub(crate) fn sort_population_ascending<V, F: PartialOrd>(candidates: &mut [V], costs: &mut [F]) {
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

impl<P, V, M, F> Solver<P, CmaEsState<V, M, F>> for CmaEs<V, M, F>
where
    F: Scalar + crate::core::parallel::MaybeSend,
    P: CostFunction<Param = V, Output = F> + crate::core::parallel::MaybeSync,
    P::Error: crate::core::parallel::MaybeSend,
    V: VectorLen
        + Clone
        + ScaledAdd<F>
        + ScaleInPlace<F>
        + ComponentMulAssign
        + NormSquared<F>
        + SampleStandardNormal
        + crate::core::parallel::MaybeSync
        + std::ops::Index<usize, Output = F>
        + std::ops::IndexMut<usize, Output = F>,
    M: MatrixIdentity
        + MatrixFromDiagonal<V>
        + MatVec<V>
        + MatTransposeVec<V>
        + ScaleInPlace<F>
        + RankOneUpdate<V, F>
        + SymmetricEigen<V>
        + Clone,
{
    type Error = P::Error;

    fn init(
        &mut self,
        problem: &mut Problem<P>,
        mut state: CmaEsState<V, M, F>,
    ) -> Result<CmaEsState<V, M, F>, Self::Error> {
        // Compute-once constants guard (cached on the solver—config
        // only). A resumed solver re-entered via `run_loop` already has
        // them, so a chain-paused CmaEs is not rebuilt on every entry.
        if self.constants.is_none() {
            let n = state.m.vec_len();
            assert!(n >= 1, "CmaEs requires a non-empty mean");
            let lambda = self
                .lambda_override
                .unwrap_or_else(|| Self::default_lambda(n));
            self.constants = Some(compute_constants::<F>(n, lambda));
        }
        let lambda = self.constants.as_ref().unwrap().lambda;

        // First generation: an empty population signals a fresh state and
        // is sampled now; a resumed / chain state arrives with a
        // populated, sorted population (and a valid `m_cost`) and is left
        // untouched. The distribution itself (`m`, `σ`, `C`, paths, `d`)
        // was fully seeded by `CmaEsState::new`/`with_stds`.
        if state.candidates.is_empty() {
            sample_generation(&mut state, lambda, &mut self.rng);
            state.costs = problem.cost_batch(&state.candidates)?;
            sort_population_ascending(&mut state.candidates, &mut state.costs);
            // Evaluate the mean—param()/cost() report `m` (xfavorite).
            let m_cost = problem.cost(&state.m)?;
            state.m_cost = Some(m_cost);
        }
        Ok(state)
    }

    fn next_iter(
        &mut self,
        problem: &mut Problem<P>,
        mut state: CmaEsState<V, M, F>,
    ) -> Result<(CmaEsState<V, M, F>, Option<TerminationReason>), Self::Error> {
        let k = self
            .constants
            .as_ref()
            .expect("CmaEs::init must run before next_iter");

        state.generation += 1;

        let one = F::one();
        let two = F::from_f64(2.0).unwrap();
        let zero = F::zero();

        // Rebuild y_{i:λ} = (x_{i:λ} − m) / σ for the *previous* m, σ.
        // (state.candidates carries the most recent generation's x's,
        // sorted ascending by cost.)
        let mut y_sorted: Vec<V> = state
            .candidates
            .iter()
            .map(|x| {
                let mut y = x.clone();
                y.scaled_add(-one, &state.m);
                y.scale_in_place(one / state.sigma);
                y
            })
            .collect();

        // ⟨y⟩_w = Σ_{i=1..µ} w_i y_{i:λ}.
        let mut y_w = state.m.clone();
        y_w.scale_in_place(zero);
        for (i, y_i) in y_sorted.iter().enumerate().take(k.mu) {
            y_w.scaled_add(k.weights[i], y_i);
        }

        // m ← m + σ ⟨y⟩_w (c_m = 1 by default).
        state.m.scaled_add(state.sigma, &y_w);

        // C^{−1/2} ⟨y⟩_w = B (D^{−1} ⊙ Bᵀ ⟨y⟩_w).
        let mut bt_y_w = state.b.mat_transpose_vec(&y_w);
        bt_y_w.component_mul_assign(&state.d_inv);
        let c_invsqrt_y_w = state.b.matvec(&bt_y_w);

        // p_σ ← (1 − c_σ) p_σ + √(c_σ(2 − c_σ) µ_eff) C^{−1/2} ⟨y⟩_w.
        state.p_sigma.scale_in_place(one - k.c_sigma);
        let coef_sigma = (k.c_sigma * (two - k.c_sigma) * k.mu_eff).sqrt();
        state.p_sigma.scaled_add(coef_sigma, &c_invsqrt_y_w);

        // σ ← σ exp((c_σ / d_σ) (‖p_σ‖ / E‖N(0,I)‖ − 1)).
        let p_sigma_norm = state.p_sigma.norm_squared().sqrt();
        let log_factor = (k.c_sigma / k.d_sigma) * (p_sigma_norm / k.expected_norm - one);
        state.sigma = state.sigma * log_factor.exp();

        // h_σ test (Hansen 2016 p. 31, denominator uses 2(g+1)).
        let g_for_h = (state.generation + 1) as i32;
        let exponent = 2 * g_for_h;
        let denom = (one - (one - k.c_sigma).powi(exponent)).sqrt();
        let h_sigma = if p_sigma_norm / denom < k.h_sigma_threshold {
            one
        } else {
            zero
        };

        // p_c ← (1 − c_c) p_c + h_σ √(c_c(2 − c_c) µ_eff) ⟨y⟩_w.
        state.p_c.scale_in_place(one - k.c_c);
        let coef_c = h_sigma * (k.c_c * (two - k.c_c) * k.mu_eff).sqrt();
        state.p_c.scaled_add(coef_c, &y_w);

        // C update (eq. 47):
        //   C ← (1 + c_1 δ_h − c_1 − c_µ Σ w_j) C
        //       + c_1 p_c p_cᵀ
        //       + c_µ Σ_i w_i° y_{i:λ} y_{i:λ}ᵀ
        // with w_i° = w_i for w_i ≥ 0, else w_i · n / ‖C^{−1/2} y_{i:λ}‖².
        let delta_h = (one - h_sigma) * k.c_c * (two - k.c_c);
        let c_scale = one + k.c_1 * delta_h - k.c_1 - k.c_mu * k.sum_w;
        state.c.scale_in_place(c_scale);
        state.c.rank_one_update(k.c_1, &state.p_c);
        // Negative-weight path rescales by n / ‖C^{−1/2} y_i‖²;
        // positive-weight path uses w_i directly (eq. 46).
        let n_f = F::from_usize(k.n).unwrap();
        for (i, y_i) in y_sorted.iter().enumerate() {
            let w_i = k.weights[i];
            let w_i_o = if w_i >= zero {
                w_i
            } else {
                // ‖C^{−1/2} y_i‖² = ‖D^{−1} ⊙ Bᵀ y_i‖² (orthogonal B).
                let mut bt_y = state.b.mat_transpose_vec(y_i);
                bt_y.component_mul_assign(&state.d_inv);
                let cinv_norm_sq = bt_y.norm_squared();
                if cinv_norm_sq > zero {
                    w_i * n_f / cinv_norm_sq
                } else {
                    // Pathological zero-direction; drop this contribution.
                    zero
                }
            };
            if w_i_o != zero {
                state.c.rank_one_update(k.c_mu * w_i_o, y_i);
            }
        }
        // Drop y_sorted now to free memory before the eigendecomposition.
        drop(std::mem::take(&mut y_sorted));

        // Refresh eigendecomposition of the new C.
        let (b_new, eigs) = match state.c.try_eigh() {
            Ok(pair) => pair,
            Err(_) => return Ok((state, Some(TerminationReason::SolverFailed))),
        };
        state.b = b_new;
        // d_i = √max(λ_i, 0); d_inv_i = 1/d_i. Floating-point can produce
        // tiny negative eigenvalues even when the algorithm preserves
        // positive definiteness; clamp to a small positive floor before
        // taking the square root.
        let eig_floor = F::from_f64(1e-30).unwrap();
        for i in 0..k.n {
            let lam = eigs[i].max(eig_floor);
            let s = lam.sqrt();
            state.d[i] = s;
            state.d_inv[i] = one / s;
        }

        // Sample new generation: x_k = m + σ B (D ⊙ z_k). Sampling is
        // sequential (the RNG draws define the reproducible trajectory);
        // the λ independent candidates then evaluate in one batch
        // (parallel under the `parallel` feature).
        let lambda = k.lambda;
        sample_generation(&mut state, lambda, &mut self.rng);
        state.costs = problem.cost_batch(&state.candidates)?;
        sort_population_ascending(&mut state.candidates, &mut state.costs);

        // Evaluate the mean so param()/cost() report `m` (xfavorite).
        let m_cost = problem.cost(&state.m)?;
        state.m_cost = Some(m_cost);

        Ok((state, None))
    }
}
