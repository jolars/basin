use std::collections::VecDeque;
use std::marker::PhantomData;

use crate::core::constraint::BoxConstraints;
use crate::core::math::{
    ClampInPlace, ComponentMulAssign, MatDiagonal, MatTransposeVec, MatVec, MatrixFromDiagonal,
    MatrixIdentity, NormSquared, RankOneUpdate, SampleStandardNormal, Scalar, ScaleInPlace,
    ScaledAdd, SymmetricEigen, VectorLen,
};
use crate::core::problem::{CostFunction, Problem};
use crate::core::rng::{ChaCha8Rng, SeedableRng};
use crate::core::solver::Solver;
use crate::core::state::CmaEsState;
use crate::core::state::cma_es::BoundPenalty;
use crate::core::termination::TerminationReason;

use super::cma_es::{CmaConstants, compute_constants, sort_population_ascending};

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
///   the problem (tenet 4 in `CONTRIBUTING.md`); handing this solver a
///   problem without `BoxConstraints` is a compile-time error.
/// - **Caller must:** ensure `lower[i] ≤ upper[i]` for every component
///   ([`f64::clamp`] panics otherwise) and `sigma > 0`.
/// - **Caller must:** hand in a
///   [`CmaEsState::new(mean, sigma)`](crate::CmaEsState::new) (optionally
///   `.with_stds(stds)`). The default `λ = 4 + ⌊3 ln n⌋` is exposed via
///   [`default_lambda`](Self::default_lambda); the penalty bookkeeping
///   is installed into the state by [`init`](Solver::init).
/// - **Implementor (this solver) must:** maintain the
///   [`PopulationState`](crate::core::state::PopulationState)
///   sorted-by-cost invariant on `state.candidates` / `state.costs`,
///   where `state.costs` carries the **penalized** fitness values
///   (raw fitness is held in `state.penalty` for the γ-update IQR). The
///   initial mean is projected onto `[lower, upper]` once at iter 0 so
///   the iter-0 search distribution is centered in feasibility.
///
/// # Result and termination
///
/// As [`CmaEs`](super::cma_es::CmaEs):
/// [`State::param`](crate::State::param) returns the mean (penalized
/// `f(m)` for [`State::cost`](crate::State::cost)),
/// [`State::best_param`](crate::State::best_param) the best evaluated
/// point. TolX is the
/// [`CmaEsTolerance`](crate::core::termination::CmaEsTolerance)
/// criterion (`σ · max_i d_i < tol_x`). Bounded-CMA-ES adds no new
/// termination criteria of its own; feasibility is enforced sample-wise
/// by construction (every evaluated point is inside the box because we
/// evaluate at `clamp(x_k, lower, upper)`), and the framework's
/// [`MaxIter`](crate::core::termination::MaxIter) /
/// [`MaxCostEvals`](crate::core::termination::MaxCostEvals) work
/// against [`CmaEsState`] without modification.
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
/// **plus** `MatDiagonal<V>` (extracts `diag(C)` for the σ²·diag(C)
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
pub struct BoundedCmaEs<V, M, F = f64> {
    lambda_override: Option<usize>,
    /// Derived constants (CMA + BoundPenalty), computed once at
    /// [`Solver::init`]. Config-only, cached on the solver.
    constants: Option<BoundedCmaConstants<F>>,
    rng: ChaCha8Rng,
    _marker: PhantomData<(V, M)>,
}

/// Derived constants for bounded CMA-ES: the shared CMA constants plus
/// the three BoundPenalty constants (`damp`, `edist_threshold`,
/// `hist_cap`). Computed once at [`Solver::init`]; the mutable penalty
/// bookkeeping lives in the [`CmaEsState`]'s `penalty` field.
pub(crate) struct BoundedCmaConstants<F = f64> {
    cma: CmaConstants<F>,
    /// `min(1, mu_eff / (10·n))`. Damping factor on the γ multiplicative
    /// update; pycma `boundary_handler.py:716`.
    damp: F,
    /// `3 · max(1, sqrt(n) / mu_eff)`. The σ-unit slack threshold
    /// before γ_i is raised on coordinate `i`; pycma `boundary_handler.py:730`.
    edist_threshold: F,
    /// Cap on the `hist` deque length: `20 + ⌊3n/λ⌋`. Pycma
    /// `boundary_handler.py:711`.
    hist_cap: usize,
}

impl<V, M, F: Scalar> BoundedCmaEs<V, M, F> {
    /// Build a bounded CMA-ES with the default population size
    /// `λ = 4 + ⌊3 ln n⌋` (Hansen 2016 eq. 48) and a seeded RNG. The
    /// initial mean / step-size / stds are supplied via [`CmaEsState`];
    /// TolX is the
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

    /// Default population size for dimension `n`: `4 + ⌊3 ln n⌋`
    /// (Hansen 2016 eq. 48). Same formula as
    /// [`CmaEs::default_lambda`](super::cma_es::CmaEs::default_lambda).
    pub fn default_lambda(n: usize) -> usize {
        4 + (3.0 * (n as f64).ln()).floor() as usize
    }
}

/// Compute the derived bounded-CMA constants (shared CMA constants plus
/// the BoundPenalty constants) for dimension `n` and population `lambda`.
fn compute_bounded_constants<F: Scalar>(n: usize, lambda: usize) -> BoundedCmaConstants<F> {
    let cma = compute_constants::<F>(n, lambda);
    let one = F::one();
    let n_f = F::from_usize(n).unwrap();
    let three = F::from_f64(3.0).unwrap();
    let ten = F::from_f64(10.0).unwrap();

    // BoundPenalty constants: damp, edist threshold, hist cap.
    let damp = (cma.mu_eff / (ten * n_f)).min(one);
    // Pycma uses `mueff` (not `max(1, mueff)`) in the denominator; we
    // mirror but defensively floor to avoid div-by-zero on pathological
    // `lambda` (mu_eff is always > 0 for lambda >= 4 with the default
    // weights).
    let edist_threshold =
        three * n_f.sqrt().max(one) / cma.mu_eff.max(F::from_f64(f64::MIN_POSITIVE).unwrap());
    let hist_cap = 20 + (3 * n) / lambda;

    BoundedCmaConstants {
        cma,
        damp,
        edist_threshold,
        hist_cap,
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
pub(crate) fn evaluate_with_penalty<P, V, F>(
    problem: &mut Problem<P>,
    x: &V,
    lower: &V,
    upper: &V,
    gamma: &V,
    n: usize,
) -> Result<(F, F), P::Error>
where
    F: Scalar,
    P: CostFunction<Param = V, Output = F>,
    V: Clone + ClampInPlace + std::ops::Index<usize, Output = F>,
{
    let mut x_rep = x.clone();
    x_rep.clamp_in_place(lower, upper);
    let raw = problem.cost(&x_rep)?;
    let mut penalty = F::zero();
    for i in 0..n {
        let dx = x[i] - x_rep[i];
        penalty = penalty + gamma[i] * dx * dx;
    }
    penalty = penalty / F::from_usize(n).unwrap();
    Ok((raw, raw + penalty))
}

/// γ adaptation. Mirrors pycma `BoundPenalty.update`
/// (`boundary_handler.py:669`–`749`). Reads the previous generation's
/// raw fitness and `γ`/`hist`/`weights_initialized` from
/// `state.penalty`, the current mean / σ / C from `state`, the penalty
/// constants from `k`, and the bounds from `problem`; writes back into
/// `state.penalty`.
fn update_gamma<P, V, M, F>(
    state: &mut CmaEsState<V, M, F>,
    k: &BoundedCmaConstants<F>,
    problem: &P,
) where
    F: Scalar,
    P: BoxConstraints<Param = V>,
    V: Clone
        + ClampInPlace
        + std::ops::Index<usize, Output = F>
        + std::ops::IndexMut<usize, Output = F>,
    M: MatDiagonal<V>,
{
    let n = k.cma.n;
    let pen = state
        .penalty
        .as_mut()
        .expect("BoundedCmaEs::init installs the penalty before next_iter");
    if pen.raw_costs.is_empty() {
        return;
    }

    let zero = F::zero();
    let two = F::from_f64(2.0).unwrap();
    let three = F::from_f64(3.0).unwrap();
    let five = F::from_f64(5.0).unwrap();
    let n_f = F::from_usize(n).unwrap();

    // varis[i] = σ² · diag(C)[i]. Per-axis variance of N(m, σ²C).
    let diag_c = state.c.diagonal();
    let mut mean_varis = zero;
    for i in 0..n {
        mean_varis = mean_varis + state.sigma * state.sigma * diag_c[i];
    }
    mean_varis = mean_varis / n_f;

    // dmean[i] = (m[i] − clamp(m)[i]) / sqrt(varis[i]). Mean violation
    // in σ-units along axis i; zero if the mean is feasible on i.
    let mut m_rep = state.m.clone();
    m_rep.clamp_in_place(problem.lower(), problem.upper());
    let mut dmean: Vec<F> = Vec::with_capacity(n);
    let mut any_violation = false;
    for i in 0..n {
        let var_i = state.sigma * state.sigma * diag_c[i];
        let d = (state.m[i] - m_rep[i]) / var_i.sqrt();
        if d != zero {
            any_violation = true;
        }
        dmean.push(d);
    }

    // Fitness IQR (pycma's offset definition: indices 3l/4 and l/4 with
    // l = 1 + λ, no interpolation), normalized by mean per-axis variance.
    let mut sorted = pen.raw_costs.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let l = 1 + sorted.len();
    let val = (sorted[3 * l / 4] - sorted[l / 4]) / mean_varis;

    // Push to hist (front), trim to hist_cap.
    if val.is_finite() && val > zero {
        pen.hist.push_front(val);
    } else if val == F::infinity() && !pen.hist.is_empty() {
        let max_hist = pen
            .hist
            .iter()
            .copied()
            .fold(F::neg_infinity(), |a, b| if b > a { b } else { a });
        pen.hist.push_front(max_hist);
    }
    while pen.hist.len() > k.hist_cap {
        pen.hist.pop_back();
    }
    if pen.hist.is_empty() {
        return;
    }

    // dfit = median(hist).
    let mut hsorted: Vec<F> = pen.hist.iter().copied().collect();
    hsorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let dfit = hsorted[hsorted.len() / 2];

    // Initialize γ on the first generation that sees an infeasible mean.
    // (We skip pycma's `countiter == 2` re-init path — see
    // references/pycma-bound-handling/NOTES.md "Implementation deltas".)
    if any_violation && !pen.weights_initialized {
        let init_val = two * dfit;
        for i in 0..n {
            pen.gamma[i] = init_val;
        }
        pen.weights_initialized = true;
    }

    // Update γ each generation once initialized:
    // - raise γ_i where |dmean_i| − edist_threshold > 0;
    // - decay γ entries that exceed 5·dfit.
    // pycma's active branch (`if 1 < 3:` at boundary_handler.py:731);
    // the elif/else legacy branches are dead code.
    if pen.weights_initialized {
        for (i, dmean_i) in dmean.iter().enumerate() {
            let edist_i = dmean_i.abs() - k.edist_threshold;
            if edist_i > zero {
                let factor = ((edist_i / three).tanh() / two * k.damp).exp();
                pen.gamma[i] = pen.gamma[i] * factor;
            }
        }
        let cap = five * dfit;
        let decay = (-k.damp / three).exp();
        for i in 0..n {
            if pen.gamma[i] > cap {
                pen.gamma[i] = pen.gamma[i] * decay;
            }
        }
    }
}

/// Sample a fresh generation into `state.candidates` (cleared first),
/// evaluating each at its repaired (clamped) point and recording the
/// penalized cost in `state.costs` and the raw cost in
/// `state.penalty.raw_costs`. The penalty must already be installed.
#[allow(clippy::too_many_arguments)]
fn sample_and_penalize<P, V, M, F>(
    state: &mut CmaEsState<V, M, F>,
    lambda: usize,
    n: usize,
    rng: &mut ChaCha8Rng,
    problem: &mut Problem<P>,
    lo: &V,
    hi: &V,
) -> Result<(), P::Error>
where
    F: Scalar,
    P: CostFunction<Param = V, Output = F>,
    V: VectorLen
        + Clone
        + ScaledAdd<F>
        + ComponentMulAssign
        + ClampInPlace
        + SampleStandardNormal
        + std::ops::Index<usize, Output = F>,
    M: MatVec<V>,
{
    let CmaEsState {
        candidates,
        costs,
        m,
        d,
        b,
        sigma,
        penalty,
        ..
    } = state;
    let pen = penalty
        .as_mut()
        .expect("BoundedCmaEs::init installs the penalty before sampling");
    candidates.clear();
    costs.clear();
    pen.raw_costs.clear();
    for _ in 0..lambda {
        let z_k = V::sample_standard_normal(m, rng);
        let mut bd_z = z_k;
        bd_z.component_mul_assign(d);
        let bd_z = b.matvec(&bd_z);
        let mut x_k = m.clone();
        x_k.scaled_add(*sigma, &bd_z);
        let (raw, p) = evaluate_with_penalty(problem, &x_k, lo, hi, &pen.gamma, n)?;
        candidates.push(x_k);
        costs.push(p);
        pen.raw_costs.push(raw);
    }
    Ok(())
}

impl<P, V, M, F> Solver<P, CmaEsState<V, M, F>> for BoundedCmaEs<V, M, F>
where
    F: Scalar,
    P: CostFunction<Param = V, Output = F> + BoxConstraints,
    V: VectorLen
        + Clone
        + ScaledAdd<F>
        + ScaleInPlace<F>
        + ComponentMulAssign
        + ClampInPlace
        + NormSquared<F>
        + SampleStandardNormal
        + std::ops::Index<usize, Output = F>
        + std::ops::IndexMut<usize, Output = F>,
    M: MatrixIdentity
        + MatrixFromDiagonal<V>
        + MatVec<V>
        + MatTransposeVec<V>
        + MatDiagonal<V>
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
        // Constants compute-once guard (cached on the solver — config
        // only; persists across `run_loop` re-entry for chain resumption).
        if self.constants.is_none() {
            let n = state.m.vec_len();
            assert!(n >= 1, "BoundedCmaEs requires a non-empty mean");
            let lambda = self
                .lambda_override
                .unwrap_or_else(|| Self::default_lambda(n));
            self.constants = Some(compute_bounded_constants::<F>(n, lambda));
        }
        let n = self.constants.as_ref().unwrap().cma.n;
        let lambda = self.constants.as_ref().unwrap().cma.lambda;

        // Install the penalty bookkeeping into the state (the bounded
        // analog of `Lbfgsb::init` installing `work`). γ starts all-ones
        // (pycma's scalar-1 default, materialized as a vector).
        if state.penalty.is_none() {
            let mut gamma = state.m.clone();
            for i in 0..n {
                gamma[i] = F::one();
            }
            state.penalty = Some(BoundPenalty {
                gamma,
                weights_initialized: false,
                hist: VecDeque::new(),
                raw_costs: Vec::with_capacity(lambda),
            });
        }

        // First generation: an empty population signals a fresh state. A
        // resumed / chain state keeps its population, distribution, and
        // penalty bookkeeping untouched.
        if state.candidates.is_empty() {
            // Project an infeasible initial mean once at iter 0 so the
            // iter-0 search distribution is centered in feasibility.
            // Mirrors ProjectedGradientDescent::init's iter-0 projection.
            let lo = problem.inner().lower().clone();
            let hi = problem.inner().upper().clone();
            state.m.clamp_in_place(&lo, &hi);

            sample_and_penalize(&mut state, lambda, n, &mut self.rng, problem, &lo, &hi)?;
            sort_population_ascending(&mut state.candidates, &mut state.costs);

            // Evaluate the mean (penalized, consistent with samples) so
            // param()/cost() report `m` (xfavorite).
            let gamma = &state.penalty.as_ref().unwrap().gamma;
            let (_raw, pen_m) = evaluate_with_penalty(problem, &state.m, &lo, &hi, gamma, n)?;
            state.m_cost = Some(pen_m);
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
            .expect("BoundedCmaEs::init must run before next_iter");
        let kc = &k.cma;

        state.generation += 1;

        let one = F::one();
        let two = F::from_f64(2.0).unwrap();
        let zero = F::zero();

        // Recombination uses the un-repaired samples. y_{i:λ} = (x_{i:λ} − m) / σ
        // for the *previous* m, σ. (state.candidates carries the most recent
        // generation's x's, sorted ascending by *penalized* cost — for
        // recombination only the rank order matters.)
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
        for (i, y_i) in y_sorted.iter().enumerate().take(kc.mu) {
            y_w.scaled_add(kc.weights[i], y_i);
        }

        // m ← m + σ ⟨y⟩_w.
        state.m.scaled_add(state.sigma, &y_w);

        // C^{−1/2} ⟨y⟩_w = B (D^{−1} ⊙ Bᵀ ⟨y⟩_w).
        let mut bt_y_w = state.b.mat_transpose_vec(&y_w);
        bt_y_w.component_mul_assign(&state.d_inv);
        let c_invsqrt_y_w = state.b.matvec(&bt_y_w);

        // p_σ ← (1 − c_σ) p_σ + √(c_σ(2 − c_σ) µ_eff) C^{−1/2} ⟨y⟩_w.
        state.p_sigma.scale_in_place(one - kc.c_sigma);
        let coef_sigma = (kc.c_sigma * (two - kc.c_sigma) * kc.mu_eff).sqrt();
        state.p_sigma.scaled_add(coef_sigma, &c_invsqrt_y_w);

        // σ ← σ exp((c_σ / d_σ) (‖p_σ‖ / E‖N(0,I)‖ − 1)).
        let p_sigma_norm = state.p_sigma.norm_squared().sqrt();
        let log_factor = (kc.c_sigma / kc.d_sigma) * (p_sigma_norm / kc.expected_norm - one);
        state.sigma = state.sigma * log_factor.exp();

        // h_σ test (Hansen 2016 p. 31, denominator uses 2(g+1)).
        let g_for_h = (state.generation + 1) as i32;
        let exponent = 2 * g_for_h;
        let denom = (one - (one - kc.c_sigma).powi(exponent)).sqrt();
        let h_sigma = if p_sigma_norm / denom < kc.h_sigma_threshold {
            one
        } else {
            zero
        };

        // p_c update.
        state.p_c.scale_in_place(one - kc.c_c);
        let coef_c = h_sigma * (kc.c_c * (two - kc.c_c) * kc.mu_eff).sqrt();
        state.p_c.scaled_add(coef_c, &y_w);

        // C update (eq. 47).
        let delta_h = (one - h_sigma) * kc.c_c * (two - kc.c_c);
        let c_scale = one + kc.c_1 * delta_h - kc.c_1 - kc.c_mu * kc.sum_w;
        state.c.scale_in_place(c_scale);
        state.c.rank_one_update(kc.c_1, &state.p_c);
        let n_f = F::from_usize(kc.n).unwrap();
        for (i, y_i) in y_sorted.iter().enumerate() {
            let w_i = kc.weights[i];
            let w_i_o = if w_i >= zero {
                w_i
            } else {
                let mut bt_y = state.b.mat_transpose_vec(y_i);
                bt_y.component_mul_assign(&state.d_inv);
                let cinv_norm_sq = bt_y.norm_squared();
                if cinv_norm_sq > zero {
                    w_i * n_f / cinv_norm_sq
                } else {
                    zero
                }
            };
            if w_i_o != zero {
                state.c.rank_one_update(kc.c_mu * w_i_o, y_i);
            }
        }
        drop(std::mem::take(&mut y_sorted));

        // Refresh eigendecomposition of the new C.
        let (b_new, eigs) = match state.c.try_eigh() {
            Ok(pair) => pair,
            Err(_) => return Ok((state, Some(TerminationReason::SolverFailed))),
        };
        state.b = b_new;
        let eig_floor = F::from_f64(1e-30).unwrap();
        for i in 0..kc.n {
            let lam = eigs[i].max(eig_floor);
            let s = lam.sqrt();
            state.d[i] = s;
            state.d_inv[i] = one / s;
        }

        // γ adaptation — runs after the m / σ / C update so it sees the
        // post-recombination state, before the new generation is sampled.
        // Consumes `state.penalty.raw_costs` (previous generation's raw
        // fitness, in sample order — γ-update only needs the IQR).
        update_gamma(&mut state, k, problem.inner());

        // Sample the new generation, evaluate at repaired points.
        let n = kc.n;
        let lambda = kc.lambda;
        let lo = problem.inner().lower().clone();
        let hi = problem.inner().upper().clone();
        sample_and_penalize(&mut state, lambda, n, &mut self.rng, problem, &lo, &hi)?;
        sort_population_ascending(&mut state.candidates, &mut state.costs);

        // Evaluate the mean (penalized) so param()/cost() report `m`.
        let gamma = &state.penalty.as_ref().unwrap().gamma;
        let (_raw, pen_m) = evaluate_with_penalty(problem, &state.m, &lo, &hi, gamma, n)?;
        state.m_cost = Some(pen_m);

        Ok((state, None))
    }
}
