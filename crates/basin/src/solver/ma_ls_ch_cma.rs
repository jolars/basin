use std::marker::PhantomData;

use crate::core::constraint::BoxConstraints;
use crate::core::executor::run_loop;
use crate::core::math::{
    ComponentMulAssign, MatTransposeVec, MatVec, MatrixFromDiagonal, MatrixIdentity, NormSquared,
    RankOneUpdate, SampleStandardNormal, SampleUniformBox, ScaleInPlace, ScaledAdd, SymmetricEigen,
    VectorLen,
};
use crate::core::problem::{CostFunction, EvalCounts, Problem};
use crate::core::rng::{ChaCha8Rng, RngExt, SeedableRng};
use crate::core::solver::Solver;
use crate::core::state::{CmaEsState, CountsMirror, PopulationState, State};
use crate::core::termination::{
    CmaEsTolerance, MaxCostEvals, TerminationCriterion, TerminationReason,
};
use crate::solver::cma_es::CmaEs;
use crate::solver::ssga::{
    bga_mutate_in_place, blx_alpha_crossover, nam_select, replace_worst_if_better,
};

/// Per-individual chain snapshot: a [`CmaEs`] carrying the derived
/// constants + RNG and the [`CmaEsState`] holding the distribution plus
/// the last generation's candidates that the next `next_iter` needs as a
/// recombination basis. Stored in [`MaLsChState::cma_chains`] as
/// `Option<ChainSlot<V, M>>` (`None` for individuals that have never
/// undergone LS).
type ChainSlot<V, M> = (CmaEs<V, M>, CmaEsState<V, M>);

/// State carried by [`MaLsChCma`]: a steady-state population plus
/// per-individual local-search chain data.
///
/// Solver-private (declared `pub` so the impl can construct it but kept
/// out of `core::state` per CONTRIBUTING.md tenet 4—one consumer, no
/// shared abstraction to design yet). Each chain entry is the saved
/// `(CmaEs, CmaEsState)` pair the inner needs for a resumed run: the
/// [`CmaEs`] carries the derived constants + RNG; the [`CmaEsState`]
/// carries the evolution state (mean, sigma, covariance, paths) and the
/// previous generation's λ candidates the next CMA `next_iter` needs as
/// the `y_sorted` basis for its m/σ/C update.
pub struct MaLsChState<V, M> {
    pub(crate) candidates: Vec<V>,
    pub(crate) costs: Vec<f64>,
    pub(crate) cma_chains: Vec<Option<ChainSlot<V, M>>>,
    /// Cost of `candidates[i]` at the end of its last LS application,
    /// or `+∞` if never LS'd. Used to evaluate the S_LS eligibility
    /// rule (Molina 2010 §4.3 step 1: `last_ls_cost − current_cost ≥
    /// δ_LS_min`).
    pub(crate) last_ls_cost: Vec<f64>,
    pub(crate) ls_application_count: Vec<u32>,
    iter: u64,
    cost_evals: u64,
    best_cost: f64,
    best_iter: u64,
    best_cost_evals: u64,
}

impl<V, M> MaLsChState<V, M> {
    /// Number of LS applications that have completed on
    /// `candidates[i]` so far. Exposed for tests that need to verify
    /// the chain machinery is firing (e.g. a single individual being
    /// re-selected and resumed).
    pub fn ls_application_count(&self, i: usize) -> u32 {
        self.ls_application_count[i]
    }
}

impl<V, M> State for MaLsChState<V, M> {
    type Param = V;
    type Float = f64;

    fn iter(&self) -> u64 {
        self.iter
    }
    fn increment_iter(&mut self) {
        self.iter += 1;
    }
    fn cost_evals(&self) -> u64 {
        self.cost_evals
    }
    fn param(&self) -> &V {
        &self.candidates[0]
    }
    fn cost(&self) -> f64 {
        self.costs[0]
    }

    fn best_param(&self) -> &V {
        // costs[0] is monotone non-increasing across iters (sort
        // invariant), so the best candidate IS candidates[0].
        &self.candidates[0]
    }
    fn best_cost(&self) -> f64 {
        self.best_cost
    }
    fn best_iter(&self) -> u64 {
        self.best_iter
    }
    fn best_cost_evals(&self) -> u64 {
        self.best_cost_evals
    }
    fn update_best(&mut self) {
        let curr = self.costs[0];
        if curr < self.best_cost {
            self.best_cost = curr;
            self.best_iter = self.iter;
            self.best_cost_evals = self.cost_evals;
        }
    }
    fn reset_best(&mut self) {
        self.best_cost = f64::INFINITY;
        self.best_iter = 0;
        self.best_cost_evals = 0;
    }
}

impl<V, M> CountsMirror for MaLsChState<V, M> {
    fn mirror(&mut self, delta: &EvalCounts) {
        // Same derivative-free convention as `BasicPopulationState`:
        // total work folds into `cost_evals` so a gradient-based inner
        // (a future LM/L-BFGS chain operator) bumps the same counter
        // as the SSGA phase's `cost` calls.
        self.cost_evals = delta.total_work();
    }
}

impl<V, M> PopulationState for MaLsChState<V, M> {
    fn candidates(&self) -> &[V] {
        &self.candidates
    }
    fn costs(&self) -> &[f64] {
        &self.costs
    }
}

impl<V, M> MaLsChState<V, M> {
    /// Build an empty state for `MaLsChCma::init` to fill. Use as the
    /// initial state passed to
    /// [`Executor`](crate::core::executor::Executor::new).
    pub fn new() -> Self {
        Self {
            candidates: Vec::new(),
            costs: Vec::new(),
            cma_chains: Vec::new(),
            last_ls_cost: Vec::new(),
            ls_application_count: Vec::new(),
            iter: 0,
            cost_evals: 0,
            best_cost: f64::INFINITY,
            best_iter: 0,
            best_cost_evals: 0,
        }
    }
}

impl<V, M> Default for MaLsChState<V, M> {
    fn default() -> Self {
        Self::new()
    }
}

/// `MA-LSCh-CMA`—memetic algorithm with local-search chains using
/// CMA-ES as the inner LS operator, per Molina et al. 2010 §4.4.
///
/// A steady-state real-coded GA (SSGA: BLX-α + NAM + BGA + replace-
/// worst) explores globally; CMA-ES exploits locally on individuals
/// that look promising. The novel piece is **chain persistence**: each
/// individual that has undergone LS keeps the *full CMA-ES evolution
/// state* (`m`, `σ`, `C`, `p_σ`, `p_c`, eigendecomposition `B/D`) so
/// that re-selecting it later resumes the same CMA-ES run from where it
/// last stopped, rather than restarting from scratch. CMA-ES adapts a
/// per-basin search distribution; the chain mechanism rewards basins
/// that keep improving by extending their LS time.
///
/// # Algorithm
///
/// One [`next_iter`](Solver::next_iter) does:
///
/// 1. **SSGA phase.** Loop SSGA offspring generation
///    (NAM → BLX-α → BGA → replace-worst) until `nfrec` cost
///    evaluations have been spent (Molina 2010 §4.3 step 2). When
///    replace-worst displaces an individual, its chain (if any) is
///    discarded—the new genome is treated as never-LS'd.
/// 2. **Build `S_LS`** = `{ i : never LS'd OR
///    last_ls_cost[i] − costs[i] ≥ δ_LS_min }` (§4.3 step 3).
/// 3. **Pick `c_LS`.** If `S_LS` non-empty, take the best individual
///    in it; otherwise take the best individual in the whole population
///    (Molina §4.3 final rule, line 371 of `references/molina-2010`).
/// 4. **Resume-or-fresh CMA-ES.**
///    - If `c_LS` has no stored chain: construct a fresh [`CmaEs`] (seed
///      derived from the outer RNG, §4.4.6) plus a
///      [`CmaEsState`]`::new(m = candidates[c_LS], σ = ½ · min_{j ≠ c_LS}
///      ‖candidates[c_LS] − candidates[j]‖)`.
///    - Otherwise: take the saved `(CmaEs, CmaEsState)` out of the chain
///      slot. The CmaEs carries the derived constants + RNG; the
///      CmaEsState carries the distribution plus the previous
///      generation's λ candidates the next CMA `next_iter` uses as the
///      recombination basis.
/// 5. **Drive the inner.** Reset the inner state's `iter` to 0, then
///    `run_loop(problem, state, &mut cma, &mut [MaxCostEvals(ls_intensity),
///    CmaEsTolerance(...)], u64::MAX)`. `CmaEs::init` is idempotent (its
///    constants cache + the non-empty population skip re-seeding), so
///    resumed runs keep their evolution state across calls.
/// 6. **Aggregate, route failures, write back.** Per CONTRIBUTING.md
///    "Solver composition" rules:
///    - Roll `inner_result.state.cost_evals()` into outer
///      `cost_evals` (rule 1: eval aggregation).
///    - Bubble `SolverFailed` (rule 3: failure routing); other
///      reasons (`MaxCostEvals`, `CmaEsTolerance`) are clean stops.
///    - If `inner_result.best_cost() < costs[c_LS]`, write the improved
///      best evaluated param/cost back (not the mean `param()` reports).
///      Always update `last_ls_cost[c_LS]` and
///      `ls_application_count[c_LS]`. Store the advanced
///      `(CmaEs, CmaEsState)` back in the slot.
/// 7. **Resort** the population (and parallel arrays) ascending.
///
/// # Default parameters
///
/// All defaults follow Molina 2010 §4.4.7 unless noted:
///
/// | Field | Default | Source |
/// |---|---|---|
/// | `pop_size` | `60` | §4.4.7 |
/// | `blx_alpha` | `0.5` | §4.4.7 |
/// | `nam_pool` | `4` (=`n_ass + 1` with `n_ass = 3`) | §4.4.7 |
/// | `mutation_prob` | `0.125` | §4.4.7 |
/// | `bga_range_fraction` | `0.1` | §4.4.4 |
/// | `ls_intensity` (`I_str`) | `300` | Bergmeir 2016 example |
/// | `ls_improvement_threshold` (`δ_LS_min`) | `1e-8` | §4.4.7 |
/// | `nfrec` | `= ls_intensity` | Derived from `r_L/G = 0.5` (§4.3) |
/// | `inner_lambda` | `CmaEs::default_lambda(D)` | Hansen 2016 |
/// | `initial_sigma_fallback` | `1.0` | when min-neighbor distance is 0 |
///
/// # Reproducibility
///
/// Carries a [`ChaCha8Rng`] seeded from the `seed: u64` passed to
/// [`new`](Self::new). Each fresh chain pulls its own per-individual
/// seed from this outer RNG, so the chain trajectories stay
/// deterministic for a fixed outer seed (load-bearing for stochastic-
/// solver reproducibility across platforms including
/// `wasm32-unknown-unknown`).
///
/// # Contract
///
/// - **Caller must:** implement [`CostFunction<Param = V, Output = f64>`]
///   *and* [`BoxConstraints<Param = V>`] on the problem. The SSGA needs
///   the box for initial sampling, BLX clipping, and BGA range; the
///   per-individual CMA-ES inner does not see the box (so the inner
///   CMA-ES is *unbounded*—chain individuals can drift outside the
///   box and be discarded only via the SSGA replace-worst feedback
///   loop). This matches Molina 2010 §4.4.6.
/// - **Caller must:** hand in a [`MaLsChState::new()`].
/// - **Implementor must:** maintain the
///   [`PopulationState`]
///   sorted-by-cost invariant at the start and end of every iteration.
///
/// # Termination
///
/// No solver-internal optimality test. Pair with framework criteria—
/// typically
/// [`MaxCostEvals`] for budget
/// control. Chain segments overshoot `I_str` by up to `λ_inner − 1`
/// evaluations (CMA-ES runs whole generations); the outer `MaxCostEvals`
/// will fire on the next outer iteration boundary, not exactly on the
/// budget. Document and accept; matches Bergmeir's reference behavior.
///
/// # Backends
///
/// Same coverage as [`CmaEs`]: `nalgebra` (`DVector`/`DMatrix`),
/// `faer` (`Col`/`Mat`), and `ndarray::Array1<f64>` with
/// `ndarray-linalg` not required (`SymmetricEigen` for `Array1` is
/// gated by the matrix backend that *isn't* available for ndarray, so
/// in practice nalgebra and faer). `Vec<f64>` produces a compile-time
/// error per tenet 5 (no honest matrix type).
///
/// # Examples
///
/// A memetic algorithm pairing a steady-state GA with CMA-ES local-search
/// chains. See [`RandomSearch`](crate::RandomSearch) for the population-
/// based `Executor` pattern.
pub struct MaLsChCma<V, M> {
    pop_size: usize,
    blx_alpha: f64,
    nam_pool: usize,
    mutation_prob: f64,
    bga_range_fraction: f64,
    ls_intensity: u64,
    ls_improvement_threshold: f64,
    nfrec: Option<u64>,
    inner_lambda: Option<usize>,
    initial_sigma_fallback: f64,
    seed: u64,
    rng: Option<ChaCha8Rng>,
    _phantom: PhantomData<(V, M)>,
}

impl<V, M> MaLsChCma<V, M> {
    /// Build a new `MaLsChCma` with the Molina 2010 §4.4.7 defaults
    /// and a PRNG seeded from `seed`.
    pub fn new(seed: u64) -> Self {
        Self {
            pop_size: 60,
            blx_alpha: 0.5,
            nam_pool: 4,
            mutation_prob: 0.125,
            bga_range_fraction: 0.1,
            ls_intensity: 300,
            ls_improvement_threshold: 1e-8,
            nfrec: None,
            inner_lambda: None,
            initial_sigma_fallback: 1.0,
            seed,
            rng: None,
            _phantom: PhantomData,
        }
    }

    /// Override the SSGA population size (default `60`).
    ///
    /// # Panics
    ///
    /// Panics if `pop_size < nam_pool`. NAM needs at least `nam_pool`
    /// individuals to sample from.
    pub fn with_pop_size(mut self, pop_size: usize) -> Self {
        assert!(
            pop_size >= self.nam_pool,
            "MaLsChCma requires pop_size >= nam_pool (got pop_size={}, nam_pool={})",
            pop_size,
            self.nam_pool
        );
        self.pop_size = pop_size;
        self
    }

    /// Override the BLX-α parameter (default `0.5`).
    ///
    /// # Panics
    ///
    /// Panics if `alpha < 0`.
    pub fn with_blx_alpha(mut self, alpha: f64) -> Self {
        assert!(alpha >= 0.0, "blx_alpha must be >= 0, got {}", alpha);
        self.blx_alpha = alpha;
        self
    }

    /// Override the NAM pool size (default `4`).
    ///
    /// # Panics
    ///
    /// Panics if `pool < 2`.
    pub fn with_nam_pool(mut self, pool: usize) -> Self {
        assert!(pool >= 2, "nam_pool must be >= 2, got {}", pool);
        self.nam_pool = pool;
        self
    }

    /// Override the per-gene BGA mutation probability (default `0.125`).
    ///
    /// # Panics
    ///
    /// Panics if `p` is not in `[0, 1]`.
    pub fn with_mutation_prob(mut self, p: f64) -> Self {
        assert!(
            (0.0..=1.0).contains(&p),
            "mutation_prob must be in [0, 1], got {}",
            p
        );
        self.mutation_prob = p;
        self
    }

    /// Override the BGA range fraction (default `0.1`).
    ///
    /// # Panics
    ///
    /// Panics if `f <= 0`.
    pub fn with_bga_range_fraction(mut self, f: f64) -> Self {
        assert!(f > 0.0, "bga_range_fraction must be > 0, got {}", f);
        self.bga_range_fraction = f;
        self
    }

    /// Override `I_str`, the per-chain LS intensity in cost-evaluation
    /// units (default `300`, Bergmeir 2016 example value). Each chain
    /// segment runs CMA-ES for whole generations until `cost_evals ≥
    /// I_str`, slightly overshooting `I_str` by up to `λ_inner − 1`.
    ///
    /// # Panics
    ///
    /// Panics if `istr == 0`.
    pub fn with_ls_intensity(mut self, istr: u64) -> Self {
        assert!(istr >= 1, "ls_intensity must be >= 1, got {}", istr);
        self.ls_intensity = istr;
        self
    }

    /// Override `δ_LS_min`, the cost-improvement threshold an
    /// already-LS'd individual must clear to be re-eligible for LS
    /// (default `1e-8`, Molina 2010 §4.4.7).
    ///
    /// # Panics
    ///
    /// Panics if `delta < 0`.
    pub fn with_ls_improvement_threshold(mut self, delta: f64) -> Self {
        assert!(
            delta >= 0.0,
            "ls_improvement_threshold must be >= 0, got {}",
            delta
        );
        self.ls_improvement_threshold = delta;
        self
    }

    /// Override `n_frec`, the number of SSGA cost evaluations performed
    /// between LS applications (default is to match `ls_intensity`,
    /// which gives the 50/50 effort split Molina 2010 §4.3 recommends
    /// from `r_L/G = 0.5`).
    ///
    /// # Panics
    ///
    /// Panics if `n == 0`.
    pub fn with_nfrec(mut self, n: u64) -> Self {
        assert!(n >= 1, "nfrec must be >= 1, got {}", n);
        self.nfrec = Some(n);
        self
    }

    /// Override the inner CMA-ES population size `λ_inner` (default is
    /// [`CmaEs::default_lambda(D)`](CmaEs::default_lambda) computed at
    /// init time from the problem's dimension).
    ///
    /// # Panics
    ///
    /// Panics if `lambda < 4` (Hansen 2016's lower bound on CMA-ES λ).
    pub fn with_inner_lambda(mut self, lambda: usize) -> Self {
        assert!(lambda >= 4, "inner_lambda must be >= 4, got {}", lambda);
        self.inner_lambda = Some(lambda);
        self
    }

    /// Override the σ value used when constructing a fresh CMA-ES
    /// chain for an individual whose nearest-neighbor distance is `0`
    /// (degenerate identical-population case). Default `1.0`.
    ///
    /// # Panics
    ///
    /// Panics if `sigma <= 0`.
    pub fn with_initial_sigma_fallback(mut self, sigma: f64) -> Self {
        assert!(
            sigma > 0.0,
            "initial_sigma_fallback must be > 0, got {}",
            sigma
        );
        self.initial_sigma_fallback = sigma;
        self
    }
}

/// Compute `0.5 · min_{j ≠ i} ‖candidates[i] − candidates[j]‖₂`, the
/// per-individual σ-init formula from Molina 2010 §4.4.6. Returns
/// `None` if there's no other individual (singleton population).
fn sigma_init_for<V>(candidates: &[V], i: usize) -> Option<f64>
where
    V: Clone + ScaledAdd<f64> + NormSquared,
{
    if candidates.len() < 2 {
        return None;
    }
    let mut best_sq = f64::INFINITY;
    for (j, x) in candidates.iter().enumerate() {
        if j == i {
            continue;
        }
        let mut diff = candidates[i].clone();
        diff.scaled_add(-1.0, x);
        let d_sq = diff.norm_squared();
        if d_sq < best_sq {
            best_sq = d_sq;
        }
    }
    Some(0.5 * best_sq.sqrt())
}

impl<P, V, M> Solver<P, MaLsChState<V, M>> for MaLsChCma<V, M>
where
    P: CostFunction<Param = V, Output = f64> + BoxConstraints<Param = V>,
    V: VectorLen
        + Clone
        + SampleUniformBox
        + SampleStandardNormal
        + ScaledAdd<f64>
        + ScaleInPlace
        + ComponentMulAssign
        + NormSquared
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
    CmaEs<V, M>: Solver<P, CmaEsState<V, M>, Error = P::Error>,
{
    type Error = P::Error;

    fn init(
        &mut self,
        problem: &mut Problem<P>,
        mut state: MaLsChState<V, M>,
    ) -> Result<MaLsChState<V, M>, Self::Error> {
        let lo = problem.inner().lower().clone();
        let hi = problem.inner().upper().clone();
        let mut rng = ChaCha8Rng::seed_from_u64(self.seed);

        // Sample the initial population uniformly in the box.
        state.candidates.clear();
        state.costs.clear();
        state.cma_chains.clear();
        state.last_ls_cost.clear();
        state.ls_application_count.clear();
        for _ in 0..self.pop_size {
            let x = V::sample_uniform_box(&lo, &hi, &mut rng);
            let c = problem.cost(&x)?;
            state.candidates.push(x);
            state.costs.push(c);
            state.cma_chains.push(None);
            state.last_ls_cost.push(f64::INFINITY);
            state.ls_application_count.push(0);
        }
        sort_parallel_arrays(&mut state);

        self.rng = Some(rng);
        Ok(state)
    }

    fn next_iter(
        &mut self,
        problem: &mut Problem<P>,
        mut state: MaLsChState<V, M>,
    ) -> Result<(MaLsChState<V, M>, Option<TerminationReason>), Self::Error> {
        let lo = problem.inner().lower().clone();
        let hi = problem.inner().upper().clone();
        let rng = self
            .rng
            .as_mut()
            .expect("MaLsChCma::init must run before next_iter");
        let nfrec = self.nfrec.unwrap_or(self.ls_intensity);

        // -- Phase 1: SSGA for nfrec evaluations. --
        // Budget the SSGA phase against the wrapper's authoritative
        // counters so a same-problem inner run earlier on this iteration
        // (none today, but the contract is uniform) wouldn't double-count.
        let phase_start_counts = *problem.counts();
        while problem.counts().cost_evals - phase_start_counts.cost_evals < nfrec {
            let (p1, p2) = nam_select(&state.candidates, self.nam_pool, rng);
            let mut child = blx_alpha_crossover(
                &state.candidates[p1],
                &state.candidates[p2],
                self.blx_alpha,
                &lo,
                &hi,
                rng,
            );
            bga_mutate_in_place(
                &mut child,
                &lo,
                &hi,
                self.mutation_prob,
                self.bga_range_fraction,
                rng,
            );
            let c_child = problem.cost(&child)?;
            if let Some(replaced_idx) =
                replace_worst_if_better(&mut state.candidates, &mut state.costs, child, c_child)
            {
                // The displaced individual's chain (if any) is orphaned:
                // the new genome is a fresh point that should start its
                // own chain on first LS pick.
                state.cma_chains[replaced_idx] = None;
                state.last_ls_cost[replaced_idx] = f64::INFINITY;
                state.ls_application_count[replaced_idx] = 0;
            }
        }
        sort_parallel_arrays(&mut state);

        // -- Phase 2: pick the LS target c_LS. --
        let mut c_ls: Option<usize> = None;
        let mut best_cost_in_s_ls = f64::INFINITY;
        for i in 0..state.candidates.len() {
            let eligible = state.cma_chains[i].is_none()
                || (state.last_ls_cost[i] - state.costs[i] >= self.ls_improvement_threshold);
            if eligible && state.costs[i] < best_cost_in_s_ls {
                best_cost_in_s_ls = state.costs[i];
                c_ls = Some(i);
            }
        }
        // Molina §4.3: when |S_LS| = 0, apply LS to the best individual
        // unconditionally.
        let c_ls = c_ls.unwrap_or(0);

        // -- Phase 3: resume or construct the inner CMA-ES. --
        let (mut cma, inner_state) = match state.cma_chains[c_ls].take() {
            Some((cma, inner_state)) => {
                let mut s = inner_state;
                // Local budget reset. `run_loop` already snapshots the
                // wrapper at entry so the inner state's `cost_evals`
                // measures per-segment work—but the iteration counter
                // is the inner's responsibility and the `MaxCostEvals`
                // criterion in Phase 4 reads `state.cost_evals()`,
                // which is the wrapper-mirrored per-run value. Reset
                // `iter` so the chain restarts at iter 0; the
                // `run_loop` baseline takes care of the eval counter.
                s.iter = 0;
                (cma, s)
            }
            None => {
                let sigma_init = sigma_init_for(&state.candidates, c_ls)
                    .filter(|s| *s > 0.0)
                    .unwrap_or(self.initial_sigma_fallback);
                let derived_seed = rng.random::<u64>();
                let mut cma = CmaEs::<V, M>::new(derived_seed);
                if let Some(lam) = self.inner_lambda {
                    cma = cma.with_lambda(lam);
                }
                // Empty inner population—CmaEs::init samples the first
                // generation. The mean / σ live on the state now.
                let inner_state =
                    CmaEsState::<V, M>::new(state.candidates[c_ls].clone(), sigma_init);
                (cma, inner_state)
            }
        };

        // -- Phase 4: drive the inner. --
        // Build per-call criteria so `MaxCostEvals` doesn't leak state
        // between chain segments (it's stateless, but the
        // `InnerExecutor` reuse pattern doesn't fit here since we hold
        // a different `cma` per individual). Allocation cost is one
        // box per chain segment—negligible against I_str evals.
        // `CmaEsTolerance` replaces the internal TolX hook the old
        // `CmaEs::terminate` provided; tol_x is `1e−12 ·` the segment's
        // starting σ (Hansen's default, made per-segment-relative so it
        // is resume-safe).
        let tol_x = 1e-12 * inner_state.sigma();
        let mut criteria: Vec<Box<dyn TerminationCriterion<CmaEsState<V, M>>>> = vec![
            Box::new(MaxCostEvals(self.ls_intensity)),
            Box::new(CmaEsTolerance::new(tol_x)),
        ];
        let inner_result = run_loop(problem, inner_state, &mut cma, &mut criteria, u64::MAX)?;

        // -- Phase 5: route failures, write back. --
        // Same-problem composition: inner evals already flowed through the
        // outer wrapper, so the `MaLsChState` mirror sees them via
        // `delta.total_work()`. `SolverFailed` is the only failure
        // reason; other reasons (`MaxCostEvals` from our budget,
        // `SolverConverged` from CMA's TolX) are clean stops the outer
        // consumes.
        if inner_result.reason.is_failure() {
            // Leave the chain dropped so a future pick would restart.
            return Ok((state, Some(inner_result.reason)));
        }

        // Adopt the chain's best *evaluated* point (xbest), not the
        // distribution mean that `param()`/`cost()` now report—the
        // memetic algorithm wants the best feasible refinement found.
        let new_cost = inner_result.best_cost();
        let new_param = inner_result.best_param().clone();
        // Conditional write-back: only adopt the LS result if it
        // improves on the current cost. Strict Molina §4.3 step 10 is
        // unconditional, but a conditional update is safer (CMA-ES is
        // genuinely non-monotone over a chain segment) and matches the
        // Rmalschains R package's behavior.
        if new_cost < state.costs[c_ls] {
            state.candidates[c_ls] = new_param;
            state.costs[c_ls] = new_cost;
        }
        state.last_ls_cost[c_ls] = state.costs[c_ls];
        state.ls_application_count[c_ls] = state.ls_application_count[c_ls].saturating_add(1);
        state.cma_chains[c_ls] = Some((cma, inner_result.state));

        // -- Phase 6: resort all parallel arrays jointly. --
        sort_parallel_arrays(&mut state);

        Ok((state, None))
    }
}

/// Joint ascending-by-cost sort over the five parallel arrays in
/// [`MaLsChState`]. The chain pointer travels with its individual
/// through the permutation—that's why the chain belongs in the
/// state, not in a side index.
fn sort_parallel_arrays<V, M>(state: &mut MaLsChState<V, M>) {
    let n = state.candidates.len();
    debug_assert_eq!(n, state.costs.len());
    debug_assert_eq!(n, state.cma_chains.len());
    debug_assert_eq!(n, state.last_ls_cost.len());
    debug_assert_eq!(n, state.ls_application_count.len());

    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&i, &j| {
        state.costs[i]
            .partial_cmp(&state.costs[j])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    apply_permutation::<V>(&mut state.candidates, &idx);
    apply_permutation::<f64>(&mut state.costs, &idx);
    apply_permutation::<Option<ChainSlot<V, M>>>(&mut state.cma_chains, &idx);
    apply_permutation::<f64>(&mut state.last_ls_cost, &idx);
    apply_permutation::<u32>(&mut state.ls_application_count, &idx);
}

/// Cycle-following in-place permutation: after the call,
/// `slice[i] = original[idx[i]]`. Mirrors the helper used by
/// `cma_es::sort_population_ascending`; inlined here because that
/// helper is module-private.
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
