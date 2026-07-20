//! Generic MA-LSCh: memetic algorithm with local-search chains, generic
//! over the resumable LS operator.
//!
//! The steady-state GA framework, the `S_LS` eligibility rule, the fixed
//! LS intensity, and the chain bookkeeping live here; everything the
//! chain operator itself must provide is the
//! [`ResumableInner`](crate::core::inner::ResumableInner) contract (seed
//! at a point + scale, snapshot, resume).
//! [`MaLsChCma`](crate::solver::MaLsChCma) (CMA-ES chains, Molina et
//! al. 2010) is the type alias `MaLsCh<V, CmaEs<V, M>>`;
//! [`MaLsChSw`](crate::solver::MaLsChSw) (Solis-Wets chains,
//! MA-SW-Chains, CEC 2010) is `MaLsCh<V, SolisWets>`.

use std::marker::PhantomData;

use crate::core::constraint::BoxConstraints;
use crate::core::executor::run_loop;
use crate::core::inner::ResumableInner;
use crate::core::math::{NormSquared, SampleUniformBox, ScaledAdd, VectorLen};
use crate::core::problem::{CostFunction, EvalCounts, Problem};
use crate::core::rng::{ChaCha8Rng, RngExt, SeedableRng};
use crate::core::solver::Solver;
use crate::core::state::{CountsMirror, PopulationState, State};
use crate::core::termination::{MaxCostEvals, TerminationCriterion, TerminationReason};
// Cycle-following in-place permutation: after the call,
// `slice[i] = original[idx[i]]`.
use crate::solver::cma_es::apply_permutation;
use crate::solver::ssga::{
    bga_mutate_in_place, blx_alpha_crossover, nam_select, replace_worst_if_better,
};

/// State carried by [`MaLsCh`]: a steady-state population plus
/// per-individual local-search chain data.
///
/// `C` is the chain-slot payload—`(LS, LS::State)` for a concrete
/// chain operator `LS:` [`ResumableInner`]—kept as a bare type
/// parameter so the struct definition carries no trait bounds (bounds
/// live on the [`Solver`] impl). Each occupied slot is the saved
/// `(solver, state)` pair the operator needs for a resumed run: the
/// solver carries its derived constants and RNG stream, the state the
/// full evolution state.
///
/// Use through the concrete aliases
/// ([`MaLsChState`](crate::solver::MaLsChState) for the CMA-ES chains)
/// or directly for a custom operator.
pub struct MaLsChGenericState<V, C> {
    pub(crate) candidates: Vec<V>,
    pub(crate) costs: Vec<f64>,
    pub(crate) chains: Vec<Option<C>>,
    /// Cost of `candidates[i]` when its last LS segment *started*, or
    /// `+∞` if never LS'd. `last_ls_cost − current_cost` is thus the
    /// improvement the last LS application obtained, which the S_LS
    /// eligibility rule compares against `δ_LS_min` (Molina 2010 §4.3
    /// step 1).
    pub(crate) last_ls_cost: Vec<f64>,
    pub(crate) ls_application_count: Vec<u32>,
    iter: u64,
    cost_evals: u64,
    best_cost: f64,
    best_iter: u64,
    best_cost_evals: u64,
}

impl<V, C> MaLsChGenericState<V, C> {
    /// Number of LS applications that have completed on
    /// `candidates[i]` so far. Exposed for tests that need to verify
    /// the chain machinery is firing (e.g. a single individual being
    /// re-selected and resumed).
    pub fn ls_application_count(&self, i: usize) -> u32 {
        self.ls_application_count[i]
    }
}

impl<V, C> State for MaLsChGenericState<V, C> {
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

impl<V, C> CountsMirror for MaLsChGenericState<V, C> {
    fn mirror(&mut self, delta: &EvalCounts) {
        // Same derivative-free convention as `BasicPopulationState`:
        // total work folds into `cost_evals` so a gradient-based inner
        // (a future LM or L-BFGS chain operator) bumps the same counter
        // as the SSGA phase's `cost` calls.
        self.cost_evals = delta.total_work();
    }
}

impl<V, C> PopulationState for MaLsChGenericState<V, C> {
    fn candidates(&self) -> &[V] {
        &self.candidates
    }
    fn costs(&self) -> &[f64] {
        &self.costs
    }
}

impl<V, C> MaLsChGenericState<V, C> {
    /// Build an empty state for `MaLsCh::init` to fill. Use as the
    /// initial state passed to
    /// [`Executor`](crate::core::executor::Executor::new).
    pub fn new() -> Self {
        Self {
            candidates: Vec::new(),
            costs: Vec::new(),
            chains: Vec::new(),
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

impl<V, C> Default for MaLsChGenericState<V, C> {
    fn default() -> Self {
        Self::new()
    }
}

/// `MA-LSCh`: memetic algorithm with local-search chains (Molina et al.
/// 2010), generic over the chain operator.
///
/// A steady-state real-coded GA (SSGA: BLX-α + NAM + BGA + replace-
/// worst) explores globally; a local-search operator exploits locally on
/// individuals that look promising. The novel piece is **chain
/// persistence**: each individual that has undergone LS keeps the
/// operator's *full evolution state* so that re-selecting it later
/// resumes the same LS run from where it last stopped, rather than
/// restarting from scratch. The operator adapts per-basin search
/// parameters; the chain mechanism rewards basins that keep improving by
/// extending their LS time.
///
/// The operator plugs in through [`ResumableInner`] (seed at a point
/// with a scale hint, snapshot the `(solver, state)` pair, resume).
/// Shipped configurations: [`MaLsChCma`](crate::solver::MaLsChCma)
/// (CMA-ES chains, `MaLsCh<V, CmaEs<V, M>>`) and
/// [`MaLsChSw`](crate::solver::MaLsChSw) (Solis-Wets chains,
/// `MaLsCh<V, SolisWets>`). Both specialized aliases have a `new(seed)`
/// constructor; a hand-configured operator prototype goes through
/// [`with_inner`](Self::with_inner).
///
/// # Algorithm
///
/// One [`next_iter`](Solver::next_iter) does:
///
/// 1. **SSGA phase.** Loop SSGA offspring generation
///    (NAM → BLX-α → BGA → replace-worst) until `nfrec` cost
///    evaluations have been spent (Molina 2010 §4.3 step 2). When
///    replace-worst displaces an individual, its chain (if any) is
///    discarded; the new genome is treated as never-LS'd.
/// 2. **Build `S_LS`** = `{ i : never LS'd OR
///    last_ls_cost[i] − costs[i] ≥ δ_LS_min }` (§4.3 step 3), where the
///    difference is the improvement the last LS segment obtained.
/// 3. **Pick `c_LS`.** If `S_LS` non-empty, take the best individual
///    in it; otherwise take the best individual in the whole population
///    (Molina §4.3 final rule, line 371 of `references/molina-2010`).
/// 4. **Resume-or-fresh operator.**
///    - If `c_LS` has no stored chain:
///      [`seed_chain`](ResumableInner::seed_chain) builds a fresh
///      `(solver, state)` pair at `candidates[c_LS]` with scale
///      `½ · min_{j ≠ c_LS} ‖candidates[c_LS] − candidates[j]‖` (σ for
///      CMA-ES, ρ for Solis-Wets) and a seed derived from the outer
///      RNG.
///    - Otherwise: take the saved pair out of the chain slot and
///      [`prepare_resume`](ResumableInner::prepare_resume) it.
/// 5. **Drive the inner.** `run_loop(problem, state, &mut ls,
///    [MaxCostEvals(ls_intensity)] + segment_criteria, u64::MAX)`. The
///    operator's [`Solver::init`] is resume-idempotent (the
///    [`ResumableInner`] contract), so resumed runs keep their
///    evolution state across calls.
/// 6. **Aggregate, route failures, write back.** Per CONTRIBUTING.md
///    "Solver composition" rules:
///    - Roll `inner_result.state.cost_evals()` into outer
///      `cost_evals` (rule 1: eval aggregation).
///    - Bubble `SolverFailed` (rule 3: failure routing); other
///      reasons (`MaxCostEvals`, operator tolerances) are clean stops.
///    - If `inner_result.best_cost() < costs[c_LS]`, write the improved
///      best evaluated param/cost back. Always update
///      `last_ls_cost[c_LS]` and `ls_application_count[c_LS]`. Store
///      the advanced `(solver, state)` pair back in the slot only when
///      the segment improved by at least `δ_LS_min`; otherwise drop the
///      chain (the reference removes exhausted chains from memory), so
///      a future pick reseeds fresh.
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
/// | `initial_scale_fallback` | `1.0` | when min-neighbor distance is 0 |
///
/// # Reproducibility
///
/// Carries a [`ChaCha8Rng`] seeded from the `seed: u64` passed to the
/// constructor. Each fresh chain pulls its own per-individual seed from
/// this outer RNG (and the operator prototype's own RNG is never drawn:
/// the [`ResumableInner`] purity contract), so the chain trajectories
/// stay deterministic for a fixed outer seed across platforms including
/// `wasm32-unknown-unknown`.
///
/// # Contract
///
/// - **Caller must:** implement [`CostFunction<Param = V, Output = f64>`]
///   *and* [`BoxConstraints<Param = V>`] on the problem. The SSGA needs
///   the box for initial sampling, BLX clipping, and BGA range; the
///   per-individual LS inner does not see the box (so the inner is
///   *unbounded*: chain individuals can drift outside the box and be
///   discarded only via the SSGA replace-worst feedback loop). This
///   matches Molina 2010 §4.4.6.
/// - **Caller must:** hand in a [`MaLsChGenericState::new()`] (or the
///   concrete alias's `new()`).
/// - **Implementor must:** maintain the [`PopulationState`]
///   sorted-by-cost invariant at the start and end of every iteration.
///
/// # Termination
///
/// No solver-internal optimality test. Pair with framework criteria,
/// typically [`MaxCostEvals`] for budget control. Chain segments
/// overshoot `I_str` slightly when the operator evaluates in batches
/// (CMA-ES runs whole generations, overshooting by up to `λ_inner − 1`
/// evaluations; Solis-Wets by at most one reversal evaluation); the
/// outer `MaxCostEvals` will fire on the next outer iteration boundary,
/// not exactly on the budget. Document and accept; matches Bergmeir's
/// reference behavior.
///
/// # Backends
///
/// The outer SSGA needs only the vector tier
/// ([`SampleUniformBox`] + [`ScaledAdd`] + [`NormSquared`] + indexing),
/// so effective coverage is set by the chain operator: all four backends
/// for both shipped configurations (CMA-ES additionally requires the
/// matrix bound
/// [`SymmetricEigen`](crate::core::math::SymmetricEigen), which every
/// backend satisfies; Solis-Wets requires no matrix type at all).
///
/// # References
///
/// - Molina, D., Lozano, M., García-Martínez, C., and Herrera, F.
///   (2010). "Memetic algorithms for continuous optimisation based on
///   local search chains." *Evolutionary Computation*, 18(1), 27-63.
///   <https://doi.org/10.1162/evco.2010.18.1.18102>
pub struct MaLsCh<V, LS> {
    pop_size: usize,
    blx_alpha: f64,
    nam_pool: usize,
    mutation_prob: f64,
    bga_range_fraction: f64,
    ls_intensity: u64,
    ls_improvement_threshold: f64,
    nfrec: Option<u64>,
    initial_scale_fallback: f64,
    seed: u64,
    rng: Option<ChaCha8Rng>,
    /// LS-operator configuration prototype: hyperparameters are copied
    /// into each fresh chain via
    /// [`ResumableInner::seed_chain`]; its own RNG is never drawn.
    pub(crate) ls: LS,
    _phantom: PhantomData<V>,
}

impl<V, LS> MaLsCh<V, LS> {
    /// Build an `MaLsCh` around an explicit LS-operator prototype, with
    /// the Molina 2010 §4.4.7 defaults and a PRNG seeded from `seed`.
    ///
    /// The prototype's hyperparameters are copied into every fresh
    /// chain; its own RNG seed is irrelevant (never drawn), so
    /// `SolisWets::new(0).with_…(…)`-style construction is fine. The
    /// shipped operators also have specialized `new(seed)` constructors
    /// on the aliases ([`MaLsChCma`](crate::solver::MaLsChCma),
    /// [`MaLsChSw`](crate::solver::MaLsChSw)) that supply a default
    /// prototype.
    pub fn with_inner(seed: u64, ls: LS) -> Self {
        Self {
            pop_size: 60,
            blx_alpha: 0.5,
            nam_pool: 4,
            mutation_prob: 0.125,
            bga_range_fraction: 0.1,
            ls_intensity: 300,
            ls_improvement_threshold: 1e-8,
            nfrec: None,
            initial_scale_fallback: 1.0,
            seed,
            rng: None,
            ls,
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
            "MaLsCh requires pop_size >= nam_pool (got pop_size={}, nam_pool={})",
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
    /// segment runs the operator until `cost_evals ≥ I_str`, slightly
    /// overshooting when the operator evaluates in batches (see the
    /// type-level "Termination" note).
    ///
    /// # Panics
    ///
    /// Panics if `istr == 0`.
    pub fn with_ls_intensity(mut self, istr: u64) -> Self {
        assert!(istr >= 1, "ls_intensity must be >= 1, got {}", istr);
        self.ls_intensity = istr;
        self
    }

    /// Override `δ_LS_min`, the cost improvement an LS segment must
    /// obtain for the individual to stay LS-eligible and for its chain
    /// to be kept for resumption (default `1e-8`, Molina 2010 §4.4.7).
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

    /// Override the scale value used when constructing a fresh chain
    /// for an individual whose nearest-neighbor distance is `0`
    /// (degenerate identical-population case). Default `1.0`. The scale
    /// becomes the operator's initial step: σ for CMA-ES, ρ for
    /// Solis-Wets.
    ///
    /// # Panics
    ///
    /// Panics if `scale <= 0`.
    pub fn with_initial_scale_fallback(mut self, scale: f64) -> Self {
        assert!(
            scale > 0.0,
            "initial_scale_fallback must be > 0, got {}",
            scale
        );
        self.initial_scale_fallback = scale;
        self
    }
}

/// Compute `0.5 · min_{j ≠ i} ‖candidates[i] − candidates[j]‖₂`, the
/// per-individual scale-init formula from Molina 2010 §4.4.6. Returns
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

impl<P, V, LS> Solver<P, MaLsChGenericState<V, (LS, <LS as ResumableInner<V>>::State)>>
    for MaLsCh<V, LS>
where
    P: CostFunction<Param = V, Output = f64> + BoxConstraints<Param = V>,
    V: VectorLen
        + Clone
        + SampleUniformBox
        + ScaledAdd<f64>
        + NormSquared
        + std::ops::Index<usize, Output = f64>
        + std::ops::IndexMut<usize, Output = f64>,
    LS: ResumableInner<V> + Solver<P, <LS as ResumableInner<V>>::State, Error = P::Error>,
{
    type Error = P::Error;

    fn init(
        &mut self,
        problem: &mut Problem<P>,
        mut state: MaLsChGenericState<V, (LS, <LS as ResumableInner<V>>::State)>,
    ) -> Result<MaLsChGenericState<V, (LS, <LS as ResumableInner<V>>::State)>, Self::Error> {
        let lo = problem.inner().lower().clone();
        let hi = problem.inner().upper().clone();
        let mut rng = ChaCha8Rng::seed_from_u64(self.seed);

        // Sample the initial population uniformly in the box.
        state.candidates.clear();
        state.costs.clear();
        state.chains.clear();
        state.last_ls_cost.clear();
        state.ls_application_count.clear();
        for _ in 0..self.pop_size {
            let x = V::sample_uniform_box(&lo, &hi, &mut rng);
            let c = problem.cost(&x)?;
            state.candidates.push(x);
            state.costs.push(c);
            state.chains.push(None);
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
        mut state: MaLsChGenericState<V, (LS, <LS as ResumableInner<V>>::State)>,
    ) -> Result<
        (
            MaLsChGenericState<V, (LS, <LS as ResumableInner<V>>::State)>,
            Option<TerminationReason>,
        ),
        Self::Error,
    > {
        let lo = problem.inner().lower().clone();
        let hi = problem.inner().upper().clone();
        let rng = self
            .rng
            .as_mut()
            .expect("MaLsCh::init must run before next_iter");
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
                state.chains[replaced_idx] = None;
                state.last_ls_cost[replaced_idx] = f64::INFINITY;
                state.ls_application_count[replaced_idx] = 0;
            }
        }
        sort_parallel_arrays(&mut state);

        // -- Phase 2: pick the LS target c_LS. --
        // S_LS membership (Molina §4.3 step 1): never LS'd, or the last
        // LS segment cleared δ_LS_min. `last_ls_cost` holds the cost the
        // last segment *started* from, so the difference is that
        // segment's improvement; it can only grow stale through
        // replace-worst, which resets the slot. Ineligibility is sticky
        // (the reference's `non_improved` marker): a failed segment also
        // drops the chain, so `chains[i].is_none()` can't stand in for
        // "never LS'd" here.
        let mut c_ls: Option<usize> = None;
        let mut best_cost_in_s_ls = f64::INFINITY;
        for i in 0..state.candidates.len() {
            let eligible = state.ls_application_count[i] == 0
                || (state.last_ls_cost[i] - state.costs[i] >= self.ls_improvement_threshold);
            if eligible && state.costs[i] < best_cost_in_s_ls {
                best_cost_in_s_ls = state.costs[i];
                c_ls = Some(i);
            }
        }
        // Molina §4.3: when |S_LS| = 0, apply LS to the best individual
        // unconditionally.
        let c_ls = c_ls.unwrap_or(0);

        // -- Phase 3: resume or construct the inner operator. --
        let (mut ls, inner_state) = match state.chains[c_ls].take() {
            Some((ls, mut s)) => {
                // Local budget reset. `run_loop` already snapshots the
                // wrapper at entry so the inner state's `cost_evals`
                // measures per-segment work, but the iteration counter
                // is the inner's responsibility and the `MaxCostEvals`
                // criterion in Phase 4 reads `state.cost_evals()`,
                // which is the wrapper-mirrored per-run value.
                // `prepare_resume` resets `iter` so the chain restarts
                // at iter 0; the `run_loop` baseline takes care of the
                // eval counter.
                ls.prepare_resume(&mut s);
                (ls, s)
            }
            None => {
                let scale = sigma_init_for(&state.candidates, c_ls)
                    .filter(|s| *s > 0.0)
                    .unwrap_or(self.initial_scale_fallback);
                let derived_seed = rng.random::<u64>();
                self.ls.seed_chain(
                    &state.candidates[c_ls],
                    state.costs[c_ls],
                    scale,
                    derived_seed,
                )
            }
        };

        // -- Phase 4: drive the inner. --
        // Build per-call criteria so `MaxCostEvals` doesn't leak state
        // between chain segments (it's stateless, but the
        // `InnerExecutor` reuse pattern doesn't fit here since we hold
        // a different operator instance per individual). Allocation
        // cost is a few boxes per chain segment, negligible against
        // I_str evals. Budget first, then the operator's per-segment
        // convergence criteria (`ResumableInner::segment_criteria`,
        // built from the segment's starting state), preserving the
        // check order.
        let mut criteria: Vec<Box<dyn TerminationCriterion<<LS as ResumableInner<V>>::State>>> =
            vec![Box::new(MaxCostEvals(self.ls_intensity))];
        criteria.extend(ls.segment_criteria(&inner_state));
        let inner_result = run_loop(problem, inner_state, &mut ls, &mut criteria, u64::MAX)?;

        // -- Phase 5: route failures, write back. --
        // Same-problem composition: inner evals already flowed through the
        // outer wrapper, so the `MaLsChGenericState` mirror sees them via
        // `delta.total_work()`. `SolverFailed` is the only failure
        // reason; other reasons (`MaxCostEvals` from our budget, the
        // operator's own tolerances) are clean stops the outer consumes.
        if inner_result.reason.is_failure() {
            // Leave the chain dropped so a future pick would restart.
            return Ok((state, Some(inner_result.reason)));
        }

        // Adopt the chain's best *evaluated* point (xbest), not
        // whatever `param()`/`cost()` now report (CMA-ES reports the
        // distribution mean); the memetic algorithm wants the best
        // feasible refinement found.
        let new_cost = inner_result.best_cost();
        let new_param = inner_result.best_param().clone();
        // Conditional write-back: only adopt the LS result if it
        // improves on the current cost. Strict Molina §4.3 step 10 is
        // unconditional, but a conditional update is safer (CMA-ES is
        // genuinely non-monotone over a chain segment) and matches the
        // Rmalschains R package's behavior.
        let pre_segment_cost = state.costs[c_ls];
        if new_cost < state.costs[c_ls] {
            state.candidates[c_ls] = new_param;
            state.costs[c_ls] = new_cost;
        }
        // Record the cost this segment started from: eligibility (Phase
        // 2) reads `last_ls_cost - costs`, the improvement obtained by
        // the previous LS application (Molina §4.3 step 1b).
        state.last_ls_cost[c_ls] = pre_segment_cost;
        state.ls_application_count[c_ls] = state.ls_application_count[c_ls].saturating_add(1);
        // Keep the chain only when the segment cleared δ_LS_min.
        // Rmalschains removes exhausted chains (`m_memory->remove`), so
        // a future pick reseeds at a fresh scale instead of resuming a
        // converged operator that would burn the whole segment budget
        // making no progress.
        if pre_segment_cost - state.costs[c_ls] >= self.ls_improvement_threshold {
            state.chains[c_ls] = Some((ls, inner_result.state));
        }

        // -- Phase 6: resort all parallel arrays jointly. --
        sort_parallel_arrays(&mut state);

        Ok((state, None))
    }
}

/// Joint ascending-by-cost sort over the five parallel arrays in
/// [`MaLsChGenericState`]. The chain pointer travels with its individual
/// through the permutation; that's why the chain belongs in the
/// state, not in a side index.
fn sort_parallel_arrays<V, C>(state: &mut MaLsChGenericState<V, C>) {
    let n = state.candidates.len();
    debug_assert_eq!(n, state.costs.len());
    debug_assert_eq!(n, state.chains.len());
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
    apply_permutation::<Option<C>>(&mut state.chains, &idx);
    apply_permutation::<f64>(&mut state.last_ls_cost, &idx);
    apply_permutation::<u32>(&mut state.ls_application_count, &idx);
}
