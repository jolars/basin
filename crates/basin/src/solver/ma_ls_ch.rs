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
//! MA-SW-Chains, CEC 2010) is `MaLsCh<V, SolisWets<V>>`.

use std::marker::PhantomData;

use crate::core::constraint::BoxConstraints;
use crate::core::inner::ResumableInner;
use crate::core::math::{
    NormSquared, SampleUniformBox, Scalar, ScaledAdd, VectorLen,
};
use crate::core::problem::{CostFunction, Problem};
use crate::core::rng::{ChaCha8Rng, RngExt, SeedableRng};
use crate::core::solver::Solver;
use crate::core::state::PopulationProgress;
use rand::distr::uniform::SampleUniform;
// Cycle-following in-place permutation: after the call,
// `slice[i] = original[idx[i]]`.
use crate::solver::cma_es::{apply_permutation, nan_last_cmp};
use crate::solver::ssga::{
    bga_mutate_in_place, blx_alpha_crossover, nam_select,
    replace_worst_if_better,
};

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
struct ChainHistory<LS, S, F> {
    chains: Vec<Option<(LS, S)>>,
    last_ls_cost: Vec<F>,
    ls_application_count: Vec<u32>,
}
impl<LS, S, F> Default for ChainHistory<LS, S, F> {
    fn default() -> Self {
        Self {
            chains: Vec::new(),
            last_ls_cost: Vec::new(),
            ls_application_count: Vec::new(),
        }
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
/// `MaLsCh<V, SolisWets<V>>`). Both specialized aliases have a `new(seed)`
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
/// 5. **Drive the inner.** Each segment uses a fresh
///    `max_cost_evals(ls_intensity)` budget and the settings supplied by
///    [`ResumableInner::configure_segment`]. New chains initialize once unless
///    their seeder supplied an initialized pair. Stored chains skip
///    [`Solver::init`] and retain their evolving model across segments.
/// 6. **Aggregate, route failures, write back.** Per CONTRIBUTING.md
///    "Solver composition" rules:
///    - The shared problem wrapper counts every inner call in its original
///      category; progress preserves all six categories.
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
/// | `initial_scale_fallback` | `1.0` | when min-neighbor distance is 0 or non-finite |
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
/// - **Caller must:** implement [`CostFunction<Param = V, Output = F>`]
///   *and* [`BoxConstraints<Param = V>`] on the problem. The SSGA needs
///   the box for initial sampling, BLX clipping, and BGA range; the
///   per-individual LS inner does not see the box (so the inner is
///   *unbounded*: chain individuals can drift outside the box and be
///   discarded only via the SSGA replace-worst feedback loop). This
///   matches Molina 2010 §4.4.6.
/// - **Caller must:** supply [`PopulationProgress::empty`] for a sampled
///   population, or [`PopulationProgress::from_population`] with the configured
///   member count and dimension. Fresh initialization projects explicit seeds
///   into the finite, ordered box before evaluation.
/// - **Implementor must:** keep population records sorted and each solver-owned
///   chain aligned with its member at every publication boundary.
///
/// # Ownership and continuation
///
/// Shared progress owns the population and historical objective incumbent.
/// The solver owns the outer RNG, local-search prototype, retained solver/state
/// pairs, and eligibility history. [`ls_application_count`](Self::ls_application_count)
/// and [`chain`](Self::chain) inspect these per-member diagnostics without
/// transferring ownership. A fresh run reevaluates its members, discards every
/// chain, and restarts the configured RNG. Exact continuation retains solver,
/// progress, and counts together and skips initialization. With `serde`, the
/// complete chain store serializes when `LS`, `LS::State`, and `F` do; the outer
/// progress additionally requires serializable parameters. Legacy MA-LS state
/// payloads are incompatible with this layout.
///
/// # Termination
///
/// No solver-internal optimality test. Pair with framework criteria,
/// typically [`max_cost_evals`](crate::Executor::max_cost_evals) for budget control. Chain segments
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
/// so effective coverage is set by the chain operator. Both shipped configurations
/// support `Vec<F>`, nalgebra `DVector<F>`, ndarray `Array1<F>`, and faer `Col<F>`,
/// for `F = f32` or `f64`. CMA chains additionally need the matrix capabilities
/// of [`CmaEs`](crate::CmaEs); Solis-Wets chains need no matrix type. Custom
/// chain operators can narrow backend support.
///
/// # References
///
/// - Molina, D., Lozano, M., García-Martínez, C., and Herrera, F.
///   (2010). "Memetic algorithms for continuous optimisation based on
///   local search chains." *Evolutionary Computation*, 18(1), 27-63.
///   <https://doi.org/10.1162/evco.2010.18.1.18102>
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(
    feature = "serde",
    serde(bound(
        serialize = "LS: serde::Serialize, LS::State: serde::Serialize, F: serde::Serialize",
        deserialize = "LS: serde::Deserialize<'de>, LS::State: serde::Deserialize<'de>, F: serde::Deserialize<'de>"
    ))
)]
pub struct MaLsCh<V, LS, F: Scalar = f64>
where
    LS: ResumableInner<V, F>,
{
    pop_size: usize,
    blx_alpha: F,
    nam_pool: usize,
    mutation_prob: F,
    bga_range_fraction: F,
    ls_intensity: u64,
    ls_improvement_threshold: F,
    nfrec: Option<u64>,
    initial_scale_fallback: F,
    seed: u64,
    rng: Option<ChaCha8Rng>,
    /// LS-operator configuration prototype: hyperparameters are copied
    /// into each fresh chain via
    /// [`ResumableInner::seed_chain`]; its own RNG is never drawn.
    pub(crate) ls: LS,
    history: ChainHistory<LS, LS::State, F>,
    _phantom: PhantomData<V>,
}

impl<V, LS, F: Scalar> MaLsCh<V, LS, F>
where
    LS: ResumableInner<V, F>,
{
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
            blx_alpha: F::from_f64(0.5).unwrap(),
            nam_pool: 4,
            mutation_prob: F::from_f64(0.125).unwrap(),
            bga_range_fraction: F::from_f64(0.1).unwrap(),
            ls_intensity: 300,
            ls_improvement_threshold: F::from_f64(1e-8).unwrap(),
            nfrec: None,
            initial_scale_fallback: F::one(),
            seed,
            rng: None,
            ls,
            history: ChainHistory::default(),
            _phantom: PhantomData,
        }
    }

    /// Number of completed local-search applications for the current member at `i`.
    /// Indices follow the published population order. Panics for an invalid index.
    pub fn ls_application_count(&self, i: usize) -> u32 {
        self.history.ls_application_count[i]
    }

    /// Borrow the retained solver and progress for a member's resumable chain.
    /// `None` means no chain is retained at that index.
    pub fn chain(&self, i: usize) -> Option<(&LS, &LS::State)> {
        self.history
            .chains
            .get(i)?
            .as_ref()
            .map(|(solver, state)| (solver, state))
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
    pub fn with_blx_alpha(mut self, alpha: F) -> Self {
        assert!(
            alpha >= F::zero(),
            "blx_alpha must be >= 0, got {:?}",
            alpha
        );
        self.blx_alpha = alpha;
        self
    }

    /// Override the NAM pool size (default `4`).
    ///
    /// # Panics
    ///
    /// Panics if `pool < 2` or `pool > pop_size`; the invariant is
    /// checked in both builders so it holds regardless of call order.
    pub fn with_nam_pool(mut self, pool: usize) -> Self {
        assert!(pool >= 2, "nam_pool must be >= 2, got {:?}", pool);
        assert!(
            pool <= self.pop_size,
            "MaLsCh requires nam_pool <= pop_size (got nam_pool={}, pop_size={})",
            pool,
            self.pop_size
        );
        self.nam_pool = pool;
        self
    }

    /// Override the per-gene BGA mutation probability (default `0.125`).
    ///
    /// # Panics
    ///
    /// Panics if `p` is not in `[0, 1]`.
    pub fn with_mutation_prob(mut self, p: F) -> Self {
        assert!(
            (F::zero()..=F::one()).contains(&p),
            "mutation_prob must be in [0, 1], got {:?}",
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
    pub fn with_bga_range_fraction(mut self, f: F) -> Self {
        assert!(f > F::zero(), "bga_range_fraction must be > 0, got {:?}", f);
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
        assert!(istr >= 1, "ls_intensity must be >= 1, got {:?}", istr);
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
    pub fn with_ls_improvement_threshold(mut self, delta: F) -> Self {
        assert!(
            delta >= F::zero(),
            "ls_improvement_threshold must be >= 0, got {:?}",
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
        assert!(n >= 1, "nfrec must be >= 1, got {:?}", n);
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
    pub fn with_initial_scale_fallback(mut self, scale: F) -> Self {
        assert!(
            scale > F::zero(),
            "initial_scale_fallback must be > 0, got {:?}",
            scale
        );
        self.initial_scale_fallback = scale;
        self
    }
}

/// Compute `0.5 · min_{j ≠ i} ‖candidates[i] − candidates[j]‖₂`, the
/// per-individual scale-init formula from Molina 2010 §4.4.6. Returns
/// `None` if there's no other individual (singleton population).
fn sigma_init_for<V, F: Scalar>(candidates: &[V], i: usize) -> Option<F>
where
    V: Clone + ScaledAdd<F> + NormSquared<F>,
{
    if candidates.len() < 2 {
        return None;
    }
    let mut best_sq = F::infinity();
    for (j, x) in candidates.iter().enumerate() {
        if j == i {
            continue;
        }
        let mut diff = candidates[i].clone();
        diff.scaled_add(-F::one(), x);
        let d_sq = diff.norm_squared();
        if d_sq < best_sq {
            best_sq = d_sq;
        }
    }
    Some(F::from_f64(0.5).unwrap() * best_sq.sqrt())
}

impl<P, V, LS, F> Solver<P, PopulationProgress<V, F>> for MaLsCh<V, LS, F>
where
    F: Scalar + SampleUniform,
    P: CostFunction<Param = V, Output = F> + BoxConstraints<Param = V>,
    V: VectorLen
        + Clone
        + SampleUniformBox
        + ScaledAdd<F>
        + NormSquared<F>
        + std::ops::Index<usize, Output = F>
        + std::ops::IndexMut<usize, Output = F>,
    LS: ResumableInner<V, F>
        + Solver<P, <LS as ResumableInner<V, F>>::State, Error = P::Error>,
{
    type Error = P::Error;

    fn init(
        &mut self,
        problem: &mut Problem<P>,
        mut state: PopulationProgress<V, F>,
    ) -> Result<PopulationProgress<V, F>, Self::Error> {
        let lo = problem.inner().lower().clone();
        let hi = problem.inner().upper().clone();
        let mut rng = ChaCha8Rng::seed_from_u64(self.seed);

        state.reset();
        self.history = ChainHistory::default();
        super::population::prepare_population(
            &mut state.candidates,
            &lo,
            &hi,
            self.pop_size,
            &mut rng,
        );
        for x in &state.candidates {
            state.costs.push(problem.cost(x)?);
            self.history.chains.push(None);
            self.history.last_ls_cost.push(F::infinity());
            self.history.ls_application_count.push(0);
        }
        sort_parallel_arrays(&mut state, &mut self.history);

        self.rng = Some(rng);
        Ok(state)
    }

    fn next_iter(
        &mut self,
        problem: &mut Problem<P>,
        mut state: PopulationProgress<V, F>,
    ) -> Result<crate::SolverStep<PopulationProgress<V, F>>, Self::Error> {
        let history = &mut self.history;
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
        while problem.counts().cost_evals - phase_start_counts.cost_evals
            < nfrec
        {
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
                self.mutation_prob.to_f64().unwrap(),
                self.bga_range_fraction,
                rng,
            );
            let c_child = problem.cost(&child)?;
            if let Some(replaced_idx) = replace_worst_if_better(
                &mut state.candidates,
                &mut state.costs,
                child,
                c_child,
            ) {
                // The displaced individual's chain (if any) is orphaned:
                // the new genome is a fresh point that should start its
                // own chain on first LS pick.
                history.chains[replaced_idx] = None;
                history.last_ls_cost[replaced_idx] = F::infinity();
                history.ls_application_count[replaced_idx] = 0;
            }
        }
        sort_parallel_arrays(&mut state, history);

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
        let mut best_cost_in_s_ls = F::infinity();
        for i in 0..state.candidates.len() {
            let eligible = history.ls_application_count[i] == 0
                || (history.last_ls_cost[i] - state.costs[i]
                    >= self.ls_improvement_threshold);
            if eligible && state.costs[i] < best_cost_in_s_ls {
                best_cost_in_s_ls = state.costs[i];
                c_ls = Some(i);
            }
        }
        // Molina §4.3: when |S_LS| = 0, apply LS to the best individual
        // unconditionally.
        let c_ls = c_ls.unwrap_or(0);

        // -- Phase 3: resume or construct the inner operator. --
        let (mut ls, inner_state, initialized) =
            match history.chains[c_ls].take() {
                Some((ls, mut s)) => {
                    // Local budget reset. `run_loop_with_control` already snapshots the
                    // wrapper at entry so the inner state's `cost_evals`
                    // measures per-segment work, but the iteration counter
                    // is the inner's responsibility and the `MaxCostEvals`
                    // criterion in Phase 4 reads `state.cost_evals()`,
                    // which is the wrapper-mirrored per-run value.
                    // `prepare_resume` resets `iter` so the chain restarts
                    // at iter 0; the `run_loop_with_control` baseline takes care of the
                    // eval counter.
                    ls.prepare_resume(&mut s);
                    (ls, s, true)
                }
                None => {
                    // Non-finite scales happen when every pairwise distance
                    // overflows (astronomically wide boxes); fall back the
                    // same way as for a duplicated point.
                    let scale = sigma_init_for(&state.candidates, c_ls)
                        .filter(|s| s.is_finite() && *s > F::zero())
                        .unwrap_or(self.initial_scale_fallback);
                    let derived_seed = rng.random::<u64>();
                    let (ls, state) = self.ls.seed_chain(
                        &state.candidates[c_ls],
                        state.costs[c_ls],
                        scale,
                        derived_seed,
                    );
                    let initialized = ls.seeded_chain_is_initialized();
                    (ls, state, initialized)
                }
            };

        // -- Phase 4: drive the inner. --
        // Build per-call criteria so `MaxCostEvals` doesn't leak state
        // between chain segments (it's stateless, but the
        // `InnerExecutor` reuse pattern doesn't fit here since we hold
        // a different operator instance per individual). Allocation
        // cost is a few boxes per chain segment, negligible against
        // I_str evals. Budget first, then the operator's per-segment
        // convergence settings built from the segment's starting state.
        // The execution budget is observed before convergence.
        let mut control = crate::RunControl::new()
            .max_iter(u64::MAX)
            .max_cost_evals(self.ls_intensity);
        ls.configure_segment(&inner_state, &mut control);
        let inner_result = crate::core::executor::run_segment_with_control(
            problem,
            inner_state,
            &mut ls,
            &mut control,
            initialized,
        )?;

        // -- Phase 5: route failures, write back. --
        // Same-problem composition: inner evals already flowed through the
        // outer wrapper, so shared progress preserves their raw categories. `SolverFailed` is the only failure
        // reason; other reasons (`MaxCostEvals` from our budget, the
        // operator's own tolerances) are clean stops the outer consumes.
        if !inner_result
            .report
            .termination
            .can_continue_as_inner(crate::PartialResultPolicy::Consume)
        {
            // Leave the chain dropped so a future pick would restart.
            return Ok(crate::SolverStep::from((
                state,
                inner_result.report.into_outer_termination(
                    crate::PartialResultPolicy::Consume,
                ),
            )));
        }

        // Adopt the chain's best *evaluated* point (xbest), not
        // whatever `param()`/`cost()` now report (CMA-ES reports the
        // distribution mean); the memetic algorithm wants the best
        // refinement found. The configured chain may search outside the box.
        let new_cost = inner_result.best_cost();
        // Conditional write-back: only adopt the LS result if it
        // improves on the current cost. Strict Molina §4.3 step 10 is
        // unconditional, but a conditional update is safer (CMA-ES is
        // genuinely non-monotone over a chain segment) and matches the
        // Rmalschains R package's behavior.
        let pre_segment_cost = state.costs[c_ls];
        if !new_cost.is_nan()
            && new_cost < F::infinity()
            && (state.costs[c_ls].is_nan() || new_cost < state.costs[c_ls])
        {
            state.candidates[c_ls] = inner_result.best_param().clone();
            state.costs[c_ls] = new_cost;
        }
        // Record the cost this segment started from: eligibility (Phase
        // 2) reads `last_ls_cost - costs`, the improvement obtained by
        // the previous LS application (Molina §4.3 step 1b).
        history.last_ls_cost[c_ls] = pre_segment_cost;
        history.ls_application_count[c_ls] =
            history.ls_application_count[c_ls].saturating_add(1);
        // Keep the chain only when the segment cleared δ_LS_min.
        // Rmalschains removes exhausted chains (`m_memory->remove`), so
        // a future pick reseeds at a fresh scale instead of resuming a
        // converged operator that would burn the whole segment budget
        // making no progress.
        if pre_segment_cost - state.costs[c_ls] >= self.ls_improvement_threshold
        {
            history.chains[c_ls] = Some((ls, inner_result.state));
        }

        // -- Phase 6: resort all parallel arrays jointly. --
        sort_parallel_arrays(&mut state, history);

        Ok(crate::SolverStep::from((state, None)))
    }
}

/// Keep each chain and its eligibility history aligned with its published member.
fn sort_parallel_arrays<V, LS, S, F: Scalar>(
    state: &mut PopulationProgress<V, F>,
    history: &mut ChainHistory<LS, S, F>,
) {
    let n = state.candidates.len();
    debug_assert_eq!(n, state.costs.len());
    debug_assert_eq!(n, history.chains.len());
    debug_assert_eq!(n, history.last_ls_cost.len());
    debug_assert_eq!(n, history.ls_application_count.len());
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&i, &j| nan_last_cmp(&state.costs[i], &state.costs[j]));
    apply_permutation(&mut state.candidates, &idx);
    apply_permutation(&mut state.costs, &idx);
    apply_permutation(&mut history.chains, &idx);
    apply_permutation(&mut history.last_ls_cost, &idx);
    apply_permutation(&mut history.ls_application_count, &idx);
    state.select_best_member();
}
