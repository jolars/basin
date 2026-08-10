use rand_distr::uniform::SampleUniform;

use crate::core::constraint::BoxConstraints;
use crate::core::math::{
    NormSquared, SampleUniformBox, Scalar, ScaledAdd, VectorLen,
};
use crate::core::problem::{CostFunction, Problem};
use crate::core::rng::{ChaCha8Rng, Rng, RngExt, SeedableRng};
use crate::core::solver::Solver;
use crate::core::state::BasicPopulationState;
use crate::core::termination::TerminationReason;
use crate::solver::cma_es::sort_population_ascending;

/// Steady-state real-coded genetic algorithm with BLX-α crossover,
/// negative assortative mating (NAM), BGA mutation, and replace-worst
/// replacement. Faithful to Molina et al. 2010 §4.4 (the SSGA component
/// of MA-LSCh-CMA); ships standalone as a citable baseline and as the
/// outer operator reused by
/// [`MaLsChCma`](crate::solver::ma_ls_ch_cma::MaLsChCma).
///
/// # Algorithm
///
/// [`init`](Solver::init) samples `pop_size` candidates uniformly in
/// the problem box `[lower, upper]`, evaluates each, and sorts the
/// population ascending by cost.
///
/// Each [`next_iter`](Solver::next_iter) produces `offspring_per_step`
/// children:
///
/// 1. **NAM selection**: pick parent 1 uniformly at random; sample
///    `nam_pool − 1` further candidates uniformly and pick the one with
///    the largest Euclidean distance from parent 1 (Fernandes & Rosa
///    2001; Molina 2010 §4.4.3 with `n_ass = nam_pool − 1`).
/// 2. **BLX-α crossover**: per dimension `i`, draw `z_i` uniformly
///    from `[min(p₁ᵢ, p₂ᵢ) − α·d, max(p₁ᵢ, p₂ᵢ) + α·d]` with
///    `d = |p₁ᵢ − p₂ᵢ|`, then clip to `[lowerᵢ, upperᵢ]` (Eshelman &
///    Schaffer 1993; Molina §4.4.2).
/// 3. **BGA mutation**: per dimension with probability `mutation_prob`
///    apply `cᵢ ← cᵢ ± rangᵢ · Σ_{k=0..15} αₖ 2⁻ᵏ`, with
///    `rangᵢ = bga_range_fraction · (upperᵢ − lowerᵢ)`, sign uniform,
///    and `αₖ ∈ {0, 1}` with `P(αₖ = 1) = 1/16`
///    (Mühlenbein & Schlierkamp-Voosen 1993; Molina §4.4.4).
/// 4. **Replace-worst**: evaluate the offspring; if it beats the
///    current population's worst member, take its slot (Molina §4.4.5
///    "standard replacement strategy").
///
/// All four parallel arrays in [`BasicPopulationState`] are re-sorted
/// ascending at the end of each iteration so `state.cost()` always
/// reports the current best.
///
/// # Default parameters
///
/// All defaults follow Molina 2010 §4.4.7:
///
/// | Field | Default | Source |
/// |---|---|---|
/// | `pop_size` | `60` | §4.4.7 |
/// | `blx_alpha` | `0.5` | §4.4.7 |
/// | `nam_pool` | `4` (=`n_ass + 1` with `n_ass = 3`) | §4.4.7 |
/// | `mutation_prob` | `0.125` (per chromosome, applied per gene) | §4.4.7 |
/// | `bga_range_fraction` | `0.1` | §4.4.4 |
/// | `offspring_per_step` | `2` | Common SSGA convention |
///
/// # Reproducibility
///
/// Carries a [`ChaCha8Rng`] seeded from the `seed: u64` passed to
/// [`new`](Self::new): same seed → same iterate trajectory on every
/// platform basin builds for (including `wasm32-unknown-unknown`).
///
/// # Contract
///
/// - **Caller must:** implement [`CostFunction<Param = V, Output = f64>`]
///   *and* [`BoxConstraints<Param = V>`] on the problem. SSGA is a
///   bounded-search method by construction.
/// - **Caller must:** hand in a
///   [`BasicPopulationState::with_size(pop_size)`](crate::BasicPopulationState::with_size)
///   matching the solver's `pop_size`.
/// - **Implementor (this solver) must:** maintain feasibility (every
///   candidate after `init` and every offspring is clipped to the box)
///   and the sorted-by-cost invariant on
///   [`PopulationState`](crate::core::state::PopulationState) at the
///   start and end of every iteration.
///
/// # Termination
///
/// No solver-internal optimality test; SSGA has no canonical
/// fixed-point criterion. Pair with framework criteria
/// [`MaxIter`](crate::core::termination::MaxIter),
/// [`MaxCostEvals`](crate::core::termination::MaxCostEvals),
/// [`MaxTime`](crate::core::termination::MaxTime),
/// [`CostTolerance`](crate::core::termination::CostTolerance), or
/// [`ParamTolerance`](crate::core::termination::ParamTolerance).
/// Replace-worst ensures `state.cost()` is non-increasing, so the
/// cost and param tolerances behave honestly under stochastic dynamics.
///
/// # Backends
///
/// Backend-generic; works with any `V` implementing
/// [`SampleUniformBox`] + [`VectorLen`] + [`ScaledAdd<F>`] +
/// [`NormSquared<F>`] + `Index<usize, Output = F>` +
/// `IndexMut<usize, Output = F>` + `Clone`. With the default `F = f64`
/// that covers `Vec<f64>`, `nalgebra::DVector<f64>` (feature `nalgebra`),
/// `ndarray::Array1<f64>` (feature `ndarray`), and `faer::Col<f64>`
/// (feature `faer`). No matrix operations are required.
///
/// # Examples
///
/// See [`RandomSearch`](crate::RandomSearch) for the population-based
/// `Executor` pattern (a `BasicPopulationState` sized to `pop_size`);
/// `Ssga` likewise requires `BoxConstraints` on the problem.
pub struct Ssga<F = f64> {
    pop_size: usize,
    blx_alpha: F,
    nam_pool: usize,
    mutation_prob: f64,
    bga_range_fraction: F,
    offspring_per_step: usize,
    seed: u64,
    rng: Option<ChaCha8Rng>,
}

impl<F: Scalar> Ssga<F> {
    /// Build a new SSGA with the Molina 2010 §4.4.7 defaults
    /// (`pop_size = 60`, `blx_alpha = 0.5`, `nam_pool = 4`,
    /// `mutation_prob = 0.125`, `bga_range_fraction = 0.1`,
    /// `offspring_per_step = 2`) and a PRNG seeded from `seed`.
    pub fn new(seed: u64) -> Self {
        Self {
            pop_size: 60,
            blx_alpha: F::from_f64(0.5).unwrap(),
            nam_pool: 4,
            mutation_prob: 0.125,
            bga_range_fraction: F::from_f64(0.1).unwrap(),
            offspring_per_step: 2,
            seed,
            rng: None,
        }
    }

    /// Override the population size (default `60`, Molina 2010 §4.4.7).
    ///
    /// # Panics
    ///
    /// Panics if `pop_size < nam_pool`; replace-worst with NAM needs at
    /// least as many individuals as the NAM pool samples.
    pub fn with_pop_size(mut self, pop_size: usize) -> Self {
        assert!(
            pop_size >= self.nam_pool,
            "Ssga requires pop_size >= nam_pool (got pop_size={}, nam_pool={})",
            pop_size,
            self.nam_pool
        );
        self.pop_size = pop_size;
        self
    }

    /// Override the BLX-α parameter (default `0.5`, Molina 2010 §4.4.7).
    /// Larger values widen the child interval beyond the parent span; the
    /// 0.5 default is what Molina argues spreads chromosome variance
    /// rather than contracts it (Nomura & Shimohara 2001 analysis).
    ///
    /// # Panics
    ///
    /// Panics if `alpha < 0`.
    pub fn with_blx_alpha(mut self, alpha: F) -> Self {
        assert!(
            alpha >= F::zero(),
            "Ssga requires blx_alpha >= 0, got {:?}",
            alpha
        );
        self.blx_alpha = alpha;
        self
    }

    /// Override the NAM pool size: total number of individuals sampled
    /// per mating event (parent 1 plus `n_ass = nam_pool − 1`
    /// candidates for parent 2). Default `4` (Molina 2010 §4.4.7 with
    /// `n_ass = 3`).
    ///
    /// # Panics
    ///
    /// Panics if `pool < 2` or `pool > pop_size`; the invariant is
    /// checked in both builders so it holds regardless of call order.
    /// A pool of 2 degenerates to plain uniform selection (no negative
    /// assortative bias).
    pub fn with_nam_pool(mut self, pool: usize) -> Self {
        assert!(pool >= 2, "Ssga requires nam_pool >= 2, got {}", pool);
        assert!(
            pool <= self.pop_size,
            "Ssga requires nam_pool <= pop_size (got nam_pool={}, pop_size={})",
            pool,
            self.pop_size
        );
        self.nam_pool = pool;
        self
    }

    /// Override the per-gene mutation probability (default `0.125`,
    /// Molina 2010 §4.4.7).
    ///
    /// # Panics
    ///
    /// Panics if `p` is not in `[0, 1]`.
    pub fn with_mutation_prob(mut self, p: f64) -> Self {
        assert!(
            (0.0..=1.0).contains(&p),
            "Ssga requires mutation_prob in [0, 1], got {}",
            p
        );
        self.mutation_prob = p;
        self
    }

    /// Override the BGA range fraction (default `0.1`, Molina 2010
    /// §4.4.4: `rang_i = 0.1 · (upper_i − lower_i)`).
    ///
    /// # Panics
    ///
    /// Panics if `f <= 0`.
    pub fn with_bga_range_fraction(mut self, f: F) -> Self {
        assert!(
            f > F::zero(),
            "Ssga requires bga_range_fraction > 0, got {:?}",
            f
        );
        self.bga_range_fraction = f;
        self
    }

    /// Number of offspring produced per [`next_iter`](crate::Solver::next_iter)
    /// (default `2`).
    /// Each offspring is generated, evaluated, and considered for
    /// replace-worst in turn.
    ///
    /// # Panics
    ///
    /// Panics if `n == 0`.
    pub fn with_offspring_per_step(mut self, n: usize) -> Self {
        assert!(n >= 1, "Ssga requires offspring_per_step >= 1");
        self.offspring_per_step = n;
        self
    }
}

/// One BLX-α crossover offspring (Eshelman & Schaffer 1993): per
/// dimension `i`, draw `z_i ~ U(min − α·d, max + α·d)` with
/// `min = min(p1[i], p2[i])`, `max = max(p1[i], p2[i])`,
/// `d = max − min`, then clip to `[lower[i], upper[i]]`.
///
/// `pub(crate)` so [`MaLsChCma`](crate::solver::ma_ls_ch_cma::MaLsChCma)
/// reuses the operator directly; not a stable public surface.
pub(crate) fn blx_alpha_crossover<V, F, R>(
    p1: &V,
    p2: &V,
    alpha: F,
    lower: &V,
    upper: &V,
    rng: &mut R,
) -> V
where
    F: Scalar,
    V: VectorLen
        + Clone
        + SampleUniformBox
        + std::ops::Index<usize, Output = F>
        + std::ops::IndexMut<usize, Output = F>,
    R: Rng + ?Sized,
{
    let n = p1.vec_len();
    // Build a per-component box [blx_lo, blx_hi] for the BLX interval,
    // pre-clipped to [lower, upper]. SampleUniformBox samples one F
    // per component from the inclusive range, which matches Eshelman &
    // Schaffer's definition.
    let mut blx_lo = lower.clone();
    let mut blx_hi = upper.clone();
    let half = F::from_f64(0.5).unwrap();
    for i in 0..n {
        let a = p1[i];
        let b = p2[i];
        let (mn, mx) = if a < b { (a, b) } else { (b, a) };
        let d = mx - mn;
        let lo = (mn - alpha * d).max(lower[i]);
        let hi = (mx + alpha * d).min(upper[i]);
        // Bounds can invert if α·d pushes both endpoints past opposite
        // sides of the global box (rare with α=0.5 and reasonable
        // populations, but possible). Pin to the midpoint of [lower,
        // upper] in that degenerate case.
        if hi >= lo {
            blx_lo[i] = lo;
            blx_hi[i] = hi;
        } else {
            let mid = half * (lower[i] + upper[i]);
            blx_lo[i] = mid;
            blx_hi[i] = mid;
        }
    }
    V::sample_uniform_box(&blx_lo, &blx_hi, rng)
}

/// Negative-assortative-mating selection (Fernandes & Rosa 2001;
/// Molina 2010 §4.4.3): pick parent 1 uniformly at random from `pop`;
/// sample `pool − 1` further candidates uniformly and return the index
/// whose Euclidean distance from parent 1 is largest.
///
/// `pub(crate)` so [`MaLsChCma`](crate::solver::ma_ls_ch_cma::MaLsChCma)
/// reuses the operator directly; not a stable public surface.
pub(crate) fn nam_select<V, F, R>(
    pop: &[V],
    pool: usize,
    rng: &mut R,
) -> (usize, usize)
where
    F: Scalar,
    V: Clone + ScaledAdd<F> + NormSquared<F>,
    R: Rng + ?Sized,
{
    debug_assert!(pop.len() >= 2, "nam_select needs at least 2 individuals");
    debug_assert!(pool >= 2, "nam_select needs nam_pool >= 2");
    debug_assert!(
        pool <= pop.len(),
        "nam_select needs nam_pool <= population size (got pool={}, pop={})",
        pool,
        pop.len()
    );
    let n = pop.len();
    let p1 = rng.random_range(0..n);
    // Seed `best` with the first sampled candidate so we always return
    // a valid index even if every distance happens to be zero (e.g.
    // identical-population pathology).
    let first = rng.random_range(0..n);
    let mut best = first;
    let mut best_d_sq = {
        let mut diff = pop[p1].clone();
        diff.scaled_add(-F::one(), &pop[first]);
        diff.norm_squared()
    };
    for _ in 1..(pool - 1) {
        let c = rng.random_range(0..n);
        let mut diff = pop[p1].clone();
        diff.scaled_add(-F::one(), &pop[c]);
        let d_sq = diff.norm_squared();
        if d_sq > best_d_sq {
            best_d_sq = d_sq;
            best = c;
        }
    }
    (p1, best)
}

/// BGA mutation operator (Mühlenbein & Schlierkamp-Voosen 1993, Molina
/// 2010 §4.4.4) applied in place. For each gene `i`:
///
/// - With probability `prob`, perturb `child[i] ← child[i] ± rang_i · S`
///   where:
///   - sign is `±` with probability 0.5 each,
///   - `rang_i = range_fraction · (upper[i] − lower[i])`,
///   - `S = Σ_{k=0..15} α_k · 2^{−k}` with each `α_k ∈ {0, 1}`
///     independently drawn with `P(α_k = 1) = 1/16`.
/// - Clip the result to `[lower[i], upper[i]]`.
///
/// `pub(crate)` so [`MaLsChCma`](crate::solver::ma_ls_ch_cma::MaLsChCma)
/// reuses the operator directly; not a stable public surface.
pub(crate) fn bga_mutate_in_place<V, F, R>(
    child: &mut V,
    lower: &V,
    upper: &V,
    prob: f64,
    range_fraction: F,
    rng: &mut R,
) where
    F: Scalar,
    V: VectorLen
        + std::ops::Index<usize, Output = F>
        + std::ops::IndexMut<usize, Output = F>,
    R: Rng + ?Sized,
{
    let n = child.vec_len();
    // BGA's α_k sum `Σ_{k=0..15} α_k · 2^{−k}` and its 1/16 Bernoulli gate
    // are F-independent, so keep them as f64 and convert the final sum to F
    // once, rather than redo `from_f64` per loop body.
    for i in 0..n {
        if rng.random::<f64>() >= prob {
            continue;
        }
        let sign_pos = rng.random::<f64>() < 0.5;
        let rang = range_fraction * (upper[i] - lower[i]);
        let mut s_f64 = 0.0_f64;
        for k in 0..16 {
            if rng.random::<f64>() < 1.0 / 16.0 {
                s_f64 += (-(k as f64)).exp2();
            }
        }
        let s = F::from_f64(s_f64).unwrap();
        let delta = if sign_pos { rang * s } else { -(rang * s) };
        let v = child[i] + delta;
        // `Float` has no `clamp`; `max(lo).min(hi)` matches `f64::clamp`
        // on finite, ordered bounds.
        child[i] = v.max(lower[i]).min(upper[i]);
    }
}

/// Replace-worst step: locate the current-worst slot (linear scan
/// since intra-iteration replacements can break any prior sort) and,
/// if `c_child` strictly improves on it, overwrite the slot and return
/// its index; otherwise return `None` (the offspring is discarded).
///
/// NaN costs are treated as worse than any finite cost so a single bad
/// evaluation can be displaced by a finite child. Callers are expected
/// to re-sort the population once per outer iteration after all
/// offspring have been processed.
///
/// `pub(crate)` so [`MaLsChCma`](crate::solver::ma_ls_ch_cma::MaLsChCma)
/// reuses the operator directly; not a stable public surface.
pub(crate) fn replace_worst_if_better<V, F: Scalar>(
    pop: &mut [V],
    costs: &mut [F],
    child: V,
    c_child: F,
) -> Option<usize> {
    let mut worst_idx = 0;
    let mut worst_cost = costs[0];
    for (i, &c) in costs.iter().enumerate().skip(1) {
        let is_worse = if worst_cost.is_nan() {
            // Already as bad as it gets.
            false
        } else if c.is_nan() {
            true
        } else {
            c > worst_cost
        };
        if is_worse {
            worst_idx = i;
            worst_cost = c;
        }
    }
    let replace = if worst_cost.is_nan() {
        !c_child.is_nan()
    } else {
        c_child < worst_cost
    };
    if replace {
        pop[worst_idx] = child;
        costs[worst_idx] = c_child;
        Some(worst_idx)
    } else {
        None
    }
}

impl<P, V, F> Solver<P, BasicPopulationState<V, F>> for Ssga<F>
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
{
    type Error = P::Error;

    fn init(
        &mut self,
        problem: &mut Problem<P>,
        mut state: BasicPopulationState<V, F>,
    ) -> Result<BasicPopulationState<V, F>, Self::Error> {
        let lo = problem.inner().lower().clone();
        let hi = problem.inner().upper().clone();
        let mut rng = ChaCha8Rng::seed_from_u64(self.seed);
        // Always reseed the population from the solver's RNG so the
        // trajectory is reproducible regardless of which
        // BasicPopulationState constructor the caller used (with_size
        // vs. from_population). Same pattern as RandomSearch.
        state.candidates.clear();
        state.costs.clear();
        for _ in 0..self.pop_size {
            let x = V::sample_uniform_box(&lo, &hi, &mut rng);
            let c = problem.cost(&x)?;
            state.candidates.push(x);
            state.costs.push(c);
        }
        sort_population_ascending(&mut state.candidates, &mut state.costs);
        self.rng = Some(rng);
        Ok(state)
    }

    fn next_iter(
        &mut self,
        problem: &mut Problem<P>,
        mut state: BasicPopulationState<V, F>,
    ) -> Result<
        (BasicPopulationState<V, F>, Option<TerminationReason>),
        Self::Error,
    > {
        let lo = problem.inner().lower().clone();
        let hi = problem.inner().upper().clone();
        let rng = self
            .rng
            .as_mut()
            .expect("Ssga::init must run before next_iter");

        for _ in 0..self.offspring_per_step {
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
            replace_worst_if_better(
                &mut state.candidates,
                &mut state.costs,
                child,
                c_child,
            );
        }

        sort_population_ascending(&mut state.candidates, &mut state.costs);
        Ok((state, None))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand_chacha::ChaCha8Rng;

    #[test]
    fn blx_alpha_samples_lie_in_expected_interval_unclipped() {
        // Parents 0 and 1 in 1D, α = 0.5, generous global box. Children
        // must land in [-0.5, 1.5].
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let p1 = vec![0.0];
        let p2 = vec![1.0];
        let lo = vec![-10.0];
        let hi = vec![10.0];
        let mut min_seen = f64::INFINITY;
        let mut max_seen = f64::NEG_INFINITY;
        for _ in 0..10_000 {
            let c = blx_alpha_crossover(&p1, &p2, 0.5, &lo, &hi, &mut rng);
            assert!(
                (-0.5..=1.5).contains(&c[0]),
                "child {} out of [-0.5, 1.5]",
                c[0]
            );
            min_seen = min_seen.min(c[0]);
            max_seen = max_seen.max(c[0]);
        }
        // 10k samples cover the interval [-0.5, 1.5] well.
        assert!(min_seen < -0.4, "min {} not near -0.5", min_seen);
        assert!(max_seen > 1.4, "max {} not near 1.5", max_seen);
    }

    #[test]
    fn blx_alpha_clips_to_global_bounds() {
        // Tight global box [0, 1] forces clipping even though the BLX
        // interval would otherwise extend to [-0.5, 1.5].
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        let p1 = vec![0.0];
        let p2 = vec![1.0];
        let lo = vec![0.0];
        let hi = vec![1.0];
        for _ in 0..2_000 {
            let c = blx_alpha_crossover(&p1, &p2, 0.5, &lo, &hi, &mut rng);
            assert!(
                (0.0..=1.0).contains(&c[0]),
                "child {} outside global box [0, 1]",
                c[0]
            );
        }
    }

    #[test]
    fn nam_picks_farthest_in_pool_deterministically_when_pool_covers_population()
     {
        // 4 individuals on a 1D line; pool=4 means every candidate is
        // sampled with high probability (uniform sampling with
        // replacement → most runs the farthest from p1 is picked).
        // Force determinism: pool size = pop size with seed where the
        // farthest gets sampled.
        let pop = vec![vec![0.0], vec![1.0], vec![2.0], vec![10.0]];
        // Run many seeds; for each, parent 1 is uniform; the farthest
        // (always index 3 if parent 1 != 3) should be picked across
        // most seeds. Smoke-check that NAM at least sometimes returns
        // the farthest.
        let mut hits_farthest = 0;
        for seed in 0..200 {
            let mut rng = ChaCha8Rng::seed_from_u64(seed);
            let (p1, p2) = nam_select(&pop, 4, &mut rng);
            if p1 != 3 && p2 == 3 {
                hits_farthest += 1;
            }
        }
        // p1 != 3 ~ 75% of seeds; conditional on that, p2 == 3 should
        // happen most of the time with pool=4 over 4 individuals.
        assert!(
            hits_farthest > 80,
            "NAM rarely picks the farthest: {} / 200",
            hits_farthest
        );
    }

    #[test]
    fn replace_worst_only_when_strictly_better() {
        let mut pop = vec![vec![0.0], vec![1.0], vec![2.0]];
        let mut costs = vec![0.0, 1.0, 2.0];
        // Tie with the worst → no replacement.
        let r = replace_worst_if_better(&mut pop, &mut costs, vec![5.0], 2.0);
        assert!(r.is_none());
        assert_eq!(pop[2], vec![2.0]);
        // Strictly better → replace.
        let r = replace_worst_if_better(&mut pop, &mut costs, vec![5.0], 1.5);
        assert_eq!(r, Some(2));
        assert_eq!(pop[2], vec![5.0]);
        assert_eq!(costs[2], 1.5);
    }

    #[test]
    fn replace_worst_treats_nan_as_worse_than_any_finite() {
        let mut pop = vec![vec![0.0], vec![1.0]];
        let mut costs = vec![0.0, f64::NAN];
        let r = replace_worst_if_better(&mut pop, &mut costs, vec![5.0], 100.0);
        assert_eq!(r, Some(1));
        assert_eq!(costs[1], 100.0);
    }

    #[test]
    fn bga_mutation_prob_zero_leaves_unchanged() {
        let mut rng = ChaCha8Rng::seed_from_u64(99);
        let mut child = vec![0.5, 0.5, 0.5];
        let lo = vec![0.0, 0.0, 0.0];
        let hi = vec![1.0, 1.0, 1.0];
        bga_mutate_in_place(&mut child, &lo, &hi, 0.0, 0.1, &mut rng);
        assert_eq!(child, vec![0.5, 0.5, 0.5]);
    }

    #[test]
    fn bga_mutation_prob_one_respects_bounds() {
        let mut rng = ChaCha8Rng::seed_from_u64(99);
        let lo = vec![0.0; 8];
        let hi = vec![1.0; 8];
        for _ in 0..200 {
            let mut child = vec![0.5; 8];
            bga_mutate_in_place(&mut child, &lo, &hi, 1.0, 0.1, &mut rng);
            for (i, &v) in child.iter().enumerate() {
                assert!(
                    v >= lo[i] && v <= hi[i],
                    "child[{}] = {} outside [0, 1]",
                    i,
                    v
                );
            }
        }
    }
}
