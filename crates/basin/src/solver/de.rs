use crate::core::constraint::BoxConstraints;
use crate::core::math::{SampleUniformBox, ScaleInPlace, ScaledAdd, VectorLen};
use crate::core::problem::{CostFunction, Problem};
use crate::core::rng::{ChaCha8Rng, Rng, RngExt, SeedableRng};
use crate::core::solver::Solver;
use crate::core::state::BasicPopulationState;
use crate::core::termination::TerminationReason;
use crate::solver::cma_es::sort_population_ascending;

/// Differential Evolution (DE/rand/1/bin) from Storn & Price 1997 (*A
/// Simple and Efficient Heuristic for Global Optimization over
/// Continuous Spaces*, J. Global Optim. 11:341–359).
///
/// Stochastic, derivative-free, population-based — the canonical
/// black-box optimizer for rugged continuous landscapes with few
/// hyperparameters and no covariance model. Sits next to
/// [`Ssga`](crate::solver::Ssga) (steady-state real-coded GA) and
/// [`CmaEs`](crate::solver::CmaEs) (Gaussian model) in basin's
/// global / stochastic family.
///
/// # Algorithm
///
/// One [`next_iter`](Solver::next_iter) = one full generation =
/// `pop_size` cost evaluations.
///
/// For each `i ∈ {0, …, NP − 1}`:
///
/// 1. **Mutation (DE/rand/1).** Pick three indices
///    `r1, r2, r3 ∈ {0, …, NP − 1} \ {i}`, pairwise distinct, and form
///    the donor `v = x[r1] + F · (x[r2] − x[r3])`.
/// 2. **Bound repair (reinit-per-coord, DEoptim style).** For each
///    coordinate `j` with `v[j] ∉ [lower[j], upper[j]]`, resample
///    `v[j] ~ U(lower[j], upper[j])`. Preserves diversity (no clipping
///    pile-up at the boundary).
/// 3. **Binomial crossover.** Pick `j_rand ∈ {0, …, D − 1}` uniformly,
///    then build the trial `u[j] = v[j]` if `j == j_rand` or
///    `rng.uniform() < CR`, else `u[j] = x[i][j]`. The `j_rand`
///    guarantee ensures `u ≠ x[i]` even when `CR = 0`.
///
/// All `NP` trials are built from the *unmodified* current generation
/// (synchronous DE — same as DEoptim and
/// [`scipy.optimize.differential_evolution`]). After evaluation, each
/// trial replaces its target if its cost is not worse
/// (`c_trial ≤ c_x[i]`), then the population is re-sorted ascending so
/// `state.cost()` always reports the current best.
///
/// # Default parameters
///
/// | Field | Default | Source |
/// |---|---|---|
/// | `pop_size` | [`default_pop_size(D)`](Self::default_pop_size) `= max(4, 10·D)` | Storn & Price's `10·D` rule |
/// | `F` (mutation scale) | `0.8` | Modern convention; matches `scipy.optimize.differential_evolution`'s upper-mutation bound |
/// | `CR` (crossover probability) | `0.9` | Modern convention for non-separable problems |
///
/// Use [`with_pop_size`](Self::with_pop_size), [`with_f`](Self::with_f),
/// [`with_cr`](Self::with_cr) to override.
///
/// # Reproducibility
///
/// Carries a [`ChaCha8Rng`] seeded from the `seed: u64` passed to
/// [`new`](Self::new) — same seed → same iterate trajectory on every
/// platform basin builds for (including `wasm32-unknown-unknown`).
///
/// # Contract
///
/// - **Caller must:** implement [`CostFunction<Param = V, Output = f64>`]
///   *and* [`BoxConstraints<Param = V>`] on the problem. DE is a
///   bounded-search method by construction (uniform initialization in
///   the box and per-coordinate bound repair both require `lower` /
///   `upper`).
/// - **Caller must:** hand in a
///   [`BasicPopulationState::with_size(λ)`](crate::BasicPopulationState::with_size)
///   with `λ ≥ 1`; the actual population size is the solver's
///   `pop_size` (default
///   [`default_pop_size(D)`](Self::default_pop_size), override via
///   [`with_pop_size`](Self::with_pop_size)), so `λ` only needs to be
///   non-zero — the solver clears the state's `candidates` / `costs`
///   and refills them from a uniform sample of the box.
/// - **Implementor (this solver) must:** maintain feasibility (every
///   candidate after `init` and every trial after crossover is repaired
///   into the box) and the sorted-by-cost invariant on
///   [`PopulationState`](crate::core::state::PopulationState) at the
///   start and end of every iteration.
///
/// # Termination
///
/// No solver-internal optimality test — classical DE has no canonical
/// fixed-point criterion. Pair with framework criteria
/// [`MaxIter`](crate::core::termination::MaxIter),
/// [`MaxCostEvals`](crate::core::termination::MaxCostEvals),
/// [`MaxTime`](crate::core::termination::MaxTime),
/// [`CostTolerance`](crate::core::termination::CostTolerance), or
/// [`ParamTolerance`](crate::core::termination::ParamTolerance).
/// Greedy selection ensures `state.cost()` is non-increasing, so the
/// cost / param tolerances behave honestly under stochastic dynamics.
///
/// # Backends
///
/// Backend-generic — works with any `V` implementing
/// [`SampleUniformBox`] + [`VectorLen`] + [`ScaledAdd<f64>`] +
/// [`ScaleInPlace`] + `Index<usize, Output = f64>` +
/// `IndexMut<usize, Output = f64>` + `Clone`. That covers `Vec<f64>`,
/// `nalgebra::DVector<f64>` (feature `nalgebra`),
/// `ndarray::Array1<f64>` (feature `ndarray`), and `faer::Col<f64>`
/// (feature `faer`). No matrix operations are required.
///
/// [`scipy.optimize.differential_evolution`]:
///     https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.differential_evolution.html
pub struct De {
    pop_size_override: Option<usize>,
    f: f64,
    cr: f64,
    seed: u64,
    rng: Option<ChaCha8Rng>,
}

impl De {
    /// Build a new DE with defaults (`F = 0.8`, `CR = 0.9`, `pop_size`
    /// resolved lazily to [`default_pop_size(D)`](Self::default_pop_size)
    /// in [`Solver::init`]) and a PRNG seeded from `seed`.
    pub fn new(seed: u64) -> Self {
        Self {
            pop_size_override: None,
            f: 0.8,
            cr: 0.9,
            seed,
            rng: None,
        }
    }

    /// Storn & Price's `10·D` rule, floored at 4 so mutation always has
    /// at least three peers to draw from (`pick_three_distinct` requires
    /// `NP ≥ 4`).
    pub fn default_pop_size(n: usize) -> usize {
        (10 * n).max(4)
    }

    /// Override the population size (default
    /// [`default_pop_size(D)`](Self::default_pop_size), resolved at
    /// [`Solver::init`]).
    ///
    /// # Panics
    ///
    /// Panics if `pop_size < 4`. DE/rand/1 mutation samples three peers
    /// distinct from the target, so at least four individuals are
    /// required.
    pub fn with_pop_size(mut self, pop_size: usize) -> Self {
        assert!(
            pop_size >= 4,
            "De requires pop_size >= 4 (DE/rand/1 mutation needs three peers distinct from the target), got {}",
            pop_size
        );
        self.pop_size_override = Some(pop_size);
        self
    }

    /// Override the mutation scale `F` (default `0.8`). Storn & Price
    /// recommend `F ∈ [0.4, 1.0]`; values near `0.5` mix conservatively,
    /// values near `1.0` explore aggressively.
    ///
    /// # Panics
    ///
    /// Panics if `f` is not strictly positive and finite.
    pub fn with_f(mut self, f: f64) -> Self {
        assert!(
            f.is_finite() && f > 0.0,
            "De requires F > 0 and finite, got {}",
            f
        );
        self.f = f;
        self
    }

    /// Override the crossover probability `CR` (default `0.9`). Larger
    /// `CR` ⇒ more donor coordinates per trial; `CR = 0` reduces to one
    /// donor coordinate (the `j_rand` guarantee), `CR ≈ 1` essentially
    /// replaces the entire target.
    ///
    /// # Panics
    ///
    /// Panics if `cr` is not in `[0, 1]`.
    pub fn with_cr(mut self, cr: f64) -> Self {
        assert!(
            (0.0..=1.0).contains(&cr),
            "De requires CR in [0, 1], got {}",
            cr
        );
        self.cr = cr;
        self
    }
}

/// Sample three pairwise-distinct indices from `0..n`, each also
/// distinct from `exclude` (the target individual). Used by DE's
/// mutation step; requires `n ≥ 4`. Rejection-sampled — `O(1)` expected
/// for the population sizes DE is typically run at.
///
/// `pub(crate)` so a future memetic / strategy-variant DE can reuse the
/// operator directly; not a stable public surface.
pub(crate) fn pick_three_distinct<R>(n: usize, exclude: usize, rng: &mut R) -> (usize, usize, usize)
where
    R: Rng + ?Sized,
{
    debug_assert!(n >= 4, "pick_three_distinct needs n >= 4 (got n = {})", n);
    let r1 = loop {
        let k = rng.random_range(0..n);
        if k != exclude {
            break k;
        }
    };
    let r2 = loop {
        let k = rng.random_range(0..n);
        if k != exclude && k != r1 {
            break k;
        }
    };
    let r3 = loop {
        let k = rng.random_range(0..n);
        if k != exclude && k != r1 && k != r2 {
            break k;
        }
    };
    (r1, r2, r3)
}

/// DE/rand/1 mutation: `v = x_r1 + F · (x_r2 − x_r3)`.
///
/// `pub(crate)` so a future memetic / strategy-variant DE can reuse the
/// operator directly; not a stable public surface.
pub(crate) fn de_rand_1_mutate<V>(x_r1: &V, x_r2: &V, x_r3: &V, f: f64) -> V
where
    V: Clone + ScaledAdd<f64> + ScaleInPlace,
{
    let mut v = x_r2.clone();
    v.scaled_add(-1.0, x_r3);
    v.scale_in_place(f);
    v.scaled_add(1.0, x_r1);
    v
}

/// Reinitialize-per-coordinate bound repair (DEoptim style): for each
/// coordinate `j` with `v[j]` outside `[lower[j], upper[j]]`, replace it
/// with a fresh uniform draw from `lower[j]..=upper[j]`. Preserves
/// diversity better than clamping, which biases the population toward
/// the boundary.
///
/// `pub(crate)` so a future memetic / strategy-variant DE can reuse the
/// operator directly; not a stable public surface.
pub(crate) fn repair_reinit_per_coord<V, R>(v: &mut V, lower: &V, upper: &V, rng: &mut R)
where
    V: VectorLen + std::ops::Index<usize, Output = f64> + std::ops::IndexMut<usize, Output = f64>,
    R: Rng + ?Sized,
{
    let n = v.vec_len();
    for j in 0..n {
        if v[j] < lower[j] || v[j] > upper[j] {
            v[j] = rng.random_range(lower[j]..=upper[j]);
        }
    }
}

/// Binomial crossover (Storn & Price `bin`): clone the target and
/// overwrite each coordinate `j` with the donor's value whenever
/// `j == j_rand` (the guaranteed coordinate) or `rng.uniform() < cr`.
///
/// `pub(crate)` so a future memetic / strategy-variant DE can reuse the
/// operator directly; not a stable public surface.
pub(crate) fn binomial_crossover<V, R>(target: &V, donor: &V, cr: f64, rng: &mut R) -> V
where
    V: VectorLen
        + Clone
        + std::ops::Index<usize, Output = f64>
        + std::ops::IndexMut<usize, Output = f64>,
    R: Rng + ?Sized,
{
    let n = target.vec_len();
    let j_rand = rng.random_range(0..n);
    let mut u = target.clone();
    for j in 0..n {
        if j == j_rand || rng.random::<f64>() < cr {
            u[j] = donor[j];
        }
    }
    u
}

impl<P, V> Solver<P, BasicPopulationState<V>> for De
where
    P: CostFunction<Param = V, Output = f64> + BoxConstraints<Param = V>,
    V: VectorLen
        + Clone
        + SampleUniformBox
        + ScaledAdd<f64>
        + ScaleInPlace
        + std::ops::Index<usize, Output = f64>
        + std::ops::IndexMut<usize, Output = f64>,
{
    type Error = P::Error;

    fn init(
        &mut self,
        problem: &mut Problem<P>,
        mut state: BasicPopulationState<V>,
    ) -> Result<BasicPopulationState<V>, Self::Error> {
        let lo = problem.inner().lower().clone();
        let hi = problem.inner().upper().clone();
        let n = lo.vec_len();
        let pop_size = self
            .pop_size_override
            .unwrap_or_else(|| Self::default_pop_size(n));
        // Re-check the with_pop_size invariant here in case the user
        // didn't go through with_pop_size (default path needs the same
        // guarantee — default_pop_size already enforces it, so this is
        // belt-and-braces but cheap).
        assert!(pop_size >= 4, "De requires pop_size >= 4 (got {pop_size})");
        let mut rng = ChaCha8Rng::seed_from_u64(self.seed);
        // Same reseed-from-scratch pattern as Ssga / RandomSearch — the
        // solver's trajectory is reproducible regardless of which
        // BasicPopulationState constructor the caller used.
        state.candidates.clear();
        state.costs.clear();
        for _ in 0..pop_size {
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
        mut state: BasicPopulationState<V>,
    ) -> Result<(BasicPopulationState<V>, Option<TerminationReason>), Self::Error> {
        let lo = problem.inner().lower().clone();
        let hi = problem.inner().upper().clone();
        let rng = self
            .rng
            .as_mut()
            .expect("De::init must run before next_iter");
        let np = state.candidates.len();

        // Synchronous DE: build all trials from the *unchanged* current
        // generation, then evaluate, then select. Holding trials in a
        // single Vec avoids the asynchronous-update variant where a
        // just-replaced x[i] would feed back into the next mutation.
        let mut trials: Vec<V> = Vec::with_capacity(np);
        let mut trial_costs: Vec<f64> = Vec::with_capacity(np);
        for i in 0..np {
            let (r1, r2, r3) = pick_three_distinct(np, i, rng);
            let mut donor = de_rand_1_mutate(
                &state.candidates[r1],
                &state.candidates[r2],
                &state.candidates[r3],
                self.f,
            );
            repair_reinit_per_coord(&mut donor, &lo, &hi, rng);
            let trial = binomial_crossover(&state.candidates[i], &donor, self.cr, rng);
            let c_trial = problem.cost(&trial)?;
            trials.push(trial);
            trial_costs.push(c_trial);
        }

        // Greedy selection: `<=` (not `<`) lets equal-cost trials take
        // over, which keeps the population moving on plateaus without
        // disturbing strict best-so-far monotonicity at index 0 (sort
        // after the sweep restores it).
        for (i, (trial, c_trial)) in trials.into_iter().zip(trial_costs).enumerate() {
            if c_trial <= state.costs[i] {
                state.candidates[i] = trial;
                state.costs[i] = c_trial;
            }
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
    fn default_pop_size_uses_ten_d_with_floor_of_four() {
        assert_eq!(De::default_pop_size(0), 4);
        assert_eq!(De::default_pop_size(1), 10);
        assert_eq!(De::default_pop_size(5), 50);
        assert_eq!(De::default_pop_size(20), 200);
    }

    #[test]
    fn pick_three_distinct_returns_pairwise_distinct_indices_avoiding_target() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        for _ in 0..2_000 {
            let (r1, r2, r3) = pick_three_distinct(10, 4, &mut rng);
            assert_ne!(r1, 4);
            assert_ne!(r2, 4);
            assert_ne!(r3, 4);
            assert_ne!(r1, r2);
            assert_ne!(r1, r3);
            assert_ne!(r2, r3);
            assert!(r1 < 10 && r2 < 10 && r3 < 10);
        }
    }

    #[test]
    fn de_rand_1_mutate_computes_x_r1_plus_f_times_diff() {
        // v = [1, 2] + 0.5 * ([4, 5] - [3, 3]) = [1 + 0.5*1, 2 + 0.5*2] = [1.5, 3]
        let x_r1 = vec![1.0, 2.0];
        let x_r2 = vec![4.0, 5.0];
        let x_r3 = vec![3.0, 3.0];
        let v = de_rand_1_mutate(&x_r1, &x_r2, &x_r3, 0.5);
        assert!((v[0] - 1.5).abs() < 1e-12, "v[0] = {}", v[0]);
        assert!((v[1] - 3.0).abs() < 1e-12, "v[1] = {}", v[1]);
    }

    #[test]
    fn repair_reinit_per_coord_only_touches_violated_coordinates() {
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        let lo = vec![0.0, 0.0, 0.0];
        let hi = vec![1.0, 1.0, 1.0];
        let mut v = vec![0.5, -2.0, 3.0]; // coord 0 in-box, 1 below, 2 above
        repair_reinit_per_coord(&mut v, &lo, &hi, &mut rng);
        assert_eq!(v[0], 0.5, "in-box coord must be untouched");
        assert!(
            (0.0..=1.0).contains(&v[1]),
            "below-bound coord must land in [0, 1], got {}",
            v[1]
        );
        assert!(
            (0.0..=1.0).contains(&v[2]),
            "above-bound coord must land in [0, 1], got {}",
            v[2]
        );
    }

    #[test]
    fn binomial_crossover_with_cr_zero_takes_exactly_one_donor_coordinate() {
        // CR = 0 means only j_rand comes from the donor; the rest stay
        // from the target. Across many seeds, exactly one coordinate
        // differs per call.
        let target = vec![0.0, 0.0, 0.0, 0.0, 0.0];
        let donor = vec![1.0, 1.0, 1.0, 1.0, 1.0];
        for seed in 0..50 {
            let mut rng = ChaCha8Rng::seed_from_u64(seed);
            let u = binomial_crossover(&target, &donor, 0.0, &mut rng);
            let donor_count = u.iter().filter(|&&x| x == 1.0).count();
            assert_eq!(
                donor_count, 1,
                "with CR = 0, exactly one donor coordinate expected; got u = {:?}",
                u
            );
        }
    }

    #[test]
    fn binomial_crossover_with_cr_one_takes_all_donor_coordinates() {
        // CR = 1 (rng.uniform() < 1.0 is always true) means every
        // coordinate is donor. The j_rand guarantee is vacuous here.
        let target = vec![0.0, 0.0, 0.0];
        let donor = vec![1.0, 1.0, 1.0];
        let mut rng = ChaCha8Rng::seed_from_u64(123);
        let u = binomial_crossover(&target, &donor, 1.0, &mut rng);
        assert_eq!(u, donor);
    }
}
