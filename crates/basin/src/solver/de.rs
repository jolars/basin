use rand_distr::uniform::SampleUniform;

use crate::core::constraint::BoxConstraints;
use crate::core::math::{
    SampleUniformBox, Scalar, ScaleInPlace, ScaledAdd, VectorLen,
};
use crate::core::problem::{CostFunction, Problem};
use crate::core::rng::{ChaCha8Rng, Rng, RngExt, SeedableRng};
use crate::core::solver::Solver;
use crate::core::state::PopulationProgress;
use crate::solver::cma_es::sort_population_ascending;

/// Mutation rule for [`De`]. See its formula table for the donor definitions.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum DeMutation {
    /// Random base plus one difference; the default.
    #[default]
    Rand1,
    /// Generation's best base plus one difference.
    Best1,
    /// Random base plus two differences; requires at least six members.
    Rand2,
    /// Generation's best base plus two differences; requires at least five members.
    Best2,
    /// Random base moved toward the best, plus one difference.
    RandToBest1,
    /// Target moved toward the best, plus one difference.
    CurrentToBest1,
}

impl DeMutation {
    fn peer_count(self) -> usize {
        match self {
            Self::Best1 | Self::CurrentToBest1 => 2,
            Self::Rand1 | Self::RandToBest1 => 3,
            Self::Best2 => 4,
            Self::Rand2 => 5,
        }
    }

    fn minimum_pop_size(self) -> usize {
        (self.peer_count() + 1).max(4)
    }
}

/// Crossover rule for [`De`], independent of its mutation rule.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum DeCrossover {
    /// Independent donor choices with one guaranteed donor coordinate; the default.
    #[default]
    Binomial,
    /// A consecutive, wrapping donor segment with at least one coordinate.
    Exponential,
}

/// Differential evolution over a finite box, with configurable mutation,
/// crossover, and generation-wise mutation dithering.
///
/// A stochastic, derivative-free optimizer for rugged continuous landscapes.
/// The default is Storn and Price's `DE/rand/1/bin`, with the original Basin
/// seeded trajectory. Select mutation and crossover independently using
/// [`with_mutation`](Self::with_mutation) and
/// [`with_crossover`](Self::with_crossover).
///
/// ```
/// use basin::{De, DeCrossover, DeMutation};
/// let solver = De::<f64>::new(42)
///     .with_mutation(DeMutation::Best1)
///     .with_crossover(DeCrossover::Exponential)
///     .with_dither(0.5, 1.0);
/// ```
///
/// # Algorithm
///
/// Each [`next_iter`](Solver::next_iter) performs one full generation and
/// `pop_size` cost evaluations. Every donor uses the frozen current population,
/// including its best member `b = x[0]`. Random peers `r0, …, r4` are pairwise
/// distinct and exclude the target `i`; the best member may also be a peer
/// when it differs from the target. The mutation scale is denoted by `F`:
///
/// | Mutation | Donor | Minimum population |
/// |---|---|---|
/// | `Rand1` (default) | `x[r0] + F·(x[r1] − x[r2])` | 4 |
/// | `Best1` | `b + F·(x[r0] − x[r1])` | 4 |
/// | `Rand2` | `x[r0] + F·(x[r1] + x[r2] − x[r3] − x[r4])` | 6 |
/// | `Best2` | `b + F·(x[r0] + x[r1] − x[r2] − x[r3])` | 5 |
/// | `RandToBest1` | `x[r0] + F·(b − x[r0]) + F·(x[r1] − x[r2])` | 4 |
/// | `CurrentToBest1` | `x[i] + F·(b − x[i] + x[r0] − x[r1])` | 4 |
///
/// The formulas follow the named strategies in SciPy 1.16.2. Basin retains
/// its own sampling order and repairs the donor *before* crossover: each
/// out-of-bounds or non-finite coordinate is replaced by a uniform draw from
/// its inclusive finite bounds. This avoids accumulating clipped donors at
/// the boundary. Overflow in mutation arithmetic is handled by the same repair.
///
/// - **Binomial crossover** independently chooses donor coordinates with
///   probability `CR`, with one uniformly chosen coordinate always taken
///   from the donor.
/// - **Exponential crossover** takes a consecutive, wrapping donor segment
///   starting at a uniformly chosen coordinate. It always copies the first
///   coordinate and continues with probability `CR`, up to the full dimension.
///
/// Both crossovers copy one donor coordinate when `CR = 0` and all coordinates
/// when `CR = 1`. Copied values can equal the target, including at fixed bounds.
/// All trials are evaluated in a batch, with deterministic ordering under
/// `parallel`. A trial replaces its target when its cost is no worse (`≤`).
/// The population is then sorted by cost, so `state.cost()` reports its best.
/// This is synchronous DE, corresponding to SciPy's `updating="deferred"`;
/// it does not reproduce SciPy's random trajectory.
///
/// # Defaults and dithering
///
/// Defaults are `Rand1`, `Binomial`, `F = 0.8`, `CR = 0.9`, and
/// [`default_pop_size(D)`](Self::default_pop_size) `= max(4, 10·D)`.
/// Use [`with_pop_size`](Self::with_pop_size), [`with_f`](Self::with_f), and
/// [`with_cr`](Self::with_cr) to override them.
/// [`with_dither(min, max)`](Self::with_dither) instead draws one `F` from
/// `[min, max)` before peer sampling in each generation. All donors share it.
/// `with_f` disables dithering; the last scale builder wins. Fixed scales and
/// dithering endpoints must be positive and finite, with `min < max`.
///
/// # Contract and lifecycle
///
/// The problem implements [`CostFunction`] and [`BoxConstraints`] with the
/// same vector type and scalar. Bounds must be non-empty, equal in length,
/// finite, and ordered (`lower[j] ≤ upper[j]`). Equal bounds pin a coordinate.
/// Invalid settings panic in their builders; initialization checks bounds and
/// the selected mutation's minimum population size, independently of builder
/// order. Typed problem errors propagate unchanged.
///
/// Supply [`PopulationProgress::empty`] to sample the configured population,
/// or [`PopulationProgress::from_population`] with that many finite vectors
/// whose dimension matches the bounds. Fresh initialization resets progress
/// and the configured [`ChaCha8Rng`], projects explicit members into the box,
/// and reevaluates every member. An explicit population with a different size
/// or dimension panics. Reusing final members is a population warm start;
/// an empty state reproduces the original seeded run.
/// [`DeInject`](crate::DeInject) accepts a configured `De` for local refinement.
///
/// The same seed and settings reproduce the trajectory, including with
/// parallel evaluation. Solver-aware exact checkpoints preserve configuration,
/// the live RNG, the population, and evaluation counts. The `serde` feature
/// enables serialization; checkpoint files require the same Basin version.
///
/// # Termination
///
/// There is no solver-internal optimality test. Configure optional
/// [`with_absolute_cost_change_tolerance`](Self::with_absolute_cost_change_tolerance)
/// or [`with_absolute_step_tolerance`](Self::with_absolute_step_tolerance),
/// and set execution budgets on [`Executor`](crate::Executor).
/// Greedy selection keeps the best cost non-increasing for finite objectives.
/// Optional change tests compare generation representatives, including
/// generations that retain the same best member.
///
/// # Backends
///
/// `Vec<F>`, `nalgebra::DVector<F>`, `ndarray::Array1<F>`, and `faer::Col<F>`
/// for every supported backend release, with `F = f32` or `f64`. Requires
/// [`SampleUniformBox`], [`VectorLen`], [`ScaledAdd<F>`], [`ScaleInPlace<F>`],
/// `Index<usize, Output = F>`, `IndexMut<usize, Output = F>`, and `Clone`.
/// No matrix operations or BLAS/LAPACK provider are required.
///
/// # References
///
/// - R. Storn and K. Price, “Differential Evolution – A Simple and Efficient
///   Heuristic for Global Optimization over Continuous Spaces,”
///   *Journal of Global Optimization* 11 (1997), 341–359.
/// - [SciPy 1.16.2 differential evolution](https://docs.scipy.org/doc/scipy-1.16.2/reference/generated/scipy.optimize.differential_evolution.html):
///   mutation formulas, binomial and exponential crossover, and generation-wise
///   dithering. Donors and solution quality are checked against this version.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct De<F: Scalar = f64> {
    pop_size_override: Option<usize>,
    f: F,
    cr: f64,
    seed: u64,
    rng: Option<ChaCha8Rng>,
    mutation: DeMutation,
    crossover: DeCrossover,
    dither: Option<(F, F)>,
}

impl<F: Scalar> De<F> {
    /// Build a new DE with defaults (`F = 0.8`, `CR = 0.9`, `pop_size`
    /// resolved lazily to [`default_pop_size(D)`](De::default_pop_size)
    /// in [`Solver::init`]) and a PRNG seeded from `seed`.
    pub fn new(seed: u64) -> Self {
        Self {
            pop_size_override: None,
            f: F::from_f64(0.8).unwrap(),
            cr: 0.9,
            seed,
            rng: None,
            mutation: DeMutation::Rand1,
            crossover: DeCrossover::Binomial,
            dither: None,
        }
    }
}

impl<F: Scalar> De<F> {
    /// Storn & Price's `10·D` rule, floored at 4 so mutation always has
    /// at least three peers to draw from. For non-empty parameter vectors,
    /// this default also satisfies every other mutation rule.
    ///
    /// `F`-free, so the `: Scalar` bound is dropped from this impl block;
    /// keeps `De::default_pop_size(D)` callable from the trait-impl
    /// `init` (`Self::default_pop_size`) and from F-agnostic test code
    /// (`De::default_pop_size(0)`, with `F` defaulting to `f64`).
    pub fn default_pop_size(n: usize) -> usize {
        (10 * n).max(4)
    }
}

impl<F: Scalar> De<F> {
    /// Select a mutation rule; defaults to [`DeMutation::Rand1`].
    ///
    /// Initialization checks the final population size against this rule,
    /// independently of builder order. `Rand2` needs six members, `Best2`
    /// needs five, and the other rules retain the minimum of four.
    pub fn with_mutation(mut self, mutation: DeMutation) -> Self {
        self.mutation = mutation;
        self
    }

    /// Select crossover independently of mutation; defaults to binomial.
    pub fn with_crossover(mut self, crossover: DeCrossover) -> Self {
        self.crossover = crossover;
        self
    }

    /// Draw one mutation scale uniformly from `[min, max)` per generation.
    ///
    /// All donors in that generation share the scale. The draw occurs before
    /// peer sampling, using the solver's seeded RNG. This overrides a previous
    /// [`with_f`](Self::with_f); calling `with_f` later disables dithering.
    /// There is no upper cap of two, matching Basin's fixed-scale API.
    ///
    /// # Panics
    ///
    /// Panics unless both endpoints are finite and `0 < min < max`.
    /// For a constant scale, use `with_f` instead of equal endpoints.
    pub fn with_dither(mut self, min: F, max: F) -> Self {
        assert!(
            min.is_finite() && max.is_finite() && min > F::zero() && min < max,
            "De dithering requires finite 0 < min < max, got {min:?}, {max:?}"
        );
        self.dither = Some((min, max));
        self
    }

    /// Override the population size (default
    /// [`default_pop_size(D)`](Self::default_pop_size), resolved at
    /// [`Solver::init`]).
    ///
    /// # Panics
    ///
    /// Panics if `pop_size < 4`. DE/rand/1 mutation samples three peers
    /// distinct from the target, so at least four individuals are
    /// required. Initialization also checks the selected mutation rule's
    /// minimum, which is five for `Best2` and six for `Rand2`.
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
    /// values near `1.0` explore aggressively. Disables previously configured
    /// [`with_dither`](Self::with_dither).
    ///
    /// # Panics
    ///
    /// Panics if `f` is not strictly positive and finite.
    pub fn with_f(mut self, f: F) -> Self {
        assert!(
            f.is_finite() && f > F::zero(),
            "De requires F > 0 and finite, got {:?}",
            f
        );
        self.f = f;
        self.dither = None;
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

// Rejection sampling preserves the legacy rand/1 draw order. A stack array
// avoids an allocation for each target, including the five-peer rand/2 rule.
fn pick_distinct<R>(
    n: usize,
    exclude: usize,
    count: usize,
    rng: &mut R,
) -> [usize; 5]
where
    R: Rng + ?Sized,
{
    debug_assert!(count <= 5 && n > count && exclude < n);
    let mut peers = [0; 5];
    for j in 0..count {
        peers[j] = loop {
            let k = rng.random_range(0..n);
            if k != exclude && !peers[..j].contains(&k) {
                break k;
            }
        };
    }
    peers
}

/// DE/rand/1 mutation: `v = x_r1 + F · (x_r2 − x_r3)`.
///
/// `pub(crate)` so a future memetic or strategy-variant DE can reuse the
/// operator directly; not a stable public surface.
pub(crate) fn de_rand_1_mutate<V, F>(x_r1: &V, x_r2: &V, x_r3: &V, f: F) -> V
where
    F: Scalar,
    V: Clone + ScaledAdd<F> + ScaleInPlace<F>,
{
    let mut v = x_r2.clone();
    v.scaled_add(-F::one(), x_r3);
    v.scale_in_place(f);
    v.scaled_add(F::one(), x_r1);
    v
}

fn mutate<V, F>(
    mutation: DeMutation,
    population: &[V],
    target: usize,
    peers: &[usize],
    f: F,
) -> V
where
    F: Scalar,
    V: Clone + ScaledAdd<F> + ScaleInPlace<F>,
{
    let best = &population[0];
    match mutation {
        DeMutation::Rand1 => de_rand_1_mutate(
            &population[peers[0]],
            &population[peers[1]],
            &population[peers[2]],
            f,
        ),
        DeMutation::Best1 => de_rand_1_mutate(
            best,
            &population[peers[0]],
            &population[peers[1]],
            f,
        ),
        DeMutation::Rand2 | DeMutation::Best2 => {
            let (base, differences) = if mutation == DeMutation::Rand2 {
                (&population[peers[0]], &peers[1..])
            } else {
                (best, peers)
            };
            let mut donor = population[differences[0]].clone();
            donor.scaled_add(F::one(), &population[differences[1]]);
            donor.scaled_add(-F::one(), &population[differences[2]]);
            donor.scaled_add(-F::one(), &population[differences[3]]);
            donor.scale_in_place(f);
            donor.scaled_add(F::one(), base);
            donor
        }
        DeMutation::RandToBest1 => {
            let base = &population[peers[0]];
            let toward_best = de_rand_1_mutate(base, best, base, f);
            de_rand_1_mutate(
                &toward_best,
                &population[peers[1]],
                &population[peers[2]],
                f,
            )
        }
        DeMutation::CurrentToBest1 => {
            let base = &population[target];
            let mut donor = best.clone();
            donor.scaled_add(-F::one(), base);
            donor.scaled_add(F::one(), &population[peers[0]]);
            donor.scaled_add(-F::one(), &population[peers[1]]);
            donor.scale_in_place(f);
            donor.scaled_add(F::one(), base);
            donor
        }
    }
}

/// Reinitialize-per-coordinate bound repair (DEoptim style): for each
/// coordinate `j` with `v[j]` non-finite or outside `[lower[j], upper[j]]`,
/// replace it with a fresh uniform draw from `lower[j]..=upper[j]`. Preserves
/// diversity better than clamping, which biases the population toward
/// the boundary.
///
/// `pub(crate)` so a future memetic or strategy-variant DE can reuse the
/// operator directly; not a stable public surface.
pub(crate) fn repair_reinit_per_coord<V, F, R>(
    v: &mut V,
    lower: &V,
    upper: &V,
    rng: &mut R,
) where
    F: Scalar + SampleUniform,
    V: VectorLen
        + std::ops::Index<usize, Output = F>
        + std::ops::IndexMut<usize, Output = F>,
    R: Rng + ?Sized,
{
    let n = v.vec_len();
    for j in 0..n {
        if !v[j].is_finite() || v[j] < lower[j] || v[j] > upper[j] {
            v[j] = rng.random_range(lower[j]..=upper[j]);
        }
    }
}

/// Binomial crossover (Storn & Price `bin`): clone the target and
/// overwrite each coordinate `j` with the donor's value whenever
/// `j == j_rand` (the guaranteed coordinate) or `rng.uniform() < cr`.
///
/// `pub(crate)` so a future memetic or strategy-variant DE can reuse the
/// operator directly; not a stable public surface.
pub(crate) fn binomial_crossover<V, F, R>(
    target: &V,
    donor: &V,
    cr: f64,
    rng: &mut R,
) -> V
where
    F: Scalar,
    V: VectorLen
        + Clone
        + std::ops::Index<usize, Output = F>
        + std::ops::IndexMut<usize, Output = F>,
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

fn exponential_crossover<V, F, R>(
    target: &V,
    donor: &V,
    cr: f64,
    rng: &mut R,
) -> V
where
    F: Scalar,
    V: VectorLen
        + Clone
        + std::ops::Index<usize, Output = F>
        + std::ops::IndexMut<usize, Output = F>,
    R: Rng + ?Sized,
{
    let n = target.vec_len();
    let start = rng.random_range(0..n);
    let mut trial = target.clone();
    trial[start] = donor[start];
    for offset in 1..n {
        if rng.random::<f64>() >= cr {
            break;
        }
        let j = (start + offset) % n;
        trial[j] = donor[j];
    }
    trial
}

impl<P, V, F> Solver<P, PopulationProgress<V, F>> for De<F>
where
    F: Scalar + SampleUniform + crate::core::parallel::MaybeSend,
    P: CostFunction<Param = V, Output = F>
        + BoxConstraints<Param = V>
        + crate::core::parallel::MaybeSync,
    P::Error: crate::core::parallel::MaybeSend,
    V: VectorLen
        + Clone
        + SampleUniformBox
        + ScaledAdd<F>
        + ScaleInPlace<F>
        + crate::core::parallel::MaybeSync
        + std::ops::Index<usize, Output = F>
        + std::ops::IndexMut<usize, Output = F>,
{
    type Error = P::Error;

    fn init(
        &mut self,
        problem: &mut Problem<P>,
        mut state: PopulationProgress<V, F>,
    ) -> Result<PopulationProgress<V, F>, Self::Error> {
        let lo = problem.inner().lower().clone();
        let hi = problem.inner().upper().clone();
        let n = lo.vec_len();
        assert!(n > 0, "De requires non-empty parameter bounds");
        let pop_size = self
            .pop_size_override
            .unwrap_or_else(|| Self::default_pop_size(n));
        // Check the final configuration so builder order cannot affect validity.
        let minimum = self.mutation.minimum_pop_size();
        assert!(
            pop_size >= minimum,
            "De {:?} requires pop_size >= {minimum} (got {pop_size})",
            self.mutation
        );
        let mut rng = ChaCha8Rng::seed_from_u64(self.seed);
        state.reset();
        super::population::prepare_population(
            &mut state.candidates,
            &lo,
            &hi,
            pop_size,
            &mut rng,
        );
        state.costs = problem.cost_batch(&state.candidates)?;
        sort_population_ascending(&mut state.candidates, &mut state.costs);
        state.select_best_member();
        self.rng = Some(rng);
        Ok(state)
    }

    fn next_iter(
        &mut self,
        problem: &mut Problem<P>,
        mut state: PopulationProgress<V, F>,
    ) -> Result<crate::SolverStep<PopulationProgress<V, F>>, Self::Error> {
        let lo = problem.inner().lower().clone();
        let hi = problem.inner().upper().clone();
        let rng = self
            .rng
            .as_mut()
            .expect("De::init must run before next_iter");
        let np = state.candidates.len();
        let f = match self.dither {
            Some((min, max)) => rng.random_range(min..max),
            None => self.f,
        };

        // Synchronous DE: build all trials from the *unchanged* current
        // generation, then evaluate, then select. Holding trials in a
        // single Vec avoids the asynchronous-update variant where a
        // just-replaced x[i] would feed back into the next mutation.
        let mut trials: Vec<V> = Vec::with_capacity(np);
        for i in 0..np {
            let count = self.mutation.peer_count();
            let peers = pick_distinct(np, i, count, rng);
            let mut donor =
                mutate(self.mutation, &state.candidates, i, &peers[..count], f);
            repair_reinit_per_coord(&mut donor, &lo, &hi, rng);
            let trial = match self.crossover {
                DeCrossover::Binomial => binomial_crossover(
                    &state.candidates[i],
                    &donor,
                    self.cr,
                    rng,
                ),
                DeCrossover::Exponential => exponential_crossover(
                    &state.candidates[i],
                    &donor,
                    self.cr,
                    rng,
                ),
            };
            trials.push(trial);
        }
        // All trials are built from the frozen current generation, so they
        // are independent, so evaluate them in one batch (parallel under the
        // `parallel` feature).
        let trial_costs = problem.cost_batch(&trials)?;

        // Greedy selection: `<=` (not `<`) lets equal-cost trials take
        // over, which keeps the population moving on plateaus without
        // disturbing strict best-so-far monotonicity at index 0 (sort
        // after the sweep restores it).
        for (i, (trial, c_trial)) in
            trials.into_iter().zip(trial_costs).enumerate()
        {
            if c_trial <= state.costs[i] {
                state.candidates[i] = trial;
                state.costs[i] = c_trial;
            }
        }
        sort_population_ascending(&mut state.candidates, &mut state.costs);
        state.select_best_member();
        Ok(crate::SolverStep::from((state, None)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand_chacha::ChaCha8Rng;

    const MUTATIONS: [DeMutation; 6] = [
        DeMutation::Rand1,
        DeMutation::Best1,
        DeMutation::Rand2,
        DeMutation::Best2,
        DeMutation::RandToBest1,
        DeMutation::CurrentToBest1,
    ];

    #[test]
    fn mutation_donors_match_scipy_1_16_2() {
        let population = vec![
            vec![1.0, 2.0, 3.0],
            vec![2.0, -1.0, 4.0],
            vec![4.0, 5.0, -2.0],
            vec![-3.0, 1.0, 2.0],
            vec![6.0, -4.0, 1.0],
            vec![0.0, 3.0, -5.0],
        ];
        let rows: Vec<_> =
            include_str!("../../tests/fixtures/de_mutations.tsv")
                .lines()
                .filter(|line| !line.starts_with('#'))
                .collect();
        assert_eq!(rows.len(), MUTATIONS.len());
        for (mutation, row) in MUTATIONS.into_iter().zip(rows) {
            let expected: Vec<f64> = row
                .split_whitespace()
                .skip(1)
                .map(|x| x.parse().unwrap())
                .collect();
            let actual =
                mutate(mutation, &population, 3, &[1, 2, 4, 5, 0], 0.5);
            assert_eq!(actual, expected, "{mutation:?}");
        }
    }

    #[test]
    fn peer_sampling_excludes_target_at_minimum_population_sizes() {
        let mut rng = ChaCha8Rng::seed_from_u64(123);
        for count in 2..=5 {
            let n = count + 1;
            for target in 0..n {
                for _ in 0..100 {
                    let peers = pick_distinct(n, target, count, &mut rng);
                    let mut actual = peers[..count].to_vec();
                    actual.sort_unstable();
                    let expected: Vec<_> =
                        (0..n).filter(|&i| i != target).collect();
                    assert_eq!(actual, expected);
                }
            }
        }
    }

    #[test]
    fn exponential_crossover_copies_one_wrapping_segment() {
        let target = vec![0.0; 5];
        let donor = vec![1.0; 5];
        let mut saw_wrapping = false;
        for seed in 0..100 {
            let mut rng = ChaCha8Rng::seed_from_u64(seed);
            let trial = exponential_crossover(&target, &donor, 0.6, &mut rng);
            assert!(trial.contains(&1.0));
            let transitions =
                (0..5).filter(|&i| trial[i] != trial[(i + 1) % 5]).count();
            assert!(transitions <= 2, "non-contiguous crossover: {trial:?}");
            saw_wrapping |=
                trial[0] == 1.0 && trial[4] == 1.0 && trial.contains(&0.0);
            let trial = exponential_crossover(&target, &donor, 0.0, &mut rng);
            assert_eq!(trial.iter().filter(|&&x| x == 1.0).count(), 1);
            assert_eq!(
                exponential_crossover(&target, &donor, 1.0, &mut rng),
                donor
            );
            for cr in [0.0, 0.5, 1.0] {
                assert_eq!(
                    exponential_crossover(&vec![0.0], &vec![1.0], cr, &mut rng),
                    vec![1.0]
                );
            }
        }
        assert!(saw_wrapping);
    }

    struct Plateau {
        lower: Vec<f64>,
        upper: Vec<f64>,
    }

    impl CostFunction for Plateau {
        type Param = Vec<f64>;
        type Output = f64;
        type Error = std::convert::Infallible;
        fn cost(&self, _: &Vec<f64>) -> Result<f64, Self::Error> {
            Ok(0.0)
        }
    }

    impl BoxConstraints for Plateau {
        fn lower(&self) -> &Vec<f64> {
            &self.lower
        }
        fn upper(&self) -> &Vec<f64> {
            &self.upper
        }
    }

    #[test]
    fn dithering_draws_one_shared_scale_before_each_generation() {
        for mutation in MUTATIONS {
            for crossover in [DeCrossover::Binomial, DeCrossover::Exponential] {
                let mut problem = Problem::new(Plateau {
                    lower: vec![-2.0; 3],
                    upper: vec![3.0; 3],
                });
                let make = || {
                    De::new(17)
                        .with_pop_size(8)
                        .with_mutation(mutation)
                        .with_crossover(crossover)
                };
                let mut dithered = make().with_dither(0.3, 1.7);
                let mut fixed = make();
                let mut actual = dithered
                    .init(&mut problem, PopulationProgress::empty())
                    .unwrap();
                let mut expected = fixed
                    .init(&mut problem, PopulationProgress::empty())
                    .unwrap();
                let mut scales = Vec::new();
                for _ in 0..4 {
                    let mut rng = dithered.rng.clone().unwrap();
                    let scale = rng.random_range(0.3..1.7);
                    scales.push(scale);
                    fixed = fixed.with_f(scale);
                    fixed.rng = Some(rng);
                    actual =
                        dithered.next_iter(&mut problem, actual).unwrap().state;
                    expected =
                        fixed.next_iter(&mut problem, expected).unwrap().state;
                    assert_eq!(
                        actual.candidates, expected.candidates,
                        "{mutation:?} {crossover:?}"
                    );
                    assert_eq!(
                        dithered.rng.as_ref().unwrap().get_word_pos(),
                        fixed.rng.as_ref().unwrap().get_word_pos()
                    );
                }
                assert!(scales.windows(2).any(|pair| pair[0] != pair[1]));
            }
        }
    }

    #[test]
    fn repair_reinitializes_non_finite_donor_coordinates() {
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        let lo = vec![-1.0; 4];
        let hi = vec![1.0; 4];
        let mut donor = vec![f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 0.5];
        repair_reinit_per_coord(&mut donor, &lo, &hi, &mut rng);
        assert!(
            donor
                .iter()
                .all(|x| x.is_finite() && (-1.0..=1.0).contains(x))
        );
        assert_eq!(donor[3], 0.5);
    }

    #[test]
    fn default_pop_size_uses_ten_d_with_floor_of_four() {
        // F-explicit so inference picks the f64 monomorphization; the
        // formula is F-free but `default_pop_size` lives on `De<F>`.
        assert_eq!(De::<f64>::default_pop_size(0), 4);
        assert_eq!(De::<f64>::default_pop_size(1), 10);
        assert_eq!(De::<f64>::default_pop_size(5), 50);
        assert_eq!(De::<f64>::default_pop_size(20), 200);
    }

    #[test]
    fn pick_three_distinct_returns_pairwise_distinct_indices_avoiding_target() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        for _ in 0..2_000 {
            let [r1, r2, r3, _, _] = pick_distinct(10, 4, 3, &mut rng);
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
        let x_r1: Vec<f64> = vec![1.0, 2.0];
        let x_r2: Vec<f64> = vec![4.0, 5.0];
        let x_r3: Vec<f64> = vec![3.0, 3.0];
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
