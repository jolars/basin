use crate::core::constraint::BoxConstraints;
use crate::core::math::{SampleUniformBox, Scalar, VectorLen};
use crate::core::problem::{CostFunction, Problem};
use crate::core::rng::{ChaCha8Rng, SeedableRng};
use crate::core::solver::Solver;
use crate::core::state::PopulationProgress;
use crate::core::termination::TerminationReason;
// Joint ascending-by-cost sort shared with the population solvers.
use crate::solver::cma_es::sort_population_ascending;

/// Elitist (1+λ) random search over a feasible box.
///
/// Samples independent candidates and retains the best member between
/// generations. Publishes [`PopulationProgress`] while owning its RNG.
///
/// # Algorithm
///
/// At [`init`](Solver::init) an empty state receives `λ` candidates drawn
/// uniformly from the problem's box `[lower, upper]`; explicit members are
/// projected into that box. The solver evaluates each and sorts by cost.
///
/// Each [`next_iter`](Solver::next_iter):
///
/// ```text
/// elite ← (candidates[0], costs[0])           # current best
/// resample λ candidates uniformly in [lower, upper]
/// evaluate each, append along with the elite (now λ + 1 entries)
/// sort ascending by cost
/// truncate back to λ                            # drops the worst
/// ```
///
/// Elite carry-over keeps the current best cost non-increasing for finite
/// objectives. Optional cost-change and step tests compare generation
/// representatives, including generations that retain the same elite.
///
/// # Reproducibility
///
/// The solver carries a [`ChaCha8Rng`] seeded from the `seed: u64`
/// passed to [`new`](Self::new). Same seed → same trajectory, on every
/// platform basin builds for (including `wasm32-unknown-unknown`). To
/// vary runs, vary the seed; to share runs, share the seed.
///
/// # Contract
///
/// - **Caller must:** implement [`BoxConstraints`] on the problem with
///   `lower[i] ≤ upper[i]` for every component. Equal bounds are
///   allowed (the corresponding component is pinned).
/// - Supply [`PopulationProgress::empty`] to sample `lambda` members, or
///   [`PopulationProgress::from_population`] with exactly `lambda` finite
///   vectors whose dimension matches the bounds. Bounds must be non-empty,
///   finite, equal in length, and ordered. Invalid shapes panic at initialization.
/// - Fresh initialization resets progress and the configured RNG, projects
///   explicit members into the box, reevaluates every member, and sorts by cost.
///   Reusing a final population is a population warm start. Supply an empty
///   state to reproduce the original seeded run.
/// - Solver-aware exact checkpoints retain the population, RNG, and raw
///   evaluation counts and skip initialization. With `serde`, both the solver
///   and progress serialize. A progress-only snapshot cannot continue exactly.
///
/// # Termination
///
/// No solver-internal optimality test; random search has no canonical
/// fixed-point criterion. Use the framework's
/// [`max_iter`](crate::Executor::max_iter),
/// [`max_cost_evals`](crate::Executor::max_cost_evals),
/// [`max_time`](crate::Executor::max_time),
/// [`with_absolute_cost_change_tolerance`](Self::with_absolute_cost_change_tolerance), or
/// [`with_absolute_step_tolerance`](Self::with_absolute_step_tolerance). The
/// elite-carryover makes cost monotonicity honest, so cost-based budgets
/// behave as expected.
///
/// # Backends
///
/// `Vec<F>`, nalgebra `DVector<F>`, ndarray `Array1<F>`, and faer `Col<F>`,
/// for all supported releases with `F = f32` or `f64`. Requires
/// [`SampleUniformBox`], [`VectorLen`], `Index<usize, Output = F>`,
/// `IndexMut<usize, Output = F>`, and `Clone`. No matrix operations are needed.
/// The problem must implement [`BoxConstraints`].
///
/// # Examples
///
/// Elitist random search over a feasible box. The problem implements
/// [`CostFunction`] and [`BoxConstraints`]; the
/// initial population is generated from an empty state via
/// [`PopulationProgress::empty`](crate::PopulationProgress::empty):
///
/// ```
/// use basin::{PopulationProgress, BoxConstraints, CostFunction, Executor, RandomSearch};
///
/// struct BoundedSphere {
///     lower: Vec<f64>,
///     upper: Vec<f64>,
/// }
/// impl CostFunction for BoundedSphere {
///     type Param = Vec<f64>;
///     type Output = f64;
///     type Error = std::convert::Infallible;
///     fn cost(&self, x: &Vec<f64>) -> Result<f64, Self::Error> {
///         Ok(x.iter().map(|xi| xi * xi).sum())
///     }
/// }
/// impl BoxConstraints for BoundedSphere {
///     fn lower(&self) -> &Vec<f64> { &self.lower }
///     fn upper(&self) -> &Vec<f64> { &self.upper }
/// }
///
/// let problem = BoundedSphere { lower: vec![-5.0, -5.0], upper: vec![5.0, 5.0] };
/// let result = Executor::new(
///     problem,
///     RandomSearch::new(16, 42),
///     PopulationProgress::<Vec<f64>>::empty(),
/// )
/// .max_iter(500)
/// .run()
/// .unwrap();
/// assert!(result.cost() < 1.0);
/// ```
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct RandomSearch {
    lambda: usize,
    seed: u64,
    rng: ChaCha8Rng,
}

impl RandomSearch {
    /// New `RandomSearch` with population size `lambda` and PRNG seed
    /// `seed`. Same `seed` → same iterate trajectory on the same
    /// problem; vary `seed` to vary the run.
    ///
    /// # Panics
    ///
    /// Panics if `lambda == 0`. A non-empty population is the smallest
    /// thing this solver can iterate on.
    pub fn new(lambda: usize, seed: u64) -> Self {
        assert!(lambda >= 1, "RandomSearch requires lambda >= 1");
        Self {
            lambda,
            seed,
            rng: ChaCha8Rng::seed_from_u64(seed),
        }
    }
}

impl<P, V, F> Solver<P, PopulationProgress<V, F>> for RandomSearch
where
    F: Scalar + crate::core::parallel::MaybeSend,
    P: CostFunction<Param = V, Output = F>
        + BoxConstraints<Param = V>
        + crate::core::parallel::MaybeSync,
    P::Error: crate::core::parallel::MaybeSend,
    V: SampleUniformBox
        + Clone
        + VectorLen
        + std::ops::Index<usize, Output = F>
        + std::ops::IndexMut<usize, Output = F>
        + crate::core::parallel::MaybeSync,
{
    type Error = P::Error;

    fn init(
        &mut self,
        problem: &mut Problem<P>,
        mut state: PopulationProgress<V, F>,
    ) -> Result<PopulationProgress<V, F>, Self::Error> {
        let lo = problem.inner().lower().clone();
        let hi = problem.inner().upper().clone();
        state.reset();
        self.rng = ChaCha8Rng::seed_from_u64(self.seed);
        super::population::prepare_population(
            &mut state.candidates,
            &lo,
            &hi,
            self.lambda,
            &mut self.rng,
        );
        state.costs = problem.cost_batch(&state.candidates)?;
        sort_population_ascending(&mut state.candidates, &mut state.costs);
        state.select_best_member();
        Ok(state)
    }

    fn next_iter(
        &mut self,
        problem: &mut Problem<P>,
        mut state: PopulationProgress<V, F>,
    ) -> Result<
        (PopulationProgress<V, F>, Option<TerminationReason>),
        Self::Error,
    > {
        // Snapshot the elite before resampling; this is what makes
        // state.cost() monotone.
        let elite_x = state.candidates[0].clone();
        let elite_c = state.costs[0];

        let lo = problem.inner().lower().clone();
        let hi = problem.inner().upper().clone();
        state.candidates.clear();
        state.costs.clear();
        state.candidates.push(elite_x);
        state.costs.push(elite_c);
        // Sample the λ fresh draws (sequential RNG) into their own buffer so
        // the elite's already-known cost is not re-evaluated, then batch the
        // independent draws (parallel under the `parallel` feature). Pushing
        // them in sample order keeps the population bit-identical to a serial
        // loop before the sort.
        let mut fresh: Vec<V> = Vec::with_capacity(self.lambda);
        for _ in 0..self.lambda {
            fresh.push(V::sample_uniform_box(&lo, &hi, &mut self.rng));
        }
        let fresh_costs = problem.cost_batch(&fresh)?;
        state.candidates.extend(fresh);
        state.costs.extend(fresh_costs);
        sort_population_ascending(&mut state.candidates, &mut state.costs);
        // Drop the worst back down to λ. Sort puts the elite first
        // when it's still the best, so truncation never drops it.
        state.candidates.truncate(self.lambda);
        state.costs.truncate(self.lambda);
        state.select_best_member();
        Ok((state, None))
    }
}
