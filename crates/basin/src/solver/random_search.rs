use crate::core::constraint::BoxConstraints;
use crate::core::math::{SampleUniformBox, Scalar};
use crate::core::problem::{CostFunction, Problem};
use crate::core::rng::{ChaCha8Rng, SeedableRng};
use crate::core::solver::Solver;
use crate::core::state::BasicPopulationState;
use crate::core::termination::TerminationReason;
// Joint ascending-by-cost sort shared with the population solvers.
use crate::solver::cma_es::sort_population_ascending;

/// Elitist (1+λ) random search over a feasible box.
///
/// The first stochastic solver in basin and the smallest vehicle for the
/// new [`BasicPopulationState`]/[`PopulationState`](crate::PopulationState) story:
/// a derivative-free, population-based method that exercises the
/// reproducible RNG infrastructure (see [`crate::core::rng`]) without
/// pulling in any covariance and distribution machinery (those land in
/// S8 alongside CMA-ES).
///
/// # Algorithm
///
/// At [`init`](Solver::init) the solver fills `state.candidates` with
/// `λ` candidates drawn component-wise uniformly from the problem's
/// box `[lower, upper]`, evaluates each, and sorts by ascending cost.
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
/// The elite carry-over keeps `state.cost()` non-increasing across
/// generations, so the framework's
/// [`CostTolerance`](crate::core::termination::CostTolerance) and
/// [`ParamTolerance`](crate::core::termination::ParamTolerance) work
/// honestly without redesign. (CMA-ES is genuinely non-monotone, and
/// the "no monotone cost" termination story will be designed alongside
/// it in S8 / S9.)
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
/// - **Caller must:** hand in a [`BasicPopulationState`] sized to match
///   the solver's `lambda`. The two natural constructors are
///   [`BasicPopulationState::with_size(lambda)`](BasicPopulationState::with_size)
///   (solver fills the population in `init`) or
///   [`from_population`](BasicPopulationState::from_population) (caller
///   supplies a custom initial distribution). `with_size` is the
///   common case.
/// - **Implementor (this solver) must:** maintain feasibility (every
///   candidate after `init` is in the box) and the sorted-by-cost
///   invariant on
///   [`PopulationState`](crate::core::state::PopulationState) at the
///   start and end of each iteration.
///
/// # Termination
///
/// No solver-internal optimality test; random search has no canonical
/// fixed-point criterion. Use the framework's
/// [`MaxIter`](crate::core::termination::MaxIter),
/// [`MaxCostEvals`](crate::core::termination::MaxCostEvals),
/// [`MaxTime`](crate::core::termination::MaxTime),
/// [`CostTolerance`](crate::core::termination::CostTolerance), or
/// [`ParamTolerance`](crate::core::termination::ParamTolerance). The
/// elite-carryover makes cost monotonicity honest, so cost-based budgets
/// behave as expected.
///
/// # Backends
///
/// Backend-generic; works with any `V` implementing
/// [`SampleUniformBox`] + `Clone`. That covers `Vec<f64>`,
/// `nalgebra::DVector<f64>` (feature `nalgebra`),
/// `ndarray::Array1<f64>` (feature `ndarray`), and `faer::Col<f64>`
/// (feature `faer`). The problem must implement [`BoxConstraints`].
///
/// # Examples
///
/// Elitist random search over a feasible box. The problem implements
/// [`CostFunction`] and [`BoxConstraints`]; the
/// population state is sized to the offspring count λ via
/// [`BasicPopulationState::with_size`](crate::BasicPopulationState::with_size):
///
/// ```
/// use basin::{BasicPopulationState, BoxConstraints, CostFunction, Executor, RandomSearch};
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
///     BasicPopulationState::<Vec<f64>>::with_size(16),
/// )
/// .max_iter(500)
/// .run()
/// .unwrap();
/// assert!(result.cost() < 1.0);
/// ```
pub struct RandomSearch {
    lambda: usize,
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
            rng: ChaCha8Rng::seed_from_u64(seed),
        }
    }
}

impl<P, V, F> Solver<P, BasicPopulationState<V, F>> for RandomSearch
where
    F: Scalar + crate::core::parallel::MaybeSend,
    P: CostFunction<Param = V, Output = F>
        + BoxConstraints<Param = V>
        + crate::core::parallel::MaybeSync,
    P::Error: crate::core::parallel::MaybeSend,
    V: SampleUniformBox + Clone + crate::core::parallel::MaybeSync,
{
    type Error = P::Error;

    fn init(
        &mut self,
        problem: &mut Problem<P>,
        mut state: BasicPopulationState<V, F>,
    ) -> Result<BasicPopulationState<V, F>, Self::Error> {
        let lo = problem.inner().lower().clone();
        let hi = problem.inner().upper().clone();
        // The state can arrive with a caller-supplied initial population
        // (`from_population`) or empty (`with_size`). The solver always
        // owns the *first* generation here: clear and resample so the
        // RNG-determined trajectory is reproducible regardless of which
        // constructor was used. Callers who genuinely want a custom
        // initial population should call `from_population` and skip
        // `init` by stepping the solver themselves; `init` is the
        // place where the solver's seeded RNG seeds the run.
        state.candidates.clear();
        state.costs.clear();
        // Sample (sequential RNG), then evaluate the independent draws in one
        // batch (parallel under the `parallel` feature).
        for _ in 0..self.lambda {
            let x = V::sample_uniform_box(&lo, &hi, &mut self.rng);
            state.candidates.push(x);
        }
        state.costs = problem.cost_batch(&state.candidates)?;
        sort_population_ascending(&mut state.candidates, &mut state.costs);
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
        Ok((state, None))
    }
}
