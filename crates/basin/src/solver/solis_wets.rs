use crate::core::inner::{InitialState, WarmStart};
use crate::core::math::{
    SampleStandardNormal, Scalar, ScaleInPlace, ScaledAdd, VectorLen,
};
use crate::core::problem::{CostFunction, Problem};
use crate::core::rng::{ChaCha8Rng, SeedableRng};
use crate::core::solver::Solver;
use crate::core::state::{PointState, State};
use crate::core::termination::Termination;
use crate::solver::cma_inject::MemeticInner;

/// Solis-Wets adaptive random local search.
///
/// A randomized hill-climber with an adaptive step size and a success-
/// direction bias: the cheapest local search in basin (O(n) memory and
/// time per iteration, cost evaluations only), and the classic
/// local-search operator of the memetic literature (MA-SW-Chains won the
/// CEC'2010 large-scale competition with it).
///
/// # Algorithm
///
/// The solver owns the bias vector `b` (init `0`), step size `ρ`, and
/// successive success/failure counters `#s`/`#f`. Each iteration:
///
/// ```text
/// d ~ N(0, ρ² I)                    # per-coordinate, σ = ρ
/// step ← b + d
/// if f(x + step) < f(x):            # forward success
///     x ← x + step;  b ← 0.2 b + 0.4 (b + d);  #s += 1, #f = 0
/// else if f(x − step) < f(x):       # reversal success
///     x ← x − step;  b ← b − 0.4 (b + d);      #s += 1, #f = 0
/// else:                             # failure (1 or 2 evals spent)
///     b ← 0.5 b;                                #f += 1, #s = 0
/// if #s ≥ 5: #s = 0, ρ ← 2 ρ        # expand
/// else if #f ≥ 3: #f = 0, ρ ← ρ/2   # contract
/// ```
///
/// All constants above are the defaults from Solis & Wets (1981) and are
/// configurable via the `with_*` builders. Two conventions from the
/// memetic lineage (matching the Rmalschains reference implementation)
/// are baked in: `ρ` is the per-coordinate standard *deviation* (the
/// paper's Algorithm 1 reads covariance `ρI`), and the streak counter
/// resets when its expansion/contraction fires (the paper's literal
/// reading would fire every iteration for the rest of the streak).
///
/// The iterate only ever moves to a strictly better point, so
/// `state.cost()` is non-increasing and equals `state.best_cost()`.
/// This is a *local* search method: the reversal step notwithstanding,
/// no global convergence claim applies (Solis & Wets 1981, local-search
/// theorem under H1 + H3).
///
/// # Reproducibility
///
/// The solver carries a [`ChaCha8Rng`] seeded from the `seed: u64`
/// passed to [`new`](Self::new). Same seed → same trajectory, on every
/// platform basin builds for (including `wasm32-unknown-unknown`). The
/// RNG advances by exactly `n` standard-normal component draws per
/// iteration regardless of which branch is taken. Fresh initialization
/// restores the configured seed, initial step size, zero bias, and streak
/// counters, and reevaluates the supplied point. Progress uses [`PointState`].
/// Exact solver-aware checkpoints preserve the RNG and adaptive model and
/// skip initialization. State-only snapshots start a fresh run.
///
/// Persistent local-search chains likewise retain the solver and state and
/// skip initialization between segments. Each segment deliberately resets its
/// budgets, convergence history, and progress metadata. With `serde`, exact
/// solver-aware serialization is available when `V` and `F` support it.
///
/// # Contract
///
/// - **Caller must:** provide only a [`CostFunction`]; the search is
///   unconstrained. Unlike the Rmalschains implementation, candidates
///   are *not* clipped to a box: box handling is adapter territory
///   (tenet 4). A problem that soft-rejects infeasible points with
///   `Ok(f64::INFINITY)` works naturally: rejected candidates register
///   as failures and `ρ` contracts back toward the feasible region.
/// - **Caller should:** pick the initial `ρ`
///   with [`with_initial_step_size`](Self::with_initial_step_size)
///   on the scale of the distance to the sought minimum; `ρ` adapts
///   quickly in either direction.
/// - Any real cost stops improving once `ρ` grows past its basin, and
///   contraction then pulls `ρ` back, so no expansion guard is shipped.
///   The truly pathological case (a cost that keeps returning improving
///   values even for non-finite candidates, i.e. one that ignores `x`)
///   can expand `ρ` all the way to `+∞`, and from there contraction
///   cannot recover it (`0.5 · ∞ = ∞`): the run makes no further
///   progress and `RhoTolerance` never fires, so budgets (`MaxIter`,
///   `MaxCostEvals`) remain the caller's job.
///
/// # Convergence
///
/// The default is budget-driven. Configure
/// [`with_absolute_step_size_tolerance`](Self::with_absolute_step_size_tolerance)
/// to stop when the mutation standard deviation `ρ` reaches the threshold.
/// `None` disables this optional check; zero tests exact collapse.
/// The reason is [`crate::TerminationCode::RhoTolerance`]. Each iteration spends
/// one or two evaluations, so an executor cost budget can be exceeded by one.
/// Optional cost- and step-change checks observe accepted moves; failed
/// proposals do not turn an unchanged point into convergence. Native step-size
/// checks remain active after rejection. Retain the solver with
/// [`Executor::run_with_solver`](crate::Executor::run_with_solver) to inspect
/// [`step_size`](Self::step_size), [`bias`](Self::bias), and streak counters.
///
/// # Backends
///
/// `Vec<F>`, `nalgebra::DVector<F>`, `ndarray::Array1<F>`, and `faer::Col<F>`
/// with `F = f64` (default) or `f32`. Custom vectors need
/// [`SampleStandardNormal`], [`ScaledAdd<F>`], [`ScaleInPlace<F>`],
/// [`VectorLen`], and `Clone`. No matrix capability is required.
///
/// # References
///
/// - Solis, F. J., and Wets, R. J.-B. (1981). "Minimization by Random
///   Search Techniques." *Mathematics of Operations Research*, 6(1),
///   19-30. <https://doi.org/10.1287/moor.6.1.19>
/// - Molina, D., Lozano, M., and Herrera, F. (2010). "MA-SW-Chains:
///   Memetic algorithm based on local search chains for large scale
///   continuous global optimization." *IEEE Congress on Evolutionary
///   Computation (CEC 2010)*, 3153-3160.
///   <https://doi.org/10.1109/CEC.2010.5586034>
///
/// # Examples
///
/// Minimize a sphere from a fixed start;
/// [`with_absolute_step_size_tolerance`](Self::with_absolute_step_size_tolerance) supplies the
/// convergence test:
///
/// ```
/// use basin::{CostFunction, Executor, SolisWets};
///
/// struct Sphere;
/// impl CostFunction for Sphere {
///     type Param = Vec<f64>;
///     type Output = f64;
///     type Error = std::convert::Infallible;
///     fn cost(&self, x: &Vec<f64>) -> Result<f64, Self::Error> {
///         Ok(x.iter().map(|xi| xi * xi).sum())
///     }
/// }
///
/// let result = Executor::from_start(Sphere, (SolisWets::new(42)).with_absolute_step_size_tolerance(1e-8), vec![2.0, -1.5])
///
///     .max_iter(10_000)
///     .run()
///     .unwrap();
/// assert!(result.cost() < 1e-6);
/// ```
#[derive(Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SolisWets<V, F: Scalar = f64> {
    step_tolerance: Option<F>,
    /// `ρ` used by [`InitialState::seed`] for fresh, unscaled starts.
    rho_init: F,
    /// Bias gain: on success, `b += bias_gain · (b + d)`.
    bias_gain: F,
    /// Bias memory on a forward success: `b ← bias_memory · b + …`.
    bias_memory: F,
    /// Bias decay on failure: `b ← bias_decay · b`.
    bias_decay: F,
    /// Successive successes before `ρ` expands (counter then resets).
    expand_threshold: u32,
    /// Successive failures before `ρ` contracts (counter then resets).
    contract_threshold: u32,
    /// `ρ` multiplier on expansion.
    expand_factor: F,
    /// `ρ` multiplier on contraction.
    contract_factor: F,
    seed: u64,
    rng: ChaCha8Rng,
    bias: Option<V>,
    rho: F,
    num_success: u32,
    num_failure: u32,
    accepted_iterate: bool,
}

impl<V, F: Scalar> SolisWets<V, F> {
    /// Current mutation standard deviation. Before initialization, this is `1`.
    /// Fresh initialization restores the configured initial step size.
    pub fn step_size(&self) -> F {
        self.rho
    }

    /// Current search bias, or `None` before initialization.
    pub fn bias(&self) -> Option<&V> {
        self.bias.as_ref()
    }

    /// Consecutive successes since the most recent failure or expansion.
    pub fn success_count(&self) -> u32 {
        self.num_success
    }

    /// Consecutive failures since the most recent success or contraction.
    pub fn failure_count(&self) -> u32 {
        self.num_failure
    }

    /// Stop when the observed radius or step size is <= the tolerance.
    ///
    /// Disabled by default. `None` disables the test and zero requests an
    /// exact-zero threshold. The tolerance must be finite and nonnegative.
    /// Checked at initialized iteration boundaries. This observation does not
    /// change the algorithm's radius or step-size update schedule.
    pub fn with_absolute_step_size_tolerance(
        mut self,
        value: impl Into<Option<F>>,
    ) -> Self {
        self.step_tolerance =
            crate::core::convergence::optional_tolerance(value);
        self
    }

    /// Build a Solis-Wets solver with the 1981 paper's defaults
    /// (`bias` constants 0.4/0.2/0.5, thresholds 5/3, factors 2/0.5,
    /// `rho_init` 1) and a [`ChaCha8Rng`] seeded from `seed`.
    pub fn new(seed: u64) -> Self {
        Self {
            rho_init: F::one(),
            step_tolerance: None,
            bias_gain: F::from_f64(0.4).unwrap(),
            bias_memory: F::from_f64(0.2).unwrap(),
            bias_decay: F::from_f64(0.5).unwrap(),
            expand_threshold: 5,
            contract_threshold: 3,
            expand_factor: F::from_f64(2.0).unwrap(),
            contract_factor: F::from_f64(0.5).unwrap(),
            seed,
            rng: ChaCha8Rng::seed_from_u64(seed),
            bias: None,
            rho: F::one(),
            num_success: 0,
            num_failure: 0,
            accepted_iterate: true,
        }
    }

    /// Configure the initial mutation standard deviation for every fresh run
    /// (default `1`). Exact checkpoints preserve the current adapted value.
    pub fn with_initial_step_size(mut self, rho_init: F) -> Self {
        assert!(
            rho_init > F::zero(),
            "rho_init must be > 0, got {:?}",
            rho_init
        );
        self.rho_init = rho_init;
        self
    }

    /// Override the bias gain (default `0.4`): the fraction of the
    /// successful step `b + d` folded into the bias.
    ///
    /// # Panics
    ///
    /// Panics if `bias_gain < 0`.
    pub fn with_bias_gain(mut self, bias_gain: F) -> Self {
        assert!(
            bias_gain >= F::zero(),
            "bias_gain must be >= 0, got {:?}",
            bias_gain
        );
        self.bias_gain = bias_gain;
        self
    }

    /// Override the bias memory (default `0.2`): the fraction of the old
    /// bias kept on a forward success.
    ///
    /// # Panics
    ///
    /// Panics if `bias_memory < 0`.
    pub fn with_bias_memory(mut self, bias_memory: F) -> Self {
        assert!(
            bias_memory >= F::zero(),
            "bias_memory must be >= 0, got {:?}",
            bias_memory
        );
        self.bias_memory = bias_memory;
        self
    }

    /// Override the bias decay (default `0.5`): the factor applied to
    /// the bias on a failed iteration.
    ///
    /// # Panics
    ///
    /// Panics if `bias_decay < 0`.
    pub fn with_bias_decay(mut self, bias_decay: F) -> Self {
        assert!(
            bias_decay >= F::zero(),
            "bias_decay must be >= 0, got {:?}",
            bias_decay
        );
        self.bias_decay = bias_decay;
        self
    }

    /// Override the expansion threshold (default `5`): successive
    /// successes before `ρ` is multiplied by
    /// [`with_expand_factor`](Self::with_expand_factor)'s factor.
    ///
    /// # Panics
    ///
    /// Panics if `expand_threshold` is `0`.
    pub fn with_expand_threshold(mut self, expand_threshold: u32) -> Self {
        assert!(
            expand_threshold >= 1,
            "expand_threshold must be >= 1, got {}",
            expand_threshold
        );
        self.expand_threshold = expand_threshold;
        self
    }

    /// Override the contraction threshold (default `3`): successive
    /// failures before `ρ` is multiplied by
    /// [`with_contract_factor`](Self::with_contract_factor)'s factor.
    ///
    /// # Panics
    ///
    /// Panics if `contract_threshold` is `0`.
    pub fn with_contract_threshold(mut self, contract_threshold: u32) -> Self {
        assert!(
            contract_threshold >= 1,
            "contract_threshold must be >= 1, got {}",
            contract_threshold
        );
        self.contract_threshold = contract_threshold;
        self
    }

    /// Override the expansion factor (default `2`).
    ///
    /// # Panics
    ///
    /// Panics if `expand_factor ≤ 0`.
    pub fn with_expand_factor(mut self, expand_factor: F) -> Self {
        assert!(
            expand_factor > F::zero(),
            "expand_factor must be > 0, got {:?}",
            expand_factor
        );
        self.expand_factor = expand_factor;
        self
    }

    /// Override the contraction factor (default `0.5`).
    ///
    /// # Panics
    ///
    /// Panics if `contract_factor ≤ 0`.
    pub fn with_contract_factor(mut self, contract_factor: F) -> Self {
        assert!(
            contract_factor > F::zero(),
            "contract_factor must be > 0, got {:?}",
            contract_factor
        );
        self.contract_factor = contract_factor;
        self
    }
}

impl<P, V, F> Solver<P, PointState<V, F>> for SolisWets<V, F>
where
    F: Scalar,
    P: CostFunction<Param = V, Output = F>,
    V: Clone
        + VectorLen
        + SampleStandardNormal
        + ScaledAdd<F>
        + ScaleInPlace<F>,
{
    type Error = P::Error;

    fn init(
        &mut self,
        problem: &mut Problem<P>,
        mut state: PointState<V, F>,
    ) -> Result<PointState<V, F>, Self::Error> {
        state.reset();
        self.reset_model(state.param(), self.rho_init, self.seed);
        let cost = problem.cost(state.param())?;
        state.replace(state.param().clone(), cost);
        Ok(state)
    }

    fn next_iter(
        &mut self,
        problem: &mut Problem<P>,
        mut state: PointState<V, F>,
    ) -> Result<crate::SolverStep<PointState<V, F>>, Self::Error> {
        let f_x = state.cost();
        self.accepted_iterate = false;
        let bias = self.bias.as_mut().expect("SolisWets must be initialized");

        // d ~ N(0, ρ² I). Sampled unconditionally first, so the RNG
        // advances by exactly n draws per iteration whichever branch
        // runs below.
        let mut d = V::sample_standard_normal(state.param(), &mut self.rng);
        d.scale_in_place(self.rho);

        // step = b + d; candidates are x + step and x − step (the
        // paper's ξ and 2x − ξ: bias and noise reverse together).
        let mut step = bias.clone();
        step.scaled_add(F::one(), &d);

        let mut candidate = state.param().clone();
        candidate.scaled_add(F::one(), &step);
        let f_forward = problem.cost(&candidate)?;

        if f_forward < f_x {
            // Forward success: b ← bias_memory · b + bias_gain · (b + d).
            state.replace(candidate, f_forward);
            self.accepted_iterate = true;
            bias.scale_in_place(self.bias_memory);
            bias.scaled_add(self.bias_gain, &step);
            self.num_success += 1;
            self.num_failure = 0;
        } else {
            let mut reversal = state.param().clone();
            reversal.scaled_add(-F::one(), &step);
            let f_reversal = problem.cost(&reversal)?;

            if f_reversal < f_x {
                // Reversal success: b ← b − bias_gain · (b + d).
                state.replace(reversal, f_reversal);
                self.accepted_iterate = true;
                bias.scaled_add(-self.bias_gain, &step);
                self.num_success += 1;
                self.num_failure = 0;
            } else {
                // Equal, NaN, and positive-infinite proposals cannot improve
                // a finite incumbent.
                bias.scale_in_place(self.bias_decay);
                self.num_failure += 1;
                self.num_success = 0;
            }
        }

        // Step-size adaptation at the end of the loop body, with the
        // counter reset on fire (Rmalschains ordering and semantics; the
        // paper's Step 1 placement is equivalent up to a one-iteration
        // phase shift).
        if self.num_success >= self.expand_threshold {
            self.num_success = 0;
            self.rho = self.rho * self.expand_factor;
        } else if self.num_failure >= self.contract_threshold {
            self.num_failure = 0;
            self.rho = self.rho * self.contract_factor;
        }

        Ok(crate::SolverStep::from((state, None)))
    }

    fn should_check_iterate_change(&self) -> bool {
        self.accepted_iterate
    }

    fn terminate(&self, _state: &PointState<V, F>) -> Option<Termination<F>> {
        let tolerance = self.step_tolerance?;
        let metric = self.rho;
        (metric.is_finite() && metric <= tolerance).then(|| {
            Termination::upper_bound(
                crate::ConvergenceTest::Radius,
                metric,
                tolerance,
                tolerance,
                None,
            )
        })
    }
}

impl<V, F: Scalar> SolisWets<V, F>
where
    V: Clone + VectorLen + ScaleInPlace<F>,
{
    fn reset_model(&mut self, x: &V, scale: F, seed: u64) {
        assert!(x.vec_len() >= 1, "SolisWets requires a non-empty x");
        assert!(scale > F::zero(), "SolisWets requires a positive step size");
        let mut bias = x.clone();
        bias.scale_in_place(F::zero());
        self.bias = Some(bias);
        self.rho = scale;
        self.num_success = 0;
        self.num_failure = 0;
        self.accepted_iterate = true;
        self.rng = ChaCha8Rng::seed_from_u64(seed);
    }
}

impl<V: Clone, F: Scalar> InitialState<V> for SolisWets<V, F> {
    type State = PointState<V, F>;

    fn seed(&self, x: &V) -> Self::State {
        PointState::new(x.clone())
    }
}

impl<V: Clone, F: Scalar> WarmStart<V> for SolisWets<V, F> {}

impl<V: Clone, F: Scalar> MemeticInner<V, F> for SolisWets<V, F> {
    /// Configure the next fresh run's initial step size from the outer scale.
    fn seed_scaled(&mut self, x: &V, sigma: F) -> Self::State {
        assert!(sigma > F::zero(), "SolisWets requires a positive step size");
        self.rho_init = sigma;
        self.seed(x)
    }
}

impl<V, F> crate::core::inner::ResumableInner<V, F> for SolisWets<V, F>
where
    F: Scalar,
    V: Clone + VectorLen + ScaleInPlace<F>,
{
    type State = PointState<V, F>;

    /// Build an initialized chain from the supplied evaluated point and scale.
    /// The prototype's RNG is untouched, and no objective call is needed.
    fn seed_chain(
        &self,
        x: &V,
        fx: F,
        scale: F,
        seed: u64,
    ) -> (Self, Self::State) {
        let mut sw = self.clone();
        sw.seed = seed;
        sw.rho_init = scale;
        sw.reset_model(x, scale, seed);
        let mut state = PointState::new(x.clone());
        state.replace(x.clone(), fx);
        (sw, state)
    }

    fn seeded_chain_is_initialized(&self) -> bool {
        true
    }

    fn prepare_resume(&self, state: &mut Self::State) {
        state.reset_progress();
    }

    fn configure_segment(
        &mut self,
        _state: &Self::State,
        _control: &mut crate::RunControl<Self::State>,
    ) {
        self.accepted_iterate = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    /// Cost that ignores `x` and returns a strictly decreasing sequence:
    /// every candidate evaluation "succeeds", forcing the forward-success
    /// branch each iteration.
    struct AlwaysImproving {
        next: Cell<f64>,
    }
    impl AlwaysImproving {
        fn new() -> Self {
            Self {
                next: Cell::new(1000.0),
            }
        }
    }
    impl CostFunction for AlwaysImproving {
        type Param = Vec<f64>;
        type Output = f64;
        type Error = std::convert::Infallible;
        fn cost(&self, _x: &Vec<f64>) -> Result<f64, Self::Error> {
            let c = self.next.get();
            self.next.set(c - 1.0);
            Ok(c)
        }
    }

    /// Constant cost: no candidate ever strictly improves, forcing the
    /// failure branch (both evals spent) each iteration.
    struct Constant;
    impl CostFunction for Constant {
        type Param = Vec<f64>;
        type Output = f64;
        type Error = std::convert::Infallible;
        fn cost(&self, _x: &Vec<f64>) -> Result<f64, Self::Error> {
            Ok(1.0)
        }
    }

    /// Sphere for real convergence-adjacent checks.
    struct Sphere;
    impl CostFunction for Sphere {
        type Param = Vec<f64>;
        type Output = f64;
        type Error = std::convert::Infallible;
        fn cost(&self, x: &Vec<f64>) -> Result<f64, Self::Error> {
            Ok(x.iter().map(|xi| xi * xi).sum())
        }
    }

    fn approx_eq(a: &[f64], b: &[f64], tol: f64) {
        assert_eq!(a.len(), b.len());
        for (ai, bi) in a.iter().zip(b) {
            assert!((ai - bi).abs() < tol, "{a:?} != {b:?}");
        }
    }

    #[test]
    fn forward_success_updates_bias_and_counters() {
        let mut solver =
            SolisWets::<_, f64>::new(1).with_initial_step_size(0.5);
        let mut problem = Problem::new(AlwaysImproving::new());
        let state = PointState::new(vec![0.0, 0.0, 0.0]);
        let state = solver.init(&mut problem, state).unwrap();

        let x_old = state.param().clone();
        let bias_old = solver.bias().unwrap().clone();
        let (state, _, reason) =
            solver.next_iter(&mut problem, state).unwrap().into_parts();
        assert!(reason.is_none());

        // Forward success moved to x + step with step = x_new − x_old,
        // so the expected bias is 0.2 b_old + 0.4 (x_new − x_old).
        let expected: Vec<f64> = (0..3)
            .map(|i| 0.2 * bias_old[i] + 0.4 * (state.param()[i] - x_old[i]))
            .collect();
        approx_eq(solver.bias().unwrap(), &expected, 1e-12);
        assert_eq!(solver.success_count(), 1);
        assert_eq!(solver.failure_count(), 0);
    }

    #[test]
    fn reversal_success_updates_bias_and_moves_backward() {
        // Sphere from the origin-adjacent point: whichever direction the
        // first candidate goes, rig it so the reversal wins by starting
        // at a point where f(x) is small but nonzero and the forward
        // candidate happens to fail. Deterministic via fixed seed: probe
        // seeds until the first iteration takes the reversal branch,
        // then assert its algebra. The probe is itself deterministic.
        for seed in 0..64 {
            let mut solver =
                SolisWets::<_, f64>::new(seed).with_initial_step_size(0.4);
            let mut problem = Problem::new(Sphere);
            let state = PointState::new(vec![0.3, -0.2]);
            let state = solver.init(&mut problem, state).unwrap();
            let x_old = state.param().clone();
            let bias_old = solver.bias().unwrap().clone();
            let f_old = state.cost();

            let (state, _, _) =
                solver.next_iter(&mut problem, state).unwrap().into_parts();
            let moved = state.param() != &x_old;
            let improved = state.cost() < f_old;
            if moved && improved && solver.success_count() == 1 {
                // Distinguish reversal from forward via the bias formula:
                // reversal ⇒ b_new = b_old + 0.4 (x_new − x_old)
                // (since x_new − x_old = −step). Forward would give
                // 0.2 b_old + 0.4 (x_new − x_old); with b_old = 0 the two
                // coincide, so only accept iterations where the reversal
                // eval count (2 evals) identifies the branch.
                let evals_this_iter = problem.counts().cost_evals;
                // init spent 1; forward spends 1 more, reversal 2 more.
                if evals_this_iter == 3 {
                    let expected: Vec<f64> = (0..2)
                        .map(|i| {
                            bias_old[i] + 0.4 * (state.param()[i] - x_old[i])
                        })
                        .collect();
                    approx_eq(solver.bias().unwrap(), &expected, 1e-12);
                    return;
                }
            }
        }
        panic!("no seed in 0..64 produced a first-iteration reversal success");
    }

    #[test]
    fn failure_decays_bias_and_counts() {
        let mut solver =
            SolisWets::<_, f64>::new(3).with_initial_step_size(0.5);
        let mut problem = Problem::new(Constant);
        let state = PointState::new(vec![1.0, 2.0]);
        let state = solver.init(&mut problem, state).unwrap();
        // Seed a nonzero bias after initialization so the decay is observable.
        solver.bias = Some(vec![0.8, -0.4]);

        let (state, _, reason) =
            solver.next_iter(&mut problem, state).unwrap().into_parts();
        assert!(reason.is_none());
        approx_eq(solver.bias().unwrap(), &[0.4, -0.2], 1e-12);
        assert_eq!(solver.failure_count(), 1);
        assert_eq!(solver.success_count(), 0);
        assert_eq!(state.param(), &vec![1.0, 2.0]); // iterate unmoved
        // Two evals spent this iteration (forward + reversal) plus init.
        assert_eq!(problem.counts().cost_evals, 3);
    }

    #[test]
    fn expansion_fires_at_threshold_and_resets_counter() {
        let mut solver = SolisWets::<_, f64>::new(5);
        let mut problem = Problem::new(AlwaysImproving::new());
        let state = PointState::new(vec![0.0; 4]);
        let mut state = solver.init(&mut problem, state).unwrap();

        for i in 1..=5 {
            let (s, _, _) =
                solver.next_iter(&mut problem, state).unwrap().into_parts();
            state = s;
            if i < 5 {
                assert_eq!(solver.success_count(), i);
                assert!(
                    (solver.step_size() - 1.0).abs() < 1e-15,
                    "rho moved early"
                );
            }
        }
        // Fifth success fires the expansion and resets the streak.
        assert!((solver.step_size() - 2.0).abs() < 1e-15);
        assert_eq!(solver.success_count(), 0);
    }

    #[test]
    fn contraction_fires_at_threshold_and_resets_counter() {
        let mut solver = SolisWets::<_, f64>::new(7);
        let mut problem = Problem::new(Constant);
        let state = PointState::new(vec![1.0; 3]);
        let mut state = solver.init(&mut problem, state).unwrap();

        for i in 1..=3 {
            let (s, _, _) =
                solver.next_iter(&mut problem, state).unwrap().into_parts();
            state = s;
            if i < 3 {
                assert_eq!(solver.failure_count(), i);
                assert!(
                    (solver.step_size() - 1.0).abs() < 1e-15,
                    "rho moved early"
                );
            }
        }
        assert!((solver.step_size() - 0.5).abs() < 1e-15);
        assert_eq!(solver.failure_count(), 0);
    }

    #[test]
    fn fresh_init_reevaluates_progress_and_resets_adaptation() {
        let mut solver =
            SolisWets::<_, f64>::new(11).with_initial_step_size(0.7);
        let mut problem = Problem::new(Sphere);
        let state = PointState::new(vec![1.5, -0.5]);
        let mut state = solver.init(&mut problem, state).unwrap();
        for _ in 0..10 {
            state = solver.next_iter(&mut problem, state).unwrap().state;
        }
        state.replace(state.param().clone(), -1000.0);
        let state = solver.init(&mut problem, state).unwrap();
        assert_eq!(state.cost(), problem.inner().cost(state.param()).unwrap());
        assert_eq!(solver.bias().unwrap(), &vec![0.0; 2]);
        assert_eq!(solver.step_size(), 0.7);
        assert_eq!(solver.success_count(), 0);
        assert_eq!(solver.failure_count(), 0);
    }

    #[test]
    fn seed_and_seed_scaled_set_rho() {
        let mut solver =
            SolisWets::<_, f64>::new(13).with_initial_step_size(0.25);
        let mut p = Problem::new(Sphere);
        let state = solver.seed(&vec![1.0, 2.0]);
        let state = solver.init(&mut p, state).unwrap();
        assert_eq!(solver.step_size(), 0.25);
        assert_eq!(state.param(), &vec![1.0, 2.0]);
        let state = solver.seed_scaled(&vec![1.0, 2.0], 0.05);
        solver.init(&mut p, state).unwrap();
        assert_eq!(solver.step_size(), 0.05);
    }
}
