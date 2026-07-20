use crate::core::inner::{InitialState, WarmStart};
use crate::core::math::{SampleStandardNormal, Scalar, ScaleInPlace, ScaledAdd, VectorLen};
use crate::core::problem::{CostFunction, Problem};
use crate::core::rng::{ChaCha8Rng, SeedableRng};
use crate::core::solver::Solver;
use crate::core::state::SolisWetsState;
use crate::core::termination::TerminationReason;
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
/// State: iterate `x`, bias vector `b` (init `0`), step size `ρ`, and
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
/// iteration regardless of which branch is taken, and is seeded once at
/// construction (never re-seeded in [`init`](Solver::init)), so a
/// paused `(SolisWets, SolisWetsState)` pair resumes its stream
/// mid-sequence—the property LS-chain persistence relies on.
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
///   ([`SolisWetsState::new`]'s second argument, or
///   [`with_rho_init`](Self::with_rho_init) when seeding through
///   [`Executor::from_start`](crate::core::executor::Executor::from_start))
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
/// # Termination
///
/// No solver-internal stop (the reference implementation is purely
/// budget-driven). The natural convergence criterion is
/// [`RhoTolerance`](crate::core::termination::RhoTolerance), which binds
/// on [`SolisWetsState`] through
/// [`RhoState`](crate::core::state::RhoState) and fires once `ρ`
/// contracts to the configured floor; combine with
/// [`MaxCostEvals`](crate::core::termination::MaxCostEvals) or
/// [`MaxIter`](crate::core::termination::MaxIter) as a budget. Note an
/// iteration spends one *or two* cost evaluations (the reversal is only
/// evaluated when the forward candidate fails), and criteria run between
/// iterations, so a `MaxCostEvals` budget can be overshot by one
/// evaluation.
///
/// # Backends
///
/// Backend-generic; works with any `V` implementing
/// [`SampleStandardNormal`] + [`ScaledAdd`] + [`ScaleInPlace`] +
/// `Clone`. That covers `Vec<f64>`, `nalgebra::DVector<f64>` (feature
/// `nalgebra`), `ndarray::Array1<f64>` (feature `ndarray`), and
/// `faer::Col<f64>` (feature `faer`). No matrix type and no `linalg`
/// tier involved.
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
/// [`RhoTolerance`](crate::core::termination::RhoTolerance) supplies the
/// convergence test:
///
/// ```
/// use basin::{CostFunction, Executor, RhoTolerance, SolisWets};
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
/// let result = Executor::from_start(Sphere, SolisWets::new(42), vec![2.0, -1.5])
///     .terminate_on(RhoTolerance::new(1e-8))
///     .max_iter(10_000)
///     .run()
///     .unwrap();
/// assert!(result.cost() < 1e-6);
/// ```
#[derive(Clone)]
pub struct SolisWets<F = f64> {
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
    rng: ChaCha8Rng,
}

impl<F: Scalar> SolisWets<F> {
    /// Build a Solis-Wets solver with the 1981 paper's defaults
    /// (`bias` constants 0.4/0.2/0.5, thresholds 5/3, factors 2/0.5,
    /// `rho_init` 1) and a [`ChaCha8Rng`] seeded from `seed`.
    pub fn new(seed: u64) -> Self {
        Self {
            rho_init: F::one(),
            bias_gain: F::from_f64(0.4).unwrap(),
            bias_memory: F::from_f64(0.2).unwrap(),
            bias_decay: F::from_f64(0.5).unwrap(),
            expand_threshold: 5,
            contract_threshold: 3,
            expand_factor: F::from_f64(2.0).unwrap(),
            contract_factor: F::from_f64(0.5).unwrap(),
            rng: ChaCha8Rng::seed_from_u64(seed),
        }
    }

    /// Override the `ρ` that [`InitialState::seed`] (and thus
    /// [`Executor::from_start`](crate::core::executor::Executor::from_start))
    /// uses for a fresh start (default `1`). States built directly via
    /// [`SolisWetsState::new`] carry their own `ρ` and ignore this.
    ///
    /// # Panics
    ///
    /// Panics if `rho_init ≤ 0`.
    pub fn with_rho_init(mut self, rho_init: F) -> Self {
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

impl<P, V, F> Solver<P, SolisWetsState<V, F>> for SolisWets<F>
where
    F: Scalar,
    P: CostFunction<Param = V, Output = F>,
    V: Clone + SampleStandardNormal + ScaledAdd<F> + ScaleInPlace<F>,
{
    type Error = P::Error;

    /// Evaluate the start point once. Resume-idempotent: a resumed chain
    /// state arrives with `cost` already populated and passes through
    /// untouched (bias, `ρ`, and counters are never reset here).
    fn init(
        &mut self,
        problem: &mut Problem<P>,
        mut state: SolisWetsState<V, F>,
    ) -> Result<SolisWetsState<V, F>, Self::Error> {
        if state.cost.is_none() {
            state.cost = Some(problem.cost(&state.x)?);
        }
        Ok(state)
    }

    fn next_iter(
        &mut self,
        problem: &mut Problem<P>,
        mut state: SolisWetsState<V, F>,
    ) -> Result<(SolisWetsState<V, F>, Option<TerminationReason>), Self::Error> {
        let f_x = state
            .cost
            .expect("SolisWets::next_iter called before init evaluated the start point");

        // d ~ N(0, ρ² I). Sampled unconditionally first, so the RNG
        // advances by exactly n draws per iteration whichever branch
        // runs below.
        let mut d = V::sample_standard_normal(&state.x, &mut self.rng);
        d.scale_in_place(state.rho);

        // step = b + d; candidates are x + step and x − step (the
        // paper's ξ and 2x − ξ: bias and noise reverse together).
        let mut step = state.bias.clone();
        step.scaled_add(F::one(), &d);

        let mut candidate = state.x.clone();
        candidate.scaled_add(F::one(), &step);
        let f_forward = problem.cost(&candidate)?;

        if f_forward < f_x {
            // Forward success: b ← bias_memory · b + bias_gain · (b + d).
            state.x = candidate;
            state.cost = Some(f_forward);
            state.bias.scale_in_place(self.bias_memory);
            state.bias.scaled_add(self.bias_gain, &step);
            state.num_success += 1;
            state.num_failure = 0;
        } else {
            let mut reversal = state.x.clone();
            reversal.scaled_add(-F::one(), &step);
            let f_reversal = problem.cost(&reversal)?;

            if f_reversal < f_x {
                // Reversal success: b ← b − bias_gain · (b + d).
                state.x = reversal;
                state.cost = Some(f_reversal);
                state.bias.scaled_add(-self.bias_gain, &step);
                state.num_success += 1;
                state.num_failure = 0;
            } else {
                // Failure (NaN/∞ costs land here too: the comparisons
                // above are strict and false for non-finite values).
                state.bias.scale_in_place(self.bias_decay);
                state.num_failure += 1;
                state.num_success = 0;
            }
        }

        // Step-size adaptation at the end of the loop body, with the
        // counter reset on fire (Rmalschains ordering and semantics; the
        // paper's Step 1 placement is equivalent up to a one-iteration
        // phase shift).
        if state.num_success >= self.expand_threshold {
            state.num_success = 0;
            state.rho = state.rho * self.expand_factor;
        } else if state.num_failure >= self.contract_threshold {
            state.num_failure = 0;
            state.rho = state.rho * self.contract_factor;
        }

        Ok((state, None))
    }
}

// -----------------------------------------------------------------------
// Composition impls: fresh-seed tiers (InitialState → WarmStart →
// MemeticInner), so SolisWets works with `Executor::from_start` and as a
// CmaInject/DeInject/BasinHopping inner.
// -----------------------------------------------------------------------

impl<V, F> InitialState<V> for SolisWets<F>
where
    F: Scalar,
    V: Clone + VectorLen + ScaleInPlace<F>,
{
    type State = SolisWetsState<V, F>;

    fn seed(&self, x: &V) -> SolisWetsState<V, F> {
        // σ-free seed at the solver's natural default scale
        // (`with_rho_init`, default 1).
        SolisWetsState::new(x.clone(), self.rho_init)
    }
}

impl<V, F> WarmStart<V> for SolisWets<F>
where
    F: Scalar,
    V: Clone + VectorLen + ScaleInPlace<F>,
{
}

impl<V, F> MemeticInner<V, F> for SolisWets<F>
where
    F: Scalar,
    V: Clone + VectorLen + ScaleInPlace<F>,
{
    /// σ-scaled seed: `ρ` tracks the outer's step-size, so the walk's
    /// exploration matches the outer distribution's spread.
    ///
    /// # Panics
    ///
    /// Panics if `sigma ≤ 0` (via [`SolisWetsState::new`]). All shipped
    /// callers pass a positive scale: `CmaInject`/`BoundedCmaInject`
    /// forward the outer CMA step-size (positive by construction: its
    /// multiplicative update never underflows to zero), and `DeInject`
    /// passes the constant `1`.
    fn seed_scaled(&self, x: &V, sigma: F) -> SolisWetsState<V, F> {
        SolisWetsState::new(x.clone(), sigma)
    }
}

impl<V, F> crate::core::inner::ResumableInner<V, F> for SolisWets<F>
where
    F: Scalar,
    V: Clone + VectorLen + ScaleInPlace<F>,
{
    type State = SolisWetsState<V, F>;

    /// Fresh chain: a copy of the prototype's hyperparameters with a
    /// private RNG stream seeded from `seed` (the prototype's own RNG is
    /// never drawn), plus a state at `(x, ρ = scale)` with the cost slot
    /// primed to `fx`—the chain snapshot for Solis-Wets is exactly
    /// `(#s, #f, bias, ρ)` (MA-SW-Chains §II.C), so a fresh chain spends
    /// zero budget re-scoring the point the outer already evaluated.
    fn seed_chain(&self, x: &V, fx: F, scale: F, seed: u64) -> (Self, Self::State) {
        // Struct-update over a clone so a future hyperparameter can't
        // be forgotten here; only the RNG stream is replaced.
        let sw = Self {
            rng: ChaCha8Rng::seed_from_u64(seed),
            ..self.clone()
        };
        let mut state = SolisWetsState::new(x.clone(), scale);
        state.cost = Some(fx);
        (sw, state)
    }

    /// Reset the local iteration counter so the resumed segment starts
    /// at iter 0; bias, `ρ`, and the streak counters persist—that's
    /// the chain.
    fn prepare_resume(&self, state: &mut Self::State) {
        state.iter = 0;
    }

    // `segment_criteria` stays the default (none): the reference
    // implementation runs Solis-Wets segments purely budget-driven.
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
        let mut solver = SolisWets::<f64>::new(1);
        let mut problem = Problem::new(AlwaysImproving::new());
        let state = SolisWetsState::new(vec![0.0, 0.0, 0.0], 0.5);
        let state = solver.init(&mut problem, state).unwrap();

        let x_old = state.x.clone();
        let bias_old = state.bias.clone();
        let (state, reason) = solver.next_iter(&mut problem, state).unwrap();
        assert!(reason.is_none());

        // Forward success moved to x + step with step = x_new − x_old,
        // so the expected bias is 0.2 b_old + 0.4 (x_new − x_old).
        let expected: Vec<f64> = (0..3)
            .map(|i| 0.2 * bias_old[i] + 0.4 * (state.x[i] - x_old[i]))
            .collect();
        approx_eq(&state.bias, &expected, 1e-12);
        assert_eq!(state.success_count(), 1);
        assert_eq!(state.failure_count(), 0);
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
            let mut solver = SolisWets::<f64>::new(seed);
            let mut problem = Problem::new(Sphere);
            let state = SolisWetsState::new(vec![0.3, -0.2], 0.4);
            let state = solver.init(&mut problem, state).unwrap();
            let x_old = state.x.clone();
            let bias_old = state.bias.clone();
            let f_old = state.cost.unwrap();

            let (state, _) = solver.next_iter(&mut problem, state).unwrap();
            let moved = state.x != x_old;
            let improved = state.cost.unwrap() < f_old;
            if moved && improved && state.success_count() == 1 {
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
                        .map(|i| bias_old[i] + 0.4 * (state.x[i] - x_old[i]))
                        .collect();
                    approx_eq(&state.bias, &expected, 1e-12);
                    return;
                }
            }
        }
        panic!("no seed in 0..64 produced a first-iteration reversal success");
    }

    #[test]
    fn failure_decays_bias_and_counts() {
        let mut solver = SolisWets::<f64>::new(3);
        let mut problem = Problem::new(Constant);
        let mut state = SolisWetsState::new(vec![1.0, 2.0], 0.5);
        // Seed a non-zero bias so the decay is observable.
        state.bias = vec![0.8, -0.4];
        let state = solver.init(&mut problem, state).unwrap();

        let (state, reason) = solver.next_iter(&mut problem, state).unwrap();
        assert!(reason.is_none());
        approx_eq(&state.bias, &[0.4, -0.2], 1e-12);
        assert_eq!(state.failure_count(), 1);
        assert_eq!(state.success_count(), 0);
        assert_eq!(state.x, vec![1.0, 2.0]); // iterate unmoved
        // Two evals spent this iteration (forward + reversal) plus init.
        assert_eq!(problem.counts().cost_evals, 3);
    }

    #[test]
    fn expansion_fires_at_threshold_and_resets_counter() {
        let mut solver = SolisWets::<f64>::new(5);
        let mut problem = Problem::new(AlwaysImproving::new());
        let state = SolisWetsState::new(vec![0.0; 4], 1.0);
        let mut state = solver.init(&mut problem, state).unwrap();

        for i in 1..=5 {
            let (s, _) = solver.next_iter(&mut problem, state).unwrap();
            state = s;
            if i < 5 {
                assert_eq!(state.success_count(), i);
                assert!((state.rho() - 1.0).abs() < 1e-15, "rho moved early");
            }
        }
        // Fifth success fires the expansion and resets the streak.
        assert!((state.rho() - 2.0).abs() < 1e-15);
        assert_eq!(state.success_count(), 0);
    }

    #[test]
    fn contraction_fires_at_threshold_and_resets_counter() {
        let mut solver = SolisWets::<f64>::new(7);
        let mut problem = Problem::new(Constant);
        let state = SolisWetsState::new(vec![1.0; 3], 1.0);
        let mut state = solver.init(&mut problem, state).unwrap();

        for i in 1..=3 {
            let (s, _) = solver.next_iter(&mut problem, state).unwrap();
            state = s;
            if i < 3 {
                assert_eq!(state.failure_count(), i);
                assert!((state.rho() - 1.0).abs() < 1e-15, "rho moved early");
            }
        }
        assert!((state.rho() - 0.5).abs() < 1e-15);
        assert_eq!(state.failure_count(), 0);
    }

    #[test]
    fn init_is_resume_idempotent() {
        let mut solver = SolisWets::<f64>::new(11);
        let mut problem = Problem::new(Sphere);
        let state = SolisWetsState::new(vec![1.5, -0.5], 0.7);
        let mut state = solver.init(&mut problem, state).unwrap();
        for _ in 0..10 {
            let (s, _) = solver.next_iter(&mut problem, state).unwrap();
            state = s;
        }

        let x = state.x.clone();
        let cost = state.cost;
        let bias = state.bias.clone();
        let rho = state.rho;
        let (ns, nf) = (state.num_success, state.num_failure);
        let evals_before = problem.counts().cost_evals;

        // Re-running init on an advanced state must not touch anything
        // and must not spend an evaluation: the resume contract.
        let state = solver.init(&mut problem, state).unwrap();
        assert_eq!(state.x, x);
        assert_eq!(state.cost, cost);
        assert_eq!(state.bias, bias);
        assert_eq!(state.rho, rho);
        assert_eq!((state.num_success, state.num_failure), (ns, nf));
        assert_eq!(problem.counts().cost_evals, evals_before);
    }

    #[test]
    fn seed_and_seed_scaled_set_rho() {
        let solver = SolisWets::<f64>::new(13).with_rho_init(0.25);
        let s: SolisWetsState<Vec<f64>> = solver.seed(&vec![1.0, 2.0]);
        assert!((s.rho() - 0.25).abs() < 1e-15);
        assert_eq!(s.x, vec![1.0, 2.0]);

        let s: SolisWetsState<Vec<f64>> = solver.seed_scaled(&vec![1.0, 2.0], 0.05);
        assert!((s.rho() - 0.05).abs() < 1e-15);
    }
}
