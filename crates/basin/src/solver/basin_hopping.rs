//! Basin-hopping: a stochastic global optimizer that walks the
//! *transformed* landscape `Ẽ(x) = min{f(x)}` (the cost of the local
//! minimum reached from `x`) with a Metropolis Monte-Carlo loop.
//!
//! Each outer iteration perturbs the current iterate, runs an inner local
//! solver to convergence, and accepts or rejects the new local minimum via
//! a temperature-parameterized Metropolis test, with an adaptive step size
//! that drives the empirical acceptance rate toward a target (0.5 by
//! default). See [`BasinHopping`] for the full description and references.

use crate::core::inner::{InitialState, InnerExecutor, WarmStart};
use crate::core::math::{SampleUniformBox, Scalar, VectorLen};
use crate::core::problem::{CostFunction, Problem};
use crate::core::rng::{ChaCha8Rng, Rng, RngExt, SeedableRng};
use crate::core::solver::Solver;
use crate::core::state::{BasicState, CountsMirror, State};
use crate::core::termination::{TerminationCriterion, TerminationReason};
use core::ops::{Index, IndexMut};

/// Perturbation strategy for [`BasinHopping`]: given the current iterate,
/// propose a displaced point to start the next local minimization from.
///
/// The step taker is the single most performance-critical, problem-
/// dependent tunable in basin-hopping (steps too small return to the same
/// basin; too large degrade into a random walk), so it is the primary
/// customization point. The shipped default is [`RandomDisplacement`]; bring
/// your own by implementing this trait and passing it to
/// [`BasinHopping::with_step_taker`].
///
/// # Adaptive scaling
///
/// [`adjust_scale`](Self::adjust_scale) is the hook the outer loop calls
/// every `interval` hops to push the acceptance rate toward its target. It
/// defaults to a no-op so structured or discrete moves that have no scalar
/// magnitude opt out cleanly (they simply ignore adaptive control).
pub trait StepTaker<V, F = f64> {
    /// Propose a perturbed point starting from `x`, consuming randomness
    /// from `rng`. Returns a fresh value; `x` is not mutated.
    fn take_step<R: Rng + ?Sized>(&mut self, x: &V, rng: &mut R) -> V;

    /// Scale the step magnitude by `factor` (`> 1` grows the step, `< 1`
    /// shrinks it). Called by [`BasinHopping`] when adaptive step control
    /// is enabled. Default: no-op.
    fn adjust_scale(&mut self, _factor: F) {}
}

/// Accept/reject rule for [`BasinHopping`]: decide whether to move the
/// current iterate to a freshly minimized candidate.
///
/// The shipped default is [`Metropolis`]; supply a custom rule via
/// [`BasinHopping::with_acceptance_test`] (e.g. a greedy "accept only if
/// strictly lower" test, or a bounds-aware rejection).
pub trait AcceptanceTest<F = f64> {
    /// Return `true` to accept the candidate cost `f_new` over the current
    /// `f_old`. `rng` supplies any randomness the rule needs.
    fn accept<R: Rng + ?Sized>(&self, f_new: F, f_old: F, rng: &mut R) -> bool;
}

/// Uniform per-coordinate displacement: `xᵢ ← xᵢ + U(−stepsize, stepsize)`,
/// the default [`StepTaker`].
///
/// This is the perturbation of Wales & Doye (1997) ("all coordinates were
/// displaced by a random number in the range [−1,1] times the step size")
/// and of SciPy's `RandomDisplacement`. [`adjust_scale`](StepTaker::adjust_scale)
/// multiplies the step size, so adaptive control widens or narrows the cube.
#[derive(Clone, Debug)]
pub struct RandomDisplacement<F = f64> {
    stepsize: F,
}

impl<F: Scalar> RandomDisplacement<F> {
    /// Build a displacement step taker with the given half-width.
    ///
    /// # Panics
    ///
    /// Panics if `stepsize <= 0`.
    pub fn new(stepsize: F) -> Self {
        assert!(
            stepsize > F::zero(),
            "RandomDisplacement requires stepsize > 0, got {stepsize:?}"
        );
        Self { stepsize }
    }

    /// Current step half-width.
    pub fn stepsize(&self) -> F {
        self.stepsize
    }
}

impl<V, F> StepTaker<V, F> for RandomDisplacement<F>
where
    F: Scalar,
    V: VectorLen
        + Clone
        + Index<usize, Output = F>
        + IndexMut<usize, Output = F>
        + SampleUniformBox,
{
    fn take_step<R: Rng + ?Sized>(&mut self, x: &V, rng: &mut R) -> V {
        // Build the cube [x − s, x + s] and reuse the tested uniform-box
        // sampler: U(xᵢ − s, xᵢ + s) ≡ xᵢ + U(−s, s).
        let mut lower = x.clone();
        let mut upper = x.clone();
        for i in 0..x.vec_len() {
            lower[i] = x[i] - self.stepsize;
            upper[i] = x[i] + self.stepsize;
        }
        V::sample_uniform_box(&lower, &upper, rng)
    }

    fn adjust_scale(&mut self, factor: F) {
        self.stepsize = self.stepsize * factor;
    }
}

/// Metropolis acceptance on minimized costs at temperature `T`: always
/// accept a downhill move, else accept with probability
/// `exp(−(f_new − f_old) / T)`. The default [`AcceptanceTest`].
///
/// Folding Boltzmann's constant into `T` matches Wales & Doye's `kT`
/// convention and SciPy's `Metropolis` (`w = exp(min(0, −(f_new − f_old)·β))`
/// with `β = 1/T`).
#[derive(Clone, Debug)]
pub struct Metropolis<F = f64> {
    beta: F,
}

impl<F: Scalar> Metropolis<F> {
    /// Build a Metropolis test at temperature `temperature` (`= T`).
    ///
    /// # Panics
    ///
    /// Panics if `temperature <= 0`.
    pub fn new(temperature: F) -> Self {
        assert!(
            temperature > F::zero(),
            "Metropolis requires temperature > 0, got {temperature:?}"
        );
        Self {
            beta: F::one() / temperature,
        }
    }
}

impl<F: Scalar> AcceptanceTest<F> for Metropolis<F> {
    fn accept<R: Rng + ?Sized>(&self, f_new: F, f_old: F, rng: &mut R) -> bool {
        if f_new <= f_old {
            return true;
        }
        // f_new > f_old, so the exponent is strictly negative and w ∈ (0, 1).
        let w = (-(f_new - f_old) * self.beta).exp();
        // One f64 draw in [0, 1) keeps reproducibility well-defined across
        // F = f32/f64 (matches the per-component draw convention in math/sample.rs).
        let r = F::from_f64(rng.random::<f64>()).unwrap();
        w >= r
    }
}

/// SciPy's success guard (`res_new.success or not res_old.success`): a candidate
/// from a failed local solve is admissible only when the incumbent also came
/// from a failed solve. ANDed with the Metropolis decision in `next_iter`.
fn accept_guard(new_success: bool, incumbent_success: bool) -> bool {
    new_success || !incumbent_success
}

/// Basin-hopping (Wales & Doye 1997): a global optimizer that explores the
/// transformed surface `Ẽ(x) = min{f(x)}`—the cost of the local minimum
/// reached from `x`, which preserves the global minimum and the relative
/// ordering of all local minima—with a Metropolis Monte-Carlo walk.
///
/// One outer iteration ([`next_iter`](Solver::next_iter)) is one *hop*:
///
/// 1. **Perturb** the current iterate with the [`StepTaker`]
///    (default [`RandomDisplacement`]: `xᵢ += U(−stepsize, stepsize)`).
/// 2. **Locally minimize** from the perturbed point by driving the inner
///    solver `I` to convergence (realizing the `Ẽ` transform).
/// 3. **Accept or reject** the new local minimum with the [`AcceptanceTest`]
///    (default [`Metropolis`] at temperature `T`). On acceptance the current
///    iterate moves to the new minimum; otherwise it stays.
/// 4. **Adapt** the step size every `interval` hops to push the empirical
///    acceptance rate toward `target_accept_rate` (Wales & Doye tuned the
///    step to a 0.5 acceptance ratio).
///
/// [`init`](Solver::init) performs one initial local minimization of the
/// starting point, so the first hop perturbs an already-relaxed iterate.
/// The global best is tracked by the framework ([`State::update_best`]); the
/// solver only maintains the Metropolis *current* iterate.
///
/// # Termination
///
/// Iteration count is framework-level (tenet 3): cap the number of hops with
/// [`Executor::max_iter`](crate::core::executor::Executor::max_iter) (SciPy's
/// `niter`) and stop on stalls with
/// [`NoImprovement`](crate::core::termination::NoImprovement) (SciPy's
/// `niter_success`) or [`TargetCost`](crate::core::termination::TargetCost).
/// The inner local solver gets its own budget via
/// [`with_inner_max_iter`](Self::with_inner_max_iter) and
/// [`inner_terminate_on`](Self::inner_terminate_on)—as Wales & Doye note,
/// the inner tolerance "need not be very tight," so a loose
/// [`SimplexTolerance`](crate::core::termination::SimplexTolerance) (for a
/// Nelder-Mead inner) keeps each hop cheap.
///
/// # Defaults
///
/// Matching SciPy's `basinhopping`: `T = 1.0`, `stepsize = 0.5`,
/// `interval = 50`, `target_accept_rate = 0.5`, `stepwise_factor = 0.9`,
/// adaptive step control on.
///
/// # Inner solver
///
/// Generic over any `I: WarmStart<V>`; the associated `I::State` determines
/// the inner state shape, so `BasinHopping::new(NelderMead::adaptive(), seed)`
/// needs no turbofish for the state type. Shipped `WarmStart` impls cover
/// [`NelderMead`](crate::solver::NelderMead),
/// [`LevenbergMarquardt`](crate::solver::LevenbergMarquardt), and
/// [`Lbfgsb`](crate::Lbfgsb); a gradient-based local solver (BFGS, L-BFGS)
/// generally finds the basin minimum in fewer hops than a derivative-free one.
///
/// # Eval aggregation
///
/// Same-problem composition: the inner shares the outer's [`Problem`]
/// wrapper, so every inner cost evaluation bumps the same
/// [`EvalCounts`](crate::core::problem::EvalCounts) as the outer's, and
/// [`BasicState`]'s [`CountsMirror`] folds the per-hop delta into
/// `cost_evals`. No manual roll-up; `MaxCostEvals` budgets and
/// `result.cost_evals()` count the local minimizations honestly. See
/// CONTRIBUTING.md "Solver composition" rule 1.
///
/// # Backends
///
/// `BasinHopping` itself is backend-generic; the supported param types are
/// the intersection of what the chosen [`StepTaker`] and inner solver
/// support. The default [`RandomDisplacement`] works on every backend that
/// implements [`SampleUniformBox`] (`Vec<f64>`, nalgebra, ndarray, faer), so
/// effective coverage is set by the inner solver. **wasm:** clean—the seeded
/// [`ChaCha8Rng`] needs no entropy source and there is no time dependency—
/// provided the inner solver is itself wasm-clean.
///
/// # References
///
/// - D. J. Wales and J. P. K. Doye, "Global Optimization by Basin-Hopping and
///   the Lowest Energy Structures of Lennard-Jones Clusters Containing up to
///   110 Atoms," *J. Phys. Chem. A* **1997**, 101, 5111–5116.
///   DOI: 10.1021/jp970984n.
/// - D. J. Wales and H. A. Scheraga, "Global Optimization of Clusters,
///   Crystals, and Biomolecules," *Science* **1999**, 285, 1368–1372.
///   DOI: 10.1126/science.285.5432.1368.
/// - Defaults and the adaptive-step/Metropolis formulation follow
///   `scipy.optimize.basinhopping` (BSD-3-Clause; behavior reproduced from
///   its documented algorithm, not ported).
///
/// # Examples
///
/// ```
/// use basin::{BasicState, BasinHopping, Executor, MaxIter, NelderMead, SimplexTolerance};
/// use basin::problems::Ackley;
///
/// let solver = BasinHopping::new(NelderMead::adaptive(), 42)
///     .with_stepsize(1.0)
///     .inner_terminate_on(SimplexTolerance::new(1e-8, 1e-8));
///
/// let result = Executor::new(Ackley::<Vec<f64>>::new(), solver, BasicState::new(vec![2.0, 2.0]))
///     .max_iter(200)
///     .run()
///     .unwrap();
///
/// // `best_cost()` is the global best the walk ever saw; the *current*
/// // iterate may be a transiently accepted uphill hop.
/// assert!(result.best_cost() < 1e-6, "Ackley best cost {}", result.best_cost());
/// ```
pub struct BasinHopping<
    I,
    V,
    F = f64,
    S = RandomDisplacement<F>,
    A = Metropolis<F>,
> where
    F: Scalar,
    I: WarmStart<V>,
{
    inner: InnerExecutor<<I as InitialState<V>>::State, I>,
    step: S,
    accept: A,
    rng: ChaCha8Rng,
    adaptive: bool,
    interval: u64,
    target_accept_rate: F,
    stepwise_factor: F,
    // Cumulative hop and acceptance counts (never reset). The adaptive
    // control uses the lifetime rate `naccept/nstep`, matching SciPy's
    // AdaptiveStepsize.
    nstep: u64,
    naccept: u64,
    // Whether the current incumbent was reached by a successful inner solve.
    // Gates SciPy's `res_new.success or not res_old.success` acceptance guard:
    // a candidate from a failed local solve is accepted only when the incumbent
    // also came from a failed solve. Set by `init`, updated on every accept.
    incumbent_success: bool,
}

impl<I, V, F> BasinHopping<I, V, F, RandomDisplacement<F>, Metropolis<F>>
where
    F: Scalar,
    I: WarmStart<V>,
    <I as InitialState<V>>::State: State + CountsMirror,
{
    /// Wrap `inner` as the local minimizer with the canonical defaults
    /// (`T = 1.0`, `stepsize = 0.5`, `interval = 50`,
    /// `target_accept_rate = 0.5`, `stepwise_factor = 0.9`, adaptive on),
    /// seeding the RNG from `seed`.
    ///
    /// The structural argument (`inner`) comes first and the RNG `seed`
    /// last, matching the rest of the stochastic family
    /// ([`RandomSearch::new`](crate::solver::RandomSearch::new),
    /// [`CmaInject::new`](crate::solver::CmaInject)).
    pub fn new(inner: I, seed: u64) -> Self {
        let half = F::from_f64(0.5).unwrap();
        Self {
            inner: InnerExecutor::new(inner),
            step: RandomDisplacement::new(half),
            accept: Metropolis::new(F::one()),
            rng: ChaCha8Rng::seed_from_u64(seed),
            adaptive: true,
            interval: 50,
            target_accept_rate: half,
            stepwise_factor: F::from_f64(0.9).unwrap(),
            nstep: 0,
            naccept: 0,
            incumbent_success: true,
        }
    }

    /// Set the default-displacement step half-width (SciPy `stepsize`,
    /// default `0.5`). Match it to the typical separation between local
    /// minima of your problem.
    ///
    /// # Panics
    ///
    /// Panics if `stepsize <= 0`.
    pub fn with_stepsize(mut self, stepsize: F) -> Self {
        self.step = RandomDisplacement::new(stepsize);
        self
    }

    /// Set the Metropolis temperature `T` (SciPy `T`, default `1.0`). Larger
    /// `T` accepts uphill hops more readily.
    ///
    /// # Panics
    ///
    /// Panics if `temperature <= 0`.
    pub fn with_temperature(mut self, temperature: F) -> Self {
        self.accept = Metropolis::new(temperature);
        self
    }
}

impl<I, V, F, S, A> BasinHopping<I, V, F, S, A>
where
    F: Scalar,
    I: WarmStart<V>,
    <I as InitialState<V>>::State: State + CountsMirror,
{
    /// Replace the step taker (perturbation strategy). Changes the `S` type
    /// parameter; configure your custom step taker before passing it.
    pub fn with_step_taker<S2>(self, step: S2) -> BasinHopping<I, V, F, S2, A> {
        BasinHopping {
            inner: self.inner,
            step,
            accept: self.accept,
            rng: self.rng,
            adaptive: self.adaptive,
            interval: self.interval,
            target_accept_rate: self.target_accept_rate,
            stepwise_factor: self.stepwise_factor,
            nstep: self.nstep,
            naccept: self.naccept,
            incumbent_success: self.incumbent_success,
        }
    }

    /// Replace the acceptance test. Changes the `A` type parameter.
    pub fn with_acceptance_test<A2>(
        self,
        accept: A2,
    ) -> BasinHopping<I, V, F, S, A2> {
        BasinHopping {
            inner: self.inner,
            step: self.step,
            accept,
            rng: self.rng,
            adaptive: self.adaptive,
            interval: self.interval,
            target_accept_rate: self.target_accept_rate,
            stepwise_factor: self.stepwise_factor,
            nstep: self.nstep,
            naccept: self.naccept,
            incumbent_success: self.incumbent_success,
        }
    }

    /// Enable or disable adaptive step-size control (default enabled). When
    /// disabled, the step taker's magnitude is held fixed.
    pub fn with_adaptive(mut self, adaptive: bool) -> Self {
        self.adaptive = adaptive;
        self
    }

    /// Number of hops between adaptive step-size adjustments (SciPy
    /// `interval`, default `50`).
    ///
    /// # Panics
    ///
    /// Panics if `interval == 0`.
    pub fn with_adaptive_interval(mut self, interval: u64) -> Self {
        assert!(interval >= 1, "BasinHopping requires interval >= 1");
        self.interval = interval;
        self
    }

    /// Target acceptance rate the adaptive control aims for (SciPy
    /// `target_accept_rate`, default `0.5`).
    ///
    /// # Panics
    ///
    /// Panics unless `0 < rate < 1`.
    pub fn with_target_accept_rate(mut self, rate: F) -> Self {
        assert!(
            rate > F::zero() && rate < F::one(),
            "BasinHopping requires 0 < target_accept_rate < 1, got {rate:?}"
        );
        self.target_accept_rate = rate;
        self
    }

    /// Multiplicative factor applied to the step size at each adjustment
    /// (SciPy `stepwise_factor`, default `0.9`). Must lie in `(0, 1)`.
    ///
    /// # Panics
    ///
    /// Panics unless `0 < factor < 1`.
    pub fn with_stepwise_factor(mut self, factor: F) -> Self {
        assert!(
            factor > F::zero() && factor < F::one(),
            "BasinHopping requires 0 < stepwise_factor < 1, got {factor:?}"
        );
        self.stepwise_factor = factor;
        self
    }

    /// Inner local-solver iteration budget per hop (default
    /// [`InnerExecutor`]'s `1000`). Pair with
    /// [`inner_terminate_on`](Self::inner_terminate_on) so the inner stops on
    /// convergence rather than always exhausting this cap.
    ///
    /// Note that an inner solve that stops on this cap terminates with
    /// [`MaxIter`](crate::core::termination::MaxIter), which counts as a
    /// *successful* solve for the acceptance guard (unlike SciPy, whose
    /// `minimize` reports `success = False` on max-iter). Only a soft
    /// [`SolverFailed`](crate::core::termination::TerminationReason::SolverFailed)
    /// marks a candidate unsuccessful.
    pub fn with_inner_max_iter(mut self, n: u64) -> Self {
        self.inner = self.inner.max_iter(n);
        self
    }

    /// Register a termination criterion on the inner local solver. Reused
    /// across every hop and reset at the start of each inner run, so
    /// stateful criteria are safe (CONTRIBUTING.md "Solver composition"
    /// rule 2).
    pub fn inner_terminate_on<C>(mut self, criterion: C) -> Self
    where
        C: TerminationCriterion<<I as InitialState<V>>::State> + 'static,
    {
        self.inner = self.inner.terminate_on(criterion);
        self
    }
}

impl<P, I, V, F, S, A> Solver<P, BasicState<V, F>>
    for BasinHopping<I, V, F, S, A>
where
    F: Scalar,
    P: CostFunction<Param = V, Output = F>,
    I: WarmStart<V>
        + Solver<P, <I as InitialState<V>>::State, Error = P::Error>,
    <I as InitialState<V>>::State: State<Param = V, Float = F> + CountsMirror,
    V: Clone,
    S: StepTaker<V, F>,
    A: AcceptanceTest<F>,
{
    type Error = P::Error;

    fn init(
        &mut self,
        problem: &mut Problem<P>,
        mut state: BasicState<V, F>,
    ) -> Result<BasicState<V, F>, Self::Error> {
        // Relax the starting point once so the Metropolis walk begins from a
        // local minimum (the `Ẽ` value of x0). Same-problem composition, so
        // the inner's evals flow into the outer wrapper transparently.
        let seeded = self.inner.solver().seed(&state.param);
        let result = self.inner.run(problem, seeded)?;
        state.param = result.state.param().clone();
        state.cost = Some(result.state.cost());
        // Record whether the initial relaxation succeeded so the first hop's
        // acceptance guard has a faithful incumbent-success flag (a hard `Err`
        // still bubbles via `?`; only soft `SolverFailed` reasons fold in here).
        self.incumbent_success = !result.reason.is_failure();
        Ok(state)
    }

    fn next_iter(
        &mut self,
        problem: &mut Problem<P>,
        mut state: BasicState<V, F>,
    ) -> Result<(BasicState<V, F>, Option<TerminationReason>), Self::Error>
    {
        let f_old = state.cost();

        // 1. Perturb the current iterate.
        let x_trial = self.step.take_step(&state.param, &mut self.rng);

        // 2. Local minimization from the perturbed point (the `Ẽ` transform).
        // A hard `Err` bubbles via `?`; a soft `SolverFailed` reason does not
        // end the walk—it marks the candidate unsuccessful for the guard below,
        // matching SciPy (which keeps walking past a failed local solve). Clean
        // stops (converged, max-iter, tolerance) leave the candidate successful.
        let seeded = self.inner.solver().seed(&x_trial);
        let result = self.inner.run(problem, seeded)?;
        let new_success = !result.reason.is_failure();

        let f_new = result.state.cost();
        let x_new = result.state.param().clone();

        // 3. Metropolis accept/reject, then SciPy's success guard
        // (`res_new.success or not res_old.success`): a failed candidate is
        // accepted only when the incumbent also came from a failed solve. The
        // Metropolis draw is evaluated first so the RNG draw (when it happens)
        // precedes the guard, mirroring SciPy's `and` short-circuit order.
        self.nstep += 1;
        let accepted = self.accept.accept(f_new, f_old, &mut self.rng)
            && accept_guard(new_success, self.incumbent_success);
        if accepted {
            state.param = x_new;
            state.cost = Some(f_new);
            self.incumbent_success = new_success;
            self.naccept += 1;
        }

        // 4. Adaptive step control. Every `interval` hops, nudge the step
        // size toward the target acceptance rate. SciPy's AdaptiveStepsize
        // uses the *cumulative* (lifetime) acceptance rate — `naccept`/`nstep`
        // are never reset — and fires when `nstep % interval == 0`.
        if self.adaptive && self.nstep % self.interval == 0 {
            let rate = F::from_u64(self.naccept).unwrap()
                / F::from_u64(self.nstep).unwrap();
            let factor = if rate > self.target_accept_rate {
                // Accepting too many → enlarge the step (factor < 1, so 1/factor > 1).
                F::one() / self.stepwise_factor
            } else {
                // Accepting too few → shrink the step.
                self.stepwise_factor
            };
            self.step.adjust_scale(factor);
        }

        Ok((state, None))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::rng::ChaCha8Rng;

    #[test]
    fn random_displacement_stays_within_cube_and_advances_rng() {
        let mut step = RandomDisplacement::new(0.5_f64);
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        let x = vec![1.0, -2.0, 3.0];
        let y = step.take_step(&x, &mut rng);
        assert_eq!(y.len(), x.len());
        for (yi, xi) in y.iter().zip(&x) {
            assert!(
                (yi - xi).abs() <= 0.5 + 1e-12,
                "component {yi} outside [{}, {}]",
                xi - 0.5,
                xi + 0.5
            );
        }
    }

    #[test]
    fn random_displacement_adjust_scale_grows_and_shrinks() {
        let mut step = RandomDisplacement::new(1.0_f64);
        // `adjust_scale` is a `StepTaker<V>` method whose signature doesn't
        // mention `V`, so a standalone call needs `V` pinned; inside the
        // solver the `S: StepTaker<V, F>` bound fixes it.
        StepTaker::<Vec<f64>>::adjust_scale(&mut step, 2.0);
        assert_eq!(step.stepsize(), 2.0);
        StepTaker::<Vec<f64>>::adjust_scale(&mut step, 0.25);
        assert_eq!(step.stepsize(), 0.5);
    }

    #[test]
    fn accept_guard_matches_scipy_success_rule() {
        // SciPy: `res_new.success or not res_old.success`. The guard is ANDed
        // with the Metropolis decision, so it only ever vetoes a candidate the
        // Metropolis test already accepted.
        // New solve succeeded → always admissible.
        assert!(accept_guard(true, true));
        assert!(accept_guard(true, false));
        // New solve failed → admissible only if the incumbent also failed.
        assert!(!accept_guard(false, true));
        assert!(accept_guard(false, false));
    }

    #[test]
    fn metropolis_always_accepts_downhill() {
        let test = Metropolis::new(1.0_f64);
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        // Strictly lower and equal costs accept unconditionally.
        assert!(test.accept(1.0, 2.0, &mut rng));
        assert!(test.accept(2.0, 2.0, &mut rng));
    }

    #[test]
    fn metropolis_uphill_acceptance_rate_matches_boltzmann() {
        // For ΔE = 1, T = 1, the acceptance probability is exp(-1) ≈ 0.3679.
        let test = Metropolis::new(1.0_f64);
        let mut rng = ChaCha8Rng::seed_from_u64(12345);
        let trials = 20_000;
        let accepts = (0..trials)
            .filter(|_| test.accept(2.0, 1.0, &mut rng))
            .count();
        let rate = accepts as f64 / trials as f64;
        let expected = (-1.0_f64).exp();
        assert!(
            (rate - expected).abs() < 0.02,
            "empirical uphill acceptance {rate} vs Boltzmann {expected}"
        );
    }

    #[test]
    fn lower_temperature_accepts_fewer_uphill_moves() {
        let mut rng = ChaCha8Rng::seed_from_u64(99);
        let hot = Metropolis::new(5.0_f64);
        let cold = Metropolis::new(0.1_f64);
        let trials = 10_000;
        let hot_accepts = (0..trials)
            .filter(|_| hot.accept(2.0, 1.0, &mut rng))
            .count();
        let cold_accepts = (0..trials)
            .filter(|_| cold.accept(2.0, 1.0, &mut rng))
            .count();
        assert!(
            hot_accepts > cold_accepts,
            "hot {hot_accepts} should accept more uphill than cold {cold_accepts}"
        );
    }
}
