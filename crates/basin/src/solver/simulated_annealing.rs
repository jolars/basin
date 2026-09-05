//! Generic simulated annealing with classical Metropolis acceptance.

use crate::core::inner::InitialState;
use crate::core::math::Scalar;
use crate::core::problem::{CostFunction, Problem};
use crate::core::rng::{ChaCha8Rng, Rng, RngExt, SeedableRng};
use crate::core::solver::Solver;
use crate::core::state::{SimulatedAnnealingState, State};
use crate::core::termination::TerminationReason;

/// Generate one candidate from the current simulated-annealing state.
///
/// The temperature is provided because useful proposal scales often cool with
/// the acceptance rule. The neighbor owns any deterministic proposal history,
/// while `rng` owns all randomness. Closures with the same signature implement
/// this trait automatically with [`Infallible`](std::convert::Infallible) as
/// their error type.
pub trait Neighbor<P, F = f64, R = ChaCha8Rng> {
    /// Error returned when a proposal cannot be generated.
    ///
    /// When used with [`SimulatedAnnealing`], this must be the same type as
    /// [`CostFunction::Error`], allowing either operation to abort the run
    /// without erasing the application error.
    type Error;

    /// Propose a candidate without mutating `current`.
    fn propose(
        &mut self,
        current: &P,
        temperature: F,
        rng: &mut R,
    ) -> Result<P, Self::Error>;
}

impl<P, F, R, N> Neighbor<P, F, R> for N
where
    N: FnMut(&P, F, &mut R) -> P,
{
    type Error = std::convert::Infallible;

    fn propose(
        &mut self,
        current: &P,
        temperature: F,
        rng: &mut R,
    ) -> Result<P, Self::Error> {
        Ok(self(current, temperature, rng))
    }
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Copy, Debug)]
enum Cooling<F> {
    Geometric { alpha: F },
    Reciprocal,
    Logarithmic,
}

/// A temperature schedule indexed by completed proposals since the most
/// recent reannealing event.
///
/// Every schedule uses the supplied initial temperature at proposal zero.
/// `steps_per_temperature` defaults to one; raising it creates temperature
/// levels with several Metropolis proposals per level, matching the original
/// simulated-annealing formulation more closely than cooling after every
/// proposal.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Copy, Debug)]
pub struct TemperatureSchedule<F = f64> {
    cooling: Cooling<F>,
    steps_per_temperature: u64,
}

impl<F: Scalar> TemperatureSchedule<F> {
    /// Geometric cooling, `T_j = T_0 α^j`.
    ///
    /// # Panics
    ///
    /// Panics unless `alpha` is finite and strictly between zero and one.
    pub fn geometric(alpha: F) -> Self {
        assert!(
            alpha.is_finite() && alpha > F::zero() && alpha < F::one(),
            "geometric cooling requires finite 0 < alpha < 1, got {alpha:?}"
        );
        Self {
            cooling: Cooling::Geometric { alpha },
            steps_per_temperature: 1,
        }
    }

    /// Reciprocal cooling, `T_j = T_0 / (j + 1)`.
    pub fn reciprocal() -> Self {
        Self {
            cooling: Cooling::Reciprocal,
            steps_per_temperature: 1,
        }
    }

    /// Normalized logarithmic cooling, `T_j = T_0 ln(2) / ln(j + 2)`.
    ///
    /// The normalization keeps proposal zero at exactly `T_0`, avoiding the
    /// initial temperature increase produced by an unshifted `T_0 / ln(i)`
    /// indexing convention.
    pub fn logarithmic() -> Self {
        Self {
            cooling: Cooling::Logarithmic,
            steps_per_temperature: 1,
        }
    }

    /// Hold each temperature for `steps` proposals.
    ///
    /// # Panics
    ///
    /// Panics if `steps == 0`.
    pub fn with_steps_per_temperature(mut self, steps: u64) -> Self {
        assert!(
            steps > 0,
            "temperature schedule requires steps_per_temperature > 0"
        );
        self.steps_per_temperature = steps;
        self
    }

    /// Number of proposals made at each temperature level.
    pub fn steps_per_temperature(&self) -> u64 {
        self.steps_per_temperature
    }

    /// Evaluate this schedule at a proposal age.
    ///
    /// Values that underflow are clamped to the scalar's smallest positive
    /// normal value, so the Metropolis rule never divides by zero.
    pub fn temperature(&self, initial_temperature: F, proposal_age: u64) -> F {
        assert!(
            initial_temperature.is_finite() && initial_temperature > F::zero(),
            "temperature schedule requires a finite initial temperature > 0"
        );
        let level = proposal_age / self.steps_per_temperature;
        let level_f = F::from_u64(level).unwrap_or_else(F::infinity);
        let temperature = match self.cooling {
            Cooling::Geometric { alpha } => {
                initial_temperature * alpha.powf(level_f)
            }
            Cooling::Reciprocal => initial_temperature / (level_f + F::one()),
            Cooling::Logarithmic => {
                let two = F::one() + F::one();
                initial_temperature * two.ln() / (level_f + two).ln()
            }
        };
        temperature.max(F::min_positive_value())
    }
}

/// Configuration for restarting a simulated-annealing temperature schedule at
/// `T_0`.
///
/// This generic restart is intentionally narrower than Ingber-style adaptive
/// simulated annealing: coordinate-sensitivity rescaling is not meaningful for
/// an arbitrary parameter type. A restart affects only the cooling age and all
/// reannealing progress; it does not reset the incumbent, best point,
/// acceptance history, or global counters.
///
/// Each constructor enables one trigger. Use
/// [`SimulatedAnnealing::with_reannealing_fixed`],
/// [`SimulatedAnnealing::with_reannealing_accepted`], and
/// [`SimulatedAnnealing::with_reannealing_best`] to compose triggers directly
/// on a solver.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Copy, Debug)]
pub struct Reannealing {
    fixed_interval: Option<u64>,
    accepted_stall: Option<u64>,
    best_stall: Option<u64>,
}

impl Reannealing {
    fn validate_threshold(threshold: u64) {
        assert!(threshold > 0, "reannealing threshold must be > 0");
    }

    fn none() -> Self {
        Self {
            fixed_interval: None,
            accepted_stall: None,
            best_stall: None,
        }
    }

    fn with_fixed_interval(mut self, interval: u64) -> Self {
        Self::validate_threshold(interval);
        self.fixed_interval = Some(interval);
        self
    }

    fn with_accepted_stall(mut self, iterations: u64) -> Self {
        Self::validate_threshold(iterations);
        self.accepted_stall = Some(iterations);
        self
    }

    fn with_best_stall(mut self, iterations: u64) -> Self {
        Self::validate_threshold(iterations);
        self.best_stall = Some(iterations);
        self
    }

    /// Restart after every `interval` completed proposals.
    ///
    /// # Panics
    ///
    /// Panics if `interval` is zero.
    pub fn fixed_interval(interval: u64) -> Self {
        Self::none().with_fixed_interval(interval)
    }

    /// Restart after `rejections` consecutive rejected proposals.
    ///
    /// # Panics
    ///
    /// Panics if `rejections` is zero.
    pub fn after_rejections(rejections: u64) -> Self {
        Self::none().with_accepted_stall(rejections)
    }

    /// Restart after `iterations` proposals without a new global best.
    ///
    /// # Panics
    ///
    /// Panics if `iterations` is zero.
    pub fn after_no_best(iterations: u64) -> Self {
        Self::none().with_best_stall(iterations)
    }

    fn update_progress(
        self,
        progress: &mut ReannealingProgress,
        accepted: bool,
        new_best: bool,
    ) {
        progress.fixed_interval = progress.fixed_interval.saturating_add(1);
        progress.accepted_stall = if accepted {
            0
        } else {
            progress.accepted_stall.saturating_add(1)
        };
        progress.best_stall = if new_best {
            0
        } else {
            progress.best_stall.saturating_add(1)
        };
    }

    fn should_restart(self, progress: ReannealingProgress) -> bool {
        self.fixed_interval
            .is_some_and(|threshold| progress.fixed_interval >= threshold)
            || self
                .accepted_stall
                .is_some_and(|threshold| progress.accepted_stall >= threshold)
            || self
                .best_stall
                .is_some_and(|threshold| progress.best_stall >= threshold)
    }
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct ReannealingProgress {
    fixed_interval: u64,
    accepted_stall: u64,
    best_stall: u64,
}

/// Classical single-proposal simulated annealing.
///
/// At temperature `T`, the solver proposes `y` with [`Neighbor`] and accepts
/// every `f(y) <= f(x)`. A strictly uphill proposal is accepted with
/// probability `exp(-(f(y) - f(x)) / T)`. This is the Metropolis rule of
/// Kirkpatrick, Gelatt, and Vecchi—not Argmin's logistic acceptance variant.
/// One successful [`Solver::next_iter`] is one proposal and one cost
/// evaluation.
///
/// The caller must choose a temperature schedule. There is no universal
/// default: the logarithmic convergence result requires finite-state
/// irreducibility assumptions and a problem-dependent coefficient, while the
/// original practical schedule holds each temperature for many proposals.
/// Fixed-interval, accepted-stall, and best-stall reannealing triggers may be
/// enabled together. A restart occurs when any enabled trigger reaches its
/// threshold and resets the progress of every trigger.
///
/// [`SimulatedAnnealingState`] owns the evolving neighbor and RNG. This makes
/// stateful proposals reproducible and permits exact serialized continuation
/// through [`Executor::resume`](crate::Executor::resume). A restored run must
/// use the same deterministic problem, scalar type, code, and resume-safe
/// termination criteria.
///
/// # Non-finite costs
///
/// Proposed `NaN` and `+∞` costs are rejected. A finite proposal replaces a
/// `+∞` incumbent, and `-∞` is accepted and stops on the following framework
/// check with [`TerminationReason::SolverConverged`]. A `NaN` incumbent stops
/// with [`TerminationReason::SolverFailed`]. Equal finite costs are accepted.
///
/// # Proposal contract
///
/// Classical Metropolis acceptance assumes a symmetric—or otherwise
/// reversible with equal forward and reverse mass—proposal kernel. This API
/// does not accept a Hastings ratio. Users with asymmetric proposals must
/// account for that ratio in a different algorithm.
///
/// # Backends
///
/// `SimulatedAnnealing` imposes no vector operations. It supports arbitrary
/// cloneable parameter types, including discrete structures, `Vec<f32/f64>`,
/// nalgebra vectors, ndarray arrays, and faer columns. The seeded `ChaCha8Rng`
/// default is wasm-safe. Each iteration has one dependent proposal, so the
/// solver does not use the `parallel` feature.
///
/// # References
///
/// - N. Metropolis, A. W. Rosenbluth, M. N. Rosenbluth, A. H. Teller, and
///   E. Teller, “Equation of State Calculations by Fast Computing Machines,”
///   *Journal of Chemical Physics* **21** (1953), 1087–1092.
///   DOI: 10.1063/1.1699114.
/// - S. Kirkpatrick, C. D. Gelatt, Jr., and M. P. Vecchi, “Optimization by
///   Simulated Annealing,” *Science* **220** (1983), 671–680.
///   DOI: 10.1126/science.220.4598.671.
/// - B. Hajek, “Cooling Schedules for Optimal Annealing,” *Mathematics of
///   Operations Research* **13** (1988), 311–329.
///   DOI: 10.1287/moor.13.2.311.
///
/// # Example
///
/// ```
/// use basin::core::rng::{ChaCha8Rng, RngExt};
/// use basin::{CostFunction, Executor, SimulatedAnnealing, TemperatureSchedule};
/// use std::convert::Infallible;
///
/// struct Sphere;
/// impl CostFunction for Sphere {
///     type Param = Vec<f64>;
///     type Output = f64;
///     type Error = Infallible;
///     fn cost(&self, x: &Vec<f64>) -> Result<f64, Infallible> {
///         Ok(x.iter().map(|v| v * v).sum())
///     }
/// }
///
/// let neighbor = |x: &Vec<f64>, temperature: f64, rng: &mut ChaCha8Rng| {
///     x.iter()
///         .map(|v| v + temperature.sqrt() * (2.0 * rng.random::<f64>() - 1.0))
///         .collect()
/// };
/// let solver = SimulatedAnnealing::new(
///     neighbor,
///     2.0,
///     TemperatureSchedule::geometric(0.995).with_steps_per_temperature(8),
///     42,
/// );
/// let result = Executor::from_start(Sphere, solver, vec![3.0, -4.0])
///     .max_iter(10_000)
///     .run()
///     .unwrap();
/// assert!(result.best_cost() < 1e-2);
/// ```
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug)]
pub struct SimulatedAnnealing<N, F = f64, R = ChaCha8Rng> {
    neighbor: N,
    initial_temperature: F,
    schedule: TemperatureSchedule<F>,
    reannealing: Option<Reannealing>,
    rng: R,
}

fn uphill_acceptance_probability<F: Scalar>(delta: F, temperature: F) -> F {
    (-delta / temperature).exp()
}

impl<N, F> SimulatedAnnealing<N, F, ChaCha8Rng>
where
    F: Scalar,
{
    /// Build a solver with Basin's seeded, wasm-safe ChaCha8 generator.
    ///
    /// # Panics
    ///
    /// Panics unless `initial_temperature` is finite and positive.
    pub fn new(
        neighbor: N,
        initial_temperature: F,
        schedule: TemperatureSchedule<F>,
        seed: u64,
    ) -> Self {
        Self::new_with_rng(
            neighbor,
            initial_temperature,
            schedule,
            ChaCha8Rng::seed_from_u64(seed),
        )
    }
}

impl<N, F, R> SimulatedAnnealing<N, F, R>
where
    F: Scalar,
{
    /// Build a solver with a caller-supplied random-number generator.
    ///
    /// # Panics
    ///
    /// Panics unless `initial_temperature` is finite and positive.
    pub fn new_with_rng(
        neighbor: N,
        initial_temperature: F,
        schedule: TemperatureSchedule<F>,
        rng: R,
    ) -> Self {
        assert!(
            initial_temperature.is_finite() && initial_temperature > F::zero(),
            "SimulatedAnnealing requires a finite initial temperature > 0"
        );
        Self {
            neighbor,
            initial_temperature,
            schedule,
            reannealing: None,
            rng,
        }
    }

    /// Replace the complete schedule-restart configuration.
    pub fn with_reannealing(mut self, reannealing: Reannealing) -> Self {
        self.reannealing = Some(reannealing);
        self
    }

    /// Restart after every `iterations` completed proposals.
    ///
    /// This composes with the accepted-stall and best-stall triggers. Calling
    /// it again replaces only the fixed-interval threshold.
    ///
    /// # Panics
    ///
    /// Panics if `iterations` is zero.
    pub fn with_reannealing_fixed(mut self, iterations: u64) -> Self {
        let reannealing = self
            .reannealing
            .unwrap_or_else(Reannealing::none)
            .with_fixed_interval(iterations);
        self.reannealing = Some(reannealing);
        self
    }

    /// Restart after `iterations` consecutive rejected proposals.
    ///
    /// This composes with the fixed-interval and best-stall triggers. Calling
    /// it again replaces only the accepted-stall threshold.
    ///
    /// # Panics
    ///
    /// Panics if `iterations` is zero.
    pub fn with_reannealing_accepted(mut self, iterations: u64) -> Self {
        let reannealing = self
            .reannealing
            .unwrap_or_else(Reannealing::none)
            .with_accepted_stall(iterations);
        self.reannealing = Some(reannealing);
        self
    }

    /// Restart after `iterations` proposals without a new global best.
    ///
    /// This composes with the fixed-interval and accepted-stall triggers.
    /// Calling it again replaces only the best-stall threshold.
    ///
    /// # Panics
    ///
    /// Panics if `iterations` is zero.
    pub fn with_reannealing_best(mut self, iterations: u64) -> Self {
        let reannealing = self
            .reannealing
            .unwrap_or_else(Reannealing::none)
            .with_best_stall(iterations);
        self.reannealing = Some(reannealing);
        self
    }
}

impl<V, N, F, R> InitialState<V> for SimulatedAnnealing<N, F, R>
where
    V: Clone,
    N: Clone,
    F: Scalar,
    R: Clone,
{
    type State = SimulatedAnnealingState<V, N, F, R>;

    fn seed(&self, x: &V) -> Self::State {
        SimulatedAnnealingState::new(
            x.clone(),
            self.neighbor.clone(),
            self.rng.clone(),
            self.initial_temperature,
            self.schedule,
            self.reannealing,
        )
    }
}

impl<P, V, N, F, R> Solver<P, SimulatedAnnealingState<V, N, F, R>>
    for SimulatedAnnealing<N, F, R>
where
    P: CostFunction<Param = V, Output = F>,
    V: Clone,
    N: Neighbor<V, F, R, Error = P::Error>,
    F: Scalar,
    R: Rng,
{
    type Error = P::Error;

    fn init(
        &mut self,
        problem: &mut Problem<P>,
        mut state: SimulatedAnnealingState<V, N, F, R>,
    ) -> Result<SimulatedAnnealingState<V, N, F, R>, Self::Error> {
        if state.cost.is_none() {
            state.cost = Some(problem.cost(&state.param)?);
        }
        Ok(state)
    }

    fn next_iter(
        &mut self,
        problem: &mut Problem<P>,
        mut state: SimulatedAnnealingState<V, N, F, R>,
    ) -> Result<
        (
            SimulatedAnnealingState<V, N, F, R>,
            Option<TerminationReason>,
        ),
        Self::Error,
    > {
        let incumbent_cost = state.cost.expect(
            "SimulatedAnnealing::next_iter called before init evaluated the start point",
        );
        let temperature = state.temperature();
        let candidate = state.neighbor.propose(
            &state.param,
            temperature,
            &mut state.rng,
        )?;
        let candidate_cost = problem.cost(&candidate)?;
        let new_best = candidate_cost < state.best_cost;

        let accepted =
            if candidate_cost.is_nan() || candidate_cost == F::infinity() {
                false
            } else if candidate_cost <= incumbent_cost {
                true
            } else {
                let probability = uphill_acceptance_probability(
                    candidate_cost - incumbent_cost,
                    temperature,
                );
                let draw = F::from_f64(state.rng.random::<f64>()).unwrap();
                draw < probability
            };

        if accepted {
            state.param = candidate;
            state.cost = Some(candidate_cost);
            state.accepted_moves = state.accepted_moves.saturating_add(1);
            state.last_accepted_iter = state.iter.saturating_add(1);
        } else {
            state.rejected_moves = state.rejected_moves.saturating_add(1);
        }

        state.cooling_age = state.cooling_age.saturating_add(1);
        if let Some(reannealing) = state.reannealing {
            reannealing.update_progress(
                &mut state.reannealing_progress,
                accepted,
                new_best,
            );
            if reannealing.should_restart(state.reannealing_progress) {
                state.cooling_age = 0;
                state.reannealing_progress = ReannealingProgress::default();
                state.reannealings = state.reannealings.saturating_add(1);
            }
        }

        Ok((state, None))
    }

    fn terminate(
        &self,
        state: &SimulatedAnnealingState<V, N, F, R>,
    ) -> Option<TerminationReason> {
        let cost = state.cost();
        if cost.is_nan() {
            Some(TerminationReason::SolverFailed)
        } else if cost == F::neg_infinity() {
            Some(TerminationReason::SolverConverged)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::uphill_acceptance_probability;

    #[test]
    fn metropolis_probability_matches_the_closed_form() {
        let probability = uphill_acceptance_probability(2.0_f64, 4.0);
        assert!((probability - (-0.5_f64).exp()).abs() < f64::EPSILON);
    }
}
