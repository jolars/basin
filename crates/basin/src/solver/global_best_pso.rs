//! Synchronous global-best particle swarm optimization.

use crate::core::constraint::BoxConstraints;
use crate::core::math::{SampleUniformBox, Scalar, VectorLen};
use crate::core::problem::{CostFunction, Problem};
use crate::core::rng::{ChaCha8Rng, Rng, RngExt, SeedableRng};
use crate::core::solver::Solver;
use crate::core::state::PopulationProgress;
use crate::core::termination::Termination;
use crate::solver::cma_es::{apply_permutation, nan_last_cmp};

/// Response applied when a particle crosses a box boundary.
///
/// Every policy first clamps the position into the feasible box. The policy
/// controls only the corresponding velocity component.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub enum PsoBoundaryHandling<F: Scalar = f64> {
    /// Set the outward velocity component to zero.
    Absorb,
    /// Preserve velocity after clamping the position.
    Preserve,
    /// Reverse velocity and multiply its magnitude by `damping`.
    Reflect {
        /// Restitution factor in `[0, 1]`.
        damping: F,
    },
}

/// Optional component-wise velocity limit.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub enum PsoVelocityLimit<F: Scalar = f64> {
    /// Do not clamp velocity.
    Unbounded,
    /// Clamp `|v_j|` to `fraction · (upper_j - lower_j)`.
    SpanFraction(F),
}

/// Synchronous global-best particle swarm optimization.
///
/// This solver implements the inertia-weight global-best rule
///
/// `vᵢⱼ ← w vᵢⱼ + c₁ r₁ (pᵢⱼ − xᵢⱼ) + c₂ r₂ (gⱼ − xᵢⱼ)`,
/// followed by `xᵢ ← xᵢ + vᵢ`. All particles in a generation use the same
/// frozen personal and global bests, and use independent `r₁, r₂ ~ U(0, 1)`
/// for each coordinate. Objective evaluations are batched after the complete
/// generation is built, preserving deterministic order under the `parallel`
/// feature.
///
/// The name is deliberately topology-specific. Standard PSO 2006 and Standard
/// PSO 2011 use changing random neighborhoods, and SPSO-2011 also changes the
/// motion distribution; those variants should be separate public solvers
/// rather than configuration flags that silently change this update rule.
///
/// # Defaults
///
/// The coefficient profile and default swarm size follow the Standard PSO 2006
/// recommendations: `w = 1/(2 ln 2)`, `c₁ = c₂ = 1/2 + ln 2`, and
/// `10 + floor(2 sqrt(D))` particles. Initial positions are uniform in the
/// box, and initial velocities are `(u - x)/2` for an independent uniform
/// `u`. The topology remains global-best, so this is not presented as a full
/// SPSO-2006 implementation. Boundary crossings are absorbed by default, and
/// velocity is otherwise unbounded.
///
/// # Contract
///
/// - **Caller must:** implement [`CostFunction`] and [`BoxConstraints`] with a
///   non-empty pair of same-shaped, finite bounds satisfying
///   `lower[j] ≤ upper[j]`.
/// - Supply [`PopulationProgress::empty`] for uniform initialization, or
///   [`PopulationProgress::from_population`] for explicit finite positions.
///   [`with_initial_velocities`](Self::with_initial_velocities) optionally
///   supplies velocities in the input member order. A configured swarm-size
///   override must match a supplied population. Otherwise its length sets the size.
/// - **Implementor (this solver) must:** keep positions feasible, keep all
///   particle arrays in parallel order, and maintain the personal/global
///   bests using strict improvements. `NaN` and `+∞` are treated as rejected
///   point costs; if initialization finds no usable cost, the run stops with
///   [`crate::TerminationCode::SolverFailed`]. A global cost of `-∞` stops with
///   [`crate::TerminationCode::SolverConverged`].
///
/// Builder methods panic immediately on invalid coefficients or policy
/// values. Invalid problem bounds or malformed warm-start vector shapes panic
/// during initialization because they violate the solver's structural
/// contract. Typed errors returned by the objective propagate unchanged. If a
/// velocity update overflows or becomes `NaN`, that coordinate stays at its
/// previous feasible position and its velocity is reset to zero, isolating the
/// failed motion from the remaining swarm.
///
/// # Progress and lifecycle
///
/// Publishes [`PopulationProgress<V, F>`]: member positions and costs live in
/// state, while the solver owns velocities, personal bests, the global-best
/// model, RNGs, and working buffers. The current record remains the global best
/// even when every current particle is worse. Shared progress retains a matching
/// incumbent and all raw evaluation categories; `cost_evals()` means cost calls.
/// Use the state's checked `current()`, `best()`, and `evaluated_members()`
/// readers to distinguish unevaluated seeds from published progress.
///
/// Read [`velocities`](Self::velocities),
/// [`personal_best_positions`](Self::personal_best_positions),
/// [`personal_best_costs`](Self::personal_best_costs), and
/// [`global_best`](Self::global_best) through the solver retained by
/// [`Executor::run_with_solver`](crate::Executor::run_with_solver) or a stepper's
/// [`solver`](crate::Stepper::solver) reader. Particle arrays remain aligned
/// when this solver sorts the population by current cost.
///
/// Fresh initialization resets progress and particle histories, clones the
/// configured RNG, repairs positions and velocities according to the boundary
/// policy, and reevaluates every member. Reusing progress supplies the current
/// member positions as seeds. Configured initial velocities are reapplied in
/// that input order; `None` samples fresh half-displacement velocities.
/// [`new`](Self::new) configures Basin's wasm-safe [`ChaCha8Rng`];
/// [`new_with_rng`](Self::new_with_rng) accepts another clonable RNG.
///
/// Exact continuation retains solver, state, and counts together and skips
/// initialization. With `serde`, the solver and shared progress serialize when
/// their vector, scalar, and RNG types do. Old `GlobalBestPsoState` payloads are
/// incompatible with this layout. State-only exact resume is unsupported.
///
/// # Termination
///
/// PSO has no single canonical convergence test. Pair it with framework
/// criteria such as [`max_iter`](crate::Executor::max_iter), [`max_cost_evals`](crate::Executor::max_cost_evals),
/// [`target_objective`](crate::Executor::target_objective), or
/// [`no_objective_improvement`](crate::Executor::no_objective_improvement).
/// Optional cost-change and step checks compare the global-best representative,
/// including generations that retain it unchanged.
///
/// # Backends
///
/// `Vec<F>`, nalgebra `DVector<F>`, ndarray `Array1<F>`, and faer `Col<F>`
/// for every supported release with `F = f32` or `f64`. Requires `Clone`,
/// [`VectorLen`], [`SampleUniformBox`], `Index<usize, Output = F>`, and
/// `IndexMut<usize, Output = F>`. No matrix operations are required.
///
/// # Example
///
/// ```
/// use basin::{
///     BoxConstraints, CostFunction, Executor, GlobalBestPso,
///     PopulationProgress,
/// };
///
/// struct Sphere {
///     lower: Vec<f64>,
///     upper: Vec<f64>,
/// }
/// impl CostFunction for Sphere {
///     type Param = Vec<f64>;
///     type Output = f64;
///     type Error = std::convert::Infallible;
///     fn cost(&self, x: &Vec<f64>) -> Result<f64, Self::Error> {
///         Ok(x.iter().map(|xi| xi * xi).sum())
///     }
/// }
/// impl BoxConstraints for Sphere {
///     fn lower(&self) -> &Vec<f64> {
///         &self.lower
///     }
///     fn upper(&self) -> &Vec<f64> {
///         &self.upper
///     }
/// }
///
/// let result = Executor::new(
///     Sphere {
///         lower: vec![-5.0; 2],
///         upper: vec![5.0; 2],
///     },
///     GlobalBestPso::new(42),
///     PopulationProgress::<Vec<f64>>::empty(),
/// )
/// .max_iter(300)
/// .run()
/// .unwrap();
/// assert!(result.cost() < 1e-6);
/// ```
///
/// # References
///
/// - Kennedy, J., and Eberhart, R. (1995). “Particle Swarm
///   Optimization.” *Proceedings of ICNN'95*, 1942–1948.
///   <https://doi.org/10.1109/ICNN.1995.488968>.
/// - Shi, Y., and Eberhart, R. (1998). “A Modified Particle Swarm
///   Optimizer.” *IEEE International Conference on Evolutionary
///   Computation*, 69–73. <https://doi.org/10.1109/ICEC.1998.699146>.
/// - Bratton, D., and Kennedy, J. (2007). “Defining a Standard for
///   Particle Swarm Optimization.” *IEEE Swarm Intelligence Symposium*,
///   120–127. <https://doi.org/10.1109/SIS.2007.368035>.
/// - Zambrano-Bigiarini, M., Clerc, M., and Rojas, R. (2013). “Standard
///   Particle Swarm Optimisation 2011 at CEC-2013: A Baseline for Future PSO
///   Improvements.” *IEEE Congress on Evolutionary Computation*, 2337–2344.
///   <https://doi.org/10.1109/CEC.2013.6557848>.
/// - Argmin 0.11.0, `ParticleSwarm`, commit
///   `c94c32adefd6c2525ce05806092ca868ec85fba4` (MIT OR Apache-2.0), was
///   used as an executable global-best reference. Its source cites SPSO-2011,
///   but implements the coordinate-wise global-best rule documented here;
///   Basin does not reproduce that citation mismatch.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug)]
pub struct GlobalBestPso<V, F: Scalar = f64, R = ChaCha8Rng> {
    swarm_size_override: Option<usize>,
    inertia: F,
    cognitive: F,
    social: F,
    boundary_handling: PsoBoundaryHandling<F>,
    velocity_limit: PsoVelocityLimit<F>,
    rng: R,
    initial_velocities: Option<Vec<V>>,
    live_rng: Option<R>,
    velocities: Vec<V>,
    personal_best_positions: Vec<V>,
    personal_best_costs: Vec<F>,
    global_best_position: Option<V>,
    global_best_cost: F,
    cognitive_draws: Vec<F>,
    social_draws: Vec<F>,
}

impl<V, F: Scalar> GlobalBestPso<V, F, ChaCha8Rng> {
    /// Construct a global-best PSO with the research-profile defaults and an
    /// RNG initialized from `seed`.
    pub fn new(seed: u64) -> Self {
        Self::new_with_rng(ChaCha8Rng::seed_from_u64(seed))
    }
}

impl<V, F: Scalar, R> GlobalBestPso<V, F, R> {
    /// Construct a global-best PSO with the research-profile defaults and a
    /// caller-supplied RNG.
    pub fn new_with_rng(rng: R) -> Self {
        let two = F::from_f64(2.0).unwrap();
        Self {
            swarm_size_override: None,
            inertia: F::one() / (two * two.ln()),
            cognitive: F::from_f64(0.5).unwrap() + two.ln(),
            social: F::from_f64(0.5).unwrap() + two.ln(),
            boundary_handling: PsoBoundaryHandling::Absorb,
            velocity_limit: PsoVelocityLimit::Unbounded,
            rng,
            initial_velocities: None,
            live_rng: None,
            velocities: Vec::new(),
            personal_best_positions: Vec::new(),
            personal_best_costs: Vec::new(),
            global_best_position: None,
            global_best_cost: F::infinity(),
            cognitive_draws: Vec::new(),
            social_draws: Vec::new(),
        }
    }

    /// Configure explicit velocities for fresh initialization, or `None` to
    /// sample the default half-displacement velocities. Entries correspond to
    /// the initial member order, before any sorting. Fresh runs reapply these
    /// configured values; exact continuation retains evolving velocities.
    /// Initialization validates their count, dimensions, and finiteness.
    pub fn with_initial_velocities(
        mut self,
        velocities: Option<Vec<V>>,
    ) -> Self {
        if let Some(values) = &velocities {
            assert!(
                !values.is_empty(),
                "GlobalBestPso requires at least one initial velocity"
            );
        }
        self.initial_velocities = velocities;
        self
    }

    /// Current velocities in the same order as the state's population members.
    /// Empty before initialization.
    pub fn velocities(&self) -> &[V] {
        &self.velocities
    }

    /// Personal-best positions in current member order. Empty before initialization.
    pub fn personal_best_positions(&self) -> &[V] {
        &self.personal_best_positions
    }

    /// Costs paired with [`personal_best_positions`](Self::personal_best_positions).
    pub fn personal_best_costs(&self) -> &[F] {
        &self.personal_best_costs
    }

    /// The swarm's eligible model best, or `None` before initialization or
    /// when every initial cost was NaN or positive infinity.
    pub fn global_best(&self) -> Option<(&V, F)> {
        self.global_best_position
            .as_ref()
            .filter(|_| usable_cost(self.global_best_cost))
            .map(|point| (point, self.global_best_cost))
    }

    /// Standard PSO 2006's `10 + floor(2 sqrt(D))` swarm-size rule.
    pub fn default_swarm_size(dimension: usize) -> usize {
        10 + (2.0 * (dimension as f64).sqrt()).floor() as usize
    }

    /// Current inertia coefficient `w`.
    pub fn inertia(&self) -> F {
        self.inertia
    }

    /// Current cognitive coefficient `c₁`.
    pub fn cognitive(&self) -> F {
        self.cognitive
    }

    /// Current social coefficient `c₂`.
    pub fn social(&self) -> F {
        self.social
    }

    /// Override the swarm size resolved during initialization.
    ///
    /// # Panics
    ///
    /// Panics if `swarm_size == 0`.
    pub fn with_swarm_size(mut self, swarm_size: usize) -> Self {
        assert!(swarm_size >= 1, "GlobalBestPso requires swarm_size >= 1");
        self.swarm_size_override = Some(swarm_size);
        self
    }

    /// Set the inertia coefficient `w`.
    ///
    /// # Panics
    ///
    /// Panics unless `inertia` is finite and nonnegative.
    pub fn with_inertia(mut self, inertia: F) -> Self {
        assert_nonnegative_finite("inertia", inertia);
        self.inertia = inertia;
        self
    }

    /// Set the cognitive coefficient `c₁`.
    ///
    /// # Panics
    ///
    /// Panics unless `cognitive` is finite and nonnegative.
    pub fn with_cognitive(mut self, cognitive: F) -> Self {
        assert_nonnegative_finite("cognitive coefficient", cognitive);
        self.cognitive = cognitive;
        self
    }

    /// Set the social coefficient `c₂`.
    ///
    /// # Panics
    ///
    /// Panics unless `social` is finite and nonnegative.
    pub fn with_social(mut self, social: F) -> Self {
        assert_nonnegative_finite("social coefficient", social);
        self.social = social;
        self
    }

    /// Set the response to box-boundary crossings.
    ///
    /// # Panics
    ///
    /// Panics unless a reflection damping factor is finite and in `[0, 1]`.
    pub fn with_boundary_handling(
        mut self,
        boundary_handling: PsoBoundaryHandling<F>,
    ) -> Self {
        if let PsoBoundaryHandling::Reflect { damping } = boundary_handling {
            assert!(
                damping.is_finite()
                    && damping >= F::zero()
                    && damping <= F::one(),
                "GlobalBestPso reflection damping must be finite and in [0, 1], got {damping:?}"
            );
        }
        self.boundary_handling = boundary_handling;
        self
    }

    /// Set a component-wise velocity limit, independently of boundary repair.
    ///
    /// # Panics
    ///
    /// Panics unless a span fraction is finite and nonnegative.
    pub fn with_velocity_limit(
        mut self,
        velocity_limit: PsoVelocityLimit<F>,
    ) -> Self {
        if let PsoVelocityLimit::SpanFraction(fraction) = velocity_limit {
            assert_nonnegative_finite("velocity span fraction", fraction);
        }
        self.velocity_limit = velocity_limit;
        self
    }
}

fn assert_nonnegative_finite<F: Scalar>(name: &str, value: F) {
    assert!(
        value.is_finite() && value >= F::zero(),
        "GlobalBestPso {name} must be finite and nonnegative, got {value:?}"
    );
}

fn usable_cost<F: Scalar>(cost: F) -> bool {
    cost < F::infinity()
}

fn strictly_better<F: Scalar>(candidate: F, incumbent: F) -> bool {
    usable_cost(candidate) && (!usable_cost(incumbent) || candidate < incumbent)
}

fn validate_box<V, F>(lower: &V, upper: &V)
where
    V: VectorLen + std::ops::Index<usize, Output = F>,
    F: Scalar,
{
    assert_eq!(
        lower.vec_len(),
        upper.vec_len(),
        "GlobalBestPso requires lower and upper bounds of equal length"
    );
    assert!(
        lower.vec_len() > 0,
        "GlobalBestPso requires a non-empty search box"
    );
    for j in 0..lower.vec_len() {
        assert!(
            lower[j].is_finite() && upper[j].is_finite(),
            "GlobalBestPso requires finite bounds, got lower[{j}] = {:?}, upper[{j}] = {:?}",
            lower[j],
            upper[j]
        );
        assert!(
            lower[j] <= upper[j],
            "GlobalBestPso requires lower <= upper, got lower[{j}] = {:?}, upper[{j}] = {:?}",
            lower[j],
            upper[j]
        );
    }
}

fn validate_particle_shape<V, F>(
    value: &V,
    dimension: usize,
    kind: &str,
    particle: usize,
) where
    V: VectorLen + std::ops::Index<usize, Output = F>,
    F: Scalar,
{
    assert_eq!(
        value.vec_len(),
        dimension,
        "GlobalBestPso {kind} {particle} has dimension {}, expected {dimension}",
        value.vec_len()
    );
    for j in 0..dimension {
        assert!(
            value[j].is_finite(),
            "GlobalBestPso {kind} {particle} contains a non-finite coordinate at index {j}"
        );
    }
}

fn repair_position<V, F>(
    position: &mut V,
    velocity: &mut V,
    lower: &V,
    upper: &V,
    handling: PsoBoundaryHandling<F>,
) where
    V: VectorLen
        + std::ops::Index<usize, Output = F>
        + std::ops::IndexMut<usize, Output = F>,
    F: Scalar,
{
    for j in 0..position.vec_len() {
        let crossed = if position[j] < lower[j] {
            position[j] = lower[j];
            true
        } else if position[j] > upper[j] {
            position[j] = upper[j];
            true
        } else {
            false
        };
        if crossed {
            match handling {
                PsoBoundaryHandling::Absorb => velocity[j] = F::zero(),
                PsoBoundaryHandling::Preserve => {}
                PsoBoundaryHandling::Reflect { damping } => {
                    velocity[j] = -damping * velocity[j];
                }
            }
        }
    }
}

impl<V, F: Scalar, R> GlobalBestPso<V, F, R> {
    fn sort_particles_ascending(
        &mut self,
        state: &mut PopulationProgress<V, F>,
    ) {
        let n = state.candidates.len();
        debug_assert_eq!(state.costs.len(), n);
        debug_assert_eq!(self.velocities.len(), n);
        debug_assert_eq!(self.personal_best_positions.len(), n);
        debug_assert_eq!(self.personal_best_costs.len(), n);
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&i, &j| nan_last_cmp(&state.costs[i], &state.costs[j]));
        apply_permutation(&mut state.candidates, &order);
        apply_permutation(&mut state.costs, &order);
        apply_permutation(&mut self.velocities, &order);
        apply_permutation(&mut self.personal_best_positions, &order);
        apply_permutation(&mut self.personal_best_costs, &order);
    }
}

impl<P, V, F, R> Solver<P, PopulationProgress<V, F>> for GlobalBestPso<V, F, R>
where
    F: Scalar + crate::core::parallel::MaybeSend,
    P: CostFunction<Param = V, Output = F>
        + BoxConstraints<Param = V>
        + crate::core::parallel::MaybeSync,
    P::Error: crate::core::parallel::MaybeSend,
    V: Clone
        + VectorLen
        + SampleUniformBox
        + crate::core::parallel::MaybeSync
        + std::ops::Index<usize, Output = F>
        + std::ops::IndexMut<usize, Output = F>,
    R: Rng + Clone,
{
    type Error = P::Error;

    fn init(
        &mut self,
        problem: &mut Problem<P>,
        mut state: PopulationProgress<V, F>,
    ) -> Result<PopulationProgress<V, F>, Self::Error> {
        state.reset();
        self.velocities.clear();
        self.personal_best_positions.clear();
        self.personal_best_costs.clear();
        self.global_best_position = None;
        self.global_best_cost = F::infinity();
        self.live_rng = None;

        let lower = problem.inner().lower().clone();
        let upper = problem.inner().upper().clone();
        validate_box(&lower, &upper);
        let dimension = lower.vec_len();
        let supplied_positions = !state.candidates.is_empty();
        let swarm_size = if supplied_positions {
            let supplied = state.candidates.len();
            if let Some(configured) = self.swarm_size_override {
                assert_eq!(
                    supplied, configured,
                    "GlobalBestPso warm start has {supplied} particles, but the configured swarm size is {configured}"
                );
            }
            supplied
        } else {
            self.swarm_size_override
                .unwrap_or_else(|| Self::default_swarm_size(dimension))
        };
        assert!(swarm_size >= 1, "GlobalBestPso requires swarm_size >= 1");

        let mut rng = self.rng.clone();
        if !supplied_positions {
            state.candidates.reserve(swarm_size);
            for _ in 0..swarm_size {
                state
                    .candidates
                    .push(V::sample_uniform_box(&lower, &upper, &mut rng));
            }
        }
        for (i, position) in state.candidates.iter().enumerate() {
            validate_particle_shape(position, dimension, "position", i);
        }

        if let Some(initial) = &self.initial_velocities {
            self.velocities.clone_from(initial);
        }
        if self.velocities.is_empty() {
            self.velocities.reserve(swarm_size);
            for position in &state.candidates {
                let target = V::sample_uniform_box(&lower, &upper, &mut rng);
                let mut velocity = target;
                let half = F::from_f64(0.5).unwrap();
                for j in 0..dimension {
                    velocity[j] = half * velocity[j] - half * position[j];
                }
                self.velocities.push(velocity);
            }
        } else {
            assert_eq!(
                self.velocities.len(),
                swarm_size,
                "GlobalBestPso requires one velocity per position"
            );
            for (i, velocity) in self.velocities.iter().enumerate() {
                validate_particle_shape(velocity, dimension, "velocity", i);
            }
        }

        for (position, velocity) in
            state.candidates.iter_mut().zip(&mut self.velocities)
        {
            repair_position(
                position,
                velocity,
                &lower,
                &upper,
                self.boundary_handling,
            );
        }
        state.costs = problem.cost_batch(&state.candidates)?;
        self.personal_best_positions = state.candidates.clone();
        self.personal_best_costs = state.costs.clone();
        self.sort_particles_ascending(&mut state);

        let best = state
            .costs
            .iter()
            .position(|&cost| usable_cost(cost))
            .unwrap_or(0);
        self.global_best_position = Some(state.candidates[best].clone());
        self.global_best_cost = state.costs[best];
        self.live_rng = Some(rng);
        self.cognitive_draws.resize(dimension, F::zero());
        self.social_draws.resize(dimension, F::zero());
        self.cognitive_draws.fill(F::zero());
        self.social_draws.fill(F::zero());
        state
            .publish_representative_from(
                self.global_best_position.as_ref().unwrap(),
                self.global_best_cost,
            )
            .expect("initialized swarm has matching member shapes");
        Ok(state)
    }

    fn next_iter(
        &mut self,
        problem: &mut Problem<P>,
        mut state: PopulationProgress<V, F>,
    ) -> Result<crate::SolverStep<PopulationProgress<V, F>>, Self::Error> {
        let lower = problem.inner().lower().clone();
        let upper = problem.inner().upper().clone();
        let global_best = self
            .global_best_position
            .as_ref()
            .expect("GlobalBestPso::init must run before next_iter")
            .clone();
        let rng = self
            .live_rng
            .as_mut()
            .expect("GlobalBestPso::init must seed the live RNG");
        let dimension = lower.vec_len();

        for i in 0..state.candidates.len() {
            // Argmin 0.11's coordinate-wise reference draws the complete
            // cognitive vector before the complete social vector. Group the
            // draws the same way, while always consuming the documented two
            // draws per coordinate even when a pull happens to be zero.
            for draw in &mut self.cognitive_draws {
                *draw = F::from_f64(rng.random::<f64>()).unwrap();
            }
            for draw in &mut self.social_draws {
                *draw = F::from_f64(rng.random::<f64>()).unwrap();
            }
            for j in 0..dimension {
                let position = state.candidates[i][j];
                let raw_velocity = self.inertia * self.velocities[i][j]
                    + self.cognitive
                        * self.cognitive_draws[j]
                        * (self.personal_best_positions[i][j] - position)
                    + self.social
                        * self.social_draws[j]
                        * (global_best[j] - position);
                if !raw_velocity.is_finite() {
                    // One numerically exploded particle must not poison the
                    // whole swarm or violate the feasible-position invariant.
                    self.velocities[i][j] = F::zero();
                    continue;
                }
                self.velocities[i][j] = match self.velocity_limit {
                    PsoVelocityLimit::Unbounded => raw_velocity,
                    PsoVelocityLimit::SpanFraction(fraction) => {
                        let span = upper[j] - lower[j];
                        let limit = if span.is_finite() {
                            fraction * span
                        } else {
                            // Scaling the endpoints first preserves finite
                            // caps when the valid finite box is too wide for
                            // its unscaled span to be represented.
                            fraction * upper[j] - fraction * lower[j]
                        };
                        raw_velocity.max(-limit).min(limit)
                    }
                };
                state.candidates[i][j] = position + self.velocities[i][j];
            }
            repair_position(
                &mut state.candidates[i],
                &mut self.velocities[i],
                &lower,
                &upper,
                self.boundary_handling,
            );
        }

        state.costs = problem.cost_batch(&state.candidates)?;
        for i in 0..state.candidates.len() {
            if strictly_better(state.costs[i], self.personal_best_costs[i]) {
                self.personal_best_positions[i] = state.candidates[i].clone();
                self.personal_best_costs[i] = state.costs[i];
            }
            if strictly_better(
                self.personal_best_costs[i],
                self.global_best_cost,
            ) {
                self.global_best_position =
                    Some(self.personal_best_positions[i].clone());
                self.global_best_cost = self.personal_best_costs[i];
            }
        }
        self.sort_particles_ascending(&mut state);
        state
            .publish_representative_from(
                self.global_best_position.as_ref().unwrap(),
                self.global_best_cost,
            )
            .expect("updated swarm has matching member shapes");
        Ok(crate::SolverStep::from((state, None)))
    }

    fn terminate(
        &self,
        _state: &PopulationProgress<V, F>,
    ) -> Option<Termination<F>> {
        let cost = self.global_best_cost;
        if !usable_cost(cost) {
            Some(Termination::numerical_failure(
                "The swarm has no incumbent with a usable objective value.",
            ))
        } else if cost == F::neg_infinity() {
            Some(Termination::custom(
                "global_best_pso.negative_infinity",
                "The published incumbent objective equals negative infinity.",
                vec![crate::Measurement {
                    name: "cost".into(),
                    value: cost,
                }],
            ))
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strict_improvement_replaces_nan_and_infinity() {
        assert!(strictly_better(1.0, f64::NAN));
        assert!(strictly_better(1.0, f64::INFINITY));
        assert!(!strictly_better(f64::NAN, 1.0));
        assert!(!strictly_better(f64::INFINITY, 1.0));
        assert!(!strictly_better(1.0, 1.0));
    }

    #[test]
    #[should_panic(expected = "swarm_size >= 1")]
    fn zero_swarm_size_is_rejected() {
        let _ = GlobalBestPso::<Vec<f64>>::new(0).with_swarm_size(0);
    }

    #[test]
    #[should_panic(expected = "inertia must be finite and nonnegative")]
    fn negative_inertia_is_rejected() {
        let _ = GlobalBestPso::<Vec<f64>>::new(0).with_inertia(-0.1);
    }

    #[test]
    #[should_panic(
        expected = "reflection damping must be finite and in [0, 1]"
    )]
    fn invalid_reflection_damping_is_rejected() {
        let _ = GlobalBestPso::<Vec<f64>>::new(0).with_boundary_handling(
            PsoBoundaryHandling::Reflect { damping: 1.1 },
        );
    }

    #[test]
    #[should_panic(
        expected = "velocity span fraction must be finite and nonnegative"
    )]
    fn negative_velocity_fraction_is_rejected() {
        let _ = GlobalBestPso::<Vec<f64>>::new(0)
            .with_velocity_limit(PsoVelocityLimit::SpanFraction(-0.1));
    }

    #[test]
    #[should_panic(expected = "requires a non-empty search box")]
    fn empty_box_is_rejected() {
        validate_box::<Vec<f64>, f64>(&vec![], &vec![]);
    }

    #[test]
    #[should_panic(expected = "requires lower <= upper")]
    fn unordered_box_is_rejected() {
        validate_box(&vec![1.0], &vec![-1.0]);
    }
}
