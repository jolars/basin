<script setup lang="ts">
    import References from "#lib/docs/References.vue";
</script>

# Global-best particle swarm optimization

Global-best particle swarm optimization (PSO) moves a population through a
bounded search space. Each particle keeps a velocity and remembers its best
position, while the entire swarm shares one global best. Basin implements
the synchronous, inertia-weight form: every particle in a generation uses
the same frozen history before the new positions are evaluated.

[API reference](https://docs.rs/basin/latest/basin/solver/global_best_pso/struct.GlobalBestPso.html)

| Property | Requirement |
| --- | --- |
| Objective | A scalar function of continuous parameters |
| Evaluations | `CostFunction` |
| Search | Global population search with personal and shared memory |
| Constraints | A box-only declaration through `Constraints`; finite bounds |
| Parameters | All four dense backends, with `f32` or `f64` |

## When to use it

Consider PSO when derivatives are unavailable and you want a population to
explore several regions while sharing useful discoveries. The method
originates with [Kennedy and Eberhart (1995)](#ref-kennedy-eberhart-1995).
Its memory distinguishes it from [random search](../random-search/),
whose new samples remain independent of previous outcomes.

Compare [differential evolution](../de/) for proposals based on differences
between population members, or [CMA-ES](../cma-es/) when adapting a correlated
search distribution is useful. For smooth local refinement from a good
candidate, a derivative-based method or a local derivative-free solver
may reach a precise solution with fewer further evaluations.

Sharing a single global best can draw the swarm together quickly, but it
can also concentrate particles in one local basin. A finite run does not
certify a global solution. Compare independent seeds and keep enough budget
for exploration before interpreting a stable best as evidence of success.

## How it works

For each particle and coordinate, the update is

```text
velocity = w * velocity
         + c1 * r1 * (personal_best - position)
         + c2 * r2 * (global_best - position)
position = position + velocity
```

The draws `r1` and `r2` are independent uniforms for each coordinate. The
inertia `w` carries the previous motion, the cognitive coefficient `c1`
attracts the particle to its own best, and the social coefficient `c2`
attracts it to the swarm's best. The inertia-weight formulation follows
[Shi and Eberhart (1998)](#ref-shi-eberhart-1998).

Basin builds every new position using the previous generation's personal
and global bests, then evaluates the whole swarm. Strict improvements
update those histories. A particle's current cost can get worse, while
the published representative continues to report the best position the
swarm has found.

The defaults use `w = 1 / (2 * ln(2))`,
`c1 = c2 = 0.5 + ln(2)`, and `10 + floor(2 * sqrt(dimension))` particles.
Positions start uniformly in the box. Each initial velocity is half the
displacement from its position to an independent uniform point. These
settings follow the Standard PSO 2006 profile discussed by
[Bratton and Kennedy (2007)](#ref-bratton-kennedy-2007). Basin keeps a
global-best topology; it does not implement that standard's random
neighborhood topology.

## Example

Minimize a bounded quadratic with its minimum at `[1.0, -2.0]`. The
problem declares the box, and an empty `PopulationProgress` asks the
solver to generate the initial swarm. The seed fixes its random draws.

<<< ../../../../../../crates/basin/examples/solver_guides/global_best_pso.rs

[View `global_best_pso.rs` on GitHub](https://github.com/jolars/basin/blob/main/crates/basin/examples/solver_guides/global_best_pso.rs)

These assertions check the quality of this seeded run. The iteration
budget is not an optimality test, and other objectives can require
different swarm sizes, coefficients, or budgets.

## Practical considerations

### Choose motion and boundary controls together

`with_inertia`, `with_cognitive`, and `with_social` accept finite,
nonnegative coefficients. Larger values can change how far particles
travel or how strongly they follow remembered positions. Check the
result across several seeds when tuning them.

Every boundary policy clamps crossed position coordinates into the box.
`PsoBoundaryHandling::Absorb`, the default, also zeros their velocity.
`Preserve` retains it, while `Reflect { damping }` reverses and scales it
by a factor from zero to one. Reflection changes the velocity after
clamping; it does not mirror the particle's overshoot back into the box.

`with_velocity_limit(PsoVelocityLimit::SpanFraction(fraction))` independently
limits each velocity component to a fraction of that coordinate's box
span. The default is `Unbounded`. If motion arithmetic overflows or
becomes `NaN`, that coordinate keeps its previous feasible position and
its velocity becomes zero.

### Account for a full swarm per iteration

Initialization evaluates every particle, and each generation evaluates
every new position. `with_swarm_size` overrides the default size. Compare
settings using objective evaluations, since the same iteration budget
costs more with a larger swarm.

Use executor controls such as `max_iter`, `max_cost_evals`,
`target_objective`, or `no_objective_improvement`. Evaluation limits are
checked between generations, so a complete swarm evaluation can cross
the requested limit. Optional step- and cost-change checks observe the
global-best representative, including unchanged generations. Zero change
can therefore mean that no particle found a new record this generation.

### Declare the full supported constraint set

PSO requires `ConstraintSet<BoxBounds<V>>`, with both vectors present,
non-empty, finite, equally sized, and ordered. Equal bounds fix a
coordinate. Additional linear or nonlinear constraint blocks are
unsupported; boundary handling only enforces the declared box.

`NaN` and positive-infinite objective values cannot update the remembered
bests. If no initial particle has a usable cost, the solver reports
failure. A negative-infinite global best terminates the run as converged.
Typed errors returned by the objective propagate through the run.

## Implementation

`PopulationProgress` owns current positions, their costs, and the
published global-best representative. The solver owns velocities,
personal bests, its global-best model, and the live RNG. Retain it with
`Executor::run_with_solver` to inspect `velocities()`,
`personal_best_positions()`, `personal_best_costs()`, and `global_best()`.
These arrays remain aligned when the solver sorts the current population.

`PopulationProgress::from_population` supplies finite starting positions
with the box's dimension. Their count determines the swarm size unless
an explicit size override is present, in which case the counts must match.
`with_initial_velocities(Some(values))` supplies finite velocities in the
input member order. Otherwise the solver samples fresh initial velocities.

Fresh initialization resets progress and personal histories, repairs
positions, and reevaluates the swarm. It restores the configured RNG, so
the same seed and inputs repeat a run. `new_with_rng` accepts another
clonable RNG. Exact continuation must preserve solver, state, and counts
together; a state-only warm start loses velocity, history, and RNG state.
The `serde` feature supports serializing compatible solver checkpoints.

Supported parameter types are `Vec<F>`, `nalgebra::DVector<F>`,
`ndarray::Array1<F>`, and `faer::Col<F>`, with `F = f32` or `f64`. No
matrix operations or BLAS/LAPACK provider are required. Independent
objective evaluations can run concurrently with `with_parallel_evaluation()`
and the `parallel` feature. Proposal order is preserved. See the API's [Backends
section](https://docs.rs/basin/latest/basin/solver/global_best_pso/struct.GlobalBestPso.html#backends)
for custom-vector requirements.

<References :entries='["kennedy-eberhart-1995", "shi-eberhart-1998", "bratton-kennedy-2007"]' />
