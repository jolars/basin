<script setup lang="ts">
    import References from "#lib/docs/References.vue";
</script>

# Differential evolution

Differential evolution (DE) searches a bounded continuous space using a
population of candidate solutions. It creates proposals from differences
between population members, combines them with the current candidates, and
keeps proposals that are at least as good. Basin defaults to the
`DE/rand/1/bin` strategy and offers several mutation and crossover rules.

[API reference](https://docs.rs/basin/latest/basin/solver/de/struct.De.html)

| Property | Requirement |
| --- | --- |
| Objective | A scalar function of continuous parameters |
| Evaluations | `CostFunction` |
| Search | Global population search, with seeded randomness |
| Constraints | A box-only declaration through `Constraints`; finite bounds |
| Parameters | All four dense backends, with `f32` or `f64` |

## When to use it

Consider DE when the objective has several local minima, derivatives are
unavailable, and meaningful finite bounds describe the search region.
[Storn and Price (1997)](#ref-storn-price-1997) introduced the method for
continuous global optimization. The population supplies its own proposal
directions and scales through differences between members, without a gradient
or covariance factorization.

Compare [particle swarm optimization](../global-best-pso/) when you want
particles to remember their personal best positions, or [CMA-ES](../cma-es/)
when learning correlations between parameters is useful. If evaluations are
expensive and a local solution is sufficient, a model-based method such as
[BOBYQA](../../derivative-free/bobyqa/) may use the budget more efficiently.

A finite DE run does not certify a global minimum. Compare multiple seeds
and inspect the objective budget, particularly as the number of parameters
grows. A population that has collapsed into one basin can stop exploring
even when another basin contains a better solution.

## How it works

With the default `Rand1` mutation, each target `x` receives a donor
`a + F * (b - c)`, using three distinct peers that exclude the target.
Binomial crossover independently takes coordinates from that donor with
probability `CR`; one randomly chosen coordinate always comes from the donor.
The trial replaces its target when its objective is no worse.

All donors use the population from the beginning of the generation. Basin
evaluates the full set of trials before accepting replacements, so an early
success cannot change later proposals in that generation. It then sorts the
population by cost and publishes the best member. Each generation uses one
objective evaluation per member, following an initial population evaluation.

The mutation and crossover choices are independent:

| Setting | Effect |
| --- | --- |
| `Rand1` | Random base plus one population difference; the default |
| `Best1` | Best member as the base, plus one difference |
| `Rand2` or `Best2` | Random or best base, plus two differences |
| `RandToBest1` | Random base moved toward the best, plus one difference |
| `CurrentToBest1` | Target moved toward the best, plus one difference |
| `Binomial` | Independent coordinate choices, with one guaranteed donor coordinate |
| `Exponential` | One consecutive, wrapping donor segment |

The named mutation formulas and crossover conventions match the
[SciPy 1.16.2 documentation](https://docs.scipy.org/doc/scipy-1.16.2/reference/generated/scipy.optimize.differential_evolution.html).
Basin uses its own RNG and sampling order. Before crossover, it replaces each
donor coordinate outside the box, or with a non-finite value, with a new
uniform draw inside that coordinate's bounds.

## Example

Minimize a bounded quadratic whose minimum is `[1.0, -2.0]`. The problem
declares its finite box independently of `CostFunction`. An empty population
asks DE to sample its initial members from that box.

<<< ../../../../../../crates/basin/examples/solver_guides/de.rs

[View `de.rs` on GitHub](https://github.com/jolars/basin/blob/main/crates/basin/examples/solver_guides/de.rs)

The assertions check this deterministic example's final accuracy. The
iteration limit is an execution budget and does not establish convergence.

## Practical considerations

### Balance population size and generations

The default population size is `max(4, 10 * dimension)`.
`with_pop_size` sets the total number of members, rather than a multiplier
of the dimension. `Rand2` requires at least six members and `Best2` at least
five; the remaining rules require at least four. A larger population gives
more initial coverage but leaves fewer generations within a fixed cost budget.

`max_cost_evals` is checked at iteration boundaries. Initialization evaluates
the entire population, and a generation completes its full batch, so the
requested limit need not be an exact cap on objective calls.

### Tune mutation and crossover together

Defaults are `F = 0.8` and `CR = 0.9`. `with_f` sets a positive finite
mutation scale, while `with_cr` accepts probabilities from zero to one.
Even at `CR = 0`, crossover copies one donor coordinate; at `CR = 1`, it
copies the whole donor. A copied coordinate can still equal the target.

`with_dither(min, max)` instead samples one mutation scale from `[min, max)`
per generation and shares it across all donors. Both endpoints must be
positive and finite, with `min < max`. Calling `with_f` later disables
dithering. Compare settings using the same budget across several seeds.

### Distinguish stagnation from optimality

DE has no default optimality test. Optional absolute or relative step- and
cost-change checks compare successive best representatives, including
generations that retain the same member. One unchanged best can therefore
satisfy a change check while the rest of the population is still searching.
An executor stagnation window can be more useful when you want to allow
several generations without improvement.

The problem must declare `ConstraintSet<BoxBounds<V>>`, with both vectors
present, non-empty, finite, equally sized, and ordered. Equal bounds pin a
coordinate. DE rejects unsupported constraint blocks; its box repair does
not enforce additional linear or nonlinear constraints.

## Implementation

`PopulationProgress` stores the member positions, matching objective values,
and published best representative. DE owns the configured seed and live
`ChaCha8Rng`. With the `parallel` feature, call `with_parallel_evaluation()` to evaluate
independent trials concurrently while preserving proposal order.

Use `PopulationProgress::from_population` to supply exactly the configured
number of finite vectors with the bounds' dimension. Fresh initialization
resets progress and the RNG, projects explicit members into the box, and
reevaluates them. Reusing final members is a population warm start. An empty
state repeats the original seeded initialization.

Exact continuation retains the solver, state, and evaluation counts together
and skips initialization. The `serde` feature supports serializing those
checkpoints; a state-only snapshot does not preserve the random trajectory.

Supported parameter types are `Vec<F>`, `nalgebra::DVector<F>`,
`ndarray::Array1<F>`, and `faer::Col<F>`, with `F = f32` or `f64`. DE needs
uniform box sampling and basic vector operations, without a matrix
factorization or BLAS/LAPACK provider. See the API's [Backends
section](https://docs.rs/basin/latest/basin/solver/de/struct.De.html#backends)
for custom-vector requirements.

<References :entries='["storn-price-1997"]' />
