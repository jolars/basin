# Getting started

## Install

Add Basin to your project:

```sh
cargo add basin
```

Basin works on plain `Vec<f64>` out of the box. Linear-algebra backends are
opt-in. For example, to use the latest supported
[nalgebra](https://nalgebra.rs/) backend (required by some solvers, see
[Solvers](../solvers/)):

```sh
cargo add basin --features nalgebra
```

The moving aliases are `nalgebra`, `ndarray`, and `faer`.
Exact version features are also available for nalgebra 0.32–0.35, ndarray
0.15–0.17, and faer 0.22–0.24. When several releases of one backend are
enabled, Basin implements each independently.
The package MSRV is Rust 1.87; nalgebra 0.35 and `nalgebra` require Rust
1.89. See [backend benchmarks](../benchmarks/backends/) for how solve time
scales with different backends.

Basin enables no features by default. Enable `problems` with
`cargo add basin --features problems` to use the standard test-function corpus
in `basin::problems`.

## A simple example

Implement `CostFunction` for your objective. Add a `Gradient` impl when the
solver needs derivatives (gradient descent does; Nelder–Mead does not). Then
hand the problem, a solver, and an initial state to the `Executor`.

```rust
use basin::{CostFunction, Executor, FirstOrderState, Gradient, GradientDescent};
use std::convert::Infallible;

struct Rosenbrock;

impl CostFunction for Rosenbrock {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = Infallible;

    fn cost(&self, x: &Vec<f64>) -> Result<f64, Self::Error> {
        Ok((1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0].powi(2)).powi(2))
    }
}

impl Gradient for Rosenbrock {
    type Gradient = Vec<f64>;

    fn gradient(&self, x: &Vec<f64>) -> Result<Vec<f64>, Self::Error> {
        Ok(vec![
            -2.0 * (1.0 - x[0]) - 400.0 * x[0] * (x[1] - x[0].powi(2)),
            200.0 * (x[1] - x[0].powi(2)),
        ])
    }
}

fn main() {
    let solver = GradientDescent::new(1e-3).with_absolute_gradient_tolerance(1e-6);
    let state = FirstOrderState::new(vec![-1.2, 1.0]);

    let result = Executor::new(Rosenbrock, solver, state)
        .max_iter(50_000)
        .run()
        .unwrap();

    println!("x = {:?}", result.param());
    println!("f = {}", result.cost());
    println!("stopped: {:?}", result.report.termination);
}
```

## Seeding from a point

`Executor::new` takes a fully-built state, so you control the initial iterate.
For the common case (start at a point with the solver's natural defaults),
`Executor::from_start` takes the bare starting vector and builds the state for
you, so you never name the concrete state type:

```rust
use basin::{Executor, GradientDescent};

let solver =
    GradientDescent::new(1e-3).with_absolute_gradient_tolerance(1e-6);
let result = Executor::from_start(Rosenbrock, solver, vec![-1.2, 1.0])
    .max_iter(50_000)
    .run()
    .unwrap();
```

Most solvers support `from_start`, including `CmaEs` and `BoundedCmaEs`: the
starting point supplies the mean, and the solver configures the initial scale.
Solvers that sample a box or need an explicit population, such as PSO, GA, DE,
and random search, require an explicit state. The scalar minimizers `Brent`,
`BrentDerivative`, and `GoldenSection` also require a state because they start
from a bracket. Use `Executor::new` for those solvers.

## Convergence and run limits

Configure numerical convergence on the solver. Configure iteration, evaluation,
and time limits on the executor, along with targets, stagnation stops, and
application conditions:

```rust
use basin::{Executor, GradientDescent};

let solver = GradientDescent::new(1e-3)
    .with_absolute_gradient_tolerance(1e-6)
    .with_relative_cost_change_tolerance(None);
let result = Executor::from_start(problem, solver, x0)
    .max_iter(50_000)
    .max_cost_evals(100_000)
    .run()
    .unwrap();
```

Choose tolerances that fit the scale of your objective and parameters. Use
`None` to disable an optional check. See your solver's guide for its defaults
and how it combines enabled checks.

Check `result.report.termination` alongside the returned point to see why the
run ended. The executor checks budgets after initialization and between
iterations, so a single iteration can take the run past an evaluation limit.

## Save and resume a run

Use `Executor::resume_from_checkpoint` to continue a saved run. An exact
checkpoint holds the solver, state, and cumulative evaluation counts. For a
completed run, capture it with
`run_with_solver()?.into_checkpoint()`. For a paused run, consume the stepper
with `Stepper::into_checkpoint`.

```rust
let checkpoint = executor.max_iter(400).run_with_solver()?.into_checkpoint();
let result = Executor::resume_from_checkpoint(problem, checkpoint)
    .max_iter(1_000)
    .run()?;
```

Supply the same problem and reattach any budgets, observers, cancellation token,
and checkpoint sinks you need. Exact continuation skips solver initialization.
The iteration limit is absolute, so a checkpoint at iteration 400 leaves up to
600 more iterations in this example.

Enable `checkpoint-postcard` or `serde` to save checkpoints to files on native
targets. Attach `ExactCheckpointWriter`
with `Executor::checkpoint_with`, then load the file with
`read_exact_checkpoint`. The reader checks the file format, exact Basin
version, and concrete solver and state types. For a fresh run from a state-only
file written by `CheckpointWriter`, load the saved state with
`read_checkpoint` and use it to seed `Executor::new`.

## Next

- [Migrating from Argmin](../migrating-from-argmin/): translate an existing
  Argmin 0.11 integration.
- Use [Choosing a solver](../solvers/) to find methods that fit your problem,
  constraints, and available derivatives.
- Browse the full API reference on [docs.rs](https://docs.rs/basin).
