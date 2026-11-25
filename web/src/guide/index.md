# Overview

**Basin** is a numerical optimization library for Rust. Solve problems with or
without derivatives, fit models with nonlinear least squares, and handle
bounds and other constraints with algorithms suited to your problem. Basin
runs natively and in the browser. Use plain Rust vectors or your choice of
nalgebra, ndarray, or faer.

To minimize a function, implement `CostFunction`, choose a solver, and supply
a starting point. Add derivatives when your solver needs them. Use
[Choosing a solver](solvers/) to find methods that fit your problem.

Bracketed scalar equations use the separate `BrentRoot::solve` API. It accepts
a fallible closure and returns the signed function value and final bracket
directly, without representing root finding as objective minimization.

## A solver run at a glance

The smallest complete program: minimize the sphere function with the
derivative-free Nelder–Mead simplex. Implement
`CostFunction`, then let
`Executor::from_start` build the state from your starting point.

```rust
use basin::{CostFunction, Executor, NelderMead};
use std::convert::Infallible;

struct Sphere;

impl CostFunction for Sphere {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = Infallible;

    fn cost(&self, x: &Vec<f64>) -> Result<f64, Self::Error> {
        Ok(x.iter().map(|xi| xi * xi).sum())
    }
}

fn main() {
    let result = Executor::from_start(Sphere, NelderMead::new(), vec![1.0, 1.0])
        .max_iter(500)
        .run()
        .unwrap();

    println!("x = {:?}", result.param()); // ~[0.0, 0.0]
    println!("f = {}", result.current_record().unwrap().cost.unwrap()); // ~0.0
}
```

Nelder–Mead needs no derivatives, so `CostFunction` is the only trait to
implement. The [Getting started](getting-started/) guide shows how to add a
`Gradient` implementation and solve the Rosenbrock problem.

## How the pieces fit

- **Problem**: what you implement. A `CostFunction` (and a `Gradient`, if your
  solver needs one) describing the objective. Constraints such as box bounds
  live here too, not on the solver.
- **Solver**: the algorithm. Gradient descent, Nelder–Mead, L-BFGS,
  Levenberg–Marquardt, CMA-ES, and more.
- **State**: the iterate(s) a solver carries: a point and cost (`PointState`),
  a point with a gradient (`FirstOrderState`), a simplex (`SimplexProgress`), a population, and so on.
- **Executor**: the driver that runs a solver to termination and hands back an
  `OptimizationResult`.
- **Convergence and run control**: configure numerical tolerances on the solver;
  configure budgets, targets, stagnation stops, and custom hooks on the executor.
- **Observers**: read-only hooks fired around the loop
  (`observe_init`/`observe_iter`/`observe_final`) for logging, progress
  reporting, or recording a trajectory. Observers run alongside the execution controls.

## Choosing a solver

Use the [solver finder](solvers/) to narrow your options. The individual
solver guides cover requirements, settings, and references.

- **Smooth objective, gradient available**: `GradientDescent` to start, then
  `Bfgs` or `Lbfgs` for faster convergence (`Lbfgs` when the dimension is
  large).
- **No derivatives, low dimension**: `NelderMead`, or Powell's model-based
  `Newuoa`/`Bobyqa` for smoother functions.
- **Sum-of-squares residuals**: the nonlinear-least-squares family,
  `GaussNewton` or `LevenbergMarquardt`, with `BoundedLevenbergMarquardt`
  and `TrustRegionReflective` for box bounds.
- **Rugged or multimodal landscape**: the global and stochastic optimizers,
  `CmaEs`, the steady-state GA, or a memetic hybrid.
- **Constraints**: box bounds, linear (in)equalities, and nonlinear constraints
  describe the *problem*, so they live problem-side. Bounded solvers (`Lbfgsb`,
  bounded `NelderMead`, `BoundedLevenbergMarquardt`, `TrustRegionReflective`)
  consume them directly. `Cobyla` handles linear and nonlinear constraints
  without derivatives.
- **Scalar equation with a sign-changing interval**: `BrentRoot` combines
  bisection, secant steps, and inverse quadratic interpolation through a direct
  root-specific API.

## Citing Basin

If you use Basin in your research, please cite the paper describing it,
[arXiv:2608.11279](https://arxiv.org/abs/2608.11279):

::: code-group

```text [APA]
Larsson, J. (2026). Basin: Efficient and Extensible Numerical Optimization in Rust (arXiv:2608.11279). arXiv. https://doi.org/10.48550/arXiv.2608.11279
```

```bibtex [BibTeX]
@misc{larsson2026basin,
  title         = {Basin: Efficient and Extensible Numerical Optimization in {{Rust}}},
  shorttitle    = {Basin},
  author        = {Larsson, Johan},
  year          = {2026},
  month         = aug,
  number        = {arXiv:2608.11279},
  eprint        = {2608.11279},
  primaryclass  = {cs.LG},
  publisher     = {arXiv},
  doi           = {10.48550/arXiv.2608.11279},
  archiveprefix = {arXiv}
}
```

```bibtex [BibLaTeX]
@online{larsson2026basin,
  title       = {Basin: Efficient and Extensible Numerical Optimization in {{Rust}}},
  shorttitle  = {Basin},
  author      = {Larsson, Johan},
  date        = {2026-08-11},
  eprint      = {2608.11279},
  eprinttype  = {arXiv},
  eprintclass = {cs.LG},
  doi         = {10.48550/arXiv.2608.11279},
  pubstate    = {prepublished}
}
```

:::

The repository's
[`CITATION.cff`](https://github.com/jolars/basin/blob/main/CITATION.cff) carries
the same reference in machine-readable form.

## Where to go next

- [Getting started](getting-started/): install Basin and run your first solve.
- [Migrating from Argmin](migrating-from-argmin/): translate problem traits,
  solver configuration, errors, bounds, and observers from Argmin 0.11.
- [Solvers](solvers/): find a starting point and browse the individual guides.

You can also watch solvers converge interactively in the
[visualizer](../visualizer/), or browse the full API on
[docs.rs](https://docs.rs/basin).
