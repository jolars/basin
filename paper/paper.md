---
title: "Basin: Efficient and Extensible Numerical Optimization in Rust"
tags:
  - Rust
  - numerical optimization
  - nonlinear optimization
  - least squares
  - derivative-free optimization
  - WebAssembly
  - scientific computing
authors:
  - name: Johan Larsson
    orcid: 0000-0002-4029-5945
    affiliation: "1"
affiliations:
  - name: Department of Mathematical Sciences, University of Copenhagen, Denmark
    index: 1
    ror: 035b05819
date: 11 July 2026
bibliography: paper.bib
---

# Summary

Basin is a numerical optimization library for the
[Rust](https://www.rust-lang.org) programming language\ [@matsakis2014].
Numerical optimization is the task of finding the inputs that minimize a
function, and it is a fundamental element across the sciences: fitting a model
to data, calibrating a simulation, training a machine learning model, or
choosing engineering parameters that minimize cost. Basin gives users a single,
consistent way to both state and solve such problems, with a broad catalog of
solvers and first-class support for constraints.

To use Basin, a user implements one or more small traits describing their
objective---at minimum a `CostFunction` that returns a value for a given input,
and optionally its derivatives (`Gradient`, `Jacobian`, or `Hessian`). The user
then hands the problem, a solver, and a starting point to an `Executor`, which
drives the optimization loop, handles stopping criteria, and returns the result.
Basin works out of the box on plain Rust vectors and, optionally, with faster
linear-algebra backends available behind feature flags. The default build
compiles to WebAssembly, which means that Basin can be used in a browser without
a native toolchain or BLAS/LAPACK support. Documentation is published at
[basin.rs](https://basin.rs), which includes a user guide, interactive
visualizer, and a benchmark suite comparing Basin to other optimization
libraries.

# Statement of Need

Rust is increasingly used for scientific and numerical computing because it
combines performance with memory safety and a strong package ecosystem.
Optimization, however, is fragmented across the ecosystem: most crates
specialize in a single family of methods and no widely used Rust crate couples a
broad solver catalog with first-class constraints and a browser-ready default
build. Basin was written to close that gap, and it targets four concrete needs.

First, Basin includes a broad catalog of solvers. Real problems rarely announce
in advance what optimization algorithm they need, so Basin includes a large set
of solvers behind a single, consistent API. The catalog includes

- first-order and quasi-Newton methods (gradient descent, SGD, BFGS, L-BFGS,
  L-BFGS-B, and a Newton trust-region
  method)\ [@nocedal2006; @byrd1995; @zhu1997];
- derivative-free methods (Nelder--Mead\ [@nelder1965], one-dimensional
  Brent\ [@brent2013] and golden-section searches, NEWUOA\ [@powell2006],
  BOBYQA\ [@powell2009], LINCOA\ [@powell2015], COBYLA\ [@powell1994], and mesh
  adaptive direct search\ [@audet2006]);
- nonlinear least squares (Gauss--Newton and
  Levenberg--Marquardt\ [@nielsen1999]);
- global and stochastic methods (random search\ [@brooks1958],
  CMA-ES\ [@hansen2016], differential evolution\ [@storn1997], a steady-state
  genetic algorithm\ [@molina2010], and basin-hopping\ [@wales1997]); and
- memetic combinations (MA-LS-Chain\ [@molina2010], plus CMA-ES and differential
  evolution injection wrappers).

Switching methods is simple, sometimes requiring changing only a single line of
code.

Second, the design enforces correctness at compile time. Solvers, termination
criteria, and observers in Basin bind on the minimum state shape they require,
which means that a method that exposes no gradient cannot be paired with a
gradient-based stopping rule---mismatches yield compilation errors rather than
runtime failures.

Third, Basin is designed to be portable. The default build targets WebAssembly
with neither BLAS/LAPACK nor concurrency dependencies, which means that Basin
can be run in a browser without a native toolchain. It also supports a low
minimum supported Rust version in order to facilitate its use in R packages and
other scientific programming languages that can be extended through Rust.

Fourth, support for constraints is first-class. Constraints are declared on the
problem, not passed to the solver call, and a solver that requires constraints
will not accept an unconstrained problem---again a compile error rather than a
runtime one. Basin supports box bounds, linear equality and inequality
constraints, and nonlinear inequality constraints, together with opt-in adapters
(log-barrier and augmented Lagrangian) that recast a constrained problem as an
unconstrained one so that any unconstrained solver can be applied to it.

The target audience is researchers, engineers, and students who need reliable
optimization in Rust or any of the scientific programming languages that can be
easily extended through Rust, such as R, Julia, and Python.

# State of the Field

The closest analog to Basin is argmin\ [@kroboth2025]: a numerical optimization
framework from which Basin takes considerable inspiration, including the
`Executor` driver loop, the `Solver`/`Problem` trait split, and per-solver
`State`. But Basin diverges elsewhere, bringing

- first-class, problem-side constraints rather than solver configuration,
- a richer linear-algebra tier implemented in pure Rust, and
- generic termination criteria shared between solvers.

gomez\ [@nevyhosteny2025] is another Rust crate with similar scope, implementing
a small set of derivative-free methods and nonlinear least-squares solvers, and
supporting constraints. Compared to Basin, it has a smaller solver catalog, only
supports box constraints, and does not have a generic backend tier for linear
algebra.

Finally, the nlopt crate provides a Rust interface to the NLopt C
library\ [@johnson2026]. Although NLopt has a broad catalog of solvers, it
requires a C toolchain to build and is not WebAssembly-compatible. It also lacks
Basin's generic termination criteria and first-class constraints support.

In summary, Basin's contribution is to bring a broad catalog natively to Rust
and WebAssembly, without linking a C or Fortran toolchain in its default
configuration.

# Example

In the following example, we implement the Rosenbrock function and its gradient,
then minimize it with gradient descent. The `Executor` driver loop handles the
iteration, stopping criteria, and error handling.

```rust
use basin::{
    BasicState, CostFunction, Executor, Gradient, GradientDescent,
    GradientTolerance,
};
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
    let result = Executor::new(
        Rosenbrock,
        GradientDescent::new(1e-3),
        BasicState::new(vec![-1.2, 1.0]),
    )
    .max_iter(50_000)
    .terminate_on(GradientTolerance(1e-6))
    .run()
    .unwrap();
}
```

# Software Design

Basin is organized as a generic core with a broad catalog of solvers layered on
top of it. The design is built on a set of principles that we think make it easy
to extend and maintain.

## Tiered Backends

Parameters and linear algebra are generic over the backend. A universal *vector
tier* (operations such as scaled addition, dot products, and norms that every
backend implements well) keeps first-order and derivative-free solvers
backend-generic across `Vec<f64>`, nalgebra\ [@crozet2026],
ndarray\ [@sverdrup2026], and faer\ [@sarrazin2026]. Each backend is activated
via a single Cargo feature pinning one major version and a backend major-version
bump becomes a Basin major-version bump. This differs from argmin, which uses
versioned backend traits and requires a new trait for each backend version. We
opted to keep the backend traits versionless and instead version the entire
crate in order to improve maintainability.

## Compile-Time Correctness

Generic stopping conditions (iteration limits, tolerance families, evaluation
budgets, and wall-clock limits) are configured uniformly on the `Executor`
rather than reimplemented per solver, and each criterion binds on the minimum
state shape it needs. This is what makes an ill-typed pairing (a gradient
tolerance on a gradient-free method) a compile error. The cost of this is more
complex generic signatures, but the benefit is that Basin users can be confident
that their stopping criteria are compatible with their solver and problem.

## Constraints

Constraints describe the *problem*, so in Basin they exist as problem-side
traits rather than in executor configuration or on the state. Solvers declare
the constraints they consume through those traits, which means that an
unconstrained problem handed to a solver that requires constraints does not
compile. For the common case of reusing an unconstrained solver, opt-in adapters
(a log-barrier method and an augmented-Lagrangian method) wrap the *problem*;
each adapter consumes the constraint trait and exposes only `CostFunction` and
`Gradient`, which is precisely what routes a constrained problem onto an
unconstrained solver.

## Compatiblitiy

Basin is WebAssembly-compatible by default. Parallelism and BLAS/LAPACK
integration are opt-in features, and default paths use a WebAssembly-safe time
shim and a seedable, WebAssembly-safe random number generator.

The minimum supported Rust version is kept deliberately low in order to comply
with the toolchain requirements of the R package network
[CRAN](https://cran.r-project.org) in order to facilitate Basin's use in R
packages such as [eulerr](https://cran.r-project.org/package=eulerr).

## Scalar Generics

The interface is generic over the scalar type, with `f64` as the default so
existing call sites resolve unchanged, while `f32` works across states, solvers,
termination criteria, and the math layer.

# Research Impact Statement

Basin is used as the optimizer for the Rust library Eunoia\ [@larsson2026a],
which in turn is used in the R package eulerr\ [@larsson2018].^[These packages
are also made by the author.] It is also used in the R package
balancing\ [@barrett2026], which calculates optimization-based balancing weights
for causal inference.

Benchmarks against competitors are available at
[basin.rs](https://basin.rs/benchmarks), showing that Basin generally
outperforms argmin and nlopt and is on par with gomez.

At the time of writing, the crate has been downloaded roughly 30,000 times on
<https://crates.io/crates/basin> over the last three months and has been
featured in *This Week in Rust*\ [@arlynx2026].

# AI Usage Disclosure

Generative AI tools were used substantially during the development of Basin:
Claude Code, running Claude Opus 4.8, Claude Opus 5, and Fable 5, was used for
code generation and refactoring, writing unit tests, writing documentation, and
reviewing this manuscript. The author made all core design decisions---the
architecture, the design tenets, and the API---and reviewed, edited, and
validated all AI-assisted contributions. In order to further verify correctness,
the Powell-family solvers were developed against PRIMA\ [@zhang2023] and
cross-validated against it numerically, and the L-BFGS-B implementation was
checked for numerical agreement with the original Fortran code\ [@zhu1997]. The
remaining solvers are covered by a test suite problems.

# Acknowledgements

As we have mentioned, Basin owes a substantial intellectual debt to
`argmin`\ [@kroboth2025]. The Powell-family derivative-free solvers are derived
from PRIMA\ [@zhang2023], Zaikun Zhang's modern-Fortran reference implementation
of M. J. D. Powell's methods, used as the authoritative source for the exact
formulas and as a cross-validation oracle. The bound-constrained L-BFGS-B solver
is a port of the L-BFGS-B version 3.0 Fortran code by Ciyou Zhu, Richard H.
Byrd, Peihuang Lu, and Jorge Nocedal, with the improvements by José Luis Morales
and Jorge Nocedal\ [@morales2011]. Both are distributed under the BSD 3-Clause
License, and their notices are retained in the Basin source tree.

# References
