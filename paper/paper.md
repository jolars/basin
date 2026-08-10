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
[`Rust`](https://www.rust-lang.org) programming language. Numerical optimization
is the task of finding the inputs that minimize a function, and is a fundamental
element across the sciences: fitting a model to data, calibrating a simulation,
training a machine learning model, or choosing engineering parameters that
minimize cost. Basin gives users a single, consistent way to both state and
solve such problems, with a broad catalog of solvers and first-class support for
constraints.

To use Basin, a user implements a small trait describing their objective---at
minimum a `CostFunction` that returns a value for a given input, and optionally
its derivatives (`Gradient`, `Jacobian`, or `Hessian`)---then hands the problem,
a solver, and a starting point to a driver loop called the `Executor`. Basin
works out of the box on plain Rust vectors and, optionally, with faster
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
specialize in a single family of methods, and no widely used library couples a
broad solver catalog with first-class constraints and a browser-ready default
build. Basin was written to close that gap, and it targets four concrete needs.

First, Basin includes a broad catalog of solvers. Real problems rarely announce
in advance what optimization algorithm they need, so Basin includes a large set
of solvers behind a single, consistent API. The catalog includes

- first-order and quasi-Newton methods (gradient descent, SGD, BFGS, L-BFGS,
  L-BFGS-B, and a Newton trust-region method);
- derivative-free methods (Nelder--Mead, one-dimensional Brent and
  golden-section searches, Powell's model-based NEWUOA, BOBYQA, LINCOA, and
  COBYLA, mesh adaptive direct search);
- nonlinear least squares (Gauss--Newton, Levenberg--Marquardt, and trust-region
  reflective); and
- global or stochastic methods (random search, CMA-ES, differential evolution, a
  steady-state genetic algorithm, basin-hopping, and memetic combinations).

Switching methods is often as simple as changing a single line of code.

Second, the design enforces correctness at compile time. Solvers, termination
criteria, and observers in Basin bind on the minimum state shape they require,
which means that a method that exposes no gradient cannot be paired with a
gradient-based stopping rule---mismatch yield compilation errors rather than
runtime failures.

Third, Basin is designed to be portable. The default build targets WebAssembly
with neither BLAS/LAPACK or concurrency dependencies, so Basin can be run in a
browser without a native toolchain.

Fourth, support for constraints is first-class. Constraints are defined on the
problem-side, rather than in the solver call. And trying to use a solver that
doesn't support constraints on a constrained problem is also a compile error.

The target audience is researchers, engineers, and students who need reliable
optimization in Rust or any of the scientific programming languages that can be
easily extended through Rust, such as R, Julia, and Python.

# State of the Field

The closest analog to Basin is `argmin`\ [@kroboth2025], a numerical
optimization framework that we taken considerable inspiration from, including
the `Executor` driver loop, `Solver`/`Problem` trait split, per-solver `State`,
and generic linear algebra backend design. But Basin diverges elsewhere:

- constraints are first-class and problem-side rather than solver configuration,
- a richer linear-algebra tier implemented in pure Rust, and
- generic termination criteria that are shared between solvers.

`gomez`\ [@nevyhosteny2025] is another similar Rust crate, which implements a
small set of derivative-free methods and nonlinear least-squares solvers as well
as supports constraints. Finally, there is also the `nlopt` crate, which
implements a Rust interface to the NLopt C library\ [@johnson2026]. Although
NLopt has a broad catalog of solvers, it requires a C toolchain to build and is
not WebAssembly-compatible. It also does not support the same kind of generic
termination criteria or first-class constraints as Basin.

Basin's contribution is to bring a broad catalog natively to Rust and
WebAssembly, without linking a C or Fortran toolchain in its default
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

fn main() {
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

    let result = Executor::new(
        Rosenbrock,
        GradientDescent::new(1e-3),
        BasicState::new(vec![-1.2, 1.0]),
    )
    .max_iter(50_000)
    .terminate_on(GradientTolerance(1e-6))
    .run()
    .unwrap();

    println!(
        "x = {:?}, f = {}, stopped: {:?}",
        result.param(),
        result.cost(),
        result.reason
    );
}
```

# Software Design

Basin is organized as a generic core with a broad category of solvers layered on
top of it. The design is built on a set of principles that make it easy to
extend and maintain.

## Tiered Backends

Parameters and linear algebra are generic over the backend. A small universal
*vector tier*---operations such as scaled addition, dot products, and norms that
every backend implements well---keeps first-order and derivative-free solvers
backend-generic across `Vec<f64>`, `nalgebra`\ [@crozet2026],
`ndarray`\ [@ndarray], and `faer`\ [@faer]. A richer *linalg tier* holds matrix
operations (matrix--vector products, Cholesky and least-squares solves,
symmetric eigendecomposition), and linear-algebra-heavy solvers bind only the
minimum subset they need, so a backend that lacks an operation produces a
compile error instead of a runtime surprise. Coverage broadens only when an
operation can be added *honestly*---in pure, WebAssembly- clean Rust with no
BLAS/LAPACK stub. A pure-Rust Jacobi eigensolver, for instance, lets CMA-ES run
on the default `Vec<f64>` backend.

## One Feature per Backend, One Pinned Version

Each backend is a single Cargo feature pinning one major version; `Vec<f64>`
needs none. A backend major-version bump becomes a Basin major-version bump.
This avoids a combinatorial explosion of per-version feature gates and keeps the
test matrix and maintenance surface small.

## Compile-Time Correctness

Generic stopping conditions---iteration limits, tolerance families, evaluation
budgets, and wall-clock limits---are configured uniformly on the `Executor`
rather than reimplemented per solver, and each criterion binds on the minimum
state shape it needs. This is what makes an ill-typed pairing (a gradient
tolerance on a gradient-free method) a compile error.

## First-Class Constraints

Constraints describe the *problem*, so they live in problem-side traits, never
as executor configuration and never on the state. Solvers declare support
through traits, so a constrained problem handed to an unconstrained solver does
not compile. For the common case of reusing an unconstrained solver, opt-in
adapters (projection, a log-barrier method, and an augmented-Lagrangian method)
wrap it; the adapters consume the constraint trait and expose only
`CostFunction` and `Gradient`, which is precisely what routes a constrained
problem onto an unconstrained solver.

## Hard and External Constraints

WebAssembly support is a hard constraint on dependencies, not a feature: the
default build must compile for `wasm32-unknown-unknown`, which is verified in
continuous integration. Anything incompatible---threads, BLAS/LAPACK, native
timers---sits behind a non-default feature, and default paths use a
WebAssembly-safe time shim and a seedable, WebAssembly-safe random number
generator. Separately, the minimum supported Rust version is treated as
*externally* constrained by downstream consumers (chiefly CRAN for the planned R
bindings) and is bumped only after checking those toolchains.

## Scalar Generics

The whole pipeline is generic over the scalar type, with `f64` as the default so
existing call sites resolve unchanged, while `f32` works across states, solvers,
termination criteria, and the math layer.

# Research Impact Statement

Basin is used as the optimizer for Eunoia\ [@larsson2018]^[This package is also
made by the author.]

# AI Usage Disclosure

Generative AI tools, including AI coding agents, were used substantially during
the development of Basin. All AI-assisted contributions were reviewed by the
author.

# Acknowledgements

Basin owes a substantial intellectual debt to `argmin`\ [@kroboth2025], from
which the overall shape of the crate---the `Executor` driver loop, the
`Solver`/`Problem` trait split, and per-solver `State`---is borrowed. The
Powell-family derivative-free solvers are derived from PRIMA\ [@zhang2023],
Zaikun Zhang's modern-Fortran reference implementation of M. J. D. Powell's
methods, used as the authoritative source for the exact formulas and as a
cross-validation oracle. The bound-constrained L-BFGS-B solver is a port of the
L-BFGS-B version 3.0 Fortran code by Ciyou Zhu, Richard H. Byrd, Peihuang Lu,
and Jorge Nocedal, with the improvements by Jos&eacute; Luis Morales and Jorge
Nocedal\ [@morales2011]. Both are distributed under the BSD 3-Clause License,
and their notices are retained in the Basin source tree.

# References
