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
date: 21 October 2026
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
solvers and first-class support for constraints. This paper describes Basin 3.0.

To use Basin, a user implements one or more small traits describing their
objective: `CostFunction` for scalar costs or `Residual` for least-squares
problems, together with derivatives required by the chosen solver (`Gradient`,
`Jacobian`, `Hessian`, or `HessianProduct`). Finite-difference adapters can
provide derivatives when analytic ones are unavailable. The user then hands the
problem, a solver, and an initial state to an `Executor`, which drives the
optimization loop, checks solver convergence and execution limits, and returns
the result. Basin works out of the box on plain Rust vectors, with optional
linear-algebra backends available behind feature flags. The default build
compiles to WebAssembly, so the compiled module can run in a browser without
BLAS/LAPACK or other native libraries. Documentation is published at
[basin.rs](https://basin.rs), which includes a user guide, interactive
visualizer, and a benchmark suite comparing Basin to other optimization
libraries.

# Statement of Need

Rust is increasingly used for scientific and numerical computing because it
combines performance with memory safety and a strong package ecosystem. Users of
optimization libraries may need to combine local and global methods,
constraints, different linear-algebra backends, and deployment in a browser.
Basin brings these capabilities together and targets four concrete needs.

First, Basin includes a broad catalog of solvers. Real problems rarely announce
in advance what optimization algorithm they need, so Basin includes a large set
of solvers behind a single, consistent API. The catalog includes

- first-order, quasi-Newton, and Newton methods (gradient descent, projected
  gradient descent, SGD, nonlinear conjugate gradient\ [@hager2005], BFGS,
  L-BFGS, L-BFGS-B, and trust-region Newton methods, including a matrix-free
  mode)\ [@nocedal2006; @byrd1995; @zhu1997];
- derivative-free methods (Nelder--Mead\ [@nelder1965], one-dimensional
  Brent\ [@brent2013] and golden-section searches, NEWUOA\ [@powell2006],
  BOBYQA\ [@powell2009], LINCOA\ [@powell2015], COBYLA\ [@powell1994], and mesh
  adaptive direct search\ [@audet2006]), together with Solis--Wets random local
  search\ [@solis1981];
- nonlinear least squares (Gauss--Newton, Levenberg--Marquardt with
  normal-equations and pivoted-QR variants, bounded Levenberg--Marquardt, and
  trust-region-reflective methods)\ [@more1978; @nielsen1999; @branch1999], with
  an optional robust-loss adapter;
- global and stochastic methods (DIRECT\ [@jones1993], globalized bounded
  Nelder--Mead\ [@luersen2004], simulated annealing\ [@kirkpatrick1983], random
  search\ [@brooks1958], CMA-ES\ [@hansen2016], differential
  evolution\ [@storn1997], global-best particle swarm\ [@shi1998], a
  steady-state genetic algorithm\ [@molina2010], and
  basin-hopping\ [@wales1997]);
- general constrained optimization through sequential least-squares programming
  (SLSQP)\ [@kraft1988], plus log-barrier and augmented-Lagrangian outer
  methods; and
- memetic combinations (MA-LS-Chain\ [@molina2010], plus CMA-ES and differential
  evolution injection wrappers).

Switching methods is simple, sometimes requiring changing only a single line of
code. Basin also provides scalar root finding through Brent, safeguarded secant,
Newton, Halley, and TOMS 748\ [@alefeld1995] methods, as well as automatic root
and minimum bracketing. These use direct callback APIs rather than the
optimization executor.

Second, the design checks interface compatibility at compile time. Solvers,
convergence checks, execution controls, and observers bind on the state and
backend capabilities they require. A solver that exposes no gradient, for
example, cannot be configured with a gradient-based convergence test.

Third, Basin is designed to be portable. The default build targets WebAssembly
without BLAS/LAPACK or threads. Its package minimum supported Rust version
(MSRV) is 1.87.0, with a newer compiler required by one optional backend
release, as discussed below. This supports integration into R packages and other
scientific programming languages that can be extended through Rust.

Fourth, support for constraints is first-class. Constraints are declared on the
problem through an optional typed declaration. Constrained solvers require a
declaration containing only blocks they support; incompatible combinations fail
to compile. Basin supports box bounds and both linear and nonlinear equality and
inequality constraints. Unconstrained solvers ignore declarations, so users must
select a compatible constrained solver or an explicit problem adapter to enforce
them.

The target audience is researchers, engineers, and students who need reliable
optimization in Rust or any of the scientific programming languages that can be
easily extended through Rust, such as R, Julia, and Python.

# State of the Field

The closest analog to Basin is argmin\ [@kroboth2025]: a numerical optimization
framework from which Basin takes considerable inspiration, including the
`Executor` driver loop, the `Solver`/`Problem` trait split, and the `State`
abstraction. But Basin diverges elsewhere, bringing

- first-class, problem-side constraints rather than solver configuration,
- a capability-based linear-algebra tier with pure-Rust dense implementations
  across four backends, and
- solver-owned numerical convergence settings with shared calculations,
  separated from executor-owned execution limits.

gomez\ [@nevyhosteny2025] is another Rust framework for optimization and
nonlinear systems of equations. It offers trust-region, LIPO, Steffensen, and
Nelder--Mead methods, using finite differences when derivatives are needed. Its
documented domains are unconstrained or box-bounded, and its linear algebra uses
nalgebra rather than a pluggable backend tier.

Finally, the nlopt crate provides a Rust interface to the NLopt C
library\ [@johnson2026]. NLopt has a broad solver catalog, supports box bounds
and nonlinear equality and inequality constraints, and provides shared stopping
tolerances and execution budgets. Its Rust interface links to the compiled C
library. Basin instead implements its solvers in Rust and expresses solver,
state, backend, and constraint requirements through Rust traits.

Basin's contribution is to combine a broad catalog with typed problem-side
constraints and multiple backends in a library whose default configuration
requires no C or Fortran libraries and compiles to WebAssembly.

# Example

In the following example, we implement the Rosenbrock function and its gradient,
then minimize it with gradient descent. The `Executor` driver loop handles the
iteration and error handling. The gradient tolerance belongs to the solver; the
iteration budget belongs to the executor.

```rust
use basin::{
    CostFunction, Executor, FirstOrderState, Gradient, GradientDescent,
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
        GradientDescent::new(1e-3).with_absolute_gradient_tolerance(1e-6),
        FirstOrderState::new(vec![-1.2, 1.0]),
    )
    .max_iter(50_000)
    .run()
    .unwrap();

    println!("x = {:?}, f = {}", result.param(), result.cost());
}
```

# Software Design

Basin is organized as a generic core with a broad catalog of solvers layered on
top of it. The design is built on a set of principles that we think make it easy
to extend and maintain.

Shared progress states record evaluated points, gradients, simplexes, or
populations. Algorithm-specific models, search histories, random number
generators, and scratch buffers belong to the solver. Resuming an optimization
exactly therefore requires both the solver and its state; initializing from a
previous point starts a fresh solve.

## Tiered Backends

Parameters and linear algebra are generic over the backend. A universal *vector
tier* (operations such as scaled addition, dot products, and norms that every
backend implements well) keeps first-order and derivative-free solvers
backend-generic across `Vec`, nalgebra\ [@crozet2026], ndarray\ [@sverdrup2026],
and faer\ [@sarrazin2026]. A richer linear-algebra tier supplies capabilities
such as matrix--vector products, factorizations, linear solves, and
eigendecompositions. All vector-based solvers support these four dense backends
with both `f32` and `f64`; custom inner solvers and strategies can impose
additional requirements.

External backends have version-specific Cargo features, such as
`nalgebra_v0_34`. The unversioned aliases `nalgebra`, `ndarray`, and `faer`
select the newest supported releases, currently 0.35, 0.17, and 0.24,
respectively. Multiple releases can be enabled together, and Basin implements
its shared math traits independently for each. Adding support for a new backend
release therefore does not require replacing the existing implementations or
bumping Basin's major version.

## Compile-Time Correctness

Numerical convergence settings belong to the solver. Shared internal
calculations implement optional absolute and relative gradient, step, and
cost-change tests, and each check requires only the state and backend
capabilities it uses. Unsupported settings, such as a gradient tolerance on
Nelder--Mead, are absent from the solver's API. `None` disables an optional
check, while zero requests an exact-zero threshold. Distinct enabled checks
generally combine with OR; Nelder--Mead's native simplex-size and cost-spread
tests form an AND group.

The `Executor` separately owns iteration, evaluation, and time budgets, target
values, stagnation stops, cancellation, and application hooks. Evaluation
budgets are checked after initialization and between iterations, so an iteration
can exceed the remaining budget. Termination reports distinguish numerical
convergence from execution limits and other stops.

## Constraints

Constraints describe the *problem*. Objective, residual, and constraint
evaluation traits stand alone; a problem optionally implements `Constraints`
with an associated typed `ConstraintSet`. The `Constrained` adapter can also
attach a complete set to an existing objective. The set distinguishes box
bounds, linear equalities and inequalities, and nonlinear equalities and
inequalities. Constrained solvers require the full declaration and reject
unsupported blocks at compile time, while runtime checks validate dimensions and
numerical data. Unconstrained solvers optimize the objective without inspecting
or enforcing the declaration.

Explicit adapters allow users to reuse unconstrained solvers for particular
constraint classes. `LogBarrier` transforms linear inequalities, and
`AugmentedLagrangian` transforms linear equalities. Both expose the transformed
cost and gradient with an empty constraint declaration and reject other blocks.
`BarrierMethod` and `AugmentedLagrangianMethod` automate the corresponding outer
iterations. General combinations of box, linear, and nonlinear constraints are
handled directly by SLSQP or COBYLA, subject to each method's derivative and
feasibility conventions.

## Compatibility

Basin is WebAssembly-compatible by default. Parallelism and BLAS/LAPACK
integration are opt-in features, and default paths use a WebAssembly-safe time
shim and a seedable, WebAssembly-safe random number generator.

Basin's package MSRV is Rust 1.87.0. The optional nalgebra 0.35 backend,
including the moving `nalgebra` feature alias, requires Rust 1.89; earlier
supported nalgebra releases remain available through version-specific features
on Rust 1.87. Maintaining this older toolchain support helps downstream users
integrate Basin into scientific software, including R packages such as
[eulerr](https://cran.r-project.org/package=eulerr).

## Scalar Generics

The interface is generic over the scalar type, with `f64` as the default. Both
`f32` and `f64` work across states, solvers, termination reports, and the math
layer.

# Research Impact Statement

Basin is used as the optimizer for the Rust library Eunoia\ [@larsson2026a],
which in turn is used in the R package eulerr\ [@larsson2018].^[These packages
are also made by the author.] It is also used in the R package
balancing\ [@barrett2026], which calculates optimization-based balancing weights
for causal inference.

Benchmarks against competitors are available at
[basin.rs](https://basin.rs/benchmarks/), comparing Basin with argmin, gomez,
nlopt, and other implementations on matched test problems. The published
convergence traces report solution quality against elapsed time. Performance
depends on the problem, solver configuration, stopping rules, and backend; these
comparisons do not establish a universal ranking of the libraries.

As of October 21, 2026, <https://crates.io/crates/basin> reports approximately
85,500 downloads over the preceding 90 days. Basin was also featured in issue
656 of *This Week in Rust*, published on June 17, 2026\ [@arlynx2026].

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
License, and their notices are retained in the Basin source tree. The SLSQP
implementation also incorporates least-squares and factor-update routines from
Jacob Williams's modern Fortran SLSQP implementation, with its notices retained
in the source tree.

# References
