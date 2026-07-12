---
title: "Basin: Extensible Numerical Optimization in Rust"
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
is the task of finding the inputs that make a function as small as possible, and
is a fundamental element across the sciences: fitting a model to data,
calibrating a simulation, training a machine learning model, or choosing
engineering parameters that minimize cost. Basin gives users a single,
consistent way to state such a problem and hand it to any of a broad catalog of
solution methods.

To use Basin, a user implements a small trait describing their objective---at
minimum a `CostFunction` that returns a value for a given input, and optionally
its derivatives (`Gradient`, `Jacobian`, or `Hessian`) when a method needs
them---then hands the problem, a solver, and a starting point to a driver loop
called the `Executor`. The same problem can be solved by many different
algorithms without rewriting it. Basin works out of the box on plain Rust
vectors (`Vec<f64>`), with optional, faster linear-algebra backends available
behind feature flags. The default build compiles to WebAssembly, so the same
code that runs on a server also runs in a web browser. Narrative documentation,
an in-browser solver visualizer, and reproducible cross-library benchmarks are
published at [basin.rs](https://basin.rs), and the programming reference is at
[docs.rs/basin](https://docs.rs/basin).

# Statement of Need

Rust is increasingly used for scientific and numerical computing because it
combines performance with memory safety and a strong package ecosystem.
Optimization, however, is fragmented across the ecosystem: most crates
specialize in a single family of methods, and no widely used library couples a
broad solver catalog with first-class constraints and a browser-ready default
build. Basin was written to close that gap, and its design targets four concrete
needs.

First, *breadth under one interface*. Real problems rarely announce in advance
which method will work, so Basin ships many families behind one problem
definition: first-order and quasi-Newton methods (gradient descent, SGD, BFGS,
L-BFGS, L-BFGS-B, and a Newton trust-region method); derivative-free methods
(Nelder--Mead; the one-dimensional Brent and golden-section searches; Powell's
model-based NEWUOA, BOBYQA, LINCOA, and COBYLA; and mesh adaptive direct
search); nonlinear least squares (Gauss--Newton, Levenberg--Marquardt, and
trust-region reflective); and global or stochastic methods (random search,
CMA-ES, differential evolution, a steady-state genetic algorithm, basin-hopping,
and memetic combinations). Switching methods is a one-line change.

Second, *safety at compile time*. Termination criteria and observers in Basin
bind on the minimum state shape they require, so a method that exposes no
gradient cannot be paired with a gradient-based stopping rule---such a mismatch
is a compilation error rather than a silent no-op or a runtime failure.
Likewise, handing a constrained problem to a solver that does not support
constraints does not compile.

Third, *portability*. The default build targets WebAssembly with neither
BLAS/LAPACK or concurrency dependencies, so Basin runs in the browser without a
native toolchain. This powers an in-browser solver visualizer that is useful for
teaching and exploration, and it keeps Basin embeddable in WebAssembly-based
scientific applications where BLAS-linked stacks cannot go.

Fourth, *reach into other research ecosystems*. Basin holds a deliberately
conservative minimum supported Rust version, chosen so that planned R (CRAN) and
Python bindings remain buildable under the toolchains those ecosystems pin. The
long-term goal is to make the same solver catalog available to researchers who
work primarily in R or Python.

The target audience is researchers, engineers, and students who need reliable
optimization in Rust today, and---through the planned bindings---the wider
scientific R and Python communities.

# State of the Field

Within Rust, the closest analog is `argmin`\ [@kroboth2025], a numerical
optimization framework that Basin openly takes as inspiration: the overall shape
of the crate---an `Executor` driver loop, the `Solver`/`Problem` trait split,
and per-solver `State`---follows argmin's conventions so that users familiar
with it feel at home. Basin diverges deliberately where it improves the design:
constraints are first-class and problem-side rather than solver configuration;
backends are tiered so that a missing linear-algebra operation is a compile-time
error; termination criteria are bound to the state shape a solver actually
exposes; and the entire numerical pipeline is generic over the scalar type, so
`f32` and `f64` both work end to end. These are the reasons Basin is a new
library rather than a set of patches to argmin. Other Rust crates are narrower:
`gomez`\ [@nevyhosteny2025] targets systems of nonlinear equations and
derivative-free optimization, and `levenberg-marquardt`\ [@schurg2026]
implements a single nonlinear-least-squares method.

Outside Rust, mature multi-method suites exist---most prominently
NLopt\ [@johnson2026], a C library with bindings for many languages, and SciPy's
`optimize` module\ [@virtanen2020] in Python. Basin's contribution is to bring a
comparably broad catalog natively to Rust and WebAssembly, without linking a C
or Fortran toolchain in its default configuration. To keep this claim honest and
current, Basin ships a reproducible benchmark harness that compares it against
argmin, `gomez`, `levenberg-marquardt`, and NLopt; results are published and
kept up to date on the [benchmarks page](https://basin.rs/benchmarks/) rather
than frozen into this paper, where they would quickly go stale.

# Software Design

Basin is organized as a small generic core with a growing set of solvers built
on top. A driver loop, the `Executor`, iterates a `Solver` over a `State`,
calling into the user-implemented `Problem` traits until a
`TerminationCriterion` fires. This uses established optimization-framework
vocabulary intentionally, to lower the barrier for users arriving from other
libraries. Several design decisions shape the API and are worth making explicit.

*Tiered, broadening backends.* Parameters and linear algebra are generic over
the backend. A small universal *vector tier*---operations such as scaled
addition, dot products, and norms that every backend implements well---keeps
first-order and derivative-free solvers backend-generic across `Vec<f64>`,
`nalgebra`\ [@crozet2026], `ndarray`\ [@ndarray], and `faer`\ [@faer]. A richer
*linalg tier* holds matrix operations (matrix--vector products, Cholesky and
least-squares solves, symmetric eigendecomposition), and linear-algebra-heavy
solvers bind only the minimum subset they need, so a backend that lacks an
operation produces a compile error instead of a runtime surprise. Coverage
broadens only when an operation can be added *honestly*---in pure, WebAssembly-
clean Rust with no BLAS/LAPACK stub. A pure-Rust Jacobi eigensolver, for
instance, lets CMA-ES run on the default `Vec<f64>` backend.

*One feature per backend, one pinned version.* Each backend is a single Cargo
feature pinning one major version; `Vec<f64>` needs none. A backend
major-version bump becomes a Basin major-version bump. This avoids a
combinatorial explosion of per-version feature gates and keeps the test matrix
and maintenance surface small.

*Framework-level termination.* Generic stopping conditions---iteration limits,
tolerance families, evaluation budgets, and wall-clock limits---are configured
uniformly on the `Executor` rather than reimplemented per solver, and each
criterion binds on the minimum state shape it needs. This is what makes an
ill-typed pairing (a gradient tolerance on a gradient-free method) a compile
error.

*First-class constraints.* Constraints describe the *problem*, so they live in
problem-side traits, never as executor configuration and never on the state.
Solvers declare support through traits, so a constrained problem handed to an
unconstrained solver does not compile. For the common case of reusing an
unconstrained solver, opt-in adapters (projection, a log-barrier method, and an
augmented-Lagrangian method) wrap it; the adapters consume the constraint trait
and expose only `CostFunction` and `Gradient`, which is precisely what routes a
constrained problem onto an unconstrained solver.

*Hard and external constraints.* WebAssembly support is a hard constraint on
dependencies, not a feature: the default build must compile for
`wasm32-unknown-unknown`, which is verified in continuous integration. Anything
incompatible---threads, BLAS/LAPACK, native timers---sits behind a non-default
feature, and default paths use a WebAssembly-safe time shim and a seedable,
WebAssembly-safe random number generator. Separately, the minimum supported Rust
version is treated as *externally* constrained by downstream consumers (chiefly
CRAN for the planned R bindings) and is bumped only after checking those
toolchains.

*Scalar generics.* The whole pipeline is generic over the scalar type, with
`f64` as the default so existing call sites resolve unchanged, while `f32` works
across states, solvers, termination criteria, and the math layer.

# Research Impact Statement

Basin is used as the optimizer for Eunoia^[This package is also made by the
author.]

# AI Usage Disclosure

Generative AI tools, including AI coding agents, were used substantially during
the development of Basin. All AI-assisted contributions were reviewed by the
author, and correctness was verified through the library's automated test suite
and, for the more complex solvers, cross-validation against authoritative
reference implementations.

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
Nocedal\ [@morales2011; @morales2011lbfgsb]. Both are distributed under the BSD
3-Clause License, and their notices are retained in the Basin source tree.

# References
