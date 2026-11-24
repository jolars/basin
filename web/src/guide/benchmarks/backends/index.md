---
title: 'Backend benchmarks – Basin'
titleTemplate: false
description: 'Choose a Basin linear-algebra backend using measured solve times for vector methods, dense matrix methods, and sparse least squares, with explanations of scaling and benchmark coverage.'
---

<script setup lang="ts">
    import BackendFigure from "#lib/benchmarks/BackendFigure.vue";
    import { BACKEND_BENCHMARKS as data } from "#lib/data/benchmarks.ts";
</script>

# Backends

Backend choice changes how Basin stores parameters and performs linear algebra.
These comparisons keep each solver and problem fixed, vary the backend, and
measure the time for a complete solve as the number of parameters $n$ grows.

The cases below progress from vector operations to dense matrices and sparse
factorizations. Read the spread between backends within each figure. Different
figures use different algorithms and problems, so their times do not compare
solver quality.

These results were measured on {{ data.generatedAt }}. Absent lines describe the
coverage of this saved run, rather than the current capabilities of a backend.
The API documentation records each solver's supported parameter and matrix
types.

## Vector methods

These methods operate on vectors without a dense Hessian or covariance matrix.
All four dense backends appear in each comparison: `Vec<f64>`, nalgebra,
ndarray, and faer.

### Gradient descent

Gradient descent on Rosenbrock provides a baseline for the cost of repeated
vector operations and evaluations. In this snapshot, `Vec<f64>` has the lowest
solve time at each measured size. With relatively cheap operations, the choice
of representation can make a visible difference to the complete solve.

<BackendFigure solver="gd" problem="rosenbrock" />

### Nelder–Mead

Nelder–Mead maintains and updates a simplex on the multimodal Ackley problem.
The measured backends remain closer together here than in the gradient-descent
case, especially at the largest size. This illustrates why the importance of
backend choice depends on the work the solver performs.

<BackendFigure solver="nm" problem="ackley" />

### L-BFGS

L-BFGS uses a short history of vector pairs to represent curvature without a
dense inverse-Hessian matrix. The Styblinski–Tang case exercises that compact
path across all four backends. `Vec<f64>` has the lowest measured time throughout
this snapshot, while the relative ordering of the other backends changes with
problem size.

<BackendFigure solver="lbfgs" problem="styblinski" />

## Dense matrix methods

Dense matrix updates and decompositions change the balance of work. The
smallest problem can favor a different backend from the largest one, so inspect
the scaling curve as well as a single timing.

### BFGS

BFGS maintains a dense inverse-Hessian approximation and updates it after each
step. On Levy, faer starts with a higher solve time but becomes the fastest of
the measured backends as the dimension grows. A backend decision based only on
the two-dimensional case would miss that crossover.

<BackendFigure solver="bfgs" problem="levy" />

This snapshot includes `Vec`, nalgebra, and faer. ndarray was not measured here;
the current [BFGS backend documentation](https://docs.rs/basin/latest/basin/solver/bfgs/struct.Bfgs.html#backends)
also lists ndarray support.

### CMA-ES

CMA-ES adds covariance adaptation and a symmetric eigendecomposition. The
Rastrigin case shows a different scaling pattern again: the `Vec` implementation
is fastest at the smallest size, while nalgebra has the lowest measured times
at the larger sizes. Supporting the same mathematical operations does not
guarantee the same cost as matrices grow.

<BackendFigure solver="cmaes" problem="rastrigin" />

Only `Vec`, nalgebra, and faer appear in this snapshot. See the
[CMA-ES backend documentation](https://docs.rs/basin/latest/basin/solver/cma_es/struct.CmaEs.html#backends)
for current support, including ndarray. The solver is stochastic; the benchmark
uses a fixed seed so the backend comparison is reproducible.

### SLSQP

SLSQP runs on unconstrained Rosenbrock with accuracy $10^{-10}$ and a
{{ data.iterations }}-iteration cap. All four backends use the same internal
least-squares kernel. Their curves are comparatively close in this snapshot,
showing a case where changing the surrounding vector and matrix representation
has a smaller effect on solve time.

<BackendFigure solver="slsqp" problem="rosenbrock" />

This is an unconstrained test of SLSQP's optimization path. It does not measure
the cost of evaluating or enforcing constraints.

## Sparse least squares

These cases use sparse Jacobians and sparse Cholesky factorizations. Only
nalgebra and faer supply the sparse matrix representations used by this
benchmark. The omitted `Vec` and ndarray lines concern this sparse setup;
both solvers also have dense implementations.

### Levenberg–Marquardt

Levenberg–Marquardt solves damped normal equations built from the sparse
Jacobian $J$. On this fixture, nalgebra has the lower measured solve time at
every size, with the relative gap narrowing as the dimension grows.

<BackendFigure solver="lm" problem="sparselsq" />

### Gauss–Newton

Gauss–Newton uses the same sparse, zero-residual design without damping. The
Jacobian has full column rank, so the normal-equation matrix $J^\mathsf{T}J$
remains positive definite for this fixture. nalgebra again has the lower
measured times in this snapshot.

<BackendFigure solver="gn" problem="sparselsq" />

This favorable least-squares design does not exercise rank deficiency or the
cases where damping helps. Use it to compare the measured sparse paths, and
choose the solver based on your problem's numerical properties.

[Benchmark methodology and reproduction](https://github.com/jolars/basin/blob/main/web/README.md#backends)
