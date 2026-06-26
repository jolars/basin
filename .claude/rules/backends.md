---
description: Detail for basin's tiered backend system (tenet 5): the shared vector tier vs the linalg tier, the trait inventory, the honest-implementability rule for adding an op to a backend, and the required per-solver "Backends" doc-comment note.
paths:
  - "crates/basin/src/core/math/**/*.rs"
---

# Backends and the math tier (tenet 5 detail)

Not every solver works on every backend today, and a few never will, but the
direction is for *most* solvers to run on *most* backends (`Vec<f64>`,
nalgebra, ndarray, faer). The tier system makes missing capability a compile
error, not a way to freeze coverage.

## Two tiers

- **Shared vector tier** (`src/core/math/`, e.g. `ScaledAdd`, `NormSquared`,
  `NormInfinity`, `Dot`, `ScaleInPlace`, `NegInPlace`, `VectorLen`, the
  component-wise ops, …) stays small and universal: only ops *every* backend
  implements well belong here. First-order and derivative-free solvers
  (gradient descent, Nelder-Mead, SA, …) bound on it and stay backend-generic.
- **`linalg` tier** (`src/core/math/linalg.rs`) holds the richer matrix ops:
  `MatVec`, `MatTransposeVec`, `GramMatrix`, `SymmetricEigen<V>`,
  `RankOneUpdate<V>` and `GeneralRankOneUpdate<V>`, `LinearSolveSpd`,
  `LinearSolveLstsq`, `MaxDiagonal`, `MatDiagonal<V>`,
  `AddDiagonalVectorInPlace<V>`, `MatrixIdentity`, `MatrixFromDiagonal<V>`,
  `DenseMatrixFromFn`, … A backend opts in by implementing them; LA-heavy
  solvers (Newton, trust-region, L-BFGS, CMA-ES, anything needing
  Cholesky, QR, or eigensolves) bound their param or matrix type on the *subset
  they actually need*, so a backend lacking an op is a compile-time error for
  that solver, not a runtime surprise. (Same spirit as tenet 3: bound on the
  minimum capability the solver needs.)

## What gates an op on a backend is honest implementability, not which backend it is

Add the impl (and broaden the solver's reach) the moment a backend can do the
op *well*: pure-Rust, wasm-clean, no BLAS/LAPACK link, no fake stub. CMA-ES is
the worked precedent: its per-iteration symmetric eigendecomposition
(`SymmetricEigen`) runs on the default `Vec<f64>` backend via a hand-rolled
cyclic Jacobi solver (`src/core/math/dense_eig.rs`), so `CmaEs` and `BoundedCmaEs`
are no longer nalgebra or faer-only. By the same reasoning, **Cholesky and
linear-solve on `Vec<f64>` or ndarray are not out of scope**: a pure-Rust
`LinearSolveSpd` and `LinearSolveLstsq` on `DenseMatrix` is welcome whenever a
solver motivates it.

What stays off a backend is only what *can't* be done honestly there: ops that
realistically need an optimized BLAS/LAPACK-class kernel (large dense
eigensolves, pivoted factorizations at scale) may never reach `Vec<f64>` or
ndarray, and that is an acceptable permanent gap: record it in the solver's
"Backends" note rather than shipping a slow or wasm-breaking impl to preserve
symmetry.

**Hard rule:** never add an op that *no* backend can implement honestly, and
never stub one out just to pass a type-check: a fake impl defeats the whole
point of the tier.

## Per-solver "Backends" note

Every solver's doc comment must include a "Backends" note listing supported
param types, mirroring the wasm compat-note pattern. This is how a permanent gap
(or current coverage) is communicated to users.
