---
description: Detail for basin's first-class constraints (tenet 4) — the three shipped constraint kinds and their feasibility mechanisms, the adapter-asymmetry rule, why there's no Constraint supertrait, and why constraints live on the problem and never on state.
paths:
  - "crates/basin/src/core/constraint.rs"
  - "crates/basin/src/core/barrier.rs"
  - "crates/basin/src/core/augmented_lagrangian.rs"
  - "crates/basin/src/solver/barrier_method.rs"
  - "crates/basin/src/solver/augmented_lagrangian_method.rs"
  - "crates/basin/src/solver/projected_gradient_descent.rs"
---

# Constraints (tenet 4 detail)

Constraints describe the *problem*, so they live problem-side. Solvers declare
support via traits; a constrained problem handed to an unconstrained solver is
a compile error, with opt-in adapters to wrap unconstrained solvers. This file
holds the detail behind that tenet.

## The three shipped kinds (all in `src/core/constraint.rs`)

Each keeps feasibility by a *different* mechanism — that is why they stay
sibling traits (see below):

- **`BoxConstraints`** (interval bounds) — kept feasible by *projection* /
  clamping (`ClampInPlace`). Used by `Brent` (1D), `ProjectedGradientDescent`,
  `Lbfgsb`, `Trf`, `BoundedCmaEs`.
- **`LinearInequalityConstraints`** (`A x ≤ b`, exposing `a()` / `b()`) — kept
  feasible by a *barrier*, no projection. Used by the log-barrier
  `BarrierMethod` (`src/solver/barrier_method.rs`) via the `LogBarrier` adapter
  (`src/core/barrier.rs`). `BarrierMethod` is a `constrOptim`-style continuation
  loop over **any gradient inner solver** (bound `So: WarmStart<V>` with
  `So::State: GradientState`: `GradientDescent`, `Bfgs`, or unbounded `Lbfgs`;
  seeded at the current iterate via `WarmStart::seed`). v1 requires a strictly
  feasible start (phase 1 deferred) and an Armijo-backtracking inner line search
  — the barrier's `+∞` wall is the only feasibility guard, so a
  Wolfe/More-Thuente inner can step through it; pair `Bfgs`/`Lbfgs` with
  `Backtracking` for the barrier.
- **`LinearEqualityConstraints`** (`A x = b`, exposing `a()` / `b()` — same
  *shape* as the inequality trait but a distinct *type*, so `≤` and `=` can't be
  confused) — kept feasible by a *quadratic penalty plus multiplier updates*
  (`L_ρ = f + λᵀc + (ρ/2)‖c‖²`, `c = A x − b`), no projection, no barrier. Used
  by `AugmentedLagrangianMethod` (`src/solver/augmented_lagrangian_method.rs`)
  via the `AugmentedLagrangian` adapter (`src/core/augmented_lagrangian.rs`).
  The outer loop minimizes `L_ρ` with any gradient inner (same `So:
  WarmStart<V>` + `So::State: GradientState` bound as the barrier), then updates
  `λ ← λ + ρ c` (or raises `ρ` when feasibility stalls). Unlike the barrier,
  `L_ρ` is finite everywhere, so it tolerates an **infeasible start** and any
  inner line search (no `+∞` wall, no phase 1). Convergence (`‖A x − b‖ ≤ tol`)
  lives in the solver's `terminate` hook, mirroring the barrier's gap test
  (tenet 3: a framework-level `FeasibilityTolerance` waits for a 2nd equality
  solver).

Both linear families run on **every backend**: they need only `MatVec` +
`MatTransposeVec` (never a solve), shipped for `Vec<f64>` (via the hand-rolled
`DenseMatrix` in `src/core/math/dense.rs`), nalgebra, faer, and ndarray.

Nonlinear equality and nonlinear (in)equality constraint kinds are not yet
designed.

## Adapters must not re-implement the constraint trait they consumed

A wrapper that converts a constrained problem into an unconstrained one (log
barrier, quadratic penalty) exposes `CostFunction + Gradient` **only**.
`LogBarrier<'a, P: LinearInequalityConstraints>` and `AugmentedLagrangian<'a,
P: LinearEqualityConstraints>` both impl `CostFunction + Gradient` and pointedly
do **not** impl the constraint trait they consumed — that asymmetry is what
flows the wrapped problem to unconstrained solvers. If a wrapper also
implemented the constraint trait, it would route back into constrained solvers
and the whole adapter model collapses. (Contrast `FiniteDiff`, which *adds* a
capability and therefore *forwards* `BoxConstraints`.) Load-bearing and
non-obvious; preserve it deliberately.

## No `Constraint` supertrait until ≥2 fundamentally different solvers share more than `a()`/`b()`

Three constraint kinds have landed and *keep confirming* the wait rather than
ending it. Each keeps feasibility by a different mechanism: box by *projection*
(`ClampInPlace`), linear-inequality by a *barrier* (`MatVec`/`MatTransposeVec`),
linear-equality by a *penalty plus multipliers* (also `MatVec`/`MatTransposeVec`,
but to assemble `∇L_ρ`, not a barrier). The two linear families share the same
*carrier ops* but no *feasibility* operation, and their data accessors
(`a()`/`b()`) are the only common surface — so `BoxConstraints`,
`LinearInequalityConstraints`, and `LinearEqualityConstraints` stay sibling
traits with no supertrait. A shared abstraction waits for a kind (nonlinear)
that genuinely shares a feasibility-check or projection op. One-member (or
no-shared-op multi-member) hierarchies are overhead with no value; designing on
paper without a solver to validate against tends to need redoing.

## Constraints live on the problem, never on state

Don't put `lower` / `upper` on `BasicState` "for convenience". State carries
iteration history; constraints define the problem. Bounds on state would
silently un-constrain a problem if a different state were swapped in, and
decouple constraint semantics from where the solver type system enforces them.
Termination criteria that need bounds (e.g. `ProjectedGradientTolerance`) clone
them at construction — that's the deliberate pattern, not a workaround.
