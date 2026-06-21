---
description: Detail for basin's first-class constraints (tenet 4) — the four shipped constraint kinds and their feasibility mechanisms, the adapter-asymmetry rule, why there's no Constraint supertrait, and why constraints live on the problem and never on state.
paths:
  - "crates/basin/src/core/constraint.rs"
  - "crates/basin/src/core/barrier.rs"
  - "crates/basin/src/core/augmented_lagrangian.rs"
  - "crates/basin/src/solver/barrier_method.rs"
  - "crates/basin/src/solver/augmented_lagrangian_method.rs"
  - "crates/basin/src/solver/projected_gradient_descent.rs"
  - "crates/basin/src/solver/cobyla.rs"
---

# Constraints (tenet 4 detail)

Constraints describe the *problem*, so they live problem-side. Solvers declare
support via traits; a constrained problem handed to an unconstrained solver is
a compile error, with opt-in adapters to wrap unconstrained solvers. This file
holds the detail behind that tenet.

## The four shipped kinds (all in `src/core/constraint.rs`)

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

- **`NonlinearInequalityConstraints`** (`c(x) ≤ 0` for an arbitrary
  vector-valued `c`, exposing `constraints(x)` + `num_constraints()`) — kept
  feasible by an *exact-penalty merit function with a geometry/acceptance test*
  (Φ = F + μ·[maxᵢ cᵢ]₊), no projection, no barrier, no multipliers. Used by
  the derivative-free `Cobyla` (`src/solver/cobyla.rs`). Also consumed by
  `Mads<Constrained>` (`Mads::constrained()`, `src/solver/mads.rs`), which keeps
  the *same trait* feasible by a **different mechanism** — the *progressive
  barrier* (Audet & Dennis 2009): an aggregate violation `h(x) = Σⱼ max(cⱼ, 0)²`
  and a threshold driven to zero around two incumbents. So the consumer↔mechanism
  map is many-to-many; the trait is just the data contract. Unlike the three other
  kinds the constraint is a **function evaluated at the iterate**, not
  matrix/vector data, so the trait carries an evaluator rather than `a()`/`b()`
  accessors. Sign convention is `cᵢ(x) ≤ 0` (feasible), matching the
  linear-inequality `≤` direction and PRIMA's modern COBYLA; Powell's 1994 paper
  writes `cᵢ ≥ 0` (negate to convert). Function-valued ⇒ needs only vector-tier
  ops, so it runs on **every backend** and wasm.

Both linear families run on **every backend**: they need only `MatVec` +
`MatTransposeVec` (never a solve), shipped for `Vec<f64>` (via the hand-rolled
`DenseMatrix` in `src/core/math/dense.rs`), nalgebra, faer, and ndarray.

Nonlinear *equality* constraints are not yet designed (express `g(x) = 0` as the
pair `g ≤ 0`, `−g ≤ 0` through `NonlinearInequalityConstraints` for now). A
`NonlinearConstraints` *aggregator* (folding nonlinear + linear-ineq/eq + box
into one `c(x) ≤ 0` vector, PRIMA's `get_nlcon`/full-COBYLA form — see "deferred
aggregator" below) is **wanted but deliberately deferred**: COBYLA ships first on
the single-kind inequality trait.

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

## No `Constraint` *parent* supertrait over the four sibling kinds

Four constraint kinds have landed and *keep confirming* the wait rather than
ending it. Each keeps feasibility by a different mechanism: box by *projection*
(`ClampInPlace`), linear-inequality by a *barrier* (`MatVec`/`MatTransposeVec`),
linear-equality by a *penalty plus multipliers* (also `MatVec`/`MatTransposeVec`,
but to assemble `∇L_ρ`, not a barrier), and nonlinear-inequality by an
*exact-penalty merit + geometry/acceptance test* (COBYLA — a derivative-free
mechanism that shares nothing with the other three). The arrival of the
nonlinear kind is the case the earlier note anticipated ("a shared parent waits
for a *nonlinear* kind") — and it landed with **yet another distinct feasibility
mechanism**, so it *still* doesn't justify a parent: there is no
feasibility-check or projection op common to all four. The two linear families
share *carrier ops* but no *feasibility* op; the nonlinear kind doesn't even
share the carrier (it's function-valued, not matrix data). So `BoxConstraints`,
`LinearInequalityConstraints`, `LinearEqualityConstraints`, and
`NonlinearInequalityConstraints` stay sibling traits with **no common parent**.
One-member (or no-shared-op multi-member) hierarchies are overhead with no
value; designing on paper without a solver to validate against tends to need
redoing.

### `LinearConstraints` is an aggregator, not that parent

`LinearConstraints` (the binding of `Lincoa`, `src/solver/lincoa.rs`) is a
*separate* trait, not the forbidden parent. LINCOA is the first solver to consume
**more than one kind at once** — box bounds + linear equalities + linear
inequalities together (PRIMA's `get_lincon` form). It folds all of them into a
**single** `A x ≤ b` system handled by **one** active-set feasibility mechanism
(`trstep`/`getact`), so for LINCOA there is exactly one feasibility op, and the
three kinds are just data to fold (`fold_constraints` in `lincoa/init.rs`).

`LinearConstraints` exposes optional `inequalities()` / `equalities()` /
`lower()` / `upper()` accessors (all defaulting to `None`); a problem implements
only the blocks it has. It is **standalone**: not a supertrait of the three
siblings, and deliberately **no blanket impl** bridges from them — a blanket
`impl<P: LinearInequalityConstraints> LinearConstraints for P` could only forward
the inequality block, silently dropping any box/equality data the problem also
carries, and would block a manual impl by coherence. The siblings remain the
right surface for their *single-kind* consumers (barrier on
`LinearInequalityConstraints`, augmented-Lagrangian on
`LinearEqualityConstraints`). So the "no parent over the siblings" rule above
still stands; LINCOA validated an aggregator, not a hierarchy.

### Deferred: a `NonlinearConstraints` aggregator for COBYLA

PRIMA's full COBYLA folds nonlinear inequalities + linear ineq/eq + box bounds
into one `constr(x) ≤ 0` vector. The maintainer wants the same eventually: a
`NonlinearConstraints` aggregator that mirrors `LinearConstraints` — a required
nonlinear block plus optional `inequalities()`/`equalities()`/`lower()`/`upper()`
blocks, all folded into the constraint vector COBYLA's merit function sees. It is
**deliberately deferred** (COBYLA ships first on the single-kind
`NonlinearInequalityConstraints`), not foreclosed. Like `LinearConstraints` it
must be **standalone**: not a parent of the sibling kinds and **no blanket impl**
bridging from them (a blanket impl could only forward the nonlinear block and
would silently drop linear/box data). Adding it later is purely additive and
breaks nothing — the single-kind trait stays the right surface for the
inequality-only consumer. See the `NonlinearInequalityConstraints` rustdoc
("Future direction").

## Constraints live on the problem, never on state

Don't put `lower` / `upper` on `BasicState` "for convenience". State carries
iteration history; constraints define the problem. Bounds on state would
silently un-constrain a problem if a different state were swapped in, and
decouple constraint semantics from where the solver type system enforces them.
Termination criteria that need bounds (e.g. `ProjectedGradientTolerance`) clone
them at construction — that's the deliberate pattern, not a workaround.
