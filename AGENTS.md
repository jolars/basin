# AGENTS.md

@CONTRIBUTING.md

The contributor guide above applies to agents and humans alike. The sections
below are agent-specific: path-scoped rules that auto-load on file access, and
deliberate non-tenet choices recorded so they don't get "fixed" by accident.

## Subsystem rules (`.claude/rules/`)

Deeper, subsystem-specific guidance is path-scoped and auto-loads when you touch
the relevant files. Don't duplicate it here:

- `constraints.md`: the three constraint kinds, adapter asymmetry,
  no-supertrait, constraints-not-on-state. Loads under
  `src/core/{constraint,barrier,augmented_lagrangian}.rs` and the constrained
  solvers.
- `backends.md`: the math tier system, trait inventory, the
  honest-implementability rule, and the per-solver "Backends" doc note. Loads
  under `src/core/math/`.
- `solver-composition.md`: running a solver as a sub-step: the three
  contracts, `InnerExecutor` vs `run_loop`, `WarmStart`/`MemeticInner`. Loads
  under `src/solver/` + `core/{inner,executor}.rs`.
- `problems.md`: test-problem corpus conventions. Loads under
  `src/problems/`.

## Deliberate non-tenet choices

- **Scalar type defaults to `f64`, but the whole pipeline is `F: Scalar`.**
  Every state (`BasicState`, `BasicSimplexState`, `BasicPopulationState`,
  `QuasiNewtonState`, `LbfgsState`, `SolisWetsState`), every solver (gradient
  descent, BFGS, both L-BFGS modes, NLLS family, CMA-ES, Solis-Wets, barrier
  and AL, line searches), and every
  shipped termination criterion carries an `F = f64` default, so existing call
  sites resolve unchanged while `f32` works end-to-end (see
  `tests/f32_round_trip.rs`). The `F = f64` default is the ergonomic choice for
  the common case, not a constraint. When adding a scalar generic to new
  surface, commit to it properly across state, solver, termination, and math
  impls rather than adding a fake generic whose defaults only work in `f64`.

- **No observer KV/metadata channel.** `Observe` (`core/observer.rs`) passes
  only `&S`. There is no argmin-style stringly-typed key-value store for
  surfacing algorithm-specific scalars (step size, barrier μ, population
  diversity), because tenet 3 makes state shape the contract: observers bind on
  the minimum `State`, `GradientState`, `SimplexState`, or `PopulationState`
  shape, and a `HashMap<String, _>` side channel would erase that compile-time
  guarantee. It can be added later without a breaking change: give `Observe` a
  new default-bodied method that forwards to `observe_iter`, switch the
  executor's call site to it, and a concrete `Kv` type keeps the trait
  object-safe for `Box<dyn Observe>`. The motivation, if it arises, is that some
  solver-internal working state (LM μ, ν, diag) lives in the solver struct
  rather than the state, so exposing it on a richer state trait would not reach
  it.

- **No `Solver::name()` introspection.** The `Solver` trait (`core/solver.rs`)
  has `type Error` plus `init`, `next_iter`, and `terminate`, but no
  `name() -> &str` (argmin has one for logging and observer display). No shipped
  observer prints the solver name, so it would be unused surface frozen into
  1.0. Adding `fn name(&self) -> &str` with a default impl is additive and
  non-breaking even post-1.0, so add it if and when an observer that displays
  the name is wanted.
