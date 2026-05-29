# AGENTS.md

@CONTRIBUTING.md

The contributor guide above applies to agents and humans alike. The sections
below are agent-specific: path-scoped rules that auto-load on file access, and
deliberate non-tenet choices recorded so they don't get "fixed" by accident.

## Subsystem rules (`.claude/rules/`)

Deeper, subsystem-specific guidance is path-scoped and auto-loads when you touch
the relevant files. Don't duplicate it here:

- `constraints.md` — the three constraint kinds, adapter asymmetry,
  no-supertrait, constraints-not-on-state. Loads under
  `src/core/{constraint,barrier,augmented_lagrangian}.rs` and the constrained
  solvers.
- `backends.md` — the math tier system, trait inventory, the
  honest-implementability rule, and the per-solver "Backends" doc note. Loads
  under `src/core/math/`.
- `solver-composition.md` — running a solver as a sub-step: the three contracts,
  `InnerExecutor` vs `run_loop`, `WarmStart` / `MemeticInner`. Loads under
  `src/solver/` + `core/{inner,executor}.rs`.
- `problems.md` — test-problem corpus conventions. Loads under `src/problems/`.

## Provisional choices (deferred, not tenets)

- **Scalar type is hardcoded to `f64`.** Solvers, `BasicState`, and tolerance
  defaults assume `f64` — simpler bounds and clearer constant defaults now, at
  the cost of a future mechanical refactor. Scalar-genericity *is* coming
  (ensmallen-style stochastic solvers want f32). **Trigger:** the first
  stochastic solver lands, or a real f32 use case appears. Plan: switch to
  `F: num_traits::Float` on `BasicState<P, F>`, `GradientDescent<F>`, etc.; the
  `ScaledAdd<S>` trait is already generic. Don't add a *fake* scalar generic
  where defaults only work in `f64` — commit to it properly or stay f64-only
  honestly.
