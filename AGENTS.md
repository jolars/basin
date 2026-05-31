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

- **Scalar type defaults to `f64`, but the whole pipeline is `F: Scalar`.**
  Every state (`BasicState`, `BasicSimplexState`, `BasicPopulationState`,
  `QuasiNewtonState`, `LbfgsState`), every solver (gradient descent, BFGS,
  both L-BFGS modes, NLLS family, CMA-ES, barrier / AL, line searches), and
  every shipped termination criterion carries an `F = f64` default. Existing
  call sites resolve unchanged; `f32` works end-to-end (see
  `tests/f32_round_trip.rs`). The `F = f64` default is preserved as the
  ergonomic choice for the majority case, not as a constraint. The rule from
  the original deferred-choice still stands: don't add a fake scalar generic
  where defaults only work in `f64` — commit to it properly across the new
  surface (state + solver + termination + math impls).
