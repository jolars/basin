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

- **No observer KV / metadata channel.** `Observe` (`core/observer.rs`) passes
  only `&S` — there is deliberately no argmin-style stringly-typed key-value
  store for solvers to surface algorithm-specific scalars (step size, barrier
  μ, population diversity). **Why:** tenet 3 — state shape is the contract;
  observers bind on the minimum `State` / `GradientState` / `SimplexState` /
  `PopulationState` shape, and a `HashMap<String, _>` side channel would erase
  that compile-time guarantee. **Not a one-way door, despite the original
  audit's framing.** A KV channel can be added post-1.0 *additively*: add a new
  default-bodied trait method (`fn observe_iter_with(&mut self, state: &S, kv:
  &Kv) { self.observe_iter(state) }`), switch the executor's internal call site
  to it, and existing `Observe` impls keep compiling via the forwarding default
  — a concrete `Kv` type keeps the trait object-safe for `Box<dyn Observe>`. So
  this is *deferred*, not foreclosed. The genuine future motivation, if it
  comes: solver-internal working state (CMA-ES σ / covariance / evolution
  paths, LM μ / ν / diag) lives in the *solver* struct, not the state, so the
  "expose it on a richer state trait" answer does not cover those scalars.
  Don't "fix" the absence of a KV channel by reflex — but don't treat it as
  permanently closed either.

- **No `Solver::name()` introspection.** The `Solver` trait (`core/solver.rs`)
  has `type Error` + `init` / `next_iter` / `terminate` and deliberately no
  `name() -> &str` (argmin requires one for logging/observer display). **Why:**
  no shipped observer prints the solver name, so it would be unused surface
  frozen into 1.0. **Safe to defer:** adding `fn name(&self) -> &str` with a
  default impl is additive and non-breaking *even post-1.0*, so there's no
  freeze-now pressure. Revisit only if/when an observer that displays the solver
  name is actually wanted — at which point add it with a default. Recorded so
  the absence reads as a deliberate choice, not an oversight.
