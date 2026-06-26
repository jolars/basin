---
description: Contracts for solvers that run another solver as a sub-step (memetic CMA, basin hopping, barrier and augmented-Lagrangian, multi-start polish): eval aggregation, inner-criteria statefulness, failure routing, and the WarmStart/MemeticInner seeding split.
paths:
  - "crates/basin/src/solver/**/*.rs"
  - "crates/basin/src/core/inner.rs"
  - "crates/basin/src/core/executor.rs"
---

# Solver composition

Some solvers run another solver as a sub-step (memetic CMA-ES + LM, basin
hopping, barrier and augmented-Lagrangian, multi-start polish, …). The
composition primitive is `run_loop(&mut problem, state, &mut solver, &mut
criteria, max_iter)` in `src/core/executor.rs`; the builder-style adapter
`InnerExecutor<S, So>` in `src/core/inner.rs` wraps it for the common case
where an outer solver stores a pre-configured inner and reuses it across
outer iters. Both take `&mut Problem<P>` (the counting wrapper), never a
raw `&P`.

## Three contracts every outer solver must follow

1. **Eval aggregation.** Two shapes, picked by *what problem the inner sees*:

   - **Same-problem inner** (e.g. `CmaInject`, `BoundedCmaInject`,
     `MaLsChCma`): the outer passes its own `&mut Problem<P>` straight to
     the inner. Inner cost/gradient/residual/Jacobian/Hessian calls
     bump the *same* `EvalCounts` as the outer's own calls, so aggregation
     happens transparently. No explicit roll-up. The outer state's
     `CountsMirror` impl decides how those counts map onto its
     `cost_evals` (and `gradient_evals`, if it carries one).

   - **Adapter-problem inner** (e.g. `BarrierMethod`/`LogBarrier`,
     `AugmentedLagrangianMethod`/`AugmentedLagrangian`): the outer
     constructs a *fresh* `Problem::new(adapter)` around its adapter type,
     runs the inner against it via `run_loop(&mut inner_wrapper, …)`, then
     folds the inner wrapper's counts back into the outer's wrapper via
     `outer.counts_mut().add(inner.counts())`. (Copy `*inner.counts()` to a
     local first if the adapter still borrows `problem.inner()`: the
     borrow checker won't let you reborrow `problem` mutably otherwise.)

   Skipping the roll-up in the adapter case silently corrupts
   `MaxCostEvals` budgets and the public `result.cost_evals()` read.
   Same-problem inners never need it because the wrapper is shared. The
   contract is spelled out on `Solver::next_iter`'s rustdoc;
   `crates/basin/tests/inner_executor.rs` asserts it.

2. **Inner termination criteria are reset per run.** An `InnerExecutor` keeps
   its `Vec<Box<dyn TerminationCriterion<S>>>` for its whole lifetime and reuses
   it on every `run()`. `run_loop` calls `TerminationCriterion::reset` on each
   criterion at entry, so any per-run internal state is cleared before each
   call: `MaxTime` (start instant), `RelativeGradientTolerance` (anchored
   `‖∇f_0‖`), and `NoImprovement` (stall counter) are all safe to reuse across
   inner runs. The contract this imposes on criterion authors: a criterion that
   holds cross-call state MUST override `reset` to clear it (the default is a
   no-op). Stateless criteria (`MaxIter`, `MaxCostEvals`, `GradientTolerance`,
   …) need no override.

3. **Failure routing.** `run()` returns a full `OptimizationResult` carrying a
   `TerminationReason`. Use `reason.is_failure()` (true only for
   `SolverFailed`) to decide whether to bubble the failure via the outer's
   mid-iter `Option<TerminationReason>` return. Everything else (`MaxIter`,
   the tolerance reasons, `SolverConverged`) is a clean stop: the outer
   consumes the inner's final iterate and continues. The common bug is
   forgetting to propagate `SolverFailed` and treating an aborted inner run as
   a successful one.

## `InnerExecutor` vs `run_loop`

Reach for `InnerExecutor` when the outer wants to expose `inner_max_iter` /
`inner.terminate_on(...)` to its users via builder methods that mirror the
framework. Reach for raw `run_loop` when the outer wants per-call criteria
passed through a different surface, or runs an adapter-problem inner, since
that case constructs a fresh `Problem::new(adapter)` per outer iter rather
than reusing the outer's wrapper. (Reconstructing criteria per call is no
longer a reason to drop to `run_loop`: stateful criteria reset per run, so an
`InnerExecutor` reuses them safely; see contract 2.)

## Per-run state counts via `CountsMirror`

`run_loop` snapshots the wrapper at entry and the executor mirrors the
*delta* (wrapper-at-now minus the entry baseline) onto the inner state's
`cost_evals`/`gradient_evals` via `CountsMirror`. So the inner state
always reflects per-run work, not cumulative-across-calls work: nested
`run_loop` calls against the same wrapper see clean per-call counters.
Each shipped state has its own mirror rule:

- Gradient states (`BasicState`, `QuasiNewtonState`, `LbfgsState`):
  `cost_evals = cost + residual`,
  `gradient_evals = gradient + jacobian + hessian`. `BasicState` now serves only
  the genuine gradient solvers (`GradientDescent`, `ProjectedGradientDescent`,
  `Sgd`).
- NLLS state (`NllsState`: `LevenbergMarquardt`, `Trf`, `GaussNewton`):
  `cost_evals = cost + residual`, plus separate `residual_evals = residual` and
  `jacobian_evals = jacobian + hessian` accessors (MINPACK `nfev`/`njev`). It
  does **not** impl `GradientState` (so framework gradient criteria are a compile
  error, not a silent no-op; see api-stabilization.md B7).
- Scalar state (`ScalarState`: `Brent` and future 1D solvers): cost-only,
  `cost_evals = cost + residual`; no gradient, no residual or Jacobian.
- Derivative-free states (`BasicSimplexState`, `BasicPopulationState`,
  `CmaEsState`, `MaLsChState`): `cost_evals = total_work()`, every kind of
  work folded in. This is what makes a CMA-ES outer (which drives a
  `CmaEsState`) with an L-BFGS inner just *work*: the inner's gradient evals
  show up in the outer's `cost_evals` honestly, with no manual cross-type fold.

User-defined state types plugging into `Executor` must impl `CountsMirror`;
it is `pub` for exactly that reason.

## Seeding an inner's state: `WarmStart` (+ `MemeticInner`)

An outer that re-solves a subproblem from the current iterate must *build* the
inner's state, not just drive it, and inners carry different state shapes
(`BasicState`, `LbfgsState`, `QuasiNewtonState`, `BasicSimplexState`).

- `WarmStart<V>` (`src/core/inner.rs`) is the minimal primitive: `type State:
  State<Param=V>` + `seed(&self, x) -> State` (σ-free, the solver's natural
  default scale).
- `MemeticInner<V>: WarmStart<V>` (`src/solver/cma_inject.rs`) extends it with
  `seed_scaled(x, σ)` (defaults to `seed`; only Nelder-Mead's σ-scaled simplex
  overrides it). No per-trait eval-aggregation hook: same-problem composition
  shares the `Problem<P>` wrapper, and the `total_work()` fold in
  `CmaEsState`'s `CountsMirror` (same rule as `BasicPopulationState`) rolls
  every kind of inner work into the outer's single `cost_evals` automatically.

Two consumer families validate the split: the barrier and AL methods bound `So:
WarmStart<V>` with `So::State: GradientState + CountsMirror` (gradient inners
only: the `GradientState` bound excludes the only σ-sensitive inner,
Nelder-Mead, so the σ-free `seed` is exactly right); CMA-injection (`CmaInject`
/ `BoundedCmaInject`) bound `I: MemeticInner<V>` with `I::State: CountsMirror`
and call `seed_scaled`. This split resolved the "dummy-σ wrinkle": barrier and
AL never pass a meaningless σ.

## Don't grow a `Composed<Outer, Inner>` type until ≥2 concrete consumers want it

Same spirit as the "no `Constraint` supertrait until two consumers" rule.
`WarmStart`/`MemeticInner` cover *state seeding* for both memetic CMA and the
barrier and AL family, but they are not a `Composed` abstraction: they say nothing
about the outer loop, eval-aggregation routing, or failure bubbling (those stay
the three contracts above). A coarser `Composed` marker is still unmotivated:
the shipped composed solvers share the three contracts and (some) `WarmStart`,
and nothing more.
