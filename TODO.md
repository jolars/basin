# TODO

Ordered by recommended sequence.

## General design

## Basin 2.0

- [ ] **Simplify and strengthen the full-form constraint API (tenet 4).**
  Consider having COBYLA consume `NonlinearConstraints` directly, removing
  the need for the `FoldedConstraints` compatibility adapter. Separate the
  constraint-value representation from `Param`, and avoid requiring an
  unused matrix type when linear blocks are absent. Define strict validation
  for NaN bounds, wrong-sign infinities, inverted bounds, and malformed
  shapes, with typed configuration errors that preserve user callback
  errors. Revisit problem typing if unconstrained solvers must reject
  constrained problems: a `CostFunction` bound alone cannot enforce that
  guarantee. Keep constraints problem-side, preserve distinct blocks, and
  require only the math capabilities each solver needs. Define migration
  from the 1.x API.

- [ ] Remove the deprecated `TerminationCriterion` facility, all shipped
  criterion types and re-exports, `Executor::terminate_on`,
  `InnerExecutor::terminate_on`, composed `inner_terminate_on` methods,
  `run_loop`, and `ResumableInner::segment_criteria`. Retain stopping
  reasons, solver convergence setters, direct execution controls, and
  closure hooks.

- [ ] Remove deprecated tolerance and algorithm-setting aliases, including
  scalar/root and line-search aliases.

- [ ] Remove `BarrierMethod::new`, `AugmentedLagrangianMethod::new`, and
  `with_inner_grad_tol`, along with their implicit inner-gradient checks.
  Use `with_inner_solver` and the supplied solver's convergence settings.

- [ ] Move remaining shared numerical calculations out of the compatibility
  criterion types and remove the compatibility-only tests and bridges.

- [ ] **Make the `problems` feature opt-in.** Set `default = []` while retaining
  `problems = []` for benchmarks, examples, and other corpus consumers. Keep
  the existing default through Basin 1.x: removing it breaks downstream
  imports that rely on implicit activation. Update documentation and ensure
  tests, examples, benchmarks, and the WASM visualizer explicitly enable
  `problems` wherever they use the corpus.

- [ ] **Reconsider exact evaluation budgets for Basin 2.0.** In Basin 1.x,
  `max_cost_evals` remains a boundary-checked execution limit:
  `Solver::init` and an active `Solver::next_iter` finish before the
  executor checks it, so initialization and batched iterations may exceed
  the threshold. A future hard-cap API would need executor/problem-level
  control flow that prevents callbacks after exhaustion, handles batch
  reservation and composed problems, and defines the outcome when the budget
  cannot complete initialization. Cover COBYLA's `n + 1`-point
  initialization and zero-to-two-evaluation steps in any resulting contract
  and tests.

## Deferred design

- [ ] **Revisit a shared constraint-violation capability (tenet 3).** COBYLA and
  constrained MADS now provide multiple consumers, but they use different
  violation measures, and only `ConstrainedMadsState` exposes its measure.
  Define common state semantics before adding a reporting API or a composite
  feasibility-and-optimality stopping rule. Do not add a standalone
  `FeasibilityTolerance`: executor criteria are combined with OR, so it
  could stop at the first feasible but nonoptimal iterate.

- [ ] **Design nonlinear equality constraints when a solver needs their
  structure (tenet 4).** For now, represent `g(x) = 0` as the pair
  `g(x) ≤ 0` and `−g(x) ≤ 0`. Do not add a dedicated trait without a
  consumer that can validate equality-specific operations and semantics.
