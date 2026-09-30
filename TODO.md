# TODO

Ordered by recommended sequence.

## Solvers and integrations

These priorities come from comparing Basin with the [SciPy optimization
catalog](https://docs.scipy.org/doc/scipy/reference/optimize.html). Favor
capabilities that help downstream integrations. Preserve Basin 1.x APIs and
behavior, the default WASM build, and honest backend support. Validate new
numerical work against analytic cases and reference implementations.

### Priority additions

- [x] **Implement SLSQP.** Add a dense, pure-Rust solver for smooth objectives
  with box bounds, nonlinear equalities, and nonlinear inequalities. Add
  compatible problem-side interfaces for nonlinear equalities and constraint
  Jacobians, with analytic derivatives and finite-difference adapters.
  Preserve the default WASM build and validate every supported backend
  against analytic cases and SciPy/NLopt reference results, including
  feasibility, stationarity, rank-deficient constraints, and failure
  handling.
- [x] **Make finite differences respect box bounds.** Add an opt-in path that
  adjusts probe directions and step sizes near bounds, including fixed
  coordinates and narrow intervals. Forwarding bounds alone does not keep
  the current probes feasible. Cover gradients and Jacobians first, with
  tests for objectives defined only inside their bounds.
- [x] **Add gradient and Jacobian checkers.** Compare analytic derivatives with
  finite differences using scale-aware error reports and optional
  directional checks. Reuse the bound-aware probe machinery when bounds are
  supplied, and distinguish non-finite evaluations from derivative
  mismatches.
- [x] **Add robust nonlinear least squares.** Added `RobustLeastSquares` with
  squared, Huber, soft-L1, Cauchy, arctangent, and custom losses and a
  residual scale. All five NLLS solvers use the robust objective, gradient,
  and safeguarded local model, with `f32` and `f64` support on their
  existing backends. Outlier-contaminated fits agree with [SciPy's
  least-squares
  API](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html).
- [x] **Add full trust-region-reflective least squares.** Added
  `TrustRegionReflective` with Coleman-Li scaling, an explicit radius,
  reflected-step selection, a rank-aware dense SVD solve, and
  fixed-coordinate elimination. All four dense backends support `f32` and
  `f64`; active bounds and rank deficiency are covered by analytic and SciPy
  1.16.2 comparisons. `Trf` retains its existing bounded-LM behavior through
  Basin 1.x. The large-scale path remains a follow-up below.
- [x] **Improve full TRF termination at the floating-point cost floor.** Found
  during the [Navette integration](https://github.com/opticsWolf/Navette/issues/2)
  and reproduced directly in Basin 1.14.0 with the `Vec<f64>` backend, using
  both an analytic Jacobian and central `BoundedFiniteDiff`. Minimize
  `r(x) = [x[0] - 0.3, x[1] - 0.7, 2]` from `[0, 1]` in the unit box,
  with absolute scaled gradient tolerance `1e-10`, relative cost-change
  tolerance `1e-12`, relative step tolerance `1e-12`, and 200 iterations.
  TRF returned `SolverFailed` after four completed iterations at approximately
  `[0.2999999969867295, 0.7000000030132705]`, with objective `0.5 * ||r||^2 = 2`.
  Further cost reductions round to zero, and the observed stopping checks
  cannot inspect the rejected trials inside the solver's loop. SciPy 1.18.1
  TRF stops on `xtol` on the same problem. Removing the constant residual or
  loosening the gradient tolerance to `1e-8` lets Basin converge. TRF now
  reports `NumericalNoProgress` when a finite equal-cost rejection is followed
  by a contracted trial that changes no parameter. It retains the last accepted
  point without claiming stationarity. Regression tests cover analytic and
  central finite-difference Jacobians, both controls, all dense backends with
  `f32` and `f64`, and genuine failures.
- [x] **Implement nonlinear conjugate gradient.** Added `NonlinearCg` with
  selectable Hager–Zhang (default) and Polak–Ribière+ updates, pluggable
  line searches, and optional periodic restarts. All four dense backends
  support `f32` and `f64`; tests cover analytic PR+ updates, descent
  safeguards, ill-conditioned problems, and Hager–Zhang's CG_DESCENT C 1.2
  reference results. This is distinct from Steihaug's linear trust-region
  CG.
- [x] **Implement DIRECT.** Added original `Direct` for deterministic global
  optimization over finite box bounds, with fixed-coordinate elimination,
  documented rectangle selection and trisection, and exact checkpoint
  support. All four dense backends support `f32` and `f64`. Solution quality
  and evaluation counts are checked against SciPy 1.16.2's original DIRECT
  on Styblinski–Tang 2D and 6D and translated Ackley 6D.

### Follow-up candidates

- [ ] **Add iterative sparse least squares and Jacobian coloring.** Extend the
  full TRF work with Jacobian and transpose-Jacobian products, an LSMR
  solve, and the two-dimensional subspace method. Add sparsity-pattern-based
  finite-difference coloring to reduce residual evaluations. Existing sparse
  storage and direct solves do not provide this large-scale algorithm.
- [x] **Expand differential evolution's core algorithms.** Added six mutation
  rules, independently selectable binomial and exponential crossover, and
  generation-wise mutation dithering. The default `DE/rand/1/bin` trajectory
  is preserved, and every combination supports exact resume and `DeInject`
  composition. All dense backends support `f32` and `f64`; mutation formulas
  and solutions are checked against [SciPy 1.16.2 differential
  evolution](https://docs.scipy.org/doc/scipy-1.16.2/reference/generated/scipy.optimize.differential_evolution.html).
- [ ] **Add differential-evolution sampling policies.** Basin 2.0 now honors
  explicit members supplied through `PopulationProgress::from_population`.
  Add Latin hypercube sampling, then assess Sobol or Halton sampling,
  including sequence quality, population-size requirements, and dependency
  costs. Backports need explicit opt-ins: Basin 1.x clears supplied members
  on fresh initialization, and its default behavior must remain compatible.
- [ ] **Add nonlinear constraints to differential evolution.** Reuse the
  problem-side constraint interfaces and assess Lampinen's feasibility-based
  selection. This needs population feasibility records and selection-aware
  incumbent reporting, since the preferred feasible candidate need not have
  the lowest objective. Keep objective-based execution controls honest.
- [ ] **Add integer variables to differential evolution.** Define a
  problem-side domain contract, validate that each integer interval contains
  an integer, and preserve integrality through sampling, mutation repair,
  and local refinement. Specify compatible `DeInject` inner solvers before
  enabling mixed-integer local search.
- [ ] **Add dedicated linear least-squares solvers.** Implement nonnegative
  least squares and general box-bounded linear least squares. Reuse suitable
  factorizations and test rank deficiency, active bounds, and optimality
  against analytic solutions and [SciPy bounded linear least
  squares](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.lsq_linear.html).
- [ ] **Add curve-fitting helpers.** Provide residual construction, weighting or
  whitening, and optional local covariance estimates around existing
  least-squares solvers. Document the approximation and behavior under rank
  deficiency, active bounds, and robust losses. Use [SciPy
  curve_fit](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html)
  as a comparison for convenience and reporting.
- [x] **Expand scalar roots and bracketing.** Added automatic root and minimum
  bracketing, safeguarded secant, Newton, and Halley methods, and TOMS 748
  (`k = 2`). Root solvers retain direct fallible callbacks, signed function
  values, and bracket-based convergence. Bracketing is an explicit first
  stage.
- [ ] **Add batched independent scalar solves when a consumer needs them.**
  Build on the direct scalar root and bracketing APIs, preserving separate
  root and minimization convergence semantics. See [elementwise
  optimization](https://docs.scipy.org/doc/scipy/reference/optimize.elementwise.html).
- [ ] **Add multivariate root solving when an integration needs it.** Start with
  a safeguarded Newton/hybrid method or Broyden, then assess Anderson
  acceleration and Newton-Krylov. Require residual-based root validation:
  convergence of a least-squares objective can leave a nonzero residual.
  Promote this work if a downstream consumer needs it, including access to
  the final Jacobian for sensitivities. See [SciPy nonlinear
  roots](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.root.html).
- [ ] **Evaluate COBYQA.** Add a quadratic-model method for nonlinear
  constraints if benchmarks justify it alongside COBYLA, BOBYQA, and LINCOA.
  Those existing methods cover different model or constraint classes.

### Longer-term candidates

- [ ] **Evaluate a trust-constr-style solver.** After SLSQP establishes
  constraint derivatives and native nonlinear equalities, assess a general
  constrained trust-region method with sparse derivatives, suitable
  factorizations, and feasibility and stationarity diagnostics. This is a
  larger project than the current barrier and augmented-Lagrangian adapters.
- [ ] **Assess remaining local methods.** Consider Powell's direction-set
  method, line-search Newton-CG, TNC, GLTR/`trust-krylov`, and SR1 Hessian
  updates when a workload demonstrates value. Powell's direction-set method
  is distinct from NEWUOA and BOBYQA; GLTR is distinct from Steihaug CG.
  Compare against [SciPy's local
  methods](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html).
- [ ] **Evaluate dogbox after strengthening TRF.** Add rectangular trust-region
  least squares if small bounded fitting problems benefit.
- [ ] **Assess complex-step differentiation.** Design an opt-in callback and
  scalar interface for complex perturbations without broadening every real
  solver's requirements. Document the analyticity requirement and operations
  that invalidate the approximation.
- [ ] **Evaluate SHGO and dual annealing after DIRECT.** Require evidence of
  useful coverage beyond the existing global methods. Treat dual annealing
  as a distinct algorithm from the current simulated annealing solver.
- [ ] **Defer LP/MILP, assignment, and isotonic regression.** Revisit these
  separate problem families only with concrete downstream demand. Assess
  optional integrations before expanding the core or adding heavy solver
  dependencies, preserving the default pure-Rust WASM build.

## General design

### State API additions for Basin 1.x

Ship these in order, preserving existing public APIs and behavior:

- [x] **Add owned checkpoint and result extraction.** Add a consuming
  `Stepper::into_checkpoint()` using `ExactCheckpoint`, then an opt-in run
  method returning the final solver, state, counts, and termination reason.
  Require neither `Clone` nor serialization.
- [x] **Introduce shared progress storage.** Start with point and first-order
  states for external and new solvers. Keep existing solver/state types,
  constructors, associated types, and serialized representations compatible.
- [x] **Add opt-in state capabilities.** Use new interfaces for checked record
  access, raw evaluation counts, and explicit incumbent-selection semantics.
  Bind new controls to the capabilities they need; preserve existing readers
  and stopping behavior.
- [x] **Expose which native stopping tests caused termination.** Added
  `NativeConvergenceDiagnostics` and `NativeConvergenceTest` for both LM
  factorizations and full TRF, including their convergence wrappers.
  `run_with_solver()` results expose all passing native tests at the stopping
  stage, including LM's trust-radius test and TRF's all-fixed case. Navette
  can distinguish these without inferring a cause from the final iterate.
  Existing stopping behavior, termination values, and fieldless enum numeric
  casts remain unchanged. Numerical safeguards retain their distinct reasons.
  Fresh runs clear records, exact checkpoints retain them, and the result
  accessor excludes earlier native diagnostics when continuation stops on a
  budget or another non-native reason. Tests cover simultaneous tests, rejected
  trials, robust objectives, continuation, and all dense backends with both
  scalar types.

### State API prototype

- [x] **Prototype shared progress states with solver-owned machinery.** Follow
  the [state and lifecycle
  contracts](CONTRIBUTING.md#state-and-lifecycle-contracts). Exercise BFGS
  and L-BFGS for ownership and backend inference, Nelder-Mead for
  observation, and simulated annealing for continuation. Preserve Basin 1.x
  public APIs and behavior while evaluating compatible additions.

#### Acceptance checks

Covered by the [executable prototype](crates/basin/tests/state_api_prototype.rs)
and its test-only support modules:

- [x] An external test solver updates shared first-order storage through public
  methods and obtains correct executor bookkeeping without custom state
  code.
- [x] Checked access distinguishes a seed, an evaluated rejected point, and an
  incumbent. Missing gradients cannot silently disable an attached check.
  Invalid structural updates preserve the previous record. Solver-owned
  memory capacity and explicit scale/simplex inputs have one construction
  path.
- [x] Nonmonotone iterations and complete population replacement preserve
  matching historical point/cost records. Equal costs retain metadata; NaN
  and positive infinity cannot poison incumbent selection; negative infinity
  does not establish unboundedness. CMA-ES considers mean and samples.
  Selection changes are explicit events, not inferred from iteration
  numbers.
- [x] A constrained transition to a higher-cost feasible incumbent preserves the
  solver's selection. Cost-only controls cannot treat an infeasible
  incumbent as a successful objective target.
- [x] Scalar, fused, batch, nested, and failing evaluations preserve raw counts.
  Mid-step publication stamps the final charged count without adding an
  iteration. Repeated boundary observation does not refresh incumbent age.
  Batches charge submitted work even on failure; fused calls charge each
  category. Shared-wrapper inner runs use deltas, and adapter counts merge
  once.
- [x] Reusing a solver for a fresh run agrees with fresh construction, including
  after a dimension change. Point warm starts reevaluate the point and reset
  momentum, model history, and convergence history. Independent chains use
  explicit seeds without consuming a prototype's live RNG.
- [x] Pausing and resuming BFGS/L-BFGS and deterministic seeded annealing
  reproduces uninterrupted evaluations and iterates, including
  iteration-zero checkpoints and stateful convergence checks. Hard errors do
  not publish partial state.
- [x] Nelder-Mead observation uses authoritative simplex storage without a
  second full copy. Final model access and checkpoint extraction consume
  ownership without imposing `Clone` or serialization on the solver.
- [x] Construction and the selected capability bounds work on every claimed
  backend, preserve `f32` round trips and the default WASM build, and keep
  matrix inference practical at ordinary BFGS call sites.

#### Findings and remaining design work

The ownership split works across the four backends and both scalar types. BFGS
infers its matrix through a backend association and accepts an explicit
override. Existing observers borrow the shared simplex, and owned checkpoints
preserve model, RNG, neighbor, and convergence history. Population updates
preserve member order so solver-owned per-member data stays aligned.

Before adding production APIs, settle the public home of the matrix association
and rank-update capability (currently a test-local adapter), factories for
resetting configured components, and integration with execution controls. The
algorithm adapters cover unbounded L-BFGS, classical Nelder-Mead, and annealing
without reannealing; serialization and performance remain migration work. Run
the prototype with `cargo test -p basin --test state_api_prototype` and the
desired backend features.

## Basin 2.0

- [ ] **Complete the stable-release handoff before publishing 2.0.0.** Follow
  [Maintenance and releases](MAINTENANCE.md#publishing-200): disable Latest
  promotion on `v1`, finish in-flight maintenance releases, record the 1.x
  support end date, and publish migration guidance. Keep the 2.0 release PR
  open until the release is ready.

- [ ] **Revise solver convergence defaults ([#109](https://github.com/jolars/basin/issues/109)).**
  Preserve existing defaults throughout Basin 1.x. Use established
  solver-specific stopping policies as the baseline for 2.0: paired absolute
  simplex-size and cost-spread tests for Nelder-Mead, projected-gradient or
  normalized cost-reduction tests for L-BFGS-B, MINPACK-style orthogonality,
  model-reduction, and scaled-step tests for LM, and cost and step tests
  alongside the bound-scaled gradient test for full TRF. Compare the precise
  formulas, observation stages, and defaults with
  [SciPy](https://docs.scipy.org/doc/scipy/reference/optimize.html) and
  [Ceres](https://ceres-solver.readthedocs.io/latest/nnls_solving.html), and
  distinguish full `TrustRegionReflective` from the legacy bounded-LM `Trf`.
  Match the mathematics rather than only similarly named tolerance settings;
  do not promise universal scale invariance. Obtain the reporter's NIST StRD
  harness and validate solution accuracy as well as termination reasons on
  both supplied starting points, rescaled objectives and parameters, zero-cost
  optima, active bounds, and stagnation away from a solution. Select and test
  defaults for both `f32` and `f64` across supported backends. Keep numerical
  no-progress safeguards distinct from convergence, account for new vector
  capability requirements in default simplex checks, and document explicit
  settings that recover the 1.x behavior.

- [ ] **Make structured termination reports part of ordinary results.** Build
  on the 1.x native diagnostics. Replace the flat `TerminationReason` with a
  payload-bearing report distinguishing convergence, execution limits,
  objective targets, numerical stalls, numerical failures, cancellation, and
  application stops. Return it with the final state and authoritative counts
  from ordinary `run()`; retaining the solver remains optional. Native and
  shared checks should return their explanation with the stopping decision,
  both at iteration boundaries and inside steps. Preserve precise criterion
  semantics, simultaneous passing tests, and relevant measurements when
  available, and provide an extension mechanism for external solvers.
  Keep iteration completion independent of termination. Reports describe a
  particular stopping event; exact continuation retains algorithm and
  convergence history and produces a report for the resumed run. Composition
  must distinguish inner convergence from outer convergence, retain relevant
  inner failure details, and explicitly decide whether to consume partial
  results. Keep typed callback aborts in `Result::Err`, distinct from numerical
  termination with a published state. Replace enum numeric casts with an
  explicit documented code mapping if bindings require one. Keep the reporting
  API compact; general tracing and exhaustive solver measurements are separate
  concerns. Document migration from the 1.x reasons and diagnostics.

- [ ] **Clean up backend compatibility aliases in Basin 2.0.0.** Retire the
  frozen unversioned `nalgebra`, `ndarray`, and `faer` feature aliases and
  their LAPACK/BLAS counterparts in favor of exact version features and the
  explicit `*_latest` aliases. Remove or replace `NalgebraQuasiNewtonState`,
  `NdarrayQuasiNewtonState`, and `FaerQuasiNewtonState`, whose concrete
  types currently change when a newer backend version is enabled. Prefer
  explicit `QuasiNewtonState<V, M, F>` types or version-specific aliases.
  Preserve independent implementations for every enabled version and
  version-specific acceleration. Document the 1.x migration and retain
  simultaneous-version and downstream Cargo feature-unification tests.

- [x] **Remove bincode in Basin 2.0.0.** Drop legacy readers for unprefixed
  state checkpoints and version 1 exact checkpoints, along with the bincode
  dependency and compatibility-only tests. Retain the postcard formats and
  document [checkpoint migration requirements](MIGRATING.md#checkpoint-files)
  for users upgrading from 1.x.

- [x] **Migrate existing solvers to shared progress states.** Build on the
  validated [prototype](#state-api-prototype) and [1.x
  additions](#state-api-additions-for-basin-1x). Replace legacy public
  solver/state types and apply uniform evaluation categories and the
  accepted initialization and continuation contracts. Document replacement
  constructors, public types, trait bounds, stopping semantics, and
  serialized-format compatibility. See the [migration guide](MIGRATING.md#shared-progress-states).

- [x] **Decide how observers access solver diagnostics before finalizing 2.0.**
  Keep `Observe<S>` for progress and add optional `ObserveSolver<S, So>` hooks
  and `Executor::observe_solver` closures that borrow the solver at the same
  observation boundaries. Preserve solver ownership, modes, and mixed
  registration order without cloning workspace or triggering evaluations.
  `Stepper::solver()` remains available for applications driving their own
  loop. See [custom diagnostic logging](MIGRATING.md#observing-solver-diagnostics).

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

- [x] Remove the deprecated `TerminationCriterion` facility, all shipped
  criterion types and re-exports, `Executor::terminate_on`,
  `InnerExecutor::terminate_on`, composed `inner_terminate_on` methods,
  `run_loop`, and `ResumableInner::segment_criteria`. Preserve stopping
  semantics through structured reports, solver convergence setters, direct
  execution controls, and closure hooks.

- [x] Remove deprecated tolerance and algorithm-setting aliases, including
  scalar/root and line-search aliases.

- [ ] Remove `BarrierMethod::new`, `AugmentedLagrangianMethod::new`, and
  `with_inner_grad_tol`, along with their implicit inner-gradient checks.
  Use `with_inner_solver` and the supplied solver's convergence settings.

- [ ] Move remaining shared numerical calculations out of the compatibility
  criterion types and remove the compatibility-only tests and bridges.

- [x] **Make the `problems` feature opt-in.** Set `default = []` while retaining
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

- [ ] **Define shared constraint-violation reporting (tenet 3).**
  Constrained MADS publishes the sum of squared positive violations in
  `SelectedState`, and SLSQP publishes the sum of absolute equality residuals
  and positive inequality violations in `SelectedFirstOrderState`. COBYLA
  publishes its maximum positive violation in `SelectedState`. Define a shared state capability with explicit semantics for
  the measure, scaling, associated iterate, and availability before
  initialization. Preserve each solver's numerical convergence semantics:
  SLSQP already combines feasibility with other convergence tests. Do not
  add a standalone `FeasibilityTolerance` to the executor: its stopping
  conditions combine with OR, so it could stop at the first feasible but
  nonoptimal iterate.
