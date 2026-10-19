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
  1.16.2 comparisons. The bounded-LM solver is named
  `BoundedLevenbergMarquardt` in Basin 2.0 and `Trf` in Basin 1.x.
  The large-scale path remains a follow-up below.
- [x] **Improve full TRF termination at the floating-point cost floor.** Found
  during the [Navette
  integration](https://github.com/opticsWolf/Navette/issues/2) and
  reproduced directly in Basin 1.14.0 with the `Vec<f64>` backend, using
  both an analytic Jacobian and central `BoundedFiniteDiff`. Minimize
  `r(x) = [x[0] - 0.3, x[1] - 0.7, 2]` from `[0, 1]` in the unit box, with
  absolute scaled gradient tolerance `1e-10`, relative cost-change tolerance
  `1e-12`, relative step tolerance `1e-12`, and 200 iterations. TRF returned
  `SolverFailed` after four completed iterations at approximately
  `[0.2999999969867295, 0.7000000030132705]`, with objective
  `0.5 * ||r||^2 = 2`. Further cost reductions round to zero, and the
  observed stopping checks cannot inspect the rejected trials inside the
  solver's loop. SciPy 1.18.1 TRF stops on `xtol` on the same problem.
  Removing the constant residual or loosening the gradient tolerance to
  `1e-8` lets Basin converge. TRF now reports `NumericalNoProgress` when a
  finite equal-cost rejection is followed by a contracted trial that changes
  no parameter. It retains the last accepted point without claiming
  stationarity. Regression tests cover analytic and central
  finite-difference Jacobians, both controls, all dense backends with `f32`
  and `f64`, and genuine failures.
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
  storage and direct solves do not provide this large-scale algorithm. The
  [iterative-TRF API prototype](#iterative-trf-api-prototype) establishes an
  operator-only capability path; the numerical algorithm remains deferred.
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
  the lowest objective. Keep objective-based execution controls honest. The
  [constrained-DE API prototype](#constrained-de-api-prototype) demonstrates
  an additive state and composition path; the numerical algorithm remains
  unimplemented.
- [ ] **Add integer variables to differential evolution.** Define a problem-side
  domain contract, validate that each integer interval contains an integer,
  and preserve integrality through sampling, mutation repair, and local
  refinement. The [integer-domain API prototype](#integer-domain-api-prototype)
  establishes the declaration, rejection, and fixed-integer refinement
  contracts. Production sampling, mutation repair, and mixed-integer injection
  remain unimplemented.
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
  `run_with_solver()` results expose all passing native tests at the
  stopping stage, including LM's trust-radius test and TRF's all-fixed case.
  Navette can distinguish these without inferring a cause from the final
  iterate. Existing stopping behavior, termination values, and fieldless
  enum numeric casts remain unchanged. Numerical safeguards retain their
  distinct reasons. Fresh runs clear records, exact checkpoints retain them,
  and the result accessor excludes earlier native diagnostics when
  continuation stops on a budget or another non-native reason. Tests cover
  simultaneous tests, rejected trials, robust objectives, continuation, and
  all dense backends with both scalar types.

### Iterative-TRF API prototype

- [x] **Prove an operator-only reflective model and robust composition.**
  The [executable prototype](crates/basin/src/core/least_squares/iterative_trf_api_prototype.rs)
  is a test-only child of the robust least-squares module. No types, counters,
  or methods from the experiment enter Basin's public API. Existing dense
  `TrustRegionReflective` keeps its constructor, bounds, and behavior.

Use separate `JacobianProduct: Residual` and
`TransposeJacobianProduct: Residual` capabilities. Their signatures are
`jacobian_product(&self, x: &Self::Param, v: &Self::Param) ->
Result<Self::Output, Self::Error>` and
`transpose_jacobian_product(&self, x: &Self::Param, v: &Self::Output) ->
Result<Self::Param, Self::Error>`. Neither extends `Jacobian` nor introduces
a matrix associated type. Products differentiate the raw residual at `x`;
they are pure and linear in `v`. Residual length stays fixed during a solve.
Parameter and residual spaces may use distinct vector types and lengths.
Do not blanket-implement the capabilities for every `Jacobian`: an explicit
dense-product adapter can provide that convenience without blocking custom
analytic products on problems that also supply a dense Jacobian.

The intended production entry is a separate
`IterativeTrustRegionReflective<F = f64>` with `PointState<V, F>` and the
existing box-problem declaration. It will require both product capabilities
and vector operations, without `Jacobian`, `MatrixIndex`, `ScaleRowsInPlace`,
matrix factorization, or matrix-entry access. This experiment implements its
model preparation and operator consumption signatures, not a `Solver` impl
or settings builders. Production LSMR tolerances, iteration limits, and
regularization must be explicit algorithm controls. Automatic Jacobian-column
scaling would require a separate capability or explicit user scales; it must
not introduce a hidden matrix-entry requirement.

The prototype establishes these contracts:

- The model freezes the accepted parameter and robust row factors. A borrowed
  linearization ties them to one counted objective. Repeated products and
  trial residual evaluations retain that model; accepting another point
  rebuilds it. Production solver-owned workspace must preserve this rule
  across rejected trials and exact continuation.
- Raw and `RobustLeastSquares` evaluations use separate internal capability
  implementations, as the dense NLLS drivers do. The robust adapter continues
  to expose its true objective rather than implementing `Residual`. The
  prototype reuses its existing loss and scale calculations: with row factors
  `a` and model residual `r_tilde`, the actions are `diag(a) J v` and
  `J^T diag(a) u`. Thus their composition supplies safeguarded curvature,
  while applying the transpose to `r_tilde` gives the robust gradient.
  Trial reductions use the true robust cost, not the model residual norm.
- The reflective augmented operator has actions for
  `A = [diag(a) J diag(d); diag(sqrt(c))]` and its transpose. Its two output
  blocks occupy residual and parameter space. Fixed coordinates use
  `d = c = 0`. These actions provide the inputs needed by LSMR and a small
  subspace model using vector storage, with no materialized normal matrix.
  Tests supply the scaling diagonals; the production driver must compute
  Coleman–Li scaling, eliminate fixed coordinates, and select feasible steps.
- Count attempted raw residual, Jv, and transpose-Jacobian requests separately,
  before invoking callbacks, including failures. Robust weighting adds no
  evaluations. Do not charge matrix `jacobian_evals` for products. Internal
  probes stay excluded under the existing wrapper-boundary contract.
  The prototype's `OperatorCounts` pairs real `EvalCounts` with two proposed
  fields and tests totals, deltas, and composition. Production must add those
  fields to the non-exhaustive `EvalCounts` and integrate state mirrors,
  budgets, checkpoint/serde storage, and inner roll-ups before shipping the
  solver. The experiment does not prove that executor integration yet.
- Callback errors propagate unchanged. Invalid initial residuals or loss
  models cannot produce a linearization; non-finite trial residuals yield
  infinite cost without replacing the model. Shape and non-finite derivative
  guards are assertions in this model consumer. The production driver must
  map numerical failures to its structured failure reports.

Analytic checks cover rectangular nonlinear products, adjoints, augmented
curvature, fixed coordinates, all five built-in losses at nonunit scale,
rejected trial reuse, refreshed accepted points, and attempted-request counts.
Tests run both precisions on every enabled dense backend version and also use
distinct Rust types for parameter and residual space. Run
`cargo test -p basin --lib iterative_trf_api_prototype`, optionally adding
`--features nalgebra,ndarray,faer`.

The operator design follows [SciPy 1.16.2's TRF implementation](https://github.com/scipy/scipy/blob/v1.16.2/scipy/optimize/_lsq/trf.py)
and [its operator and robust-scaling helpers](https://github.com/scipy/scipy/blob/v1.16.2/scipy/optimize/_lsq/common.py).
LSMR, the two-dimensional subspace solve, reflective step selection, sparse
adapters, coloring, and production accounting/checkpoint integration remain
future work. This prototype establishes the additive API path, not numerical
convergence or performance of a large-scale solver.

### Constrained-DE API prototype

- [x] **Prove selection-aware population storage and constrained injection.**
  The [executable prototype](crates/basin/tests/constrained_de_api_prototype.rs)
  uses the real executor, inner executor, canonical problem declarations,
  and evaluation counters. Its support types remain test-only. Scripted
  populations exercise selection without implementing mutation or sampling.

The additive production path is a separate `ConstrainedDe<F>` constructed
with `ConstrainedDe::new(de: De<F>)`, paired with
`SelectedPopulationProgress<V, F>`. A sibling
`ConstrainedDeInject::with_inner_solver(outer: ConstrainedDe<F>, inner: I)`
can compose local refinement. These are proposed signatures, not public
exports. Ordinary `De`, `DeInject`, and `PopulationProgress` retain their
existing types and behavior.

The prototype establishes these contracts:

- Member records pair parameters with componentwise violations and an
  optional objective. Checked replacement validates the entire population
  before changing it and retains member order. A separate population
  capability exposes these records without promising a cost for every
  member. Seeds have no evaluated records.
- Feasible members have evaluated objectives. An all-infeasible population
  evaluates its selected representative before publication, leaving other
  infeasible objectives optional. `State::cost()` therefore remains an
  actual evaluated objective. A representative callback error propagates.
  This choice requires an objective that can evaluate an infeasible
  representative; an objective defined only on feasible points needs a
  different, future availability contract.
- Feasible trials beat infeasible members. Feasible pairs compare objective
  values; infeasible pairs use componentwise non-worsening violations,
  following [SciPy 1.16.2's Lampinen-based
  selection](https://github.com/scipy/scipy/blob/v1.16.2/scipy/optimize/_differentialevolution.py).
  Representatives use the lowest feasible objective or, before feasibility,
  the smallest unscaled sum of violations. Exact ranking ties retain the
  historical incumbent. This sum is a prototype policy, not a common
  constraint-reporting norm.
- Explicit selection events retain independent point, objective, and
  violation snapshots. The executor stamps iteration and all raw counts at
  publication, including clean mid-step stops. Republishing a selection
  preserves its age. The state implements `IncumbentState` but neither
  `ObjectiveIncumbentState` nor `PopulationState`; objective targets and
  objective stalls cannot attach to it.
- An injection inner must accept the original constrained problem. Seeds
  follow the representative ranking. Every returned point is repaired to
  the box and its constraints are reevaluated before applying the same trial
  selection rule. The prototype permits infeasible local starts and partial
  results; it does not permit a local solver to discard nonlinear blocks.
- Nonempty nonlinear callback attempts charge `residual_evals`, including
  failures and malformed output. Zero-count blocks skip callbacks. Objective
  and derivative calls preserve their existing categories. Same-problem
  inner execution shares the outer wrapper and never adds its counts twice.
- Fresh initialization reevaluates member seeds and resets script position
  and bookkeeping. Exact checkpoints preserve the script position, members,
  refinement schedule, inner solver, selection metadata, and counts.
  Tests cover both precisions on all four dense backends and serialized
  `Vec` checkpoints.

The fixture exercises finite boxes and nonlinear inequalities. Other
constraint blocks, feasibility tolerances, production mutation and sampling,
and nonlinear relaxation adapters remain future work. The `De` configuration
is retained as opaque input to the wrapper; the script does not interpret
its settings or exercise its RNG. Run the experiment with
`cargo test -p basin --test constrained_de_api_prototype`, optionally adding
`--features nalgebra,ndarray,faer,serde,parallel`.

### Integer-domain API prototype

- [x] **Prove an additive domain declaration and safe local refinement.**
  The [executable prototype](crates/basin/src/core/constraint/model/integer_domain_api_prototype.rs)
  extends the sealed canonical declaration inside a test-only child module.
  It uses the real problem traits, adapters, executor, and inner executor.
  No prototype types or methods enter the public API.

Integrality belongs in the canonical problem-side declaration, alongside
bounds and constraint blocks. The proposed public spelling is
`Constraints<B, L, N, D = ContinuousDomain>`, with an explicit builder that
changes the domain type. Existing declarations retain their continuous
meaning. Do not add a required `Domain` associated type to every problem.

The prototype wraps the existing constraint set in
`DomainConstraints<C, D = ContinuousDomain>` rather than changing the public
`Constraints` type. This demonstrates a separate additive implementation path
through the sealed `ConstraintDescription` trait. It does not prove source or
serialization compatibility for the proposed fourth parameter. Production
integration must check generic builders, borrowed views, and serialization
before choosing that spelling. Existing continuous declarations and DE runs
remain unchanged in this experiment.

The prototype establishes these contracts:

- Bounds describe intervals; a separate coordinate mask declares integrality.
  The mask and bounds have the parameter's length and remain fixed during a
  solve. A borrowed declaration shares both the original bounds and mask.
  Replacing bounds preserves the domain block. Finite-difference and
  robust-loss adapters preserve the complete declaration and its type.
- Unsupported solvers reject integer declarations through their existing
  trait bounds. Compile-time checks cover continuous DE, ordinary and
  projected Nelder-Mead, `DeInject`, and `RelaxedInner`, including relaxed
  injection. The box-relaxation adapter is not permission to discard an
  integer domain. Future constraint transformations must consume only blocks
  they support and retain any unconsumed domain declaration.
- Initialization validates finite, ordered bounds and requires
  `ceil(lower) <= floor(upper)` for every integer coordinate. The effective
  integer endpoints must lie in the consecutive exactly representable range:
  `[-2^24, 2^24]` for `f32` and `[-2^53, 2^53]` for `f64`. This deliberately
  conservative prototype policy rejects wider integer intervals, including
  singletons outside that range. Seeds must already satisfy bounds and
  integrality; validation never silently rounds them or evaluates the
  objective.
- Continuous local refinement requires an explicit adapter that fixes integer
  coordinates at the selected member's values. The prototype eliminates
  those coordinates and fixed continuous coordinates, runs projected
  Nelder-Mead on the remaining box, and reconstructs the full parameter
  before every callback. All-integer or otherwise fully fixed problems skip
  refinement without evaluations. Rounding only the returned point is
  insufficient because intermediate callbacks must also preserve integrality.
- The adapter preserves original callback errors and folds its evaluation
  counts into the outer problem exactly once, including failing attempts.
  Refinement returns its inner termination report with the matching point
  and cost. An iteration limit retains partial progress; production injection
  must apply the normal explicit inner-stop policy and outer selection rule.

Tests exercise both precisions on each enabled dense backend version.
Run `cargo test -p basin --lib integer_domain_api_prototype`, optionally adding
`--features nalgebra,ndarray,faer,parallel,serde`.

Production integer-aware DE, integer sampling and mutation repair, integer
checkpoint storage, nonlinear constrained refinement, and derivative-based
reduced problems remain future work. An explicit continuous relaxation that
allows fractional objective calls would require its own problem transformation
and callback-domain contract; `RelaxedInner` does not provide that contract.

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

- [x] **Rename bounded LM to distinguish it from reflective trust regions.**
  `BoundedLevenbergMarquardt` replaces `Trf`, including its defining module,
  without a compatibility alias. Its algorithm, settings, and dense and sparse
  support are preserved. `TrustRegionReflective` keeps its name. See the
  [migration guide](MIGRATING.md#bounded-least-squares-solver-name).

- [ ] **Complete the stable-release handoff before publishing 2.0.0.** Follow
  [Maintenance and releases](MAINTENANCE.md#publishing-200): disable Latest
  promotion on `v1`, finish in-flight maintenance releases, record the 1.x
  support end date, and publish migration guidance. Keep the 2.0 release PR
  open until the release is ready.

- [x] **Enable paired Nelder-Mead convergence defaults
  ([#109](https://github.com/jolars/basin/issues/109)).** Both simplex size
  and cost spread use absolute floors of `1e-8` and relative tolerances of
  `sqrt(f64::EPSILON)` for `f64` and `32 * f32::EPSILON` for `f32`. The
  relative cost reference includes the initial spread so it remains useful
  near zero minima. Tests cover objective and parameter scales, flat
  objectives, projected collapse, high-dimensional Rosenbrock, both
  precisions, and every dense backend version. A corpus audit across
  multiple starts and dimensions informed the precision-specific defaults.
  See the [migration guide](MIGRATING.md#nelder-mead-convergence-defaults)
  for settings, checkpoint changes, and the limits imposed by fixed floors.

- [x] **Revisit convergence defaults for LM, L-BFGS-B, bounded LM, and
  TRF ([#109](https://github.com/jolars/basin/issues/109)).**
  Bounded `f64` L-BFGS now uses projected-gradient
  tolerance `1e-9` and signed cost-reduction tolerance `10*f64::EPSILON`,
  after a bounded NIST investigation and a fix for nonfinite selected
  evaluations. The `f32` defaults are unchanged. The [NIST
  fixtures](crates/competitor-bench/fixtures/nist/README.md) remain
  available for future solver comparisons. See the [migration
  guide](MIGRATING.md#l-bfgs-b-stopping-defaults) for the individual knobs
  and accuracy tradeoffs. A fixed, source-informed comparison on both NIST
  starts and curated scaling cases supports retaining the existing LM, bounded
  LM, and TRF defaults: the tested relative policies saved evaluations but
  reduced independently verified recovery. Relative checks remain opt-in. The
  [decision and scaling guidance](MIGRATING.md#lm-and-trf-stopping-defaults)
  explain the formulas, evidence, and limitations; this decision does not
  claim scale-invariant convergence.

- [x] **Make structured termination reports part of ordinary results.** Ordinary
  `run()` returns an owned report, final state, and authoritative counts.
  Reports distinguish convergence, limits, targets, stalls, failures,
  cancellation, and application stops, retaining simultaneous criteria and
  their measurements. `SolverStep` separates completion from termination;
  composed runs retain inner failure details and use explicit policies for
  partial results. Exact continuation produces a new report, and typed
  callback aborts remain `Result::Err`. See the [migration
  guide](MIGRATING.md#structured-termination-reports) for the API, binding
  code mapping, observer lifecycle, and checkpoint changes.

- [x] **Make evaluation counters extensible.** Mark `EvalCounts` as
  non-exhaustive so future releases can add categories without breaking
  downstream code. Construct counters with `Default` and assign public fields;
  destructuring patterns include `..`.
  See [evaluation-counter migration](MIGRATING.md#evaluation-counters).

- [x] **Make results and diagnostic records extensible.** Mark optimization
  results, termination reports, diagnostic records, and their growing reason
  and phase enums as non-exhaustive. Provide result and report constructors,
  retain public field access, and document fallback matches and destructuring
  with `..`. See the [migration
  guide](MIGRATING.md#extensible-results-and-diagnostics).

- [x] **Make solver publication types extensible.** Mark `SolverStep` and
  `IncumbentRef` as non-exhaustive with complete public constructors and public
  fields. `SolverStep::partial` covers continuing partial steps, and `new`
  supports forwarding dynamic decisions. Future optional metadata must default
  to behavior compatible with existing constructors. See the [publication
  migration guide](MIGRATING.md#extensible-solver-publications).

- [x] **Set extension points for smaller stopping records.**
  `ConvergenceCriterion` and `ApplicationStop` are non-exhaustive, retain public
  fields, and provide straightforward construction, including the general
  `ConvergenceCriterion::new(test, evidence)`. Future optional metadata must
  default to behavior compatible with existing constructors. `Threshold`
  remains exhaustive as a complete mathematical comparison. See the
  [stopping-record migration guide](MIGRATING.md#extensible-stopping-records).

- [x] **Make line-search configurations extensible.** `Backtracking`, `Wolfe`,
  `MoreThuente`, and `HagerZhang` are non-exhaustive, with public fields,
  default constructors, and builders. Future controls must default to behavior
  compatible with existing constructors. See the [migration
  guide](MIGRATING.md#extensible-line-search-configurations) for constructors,
  public field assignment, and destructuring with `..`.

- [x] **Make state shape errors extensible.** `SimplexShapeError` and
  `PopulationShapeError` are non-exhaustive, with directly constructible
  variants and unchanged payloads and validation contracts. Matches include a
  fallback arm for future variants. See the [state shape error migration
  guide](MIGRATING.md#extensible-state-shape-errors).

- [ ] **Settle the extension policy for enum payloads.** An enum-level
  `#[non_exhaustive]` permits new variants but does not permit adding fields to
  existing variants. For evolving payloads, especially `ConvergenceEvidence`,
  choose between non-exhaustive variants with public constructors and new
  variants for richer evidence. Keep serialization compatibility a separate
  decision. Preserve exhaustive matching for deliberately closed contracts
  such as `BoundSide`, `ConstraintAxis`, `StepOutcome`, `TerminationStage`, and
  `PartialResultPolicy`, and literal construction for complete mathematical
  values such as `BinaryValue`, `Measurement`, and `LossEvaluation`.

- [x] **Clean up backend compatibility aliases in Basin 2.0.0.** Make
  unversioned `nalgebra`, `ndarray`, and `faer` features and their
  LAPACK/BLAS counterparts select the newest supported releases. Retain the
  `*_latest` features as deprecated synonyms and exact version features for
  pinned dependencies. Remove the backend-specific quasi-Newton state
  aliases in favor of `FirstOrderState<V, F>` with a solver-owned matrix.
  Preserve independent implementations for every enabled version and
  version-specific acceleration. Document the 1.x migration and retain
  simultaneous-version and downstream Cargo feature-unification tests.

- [x] **Remove bincode in Basin 2.0.0.** Drop legacy readers for unprefixed
  state checkpoints and version 1 exact checkpoints, along with the bincode
  dependency and compatibility-only tests. Retain the postcard formats and
  document [checkpoint migration
  requirements](MIGRATING.md#checkpoint-files) for users upgrading from 1.x.

- [x] **Migrate existing solvers to shared progress states.** Build on the
  validated [prototype](#state-api-prototype) and [1.x
  additions](#state-api-additions-for-basin-1x). Replace legacy public
  solver/state types and apply uniform evaluation categories and the
  accepted initialization and continuation contracts. Document replacement
  constructors, public types, trait bounds, stopping semantics, and
  serialized-format compatibility. See the [migration
  guide](MIGRATING.md#shared-progress-states).

- [x] **Decide how observers access solver diagnostics before finalizing 2.0.**
  Keep `Observe<S>` for progress and add optional `ObserveSolver<S, So>`
  hooks and `Executor::observe_solver` closures that borrow the solver at
  the same observation boundaries. Preserve solver ownership, modes, and
  mixed registration order without cloning workspace or triggering
  evaluations. `Stepper::solver()` remains available for applications
  driving their own loop. See [custom diagnostic
  logging](MIGRATING.md#observing-solver-diagnostics).

- [x] **Simplify and strengthen the full-form constraint API (tenet 4).**
  Share one canonical `ProblemConstraints` declaration between objectives and
  residuals, enforce solver support through its type, and let COBYLA consume
  every declared block directly. Separate constraint values, right-hand sides,
  linear matrices, and nonlinear Jacobians from the parameter representation.
  Statically absent blocks require no unused matrix or callback. Validate
  bounds, shapes, and counts with `ConstraintSolveError`, preserving original
  callback errors. Keep MA-LS sampling domains distinct from mathematical
  constraints, and require explicit local relaxation in bounded memetic runs.
  See [constraint migration](MIGRATING.md#declaring-constraints).

- [x] Remove the deprecated `TerminationCriterion` facility, all shipped
  criterion types and re-exports, `Executor::terminate_on`,
  `InnerExecutor::terminate_on`, composed `inner_terminate_on` methods,
  `run_loop`, and `ResumableInner::segment_criteria`. Preserve stopping
  semantics through structured reports, solver convergence setters, direct
  execution controls, and closure hooks.

- [x] Remove deprecated tolerance and algorithm-setting aliases, including
  scalar/root and line-search aliases.

- [x] Remove `BarrierMethod::new`, `AugmentedLagrangianMethod::new`, and
  `with_inner_grad_tol`, along with their implicit inner-gradient checks.
  Use `with_inner_solver` and the supplied solver's convergence settings.

- [x] Move remaining shared numerical calculations out of the compatibility
  criterion types and remove the compatibility-only tests and bridges.

- [x] **Make the `problems` feature opt-in.** Set `default = []` while retaining
  `problems = []` for benchmarks, examples, and other corpus consumers. Keep
  the existing default through Basin 1.x: removing it breaks downstream
  imports that rely on implicit activation. Update documentation and ensure
  tests, examples, benchmarks, and the WASM visualizer explicitly enable
  `problems` wherever they use the corpus.

- [x] **Retain soft evaluation limits for Basin 2.0.** `max_cost_evals`,
  `max_gradient_evals`, and `max_evaluations` remain checked after
  initialization and between iterations. Initialization and an in-progress
  iteration may exceed a threshold before the executor checks it. Keep
  budget checks outside callbacks to avoid per-evaluation overhead. A
  separate hard-cap API is outside the planned 2.0 scope.

- [x] **Retain callback-request evaluation counts for Basin 2.0.** Counters
  measure requests at the `Problem` boundary, with each fused category and
  submitted batch item charged even on failure. Internal adapter probes are
  excluded from existing counters and `total_work`. Keep future opt-in probe
  accounting separate. Document the implications for finite differences and
  execution budgets in [evaluation counters](MIGRATING.md#evaluation-counters).

## Deferred design

- [ ] **Add opt-in internal evaluation accounting when a consumer needs it.**
  Report underlying cost and residual probes separately from existing
  callback-request counters, including any savings from Jacobian coloring.
  Define accounting for failed and submitted probes, parallel evaluation,
  fused callbacks, and composed adapters without double counting. Preserve
  the existing meanings of `EvalCounts`, `total_work`, and evaluation budgets.

- [ ] **Define shared constraint-violation reporting (tenet 3).** Constrained
  MADS publishes the sum of squared positive violations in `SelectedState`,
  and SLSQP publishes the sum of absolute equality residuals and positive
  inequality violations in `SelectedFirstOrderState`. COBYLA publishes its
  maximum positive violation in `SelectedState`. Define a shared state
  capability with explicit semantics for the measure, scaling, associated
  iterate, and availability before initialization. Preserve each solver's
  numerical convergence semantics: SLSQP already combines feasibility with
  other convergence tests. Do not add a standalone `FeasibilityTolerance` to
  the executor: its stopping conditions combine with OR, so it could stop at
  the first feasible but nonoptimal iterate.
