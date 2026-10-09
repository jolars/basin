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
- [ ] **Add integer variables to differential evolution.** Define a problem-side
  domain contract, validate that each integer interval contains an integer,
  and preserve integrality through sampling, mutation repair, and local
  refinement. Specify compatible `DeInject` inner solvers before enabling
  mixed-integer local search.
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

- [ ] **Revise solver convergence defaults
  ([#109](https://github.com/jolars/basin/issues/109)).** Follow the
  [multi-session investigation plan](#convergence-defaults-investigation)
  below. Cover every solver in Basin, including scalar roots, stochastic
  methods, aliases, and composed solvers. Use established solver-specific
  policies as baselines, then calibrate and independently validate their
  accuracy and work before selecting 2.0 defaults. Uniformity means clear
  semantics and comparable expectations, not identical tolerance values or
  guaranteed objective accuracy. Preserve existing defaults throughout Basin
  1.x and document explicit settings that recover its behavior.

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

### Convergence defaults investigation

Treat this as a sequence of reviewable sessions on a dedicated development
branch from `main`. Each phase produces evidence for the next; a corpus sweep
alone does not settle the defaults. Every solver needs a recorded disposition,
including justified decisions to retain its defaults or rely on execution
budgets. Do not promise universal scale invariance or interpret a local stop as
a certificate of global optimality.

Progress: the [inventory](dev/convergence-defaults/inventory.md) has an initial
source-checked stopping record for all 47 public solver names, and the
[dependency audit](dev/convergence-defaults/dependencies.md) covers line
searches, bracketers, and principal subproblems. Reference surveys and draft
candidate policies cover all 47 names; the reviewed [CDP-1
protocol](dev/convergence-defaults/protocol.md) fixes the experimental design.
Step 5 has an initial [measurement harness](dev/convergence-defaults/harness.md)
and [nine analytic validation
runs](dev/convergence-defaults/runs/2026-10-09-analytic-001.md), plus an
explicit [forward-difference stopping
configuration](dev/convergence-defaults/runs/2026-10-09-forward-001.md). NIST
model adapters, full pilots, calibration, solver decisions, and verification
remain open; the coverage boxes below require all applicable stages.

- [x] **1. Establish the branch and durable session records.** Created the
  `convergence-defaults` branch and temporary development documents under
  [dev/convergence-defaults/](dev/convergence-defaults/README.md):
  `README.md` for status and session handoffs, `inventory.md` for solver
  coverage and references, `protocol.md` for experimental design, and
  `decisions.md` for conclusions and unresolved questions. Keep run
  manifests and concise result summaries alongside them; put bulk traces and
  exploratory output in ignored `target/convergence-defaults/`. Each session
  records its starting revision, commands, evidence, decisions, open
  questions, and next concrete task. Distinguish proposals from accepted
  decisions and revisit decisions when new evidence contradicts them.

- [ ] **2. Inventory every solver and its stopping behavior.** Reconcile
  `crates/basin/src/lib.rs`, the public `solver` and `root` modules, and the
  web solver catalogue. Track references, current defaults, candidate
  policies, experiment status, decision, implementation, and verification
  for each solver. Give aliases and configurable modes separate entries
  wherever their stopping behavior differs. Reconcile the inventory again
  before completion so solvers added during the investigation are included.
  The initial coverage checklist is:

  - [ ] Scalar minimization: `Brent`, `BrentDerivative`, `GoldenSection`.
  - [ ] Scalar roots: `BrentRoot`, `SecantRoot`, `NewtonRoot`, `HalleyRoot`,
    `Toms748Root`.
  - [ ] First-order, quasi-Newton, and Newton methods: `GradientDescent`,
    `ProjectedGradientDescent`, `NonlinearCg`, `Bfgs`, `Lbfgs`, `Lbfgsb`,
    `TrustRegion`, and `Sgd`.
  - [ ] Least squares: `GaussNewton`, `LevenbergMarquardt`,
    `LevenbergMarquardtQr`, `Trf`, and `TrustRegionReflective`.
  - [ ] Derivative-free local methods: `NelderMead`, `Newuoa`, `Bobyqa`,
    `Lincoa`, `Cobyla`, `Mads`, and `SolisWets`.
  - [ ] Other constrained methods: `Slsqp`, `BarrierMethod`, and
    `AugmentedLagrangianMethod`.
  - [ ] Global and population methods: `Direct`, `Gbnm`, `RandomSearch`,
    `SimulatedAnnealing`, `CmaEs`, `BoundedCmaEs`, `De`, `GlobalBestPso`,
    and `Ssga`.
  - [ ] Composed methods: `BasinHopping`, `CmaInject`, `BoundedCmaInject`,
    `DeInject`, `MaLsCh`, `MaLsChCma`, and `MaLsChSw`.

  Cover constrained modes, trust-region strategies, damping choices, stochastic
  updates, and inner-solver configurations where they affect stopping. Audit
  line searches, root/minimum bracketers, and subproblem tolerances as
  dependencies, keeping their acceptance or completion tests distinct from
  convergence of the enclosing solve. Check these coverage boxes only after all
  applicable decisions and verification are complete.

- [x] **3. Survey references and specify candidate policies.** Record versioned
  primary documentation, research, and reference implementations. Start with
  [SciPy](https://docs.scipy.org/doc/scipy/reference/optimize.html),
  [Ceres](https://ceres-solver.readthedocs.io/latest/nnls_solving.html),
  [NLopt](https://nlopt.readthedocs.io/en/stable/NLopt_Algorithms/), and
  algorithm-author implementations where available. Document exact formulas,
  norms, scaling, observation stages, AND/OR composition, enabled defaults,
  precision assumptions, and failure/no-progress safeguards. Equal tolerance
  names do not establish equivalent mathematics or algorithm variants.
  Initial candidates include paired absolute simplex-size and cost-spread
  tests for Nelder-Mead, projected-gradient or normalized cost-reduction
  tests for L-BFGS-B, MINPACK-style orthogonality, model-reduction, and
  scaled-step tests for LM, and cost and step tests alongside the
  bound-scaled gradient test for full TRF. Distinguish
  `TrustRegionReflective` from the legacy bounded-LM `Trf`. Review the
  candidate set and evidence gaps before launching large sweeps; these
  examples do not limit the solver inventory. The [step 3
  review](dev/convergence-defaults/review-step3.md) covers all 47 names and
  records the evidence gates before large sweeps.

- [x] **4. Define and review the experimental protocol.** Specify success
  independently of each solver's stopping predicate. Use multiple objective
  accuracy targets with known or independently validated reference values,
  explicit absolute and relative scales, and attainable precision floors.
  Add stationarity and feasibility checks for applicable problems, parameter
  error where identifiable, and bracket/root accuracy for scalar roots.
  Distinguish a different local optimum from premature termination. For
  stochastic and global methods, assess repeated-run success and quality
  within budgets without treating population collapse as global convergence.
  Use data profiles and work to reach common targets as methodological
  references: [Moré and
  Wild](https://www.mcs.anl.gov/~wild/papers/2009/JJMSMW07.html) and
  [COCO](https://numbbo.github.io/coco-doc/perf-assessment/).

  Stratify Basin's problem corpus by solver applicability, family, dimension,
  constraints, and conditioning. Obtain the reporter's NIST StRD harness and
  include both supplied starts; independently assemble those cases if the
  harness remains unavailable. Include zero and nonzero residual minima,
  objective rescaling and offsets, parameter rescaling, active bounds, fixed
  coordinates, rank deficiency, non-finite inputs, and stagnation away from a
  solution. Cover analytic and finite-difference derivatives and noisy
  evaluations where supported. Hold out whole problem families, keeping their
  dimensions, starts, and transformed copies in the same partition. Fix family
  weights, budgets, seeds, target grids, and selection criteria before tuning.
  Define how missing applicable problems will be supplied; add new corpus
  problems through the project workflow. The [CDP-1
  protocol](dev/convergence-defaults/protocol.md), [case
  register](dev/convergence-defaults/cases.md), and [step 4
  review](dev/convergence-defaults/review-step4.md) fix the design and record
  gates before calibration. All 27 public NIST input datasets and both starts
  are assembled; executable model adapters are validated in step 5, while
  reference refinement and precision certificates remain open.

- [ ] **5. Build a reproducible measurement harness and pilot it.** Extend
  `crates/competitor-bench` using the existing convergence traces and LM
  stopping probes. Record returned-point quality, all passing stopping
  criteria and measurements, termination stage, accepted/rejected steps,
  iterations, authoritative evaluations by kind, underlying
  finite-difference calls, and elapsed time. Separate verification work from
  solve costs, and distinguish returned points from best sampled trials.
  Keep failures, budgets, no-progress stops, and infeasible results in the
  summaries. Preserve numerical safeguards and account for inner-solver and
  bracketing work. Validate the harness on analytic cases and matching
  references before running the corpus. Start with the issue's Nelder-Mead,
  L-BFGS-B, LM, and TRF cases, but retain the full coverage checklist.

  The analytic measurement foundation, all 27 NIST model adapters, and the
  [development reference preflight](dev/convergence-defaults/nist-reference.md)
  are implemented. The preflight certifies 15 local references and 321 of 330
  analytic target combinations; nine `f32` targets remain pending. The [LM/TRF
  measurement pilot](dev/convergence-defaults/least-squares-pilot.md) now covers
  both factorizations and damping modes, legacy TRF, and full TRF on the
  development NIST subset in both precisions. The [bounded full-TRF analytic
  extension](dev/convergence-defaults/runs/2026-10-09-bounded-trf-001.md) adds
  active bounds, fixed coordinates, and a stationary nonminimum control. The
  [robust-loss pilot](dev/convergence-defaults/runs/2026-10-09-robust-ls-001.md)
  adds Huber, soft-L1, and Cauchy measurements with known minima and paired
  default continuations. The [legacy TRF finite-model safeguard
  recheck](dev/convergence-defaults/runs/2026-10-09-trf-finite-model-001.md)
  removes repeated non-finite predictions, with main and 1.x fixes proposed.
  Robust stopping composition, remaining loss/constraint variants, additional
  backends, full inner-work accounting, and other solver families retain gates.

- [ ] **6. Calibrate by solver family and validate independently.** Compare
  current defaults, reference policies, and a small logarithmic sweep of
  defensible candidates. Vary enabled criteria and composition as well as
  thresholds. Where thresholds only affect termination, record candidate
  stopping points along a stricter run; confirm shortlisted policies with
  actual solves and correct observation stages. Measure early termination,
  improvement available under stricter continuation, work after first
  attaining each target, and final quality. Compare attainable targets
  across precisions without blindly transferring double-precision constants
  or epsilon formulas to `f32`. Use paired seeds and report uncertainty for
  stochastic methods. Validate finalists across every supported backend.

  Prefer a simple policy in a stable region of the accuracy/work tradeoff.
  Inspect worst cases and family-level results rather than selecting on a pooled
  average. Record each solver's recommended policy, alternatives, evidence for
  retaining or changing defaults, precision behavior, and known limitations.
  Review those decisions before implementation. Evaluate frozen candidates on
  held-out families; if those results cause retuning, record that the set has
  become development data and arrange fresh validation.

- [ ] **7. Implement and verify the reviewed decisions.** Add regression tests
  before behavioral fixes and implement one solver family at a time. Keep
  numerical no-progress safeguards and execution controls distinct from
  convergence. Preserve fresh-run resets, exact continuation, and diagnostic
  observation stages. Verify outer/inner stopping interactions and partial
  results for composed solvers. Account for any new vector capabilities
  required by default checks, especially simplex size, without silently
  dropping backend support. Run focused tests during development, then the
  applicable formatting, pure-Rust tests, clippy, rustdoc, WASM, and web
  checks from `AGENTS.md`. Cover both `f32` and `f64`, supported backend
  versions, disabled checks, exact-zero tolerances, and relevant degenerate
  inputs.

- [ ] **8. Close the inventory and publish the lasting rationale.** Require a
  supported disposition and completed validation for every solver and
  applicable variant, even when no default changes. Rerun the frozen
  protocol against the final implementation and explain material deviations
  from the selected policies. Document formulas, scaling, defaults,
  composition, termination meaning, and precision limitations in rustdoc and
  the relevant guides. Add migration settings that recover 1.x behavior and
  synchronize the web catalogue when references or backend support change.
  Retain the reusable harness, regression cases, reproduction manifests, and
  concise decision rationale in permanent repository locations before
  removing the temporary development documents. Leave unresolved decisions
  visible and keep the parent task open until the full inventory is
  complete.

## Deferred design

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
