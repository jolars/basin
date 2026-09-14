# TODO

Ordered by recommended sequence.

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
- [ ] **Add opt-in state capabilities.** Use new interfaces for checked record
  access, raw evaluation counts, and explicit incumbent-selection semantics.
  Bind new controls to the capabilities they need; preserve existing readers
  and stopping behavior.

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

## Performance investigations

### globalsearch comparison (2026-09-14)

Compared Basin 1.11.0 with argmin 0.11.0 through globalsearch's public local
solver adapters at commit `70ad94205235a682d9376663a858be4aa3cb3b53`. The
release benchmark used five problems, 24 shared starts per problem, and nine
timing rounds on a Ryzen 9 7900. Times below compare identical starts where both
backends evaluated a point with `f(x) - f* <= 1e-6`, with limits of 1,000 outer
iterations and 20,000 objective evaluations. Native stopping tests were disabled
for this comparison; reaching the target at a trial point does not certify a
returned solution or stationarity.

Steihaug was slower in Basin only on the ill-conditioned quadratic:

  | Problem                        | Basin relative to argmin |
  | ------------------------------ | ------------------------ |
  | Sphere, 20D                    | 1.21x faster             |
  | Ill-conditioned quadratic, 20D | 1.93x slower             |
  | Rosenbrock, 2D                 | 2.19x faster             |
  | Rosenbrock, 20D                | 1.74x faster             |
  | Six-hump camel, 2D             | 3.02x faster             |

Across these five problems, Steihaug's geometric mean speedup was 1.49x, with
103/120 target hits for Basin versus 100/120 for argmin. These measurements
include adapter overhead and use inexpensive analytic derivatives.

- [x] **Investigate Steihaug's inner CG stopping rule.** Added
  `Steihaug::with_forcing_parameters(kappa, theta)` for adaptive or fixed
  residual thresholds, preserving the default forcing rule and iteration
  cap.
- [x] **Measure Steihaug with expensive derivatives.** Reproduced the evaluation
  gap. Added Hessian work reverses the timing advantage at about 0.5 µs/call
  in 2D Rosenbrock and 1.5 µs/call in 20D, on shared successful starts; equal
  work in both derivative callbacks roughly halves those thresholds. See the
  [measurements and reproducible probe](crates/competitor-bench/investigations/steihaug-derivatives/README.md).
- [x] **Investigate gradient descent's extra evaluations on sphere.** Reproduced
  6 objective / 5 gradient calls versus argmin's 3 / 2. One extra pair is an
  accepted-point reevaluation; the remaining two arise from Basin's
  reference-compatible 0.4995 step versus argmin's 0.5. Both initialize at 1.
  Verified against the original Fortran line search and 72 deterministic starts.
- [x] **Reuse gradient descent's accepted line-search evaluation.** Plain descent
  adopts retained values through `next_with_evaluation`, reducing the sphere
  run to 5 objective / 4 gradient calls without changing trial points. Searches
  without retained values and momentum steps still evaluate the actual iterate.

Local artifacts: [report and
methodology](../globalsearch-rs/target/backend-comparison/REPORT.md),
[reproducible
harness](../globalsearch-rs/target/backend-comparison/src/main.rs), and
[plots](../globalsearch-rs/target/backend-comparison/plots/index.html). These
live in globalsearch's ignored `target/` directory.

## Basin 2.0

- [ ] **Migrate existing solvers to shared progress states.** Build on the
  validated [prototype](#state-api-prototype) and [1.x
  additions](#state-api-additions-for-basin-1x). Replace legacy public
  solver/state types and apply uniform evaluation categories and the
  accepted initialization and continuation contracts. Document replacement
  constructors, public types, trait bounds, stopping semantics, and
  serialized-format compatibility.

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
