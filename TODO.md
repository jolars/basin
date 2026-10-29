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
  `BoundedLevenbergMarquardt` in Basin 2.0 and `Trf` in Basin 1.x. The
  large-scale path remains a follow-up below.
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

- [x] **Investigate central-difference Jacobian overhead.** Compared matched
  stencils against Navette's Jacobian builder, following the
  [integration review](https://github.com/opticsWolf/Navette/pull/3#issuecomment-6431966233).
  Unbounded central Jacobians now avoid the base-residual evaluation, and
  fused finite-difference Jacobians reuse the base residual for forward and
  bounded stencils. Regression tests verify callback counts and numerical
  results across the dense backends, with checks for feasible probes and
  empty shapes.
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
  refinement. The [integer-domain API
  prototype](#integer-domain-api-prototype) establishes the declaration,
  rejection, and fixed-integer refinement contracts. Production sampling,
  mutation repair, and mixed-integer injection remain unimplemented.
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
  bracketing, bisection, safeguarded secant, Newton, and Halley methods, and
  TOMS 748 (`k = 2`). Root solvers retain direct fallible callbacks, signed
  function values, and bracket-based convergence. Bracketing is an explicit
  first stage.
- [ ] **Evaluate ITP for bracketed scalar roots.** Implement the Interpolate,
  Truncate, Project method of Oliveira and Takahashi (2021),
  [doi:10.1145/3423597](https://doi.org/10.1145/3423597), and benchmark it
  against Brent and TOMS 748. Verify its bisection-based worst-case
  iteration bound, including the zero-slack configuration, under the
  documented tolerance and floating-point safeguards.
- [ ] **Add Ridder's scalar root method.** Keep the direct fallible callback,
  signed-value, and bracket-preservation contracts. Use analytic cases and
  [SciPy's Ridder
  implementation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.ridder.html)
  as references, and cover extreme function scales and non-finite
  arithmetic.
- [ ] **Add unbracketed Newton and secant root solving.** Accept initial guesses
  without requiring a sign-changing interval, including for
  even-multiplicity roots such as `(x - 1)^2 = 0`. Require explicit residual
  validation rather than accepting small steps alone. Keep these convergence
  and result contracts distinct from bracketed solves, and test zero
  derivatives, stagnation, divergence, and non-finite evaluations. See
  [SciPy's Newton
  API](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.newton.html).
- [ ] **Add batched independent scalar solves when a consumer needs them.**
  Build on the direct scalar root and bracketing APIs, preserving separate
  root and minimization convergence semantics. See [elementwise
  optimization](https://docs.scipy.org/doc/scipy/reference/optimize.elementwise.html).
- [ ] **Add multivariate root solving when an integration needs it.** Start with
  Powell's hybrid method or damped Newton, followed by Broyden, then assess
  Anderson acceleration and Newton-Krylov. Require residual-based root
  validation: convergence of a least-squares objective can leave a nonzero
  residual. Promote this work if a downstream consumer needs it, including
  access to the final Jacobian for sensitivities. See [SciPy nonlinear
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

## Competitor benchmarks

Investigation dated 2026-10-22. Scope: Rust solver packages, with C libraries
considered when they already have Rust bindings. Keep every comparison in a Rust
harness. A C interface to a Fortran or C++ implementation does not by itself put
that implementation in scope.

The site's competitor page currently covers gradient descent, Nelder-Mead,
L-BFGS, NEWUOA, and unconstrained SLSQP. Four cases use Rosenbrock at `n = 2`;
NEWUOA uses Styblinski-Tang at `n = 5`. LM and COBYLA comparisons, all 27 NIST
regression datasets with two starts, and constrained SLSQP workloads already
exist in `crates/competitor-bench`, but are not exposed on the competitor page.
Prioritize those assets and uncovered algorithm families before adding more
libraries to the same cases. New packages below are research candidates; build
compatibility and numerical behavior still need verification.

### Measurement and publication prerequisites

- [ ] **Extend competitor results beyond objective-versus-time traces.** Record
  package and native-library versions or commits, algorithm variants,
  dimensions, starts, settings, thread counts, and timing boundaries. Give
  each configuration a stable case ID; the current solver/problem key cannot
  distinguish dimensions or starts. Include initialization consistently and
  report repeated solve timings separately from instrumented traces. Record
  actual objective, derivative, residual, and constraint calls, termination,
  independent solution checks, and budget overruns. Basin's iteration limits
  and NLopt's evaluation limits do not establish equal work. For constrained
  curves, track the best feasible objective and violation separately; an
  infeasible trial with a low cost must not look like a successful solve.
- [ ] **Broaden workloads and summarize reliability.** Keep a few readable
  convergence examples, then add time and evaluations to common accuracy and
  feasibility targets, success rates, and performance profiles across cases.
  Retain failures. Reuse NIST's certified values and both starts for
  fitting; add active-bound, rank-deficient, ill-conditioned, nonsmooth, and
  multimodal cases from existing fixtures. Sweep dimensions appropriate to
  each method and use multiple starts. For stochastic solvers, use a
  recorded seed set and uncertainty intervals; equal seed integers across
  different RNGs do not imply identical random draws. Start with serial runs
  and record backend versions, BLAS providers, and any unavoidable adapter
  work.

### First additions

- [x] **Publish the existing LM and COBYLA comparisons.** The competitor page
  shows exponential-fit LM against `levenberg-marquardt` and a constrained
  quadratic against `cobyla`, using 11 repeated convergence traces. Exposed
  settings match where possible, with algorithm differences documented.
  COBYLA reports best feasible objective and trial violation separately.
  Broader aggregate reporting remains part of the publication prerequisites
  above.

- [ ] **Expand comparisons using existing argmin and NLopt dependencies.** Use
  [argmin](https://docs.rs/argmin/0.11.0/argmin/solver/index.html) and the C
  implementations exposed through the existing [`nlopt` Rust
  bindings](https://docs.rs/nlopt/0.8.1/nlopt/). Start with bounded BOBYQA
  and constrained SLSQP. For constrained comparisons, check feasibility,
  stationarity, and complementarity where available.

  - [ ] **Compare bounded BOBYQA against NLopt.**
  - [ ] **Compare constrained SLSQP against NLopt.** Reuse HS71 and an analytic
    constrained quadratic.
  - [ ] **Compare BFGS against argmin.**
  - [ ] **Compare nonlinear CG against argmin with PR+.** Match Basin's update
    explicitly: its default is Hager-Zhang, while argmin 0.11 has no
    built-in Hager-Zhang update.
  - [ ] **Compare trust-region Steihaug against argmin.**
  - [ ] **Compare trust-region dogleg against argmin.**
  - [ ] **Compare particle swarm against argmin.** Reuse the existing PSO
    reference fixture, checking initialization, topology, and boundary
    handling before claiming a matched comparison.
  - [ ] **Compare Gauss-Newton against argmin.**
  - [ ] **Compare scalar methods against argmin.**
  - [x] **Compare COBYLA against NLopt.** Added native NLopt to the migration
    workload timings and verification, plus the published 30-variable
    constrained quadratic traces. Match starts, objective budgets, and
    radii; report feasibility, callback counts, and termination separately.
  - [ ] **Compare DIRECT against NLopt.** Distinguish the DIRECT variants.
  - [ ] **Compare augmented Lagrangian methods against NLopt.** Match inner
    solvers and account for all nested evaluation work.
  - [ ] **Add the `slsqp` crate as another Rust reference.** Record its shared
    NLopt lineage.

- [ ] **Benchmark actual L-BFGS-B with active bounds.** The [`lbfgsb`
  crate](https://github.com/ybyygu/rust-lbfgsb) wraps Stephen Becker's C
  translation of L-BFGS-B and is the preferred C-backed candidate. Its build
  uses a C compiler and bindgen/libclang. Also assess the native
  [`lbfgsb-rs-pure` crate](https://docs.rs/crate/lbfgsb-rs-pure/0.1.2),
  which advertises a v3.0 port; verify its numerical behavior before using
  it as a reference. Reuse existing parity inputs and certified solutions,
  including optima on box faces, fixed coordinates, and larger dimensions.
  Match memory size, line-search settings, and projected-gradient checks.
  The existing unconstrained L-BFGS chart and NLopt's L-BFGS are not
  coverage of Basin's L-BFGS-B algorithm.

  - [x] Publish one larger bounded example against the C-backed `lbfgsb`
    reference on the competitor benchmark page. The 20,000-variable extended
    Rosenbrock driver takes at least 20 ms per solve for both
    implementations on the measurement machine. Broader fixture coverage and
    assessment of `lbfgsb-rs-pure` remain open.

- [ ] **Add active CMA-ES against the Rust `cmaes` crate.** [`cmaes`
  0.2.2](https://docs.rs/cmaes/0.2.2/cmaes/) supplies active covariance
  adaptation and automatic restart strategies. Explicitly select negative
  recombination weights to match Basin's active update. Match population
  size, initial mean, sigma, and stopping policy, recording the differing
  nalgebra versions. Test ill-conditioned and nonseparable problems as well
  as multimodal problems over multiple seeds. Compare standalone CMA-ES
  first; report IPOP/BIPOP restarts and local injections separately. Do not
  infer bounded-CMA equivalence from unconstrained CMA-ES results.

  - [x] Publish standalone active CMA-ES against `cmaes` 0.2.2 on a
    40-variable rotated ellipsoid and Ackley, each with five fixed seeds.
    Match population, mean, sigma, target, and evaluation budget; report
    per-seed traces, success counts, actual objective calls, and backend
    differences. Broader workloads and restart comparisons remain open.

- [ ] **Add bounded least squares against `rmpfit`.** [`rmpfit`
  2.0.0](https://docs.rs/rmpfit/2.0.0/rmpfit/) is a pure-Rust CMPFIT port
  with analytic or numerical derivatives, parameter bounds, and fixed
  parameters. Compare with Basin's LM and bounded LM using existing NIST
  fixtures plus fitting cases with active bounds and fixed coordinates.
  Normalize half-RSS versus RSS, derivative settings, scaling, and stopping
  tests. Compare TRF on the same problems for solver effectiveness, while
  keeping the algorithm distinction explicit: CMPFIT is bounded LM, not TRF.

### Additional Rust packages

- [ ] **Compare constrained optimization with `optimization_engine`.**
  [`optimization_engine`
  0.12.0](https://docs.rs/optimization_engine/0.12.0/optimization_engine/)
  implements PANOC and augmented Lagrangian methods in Rust. Compare its
  rectangle-constrained solves with Basin's projected gradient and L-BFGS-B,
  and its outer constrained solves with Basin's augmented Lagrangian method.
  Match problem definitions, feasibility targets, and all inner work, while
  labeling different inner algorithms. This would cover practical
  constrained performance beyond the current unconstrained SLSQP example.
- [ ] **Evaluate Rust differential-evolution competitors.**
  [`math-differential-evolution`
  0.3.3](https://docs.rs/math-differential-evolution/0.3.3/math_audio_differential_evolution/)
  exposes classical strategies, binomial and exponential crossover, seeded
  runs, deferred updates, and SHADE/L-SHADE variants. Start with matched
  `rand/1/bin`, fixed mutation and crossover settings, population sizes, and
  no local polishing. Check initialization and bound repair in source. Its
  released manifest enables ndarray BLAS and Rayon, so establish the link
  provider and serial thread settings during preflight. Treat adaptive DE
  and population reduction as separate effectiveness comparisons. The older
  [`differential-evolution`
  crate](https://docs.rs/differential-evolution/0.2.2/differential_evolution/)
  is another self-adaptive candidate if its seed and precision controls fit
  reproducible comparisons.
- [ ] **Evaluate the Rust `dogleg` least-squares solver.** [`dogleg`
  0.1.1](https://docs.rs/dogleg/0.1.1/dogleg/) implements Powell's
  least-squares dogleg method with nalgebra and faer backends and an adapter
  for `levenberg-marquardt` problems. Reuse NIST fixtures and analytic
  Jacobians. Compare with Basin LM for solver effectiveness; a matched
  dogleg comparison needs a Gauss-Newton model rather than blindly using an
  exact objective Hessian with Basin's general trust-region solver. Keep
  this distinct from rectangular dogbox and reflective least squares.
- [ ] **Assess the native Rust `liblbfgs` port as a specialist baseline.**
  [`liblbfgs` 0.1.0](https://docs.rs/crate/liblbfgs/0.1.0) is a Rust
  implementation derived from the C library, despite its name. It offers
  L-BFGS and OWL-QN. Add an unconstrained L-BFGS comparison only if it
  contributes beyond argmin and NLopt, matching memory, line search, and
  fused objective/gradient callbacks. An OWL-QN comparison would require an
  explicit L1-regularized workload and is a different algorithm.

### Algorithm gaps to revisit after comparisons

Most immediate gaps are in benchmark coverage. Within this shortlist, no
matching reference has been verified for Basin's full TRF, LINCOA, or MADS;
comparisons with other algorithms should be labeled as solver effectiveness.
Keep existing correctness fixtures separate from the in-scope benchmark
contestants. The existing solver backlog already tracks sparse iterative TRF,
Jacobian coloring, COBYQA, general sparse constrained optimization, and dogbox.
Potential additions motivated by Rust competitors are CMA-ES restart policies,
adaptive DE, PANOC, least-squares-specific dogleg, and OWL-QN. Promote these
only when workloads show useful coverage beyond existing methods. These are
investigation leads, not measured performance findings or commitments to new
production solvers.

## General design

### Finite-difference performance

Source review of `core/numdiff.rs` and its child modules on 2026-10-27.
These are optimization candidates; speedups have not been measured. Preserve
stencils, step rules, feasible probes, error propagation, and request-count
semantics. Keep each entry's arithmetic order unchanged to preserve bitwise
results for deterministic callbacks. Verify changed kernels across supported
backends and scalar types.

- [x] **Establish finite-difference benchmarks.** Add focused cases for
  gradients, rectangular Jacobians, Hessians, and Hessian-vector products,
  covering owned and `*_into` calls, cheap and expensive callbacks, and small
  and large dimensions. Compare sequential builds with `parallel` at one and
  multiple threads. Measure raw callback counts and allocations separately
  from uninstrumented timings, including repeated calls with reused outputs.
  See the [benchmark protocol](crates/competitor-bench/NUMDIFF.md).
- [x] **Reuse base costs in fused derivative calls.** Pass the cost already
  computed by `cost_and_gradient_into` into forward-gradient and bounded
  probe kernels. Share it with the Hessian in the three-output callback,
  and route owned fused methods through the same implementation. An
  unbounded forward cost-and-gradient request should use `n + 1` underlying
  cost calls instead of `n + 2`. Add callback-count and numerical regression
  tests for both wrappers, stencil choices, and empty parameter vectors.
- [x] **Remove redundant Hessian assembly storage.** Both Hessian kernels
  allocate explicit upper-triangle index pairs, packed values, and a full
  temporary matrix before filling the output. Generate indices implicitly
  or in row blocks, and fill the backend matrix from packed values directly.
  Compare allocation counts, requested bytes, and timings at larger `n`;
  preserve exact symmetry and each entry's arithmetic order.
- [x] **Use scalar probe results for bounded gradients.** Replace the
  `vec![f]` callback adapter and one-element column vectors with a scalar
  kernel that shares the bound-aware stencil rules. Cover `f32` and `f64`,
  fixed coordinates, adjacent-float bounds, and one-sided central stencils.
  Measure allocations with a cheap objective to expose the adapter overhead.
- [x] **Reduce gradient result staging.** Stage up to 32 scalar results on
  the stack and allocate larger buffers at their final size. Fill parallel
  staging slots directly instead of collecting per-task result vectors.
  Keep output publication after successful probes to preserve shape
  replacement and unchanged outputs on errors. Benchmarked repeated calls
  with sized outputs on all four backends, sequentially and with Rayon.
- [ ] **Reduce Jacobian column allocation and copying.** Both unbounded
  Jacobian kernels clone residual vectors into separately allocated columns,
  then copy those columns into the backend matrix. Compare flat or chunked
  staging, reusable column storage, and direct sequential matrix fills.
  Include tall and wide Jacobians, storage orders across backends, and
  callbacks that override `residual_into` to reuse their output buffers.
- [x] **Reuse HVP probe and gradient scratch.** The HVP helpers clone `x`
  to construct directional probes and temporary gradients on every call.
  Measure repeated products at one point and assess reusing those buffers
  through caller-owned scratch where justified. Preserve zero-direction
  behavior, output resizing, thread safety, and the wrappers' existing
  `Clone` and `Copy` contracts.
- [ ] **Tune Rayon task and scratch granularity.** The parallel map helpers
  use `map_init`, which initializes scratch per job rather than guaranteeing
  one buffer per worker. Count probe-buffer clones and compare minimum job
  lengths or chunked tasks for gradients, Jacobians, and Hessians. Measure
  small cheap cases as well as expensive callbacks before choosing a serial
  fallback or exposing a granularity control. Correct the per-thread wording
  in the helper documentation.
- [ ] **Remove forward-Hessian probe barriers.** Schedule base,
  single-coordinate, and two-coordinate evaluations together, then combine
  their results. The probe evaluations do not depend on each other; their
  final arithmetic needs the shared results. Compare low-dimensional cases
  and uneven callback costs across thread counts, keeping probe counts fixed.
- [ ] **Overlap base evaluations with derivative probes.** Evaluate `f(x)`
  or `r(x)` concurrently with the independent probes for forward gradients
  and Jacobians, then form differences once the base is available. Apply the
  same assessment to the central Hessian's base cost. Preserve residual shape
  checks, empty-input behavior, base reuse, and hard-error propagation.
- [ ] **Overlap independent work in fused derivative calls.** Compare
  concurrent cost and central-gradient evaluations, overlapping gradient and
  Hessian probe sets, and concurrent base-residual and central-Jacobian
  evaluations. Reuse shared base values in forward stencils first. Measure
  the benefit when inner derivative loops already use the Rayon pool, and
  account for any additional vector or matrix thread-safety bounds.
- [ ] **Parallelize independent HVP gradient evaluations.** Compare a
  feature-gated `rayon::join` for the two gradient calls in both forward and
  central HVPs. Include expensive analytic gradients and synthesized
  gradients to expose nested overhead. Explicitly address compatibility
  before adding `Send` or `Sync` bounds to the currently unrestricted HVP
  free functions, and retain the zero-direction shortcut.
- [ ] **Evaluate parallelism within central stencils.** Expose the positive
  and negative gradient and Jacobian probes, and the four off-diagonal
  Hessian probes, as separate tasks when the existing column or entry tasks
  cannot occupy the pool. Compare flattened probe batches with selective
  joins, including extra scratch memory, cloning, and scheduling costs.
  Keep all new threading behind `parallel` and preserve the sequential WASM
  path.

### In-place problem evaluation

- [ ] **Make evaluation buffer reuse automatic in Basin 4.0.** Replace the
  compatible opt-in evaluation storage on solvers and the robust
  least-squares adapter with typed storage by default. Basin 4.0 may change
  public type parameters and their `Send`, `Sync`, `Clone`, and `Copy`
  implementations as needed. Document the migration from the opt-in builders
  and preserve callback dispatch, evaluation counts, fresh initialization,
  and exact continuation.

### Iterative-TRF API prototype

- [x] **Prove an operator-only reflective model and robust composition.** The
  [executable
  prototype](crates/basin/src/core/least_squares/iterative_trf_api_prototype.rs)
  is a test-only child of the robust least-squares module. No types,
  counters, or methods from the experiment enter Basin's public API.
  Existing dense `TrustRegionReflective` keeps its constructor, bounds, and
  behavior.

Use separate `JacobianProduct: Residual` and
`TransposeJacobianProduct: Residual` capabilities. Their signatures are
`jacobian_product(&self, x: &Self::Param, v: &Self::Param) -> Result<Self::Output, Self::Error>`
and
`transpose_jacobian_product(&self, x: &Self::Param, v: &Self::Output) -> Result<Self::Param, Self::Error>`.
Neither extends `Jacobian` nor introduces a matrix associated type. Products
differentiate the raw residual at `x`; they are pure and linear in `v`. Residual
length stays fixed during a solve. Parameter and residual spaces may use
distinct vector types and lengths. Do not blanket-implement the capabilities for
every `Jacobian`: an explicit dense-product adapter can provide that convenience
without blocking custom analytic products on problems that also supply a dense
Jacobian.

The intended production entry is a separate
`IterativeTrustRegionReflective<F = f64>` with `PointState<V, F>` and the
existing box-problem declaration. It will require both product capabilities and
vector operations, without `Jacobian`, `MatrixIndex`, `ScaleRowsInPlace`, matrix
factorization, or matrix-entry access. This experiment implements its model
preparation and operator consumption signatures, not a `Solver` impl or settings
builders. Production LSMR tolerances, iteration limits, and regularization must
be explicit algorithm controls. Automatic Jacobian-column scaling would require
a separate capability or explicit user scales; it must not introduce a hidden
matrix-entry requirement.

The prototype establishes these contracts:

- The model freezes the accepted parameter and robust row factors. A borrowed
  linearization ties them to one counted objective. Repeated products and trial
  residual evaluations retain that model; accepting another point rebuilds it.
  Production solver-owned workspace must preserve this rule across rejected
  trials and exact continuation.
- Raw and `RobustLeastSquares` evaluations use separate internal capability
  implementations, as the dense NLLS drivers do. The robust adapter continues to
  expose its true objective rather than implementing `Residual`. The prototype
  reuses its existing loss and scale calculations: with row factors `a` and
  model residual `r_tilde`, the actions are `diag(a) J v` and `J^T diag(a) u`.
  Thus their composition supplies safeguarded curvature, while applying the
  transpose to `r_tilde` gives the robust gradient. Trial reductions use the
  true robust cost, not the model residual norm.
- The reflective augmented operator has actions for
  `A = [diag(a) J diag(d); diag(sqrt(c))]` and its transpose. Its two output
  blocks occupy residual and parameter space. Fixed coordinates use `d = c = 0`.
  These actions provide the inputs needed by LSMR and a small subspace model
  using vector storage, with no materialized normal matrix. Tests supply the
  scaling diagonals; the production driver must compute Coleman–Li scaling,
  eliminate fixed coordinates, and select feasible steps.
- Count attempted raw residual, Jv, and transpose-Jacobian requests separately,
  before invoking callbacks, including failures. Robust weighting adds no
  evaluations. Do not charge matrix `jacobian_evals` for products. Internal
  probes stay excluded under the existing wrapper-boundary contract. The
  prototype's `OperatorCounts` pairs real `EvalCounts` with two proposed fields
  and tests totals, deltas, and composition. Production must add those fields to
  the non-exhaustive `EvalCounts` and integrate state mirrors, budgets,
  checkpoint/serde storage, and inner roll-ups before shipping the solver. The
  experiment does not prove that executor integration yet.
- Callback errors propagate unchanged. Invalid initial residuals or loss models
  cannot produce a linearization; non-finite trial residuals yield infinite cost
  without replacing the model. Shape and non-finite derivative guards are
  assertions in this model consumer. The production driver must map numerical
  failures to its structured failure reports.

Analytic checks cover rectangular nonlinear products, adjoints, augmented
curvature, fixed coordinates, all five built-in losses at nonunit scale,
rejected trial reuse, refreshed accepted points, and attempted-request counts.
Tests run both precisions on every enabled dense backend version and also use
distinct Rust types for parameter and residual space. Run
`cargo test -p basin --lib iterative_trf_api_prototype`, optionally adding
`--features nalgebra,ndarray,faer`.

The operator design follows [SciPy 1.16.2's TRF
implementation](https://github.com/scipy/scipy/blob/v1.16.2/scipy/optimize/_lsq/trf.py)
and [its operator and robust-scaling
helpers](https://github.com/scipy/scipy/blob/v1.16.2/scipy/optimize/_lsq/common.py).
LSMR, the two-dimensional subspace solve, reflective step selection, sparse
adapters, coloring, and production accounting/checkpoint integration remain
future work. This prototype establishes the additive API path, not numerical
convergence or performance of a large-scale solver.

### Constrained-DE API prototype

- [x] **Prove selection-aware population storage and constrained injection.**
  The [executable
  prototype](crates/basin/tests/constrained_de_api_prototype.rs) uses the
  real executor, inner executor, canonical problem declarations, and
  evaluation counters. Its support types remain test-only. Scripted
  populations exercise selection without implementing mutation or sampling.

The additive production path is a separate `ConstrainedDe<F>` constructed with
`ConstrainedDe::new(de: De<F>)`, paired with `SelectedPopulationProgress<V, F>`.
A sibling
`ConstrainedDeInject::with_inner_solver(outer: ConstrainedDe<F>, inner: I)` can
compose local refinement. These are proposed signatures, not public exports.
Ordinary `De`, `DeInject`, and `PopulationProgress` retain their existing types
and behavior.

The prototype establishes these contracts:

- Member records pair parameters with componentwise violations and an optional
  objective. Checked replacement validates the entire population before changing
  it and retains member order. A separate population capability exposes these
  records without promising a cost for every member. Seeds have no evaluated
  records.
- Feasible members have evaluated objectives. An all-infeasible population
  evaluates its selected representative before publication, leaving other
  infeasible objectives optional. `State::cost()` therefore remains an actual
  evaluated objective. A representative callback error propagates. This choice
  requires an objective that can evaluate an infeasible representative; an
  objective defined only on feasible points needs a different, future
  availability contract.
- Feasible trials beat infeasible members. Feasible pairs compare objective
  values; infeasible pairs use componentwise non-worsening violations, following
  [SciPy 1.16.2's Lampinen-based
  selection](https://github.com/scipy/scipy/blob/v1.16.2/scipy/optimize/_differentialevolution.py).
  Representatives use the lowest feasible objective or, before feasibility, the
  smallest unscaled sum of violations. Exact ranking ties retain the historical
  incumbent. This sum is a prototype policy, not a common constraint-reporting
  norm.
- Explicit selection events retain independent point, objective, and violation
  snapshots. The executor stamps iteration and all raw counts at publication,
  including clean mid-step stops. Republishing a selection preserves its age.
  The state implements `IncumbentState` but neither `ObjectiveIncumbentState`
  nor `PopulationState`; objective targets and objective stalls cannot attach to
  it.
- An injection inner must accept the original constrained problem. Seeds follow
  the representative ranking. Every returned point is repaired to the box and
  its constraints are reevaluated before applying the same trial selection rule.
  The prototype permits infeasible local starts and partial results; it does not
  permit a local solver to discard nonlinear blocks.
- Nonempty nonlinear callback attempts charge `residual_evals`, including
  failures and malformed output. Zero-count blocks skip callbacks. Objective and
  derivative calls preserve their existing categories. Same-problem inner
  execution shares the outer wrapper and never adds its counts twice.
- Fresh initialization reevaluates member seeds and resets script position and
  bookkeeping. Exact checkpoints preserve the script position, members,
  refinement schedule, inner solver, selection metadata, and counts. Tests cover
  both precisions on all four dense backends and serialized `Vec` checkpoints.

The fixture exercises finite boxes and nonlinear inequalities. Other constraint
blocks, feasibility tolerances, production mutation and sampling, and nonlinear
relaxation adapters remain future work. The `De` configuration is retained as
opaque input to the wrapper; the script does not interpret its settings or
exercise its RNG. Run the experiment with
`cargo test -p basin --test constrained_de_api_prototype`, optionally adding
`--features nalgebra,ndarray,faer,serde,parallel`.

### Integer-domain API prototype

- [x] **Prove an additive domain declaration and safe local refinement.** The
  [executable
  prototype](crates/basin/src/core/constraint/model/integer_domain_api_prototype.rs)
  extends the sealed canonical declaration inside a test-only child module.
  It uses the real problem traits, adapters, executor, and inner executor.
  No prototype types or methods enter the public API.

Integrality belongs in the canonical problem-side declaration, alongside bounds
and constraint blocks. The proposed public spelling is
`ConstraintSet<B, L, N, D = ContinuousDomain>`, with an explicit builder that
changes the domain type. Existing declarations retain their continuous meaning.
Do not add a required `Domain` associated type to every problem.

The prototype wraps the existing constraint set in
`DomainConstraints<C, D = ContinuousDomain>` rather than changing the public
`ConstraintSet` type. This demonstrates a separate additive implementation path
through the sealed `ConstraintDescription` trait. It does not prove source or
serialization compatibility for the proposed fourth parameter. Production
integration must check generic builders, borrowed views, and serialization
before choosing that spelling. Existing continuous declarations and DE runs
remain unchanged in this experiment.

The prototype establishes these contracts:

- Bounds describe intervals; a separate coordinate mask declares integrality.
  The mask and bounds have the parameter's length and remain fixed during a
  solve. A borrowed declaration shares both the original bounds and mask.
  Replacing bounds preserves the domain block. Finite-difference and robust-loss
  adapters preserve the complete declaration and its type.
- Constraint-aware solvers reject unsupported integer declarations through their
  trait bounds. Compile-time checks cover continuous DE, projected Nelder-Mead,
  `DeInject`, and `RelaxedInner`, including relaxed injection. Ordinary
  unconstrained solvers ignore these declarations and may evaluate fractional
  coordinates. The box-relaxation adapter is not permission to discard an
  integer domain. Future constraint transformations must consume only blocks
  they support and retain any unconsumed domain declaration.
- Initialization validates finite, ordered bounds and requires
  `ceil(lower) <= floor(upper)` for every integer coordinate. The effective
  integer endpoints must lie in the consecutive exactly representable range:
  `[-2^24, 2^24]` for `f32` and `[-2^53, 2^53]` for `f64`. This deliberately
  conservative prototype policy rejects wider integer intervals, including
  singletons outside that range. Seeds must already satisfy bounds and
  integrality; validation never silently rounds them or evaluates the objective.
- Continuous local refinement requires an explicit adapter that fixes integer
  coordinates at the selected member's values. The prototype eliminates those
  coordinates and fixed continuous coordinates, runs projected Nelder-Mead on
  the remaining box, and reconstructs the full parameter before every callback.
  All-integer or otherwise fully fixed problems skip refinement without
  evaluations. Rounding only the returned point is insufficient because
  intermediate callbacks must also preserve integrality.
- The adapter preserves original callback errors and folds its evaluation counts
  into the outer problem exactly once, including failing attempts. Refinement
  returns its inner termination report with the matching point and cost. An
  iteration limit retains partial progress; production injection must apply the
  normal explicit inner-stop policy and outer selection rule.

Tests exercise both precisions on each enabled dense backend version. Run
`cargo test -p basin --lib integer_domain_api_prototype`, optionally adding
`--features nalgebra,ndarray,faer,parallel,serde`.

Production integer-aware DE, integer sampling and mutation repair, integer
checkpoint storage, nonlinear constrained refinement, and derivative-based
reduced problems remain future work. An explicit continuous relaxation that
allows fractional objective calls would require its own problem transformation
and callback-domain contract; `RelaxedInner` does not provide that contract.

### State API prototype

- [x] **Prototype shared progress states with solver-owned machinery.** Follow
  the [state and lifecycle
  contracts](CONTRIBUTING.md#state-and-lifecyclecontracts). Exercise BFGS
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

## Basin 3.0

Upcoming breaking changes for Basin 3.0.

## Deferred design

- [ ] **Add opt-in internal evaluation accounting when a consumer needs it.**
  Report underlying cost and residual probes separately from existing
  callback-request counters, including any savings from Jacobian coloring.
  Define accounting for failed and submitted probes, parallel evaluation,
  fused callbacks, and composed adapters without double counting. Preserve
  the existing meanings of `EvalCounts`, `total_work`, and evaluation
  budgets.
