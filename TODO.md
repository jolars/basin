# TODO

Ordered by recommended sequence.

## General design

- [x] **Investigate the COBYLA performance gap observed during the
  GlobalSearch-rs migration.** This item records the complete initial
  observation; no external discussion is needed to interpret or reproduce it.

  **Investigation:** the [reproducer and findings](crates/competitor-bench/investigations/cobyla-lm/README.md)
  locate the main overhead in COBYLA's numerical driver, including scratch
  allocations, inverse checks, and repeated model construction. The
  [implemented optimization and results](crates/competitor-bench/investigations/cobyla-lm/cobyla-optimization.md)
  retain numerical traces, callback-error and budget coverage, a driver
  benchmark, and allocation ceilings. The driver is 1.9–2.8 times faster on
  these cases, with 94–97% fewer allocation requests. A smaller performance
  gap to `cobyla` 1.0.2 remains.

  **Implementations.** The comparison used GlobalSearch-rs commit
  `4bf3eaa6b3677e0a2cc18f61f81603612db66b13`, whose COBYLA adapter uses Basin
  1.8.0, and its direct parent
  `1f44818e396567d22cb7137c267c66da2e19f334`, whose otherwise equivalent
  adapter uses `cobyla` 1.0.2. Both revisions were compiled with Rust 1.88.0
  using `cargo build --release --no-default-features`. Measurements ran on an
  Intel Core Ultra 7 155U under NixOS. Hyperfine 1.20.0 performed three
  process-level warmups followed by 15 timed runs. Each process also performed
  16 untimed solves before its measured loop. Objective-evaluation tracking
  was enabled for every solve.

  **Problems and solver settings.** The descriptions below use GlobalSearch's
  nonnegative-is-feasible convention. Its Basin adapter negates user
  constraints into Basin's nonpositive-is-feasible convention. At the recorded
  migration commit, the adapter also floors the final radius at
  `sqrt(f64::EPSILON) * initial_step_size`, even with zero parameter
  tolerances. For these cases, that floor is approximately `7.45e-9`.

  1. Six-hump camel used
     `f(x,y) = (4 - 2.1*x^2 + x^4/3)*x^2 + x*y + (-4 + 4*y^2)*y^2`, bounds
     `x in [-3,3]` and `y in [-2,2]`, start `(0,0)`, initial radius `0.5`, an
     objective budget of 50, and zero function and parameter tolerances. Each
     timed process performed 5,000 solves.
  2. The 10-dimensional sphere used `f(x) = sum(x_i^2)`, bounds `[-5,5]` for
     every coordinate, start
     `(2.5,-2,1.5,-1,0.5,2.25,-1.75,1.25,-0.75,0.25)`, initial radius `0.5`,
     an objective budget of 200, and zero function and parameter tolerances.
     Each timed process performed 400 solves.
  3. The constrained quadratic used
     `f(x,y) = (x - 1)^2 + (y - 1)^2`, bounds `[0,2]` for both coordinates,
     start `(0.5,0.5)`, constraint `1.5 - x - y >= 0`, initial radius `0.5`, an
     objective budget of 100, and zero function and parameter tolerances. Each
     timed process performed 3,000 solves.

  **Results.** Times are the mean wall time per local solve; the parenthesized
  number is the mean objective-evaluation count per solve.

  | Problem | `cobyla` 1.0.2 | Basin 1.8.0 | Slowdown | Final objective: old / Basin |
  |---|---:|---:|---:|---:|
  | Six-hump camel | 10.3 us (50) | 84.7 us (50) | 8.23x | -1.0316284534 / -1.0316284527 |
  | 10D sphere | 620 us (200) | 1.97 ms (200) | 3.18x | 9.65e-7 / 5.73e-9 |
  | Constrained quadratic | 35.9 us (100) | 157 us (61) | 4.39x | 0.1250000000 / 0.1249999974 |

  A second six-hump measurement used the public default settings: budget 300,
  initial radius `0.5`, relative and absolute function tolerances `1e-6` and
  `1e-8`, respectively, and zero parameter tolerances. The old adapter took
  10.9 us and 52 evaluations per solve; the Basin adapter took 88.3 us and 50
  evaluations, an 8.11x slowdown, with the same final objectives as the
  fixed-budget row.

  **Caveats and deliverable.** These numbers compare the complete
  GlobalSearch-rs adapters, not bare solver kernels. In particular, the Basin
  adapter projects objective and user-constraint callbacks into the box and
  represents the box as `2*n` nonlinear inequalities, whereas the old adapter
  passed bounds through `cobyla`'s native bounds argument. The Basin adapter
  also adds a hard objective-budget guard and executor termination criteria.
  Reproduce the three cases directly against both solver crates in
  `competitor-bench`, and separately benchmark the two GlobalSearch adapters.
  Profile allocations, state cloning, constraint evaluation, and executor
  overhead. Determine which layer owns the gap, retain a benchmark that guards
  the affected layer, and improve it without weakening numerical behavior,
  bounds, error propagation, or strict callback-budget handling.

- [x] **Add a pivoted-QR solve path for Levenberg-Marquardt.** Implemented
  `.with_pivoted_qr()` and `LevenbergMarquardtQr` with reusable column-pivoted
  QR, explicit rank-loss handling, and all four dense backends. Cholesky
  remains the default. See the [implementation and production comparisons](crates/competitor-bench/investigations/cobyla-lm/lm-qr.md)
  for conditioning, raw SVI, SSVI, backend coverage, and sparse limitations.

- [ ] **Investigate LM damping selection and relative stopping tests.**
  Production QR improves small-damping step accuracy but preserves the
  observed narrow-SVI convergence gap and premature stopping on nearly
  collinear linear problems. Investigate Nielsen damping versus MINPACK's
  trust-radius parameter selection independently of factorization. Validate
  calibration Jacobians against their actual residual formulas, and account
  for parameter nonidentifiability before drawing migration conclusions.
  Retain the [production probes](crates/competitor-bench/investigations/cobyla-lm/lm-qr.md)
  and compare convergence, termination, parameter recovery, and callbacks
  from all retained starts.

- [ ] **Add the full-form `NonlinearConstraints` aggregator (tenet 4).** Model
  PRIMA's full COBYLA input by folding nonlinear inequalities, optional
  linear inequalities and equalities, and optional box bounds into one
  `c(x) ≤ 0` vector. Keep the trait standalone like `LinearConstraints`: it
  must not be a parent of the sibling constraint traits, and blanket bridges
  must not silently discard constraint blocks. Preserve the existing
  `NonlinearInequalityConstraints` API, all four backends, and wasm support.

## Deferred design

- [ ] **Reconsider exact evaluation budgets for Basin 2.0.** In Basin 1.x,
  `MaxCostEvals` remains a boundary-checked stopping criterion: `Solver::init`
  and an active `Solver::next_iter` finish before the executor checks it, so
  initialization and batched iterations may exceed the threshold. A future
  hard-cap API would need executor/problem-level control flow that prevents
  callbacks after exhaustion, handles batch reservation and composed problems,
  and defines the outcome when the budget cannot complete initialization. Cover
  COBYLA's `n + 1`-point initialization and zero-to-two-evaluation steps in any
  resulting contract and tests.

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

See `CONTRIBUTING.md` for the design tenets and constraints that shape these
decisions.

## Outreach plan

Research snapshot: 2026-09-03

This section turns the current crates.io reverse-dependency audit into an
outreach plan for [Basin](https://github.com/jolars/basin). Basin now provides
version-specific backend features and supports the backend matrix below.

### Executive summary

The best immediate targets are **GlobalSearch-rs**, **stochastic-rs**,
**lme-rs**, **PMcore**, **crabSAXS**, **molex**, the
Levenberg–Marquardt/Gauss–Newton path in **system_solver**, and the focused root
solves in **finql** and **kde_diffusion**. These projects already map well to
Basin's solver set, and several have a concrete reason to prefer Basin: native
bounds and constraints, L-BFGS-B, derivative-free constrained solvers,
global/memetic solvers, native Levenberg–Marquardt, direct bracketed root
finding, or first-class cancellation.

The audit identified four principal compatibility gaps; all are now closed:

1. **Reliable checkpoint/resume for stochastic and population solvers.** Exact
   solver-aware continuation is tested for simulated annealing, DE, SSGA,
   CMA-ES, basin-hopping, and PSO; PSO also supports state-only exact resume.
2. **Particle swarm optimization.** `GlobalBestPso` supplies the bounded,
   seeded global-best route needed by the audited PSO consumers.
3. **Brent root finding.** `BrentRoot` directly solves a bracketed scalar
   equation while preserving the signed value, bracket, and typed callback
   errors outside the optimization `Executor` model.
4. **Hager–Zhang line search.** `HagerZhang` covers the option exposed by
   GlobalSearch-rs, including its Ackley example with unbounded L-BFGS.

The backend matrix and cancellation API are also complete. The remaining
pre-outreach work is migration polish: finish the checkpoint documentation,
land a proof-of-concept migration, and publish reproducible benchmark methods.

### Backend compatibility

  | Backend                              | Supported versions | Why                                                                                               |
  | ------------------------------------ | -----------------: | ------------------------------------------------------------------------------------------------- |
  | `ndarray`                            |   0.15, 0.16, 0.17 | Covers EnzymeML, GlobalSearch/lme-rs/crabSAXS, and current stochastic-rs/PMcore respectively.     |
  | `nalgebra`                           |          0.32–0.35 | Covers the target projects, preserves the older ecosystem range, and includes the latest release. |
  | `faer`                               |   0.22, 0.23, 0.24 | Covers the main modern Faer range seen in the reverse-dependency audit.                           |
  | `Vec<f32/f64>` and primitive scalars |            current | Needed by projects such as molex and argtuner and useful as the lowest-friction migration path.   |

The most valuable target-specific versions—`ndarray` 0.15–0.17, `nalgebra`
0.33–0.34, and `faer` 0.24—are covered. Older backend releases exist in the full
reverse-dependency set, but supporting them has a lower outreach return.

### What Basin already has

Basin is already unusually strong for the projects in this audit:

- Local smooth optimization: BFGS, L-BFGS, L-BFGS-B, gradient descent,
  stochastic gradient descent, Newton trust-region methods, Gauss–Newton,
  Levenberg–Marquardt, and trust-region reflective least squares.
- Derivative-free optimization: Nelder–Mead, Brent minimization, golden section,
  NEWUOA, BOBYQA, LINCOA, COBYLA, and MADS.
- Scalar root finding: direct bracketed Brent with typed callback errors and
  explicit endpoint-root and invalid-bracket handling.
- Global and hybrid optimization: CMA-ES, bounded CMA-ES, differential
  evolution, steady-state genetic algorithm, random search, Solis–Wets, basin
  hopping, generic simulated annealing, and memetic CMA/DE/MA-LSCh methods.
- Constraints: box, linear, and nonlinear constraints, plus augmented-Lagrangian
  and barrier approaches.
- Framework capabilities: iteration/time/evaluation limits, typed errors,
  observers, one-step execution, deterministic seeded RNGs, finite-difference
  gradients/Jacobians/Hessians/Hessian-vector products, and parallel batch
  evaluation.

The main task is therefore compatibility and migration polish, not reproducing
Argmin's whole solver catalog.

### Recommended Basin roadmap

#### P0: complete before broad outreach

##### 1. Backend-version matrix and CI (complete)

Basin provides exact-version features, frozen Basin 1.x aliases, and moving
`*_latest` aliases. Because Cargo features are additive, enabling several
versions of one backend selects the newest enabled release. CI builds and tests
every exact version independently, exercises the frozen and moving aliases, and
checks the MSRV, accelerated features, and WASM-compatible configurations.

Acceptance criteria:

- [x] Each backend/version combination compiles in isolation.
- [x] Feature unification selects the newest enabled release of a backend.
- [x] The docs show a complete feature-to-version table.
- [x] At least one solver test runs for every backend/version combination.

##### 2. Define checkpoint guarantees accurately

The current state-only checkpoint writer is useful for some deterministic local
solvers, but it is not sufficient to promise exact continuation of stochastic
solvers. Several solvers keep RNG state inside the solver, and some
initialization paths rebuild or clear population state.

Choose and document two distinct concepts:

- **Warm start:** resume from the best point or a saved population, without
  promising an identical future trajectory.
- **Exact resume:** serialize all state needed to continue identically,
  including population/simplex/history, solver phase, counters, and RNG state.

The abstraction boundary is now explicit:

- [x] Exact checkpoints store the solver, state, and authoritative evaluation
  counters. `Executor::resume_from_checkpoint` restores the three together
  and skips `init`, so evolving data can remain where it naturally belongs.
  Simulated annealing retains its stateful neighbor and RNG in
  `SimulatedAnnealingState` for compatibility with state-only exact resume.
  The separate state-only `CheckpointWriter` remains a warm-start facility.
- [x] The exact file format records a format version, the exact Basin version,
  and concrete solver/state type names, rejecting incompatible files before
  decoding their payload.

Acceptance criteria for exact resume across the remaining solvers:

- Store and serialize every evolving component.
- Add serialization to population, CMA-ES, simplex, and nonlinear least-squares
  states as appropriate.
- Restore through `Executor::resume_from_checkpoint`, which does not call
  `init`.
- [x] Test uninterrupted versus save/reload/`Executor::resume_from_checkpoint`
  for simulated annealing, including a stateful neighbor and bit-identical
  serialized output.
- [x] Add the same coverage for DE, SSGA, CMA-ES, and basin hopping. The future
  PSO task below independently requires warm-start and exact-resume tests.
- [x] Version the checkpoint format and reject incompatible checkpoints cleanly.

Until the wider work is complete, describe ordinary state checkpointing as a
warm start. Promise exact continuation only for a solver/state pair captured by
the exact checkpoint API and restored through
`Executor::resume_from_checkpoint`.

##### 3. Publish an Argmin-to-Basin migration guide

Include compilable before/after examples for the patterns that dominate the
audit:

| Argmin pattern | Basin equivalent or guidance |
|---|---|
| `Executor::new(...).configure(|state| state.param(x).max_iters(n))` | `Executor::from_start(...)`, `.max_iter(n)`, and explicit termination criteria. |
| `NelderMead::new(simplex).with_sd_tolerance(tol)` | Basin Nelder–Mead plus `SimplexTolerance`. |
| `LBFGS::new(MoreThuenteLineSearch::new(), m)` | Basin L-BFGS or L-BFGS-B with the matching line search and memory. |
| `GaussNewtonLS` | Basin Gauss–Newton or Levenberg–Marquardt, depending on the problem. |
| `CostFunction`, `Gradient`, `Hessian` | Basin's corresponding problem traits and typed error associated type. |
| Manual central differences | Basin's finite-difference adapters. |
| Box constraints encoded by transforms or penalties | `BoxConstraints` plus L-BFGS-B, bounded Nelder–Mead, TRF, BOBYQA, or bounded global solvers. |
| Argmin's general `Error` | A project error type or `anyhow::Error`; explain the typed-error migration explicitly. |
| `Observe` for progress | Basin observer receiving state. |
| Observer error used to stop | Cancellation token or a typed problem error. |

Also publish one complete example using each supported backend version. A
maintainer is far more likely to try Basin if they can copy a working
`Cargo.toml` line and a 30-line migration.

#### P1: solver gaps with a measurable outreach payoff

##### 1. Generic simulated annealing (complete)

`SimulatedAnnealing` is generic over arbitrary cloneable parameter types and a
user-supplied `Neighbor` trait or closure. It implements the classical
Metropolis rule rather than Argmin's logistic acceptance variant, requires an
explicit geometric, reciprocal, or normalized-log schedule, supports several
proposals per temperature level, accepts a seed or custom RNG, and offers one
optional fixed-interval, rejection-stall, or best-stall reannealing trigger.

`SimulatedAnnealingState` owns the evolving neighbor, RNG, cooling and chain
progress, counters, incumbent, and best point. With `serde`, the solver/state
pair can be checkpointed and restored through `Executor::resume_from_checkpoint`
for bit-identical continuation. Public tests cover continuous vectors, discrete
permutations, all numeric backends, `f32`, custom RNGs, non-finite costs,
reannealing, stall criteria, and exact resume.

This enables complete migrations for system_solver's SA route,
`saltine-gromark`, `scattr`, and `aminograph`. Continuous CMA-ES/DE is not a
drop-in substitute for their custom transition rules.

- [x] Match Argmin's composable reannealing configuration: allow fixed,
  accepted-stall, and best-stall triggers to be enabled simultaneously, and
  provide migration-friendly `with_reannealing_fixed`,
  `with_reannealing_accepted`, and `with_reannealing_best` builder methods.
  Reanneal when any enabled trigger fires, reset all reannealing counters,
  and retain Basin's exact threshold semantics and classical Metropolis
  acceptance.
- [x] Make `Neighbor::propose` return `Result<P, Self::Error>` before the API is
  released. Require `Neighbor` and `CostFunction` to share the run's typed
  application error so `SimulatedAnnealing` can continue exposing that error
  directly as `Solver::Error`; use `std::convert::Infallible` when neither
  the proposal nor the objective can fail. Update the closure blanket
  implementation and add coverage for both infallible and fallible
  proposals.

##### 2. Particle swarm optimization (complete)

PSO is the largest exact solver-name gap in the current reverse-dependency set.
Implement bounded continuous PSO with:

- inertia, cognitive, and social coefficients;
- configurable swarm size and velocity handling;
- seeded/custom RNG;
- parallel objective evaluation;
- serializable particles, velocities, personal/global bests, and RNG state;
- warm-start and exact-resume tests.

It unlocks or simplifies outreach to `argtuner`, EnzymeML, `lightcurve-fitting`,
`atmosim`, `rssn`, and RustQuant data code. It also avoids asking maintainers to
validate a change of algorithm while simultaneously changing frameworks.

`GlobalBestPso` implements the synchronous inertia-weight global-best rule with
the Standard PSO 2006 coefficient and swarm-size profile. Boundary response and
velocity limiting are separate policies, allowing absorb, preserve, or damped
reflection without conflating them with a span-relative speed cap. The state
owns current particles, velocities, personal/global bests, evaluation counters,
and the live RNG, so both state-only and solver-aware checkpoint continuation
are bit-identical. Public tests cover an Argmin 0.11 one-generation fixture,
uniform and warm initialization, non-finite costs, all numeric backends, `f32`,
parallel counts/reproducibility, and exact resume.

The topology-specific name leaves `StandardPso2006` and `StandardPso2011`
available for faithful future implementations. Those algorithms require random
neighborhood topology, and SPSO-2011 also changes the motion distribution, so
they are not represented as modes of the global-best update.

##### 3. Brent root solver (complete)

`BrentRoot<F>` combines bisection, secant steps, and inverse quadratic
interpolation behind a direct fallible-closure API. It deliberately remains
outside the optimization `Solver`/`State`/`Executor` path, where the signed
function value and sign-changing bracket would otherwise be misrepresented as
an objective cost and box constraint. `RootResult` reports the signed value,
final bracket, iterations, evaluations, and clean termination status;
`BrentRootError` distinguishes invalid or non-bracketing intervals, non-finite
values, and typed callback failures. Endpoint roots return successfully without
iterating. Public tests cover analytic roots, `f32`, evaluation counts, clean
iteration limits, structural and callback error paths, and overflow-safe
bracket validation.

This enables focused migrations for
[finql](https://github.com/xemwebe/finql) and
[kde_diffusion](https://crates.io/crates/kde_diffusion).

##### 4. Hager–Zhang line search (complete)

`HagerZhang<F>` implements the reference expansion, contraction, interval
update, and double-secant procedure with ordinary and approximate-Wolfe
acceptance. Its public configuration covers `delta`, `sigma`, `epsilon`,
`theta`, `gamma`, the initial step and bounds, the expansion factor, and a hard
fused-evaluation budget. The conjugate-gradient-only `eta` parameter is
intentionally absent. Tests cover cancellation-sensitive acceptance, strict
failure, non-finite trials, typed errors, all numeric backends, `f32`, and the
GlobalSearch Ackley example with unbounded L-BFGS.

#### P2: useful migration accelerators

##### Argmin problem adapter

Consider a small optional `basin-argmin` compatibility crate containing local
newtype wrappers. A wrapper can delegate an existing Argmin `CostFunction`,
`Gradient`, `Hessian`, `Residual`, or `Jacobian` implementation to Basin. That
lets a maintainer benchmark Basin solvers before rewriting all problem traits.

Keep the scope narrow: adapting problem definitions is straightforward; adapting
arbitrary Argmin solvers and their state machines is not. The adapter should be
presented as an evaluation/migration bridge, not a permanent requirement.

##### Convenience problem builders

Closure-based builders would reduce boilerplate in projects whose optimization
problem is local to one function:

```rust
let problem = Function::new(cost)
    .gradient(gradient)
    .bounds(lower, upper);
```

A `BoundedProblem` delegating wrapper would also make it easy to attach box
constraints without modifying an existing problem type.

##### Noisy-objective support

PMcore explicitly averages repeated particle-filter evaluations before making
Nelder–Mead decisions. Basin currently describes cost functions as
pure/deterministic, while noisy objectives require additional semantics.

At minimum, document that solvers may cache or reuse evaluations and show a safe
averaging wrapper. A reusable `Replicated`/`Averaged` problem adapter—with
configurable repetitions and deterministic seed handling—would make Basin more
credible for stochastic simulation and pharmacometrics workloads.

##### Observer integrations

A small satellite crate for `tracing` and/or `indicatif` would make progress
reporting easier, but this is not a core blocker. The audited custom observers
mostly read generic state and ignore Argmin's solver-specific key-value
metadata.

#### Lower priority solver gaps

- **SR1 trust region:** used by EnzymeML. Basin's existing Newton trust-region
  and quasi-Newton methods provide alternatives, but an SR1 implementation would
  preserve behavior.
- **Landweber iteration:** only one low-priority audited package mentions it; a
  specialized inverse-problems audience may value it, but it should not delay
  outreach.
- **Solver-specific observer metadata:** useful for rich diagnostics, but not
  required by the strongest migration candidates. Add structured diagnostics
  only if real integrations request them.

### Outreach targets

#### Ready now

  | Rank | Project                                                                                                                                                   | Current use                                                                                                      | Basin opportunity                                                                                                                                                                                       | Remaining caveat                                                                                                                                   | Suggested offer                                                                                                    |
  | ---: | --------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------ |
  |    1 | [GlobalSearch-rs](https://github.com/GermanHeim/globalsearch-rs)                                                                                          | Argmin 0.11; `ndarray` 0.16; L-BFGS, Newton-CG/trust-region, steepest descent, Nelder–Mead; optional Hager–Zhang | Basin can supply the local solvers, including Hager–Zhang, and adds native constrained/local methods that fit a global-search framework.                                                                | No substantial solver blocker. Their outer checkpointing is project-owned, so Basin runner checkpoints are not a prerequisite.                     | Offer a Basin local-solver backend behind a feature flag and volunteer the first PR.                               |
  |    2 | [stochastic-rs](https://github.com/rust-dd/stochastic-rs)                                                                                                 | Argmin 0.11; `ndarray` 0.17; L-BFGS, manually projected bounds, Nelder–Mead; separate LM crate                   | Replace projected L-BFGS with native L-BFGS-B; reuse Basin finite differences; optionally consolidate least-squares work onto Basin LM/TRF. Basin hopping and global solvers fit calibration workloads. | Broad surface area means a staged migration is safer than a wholesale replacement.                                                                 | Start with one bounded MLE/Whittle routine and provide a benchmark plus matching tolerances.                       |
  |    3 | [lme-rs](https://github.com/x4g4p3x/lme-rs)                                                                                                               | Argmin 0.11; `ndarray` 0.16; a compact Nelder–Mead integration                                                   | Very small migration; bounded or transformed model parameters could later use BOBYQA, L-BFGS-B, or MADS.                                                                                                | No substantial solver blocker.                                                                                                                     | Offer a focused PR replacing the optimizer module and preserving outputs on existing tests.                        |
  |    4 | [PMcore](https://github.com/LAPKB/PMcore)                                                                                                                 | Argmin 0.11; `ndarray` 0.17/Faer 0.24; Nelder–Mead for dose and noisy IOV objectives                             | Basin Nelder–Mead is a direct path; MADS/BOBYQA and constrained methods are relevant follow-up experiments.                                                                                             | IOV costs are noisy and averaged. Document evaluation/caching semantics and ideally provide an averaging adapter before pitching MADS as superior. | Migrate deterministic `bestdose` first; treat IOV as a separate benchmark.                                         |
  |    5 | [crabSAXS](https://github.com/Ojas-Singh/crabSAXS)                                                                                                        | Argmin 0.10; `ndarray` 0.16/`nalgebra` 0.33; Nelder–Mead fitting                                                 | Direct migration, with constrained derivative-free alternatives for physical fitting parameters.                                                                                                        | No substantial solver blocker.                                                                                                                     | Offer a small PR and compare Nelder–Mead with bounded BOBYQA or MADS on one fit.                                   |
  |    6 | [system_solver](https://github.com/bcolloran/system_solver)                                                                                               | Argmin 0.11; `nalgebra` 0.34; Gauss–Newton with line search, L-BFGS, and simulated annealing                     | Basin LM/Gauss–Newton, L-BFGS, and generic simulated annealing now cover all named paths; constraints and TRF may improve robustness.                                                                   | Basin uses classical Metropolis acceptance rather than Argmin's logistic variant, so the SA route needs behavior-level validation.                 | Offer a feature-gated Basin backend, starting with Gauss–Newton/LM and then porting SA with numerical comparisons. |
  |    7 | [inlier](https://github.com/soraxas/inlier)                                                                                                               | Argmin 0.11; `nalgebra` 0.33; hand-built LM-style bundle adjustment                                              | Basin's native LM/TRF and residual/Jacobian traits are a strong conceptual match.                                                                                                                       | The current implementation is specialized; migration value must be demonstrated with numerical and performance tests.                              | Propose a benchmark branch, not an immediate dependency switch.                                                    |
  |    8 | [molex](https://github.com/foldit-org/molex)                                                                                                              | Argmin 0.11; `Vec`; L-BFGS with progress-driven cancellation                                                     | Basin L-BFGS and `CancellationToken` replace the solver and observer-error cancellation pattern directly.                                                                                               | The progress callback still needs an observer, but it can cancel a cloned token without making observation fallible.                               | Offer a focused migration preserving progress updates and clean cancellation.                                      |
  |    9 | [argtuner](https://github.com/jzombie/rust-argtuner)                                                                                                      | Argmin PSO over `Vec`; population and completed-trial checkpointing                                              | `GlobalBestPso` covers the central algorithm and preserves particles, velocities, personal/global bests, and the live RNG for bit-identical continuation.                                               | Completed-trial records and persistence orchestration remain application-owned; validate their mapping to Basin's checkpoint format.               | Port the PSO sampler and compare uninterrupted and resumed expensive-trial runs.                                   |
  |   10 | [EnzymeML](https://github.com/enzymeml/enzymeml-rs)                                                                                                       | Argmin BFGS, L-BFGS, PSO, and SR1 trust-region; an `egobox-ego` integration                                      | Basin now covers BFGS, L-BFGS, and PSO, including deterministic PSO resume.                                                                                                                             | A full switch still requires SR1 trust-region and a separate plan for the EGO runner, which is built on Argmin.                                    | Offer a Basin backend for the native quasi-Newton and PSO paths; scope EGO as a separate integration.              |
  |   11 | [scattr](https://crates.io/crates/scattr), [saltine-gromark](https://crates.io/crates/saltine-gromark), [aminograph](https://crates.io/crates/aminograph) | Argmin simulated annealing over custom transitions or discrete states                                            | Basin's generic `Neighbor`, explicit cooling/reannealing configuration, arbitrary cloneable parameters, and exact resume now cover these use cases.                                                     | Basin uses classical Metropolis acceptance rather than Argmin's logistic variant, so behavior and tuning will not match mechanically.              | Offer separate small migrations that preserve each project's transition rule and validate its acceptance behavior. |
  |   12 | [finql](https://github.com/xemwebe/finql)                                                                                                                 | Argmin `BrentRoot` for fixed-income yield calculations                                                           | `BrentRoot` directly replaces the bracketed scalar solve without routing it through optimization state.                                                                                                | Confirm tolerance semantics and error mapping against the current yield routines.                                                                 | Offer a focused replacement and compare roots, iteration counts, and failure behavior.                             |
  |   13 | [kde_diffusion](https://crates.io/crates/kde_diffusion)                                                                                                   | Argmin `BrentRoot` in a scalar bandwidth calculation                                                             | Its concentrated Argmin dependency can move to Basin's direct `BrentRoot` API.                                                                                                                         | Confirm the current crate release still has no other Argmin-dependent paths.                                                                      | Offer a minimal dependency replacement with numerical regression tests.                                            |

#### Deprioritize

- **linfa-linear / linfa-logistic / linfa-ftrl:** high-profile, but Argmin is
  embedded in established public algorithms and broad ecosystem compatibility
  matters more than Basin's extra solvers.
- **augurs-forecaster:** optimization is only one component of a larger
  forecasting stack; migration benefit is not obvious enough for cold outreach.
- **curvo:** Argmin numeric traits are spread through the codebase, making this
  more than a solver replacement.
- **stem_material:** the relevant optimization path is specialized and the
  benefit of switching is weak.
- **hawkes-rs:** current use is small and does not strongly benefit from
  Basin-specific capabilities.
- **egobox-ego:** it implements an Argmin solver and custom state rather than
  merely consuming a solver. Porting it is an integration project, not a normal
  dependency switch.
- **argmin observer/checkpoint crates and `cobyla-argmin`:** these are
  extensions of Argmin itself. Basin already has native COBYLA, and the
  observer/checkpoint crates are not plausible switch targets.

### Contact sequence

1. Publish the migration guide; the backend matrix and cancellation API are
   complete.
2. Offer molex a focused migration that preserves its progress callback and
   replaces observer-error cancellation with `CancellationToken`.
3. Prepare one small proof-of-concept PR for lme-rs or crabSAXS to validate the
   guide on a real codebase.
4. Approach GlobalSearch-rs with a feature-gated Basin backend proposal.
5. Approach stochastic-rs with a narrow L-BFGS-B migration and benchmark, not a
   request to replace every optimizer at once.
6. Approach PMcore with deterministic `bestdose` first and a separate
   noisy-objective experiment.
7. Approach system_solver and inlier with LM/Gauss–Newton comparison branches.
8. Approach argtuner and EnzymeML with matching PSO paths and resume tests.
9. Offer focused `BrentRoot` replacements to finql and kde_diffusion.
10. Approach the generic-SA targets with migrations that preserve their custom
    transition rules.

### Outreach principles

- Open an issue only after confirming that the default branch still contains the
  audited code.
- Lead with the project's concrete problem, not a general claim that Basin is
  better.
- Offer a small PR or benchmark. Do not ask maintainers to perform an unproven
  migration for you.
- Preserve the current algorithm first. Introduce a different Basin solver as an
  opt-in comparison.
- State backend-version support explicitly in the first message.
- Be candid about missing solver parity and checkpoint semantics.
- Include numerical equivalence tests, termination behavior, evaluation counts,
  and runtime in any PR.
- Avoid mass-produced outreach. Each message should cite the exact file or
  routine that motivated it.

### Pre-outreach release checklist

- [x] Backend feature table published.
- [x] CI covers every advertised backend version.
- [x] Feature unification selects the newest enabled release of a backend.
- [x] Argmin-to-Basin migration guide published.
- [x] Typed-error migration example published.
- [x] Bounds and constraint migration example published.
- [x] Cancellation returns a normal `Cancelled` result with best-so-far state.
- [ ] Checkpoint documentation distinguishes warm start from exact resume.
- [ ] One external proof-of-concept migration passes upstream tests.
- [ ] Benchmark methodology and reproducible commands are included.
- [ ] Each outreach issue is rechecked against the project's current default
  branch.

### Audit notes and evidence

The audit inspected 88 current package source releases in Argmin's crates.io
reverse-dependency set, then inspected the default branches of the strongest
candidates. Solver-name counts are discovery signals, not ecosystem market-share
estimates: source archives can contain examples, tests, old code, or commented
integrations.

Key source locations:

- GlobalSearch-rs local solver runner:
  <https://github.com/GermanHeim/globalsearch-rs/blob/main/src/local_solver/runner.rs>
- stochastic-rs MLE fitting:
  <https://github.com/rust-dd/stochastic-rs/blob/main/stochastic-rs-stats/src/mle/fit.rs>
- stochastic-rs SABR objective:
  <https://github.com/rust-dd/stochastic-rs/blob/main/stochastic-rs-quant/src/vol_surface/sabr_smile/objective.rs>
- lme-rs optimizer:
  <https://github.com/x4g4p3x/lme-rs/blob/master/src/optimizer.rs>
- PMcore noisy IOV optimizer:
  <https://github.com/LAPKB/PMcore/blob/main/src/iov/optimizer.rs>
- PMcore dose optimization:
  <https://github.com/LAPKB/PMcore/blob/main/src/bestdose/optimization.rs>
- crabSAXS fitting:
  <https://github.com/Ojas-Singh/crabSAXS/blob/main/src/fit.rs>
- system_solver solver integrations:
  <https://github.com/bcolloran/system_solver/tree/main/src/equation_system/sub_problem/solve_subproblem>
- inlier bundle adjustment:
  <https://github.com/soraxas/inlier/blob/main/src/bundle_adjustment.rs>
- EnzymeML optimizers:
  <https://github.com/enzymeml/enzymeml-rs/tree/master/src/optim/optimizers>
- molex cancellation-through-observer pattern:
  <https://github.com/foldit-org/molex/blob/dev/src/xtal/bfactor_refine.rs>
- argtuner PSO and checkpointing:
  <https://github.com/jzombie/rust-argtuner/blob/main/src/sampler/pso.rs>
- Basin observer API:
  <https://github.com/jolars/basin/blob/main/crates/basin/src/core/observer.rs>
- Basin checkpoint writer:
  <https://github.com/jolars/basin/blob/main/crates/basin/src/core/observer/checkpoint.rs>
- Basin exact-checkpoint API:
  <https://github.com/jolars/basin/blob/main/crates/basin/src/core/checkpoint.rs>
- Basin termination reasons:
  <https://github.com/jolars/basin/blob/main/crates/basin/src/core/termination.rs>

Re-run the reverse-dependency and default-branch checks immediately before
opening issues; this list is a dated prospecting snapshot, not a permanent
compatibility claim.
