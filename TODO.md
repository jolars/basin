# TODO

Ordered by recommended sequence.

## General design

- [ ] **Add the full-form `NonlinearConstraints` aggregator (tenet 4).** Model
  PRIMA's full COBYLA input by folding nonlinear inequalities, optional
  linear inequalities and equalities, and optional box bounds into one
  `c(x) ≤ 0` vector. Keep the trait standalone like `LinearConstraints`: it
  must not be a parent of the sibling constraint traits, and blanket bridges
  must not silently discard constraint blocks. Preserve the existing
  `NonlinearInequalityConstraints` API, all four backends, and wasm support.

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

See `CONTRIBUTING.md` for the design tenets and constraints that shape these
decisions.

## Outreach plan

Research snapshot: 2026-09-03

This section turns the current crates.io reverse-dependency audit into an
outreach plan for [Basin](https://github.com/jolars/basin). Basin now provides
version-specific backend features and supports the backend matrix below.

### Executive summary

The best immediate targets are **GlobalSearch-rs**, **stochastic-rs**,
**lme-rs**, **PMcore**, **crabSAXS**, **molex**, and the
Levenberg–Marquardt/Gauss–Newton path in **system_solver**. These projects
already map well to Basin's solver set, and several have a concrete reason to
prefer Basin: native bounds and constraints, L-BFGS-B, derivative-free
constrained solvers, global/memetic solvers, native Levenberg–Marquardt, or
first-class cancellation.

The largest remaining compatibility gaps are:

1. **Reliable checkpoint/resume for stochastic and population solvers.** This is
   a blocker for `argtuner` and any long-running global optimization workflow.
   The checkpoint must include solver state and RNG state, not only the generic
   optimization state.
2. **Generic simulated annealing.** Active Argmin users apply it to arbitrary
   and sometimes discrete parameter types, which Basin's continuous vector
   optimizers cannot replace directly.
3. **Particle swarm optimization.** Seven audited package trees contain PSO
   usage or integration. CMA-ES and differential evolution are alternatives, but
   PSO support makes migration much less disruptive.
4. **Brent root finding.** Two current packages use `BrentRoot`. Basin's
   existing Brent implementation minimizes a scalar function; it does not
   replace a bracketed root solver.
5. **Hager–Zhang line search.** This matters particularly to GlobalSearch-rs,
   which exposes it as a supported option and has an example where it
   outperforms the default More–Thuente setup.

Do not hold the first outreach round for every solver gap. The backend matrix
and cancellation API are complete; ship migration examples and honest checkpoint
documentation, then contact the ready targets. Add simulated annealing, PSO, and
exact stochastic resume before approaching projects that depend on those
capabilities.

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
- Global and hybrid optimization: CMA-ES, bounded CMA-ES, differential
  evolution, steady-state genetic algorithm, random search, Solis–Wets, basin
  hopping, and memetic CMA/DE/MA-LSCh methods.
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

Acceptance criteria for exact resume:

- Serialize a complete runner checkpoint, not only `State`.
- Add serialization to population, CMA-ES, simplex, and nonlinear least-squares
  states as appropriate.
- Ensure `init` recognizes restored state and does not discard it.
- Test uninterrupted versus save/reload/resume runs for DE, SSGA, CMA-ES, basin
  hopping, and any future PSO/SA solver.
- Version the checkpoint format or record enough metadata to reject incompatible
  checkpoints cleanly.

Until this is complete, describe Basin's current facility as state
checkpointing/warm start, not exact stochastic resume.

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

##### 1. Generic simulated annealing

This is the most important *capability* gap because it covers problems that are
not naturally dense real vectors. Design it over an arbitrary parameter type
with a user-supplied move/neighbor operation, rather than tying it to a numeric
backend.

Useful API elements:

- `Anneal`/`Neighbor` trait or closure that proposes a state from the current
  state and temperature.
- Built-in temperature schedules matching common Argmin options.
- Caller-supplied seed and preferably caller-supplied RNG.
- Reannealing and stall limits.
- Serializable solver/RNG state for exact resume.
- Examples for both a continuous vector and a discrete/combinatorial state.

This enables complete migrations for system_solver's SA route,
`saltine-gromark`, `scattr`, and `aminograph`. Continuous CMA-ES/DE is not a
drop-in substitute for their custom transition rules.

##### 2. Particle swarm optimization

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

##### 3. Brent root solver

Add a bracketed scalar root solver separately from Brent minimization. This is a
relatively small implementation with two immediate targets:
[finql](https://github.com/xemwebe/finql) and
[kde_diffusion](https://crates.io/crates/kde_diffusion). Its API should make
invalid brackets and endpoint roots explicit.

##### 4. Hager–Zhang line search

This is lower-volume than PSO or SA, but unusually relevant to the top-ranked
GlobalSearch-rs target. More–Thuente is an acceptable default replacement for
many callers; Hager–Zhang is needed to preserve every exposed GlobalSearch
configuration and its Ackley example behavior.

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

#### Ready now that the backend matrix and cancellation API are complete

  | Rank | Project                                                          | Current use                                                                                                      | Basin opportunity                                                                                                                                                                                       | Remaining caveat                                                                                                                                                                                  | Suggested offer                                                                                               |
  | ---: | ---------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
  |    1 | [GlobalSearch-rs](https://github.com/GermanHeim/globalsearch-rs) | Argmin 0.11; `ndarray` 0.16; L-BFGS, Newton-CG/trust-region, steepest descent, Nelder–Mead; optional Hager–Zhang | Basin can supply the local solvers and adds native constrained/local methods that fit a global-search framework.                                                                                        | Hager–Zhang configurations need More–Thuente initially; exact parity requires adding Hager–Zhang. Their outer checkpointing is project-owned, so Basin runner checkpoints are not a prerequisite. | Offer a Basin local-solver backend behind a feature flag and volunteer the first PR.                          |
  |    2 | [stochastic-rs](https://github.com/rust-dd/stochastic-rs)        | Argmin 0.11; `ndarray` 0.17; L-BFGS, manually projected bounds, Nelder–Mead; separate LM crate                   | Replace projected L-BFGS with native L-BFGS-B; reuse Basin finite differences; optionally consolidate least-squares work onto Basin LM/TRF. Basin hopping and global solvers fit calibration workloads. | Broad surface area means a staged migration is safer than a wholesale replacement.                                                                                                                | Start with one bounded MLE/Whittle routine and provide a benchmark plus matching tolerances.                  |
  |    3 | [lme-rs](https://github.com/x4g4p3x/lme-rs)                      | Argmin 0.11; `ndarray` 0.16; a compact Nelder–Mead integration                                                   | Very small migration; bounded or transformed model parameters could later use BOBYQA, L-BFGS-B, or MADS.                                                                                                | No substantial solver blocker.                                                                                                                                                                    | Offer a focused PR replacing the optimizer module and preserving outputs on existing tests.                   |
  |    4 | [PMcore](https://github.com/LAPKB/PMcore)                        | Argmin 0.11; `ndarray` 0.17/Faer 0.24; Nelder–Mead for dose and noisy IOV objectives                             | Basin Nelder–Mead is a direct path; MADS/BOBYQA and constrained methods are relevant follow-up experiments.                                                                                             | IOV costs are noisy and averaged. Document evaluation/caching semantics and ideally provide an averaging adapter before pitching MADS as superior.                                                | Migrate deterministic `bestdose` first; treat IOV as a separate benchmark.                                    |
  |    5 | [crabSAXS](https://github.com/Ojas-Singh/crabSAXS)               | Argmin 0.10; `ndarray` 0.16/`nalgebra` 0.33; Nelder–Mead fitting                                                 | Direct migration, with constrained derivative-free alternatives for physical fitting parameters.                                                                                                        | No substantial solver blocker.                                                                                                                                                                    | Offer a small PR and compare Nelder–Mead with bounded BOBYQA or MADS on one fit.                              |
  |    6 | [system_solver](https://github.com/bcolloran/system_solver)      | Argmin 0.11; `nalgebra` 0.34; Gauss–Newton with line search, L-BFGS, and simulated annealing                     | Basin LM/Gauss–Newton and L-BFGS cover the main continuous paths; constraints and TRF may improve robustness.                                                                                           | Full removal of Argmin is blocked by generic simulated annealing.                                                                                                                                 | Offer a feature-gated Basin implementation for Gauss–Newton/LM first; revisit full migration after SA exists. |
  |    7 | [inlier](https://github.com/soraxas/inlier)                      | Argmin 0.11; `nalgebra` 0.33; hand-built LM-style bundle adjustment                                              | Basin's native LM/TRF and residual/Jacobian traits are a strong conceptual match.                                                                                                                       | The current implementation is specialized; migration value must be demonstrated with numerical and performance tests.                                                                             | Propose a benchmark branch, not an immediate dependency switch.                                               |
  |    8 | [molex](https://github.com/foldit-org/molex)                     | Argmin 0.11; `Vec`; L-BFGS with progress-driven cancellation                                                     | Basin L-BFGS and `CancellationToken` replace the solver and observer-error cancellation pattern directly.                                                                                               | The progress callback still needs an observer, but it can cancel a cloned token without making observation fallible.                                                                              | Offer a focused migration preserving progress updates and clean cancellation.                                 |

#### Contact after one specific feature lands

  | Project                                                                                                                                                   | Wait for                                       | Why it becomes compelling                                                                                                                                                                                           |
  | --------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
  | [argtuner](https://github.com/jzombie/rust-argtuner)                                                                                                      | PSO plus tested population/RNG resume          | PSO is central, and the project saves/restores population state while preserving completed expensive trials. Offering a different global algorithm is not an equivalent migration.                                  |
  | [EnzymeML](https://github.com/enzymeml/enzymeml-rs)                                                                                                       | PSO; ideally SR1 trust-region; clear EGO story | Basin already covers BFGS/L-BFGS and observers, but a full switch must also address PSO, SR1 trust-region, and the `egobox-ego` integration, whose runner is built on Argmin. A partial backend is possible sooner. |
  | [finql](https://github.com/xemwebe/finql)                                                                                                                 | Brent root solver                              | Fixed-income yield calculations use bracketed root finding, not minimization.                                                                                                                                       |
  | [kde_diffusion](https://crates.io/crates/kde_diffusion)                                                                                                   | Brent root solver                              | Its Argmin dependency is concentrated in a scalar bracketed root solve.                                                                                                                                             |
  | [scattr](https://crates.io/crates/scattr), [saltine-gromark](https://crates.io/crates/saltine-gromark), [aminograph](https://crates.io/crates/aminograph) | Generic simulated annealing                    | These use annealing-style custom transitions/discrete states that continuous vector solvers do not replace cleanly.                                                                                                 |

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
8. After PSO and exact resume land, approach argtuner and EnzymeML.
9. After generic SA and Brent root land, approach the corresponding exact-solver
   targets.

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
- [ ] Argmin-to-Basin migration guide published.
- [ ] Typed-error migration example published.
- [ ] Bounds and constraint migration example published.
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
- Basin termination reasons:
  <https://github.com/jolars/basin/blob/main/crates/basin/src/core/termination.rs>

Re-run the reverse-dependency and default-branch checks immediately before
opening issues; this list is a dated prospecting snapshot, not a permanent
compatibility claim.
