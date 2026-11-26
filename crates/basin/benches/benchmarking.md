# Benchmarking Basin

Run these commands from the Basin repository root.

## Reuse the existing harnesses

Inspect implementation and manifests before trusting benchmark comments; some
older comments still call `competitor-bench` by its former name, `lm-bench`.

  | Question                                      | Starting point                                                                     |
  | --------------------------------------------- | ---------------------------------------------------------------------------------- |
  | Gradient-descent step versus full run         | `crates/basin/benches/rosenbrock.rs` (custom timing loop, not Criterion)           |
  | Same solver across backends and sizes         | `crates/basin/benches/solver_backends.rs`                                          |
  | LM matrix kernels versus full solve           | `crates/basin/benches/lm_backends.rs`                                              |
  | LM versus `levenberg-marquardt`               | `crates/competitor-bench/benches/compare.rs` and `src/bin/verify.rs` in that crate |
  | GD or Nelder-Mead versus argmin               | `crates/competitor-bench/benches/gd_nm.rs` and `src/bin/verify_gd_nm.rs`           |
  | Private COBYLA driver and allocation guards   | `crates/competitor-bench/benches/cobyla.rs` and `tests/cobyla_allocations.rs`      |
  | Public COBYLA versus the `cobyla` crate       | `crates/competitor-bench/benches/gd_nm.rs` and `src/bin/verify_gd_nm.rs`           |
  | Convergence across libraries                  | `crates/competitor-bench/src/bin/trace.rs`                                         |
  | Population solver thread scaling              | `crates/competitor-bench/benches/population_scaling.rs`                            |
  | Active CMA-ES solver and eigen-kernel comparisons | `crates/competitor-bench/src/bin/verify_cmaes.rs` and `benches/cmaes.rs` in that crate |
  | Finite-difference kernels, callback work, and allocations | `crates/competitor-bench/benches/numdiff.rs`, `src/bin/verify_numdiff.rs`, and `NUMDIFF.md` in that crate |

For example, verify and then list or run a focused comparison:

```sh
cargo run --release -p competitor-bench --bin verify_gd_nm
cargo bench -p competitor-bench --bench gd_nm -- --list
cargo bench -p competitor-bench --bench gd_nm -- 'gd_rosenbrock_n2/'
cargo run --release -p competitor-bench --bin verify
cargo bench -p competitor-bench --bench compare -- 'exp_fit/'
cargo bench -p basin --features nalgebra_latest,faer_latest,problems \
  --bench lm_backends -- 'lm_gram/'
```

Run only cases relevant to the question. Inspect timed boundaries: setup
exclusion in `iter_batched` does not prove contestants charge initialization,
allocation, conversion, and teardown symmetrically. Trace observers can add
cost, and sampling by evaluation versus iteration is not interchangeable.

The LM competitor harness normally matches nalgebra 0.34 on both sides;
`basin-latest` changes Basin's side. Avoid workspace-wide or `--all-features`
timings that silently change backends or enable parallelism. Record resolved
backend versions and features; moving `*_latest` aliases are not lasting pins.

COBYLA's public comparison is part of `gd_nm`; its private driver has a
separate benchmark and allocation tests. For LM, use `src/bin/verify.rs`,
`verify_lm_qr.rs`, and `verify_lm_stopping.rs` in `competitor-bench`.
Neither maintained comparison needs GlobalSearch.
Historical measurements are leads to remeasure, not current facts.

For CMA-ES, `verify_cmaes --untraced` checks the fixed published seed without trace
sampling. It also accepts `--dimension`, `--eigen-update-period`,
`--generations` (fixed work with the target disabled), and `--repetitions`.
The default retains the published settings, including eigen refresh every three
generations; changing the refresh period changes trajectories and must be
reported explicitly. The `cmaes` Criterion benchmark
isolates eigendecomposition on identical covariances, including a reference
with Basin's tolerance and iteration cap.

Run the Basin-only refresh comparison in release mode:

```sh
cargo run --locked --release -p competitor-bench --bin verify_cmaes -- --refresh-sweep
```

The CSV reports a median and range for three timed solves per case and period,
plus the actual objective calls, stop reason, and target attainment. Use
`--repetitions N` for a different positive odd sample count. A successful row
requires an independently checked objective and box violation within the common
limits. Compare timings within a matched workload, solver, backend, precision,
and seed. Identical resolved periods share one measurement.

Extend a nearby harness when needed. Keep native competitors and instrumentation
in `competitor-bench` or an isolated investigation workspace. Profile the actual
production implementation; avoid a copied kernel that can drift from it, and do
not expose private internals as new public API solely for a probe.

## CMA-ES refresh adoption

The Basin 4.0 default remains one generation. In the initial sweep, the
candidate period was two generations on the 48-dimensional rotated ellipsoid
with population 8 and condition number `1e2`. Both schedules reached the
target in all 96 matched runs. The median paired solve time fell by 43%, and
the largest increase in objective calls was 10.7%.

With condition number `1e6`, the same candidate missed a target that period 1
reached. On bounded CMA-ES with faer `f32` and seed 0, period 1 reached the
objective target of `1e-8` in 88,155 calls. The candidate stopped at the
10,000-generation limit after 90,000 calls with an objective of
`3.31e-8`. Both returned feasible points. The adoption gate required
preserving every target reached with period 1, so this case ruled out the
change. The expanded sweep also includes 48-dimensional Ackley and a
96-dimensional rotated ellipsoid with the default population size.

Reproduce the regression by running the same case with fixed periods 1
and 3 and the candidate's resolved period:

```sh
cargo run --locked --release -p competitor-bench --bin verify_cmaes -- \
  --refresh-sweep --regression-only
```

## Constrained and least-squares diagnostics

Run the diagnostics comparison and its focused checks from the workspace root:

```sh
cargo run --locked --release -p competitor-bench --bin verify_diagnostics
cargo test --locked -p competitor-bench --bin verify_diagnostics
```

The comparison uses `f64`, Basin's `Vec` backend, NLopt 2.9.1 through
`nlopt` 0.8.1, and `rmpfit` 2.0.0. It reports native stops and counted
solve work alongside independent checks at the returned points. See the
public result APIs for [NLopt](https://docs.rs/nlopt/0.8.1/nlopt/struct.Nlopt.html)
and [MPFIT](https://docs.rs/rmpfit/2.0.0/rmpfit/struct.MPStatus.html).

### SLSQP against NLopt

Both implementations start at the same point on HS71 and the 30-variable
quadratic with an active ball constraint. They use analytic derivatives and
an objective evaluation budget of 200. Basin checks Kraft's absolute
accuracy at `1e-10`, with an additional iteration limit of 200. NLopt checks
absolute objective change at `1e-10` and uses a constraint tolerance of
`1e-8`. Basin counts nonlinear value blocks and whole constraint Jacobians.
NLopt's scalar constraint callbacks can also fill one derivative row.

Basin's selected record supplies the point, objective, objective gradient,
and an unscaled L1 violation. The retained solver supplies QP multiplier
estimates, the projected Lagrangian-gradient infinity norm, and nonlinear
inequality complementarity. Reevaluating the constraints and derivatives at
the returned point reproduces both metrics with those multipliers. The stop
report identifies the convergence predicate and its operands. NLopt's Rust
result supplies the point, objective, and status. Rebuilding the KKT evidence
also requires estimating the multipliers.

The audit fits nonlinear multipliers by least squares on coordinates more
than `1e-6` from each box face, with the known active inequality on each
fixture. It then checks stationarity on every coordinate, checks the
inequality multiplier sign, and measures nonlinear and box complementarity
separately. It uses `1e-8` for total violation and complementarity and
`1e-5` for stationarity.

| Case and implementation | Objective calls | Recomputed L1 violation | Projected stationarity | Nonlinear complementarity |
| ----------------------- | --------------: | ----------------------: | ---------------------: | ------------------------: |
| HS71, Basin             |               6 |                6.64e-11 |                8.62e-9 |                  1.71e-11 |
| HS71, NLopt             |               7 |                6.07e-11 |                8.62e-9 |                  1.41e-11 |
| Quadratic, Basin        |              45 |                6.33e-15 |               8.40e-10 |                  6.33e-14 |
| Quadratic, NLopt        |              65 |                 2.12e-9 |                2.08e-5 |                   2.12e-8 |

Both implementations pass the HS71 checks. NLopt reports `FtolReached` on
the quadratic with stationarity and complementarity above the audit thresholds.
The separately reconstructed box complementarity is below `1e-8` on all
four runs.

### Least squares against MPFIT

The harness uses both starts on all 27 NIST datasets and an analytic fit
with residuals `[a² - 4, b - a, c - 3]`. The analytic fit has an active
upper bound `a <= 1`. The fixed variant also sets `c = 2`. The independent
checks use RSS (twice Basin's cost), box violation, the projected `Jᵀr`
infinity norm, and box complementarity estimated from `Jᵀr` on near-active
faces. NIST recovery requires relative errors within `1e-4` for RSS and
every parameter.

All implementations use the same analytic models. Basin LM and MPFIT use
Jacobian-column scaling, an initial step factor of 100, and
orthogonality, relative model reduction, and scaled-radius tests at
`1e-10`. Their update rules and the timing of their checks differ.
Bounded LM uses Coleman–Li scaling and Nielsen damping. TRF uses
Coleman–Li scaling and a reflective trust region. Both use their native
scaled-gradient test plus observed relative cost and step checks at
`1e-10`. The harness counts residual and Jacobian work separately.
MPFIT's analytic derivative callback also evaluates the residual. Version
2.0.0 starts `n_fev` at one, and the audit reports it alongside the actual
count. The budgets are 20,000 residual evaluations and 2,000 Basin
iterations or MPFIT acceptances.

| Implementation   | NIST fits recovered | Most recent Jacobian callback at returned point |
| ---------------- | ------------------: | ----------------------------------------------: |
| Basin LM         |               52/54 |                                           14/54 |
| Basin bounded LM |               41/54 |                                            5/54 |
| Basin TRF        |               50/54 |                                           54/54 |
| MPFIT            |               51/54 |                                           12/54 |

Bounded LM, TRF, and MPFIT recover the active-bound fit (RSS 9) in 5,
4, and 6 residual evaluations, respectively. The direct bounded LM run
with a fixed coordinate moves that coordinate by about `2e-10` during
strict-interior initialization and fails with non-finite damping. TRF and
MPFIT handle the fixed coordinate directly. Eliminating it in the bounded
LM adapter also recovers the constrained fit (RSS 10). Those three
successful runs use 4, 6, and 5 residual evaluations, respectively.

Basin's `PointState` publishes the accepted point and cost. The termination
report carries the tests that passed. Opt-in model observations can carry a
scaled-gradient or orthogonality comparison at the **base point** of the last
trial, even when the trial moves to a different point. MPFIT returns the
accepted parameters and residuals, a stop code, covariance, uncertainty, and
the count of pegged coordinates. Exact feasibility and projected stationarity
and complementarity at the fit can be checked by reevaluating the model.

The one-step checks use a Jacobian that changes with the fitted parameters.
LM, bounded LM, and MPFIT evaluate it at the seed and return the next accepted
point. TRF calls the Jacobian at the acceptance point before publication,
including when the iteration limit fires. Callers can save that Jacobian in
their callback and match its coordinates against the returned point. MPFIT
also provides a matching callback on the analytic fits that stop on
orthogonality, with columns only for the free coordinates on the fixed fit.
To obtain a full Jacobian, evaluate the original model's analytic Jacobian
at the returned parameters, or reuse a callback capture whose point and
column set match.

## Account for Basin's execution model

- `crates/basin/src/core/problem.rs`: `Problem::counts()` holds authoritative
  per-kind `EvalCounts`. A fused `cost_and_gradient` call increments both the
  cost and gradient counters; `residual_and_jacobian` likewise increments two
  counters. Their sum is not a count of separate user callback invocations or an
  estimate of elapsed cost. Record whether the problem overrides these fused
  methods when comparing callback work.
- `crates/basin/src/core/state.rs`: `CountsMirror` maps per-run wrapper deltas
  into the concrete state's counters. Some derivative-free outer states fold all
  evaluation kinds into `cost_evals()`, including work from inner solvers.
  Inspect the state's mapping before comparing that number with a reference's
  objective-evaluation count.
- `crates/basin/src/core/numdiff.rs`: `FiniteDiff` expands a derivative request
  into multiple underlying cost or residual calls. Count those raw callbacks
  separately when comparing analytic derivatives, finite differences, or
  libraries with different evaluation accounting. Nonlinear constraint-vector
  calls also need separate instrumentation; `EvalCounts` has no constraint
  counter.
- `crates/basin/src/core/executor.rs` and `crates/basin/src/core/solver.rs`: for
  an executor-overhead comparison, a manual loop must call `Solver::init` before
  `next_iter`, honor framework and solver stops, and maintain equivalent state
  bookkeeping. The executor moves the state into `next_iter`; do not presume it
  clones the solver workspace every iteration. A mid-iteration stop can perform
  work without incrementing the completed-iteration count.
- `crates/basin/src/core/inner.rs`: distinguish a shared `Problem` wrapper from
  a fresh wrapper around an adapter. Shared-wrapper counts already include inner
  work; adapter-wrapper counts require the documented roll-up. Match restart,
  warm-start, and resume behavior when timing solver composition.
- `crates/basin/src/core/constraint.rs`: Basin's nonlinear inequalities use
  `c(x) <= 0`. Preserve feasibility when adapting a reference that uses the
  opposite sign. Charge projection, barrier, and penalty work to the layer that
  actually performs it.

Keep the exact backend release fixed for implementation comparisons. Basin's
`nalgebra_v0_34`, `ndarray_v0_17`, and `faer_v0_24` are examples of versioned
features; consult `crates/basin/Cargo.toml` for the intended release. Feature
unification enables implementations for every requested release. Inspect the
resolved feature graph with the same features as the benchmark and check the
actual vector and matrix types. Legacy backend-specific quasi-Newton state
aliases still select the newest enabled version; use explicit generic state
types to keep a comparison fixed.

## Measure a stable baseline

Build optimized binaries before measurement. Record revisions and local diffs,
lockfiles, compiler, build flags, CPU, thread counts, workload, and timing
boundaries. Keep compiler settings and dependency versions matched unless they
are the variable under investigation. Use a separate worktree or archive and
separate build outputs for historical revisions; preserve the user's checkout.

Measure uninstrumented code on an otherwise idle machine. Warm up, repeat, and
retain samples plus a median and spread or confidence interval. For tiny
operations, batch enough fresh operations to amortize clock and process costs,
consume results with `std::hint::black_box`, and reset mutable solver state
between independent solves. Measure cold initialization separately if relevant.

Criterion supports a named baseline before a change and comparison afterward:

```sh
cargo bench -p competitor-bench --bench gd_nm -- \
  'gd_rosenbrock_n2/' --save-baseline before
# After the candidate change, using the same feature and build configuration:
cargo bench -p competitor-bench --bench gd_nm -- \
  'gd_rosenbrock_n2/' --baseline before
```

Named results normally live under `target/criterion`; separate target
directories need an explicit plan to retain and compare their samples. For small
effects or machine drift, prebuild both contestants and alternate or randomize
their order across rounds. Hyperfine is useful for a repeated-solve executable,
not for timing Cargo builds or Criterion's fixed-duration statistical harness.

Use the same available CPU affinity for serial contestants when helpful. Record
Rayon and BLAS thread settings. For parallel scaling, use the existing
`population_scaling` protocol with `RAYON_NUM_THREADS=1` and the desired pool
size in separate processes; do not pin a multicore run to one CPU. Report
single-thread latency separately from parallel throughput. Native results do not
establish browser/WASM performance; measure the requested runtime.

## Establish the comparison

Keep the user's solver, variant, workload, and reference. If none is specified,
choose a representative existing case and state the choice. For a regression,
compare against a known earlier revision.

Before choosing kernel optimizations for an implementation comparison, establish
whether the competitor uses the same algorithm variant. If it differs, retain
the user's competitor as the practical performance target and identify a
matching reference for implementation-cost comparisons. Prefer author-maintained
reference code or an established library implementing the same variant. Record
versions or commits and material differences; use primary documentation or
source to resolve ambiguous reference semantics.

When whole-solver trajectories differ, start the implementation comparison with
kernels on identical inputs. If no matching reference is available, document
that limitation and use matched before/after work to support implementation
speedups without attributing the whole competitor gap to implementation
overhead.

Before timing, run each contestant once and record:

- Problem dimensions, data, starting point, scalar type, bounds, and
  constraints. Check constraint signs, residual scaling, and whether the
  reported objective is a sum of squares or half that sum.
- Algorithm settings: line search, memory size, initial simplex or population,
  trust-region radii, scaling, restart rules, and random seeds as applicable.
- Actual stopping conditions, including implicit solver stops, adapter tolerance
  floors, and the meaning of each budget. Equal tolerance values or iteration
  caps do not establish equal work across libraries.
- Returned objective, feasibility, gradient or residual norm when relevant,
  termination reason, iterations, and objective, derivative, and constraint
  evaluations. Validate the returned point independently outside the timer.
  Distinguish current and best-so-far state, and count verification calls
  separately. Budget exhaustion alone does not establish convergence;
  distinguish budget-limited output from verified success and invalid non-finite
  results.

Choose the measurement that answers the question:

- **Implementation cost:** matched work, such as a kernel on identical inputs or
  the same solver steps. Verify evaluation counts and numerical results; equal
  iterations alone can hide different line-search or inner-solver work.
- **Solver effectiveness:** time and evaluations to a common accuracy and
  feasibility target, or quality achieved within a common budget. Retain failed
  cases and success rates. Use paired seeds and multiple starts when stochastic
  variation matters. Convergence traces help explain differing trajectories.

Use both views when fewer evaluations could explain the speed difference. For
non-solver components, establish equivalent outputs and the relevant work unit
instead of imposing solver convergence criteria.
