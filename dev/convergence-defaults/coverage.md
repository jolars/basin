# Experiment coverage

Status: fixtures and baseline pilots are implemented. Partitions and case
weights remain proposed until the [protocol](protocol.md) is frozen. No
numerical default has been selected. The [coverage
pilot](runs/2026-10-02-coverage-002.md) exercises development cases only.

The public corpus still has 28 catalog entries: 16 fixed two-dimensional
problems, one fixed four-dimensional problem, and 11 scalable problems. It
already supplies many multimodal and nonsmooth objectives, but only five
families expose residuals and Jacobians, its constrained examples are linear
quadratics, and its problem implementations use `f64`. Counting catalog entries
therefore overstated its coverage for this investigation.

The added fixtures live in
[`competitor-bench::convergence`](../../crates/competitor-bench/src/convergence.rs).
They are benchmark fixtures, separate from Basin's public problem API and
visualizer. This lets diagnostic constructions support the investigation without
making each one a supported library type.

## Added cases

  | Stratum                      | Definitions | Starts or intervals              | Precision    | Reference and purpose                                                                                                             |
  | ---------------------------- | ----------: | -------------------------------: | ------------ | --------------------------------------------------------------------------------------------------------------------------------- |
  | NIST nonlinear least squares |          27 |            Both published starts | `f32`, `f64` | All StRD certificates, near-zero and nonzero residuals, and all published difficulty levels                                       |
  | Analytic least squares       |          15 |                  Two starts each | `f32`, `f64` | Dimensions 2, 5, and 10; Hessian condition numbers 1, `1e4`, and `1e8`; rotation, rank deficiency, and paired observation noise   |
  | Constrained                  |           8 |                  Two starts each | `f32`, `f64` | Active and inactive disks, a scaled ellipse, a parabolic inequality, linear equality, active bounds, a fixed coordinate, and HS71 |
  | Scalar roots                 |           4 | Two sign-changing intervals each | `f32`, `f64` | Simple, rescaled, multiple, and trigonometric roots; position error and residual are separate                                     |
  | Scalar minima                |           3 |               Two intervals each | `f32`, `f64` | Quadratic, flat quartic, and asymmetric exponential minima                                                                        |

There are **57 fixture definitions**, before expanding starts, precision,
solver, or seed. Three analytic definitions also expose a finite-sum gradient
with nonzero sample noise at the optimum. These are the same cases, not three
extra families. Full objective callbacks remain deterministic. SGD draws
minibatches with an explicit seed; the pilot pairs seeds `0..19` across starts
and precisions.

Generated manifests list [vector cases](cases.csv), [scalar
roots](scalar-roots.csv), and [scalar minima](scalar-minima.csv), including
references and both starts or intervals.

All vector fixtures currently use `Vec`. Production derivatives use forward
differentiation, with independent finite-difference checks in tests. Constants
and observations are converted before arithmetic in `f32` runs. Diagnostics
evaluate the returned parameters in `f64`; differences can expose cancellation
or coefficient rounding, but do not establish attainable targets.

Analytic least-squares controls apply an orthogonal Householder reflection to a
diagonal spectrum. Paired residual errors preserve the known minimizer and give
a reference cost of `n * noise²`. Rank-deficient controls report parameter error
as unavailable. NIST exponential sums, Gaussian mixtures, and harmonic models
also omit a unique-parameter error claim. Other parameter errors are diagnostics
against the supplied reference, not assertions of global identifiability.

Constrained references are analytic except HS71, whose reference is shared with
the existing [SLSQP tests](../../crates/basin/tests/slsqp.rs). The parabolic
reference solves `2x³ + x - 1 = 0`, with `y = x²`. The recorder independently
checks bound and constraint violations. It does not yet compute a KKT residual
or report that test as passed.

## Development and validation

Related starts, dimensions, precisions, and transformed copies stay together.
The disk and its scaled ellipse share a development family. All analytic
quadratics, including noisy and rank-deficient variants, remain in development.

  | Group                  | Development families                                                        | Reserved validation families                                       |
  | ---------------------- | --------------------------------------------------------------------------- | ------------------------------------------------------------------ |
  | NIST                   | Saturation, exponential ratio, exponential sum, power, arctangent, logistic | Gaussian, rational, log response, harmonic, reciprocal exponential |
  | Analytic least squares | Quadratic                                                                   | None yet; independent NIST families provide least-squares holdouts |
  | Constrained            | Disk, linear equality, box quadratic                                        | Parabola, HS71                                                     |
  | Scalar                 | Square root, flat polynomial, quadratic                                     | Trigonometric, scalar exponential                                  |

The vector runner excludes 11 reserved NIST datasets and two constrained
definitions by default. The scalar runner excludes two reserved definitions.
`--list` includes them for review without solving. `--include-validation`
explicitly enables solving them and should be used after the candidate and
selection rule are frozen. Unit tests check their reference formulas and
derivatives without optimizing them or choosing solver settings from them.

This split does not establish sufficient independent validation in every planned
stratum. In particular, the stochastic controls share the quadratic family. Add
independent stochastic families before selecting SGD defaults.

## Solver applicability and executable coverage

The table accounts for all 47 names in the [inventory](inventory.md). “Wired”
means a baseline entry point exists, not that calibration is complete.

  | Solver names                                                                                                   | Applicable fixtures                                              | Current runner                                                                                                |
  | -------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
  | `Brent`, `BrentDerivative`, `GoldenSection`                                                                    | Scalar minima                                                    | Wired in `convergence_scalar --minima`                                                                        |
  | `BrentRoot`, `SecantRoot`, `NewtonRoot`, `HalleyRoot`, `Toms748Root`                                           | Scalar roots                                                     | Wired in `convergence_scalar`                                                                                 |
  | `GradientDescent`, `NonlinearCg`, `Bfgs`, `Lbfgs`                                                              | NIST and analytic differentiable costs                           | Traits ready; wiring pending                                                                                  |
  | `ProjectedGradientDescent`, `Lbfgsb`                                                                           | Differentiable costs with bounds                                 | `Lbfgsb` wired on unbounded NIST and analytic controls; finite-bound wiring pending                           |
  | `TrustRegion`                                                                                                  | Smooth costs with Hessians or Hessian products                   | Second-derivative fixture interface pending                                                                   |
  | `Sgd`                                                                                                          | Paired-noise finite-sum controls                                 | Wired with seed pairing; momentum and refresh variants pending                                                |
  | `GaussNewton`, `LevenbergMarquardt`, `LevenbergMarquardtQr`, `Trf`, `TrustRegionReflective`                    | NIST and analytic residuals/Jacobians                            | `LevenbergMarquardt` and `TrustRegionReflective` wired; other solvers and bounded least-squares modes pending |
  | `NelderMead`, `Newuoa`                                                                                         | NIST and analytic costs                                          | `NelderMead` wired; `Newuoa` pending                                                                          |
  | `Bobyqa`, `Lincoa`, `Mads`, `SolisWets`                                                                        | Bound or linear constraint controls, as supported by each solver | Constraint adapters and wiring pending                                                                        |
  | `Cobyla`, `Slsqp`                                                                                              | General constrained controls                                     | `Slsqp` wired; `Cobyla` pending                                                                               |
  | `BarrierMethod`, `AugmentedLagrangianMethod`                                                                   | Constrained controls with appropriate inner solvers              | Adapter, stationarity, and nested work records pending                                                        |
  | `Direct`, `Gbnm`, `RandomSearch`, `SimulatedAnnealing`, `CmaEs`, `BoundedCmaEs`, `De`, `GlobalBestPso`, `Ssga` | Existing multimodal corpus plus bounded controls                 | Integration into this recorder and independent family splits pending                                          |
  | `BasinHopping`, `CmaInject`, `BoundedCmaInject`, `DeInject`, `MaLsCh`, `MaLsChCma`, `MaLsChSw`                 | Applicable local/global cases and inner-solver configurations    | Nested work and outer/inner stage records pending                                                             |

## Run and inspect

From the repository root:

```sh
cargo test --release -p competitor-bench --lib convergence
cargo build --release -p competitor-bench --bin convergence_suite --bin convergence_scalar
target/release/convergence_suite --list
target/release/convergence_suite --suite nist --case Misra1a
target/release/convergence_suite --suite analytic
target/release/convergence_suite --suite constraints
target/release/convergence_suite --suite stochastic --seeds 20
target/release/convergence_scalar
target/release/convergence_scalar --minima
```

`convergence_suite` accepts `--precision f32|f64|both`, `--max-iter`,
`--max-passes`, and `--final-only`. Defaults are both precisions, 200
iterations, 2,000 physical passes, and three seeds for a smoke run. The scalar
entry point uses both precisions, two intervals, default root budgets, and a
200-iteration/2,000-pass cap for minima. Root records use a separate schema
because signed values and brackets differ from optimization costs and states.

Vector and minimum traces contain initialization, published boundaries, the
owned stopping report and stage, authoritative callback categories, value and
derivative passes, returned-point diagnostics, best sampled points, and
solver-only elapsed time. A forward derivative pass propagates all coordinates
and need not cost the same as a value pass. Minibatch gradients currently
evaluate the full residual model and charge a full derivative pass, plus the
selected sample count. Constraint objective and constraint-block callbacks each
charge their actual pass. Compare these measurements within a case; they do not
imply equal costs across unrelated families.

Pass budgets are checked at published boundaries, so a step can overshoot the
cap. The actual count is preserved. The best sampled cost may come from a
rejected or infeasible point and is not a success claim. Derivative passes also
produce residual values and can update that diagnostic incumbent. Independent
diagnostics do not affect solver counters or timing. Non-finite model results
remain in the records.

The trace does not identify every internal rejected trial, collect unavailable
stopping tests, or handle typed application errors in vector fixtures, which use
`Infallible`. Root errors retain callback counts. The full measurement gates in
the protocol remain open.

## Policy and timing probes

`convergence_policies` compares the current defaults, named probes, and stricter
controls for Nelder–Mead, L-BFGS-B, Nielsen LM, and TRF. It runs Misra1a,
Chwirut2, DanWood, and a rotated five-dimensional quadratic with condition
number `1e8`, using both starts and precisions. `--case` can select other
development fixtures; validation families are rejected. The probes exercise
available checks from [candidate-pilot.md](candidate-pilot.md), rather than
implementing complete reference-library policies. Strict controls restart from
the same input with optional approximate checks disabled; native numerical
safeguards remain enabled. Nelder–Mead's default already disables those checks,
so its default and strict policies coincide.

`--mode trace` records every boundary, `--mode final` omits intermediate
diagnostics, and `--mode plain` runs an ordinary Executor with fixture recording
disabled. Compare timings only after returned points, stopping reports, and
authoritative counts match. `solver_seconds` sums timed initialization and steps
in recording modes, whereas plain mode times the whole Executor call.
`harness_seconds` includes diagnostics and trace construction, but excludes CSV
formatting, fixture construction, and solver construction. These controls expose
both charged-time perturbation and the additional cost outside the budget.

`--seconds` adds a time cap at published boundaries. Actual time, the largest
timed segment, and separate time/pass limit flags retain overshoot, including on
native stopping steps. A point published after a cap does not establish
attainment within that cap. Plain mode rejects time/pass caps. The runner also
emits the initial objective after rounding the start to the requested precision,
the reference objective, and its independent reevaluation.

The [serial
driver](../../crates/competitor-bench/scripts/convergence_policies.py) builds no
code while measuring. Its `controls`, `traces`, and `budgets` phases create
fresh output directories, run warmups, alternate process order, and record exact
commands and environment metadata. `summarize` checks numerical matching and
monotone work/time, then retains per-case timing quartiles, signed gaps,
observed target-grid crossings, and the last boundary within each budget. Grid
crossings are descriptive observations, not scored successes: reference
uncertainties and precision floors still require validation.

## Before calibration

The expanded set is sufficient for developing and piloting the harness. Next,
freeze the design, establish attainable targets and reference uncertainties,
finish solver wiring, and supply missing families identified by the matrix. A
larger undifferentiated catalog would not resolve those gaps.

Remaining additions include the fractional design for objective offsets,
objective and coordinate rescaling, finite-difference derivatives, explicit
non-finite probes, and deliberate stagnation. Add bounded least-squares cases,
an independent stochastic family, and the global/composed family split. Validate
finalists on the other three dense backends. Use equal-family summaries without
application preferences, and pilot the common time budgets and separate work
measurements described in the [protocol](protocol.md). Establish numerical
effect thresholds from development evidence before calibration. Bring any
remaining consequential accuracy/work tradeoff to the maintainer as a concrete
comparison.
