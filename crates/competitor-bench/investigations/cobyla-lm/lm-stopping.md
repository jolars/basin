# Relative stopping in Levenberg–Marquardt

Investigated September 11, 2026, from Basin revision
`5ae54633ba7e89a131c670ff3d6a2cb0c445f432`, following the [damping
comparison](lm-damping.md). Production solver behavior was unchanged during
that investigation. The later [arithmetic correction](#arithmetic-correction)
fixes the reproduced false orthogonality stop.

The current relative tests can mistake a heavily damped step for convergence.
Disabling them exposes a separate problem: an accurate rounded solution can
retain nonzero gradient and a substantial residual/Jacobian cosine, so Basin
keeps trying steps until its budget expires. The retained experiments isolate
both effects. Neither removing one relative test nor setting both tolerances to
zero solves them together.

The next implementation should distinguish numerical inability to progress from
convergence, preserve the existing optional stopping contracts, and make any
scaled trust-radius criterion explicit. A universal replacement for the current
tests is not established by these workloads.

## Reproduce and interpret

```sh
cargo run -p competitor-bench --release --bin verify_lm_qr -- --damping \
  > target/lm-stopping-baseline.csv
cargo run -p competitor-bench --release --bin verify_lm_qr -- --stopping \
  > target/lm-stopping-profiles.csv
cargo run -p competitor-bench --release --bin verify_lm_stopping \
  > target/lm-stopping-diagnostics.csv
```

Use the default competitor-bench features: both libraries use nalgebra 0.34.2
and `f64`; the reference is `levenberg-marquardt` 0.15.0. Adding `basin-latest`
or workspace-wide features changes Basin's backend version. This run used Rust
1.89.0 on NixOS. These are deterministic solver-effectiveness experiments, with
no timing or implementation-speed claim.

The [profile results](lm-stopping-profiles.csv) retain all 500 combinations of
the original 20 starts, five solver routes, and five stopping profiles. Models,
observations, analytic Jacobian validation, and callback budgets are unchanged
from the damping comparison. Every run permits `200*(n+1)` residual callbacks,
including initialization; Jacobian calls are counted separately. Instrumented
residual counts are checked against solver reports. Independent verification
calls are excluded from those counters.

All Basin profiles enable exact-zero absolute gradient detection and
orthogonality tolerance `1e-12`. The relative settings are:

  | Profile          | Model-reduction tolerance | Step tolerance |
  | ---------------- | ------------------------: | -------------: |
  | `relative`       |                   `1e-12` |        `1e-12` |
  | `gradient-only`  |                  Disabled |       Disabled |
  | `model-only`     |                   `1e-12` |       Disabled |
  | `step-only`      |                  Disabled |        `1e-12` |
  | `exact-progress` |                       `0` |            `0` |

For Basin, disabled means `None` and zero means an enabled exact-zero test. The
reference represents disabled relative tests with zero tolerances, but retains
its machine-precision stops. Its `gradient-only` and `exact-progress`
configurations are consequently identical. Equal numeric tolerances do not make
the libraries' stopping rules equivalent.

Fit success means `||r|| / ||y|| <= 1e-10`; recovery means maximum equivalent
parameter error `<= 1e-6`. These are assessment targets, separate from the
solver's successful-stop flag. Recovery excludes the two exactly dependent
linear starts, and SVI sigma is replaced by its absolute value before
comparison. A solver-declared convergence below can still miss either target.

## Which test stops the collinear solves?

The full-rank linear problem has rows `[1, 1]`, `[1, 1+1e-8]`, and
`[1, 1-1e-8]`, with observations generated from `[1, -1]`.

  | Start    | Nielsen QR profile | Residual calls | Relative residual | Parameter error | Termination   |
  | -------- | ------------------ | -------------: | ----------------: | --------------: | ------------- |
  | `[0, 0]` | Relative           |              2 |               `1` |             `1` | Converged     |
  | `[0, 0]` | Model only         |              2 |               `1` |             `1` | Converged     |
  | `[0, 0]` | Step only          |             36 |               `0` |       `1.99e-9` | Converged     |
  | `[0, 0]` | Gradient only      |             36 |               `0` |       `1.99e-9` | Converged     |
  | `[0, 0]` | Exact progress     |             36 |               `0` |       `1.99e-9` | Converged     |
  | `[2, 1]` | Relative           |              8 |             `0.5` |           `0.5` | Converged     |
  | `[2, 1]` | Step only          |              8 |             `0.5` |           `0.5` | Converged     |
  | `[2, 1]` | Model only         |              9 |             `0.5` |           `0.5` | Converged     |
  | `[2, 1]` | Exact progress     |             39 |             `0.5` |           `0.5` | Converged     |
  | `[2, 1]` | Gradient only      |             52 |             `0.5` |           `0.5` | Solver failed |

Cholesky has the same early inaccurate stops: two calls from zero, eight from
the other start, or nine with model stopping alone. Unlike QR, disabling model
stopping from zero does not yield exact residuals: step-only Cholesky takes 105
calls and finishes at relative residual `4.44e-8`. Its parameter error `4.76e-8`
passes recovery, while its fit misses the deliberately strict target relative to
the tiny observations. Gradient-only Cholesky exhausts 600 calls from both
starts.

Trust-region QR solves both starts in three calls under every profile, with
exact floating-point residuals and parameter errors below `4e-9`. The MINPACK
reference needs two or three calls. Trust-region Cholesky reaches the recovery
target from zero but also misses the relative fit target; factorization remains
a separate consideration when damping permits nearly undamped steps.

## Why the predicted-reduction check is insufficient

The shared Cholesky/QR implementation already tests

```text
|actual| <= ftol * old_cost
predicted <= ftol * old_cost
gain_ratio <= 2
```

where `predicted = (mu * hᵀDh - hᵀg) / 2`. Assuming an accurate damped solve,
this is the reduction of the unregularized linear least-squares model evaluated
at the **damped step**. It is not the maximum possible reduction in that model.
Large damping can suppress the weak direction and make both predicted and actual
reductions small, even on a linear problem that admits an exact fit.

This is already algebraically the MINPACK reduction test. The complete
algorithms differ: MINPACK chooses damping against a scaled trust radius and
tests that updated radius against `xtol * ||sqrt(D) x||`. Basin tests the
unscaled internal attempted step against `xtol * ||x||`, under either damping
policy. Both libraries evaluate progress tests after rejected trials as well as
accepted ones. Basin accepts positive gains; MINPACK requires a gain ratio of at
least `1e-4`. See the primary [`lmder`
source](https://netlib.org/minpack/lmder.f) and the [reference
implementation](https://docs.rs/levenberg-marquardt/0.15.0/src/levenberg_marquardt/lm.rs.html).

Simply requiring acceptance would still allow stops at tiny accepted steps.
Replacing the attempted step with an observed iterate change would instead make
every rejection look like a zero step. Changing the norm to a scaled norm
addresses coordinate sensitivity but cannot by itself undo excessive damping in
a weak direction.

The [diagnostic trace](lm-stopping-diagnostics.csv) confirms this mechanism.
From zero, Nielsen QR's first accepted trial reduces cost by `6.66e-14` relative
to the old cost; its independently reconstructed model reduction is `6.67e-14`.
Both pass `ftol=1e-12`. The relative step is `1`, so step stopping does not
explain this result. An SVD solve at that base point retains both singular
directions and predicts essentially all of the residual can be removed. At the
other start's stopping trial, both unscaled and scaled step ratios are
`6.77e-13`, while the SVD model still predicts complete reduction with a
relative step of about `1`. Scaling the step norm alone would still stop this
case.

These diagnostics wrap the production solver through `Executor`. They
reconstruct steps by subtracting the base coordinates from residual-callback
trial coordinates; they cannot recover the internal step lost to rounding. Their
model reduction is evaluated directly from `J` and that reconstructed step,
independently of the solver's damped-equation identity. Differences at rounded
solutions must not be interpreted as the internal predicted reduction. The rank
threshold is `epsilon * max(m,n) * sigma_max`; the reported potential is the
fraction of squared residual projected onto the retained left singular vectors.
Diagnostic evaluations and SVDs do not enter callback counts or affect stopping.
Boundary rows describe the base point; `final` rows describe the returned point.
The trace retains the first ten boundaries, first unchanged trial, last
boundary, and final state of each of 21 solves.

## Why accurate rounded solutions keep running

At the favorable narrow-SVI start `[0.035, 0.12, -0.2, 0.01, 0.25]`:

  | Solver          | Profile                                     | Residual calls | Relative residual | Equivalent parameter error | Termination             |
  | --------------- | ------------------------------------------- | -------------: | ----------------: | -------------------------: | ----------------------- |
  | Trust-region QR | Relative / step only                        |             13 |        `3.13e-17` |                 `1.01e-10` | Converged               |
  | Trust-region QR | Model only                                  |             54 |        `3.13e-17` |                 `1.01e-10` | Converged               |
  | Trust-region QR | Gradient only / exact progress              |          1,200 |        `3.13e-17` |                 `1.01e-10` | Budget                  |
  | MINPACK         | Relative / step only                        |             10 |        `3.61e-17` |                 `1.38e-11` | Relative step           |
  | MINPACK         | Gradient only / model only / exact progress |             15 |        `2.55e-17` |                 `1.41e-11` | No improvement possible |

Basin's final QR residual norm is `1.20e-17`, but its normalized orthogonality
is about `0.271`, far above `1e-12`. Normalizing a tiny nonzero residual does
not make that direction cosine small. Exact-zero gradient and exact-zero
progress are therefore unsuitable substitutes for a finite precision stopping
policy.

The gradient-only trace first evaluates exactly unchanged trial coordinates on
residual call 54, yet continues to call 1,200. Its final SVD model potential is
still about `0.126`. Thus a small *relative* undamped model potential is also
not a universal way to recognize a rounded solution: the remaining absolute
residual is already tiny.

The reference retries its reduction, scaled-radius, and orthogonality tests at
machine epsilon, returning `NoImprovementPossible` when a stopping inequality
passes at machine epsilon but not at the requested tolerance. This is
**unsuccessful** according to its API, even when both assessment targets pass.
Basin currently has no equivalent native precision stop. See the reference's
[termination
API](https://docs.rs/levenberg-marquardt/0.15.0/levenberg_marquardt/enum.TerminationReason.html).

Across all 20 starts, trust-region QR reaches the same 18 fit and 16 recovery
targets under every profile. Its residual-call totals are 2,524 with relative or
step-only stopping, 2,860 with model-only stopping, 9,754 with exact-progress
stopping, and 10,461 with gradient-only stopping. These totals retain every
unsuccessful run.

The other two narrow-SVI starts exhaust all 1,200 residual calls under every
solver route and every profile, missing both targets. Their persistence does not
support a claim that stopping changes would solve their trajectory problem.

## Identifiability limits

The `1e-8` linear system is mathematically full rank, and QR with trust-region
damping resolves it. Its inaccurate Nielsen stops cannot be dismissed as exact
nonidentifiability. The exactly dependent case, in contrast, has many valid
parameter vectors; recovery of the chosen generating vector is not a correctness
criterion.

Raw SVI has an exact sigma-sign symmetry and a narrow-window Jacobian condition
number about `1.48e8` at the generating parameters. Fit, stationarity, and
parameter recovery must remain separate outcomes. Even MINPACK describes
orthogonality as having no general relationship to solution accuracy, and its
scaled parameter criterion primarily protects larger scaled coordinates. See the
original [MINPACK
documentation](https://www.math.utah.edu/software/minpack/minpack/lmder1.html).

## Coordinate sensitivity and an arithmetic bug

A separate linear check fits `[x1, x2/s]` to `[1, 1]`, starting at `[0, s]`.
Changing `s` changes only the second coordinate's units. With `s=1`, Nielsen QR
uses six residual calls and obtains an exact fit. With `s=1e13`, it stops after
two calls with relative residual `7.06e-4` and first-coordinate error `9.99e-4`.
The stopping trial has unscaled relative step `9.99e-14` but scaled relative
step `0.707`. Both columns remain above the diagnostic SVD rank threshold. This
demonstrates the documented unscaled step criterion's unit dependence,
separately from the collinear damping issue.

The probe also reproduces an arithmetic defect in orthogonality. For the
one-dimensional residual `1e-100 * (x+1)` at `x=0`, the Jacobian and residual
are both `1e-100`, the gradient is `1e-200`, and the true cosine is `1`. Basin
squares the gradient before dividing by the squared column norm; that square
underflows to zero and incorrectly passes orthogonality tolerance `1e-12`. With
absolute gradient tolerance zero and both progress checks disabled, QR reports
convergence after initialization without moving, at relative residual and
parameter error `1`. This is a reproducible arithmetic bug, not a choice of
stopping policy.

Normalize before squaring or use safe scaled comparisons in a separate fix, with
regression coverage for `f32`, `f64`, and the supported backends. Audit the
attempted-step comparison's squared norms at the same time: source inspection
also shows possible overflow and zero-times-infinity behavior there, but this
probe does not exercise those additional extreme-step cases.

## Recommended implementation boundaries

Fix the reproduced orthogonality arithmetic bug independently. For stopping
policy, pursue the following separately:

1. **Add explicit numerical no-progress handling before pursuing a new
   convergence claim.** A precision-limited stop should be distinguishable from
   convergence and budget exhaustion. Test it at accurate rounded fits and at
   inaccurate over-damped points. It may limit wasted work in both cases; it
   does not guarantee recovery. Define its opt-in behavior and composition
   semantics before implementation.
2. **Consider an opt-in scaled trust-radius criterion specifically for
   trust-region damping.** This has a direct reference contract. Preserve the
   documented unscaled attempted-step setter and do not manufacture a radius
   from one Nielsen step.
3. **Treat a damping-independent model check as a separate experiment.** A
   rank-aware undamped least-squares solve can show remaining model reduction
   and veto a premature progress stop. It needs an explicit numerical rank
   convention, backend support, and cost accounting. On nonlinear models its
   unrestricted step can leave the valid local neighborhood; a large relative
   potential reduction at a rounded residual also need not justify more work.
   Testing lower damping is another candidate, but changes the search policy.

Do not silently floor configured tolerances at epsilon or enable unconditional
zero-residual stopping. The current contract distinguishes `None` from zero, and
the existing disabled-stopping regression explicitly expects a budget stop at
zero residual. Reusing `SolverFailed` for all rounded fits also affects
composition because that reason propagates as an unrecoverable inner failure.
These are observable design choices, not internal arithmetic substitutions.

## Verification

The initial 200-row damping output matches the previously retained results
byte-for-byte. The extended probe preserves the ordinary, `--steps`, and
`--damping` outputs byte-for-byte and validates all 500 stopping-profile runs.
The diagnostic probe validates callback budgets and residual/Jacobian accounting
for 21 additional solves, checks the original models' analytic Jacobians, and
asserts the tiny-orthogonality reproduction. Its overlapping final results agree
with the unwrapped profile runs. Rust formatting and workspace all-target,
all-feature Clippy pass. No production library or web code changed.

## Arithmetic correction

The shared Cholesky/QR stopping implementation now compares the unsquared
orthogonality and attempted-step inequalities using normalized binary
significands and separate exponents. Norms remain factored by their infinity
norm, so neither a large norm nor a tiny tolerance must be squared or
materialized as an overflowing or underflowing product. Checking each gradient
component against its column norm also avoids losing a subnormal projection
during division. Exact-zero tests inspect the original gradient or step;
`None` still disables each check.

The current `verify_lm_stopping` probe asserts the corrected result: the
`1e-100 * (x+1)` case takes seven residual calls and reaches `x=-1` with zero
residual, instead of stopping at initialization. The other 20 final diagnostic
rows match the retained investigation results exactly. The CSV linked above
remains the historical baseline.

Regressions in `crates/basin/tests/levenberg_marquardt_stopping.rs` cover
`f32` QR and `f64` Cholesky/QR on all four dense backends, plus both sparse
Cholesky backends. They exercise gradient and step square underflow and
overflow, tiny tolerances beside large coordinates, and disabled versus
exact-zero settings. Unit tests cover subnormal projections, norms beyond
the scalar range, zero columns, and non-finite inputs. This correction retains
the unscaled internal attempted-step criterion and its evaluation stage;
numerical no-progress handling remains a separate design task.
