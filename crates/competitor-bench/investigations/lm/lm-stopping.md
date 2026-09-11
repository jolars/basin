# Relative stopping in Levenberg–Marquardt

Investigated September 11, 2026, from Basin revision
`5ae54633ba7e89a131c670ff3d6a2cb0c445f432`, following the [damping
comparison](lm-damping.md). Production solver behavior was unchanged during
that investigation. The later [arithmetic correction](#arithmetic-correction)
fixes the reproduced false orthogonality stop.
The subsequent [numerical no-progress safeguard](#numerical-no-progress-safeguard)
changes the default stopping policy; the original findings below describe the
historical baseline with that safeguard disabled.
The later [scaled trust-radius option](#scaled-trust-radius-convergence) adds
an independent, disabled-by-default convergence test for trust-region damping.

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
  --disable-numerical-no-progress \
  > target/lm-stopping-baseline.csv
cargo run -p competitor-bench --release --bin verify_lm_qr -- --stopping \
  --disable-numerical-no-progress \
  > target/lm-stopping-profiles.csv
cargo run -p competitor-bench --release --bin verify_lm_stopping -- \
  --disable-numerical-no-progress \
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
the unscaled internal attempted-step criterion and its evaluation stage.
Numerical no-progress handling was implemented separately, as described below.

## Numerical no-progress safeguard

Collected September 11, 2026, using Rust 1.89.0 and the default
competitor-bench features, from Basin base revision
`25d1b04fc2386bd3a08b75fb8667c1d3d30c88fa` plus this safeguard.

Both LM factorizations now report `NumericalNoProgress` after one rejected
finite trial for which the computed `x + h` equals the base `x` componentwise.
The comparison uses the actual trial coordinates: rejection alone always leaves
the stored iterate unchanged and is insufficient. Parameters, step, residuals,
gradient, costs, and reduction diagnostics must be finite. The regular residual
callback and damping/cache updates still run, and native convergence tests take
precedence. Callback errors and model-solve failures retain their existing paths.
The stop occurs mid-iteration, before observed-step or cost-change checks at
the next boundary; those checks can be isolated by disabling the safeguard.

This safeguard is enabled by default. `with_no_progress_check(false)` on
either solver restores the previous behavior. It is independent of convergence
tolerances: `None` still disables a particular test, and zero still requests an
exact-zero threshold. With all convergence tests disabled, zero residual now
produces a numerical no-progress stop unless the safeguard is also disabled.
No tolerance is floored at epsilon. Scaled-radius convergence and
damping-independent model checks remain separate work.

`NumericalNoProgress` is neither convergence nor budget exhaustion, and
`is_failure()` returns false so an outer solver can consume the finite result
and continue. It does not establish mathematical stagnation, fit accuracy, or
parameter recovery. In particular, excessive damping can trigger it while a
weak direction remains unresolved. It also need not detect stagnation involving
distinct trial coordinates.

The [enabled](lm-no-progress-enabled.csv) and
[disabled](lm-no-progress-disabled.csv) profile results each retain all 500
combinations, with the same models, starts, callback budgets, and independent
assessment targets. They compare the safeguard on and off on the same
implementation, including the earlier arithmetic correction. Historical CSVs
are unchanged. Across the 400 Basin runs, all reported parameter vectors and
fit/recovery target outcomes are identical between settings. The 100 reference
runs are unchanged in every field.

| Basin route | Residual calls, disabled | Residual calls, enabled | Numerical stops | Fit targets | Recovery targets |
| --- | ---: | ---: | ---: | ---: | ---: |
| Nielsen Cholesky | 26,407 | 19,512 | 20 | 75 | 68 |
| Nielsen QR | 19,936 | 19,414 | 17 | 78 | 68 |
| Trust-region Cholesky | 31,939 | 13,748 | 26 | 85 | 85 |
| Trust-region QR | 28,830 | 13,251 | 20 | 90 | 80 |

Totals fall from 107,112 to 65,925 residual calls, saving 41,187. The 83 new
numerical stops replace 44 budget stops, seven `SolverFailed` stops, and 32
later `SolverConverged` stops. The last group confirms that solver-declared
convergence and independently assessed accuracy remain separate outcomes.
This is evaluation accounting, not a timing comparison or a general guarantee
that stopping earlier preserves accuracy.

The [updated diagnostic trace](lm-no-progress-diagnostics.csv) includes the
accurate narrow-SVI trust-region QR fit: gradient-only stopping now exits at
residual call 54 instead of 1,200, retaining relative residual `3.13e-17` and
equivalent parameter error `1.01e-10`. From the inaccurate nonzero collinear
start, Nielsen Cholesky and QR both stop at call 12 with relative residual and
parameter error about `0.5`. The two difficult narrow-SVI starts still exhaust
their budgets under every profile and route, missing both targets.

Reproduce the new comparison with the default competitor-bench features:

```sh
cargo run -p competitor-bench --release --bin verify_lm_qr -- --stopping \
  > target/lm-no-progress-enabled.csv
cargo run -p competitor-bench --release --bin verify_lm_qr -- --stopping \
  --disable-numerical-no-progress > target/lm-no-progress-disabled.csv
cargo run -p competitor-bench --release --bin verify_lm_stopping \
  > target/lm-no-progress-diagnostics.csv
```

Regression coverage includes rounded quadratic roots with nonzero residuals,
inaccurate heavily damped affine fits, zero steps, rejected distinct trials,
and native-convergence precedence across the supported dense and sparse
backends. Additional tests preserve callback errors and non-finite trial
rejections, verify opt-out forwarding and QR conversion, and check that
`CmaInject` continues after fresh LM inner runs with exact evaluation accounting.

## Scaled trust-radius convergence

Collected September 11, 2026, using Rust 1.89.0 on NixOS, from Basin base
revision `16c12c6d698465bcde052ae1eb9e09845c1d4bf0` plus this change. The default
competitor-bench features retain nalgebra 0.34.2 on both sides and
`levenberg-marquardt` 0.15.0 as the reference. These are deterministic
solver-effectiveness comparisons, with no timing claim.

Both LM factorizations now offer
`with_relative_trust_radius_tolerance(value)`. It checks
`delta <= tolerance * ||sqrt(D) x||` after the radius update and acceptance
decision, using the current monotone Marquardt diagonal and the accepted
iterate, or the base after rejection. This follows the observation stage of
[MINPACK's `lmder`](https://netlib.org/minpack/lmder.f), while retaining Basin's
positive-gain acceptance rule and damping search. Basin recomputes the norm
with the current diagonal even after rejection; MINPACK retains its previously
stored `xnorm` until acceptance. This is not a claim of complete MINPACK parity.

The setting defaults to `None` and remains inactive under Nielsen damping.
Zero requests an exactly zero radius, not an exactly zero step; the existing
positive radius safeguards remain. The check requires finite trial quantities
and reports `SolverConverged` before numerical no-progress handling. The norm
comparison factors weighted coordinates and binary exponents, avoiding
overflow or underflow from materializing the norm or tolerance product.
The unscaled attempted-step setter keeps its existing meaning. No solver
default, damping arithmetic, or recovery guarantee changes.

The [enabled](lm-trust-radius-enabled.csv) and
[disabled](lm-trust-radius-disabled.csv) results each contain 180 rows:
20 starts, three solver routes, and three profiles. The filenames refer to
the numerical no-progress safeguard. All Basin profiles retain exact-zero
absolute gradient detection and orthogonality tolerance `1e-12`, with model
reduction disabled. `gradient-only` disables both progress checks;
`step-only` enables unscaled attempted-step tolerance `1e-12`; `radius-only`
instead enables scaled-radius tolerance `1e-12`. Budgets remain
`200*(n+1)` residual callbacks, including initialization, with independent
verification excluded. Reported residual counts match instrumented callbacks.

With numerical no-progress enabled:

| Basin route | Profile | Residual calls | Jacobian calls | Fit targets / 20 | Recovery targets / 18 |
| --- | --- | ---: | ---: | ---: | ---: |
| Trust-region Cholesky | Gradient only | 2,858 | 2,571 | 17 | 16 |
| Trust-region Cholesky | Step only | 2,607 | 2,554 | 17 | 16 |
| Trust-region Cholesky | Radius only | 2,605 | 2,554 | 17 | 16 |
| Trust-region QR | Gradient only | 2,747 | 2,497 | 18 | 16 |
| Trust-region QR | Step only | 2,524 | 2,478 | 18 | 16 |
| Trust-region QR | Radius only | 2,524 | 2,478 | 18 | 16 |

Fit means relative residual at most `1e-10`; recovery means equivalent
parameter error at most `1e-6`, excluding both exactly dependent linear
starts. The CSV retains the probe's raw parameter-target flags, which can be
true by coincidence for a nonidentifiable generating vector; the table
excludes those starts explicitly.

Radius-only and step-only return identical reported parameters and assessment
outcomes for every Basin case. Cholesky saves one residual call on the
`collinear-1e-8` zero start and one on the accurate narrow-SVI start. QR uses
two extra calls on the `collinear-1e-4` zero start and saves two on the accurate
narrow-SVI start, stopping there at call 11 with relative residual `3.13e-17`
and equivalent parameter error `1.01e-10`. Both difficult narrow-SVI starts
still exhaust 1,200 calls and miss both targets. The analytic regression
tests also demonstrate that a deliberately small radius can satisfy this
criterion far from a solution. Radius convergence does not establish recovery.

Disabling numerical no-progress changes only the Basin gradient-only rows:
residual totals rise to 11,855 for Cholesky and 10,461 for QR. All step-only,
radius-only, and reference rows remain identical. The reference's step-only
and radius-only profiles intentionally use the same `xtol` setting because
MINPACK already tests its scaled radius. Each uses 2,524 residual and 2,459
Jacobian calls, meeting 18 fit and 16 identifiable recovery targets. Its
gradient-only profile uses 2,548 residual and 2,472 Jacobian calls and retains
machine-precision stops. Matching tolerances do not make the full stopping
policies equivalent.

Reproduce with the default competitor-bench features:

```sh
cargo run -p competitor-bench --release --bin verify_lm_qr -- --trust-radius \
  > target/lm-trust-radius-enabled.csv
cargo run -p competitor-bench --release --bin verify_lm_qr -- --trust-radius \
  --disable-numerical-no-progress > target/lm-trust-radius-disabled.csv
```

The existing 500-row `--stopping` output is byte-identical before and after
this change with the radius tolerance left disabled. Focused regressions
cover acceptance and rejection, monotone scaling, coordinate rescaling,
exact-zero semantics, non-finite trials, callback errors, model failure,
native-stop precedence, and builder forwarding. They run across all four
dense backends, `f32` QR, and sparse Cholesky; arithmetic unit tests cover
extreme `f32` and `f64` weighted norms.
