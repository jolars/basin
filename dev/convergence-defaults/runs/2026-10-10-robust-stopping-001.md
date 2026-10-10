# Robust LM stopping composition review

The [manifest](2026-10-10-robust-stopping-001.toml) and [retained
report](../robust-stopping-results.json) record 336 stopping ablations and a
352-case reproduction of the previous robust pilot. Source commit `2a5e89e` and
planned manifest `510d5db` preceded execution. All settings use existing public
builders; the solver algorithm, defaults, and safeguards are unchanged. This is
a step 5 measurement review, not calibration or held-out validation.

## Design and verification

Six existing unbounded robust fixtures cross normal equations and pivoted QR,
Nielsen and trust-region damping, and native `f32` and `f64`. The fixtures
retain their raw residuals, Huber, soft-L1, or Cauchy loss, starts, scales, and
analytic minima. Every solve has a physical cap of 4000 and an iteration cap of
10,000.

Seven configurations isolate the default absolute gradient, all existing
relative probes, model reduction alone, step alone, model reduction OR step,
robust normalized gradient alone, and trust radius alone. Relative tolerances
remain `1e-8` in `f64` and `1e-4` in `f32`. Each relative probe retains the
exact-zero absolute gradient check. The radius test is inactive under Nielsen
damping, so that route is an exact-gradient control rather than a radius test.

The independent checker uses the previous pilot's 100-digit robust objective,
gradient, model, and publication verification. Every configuration receives the
same parameter and stationarity limits: `1e-6` in `f64` and `1e-3` in `f32`.
These unit-scale measurement checks are stricter than the previous relative
probe checks. They do not certify CDP-1 target eligibility or select a
tolerance. Every ablation shares its entire common callback prefix with its
default control. All 96 default and combined-relative controls reproduce their
baseline callbacks, publications, native observations, checks, counts, and
outcomes exactly. Verification stays outside solve ledgers and timers.

## Results

Each configuration has 48 solves. Quality counts include returned stalled
points; convergence codes remain separate. A premature stop below means that a
converged point misses the common quality check and the matching default
continuation reaches it within the same budget.

  | Configuration                  | Converged | Stalled | Quality passed | Confirmed premature | Physical calls |
  | ------------------------------ | --------- | ------- | -------------- | ------------------- | -------------- |
  | Default absolute gradient      | 33        | 15      | 48             | 0                   | 973            |
  | All relative probes            | 48        | 0       | 38             | 10                  | 564            |
  | Model reduction                | 48        | 0       | 38             | 10                  | 582            |
  | Trial step                     | 48        | 0       | 44             | 4                   | 776            |
  | Model reduction OR trial step  | 48        | 0       | 38             | 10                  | 572            |
  | Robust normalized gradient     | 36        | 12      | 48             | 0                   | 1025           |
  | Trust radius or exact gradient | 30        | 18      | 48             | 0                   | 1080           |

The 336 runs use 5572 physical calls and return 291 convergence outcomes and 45
stalls. All 1928 accepted trials are published. There are no callback denials or
non-finite model predictions; 224 trial objectives are non-finite and rejected.
The 352-case baseline reproduces all 348 unaffected prior outcomes and the four
known legacy-TRF corrections, using 2175 physical calls.

### Cauchy identifies the mechanism

Both Nielsen factorizations reproduce the four severe Cauchy stops. The
model-only, step-only, combined-relative, and model-OR-step probes stop at the
same trial in each route and precision. The gradient default continues along the
identical callback prefix to the analytic minimum.

  | Precision | Probe calls | Parameter error | Robust gradient magnitude | Gain ratio | Default calls | Normalized-gradient calls |
  | --------- | ----------- | --------------- | ------------------------- | ---------- | ------------- | ------------------------- |
  | `f64`     | 21          | 0.9503          | 0.4994                    | 0.9984     | 96            | 100                       |
  | `f32`     | 18          | 0.2445          | 0.2307                    | 1.1682     | 55            | 55                        |

The Cauchy residual is `x-1`, with start `x=-1`. Its initial residual curvature
is negative, so the safeguarded model clips it to native epsilon. Rejections
increase Nielsen damping. At the premature `f64` trial, damping is about
`9.11e14`, the Marquardt diagonal is about `0.02676`, and the step is about
`2.05e-14`. In one dimension, the damped solve gives
`h = -g / (J_model² + mu*D)`: large damping makes the step and its predicted
reduction tiny without making the gradient small.

Both model reduction and step pass at all four stopped trials. Each trial has a
positive actual reduction, is accepted and published, and has a gain ratio above
0.25. Consequently, requiring both tests, requiring acceptance, or adding that
model-agreement threshold would still stop at these same inaccurate points. This
conclusion uses measured terminal predicates; it does not claim to run a new
conjunction API.

The 0.25 comparison comes from [SciPy 1.16.2's TRF completion
test](https://github.com/scipy/scipy/blob/v1.16.2/scipy/optimize/_lsq/common.py#L652-L676),
which also clips robust curvature. It is a diagnostic comparison, not an
equivalent LM algorithm or a transferred stopping policy. Both trust-region LM
routes pass the Cauchy quality checks in this pilot.

The six additional common-quality misses concern the non-finite-domain soft-L1
fixture: all four `f64` routes and the two Nielsen `f32` routes. Their relative
model-reduction stops pass the old, looser probe-quality checks, but default
continuations improve them to the common thresholds. They are tolerance and
accuracy findings distinct from the severe Cauchy mechanism.

## Disposition and limits

The existing absolute-gradient control avoids all measured premature stops.
Public LM and QR rustdoc now explain that progress tests combine with OR and
that neither small progress nor a positive gain ratio requires stationarity. For
a solve requiring stationarity, keep gradient checks alone and disable native
and observed progress checks. Thresholds still depend on objective and parameter
scales; this evidence does not select a general default.

The robust normalized gradient also returns good points here, but its
denominator shrinks near a zero-residual minimum. On the single-residual Cauchy
fixture, its value tends to one until the residual is exactly zero, explaining
the extra `f64` work. It remains a separate calibration candidate, not a
replacement chosen from these small controls.

The [decision
record](../decisions.md#d005-screen-robust-lm-progress-tests-against-stationarity)
rules out the demonstrated unguarded progress configurations for these controls.
Other losses, dimensions, fixed robust coordinates, derivative modes,
transformations, precision certificates, and complete inner work retain their
gates. Remaining backend versions also require validation. Step 5 stays open.
The next task is to extend these controls to larger and rank-deficient robust
problems and arctangent loss before testing a stationarity guard or sweeping
robust candidates.

## Validation

The manifest records reproduction commands and hashes. The independent checker,
four composition tests with six corruption controls, and the twelve existing
checker tests pass on the ablation; existing checker tests also pass on the
baseline. All 37 measurement tests pass on nalgebra 0.34 and 0.35. Permanent
Cauchy gradient-control checks cover both damping modes and dense LM
factorizations on Vec, nalgebra, ndarray, and faer in both precisions, plus
supported sparse normal-equation paths. The full pure-Rust solver suite,
workspace all-target/all-feature clippy with warnings denied, public rustdoc,
Rust formatting, and documentation checks pass.

A verifier-only correction recognizes that native trust-radius evidence is
unavailable on a non-finite trial. The original solve CSVs and every setting
remain unchanged. The final checker also compares all baseline control records,
not timing fields. An initial test compile needed an explicit state constructor
for its generic solver bound; this did not affect measured solver code.
