# Configurable damping for Levenberg–Marquardt

Investigated September 11, 2026, from Basin revision
`37eb2824c4895a48b90ed0f4c9f8e6a7f3562b46`. This follows the [production QR
comparison](lm-qr.md) and separates damping selection from factorization and
stopping rules.

The retained results support an explicit trust-region damping option. It closes
the narrow-SVI gap from the favorable third start and improves the nearly
collinear solves. It does not solve the other two difficult SVI starts within
their budgets, and it needs more callbacks on one scaled linear start. Nielsen
damping remains the default.

```rust
use basin::{LevenbergMarquardt, LmDamping};

let solver = LevenbergMarquardt::new()
    .with_damping(LmDamping::TrustRegion)
    .with_pivoted_qr();
```

The new choice also works with the existing Cholesky route. Its scalar search
chooses damping to approximately satisfy a bound on `sqrt(hᵀDh)`, where `D` is
Basin's existing squared column scaling. The initial bound is
`factor * sqrt(xᵀDx)`, or `factor` at zero, with factor 100 by default and an
explicit `.with_initial_step_bound(factor)` setting. The search reuses the same
prepared linear model for damping attempts.

The initialization and radius updates follow MINPACK's trust-region approach,
but the parameter search uses safeguarded interpolation with the existing
regularized solves. It is not a port of
[`lmpar`](https://netlib.org/minpack/lmpar.f), which uses triangular-factor
information for its Newton correction. Basin also retains its positive-gain
acceptance rule and existing stopping tests. The reference accepts gains of at
least `1e-4` and has additional machine-precision stops. These differences
prevent an exact MINPACK equivalence claim.

## Reproduce

```sh
cargo run -p competitor-bench --release --bin verify_lm_qr \
  > target/lm-damping-baseline.csv
cargo run -p competitor-bench --release --bin verify_lm_qr -- --damping \
  > target/lm-damping-results.csv
```

The ordinary invocation and `--steps` retain their previous comparisons.
`--damping` crosses Nielsen and trust-region damping with Cholesky and pivoted
QR, alongside `levenberg-marquardt` 0.15.0. Both libraries use nalgebra 0.34.2,
`f64`, analytic Jacobians, monotone column scaling, and the same synthetic
observations and starts. The default competitor-bench features select Basin's
nalgebra 0.34 backend. Do not add `basin-latest` or workspace-wide features when
reproducing this comparison.

The run used Rust 1.89.0 on NixOS with an Intel Core Ultra 7 155U. It measures
solver effectiveness through callbacks and achieved accuracy. It contains no
wall-time measurements and supports no implementation-speed claim.

## Workload and validation

All 20 original starts remain:

- Two starts each for linear systems with column separation `1e-4`, `1e-8`, and
  exact dependence.
- Two starts for a diagonal system scaled from `1e-8` to `1e8`.
- Three starts each for raw SVI observations over `[-0.5, 0.5]` and
  `[-0.01, 0.01]`.
- Three starts each for SSVI observations at separated and closely spaced
  maturities.

The shared [models](support/lm_models.rs) check every analytic Jacobian column
against centered finite differences at the generating parameters and every start
before solving. In particular, SSVI differentiates its actual
`eta / (theta^gamma * (1+theta)^(1-gamma))` formula, including the `ln(1+theta)`
derivative term. The historical downstream Jacobian mismatch is described in the
[original
investigation](README.md#ssvi-jacobian-mismatch-in-the-downstream-source).

Each solve permits `200*(n+1)` residual callbacks including initialization.
Basin's iteration cap is one less because each completed outer iteration
attempts one residual evaluation after initialization. Linear solve retries do
not call the residual. The callback enforces this budget; actual Basin residual
counts and the reference's reported evaluations are asserted against
instrumented calls. Jacobian callbacks are counted separately. These models do
not override the fused residual/Jacobian callback. Final residuals, Jacobians,
and parameter errors are independently recomputed outside the solve and excluded
from its counters.

Each CSV row records the solver's own successful-stop flag and termination
reason separately from two shared accuracy targets:

- Fit: `||r|| / ||y|| <= 1e-10`; a zero observation norm uses the smallest
  positive `f64` denominator.
- Recovery: maximum absolute parameter error `<= 1e-6`, after replacing raw
  SVI's sigma by its absolute value.

These are workload-specific comparison targets, not new solver defaults. Exactly
dependent systems have nonunique solutions, so their recovery flag cannot
determine solver correctness. Raw SVI's sigma sign is unidentifiable. The
narrow-SVI Jacobian has condition number about `1.48e8` at the truth; accurate
fitted observations alone do not establish accurate parameters. The CSV
therefore retains residual norm, raw and equivalent parameter errors, gradient
infinity norm, and residual/Jacobian-column orthogonality.

## Stopping profiles

The `relative` profile requests orthogonality, relative model reduction, and
relative step tolerances of `1e-12`. Basin's absolute gradient tolerance is
zero, so it retains exact-zero detection. The `gradient-only` profile disables
Basin's model-reduction and step checks with `None`, keeping the same
orthogonality and exact-zero gradient checks.

The reference has no optional tolerance representation: its gradient-only
profile sets `ftol = xtol = 0`. It still has zero-residual termination and
machine-epsilon no-improvement stops. Its step test uses a scaled trust radius,
while Basin's existing test uses the attempted unscaled step. Matching numeric
tolerance values does not make these termination rules equivalent. In
particular, a no-improvement report or exhausted budget is not counted as
solver-declared convergence even when the independent fit target passes.

The earlier ordinary probe uses the reference's `.with_tol(1e-12)`, which sets
`ftol = xtol = 1e-12` and **`gtol = 0`**. The earlier QR report's claim that all
three reference relative tolerances were `1e-12` was inaccurate. The new
`--damping` comparison explicitly sets `gtol = 1e-12` for both profiles; the
ordinary probe stays unchanged for reproducibility.

## Unchanged baseline

The initial production run reproduced 17 successful stops out of 20 starts for
both Basin routes and 18 for the reference. These success counts include the
nearly collinear false-positive stops. For separation `1e-8`, Basin stops after
two residual calls from zero with parameter error `1`, and after eight calls
from the other start with error `0.5`. Both factorizations have the same
behavior. The reference recovers the generating parameters in two or three
calls.

Both Basin routes exhaust 1,200 residual calls at every narrow-SVI start. From
`[0.035, 0.12, -0.2, 0.01, 0.25]`, both finish with residual norm about
`1.97e-9` and maximum parameter error `0.038`. The reference uses ten residual
calls and seven Jacobians, with residual norm `1.39e-17` and parameter error
`1.38e-11`. It exhausts the same budget at the other two narrow-SVI starts. This
baseline confirms that the original damping question remains relevant after QR
was added.

## Configurable-damping results

The [retained CSV](lm-damping-results.csv) contains all 200 combinations of 20
starts, five solver routes, and two stopping profiles. Under the `relative`
stopping profile:

  | Solver                  | Successful stops / 20 | Fit target / 20 | Recovery target / 18 | Residual calls, total |
  | ----------------------- | --------------------: | --------------: | -------------------: | --------------------: |
  | Nielsen / Cholesky      |                    17 |              15 |                   13 |                 3,807 |
  | Nielsen / QR            |                    17 |              15 |                   13 |                 3,808 |
  | Trust region / Cholesky |                    18 |              17 |                   16 |                 2,607 |
  | Trust region / QR       |                    18 |              18 |                   16 |                 2,524 |
  | MINPACK reference       |                    18 |              18 |                   16 |                 2,524 |

The recovery denominator excludes the two exactly dependent linear starts.
Totals retain budget exhaustion and inaccurate successful stops; they are not
timings or an estimate of callback cost.

At the favorable narrow-SVI start `[0.035, 0.12, -0.2, 0.01, 0.25]`:

  | Solver                  | Residual / Jacobian calls | Residual norm | Equivalent parameter error | Stop          |
  | ----------------------- | ------------------------: | ------------: | -------------------------: | ------------- |
  | Nielsen / Cholesky      |             1,200 / 1,199 |     `1.97e-9` |                  `3.80e-2` | Budget        |
  | Nielsen / QR            |             1,200 / 1,199 |     `1.97e-9` |                  `3.80e-2` | Budget        |
  | Trust region / Cholesky |                   13 / 11 |    `1.39e-17` |                 `4.80e-11` | Converged     |
  | Trust region / QR       |                    13 / 8 |    `1.20e-17` |                 `1.01e-10` | Converged     |
  | MINPACK reference       |                    10 / 7 |    `1.39e-17` |                 `1.38e-11` | Relative step |

At the other two narrow-SVI starts, all five routes exhaust 1,200 residual
calls. Trust-region damping improves the fitted residuals over Nielsen, but it
does not reach either comparison target. The reference retains the best residual
and equivalent parameter error at both starts. Broad SVI and both SSVI datasets
reach the targets under all routes. The closely spaced SSVI cases use seven
residual calls at all three starts under either trust-region route, versus 13–32
under Nielsen.

For the `1e-8` collinear system, trust-region QR reaches exact floating-point
residuals in three calls from either start, with parameter errors about
`2.24e-9` and `3.31e-9`. Trust-region Cholesky needs 84 calls from zero and
finishes with residual norm `4.71e-16`. That is good absolute accuracy and meets
the recovery target, but misses the fit target relative to the tiny observation
norm. It needs three calls from the other start. The factorization therefore
still matters when the damping policy permits nearly undamped steps.

The scaled diagonal system supplies a contrasting case: trust-region damping
needs 22–23 residual calls from zero, versus six under Nielsen. At zero the
initial scaled radius is 100, while the scaled solution norm is about `1e8`.
Successive radius increases account for the extra work. From `[2, 2, 2]`, the
initial radius admits the solution step and either trust-region route needs two
calls. This dependence on the initial radius supports retaining an explicit
policy and step-bound setting.

## Relative stopping remains a separate choice

Disabling both relative progress checks changes Nielsen QR's zero-start
collinear result from an inaccurate two-call stop to a recovered solution after
36 calls. From the other start, however, it fails after 52 calls with the same
parameter error `0.5`. Disabling the checks alone therefore does not resolve the
full collinear problem.

The narrow-SVI trajectories at the two difficult starts are unchanged when the
relative checks are disabled. From the favorable start, trust-region damping
still reaches residuals near `1e-17`, but both Basin routes run to the
1,200-call budget because neither exact-zero gradient nor orthogonality is
satisfied at the rounded solution. The reference stops after 15 calls with
`NoImprovementPossible("xtol")`, which is unsuccessful by its API but meets both
independent accuracy targets.

Across all starts, the gradient-only profile retains the same number of fit and
recovery successes for both trust-region routes as the relative profile, while
their total residual calls rise to 11,855 and 10,461. This supports keeping
numerical stopping configurable and reporting final accuracy separately from
termination. The damping option preserves the existing relative stopping
contracts and defaults; this investigation does not establish a universal
replacement for those tests.

## Verification

The unchanged baseline was collected before production edits. After adding the
policy, the ordinary probe's complete CSV remained byte-for-byte equal to that
baseline. The new comparison passed all Jacobian checks, finite parameter
checks, residual-budget guards, and callback-accounting assertions across its
200 runs. Numerical regression and backend coverage for the production policy
live with the solver tests.

The routine pure-Rust suite passes, including existing `f32` round-trip tests.
Focused tests cover both damping strategies on all four dense backends, `f32`
QR, both sparse Cholesky backends, analytic radius checks, deficient models,
callback reuse and errors, initialization resets, and large inactive parameters
whose squared scaled norm overflows. Workspace all-feature Clippy, Rust
formatting, rustdoc, and both default and no-default-feature WASM builds pass.
Web formatting, type checking, and the production build pass; web lint reports
existing diagnostics in files unchanged from the parent revision.
