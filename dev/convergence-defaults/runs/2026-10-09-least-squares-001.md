# LM/TRF measurement pilot, 2026-10-09

The [frozen manifest](2026-10-09-least-squares-001.toml) and [retained machine
report](../least-squares-results.json) cover 80 analytic checks and 360 default
NIST solves: 15 development datasets, both published starts, both native
precisions, and six LM/TRF routes. Native diagnostics and callback accounting
pass verification. Independent returned-point checks use the [frozen local
references](../nist-reference.md). This run changes no default, safeguard,
reference eligibility, or quality target, and selects no policy.

## Findings

Native outcomes are 101 convergence reports, 177 numerical stalls, 40 numerical
failures, eight iteration limits, and 34 callback budget errors. No wall guard
fired. A coherent point returned with a stall or numerical failure can meet the
external target; a convergence report can miss it. Callback errors have no
returned point, even when an earlier published point attained the target.

At designated targets `1e-6` for `f64` and `1e-3` for `f32`, returned-point
success is 163/180 and 115/174 eligible cases, respectively. Nelson start 2's
`f32` designated target remains reference-pending for all six routes and is
withheld from those scores. All nine frozen pending precision/start/target
combinations remain visible: 54 withheld route/target records across the grid.

  | Route                        | `f64` returned / eligible | `f64` first attained | `f32` returned / eligible | `f32` first attained |
  | ---------------------------- | ------------------------- | -------------------- | ------------------------- | -------------------- |
  | LM normal, Nielsen           | 28/30                     | 28                   | 23/29                     | 23                   |
  | LM normal, trust radius      | 29/30                     | 29                   | 23/29                     | 23                   |
  | LM pivoted QR, Nielsen       | 28/30                     | 28                   | 24/29                     | 24                   |
  | LM pivoted QR, trust radius  | 29/30                     | 29                   | 24/29                     | 24                   |
  | Legacy `Trf`                 | 20/30                     | 28                   | 7/29                      | 16                   |
  | Full `TrustRegionReflective` | 29/30                     | 29                   | 14/29                     | 14                   |

First attainment uses published points, separately from sampled trials. Legacy
TRF illustrates the distinction: eight `f64` and nine `f32` cases attained the
designated target but subsequently exhausted their callback budget and exposed
no returned point. Its seven successful `f32` returns came at iteration limits,
with median 9,920 and maximum 10,017 physical calls after first attainment. For
other routes, the median additional work among eligible cases that attained the
target and returned a point ranges from 11 to 18 calls; the maximum is 255.
These are observations under the frozen controls, not selected policies or
performance rankings.

  | Development family      | `f64` returned / eligible | `f32` returned / eligible |
  | ----------------------- | ------------------------- | ------------------------- |
  | Exponential mixtures    | 47/48                     | 41/48                     |
  | Logistic models         | 22/24                     | 20/24                     |
  | MGH10                   | 8/12                      | 5/12                      |
  | Misra models and BoxBOD | 53/60                     | 38/60                     |
  | Nelson                  | 9/12                      | 1/6                       |
  | Power models            | 24/24                     | 10/24                     |

The family table pools routes only to expose coverage and difficult cases;
CDP-1's equal family weights remain unchanged. No pooled selection score is
computed. Per-route outcomes, target grids, first attainment, and returned and
last-published quality remain in the machine report.

Four convergence reports miss an eligible designated target. Trust-radius LM
with both factorizations on BoxBOD start 1 in `f64`, and normal-equation
trust-radius LM on that start in `f32`, reach a saturated exponential with tiny
stationarity but normalized objective gap about 0.04645 and parameter error
about 113.2. Legacy TRF on MGH17 start 1 in `f64` has objective gap about
`2.86e-10` and stationarity about `4.82e-13`, but parameter error about 0.7675
even after pair symmetry. These reports describe their native predicate; they do
not certify the desired parameter solution. Local reference certificates and
pending start-basin classification do not justify labeling these cases as
premature termination or claiming stricter tolerances would fix them.

## Accounting and diagnostic validation

The NIST ledger contains 372,015 physical calls: 360 fused residual/Jacobian
initializations, 339,407 residual calls, and 32,248 Jacobian calls.
Authoritative logical counts are 339,799 residual and 32,610 Jacobian requests,
with zero cost requests. The difference is exactly 360 fused categories plus 34
denied requests: 32 residual denials and two Jacobian denials. Residual-derived
objectives charge neither a physical cost callback nor a `CostFunction` request.
No denied request overshoots the cap.

Native records preserve 339,439 attempted trials, 32,250 accepted and published
trials, 307,157 rejections, 139 completed non-finite trials, and 32 denied trial
callbacks. All 101 passing native convergence comparisons match their
termination reports; 338,709 failing and 132,162 disabled checks are retained.
The analytic phase independently exercises two accepted full-TRF trials whose
subsequent Jacobian budget denial prevents publication. It also preserves
non-finite rejections, rich LM relative-test operands, and initialization caps.

The verifier checks actual and predicted model decrease, safeguarded ratios,
full TRF radius updates, native norm/reference operands, and publication
ownership. Identity bounds use absolute operation terms and explicitly allow
underflow of squared steps before damping rescales them. They are not accuracy
certificates. Independent quality verification performs 11,253 native point
probes and 3,670 interval quality evaluations outside solve ledgers and timers.
Target-specific parameter bounds skip points that cannot pass the joint target;
every final publication receives a full check.

## Freeze, corrections, and verification

Solver and harness source commit `0ed083c` and planned manifest commit `dd01375`
preceded retained execution. The analytic phase passed before the corpus ran.
The frozen solver source remains unchanged throughout the pilot.

The first corpus identity pass rejected Bennett5 start 1's `f64` QR/Nielsen
prediction at sequence 546 because its allowance scaled the tiny result rather
than absolute terms within the dot product. Commit `bbcde7a` corrects that bound
and adds a cancellation regression. It also conservatively bounds parameter
uncertainty across all symmetry branches. No multiplier, target, partition,
reference eligibility, or solver setting changed.

An initial quality pass was manually stopped after native validation to avoid
fully evaluating every near-reference trajectory point. Commit `25cbcf7` freezes
target-specific parameter screening and first-attainment searches. Final
verification uses fresh probe files under `nist-checked/` and identical
hard-linked solve CSVs. Original solve outputs and the interrupted pass's probe
files remain in `nist/`; their hashes are recorded. The final verifier passes
all 360 cases and retains every outcome.

Validation includes the pure-Rust solver suite, 39 competitor tests on nalgebra
0.34, all 39 corresponding tests on nalgebra 0.35 across focused commands, four
Python tests with seven mutation controls, workspace all-target/all-feature
clippy with warnings denied, rustfmt, public rustdoc, and both WASM builds.
Recording-on/off tests preserve exact trajectories, counts, and outcomes for all
six routes and both precisions. Lifecycle tests cover fresh initialization and
exact checkpoint continuation. Documentation formatting, lint, and local links
are checked separately.

Use the manifest's commands with fresh output paths. For exact reproduction, use
its frozen solver revision and the final verification revision. On that combined
source, the final checker can read freshly generated `nist/` outputs directly;
`nist-checked/` was needed only to preserve the interrupted pass's probe files.
Both binaries and the checker refuse overwriting evidence.

## Remaining gates

This completes the analytic unconstrained nalgebra LM/TRF measurement pilot, not
all of step 5. Bound-active and fixed-coordinate cases, robust losses, finite
differences, transformed problems, other backends, failed model attempts before
residual trials, and full secular-search work remain open. Native probe
arithmetic screens cover the tested points and adapter, not arbitrary backend
linear-algebra errors. Holdout candidate outcomes remain sealed. Timing includes
instrumentation and an unoptimized build and is not speed evidence.

Next, extend the analytic diagnostics to bound-active and fixed-coordinate TRF
and robust-loss LM/TRF, with independently known KKT or robust references. Keep
reference branch classification and precision gates explicit before any
candidate sweep. No tolerance recommendation follows from this run.
