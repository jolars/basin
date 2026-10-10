# Larger and rank-deficient robust LM controls

The [manifest](2026-10-10-robust-extended-001.toml) and [retained
report](../robust-extended-results.json) record 504 full-budget stopping
ablations and 216 budget interruptions. Source `aa8e078` and planned manifest
`43e518e` preceded execution. All four built-in robust losses now have larger
and rank-deficient analytic controls. The original 336-run ablation reproduces
all five CSV files exactly, excluding elapsed time. This advances step 5;
calibration and holdout evaluation remain unopened.

## Models and independent quality

Nine fixtures cross normal equations and pivoted QR, Nielsen and trust-region
damping, and native `f32` and `f64`. They use the seven existing stopping
configurations, unchanged algorithm settings, unit loss scales, and analytic
derivatives. Every full solve has caps of 4000 physical callbacks and 10,000
completed iterations. Default-gradient runs also receive physical caps 0, 1, and 2.

The scalar arctangent fixture has residual `x-1`, start `-1`, and minimum `1`.
For each of Huber, soft-L1, Cauchy, and arctangent:

- The full-rank fixture has four parameters and twelve residuals. Each
  coordinate difference from `(1,-1,0.5,-0.5)` contributes `d`, `2d`, and `-d`.
  Its start is `(-1,1,-0.5,0.5)`, and its unique minimum has zero residuals.
- The rank-deficient fixture has four parameters and eight residuals. The sums
  `x0+x1-1` and `x2+x3+1` each contribute `d`, `2d`, and `-d`, followed by
  constant residuals `2` and `-2`. Its start is `(-0.5,-0.5,0.5,0.5)`. The
  Jacobian has rank two, and the global minimizer set is `x0+x1=1`, `x2+x3=-1`.
  The constant residuals give nonzero minimum objectives: `3`, `2*(sqrt(5)-1)`,
  `ln(5)`, and `atan(4)`, respectively.

Every variable-dependent loss term increases with squared residual, proving
these minima without a numerical reference solve. The checker measures
rank-deficient parameter error as infinity-norm distance to the affine minimizer
set: half the largest sum error. A valid nullspace displacement therefore
passes. All starts, coefficients, and representative minima are exactly
representable in both precisions.

The independent checker evaluates robust objectives, vector gradients, and
safeguarded models at 100-digit Decimal precision. Arctangent uses half-angle
reduction and an alternating series. All configurations share parameter and
stationarity limits of `1e-6` in `f64` and `1e-3` in `f32`. These are unit-scale
measurement checks, not CDP-1 precision eligibility certificates. Verification
adds no solve callbacks or timed work.

## Results

Each configuration has 72 full-budget solves. Point-quality counts include
stalled returns. A confirmed premature stop requires convergence at an
inaccurate point, an identical callback prefix, and a default continuation that
reaches the common quality check within the same budget.

  | Configuration                  | Converged | Stalled | Quality passed | Confirmed premature | Physical calls |
  | ------------------------------ | --------- | ------- | -------------- | ------------------- | -------------- |
  | Default absolute gradient      | 62        | 10      | 68             | 0                   | 3552           |
  | All relative probes            | 72        | 0       | 44             | 24                  | 2090           |
  | Model reduction                | 72        | 0       | 44             | 24                  | 2109           |
  | Trial step                     | 72        | 0       | 46             | 22                  | 2183           |
  | Model reduction OR trial step  | 72        | 0       | 44             | 24                  | 2093           |
  | Robust normalized gradient     | 67        | 5       | 68             | 0                   | 3565           |
  | Trust radius or exact gradient | 56        | 16      | 68             | 0                   | 3648           |

Full-budget solves use 19,240 physical calls and yield 473 convergences and 31
stalls. The 216 budget controls use 216 calls, retain 72 initialization errors
and 144 callback errors, and return no point. All 720 runs retain native stage,
completed iterations when available, logical callback categories, physical calls
by kind, and published points. They record 24,991 model-solve attempts, 7579
accepted and published trials, 3821 rejected trials, and 144 denied trial
callbacks. No trial objective or model prediction is non-finite. Complete work
before unrecorded model failures and within composed solvers remains gated.

### Progress stops extend beyond Cauchy

All 24 confirmed stops under the combined-relative configuration occur with
Nielsen damping. They include scalar arctangent and the larger Huber, Cauchy,
and arctangent models. Full-rank soft-L1 and rank-deficient soft-L1 pass all
measured quality checks.

On scalar arctangent, `f64` progress probes stop after 23 calls at parameter
error about `0.7581` and robust-gradient magnitude `0.5699`; default controls
continue to the exact minimum after 96 calls. The `f32` probes stop after 14
calls at parameter error about `0.3295` and gradient magnitude `0.3257`, while
defaults need 49 calls. Both factorizations reproduce this behavior.

Of the 24 confirmed combined-relative stops, 22 pass both model and step tests
on accepted, published trials with gain ratios above 0.25. Requiring
conjunction, acceptance, or that model-agreement threshold would retain those
stops. The two `f32` full-rank Cauchy stops pass only model reduction, so
conjunction would exclude those two particular decisions. These are
terminal-predicate comparisons; no new solver stopping composition is
implemented.

### Four gradient controls expose a recovery gap

The default gradient and normalized-gradient configurations pass all 36 `f64`
quality checks and 32 of 36 `f32` checks. Their four poor returns are Nielsen
runs on the rank-deficient Huber and arctangent models, with both
factorizations. They stop as **stalled**, with no convergence criterion.

  | Loss       | Physical calls | Distance to minimizer set | Gradient infinity norm | Objective gap     |
  | ---------- | -------------- | ------------------------- | ---------------------- | ----------------- |
  | Huber      | 16             | about `0.05794`           | about `0.6953`         | about `0.08058`   |
  | Arctangent | 16             | about `0.002168`          | about `0.02602`        | about `0.0001128` |

Their terminal rejected trials have zero measured actual reduction and damping
about `7.39e6` for Huber or `2.63e6` for arctangent. Native step components are
about `1.57e-8` or `1.65e-9`, respectively. The trial rounds to the published
point despite a substantial independently verified gradient. Trust-region
damping passes the same fixtures in both factorizations and precisions.

These stalls show that a stationarity guard alone cannot ensure recovery: it can
reject an inaccurate convergence claim but cannot enlarge a rounded-away step.
Four additional poor progress returns share poor default continuations; this
pilot does not classify them as confirmed premature convergence. All poor
outcomes remain in the report.

## QR identity screen correction

The first independent-model check failed on a full-rank Huber QR/trust-region
trial. Its native predicted decrease was `7.090717309744541e-7`, versus direct
model decrease `7.090717312665235e-7`. The discrepancy was about `2.92e-16`,
above the old local allowance of `6.45e-19`. Curvature clipping creates large
model residuals, and projecting that right-hand side through QR introduces an
error that the unsummed-gradient allowance alone misses.

The checker now adds `||h|| * ||J_model||_F * ||r_model||` to the QR projection
roundoff scale. It still checks the native prediction formula, independent
objective, gradient, diagonal, acceptance, and reported stopping operands. It
retains a count of 789 QR predictions that differ from direct model decrease
beyond local relative roundoff. This is an arithmetic identity screen, not a
backward-error certificate or evidence of accurate model solves. A regression
control reproduces the discrepancy and rejects a corrupted prediction. Solver
code, settings, and original solve CSVs are unchanged.

## Disposition and validation

[D006](../decisions.md#d006-extend-robust-controls-and-gate-nielsen-recovery)
extends the progress-test screening restriction and records the `f32`
rank-deficient recovery gate. No numerical default or stationarity-guard policy
is selected. The next task is to investigate and recover these four Nielsen
stalls before calibrating a guarded progress policy on the enlarged controls.
Finite differences, robust bounds and fixed coordinates, nonlinear regression
models, transformations, other backend measurements, precision certificates, and
complete inner work retain their gates. Step 5 stays open.

All 39 measurement tests pass with nalgebra 0.34 and 0.35, including model rank,
nullspace, analytic residual changes, observation neutrality, and budget
interruptions. All 17 independent checker tests and five stopping/reproduction
tests pass; mutation controls reject eleven general corruptions and nine
stopping/extended-model corruptions. Workspace all-target/all-feature clippy
with warnings denied, Rust formatting, and documentation formatting and lint
checks pass. The manifest records commands, source/output hashes, and the
verifier correction. Timing remains instrumented and supports no speed claim.
