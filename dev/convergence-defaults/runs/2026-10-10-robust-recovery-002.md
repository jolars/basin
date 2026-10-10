# Robust Nielsen curvature recovery

The [manifest](2026-10-10-robust-recovery-002.toml) and [retained
report](../robust-recovery-results.json) record a successful accuracy recovery
after [the rejected first correction](2026-10-10-robust-recovery-001.md). Source
`d009a09` and planned manifest `f121c10` preceded 720 extended runs and 336
original ablations. Every gradient-control configuration now passes all 72
extended quality checks, including the four formerly inaccurate rank-deficient
`f32` returns. Those four runs still report **stalled** after reaching accurate
points; the correction does not relabel their termination as convergence. Step 5
remains open.

## Cause and correction

Clipped robust curvature initially gives diagonal entries near `6*epsilon`.
Rejection escalation accumulates a large dimensionless Nielsen damping parameter
while its product with this diagonal remains usable. An accepted move then
restores positive curvature and raises the diagonal to about `6`. Keeping the
accumulated parameter multiplies effective damping by about eight million in
`f32`, reducing the next step until it rounds away at an inaccurate point.

Before updating the monotone scaling diagonal, Basin now applies

```text
D_new[j] = max(D_old[j], diag(J_model^T J_model)[j])
alpha = max_j(D_old[j] / D_new[j])
mu_new = mu_old * alpha
```

This compensates for growth shared by every coordinate. Under uniform growth,
effective damping stays the same. Under unequal growth, each coordinate retains
at least its previous effective damping, apart from arithmetic roundoff. If any
entry stays unchanged, the parameter stays unchanged. Two square-root factors
retain products whose diagonal ratio would underflow first; an underflowed
product receives a positive scalar floor. Invalid scales and damping retain the
existing failure path.

The largest ratio matters. The rejected smallest-ratio correction recovered the
four original failures but removed damping from still-clipped coordinates on
full-rank arctangent. Some parameters escaped into the loss's flat tails while
the total objective improved. Its four new poor default returns failed the
unchanged quality gate. Permanent mixed-curvature regressions now protect
against that escape.

The safeguard extends the [Nielsen damping
update](https://www.imm.dtu.dk/documents/ftp/tr99/tr05_99.abstract.html) for
Basin's varying robust diagonal. Normal equations and pivoted QR share it.
Gain-ratio acceptance, rejection escalation, retained caches, native no-progress
checks, and solver convergence comparisons continue to govern the run. The
adjustment adds no problem callbacks or allocations. Ordinary least squares and
trust-region damping retain their existing paths; no convergence threshold or
composition is calibrated.

## Returned quality and work

Both factorizations show the following recovery under the default
absolute-gradient control. Errors and gradient magnitudes come from the
independent 100-digit verifier. Ranges cover the normal and QR variants.

  | Loss       | Physical calls before → after | Distance to minimizer set before → after | Gradient infinity norm before → after | Objective gap after |
  | ---------- | ----------------------------- | ---------------------------------------- | ------------------------------------- | ------------------- |
  | Huber      | 16 → 29                       | about `0.05794` → `1.73–1.76e-6`         | about `0.6953` → `2.07–2.11e-5`       | below `7.5e-11`     |
  | Arctangent | 16 → 26                       | about `0.002168` → `1.46e-5`             | about `0.02602` → `1.75e-4`           | about `5.1e-9`      |

The first changed Huber trial uses damping about `0.8815` instead of `7.39e6`,
giving a step component about `0.04022` instead of `1.57e-8`. Arctangent uses
about `0.3133` instead of `2.63e6`, giving about `0.001875` instead of
`1.65e-9`. The checker verifies that the first changed trial follows an accepted
curvature transition and that its damping matches the diagonal compensation.

The unchanged quality limits are `1e-3` for `f32` and `1e-6` for `f64`, applied
to both independent parameter and gradient errors on every stopping
configuration. The recovered objective gaps are below the spacing of the native
nonzero objectives. Further objective reductions round away, and the native
safeguard eventually reports no progress. This supplies accurate returned
points, not exact stationarity or a complete CDP-1 precision certificate.

  | Configuration                  | Converged | Stalled | Quality passed | Confirmed premature | Physical calls |
  | ------------------------------ | --------- | ------- | -------------- | ------------------- | -------------- |
  | Default absolute gradient      | 62        | 10      | 72             | 0                   | 2972           |
  | All relative probes            | 72        | 0       | 54             | 18                  | 2196           |
  | Model reduction                | 72        | 0       | 54             | 18                  | 2231           |
  | Trial step                     | 72        | 0       | 62             | 10                  | 2327           |
  | Model reduction OR trial step  | 72        | 0       | 54             | 18                  | 2199           |
  | Robust normalized gradient     | 69        | 3       | 72             | 0                   | 2977           |
  | Trust radius or exact gradient | 56        | 16      | 72             | 0                   | 3072           |

The 504 full-budget runs retain 475 convergences and 29 stalls, using 17,974
physical callbacks. The 216 interruptions use 216 callbacks, retain 72
initialization errors and 144 callback errors, and return no point. Native
observations retain 6897 accepted and published trials, 3919 rejected trials,
144 denied trial callbacks, and no non-finite objective trials. Verification
stays outside the solve ledger and timer. Timing is instrumented and supplies no
speed comparison.

Progress tests still have 18 confirmed premature combined-relative returns. Ten
satisfy both model and step tests on accepted trials with gain ratio above 0.25.
The correction therefore closes the demonstrated accuracy-recovery gap while
preserving the stationarity-guard gate. It does not justify selecting
progress-only defaults.

## Reproduction and validation

All five CSV tables match the previous run exactly, excluding elapsed time, for
360 trust-region cases and 108 Nielsen interruptions. The original 336-ablation
rerun also preserves all 168 trust-region traces. Its three gradient-control
configurations pass all 48 quality checks each, while model and
combined-relative settings retain six confirmed premature returns on the
non-finite-domain fixture. Changed Nielsen work and all native stalled outcomes
remain in the summaries.

Thirty-six permanent backend tests cover 320 solves across every supported dense
backend release and supported sparse path in both precisions. They include
rank-deficient Huber/arctangent recovery and full-rank mixed-curvature
arctangent controls, both damping modes, and both dense factorizations. Unit
tests cover uniform and unequal growth, inactive growth, invalid inputs, extreme
scales, and representable products. The regression failed on the original
inaccurate return before the fix.

All 43 measurement tests pass on nalgebra 0.34 and 0.35. Seventeen independent
checker tests and six stopping/recovery tests pass on both output sets,
including CSV corruption controls and an asymmetric-ratio negative control. The
full pure-Rust solver suite, all-target/all-feature workspace clippy with
warnings denied, rustdoc, both default and no-default WASM builds, Rust
formatting, and Markdown checks pass.

[D007](../decisions.md#d007-preserve-effective-damping-through-robust-curvature-growth)
records the safeguard and the rejected alternative. These remain linear
development controls with analytic derivatives and unit scales. Nonlinear
models, derivative and constraint modes, backend measurements, precision
certificates, and complete inner-work accounting remain prerequisites for
affected calibration sweeps. Holdout candidate outcomes remain sealed.
