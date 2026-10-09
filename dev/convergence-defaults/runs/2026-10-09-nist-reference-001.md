# Development NIST reference preflight

Run `2026-10-09-nist-reference-001` completes the analytic development-reference
preflight described in [nist-reference.md](../nist-reference.md). The
[manifest](2026-10-09-nist-reference-001.toml) was committed before execution.
Reference source revision is `6f3ab0f14f08e69491fb43419b8d278b07ca6822`;
execution revision is `9a11eb3`, with a clean working tree when preparation
started. This is independent reference work, not a Basin solver run,
calibration, or a stopping-policy comparison.

## Results

All 15 development datasets in six families have `validated-local` reference
certificates. Full-Hessian Newton and independently implemented Gauss–Newton
agree within the fixed scaled `1e-35` bound. Every reference passes strict
Krawczyk inclusion, contraction below one, and positive interval LDL pivots of
the complete objective Hessian. Refined RSS intervals overlap the published RSS
rounding intervals. The certificates establish strict local minima and local
parameter identifiability; neither start's basin membership nor global
optimality is established.

The native probe evaluated 294 points: 60 start evaluations and 234 rounded
reference or coordinate-neighbor witnesses across `f64` and `f32`. Independent
interval arithmetic checked them against both original decimal data and
native-rounded data. Solve work is zero; all evaluations are verification work.

The [retained report](../nist-reference-eligibility.json) records 60
precision/start cases and all 330 target combinations. It retains complete
reference certificates, scales, target status, selected witness coordinates,
quality bounds, and separate uncertainty operands.

  | Precision | Eligible targets | Pending targets | Eligible designated targets |
  | --------- | ---------------- | --------------- | --------------------------- |
  | `f64`     | 180 of 180       | 0               | 30 of 30                    |
  | `f32`     | 141 of 150       | 9               | 29 of 30                    |

All pending targets are `reference-pending`. The tested points do not prove
impossibility or justify loosening the fixed allowances:

  | Dataset | Precision | Start | Pending `q` values             |
  | ------- | --------- | ----- | ------------------------------ |
  | MGH10   | `f32`     | 2     | `1e-5`, `1e-6`                 |
  | Nelson  | `f32`     | 1     | `1e-4`, `1e-5`, `1e-6`         |
  | Nelson  | `f32`     | 2     | `1e-3`, `1e-4`, `1e-5`, `1e-6` |

The rounded witness fails the stationarity uncertainty margin at every pending
target; its stationarity measure also fails at some tighter targets. The full
report retains every tested neighbor rather than only the rounded point. Nelson
start 2's selected witness has normalized stationarity uncertainty about
`0.003331`, so ten times that uncertainty exceeds `sqrt(1e-3)`, about
`0.031623`. The fixed start-dependent objective normalization explains why
Nelson start 1 can certify the designated target while start 2 remains pending.

Lanczos1 illustrates the reference-refinement requirement: RSS at the exact
printed parameters is about `3.98336e-21`, while the certified refined local
reference has RSS about `1.430786772078e-25`, consistent with the published RSS.
The refined box, not the printed parameter midpoint, supplies reference
uncertainty.

## Verification and reproducibility

The six analytic certificate tests pass, including exact-rational interval
enclosures, elementary-function enclosures, second-order chain rules,
minimum/maximum discrimination, tenfold uncertainty margins, native neighbors,
and Hessian checks across all development models. All 24 existing NIST and
measurement integration tests pass. Workspace all-target and all-feature clippy
passes with warnings denied, as do rustfmt and documentation checks.

The [retained-report checker](../check-nist-reference.py) verifies each stored
witness and exercises eight rejection controls: truncated references, changed
source hashes, missing or duplicate native points, altered coordinates,
independently inconsistent native costs, holdout probe requests, and output
overwrite attempts. Raw outputs and validation logs live under ignored
`target/convergence-defaults/2026-10-09-nist-reference-001/`; their hashes are
retained in the manifest.

Exploratory development runs preceded this freeze and selected no solver
policies. They exposed mixed interval/AD operand dispatch and an overly
conservative cancellation screen. The final screen applies to the actual
objective and gradient accumulation terms; interval comparisons with native
outputs independently bound cancellation. Thresholds, target grids, and family
partitions did not change. The exploratory work has no retained calibration
status. Disk exhaustion and concurrent removal of the earlier generated build
directory interrupted Rust verification; a separate build with debug information
and incremental compilation disabled completed the checks.

## Remaining gates

This supplies the analytic development subset of G403 and development reference
refinement in G404. Both gates remain open for their other strata. Finite
differences, pending native targets, holdout references, start-basin
classification, conditioning metadata, transformations, and full backend
coverage remain separate work. Witness arithmetic bounds apply at the tested
points, not throughout solver trajectories.

Next, connect these development references and frozen eligibility records to the
measured LM/TRF pilot, crossing both LM factorizations with both damping modes
and retaining legacy and full TRF separately. Validate rejected-trial
diagnostics and work accounting before interpreting stopping outcomes. Keep
uncertified targets visible and out of their target scores. No solver default or
numerical safeguard changed.
