# Legacy TRF finite-model safeguard recheck

The [manifest](2026-10-09-trf-finite-model-001.toml) and [retained
comparison](../trf-finite-model-results.json) recheck the frozen 352-case
[robust pilot](2026-10-09-robust-ls-001.md) after correctness fix `24cb364`.
Planned manifest commit `cbb646a` preceded execution. Settings, starts, losses,
bounds, targets, budgets, and partitions are identical to the baseline. No
convergence default or candidate policy was selected.

## Fix and regression coverage

Legacy TRF previously checked non-finite damping only after a linear-solve
error. A backend can instead return a zero step from an infinite damped matrix,
leading to a non-finite prediction, another residual callback, and repeated
rejections until a budget expires.

The fix checks damping and its escalation factor before solving, and checks
feasible-step scaling, predicted reduction, and the Coleman–Li correction before
requesting the trial residual. An invalid model stops with numerical failure and
preserves the current point, cost, counts, and completed iteration number.
Cached residuals and the Jacobian remain attached to that point. Finite rejected
trials and non-finite trial objectives retain their existing behavior.

Regression checks exercise both initial infinite damping and damping escalation
following finite rejections on all four dense backends in both precisions, plus
supported sparse backends. A custom Jacobian deliberately returns an extreme
step to verify that a non-finite prediction stops before any trial callback. No
public trait requirement or signature changes.

## Before and after

The independent robust verifier passes all 352 recheck cases. The comparison
checker confirms that every new physical callback trace is a prefix of its
frozen baseline, and that all final published points and costs are identical.
Exactly four outcomes change, all default `f32` legacy TRF cases:

  | Fixture                  | Previous calls | Fixed calls | Previous outcome      | Fixed outcome     |
  | ------------------------ | -------------- | ----------- | --------------------- | ----------------- |
  | Huber outlier, scale 1   | 4000           | 20          | Callback budget error | Numerical failure |
  | Huber outlier, scale 0.5 | 4000           | 20          | Callback budget error | Numerical failure |
  | Symmetric soft-L1        | 4000           | 22          | Callback budget error | Numerical failure |
  | Non-finite trial domain  | 4000           | 44          | Callback budget error | Numerical failure |

Previously, callback errors exposed only last publications. The fixed runs
return ordinary solver-failure results containing those same points and costs.
All four points pass the existing unit-scale measurement quality checks; failure
remains distinct from convergence. Total physical calls fall from 18,069 to
2,175, eliminating the 15,894 non-finite model predictions. The other 348
outcomes retain their original classifications and work.

The four premature Cauchy relative-probe stops remain unchanged. Their robust
stopping composition still needs assessment before an affected candidate sweep.
This safeguard fix does not modify those tests or their tolerances.

## Verification and backport

The full pure-Rust solver suite passes on the convergence branch, the clean main
fix branch, and the adapted 1.x branch. Workspace all-target/all-feature clippy
with warnings denied passes on all three. Rustdoc and both default and
no-default WASM builds pass on the convergence branch and on 1.x. Rust
formatting passes everywhere.

The robust verifier and its 12 Python tests pass on the recheck. Eleven
corrupted-record controls apply to the new traces; the extra non-finite
prediction acceptance control still applies to the original traces, where all 12
controls pass. The comparison checker also validates the exact affected case
set, all callback prefixes, and final publications. The manifest records source
and output hashes. Its comparison command takes already verified traces and
refuses to overwrite the report.

The main fix is proposed in [PR #116](https://github.com/jolars/basin/pull/116),
commit `250f59e`. The adapted 1.x backport is proposed in [PR
#117](https://github.com/jolars/basin/pull/117), commit `a8dd82c`, based on
`origin/v1`. Both are draft PRs. The backport preserves the existing fieldless
`TerminationReason::SolverFailed`, state API, convergence defaults,
dependencies, and branch-owned release bookkeeping. It should follow main review
and merge.

Step 5 remains open for the other recorded gates. The next numerical-policy work
is robust step/model-reduction composition; this run establishes no new
tolerance recommendation or holdout result.
