# Run records

The [expanded coverage pilot](2026-10-02-coverage-002.md) records development
cases from the 57 experimental fixtures, both precisions, and 14 solver names.

The [Misra1a pilot](2026-10-02-misra1a-001.md) is the first focused numerical
run. This directory also establishes conventions for the later harness and
experiments.

Use a stable run ID such as `YYYY-MM-DD-family-NNN`. Keep the manifest in
`<run-id>.toml` and a concise interpretation in `<run-id>.md`. Commit small case
definitions and configuration files needed for reproduction alongside them.
Store bulk traces and exploratory output in
`target/convergence-defaults/<run-id>/` relative to the repository root. Do not
make ignored output the only record of a decision's evidence.

## Manifest requirements

Before execution, record:

- Run ID, purpose, planned status, and linked protocol and candidate IDs.
- Exact Basin revision, reference versions or commits, lockfile identity, and
  input/configuration paths and hashes. Commit experimental source and small
  inputs before collecting evidence used in a decision.
- Exact commands, working directory, enabled features, backend versions, scalar
  types, compiler, build profile, and relevant hardware/thread settings.
- Case families, dimensions, starts, bounds/constraints, transformations,
  derivative source, calibration/validation partition, seeds, and repetitions.
- Every stopping setting, algorithm variant, initialization choice, execution
  budget, external quality target, reference value, and scaling convention.
- Output paths, measurement schema version, and verification/accounting rules.

After execution, record completion status, failures or partial execution, actual
case coverage, output hashes, and validation commands and results. Retain failed
and incomplete runs. Never label a planned run as measured. Resolve the
executable schema while developing the harness in step 5.

## Result summary requirements

State what the run establishes and link its manifest. Report quality and target
success by family, evaluation counts by kind, work after attaining targets,
termination reasons, uncertainty where applicable, and important failures.
Explain algorithm or accounting differences from references.

Keep observations, hypotheses, and decisions distinct. Link any policy
conclusion to [decisions.md](../decisions.md), and state what should be tested
next. Summaries must remain useful if local raw output is deleted; manifests and
retained inputs must make the experiment reproducible.
