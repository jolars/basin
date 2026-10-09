# Analytic measurement harness

The first step 5 implementation lives in
[`competitor_bench::convergence`](../../crates/competitor-bench/src/convergence.rs).
It validates measurement semantics on development fixtures. It does not
implement the entire case register or authorize calibration.

## Components

- `ledger.rs` records physical leaf calls, kind, scope, coordinates, sampled
  objective, and completion or callback error. Cloned ledgers share an atomic
  cap reservation. Budget denials occur before user code and consume no physical
  call; failed evaluations consume one. Cache hits consume none. Finite
  differences wrap an instrumented leaf problem, not another charged adapter.
- `quality.rs` checks conservative objective gaps, normalized gradient or box
  stationarity, bound feasibility, and distance to an explicit solution set.
  Scalar root tests require independent position and residual accuracy, with
  bracket validity and the exact-root exception. Unverified or unattainable
  targets remain distinct from success. General constrained KKT verification is
  not implemented.
- `runner.rs` drives the public `Stepper`, records initialization and successful
  publication boundaries with authoritative `EvalCounts`, and retains native
  termination reports. Typed initialization errors expose no logical counters or
  returned point. Typed step errors preserve readable counters and the previous
  published recommendation; the consumed state is not a final checkpoint.
- `fixtures.rs` supplies a native `f32`/`f64` quadratic with exact dyadic
  optimum `(1,-2)`. Its independently evaluated objective excludes additive
  offsets. These benchmark fixtures do not add public corpus problems.

`EvalCounts` counts logical requests, including denied requests and submitted
batch work. The physical ledger counts actual leaf evaluations. Their difference
must be explained by fusion, adapter expansion, batching, and denials rather
than forced to zero. A fused cost/gradient evaluation produces two logical
categories for one physical call. A derivative adapter may produce many cost
calls while recording only one logical gradient request.

## Validation command and output

```sh
CARGO_TARGET_DIR=target/convergence-defaults/build \
  cargo test -p competitor-bench --test convergence_measurement
CARGO_TARGET_DIR=target/convergence-defaults/build \
  cargo run -p competitor-bench --release --bin verify_convergence -- \
  --output-dir target/convergence-defaults/<new-run-id>
```

The separate build directory prevents compiler/CMake cache conflicts with the
editor's background checks. The binary refuses to overwrite an earlier output
directory. Commit source and a planned run manifest before retaining evidence.

CSV schema 1 emits:

- `summary.csv`: physical cap/work, denials, authoritative logical counts when
  available, owned native outcome, returned-point and last-published target
  status, first published target attainment, verification evaluations, elapsed
  nanoseconds, and the last published coordinates.
- `<case>-recommendations.csv`: published coordinates and solver cost,
  independently checked quality, stage, iteration, work, and all six logical
  categories. Boundary publication does not by itself prove step acceptance.
- `<case>-leaves.csv`: every physical leaf point, callback kind/scope, sampled
  objective and independent target status where present, and callback outcome.
  Sampled trial success never enters recommendation first-attainment scores.

Scalar numbers use round-trip decimal output. Compound cells are CSV-escaped
Rust debug text, not JSON or a stable serialization of Basin's public types. A
later full harness must version any structured replacement. Verification is
performed after the solve timer stops and charges no solve work. Elapsed time
includes instrumentation and is not evidence for a speedup.

The command uses the development quadratic, start `(4,3)`, bounds `[-8,8]`,
fixed coordinate units 1, objective scale 17, and designated targets `1e-6`
(`f64`) and `1e-3` (`f32`). The exact native witness and zero uncertainty are
checked before solves. Thresholds on the two paired-simplex fixtures are
measurement fixtures, not calibrated recommendations. It also exercises analytic
default L-BFGS-B, forward/central finite differences, default budget-driven
Nelder-Mead, and exact initialization/step budget interruptions.

For the forward-difference quadratic, pass `--forward-tolerance 1e-7` after
`--output-dir <new-directory>` to configure the bounded L-BFGS gradient test
explicitly. This replaces the forward case with
`lbfgsb-f64-forward-configured`; all other cases keep their settings. Validate
that output with
`python dev/convergence-defaults/check-analytic.py <new-directory> --forward-configured`.
Without the option, the strict default remains the control. The configured
threshold addresses this fixture's derivative bias, not a general default.

## Remaining gates

This covers an initial subset of G402/G405, not their completion. Unit fixtures
exercise rejected trials and nested budgets, but do not establish correct native
LM/TRF trial diagnostics or every composed adapter's count roll-up. Passing
predicates in native termination reports are retained; failing clauses, trial
acceptance decisions, and unavailable model operands are not reconstructed.

The runner checks a wall limit between steps. It cannot interrupt a blocked
callback or initialization; a process-level timeout is still needed for broad
sweeps. Callback panics are outside its typed-error contract.

Executable NIST adapters, residual conventions, general feasibility/KKT checks,
independent precision certificates for nonanalytic models, all claimed backend
versions, and the remaining variant audit still block affected pilots or sweeps.
There is no holdout candidate evaluation, reference implementation comparison,
default selection, or protocol amendment in this initial implementation.
