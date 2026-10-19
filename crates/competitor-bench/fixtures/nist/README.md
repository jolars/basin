# NIST nonlinear least-squares fixtures

These snapshots contain all 27 nonlinear-regression datasets from the
[NIST Statistical Reference Datasets collection](https://www.itl.nist.gov/div898/strd/nls/nls_main.shtml),
including both supplied starts, certified parameter values, residual sums of
squares, and all 2,176 observations. `manifest.json` records source URLs,
retrieval dates, original and normalized SHA-256 hashes, dimensions, and printed
reference values. The `.dat` snapshots retain the published decimal precision.

The benchmark crate exposes them through `competitor_bench::nist`:

```rust
use competitor_bench::nist::{Nist, datasets};

let dataset = datasets()?.into_iter().find(|d| d.id == "Misra1c").unwrap();
let start = dataset.start::<f64>(0);
let model = Nist::<f64>::new(dataset);
let rss = model.rss(&start)?;
```

`Nist<F>` implements Basin's cost, gradient, residual, and Jacobian traits for
the benchmark crate's nalgebra version, using native `f32` or `f64` arithmetic.
Slice-based methods expose responses, analytic derivatives, residuals, and RSS
for other adapters. Cost is half RSS; residuals are model minus response.
Nelson uses the natural logarithm of the response, and Roszman1 uses the
principal arctangent. Invalid model domains remain visible as nonfinite values.

`PrintedValue` preserves each decimal string and its last-place rounding
interval. Those intervals describe the published precision, rather than an
attainable solver tolerance. The models remain benchmark fixtures and are not
part of Basin's public problem corpus.

The Rust tests check dimensions, both starts, reference values, analytic
derivatives, adapter consistency, and model-specific conventions:

```sh
cargo test --locked -p competitor-bench --test nist
cargo test --locked -p competitor-bench --test nist --features basin-latest
```

## Basin 2.0 stopping-policy decision

The 2.0 decision retains the existing LM, bounded LM, and TRF defaults and keeps
relative checks opt-in. The comparison uses two fixed policies per solver,
all 27 NIST datasets and both starts, plus four affine variants isolating residual units,
parameter units, and an irreducible residual floor. It does not search a
tolerance grid, change damping, or compare runtime.

### Sources and candidate selection

- [SciPy 1.16.2 `least_squares`](https://docs.scipy.org/doc/scipy-1.16.2/reference/generated/scipy.optimize.least_squares.html)
  uses `ftol = xtol = gtol = 1e-8`, with different formulas for LM and TRF.
  LM uses gradient orthogonality, native model reduction, and a scaled
  trust-radius test. TRF uses a bound-scaled gradient, a cost test gated by
  model agreement, and a mixed absolute/relative trial-step test.
- [SciPy 1.16.2 `leastsq`](https://docs.scipy.org/doc/scipy-1.16.2/reference/generated/scipy.optimize.leastsq.html)
  uses `ftol = xtol = 1.49012e-8` and `gtol = 0`. The
  [MINPACK source](https://netlib.org/minpack/lmder.f) defines the native
  model-reduction and scaled-radius conditions. This is the same family
  of tests, rather than a reason to sweep nearby tolerance values.
- [Moré (1978), section 6](https://www.osti.gov/servlets/purl/7256021)
  used `FTOL = XTOL = 1e-8` in the reported experiments. Its scale-invariance
  discussion depends on the algorithm's diagonal scaling and native tests.
- [Madsen, Nielsen, and Tingleff (2004), pp. 26–28](https://www2.imm.dtu.dk/pubdb/edoc/imm3215.pdf)
  use absolute gradient and mixed absolute/relative step tests, with both
  tolerances `1e-8` in Example 3.7. The thresholds are user choices, not
  universal recommendations; increasing damping can force a small step.
- [Ceres 2.2.0](https://github.com/ceres-solver/ceres-solver/blob/2.2.0/docs/source/nnls_solving.rst)
  defaults to function tolerance `1e-6`, projected-gradient tolerance
  `1e-10`, and parameter tolerance `1e-8`. Its formulas differ from MINPACK
  and Basin's observed tests. It supplies survey context, not another run.

For both LM factorizations, `relative-model` disables the absolute gradient
check and enables gradient orthogonality and native relative model reduction
at `1e-8`. Both LM policies explicitly retain Nielsen damping, the default
when this stopping comparison was recorded, and omit the scaled-radius check,
which requires trust-region damping. This isolates stopping changes without
claiming to reproduce SciPy LM. All other LM settings remain unchanged.

For bounded LM and TRF, `observed-relative` retains the native scaled
gradient tolerance `1e-8` and adds Basin's observed relative cost and step
checks at `1e-8`. This is one curated Basin policy. It is not SciPy TRF:
Basin's cost check lacks its model-agreement gate, and its step formula
and observation stage differ. No new native TRF criteria were introduced.

### Reproduction and measurements

Run from the repository root:

```sh
mkdir -p target/nlls-defaults-decision
cargo run --locked --release -p competitor-bench --bin verify_nlls_defaults \
  > target/nlls-defaults-decision/results.csv
```

The recorded run used Basin library revision `315e4f7` with the retained probe,
Rust 1.89.0, nalgebra 0.34, and `f64`, with analytic Jacobians and explicit
infinite bounds for bounded LM and TRF.
Each solve has limits of 2,000 iterations and 20,000 total evaluation-category
charges. Evaluation limits are soft. The CSV retains all 464 rows, including
failures, termination details, authoritative evaluation counts, and independently
recomputed RSS, gradient, and orthogonality. Diagnostic evaluations happen
outside the solver and its counts. State and recomputed RSS agreed in this run.

Recovery below requires both relative RSS error and maximum componentwise
relative parameter error to be at most the stated threshold against the
printed NIST references. It does not require a convergence termination code.
Zero-reference RSS in the analytic cases is assessed using absolute RSS and
parameter error instead. Counts are residual/Jacobian evaluation categories,
not elapsed time or a claim of equal work across algorithms.

| Solver | Policy | Recovery at `1e-4` | Recovery at `1e-6` | Residual evaluations | Jacobian evaluations |
| --- | --- | --- | --- | --- | --- |
| LM | Current | 50/54 | 44/54 | 5,078 | 4,536 |
| LM | Relative model | 47/54 | 27/54 | 4,036 | 3,804 |
| QR LM | Current | 50/54 | 44/54 | 5,067 | 4,549 |
| QR LM | Relative model | 47/54 | 27/54 | 4,033 | 3,808 |
| Bounded LM | Current | 50/54 | 44/54 | 6,738 | 5,248 |
| Bounded LM | Observed relative | 37/54 | 17/54 | 2,914 | 2,683 |
| TRF | Current | 51/54 | 48/54 | 3,581 | 2,613 |
| TRF | Observed relative | 46/54 | 32/54 | 2,905 | 2,416 |

The LM relative policy increased convergence reports from 21 to 52 (QR: 24 to
51), while losing both ENSO starts and the first MGH17 start at `1e-4`.
For MGH17's first start, it stopped with relative RSS error about `0.46` and
maximum parameter relative error about `82`; the current defaults recovered
the reference solution. This concrete regression outweighs the evaluation
savings for a general default. Some tighter recoveries improved, including
Lanczos2, so the current absolute test is not uniformly better either.

Many accurate default runs returned `NumericalNoProgress` or `SolverFailed`;
the recovery totals above do not relabel them as convergence. For current
LM/QR, the counts were respectively 32/29 numerical stalls and one iteration
limit each. Bounded LM and TRF reported 24 and 25 solver failures, respectively.
This comparison does not resolve all numerical termination behavior, establish global
optimality, or evaluate robust losses and finite-difference noise.

All four defaults accepted the small-residual affine starting point without
moving, demonstrating their unit dependence. LM's alternative recovered it;
the retained absolute tests for bounded LM and TRF still stopped immediately.
In the parameter-unit case, bounded LM's observed-step policy stopped with
about `0.43` relative parameter error, compared with about `0.0023` under its default. Thus neither
policy makes scaling unnecessary. See the
[migration guide](https://basin.rs/docs/migrating-to-v2/#lm-and-trf-stopping-defaults)
for explicit scaling and configuration guidance.

## Comparing LM damping and factorization

`verify_lm_defaults` compares Nielsen and trust-region damping with both
Cholesky and column-pivoted QR. It reuses these fixtures through thin adapters
for Vec, nalgebra, ndarray, and faer. All 27 datasets, both starts, both scalar
types, and all four configurations produce 1,728 accuracy rows. Each backend
uses the same native fixture arithmetic; callback conversion and allocation
are included in solve timings.

Only damping and factorization change, and each configuration selects its
damping policy explicitly. Other constructor settings remain fixed,
including the absolute gradient tolerance `1e-8`, disabled optional convergence
checks, and numerical no-progress handling. Every solve has limits of 2,000
iterations and 20,000 evaluation-category charges. Limits are soft, and an
evaluation category is not an elapsed-work estimate. There is no tolerance or
initial-radius sweep.

The CSV records the returned parameters, termination, authoritative counts,
native and state RSS, and independent diagnostics. Independent diagnostics
reevaluate the returned native parameters in `f64` against the original
fixture, so `f32` rounding remains visible. Joint recovery requires both
relative RSS error and maximum componentwise relative parameter error below
`1e-3`, `1e-4`, or `1e-6`. Interpret the `f32` results at `1e-3` separately from
the primary `f64` results at `1e-4` and the tighter `1e-6` check. Relative RSS
error can magnify roundoff when the certified RSS is almost zero, particularly
on Lanczos1. Recovery does not relabel a numerical stall as convergence.

Run from the repository root:

```sh
mkdir -p target/lm-defaults-decision
cargo build --locked --release -p competitor-bench --bin verify_lm_defaults
target/release/verify_lm_defaults > target/lm-defaults-decision/results.csv
target/release/verify_lm_defaults --trace > target/lm-defaults-decision/traces.csv
target/release/verify_lm_defaults --reference > target/lm-defaults-decision/reference.csv
target/release/verify_lm_defaults --timing > target/lm-defaults-decision/timings.csv
python3 crates/competitor-bench/scripts/summarize_lm_defaults.py \
  target/lm-defaults-decision/results.csv \
  --timings target/lm-defaults-decision/timings.csv \
  --output target/lm-defaults-decision/summary.json
```

Use `--backend`, `--scalar`, `--dataset`, and `--start` to narrow a run.
`--scaling` reuses the existing Variably Dimensioned formulas at parameter
counts 10, 30, 100, and 300, in `f64`. For these zero-RSS cases, the RSS error
column reports absolute RSS and the parameter error compares against ones.

Timing runs disable tracing and exclude fixture preparation, diagnostics,
and CSV output. They include a fresh solver, starting-point allocation,
initialization, the executor, all callbacks, and teardown. Each case warms up
all configurations and records 32 paired rounds, with batches calibrated to
at least 2 ms where the repetition cap permits. A balanced four-round order
rotates configuration positions and predecessors. `--samples` and
`--sample-ms` control this protocol. Run timing builds without all-feature
unification, concurrent builds, or profiling, and record CPU affinity and
thread settings.

The summarizer retains failures in accuracy totals and performance profiles.
It reports paired runtime ratios on jointly accurate cases separately from
ratios across all runs. Its 95% intervals use 1,000 deterministic bootstrap
samples: paired rounds for individual cases, and datasets with their starts
kept together for aggregate ratios. The reference uses
`levenberg-marquardt` 0.15 on nalgebra 0.34, its own constructor stopping
settings, and a residual-call cap rounded down from 20,000. Its work limits
and stopping policy differ from Basin's; it supplies context rather than a
fifth matched configuration.

The `lm_defaults` Criterion benchmark uses the same adapter and solver runner
for longer measurements of selected cases:

```sh
cargo bench --locked -p competitor-bench --bench lm_defaults -- \
  'lm_defaults/vec/f64/(Bennett5|MGH10)/start1'
```

Tracing runs summarize each native subproblem once; trial rows repeat those
counts and must not be added again. Accepted and rejected trials are counted
independently. Focused adapter and accounting checks run with
`cargo test --locked -p competitor-bench --test lm_defaults --test nist`.

### Recorded damping comparison

The 2026-10-16 run used Basin library revision `ae04e00`, Rust 1.89.0,
nalgebra 0.34.2, ndarray 0.17.2, and faer 0.24.4 on an AMD Ryzen 9 7900.
The table gives ranges across the four backends. Each recovery denominator
is 54 starts. Runtime ratios compare jointly recovered starts with Nielsen
and Cholesky, using the geometric mean of paired median ratios.

| Configuration | `f64` recovery at `1e-4` | `f64` recovery at `1e-6` | `f32` recovery at `1e-3` | Relative `f64` runtime |
| --- | --- | --- | --- | --- |
| Nielsen, Cholesky | 50/54 | 44/54 | 40–42/54 | 1.00 |
| Nielsen, QR | 50/54 | 44/54 | 39–41/54 | 1.76–3.37 |
| Trust region, Cholesky | 51–52/54 | 48/54 | 39–41/54 | 1.02–1.16 |
| Trust region, QR | 51/54 | 48/54 | 41–42/54 | 1.84–3.19 |

Basin 2.0 selects trust-region damping with Cholesky as one default for both
precisions, favoring the improved `f64` recovery while accepting mixed runtime
results and some `f32` recovery losses. Nielsen remains an explicit option.
Trust-region damping recovered MGH10's first start after Nielsen exhausted
the iteration budget, and improved both Lanczos2 starts and the second
Lanczos3 start at `1e-6`. It also lost existing `f32` recoveries, including
three starts on faer. On Vec, trust-region damping used 26% fewer Jacobian
evaluations but 2.58 times as many model solves. Its runtime varied strongly
by problem and start, and its aggregate runtime intervals included both a
speedup and a slowdown.

QR improved the ill-conditioned matched-step comparison against an independent
SVD reference, supporting its use when normal-equation roundoff matters.
Its runtime cost and lack of sparse backend coverage argue against replacing
the general linear-solve default. At 300 parameters, trust-region damping
with Cholesky was about 36% faster than Nielsen on Vec, nalgebra, and ndarray,
but 32% slower on faer because of additional model solves.

Timings pinned one CPU and fixed thread counts to one, but unrelated builds
were active on other cores. Longer Criterion checks confirmed the broad
patterns while showing variation in some absolute timings; all eight selected
preflight runs matched the main probe's parameters and work counts exactly.
Treat small timing differences as inconclusive. The complete local report,
raw samples, plots, and reproduction script are retained in
`target/lm-defaults-decision/`. The policies in the probe remain explicit,
so changing Basin's default does not change this comparison.
