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

The 2.0 decision retains the existing LM and TRF defaults and keeps relative
checks opt-in. The comparison uses two fixed policies per solver, all 27 NIST
datasets and both starts, plus four affine variants isolating residual units,
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
at `1e-8`. It retains Nielsen damping and omits the scaled-radius check,
which requires trust-region damping. This isolates stopping changes without
claiming to reproduce SciPy LM. All other LM settings remain unchanged.

For both TRF implementations, `observed-relative` retains the native scaled
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
infinite bounds for TRF.
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
| Legacy TRF | Current | 50/54 | 44/54 | 6,738 | 5,248 |
| Legacy TRF | Observed relative | 37/54 | 17/54 | 2,914 | 2,683 |
| Full TRF | Current | 51/54 | 48/54 | 3,581 | 2,613 |
| Full TRF | Observed relative | 46/54 | 32/54 | 2,905 | 2,416 |

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
limit each. Legacy/full TRF reported 24/25 solver failures. This comparison
does not resolve all numerical termination behavior, establish global
optimality, or evaluate robust losses and finite-difference noise.

All four defaults accepted the small-residual affine starting point without
moving, demonstrating their unit dependence. LM's alternative recovered it;
TRF's retained absolute test still stopped immediately. In the parameter-unit
case, legacy TRF's observed-step policy stopped with about `0.43` relative
parameter error, compared with about `0.0023` under its default. Thus neither
policy makes scaling unnecessary. See the
[migration guide](../../../../MIGRATING.md#lm-and-trf-stopping-defaults)
for explicit scaling and configuration guidance.
