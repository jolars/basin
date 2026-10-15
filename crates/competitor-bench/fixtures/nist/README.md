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
