# NIST executable model validation

The benchmark-only [`nist`
module](../../crates/competitor-bench/src/convergence/nist.rs) implements all 27
frozen StRD snapshots, all 2176 observations, and both published starts. It
retains each family's development or validation partition. Reading or validating
a held-out formula does not evaluate a candidate stopping policy on that family.

## Model and objective conventions

The adapter uses hand-derived analytic response derivatives. Residuals are
predicted response minus observed response, and cost is half RSS; the gradient
is `Jᵀr`. Nelson fits the natural logarithm of the response. Roszman1 uses the
principal `atan(b3 / (x - b4))` branch and the native rounding of NIST's printed
pi, rather than `atan2`. The logistic models use stable softplus arithmetic.
Invalid model domains produce visible non-finite values; the adapter adds no
bounds, clamps, or numerical stopping safeguards.

Data, parameters, and predictors are rounded to the selected native scalar
before evaluation. The models then compute in `f32` or `f64`, including Nelson's
log transformation. The Basin trait adapters use the benchmark-selected nalgebra
version: 0.34 normally and 0.35 with `basin-latest`. Raw slice methods also
expose residuals, Jacobian rows, and RSS. Other dense backends and backend
versions remain a separate pilot gate.

Printed parameter and RSS decimals retain their midpoint and half a unit in the
last published decimal place. Parameter standard deviations do not enter these
rounding widths. These widths describe printed values; they do not certify an
optimum, reference uncertainty, or attainable solver accuracy.

## Independent checks

The Rust integration tests check dimensions, both starts, all observation rows,
analytic derivatives in both precisions, objective and trait consistency,
response transformation, arctangent branch, invalid dimensions, singular models,
and stable logistic tails. Multiple finite-difference stencil sizes separate
cancellation from truncation error; an absolute screen covers negligible
Gaussian tails.

`verify_nist` emits schema 1 CSV rows for both starts and the printed reference
point in each precision. Each row contains family and partition, response,
residual, RSS, and every analytic derivative. It runs no solver and refuses to
overwrite its output file.

[`check-nist.py`](check-nist.py) independently evaluates the published formulas
with Python's standard-library Decimal at 100 digits. Its own trigonometric and
principal-arctangent series avoid a new dependency. It checks snapshot hashes
and manifest extraction, all emitted rows and derivatives, native input
rounding, residual conventions, and RSS accumulation. It differentiates the
independent high-precision formulas with tiny central perturbations, separately
from the native Rust stencil tests. Error allowances include native arithmetic,
conditioning, and absolute phase error at trigonometric zeros.

The checker also reevaluates RSS at the exact printed parameter decimals and
compares it with published RSS using a local sensitivity screen for parameter
rounding and the printed RSS width. This screen is not a rigorous interval
certificate. In particular, a tiny published RSS need not equal the RSS at
rounded printed parameters. Keep both values in the report rather than treating
the printed parameters as an exact numerical optimum.

## Reproduction

From the repository root, with a new output filename:

```sh
CARGO_TARGET_DIR=target/convergence-defaults/build \
  cargo test -p competitor-bench --test nist_models
CARGO_TARGET_DIR=target/convergence-defaults/build \
  cargo test -p competitor-bench --test nist_models \
  --features parallel,basin-latest
CARGO_TARGET_DIR=target/convergence-defaults/build \
  cargo run -p competitor-bench --bin verify_nist -- \
  --output target/convergence-defaults/nist-models.csv
python dev/convergence-defaults/check-nist.py \
  target/convergence-defaults/nist-models.csv \
  > target/convergence-defaults/nist-models-check.json
```

This is model validation, not retained calibration evidence or a frozen solver
run. Freeze source, lockfile, inputs, policies, and a planned run manifest
before collecting the next solver pilot.

## Remaining gates

The executable-model portion of G404 is covered. Independent reference
refinement where needed, rigorous precision certificates, native witnesses,
target eligibility, derivative-bias eligibility for finite differences, and
identifiability checks remain open. G403 and the full G404 gate remain open.
These checks select no defaults and authorize no candidate sweep.

Next, establish reference and precision eligibility for the development NIST
models, then connect them to the measured LM/TRF pilot. Cross both LM
factorizations with both damping modes, retain legacy and full TRF as separate
strata, and validate rejected-trial diagnostics and evaluation accounting before
interpreting stopping outcomes.
