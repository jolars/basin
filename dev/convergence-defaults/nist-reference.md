# Development NIST reference preflight

The [`reference-tools/preflight.py`](reference-tools/preflight.py) workflow
refines the 15 development datasets in six NIST families without running a Basin
solver. It uses both supplied starts to fix coordinate and objective scales, but
does not solve from those starts. Holdout candidate outcomes remain sealed. This
preflight supplies local reference certificates and analytic native-precision
witnesses for step 5, not calibrated stopping policies.

## Reference checks

All reference arithmetic uses 100 decimal digits and the frozen decimal input
snapshots. Newton refinement differentiates the complete half-RSS objective with
second-order automatic differentiation. The second refinement uses Gauss–Newton,
the independent formulas in `check-nist.py`, and central numerical response
derivatives. Both begin at NIST's printed parameters. Their scaled parameter
difference must be below `1e-35`; the numerical derivatives have `O(1e-44)`
truncation bias. Agreement is a cross-check, not the certificate.

Outward-rounded Decimal intervals enclose every mathematical operation. Basic
arithmetic uses directed rounding. For `exp`, `ln`, and `sqrt`, the code expands
correctly rounded endpoint results to their adjacent Decimal neighbors. Python
documents these functions' rounding behavior in its [Decimal
reference](https://docs.python.org/3/library/decimal.html).

The certificate applies the Krawczyk operator to the objective gradient on a box
with coordinate radii `1e-50 * s_i`. Strict interior inclusion and a scaled
contraction bound below one establish a unique stationary point inside the box.
Positive interval LDL pivots establish a positive definite Hessian throughout
the box, proving that the point is a strict local minimum. This follows the
verification approach described by
[Rump](https://www.tuhh.de/ti3/rump/Research_Rump/topics.shtml). The code
reevaluates the objective over the entire box and retains its interval, gradient
intervals, parameter box, Krawczyk image, contraction bound, and LDL pivots. It
also checks overlap with the printed RSS rounding interval.

These references are `validated-local`. They certify a locally identifiable
parameter branch, not a global minimum or either start's basin membership.
Lanczos amplitude/rate permutations and MGH17's two-component exchange remain
equivalent parameter branches. Printed decimal widths are retained separately
from the refined parameter enclosure; parameter standard deviations never enter
numerical uncertainty.

## Native witnesses and target eligibility

For each precision, prepare the rounded refined point and both adjacent
representable values of each coordinate, changing one coordinate at a time. Also
evaluate the two native starts. `verify_nist_witness` computes cost, gradient,
and absolute accumulation terms using the existing `Nist<f32>` or `Nist<f64>`
adapter. It refuses holdout datasets, nonrepresentable coordinates, and existing
output files. It runs no solver.

The checker independently encloses objective, stationarity, and parameter error
at every native witness using the original decimal data. It separately encloses
the model with native-rounded data. Comparing both with actual native outputs
separates data conversion from arithmetic error, including cancellation and
Nelson's native log-response conversion. These error bounds apply to the tested
points; they do not bound derivative error throughout a solver trajectory.

The rounding screen is at least eight unit roundoffs times the sum of absolute
objective or gradient accumulation terms. Cancellation is checked through
independent intervals and actual native output rather than certified by this
screen alone. Each eligible joint target needs a stored representable witness
passing all three quality tests and CDP-1's tenfold uncertainty margin.

Coordinate scales use the published reference and both starts. Objective scales
use modeled-response variation and the native start's independent reference gap,
as specified by CDP-1. The report retains scale intervals and freezes their
positive lower bounds as conservative normalizations. Each start and precision
has its own target record. Reference uncertainty, data conversion, native
arithmetic error, and the rounding screen remain separate report operands.

A target without a passing tested witness is `reference-pending`, not
`precision-ineligible`: this finite neighbor search does not prove that no
representable solution exists. The report lists the rounded witness's failing
clauses. Finite-difference eligibility requires a separate derivative-bias
preflight.

## Reproduction

The [planned run manifest](runs/2026-10-09-nist-reference-001.toml) fixes the
source, configuration, inputs, commands, and output locations. Run from the
repository root with new output paths:

```sh
PYTHONDONTWRITEBYTECODE=1 python \
  dev/convergence-defaults/reference-tools/test_reference.py
CARGO_PROFILE_DEV_DEBUG=0 CARGO_INCREMENTAL=0 \
  CARGO_TARGET_DIR=target/nist-reference-build \
  cargo build -p competitor-bench --bin verify_nist_witness
python dev/convergence-defaults/reference-tools/preflight.py prepare \
  target/convergence-defaults/<new-reference-run>
target/nist-reference-build/debug/verify_nist_witness \
  --points target/convergence-defaults/<new-reference-run>/points.csv \
  --output target/convergence-defaults/<new-reference-run>/native.csv
python dev/convergence-defaults/reference-tools/preflight.py check \
  target/convergence-defaults/<new-reference-run> \
  --output target/convergence-defaults/<new-reference-run>/summary.json
```

Preparation snapshots source hashes. Checking refuses changed sources, inputs,
missing or duplicated native rows, altered coordinates, or changed interval
certificates. It also screens native outputs against independent model
arithmetic. The compact report retains every target and its selected witness;
the full report retains all neighbor measurements. Reproducing an older run
requires its source revision.

## Remaining work

This addresses development-model reference refinement and analytic witness
eligibility only. Pending targets, finite differences, holdout references,
start-basin classification, conditioning metadata, transformed cases, and full
backend coverage retain their gates. The LM/TRF measurement pilot must still
validate accepted/rejected trial diagnostics and work accounting. No default,
numerical safeguard, calibration result, or protocol amendment follows from this
preflight alone.
