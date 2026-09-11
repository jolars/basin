# Pivoted QR for Levenberg–Marquardt

Implemented September 9, 2026, against parent revision `47d4c5203a9fd3dd625926f9dc79f33874344ae9`.
QR is an explicit option; Cholesky remains the default. The retained production
comparisons confirm better small-damping step accuracy, while the narrow-SVI
convergence gap remains a separate damping and stopping question.

```rust
let solver = LevenbergMarquardt::new()
    .with_tol_grad(1e-10)
    .with_pivoted_qr();
```

`LevenbergMarquardtQr<V, M, F = f64>` also has a scalar-generic constructor.
Both routes share the LM loop, including Nielsen damping, monotone Marquardt
scaling, gain ratio, stopping rules, and callback accounting. The new type
supports `Executor::from_start`, warm starts, and memetic composition. Existing
LM annotations and Cholesky-only capabilities retain their original bounds.

## Factorization and rank contract

`FactorizePivotedQr` produces a reusable `RegularizedQrSolve` factor for a
fixed right-hand side. Columns are equilibrated before numerical pivoting;
`R` is restored to the original matrix scaling. This prevents parameter units
from determining the Householder order. Tests with column magnitudes from
`1e-8` to `1e8` exposed avoidable right-hand-side cancellation without this
initial equilibration, including in the native QR implementations.

The factor retains the permutation, `R`, `Qᵀb`, and original column norms.
Each damping attempt uses Givens rotations to solve the equivalent triangular
augmented system, following MINPACK's
[`qrsolv`](https://netlib.org/minpack/qrsolv.f). Rejected steps and damping
retries reuse the factor. Accepted steps retain the trial residual and
invalidate the factor and gradient. Neither QR operation forms a Gram matrix.

Rank checks apply after regularization and unit-column equilibration of the
augmented system. A triangular pivot at or below `epsilon(F) * (m+n)` returns
`RankDeficient`; `.with_rank_tolerance(tol)` overrides this threshold with a
finite value in `[0,1)`. Zero requests exact-zero detection. The check is a
numerical safeguard, not a singular-value estimate of the original Jacobian.
Deficient Jacobians may still produce valid damped steps. Deficient augmented
systems trigger damping retries, then `SolverFailed` if retries are exhausted.
Zero-damping deficient systems report rank loss; there is no truncated or
minimum-norm solution and no normal-equation fallback. Non-finite factorization
or solve arithmetic fails immediately, and typed callback errors propagate.

Dense nalgebra and faer use native Householder QR. `DenseMatrix` and ndarray
share the pure-Rust kernel. All four dense backends support `f32` and `f64`,
including rectangular and underdetermined systems. A zero faer Jacobian uses
the exact identity transform because its native reflector coefficients are
undefined for that case. Sparse implementations are absent: nalgebra-sparse
has no QR, and faer's sparse QR uses fill-reducing ordering instead of numerical
column pivoting. Existing sparse Cholesky LM remains available.

## Reproduce

No GlobalSearch checkout is needed:

```sh
cargo run -p competitor-bench --release --bin verify_lm_qr > lm-qr-results.csv
cargo run -p competitor-bench --release --bin verify_lm_qr -- --steps > lm-qr-steps.csv
cargo bench -p competitor-bench --bench compare -- \
  'exp_fit|qr_regularization' --sample-size 10 --warm-up-time 1 --measurement-time 2
```

The comparison uses Basin's nalgebra 0.34 backend and `levenberg-marquardt`
0.15 with the same nalgebra patch (0.34.2). `basin-latest` selects nalgebra
0.35 for the Basin side during all-feature maintenance builds. Model formulas,
Jacobian checks, and all starts are shared with the historical probe.
Observations are deterministic synthetic data, not market data. Analytic
Jacobians are checked against centered finite differences at the truth and
every start, including the actual SSVI phi formula discussed in the
[investigation](README.md#ssvi-jacobian-mismatch-in-the-downstream-source).

Both production routes disable absolute gradient stopping and set all three
relative tolerances to `1e-12`. Each solve permits `200*(n+1)` residual calls,
including initialization. The callback asserts the budget, and Basin's
executor counts are checked against actual callbacks. Residuals used for final
reporting and derivative validation are outside the solve's counters.

## Results

[Linear-step results](lm-qr-steps.csv) compare each step with an independent
SVD of the augmented system. For column separation `1e-8`:

| Damping | Cholesky step error | QR step error |
|---|---:|---:|
| `1e-10` | `1.19e-13` | `5.35e-16` |
| `1e-16` | `3.51e-2` | `2.01e-10` |
| `1e-20` | Factorization failed | `1.60e-9` |

Errors are Euclidean distances from the SVD reference, which also has finite
precision. An additional test uses a cancellation-free analytic solution for
a two-column system whose Gram matrix loses positive definiteness in `f64`.
Exactly deficient undamped systems report rank loss as specified.

[Calibration results](lm-qr-results.csv) record convergence, termination,
residual and Jacobian calls, final residual norm, parameters, and recovery
errors for every retained start. On narrow raw SVI from
`[0.035, 0.12, -0.2, 0.01, 0.25]`:

| Solver | Residual/Jacobian calls | Residual norm | Parameter error | Stop |
|---|---:|---:|---:|---|
| Cholesky | 1200 / 1199 | `1.97e-9` | `3.80e-2` | MaxIter |
| QR | 1200 / 1199 | `1.97e-9` | `3.80e-2` | MaxIter |
| MINPACK | 10 / 7 | `1.39e-17` | `1.38e-11` | Relative step |

All three solvers exhaust the budgets at the other two narrow-SVI starts.
Both Basin routes converge on broad SVI and both SSVI datasets. Closely spaced
SSVI maturities use 13–32 residual calls in Basin and seven in MINPACK. The
poorly scaled diagonal problem converges in six calls on both Basin routes.
The nearly collinear linear example still demonstrates premature stopping
with enabled relative tolerances; improved factorization does not fix it.

Recovery is nonunique for exactly deficient systems. Raw SVI also has a sigma
sign symmetry: the CSV reports both raw parameter error and an error after
replacing sigma by its absolute value. A tiny residual or successful stop is
not, by itself, evidence of unique parameter recovery.

These results support keeping QR explicit. It addresses normal-equation
roundoff, while replacing the default would not resolve the observed nonlinear
convergence difference and would remove sparse support. MINPACK's trust-radius
parameter selection and Basin's relative stopping tests remain follow-ups.

## Timing

[Retained timings](lm-qr-timing.csv) use Criterion's slope estimate and 95%
confidence interval, ten samples, one second of warmup, and two seconds of
measurement on an AMD Ryzen 9 7900, NixOS, and Rust 1.89.0. The recorded run
had no concurrent builds from this task. It was not CPU-pinned; these are
short local measurements, not a general performance guarantee.

| Exponential fit route | Time per solve |
|---|---:|
| MINPACK | 1.75 us |
| Basin / nalgebra / Cholesky | 1.82 us |
| Basin / nalgebra / QR | 5.10 us |
| Basin / faer / Cholesky | 4.62 us |
| Basin / faer / QR | 9.08 us |

These are full solves under the existing benchmark's tolerances; iteration
counts may differ. QR costs more on this small problem. The separate
`DenseMatrix` 128-by-8 benchmark retains guards for factorization and repeated
damping solves: QR factorization took 78.3 us, a QR damping
retry 0.767 us, and a Cholesky retry from a cached Gram
0.234 us. Retrying QR avoids repeating the much more
expensive Jacobian factorization. The comparison harness retains both paths
for exponential fitting, Powell singular, variable dimension, and
underdetermined problems.

## Verification

The focused suite checks all supported backend versions, both scalar types,
permutations, rectangular systems, inconsistent data, exact and numerical rank
loss, zero columns, tiny damping, non-finite inputs, and analytic/SVD references.
LM tests cover factor reuse, accepted/rejected steps, initialization reuse,
damping exhaustion without trial callbacks, unchanged callback errors, legacy
Cholesky-only capabilities, and a QR-only downstream matrix.

The routine pure-Rust suite, all-feature workspace Clippy, and rustdoc with
warnings denied pass. Focused tests pass on all ten supported backend releases
(nalgebra 0.32–0.35, ndarray 0.15–0.17, and faer 0.22–0.24). The Rust 1.87
all-target MSRV check passes with CI's feature set. Both default and
no-default-feature WASM builds pass. Web formatting, type checking, and the
production build pass. Web lint reports existing errors in HTML, TypeScript,
and SVG files; all ten diagnostic files match the parent revision exactly.
