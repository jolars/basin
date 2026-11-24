<script setup lang="ts">
    import References from "#lib/docs/References.vue";
</script>

# Levenberg–Marquardt QR

This Levenberg–Marquardt variant solves the damped linear least-squares problem
with column-pivoted QR. It avoids forming the Gram matrix `JᵀJ`, while retaining
the Cholesky variant's damping choices and convergence rules.

[API
reference](https://docs.rs/basin/latest/basin/solver/levenberg_marquardt/struct.LevenbergMarquardtQr.html)

  | Property    | Requirement                                                  |
  | ----------- | ------------------------------------------------------------ |
  | Objective   | A smooth sum of squared residuals, or an adapted robust loss |
  | Evaluations | `Residual` and `Jacobian`                                    |
  | Search      | Local, from one starting point                               |
  | Constraints | Unconstrained                                                |
  | Parameters  | All four dense backends, with `f32` or `f64`; no sparse QR   |

## When to use it

Consider pivoted QR when forming normal equations loses too much accuracy,
particularly for poorly conditioned dense Jacobians. Compare the [Cholesky
variant](../levenberg-marquardt/) when sparse Jacobians are needed. QR changes
the linear solve; it does not make a nonlinear fit globally optimal or remove
sensitivity to parameter and residual scaling.

The method follows [Madsen, Nielsen, and Tingleff
(2004)](#ref-madsen-nielsen-tingleff-2004), §3.2, and MINPACK's
[`qrfac`](https://netlib.org/minpack/qrfac.f) and
[`qrsolv`](https://netlib.org/minpack/qrsolv.f) routines.

## How it works

The solver factors the Jacobian with column pivoting and solves the stacked
system `[J; sqrt(μD)] h ≈ [-r; 0]`. Diagonal regularization is applied through
the QR solve rather than by constructing `JᵀJ`. The factorization can be reused
while damping changes at the same iterate.

Scaled trust-region damping is the default, following the radius updates of
[Moré (1978)](#ref-more-1978) and MINPACK's `lmder` with Basin's bracketed
parameter search. Nielsen damping is also available. Both variants use the same
gain-ratio acceptance test and stopping settings as Cholesky LM.

## Example

The same residual problem can run with either LM factorization. This example
selects QR through `with_pivoted_qr()`; `LevenbergMarquardtQr::new()` is also
available.

<<< ../../../../../../crates/basin/examples/solver_guides/levenberg_marquardt_qr.rs

[View `levenberg_marquardt_qr.rs` on GitHub](https://github.com/jolars/basin/blob/main/crates/basin/examples/solver_guides/levenberg_marquardt_qr.rs)

## Practical considerations

### Treat numerical rank as a solve safeguard

Rank is checked after regularization and column equilibration. The default
relative threshold is `epsilon(F) * (m + n)` for `m` residuals and `n`
parameters; `with_relative_rank_tolerance` changes it. This check concerns the
regularized solve and does not establish that the original parameters are
identifiable. There is no truncated solution or normal-equation fallback.

Nielsen damping increases after rank loss; trust-region damping uses rank loss
to bound its parameter search. An unusable solve or exhausted retries can
produce `SolverFailed`. For bounded problems needing minimum-norm steps under
rank deficiency, compare [trust-region reflective](../trust-region-reflective/).

### Keep damping and stopping choices explicit

The default absolute gradient threshold is `1e-8`. The damping, optional
progress tests, and numerical no-progress safeguard behave as described in the
[LM guide](../levenberg-marquardt/#practical-considerations). A small damped
step does not establish stationarity. Put iteration and evaluation budgets on
the executor, and inspect both termination and fit quality.

## Implementation

Levenberg–Marquardt QR publishes matching parameters and cost through
`PointState`. It computes the gradient internally; the state does not expose it
through `GradientState`. Residuals, Jacobians, and model caches belong to the
solver. Fresh runs reset them, while exact solver-and-state checkpoints retain
them.

For ordinary residuals, the reported cost is `0.5 * sum(r[i]^2)`. The solver
requires `Residual` and `Jacobian`, without a `CostFunction` implementation.
Wrapping the problem in
[`RobustLeastSquares`](https://docs.rs/basin/latest/basin/core/least_squares/struct.RobustLeastSquares.html)
selects a robust loss and residual scale. The adapter supplies a safeguarded
model, while costs and convergence use the robust objective and its gradient.
Custom Jacobians additionally need `ScaleRowsInPlace` for robust losses.

Supported dense pairs are `Vec<F>`/`DenseMatrix<F>`,
`nalgebra::DVector<F>`/`DMatrix<F>`, `ndarray::Array1<F>`/`Array2<F>`, and
`faer::Col<F>`/`Mat<F>`, with `F = f32` or `f64`. The default `Vec` path needs
no BLAS/LAPACK provider.

Sparse Jacobians are unsupported: the sparse backends do not provide the
required numerical column pivoting. The dense paths use pure Rust. Use
`LevenbergMarquardtQr::default()` for `f32`.

<References :entries='["madsen-nielsen-tingleff-2004", "more-1978"]' />
