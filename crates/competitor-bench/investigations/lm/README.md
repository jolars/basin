# Levenberg–Marquardt investigation

The initial conditioning and calibration investigation ran September 8, 2026,
against Basin `260c0ffe87bb95e1781a627f244bdf252e81242e` (1.9.0). The findings
below describe that historical implementation. The production follow-ups cover
[pivoted QR](lm-qr.md), [damping](lm-damping.md), and
[stopping](lm-stopping.md). Their maintained verification commands reproduce the
current solver comparisons.

Shared formulas and analytic Jacobian checks live in [verification
support](../../src/bin/support/lm_models.rs). The original combined COBYLA/LM
prototype and GlobalSearch reproduction scripts are archived at Basin commit
`ad4ef87ee879df65b7a9ce9b9c3baf60e6470089` under
`crates/competitor-bench/investigations/cobyla-lm/`. The completed COBYLA
investigation has a separate [closure record](../cobyla.md).

## LM: accuracy and convergence are separate questions

### Fixed damped linear systems

The probe compares Cholesky of `J^T J + mu*D`, pivoted QR of the augmented
system, and an SVD reference. For

```text
J = [[1, 1], [1, 1 + 1e-8], [1, 1 - 1e-8]],
D = diag(J^T J), r = -J * [1, -1],
```

at `mu = 1e-16`, the step's absolute error against the SVD reference is
approximately `3.51e-2` for Cholesky and `7.86e-10` for QR. At `mu = 1e-20`,
Cholesky fails while QR succeeds. At `mu = 1e-3`, both routes agree closely.
This isolates the benefit of avoiding the rounded Gram matrix as damping becomes
small. Exact rank deficiency with zero damping needs a specified solution
policy; the diagnostic triangular solve is not such a policy.

### Whole solves and calibration probes

[lm-results.txt](lm-results.txt) records residual and Jacobian evaluations,
residual norms, parameter errors, and termination reasons for:

- Nearly collinear linear problems from two starts, an exactly rank-deficient
  problem, and a diagonal problem scaled across 16 orders of magnitude.
- Raw SVI with 41 log-moneyness observations over `[-0.5, 0.5]` and
  `[-0.01, 0.01]`, each from three starts.
- SSVI across four separated maturities and three closely spaced maturities,
  each from three starts.

These are deterministic synthetic observations with known parameters, not market
calibration data. Analytic Jacobians are checked by centered finite differences
before solving. Relative tolerances are `1e-12`; Basin's absolute gradient
tolerance is disabled. Each solver has at most `200*(n+1)` residual evaluations.
The prototype uses the same budget and Nielsen damping logic as Basin, with
either Cholesky or QR. It is diagnostic code, not a public solver or a proposed
rank-deficiency contract.

The narrow SVI case has Jacobian condition number about `1.48e8` at the known
solution. From start `[0.035, 0.12, -0.2, 0.01, 0.25]`:

  | Solver                      | Residual/Jacobian calls | Residual norm | Maximum parameter error | Stop          |
  | --------------------------- | ----------------------: | ------------: | ----------------------: | ------------- |
  | Basin                       |             1200 / 1199 |      1.974e-9 |                3.798e-2 | `MaxIter`     |
  | Nielsen prototype, Cholesky |             1200 / 1199 |      1.974e-9 |                3.798e-2 | `MaxIter`     |
  | Nielsen prototype, QR       |             1200 / 1199 |      1.974e-9 |                3.798e-2 | `MaxIter`     |
  | `levenberg-marquardt` 0.15  |                  10 / 7 |     1.388e-17 |               1.380e-11 | Relative step |

Both solvers exhaust their budgets from the other two narrow-SVI starts,
although MINPACK reaches smaller residuals. Broad SVI and both SSVI cases reach
small residuals with both solvers. Closely spaced SSVI maturities need 13–32
residual evaluations in Basin versus seven in MINPACK; replacing only the
factorization largely preserves Basin's counts.

The linear problem with column separation `1e-8` exposes an independent stopping
issue: starting from zero, both Nielsen prototypes stop on relative cost
reduction after two residual calls with parameter error about one. MINPACK
recovers the known parameters. The QR step is still heavily damped, so more
accurate factorization does not prevent the premature stop.

MINPACK's [lmder](https://netlib.org/minpack/lmder.f) factorizes the Jacobian
with column pivoting and calls a separate damping-parameter routine tied to its
trust-region radius. Its [lmpar](https://netlib.org/minpack/lmpar.f) and
[qrsolv](https://netlib.org/minpack/qrsolv.f) reuse that factorization while
applying diagonal regularization. Basin uses Nielsen's scalar damping update.
The probes therefore support investigating damping selection and stopping tests
independently of adding QR. Merely matching named tolerances does not make the
two complete algorithms equivalent.

Parameter recovery also needs an identifiability qualification. Exactly
rank-deficient systems have nonunique solutions. Unconstrained raw SVI has the
same residuals for positive and negative `sigma`; one broad-SVI start reaches
`sigma = -0.2` in both solvers. Its raw parameter error is 0.4 despite an
essentially exact fit.

### SSVI Jacobian mismatch in the downstream source

At stochastic-rs commit `1560c6cfc484327c3fe8f80092f0998f54252e06`,
[`SsviParams::phi`](https://github.com/rust-dd/stochastic-rs/blob/1560c6cfc484327c3fe8f80092f0998f54252e06/stochastic-rs-quant/src/vol_surface/ssvi/params.rs#L24)
uses `eta / (theta^gamma * (1+theta)^(1-gamma))`. The [calibration
Jacobian](https://github.com/rust-dd/stochastic-rs/blob/1560c6cfc484327c3fe8f80092f0998f54252e06/stochastic-rs-quant/src/vol_surface/ssvi/calibrate.rs#L146)
instead differentiates `eta * theta^(-gamma)`.

For the implemented residual, `dphi/dgamma` is
`phi * (ln(1+theta) - ln(theta))`. At `(rho, eta, gamma) = (-0.3, 0.5, 0.5)`,
`k = 0.5`, and `theta = 0.25`, the correct Jacobian row is approximately
`[0.113817, -0.016489, -0.013269]`; the downstream calibration code produces
`[0.126624, -0.011851, -0.008214]`. The retained SSVI probe differentiates the
actual residual consistently. Fix or account for this mismatch before using that
downstream calibration as evidence about either LM implementation.
