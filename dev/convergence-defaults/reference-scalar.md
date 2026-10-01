# Scalar minimization and root references

Status: reference comparison and draft candidates for all eight public scalar
solver names. The [inventory](inventory.md#scalar-stopping-records) records
Basin's current formulas. No defaults have been selected.

## Scalar minimum references

SciPy 1.16.2's
[`Brent`](https://github.com/scipy/scipy/blob/v1.16.2/scipy/optimize/_optimize.py)
uses `xtol=1.48e-8` and `maxiter=500` by default. At its current best interior
point `x`, with bracket midpoint `m=(a+b)/2`, it tests
`|x-m| < 2(xtol*|x| + 1e-11) - (b-a)/2`. The `1e-11` term is an implementation
floor; the public [`minimize_scalar`
documentation](https://docs.scipy.org/doc/scipy-1.16.2/reference/optimize.minimize_scalar-brent.html)
calls `xtol` a relative solution tolerance. Basin's `Brent` uses a similar
geometry but a non-strict comparison, `sqrt(ε_F)` relative tolerance, and
`1e-12` absolute tolerance. SciPy's constants are `f64` values; its local
iteration limit is not an accuracy criterion. Basin's limit belongs to the
executor. Bracketing expansion and interpolation acceptance are dependencies,
not completion of the enclosing minimum search.

SciPy 1.16.2's
[`Golden`](https://github.com/scipy/scipy/blob/v1.16.2/scipy/optimize/_optimize.py)
uses `xtol=√ε_float64` and `maxiter=5000` by default. Its stop compares the full
outer bracket width with `xtol*(|x_1|+|x_2|)`, where `x_1` and `x_2` are the two
interior probes. Basin's `GoldenSection` instead uses
`b-a <= 2(sqrt(ε_F)*|x_best| + 1e-12)`. These rules can stop at different widths
near zero or when the two interior probes are asymmetric. Neither rule tests
objective accuracy. SciPy has no derivative-guided counterpart for
`BrentDerivative`; Basin's derivative is a trial-selection input, with an
optional independent absolute derivative stop.

## Scalar root references

SciPy 1.16.2's bracketed
[`brentq`](https://docs.scipy.org/doc/scipy-1.16.2/reference/generated/scipy.optimize.brentq.html)
and
[`toms748`](https://docs.scipy.org/doc/scipy-1.16.2/reference/generated/scipy.optimize.toms748.html)
default to `xtol=2e-12`, `rtol=4ε_float64`, and `maxiter=100`. They document
position accuracy as `|x_returned-x_root| <= xtol+rtol*|x_returned|` for a
continuous sign-changing bracket. The TOMS 748 method uses inverse cubic and
Newton-quadratic steps; its `k=1` default means one Newton-quadratic step per
iteration. The two algorithms share tolerances without sharing a step path or
iteration cost. Basin's five root solvers default to `1e-12`, `4ε_F`, and a
100-step local limit. Their success rule is an exact zero at an evaluated point
**or** a sufficiently narrow sign-changing bracket, which is an algorithmic
width check rather than direct knowledge of the unknown root.

SciPy 1.16.2's unbracketed
[`newton`](https://docs.scipy.org/doc/scipy-1.16.2/reference/generated/scipy.optimize.newton.html)
uses a step-size test, with `tol=1.48e-8`, `rtol=0`, and `maxiter=50` by
default. It selects secant, Newton, or Halley updates from the supplied
derivatives. Its documentation warns that a small step does not guarantee a
root. Basin's `SecantRoot`, `NewtonRoot`, and `HalleyRoot` are **bracketed**
algorithms with bisection safeguards and the shared exact-zero-or-width rule.
SciPy `newton` is therefore a path and safeguard comparator, not a stopping
policy to copy into those Basin types.

The absolute floors and relative coefficients above assume SciPy's `f64`
arithmetic. Basin's `4ε_F` coefficient changes with `F`, but its absolute
`1e-12` root floor does not. An `f32` root near zero can reach a representable
exact zero, while a root far from zero may hit an adjacent-float bracket well
before an absolute `1e-12` width is meaningful. Test both cases. A small
`|f(x)|` can also be misleading when the derivative is tiny; use an independent
root-position or bracket target when a reference root is known.

## Candidate policies to measure

Retain every current default as control. Compare finite positive absolute and
relative position tolerances within each setter's accepted domain; these
algorithms intentionally require positive position tolerances, and the root
relative setter requires at least `4ε_F`. Evaluate each candidate on the bracket
maintained by that solver, with its own strict or non-strict comparison. Keep
exact-zero success as a separate OR branch for roots. Keep invalid brackets,
non-finite function values, representational stalls, and iteration limits as
nonconvergence outcomes.

  | Basin solver                                                         | Candidate and observation                                                                                                                                                                                                  | Independent quality measure                                                                                                                                        |
  | -------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
  | `Brent`                                                              | Current `sqrt(ε_F)` and `1e-12` bracket rule versus a small scale-aware grid for each threshold. Check the current best interior point after initialization and each update.                                               | Parameter error where the local minimizer is known; objective excess and bracket containment otherwise.                                                            |
  | `BrentDerivative`                                                    | Same bracket grid; separately measure an optional `abs(f′(x_best)) <= gtol` OR branch and a bracket **AND** derivative alternative.                                                                                        | Parameter and objective error, plus derivative residual only where the derivative is a reliable stationarity measure.                                              |
  | `GoldenSection`                                                      | Current rule versus candidate absolute and relative scales using its full bracket width.                                                                                                                                   | Parameter and objective error; compare work at matched target widths with `Brent`.                                                                                 |
  | `BrentRoot`, `SecantRoot`, `NewtonRoot`, `HalleyRoot`, `Toms748Root` | Current exact-zero-or-width rule versus per-precision absolute and relative floors, with shared thresholds but distinct step paths. Inspect the returned endpoint, bracket width, and evaluated residual at each decision. | Root-position error for a known root, or independently certified enclosing bracket width; residual accuracy is additional evidence, not a substitute for position. |

The experiment must include roots near zero and far from zero, steep and flat
functions, a root at an endpoint, derivative zero or near zero,
finite-difference derivatives, a sign-changing discontinuity that must not be
counted as a root, and non-finite callbacks. For minimization, include shifted
and rescaled objectives, shallow and asymmetric wells, and minima near zero and
at large parameter magnitude. Compare `f32` and `f64` separately. The protocol
still needs to specify target grids and attainable floors before any position
threshold can become a proposed default.
