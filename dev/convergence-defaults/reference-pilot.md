# Versioned reference pilot

This first reference pass compares *stopping semantics* for the issue's four
pilot families. It selects no Basin default. SciPy 1.16.2 provides a versioned
implementation for comparison; its algorithm, arithmetic, and execution budgets
can differ from Basin's. In particular, a shared tolerance name does not imply a
shared formula or observation stage. Comparisons for other solvers and research
references remain pending.

## Nelder–Mead

[SciPy 1.16.2's
implementation](https://github.com/scipy/scipy/blob/v1.16.2/scipy/optimize/_optimize.py#L3302-L3309)
defaults to `xatol=fatol=1e-4` and requires both
`max(abs(sim[1:] - sim[0])) <= xatol` and
`max(abs(fsim[1:] - fsim[0])) <= fatol` at the top of its iteration
([source](https://github.com/scipy/scipy/blob/v1.16.2/scipy/optimize/_optimize.py#L3578-L3587)).
It clips simplex vertices when bounds are supplied
([manual](https://docs.scipy.org/doc/scipy-1.16.2/reference/optimize.minimize-neldermead.html)).
[Basin's `NelderMead`](../../crates/basin/src/solver/nelder_mead.rs) exposes the
same two maximum-spread shapes against the best vertex, but disables both by
default. The bounded mode projects trials. The pair `1e-4, 1e-4` is a *reference
candidate*, not yet a quality or precision recommendation. Its absolute
parameter and objective scales require transformed-case tests.

## Bounded L-BFGS

[SciPy 1.16.2's L-BFGS-B
wrapper](https://github.com/scipy/scipy/blob/v1.16.2/scipy/optimize/_lbfgsb_py.py#L1945-L1952)
uses `gtol=1e-5` and `ftol=2.2204460492503131e-9` by default. Its documented
tests are maximum projected-gradient component at most `gtol`, and
`(f[k]-f[k+1])/max(abs(f[k]), abs(f[k+1]), 1) <= ftol`
([manual](https://docs.scipy.org/doc/scipy-1.16.2/reference/optimize.minimize-lbfgsb.html)).
The latter corresponds to `factr=1e7` times double-precision epsilon in its
legacy interface. [Basin's bounded `Lbfgs` and `Lbfgsb`
alias](../../crates/basin/src/solver/lbfgs.rs) enable only the absolute
projected-gradient test, with threshold `1e-10`. Its optional observed relative
cost-change check needs a formula and observation-stage comparison before it can
stand in for SciPy's `ftol`. The unbounded Basin mode has no enabled gradient
test and needs a separate unconstrained reference.

## Levenberg–Marquardt and trust-region reflective least squares

[SciPy 1.16.2's `least_squares`
manual](https://docs.scipy.org/doc/scipy-1.16.2/reference/generated/scipy.optimize.least_squares.html)
defaults to `ftol=xtol=gtol=1e-8` for both `lm` and `trf`, but gives each method
different tests:

  | Method | Reference test                                                                                                                                                                                                                                                    | Basin comparison                                                                                                                                                                                                                                                                                                                                                                                                    |
  | ------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
  | `lm`   | `gtol`: largest absolute cosine between a Jacobian column and residual is below threshold, or residual is zero. `xtol`: scaled trust radius below threshold times scaled parameter norm. `ftol`: sufficiently small cost reduction with adequate model agreement. | [Normal-equation LM and QR LM](../../crates/basin/src/solver/levenberg_marquardt.rs) enable `‖Jᵀr‖∞ <= 1e-8`; orthogonality, model reduction, relative step, and radius tests are opt-in. Both damping modes share these settings, but Basin's radius test applies only to trust-region damping. The absolute-gradient default is therefore not equivalent to SciPy's `gtol`.                                       |
  | `trf`  | `gtol`: infinity norm of bound-scaled gradient below threshold. `xtol`: `‖dx‖₂ < xtol(xtol + ‖x‖₂)`. `ftol`: sufficiently small cost reduction with adequate model agreement.                                                                                     | [Legacy `Trf`](../../crates/basin/src/solver/trf.rs) and [full `TrustRegionReflective`](../../crates/basin/src/solver/trust_region_reflective.rs) enable a bound-scaled gradient at `1e-8`; both leave observed cost and step checks off. The full variant tests free coordinates and accepted iterates. Basin's optional observed checks must be compared individually with SciPy's formulas and acceptance stage. |

SciPy 1.16.2 uses Jacobian scaling by default for `lm` and unit scaling for
`trf`; its default evaluation budget is `100*n`. These choices affect a
performance comparison even if stopping thresholds agree. SciPy's dense TRF and
Basin's full TRF are closer algorithmic comparators than SciPy's TRF and Basin's
legacy `Trf`, which uses simplified bounded LM. The reference's strict `<` tests
and Basin's `<=` tests can differ at an exact threshold. The reference's `f64`
defaults do not determine suitable `f32` settings.

### LM observation stages and MINPACK comparison

SciPy 1.16.2 calls MINPACK's `lmder` for `method='lm'`, with `diag=None` for its
default Jacobian scaling and `max_nfev=100*n` unless overridden ([tagged
wrapper](https://github.com/scipy/scipy/blob/v1.16.2/scipy/optimize/_lsq/least_squares.py#L46-L77)).
In [MINPACK's original `lmder`](https://www.netlib.org/minpack/lmder.f), the
orthogonality test runs after the Jacobian and its QR factorization at the
current point, before an inner trial. It checks the maximum absolute cosine
against `gtol`, including the zero-residual case. After a trial, MINPACK updates
the radius, accepts the trial only for gain ratio at least `1e-4`, and then
tests `|actred| <= ftol`, `prered <= ftol`, and `ratio <= 2`, or
`delta <= xtol * norm(diag*x)`. These last tests also run after rejection; the
returned point is then the retained base point. `actred` and `prered` are
relative to the base sum of squares. See the [gradient
check](https://www.netlib.org/minpack/lmder.f), the inner-loop acceptance, and
the convergence checks in that source.

[Basin's LM engine](../../crates/basin/src/solver/levenberg_marquardt.rs)
likewise checks its optional cosine criterion before a trial and its optional
model reduction and step criteria after a trial, including a rejected one. For
ordinary least squares, its model reduction test has the same three-part shape
as MINPACK's `ftol` test after normalization by the base cost. Basin accepts any
positive gain ratio, while MINPACK requires at least `1e-4`; their returned
points and subsequent radius histories can therefore differ. Basin's
`with_relative_step_tolerance` checks the unscaled trial step
`‖h‖ <= tolerance * ‖x‖`, which is not MINPACK's radius test. The separate
`with_relative_trust_radius_tolerance` checks
`radius <= tolerance * sqrt(xᵀ D x)` after the radius update, but only with
trust-region damping. Nielsen damping has no corresponding radius. The normal
equations and QR variants share these stopping settings, though their steps and
rank handling differ. For robust losses, Basin's gradient-orthogonality builder
tests a different normalized robust gradient; MINPACK's residual cosine is not a
reference for that mode.

### TRF observation stages and optional Basin checks

In [SciPy's tagged bounded
TRF](https://github.com/scipy/scipy/blob/v1.16.2/scipy/optimize/_lsq/trf.py#L263-L274),
the bound-scaled gradient test runs at the current accepted point, including the
initial point. During each inner attempt, the solver evaluates a trial, updates
its radius, and calls the shared
[`check_termination`](https://github.com/scipy/scipy/blob/v1.16.2/scipy/optimize/_lsq/common.py#L705-L717)
*before* accepting that trial. Its cost test requires
`actual_reduction < ftol * base_cost` **and** `ratio > 0.25`; its step test
requires `‖step‖₂ < xtol * (xtol + ‖base_x‖₂)`. The two tests combine with OR,
and a simultaneous pass has its own status. A rejected or equal-cost trial can
therefore satisfy the step test while SciPy returns the unchanged base point
([tagged trial
loop](https://github.com/scipy/scipy/blob/v1.16.2/scipy/optimize/_lsq/trf.py#L349-L385)).

[Basin's full
`TrustRegionReflective`](../../crates/basin/src/solver/trust_region_reflective.rs)
checks its native bound-scaled gradient at the initial point and after an
accepted step. Its optional shared observed cost test uses the absolute change
between accepted costs, `|F_k - F_(k-1)| <= tolerance * |F_(k-1)|`; its optional
step test uses `‖x_k - x_(k-1)‖₂ <= tolerance * ‖x_k‖₂`. Neither tests rejected
trials, includes SciPy's gain-ratio condition or additive `xtol²` term, or
reproduces SciPy's strict inequality. Its numerical no-progress stop is a
separate failure path. The legacy [`Trf`](../../crates/basin/src/solver/trf.rs)
uses a different bounded-LM model and its own native scaled-gradient check;
SciPy's TRF is not an algorithmic match for it.

## Next checks

The [pilot candidate register](candidate-pilot.md) records testable formulas and
variant distinctions without selecting defaults. Compare the original L-BFGS-B
acceptance and cost rule, then add independent quality targets and reference
versions for the remaining solver families. The shared observed check lifecycle
and robust-loss variants need focused traces before calibration.
