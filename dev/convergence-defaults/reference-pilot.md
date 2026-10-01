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

## Next checks

Read the versioned implementation paths for SciPy's LM and TRF termination
stages and the original MINPACK and L-BFGS-B rules, then compare Basin's
optional observed checks line by line. Add independent quality targets and
reference versions for the remaining solver families before any default
decision. Do not promote these reference candidates into the per-solver
candidate or decision tracker until their exact Basin formulas and tests are
specified.
