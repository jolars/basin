# Pilot stopping-policy candidates

Status: draft candidates for the Nelder–Mead, L-BFGS, LM, and TRF pilot. These
formulas are testable alternatives, not selected defaults or implementation
instructions. The [reference comparison](reference-pilot.md) separates similar
tolerance names with different mathematics. All candidates require calibration
against independent solution quality, a separate `f32` policy, held-out
validation, and an explicit decision in [decisions.md](decisions.md). Keep the
current defaults as the control in every comparison.

## Nelder–Mead

For unbounded standard, adaptive, and custom coefficients, compare the current
budget-driven policy with a paired absolute simplex rule. At each initialized or
updated simplex, let `b` be its lowest-cost vertex and require both
`max_i ‖x_i - x_b‖∞ <= xatol` **and** `max_i |F_i - F_b| <= fatol`. This is
already expressible with Basin's two simplex builders. SciPy 1.16.2's
`xatol=fatol=1e-4` is an `f64` reference anchor, not a scale-independent
proposed default. Compare smaller and larger values after the protocol fixes
problem scales and quality targets. Run the same candidate for projected
Nelder–Mead as a separate stratum: projection can collapse vertices at an active
bound without establishing local optimality. Require independent feasible
quality checks there. Neither observed best-cost stagnation nor a single small
simplex spread should replace the conjunction.

## Bounded and unbounded L-BFGS

For bounded `Lbfgs::new()` and its `Lbfgsb` alias, compare the current
`‖x - projection(x - g)‖∞ <= 1e-10` rule with the following alternatives at the
current accepted iterate: (A) projected gradient alone at calibrated absolute
thresholds; (B) projected gradient **or** normalized accepted cost decrease
`(F_(k-1) - F_k) / max(|F_(k-1)|, |F_k|, 1) <= ftol`, guarded by a finite,
nonnegative decrease and a successfully accepted line-search step. SciPy 1.16.2
supplies `gtol=1e-5` and `ftol=2.2204460492503131e-9` as `f64` reference
anchors. Basin's shared relative-cost builder checks
`|F_(k-1) - F_k| <= tolerance * |F_(k-1)|`; it cannot directly express (B).
Record line-search failure and a zero accepted step separately. Test custom line
searches as distinct paths because they can change acceptance and evaluation
work. For unbounded `Lbfgs::unbounded()`, compare its current budget-driven
policy with a Euclidean gradient-norm stop at calibrated absolute or relative
thresholds. Do not transfer bounded projected-gradient or SciPy L-BFGS-B cost
thresholds to that different mode without an unconstrained reference comparison.

## Levenberg–Marquardt

For ordinary least squares, compare the current absolute gradient-only stop
`‖Jᵀr‖∞ <= 1e-8` with (A) orthogonality alone,
`max_j |(Jᵀr)_j| / (‖J[:,j]‖₂ ‖r‖₂) <= gtol`, and (B) that criterion **or** the
paired model reduction condition `|F(x)-F(x+h)| <= ftol * F(x)`,
`predicted_reduction <= ftol * F(x)`, and `gain_ratio <= 2`. The zero-residual
case is an exact-gradient success. For trust-region damping, add a separate
alternative that ORs the updated-radius criterion
`radius <= xtol * sqrt(xᵀ D x)` with (B). These are Basin's native tests;
SciPy's three `1e-8` defaults supply only `f64` starting anchors. Keep Nielsen
damping separate because its radius criterion is inactive. Do not treat Basin's
unscaled relative trial-step builder as MINPACK `xtol`.

Cross each damping choice with normal-equations and pivoted-QR factorization in
the pilot, using the same stopping formulas but separate outcomes. Preserve the
default finite rejected-trial no-progress safeguard and the inner retry limit;
report them separately from convergence. For robust objectives, retain the
existing absolute gradient control and assess Basin's robust normalized gradient
test separately. MINPACK's residual-angle rule does not define a robust-loss
candidate. Any model reduction criterion in that mode needs a separate
model-agreement and quality check. The [robust analytic
pilot](runs/2026-10-09-robust-ls-001.md) demonstrates that tiny accepted steps
under large Nielsen damping can pass both optional model-reduction and
trial-step tests on Cauchy while the robust gradient remains large. Paired
default continuations reach the known minimum. Review this composition before
sweeping robust candidates; the explicit measurement probes select no policy.
The [stopping composition review](runs/2026-10-10-robust-stopping-001.md)
isolates model-only and step-only stops and verifies their exact callback
prefixes against gradient continuations. Both predicates pass at the same
accepted Cauchy trials with good gain ratios, so a conjunction or
acceptance/model-agreement filter alone cannot fix those stops. Apply
[D005](decisions.md#d005-screen-robust-lm-progress-tests-against-stationarity)
before carrying robust progress candidates into calibration. Keep the absolute
gradient default as the control and assess normalized gradients separately,
including their zero-residual behavior.

## Trust-region reflective least squares

For full `TrustRegionReflective`, compare the current bound-scaled gradient test
`‖v ⊙ Jᵀr‖∞ <= 1e-8` with (A) the same metric at calibrated absolute thresholds
and (B) that test **or** a trial cost rule `F(x)-F(x_trial) < ftol * F(x)` with
gain ratio `> 0.25` **or** a trial step rule
`‖x_trial-x‖₂ < xtol * (xtol + ‖x‖₂)`. Evaluate the trial rules after a finite
trial and radius update but before acceptance; return the accepted base when a
rejected trial triggers a stop. SciPy 1.16.2's three `1e-8` values are `f64`
reference anchors. Candidate (B) requires native trial diagnostics and cannot be
reproduced with Basin's shared observed builders. Also compare an
accepted-step-only alternative to isolate the effect of SciPy's rejected-trial
step stop. Record zero-cost and zero-step cases and keep numerical no-progress
distinct from convergence. Test active bounds, fixed coordinates, deficient
rank, and robust objectives as separate strata; the all-fixed structural stop
remains independent of a tolerance.

For legacy `Trf`, retain its bounded-LM scaled-gradient default as the control
and compare calibrated thresholds for that native metric. Its model and
acceptance path differ from full TRF, so the SciPy trial formulas above are not
a direct candidate for this type. Consider optional accepted cost or step checks
only after a legacy-specific reference and stagnation review.

## Before calibration

Specify a finite `f64` and `f32` threshold grid around each applicable anchor,
with units and attainable floors stated for each problem family. Freeze
problem-family partitions, quality targets, evaluation budgets, failure
classification, and policy selection rules in the [protocol](protocol.md). The
first traces must show each test's value and observation stage, including
rejected trials, as well as the returned point. Policies requiring new native
checks remain proposals until code, regression tests, and migration guidance are
reviewed.
