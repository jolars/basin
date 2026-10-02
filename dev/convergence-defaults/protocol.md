# Experimental protocol

Status: **proposed for maintainer review**, not frozen. This protocol fixes the
measurement definitions and proposes the case mix, budgets, and selection rule
for steps 4–6. The [Misra1a pilot](runs/2026-10-02-misra1a-001.md) checks one
case and four solver families; it does not justify a default. The maintainer
must review the proposed case weights and reliability/work tradeoff before a
large calibration sweep. Record the accepted revision and decision ID here.

## Unit of comparison and independent success

One case is a problem family, dimension, start, transformation, derivative
source, precision, and constraint configuration. A solver enters a case only
when its public traits and documented domain make it applicable. Report the
returned point and stop separately from the best sampled point. A
`SolverConverged` code is a claim by the solver, not an independent success.

For a minimization case, record a reference objective `f_ref`, its provenance,
and uncertainty. Reevaluate the returned parameter with the unwrapped reference
formula after the solve. Its objective gap is `max(0, f(returned) - f_ref)`.
Keep a negative difference as a reference discrepancy for investigation, rather
than silently scoring it as success. Use `s_f = |f(start) - f_ref|` when this
exceeds the verified numerical floor; otherwise use the gap at a preregistered
perturbation of the reference point. Success at target `rho` means the gap is at
most `rho * s_f`, subject to a case- and precision-specific absolute floor. The
floor is a **resolution label**, not permission to mark a target below that
floor successful. Report absolute gap alongside relative gap so an objective
rescaling cannot hide a failure. An additive cost offset leaves both gaps
unchanged.

The proposed objective target grid is `rho = 1e-2, 1e-4, 1e-6, 1e-8` for `f64`
and `1e-2, 1e-3, 1e-4` for `f32`. A target enters a case only after the
reference and a strict run demonstrate that it is attainable at that precision.
Never replace a missing target by an easier one in the denominator. Record
attainment at each target and work to first attainment, including failure and
budget exhaustion. For local methods, distinguish a different validated local
minimum from premature termination; report both the local stationarity result
and the gap to the chosen global reference.

Use the following additional diagnostics where mathematically applicable. All
are recomputed after the run and excluded from solve costs.

  | Case                   | Independent diagnostic                                                                                                                                                     |
  | ---------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
  | Smooth unconstrained   | `‖D grad f(x)‖∞ / s_f`, where `D` contains declared coordinate scales; also report the unscaled gradient.                                                                  |
  | Bounded or constrained | Maximum bound/constraint violation and a projected-gradient or KKT residual using independently computed multipliers when available; a small step alone is insufficient.   |
  | Least squares          | Residual norm, `‖Jᵀr‖∞`, and identifiable parameter error; report zero and nonzero residual minima separately.                                                             |
  | Scalar minimum         | Objective gap and bracket/position error against an independently checked minimizer.                                                                                       |
  | Scalar root            | Absolute residual, root position error, and final sign-changing bracket width; do not substitute a small residual for root accuracy.                                       |
  | Stochastic or global   | Returned and best-sampled quality distributions, target-hit rate, and work under the same budget; a compact population or small rectangle is only a resolution diagnostic. |

`D` and all feasibility units belong in each case manifest. If the optimizer
cannot supply a multiplier or a local optimum is nonidentifiable, record that
fact and use the strongest valid diagnostic; do not label an uncomputed KKT test
as passed. Parameter error is secondary when equivalent parameters fit the same
observations. A solver failure at a high-quality point remains a termination
misclassification in the summary.

## Cases and partition

The first focused pilot uses NIST StRD Misra1a with both published starts, its
14 observations, certified parameters, and certified residual sum of squares.
NIST publishes the
[index](https://www.itl.nist.gov/div898/strd/nls/nls_main.shtml) and the
[Misra1a data and
certificate](https://www.itl.nist.gov/div898/strd/nls/data/LINKS/DATA/Misra1a.dat),
so this case does not depend on the reporter's private harness. Reproduce all 27
NIST nonlinear regression cases from those primary files, including both starts,
before claiming that the reported 54-run failure count is replicated. The
reporter's harness would help cross-check implementation and counting, but is
not required. Keep NIST model classes and difficulty levels visible.

For the broader sweep, assemble cases from the [Basin
corpus](../../crates/basin/src/problems.rs) and analytic constructions before
freezing a manifest. Proposed strata are smooth convex, curved nonconvex,
discontinuous/nonsmooth, multimodal, zero- and nonzero-residual least squares,
scalar roots/minima, active and inactive bounds, equality and nonlinear
inequality constraints, and noisy/stochastic objectives. Include dimensions 2,
5, and 10 for scalable problems where the algorithm applies, plus native
dimensions. Use at least two non-equivalent starts. For each relevant family,
include cost factors `1e-6, 1, 1e6`, cost offsets `0` and `1e6`, and coordinate
factors `1e-3, 1, 1e3` when the reference remains representable. Do not cross
every transformation blindly; use a preregistered fractional design that covers
each factor and keeps its untransformed anchor. Include fixed coordinates, rank
deficiency, non-finite probes, and stagnation away from a solution as diagnostic
cases.

Partition by **whole problem family** before calibration. Every start,
dimension, precision, and transformed copy of a family stays in the same
partition. Reserve at least one third of families in each applicable stratum for
validation. No selected setting may inspect those results until its candidate
and selection rule are frozen. If validation causes retuning, move the affected
families into development and reserve fresh families. A case manifest must list
all applicable solvers and expose gaps; do not infer broad coverage from a
solver's existing tests or from NIST alone. Add missing corpus problems through
the project's one-problem workflow.

## Candidates, resources, and selection proposal

For each solver, compare its current default, the versioned reference candidate
in [step 3](review-step3.md), and a strict continuation control. When a
numerical threshold is meaningful, use a small grid centered on its reference
value: one decade tighter and one decade looser, separately for `f32` and `f64`.
Include `None` and exact zero only when they express an applicable policy. Vary
which checks are enabled and their AND/OR composition as separate named
policies; never reinterpret a named tolerance from another algorithm. Use a
pilot to prune invalid candidates before the full grid. For DIRECT, exclude zero
geometry tolerances from the candidate grid: a finite subdivision of a nonfixed
box has positive exact radius and volume. Its setter and implementation
discrepancy remains Q006 for a separate behavioral fix.

The proposed per-case caps are `1,000 * max(n, 2)` full base-model passes for
local methods and `10,000 * max(n, 2)` for global/population methods. An
instrumented problem must count physical model passes inside each cost,
gradient, residual, or finite-difference callback; raw Basin counters remain
separate. Record family-specific overrides **before** calibration when one pass
has a different cost. Compare policies within the same solver and case at the
same cap, and report elapsed time as a secondary measure. An iteration cap is a
safety limit, not an equal-work unit. Pilot stochastic policies with 20 paired
seeds and validate finalists with 50 paired seeds. Use the same seed and initial
population for a paired comparison. Summarize hit-rate uncertainty with Wilson
intervals and paired work/quality differences with family-level bootstrap
intervals. These numbers are proposals, not accepted thresholds or a license to
omit expensive solver families.

The proposed selection rule is lexicographic within a solver family: reject
policies with a material increase in premature stops or false convergence at any
attainable target; among the remaining policies, prefer a simple one in a stable
region of the work/accuracy tradeoff. Inspect every family and worst case; a
pooled average cannot overrule a severe regression. Record the threshold for a
*material* reliability or work difference in the accepted protocol before
tuning. The maintainer's input is needed here because the choice expresses
Basin's intended reliability and computation tradeoff. Budget-driven methods may
retain their existing behavior with evidence; they need no invented
local-optimality test.

## Measurement and validation gates

Use `Stepper` or solver observers to record initialization, accepted points,
complete populations/sweeps, rejected trials, and completed inner segments as
distinct stages. At each boundary record all passing native tests, their
measurements and composition, the report stage/code, accepted and rejected work,
returned and best-sampled quality, and authoritative `Problem` counts. Fused
calls charge each produced category but count as one underlying callback when
measuring actual work. Count finite-difference probes and inner/bracketing calls
as physical work. Verification evaluations stay outside the solve totals. Keep
typed errors, non-finite cases, budgets, no-progress stops, and infeasible
results in the data.

Validate the recorder on an analytic quadratic and matching reference
implementation before interpreting a corpus run. Its accounting must agree with
`Stepper::counts`, including a stopped step and failed or rejected trial.
Trace-based threshold replay is allowed only when changing the threshold cannot
change the trajectory; confirm every finalist with a new solve at the correct
observation stage. Test both precisions and all claimed backends after
selection. Specifically resolve the [step 3
gates](review-step3.md#evidence-gaps-and-gates) for L-BFGS-B, SLSQP, constrained
stationarity, and composed inner/outer work before broad comparisons of those
families. The focused [Backtracking exhaustion
fix](../../crates/basin/src/line_search/backtracking.rs) now reports failure to
outcome-aware callers; verify its propagation in each affected solver during
step 5.

Each run needs a checked-in manifest and concise summary under
[runs](runs/README.md), with raw traces under ignored
`target/convergence-defaults/`. Freeze the final case list, reference answers
and floors, partitions, candidate IDs, budget overrides, and the
material-difference rule in a versioned protocol revision before large sweeps. A
numerical default changes only after a linked decision record and held-out
validation.
