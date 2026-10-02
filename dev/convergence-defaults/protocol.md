# Experimental protocol

Status: **proposed for maintainer review**, not frozen. This protocol fixes the
measurement definitions and proposes the case mix, budgets, and selection rule
for steps 4–6. The [Misra1a pilot](runs/2026-10-02-misra1a-001.md) checks one
case and four solver families; it does not justify a default. Basin is a
general-purpose library with no preferred application domain. The investigation
should establish broad coverage and expose accuracy/work tradeoffs without
requiring application weights or a guessed universal price for accuracy. Use
development pilots to make the remaining numerical choices concrete, then freeze
the design before calibration. Record the accepted revision and decision ID
here.

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

The target grid samples a range of achieved accuracies. Its tightest target is
not a universal requirement for every default. Missing that target alone does
not establish false convergence. Report accuracy at termination, work to each
attainable target, and any additional improvement from a stricter control.

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

The [coverage matrix](coverage.md) and generated [vector case list](cases.csv)
now supply the first implementation of these strata. All 27 NIST datasets,
analytic least-squares controls, nonlinear constraints, scalar cases, and native
`f32` evaluation are available in `competitor-bench`. The [expanded baseline
pilot](runs/2026-10-02-coverage-002.md) uses development families only. The
matrix identifies missing variants and solver wiring; it does not freeze the
partition, establish precision floors, or authorize default selection.

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

Give no application domain preferential weight. Report results by problem family
and stratum, retaining dimension, precision, and target as separate axes. If a
pooled summary is useful, weight distinct applicable families equally and
normalize starts, instances, and transformed copies within each family. Adding
ten rotations of a quadratic must not give quadratics ten times the influence.
This convention controls duplication in the experiment; it is not an estimate of
how often users encounter each family. Keep individual families and worst cases
visible, and compare solvers on the same applicable cases when making a
cross-solver claim.

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

Separate the experiment's resource limit from a solver's numerical stopping
policy. A budget is permission to continue, not a requirement to spend the whole
allowance or evidence of convergence. The primary comparison for choosing a
default is between policies of the **same solver on the same case**. Give those
policies and the strict control the same upper allowance. If budget exhaustion
hides their stopping behavior, extend all policies in that comparison during
development rather than interpreting the limit as premature convergence.

For comparisons across solvers, record time to common attainable targets and
quality at a common sequence of elapsed-time budgets. More expensive iterations
then consume more of the allowance automatically. Choose a logarithmic budget
range from fresh development pilots, fix hardware and thread settings, and
freeze that range before calibration. Do not increase a solver's allowance
merely because its implementation does more work. Initialization, callbacks,
linear algebra, line searches, and inner solves count toward elapsed solve time;
diagnostic checks and trace output do not. Check time limits at supported
boundaries and report any overshoot. Use repeated, isolated timings and check
instrumentation overhead. The concurrent coverage pilot's timings cannot set
these budgets or support speed comparisons.

Record which resource actually ended each run. A separate evaluation or
iteration safety cap can interrupt a time-budget experiment; such a run has not
demonstrated what that solver would achieve with the full time allowance.

Also report evaluation work separately. An instrumented problem must count
physical value and derivative passes, finite-difference probes, and other
callback work; raw Basin counters remain separate. One Jacobian or forward
derivative pass need not cost the same as one objective evaluation. Equal
iteration counts or a sum of callback categories therefore do not establish
equal work or equal time. A comparison using a combined work unit needs a
declared cost model. Keep the underlying categories visible so results remain
useful when user objectives are much more expensive than the cheap fixtures.

Global and more general constrained methods can spend additional work on
exploration or feasibility. Cover longer budgets where needed to observe that
behavior, but offer the same extended budgets to every solver in a stated
comparison. Report local stationarity, feasibility, and global target attainment
separately. A valid local minimum is not a failed local stopping rule merely
because another basin has a better objective. Longer runs of a global method do
not certify global optimality. The earlier proposal of separate `1,000 * n` and
`10,000 * n` pass caps is superseded as a cross-solver comparison rule;
historical pilot budgets remain as recorded in their manifests.

This use of target-attainment and budget curves follows the distinction in
[COCO's performance
assessment](https://numbbo.github.io/coco-doc/perf-assessment/). COCO emphasizes
function evaluations and treats CPU timing separately. Here, both views matter
because Basin supports methods with different derivative and linear-algebra
costs. COCO also distinguishes the experiment budget from algorithm parameters
in its [experimental
guidance](https://coco-platform.org/getting-started/index.html). These sources
support the measurement approach; they do not supply Basin's numerical
thresholds or imply that an application-independent best default exists.

Pilot stochastic policies with 20 paired seeds and validate finalists with 50
paired seeds. Use the same seed and initial population for paired comparisons.
Summarize hit-rate uncertainty with Wilson intervals and paired work/quality
differences with family-level bootstrap intervals. These repetition counts
remain proposals, not a license to omit expensive solver families.

Select within each solver using the observed accuracy/work tradeoff. First
investigate incorrect stopping evidence, infeasible returned solutions, and
stops followed by substantial improvement under the strict control. Preserve the
distinction between a numerical failure, a budget limit, a valid local solution,
and a missed accuracy target. Among credible candidates, exclude policies that
cost more without improving accuracy or reliability, accounting for measurement
uncertainty. Prefer a simple policy whose behavior is stable under nearby
settings and across families. A pooled gain cannot conceal a severe
family-specific regression.

Development pilots should establish proposed material-effect thresholds using
reference uncertainty, repeated-run variability, target attainment, and the
extra work of strict controls. Record their rationale before calibration. There
is no requirement for the maintainer to guess a percentage tradeoff before
seeing evidence. If candidates retain a consequential accuracy/work tradeoff,
present that concrete comparison for a decision; retain the current default when
the evidence does not justify a change. Budget-driven methods may retain their
existing behavior with evidence and need no invented local-optimality test.

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
