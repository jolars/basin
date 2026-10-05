# Experimental protocol

Protocol `CDP-1`, reviewed 2026-10-05. This fixes the step 4 design under
[D003](decisions.md#d003-use-cdp-1-for-calibration). It selects no solver
defaults and reports no numerical experiments. The [review](review-step4.md)
records the checks and step 5 gates. Freeze executable case and candidate
manifests, reference certificates, and their hashes after the pilot and before
calibration. A pilot may expose a design defect; record an amendment before
using affected measurements for selection.

## Questions and experimental units

For every solver and applicable variant, determine whether stopping returns
useful accuracy, loses attainable accuracy, or spends substantial work after
reaching it. Compare policies within the same algorithm, initialization,
derivative implementation, precision, and backend. An external implementation is
a reference comparison only after documenting its algorithm differences.

One run is a policy applied to a case, start, precision, backend, derivative
mode, algorithm variant, and seed. A case specifies its mathematical objective,
constraints, domain, dimension, and transformation. A family contains related
models and every start, dimension, transformation, and residual representation
of those models. The [case register](cases.md) fixes families and partitions.

Maintain separate records for all names and variants in the
[inventory](inventory.md), including both LM factorizations and damping modes,
bounded and unbounded L-BFGS, legacy `Trf` and full `TrustRegionReflective`,
trust-region strategies, robust losses, stochastic schedules, and inner solvers.
An unavailable implementation is a coverage gap, not a failed solve or implicit
exclusion. Fix mathematical incompatibilities before execution. Each family
sweep requires its step 2 audit and step 3 evidence gates to be closed.

## Independent reference certificates and scales

Each case needs a certificate produced without candidate results. Prefer an
analytic optimum or certified source. Otherwise use at least two independent
algorithms or implementations, analytic derivatives where possible, and at least
100 decimal digits for reference refinement and residual checks. Agreement alone
does not prove global optimality: label the reference `analytic`,
`certified-best-available`, `validated-local`, or `unknown`. Preserve its point,
objective interval, derivative and feasibility residuals, parameter uncertainty,
precision, commands, source version, and input hashes. Unknown references permit
diagnostic runs but cannot enter objective-target selection.

Let `f` be the actual objective. Least squares uses the explicit convention
`f = kappa * sum(r_i^2)`, with `kappa` recorded for each adapter. Convert NIST
RSS and its uncertainty accordingly. Robust loss requires its own objective and
validated reference; an ordinary least-squares optimum is not a robust
reference. A barrier or penalty objective never replaces the original quality
measure.

Before observing candidates, record positive coordinate scales `s_i`, constraint
scales, objective absolute scale `A_f`, and origin `o`. Define
`z_i = (x_i - o_i) / s_i`. Prefer physical units; otherwise use the fixed domain
width, or `s_i = 1` for dimensionless unbounded fixtures. For NIST use
`s_i = max(abs(beta_ref_i), abs(start1_i), abs(start2_i))`, replacing zero by
one in that coordinate's units. These are verifier scales, not secret
preconditioning supplied to candidates. Never infer scales from a returned point
or stop history.

For a linear constraint row use scale
`max(abs(b_j), sum_i abs(A_ji)*s_i, one constraint unit)`. Dimensionless
nonlinear fixtures use one; other nonlinear constraints must declare a physical
unit in their certificate. Bounds use their coordinate scales. These constants
remain fixed across starts and candidates, with explicit changes under
transformations.

For each start, set `R_f = abs(f(x0) - f_ref)` and `S_f = max(A_f, R_f)`. The
absolute and relative allowances are `q * A_f` and `q * R_f`; the objective
allowance is their maximum, `q * S_f`. A zero initial gap is therefore defined.
Use `A_f = 1` for dimensionless analytic objectives. For regression use
`A_f = kappa * sum((y_i - mean(y))^2)` in the modeled response space, with
`kappa * sum(y_i^2)` as the zero-variance fallback and one unit of squared
response as the all-zero fallback. For Nelson, use its log-response model.
Freeze the resulting numbers. Do not scale by `abs(f_ref)`: objective offsets
must not loosen the verifier.

For `f_t(z) = a * f(o + D*z) + b`, `a > 0`, transform references and objective
scales by `a`, constraints by their declared positive scaling, and coordinate
scales through `D`. Offset `b` changes neither target nor stationarity. Compute
gaps with the unshifted formula in the independent verifier to avoid subtracting
large rounded costs. Also retain the actual solver cost and quantify information
lost in that evaluation.

## Accuracy targets and attainable precision

Use every eligible target below. The designated target guides selection;
neighbors expose sensitivity. These are experimental accuracy requirements, not
proposed stopping tolerances.

  | Precision | Objective and feasibility grid `q`     | Designated target |
  | --------- | -------------------------------------- | ----------------- |
  | `f64`     | `1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12` | `1e-6`            |
  | `f32`     | `1e-2, 1e-3, 1e-4, 1e-5, 1e-6`         | `1e-3`            |

For each case, derivative mode, and precision, preflight an attainable floor
before tuning. Bound reference rounding, data conversion, cancellation, and
derivative error in the measures below. Evaluate a rounded reference and its
representable neighbors with actual scalar arithmetic and an independent
high-precision check. Refine with independent methods if needed. Store both
uncertainty bounds and a representable witness meeting each eligible joint
target. Require every allowance to exceed ten times its uncertainty bound. Use
at least eight unit roundoffs times the sum of absolute evaluation terms as a
rounding-error screen, then validate cancellation-prone and ill-conditioned
cases separately; the screen alone does not certify attainability.

A target below a demonstrated floor is `precision-ineligible`, never a success
at a silently enlarged tolerance. An unverified floor is `reference-pending`.
Neither may be inferred from poor candidate performance. Fix eligibility for all
candidates together and list omitted targets/cases in each precision. Retain
their final-quality and failure reports. If the designated target is ineligible,
show other targets but exclude the case from its score. A family with no
eligible designated-target case blocks a default claim for that stratum.
Separate reference rounding, arithmetic limits, and finite-difference error.
Statistical noise in fixed observations is not fresh evaluation noise.

## Quality tests at a returned point

Reevaluate saved coordinates independently, without trusting cached costs or
termination codes. Verification calls remain outside the solve ledger.

  | Measure                          | Definition and passing allowance                                                                                                                                                                                                                         |
  | -------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
  | Objective error                  | Conservative upper gap `max(0, f_upper(x) - f_ref_lower) / S_f <= q`. If an objective interval lies wholly below the reference interval beyond recorded uncertainty, invalidate the reference and review it; do not clamp that discrepancy into success. |
  | Unconstrained stationarity       | `g_z = D_s * grad(f) / S_f`; require `norm_inf(g_z) <= sqrt(q)`. Differentiate the recorded least-squares convention and actual loss.                                                                                                                    |
  | Box stationarity                 | `norm_inf(z - project_[l_z,u_z](z - g_z)) <= sqrt(q)`, plus feasibility. Fixed coordinates have zero projected displacement but still need a feasibility check.                                                                                          |
  | Feasibility                      | Maximum of scaled absolute equality residuals, positive inequality residuals, and positive bound violations, using `h = 0`, `c <= 0`; require `v <= q`. Retain each category.                                                                            |
  | General constrained stationarity | In scaled coordinates, minimize the maximum of `norm_inf(g_z + J_h^T*lambda + J_c^T*mu)` and `norm_inf(mu .* c_scaled)` over free `lambda` and `mu >= 0`. Include bounds as inequalities. Require residual `<= sqrt(q)` and feasibility `<= q`.          |
  | Identifiable parameter error     | Distance to the certified solution set: `inf_xref norm_inf((x - xref) / s) <= sqrt(q)`. Respect known permutations/sign symmetries. Require this in parameter-recovery strata; otherwise report it as a diagnostic.                                      |

The constrained verifier solves its multiplier subproblem independently and
retains multipliers, residuals, rank, and constraint qualification concerns.
Validate its residual rather than copying solver multipliers or trusting its
termination flag. Large cancelling multipliers or singular constraints need an
analytic certificate or leave stationarity indeterminate. KKT residuals are
first-order diagnostics, not global certificates. Barrier centering/duality-gap
measurements and ALM inner stationarity remain additional diagnostics with inner
capabilities recorded.

On smooth problems, **joint target success** requires objective accuracy,
stationarity, feasibility if constrained, and parameter accuracy where required.
Derivative-free solvers on smooth problems receive the same external derivative
check without gaining derivative access during the solve. On nonsmooth or
discontinuous problems without a stationarity certificate, report objective and
feasibility success and mark stationarity unavailable; keep this stratum
separate. Rank-deficient and nonidentifiable problems do not fail merely because
a parameter vector differs from a chosen representative.

### Scalar minimization and roots

For scalar minima, use objective and identifiable-position tests. Check
derivative stationarity where it exists; retain final brackets and their
containment/width separately. A narrow bracket alone is not objective accuracy.

For a root of `h`, use the same `q` grid with position allowance `q * S_x`,
where `S_x = max(A_x, abs(x0 - x_ref))` and `A_x` is the positive coordinate
unit. Require distance to the certified root set within that allowance and
`abs(h(x)) <= q * S_h`, where `S_h = max(A_h, abs(h(x0)))` and `A_h` is the
positive function unit. There is no objective-gap requirement. Flat/multiple
roots still require position accuracy. For bracketed methods also require finite
ordered endpoints, certified root containment, and half-width `<= q * S_x`,
unless an exact-root exit independently passes the position and residual tests.
Such an exit may leave an older wide bracket, which remains visible as a bracket
diagnostic. Record sign enclosure, width, endpoint ULPs, and exact zero
separately. A floating-point callback zero is not itself a reference
certificate. Unbracketed methods have no invented bracket requirement. Apply the
same precision eligibility checks to all allowances.

### Local minima and premature termination

A missed global-reference target is not by itself premature local termination.
Keep global-reference success and local stopping adequacy as separate columns.
For a smooth alternate point, check stationarity and feasibility, then establish
local minimality analytically or combine feasible-direction curvature checks
with independent refinement from the point and small feasible perturbations. Use
`+/- 1e-3` in each scaled free coordinate and record constraint restrictions. A
stationary saddle is not an alternate minimum. Singular/nonsmooth evidence
without a certificate remains `unresolved`.

Run stricter continuation for every shortlisted deterministic policy and paired
stochastic control, preserving safeguards. Prefer exact solver/state/RNG/counts
continuation disabling only the candidate stop. If unavailable, run a fresh
stricter solve from the original start and seed; label a restart from the
stopped point separately because it resets history. Controls share the same
total budget, including work before a candidate stop. A missed target reached by
this control on a verified matching trajectory is confirmed premature
termination. If changing a schedule or inner tolerance changes the trajectory,
report the improvement as policy sensitivity rather than proof about that
stopping point. Failure to improve does not prove optimality. A supported
alternate minimum remains a global-target miss but is not a premature stop. Also
report adequate answers returned with failure or no-progress codes.

## Transformations, derivatives, and noisy evaluations

The [case register](cases.md) fixes dimensions, starts, and case supply. Apply
this transform set where mathematically supported: baseline; objective
multipliers `1e-8, 1e8`; offsets `-1e8*A_f, +1e8*A_f`; uniform coordinate
multipliers `1e-4, 1e4`; alternating coordinate multipliers `1e-4, 1e4`; and the
combination of objective multiplier `1e8` with alternating coordinate scaling.
Transform starts, domains, constraints, and references consistently. Do not take
a full Cartesian product. Offset cases test information loss and may have
ineligible targets. Keep every transform visible in reports.

Residual models implement objective scaling by multiplying residuals/Jacobians
by `sqrt(a)`. Arbitrary additive offsets cannot be supplied through a pure
residual API; mark them inapplicable there and use cost adapters where
supported. Appending a constant residual is a separate nonzero-residual case
because it changes residual-angle diagnostics. For smooth derivative-capable
cases, compare analytic derivatives with forward and central finite differences
wherever the adapter is supported. Freeze production adapter defaults and record
actual perturbations, bound handling, and underlying calls. Verify using
independent analytic/high-precision derivatives. Add Hessians only with honest
supported implementations. Keep derivative modes separate.

For supported noisy methods, add zero-mean Gaussian objective noise with
standard deviation `1e-6*S_f` and `1e-3*S_f` to the quadratic and Rastrigin
development families and Levy validation family. SGD also gets a finite-sum
quadratic with an exactly known full objective/gradient and fixed sampling
scheme. Record RNG and sample-index streams. Pair seeded noise by callback kind
and call number; different trajectories need not see noise at identical
coordinates. Score saved points on the latent objective/full gradient. Keep this
separate from regression with fixed noisy data. Real noisy objectives lacking a
latent reference need independent replicated verification and confidence bounds;
they remain exploratory and cannot enter `CDP-1` default selection.

## Work ledger, budgets, and observation stages

Extend the [trace harness](../../crates/competitor-bench/src/bin/trace.rs) and
[LM probe](../../crates/competitor-bench/src/bin/verify_lm_stopping.rs) in step 5.
Preserve `Problem::EvalCounts` as authoritative public categories and add a
reconciled ledger of physical leaf oracle calls. One full residual vector is one
residual call, not one per observation. A fused cost/gradient call is one
physical call with both logical categories. Finite differences count every leaf
call, including retries, without counting the adapter wrapper again. Retain
analytic gradient, Jacobian, Hessian, Hessian-vector, and constraint counts. A
batch of `k` evaluated points counts `k` point calls. Record cache hits without
charging another call. Fused model/constraint evaluations follow the same rule.

Define `W` as all physical leaf point calls, with unit weight per call. This
administrative budget supports comparisons within a stratum; it does not claim a
Jacobian and cost have equal computational cost. Report the category vector,
residual dimension, and time beside `W`. Compare policies with identical
callback contracts; do not rank unrelated derivative regimes using pooled `W`.

  | Run class                                   | Total leaf-call cap `B` | Completed-iteration cap | Wall cap per run |
  | ------------------------------------------- | ----------------------- | ----------------------- | ---------------- |
  | Scalar minimum or root                      | `1000`                  | `1000`                  | 60 seconds       |
  | Local vector, least squares, or constrained | `2000 * (n_free + 1)`   | `10000`                 | 10 minutes       |
  | Global, population, composed global, or SGD | `10000 * (n_free + 1)`  | `100000`                | 30 minutes       |

`n_free` removes fixed coordinates; retain original dimension. Outer and inner
solves, initialization, bracketing, finite differences, failed calls, and
rejected trials share the cap. All-fixed cases still get one unit in the
dimension factor and verify their supplied point. Observe quality at fractions
`0.01, 0.1, 1` of `B`, using the last published recommendation at or before that
budget. For an earlier stop, carry its returned recommendation into fixed-budget
quality while showing actual stop/work. Do not pad its evaluation count or
trace.

Enforce the cap before a leaf call. If initialization or an indivisible
operation cannot finish, retain the last coherent state and classify an
incomplete/budget outcome. If the API cannot safely return it, capture callback
history and mark the returned point unavailable. An overshoot is a harness
defect. Native limits may additionally be set but are not this aggregate cap.
The pilot must test exact boundaries, initialization, and nested calls. Wall
caps protect the runner; wall-limited cases are censored and cannot establish a
work advantage.

At initialization, accepted outer points, complete populations/DIRECT sweeps,
rejected-trial stop decisions, and inner-segment completion, retain the
published recommendation, stage, and counts. A callback trial is not a
recommendation; keep its quality in a separate best-sampled series. A rejected
TRF trial does not replace the accepted base. Preserve actual constrained
selection ordering rather than choosing the lowest objective offline.

Record all passing/failing stop clauses, operands, thresholds, comparators,
norms, enabled states, composition, and winning code. Missing diagnostics remain
missing; do not infer them from unchanged coordinates. Distinguish native
convergence, known/exact-root exits, structural no progress, callback/domain
failures, budgets, application stops, and inner-only completion. Retain
non-finite values and their evaluation stages.

Timing uses release builds, one solver thread, fixed hardware/thread settings,
and separate uninstrumented finalist replays: one warm-up, then five measured
runs in interleaved policy order, reporting median and range. Offline
verification is never solve work. Timing cannot rescue a policy that fails
quality gates.

## Seeds, repetitions, and candidate grids

Deterministic runs use each fixed start once per configuration. Stochastic runs
use 30 paired seeds per case/start for development screening, extend shortlisted
development candidates to 100, and use 100 on validation families. The extension
uses repetitions 30 through 99 rather than replacing the first 30. Derive a
64-bit seed from the first eight bytes, big-endian, of SHA-256 of UTF-8
`CDP-1|<stream>|<family>|<case>|<start>|<rep>`. Repetitions start at zero;
stream is `solver`, `noise`, or `start`. The case field is the untransformed
case ID. Exclude policy, backend, and precision so comparisons are paired. Pin
RNG algorithm/version and materialize all seeds. Pair initial states too;
identical seeds do not ensure pairing when initialization differs. Numerical
failures receive no replacement seeds. Infrastructure retries retain both
attempts under the same ID/seed.

Include current defaults, applicable step 3 reference policies, and a stricter
budget-driven control retaining safeguards. For a candidate with a transferable
positive `f64` anchor `t0`, use `t0 * {1e-2, 1e-1, 1, 1e1, 1e2}`. Without an
anchor, use `{1e-12, 1e-10, 1e-8, 1e-6, 1e-4}` in documented units.
Independently use `{1e-6, 1e-5, 1e-4, 1e-3, 1e-2}` for `f32`, plus its current
default. Formula-specific representability restrictions and inactive criteria
are explicit exclusions. Exclude DIRECT zero until its setter/implementation
discrepancy is resolved.

For multithreshold policies first use five diagonal settings, taking the same
grid index for each criterion. On development data only, take the best eligible
diagonal setting under the rules below, then vary each criterion by one adjacent
grid position while holding others fixed. This prespecified second stage adds at
most `2*k` settings for `k` thresholds. If none qualifies, do not expand the
grid automatically. Compare enabled criteria and AND/OR combinations only as
specified in step 3. `None` and exact zero are separate semantic probes, never
synonyms. Fix native schedules, damping, population sizes, and line searches
unless a candidate explicitly changes that algorithm variant.

Trace replay may screen a threshold only if it affects termination alone.
Preserve observation stages and confirm every finalist with actual solves.
Schedules/inner tolerances that change trajectories need separate runs. Freeze
expanded candidate IDs, formulas, units, grids, and exclusions before
calibration, and finalists before opening validation outcomes.

## Aggregation and policy selection

Within each precision, backend, variant, and derivative/noise stratum, weight
eligible families equally. Within a family weight base models, dimensions,
starts, and transforms equally at each successive level, then average seeds
within cells. Numerous datasets/transforms must not dominate other families.
Keep dimensions and transformations visible; do not pool derivative regimes,
noise levels, roots, nonsmooth objectives, or local/global objectives into one
league table.

Let `T_i(q)` be cumulative `W` at the first published recommendation passing its
joint target, or infinity if never attained, including failures and budgets. An
unavailable, non-finite, or infeasible recommendation has infinite quality error
in fixed-budget summaries; retain its reason rather than dropping it from
quantiles. An indeterminate certificate blocks the affected score until
reviewed. Plot the data profile
`d(q, alpha) = sum_i w_i * 1[T_i(q) / (n_free_i + 1) <= alpha]` for vector runs;
use unnormalized calls for scalar runs. Also report returned-point success,
fixed-budget quality quantiles, and work beyond first attainment
`max(0, W_stop - T_i(q))` for attained targets. A run can attain a target and
later return a worse point; first attainment does not make that returned point a
success. For stochastic methods report success fractions, median/10th/90th
quality quantiles, and `sum_i min(T_i(q), W_stop_i) / number_of_successes` as a
restart-cost diagnostic within identically distributed cells. With no successes
it is infinite. This does not interpret population collapse as global
convergence. Show failure fractions and budgets alongside it.

Report 95% Wilson intervals for repeated-run success fractions. Use 10000 paired
bootstrap resamples for candidate-minus-control differences, resampling families
and paired replicate IDs. Reuse each sampled replicate index across all
transforms of its base case/start to preserve their dependence. Keep starts and
transforms together in each family. Use bootstrap seed `1092026`. Show
family-level intervals, counts, and worst cases as well as aggregates; few
families limit generalization. These margins are engineering requirements fixed
here, not results from the methodological references.

An all-equal paired sample must not produce a claim of zero uncertainty. When
every candidate/control target indicator agrees in a family, replace the
degenerate bootstrap lower bound by `-(1 - 0.05^(1/N))`, where `N` is the number
of independent replicate batches, each containing that family's full case set.
This bounds the probability of any lost success in a future batch,
conservatively bounding its weighted success difference. It is about `-0.095` at
30 batches and `-0.030` at 100. Treat other degenerate interval calculations as
inconclusive until independently checked. The intervals describe individual
comparisons; they are not simultaneous guarantees across the whole solver
catalogue.

Apply these rules on development data, then once to frozen finalists on holdout.
For the 30-seed screening stage only, use point estimates in rule 3 and call
eligibility provisional. Extend the best three provisionally eligible settings
per formula, plus the adjacent settings needed by rule 4, to 100 development
seeds. Resolve ties by fewer criteria, then grid index. Only settings passing
the full interval rules may become finalists. This expansion is fixed in
advance.

1. Reject new false convergence claims on mandatory analytic adversarial cases,
   non-finite/infeasible claimed solutions, and lifecycle/accounting violations.
   Local heuristics retain their documented meaning; population collapse never
   certifies global optimality.
2. For deterministic local/root strata, require at least 95% family-weighted
   designated-target success on independently certified reference-basin cases,
   no lost designated-target successes of the current default on those cases,
   and at most 1% confirmed premature stops. Require no family-level regression
   at the adjacent looser target or in mandatory edge cases. Alternate minima
   and unresolved cases stay in global-target reports. Fix reference-basin
   membership independently before candidate evaluation, never to rescue a
   policy.
3. For stochastic/global policies compare with both current defaults and the
   equal-total-budget control at all three checkpoints. Require the lower 95%
   paired confidence bound on designated-target success difference to be at
   least `-0.05` for each family and the weighted aggregate. The 90th-percentile
   normalized objective error must not exceed twice the control error plus `q`;
   report feasibility separately. If 100 repeats cannot resolve the margin, call
   the result inconclusive, not equivalent. Deterministic global methods use one
   run per fixed case, paired case outcomes, and family resampling; they have no
   invented seed-level uncertainty. Show every family's observed success
   difference and require it to be at least `-0.05` as well.
4. Among eligible policies retain those within 10% of the best capped-work score
   `sum_i w_i * min(T_i(q), B_i) / B_i`. Failures cost the full cap here even
   when they stop early. Prefer fewer independent criteria/thresholds. Require
   an adjacent tested setting that also passes quality gates and has work within
   20%, demonstrating a stable region. Report median/90th-percentile work after
   attainment and every family with over twice the control's work.
5. Change a default only for a documented quality gain or at least 10% less
   capped work/total stopping work without quality regression. Otherwise retain
   the current policy if it passes. If none passes, record the solver unresolved
   or recommend documented budget-driven operation; do not claim the current
   default was validated.

Apply the rules independently to `f32` and `f64`. Verify finalists across every
supported backend version on applicable cases using native scalar evaluations;
casting completed `f64` results to `f32` is not coverage. Diagnose precision or
backend failures before a general recommendation. Conclusions cover only the
recorded applicability strata.

## Freeze, pilot gates, and references

Before sweeps, commit expanded cases, seeds, scales, reference certificates,
eligibility, budgets, candidates, source/lockfile hashes, and measurement schema
as required by the [run conventions](runs/README.md). Pilot analytic fixtures
must validate targets, stages, rejected trials, fused/finite-difference calls,
nested budgets, initialization exhaustion, and returned-point identity. Resolve
every applicable gate in the [step 3
review](review-step3.md#evidence-gaps-and-gates). Pilot performance must not
open the holdout or silently revise this protocol.

Build/reference checks may inspect holdout definitions and known answers, but
may not compare candidate outcomes. If holdout outcomes prompt retuning, mark
whole related families as development data and procure fresh independent
families before claiming validation. Preserve failed holdout results. Record
every deviation, its reason, affected runs, and whether outcomes were observed.

The profile follows the budget-versus-success perspective of [Moré and Wild,
*Benchmarking Derivative-Free Optimization Algorithms*, SIAM J. Optimization
20(1), 172–191 (2009)](https://www.mcs.anl.gov/~wild/papers/2009/JJMSMW07.html).
Our joint targets and leaf-call budget extend that perspective to derivative and
constrained solvers. [COCO performance assessment
(2016)](https://numbbo.github.io/coco-doc/perf-assessment/) motivates work to
predefined targets and retaining unsuccessful runs. Our family weights,
returned-point series, and selection margins are explicit adaptations.

[NIST's regression
background](https://www.itl.nist.gov/div898/strd/nls/nls_info.shtml) describes
best-available solutions checked with multiple implementations and 128-bit
arithmetic, and limitations involving alternate minima, finite differences, and
single precision. Published precision is evidence, not global proof or an
assurance that every `f32` target is attainable. Sources were checked
2026-10-05; preserve source snapshots/hashes with executable certificates.
