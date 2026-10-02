# Decisions and open questions

This log distinguishes agreed project direction from proposed numerical
policies. No solver-specific defaults have been selected.

Use stable decision IDs. Each record states its status, scope, rationale,
supporting sources or run IDs, alternatives, limitations, and follow-up work.
Allowed statuses are proposed, accepted, rejected, and superseded. When evidence
changes a decision, preserve its history and link its successor.

## D001: Cover every solver in Basin

- Status: accepted project scope, 2026-10-01.
- Basis: the user's explicit clarification that the investigation covers every
  solver in Basin and the resulting [TODO
  plan](../../TODO.md#convergence-defaults-investigation).
- Scope: optimization solvers, scalar roots, stochastic and global methods,
  aliases, and composed solvers, including relevant configurable variants.
- Rationale: the issue's original examples provide a useful pilot, but they do
  not establish suitable defaults for the rest of the library.
- Consequence: every solver needs a supported disposition, including decisions
  to retain defaults or rely on execution budgets. Completion is tracked in the
  [inventory](inventory.md).
- Alternative rejected: treating the initial Nelder-Mead, L-BFGS-B, LM, and TRF
  pilot as completion of the investigation.
- Limitation: coverage does not imply that every solver uses the same tests or
  can certify the same quality.

## D002: Preserve evidence between sessions

- Status: accepted investigation workflow, 2026-10-01.
- Basis: the user's request for a branch and temporary development documents,
  the TODO plan, and the repository's [convergence
  contracts](../../CONTRIBUTING.md#design-tenets).
- Scope: work on `convergence-defaults`, keeping small records in this directory
  and bulk output under ignored `target/convergence-defaults/`.
- Rationale: reference policies, experiments, and explicit decision records
  provide a basis for reviewing changes and resuming work across sessions.
- Consequence: preserve 1.x defaults, use solver-specific reference policies as
  candidates, calibrate against independent quality measures, and validate
  before changing 2.0 defaults. Uniformity concerns documented semantics and
  expectations; identical tolerance values do not establish equal accuracy.
- Alternative rejected: selecting defaults solely from a pooled corpus score or
  copying a common tolerance value across solvers.
- Limitation: the protocol is still a draft. This decision selects no
  thresholds, candidates, or numerical acceptance criteria.
- Follow-up: retain lasting evidence in permanent repository locations before
  deleting these temporary records in step 8.

## D003: Adopt the experimental protocol

- Status: proposed, 2026-10-02; maintainer review pending.
- Scope: the [protocol](protocol.md)'s independent quality targets, whole-family
  partition, proposed case strata, resource measurements, paired seeds, and
  selection from the observed accuracy/work tradeoff.
- Basis: the [step 3 evidence gates](review-step3.md#evidence-gaps-and-gates),
  [NIST's public StRD
  files](https://www.itl.nist.gov/div898/strd/nls/nls_main.shtml), and the
  [Misra1a pilot](runs/2026-10-02-misra1a-001.md). The pilot supports measuring
  returned quality independently of termination, but cannot set a default or
  validate the proposed resource limits.
- Proposed choice: accept the measurement definitions and family-level
  partition, use equal-family summaries without application preferences, and
  finalize the target grid, budgets, and numerical effect thresholds from
  development evidence before calibration. D005 records the scope clarification
  and revised resource comparison.
- Alternatives: adjust those proposed values for Basin's intended use cases, or
  retain budget-driven stopping for methods without a justified numerical
  policy. Do not treat a one-case pilot or pooled average as sufficient.
- Limitation: the [coverage matrix](coverage.md) now supplies fixtures and
  baseline entry points, but precision floors, candidate policies, the full
  recorder, and remaining solver wiring are not frozen. Acceptance of this
  protocol would authorize measurement, not any solver default change.

## D004: Broaden the experimental fixtures

- Status: accepted implementation scope, 2026-10-02, following the user's
  request to broaden coverage before the experiments.
- Scope: preserve all 27 NIST source files and add analytic least-squares,
  constrained, scalar, and stochastic controls under `competitor-bench`, with
  native `f32` and `f64` evaluation and two starts or intervals each.
- Evidence: the [coverage matrix](coverage.md), generated case lists, and
  [development pilot](runs/2026-10-02-coverage-002.md) record 57 definitions and
  executable coverage of 14 solver names. Tests check certificates, derivatives,
  reference feasibility, seed reproducibility, and recorder accounting.
- Consequence: keep these diagnostic fixtures outside Basin's public corpus. The
  original Misra1a pilot remains available at its original entry point. The
  expanded NIST runner uses the published unconstrained problems, without the
  first pilot's additional finite bounds.
- Limitation: family partitions remain proposed, validation families have not
  been optimized, and no success targets or solver defaults have been selected.
  D003 and the remaining measurement gates still apply.

## D005: Use broad coverage and distinguish budgets from stopping

- Status: the user's general-purpose scope is confirmed, 2026-10-02. The
  measurement design below remains proposed until the protocol is frozen.
- Basis: the user identifies no preferred application or problem family and
  suggests comparable time allowances while recognizing that more general
  methods may need more work. This removes the need to ask for application
  weights or an arbitrary price for accuracy.
- Proposed measurement: report families separately; use equal-family weights for
  any pooled summary, normalizing related instances within a family. Compare
  stopping policies primarily within the same solver. For cross-solver
  comparisons, use common elapsed-time budget curves and time to common targets,
  alongside physical callback work. Offer longer budgets consistently within
  each comparison rather than granting an automatic multiplier by solver name.
- Evidence: [COCO's performance
  assessment](https://numbbo.github.io/coco-doc/perf-assessment/) distinguishes
  target-based and budget-based measurements and explains its focus on
  evaluation counts. The [protocol](protocol.md) adapts that framework to
  Basin's differing callback and linear-algebra costs. No new experiment ran in
  this session; the existing pilot timings remain unsuitable for speed
  comparisons.
- Consequence: the protocol now asks development pilots to establish concrete
  resource and effect thresholds. A maintainer decision is useful when observed
  candidates present a consequential unresolved tradeoff, rather than as a
  prerequisite to inventing application priorities.
- Limitation: this clarification freezes no time caps, effect thresholds,
  accuracy targets, or solver defaults. The existing measurement and validation
  gates remain.

## D006: Retain timing and reference gates after the policy pilot

- Status: measurement finding, 2026-10-02; no numerical default selected.
- Evidence: [run 003](runs/2026-10-02-policies-003.md) compares three policies
  for four solvers on four development families. Recording modes agree on
  numerical work and returned results, but their charged timings differ. A
  common boundary budget also permits overshoot, which remains visible even when
  the solver stops natively.
- Consequence: do not freeze cross-solver time allowances from the instrumented
  pilot or subtract a single overhead factor. Compare an unrecorded boundary
  driver with ordinary Executor execution, preserving initialization in the
  charged budget. Retain separate physical work and within-budget quality.
- Reference finding: independent high-precision reevaluation explains all
  negative NIST differences within printed certificate rounding. Strict runs
  demonstrate all 42 proposed target combinations for the three NIST cases; this
  is evidence for that subset, not a floor certificate for every family.
- Follow-up: test nearby Nelder–Mead and LM policies, extend reference checks,
  and diagnose persistent `f32` failures/stagnation. Human application weights
  and an arbitrary accuracy/work exchange rate are not prerequisites. D003,
  held-out validation, and the solver-specific gates remain open.

## Numerical decision template

For each policy decision, record the solver and variants, current and proposed
formulas and defaults for both precisions, exact reference versions, candidate
alternatives, calibration and validation run IDs, family-level and worst-case
outcomes, sensitivity to nearby settings, unresolved limitations, and required
regressions or migration settings. State why the evidence supports changing or
retaining the default. A proposal becomes accepted only after review of that
evidence; implementation and verification remain separate work.

## Open questions

  | ID   | Question                                                                                                                                                                                                                                                              | Resolve in        |
  | ---- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------- |
  | Q001 | No application preference is intended. Verify broad solver applicability and the proposed equal-family summaries; see D005.                                                                                                                                           | Steps 2 and 4     |
  | Q002 | Which external accuracy targets and precision floors are attainable and useful for each solver family?                                                                                                                                                                | Steps 3 and 4     |
  | Q003 | The reporter's harness has not been obtained; NIST publishes all 27 cases. Which additional cases are needed for full solver coverage?                                                                                                                                | Steps 4 and 5     |
  | Q004 | Which stochastic methods should retain budget-driven termination, and how should their numerical stops be interpreted?                                                                                                                                                | Steps 3, 4, and 6 |
  | Q005 | Which common budget ranges, repetition counts, and effect thresholds are justified by development pilots? See D005.                                                                                                                                                   | Step 4            |
  | Q006 | Should DIRECT's documented zero radius and volume tolerances use exact-zero checks, or should their setter docs explicitly disallow zero? The current termination code gates both on a positive threshold.                                                            | Steps 2 and 7     |
  | Q007 | Resolved in the development branch: outcome-aware `Backtracking` now reports `Failed` on exhausted Armijo trials, with unit and gradient-descent regression tests. The legacy `next` method retains its step-only fallback. Verify other owning solvers' propagation. | Step 5            |

These questions do not prevent the step 2 inventory. Add concrete findings and
new questions as that audit proceeds.
