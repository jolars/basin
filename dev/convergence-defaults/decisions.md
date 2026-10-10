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
- Limitation at acceptance: the protocol was still a draft. D003 now fixes the
  experimental design; this workflow decision selects no solver defaults.
- Follow-up: retain lasting evidence in permanent repository locations before
  deleting these temporary records in step 8.

## D003: Use CDP-1 for calibration

- Status: accepted experimental design, 2026-10-05; solver policies remain
  unselected.
- Basis: the requested step 4 definition and review, the
  [protocol](protocol.md), [case register](cases.md), and [design
  review](review-step4.md).
- Scope: independent quality measures, precision eligibility, family partitions,
  transforms, derivative/noise strata, budgets, seeds, candidate grid expansion,
  work accounting, aggregation, and selection criteria for all solver families.
- Rationale: a protocol fixed before candidate outcomes makes premature stops,
  wasted work, alternate minima, and failures distinguishable and prevents
  tuning the benchmark to a favored policy.
- Consequence: implement and pilot CDP-1 in step 5, close its gates, and commit
  expanded manifests and reference certificates before calibration. The NIST
  fallback has all 27 public input snapshots and 54 starts; executable model
  adapters and independent numerical verification remain required.
- Alternatives rejected: using native success codes as accuracy labels, pooling
  raw cases without family weights, dropping failures, copying `f64` constants
  into `f32`, or changing targets/budgets after observing candidate performance.
- Limitations: this is an author design review, not independent empirical
  validation. Reliability/work margins are engineering choices; coverage and
  attainable floors must be demonstrated. No numerical run or default change is
  authorized by the completion status alone.
- Follow-up: use G401-G408 in the review. Amend the protocol explicitly if
  pilots expose defects, and consume holdout families if their outcomes inform
  retuning.

## D004: Make the analytic forward-difference setting explicit

- Status: accepted fixture configuration and documentation fix, 2026-10-09; no
  solver default selected.
- Basis: the requested fix following S012's forward-difference finding and the
  [paired validation](runs/2026-10-09-forward-001.md).
- Scope: the unit-scale analytic quadratic, bounded L-BFGS, and forward
  finite-difference gradients. Preserve the strict default as a control.
- Rationale: at the exact minimum, the native `f64` approximate gradient norm is
  `1.49e-8`, above the solver's `1e-10` stopping threshold. Explicit `1e-7`
  stops with convergence after 8 physical calls; the strict control fails its
  line search after 112. The return and the first eight calls are identical.
- Consequence: add an explicit probe option, permanent numerical-gradient
  guidance, a compiling example, and regression coverage for both precisions and
  all supported dense backend versions.
- Alternatives rejected: automatically calling the native failure convergence or
  loosening the general default using this one fixture.
- Limitations: extreme scaling did not support comparable solve accuracy;
  witness bias alone is not a general gradient-error bound. This is no
  calibration or independent holdout evidence.
- Follow-up: establish derivative accuracy and attainable solver thresholds in
  each pilot stratum before calibration, including NIST and LM/TRF variants.

## D005: Screen robust LM progress tests against stationarity

- Status: accepted pilot interpretation and screening restriction, 2026-10-10;
  no solver default or replacement composition selected.
- Basis: the requested next step following S019 and the [336-run stopping
  review](runs/2026-10-10-robust-stopping-001.md), with independently verified
  points and matching default callback prefixes.
- Scope: the six unbounded analytic robust fixtures, four LM routes, and both
  precisions on nalgebra 0.34. Backend regressions separately verify the
  existing gradient control on all dense backends and supported sparse paths.
- Rationale: model-only and step-only tests each reproduce the four severe
  Nielsen/Cauchy stops. Both tests pass on accepted, published trials with gain
  ratios above 0.25, despite large robust gradients. Default continuations reach
  the known minimum. Six other model-reduction stops miss the stricter common
  quality checks on the non-finite-domain fixture.
- Consequence: keep the absolute-gradient default as the control. Do not carry
  the demonstrated progress-only probes into robust calibration as admissible
  candidates without a stationarity or independently validated accuracy guard.
  Document the OR composition and the limits of progress tests. Normalized
  robust gradients remain a separate candidate with zero-residual limitations.
- Alternatives rejected for these controls: merely requiring both model and step
  tests, accepting only published trials, or adding gain ratio above 0.25. All
  would still pass at the same four Cauchy stops. This terminal-predicate
  comparison does not implement or calibrate a new conjunction API.
- Limitations: small development fixtures, unit-scale measurement quality,
  incomplete precision certificates, and no held-out selection. Gradient
  thresholds still require scale and accuracy evidence. Good stalled points
  remain stalled outcomes. The result does not justify changing ordinary
  least-squares semantics or selecting general robust defaults.
- Follow-up: add larger and rank-deficient robust controls and arctangent loss,
  then assess stationarity guards and derivative/backend coverage before
  affected sweeps. Step 5 remains open.

## D006: Extend robust controls and gate Nielsen recovery

- Status: accepted measurement interpretation and recovery gate, 2026-10-10; no
  solver default or guarded progress policy selected.
- Basis: the [504-ablation extension](runs/2026-10-10-robust-extended-001.md),
  216 budget interruptions, and exact reproduction of the original 336-run
  ablation's five CSV files, excluding elapsed time.
- Scope: scalar arctangent and all four built-in losses on four-parameter
  full-rank and rank-deficient models, four LM routes, both precisions, and
  nalgebra 0.34. Independent quality uses distance to the known minimizer set
  under rank deficiency.
- Rationale: 24 combined-relative stops are confirmed premature by matching
  successful default continuations. Of these, 22 satisfy both model and step
  tests on accepted, published trials with gain ratios above 0.25. Progress
  conjunction and these filters remain insufficient on the enlarged controls.
- Recovery evidence: default and normalized-gradient controls each pass 68 of 72
  quality checks. Four Nielsen `f32` runs on rank-deficient Huber and arctangent
  models stall with substantial gradients and rounded-away steps. Trust-region
  damping passes those same fixtures. The stalls are failures, and four poor
  progress returns with poor controls remain unconfirmed.
- Consequence: extend D005's screening restriction to these measured models.
  Investigate the Nielsen recovery gap before calibrating a guarded progress
  policy; rejecting false convergence alone cannot restore a usable step.
- Limitations: linear development controls, unit scales, analytic derivatives,
  and no held-out outcomes or complete precision certificates. QR model checks
  include a projection roundoff screen, not a model-solve accuracy certificate.
  Other backends are tested for observation integrity but not measured here.
- Follow-up: recover the four `f32` stalls, then assess stationarity guards,
  nonlinear robust models, derivatives, bounds/fixed coordinates, backend
  versions, and precision eligibility before affected sweeps. Step 5 stays open.

## Numerical decision template

For each policy decision, record the solver and variants, current and proposed
formulas and defaults for both precisions, exact reference versions, candidate
alternatives, calibration and validation run IDs, family-level and worst-case
outcomes, sensitivity to nearby settings, unresolved limitations, and required
regressions or migration settings. State why the evidence supports changing or
retaining the default. A proposal becomes accepted only after review of that
evidence; implementation and verification remain separate work.

## Open questions

  | ID   | Question                                                                                                                                                                                                   | Resolve in        |
  | ---- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------- |
  | Q001 | Resolved by D003: case register partitions and equal family weights. Executable applicability and missing-case coverage still need G401/G408.                                                              | Steps 5 and 6     |
  | Q002 | D003 fixes quality formulas and target grids. Which targets have independently demonstrated precision floors on each executable case?                                                                      | Step 5, G402/G403 |
  | Q003 | No reporter harness was available in the issue on 2026-10-05. All 27 public datasets and both starts are assembled; executable NIST models and other coverage gaps remain.                                 | Step 5, G401/G404 |
  | Q004 | Which stochastic methods should retain budget-driven termination, and how should their numerical stops be interpreted?                                                                                     | Steps 3, 4, and 6 |
  | Q005 | Resolved by D003: fixed per-class budgets, 30 screening seeds extended to 100 for development finalists, 100 validation seeds, reliability/work margins, and amendment rules.                              | Resolved          |
  | Q006 | Should DIRECT's documented zero radius and volume tolerances use exact-zero checks, or should their setter docs explicitly disallow zero? The current termination code gates both on a positive threshold. | Steps 2 and 7     |
  | Q007 | Should `Backtracking` report `Failed` when all Armijo trials fail? It now returns an untested reduced step, and the default outcome wrapper labels it `Step`.                                              | Steps 2 and 7     |

These questions do not prevent the step 2 inventory. Add concrete findings and
new questions as that audit proceeds.
