# Convergence defaults investigation

This directory holds temporary development records for the [Basin 2.0
convergence plan](../../TODO.md#convergence-defaults-investigation) and [issue
#109](https://github.com/jolars/basin/issues/109). The scope is every solver in
Basin, including scalar roots, stochastic methods, aliases, and composed
solvers. A decision to keep existing defaults still needs evidence.

The development branch is `convergence-defaults`, created from `main` at
`53dc844f003353cb79a9caad6e2b72445a770ca5`. Preserve Basin 1.x defaults. These
records do not replace the repository's [agent guide](../../AGENTS.md) or
[design contracts](../../CONTRIBUTING.md).

## Status

Step 1 is complete. Step 2 has a first source pass for all 47 public solver
names and key dependencies, with variant and work-accounting checks remaining.
Step 3 has a reference survey and draft candidate policies for all 47 public
solver names. The [candidate and evidence review](review-step3.md) records the
remaining gaps and gates before large sweeps. Step 4 now has the reviewed [CDP-1
protocol](protocol.md), [case partitions](cases.md), and all 27 NIST input
datasets with both starts. The [step 4 review](review-step4.md) records the
implementation and pilot gates. Step 5 has an initial [analytic measurement
harness](harness.md) and a [nine-run validation
record](runs/2026-10-09-analytic-001.md), with verifier/accounting checks in
both native precisions. A [forward-difference configuration
fix](runs/2026-10-09-forward-001.md) now covers the analytic stopping mismatch.
The [NIST analytic adapters](nist-models.md) now pass independent model
validation in both precisions. The [development reference
preflight](runs/2026-10-09-nist-reference-001.md) certifies 15 local references
and 321 of 330 analytic target combinations. Nine `f32` targets remain pending,
as do finite-difference eligibility and holdout references. The [LM/TRF
measurement pilot](runs/2026-10-09-least-squares-001.md) adds opt-in native
diagnostics and 360 measured development NIST solves across six routes in both
precisions. Remaining bound and loss variants, other backends, complete inner
work, and other families remain open. The [bounded full-TRF analytic
pilot](runs/2026-10-09-bounded-trf-001.md) adds active bounds, mixed and
all-fixed coordinates, budget interruptions, and a stationary nonminimum
control. The [robust-loss LM/TRF pilot](runs/2026-10-09-robust-ls-001.md) adds
352 analytic measurements with Huber, soft-L1, and Cauchy losses, independent
robust model checks, and paired default continuations. No solver policies have
been selected and calibration has not started. The [legacy TRF safeguard
recheck](runs/2026-10-09-trf-finite-model-001.md) eliminates repeated non-finite
model predictions without changing any final published point or cost; main and
1.x fixes are proposed in draft PRs. The [robust LM stopping
review](runs/2026-10-10-robust-stopping-001.md) adds 336 ablations and
reproduces the 352-case robust baseline. Both gradient configurations pass
common point quality on all 48 cases; progress-only settings have confirmed
premature stops. Pairing model and step tests or requiring accepted trials and
good model agreement does not prevent the four severe Cauchy stops. Public
guidance and backend regressions cover the existing gradient control; broad
policy selection and remaining robust coverage stay open. The [larger robust
controls](runs/2026-10-10-robust-extended-001.md) add all four built-in losses
on full-rank and rank-deficient four-parameter models and scalar arctangent.
Their 504 ablations confirm 24 premature combined-relative stops; 216 exact
budget interruptions pass. Gradient controls expose four inaccurate `f32`
Nielsen stalls, retaining a recovery gate before stationarity-guard calibration.

  | Step | Deliverable                                   | Status      |
  | ---- | --------------------------------------------- | ----------- |
  | 1    | Branch and session records                    | Complete    |
  | 2    | Inventory of stopping behavior and variants   | In progress |
  | 3    | Reference survey and candidate policies       | Complete    |
  | 4    | Reviewed experimental protocol                | Complete    |
  | 5    | Measurement harness and pilot                 | In progress |
  | 6    | Calibration and independent validation        | Pending     |
  | 7    | Implementation and verification               | Pending     |
  | 8    | Complete coverage and permanent documentation | Pending     |

## Records

- [Inventory](inventory.md): initial coverage, audit fields, and completion
  requirements for each solver and relevant variant.
- [Dependencies](dependencies.md): source audit of line searches, bracketers,
  and inner model stops.
- [Reference pilot](reference-pilot.md): first versioned stopping-formula
  comparison for Nelder-Mead, bounded L-BFGS, LM, and TRF, including LM and TRF
  observation stages.
- [Pilot candidates](candidate-pilot.md): testable policy alternatives and
  variant distinctions for the four pilot families; no selected defaults.
- [First-order references and candidates](reference-first-order.md): versioned
  SciPy, Ceres, and NLopt comparisons and testable policies for first-order,
  Newton, Gauss–Newton, and SLSQP names; named evidence gaps remain.
- [Scalar references and candidates](reference-scalar.md): versioned SciPy
  comparisons and testable bracket, position, and derivative policies for all
  three scalar minimizers and five root solvers.
- [Derivative-free local references and
  candidates](reference-derivative-free.md): versioned PDFO, PRIMA, SciPy,
  NLopt, NOMAD, and Solis–Wets comparisons for Powell-model methods, MADS, and
  Solis–Wets.
- [Global and population references and candidates](reference-global.md):
  versioned SciPy and pycma comparisons, primary research, and explicit
  budget-driven controls for all nine names.
- [Constrained and composed references and candidates](reference-composed.md):
  log-barrier and augmented-Lagrangian checks, basin hopping, injections, and
  local-search chains with separate inner and outer stop owners.
- [Step 3 review](review-step3.md): 47-name coverage, candidate interpretation,
  evidence gaps, and gates before large sweeps.
- [Protocol](protocol.md): CDP-1 quality measures, precision eligibility,
  budgets, seeds, candidate grids, aggregation, and selection rules.
- [Case register](cases.md): corpus/NIST families, held-out partitions,
  transformations, and explicit supply gates for missing cases and capabilities.
- [NIST inputs](nist/manifest.json): 27 source snapshots, both starts, reference
  strings, source/snapshot hashes, and family partitions; analytic model
  adapters validated in both precisions.
- [NIST reference preflight](nist-reference.md): independent local reference
  refinement, interval certificates, native witnesses, and frozen analytic
  eligibility for development families.
- [Step 4 review](review-step4.md): design checks, corrections, and gates before
  the harness, calibration, and independent validation.
- [Analytic harness](harness.md): measurement API, CSV schema, analytic
  fixtures, commands, and remaining diagnostic/coverage limitations.
- [Least-squares pilot](least-squares-pilot.md): six native routes, trial
  diagnostics, callback accounting, independent point quality, and reproducible
  analytic and development NIST measurements.
- [Robust-loss pilot](runs/2026-10-09-robust-ls-001.md): analytic robust
  objectives, gradients, models, and default continuations across LM/TRF.
- [Robust LM stopping review](runs/2026-10-10-robust-stopping-001.md): isolated
  predicates, matched gradient continuations, and limits of progress
  conjunctions and model-agreement filters.
- [Larger robust controls](runs/2026-10-10-robust-extended-001.md): all built-in
  losses, identifiable parameter quality under rank deficiency, and a measured
  Nielsen recovery gap in `f32`.
- [Decisions](decisions.md): agreed direction, proposals, evidence, and open
  questions. Numerical decisions remain pending.
- [Run records](runs/README.md): conventions for tracked manifests and concise
  results. Bulk traces belong in `target/convergence-defaults/`, which the
  repository already ignores.

Keep these documents and small reproduction inputs in Git. Before removing this
directory in step 8, move the reusable harness, regression cases, manifests, and
lasting decision rationale to permanent repository locations.

## Session handoffs

At the start of a session, read this status, the latest handoff, the inventory,
and the applicable decisions. Check the current branch, revision, and working
tree before editing. Review the protocol before interpreting experiment data.

Append a handoff at the end of each session with its starting revision and
working-tree state, scope, exact commands, evidence or run IDs, decisions,
validation, unresolved questions, and next concrete task. Update the status
table and the TODO checkboxes only when their completion criteria are met. Link
results to evidence and keep proposals distinct from accepted decisions. If
later evidence contradicts a decision, supersede its record rather than silently
rewriting the rationale.

### S001: Establish the investigation, 2026-10-01

- Starting revision: `53dc844f003353cb79a9caad6e2b72445a770ca5` on `main`
  (`docs: outline convergence criteria plan`).
- Starting working tree: `TODO.md` contained the previous planning session's
  uncommitted formatting and wording refinements. They were carried onto the
  development branch.
- Scope: create the branch, seed these records, and mark step 1 complete. The
  inventory is a scope seed, not a completed audit.
- Evidence: the plan lists 47 public solver names. Implementation paths,
  configurable variants, formulas, defaults, and catalogue reconciliation still
  need the step 2 audit. The existing `/target` ignore rule covers the proposed
  raw-output directory.
- Decisions: recorded the agreed scope and workflow as D001 and D002. No
  numerical defaults or experimental target values were chosen.
- Validation: formatting and lint pass for all five new documents. A local link
  check resolved all 23 file links and anchors. The coverage seed matches all 47
  publicly re-exported solver names; this does not replace the behavior and
  public-submodule audit. The edited TODO item passes formatting, and
  `git diff --check` passes. No Rust tests apply to this documentation setup.
- Open questions: Q001-Q005 in [decisions.md](decisions.md).
- Next task: reconcile the scope seed against `lib.rs`, the public `solver` and
  `root` modules, and the web catalogue. Create a stopping-behavior record for
  each solver and variant, starting with the issue's Nelder-Mead, L-BFGS-B, LM,
  and TRF cases. Keep step 2 open until every solver is audited.

Commands run from the repository root:

```sh
git status --short
git branch --show-current
git log -1 --format='%H%n%s'
git branch --list main convergence-defaults
git worktree list
git diff -- TODO.md
git switch -c convergence-defaults main
git check-ignore -v target/convergence-defaults/.probe
panache format dev/convergence-defaults
panache format --check dev/convergence-defaults
panache lint dev/convergence-defaults
git diff --check
git add -- TODO.md dev/convergence-defaults
git diff --cached --check
git commit -m "docs: start convergence defaults investigation" -m "Refs #109"
```

An ad hoc Python check verified local link targets and heading anchors and
compared the seed's solver names with the public `solver` and `root` exports,
excluding strategy, configuration, and result types. It did not audit solver
implementations. The setup checkpoint is the Git commit titled
`docs: start convergence defaults investigation` that introduces this log.

### S002: Reconcile public solvers and begin the stop audit, 2026-10-01

- Starting revision: `3ae7be6977334cd50554217c4f3f013f4cb6bf1c` on
  `convergence-defaults`; the working tree was clean.
- Scope: step 2 only. Compared the 47-name seed against the crate-root,
  `solver`, and `root` exports and the web catalogue. Inspected source for type
  aliases and configurable modes, then recorded initial native stopping behavior
  for Nelder–Mead, L-BFGS-B/unbounded L-BFGS, both LM factorizations and damping
  modes, legacy `Trf`, full `TrustRegionReflective`, three scalar minimizers,
  and five scalar root solvers.
- Evidence: [inventory.md](inventory.md#public-api-reconciliation) records the
  public-surface comparison, [variant register](inventory.md#variant-register),
  [initial stopping records](inventory.md#initial-stopping-records), [scalar
  stopping records](inventory.md#scalar-stopping-records), and a [47-name
  evidence tracker](inventory.md#per-solver-evidence-tracker). The source links
  there identify the implementations. `Lbfgsb` is the bounded `Lbfgs` alias; the
  two LM damping modes and the two TRF implementations need separate records
  despite similar tolerance names.
- Decisions: none. Every candidate policy, experiment, and solver disposition
  remains pending. This source audit does not justify changing defaults.
- Validation: an ad hoc Python name-set check matched all 47 seed names to the
  catalogue and crate-root exports, all five root names to the `root` exports,
  and the 42 optimization names to 40 direct `solver` re-exports plus the two
  public `solver::lbfgs` names.
  `panache format --check dev/convergence-defaults`,
  `panache lint dev/convergence-defaults`, and `git diff --check` pass. No Rust
  code changed and no numerical test ran.
- Open questions: complete the explicit native and shared opt-in stop audit for
  the initial records; then inspect the remaining solvers and their modes, line
  searches, bracket searches, and subproblem completion rules. Reconcile names
  again at the final revision.
- Next task: build one source-backed record per remaining solver and relevant
  mode, including defaults, formula, observation stage, and the separation of
  convergence from safeguards and execution budgets. Keep the TODO boxes open
  until decisions and validation exist for every applicable variant.

### S003: Audit first-order and derivative-free stops, 2026-10-01

- Starting revision: `7bb53a2b5763b497b23e1eaee503b6c2ec7f82f0` on
  `convergence-defaults`; the working tree was clean.
- Scope: source audit of `GradientDescent`, `ProjectedGradientDescent`,
  `NonlinearCg`, `Bfgs`, `TrustRegion`, `Sgd`, and `GaussNewton`, followed by
  `Newuoa`, `Bobyqa`, `Lincoa`, `Cobyla`, all three `Mads` modes, and
  `SolisWets`. Inspected stopping-relevant updates, trust-region modes,
  objective-refresh choices, and radius or poll-size schedules.
- Evidence: [first-order and Gauss-Newton
  records](inventory.md#first-order-and-gauss-newton-stopping-records) and
  [derivative-free local
  records](inventory.md#derivative-free-local-stopping-records) link to each
  defining source and the shared configured-check builders. The per-solver
  tracker distinguishes this source observation from the pending reference,
  candidate, experiment, decision, implementation, and verification stages.
- Decisions: none; all numerical dispositions remain pending.
- Validation: `panache format --check dev/convergence-defaults`,
  `panache lint dev/convergence-defaults`, and `git diff --check` pass. An ad
  hoc local Markdown link and heading-anchor check passes. No Rust code or
  numerical behavior changed.
- Open questions: audit the inner trust-region strategies and line searches
  separately, especially failure propagation, then continue through the
  remaining constrained, global, and composed solvers.
- Next task: record source-backed stops for the remaining solvers and build
  explicit dependency records for line searches, bracketers, and subproblems.

### S004: Audit global and population stops, 2026-10-01

- Starting revision: `aac2c51` on `convergence-defaults`; the working tree was
  clean.
- Scope: inspect the native stopping behavior and stopping-relevant variants of
  `Direct`, `Gbnm`, `RandomSearch`, `SimulatedAnnealing`, `CmaEs`,
  `BoundedCmaEs`, `De`, `GlobalBestPso`, and `Ssga`.
- Evidence: [global and population stopping
  records](inventory.md#global-and-population-stopping-records) distinguish
  solver stops from local restarts, generation representatives, distribution
  collapse, and executor budgets. The per-solver tracker links the nine names to
  these source observations.
- Finding: DIRECT's zero-threshold setters promise exact-zero checks, but the
  current termination code requires positive thresholds. This is tracked as Q006
  in [decisions.md](decisions.md#open-questions), pending a test-first
  behavioral fix or documented contract change.
- Decisions: none. The source findings do not yet select or retain defaults.
- Validation: `panache format --check dev/convergence-defaults`,
  `panache lint dev/convergence-defaults`, and `git diff --check` pass. An ad
  hoc local Markdown link and heading-anchor check passes. No Rust code or
  numerical behavior changed.
- Open questions: complete the constrained and composed solver source audit,
  then audit line searches, bracketers, and inner subproblem completion.
- Next task: record the remaining ten public solver names and their inner-stop
  interactions, then reconcile the full source audit before reference review.

### S005: Complete the first source pass, 2026-10-01

- Starting revision: `2b543efadde2d9169bae383c9c77fbd950cfe30f` on
  `convergence-defaults`; the working tree was clean. This session continued
  after an interruption with the constrained and composed records uncommitted.
- Scope: inspect the remaining ten solver names and record their outer and inner
  stops. Audit public line searches, root and minimum bracketers, and
  trust-region, LM, and SLSQP subproblem completion as dependencies.
- Evidence: all 47 names have a source-checked entry in the
  [tracker](inventory.md#per-solver-evidence-tracker). The [constrained and
  composed records](inventory.md#constrained-and-composed-stopping-records) and
  [dependency audit](dependencies.md) identify the distinct stop owners,
  defaults, and budget or failure paths. This is a source pass, not a reference
  comparison or empirical validation.
- Finding: `Backtracking` exhausts its Armijo trials by returning a further
  reduced, untested step; the default outcome wrapper labels it `Step`. Q007
  records the needed disposition. The existing DIRECT mismatch remains Q006.
- Decisions: none. Candidate policies, experiments, solver dispositions,
  implementation, and verification remain pending. No TODO coverage box is
  complete.
- Validation: `panache format --check dev/convergence-defaults`,
  `panache lint dev/convergence-defaults`, and `git diff --check` pass. An ad
  hoc local file and heading-anchor check passes, and the tracker has 47 rows
  with no pending current-stop entry. No Rust behavior changed.
- Open questions: Q001-Q007 in [decisions.md](decisions.md#open-questions).
- Next task: reconcile the first source pass against current exports and the web
  catalogue, then begin versioned reference comparisons and candidate records
  for the initial Nelder-Mead, L-BFGS-B, LM, and TRF pilot. Expand the
  subproblem-work and composed outcome audits before designing measurements.

### S006: Start versioned reference comparisons, 2026-10-01

- Starting revision: `41f9ca5` on `convergence-defaults`; the working tree was
  clean.
- Scope: compare the first four pilot families against SciPy 1.16.2's public
  documentation and tagged source. Keep similarly named tolerances distinct.
- Evidence: [reference-pilot.md](reference-pilot.md) records SciPy's Nelder-Mead
  two-spread rule, L-BFGS-B projected gradient and relative cost rules, and
  method-specific LM and TRF criteria beside Basin's current rules. Each
  external claim links to versioned primary documentation or source.
- Decisions: none. The values are reference candidates, not accepted defaults;
  precise optional-check comparison, independent quality targets, `f32`
  calibration, and validation remain pending. Tracker reference fields remain
  pending until each solver's comparison is complete.
- Validation: `panache format --check dev/convergence-defaults`,
  `panache lint dev/convergence-defaults`, and `git diff --check` pass. Local
  file and heading-anchor links resolve. No Rust code changed.
- Next task: compare SciPy's tagged LM and TRF implementation stages and
  MINPACK's original tests with Basin's optional checks. Then record candidate
  formulas for each pilot variant and design scaled, held-out tests.

### S007: Specify pilot stopping candidates, 2026-10-01

- Starting revision: `b415db7611678493bc2a3905229ee9143aa1c10a` on
  `convergence-defaults`; the working tree was clean.
- Scope: trace SciPy 1.16.2's LM wrapper and dense TRF trial loop, MINPACK's
  original `lmder`, and Basin's native and shared optional checks. Draft
  candidate policies for all four pilot families and their relevant modes.
- Evidence:
  [reference-pilot.md](reference-pilot.md#lm-observation-stages-and-minpack-comparison)
  now records the exact LM and TRF observation stages and mismatches;
  [candidate-pilot.md](candidate-pilot.md) states formulas, composition,
  reference anchors, and variant boundaries. The per-solver tracker remains
  pending until each candidate has a complete reference and precision review.
- Findings: Basin's ordinary LM model-reduction test closely matches MINPACK's
  normalized three-part test, but acceptance and radius histories differ. The
  unscaled LM step check is not MINPACK's `xtol`. SciPy TRF checks cost and step
  on a trial before acceptance; Basin's shared observed checks only see accepted
  states and use different formulas. A SciPy-like TRF policy would require a
  native trial-stage implementation.
- Decisions: none. External `f64` defaults are candidate anchors, not selected
  Basin settings; `f32` thresholds and experiment grids are unchosen.
- Validation: documentation formatting, lint, links, and `git diff --check` are
  recorded below. No Rust code or numerical behavior changed.
- Open questions: original L-BFGS-B cost and acceptance details, an
  unconstrained L-BFGS reference, robust-loss candidates, independent targets,
  and the remaining solver families.
- Next task: finish the L-BFGS-B and unconstrained reference comparison; specify
  and review the experimental protocol before calibration.

Commands run from the repository root:

```sh
panache format dev/convergence-defaults
panache format --check dev/convergence-defaults
panache lint dev/convergence-defaults
git diff --check
```

### S008: Extend the reference survey, 2026-10-01

- Starting revision: `4f3655093e67771b2fe7855e02605e7e76f1c88a` on
  `convergence-defaults`; the working tree was clean.
- Scope: step 3 reference and candidate pass for `GradientDescent`,
  `ProjectedGradientDescent`, `NonlinearCg`, `Bfgs`, `TrustRegion`, `Sgd`,
  `GaussNewton`, `Slsqp`, the three scalar minimizers, and all five root
  solvers.
- Evidence: [reference-first-order.md](reference-first-order.md) records
  versioned SciPy 1.16.2, Ceres 2.2.0, and NLopt 2.10.0 references beside
  Basin's current source audit. It specifies observation stage, norm,
  composition, default, precision caveat, and distinct no-progress paths where
  the reference establishes them. The candidate table keeps variants separate.
  [reference-scalar.md](reference-scalar.md) compares SciPy's scalar minimum and
  root rules with Basin's bracket tests, including the difference between
  SciPy's unbracketed Newton family and Basin's bracketed variants.
- Finding: SciPy's default BFGS and CG gradient threshold uses the infinity
  norm; Basin's optional shared gradient threshold uses the Euclidean norm.
  SciPy's trust-region gradient anchor is also algorithm-specific. Ceres'
  projected-gradient rule provides a closer mathematical reference for Basin's
  projected method than an ordinary gradient threshold.
- Decisions: none. No threshold, candidate, or solver disposition is accepted.
  The per-solver tracker remains pending until the named evidence gaps and
  precision review are resolved.
- Validation: documentation formatting, lint, local links, and diff checks are
  recorded in this session's command output. No Rust behavior changed.
- Open questions: source-level SciPy/NLopt SLSQP branch composition, Hager–Zhang
  and projected-line-search reference behavior, SGD noise-aware stopping, scalar
  `f32` floors, and the still unsurveyed derivative-free, population, and
  composed families.
- Next task: survey versioned primary references and specify candidates for
  Powell-model derivative-free methods, then global and population methods.

Commands run from the repository root:

```sh
git status --short --branch
git log -1 --format='%H%n%s'
panache format dev/convergence-defaults TODO.md
panache format dev/convergence-defaults/reference-scalar.md
panache format --check dev/convergence-defaults TODO.md
panache lint dev/convergence-defaults TODO.md
git diff --check
```

The reference pass also opened the tagged SciPy 1.16.2, Ceres 2.2.0, and NLopt
2.10.0 sources linked from the new records. A local Markdown file-link check
resolved the new links. No corpus sweep or numerical test ran.

### S009: Survey derivative-free local stops, 2026-10-01

- Starting revision: `8982cbb` on `convergence-defaults`; the working tree was
  clean.
- Scope: step 3 reference and candidate pass for `Newuoa`, `Bobyqa`, `Lincoa`,
  `Cobyla`, `Mads`, and `SolisWets`, with constrained modes treated separately.
- Evidence: [reference-derivative-free.md](reference-derivative-free.md)
  compares Basin with PDFO 1.3, PRIMA v0.7.1 source, SciPy 1.16.2 COBYLA, NLopt
  2.10.0 documentation, NOMAD 4.5.0, and the Solis–Wets paper. It distinguishes
  a radius schedule's final stage from an optional early radius test, and a
  small search scale from returned-point feasibility.
- Finding: SciPy's COBYLA wrapper requires both an acceptable constraint
  violation and a radius or target code for success. Basin's current COBYLA
  completion reads only its final-radius schedule. NOMAD distinguishes mesh and
  frame sizes, while Basin's MADS has one scalar poll-size floor.
- Decisions: none. Candidate thresholds and solver dispositions remain
  unselected; the step 3 checklist is open.
- Validation: documentation formatting and lint, local links, and Git diff
  checks are recorded below. No Rust code or numerical experiment changed.
- Open questions: exact PRIMA/Basin model-path differences, constrained
  violation scales and reporting, Solis–Wets finite-run reference details, and
  all remaining global, population, and composed methods.
- Next task: survey versioned primary sources for global and population methods,
  then composed methods and the remaining constrained adapters.

Commands run from the repository root:

```sh
git status --short --branch
git log -5 --oneline
panache format dev/convergence-defaults/reference-derivative-free.md
panache lint dev/convergence-defaults/reference-derivative-free.md
git diff --check
```

### S010: Finish the reference and candidate survey, 2026-10-01

- Starting revision: `cb8cd60` on `convergence-defaults`; the working tree was
  clean.
- Scope: step 3 reference and candidate passes for the nine global and
  population names, two remaining constrained adapters, and seven composed
  names. Reviewed candidate coverage and evidence gaps across all 47 public
  names and marked step 3 complete.
- Evidence: [global reference policies](reference-global.md) compare Basin's
  DIRECT geometry with SciPy 1.16.2, CMA distribution TolX with pycma r4.3.0,
  and DE population-energy spread with SciPy 1.16.2; primary research anchors
  the budget-driven stochastic controls. [Constrained and composed
  policies](reference-composed.md) compare barrier and augmented-Lagrangian
  checks, SciPy basin hopping, injection inheritance, and local-search chain
  segments. The [step 3 review](review-step3.md) enumerates 47 names, candidate
  meanings, and pre-sweep gates.
- Findings: external thresholds with equal names are often different
  measurements. In particular, SciPy's default DIRECT_L length differs from
  Basin's original DIRECT half-diagonal; pycma TolX includes coordinate and
  evolution-path checks while Basin has one principal-axis check; SciPy DE
  observes full population objective spread while Basin's outer default is
  budget-driven. Inner completion does not terminate a composed outer solve.
- Decisions: the survey and draft candidate set are complete. No candidate
  default or threshold was selected; the review blocks large sweeps until the
  protocol resolves its stated quality, precision, and work-accounting gates.
- Validation: formatting, lint, local Markdown links, a 47-name coverage check,
  and `git diff --check` pass. No Rust code or numerical behavior changed.
- Open questions: the high-priority and follow-up gaps in
  [review-step3.md](review-step3.md#evidence-gaps-and-gates), including exact
  L-BFGS-B and SLSQP reference branches, constrained stationarity diagnostics,
  and DIRECT's zero-threshold discrepancy.
- Next task: finish and review the [experimental protocol](protocol.md) with
  independent success, precision floors, diagnostic stages, seed pairing, and
  inner/outer evaluation accounting. Resolve the gate items with focused checks
  before any large sweep.

Commands run from the repository root:

```sh
git status --short --branch
git log -3 --oneline
panache format dev/convergence-defaults TODO.md
panache format --check dev/convergence-defaults TODO.md
panache lint dev/convergence-defaults TODO.md
git diff --check
```

### S011: Define and review the protocol, 2026-10-05

- Starting revision: `06a8d4fd47ef93eadb515b1a4f395bc561e43120` on
  `convergence-defaults`; the working tree was clean.
- Scope: finish step 4 with CDP-1, fixed corpus/NIST family partitions,
  independent target formulas, precision eligibility, work budgets, seeds,
  candidate grids, selection rules, and an author design review. Update run
  conventions and mark step 4 complete while preserving downstream gates.
- Evidence: [protocol](protocol.md), [case register](cases.md), [NIST
  manifest](nist/manifest.json), and [review](review-step4.md). Checked
  Moré–Wild, COCO, NIST source documentation, issue #109, all 28 corpus specs,
  and the existing trace and LM probe. No reporter harness was available in the
  visible issue; independently assembled all 27 NIST inputs and 54 starts.
- Findings: native codes cannot substitute for solution quality; offsets must
  not enlarge accuracy allowances; alternate local minima need separate
  evidence; exact-root exits can retain wide brackets; changed continuation
  trajectories do not establish premature termination; most corpus wrappers
  still need native `f32` coverage. Public NIST files require
  objective-convention and rounding checks before executable comparisons.
- Decision: D003 accepts the experimental design. No solver default, reference
  certificate, attainable floor, or numerical outcome was selected or measured.
- Validation: Markdown formatting/lint, local file/anchor checks, NIST source
  and snapshot hashes, numeric row/start counts, family membership, analytic
  design checks, and Git whitespace checks pass. Downloaded data contain 2176
  observations. No Rust code changed and no solver experiment ran.
- Open work: G401-G408 in the review, remaining step 2 paths, and applicable
  step 3 source/diagnostic gates. The protocol is complete; executable
  manifests, model adapters, independent certificates, and harness validation
  are not.
- Next task: implement the step 5 verifier and work ledger on analytic fixtures,
  validate NIST models and both starts, then pilot Nelder–Mead, L-BFGS-B, both
  LM factorizations/damping modes, and legacy/full TRF. Freeze expanded
  manifests before calibration and keep candidate holdout outcomes sealed.

Commands run from the repository root:

```sh
git status --short
git branch --show-current
git log -1 --format='%H%n%s'
gh issue view 109 --repo jolars/basin --json body,comments,url
panache format dev/convergence-defaults TODO.md
panache format --check dev/convergence-defaults TODO.md
panache lint dev/convergence-defaults TODO.md
git diff --check
```

The NIST download used
`curl --silent --show-error --fail --location --max-time 30 --user-agent 'Mozilla/5.0'`
against each exact manifest URL. Python's default HTTP client received HTTP 403,
so no input was taken from that failed attempt. Source bytes were downloaded
under ignored `target/convergence-defaults/`, validated, and normalized to LF
with trailing whitespace removed. The manifest retains both byte hashes and
unrounded decimal strings. Ad hoc Python checks reparsed all snapshots,
reconciled corpus families and NIST names, checked local links/anchors, and
exercised the protocol's analytic examples and accounting formulas. These checks
establish input/design consistency, not solver behavior.

### S012: Validate the analytic measurement layer, 2026-10-09

- Starting revision: `0ea3a6d566580f3e2adcef9eb9608b5f00c84b3c` on
  `convergence-defaults`; the working tree was clean.
- Scope: first step 5 implementation in `competitor-bench`: physical leaf-call
  ledger with a shared exact cap, smooth/box and scalar-root quality predicates,
  published-point measurement runner, native-scalar quadratic fixtures, CSV
  output, and independent artifact checks. Production solvers and defaults did
  not change.
- Evidence: [harness](harness.md), 16 integration tests, and
  [2026-10-09-analytic-001](runs/2026-10-09-analytic-001.md). Source was
  committed at `35128808152050be4dd12dd034f231fe65f39328`, and the planned
  manifest/checker at `c11acfe952564a520ec18f8bef08b2e8ed88a7d6`, before the
  recorded release execution. All nine runs and 19 CSV files pass schema, work,
  stage, quality, and authoritative-count reconciliation checks. Output hashes
  are tracked.
- Findings: fused analytic L-BFGS-B uses two physical calls for four logical
  categories; finite differences expand gradient requests into individual
  counted probes. Forward-difference L-BFGS-B returns the exact witness with a
  line-search failure code. Default NM reaches the designated target at 71
  calls, then continues to its 6000-call physical cap; typed interruption leaves
  a passing prior recommendation but no final returned checkpoint. Caps at 2 and
  4 correctly distinguish incomplete initialization from interrupted steps.
- Decisions: no solver policy selected and no protocol amended. The first
  measurement layer is validated on analytic fixtures only. Step 5 remains in
  progress, and G401-G408 remain open for their full scope.
- Validation: all 16 tests pass with default features and with
  `parallel,basin-latest`; workspace all-target/all-feature clippy passes with
  warnings denied; rustfmt, documentation formatting/lint, manifest/hash, local
  link, and whitespace checks pass. The run checker independently parses the
  emitted artifacts and confirms no physical overshoot and no solve charge for
  verification. Every published point has zero bound violation.
- Development corrections: the initial native-f32 test omitted
  `BoundedFiniteDiff`'s base evaluation; its expected physical total was
  corrected from 6 to 7 after checking the adapter. The CSV summary placeholder
  count and clippy findings were corrected before committing source. Initial
  shared-target builds hit NLopt CMake compiler-cache conflicts with background
  checks; the isolated `target/convergence-defaults/build` directory resolved
  them. Those failed builds are not numerical runs.
- Open work: NIST residual models/Jacobians and published RSS checks; native
  LM/TRF rejected-trial operands and acceptance diagnostics; general constrained
  KKT certificates; precision eligibility beyond this exact analytic fixture;
  all backend versions; remaining solver/variant and composed accounting paths;
  process-level wall timeouts for broad sweeps. Passing native report evidence
  is preserved; unavailable failing-clause data is not inferred.
- Next task: validate executable NIST adapters against all observations, both
  starts, analytic Jacobians, and published reference rounding, then extend the
  pilot through both LM factorizations/damping modes and legacy/full TRF. Keep
  holdout candidate outcomes sealed and freeze expanded cases before
  calibration.

Commands run from the repository root after source corrections:

```sh
cargo fmt --all -- --check
CARGO_TARGET_DIR=target/convergence-defaults/build \
  cargo test -p competitor-bench --test convergence_measurement
CARGO_TARGET_DIR=target/convergence-defaults/build \
  cargo test -p competitor-bench --test convergence_measurement \
  --features parallel,basin-latest
CARGO_TARGET_DIR=target/convergence-defaults/build \
  cargo clippy --workspace --all-targets --all-features -- -D warnings
CARGO_TARGET_DIR=target/convergence-defaults/build \
  cargo build -p competitor-bench --release --bin verify_convergence
RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  target/convergence-defaults/build/release/verify_convergence \
  --output-dir target/convergence-defaults/2026-10-09-analytic-001
python dev/convergence-defaults/check-analytic.py \
  target/convergence-defaults/2026-10-09-analytic-001
panache format dev/convergence-defaults TODO.md
panache format --check dev/convergence-defaults TODO.md
panache lint dev/convergence-defaults TODO.md
git diff --check
```

The manifest/hash and local link checks use the tracked file sets and formulas
described in the run record. Small ad hoc Python checks confirmed TOML parsing,
all source/output SHA-256 values, and zero published bound violations. No
candidate performance on holdout families was evaluated.

### S013: Fix the forward-difference fixture, 2026-10-09

- Starting revision: `4640a5db73a5078cef1030b990873d834f92bb36` on
  `convergence-defaults`; the working tree was clean.
- Scope: address the forward-difference and bounded L-BFGS stopping mismatch
  found in S012. Add regression coverage, explicit fixture configuration, and
  permanent rustdoc explaining derivative accuracy and solver tolerances.
- Source: `a73c8a5`; the planned comparison manifest was committed at `33c43f0`
  before execution. The [run record](runs/2026-10-09-forward-001.md) and
  [manifest](runs/2026-10-09-forward-001.toml) retain commands, inputs, source
  hashes, native outcomes, and output hashes.
- Evidence: paired runs from the same release binary passed independent checks
  for 18 cases and 38 CSVs. The forward stencil's exact-witness gradient norm is
  `1.4901161193847656e-8`. Explicit threshold `1e-7` stops at the exact minimum
  with convergence after 8 physical calls, versus 112 and line-search failure
  for the strict `1e-10` control. The first eight leaf calls and published
  history before stopping match. The other eight control fixtures match except
  elapsed time. This is work reduction, not a timing claim.
- Decision: D004 accepts the explicit fixture setting and documentation. No
  general solver default or numerical safeguard changes; calibration and holdout
  outcomes remain unopened. Step 5 remains in progress.
- Validation: all 23 focused regression checks pass across both precisions and
  every supported Vec, nalgebra, ndarray, and faer version. The routine Basin
  suite, including doctests, passes, as do all 16 measurement tests, workspace
  all-target/all-feature clippy with warnings denied, rustfmt, and public
  documentation. The new L-BFGS rustdoc example also passes independently.
- Development corrections: an initial scaled-solve assertion exposed `f32`
  line-search failure and failed `f64` parameter accuracy at multiplier `1e-8`.
  Scaled tests now check witness derivative bias only; solve tests establish
  unit-scale accuracy. One redundant constructor closure was removed after
  clippy flagged it. Neither correction changed a numerical safeguard.
- Open work and next task: validate executable NIST residual models and analytic
  Jacobians against both starts and published RSS, establish derivative accuracy
  and precision eligibility, and extend the measurement pilot through both LM
  factorizations/damping modes and legacy/full TRF. Freeze expanded cases and
  close pilot gates before calibration.

Verification commands from the repository root:

```sh
cargo fmt --all -- --check
CARGO_TARGET_DIR=target/convergence-defaults/build \
  cargo test -p basin --test finite_diff_stopping \
  --features nalgebra_v0_32,nalgebra_v0_33,nalgebra_v0_34,nalgebra_v0_35,ndarray_v0_15,ndarray_v0_16,ndarray_v0_17,faer_v0_22,faer_v0_23,faer_v0_24,parallel
CARGO_TARGET_DIR=target/convergence-defaults/build \
  cargo test -p basin --features nalgebra,ndarray,faer,problems,parallel
CARGO_TARGET_DIR=target/convergence-defaults/build \
  cargo test -p basin --doc solver::lbfgs \
  --features nalgebra,ndarray,faer,problems,parallel
CARGO_TARGET_DIR=target/convergence-defaults/build \
  cargo test -p competitor-bench --test convergence_measurement
CARGO_TARGET_DIR=target/convergence-defaults/build \
  cargo clippy --workspace --all-targets --all-features -- -D warnings
CARGO_TARGET_DIR=target/convergence-defaults/build \
  cargo doc --no-deps -p basin \
  --features nalgebra-lapack,ndarray-blas,faer,parallel,problems,serde
panache format --check TODO.md dev/convergence-defaults/README.md \
  dev/convergence-defaults/decisions.md dev/convergence-defaults/harness.md \
  dev/convergence-defaults/runs/2026-10-09-forward-001.md
panache lint TODO.md dev/convergence-defaults/README.md \
  dev/convergence-defaults/decisions.md dev/convergence-defaults/harness.md \
  dev/convergence-defaults/runs/2026-10-09-forward-001.md
git diff --check
```

The run manifest lists the build, both executions, the per-run independent
checks, and the paired comparison command. Raw artifacts and check logs live
under ignored `target/convergence-defaults/`. Source hashes describe the frozen
source revision, including the pre-run layout of `harness.md`; its post-run
formatting change only rewraps prose. Ad hoc checks verified frozen source
hashes through `git show`, output hashes, and local links.

### S014: Validate executable NIST models, 2026-10-09

- Starting revision: `998871a15d76b61005107a743a436db56e9d4668` on
  `convergence-defaults`; the working tree was clean.
- Scope: benchmark-only adapters for all 27 frozen NIST formulas, 2176
  observations, both primary starts, and printed reference points. Add
  hand-derived analytic Jacobians, native `f32`/`f64` evaluations, half-RSS cost
  and gradient adapters, and printed decimal rounding widths. Preserve family
  partitions, Nelson's log response, and Roszman1's principal arctangent branch.
- Evidence: [model documentation](nist-models.md) and the [validation
  report](nist-model-validation.json) record 13,056 independently checked CSV
  rows and 77,670 derivative values. The standard-library Decimal checker uses
  independent formulas and 100-digit arithmetic. Source hashes identify this
  working-tree validation; it is not a committed calibration freeze or a solver
  run. The CSV and complete check output live under ignored
  `target/convergence-defaults/`.
- Rounding finding: Lanczos1's RSS at the exact printed parameters is about
  `3.98336e-21`, versus published optimum RSS `1.43079e-25`. The local parameter
  rounding screen admits this discrepancy, but does not provide an interval
  certificate. Independent reference refinement remains necessary.
- Verification: all eight NIST integration tests pass with default features and
  with `parallel,basin-latest`, covering nalgebra 0.34 and 0.35 in both
  precisions. All 16 existing measurement tests pass. Workspace all-target and
  all-feature clippy passes with warnings denied, as do rustfmt, documentation
  formatting/lint, and whitespace checks. Independent elementary-function checks
  pass; a deliberately corrupted response is rejected by the checker.
- Development corrections: native finite-difference tests needed multiple
  stencil sizes and parameter-relative scales for narrow Gaussian peaks and tiny
  rational coefficients. The independent checker needed 100-digit arithmetic for
  tiny Gaussian derivatives, absolute phase-error allowances at trigonometric
  zeros, and explicit rational cancellation amplification. These corrections
  affect validation arithmetic, not solver policies.
- Gates: executable-model validation covers part of G404. Independent reference
  refinement and rigorous precision certificates remain open; full G404 and G403
  are not closed. No candidate policy, holdout solver outcome, default,
  numerical safeguard, or protocol amendment was evaluated or selected.
- Next task: establish reference and precision eligibility for development NIST
  cases, then extend the measured pilot through both LM factorizations and
  damping modes, legacy TRF, and full TRF. Validate rejected-trial diagnostics
  and accounting before interpreting stopping outcomes.

Reproduction and verification commands from the repository root:

```sh
CARGO_TARGET_DIR=target/convergence-defaults/build \
  cargo test -p competitor-bench --test nist_models --test convergence_measurement
CARGO_TARGET_DIR=target/convergence-defaults/build \
  cargo test -p competitor-bench --test nist_models \
  --features parallel,basin-latest
CARGO_TARGET_DIR=target/convergence-defaults/build \
  cargo run -p competitor-bench --bin verify_nist -- \
  --output target/convergence-defaults/nist-models-002.csv
python dev/convergence-defaults/check-nist.py \
  target/convergence-defaults/nist-models-002.csv \
  > target/convergence-defaults/nist-models-002-check.json
CARGO_TARGET_DIR=target/convergence-defaults/build \
  cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo fmt --all -- --check
panache format --check TODO.md dev/convergence-defaults/README.md \
  dev/convergence-defaults/cases.md dev/convergence-defaults/harness.md \
  dev/convergence-defaults/nist-models.md
panache lint TODO.md dev/convergence-defaults/README.md \
  dev/convergence-defaults/cases.md dev/convergence-defaults/harness.md \
  dev/convergence-defaults/nist-models.md
git diff --check
```

The output command requires a new filename when rerun. The first intermediate
CSV preceded the schema/family columns; only the final schema 1 CSV appears in
this report. Failed checker executions during arithmetic validation are not
solver runs or candidate outcomes.

### S015: Certify development NIST references and witnesses, 2026-10-09

- Starting revision: `e7b9a467281968a18613801921a4dd326f6d2c1e` on
  `convergence-defaults`; the working tree was clean.
- Scope: independent reference refinement and analytic precision eligibility for
  the 15 development NIST datasets in six families. Add 100-digit Decimal
  interval arithmetic, second-order AD, full-Hessian Newton refinement,
  independent Gauss–Newton refinement, Krawczyk inclusion, and positive interval
  LDL checks. Evaluate rounded references, coordinate neighbors, and both starts
  with the existing native `Nist<f32>`/`Nist<f64>` adapter. No Basin solver ran.
- Freeze: source commit `6f3ab0f` and planned-manifest commit `9a11eb3` preceded
  retained run `2026-10-09-nist-reference-001`. Its
  [manifest](runs/2026-10-09-nist-reference-001.toml) records exact revisions,
  input/source hashes, configuration, and output hashes.
- Evidence: the [run summary](runs/2026-10-09-nist-reference-001.md), [method
  documentation](nist-reference.md), and [retained
  report](nist-reference-eligibility.json) preserve 15 strict local minimum
  certificates, 294 native evaluations, 60 precision/start records, and all 330
  target combinations. Analytic witness eligibility covers 321 targets; nine
  `f32` targets remain `reference-pending` for MGH10 or Nelson. All `f64`
  targets pass. Nelson start 2 remains uncertified at designated `f32` target
  `1e-3`, because its stationarity uncertainty lacks the tenfold margin.
- Reference finding: Lanczos1's refined local RSS is about `1.430786772078e-25`,
  consistent with published RSS, while RSS at printed parameter midpoints
  remains about `3.98336e-21`. The refined interval supplies reference
  uncertainty. All 15 refined RSS intervals overlap their published rounding
  intervals.
- Verification: six analytic Python tests, all 24 existing NIST/measurement Rust
  tests, and eight negative controls pass. Workspace all-target/all-feature
  clippy denies warnings; rustfmt, documentation formatting/lint, and whitespace
  checks pass. Negative controls cover altered evidence, independent native-cost
  consistency, holdout rejection, and output overwrite refusal.
- Development corrections: mixed interval/AD operands needed reflected operator
  dispatch. The rounding screen now uses actual objective/gradient accumulation
  terms; independent interval comparisons separately bound cancellation. These
  corrections changed no target grid, threshold, family partition, or protocol.
  Disk exhaustion and removal of an earlier generated build directory
  interrupted preliminary Rust checks. Verification completed in a separate
  build with debug information and incremental compilation disabled.
- Gates: the analytic development subset of G403 and development reference
  refinement in G404 are covered; the complete gates remain open. Witness errors
  are bounded only at tested points. Pending targets, finite-difference bias,
  holdout references, start-basin classification, conditioning metadata,
  transformed cases, and full backend coverage remain separate work. No global
  reference claim, candidate policy, holdout solver outcome, default, numerical
  safeguard, or protocol amendment was selected. Step 5 remains in progress.
- Next task: connect the frozen development references and eligibility to the
  measured pilot through both LM factorizations/damping modes, legacy TRF, and
  full TRF. Validate rejected-trial diagnostics and evaluation accounting before
  interpreting stopping outcomes. Keep uncertified targets visible and out of
  target scores.

Reproduction and verification commands from the repository root:

```sh
PYTHONDONTWRITEBYTECODE=1 python \
  dev/convergence-defaults/reference-tools/test_reference.py
CARGO_PROFILE_DEV_DEBUG=0 CARGO_PROFILE_TEST_DEBUG=0 CARGO_INCREMENTAL=0 \
  CARGO_TARGET_DIR=target/nist-reference-build \
  cargo test -p competitor-bench --test nist_models --test convergence_measurement
CARGO_PROFILE_DEV_DEBUG=0 CARGO_INCREMENTAL=0 \
  CARGO_TARGET_DIR=target/nist-reference-build \
  cargo build -p competitor-bench --bin verify_nist_witness
python dev/convergence-defaults/reference-tools/preflight.py prepare \
  target/convergence-defaults/2026-10-09-nist-reference-001
target/nist-reference-build/debug/verify_nist_witness \
  --points target/convergence-defaults/2026-10-09-nist-reference-001/points.csv \
  --output target/convergence-defaults/2026-10-09-nist-reference-001/native.csv
python dev/convergence-defaults/reference-tools/preflight.py check \
  target/convergence-defaults/2026-10-09-nist-reference-001 \
  --output dev/convergence-defaults/nist-reference-eligibility.json
python dev/convergence-defaults/check-nist-reference.py \
  target/convergence-defaults/2026-10-09-nist-reference-001 \
  dev/convergence-defaults/nist-reference-eligibility.json \
  --probe target/nist-reference-build/debug/verify_nist_witness
CARGO_PROFILE_DEV_DEBUG=0 CARGO_INCREMENTAL=0 \
  CARGO_TARGET_DIR=target/nist-reference-build \
  cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo fmt --all -- --check
panache format --check TODO.md dev/convergence-defaults/README.md \
  dev/convergence-defaults/cases.md dev/convergence-defaults/harness.md \
  dev/convergence-defaults/nist-models.md dev/convergence-defaults/nist-reference.md \
  dev/convergence-defaults/runs/README.md \
  dev/convergence-defaults/runs/2026-10-09-nist-reference-001.md
panache lint TODO.md dev/convergence-defaults/README.md \
  dev/convergence-defaults/cases.md dev/convergence-defaults/harness.md \
  dev/convergence-defaults/nist-models.md dev/convergence-defaults/nist-reference.md \
  dev/convergence-defaults/runs/README.md \
  dev/convergence-defaults/runs/2026-10-09-nist-reference-001.md
git diff --check
```

The preflight refuses existing preparation directories and output reports. For a
rerun, use new paths and its source revision. Raw neighbor records remain in the
ignored full report; all certificates, eligibility decisions, and selected
witnesses are retained in Git. The negative-control checker was added after
source freeze and only validates evidence; its hash is retained separately.

### S016: Measure native LM/TRF trials and development NIST, 2026-10-09

- Starting revision: `12f4ced202a17ee87857710e77604ec773b2154e` on
  `convergence-defaults`, with a clean working tree.
- Scope: opt-in solver-owned model/trial observations for normal-equation and
  pivoted-QR LM with both damping modes, legacy TRF, and full TRF. Connect them
  to physical callbacks, authoritative counts, and publication boundaries.
  Preserve native defaults and safeguards. Run analytic validation before the
  unconstrained development NIST pilot in both precisions.
- Freeze: source `0ed083c` and planned manifest `dd01375` precede retained run
  `2026-10-09-least-squares-001`. Verifier-only corrections are frozen in
  `bbcde7a` and `25cbcf7`; the solver source and measured CSVs are unchanged.
- Evidence: the [run summary](runs/2026-10-09-least-squares-001.md),
  [manifest](runs/2026-10-09-least-squares-001.toml), [method
  documentation](least-squares-pilot.md), and [retained
  report](least-squares-results.json) preserve 80 analytic checks, 360 NIST
  solves, all outcomes and target grids, and independent point verification.
  Designated returned quality passes in 163/180 eligible `f64` and 115/174
  eligible `f32` cases. Six Nelson start 2 route cases remain withheld at the
  designated `f32` target. Four convergence reports miss eligible designated
  quality; local reference/start-basin limits prevent a premature-stop claim.
- Accounting: 372,015 physical calls reconcile with 339,799 residual and 32,610
  Jacobian requests, 360 fused initializations, and 34 budget denials.
  Residual-derived objectives add no cost request. NIST native observations
  retain 32,250 accepted publications, 307,157 rejections, 139 non-finite
  trials, and 32 denied trial callbacks. Analytic caps demonstrate that a native
  acceptance followed by Jacobian denial creates no publication or returned
  point.
- Verification: the pure-Rust solver suite and focused non-finite evidence
  checks pass. All 39 competitor checks pass on nalgebra 0.34 and 0.35 across
  focused commands. Four Python tests include seven mutation controls and
  independent first-order/full-Hessian AD agreement for every development model.
  Workspace all-target/all-feature clippy denies warnings. Rustfmt, rustdoc,
  default/no-default WASM builds, documentation format/lint, local links, and
  whitespace checks pass.
- Development corrections: model-prediction roundoff bounds need absolute terms
  within cancelling dot products; native squared steps can underflow before
  damping rescales them. Parameter uncertainty must cover symmetry branches. An
  initial corpus quality pass was stopped after native validation to use
  provable target-specific parameter screens. Original outputs and probe files
  remain retained, with fresh final verification under `nist-checked/`. These
  corrections change no numerical setting, target, or protocol rule.
- Decisions: none. No holdout solver outcome, policy, default, safeguard, or
  protocol amendment was selected. Native convergence remains distinct from
  external point quality. Full TRF failures and legacy TRF budget errors retain
  their original classification even when a published point attained a target.
  Step 5 remains in progress.
- Next task: extend analytic diagnostics and accounting to bound-active and
  fixed-coordinate TRF and robust-loss LM/TRF, using independently known KKT or
  robust references. Then close remaining backend, derivative, inner-work,
  transformed-case, and reference-branch gates before affected candidate sweeps.

Reproduction commands are frozen in the run manifest. Core verification used:

```sh
CARGO_PROFILE_DEV_DEBUG=0 CARGO_PROFILE_TEST_DEBUG=0 CARGO_INCREMENTAL=0 \
  CARGO_TARGET_DIR=target/nist-reference-build \
  cargo test -p basin --features nalgebra,ndarray,faer,problems,parallel
CARGO_PROFILE_DEV_DEBUG=0 CARGO_PROFILE_TEST_DEBUG=0 CARGO_INCREMENTAL=0 \
  CARGO_TARGET_DIR=target/nist-reference-build \
  cargo test -p competitor-bench --test least_squares_measurement \
  --test nist_models --test convergence_measurement
CARGO_PROFILE_DEV_DEBUG=0 CARGO_PROFILE_TEST_DEBUG=0 CARGO_INCREMENTAL=0 \
  CARGO_TARGET_DIR=target/nist-reference-build \
  cargo test -p competitor-bench --features basin-latest \
  --test least_squares_measurement --test nist_models --test convergence_measurement
CARGO_PROFILE_DEV_DEBUG=0 CARGO_INCREMENTAL=0 \
  CARGO_TARGET_DIR=target/nist-reference-build \
  cargo clippy --workspace --all-targets --all-features -- -D warnings
CARGO_PROFILE_DEV_DEBUG=0 CARGO_INCREMENTAL=0 \
  CARGO_TARGET_DIR=target/nist-reference-build \
  cargo doc --no-deps -p basin \
  --features nalgebra-lapack,ndarray-blas,faer,parallel,problems,serde
CARGO_PROFILE_DEV_DEBUG=0 CARGO_INCREMENTAL=0 \
  CARGO_TARGET_DIR=target/least-squares-wasm \
  cargo build --target wasm32-unknown-unknown
CARGO_PROFILE_DEV_DEBUG=0 CARGO_INCREMENTAL=0 \
  CARGO_TARGET_DIR=target/least-squares-wasm \
  cargo build --target wasm32-unknown-unknown --no-default-features
cargo fmt --all -- --check
PYTHONDONTWRITEBYTECODE=1 python dev/convergence-defaults/test-least-squares.py \
  target/convergence-defaults/2026-10-09-least-squares-001/analytic
git diff --check
```

The default-feature run uses nalgebra 0.34.2; `basin-latest` tests use 0.35.0.
The final quality pass performs 11,253 independent native probes and 3,670
interval quality evaluations outside the solve ledger and timer. Bulk traces
remain under the ignored run directory; the retained summary and manifest remain
sufficient to inspect outcomes and reproduce the measurements.

### S017: Extend bounded full-TRF measurements, 2026-10-09

- Starting revision: `ce22b005c16b07fad119688a9163c42d5969e509`, clean working
  tree on `convergence-defaults`.
- Scope: analytic active-bound, mixed-fixed, all-fixed, and stationary
  nonminimum full-TRF controls in both precisions, with caps 0, 1, 2, and 4000.
- Freeze and evidence: source/planned manifest `e2c9a6a` preceded run
  [2026-10-09-bounded-trf-001](runs/2026-10-09-bounded-trf-001.md). Its manifest
  records commands and hashes; the retained report includes all 32 outcomes.
- Findings: both unrestricted identity `f64` solves converge. Accurate `f32`
  returns retain native failure and stall outcomes. All-fixed solves skip the
  Jacobian; stationary nonminimum controls converge but fail minimum quality.
  Four accepted trials denied their Jacobian request remain unpublished.
- Validation: 17 measurement tests on each nalgebra version, six Python tests
  with seven mutation controls, workspace all-target/all-feature clippy with
  warnings denied, Rust formatting, and documentation checks.
- Correction: returned quality must not require native convergence. The
  verifier-only correction preserves original solve evidence and settings.
- Decisions: none; step 5 remains open and holdout outcomes remain sealed.
- Next task: robust-loss LM/TRF analytic fixtures with independently known
  minima and gradients. Legacy bounded TRF, other backends, derivatives,
  transformations, rank deficiency, and full inner work retain their gates.

### S018: Measure robust-loss LM/TRF, 2026-10-09

- Starting revision: `30ebdbc`, clean tree on `convergence-defaults`.
- Scope: Huber outliers at two scales, Huber transition, symmetric soft-L1,
  Cauchy negative curvature, non-finite trials, and an active robust bound; both
  precisions and six native routes, with only TRF on the bounded case.
- Freeze: source/planned manifest `f0c6d05` precedes run
  [2026-10-09-robust-ls-001](runs/2026-10-09-robust-ls-001.md). Verifier-only
  corrections and continuation checks are frozen in `68c6a28`. The manifest
  records commands and source/output hashes; solve source and CSVs are
  unchanged.
- Evidence: all 38 unrestricted default `f64` cases converge with good
  measurement quality. Of 38 `f32` controls, 34 return good points (16
  converged, 18 stalled); four legacy TRF budget errors have good last
  publications but no returned points. These four runs include 15,894 non-finite
  predictions.
- Stopping finding: all 48 explicit LM relative probes report convergence; four
  Nielsen/Cauchy probes fail independent quality. Default continuations share
  their exact callback prefixes and reach the analytic minimum. Tiny accepted
  steps under large damping pass both model reduction and step tests.
- Accounting: 18,069 physical calls and 232 denials reconcile per run with
  authoritative counters. Fourteen accepted trials denied their Jacobian remain
  unpublished. No cost callbacks or logical cost requests occur.
- Verification: 29 Rust measurement tests on each nalgebra version; 12 Python
  tests with 12 robust corruption controls; previous analytic and bounded
  checker/test passes; workspace all-target/all-feature clippy with warnings
  denied, Rust formatting, and documentation/link/hash checks.
- Corrections: prediction roundoff includes unsummed robust gradient terms
  involved in QR cancellation and scalar-product underflow. Non-finite
  predictions remain non-finite rejected decisions. Mutation controls target
  completed decisions. Last publications stay separate from returned quality.
- Decisions: none. Step 5 remains open; no solver default, safeguard,
  calibration policy, or holdout outcome changed.
- Next task: investigate legacy TRF finite-model/no-progress safeguards and
  robust step/model-reduction composition before affected sweeps. Other losses,
  backends, derivatives, larger and fixed-coordinate robust cases, precision
  certificates, and full inner work retain their gates.

### S019: Fix and backport legacy TRF non-finite models, 2026-10-09

- Starting revision: `7882623`, clean tree on `convergence-defaults`.
- Scope: stop before solving with non-finite damping or evaluating a non-finite
  trial model. Preserve current point, cost, counts, caches, and completed
  iteration count; retain convergence defaults and finite rejections.
- Fix: `24cb364` adds guards and regression checks across dense and sparse
  backends in both precisions, with a downstream Jacobian prediction check.
- Evidence: planned manifest `cbb646a` precedes
  [2026-10-09-trf-finite-model-001](runs/2026-10-09-trf-finite-model-001.md).
  All 352 cases verify independently. Every callback trace matches its baseline
  prefix; every final published point and cost is identical. Exactly four `f32`
  legacy TRF budget errors become returned numerical failures after 20, 20, 22,
  and 44 calls, eliminating 15,894 non-finite predictions.
- Backports: clean main commit `250f59e` is draft [PR
  #116](https://github.com/jolars/basin/pull/116). Compatible 1.x commit
  `a8dd82c`, cherry-picked with provenance and adapted to its existing state and
  failure enum, is draft [PR #117](https://github.com/jolars/basin/pull/117).
  Main should merge first. No release bookkeeping, dependency, or public API
  changed on 1.x.
- Verification: full pure-Rust solver suites and all-target/all-feature clippy
  pass on all three branches. Rustdoc and both WASM configurations pass on the
  convergence and 1.x branches. Rust formatting, robust checker/tests, baseline
  corruption controls, exact comparison, documentation, links, and hashes pass.
- Decisions: correctness safeguard only. Step 5 remains open. No numerical
  tolerance, candidate policy, or holdout outcome selected.
- Next task: review robust step/model-reduction composition using the unchanged
  Cauchy controls before affected sweeps. Remaining variants, backends,
  derivatives, reference certificates, and complete inner work retain gates.

### S020: Review robust LM stopping composition, 2026-10-10

- Starting revision: `68b96a8`, clean tree on `convergence-defaults`.
- Scope: seven stopping configurations on six unbounded robust fixtures, four LM
  routes, and both native precisions. Isolate model reduction, trial step,
  normalized gradient, and radius tests while retaining the existing default
  gradient control and numerical safeguards.
- Freeze: source `2a5e89e` and planned manifest `510d5db` precede
  [2026-10-10-robust-stopping-001](runs/2026-10-10-robust-stopping-001.md).
  Verifier corrections, public guidance, and permanent backend checks are in
  `6f2aec3`. The manifest records source/output hashes and exact commands.
- Evidence: 336 ablations and a 352-case baseline reproduction. Every ablation
  shares its common callback prefix with its default continuation. All 96
  default and combined-relative controls match baseline callbacks, publications,
  native observations, checks, counts, and outcomes exactly. The baseline also
  agrees with all previously recorded outcomes, including the four known
  legacy-TRF corrections.
- Findings: model-only and step-only tests each reproduce the four severe
  Nielsen/Cauchy stops. Both tests pass on accepted, published trials with gain
  ratios above 0.25, so pairing them or adding those filters still stops
  prematurely. Default and normalized-gradient configurations pass common point
  quality on all 48 cases each. Six additional model-reduction stops on the
  non-finite-domain fixture miss the stricter common quality thresholds; these
  are separate accuracy findings. Good stalled points retain their native
  classification.
- Guidance: public LM and QR rustdoc explain progress versus stationarity and
  gradient-only configurations. Permanent negative-curvature Cauchy controls
  verify default-gradient recovery across dense and supported sparse backends,
  both precisions, and both LM damping modes.
- Validation: all 37 measurement tests pass on nalgebra 0.34 and 0.35. Four
  composition tests include six corruption controls; all 12 existing checker
  tests pass on the ablation and baseline. The full pure-Rust solver suite,
  workspace all-target/all-feature clippy with warnings denied, rustdoc,
  formatting, and documentation checks pass.
- Corrections: native radius evidence is unavailable on non-finite trials, so
  the verifier recognizes its documented observation eligibility. A generic
  backend test needed an explicit state constructor. These corrections change no
  measured solver code, settings, targets, or raw CSVs.
- Decision:
  [D005](decisions.md#d005-screen-robust-lm-progress-tests-against-stationarity)
  restricts the demonstrated progress-only candidates. No replacement
  composition or general default selected; calibration remains pending and
  holdout outcomes remain sealed. Step 5 stays open.
- Next task: larger and rank-deficient robust controls and arctangent loss,
  followed by stationarity guards, derivative modes, remaining backend versions,
  precision eligibility, transformations, and full inner-work measurements
  before affected candidate sweeps.

### S021: Extend robust controls and retain recovery failures, 2026-10-10

- Starting revision: `b7f4ab5`, clean tree on `convergence-defaults`.
- Scope: scalar arctangent and all four built-in robust losses on full-rank and
  rank-deficient four-parameter models. Retain the seven stopping
  configurations, four LM routes, both precisions, analytic derivatives, and
  unit scales.
- Freeze: source `aa8e078` and planned manifest `43e518e` precede
  [2026-10-10-robust-extended-001](runs/2026-10-10-robust-extended-001.md).
  Verifier correction `57a30d5` changes no solver code, setting, or solve CSV.
- Evidence: 504 full-budget ablations, 216 exact callback-budget interruptions,
  and 336 original ablations reproduced with identical outcomes, callbacks,
  publications, observations, and checks, excluding elapsed time. Independent
  rank-deficient quality measures distance to the minimizer set, with explicit
  nullspace checks.
- Findings: 24 combined-relative stops are confirmed premature. Of these, 22
  pass model and step tests on accepted, published trials with gain ratios above
  0.25. Both gradient configurations pass 68 of 72 point-quality checks. Their
  four poor points are Nielsen `f32` stalls on rank-deficient Huber/arctangent
  models under both factorizations, with substantial gradients and steps that
  round away. Trust-region damping passes these same fixtures.
- Correction: the independent model-identity checker accounts for QR projection
  roundoff with large clipped residuals. Native operands and formula checks stay
  intact, and 789 discrepancies above local relative roundoff remain recorded.
  This screen supplies no precision or model-solve accuracy certificate.
- Validation: all 39 measurement tests pass on nalgebra 0.34 and 0.35. All 17
  independent checker tests and five stopping/reproduction tests pass, including
  twenty CSV corruption controls. Workspace all-target/all-feature clippy with
  warnings denied, Rust formatting, and documentation checks pass.
- Decision:
  [D006](decisions.md#d006-extend-robust-controls-and-gate-nielsen-recovery)
  extends the progress-test screen and retains the measured recovery gate. No
  default or guarded policy selected; holdout outcomes remain sealed.
- Next task: investigate and recover the four rank-deficient Nielsen `f32`
  stalls before calibrating stationarity-guarded progress. Remaining nonlinear
  robust models, derivatives, bounds/fixed coordinates, transformations, backend
  measurements, precision certificates, and complete inner work retain gates.
