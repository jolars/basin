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
remaining gaps and gates before large sweeps. The [protocol](protocol.md) now
contains a concrete proposal for maintainer review, and the [Misra1a
pilot](runs/2026-10-02-misra1a-001.md) supplies the first focused numerical
evidence. The [coverage matrix](coverage.md) now documents 57 experimental
fixture definitions, native `f32`/`f64` evaluation, proposed family partitions,
and executable coverage of 14 solver names. The [expanded
pilot](runs/2026-10-02-coverage-002.md) exercises development cases only. No
solver policies have been selected or independently validated.

  | Step | Deliverable                                   | Status      |
  | ---- | --------------------------------------------- | ----------- |
  | 1    | Branch and session records                    | Complete    |
  | 2    | Inventory of stopping behavior and variants   | In progress |
  | 3    | Reference survey and candidate policies       | Complete    |
  | 4    | Reviewed experimental protocol                | Proposed    |
  | 5    | Measurement harness and pilot                 | Pilot begun |
  | 6    | Calibration and independent validation        | Pending     |
  | 7    | Implementation and verification               | Pending     |
  | 8    | Complete coverage and permanent documentation | Pending     |

## Records

- [Experiment coverage](coverage.md): fixture strata, proposed family splits, a
  47-name applicability matrix, runnable entry points, and remaining gaps.

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

- [Protocol](protocol.md): proposed success definitions, case strata, budgets,
  and selection rule. It awaits maintainer review before a large sweep.

- [Decisions](decisions.md): agreed direction, proposals, evidence, and open
  questions. Numerical decisions remain pending.

- [Run records](runs/README.md): the first NIST pilot, conventions for tracked
  manifests, and concise results. Bulk traces belong in
  `target/convergence-defaults/`, which the repository already ignores.

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

### S011: Propose the protocol and run the first NIST pilot, 2026-10-02

- Starting revision: `9ef821dc3496c7dc06eae7628c7ff79ef9e397ad` on
  `convergence-defaults`; the working tree was clean.
- Scope: turn the draft [protocol](protocol.md) into a reviewable proposal,
  reproduce one public NIST StRD case from both starts across the four pilot
  solvers, and resolve Backtracking's exhausted Armijo outcome with a test-first
  fix.
- Evidence: [run 2026-10-02-misra1a-001](runs/2026-10-02-misra1a-001.md) records
  the NIST source, certified answer, exact command and hashes, returned quality,
  termination, and callback categories. Five of six runs with relative parameter
  error below `1e-8` ended without `SolverConverged`; L-BFGS-B remained less
  accurate at the 500-iteration pilot cap. This is one development case, not a
  27-case replication or a selected default.
- Decisions: no numerical default or selection threshold was accepted. Q003 no
  longer requires the reporter's private harness because NIST publishes the 27
  primary files. Q007 is resolved for outcome-aware Backtracking callers; its
  legacy step-only method remains source-compatible. Step 4 awaits maintainer
  review and freezing before a large calibration sweep.
- Validation: the Backtracking regression failed before the fix and passed after
  it; gradient descent now reports `SolverFailed` and retains the accepted point
  on exhausted backtracking. The Misra1a certificate and analytic Jacobian test
  passes. The routine pure-Rust feature test suite, all-target all-feature
  clippy, rustdoc, both WASM builds, rustfmt, Panache format/lint, and
  `git diff --check` pass.
- Open questions: the proposed case weights, target grid, resource budgets, and
  material reliability/work threshold need maintainer review. Step 5 still needs
  stage-aware tracing, physical model-call accounting, and additional case
  families; Q006 and the other [step 3
  gates](review-step3.md#evidence-gaps-and-gates) remain open.
- Next task: obtain maintainer review of the [protocol](protocol.md), then
  freeze its accepted revision and implement the full measurement schema before
  calibration. Continue assembling NIST and non-NIST cases and resolve
  solver-specific source gates without treating this pilot as a default choice.

Commands run from the repository root:

```sh
cargo test -p basin --lib line_search::backtracking::tests
cargo test -p basin --test gradient_descent_line_search_evaluations exhausted_backtracking_keeps_the_accepted_iterate
cargo test --release -p competitor-bench --bin convergence_pilot
cargo run -p competitor-bench --release --bin convergence_pilot > target/convergence-defaults/2026-10-02-misra1a-001/raw.csv
cargo test -p basin --features nalgebra,ndarray,faer,problems,parallel
CARGO_INCREMENTAL=0 cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo doc --no-deps -p basin --features nalgebra-lapack,ndarray-blas,faer,parallel,problems,serde
cargo build --target wasm32-unknown-unknown
cargo build --target wasm32-unknown-unknown --no-default-features
cargo fmt --all -- --check
panache format --check dev/convergence-defaults TODO.md
panache lint dev/convergence-defaults TODO.md
git diff --check
```

### S012: Broaden fixtures and validate the recorder, 2026-10-02

- Starting revision: `7065d6d4d94048d9a01f4c176352cf6447c2b301` on
  `convergence-defaults`; the working tree was clean.
- Scope: implement the requested coverage expansion in `competitor-bench`,
  preserve all 27 NIST datasets, add native-precision analytic, constrained,
  scalar, and stochastic controls, and record baseline runs.
- Evidence: the [coverage matrix](coverage.md) accounts for all 47 solver names
  and documents 57 fixture definitions. Generated case lists include both starts
  or intervals and references. The [expanded
  pilot](runs/2026-10-02-coverage-002.md) contains 844 development solves across
  14 names, both precisions, and 20 paired SGD seeds.
- Decisions: D004 records the authorized fixture expansion. D003 remains
  proposed; no numerical defaults, success thresholds, precision floors, or
  final case weights have been selected. Validation families remain unused for
  optimization.
- Validation: 16 focused tests, workspace all-target/all-feature clippy,
  rustfmt, Panache format/lint, checksums, local links, case-list regeneration,
  CSV schema/partition checks, and `git diff --check` pass. Certificate and
  derivative tests cover all NIST datasets. Recorder tests cover initialization,
  fused calls, a failed partial step with rejected trial work, pass budgets, and
  agreement with an ordinary Executor run.
- Environment: disk space ran out while verification was in progress. Cleaning
  this project's generated debug artifacts with
  `cargo clean -p basin --profile dev` recovered space; inputs and raw
  experiment records were retained. The affected new runner source was restored,
  rebuilt, tested, and rerun before recording final hashes.
- Open questions: the [coverage gaps](coverage.md#before-calibration),
  attainable targets, protocol choices, and existing solver-specific gates
  remain. Vector fixtures use `Vec`; internal trial stages, typed application
  errors, and composed solver accounting are not yet fully covered.
- Handoff: the maintainer should review the proposed family weights, resource
  budgets, and material reliability/work tradeoff in the
  [protocol](protocol.md). Implementation can then continue with
  attainable-target checks, remaining solver wiring, and the explicitly
  identified missing families before calibration. No further broad,
  undifferentiated corpus expansion is needed to begin that work.

Commands run from the repository root:

```sh
CARGO_INCREMENTAL=0 cargo test --release -p competitor-bench --lib --bin convergence_suite --bin convergence_scalar --bin convergence_pilot
CARGO_INCREMENTAL=0 cargo build --release -p competitor-bench --bin convergence_suite --bin convergence_scalar
CARGO_INCREMENTAL=0 cargo clippy --workspace --all-targets --all-features -- -D warnings
target/release/convergence_suite --list > dev/convergence-defaults/cases.csv
target/release/convergence_scalar --list > dev/convergence-defaults/scalar-roots.csv
target/release/convergence_scalar --minima --list > dev/convergence-defaults/scalar-minima.csv
cargo fmt --all -- --check
panache format --check dev/convergence-defaults TODO.md crates/competitor-bench/data/nist/README.md
panache lint dev/convergence-defaults TODO.md crates/competitor-bench/data/nist/README.md
sha256sum --check dev/convergence-defaults/runs/2026-10-02-coverage-002.sources.sha256
git diff --check
```

The [run manifest](runs/2026-10-02-coverage-002.toml) contains each pilot
command, output path, and output hash.
