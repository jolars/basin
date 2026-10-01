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

Step 1 is complete. Step 2 has reconciled the public solver names and begun the
variant and stopping-behavior audit. No solver policies have been selected and
no numerical experiments have run.

  | Step | Deliverable                                   | Status      |
  | ---- | --------------------------------------------- | ----------- |
  | 1    | Branch and session records                    | Complete    |
  | 2    | Inventory of stopping behavior and variants   | In progress |
  | 3    | Reference survey and candidate policies       | Pending     |
  | 4    | Reviewed experimental protocol                | Pending     |
  | 5    | Measurement harness and pilot                 | Pending     |
  | 6    | Calibration and independent validation        | Pending     |
  | 7    | Implementation and verification               | Pending     |
  | 8    | Complete coverage and permanent documentation | Pending     |

## Records

- [Inventory](inventory.md): initial coverage, audit fields, and completion
  requirements for each solver and relevant variant.
- [Protocol](protocol.md): experimental requirements and choices that must be
  resolved before calibration. It remains a draft.
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

### S003: Audit first-order and Gauss–Newton stops, 2026-10-01

- Starting revision: `7bb53a2b5763b497b23e1eaee503b6c2ec7f82f0` on
  `convergence-defaults`; the working tree was clean.
- Scope: source audit of `GradientDescent`, `ProjectedGradientDescent`,
  `NonlinearCg`, `Bfgs`, `TrustRegion`, `Sgd`, and `GaussNewton`, including
  stopping-relevant updates, trust-region modes, and objective-refresh choices.
- Evidence: [first-order and Gauss-Newton
  records](inventory.md#first-order-and-gauss-newton-stopping-records) link to
  each defining source and the shared configured-check builders. The per-solver
  tracker distinguishes this source observation from the pending reference,
  candidate, experiment, decision, implementation, and verification stages.
- Decisions: none; all numerical dispositions remain pending.
- Validation: `panache format --check dev/convergence-defaults`,
  `panache lint dev/convergence-defaults`, and `git diff --check` pass. An ad
  hoc local Markdown link and heading-anchor check passes. No Rust code or
  numerical behavior changed.
- Open questions: audit the inner trust-region strategies and line searches
  separately, especially failure propagation, then continue through the
  remaining derivative-free, constrained, global, and composed solvers.
- Next task: record source-backed stops for the remaining solvers and build
  explicit dependency records for line searches, bracketers, and subproblems.
