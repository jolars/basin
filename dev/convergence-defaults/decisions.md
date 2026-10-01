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

## Numerical decision template

For each policy decision, record the solver and variants, current and proposed
formulas and defaults for both precisions, exact reference versions, candidate
alternatives, calibration and validation run IDs, family-level and worst-case
outcomes, sensitivity to nearby settings, unresolved limitations, and required
regressions or migration settings. State why the evidence supports changing or
retaining the default. A proposal becomes accepted only after review of that
evidence; implementation and verification remain separate work.

## Open questions

  | ID   | Question                                                                                                               | Resolve in        |
  | ---- | ---------------------------------------------------------------------------------------------------------------------- | ----------------- |
  | Q001 | Which problem families and applications should determine coverage and weights?                                         | Steps 2 and 4     |
  | Q002 | Which external accuracy targets and precision floors are attainable and useful for each solver family?                 | Steps 3 and 4     |
  | Q003 | Is the reporter's NIST harness available, and which additional cases are needed for full solver coverage?              | Steps 2 and 4     |
  | Q004 | Which stochastic methods should retain budget-driven termination, and how should their numerical stops be interpreted? | Steps 3, 4, and 6 |
  | Q005 | What experiment budgets, repetition counts, and reliability/work thresholds should guide selection?                    | Step 4            |

These questions do not prevent the step 2 inventory. Add concrete findings and
new questions as that audit proceeds.
