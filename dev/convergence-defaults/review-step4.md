# Step 4 protocol review

Reviewed 2026-10-05 against the complete step 4 TODO item, the [step 3 evidence
gates](review-step3.md#evidence-gaps-and-gates), all 28 corpus specifications,
the existing trace/LM helpers, and the primary sources linked in
[CDP-1](protocol.md). This is a design review by the author of this change, not
an independent reviewer assessment or empirical validation.

**Disposition:** step 4 is complete. `CDP-1` and the [case register](cases.md)
are ready for implementation and focused pilots in step 5. No sweep, candidate
default, precision floor, or numerical reference certificate is approved by this
review. The gates below must close before calibration of affected families.

## Requirement review

  | Requirement                                   | Design decision                                                                                                                                                         | Review result                                                                                                                                                                               |
  | --------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
  | Independent success                           | Reevaluate returned points, with explicit objective, stationarity, feasibility, identifiable-parameter, and root-position tests.                                        | Native codes cannot establish success. KKT/gradient diagnostics do not certify global optimality. Nonsmooth cases retain a separate applicable target.                                      |
  | Absolute/relative scales and multiple targets | Fixed physical/case scales, initial-gap relative scale, separate `f64`/`f32` grids, and designated targets.                                                             | Offsets cannot inflate the verifier allowance; objective transforms carry their scales. Units and normalizations are explicit.                                                              |
  | Attainable floors                             | Independent uncertainty bounds and representable witnesses before tuning; eligibility shared by candidates.                                                             | No tolerance relaxation based on solver performance. Missing certificates stay visible and block affected selection.                                                                        |
  | Local versus premature stops                  | Separate all-reference profiles, independently fixed reference-basin strata, alternate-minimum evidence, and matched continuation.                                      | A missed global optimum is not sufficient evidence of a stopping defect. Changed trajectories establish sensitivity rather than causal evidence about an earlier stop.                      |
  | Global/stochastic methods                     | Repeated seeds, equal-total-budget controls, fixed-budget quality, target-attainment profiles, and uncertainty.                                                         | Population collapse is only a diagnostic. Deterministic global methods use case outcomes rather than artificial seed replication.                                                           |
  | Corpus stratification                         | All 28 `ALL_SPECS` entries assigned to 19 corpus families; related wrappers/transforms inherit membership.                                                              | Applicability, dimension, constraints, conditioning, derivative availability, and precision are explicit. Additional scalar/constrained/SGD fixtures have named supply gates.               |
  | NIST reproduction                             | Public issue checked; 27 source snapshots, both starts, dimensions, decimal reference strings, provenance, hashes, and related-model partitions retained.               | The missing reporter harness does not block input assembly. Executable model validation is separate and still required.                                                                     |
  | Stress coverage                               | Scale/offset transforms, nonzero residuals, active/fixed bounds, rank deficiency, non-finite inputs, stagnation, finite differences, and supported noise.               | Safeguard diagnostics cannot inflate accuracy/work scores. Native `f32` and backend gaps cannot be hidden by casting or exclusion.                                                          |
  | Leakage and weighting                         | Whole families held out, equal family weights, nested weights within families, sealed candidate outcomes.                                                               | Picheny/Goldstein–Price, quadratic representations, scalar powers, and related NIST models remain together. Retuning consumes the whole holdout family.                                     |
  | Work to common targets                        | Physical leaf-call ledger reconciled with authoritative categories, nested budgets, published-point first attainment, and full-cap penalties for failures in selection. | Early failures cannot appear efficient in selection. Fused/finite-difference work and inner/bracketing work have explicit rules. Cross-regime comparisons require category counts and time. |
  | Selection fixed before tuning                 | Explicit candidate grid expansion, 95% deterministic reliability, no lost baseline successes, stochastic noninferiority margin, stable-neighbor/work rules.             | Thresholds are declared engineering choices. Failure to meet them means unresolved evidence, not permission to weaken them after seeing results.                                            |

## Review findings and corrections

1. The initial outline left every operational choice pending. `CDP-1` now fixes
   grids, budgets, repetitions, seed derivation, weighting, partition
   membership, candidate-search rules, and acceptance margins. Executable
   expansion belongs to step 5 and must preserve these choices or record an
   amendment.
2. Absolute objective magnitude would make a large offset loosen quality checks.
   The verifier uses a declared absolute unit and initial reference gap, and
   checks transformed solver arithmetic separately. Precision loss can make a
   target ineligible, but cannot turn a poor answer into success.
3. A single best-known value cannot distinguish alternate minima, stationary
   saddles, and early stopping. Local evidence, reference-basin membership,
   curvature checks, and continuation now have separate roles. Candidate results
   cannot determine which cases count as reference-basin cases.
4. A root solver may return an exact root without shrinking its saved bracket.
   Independent position and residual checks admit that outcome while retaining
   bracket width separately. A rounded callback zero alone is insufficient.
5. A stricter schedule or inner solve can change the trajectory. Improvement
   from such a run is policy sensitivity. Confirmed premature termination
   requires a matching continuation/trajectory with the same total budget and
   safeguards.
6. A fast failure can distort a work average, and a rejected trial can look
   better than the returned point. Failures receive the full cap in selection;
   returned recommendations and sampled trials have distinct series.
   Infeasible/missing results stay in fixed-budget quantiles as infinite error.
7. Corpus membership does not guarantee executable precision/backend coverage.
   Most corpus wrappers currently use `f64`. Native `f32` models, derivatives,
   and all claimed backend versions are explicit supply gates.
8. NIST's available inputs have different residual conventions and published
   rounding. Nelson's response transformation and Roszman1's arctangent branch
   need dedicated checks. Parameter standard deviations do not define numerical
   certificate uncertainty. Input extraction does not certify model code.
9. Identical paired outcomes give a degenerate bootstrap interval. The protocol
   now substitutes a conservative zero-event bound and extends shortlisted
   development policies from 30 to 100 fixed seeds. Thirty seeds screen
   policies; full confidence checks precede finalist selection. Shared
   transforms reuse bootstrap replicate indices to preserve dependence.
   Intervals describe individual comparisons, not a simultaneous catalogue-wide
   guarantee.

## Gates for step 5 and calibration

  | Gate                             | Required evidence                                                                                                                                                                                                                        | Blocks                                                                     |
  | -------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------- |
  | G401: Executable cases           | Expanded domain/start/transform/derivative manifests, inherited family IDs, native scalar/backend coverage, no duplicate weighting, and solver applicability reconciliation.                                                             | Affected solver/variant sweeps.                                            |
  | G402: Independent verifier       | Analytic checks for all quality measures, objective conventions, scale transforms, solution-set distance, constrained multiplier residuals, and exact/flat root accuracy.                                                                | Any use of target success for policy selection.                            |
  | G403: Precision certificates     | Reference intervals, independent native-precision witnesses, uncertainty bounds, and frozen target eligibility for each case/derivative/precision.                                                                                       | Affected target scores; broad claims where an entire family is ineligible. |
  | G404: NIST model adapters        | All 27 formulas, observations, both starts, analytic Jacobians, RSS checks, published rounding, and independent reference refinement where needed.                                                                                       | NIST pilot interpretation and corpus calibration.                          |
  | G405: Accounting and stages      | Tested reconciliation of logical counts with physical calls; fused/finite-difference calls; rejected trials; initialization and mid-inner budget exhaustion; coherent returned points.                                                   | Work profiles and stopping interpretation.                                 |
  | G406: Candidate/variant audit    | Close applicable step 2 paths and step 3 gates: L-BFGS-B/SLSQP reference branches, line-search outcomes, CMA genotype/phenotype scales, chain state, constrained inner capabilities, and DIRECT zero behavior or its declared exclusion. | Corresponding candidate sweeps.                                            |
  | G407: Precalibration freeze      | Committed source/lockfile/input hashes, exact commands, schema, complete seed list, policy configurations, scales, targets, budgets, exclusions, and pilot report.                                                                       | All large sweeps.                                                          |
  | G408: Holdout and generalization | Freeze finalists before candidate evaluation; satisfy minimum distinct-family coverage and verify all claimed backends/precisions.                                                                                                       | Default recommendation and broad coverage claim.                           |

No gate requires waiting for the issue reporter. Supply NIST adapters and
missing fixtures locally using the documented project workflow. This review does
not close the remaining inventory audit, harness work, calibration, or
implementation.

## Verification of this change

- Reconciled all 28 public corpus specifications with the family table and all
  27 NIST dataset names with the source index and manifest. Both supplied starts
  are present for every dataset, totaling 54 primary starts and 2176
  observations.
- Checked snapshot SHA-256 hashes, parameter/start dimensions, numeric data row
  counts, predictor dimensions, decimal reference strings, family partitions,
  and normalization against the downloaded source bytes.
- Checked protocol seed derivation, transformed objective allowances, full-cap
  treatment of early failures, scalar example brackets, and the unit-circle KKT
  certificate with small analytic calculations. These are design checks, not
  solver runs or certified precision-floor measurements.
- Ran Markdown formatting/lint, local file/anchor validation, and
  `git diff --check`. Commands and starting revision are in the [session
  handoff](README.md#s011-define-and-review-the-protocol-2026-10-05).

Remaining limitations are substantive: reliability margins are prespecified
judgment calls; the finite suite cannot establish universal scale invariance;
very tight targets may be unattainable; sparse and custom-inner variants need
honest capability coverage; and stochastic noninferiority can remain unresolved
at the fixed sample sizes. Preserve these limitations in later decisions.
