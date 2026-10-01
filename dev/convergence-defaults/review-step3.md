# Step 3 candidate and evidence review

Reviewed 2026-10-01 against the 47 public solver names in the
[inventory](inventory.md#per-solver-evidence-tracker). This review closes the
**survey and candidate specification**, not a choice of new defaults. The source
records and versioned or dated primary references are linked below. Where a
precise comparable reference does not exist, the record states that limit and
supplies a budget-driven control rather than borrowing a threshold from a
different algorithm. Every candidate remains provisional until the
[protocol](protocol.md), measurements, and independent validation establish the
accuracy and work tradeoff.

  | Family                               | Public names                                                                                                                                 | Reference and candidate record                                    | Review disposition                                                                                                                                                                                                                                                  |
  | ------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
  | Pilot, 7                             | `NelderMead`, `Lbfgs`, `Lbfgsb`, `LevenbergMarquardt`, `LevenbergMarquardtQr`, `Trf`, `TrustRegionReflective`                                | [Reference](reference-pilot.md), [candidates](candidate-pilot.md) | Keep method and mode distinctions, especially bounded versus unbounded L-BFGS and legacy `Trf` versus full TRF. Candidates are ready for protocol design; original L-BFGS-B branch behavior and precision floors still need checks.                                 |
  | First-order and SLSQP, 8             | `GradientDescent`, `ProjectedGradientDescent`, `NonlinearCg`, `Bfgs`, `TrustRegion`, `Sgd`, `GaussNewton`, `Slsqp`                           | [First-order record](reference-first-order.md)                    | Compare accepted-point gradients with the stated 2- or infinity-norm, and keep line-search/subproblem failures distinct. SGD needs replicated quality rather than a raw one-step convergence claim; SLSQP's stricter KKT candidate requires multiplier diagnostics. |
  | Scalar, 8                            | `Brent`, `BrentDerivative`, `GoldenSection`, `BrentRoot`, `SecantRoot`, `NewtonRoot`, `HalleyRoot`, `Toms748Root`                            | [Scalar record](reference-scalar.md)                              | Keep exact-zero root exits and bracket-width exits distinct. Compare `f32` and `f64` representable floors and evaluate position error independently of residual.                                                                                                    |
  | Derivative-free local, 6             | `Newuoa`, `Bobyqa`, `Lincoa`, `Cobyla`, `Mads`, `SolisWets`                                                                                  | [Derivative-free record](reference-derivative-free.md)            | Distinguish native radius-schedule completion from an earlier radius check. Feasibility must accompany constrained floor completion; stochastic Solis–Wets remains a budget-driven control.                                                                         |
  | Global and population, 9             | `Direct`, `Gbnm`, `RandomSearch`, `SimulatedAnnealing`, `CmaEs`, `BoundedCmaEs`, `De`, `GlobalBestPso`, `Ssga`                               | [Global record](reference-global.md)                              | A small rectangle or distribution describes resolution, not global error. Compare population-spread and stall heuristics only against equal-work, paired-seed controls; distinguish genotype from clipped phenotype.                                                |
  | Constrained adapters and composed, 9 | `BarrierMethod`, `AugmentedLagrangianMethod`, `BasinHopping`, `CmaInject`, `BoundedCmaInject`, `DeInject`, `MaLsCh`, `MaLsChCma`, `MaLsChSw` | [Composed record](reference-composed.md)                          | Keep inner and outer stop owners, codes, and work separate. Joint feasibility/stationarity candidates need gradient-capable inner solves; all outer global searches retain budget-driven controls.                                                                  |

The six rows are disjoint and total 47. Aliases, constrained modes, trust-region
strategies, damping choices, stochastic schedules, injection frequency, and
custom inner settings retain their own strata within the linked records. The
[dependency audit](dependencies.md) shows why an accepted line-search step,
bracketer completion, or inner model stop cannot be counted as outer
convergence. Its exact exhaustion and work-accounting paths remain step 2 and
protocol tasks.

## Cross-family decisions for the experiment design

1. **Meaning of a stop.** Classify solver numerical convergence, exact-root or
   known-target success, structural no progress, callback failure, executor
   budget, application stop, and inner-only completion separately. Do not pool
   an OR branch into an AND claim. A small step or unchanged elite by itself can
   be caused by a search failure, projection, rejection, or stochastic sampling.
2. **Observation stage.** Record seed checks, accepted outer points, complete
   populations or DIRECT sweeps, rejected trial updates, and completed inner
   segments where relevant. The [pilot](reference-pilot.md),
   [first-order](reference-first-order.md),
   [derivative-free](reference-derivative-free.md), and
   [composed](reference-composed.md) records state the stages for each
   candidate. The protocol must capture the stage in every trace.
3. **Units and precision.** Preserve the published norm, objective or coordinate
   units, normalization, and `<` versus `<=` in each comparison. Calibrate `f32`
   and `f64` independently. External `f64` constants are anchors, not Basin
   defaults. Treat `None` as disabled and zero according to each actual
   implementation; [DIRECT's
   discrepancy](inventory.md#global-and-population-stopping-records) needs a
   test-first fix before a zero-threshold experiment.
4. **Work and quality.** Compare candidates at equal raw evaluation budgets and
   use quality targets independent of the stopping predicate. For stochastic
   methods, use paired seeds and summarize failure rates and quality
   distributions. For composed solvers, sum outer and inner work and report both
   termination reasons. The [protocol](protocol.md) must fix the exact
   accounting and success rules before a large sweep.

## Evidence gaps and gates

These gaps were reviewed, not silently filled by a same-named external
tolerance. They are prioritized by whether they can reverse a candidate's
interpretation.

  | Priority  | Gap                                                                                                                                                                                 | Required resolution before a large sweep                                                                                                                    |
  | --------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
  | Gate      | Independent, scale-aware quality and feasibility targets, attainable `f32` floors, case mix, paired seeds, and complete inner/outer work accounting.                                | Finish and review the [protocol](protocol.md); run focused measurement checks first.                                                                        |
  | Gate      | `BarrierMethod` gap without centering and `AugmentedLagrangianMethod` feasibility without stationarity can be false success; custom inner solvers change the available certificate. | Choose explicit diagnostic formulas and accepted inner capability strata before testing joint stops. Keep current policies as controls.                     |
  | Gate      | `Direct` zero-tolerance setter behavior contradicts its implementation; candidate geometry is dimension dependent.                                                                  | Fix with a regression test or remove zero from the candidate grid; stratify length and volume by dimension.                                                 |
  | Gate      | Original L-BFGS-B cost/gradient branch composition, SciPy/NLopt SLSQP branches, and line-search warning or exhaustion paths can change what an apparent stop means.                 | Perform focused source comparisons and outcome tests before interpreting these families' broad sweep results.                                               |
  | Gate      | CMA-ES genotype TolX, clipped phenotype quality, path diagnostics, and MA-LS-Ch per-segment TolX use different scales.                                                              | Capture all scales and chain states in pilot traces; do not reuse a threshold across them by name.                                                          |
  | Follow-up | Exact comparator for GBNM, `RandomSearch`, `GlobalBestPso`, `Ssga`, and custom injection and chain variants is absent or algorithmically different.                                 | Retain the budget-driven control and label stall/spread options as exploratory. Search specific implementation sources if an early stop survives the pilot. |
  | Follow-up | Noise-aware SGD, robust-loss Gauss–Newton, projected line searches, Hager–Zhang variant matching, and Solis–Wets finite-run policies lack a complete transferable reference.        | Test independent diagnostics in focused pilots, and narrow or reject any candidate that cannot be justified for the actual variant.                         |

The candidate set is ready for **protocol review and focused pilots**. It is not
yet ready for a large sweep or a default change. No acceptance threshold,
candidate policy, or solver disposition was approved in this review.
