# Solver inventory

Status: scope seed only. The stopping-behavior audit in step 2 has not begun.
This seed copies the agreed coverage from the [TODO
plan](../../TODO.md#convergence-defaults-investigation) at the [starting
revision](README.md). All defaults and dispositions remain pending.

## Coverage seed

The 47 names below include aliases. They are not a claim of 47 independent
algorithms or an exhaustive list of configurable variants.

  | Family                                | Solver names                                                                                                   | Audit status |
  | ------------------------------------- | -------------------------------------------------------------------------------------------------------------- | ------------ |
  | Scalar minimization                   | `Brent`, `BrentDerivative`, `GoldenSection`                                                                    | Pending      |
  | Scalar roots                          | `BrentRoot`, `SecantRoot`, `NewtonRoot`, `HalleyRoot`, `Toms748Root`                                           | Pending      |
  | First-order, quasi-Newton, and Newton | `GradientDescent`, `ProjectedGradientDescent`, `NonlinearCg`, `Bfgs`, `Lbfgs`, `Lbfgsb`, `TrustRegion`, `Sgd`  | Pending      |
  | Least squares                         | `GaussNewton`, `LevenbergMarquardt`, `LevenbergMarquardtQr`, `Trf`, `TrustRegionReflective`                    | Pending      |
  | Derivative-free local                 | `NelderMead`, `Newuoa`, `Bobyqa`, `Lincoa`, `Cobyla`, `Mads`, `SolisWets`                                      | Pending      |
  | Other constrained                     | `Slsqp`, `BarrierMethod`, `AugmentedLagrangianMethod`                                                          | Pending      |
  | Global and population                 | `Direct`, `Gbnm`, `RandomSearch`, `SimulatedAnnealing`, `CmaEs`, `BoundedCmaEs`, `De`, `GlobalBestPso`, `Ssga` | Pending      |
  | Composed                              | `BasinHopping`, `CmaInject`, `BoundedCmaInject`, `DeInject`, `MaLsCh`, `MaLsChCma`, `MaLsChSw`                 | Pending      |

Reconcile this seed against the [public exports](../../crates/basin/src/lib.rs),
the [solver module](../../crates/basin/src/solver.rs), the [root
module](../../crates/basin/src/root.rs), and the [web
catalogue](../../web/src/routes/docs/solvers/+page.svx). Check public submodules
for APIs absent from the root re-exports. Repeat this reconciliation before
closing the investigation so later additions are included.

## Record required for each solver

Create one record per solver, with additional records for variants whose
stopping behavior differs. Link aliases to their defining implementation; record
both public entry points. Do not infer a formula from a setter's name.

  | Field            | Evidence to record                                                                            |
  | ---------------- | --------------------------------------------------------------------------------------------- |
  | Identity         | Public name, variant, defining source, and audited revision                                   |
  | Applicability    | Problem traits, constraints, scalar types, backends, and feature versions                     |
  | Current defaults | Enabled tests and actual values for `f32` and `f64`, including implicit stops                 |
  | Mathematics      | Exact formulas, norms, scaling, absolute/relative meanings, and AND/OR composition            |
  | Observation      | Initialization, trial or accepted step, generation, inner solve, and returned-point semantics |
  | Ownership        | Solver convergence, algorithm controls, numerical safeguards, and executor limits             |
  | Lifecycle        | Fresh initialization, warm starts, exact continuation, and diagnostic history                 |
  | Dependencies     | Line search, bracketing, subproblem, or inner-solver settings that affect stopping            |
  | References       | Versioned primary sources, algorithm match, and unresolved differences                        |
  | Candidates       | Proposed formulas and settings, rationale, and relevant decision IDs                          |
  | Experiments      | Applicable cases, calibration/validation status, and run records                              |
  | Disposition      | Pending, retain defaults, change defaults, or retain reliance on budgets, with evidence       |
  | Delivery         | Implementation, verification, migration guidance, and remaining work                          |

Keep audit, reference, experiment, decision, and delivery status separate. A
completed inventory record does not mean its numerical decision is settled. Use
an explicit reason for fields that do not apply.

## Variants and dependencies to check

Inspect constrained modes, trust-region strategies, damping and factorization
choices, stochastic updates, and composed inner solvers. In particular, keep
`Lbfgs` and `Lbfgsb`, both LM factorizations, and `Trf` and
`TrustRegionReflective` visible as separate entries.

Audit the public [line searches](../../crates/basin/src/line_search.rs),
[bracketers](../../crates/basin/src/bracket.rs), and subproblem tolerances as
dependencies. Their acceptance and completion tests have different purposes from
convergence of the enclosing solve. Record failure propagation and work
accounting for composed methods.

A solver's coverage is complete only when its applicable variants have a
supported disposition, validation, and delivered documentation. No solver is
complete at this stage.
