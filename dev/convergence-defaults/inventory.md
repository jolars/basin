# Solver inventory

Status: API reconciliation complete at
`3ae7be6977334cd50554217c4f3f013f4cb6bf1c`; behavior audit in progress. This
inventory copies the agreed coverage from the [TODO
plan](../../TODO.md#convergence-defaults-investigation). Numerical references,
candidates, experiments, decisions, implementation, and verification remain
pending unless a record below says otherwise.

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

## Public API reconciliation

At the audited revision, the [crate-root
exports](../../crates/basin/src/lib.rs), the public [solver
module](../../crates/basin/src/solver.rs), the public [root
module](../../crates/basin/src/root.rs), and the [web
catalogue](../../web/src/routes/docs/solvers/+page.svx) cover the same 47 solver
names in the coverage seed. All are available at the crate root and in their
defining public modules. The `solver` module directly re-exports 40 of its 42
names; `Lbfgs` and `Lbfgsb` are reached through `solver::lbfgs`. `Lbfgsb`,
`MaLsChCma`, and `MaLsChSw` are type aliases; the other names denote concrete
solver types. The catalogue also mentions `RootBracketer` and
`MinimumBracketer`, which are bracket searches, and the line-search strategies,
which are solver dependencies. The exported strategy, result, error, and
type-state names are not additional solvers.

This reconciles public names, not stopping behavior. Repeat the comparison at
the final revision so a solver added during the investigation cannot be missed.

## Variant register

Each row below needs a separate stopping record or an explicit proof that its
stop is identical to the parent record. The list is a source-backed starting
point, not a completed variant audit.

  | Solver family                                 | Public variants and current stopping distinction                                                                                                                                                             | Source                                                                                                                                 |
  | --------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------- |
  | `NelderMead`                                  | Unbounded and projected; standard, adaptive, and custom coefficients. Both modes have all native tests disabled by default. Projection can collapse vertices on a bound.                                     | [nelder_mead.rs](../../crates/basin/src/solver/nelder_mead.rs)                                                                         |
  | `Lbfgs` / `Lbfgsb`                            | `Lbfgs::new()` and `Lbfgsb` are the bounded mode; `unbounded()` is a separate mode. Bounded has a default projected-gradient test; unbounded has no default gradient test. Both accept custom line searches. | [lbfgs.rs](../../crates/basin/src/solver/lbfgs.rs)                                                                                     |
  | `LevenbergMarquardt` / `LevenbergMarquardtQr` | Normal-equations and pivoted-QR implementations each accept Nielsen or trust-region damping. QR forwards the convergence settings to the same LM engine; the rank cutoff is a numerical safeguard.           | [levenberg_marquardt.rs](../../crates/basin/src/solver/levenberg_marquardt.rs)                                                         |
  | `Trf` / `TrustRegionReflective`               | Separate implementations and stop paths. `Trf` is the legacy bounded-LM algorithm; `TrustRegionReflective` has a radius, reflected trials, and fixed-coordinate elimination.                                 | [trf.rs](../../crates/basin/src/solver/trf.rs), [trust_region_reflective.rs](../../crates/basin/src/solver/trust_region_reflective.rs) |
  | `Mads`                                        | Unbounded, box-bounded, and progressive-barrier constrained modes have separate `Solver` implementations and state types.                                                                                    | [mads.rs](../../crates/basin/src/solver/mads.rs)                                                                                       |
  | `TrustRegion`                                 | Exact-Hessian and matrix-free modes; `Steihaug`, `CauchyPoint`, `Dogleg`, and `MoreSorensen` subproblem strategies have distinct inner completion rules.                                                     | [trust_region.rs](../../crates/basin/src/solver/trust_region.rs)                                                                       |
  | `NonlinearCg`                                 | Hager–Zhang and Polak–Ribière+ updates; line search and periodic restart settings may alter progress or failure.                                                                                             | [nonlinear_cg.rs](../../crates/basin/src/solver/nonlinear_cg.rs)                                                                       |
  | `De`                                          | Mutation, crossover, and dithering settings alter generation behavior; check whether they share the outer stop.                                                                                              | [de.rs](../../crates/basin/src/solver/de.rs)                                                                                           |
  | `GlobalBestPso`                               | Boundary handling and velocity-limit policies can change population progress; check outer stop separately.                                                                                                   | [global_best_pso.rs](../../crates/basin/src/solver/global_best_pso.rs)                                                                 |
  | `SimulatedAnnealing`                          | Cooling schedule and optional reannealing triggers; distinguish a schedule restart from termination.                                                                                                         | [simulated_annealing.rs](../../crates/basin/src/solver/simulated_annealing.rs)                                                         |
  | Composed solvers                              | `BasinHopping`, `CmaInject`, `BoundedCmaInject`, `DeInject`, `MaLsCh`, `MaLsChCma`, `MaLsChSw`, `BarrierMethod`, and `AugmentedLagrangianMethod` depend on the configured inner solver's stop and budget.    | [solver.rs](../../crates/basin/src/solver.rs)                                                                                          |

## Initial stopping records

These are source observations, not proposed defaults or accepted numerical
decisions. The shared opt-in observed cost and step checks need their own
formula and observation-stage audit before any record is complete.

  | Solver or mode                                              | Current native stop and default, for `f32` and `f64`                                                                                                                        | Stage and distinction                                                                                                                    | Reference, candidate, experiment, decision, delivery                                                       |
  | ----------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
  | `NelderMead` unbounded and projected                        | No native convergence test enabled; optional absolute simplex size and cost spread require both thresholds. Standard, adaptive, and custom coefficients share this default. | Whole simplex after initialization or an iteration; projected vertices are clamped before evaluation.                                    | Reference comparison, candidate policy, experiment, disposition, implementation, and verification pending. |
  | `Lbfgs` bounded / `Lbfgsb` alias                            | `‖x - projection(x - g)‖∞ ≤ 1e-10`; opt-in observed step and cost checks combine with OR.                                                                                   | Current point at the top of an iteration; the alias uses exactly this implementation.                                                    | Fortran v3.0 named in rustdoc; precise comparison and all later stages pending.                            |
  | `Lbfgs` unbounded                                           | No native gradient test enabled; optional absolute or relative Euclidean gradient norm.                                                                                     | Current iterate; custom line searches have their own acceptance or failure rules.                                                        | Nocedal–Wright reference named in rustdoc; precise comparison and all later stages pending.                |
  | `LevenbergMarquardt` normal equations, Nielsen damping      | `‖Jᵀr‖∞ ≤ 1e-8`. Orthogonality, relative model reduction, relative trial step, and radius tests disabled.                                                                   | Gradient before a trial; opt-in model and step tests use trial diagnostics, including rejected trials.                                   | MINPACK and Madsen et al. named in rustdoc; precise comparison and all later stages pending.               |
  | `LevenbergMarquardt` normal equations, trust-region damping | Same enabled default; opt-in relative radius test is active only in this damping mode.                                                                                      | Radius test uses the updated radius; a small radius alone does not establish fit accuracy.                                               | MINPACK named in rustdoc; precise comparison and all later stages pending.                                 |
  | `LevenbergMarquardtQr`, both damping modes                  | Forwards the same convergence settings to LM; default `‖Jᵀr‖∞ ≤ 1e-8`.                                                                                                      | Pivoted QR changes the linear solve and rank safeguard; damping still determines radius-test availability.                               | MINPACK named in rustdoc; precise comparison and all later stages pending.                                 |
  | `Trf`                                                       | `maxᵢ abs((Jᵀr)ᵢ)·abs(vᵢ) ≤ 1e-8`; opt-in observed cost and step checks combine with OR.                                                                                    | Coleman–Li bound scaling in the legacy bounded-LM implementation.                                                                        | Branch–Coleman–Li named in rustdoc; precise comparison and all later stages pending.                       |
  | `TrustRegionReflective`                                     | Same named scaled-gradient formula and `1e-8` default, computed over free coordinates; all-fixed termination is structural.                                                 | Initial point and accepted iterates; observed checks exclude rejected inner trials. Rank cutoff and inner-attempt limits are safeguards. | SciPy 1.16.2 comparison named in rustdoc; precise comparison and all later stages pending.                 |

## Per-solver evidence tracker

This tracker gives every public name an evidence slot. `Source checked` means
only that the initial native-stop observation above was read from source; it
does not close the reference, variant, or numerical audit. `Pending` means no
claim of completion. Aliases retain their own rows because callers see those
names and may configure different modes through the defining type.

  | Public name                 | Defining source                                                                                | Current stop                                                | Reference match | Candidate | Experiment | Decision | Implementation | Verification |
  | --------------------------- | ---------------------------------------------------------------------------------------------- | ----------------------------------------------------------- | --------------- | --------- | ---------- | -------- | -------------- | ------------ |
  | `Brent`                     | [brent.rs](../../crates/basin/src/solver/brent.rs)                                             | Pending                                                     | Pending         | Pending   | Pending    | Pending  | Pending        | Pending      |
  | `BrentDerivative`           | [brent_derivative.rs](../../crates/basin/src/solver/brent_derivative.rs)                       | Pending                                                     | Pending         | Pending   | Pending    | Pending  | Pending        | Pending      |
  | `GoldenSection`             | [golden_section.rs](../../crates/basin/src/solver/golden_section.rs)                           | Pending                                                     | Pending         | Pending   | Pending    | Pending  | Pending        | Pending      |
  | `BrentRoot`                 | [brent.rs](../../crates/basin/src/root/brent.rs)                                               | Pending                                                     | Pending         | Pending   | Pending    | Pending  | Pending        | Pending      |
  | `SecantRoot`                | [secant.rs](../../crates/basin/src/root/secant.rs)                                             | Pending                                                     | Pending         | Pending   | Pending    | Pending  | Pending        | Pending      |
  | `NewtonRoot`                | [newton.rs](../../crates/basin/src/root/newton.rs)                                             | Pending                                                     | Pending         | Pending   | Pending    | Pending  | Pending        | Pending      |
  | `HalleyRoot`                | [halley.rs](../../crates/basin/src/root/halley.rs)                                             | Pending                                                     | Pending         | Pending   | Pending    | Pending  | Pending        | Pending      |
  | `Toms748Root`               | [toms748.rs](../../crates/basin/src/root/toms748.rs)                                           | Pending                                                     | Pending         | Pending   | Pending    | Pending  | Pending        | Pending      |
  | `GradientDescent`           | [gradient_descent.rs](../../crates/basin/src/solver/gradient_descent.rs)                       | Pending                                                     | Pending         | Pending   | Pending    | Pending  | Pending        | Pending      |
  | `ProjectedGradientDescent`  | [projected_gradient_descent.rs](../../crates/basin/src/solver/projected_gradient_descent.rs)   | Pending                                                     | Pending         | Pending   | Pending    | Pending  | Pending        | Pending      |
  | `NonlinearCg`               | [nonlinear_cg.rs](../../crates/basin/src/solver/nonlinear_cg.rs)                               | Pending                                                     | Pending         | Pending   | Pending    | Pending  | Pending        | Pending      |
  | `Bfgs`                      | [bfgs.rs](../../crates/basin/src/solver/bfgs.rs)                                               | Pending                                                     | Pending         | Pending   | Pending    | Pending  | Pending        | Pending      |
  | `Lbfgs`                     | [lbfgs.rs](../../crates/basin/src/solver/lbfgs.rs)                                             | Source checked; [variants above](#initial-stopping-records) | Pending         | Pending   | Pending    | Pending  | Pending        | Pending      |
  | `Lbfgsb`                    | [lbfgs.rs](../../crates/basin/src/solver/lbfgs.rs)                                             | Source checked; [variants above](#initial-stopping-records) | Pending         | Pending   | Pending    | Pending  | Pending        | Pending      |
  | `TrustRegion`               | [trust_region.rs](../../crates/basin/src/solver/trust_region.rs)                               | Pending                                                     | Pending         | Pending   | Pending    | Pending  | Pending        | Pending      |
  | `Sgd`                       | [sgd.rs](../../crates/basin/src/solver/sgd.rs)                                                 | Pending                                                     | Pending         | Pending   | Pending    | Pending  | Pending        | Pending      |
  | `GaussNewton`               | [gauss_newton.rs](../../crates/basin/src/solver/gauss_newton.rs)                               | Pending                                                     | Pending         | Pending   | Pending    | Pending  | Pending        | Pending      |
  | `LevenbergMarquardt`        | [levenberg_marquardt.rs](../../crates/basin/src/solver/levenberg_marquardt.rs)                 | Source checked; [variants above](#initial-stopping-records) | Pending         | Pending   | Pending    | Pending  | Pending        | Pending      |
  | `LevenbergMarquardtQr`      | [levenberg_marquardt.rs](../../crates/basin/src/solver/levenberg_marquardt.rs)                 | Source checked; [variants above](#initial-stopping-records) | Pending         | Pending   | Pending    | Pending  | Pending        | Pending      |
  | `Trf`                       | [trf.rs](../../crates/basin/src/solver/trf.rs)                                                 | Source checked; [variants above](#initial-stopping-records) | Pending         | Pending   | Pending    | Pending  | Pending        | Pending      |
  | `TrustRegionReflective`     | [trust_region_reflective.rs](../../crates/basin/src/solver/trust_region_reflective.rs)         | Source checked; [variants above](#initial-stopping-records) | Pending         | Pending   | Pending    | Pending  | Pending        | Pending      |
  | `NelderMead`                | [nelder_mead.rs](../../crates/basin/src/solver/nelder_mead.rs)                                 | Source checked; [variants above](#initial-stopping-records) | Pending         | Pending   | Pending    | Pending  | Pending        | Pending      |
  | `Newuoa`                    | [newuoa.rs](../../crates/basin/src/solver/newuoa.rs)                                           | Pending                                                     | Pending         | Pending   | Pending    | Pending  | Pending        | Pending      |
  | `Bobyqa`                    | [bobyqa.rs](../../crates/basin/src/solver/bobyqa.rs)                                           | Pending                                                     | Pending         | Pending   | Pending    | Pending  | Pending        | Pending      |
  | `Lincoa`                    | [lincoa.rs](../../crates/basin/src/solver/lincoa.rs)                                           | Pending                                                     | Pending         | Pending   | Pending    | Pending  | Pending        | Pending      |
  | `Cobyla`                    | [cobyla.rs](../../crates/basin/src/solver/cobyla.rs)                                           | Pending                                                     | Pending         | Pending   | Pending    | Pending  | Pending        | Pending      |
  | `Mads`                      | [mads.rs](../../crates/basin/src/solver/mads.rs)                                               | Pending                                                     | Pending         | Pending   | Pending    | Pending  | Pending        | Pending      |
  | `SolisWets`                 | [solis_wets.rs](../../crates/basin/src/solver/solis_wets.rs)                                   | Pending                                                     | Pending         | Pending   | Pending    | Pending  | Pending        | Pending      |
  | `Slsqp`                     | [slsqp.rs](../../crates/basin/src/solver/slsqp.rs)                                             | Pending                                                     | Pending         | Pending   | Pending    | Pending  | Pending        | Pending      |
  | `BarrierMethod`             | [barrier_method.rs](../../crates/basin/src/solver/barrier_method.rs)                           | Pending                                                     | Pending         | Pending   | Pending    | Pending  | Pending        | Pending      |
  | `AugmentedLagrangianMethod` | [augmented_lagrangian_method.rs](../../crates/basin/src/solver/augmented_lagrangian_method.rs) | Pending                                                     | Pending         | Pending   | Pending    | Pending  | Pending        | Pending      |
  | `Direct`                    | [direct.rs](../../crates/basin/src/solver/direct.rs)                                           | Pending                                                     | Pending         | Pending   | Pending    | Pending  | Pending        | Pending      |
  | `Gbnm`                      | [gbnm.rs](../../crates/basin/src/solver/gbnm.rs)                                               | Pending                                                     | Pending         | Pending   | Pending    | Pending  | Pending        | Pending      |
  | `RandomSearch`              | [random_search.rs](../../crates/basin/src/solver/random_search.rs)                             | Pending                                                     | Pending         | Pending   | Pending    | Pending  | Pending        | Pending      |
  | `SimulatedAnnealing`        | [simulated_annealing.rs](../../crates/basin/src/solver/simulated_annealing.rs)                 | Pending                                                     | Pending         | Pending   | Pending    | Pending  | Pending        | Pending      |
  | `CmaEs`                     | [cma_es.rs](../../crates/basin/src/solver/cma_es.rs)                                           | Pending                                                     | Pending         | Pending   | Pending    | Pending  | Pending        | Pending      |
  | `BoundedCmaEs`              | [bounded_cma_es.rs](../../crates/basin/src/solver/bounded_cma_es.rs)                           | Pending                                                     | Pending         | Pending   | Pending    | Pending  | Pending        | Pending      |
  | `De`                        | [de.rs](../../crates/basin/src/solver/de.rs)                                                   | Pending                                                     | Pending         | Pending   | Pending    | Pending  | Pending        | Pending      |
  | `GlobalBestPso`             | [global_best_pso.rs](../../crates/basin/src/solver/global_best_pso.rs)                         | Pending                                                     | Pending         | Pending   | Pending    | Pending  | Pending        | Pending      |
  | `Ssga`                      | [ssga.rs](../../crates/basin/src/solver/ssga.rs)                                               | Pending                                                     | Pending         | Pending   | Pending    | Pending  | Pending        | Pending      |
  | `BasinHopping`              | [basin_hopping.rs](../../crates/basin/src/solver/basin_hopping.rs)                             | Pending                                                     | Pending         | Pending   | Pending    | Pending  | Pending        | Pending      |
  | `CmaInject`                 | [cma_inject.rs](../../crates/basin/src/solver/cma_inject.rs)                                   | Pending                                                     | Pending         | Pending   | Pending    | Pending  | Pending        | Pending      |
  | `BoundedCmaInject`          | [bounded_cma_inject.rs](../../crates/basin/src/solver/bounded_cma_inject.rs)                   | Pending                                                     | Pending         | Pending   | Pending    | Pending  | Pending        | Pending      |
  | `DeInject`                  | [de_inject.rs](../../crates/basin/src/solver/de_inject.rs)                                     | Pending                                                     | Pending         | Pending   | Pending    | Pending  | Pending        | Pending      |
  | `MaLsCh`                    | [ma_ls_ch.rs](../../crates/basin/src/solver/ma_ls_ch.rs)                                       | Pending                                                     | Pending         | Pending   | Pending    | Pending  | Pending        | Pending      |
  | `MaLsChCma`                 | [ma_ls_ch_cma.rs](../../crates/basin/src/solver/ma_ls_ch_cma.rs)                               | Pending                                                     | Pending         | Pending   | Pending    | Pending  | Pending        | Pending      |
  | `MaLsChSw`                  | [ma_ls_ch_sw.rs](../../crates/basin/src/solver/ma_ls_ch_sw.rs)                                 | Pending                                                     | Pending         | Pending   | Pending    | Pending  | Pending        | Pending      |

Repeat this tracker against the public API at the final revision.

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
