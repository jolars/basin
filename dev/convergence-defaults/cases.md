# Case register for CDP-1

The [protocol](protocol.md) fixes quality, work, and selection rules. This
register fixes family membership and partitions before tuning. `D` means
development and `V` means held-out validation. Existing problem definitions were
reconciled against all 28 entries in
[`ALL_SPECS`](../../crates/basin/src/problems.rs) on 2026-10-05. Corpus metadata
supplies useful properties but does not provide a complete benchmark manifest,
reference certificate, or native `f32` implementation.

## Corpus families and applicability

All named models, their residual/boxed wrappers, dimensions, starts,
constraints, and transformed copies stay in the same family. Grouping quadratic
models conservatively prevents a rotated or constrained copy from leaking into
holdout. A problem's plotting domain is a search domain, not an implicit
constraint on an unconstrained solve. Freeze actual bounds per
constrained/global case.

  | Family ID                 | Corpus models                                                                                 | Partition | Main strata                                                                                                                                                                                 |
  | ------------------------- | --------------------------------------------------------------------------------------------- | --------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
  | `quadratic`               | Sphere, Booth, Matyas, ConstrainedQuadratic, EqualityConstrainedQuadratic, SparseLeastSquares | D         | Convex, coupled/separable, box/linear equality/inequality constraints, zero/nonzero residual, conditioned/rank-deficient linear systems. Sparse variants only on supported sparse backends. |
  | `rosenbrock`              | Rosenbrock and RosenbrockResiduals                                                            | D         | Curved valleys, dimensionality, smooth cost/residual forms, alternate minima.                                                                                                               |
  | `powell-singular`         | PowellSingular                                                                                | V         | Four parameters, singular Jacobian at the zero-residual minimum.                                                                                                                            |
  | `exponential-fit`         | ExponentialFit                                                                                | D         | Two-parameter nonlinear regression, zero/nonzero residual.                                                                                                                                  |
  | `beale`                   | Beale                                                                                         | V         | Two-parameter smooth nonquadratic objective.                                                                                                                                                |
  | `goldstein-price`         | GoldsteinPrice and Picheny                                                                    | D         | Same landscape under the documented coordinate/objective transform; one family.                                                                                                             |
  | `rastrigin`               | Rastrigin                                                                                     | D         | Separable multimodal search and synthetic noisy evaluation.                                                                                                                                 |
  | `ackley`                  | Ackley                                                                                        | D         | Multimodal search, nonsmooth optimum; no smooth stationarity claim there.                                                                                                                   |
  | `levy`                    | Levy                                                                                          | V         | Multimodal nonseparable search, including noisy evaluation.                                                                                                                                 |
  | `styblinski-tang`         | StyblinskiTang                                                                                | D         | Multiple wells, local versus global quality.                                                                                                                                                |
  | `himmelblau`              | Himmelblau                                                                                    | V         | Multiple equivalent minimizers; distance to their set.                                                                                                                                      |
  | `three-hump-camel`        | ThreeHumpCamel                                                                                | V         | Smooth nonconvex polynomial.                                                                                                                                                                |
  | `mccormick`               | McCormick                                                                                     | D         | Smooth nonconvex objective and bounded domain.                                                                                                                                              |
  | `schaffer`                | SchafferN2 and SchafferN4                                                                     | V         | Related oscillatory models; differentiability differs.                                                                                                                                      |
  | `bukin`                   | BukinN6                                                                                       | D         | Nonsmooth valley.                                                                                                                                                                           |
  | `oscillatory-exponential` | CrossInTray and HolderTable                                                                   | D         | Related nonsmooth oscillatory/exponential landscapes.                                                                                                                                       |
  | `easom`                   | Easom                                                                                         | V         | Narrow basin and flat distant regions.                                                                                                                                                      |
  | `eggholder`               | Eggholder                                                                                     | V         | Nonsmooth multimodal bounded search.                                                                                                                                                        |
  | `degenerate`              | Step and Zero                                                                                 | D         | Plateaus, nonunique answers, and immediate/exact stops; safeguard fixtures, excluded from aggregate accuracy/work scores.                                                                   |

For scalable vector models use dimensions `2, 10, 50` when mathematically
supported. Fixed-dimensional models keep their declared dimension. Linear
least-squares fixtures use `m = 2*n + 1`, singular-value condition numbers
`1, 1e4, 1e8`, and a rank-deficient case with one zero singular value. Retain
larger condition numbers in `f32` reports even when their targets are
ineligible. For nonlinear models record singular values of the scaled Jacobian
or Hessian at the reference; label a singular problem separately rather than
inventing a finite condition number. ExponentialFit uses `m = 21`, `t_i = i/20`,
`(a_ref, b_ref) = (2, -1)`, and starts `(1, 0)` and `(4, -2)`; add fixed data
perturbations `0.01 * sin(i)` as a separate nonzero-residual instance whose
reference must be independently solved.

For each other base case use its documented literature start when available,
plus three starts at coordinate fractions `0.25, 0.5, 0.75` of the recorded
search domain. If no domain exists, use `o_i + c*s_i` for `c = -1, 0.5, 2`.
Deduplicate equal starts. Population methods generate their initial populations
with pinned solver defaults and paired seeds in that same domain; record the
whole initial population. Do not start scored runs at the known optimum. An
accidentally optimal start becomes a separate initialization fixture, and the
scored start uses fraction `0.375`. Reference-at-start tests remain mandatory
diagnostics. NIST uses its own two starts instead of this rule. Freeze
coordinates/domain sources before pilot outcome comparisons.

Reference-basin membership for deterministic local selection is established by
analytic convexity/uniqueness or by independent reference solves from the fixed
start with matching local values and curvature checks. Other starts still enter
all-reference target profiles and alternate-minimum reports, but cannot be
relabeled in response to a candidate outcome. A solver/variant needs at least
three development families and two validation families with eligible designated
targets before a broad vector-solver recommendation. Scalar methods require two
development and two validation families. A narrower coverage claim must identify
its limitation explicitly; additional related starts do not supply new families.

## NIST StRD inputs and model families

[Issue #109](https://github.com/jolars/basin/issues/109) and its visible
comments were checked 2026-10-05. No reporter harness or attachment was
available there. The fallback is an independent assembly of all 27 datasets from
the [NIST nonlinear regression
index](https://www.itl.nist.gov/div898/strd/nls/nls_main.shtml). The [input
manifest](nist/manifest.json) records original URLs, retrieval date, SHA-256
hashes, dimensions, both starts, certified parameters, RSS, and partitions. The
neighboring `.dat` snapshots retain the model, observations, and printed
precision, with line endings and trailing whitespace normalized as recorded in
the manifest. They define 54 primary starts; the certified parameters are
additional initialization diagnostics and do not increase the primary count.

  | Family ID                  | Datasets                                   | Partition |
  | -------------------------- | ------------------------------------------ | --------- |
  | `nist-misra`               | Misra1a, Misra1b, Misra1c, Misra1d, BoxBOD | D         |
  | `nist-chwirut`             | Chwirut1, Chwirut2                         | V         |
  | `nist-exponential-mixture` | Lanczos1, Lanczos2, Lanczos3, MGH17        | D         |
  | `nist-gaussian`            | Gauss1, Gauss2, Gauss3, Eckerle4           | V         |
  | `nist-power`               | DanWood, Bennett5                          | D         |
  | `nist-rational`            | Kirby2, Hahn1, MGH09, Thurber              | V         |
  | `nist-nelson`              | Nelson                                     | D         |
  | `nist-roszman`             | Roszman1                                   | V         |
  | `nist-periodic`            | ENSO                                       | V         |
  | `nist-logistic`            | Rat42, Rat43                               | D         |
  | `nist-mgh10`               | MGH10                                      | D         |

These groupings keep related response models and dataset variants together. They
are not NIST difficulty labels: report lower/average/higher difficulty
separately. Both starts enter each applicable solver comparison. Run cost-only
adapters for Nelder–Mead, analytic cost/gradient adapters for L-BFGS, and
residual/ Jacobian adapters for LM and unbounded-capable TRF modes. Apply
finite-difference variants where supported. Do not add bounds that silently
change the published problem to accommodate a bounded solver; a derived bounded
case needs a separate case ID and reference, retaining its family/partition.

Step 5 must implement and verify the 27 models, data parsing, analytic
Jacobians, objective conventions, and reference rounding intervals. Nelson fits
a transformed response; Roszman1 requires the published arctangent branch and
its stated value of pi. Check every implementation's RSS at the published
parameters and derivative checks away from the optimum. Source data and start
extraction do not yet validate an executable model. Use printed rounding
intervals, never parameter standard deviations, as numerical reference
uncertainty. Published best-available optima are not proofs that all starts
reach the same minimum. Retain the original 54-run report as a separate
reproduction table; family weighting governs selection.

Treat identifiable NIST and ExponentialFit instances as parameter-recovery
strata requiring the parameter test. Verify rank and known model symmetries
independently before classifying an instance as nonidentifiable.

## Missing applicable cases and how to supply them

  | Gap and family                                                        | Partition | Required case or supply action before its sweep                                                                                                                                                                                                                                                                 |
  | --------------------------------------------------------------------- | --------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
  | Scalar polynomial (`scalar-polynomial`)                               | D         | Minimize `(x-1)^2` and `(x-1)^4` on `[-2,4]`; solve `(x-1)`, `(x-1)^2`, `(x-1)^3` with root 1. Use minimum bracket `(-2,0,4)`, root bracket `[-2,4]` for odd multiplicities, Newton/Halley start 3, and secant starts 2 and 3. Even multiplicity has no sign bracket and is excluded from sign-bracket methods. |
  | Scalar rational (`scalar-rational`)                                   | D         | Root of `x/(1+x^2)` on `[-1,1]`, unbracketed start `0.5`, secant starts `0.25,0.5`; minimum of `x^2/(1+x^2)` with bracket `(-1,0.25,1)`. Reference zero.                                                                                                                                                        |
  | Scalar flat (`scalar-flat`)                                           | D         | Minimum `x^6` with bracket `(-1,0.25,1)`; root `x^5`, bracket `[-1,1]`, start `0.5`, secant starts `0.25,0.5`. These related powers form one family.                                                                                                                                                            |
  | Scalar exponential (`scalar-exponential`)                             | V         | Root `exp(x)-2`, bracket `[0,2]`, start 1, secant starts `0,1`, reference `log(2)`; minimize `exp(x)-x` with bracket `(-1,0.25,1)`, reference 0.                                                                                                                                                                |
  | Scalar trigonometric (`scalar-trigonometric`)                         | V         | Root `cos(x)-x`, bracket `[0,1]`, start 1, secant starts `0,1`, independently certified reference; minimize `1-cos(x)` with bracket `(-1,0.25,1)`, reference 0.                                                                                                                                                 |
  | Active/fixed bounds and linear constraints (`quadratic`)              | D         | Construct known KKT solutions with inactive/active bounds, one/all fixed coordinates, equality rows, and inequality rows. Include feasible and infeasible starts where the solver accepts them. Add inconsistent constraints as expected-failure diagnostics, not accuracy cases.                               |
  | Nonlinear constraints (`unit-circle`)                                 | V         | Minimize `-x1` under `sum(x_i^2) <= 1`, then equality `sum(x_i^2) = 1`, at `n=2,10`; reference `(1,0,...)`, value `-1`. Use feasible interior/boundary and supported infeasible starts. Independent analytic multipliers test feasibility, complementarity, and inner centering.                                |
  | Constrained nonquadratic coverage (existing families)                 | Inherited | Add known inactive/active bound and linear-constraint cases from Rosenbrock (D), Beale (V), and Himmelblau (V), with independently certified optima. They remain their parent families. This supplies validation for bound/linear-only solvers that cannot use the circle case.                                 |
  | Nonzero residual and rank deficiency (`quadratic`, `powell-singular`) | Inherited | For linear systems use `b = A*x_ref + e` with `A^T*e = 0` and declared residual norm; use full-rank and deficient matrices. Add a constant residual to PowellSingular as a separate nonzero-minimum instance. Do not confuse an unidentifiable direction with parameter error.                                  |
  | LM regression models (`volatility-surface`)                           | D         | Reuse SVI/SSVI definitions and starts from the existing LM model helper; group both models together. Independently check Jacobians, sign symmetry, and narrow-data identifiability. Its linear models belong to `quadratic`.                                                                                    |
  | SGD/robust loss (parent families)                                     | Inherited | Supply a finite-sum quadratic with a known full gradient and robust versions of regression models with independently solved robust references. Record batch schedule, loss/scale, and inner capabilities. Ordinary least-squares references cannot substitute.                                                  |
  | Native `f32` and missing backend/derivative implementations           | Inherited | Implement native arithmetic and truthful traits before claiming coverage. Existing corpus wrappers are predominantly `f64`; do not silently evaluate in `f64` and cast. Verify every backend version a finalist claims.                                                                                         |

The scalar polynomial and flat fixtures are related algebraically. Merge both
into `scalar-polynomial` for weighting and partitioning. The rational family
supplies the second development family; exponential and trigonometric models
supply two validation families. Affine transformations do not add families.

Supply small diagnostic fixtures and adapters in `crates/competitor-bench`
during step 5. Any new reusable corpus problem must use the project's
`add_test_problem` subagent, exactly one problem per task, with metadata,
symbolic production derivatives, supported backends, and required verification.
The downloaded NIST inputs are reproduction fixtures, not new public corpus
implementations. If a required gap remains, block that solver/variant's broad
sweep instead of dropping the case. New model families require a protocol
amendment before outcome access.

## Mandatory adversarial diagnostics

Keep these outside aggregate work rankings and run each relevant precision/mode:

- An exact initial solution; zero objective; nonzero residual with zero
  gradient; one/all fixed coordinates; and an initial guess at an active bound.
- A tiny gradient caused by objective scaling away from the solution; an offset
  that rounds distinct objective values together; and a rank-deficient Jacobian.
- A collapsed simplex/population or projected zero step at a nonstationary
  point; rejected LM/TRF trials with an unchanged accepted point; line-search
  failure; and an exhausted inner solve. Independent verification must detect
  inadequacy.
- NaN/infinity at initialization, non-finite trial evaluations with a finite
  incumbent, callback errors, invalid root brackets, absent roots, derivative
  zeros away from a root, and neighboring representable bracket endpoints.
- Budgets below initialization cost, exactly at a leaf call, during finite
  differences, and within an inner solve. Retain partial outcomes and charges.

Adversarial perturbations inherit their mathematical family's partition. Generic
lifecycle/budget tests use analytic development fixtures. Building the verifier
may check holdout formulas, but candidate performance on holdout remains sealed.
