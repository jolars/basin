# Constrained and composed reference policies

Status: reference comparison and draft candidates for `BarrierMethod`,
`AugmentedLagrangianMethod`, `BasinHopping`, `CmaInject`, `BoundedCmaInject`,
`DeInject`, `MaLsCh`, `MaLsChCma`, and `MaLsChSw`.
[`Slsqp`](reference-first-order.md) has its own reference pass. The [source
inventory](inventory.md#constrained-and-composed-stopping-records) supplies
current formulas and defaults. None of the candidates below is a selected Basin
default.

## Constrained outer solves

The [Boyd and Vandenberghe *Convex Optimization*
book](https://web.stanford.edu/~boyd/cvxbook/), §11.3–11.4, gives the
log-barrier central-path gap `m/t` after a centered subproblem, with `m`
inequalities and barrier weight `μ = 1/t`. Basin's Phase II stops at
`m*μ <= 1e-8` by default after an inner solve at that `μ`; Phase I seeks a
strict interior and has its own `1e-8` gap scale for declaring that it could not
find one. Both measure absolute objective units. The textbook gap bound requires
the convexity and sufficiently centered subproblem assumptions; for nonconvex
objectives or a truncated inner run, `m*μ` alone is a schedule measure. A
feasible point with a small gap parameter can still have an unacceptably large
stationarity residual. Each inner run has a default 50-iteration cap, and its
stop reason must be retained. Phase II optional cost and step checks observe
accepted outer points only. Phase I failure, loss of strict feasibility, a
non-finite callback, and inner failure are separate from outer convergence.

[SciPy 1.16.2
`trust-constr`](https://docs.scipy.org/doc/scipy-1.16.2/reference/optimize.minimize-trustconstr.html)
uses a different interior-point/trust-region algorithm, so its numerical
threshold is a comparator rather than a direct default. It requires **both**
Lagrangian-gradient infinity norm and maximum constraint violation below
`gtol=1e-8` for its gradient success test; with inequalities, termination also
requires its barrier parameter below `barrier_tol=1e-8`. Its trust-radius
`xtol=1e-8` is a separate criterion. Basin's `m*μ` has objective units and is
not equal to SciPy's barrier-parameter test or to a KKT residual. A candidate
joint stop for Basin must state `m*μ <= a_gap` **AND** strict feasibility
**AND** a scale-aware inner stationarity test, with the caveat that
derivative-free custom inners cannot supply that gradient. This composition
would need a separate outcome for a small gap with unconverged inner solve.

Basin's `AugmentedLagrangianMethod` follows Nocedal and Wright, *Numerical
Optimization*, second edition, §17.3: it minimizes
`L_ρ(x,λ) = f(x) + λᵀ(Ax-b) + (ρ/2)‖Ax-b‖₂²`, then updates multipliers or the
penalty according to feasibility progress. Its sole enabled outer test is
`‖Ax-b‖₂ <= 1e-8` after a complete surrogate inner solve. This is an unscaled
absolute residual, and it does not establish approximate stationarity if the
inner 50-iteration budget stopped first. [SciPy's `trust-constr`
documentation](https://docs.scipy.org/doc/scipy-1.16.2/reference/optimize.minimize-trustconstr.html)
provides a useful independent **AND** comparator with Lagrangian gradient and
feasibility, though its algorithm and norm differ. A candidate Basin test is
`‖Ax-b‖₂ <= a_feas` **AND** `‖∇f(x)+Aᵀλ‖∞ <= a_stat` at the completed outer
point; for an inexact surrogate solve, also record `‖∇_x L_ρ(x,λ)‖∞` using the
multiplier value that defined that surrogate. State explicitly which multiplier
estimate the observation uses. A custom inner without gradients cannot use this
branch. Preserve a feasibility-only control and report its stationarity
independently. Test row scaling and redundant or inconsistent constraints before
assigning a common `a_feas`.

## Hops, injections, and local-search chains

[SciPy 1.16.2
`basinhopping`](https://docs.scipy.org/doc/scipy-1.16.2/reference/generated/scipy.optimize.basinhopping.html)
defaults to `niter=100` outer hops, hence 101 local minimizations, and disables
its best-unchanged `niter_success` stop by default. The optional rule stops when
the global-best candidate remains the same for a specified number of hops. SciPy
describes repeated runs as a consistency check, not a global certificate. Basin
`BasinHopping` likewise has no native outer optimality test. Its initial local
solve and every later hop use a fresh inner solve, default cap 1000 iterations.
A clean inner budget stop may provide a usable candidate; inner convergence does
not imply the outer landscape search has converged. Rejected hops do not observe
cost or step change, whereas a best-unchanged executor stall can count them if
configured. These are different observation sequences, so a SciPy-style count
must be specified explicitly before comparison.

`CmaInject` and `BoundedCmaInject` add one local refinement per default outer
generation and delegate their outer stop to `CmaEs` and `BoundedCmaEs`,
respectively. Thus the parent [CMA distribution
comparison](reference-global.md#distribution-and-population-spread) applies to
their TolX, disabled by default. Their fresh inner runs default to 50
iterations, but can use a custom solver with different convergence settings. An
inner stop is consumed as candidate production, not an outer optimality event.
The bounded injection also clips phenotypes and ranks penalized genotypes. A
local refinement can change the outer distribution's trajectory, so compare the
same TolX threshold with and without injection and charge inner evaluations to
total work. No separate published reference implementation establishes a
universal injection-specific stop.

`DeInject` delegates outer termination to [Basin's
`De`](reference-global.md#distribution-and-population-spread): no native
optimality test, with a default 50-iteration fresh inner refinement each
generation. It writes back only strict improvements, and a clean inner
termination does not stop DE. [Noman and Iba
(2008)](https://doi.org/10.1109/TEVC.2007.895272) study DE with adaptive local
search, but their simplex-crossover inner algorithm and adaptive search length
are different from Basin's generic inner. Compare inner stopping choices at the
same **total** raw evaluation budget. Include a control that refines less often,
because an inner threshold can change the balance of global exploration and
local work without changing outer stopping semantics.

[Molina et al. (2010)](https://doi.org/10.1162/evco.2010.18.1.18102) introduce
local-search chains that resume stored local searches instead of restarting each
selected basin. Basin's generic `MaLsCh` has no native outer numerical stop and
defaults to an inner segment budget of 300 cost evaluations. A clean segment
budget or operator-tolerance stop is consumed; a hard inner failure propagates.
Insufficient improvement can discard the chain and cause a later fresh seed, not
outer convergence. The alias `MaLsChCma` keeps CMA history across segments and,
unless overridden, applies a per-segment TolX of `1e-12 * starting_sigma`
alongside that budget. This threshold has parameter units and is reset to the
segment's starting scale; do not equate it to a single run-wide CMA TolX.
`MaLsChSw` resumes Solis–Wets bias and radius, but its segments have no default
tolerance and are budget-driven. The [MA-SW-Chains
paper](https://doi.org/10.1109/CEC.2010.5586034) is an algorithmic reference,
not an outer convergence certificate. For each alias, record the operator's
segment stop, whether the chain survived, the incumbent change, and the outer
budget result separately.

## Candidate policies to measure

Keep the current outer and inner settings as controls. The two constrained
methods need independent feasible-quality targets, scaled residuals, and
`f32`/`f64` calibration. A `1e-8` absolute threshold can be below useful
single-precision accuracy after badly scaled arithmetic. For all composed
methods, count inner and outer evaluations in the same six raw categories,
retain inner termination codes, and compare at equal total work and paired
seeds. Treat a custom inner's capabilities and settings as part of the solver
variant. Executor budgets, application stops, and objective targets remain
separate **OR** exits from any solver-specific condition.

  | Solver or mode                  | Candidate comparison                                                                                                                                            | Observation and safeguard                                                                                                                                                                                                               |
  | ------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
  | `BarrierMethod`, Phase I and II | Current `m*μ <= 1e-8` control versus calibrated gap grid; compare `gap AND strict feasibility AND inner stationarity` where a gradient-based inner supports it. | Observe the completed centered subproblem before shrinking `μ`. Record Phase I interior margin, Phase II slack, inner gradient norm and code, and `m*μ`; separate a small schedule gap from poor centering and infeasible failure.      |
  | `AugmentedLagrangianMethod`     | Current `‖Ax-b‖₂ <= 1e-8` control versus calibrated feasibility grid; compare `feasibility AND KKT stationarity` for gradient-capable inners.                   | Observe after a full surrogate solve, with the multiplier stage specified. Record row-scaled and raw residuals, surrogate gradient, true Lagrangian gradient, and inner code. Feasible but unstationary is a distinct outcome.          |
  | `BasinHopping`                  | Budget-driven control versus optional outer best-unchanged hop count, with the count defined over completed hops; vary inner precision separately.              | Include initial and hop-local work, accepted/rejected hops, and historical best. Compare paired seeds at equal total evaluations; inner convergence and repeated local basin visits are not global convergence.                         |
  | `CmaInject`, `BoundedCmaInject` | Parent CMA TolX grid with and without injection; vary inner stop and 50-iteration cap separately.                                                               | Observe outer live distribution after completed generations and inner final code per injected candidate. For bounded injection record genotype, clipped phenotype, penalty, and raw objective; no inner code alone stops the outer run. |
  | `DeInject`                      | Parent DE budget/spread candidates with and without injection; vary refinement frequency and inner tolerance separately.                                        | Observe the full post-generation population and total work. Separate an unaccepted inner candidate, clean inner budget stop, hard inner failure, and outer result.                                                                      |
  | `MaLsCh`, generic operator      | Budget-driven outer control, with optional executor best-improvement stall; vary inner segment budget and operator tolerance as distinct controls.              | Record resumed versus fresh chain, segment termination, chain discard, and historical best. Specify the custom operator before pooling outcomes.                                                                                        |
  | `MaLsChCma`                     | Same outer control; compare default per-segment `1e-12 * starting_sigma` TolX against calibrated segment-scale alternatives and budget-only chains.             | Record segment start sigma, live axis scale, segment budget overshoot, and retained history. A TolX stop belongs to that chain segment, not the outer search.                                                                           |
  | `MaLsChSw`                      | Same outer control; optional Solis–Wets radius threshold studied only as chain-scale exhaustion.                                                                | Record resumed bias, radius, accepted improvements, and segment budget. A small mutation radius or no-improvement segment is not outer optimality.                                                                                      |

Evidence gaps before a large sweep: feasibility and stationarity scales for both
constrained methods; whether `BarrierMethod` can expose a reliable
centered-inner certificate for every supported custom inner; the exact
multiplier stage and derivative accounting for a composite ALM KKT check; custom
inner outcome matrices for all injections and hops; and reference implementation
matching for `MaLsCh` segment stopping. Define independent quality, feasibility,
and work measures in the [protocol](protocol.md) before calibrating any
threshold.
