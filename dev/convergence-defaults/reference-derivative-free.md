# Derivative-free local reference policies

Status: versioned reference comparison and draft candidates for `Newuoa`,
`Bobyqa`, `Lincoa`, `Cobyla`, `Mads`, and `SolisWets`. The [Nelder–Mead
pilot](candidate-pilot.md#neldermead) covers the seventh name in this family.
This record selects no Basin default. The [source
inventory](inventory.md#derivative-free-local-stopping-records) supplies the
current rules and variants.

## Powell-model methods

[PDFO 1.3](https://pdfo.github.io/docs.html#options), developed from Powell's
Fortran solvers by Ragonneau and Zhang, calls `rhoend` the final trust-region
radius and defaults it to `1e-6`; its unscaled `rhobeg` default is `1`. The
radius belongs to the coordinates used by the solver. PDFO's optional bound
scaling maps all finite box coordinates to `[-1,1]` before applying the radius,
so equal `rhoend` values can describe different physical precision. Its
`ftarget` is an objective target for a feasible point, and its `maxfun` is an
evaluation budget. They are not radius tests.

The algorithm-author [PRIMA v0.7.1
constants](https://github.com/libprima/prima/blob/v0.7.1/fortran/common/consts.F90)
also set `RHOBEG_DFT=1` and `RHOEND_DFT=1e-6` for the four relevant solvers. Its
[`NEWUOA`
loop](https://github.com/libprima/prima/blob/v0.7.1/fortran/newuoa/newuob.f90),
[`BOBYQA`
loop](https://github.com/libprima/prima/blob/v0.7.1/fortran/bobyqa/bobyqb.f90),
[`LINCOA`
loop](https://github.com/libprima/prima/blob/v0.7.1/fortran/lincoa/lincob.f90),
and [`COBYLA`
loop](https://github.com/libprima/prima/blob/v0.7.1/fortran/cobyla/cobylb.f90)
report `SMALL_TR_RADIUS` only at a final-radius completion branch. Merely
setting the current schedule radius to `rhoend` is not that completion event.
Each loop has short-step, insufficient-model-reduction, geometry, and radius
updates before the final stop. PRIMA separates evaluation-budget and target
codes from the radius code. Basin likewise has an optional
`with_absolute_radius_tolerance` checked at an initialized or updated iteration
boundary; enabling it at `rhoend` can stop before the native final stage
finishes. Treat it as a distinct candidate, not an equivalent spelling of
`with_final_radius`.

For BOBYQA, PRIMA's
[preprocessor](https://github.com/libprima/prima/blob/v0.7.1/fortran/common/preproc.f90)
can revise the radii to fit a narrow box. Basin also adjusts its effective radii
for narrow boxes. For LINCOA and COBYLA, PRIMA v0.7.1 has `CTOL_DFT=sqrt(ε)` for
deciding whether the returned point is feasible in its selection logic. The
default has a precision dependence; Basin's current radius-only completion has
no matching feasibility gate.

[SciPy 1.16.2
COBYLA](https://github.com/scipy/scipy/blob/v1.16.2/scipy/optimize/_cobyla_py.py)
uses a PRIMA-derived Python implementation with `rhobeg=1`, `tol=1e-4` as its
final radius, `maxiter=1000` as an evaluation budget, and default
`catol=sqrt(ε_float64)` in the wrapper. After the solve, its public `success` is
false if returned maximum constraint violation exceeds `catol`; otherwise
success requires a small-radius or feasible-target stop code. Thus its `tol` and
`catol` compose with **AND** for a small-radius success result, while a feasible
objective target is a separate **OR** success branch. An exhausted budget is not
successful convergence even if it returns a useful point. [NLopt
2.10.0](https://github.com/stevengj/nlopt/blob/v2.10.0/doc/docs/NLopt_Algorithms.md)
documents adapted COBYLA, BOBYQA, and NEWUOA variants with NLopt's general outer
stopping controls. Its bound support and internal model changes prevent
one-to-one interpretation of an equal radius or `xtol` name.

## MADS and Solis–Wets

[NOMAD
4.5.0](https://nomad-4-user-guide.readthedocs.io/en/v.4.5.0/HowToUseNomad.html)
distinguishes the poll frame size `Δ_k` from mesh size `δ_k`, supports
coordinate-specific `MIN_FRAME_SIZE` and `MIN_MESH_SIZE`, and offers `ORTHO 2N`
as one direction strategy. Its [parameter
list](https://nomad-4-user-guide.readthedocs.io/en/v.4.5.0/Appendix.html) lists
no numerical default for either minimum-size termination setting. NOMAD's
extreme barrier rejects infeasible points, while its progressive barrier can use
infeasible intermediate points. Basin's scalar poll-size floor `1e-6` and its
`2n` polling strategy are related but not equivalent to NOMAD's separate
mesh/frame and per-coordinate rules. Do not copy a NOMAD size threshold by name.
In Basin's progressive-barrier mode, reaching the poll floor does not alone
establish feasibility.

[Solis and Wets (1981)](https://doi.org/10.1287/moor.6.1.19) study convergence
of random-search methods, but an asymptotic convergence result does not supply a
finite-run certificate. Basin's `SolisWets` has no enabled approximate stop; its
optional `rho <= tolerance` measures the mutation standard deviation, not
objective error. A rejected proposal changes adaptation history without
publishing a better point. The paper is an algorithmic research reference, not a
source for a transferable numerical default. The exact finite-run stopping rule
of the particular bias adaptation used by Basin still needs an
implementation-level reference check.

## Candidate policies to measure

Retain each native Basin default as control. Candidate absolute radii and poll
sizes must be calibrated in problem coordinates separately for `f32` and `f64`.
Native schedule completion, opt-in early size checks, feasibility, objective
targets, and budgets require separate diagnostic codes. Do not count a model
failure or a non-finite callback as convergence.

  | Solver and mode                                  | Candidate comparison                                                                                                                       | Observation and required safeguard                                                                                                                                                                            |
  | ------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
  | `Newuoa`                                         | Current `rhoend=1e-6` schedule versus nearby final radii; compare optional `rho_k <= a_r` as a distinct early-stop policy.                 | Record radius stage, model predicted reduction, accepted/rejected trial, and returned objective. A short step at `rhoend` is not automatically objective accuracy.                                            |
  | `Bobyqa`                                         | Same radius comparisons, stratified by narrow and wide boxes.                                                                              | Record configured and effective radii. Check returned box feasibility and quality, including active and fixed coordinates; do not ascribe work differences to stopping when preprocessing changed the radius. |
  | `Lincoa`                                         | Same radius comparisons; separately evaluate `final-radius completion AND maximum linear violation <= ctol`.                               | Report an infeasible final-radius result as a distinct outcome. Establish the scale and precision of `ctol` before proposing it as a default.                                                                 |
  | `Cobyla`, inequality-only and folded constraints | Current radius completion versus calibrated radii; separately evaluate `final-radius completion AND maximum constraint violation <= ctol`. | Study feasible objective-target exits independently. Folded equalities use two inequalities; check their original equality residual too. Model failure and infeasible completion stay visible.                |
  | `Mads`, unbounded and box modes                  | Current `poll_size_min=1e-6` versus calibrated scalar poll floors; compare optional earlier poll-size stop separately.                     | Include extreme-barrier rejected polls, active bounds, and fixed coordinates. A small poll size measures local resolution, not a proven optimum.                                                              |
  | `Mads`, progressive barrier                      | Same poll-size grid, with an additional `poll-floor completion AND violation <= ctol` candidate.                                           | Keep infeasible selected states and feasibility progress separate from cost progress. If the floor arrives first, report an infeasible/no-progress outcome rather than convergence.                           |
  | `SolisWets`                                      | Budget-driven control versus optional `rho <= a_r`, with the latter described as search-scale exhaustion.                                  | Use paired seeds and repeated-run quality at equal evaluation budgets. Record success/failure streaks and bias; zero accepted movement is not stationarity.                                                   |

Reference gaps before a large sweep: PRIMA/Basin model-path and evaluation
accounting differences; the exact constraint violation norm and success
composition for LINCOA and Basin's folded COBYLA; NOMAD's scalar-to-vector size
mapping and default run limits; and an implementation-level Solis–Wets
finite-run policy. The [experimental protocol](protocol.md) must set feasible
quality targets and coordinate scales before thresholds can be selected.
