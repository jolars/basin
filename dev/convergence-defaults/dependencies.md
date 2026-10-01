# Stopping dependencies

These are source observations for step 2 of the [investigation](README.md).
Their checks select a trial step, finish a bracket search, or solve an inner
model. None alone certifies convergence of the enclosing solver. Reference
comparisons, candidate policies, and validation remain pending.

## Line searches

The [line-search contract](../../crates/basin/src/line_search.rs) distinguishes
`Step` from `Failed`. The default `next_with_outcome` wraps the legacy `next`
result in `Step`; the default bounded search returns `Failed` without a trial.
Problem calls inside a search count through `Problem`. An accepted evaluation
can be returned to avoid repeating it in the caller.

  | Search                                                               | Current acceptance and default budget                                                                                                                                                                                             | Exhaustion and relevant mode                                                                                                                                                                                                                                                              |
  | -------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
  | [`Constant`](../../crates/basin/src/line_search.rs)                  | Returns its configured step without evaluating a condition. No default step or trial budget.                                                                                                                                      | Bounded mode caps a positive finite step at the supplied maximum and fails on an invalid configured step.                                                                                                                                                                                 |
  | [`Backtracking`](../../crates/basin/src/line_search/backtracking.rs) | Armijo decrease; initial step 1, reduction factor 0.5, coefficient `1e-4`, at most 50 cost trials.                                                                                                                                | On exhaustion, `next` returns the next reduced step without evaluating it. The default outcome wrapper calls this `Step`, including in bounded mode. See [Q007](decisions.md#open-questions).                                                                                             |
  | [`Wolfe`](../../crates/basin/src/line_search/wolfe.rs)               | Armijo and strong curvature; `c1=1e-4`, `c2=0.9`, initial step 1, upper step 10, at most 25 outer/zoom trials.                                                                                                                    | Search or zoom can return `Failed`. Bounded mode restricts trials and may accept sufficient decrease at the feasible upper step.                                                                                                                                                          |
  | [`MoreThuente`](../../crates/basin/src/line_search/more_thuente.rs)  | Armijo and strong curvature; `ftol=1e-3`, `gtol=0.9`, relative bracket width `xtol=0.1`, initial step 1, step interval `[0, 1e10]`, at most 20 evaluations.                                                                       | Rounding, bracket-width, or step-bound warnings return a step; exhausted evaluations return the bracket's best step. Invalid initial slope or step returns zero, which callers must classify. Bounded mode has a separate failure path. A warning step need not satisfy both Wolfe tests. |
  | [`HagerZhang`](../../crates/basin/src/line_search/hager_zhang.rs)    | Wolfe or approximate Wolfe; `delta=0.1`, `sigma=0.9`, relaxation `epsilon=1e-6`, interpolation `theta=0.5`, contraction `gamma=0.66`, initial step 1, step interval `[epsilon(F), 1e5]`, growth `rho=5`, at most 100 evaluations. | Precision, bound, or budget exhaustion can return `Failed`. Bounded mode may accept an Armijo-decreasing upper endpoint with negative slope.                                                                                                                                              |

## Bracketers

The shared [search configuration](../../crates/basin/src/bracket.rs) has growth
factor 2, at most 1000 expansion rounds, and no bounds by default. Bracketer
results report `Bracketed`, `BoundsReached`, `NoProgress`, or `MaxIter`; the
last three do not establish a bracket. Their evaluation counts are search work,
not the enclosing root or minimization solver's iteration count.

  | Search                                                          | Completion test and other exits                                                                                                                                                                   |
  | --------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
  | [`RootBracketer`](../../crates/basin/src/bracket/root.rs)       | Two distinct points with opposite signs, or an exact root at an endpoint. It expands both sides unless a bound or numerical stagnation stops them; the configured round cap can stop first.       |
  | [`MinimumBracketer`](../../crates/basin/src/bracket/minimum.rs) | Three ordered points whose middle cost is no greater than both neighbors and strictly less than at least one. It expands toward the lower-cost side until bracketed, bounded, stalled, or capped. |

## Model and constrained subproblems

  | Owner                                                                                            | Inner completion and default                                                                                                                                                                                            | Outer interpretation                                                                                                    |
  | ------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
  | [`TrustRegion` with `CauchyPoint`](../../crates/basin/src/solver/trust_region.rs)                | Closed-form steepest-descent model step, clipped to the radius. No inner tolerance or iteration budget.                                                                                                                 | A model step, including a zero step, is not an outer convergence certificate.                                           |
  | [`TrustRegion` with `Dogleg`](../../crates/basin/src/solver/trust_region/dogleg.rs)              | Full SPD Newton step if interior; otherwise a point on the two-leg path. Falls back to Cauchy when the SPD solve fails. No inner tolerance.                                                                             | The fallback changes step quality, not the outer stopping test.                                                         |
  | [`TrustRegion` with `Steihaug`](../../crates/basin/src/solver/trust_region/steihaug.rs)          | Truncated CG stops at boundary, nonpositive curvature, exact-zero residual, or residual below `min(0.5, ‖g‖^0.5)‖g‖`; default cap is dimension `n`. Forcing parameters and cap are configurable.                        | Residual and cap govern model accuracy and Hessian-product work. A cap stop is not outer convergence.                   |
  | [`TrustRegion` with `MoreSorensen`](../../crates/basin/src/solver/trust_region/more_sorensen.rs) | Up to 50 trials each to establish a feasible shift and solve the secular equation; accepts a radius residual scaled by `100·epsilon(F)`, with endpoint, hard-case, bracket-collapse, and Cauchy fallback paths.         | Inner precision and fallback affect model quality and radius updates, not outer stationarity.                           |
  | [LM trust-region damping](../../crates/basin/src/solver/levenberg_marquardt/damping.rs)          | At most 50 attempts by default through LM's `max_inner_attempts`. Accepts an undamped feasible step or a damped step whose radius ratio is within 10%; otherwise can retain the best feasible regularized step or fail. | Damping search completion is distinct from LM's gradient, cost, and step convergence tests.                             |
  | [`Slsqp` NNLS and merit search](../../crates/basin/src/solver/slsqp.rs)                          | Default NNLS active-set cap is three times its column count; the inexact L1 merit search permits eleven trials.                                                                                                         | Subproblem failure and unsuccessful globalization must be accounted separately from the native composite accuracy test. |

The derivative-free model solvers also have internal trust-region work. Their
outer radius and step observations are recorded in the
[inventory](inventory.md#derivative-free-local-stopping-records); a deeper
subproblem-work audit is still needed before experiment design.
