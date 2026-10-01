# First-order and constrained reference survey

Status: versioned reference comparison and draft candidate policies for the
first-order, Newton, and SLSQP names outside the [four-family
pilot](candidate-pilot.md). This selects no default. The [source
inventory](inventory.md) describes Basin's current stopping rules. The external
values below are `f64` reference anchors; none is an automatic `f32` default.

## Reference policies

SciPy 1.16.2 uses `‖g_k‖∞ <= 1e-5` by default for both
[`BFGS`](https://github.com/scipy/scipy/blob/v1.16.2/scipy/optimize/_optimize.py)
and
[`CG`](https://docs.scipy.org/doc/scipy-1.16.2/reference/optimize.minimize-cg.html).
The BFGS implementation tests the initial gradient and then each new gradient
after a line-search step. Its optional `xrtol`, zero by default, stops after a
step when `‖α_k p_k‖₂ <= xrtol * (xrtol + ‖x_(k+1)‖₂)`. An unsuccessful line
search or non-finite result has a separate exit; neither establishes
stationarity. SciPy's [BFGS
documentation](https://docs.scipy.org/doc/scipy-1.16.2/reference/optimize.minimize-bfgs.html)
explicitly cautions that single-precision numerical gradients may need a looser
`gtol`, giving `1e-3` as an example. Its CG source uses a Polak–Ribière+ update
and a Wolfe line search; this is a closer algorithmic comparator for Basin's
Polak–Ribière+ mode than for its Hager–Zhang mode. SciPy CG has no default step
or cost-change convergence test.

SciPy 1.16.2's
[`trust-ncg`](https://github.com/scipy/scipy/blob/v1.16.2/scipy/optimize/_trustregion.py)
driver stops when the Euclidean norm of the model gradient is **below** its
`gtol=1e-4` default. It checks the seed and the model rebuilt after each
accepted step. A rejected trial changes the radius and does not satisfy the
gradient check through an unchanged point. Its `eta=0.15` acceptance threshold
and `max_trust_radius=1000` are algorithm controls; subproblem completion and
radius exhaustion are separate. This reference matches Basin's matrix-free
`TrustRegion` more closely than its exact-Hessian Dogleg or More–Sorensen modes,
but the model, forcing sequence, and failure handling still need comparison.

[Ceres Solver
2.2.0](https://github.com/ceres-solver/ceres-solver/blob/2.2.0/docs/source/nnls_solving.rst)
uses `‖x - Π(x ⊞ -g(x))‖∞ <= 1e-10` for its default gradient tolerance, where
`Π` projects onto bounds and `⊞` applies its manifold update. For an unbounded
Euclidean problem this becomes `‖g‖∞ <= 1e-10`; Basin's optional
`GradientDescent`, `Bfgs`, `NonlinearCg`, and `TrustRegion` gradient check is
instead `‖g‖₂ <= tolerance`. Ceres also documents a relative function test
`|ΔF|/F <= 1e-6` and a parameter test `‖Δx‖ <= (‖x‖ + 1e-8) * 1e-8`. These tests
apply to its nonlinear least-squares driver with its own acceptance rules. The
cost ratio is undefined at zero cost without an implementation guard, and
neither formula can be transferred to Basin's shared observed checks by matching
a tolerance name. Ceres allows up to five consecutive invalid trust-region steps
and a minimum radius of `1e-32` by default; these are no-progress safeguards,
not quality tests.

SciPy 1.16.2's [`SLSQP`
documentation](https://docs.scipy.org/doc/scipy-1.16.2/reference/optimize.minimize-slsqp.html)
uses one `ftol=1e-6` accuracy parameter for Lagrangian-gradient, summed
constraint-violation, step, and objective-change checks. Its wrapper passes this
value as `acc` to the [`slsqp`
implementation](https://github.com/scipy/scipy/blob/v1.16.2/scipy/optimize/_slsqp_py.py).
The public prose does not establish a conjunction of all four quantities, or the
exact composition of the Fortran tests. Basin's Kraft-style composite test also
defaults to `1e-6`; the same number does not establish equivalent stopping
because its feasibility gate, QP directional quantity, line search, and
Hessian-reset relaxation need a branch-by-branch comparison. NLopt 2.10.0
describes its [`SLSQP`
implementation](https://github.com/stevengj/nlopt/blob/v2.10.0/doc/docs/NLopt_Algorithms.md)
as a modified Kraft implementation with NLopt's own termination tests. It is
another comparator, not a single canonical `ftol` formula.

NLopt 2.10.0's
[`general stop API`](https://github.com/stevengj/nlopt/blob/v2.10.0/doc/docs/NLopt_Reference.md)
combines enabled outer tests with OR and leaves them disabled by default. Its
`ftol_rel` uses objective change times an absolute function-value scale, while
`xtol_rel` uses a weighted parameter-change measure. Their precise observation
can vary by algorithm. NLopt treats a roundoff-limited exit separately from a
tolerance result. This supports keeping Basin's execution budgets and numerical
stalls distinct from solver convergence. It does not justify a universal cost or
step tolerance for first-order methods.

## Candidate policies to measure

Each row retains the current Basin default as control. `a_g`, `r_g`, and `a_p`
are finite nonnegative candidate thresholds, calibrated by problem scale and
precision. Unless stated otherwise, an approximate test runs at the initialized
point and each coherent accepted point with a finite objective and full
gradient. If several checks are enabled, they compose with OR. A zero threshold
means exact zero, whereas `None` disables the check. Do not count line-search
failure, trust-region rejection, or exhausted budget as convergence.

  | Basin solver and variants                                            | Candidate comparison                                                                                                                                                                                                                     | Essential separation                                                                                                                                                                                                                        |
  | -------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
  | `GradientDescent`, fixed or searched step, with and without momentum | Budget-driven control versus `‖g_k‖₂ <= a_g`; also test `‖g_k‖₂ <= r_g ‖g_0‖₂` on scaled objectives.                                                                                                                                     | Momentum changes the accepted parameter after the searched point. Verify the gradient belongs to the published parameter before evaluating a new default. A small step or cost change alone can reflect a poor fixed step or failed search. |
  | `ProjectedGradientDescent`, fixed or searched step                   | Budget-driven control versus `‖x_k - Π(x_k-g_k)‖∞ <= a_p`, using the current box and accepted point.                                                                                                                                     | Test projected starts and active or fixed bounds separately. A custom search evaluates an unprojected trial; verify descent and stationarity again after projection.                                                                        |
  | `NonlinearCg`, Polak–Ribière+ and Hager–Zhang                        | Exact-zero-gradient control versus accepted-point `‖g_k‖₂ <= a_g`; measure a relative-initial-gradient alternative separately.                                                                                                           | SciPy's `1e-5` infinity-norm anchor is a comparator only for the Polak–Ribière+ stratum. Restart and line-search failure remain separate.                                                                                                   |
  | `Bfgs`, default and custom line searches                             | Budget-driven control versus accepted-point `‖g_k‖₂ <= a_g`; compare an infinity-norm test at SciPy's `1e-5` anchor only if implemented across all claimed backends.                                                                     | Study an optional guarded accepted-step test separately. Do not turn failed Wolfe search or a skipped curvature update into convergence.                                                                                                    |
  | `TrustRegion`, exact-Hessian and matrix-free strategies              | Budget-driven control versus `‖g_k‖₂ <= a_g`; use SciPy's `1e-4` as one `f64` matrix-free anchor.                                                                                                                                        | Stratify Steihaug, Cauchy, Dogleg, and More–Sorensen where applicable. Record rejected trials, inner residual/radius exits, and outer model stalls independently.                                                                           |
  | `Sgd`, plain and momentum, by full-cost refresh period               | Retain the budget-driven policy as the first candidate. Measure returned-point quality independently at fixed budgets; specify a replicated full-gradient or validation criterion only after finding a defensible noise-aware reference. | A mini-batch gradient is noisy and Basin publishes a full-cost point only on refresh. Repeated unchanged publication is not stationarity. Do not make raw observed cost or step change a default without replicated noise-aware evidence.   |
  | `GaussNewton`, ordinary and robust loss                              | Current `‖Jᵀr‖∞ <= 1e-8` control versus calibrated absolute thresholds for the same tested objective gradient.                                                                                                                           | A small residual is sufficient but not necessary at a nonzero-residual minimum. Robust and ordinary gradients need separate reference and finite-difference checks; Ceres' `1e-10` is an unbounded `f64` anchor, not an equivalent default. |
  | `Slsqp`, inequalities, equalities, and active boxes                  | Current composite `1e-6` control versus calibrated composite accuracy, and a stricter candidate requiring primal violation `<= ctol` **AND** a computed KKT residual `<= ktol`.                                                          | A tolerance on feasibility alone does not prove stationarity. The KKT candidate needs a documented multiplier estimate and an explicit AND/OR rule; keep it provisional until that diagnostic exists.                                       |

The candidate set still lacks an algorithm-author reference for the specific
Hager–Zhang implementation, an exact SciPy/NLopt SLSQP branch comparison, a
reference for projected gradient with a projected line search, and a noise-aware
stopping source for SGD. Resolve those gaps before declaring this family's
reference field complete or starting a large sweep. The [protocol](protocol.md)
must also fix independently checked targets and `f32` floors before thresholds
are selected.
