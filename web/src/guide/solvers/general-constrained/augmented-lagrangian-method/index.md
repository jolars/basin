<script setup lang="ts">
    import References from "#lib/docs/References.vue";
</script>

# Augmented Lagrangian

`AugmentedLagrangianMethod` minimizes a scalar objective subject to linear
equalities `A x = b`. It combines multiplier estimates and a quadratic penalty
with a configurable unconstrained inner solver. The starting point may be
infeasible, and inner iterates can move freely off the constraint surface.

[API
reference](https://docs.rs/basin/latest/basin/solver/augmented_lagrangian_method/struct.AugmentedLagrangianMethod.html)

  | Property    | Requirement                                                            |
  | ----------- | ---------------------------------------------------------------------- |
  | Objective   | A differentiable scalar function, including off the constraint surface |
  | Evaluations | `CostFunction` and `Gradient`, plus a compatible `WarmStart` inner     |
  | Search      | Local continuation through augmented-Lagrangian subproblems            |
  | Constraints | A linear equality block                                                |
  | Parameters  | All four dense backends with a compatible inner                        |

## When to use it

Use this wrapper when your constraints are linear equalities and you want to
reuse an unconstrained solver. Multiplier estimates can reduce the need for an
extremely large quadratic penalty. Basin follows a simplified LANCELOT-style
outer loop from [Nocedal and Wright (2006)](#ref-nocedal-wright-2006), §17.3,
Algorithm 17.4.

The complete declaration must contain only a linear equality block. Bounds,
inequalities, and nonlinear blocks are unsupported. For mixed constraint blocks,
compare [SLSQP](../slsqp/). For linear inequalities, compare the [barrier
method](../barrier-method/). This is a local method, so a nonconvex objective
can lead to different solutions from different starts.

## How it works

For residual `c(x) = A x - b`, the surrogate is
`L_rho(x, lambda) = f(x) + lambda^T c(x) + (rho/2) * ||c(x)||^2`. The wrapper
initializes `lambda` to zero and `rho` to the initial penalty. Each outer
iteration starts a fresh inner solve at the current point with those surrogate
parameters held fixed.

If the returned equality residual norm decreases enough, the wrapper updates the
multipliers with `lambda = lambda + rho * c(x)`. Otherwise, it increases `rho`
and keeps the multipliers. The surrogate gradient is
`grad f(x) + A^T (lambda + rho * c(x))`; the outer method independently checks
it at the returned point before accepting convergence.

The penalty is finite away from the constraint surface. Inner solves can
therefore use ordinary backtracking or Wolfe line searches, and no Phase I
feasibility search is needed. The objective and gradient must be defined at the
infeasible points those solves explore.

## Example

This quadratic has its unconstrained minimum at `[2.0, 2.0]`. The equality
`x[0] + x[1] = 2.0` moves the solution to `[1.0, 1.0]`, with cost `2.0`. The
start `[0.0, 0.0]` violates the equality. The example checks both the returned
point and the selected incumbent's feasibility.

<<< ../../../../../../crates/basin/examples/solver_guides/augmented_lagrangian_method.rs

[View `augmented_lagrangian_method.rs` on GitHub](https://github.com/jolars/basin/blob/main/crates/basin/examples/solver_guides/augmented_lagrangian_method.rs)

## Practical considerations

### Configure the inner solve and penalty together

Use `with_inner_solver` with gradient descent, BFGS, unbounded L-BFGS, or a
compatible derivative-free method. Inners implement `WarmStart` and consume a
`CostFunction + Gradient` surrogate. Least-squares solvers cannot consume this
adapter because it provides no residual or Jacobian interface. The original
problem needs `Gradient` even with a derivative-free inner because the outer
method verifies stationarity independently.

Defaults are `rho0 = 10.0`, `with_rho_increase(10.0)`,
`with_feasibility_decrease(0.25)`, and `50` inner iterations per outer step. The
ratio compares consecutive equality residual norms. A larger penalty pushes
toward feasibility but can make inner minimization harder, so scale the
constraint rows and allow enough inner work before increasing it aggressively.

### Check feasibility and stationarity

`with_absolute_feasibility_tolerance` tests `||A x - b||_2` and defaults to
`1e-8`. Outer convergence also requires the surrogate-gradient infinity norm to
be at most `1e-8`, or at most `1e-3` of its norm at the inner seed. Configure
those alternatives with `with_absolute_stationarity_tolerance` and
`with_relative_stationarity_tolerance`; they combine with OR and use the same
pre-update multipliers and penalty.

`None` disables an optional test, and zero requests exact-zero comparisons.
Disabling both stationarity guards permits feasibility-only stopping. Otherwise,
a derivative-free or capped inner run still has to pass independent
verification. Optional outer cost- and step-change tests require both
feasibility and stationarity. These are numerical stopping tests, so inspect the
returned objective and residual as well as the report.

### Interpret the selected incumbent

`SelectedState` publishes the original objective and an unscaled L2 violation
report. The incumbent prefers feasible points at the configured tolerance, then
lower costs among feasible points. Before finding one, it prefers smaller
residual norms, using the objective to break exact ties. A selected point can
therefore have a higher cost than an earlier infeasible point.

Use `state.best()` for checked access and read its matching violation report.
Objective-only executor targets and stalls are unavailable for this state;
application stopping rules should inspect feasibility alongside cost.

## Implementation

The solver owns the multipliers, penalty history, and inner executor. The outer
state exposes no objective gradient. Every inner solve starts fresh, while a
fresh outer run resets multipliers and penalty history. An `ExactCheckpoint`
retains the solver, selected progress, and evaluation counts for continuation.
With `serde`, checkpoints can be serialized when their constituent types can.

Inner work is charged to its corresponding evaluation categories. Independent
stationarity checks add up to two gradient evaluations per consumed result.
Enable `with_inner_diagnostics(true)` and retain the solver with
`run_with_solver` to inspect the last native inner outcome and the multipliers
and penalty that defined it. Outer executor limits are checked between segments,
so one inner run can exceed a remaining evaluation allowance.

Supported dense pairs are `Vec<F>`/`DenseMatrix<F>`,
`nalgebra::DVector<F>`/`DMatrix<F>`, `ndarray::Array1<F>`/`Array2<F>`, and
`faer::Col<F>`/`Mat<F>`, with `F = f32` or `f64`, when the chosen inner supports
them. The outer method needs matrix-vector and transpose products, without a
linear solve. Custom inners can add requirements; the `Vec` example needs no
BLAS/LAPACK provider.

<References :entries='["nocedal-wright-2006"]' />
