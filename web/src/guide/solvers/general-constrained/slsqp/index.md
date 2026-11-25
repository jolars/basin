<script setup lang="ts">
    import References from "#lib/docs/References.vue";
</script>

# SLSQP

Sequential least-squares programming (SLSQP) minimizes smooth scalar functions
with bounds, linear constraints, and nonlinear constraints. It combines
objective and constraint derivatives with a dense approximation to the
Lagrangian Hessian. You do not need to supply a Hessian.

[API
reference](https://docs.rs/basin/latest/basin/solver/slsqp/struct.Slsqp.html)

  | Property    | Requirement                                                                           |
  | ----------- | ------------------------------------------------------------------------------------- |
  | Objective   | A smooth scalar function                                                              |
  | Evaluations | `CostFunction`, `Gradient`, and derivatives of present nonlinear blocks               |
  | Search      | Local, from one starting point                                                        |
  | Constraints | Bounds, linear equalities and inequalities, and nonlinear equalities and inequalities |
  | Parameters  | All four dense backends, with `f32` or `f64`                                          |

## When to use it

Consider SLSQP for small and medium dense problems with reliable derivatives and
several kinds of constraints. The starting point may violate linear or nonlinear
constraints. Basin clips initial and trial points to any declared box and
eliminates fixed coordinates internally.

For box bounds alone, compare [L-BFGS-B](../../quasi-newton/l-bfgs-b/). For
linear constraints and an existing unconstrained inner solver, compare the
[barrier method](../barrier-method/) for inequalities or the [augmented
Lagrangian method](../augmented-lagrangian-method/) for equalities. If
derivatives are unreliable, compare [COBYLA](../../derivative-free/cobyla/).
SLSQP finds a local solution, so the starting point still matters.

## How it works

Basin implements Kraft's Han–Powell method from the [1988
report](#ref-kraft-1988), §§2.2 and 3.2. At each accepted point, it linearizes
the constraints and builds a quadratic objective model using a positive
definite, damped BFGS approximation to the Lagrangian Hessian. Constrained
least-squares routines solve the resulting quadratic subproblem.

An inexact line search balances the objective against an L1 constraint penalty.
An additional slack variable can recover from an inconsistent linearization, and
Hessian resets can recover a descent direction. Failed factorizations or
exhausted recovery produce a numerical failure.

The public convention is `h(x) = 0` for equalities and `c(x) <= 0` for
inequalities. Linear blocks supply exact values and derivatives. Each present
nonlinear block needs its value callback, and `ConstraintJacobian` supplies rows
for nonlinear equalities first, followed by nonlinear inequalities. Linear rows
are excluded from this Jacobian.

## Example

The unconstrained minimum of this quadratic is `[2.0, 0.0]`. Restricting the
point to the unit disk moves the solution to `[1.0, 0.0]`, with cost `1.0`. The
example starts outside the disk and supplies analytic derivatives of both the
objective and the nonlinear inequality.

<<< ../../../../../../crates/basin/examples/solver_guides/slsqp.rs

[View `slsqp.rs` on GitHub](https://github.com/jolars/basin/blob/main/crates/basin/examples/solver_guides/slsqp.rs)

## Practical considerations

### Declare every constraint block

Attach the complete typed `ConstraintSet` with `Constrained`, or implement
`Constraints` directly. Constraint counts, dimensions, bounds, and linear
coefficients must stay fixed during the solve. Linear-only declarations need no
nonlinear callback or `ConstraintJacobian` implementation.

[`FiniteDiff`](https://docs.rs/basin/latest/basin/core/numdiff/struct.FiniteDiff.html)
can synthesize derivatives. Use
[`BoundedFiniteDiff`](https://docs.rs/basin/latest/basin/core/numdiff/bounded/struct.BoundedFiniteDiff.html)
when numerical perturbations must respect the box. Callbacks must still accept
points that violate linear or nonlinear constraints.

### Scale constraints and interpret accuracy

`with_absolute_accuracy_tolerance` defaults to `1e-6`. Kraft's composite tests
require a small sum of constraint violations together with a small QP optimality
measure, objective change, or step norm. After five Hessian resets, the
reference also permits the objective-change or step test at ten times that
accuracy. These tests do not establish a strict KKT certificate.

Keep constraint residuals and objective values on useful scales. A tolerance on
a sum of violations depends on both the units and the number of rows. `f32`
generally needs a looser accuracy. `None` disables the convergence tests, and
zero requests exact-zero comparisons.

### Inspect diagnostics and budget all evaluations

Retain the solver with `run_with_solver` to read `constraint_violation()`,
`stationarity()`, `complementarity()`, and multiplier estimates. Stationarity
uses the projected Lagrangian gradient. Complementarity and returned inequality
multipliers exclude bound multipliers. Diagnostic readers return `None` until
their values are available.

Numerical failures report `SolverFailed` with details in `failure()`; callback
errors propagate through the typed error. `with_max_subproblem_iterations`
overrides the NNLS active-set limit. Keep overall limits on the executor: one
iteration can contain several line-search trials, constraint values count as
residual evaluations, and constraint derivatives count as Jacobian work.

## Implementation

`SelectedFirstOrderState` publishes the latest accepted point with its matching
objective, objective gradient, and unscaled L1 violation report. Its `best_*`
readers follow the solver's explicit selection of accepted iterates. Inspect
feasibility alongside cost when reporting a result.

Fresh initialization reevaluates the clipped seed and rebuilds the Hessian model
and constraint layout. An `Checkpoint` retains the solver and state for
continuation. With `serde`, checkpoints support serialization and rebuild
disposable scratch on demand.

Supported dense pairs are `Vec<F>`/`DenseMatrix<F>`,
`nalgebra::DVector<F>`/`DMatrix<F>`, `ndarray::Array1<F>`/`Array2<F>`, and
`faer::Col<F>`/`Mat<F>`, with `F = f32` or `f64`. Constraint blocks can use
independent storage. Internal dense factorizations need no BLAS/LAPACK provider
and work on WASM; storage grows quadratically with problem dimensions.

Reference comparisons use [Jacob Williams's SLSQP
1.6.1](https://github.com/jacobwilliams/slsqp/tree/97884f98042624007f736dc536fa5636906ff26a)
with the original NNLS kernel and inexact line search. Basin does not expose the
reference's exact line search or alternative BVLS kernel.

See the [library benchmark](../../../benchmarks/constrained/#slsqp) for this solver.

<References :entries='["kraft-1988"]' />
