<script setup lang="ts">
    import References from "#lib/docs/References.vue";
</script>

# BOBYQA

BOBYQA is a local optimizer for smooth functions with box bounds when
derivatives are unavailable. It builds a quadratic model from objective
values and searches within the intersection of a trust region and the box.
The bounds belong to the problem, so every objective evaluation can remain
inside the declared feasible region.

[API
reference](https://docs.rs/basin/latest/basin/solver/bobyqa/struct.Bobyqa.html)

  | Property    | Requirement                                  |
  | ----------- | -------------------------------------------- |
  | Objective   | A smooth scalar function                     |
  | Evaluations | `CostFunction`                               |
  | Search      | Local, from one starting point               |
  | Constraints | A box-only declaration through `Constraints` |
  | Parameters  | All four dense backends, with `f32` or `f64` |

## When to use it

Consider BOBYQA when variables have lower and upper bounds, gradients are
unavailable, and objective values can support a smooth local model.
[Powell (2009)](#ref-powell-bobyqa-2009) describes the method, including the
steps that maintain interpolation geometry near active bounds.

Use [NEWUOA](../newuoa/) for an unconstrained problem,
[LINCOA](../lincoa/) for linear inequalities, or [COBYLA](../cobyla/)
when nonlinear inequalities are needed. If gradients are available, compare
[L-BFGS-B](../../quasi-newton/l-bfgs-b/). For a nonsmooth objective, compare
[MADS](../mads/).

## How it works

BOBYQA interpolates objective values with a quadratic model. Its updates
minimize the Frobenius norm of the change to the model Hessian while
satisfying the interpolation conditions. The solver approximately minimizes
that model subject to both the current trust region and the box bounds.
Agreement between predicted and observed improvement controls the trust
region. [Powell (2009)](#ref-powell-bobyqa-2009) gives the model and
subproblem details.

Basin also takes feasible geometry steps that improve the arrangement of
interpolation points. The RESCUE procedure can restore the interpolation
system when rounding errors damage its algebra. Neither operation requires
the objective to decrease at every evaluation.

A separate resolution `rho` controls the stages of the search. The solver
reduces it toward the final radius and finishes after completing that final
stage.

## Example

The unconstrained minimum of this quadratic is `[2.0, -1.0]`, but the first
coordinate cannot exceed `1.0`. The constrained minimum is therefore
`[1.0, -1.0]`, with objective value `1.0`. The problem declares its complete
constraint set independently of its objective implementation.

<<< ../../../../../../crates/basin/examples/solver_guides/bobyqa.rs

[View `bobyqa.rs` on GitHub](https://github.com/jolars/basin/blob/main/crates/basin/examples/solver_guides/bobyqa.rs)

## Practical considerations

### Leave room for initial sampling

`with_initial_radius` defaults to `1.0`. Initialization needs each interval
to span at least twice that radius. If the requested radius is too large,
Basin reduces it to one quarter of the narrowest interval and, if necessary,
reduces the final radius as well. A fixed coordinate with equal lower and
upper bounds leaves no room for interpolation and causes initialization to
panic; remove fixed coordinates from the optimization variables.

Basin moves an infeasible starting point into the box and may adjust a point
near a bound to make room for the initial interpolation set. The first
evaluated point can therefore differ from the point you supplied. Scale
variables with very different units before choosing a common radius.

### Declare the supported constraints

BOBYQA requires `ConstraintSet<BoxBounds<V>>`, with both bound vectors
provided and matching the parameter dimension. A declaration containing
additional constraint blocks is not supported. The example uses finite
bounds with positive interval widths.

`with_npt` defaults to `2n + 1`. Basin currently supports
`2n + 1 <= npt <= (n + 1)(n + 2) / 2`; the smaller partial-model sizes of
the original algorithm are not implemented. Initialization evaluates all
`npt` points, and geometry restoration can require additional evaluations.

### Interpret the final radius

`with_final_radius` defaults to `1e-6`; the radii must satisfy
`initial_radius > final_radius > 0` after any narrow-box adjustment. Choose
the final radius for the variables, objective accuracy, and scalar precision.
Noisy values and discontinuities can undermine the interpolation model.

Completing the final radius stage is a solver convergence condition, rather
than a certificate of global optimality or parameter accuracy.
`with_absolute_radius_tolerance` optionally stops at a coarser `rho` without
changing the schedule and is disabled by default. Keep evaluation budgets on
the executor and inspect the termination report after the run.

## Implementation

BOBYQA publishes the current evaluated point and its historical incumbent
through `PointState`. The solver owns the model, interpolation geometry,
shifted bounds, and radius schedule. Fresh initialization rebuilds this work;
exact solver-and-state checkpoints preserve it. Use
`Executor::run_with_solver` to retain the solver and inspect `rho()` or
`geometry_diagnostics()`. Its diagnostic gradient belongs to the model and
does not evaluate an objective derivative.

Supported parameter types are `Vec<F>`, `nalgebra::DVector<F>`,
`ndarray::Array1<F>`, and `faer::Col<F>`, with `F = f32` or `f64`. The
quadratic model uses internal vectors and pure-Rust algebra without a
BLAS/LAPACK provider. Basin's implementation includes the bound-aware
subproblem, geometry steps, and RESCUE procedure, and has been
cross-validated against [PRIMA](https://github.com/libprima/prima) v0.7.2.

<References :entries='["powell-bobyqa-2009"]' />
