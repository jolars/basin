<script setup lang="ts">
    import References from "#lib/docs/References.vue";
</script>

# L-BFGS

L-BFGS is a local optimizer for smooth functions when you can compute a
gradient. It learns how to scale and combine gradient directions from recent
steps, often making more progress than steepest descent without requiring a
Hessian. Its limited memory makes it useful when a full BFGS matrix would be too
large.

[API
reference](https://docs.rs/basin/latest/basin/solver/lbfgs/struct.Lbfgs.html)

  | Property    | Requirement                                                        |
  | ----------- | ------------------------------------------------------------------ |
  | Objective   | A smooth scalar function                                           |
  | Evaluations | `CostFunction` and `Gradient`                                      |
  | Search      | Local, from one starting point                                     |
  | Constraints | Unconstrained; use [L-BFGS-B](../l-bfgs-b/) for box bounds        |
  | Parameters  | Contiguous vectors on all four dense backends, with `f32` or `f64` |

## When to use it

Consider L-BFGS when the gradient is available and storing a dense approximation
to the inverse Hessian would be expensive. The history length lets you trade
memory for additional curvature information. The original numerical study is
[Liu and Nocedal (1989)](#ref-liu-nocedal-1989).

For a smaller problem where a dense approximation is affordable,
[BFGS](../bfgs/) is another option. If you can supply useful Hessian-vector
products, consider [trust-region Newton](../../trust-region/trust-region-newton/).
For curve fitting expressed through residuals and a Jacobian, compare the methods
in the [nonlinear least-squares benchmarks](../../../benchmarks/least-squares/).

L-BFGS follows local descent directions, so different starting points can lead
to different minima. Its curvature estimates also depend on changes in the
gradient. If your objective is noisy or nonsmooth, compare the methods in the
[derivative-free benchmarks](../../../benchmarks/derivative-free-and-global/).

## How it works

After an accepted step, L-BFGS forms two vectors: the change in the parameters
and the change in the gradient. Together, they describe how the local slope
changed as the solver moved. It retains at most `m` of these pairs.

The next iteration uses this history to transform the negative gradient into a
search direction. A two-loop recursion applies the inverse-Hessian approximation
without constructing a dense matrix. Basin follows Algorithm 7.4 in [Nocedal and
Wright (2006)](#ref-nocedal-wright-2006).

A line search then chooses how far to move along that direction. Basin uses the
[Moré–Thuente line
search](https://docs.rs/basin/latest/basin/line_search/more_thuente/struct.MoreThuente.html)
by default, which seeks sufficient decrease and a strong Wolfe curvature
condition ([Moré and Thuente, 1994](#ref-more-thuente-1994)). One solver
iteration can therefore require several objective and gradient evaluations.

## Example

This quadratic has its minimum at `[1.0, -2.0]`. The example supplies an
analytic gradient, sets a gradient tolerance on the solver, and limits
iterations through the executor.

<<< ../../../../../../crates/basin/examples/solver_guides/lbfgs.rs

[View `lbfgs.rs` on GitHub](https://github.com/jolars/basin/blob/main/crates/basin/examples/solver_guides/lbfgs.rs)

Call `.unbounded()` to select L-BFGS. `Lbfgs::new()` alone selects the
box-constrained L-BFGS-B mode, also available through the `Lbfgsb` alias.

## Practical considerations

### Scale the parameters

When parameter units differ greatly, rescaling can make the curvature
information and gradient tolerance easier to interpret. An absolute tolerance of
`1e-8` is a choice for the example above; choose your threshold for the units,
precision, and derivative accuracy of your problem.

### Start with the default history length

Basin retains up to ten curvature pairs by default. `with_m_capacity(m)` changes
this capacity for a fresh solve. Increasing it stores more information and uses
more memory, but does not guarantee faster convergence.

## Implementation

The unconstrained mode uses the two-loop recursion with a scaled identity
initial approximation. Basin skips curvature updates that fail its curvature
safeguard. It publishes the current point, objective, and gradient through
`FirstOrderState`; the solver retains the curvature history.

The default line search is a port of MINPACK-2's `dcsrch` and `dcstep` routines.
You can supply another strategy through
`Lbfgs::with_line_search(search).unbounded()`. The related L-BFGS-B mode is a
port of the L-BFGS-B 3.0 Fortran implementation and adds box-specific steps; it
is a distinct algorithm from unconstrained L-BFGS.

Supported parameter types are `Vec<F>`, `nalgebra::DVector<F>`,
`ndarray::Array1<F>`, and `faer::Col<F>`, with `F = f32` or `f64`. Parameter
storage must be contiguous; strided ndarray views are unsupported. See the API's
[Backends
section](https://docs.rs/basin/latest/basin/solver/lbfgs/struct.Lbfgs.html#backends)
for the exact capability requirements.

See the [library benchmark](../../../benchmarks/smooth-unconstrained/#l-bfgs) for this solver.

<References
:entries='["liu-nocedal-1989", "nocedal-wright-2006", "more-thuente-1994"]' />
