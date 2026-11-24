<script setup lang="ts">
    import References from "#lib/docs/References.vue";
</script>

# Stochastic gradient descent

Stochastic gradient descent (SGD) is a local optimizer for objectives that
average a loss over many samples. Each step uses the gradient from a small
batch, so you can update the parameters without computing the full gradient.
Basin uses a constant learning rate and offers optional heavy-ball momentum.

[API reference](https://docs.rs/basin/latest/basin/solver/sgd/struct.Sgd.html)

  | Property    | Requirement                                                 |
  | ----------- | ----------------------------------------------------------- |
  | Objective   | An average of differentiable per-sample losses              |
  | Evaluations | `CostFunction` and `MiniBatchGradient`                      |
  | Search      | Local, from one starting point, with seeded batch shuffling |
  | Constraints | Unconstrained                                               |
  | Parameters  | All four dense vector backends, with `f32` or `f64`         |

## When to use it

Consider SGD when a full gradient is expensive because the objective includes
many samples. Small batches trade a less accurate direction for cheaper steps.
[Bottou (2012)](#ref-bottou-2012) discusses this trade-off for learning problems
where computing time limits the amount of data you can process.

If full gradients are affordable, compare [gradient
descent](../gradient-descent/), [nonlinear conjugate
gradient](../nonlinear-cg/), or [L-BFGS](../../quasi-newton/l-bfgs/). These
methods can make better use of accurate derivatives when you need a precise
solution. For a smooth objective with a projection onto its feasible set, see
[projected gradient descent](../projected-gradient-descent/).

Stochastic approximation goes back to [Robbins and Monro
(1951)](#ref-robbins-monro-1951). Its convergence results depend on the sampling
scheme, learning rates, and objective assumptions. Basin's fixed learning rate
does not provide a general guarantee of convergence to an exact minimizer; batch
gradients can remain noisy near a solution.

## How it works

The problem supplies the number of samples and a method that averages their
gradients over a requested batch of indices. For a batch gradient `g`, SGD takes
the step `x = x - alpha * g`. It uses the same learning rate `alpha` throughout
the run and performs no line search.

Basin shuffles the sample indices at initialization, consumes batches without
replacement, and reshuffles for the next epoch. If the remaining indices cannot
fill a batch, it discards that short tail. A fixed seed makes the batch order
reproducible. A batch containing all samples gives a full-gradient step.

With `with_momentum(beta)`, the solver carries a velocity between steps:
`v = beta * v - alpha * g`, followed by `x = x + v`. It starts with zero
velocity. This is the heavy-ball form associated with [Polyak
(1964)](#ref-polyak-1964); `beta = 0` selects ordinary SGD.

## Example

Fit a constant to four observations by minimizing their mean squared error. The
exact minimizer is their mean, `2.0`, and the minimum cost is `0.625`. The
example uses two samples per step and checks that the reported point is close to
this solution after a fixed budget.

<<< ../../../../../../crates/basin/examples/solver_guides/sgd.rs

[View `sgd.rs` on GitHub](https://github.com/jolars/basin/blob/main/crates/basin/examples/solver_guides/sgd.rs)

`batch_gradient` must return the average gradient over the requested samples.
Returning their sum would change the effective learning rate when the batch size
changes. The assertions allow for batch noise; reaching the iteration budget
does not establish convergence.

## Practical considerations

### Tune the learning rate

A large rate can produce unstable steps, while a small rate can make progress
slow. Scale the input features and parameters, and compare a few rates on a
representative subset. Basin does not decay the rate automatically. The
decreasing-rate convergence discussion in [Bottou (2012)](#ref-bottou-2012) does
not directly apply to this fixed-rate solver.

### Choose the batch size deliberately

Larger batches reduce the variation in the gradient but require more work per
step. The batch size must be positive; Basin clamps it to the number of samples
if necessary. Use a size that divides the sample count when every sample must
participate in each epoch.

### Retune when adding momentum

`with_momentum(beta)` can carry progress through successive directions, but it
also changes stability. Reduce the learning rate if the parameters oscillate or
diverge. Momentum does not remove batch noise or guarantee faster convergence.

## Implementation

Basin evaluates the full objective at the starting point, then refreshes it
every `n_samples / batch_size` mini-batch steps by default, using the clamped
batch size. `with_cost_eval_every(period)` changes that frequency. A period of
`1` evaluates the full objective after every step.

`PointState` keeps the latest evaluated point and its matching objective
together. Between refreshes, the working iterate advances inside the solver,
while ordinary results and observers retain the last evaluated point. Use
`Executor::run_with_solver()` and `Sgd::working_param()` to inspect that working
iterate, or refresh every step to publish it immediately. Reading either view
does not evaluate the objective. Each mini-batch gradient counts as one raw
gradient evaluation, regardless of batch size.

Supported parameter types are `Vec<F>`, `nalgebra::DVector<F>`,
`ndarray::Array1<F>`, and `faer::Col<F>`, with `F = f32` or `f64`. Custom
vectors need `ScaledAdd<F>`, `ScaleInPlace<F>`, and `Clone`; see the API's
[Backends
section](https://docs.rs/basin/latest/basin/solver/sgd/struct.Sgd.html#backends).

<References :entries='["robbins-monro-1951", "polyak-1964", "bottou-2012"]' />
