<script setup lang="ts">
    import CompetitorFigure from "#lib/benchmarks/CompetitorFigure.vue";
    import { COMPETITOR_BENCHMARKS as data } from "#lib/data/competitors.ts";
</script>

# Smooth unconstrained optimization

Compare gradient-based solvers on the 100-variable chained Rosenbrock problem.

## Gradient descent

Basin and [argmin](https://argmin-rs.org/) {{ data.versions.argmin }} use steepest
descent with a Moré–Thuente line search on chained Rosenbrock with 100
variables. Both start from the alternating vector $(-1.2,1,-1.2,1,\ldots)$. They
seek $f \le 10^{-10}$ with a {{ data.iterations }}-iteration safety cap.

<CompetitorFigure solver="gd" problem="rosenbrock" />

Read the time at a common objective gap as well as the endpoints: faster
iterations alone do not establish a faster solve.

## Nonlinear conjugate gradient

Basin and argmin {{ data.versions.argmin }} use Polak–Ribière+ on the same
100-variable chained Rosenbrock problem. Both start at
$(-1.2, 1, -1.2, 1, \ldots)$ and use the same analytic objective and gradient
with `Vec<f64>`. Both aim for an objective of $10^{-10}$ with a
10,000-iteration safety cap.

Both use Moré–Thuente with sufficient decrease $10^{-4}$, curvature $0.1$,
initial step $1$, relative bracket tolerance $10^{-10}$, and step bounds
$[\sqrt{\epsilon}, 10^{10}]$, where $\epsilon$ is machine epsilon for `f64`.
Optional restart rules are disabled. Basin still restarts when its direction
fails the sufficient descent test and retries a failed conjugate-direction
search along the negative gradient. Basin reuses accepted line-search
evaluations. Argmin reevaluates the objective and gradient at the accepted point.

<CompetitorFigure solver="nonlinear_cg" problem="rosenbrock" />

The chart follows the best objective among accepted iterates.

## BFGS

Basin and [argmin's BFGS](https://docs.rs/argmin/0.11.0/argmin/solver/quasinewton/struct.BFGS.html)
{{ data.versions.argmin }} solve the same 100-variable chained Rosenbrock problem
from the alternating start used for gradient descent. Both use `Vec<f64>`, analytic gradients,
and an identity initial inverse Hessian. The harness disables Basin's default
first-update rescaling to match argmin's unscaled identity seed.

Both use a Moré–Thuente line search with sufficient-decrease coefficient
$10^{-4}$, curvature coefficient $0.9$, relative bracket tolerance $10^{-10}$,
an initial trial step of 1, and step bounds $[\sqrt{\epsilon},10^{10}]$, where
$\epsilon$ is `f64` machine epsilon. Basin caps each search at 20 evaluations;
argmin exposes no equivalent cap. Both seek $f \le 10^{-10}$ with a
{{ data.iterations }}-iteration safety cap. Argmin's optional gradient and
cost-change stops are zeroed; Basin BFGS has no such optional stops.

<CompetitorFigure solver="bfgs" problem="rosenbrock" />

Basin applies the inverse-BFGS update through in-place rank-one operations,
requiring quadratic work in the dimension. Argmin 0.11 evaluates the factored
update with dense matrix products, requiring cubic work. Curvature safeguards,
line-search details, and rounding can also change trajectories. The curves show
effectiveness at this dimension, including those implementation differences.

Samples follow initialization and accepted iterates, with an endpoint after
solver teardown. The run details report objective and gradient callback counts,
including repeated evaluations at line-search endpoints. Independent checks
recompute the returned point's objective and require the common target outside
the timer. A failed or budget-limited run remains in the data.

## L-BFGS

L-BFGS uses the same 100-variable Rosenbrock fixture and start as gradient
descent. All three retain ten curvature pairs. Basin and argmin use a
Moré–Thuente line search. NLopt supplies its bundled L-BFGS implementation,
whose line search is not configurable through the harness's API.

All seek $f \le 10^{-10}$. Basin and argmin have a {{ data.iterations }}-iteration
safety cap, with argmin's gradient and cost-change tolerances zeroed. NLopt has
a {{ data.iterations }}-objective-evaluation cap. These limits bound the runs; they
do not prescribe equal work.

<CompetitorFigure solver="lbfgs" problem="rosenbrock" />

[Benchmark methodology and reproduction](https://github.com/jolars/basin/blob/main/web/README.md#competitors)
