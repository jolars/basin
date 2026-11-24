<script setup lang="ts">
    import CompetitorFigure from "#lib/benchmarks/CompetitorFigure.vue";
    import { COMPETITOR_BENCHMARKS as data } from "#lib/data/competitors.ts";
</script>

# Constrained optimization

Compare solvers on problems with box bounds, a linear sum constraint, or a nonlinear ball constraint.

## L-BFGS-B

Basin is compared with [lbfgsb](https://docs.rs/lbfgsb/0.1.1/lbfgsb/)
{{ data.versions.lbfgsb }}, which wraps Stephen Becker's C translation of L-BFGS-B
3.0. This case scales the reference's [extended Rosenbrock
driver](https://github.com/stephenbeckr/L-BFGS-B-C/blob/master/src/driver3.c)
to 20,000 variables:

$$
f(\mathbf{x}) = (x_1 - 1)^2 + 4\sum_{i=2}^{n}(x_i - x_{i-1}^2)^2.
$$

Every coordinate starts at 3. Lower bounds alternate between 1 and $-100$,
starting with 1, and every upper bound is 100. The all-ones vector attains
$f^\star = 0$ on the box faces. Both implementations use `Vec<f64>`, fused
analytic objective and gradient evaluations, and ten curvature pairs.

Both stop when the projected-gradient infinity norm reaches $10^{-8}$.
The signed relative-reduction tolerance is zero (`factr = 0`), so a
nonpositive reduction can also stop a run. Both use a Moré–Thuente line search
with sufficient-decrease coefficient $10^{-3}$, curvature coefficient 0.9,
relative bracket tolerance 0.1, and at most 20 evaluations per search. Safety
limits are 2,000 iterations and 10,000 objective calls, checked at accepted
iteration boundaries.

<CompetitorFigure solver="lbfgsb" problem="extended_rosenbrock" />

The curves sample initialization and accepted iterates, followed by an endpoint
that includes solver teardown. Independent checks recompute the objective and
projected gradient, require an objective gap at most $10^{-10}$, and check
returned and trial bound violations against $10^{-12}$. They check objective
accuracy and stationarity, rather than coordinate accuracy along this
ill-conditioned valley. The problem size gives both implementations at least
20 ms per solve on the measurement machine, including with intermediate trace
recording disabled.

## SLSQP

SLSQP minimizes the same coupled quadratic with $n = 100$, subject to the
nonlinear constraint

$$
\frac{1}{n}\sum_{i=1}^{n}x_i^2 \le 0.75^2.
$$

The unique optimum again has every coordinate equal to $0.75$, with
$f^\star = 6.25$. This case uses the same starting-point formula as COBYLA,
analytic objective gradients, and an analytic constraint Jacobian. Normalizing
the constraint by $n$ keeps its scale independent of dimension.

Basin is compared with [slsqp](https://docs.rs/slsqp/latest/slsqp/)
{{ data.versions.slsqp }}, a Rust translation of NLopt, and native NLopt. Basin uses
Kraft's composite accuracy test at $10^{-10}$. The references use an absolute
function-change tolerance of $10^{-10}$, with their other optional objective and
step tolerances disabled. Basin has a 200-iteration safety cap; the references
have a 200-objective-evaluation cap. The objective curve uses a
constraint-violation tolerance of $10^{-8}$.

<CompetitorFigure solver="slsqp" problem="constrained_quadratic" />

Termination statuses and independent accuracy and feasibility checks are
reported in the run details. Passing an independent solution check does not mean
that the solver reported convergence. The main curve records best feasible
trials, which can differ from the returned point.

## COBYLA

The constrained fixture minimizes a coupled quadratic,

$$
f(\mathbf{x}) = \sum_{i=1}^{n}(x_i - 1)^2 + \sum_{i=1}^{n-1}(x_{i+1} - x_i)^2,
$$

with $n = 30$, $0 \le x_i \le 2$, and $\sum_i x_i \le 0.75n$. The unique optimum
has every coordinate equal to $0.75$ and $f^\star = n/16 = 1.875$. The
deterministic start is $x_i = 0.3 + 0.4(i - 1)/n$.

Basin's PRIMA implementation is compared with [cobyla](https://docs.rs/cobyla/)
{{ data.versions.cobyla }}, derived from NLopt, and native NLopt 2.9.1 through the
[nlopt](https://docs.rs/nlopt/) {{ data.versions.nlopt }} Rust bindings. All three
use an initial radius of $0.5$, a final radius of about $7.45 \times 10^{-9}$,
a 10,000-objective-call budget, and disabled optional objective-change stops.
The references use native bounds and a zero tolerance for the sum constraint.
Objective trials enter the main curve only when their maximum constraint
violation is at most $1.49 \times 10^{-8}$.

The [NLopt variant](https://nlopt.readthedocs.io/en/latest/NLopt_Algorithms/#cobyla-constrained-optimization-by-linear-approximations)
modifies Powell's original implementation with explicit bound handling,
trust-region increases, and deterministic simplex perturbations. Basin follows
modern PRIMA, so matched budgets and radii do not imply identical trajectories.

<CompetitorFigure solver="cobyla" problem="quadratic_30d" />

Read the violation plot alongside objective progress: an infeasible trial with a
lower objective is not a better constrained solution. Small negative objective
gaps permitted by the feasibility tolerance appear at the display floor.

Constraint counts use different callback units. Basin records vector callbacks
including bounds, while both references record scalar nonlinear callbacks and
exclude native bounds.

[Benchmark methodology and reproduction](https://github.com/jolars/basin/blob/main/web/README.md#competitors)
