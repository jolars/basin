<script setup lang="ts">
    import CompetitorFigure from "#lib/benchmarks/CompetitorFigure.vue";
    import { COMPETITOR_BENCHMARKS as data } from "#lib/data/competitors.ts";

    const cmaesNalgebraVersion = data.versions["nalgebra-cmaes"];
</script>

# Derivative-free and global search

Compare derivative-free methods on Rosenbrock and Styblinski–Tang, and active CMA-ES on rotated ellipsoid and Ackley problems.

## Nelder–Mead

The simplex comparison uses 12-variable chained Rosenbrock, with the same
alternating starting-point formula. Basin and argmin use identical simplex
coordinates and standard coefficients. [gomez](https://docs.rs/gomez/)
{{ data.versions.gomez }} and NLopt construct their own simplexes with their default
coefficients.

All seek $f \le 10^{-10}$. Basin, argmin, and gomez have a
{{ data.iterations }}-iteration cap; NLopt has a
{{ data.iterations }}-objective-evaluation cap.

<CompetitorFigure solver="nm" problem="rosenbrock" />

## NEWUOA

Basin and NLopt run Powell's model-based derivative-free method on
30-dimensional Styblinski–Tang, starting at the origin. Basin follows the
modernized PRIMA implementation, while NLopt uses Powell's reference.

The initial radius is $0.5$, the final radius is $10^{-6}$, and the
interpolation set has $2n + 1$ points. Basin has a 10,000-iteration safety cap
and NLopt a 10,000-evaluation cap.

<CompetitorFigure solver="newuoa" problem="styblinski" />

A single start does not measure the probability of finding the global minimum on
a multimodal problem.

## Active CMA-ES

Basin is compared with [cmaes](https://docs.rs/cmaes/0.2.2/cmaes/)
{{ data.versions.cmaes }} on two unconstrained problems with 40 variables. Both
use active covariance adaptation; the reference explicitly selects
`Weights::Negative`. Both start with mean $(3,\ldots,3)$, identity covariance,
step size $\sigma_0 = 2$, and a population of 16. Runs use a fixed seed of 0,
without restarts or local search. Matching seed numbers do not produce the same
random samples across libraries.

The rotated ellipsoid is a nonseparable quadratic with condition number $10^6$:

$$
f(\mathbf{x}) = \sum_{i=0}^{39} 10^{6i/39}(Q\mathbf{x})_i^2,
$$

where $Q$ is the orthonormal discrete cosine transform, with
$Q_{0j} = 1/\sqrt{40}$ and
$Q_{ij} = \sqrt{2/40}\cos(\pi i(j+1/2)/40)$ for $i > 0$. Its global minimum
is zero at the origin. The dense rotation prevents the solvers from exploiting
coordinate separability.

<CompetitorFigure solver="cmaes" problem="rotated_ellipsoid" />

[Ackley](https://docs.rs/basin/latest/basin/problems/ackley/struct.Ackley.html)
has a broad outer funnel and many local minima. We use Basin's standard formula
with $a = 20$, $b = 0.2$, and
$c = 2\pi$, whose global minimum is also zero at the origin. These runs impose
no box bounds.

<CompetitorFigure solver="cmaes" problem="ackley" />

Both drivers seek a best evaluated objective of at most $10^{-8}$, with safety
caps of 100,000 objective calls and 10,000 sampled generations. Budgets are
checked at generation boundaries, so a run can exceed its evaluation cap by
less than one generation. The reference's optional objective-change and
distribution-size tolerances are zeroed, and its stagnation window lies beyond
the generation cap. Its intrinsic numerical safeguards remain enabled, and
their termination reasons remain visible. Basin's optional distribution-size
check is disabled. Independent checks recompute the best returned point's
objective outside the timer; a budget or safeguard stop alone does not count
as success.

Basin uses nalgebra {{ data.versions.nalgebra }}, while `cmaes` uses nalgebra
{{ cmaesNalgebraVersion }}. Basin evaluates the mean as well as the 16
population members, charging 17 objective calls per sampled generation; the
reference evaluates only the 16 members. Basin uses
`with_eigen_update_period(3)` to refresh its eigendecomposition every three
generations, matching the reference's cadence for this workload. Basin's solver
default remains every generation. Periodic refresh changes seeded trajectories
and can affect convergence.

Each run warms up once and uses {{ data.repetitions }} timing repetitions of the
same numerical path. A single seed does not measure stochastic reliability.

[Benchmark methodology and reproduction](https://github.com/jolars/basin/blob/main/web/README.md#competitors)
