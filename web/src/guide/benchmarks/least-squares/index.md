<script setup lang="ts">
    import CompetitorFigure from "#lib/benchmarks/CompetitorFigure.vue";
    import DoglegFigure from "#lib/benchmarks/DoglegFigure.vue";
    import { COMPETITOR_BENCHMARKS as data } from "#lib/data/competitors.ts";

    const lmVersion = data.versions["levenberg-marquardt"];
</script>

# Nonlinear least squares

Compare Levenberg–Marquardt on an exponential fit and Powell dogleg on NIST’s BoxBOD regression problem.

## Levenberg–Marquardt

Basin {{ data.versions.basin }} and
[levenberg-marquardt](https://docs.rs/levenberg-marquardt/latest/levenberg_marquardt/)
{{ lmVersion }} jointly fit 16 exponential response curves. Each curve has an
amplitude and a decay rate, giving 32 parameters. Each has 256 exact
observations at evenly spaced times in $[0,4]$, giving 4,096 residuals. For
curve $k = 0,\ldots,15$, the true parameters are $a_k = 100\,000 + 1\,000k$ and
$b_k = -1 - 0.02k$. Every pair starts at $(50\,000,-0.3)$.

Both use nalgebra {{ data.versions.nalgebra }}, the same analytic Jacobian, column
scaling, pivoted QR, and an initial scaled step-bound factor of 100. Although
curves have separate parameters, the harness supplies a single dense Jacobian
and both libraries solve the complete dense least-squares system. This tests the
dense QR path, rather than a specialized solver that exploits the blocks.

Function-reduction, orthogonality, and scaled-radius tolerances are 30 times
`f64` machine epsilon, about $6.66 \times 10^{-15}$. Basin uses trust-region
damping and disables its absolute-gradient and unscaled-step checks. Both have a
330-residual-call budget; Basin also has a 200-iteration safety cap.

<CompetitorFigure solver="lm" problem="exp_fit" />

The curve records the best objective seen at residual evaluations, including
rejected trials. Internal damping searches and acceptance rules still differ
from the reference's MINPACK path.

## Powell dogleg on NIST least squares

The plot shows the first published start for BoxBOD, one of the 27 NIST
Statistical Reference Datasets for nonlinear regression. All three
configurations use the same residuals and analytic Jacobian, with `f64` and
nalgebra 0.34.

Basin's dogleg uses the Gauss–Newton least-squares model, with Cholesky on
$J^\top J$ and a Cauchy step if the solve fails. The
[`dogleg` 0.1.1 crate](https://docs.rs/dogleg/0.1.1/dogleg/) uses an SVD of
the Jacobian. The unscaled configuration shares Basin's Euclidean trust region
and initial radius of 1. The default configuration uses the crate's own
scaling and initial radius. Each implementation applies its own acceptance
and stopping rules.

<DoglegFigure />

[Benchmark methodology and reproduction](https://github.com/jolars/basin/blob/main/web/README.md#competitors)
