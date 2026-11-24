<script setup lang="ts">
    import SolverFinder from "#lib/docs/SolverFinder.vue";
</script>

# Choosing a solver

Choose your problem type and describe the objective, constraints, and
available derivatives. The sidebar lists solvers by algorithm family.

## Where to start

Use the solver helper below to suggest a solver for your problem.

<SolverFinder />

::: tip Note
These choices are starting points for comparison. A local method depends on its
starting point, and a global search does not guarantee that a finite run will
find the global minimum. Compare evaluation counts and the quality of the
returned point on your own problem.
:::

## Constraints and derivatives

Choose a solver that handles every constraint block your problem declares. Box
bounds, linear constraints, and nonlinear constraints are different
capabilities. An unconstrained solver ignores constraint declarations.
Projection, barrier, and penalty adapters are explicit choices.

When derivatives are unavailable, you can use a derivative-free solver or
synthesize derivatives with
[`FiniteDiff`](https://docs.rs/basin/latest/basin/core/numdiff/struct.FiniteDiff.html).
Finite differences add evaluations and introduce approximation error; take both
into account when choosing a method and its stopping tolerances.

The Guide compares solver implementations on shared problems in [smooth
unconstrained optimization](../benchmarks/smooth-unconstrained/), [constrained
optimization](../benchmarks/constrained/), [derivative-free and global
search](../benchmarks/derivative-free-and-global/), and [nonlinear least
squares](../benchmarks/least-squares/).
