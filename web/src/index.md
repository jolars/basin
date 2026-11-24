---
layout: home
title: Numerical optimization in Rust
hero:
  name: Basin
  text: Numerical optimization in Rust
  tagline: Solve optimization problems with your choice of solver and linear-algebra backend. Run natively or in the browser.
  image:
    light: /logo.svg
    dark: /logo-dark.svg
    alt: Optimization trajectories on the Rosenbrock function
  actions:
    - theme: brand
      text: What is Basin?
      link: /guide/
    - theme: alt
      text: Get started
      link: /guide/getting-started/
    - theme: alt
      text: Open the visualizer
      link: /visualizer/
features:
  - title: "A wide range of solvers"
    details: "Choose from gradient-based methods like L-BFGS, derivative-free methods like Nelder–Mead, least-squares solvers like Levenberg–Marquardt, and global methods like CMA-ES."
  - title: "Multiple backends"
    details: "Run on plain Vec<f64>, or select a compatible nalgebra, ndarray, or faer release with exact-version and latest feature aliases."
  - title: "Constrained optimization"
    details: "Solve problems with box bounds, linear constraints, or nonlinear constraints using solvers that support them."
---

<script setup lang="ts">
import Playground from "#lib/playground/Playground.vue";
</script>

## A small example

Implement `CostFunction` and `Gradient`, then hand your problem, a solver, and a
starting point to the `Executor`.

Here's a live example of gradient descent on the Rosenbrock valley. Drag the
sliders or click the plot to move the start, and watch the run and the code
update together.

To try more problems and solvers, open the [visualizer](/visualizer/).

<div class="vp-raw">
    <Playground />
</div>
