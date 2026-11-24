---

"title": "Benchmarks – Basin"
"description": "Read Basin's benchmark results: compare competing libraries and linear-algebra backends, with explanations of the measurements and their limits."
---

# Benchmarks

The benchmark suite compares Basin with other optimization libraries and
measures how linear-algebra backends affect solve time. These guides explain
the setup, results, and limits of each comparison.

## Comparing libraries

**How does Basin compare with other implementations?** The [library
comparisons](./competitors/) examine individual solvers, including L-BFGS,
Nelder–Mead, Levenberg–Marquardt, and COBYLA. Each section explains what the
harness matches and where the implementations differ.

These curves measure the complete optimization path recorded by each harness.
Differences in line searches, stopping rules, and numerical safeguards can
affect both the work performed and the result.

## Choosing a backend

**How does backend choice affect the cost of a solve as the problem grows?** The
[backend comparisons](./backends/) hold the solver and problem fixed while
varying the parameter and matrix types. They cover vector operations, dense
matrix methods, and sparse least squares.

Use these results to compare backends within a case. The cases use different
problems and stopping rules, so their solve times do not rank the algorithms.
