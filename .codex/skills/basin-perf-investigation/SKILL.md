---
name: basin-perf-investigation
description: Investigate performance in this Basin repository using its solver
  and backend benchmarks, competitor harnesses, evaluation accounting, and
  profiling tools. Use for Basin solver slowdowns, executor or adapter overhead,
  math-kernel hotspots, allocation costs, or measured optimization work.
---

# Investigate Basin Performance

Reproduce the reported workload, establish a fair reference, and explain the
cost with measurements. An investigation can finish with a supported diagnosis
and ranked next steps. When the user requests a speedup, carry promising fixes
through numerical verification and fresh timing. Do not turn an investigation
alone into a solver redesign or publication of benchmark results.

This is a project-local skill for Basin's Rust optimization library and its
workspace. Run commands from the repository root. Read `AGENTS.md`, the relevant
benchmark source, and the target implementation. Consult `CONTRIBUTING.md`
before changes to architecture, public APIs, dependencies, or platform support.

## Establish the comparison

Keep the user's solver, variant, workload, and reference. If none is specified,
choose a representative existing case and state the choice. For a regression,
compare against a known earlier revision. For an implementation comparison,
prefer author-maintained reference code or an established library implementing
the same variant. Record its version or commit and material differences; a
different algorithm is a practical competitor, not an implementation oracle. Use
primary documentation or source to resolve ambiguous reference semantics.

Before timing, run each contestant once and record:

- Problem dimensions, data, starting point, scalar type, bounds, and
  constraints. Check constraint signs, residual scaling, and whether the
  reported objective is a sum of squares or half that sum.
- Algorithm settings: line search, memory size, initial simplex or population,
  trust-region radii, scaling, restart rules, and random seeds as applicable.
- Actual stopping conditions, including implicit solver stops, adapter tolerance
  floors, and the meaning of each budget. Equal tolerance values or iteration
  caps do not establish equal work across libraries.
- Returned objective, feasibility, gradient or residual norm when relevant,
  termination reason, iterations, and objective, derivative, and constraint
  evaluations. Validate the returned point independently outside the timer.
  Distinguish current and best-so-far state, and count verification calls
  separately. Budget exhaustion alone does not establish convergence;
  distinguish budget-limited output from verified success and invalid non-finite
  results.

Choose the measurement that answers the question:

- **Implementation cost:** matched work, such as a kernel on identical inputs or
  the same solver steps. Verify evaluation counts and numerical results; equal
  iterations alone can hide different line-search or inner-solver work.
- **Solver effectiveness:** time and evaluations to a common accuracy and
  feasibility target, or quality achieved within a common budget. Retain failed
  cases and success rates. Use paired seeds and multiple starts when stochastic
  variation matters. Convergence traces help explain differing trajectories.

Use both views when fewer evaluations could explain the speed difference. For
non-solver components, establish equivalent outputs and the relevant work unit
instead of imposing solver convergence criteria.

## Benchmark and profile

Read [Benchmarking Basin](references/benchmarking.md) when selecting a harness
or preparing timings. It maps the existing benchmarks and verification probes,
explains Basin's counter and solver-lifecycle semantics, and gives commands for
reference comparisons and before/after baselines. Check those semantics before
using a state's `cost_evals()` as a measure of objective calls.

Establish an uninstrumented baseline before changing production code. Record
versions, features, workload, timing boundaries, and sample uncertainty. Isolate
historical builds from the user's checkout, and alternate prebuilt contestants
when drift could obscure a small difference. Match numerical work and thread
settings before attributing a gap to implementation cost.

Read [Profiling Basin](references/profiling.md) when collecting a profile. It
provides a focused Criterion flamegraph command, optimized build settings,
stack-validation guidance, and fallback options. Keep profiling and allocation
instrumentation separate from timing builds; profile percentages alone cannot
establish a speedup.

## Locate the cause and test a hypothesis

Read inclusive costs before self-time leaves. Separate problem callbacks, solver
work, backend kernels, executor bookkeeping, and outer adapters. Then follow the
hot callers into the actual implementation:

- **Extra evaluations or inner iterations:** inspect termination semantics,
  line-search trials, rejection steps, derivative reuse, and callback counts.
- **Allocation and copying:** trace allocator frames to workspace creation,
  vector clones, conversions, or state snapshots. Measure requests and bytes in
  a separate instrumented run; cumulative requested bytes are not peak memory.
- **Matrix work:** isolate factorization, matrix products, layouts, dispatch,
  and scratch allocation at representative dimensions and conditioning. Check
  scaling over sizes before proposing a backend switch.
- **Framework or adapter overhead:** compare a valid initialized solver loop
  with `Executor`, then add the adapter or observers. Keep the callbacks,
  stopping policy, and returned-state semantics equivalent across layers.

Label the denominator for reported percentages; nested inclusive shares cannot
be summed. A smaller hotspot share alone does not establish less elapsed time.
Use source inspection and a focused experiment to connect the profile to a
specific cause.

When implementing a fix, test one supported hypothesis at a time. Preserve
numerical safeguards, constraint handling, evaluation accounting, and public
contracts. Looser tolerances, removed recovery checks, changed precision, or
different algorithms require separate justification as numerical tradeoffs.
Floating-point bit identity is not generally required, but approximate result
checks, feasibility, and relevant degenerate or ill-conditioned cases are.

Rerun focused correctness tests, reference checks, and uninstrumented benchmarks
on the original case plus representative contrasting sizes or workloads. Follow
the repository's scope-appropriate verification, including supported backends
and `f32` coverage when affected. Keep default WASM and feature guarantees;
all-feature tests need an explicit BLAS/LAPACK provider. Wall-time thresholds
are unsuitable unit-test assertions; deterministic work or allocation ceilings
can guard a demonstrated regression when stable and paired with result checks.

If the difference remains within measurement uncertainty, report it as
inconclusive. Discard unsupported experimental edits without disturbing user
changes, and record the finding rather than accumulating speculative rewrites.

## Leave reproducible evidence

Keep raw profiles and exploratory output in ignored `target/` directories or a
temporary workspace. Preserve useful probes and compact findings under
`crates/competitor-bench/investigations/<topic>/` when durable artifacts are
part of the task. Record enough metadata and exact commands to repeat the run.
The `task bench:*` commands also regenerate published web data; use the direct
harness for investigation and refresh website results only within the requested
scope.

Report the reference and workload, numerical comparability and remaining
asymmetries, timings with uncertainty, evaluation counts or success rates,
profile evidence identifying the responsible layer and function, and any
verified change. Include validation results, attempted ideas that did not pay,
artifact paths, and the next experiment supported by the evidence. Distinguish
measured findings from hypotheses and tooling limitations.
