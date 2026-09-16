# SLSQP performance investigation

The [workspace-reuse follow-up](workspace-reuse.md) reduces the reported
Rosenbrock solve from 13.802 to 10.509 microseconds (24%) against the first
pass below. The reference medians are 10.231–10.312 microseconds, leaving
approximately 2–3%, within the observed run variation. Constrained HS71 also
improves by 16%. Numerical results, evaluation counts, and stopping rules are
preserved.

## First pass: temporary allocations

Measured September 16, 2026, against baseline
`2190280d46662b6c916643a861bac88b062b0675`. The reported case is the SLSQP
plot on `/benchmarks/competitors/`. Removing redundant temporary allocations
reduced its median complete-solve time from 18.964 to 14.059 microseconds
(26%). The references remain faster at approximately 10.1–10.2 microseconds.
These results do not cover the separate NEWUOA comparison on that page.

### Numerical comparison

The existing `src/slsqp.rs` harness runs analytic, unconstrained Rosenbrock in
two dimensions, using `Vec<f64>`, start `[-1.2, 1.0]`, and no finite bounds.
The objective is the conventional Rosenbrock sum, without a half-sum scaling.
All runs are deterministic and all three returned points pass independent
objective and gradient checks outside the timer.

| Implementation | Objective calls | Gradient calls | Stop |
| --- | ---: | ---: | --- |
| Basin, before and after | 49 | 36 | `SolverConverged` |
| `slsqp` 1.0.2 | 59 | 46 | `FtolReached` |
| `nlopt` 0.8.1, bundled NLopt 2.9.1 | 59 | 46 | `FtolReached` |

Every implementation returns exactly the same printed point,
`[0.9999999967339014, 0.9999999946665349]`, and objective,
`1.54363268165856448e-16`. Its independently evaluated gradient infinity norm
is `4.860250410903085e-7`; feasibility is vacuous. Verification adds one
objective and one gradient calculation, excluded from those counts. Basin
evaluates a gradient at initialization and each accepted step (35 steps here).
The reference interfaces do not report iteration counts.

Basin uses Kraft's composite accuracy test at `1e-10` and a 200-iteration cap.
The references use absolute objective-change tolerance `1e-10`, other optional
tolerances disabled, and a 200-evaluation cap. None exhausts its budget. Basin's
current and best accepted records coincide for this run; the graph instead
records objective improvements at trial callbacks. All 37 graph samples,
including the initial and final endpoints, retain identical objective values
before and after this change.

The implementations share Kraft's Han–Powell SLSQP, damped BFGS, original NNLS,
and inexact L1 merit line search. Basin follows
[Williams SLSQP 1.6.1](https://github.com/jacobwilliams/slsqp/tree/97884f98042624007f736dc536fa5636906ff26a).
The `slsqp` crate translates NLopt 2.7.1.
[NLopt documents](https://nlopt.readthedocs.io/en/latest/NLopt_Algorithms/#slsqp)
its different stopping rules and first-trial fused objective/gradient callback.
Thus equal tolerance values do not mean equal work. The before/after Basin
comparison preserves its evaluation counts and results; the kernel comparison
uses identical inputs to isolate implementation cost.

### Cause and changes

A separate optimized, frame-pointer-enabled `perf` build collected about
8,000 user-cycle samples with resolved callchains and no lost samples.
Percentages below use all sampled process cycles as their denominator:

- `Work::prepare`: 41.5% inclusive; its `Work::qp` child: 38.9% inclusive.
- `least_squares::lsei`, within the QP: 21.7% inclusive.
- `Factor::update`: 12.0% inclusive.
- `memmove`: 11.4% self time; allocator/free functions are also prominent.

Nested inclusive percentages must not be added. The profile points to the
solver and least-squares math, rather than expensive objective callbacks.
Separate allocator instrumentation counted 1,266 requests and 25,095 requested
bytes per published solve. These are cumulative requests, not peak memory.

The retained changes:

- Compute Householder column updates and norms directly from matrix storage.
  Allocate only the reflection tail, instead of copying a whole column and
  then copying its tail again.
- With no equalities, pass the original matrices to LSI. This avoids a
  duplicate reduced problem and an unused equality-multiplier residual.
- Evaluate Lagrangian components and validate gradients without temporary
  vectors; evaluate linear constraints directly from the backend vector.
- Reuse the owned BFGS difference vector and matrix-product output. Allocate
  the rank-one update's auxiliary vector only for the negative update.

No convergence setting, line-search rule, rank threshold, non-finite check,
public API, or persistent solver/checkpoint layout changes. The allocation
count falls to 658 requests and 15,351 requested bytes. The allocation test
failed before the change and now passes while checking the original evaluation
counts and objective.

### Timings

Linux x86-64, AMD Ryzen 9 7900, Rust 1.89.0, release profile with thin LTO and
one codegen unit; workspace lockfile unchanged. `competitor-bench` default
features: Basin includes `problems`, nalgebra 0.34.2, and faer 0.24.4, with no
parallel feature. All measured math uses serial `Vec<f64>` storage. Processes
were pinned to CPU 4; no profiling or allocation instrumentation was linked
into the timing binaries.

`slsqp_probe` warms each case, then records 21 batches. Full-solve batches have
1,000 fresh solves, including initialization, callbacks, trace logging,
termination, and result destruction. Kernel batches have `1000 / n` fresh
solves, including input matrix clones. Checks run outside the timer. Before
and after binaries were prebuilt, then alternated in three rounds (AB, BA, AB).
The table pools 63 batch averages and reports median and interquartile range;
the IQR is descriptive spread, not a confidence interval.

| Case | Before, µs (IQR) | After, µs (IQR) | Speedup |
| --- | ---: | ---: | ---: |
| basin | 18.964 (18.926–20.633) | 14.059 (13.940–14.372) | 1.35× |
| slsqp | 10.161 (10.134–10.187) | 10.083 (10.040–10.113) | 1.01× |
| nlopt | 10.242 (10.197–10.297) | 10.212 (10.120–10.242) | 1.00× |
| lsei_n2_constrainedfalse | 0.174 (0.173–0.175) | 0.122 (0.121–0.123) | 1.43× |
| lsei_n2_constrainedtrue | 0.246 (0.244–0.249) | 0.240 (0.235–0.245) | 1.03× |
| lsei_n8_constrainedfalse | 2.754 (2.733–2.775) | 2.135 (2.106–2.158) | 1.29× |
| lsei_n8_constrainedtrue | 1.286 (1.266–1.347) | 1.045 (1.041–1.055) | 1.23× |
| lsei_n32_constrainedfalse | 169.333 (164.295–171.674) | 140.832 (139.596–147.594) | 1.20× |
| lsei_n32_constrainedtrue | 25.906 (25.623–26.415) | 17.183 (17.125–17.328) | 1.51× |

The two-dimensional constrained kernel's small change overlaps run variation
and is inconclusive. The larger kernels and reported full solve show clear
improvements. References changed by less than 1%, which helps rule out a
general machine-speed change as the explanation. Early non-alternating runs
varied substantially (Basin's initial baseline was about 24 microseconds), so
the final claim uses the alternating measurements above.

The kernel builds an upper triangular E with diagonal `1 + i` and strict
upper entries `0.25`, with `f = E * ones(n)`. The constrained variant adds
`sum(x) = n` and `x[0] >= 0`; both variants independently check `x ≈ ones(n)`.
Sizes 2, 8, and 32 exercise the production LSEI source directly, without adding
public kernel APIs or maintaining a copied implementation. These kernels do
not establish full-solver performance on large or heavily active constraint
systems.

### Reproduction and validation

From the repository root:

```sh
cargo build --release -p competitor-bench --bin slsqp_probe --bin trace
mkdir -p target/perf-investigation/slsqp
taskset -c 4 target/release/slsqp_probe > target/perf-investigation/slsqp/current.csv
cargo test --release -p competitor-bench --test slsqp_allocations -- --nocapture
cargo run --release -p competitor-bench --bin trace > target/perf-investigation/slsqp/trace.json
```

For a historical baseline, create an isolated worktree at the revision above,
copy only `crates/competitor-bench/src/bin/slsqp_probe.rs` into it, and build in
its own target directory with the same toolchain and lockfile. The probe's
relative module path will compile that checkout's production LSEI code. Save
both prebuilt executables, run them in AB/BA order, and aggregate each case's
`ns` column. Do not compile or run other benchmarks during timing.

For profiling (separate from ordinary timing):

```sh
RUSTFLAGS='-C force-frame-pointers=yes' cargo build --profile profiling -p competitor-bench --bin slsqp_probe
perf record -e cycles:u -F 997 --call-graph fp \
  -o target/perf-investigation/slsqp/profile.data \
  -- taskset -c 4 target/profiling/slsqp_probe --profile
perf report --stdio --children --no-inline -i target/perf-investigation/slsqp/profile.data
```

Validation covers the full pure-Rust backend suite, `f32` round trips, focused
SLSQP tests with `serde` checkpoint round trips, HS71 reference trajectories,
active constraints, singular/non-finite subproblems, slack recovery, all-feature
workspace Clippy, formatting, and both default and no-default-feature WASM
builds. The published web data is not regenerated by this investigation.

This pass left approximately 39% against the Rust reference. The
[follow-up](workspace-reuse.md) implements QP/BFGS workspace reuse and records
its lifecycle checks, constrained workloads, allocation counts, and timings.
